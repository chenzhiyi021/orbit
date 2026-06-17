import logging

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from orbit.backends.sglang_utils.sglang_engine import SGLangEngine
from orbit.utils.logging_utils import configure_logger

from .utils import build_noset_visible_devices_env_vars

logger = logging.getLogger(__name__)

@ray.remote
class TeacherManager:
    """
    Ray actor wrapping a SGLang engine for frozen teacher inference.

    One TeacherManager instance handles a single teacher model.
    For MOPD with multiple teachers, instantiate one TeacherManager per
    teacher model and route samples from train_opd.py.

    Public API:
        score(token_ids, attention_mask) -> dict
        offload() / onload()             -> GPU memory management
    """

    def __init__(self, args, pg):
        configure_logger()

        self.args = args
        self.loss_type = args.opd_loss_type
        self.topk_k = args.opd_topk_k

        pg_obj, bundle_indices, gpu_ids = pg
        base_gpu_id = int(gpu_ids[0])

        # tp_size engines share the same pg; each takes one bundle slot.
        self._engines = []
        for i in range(args.opd_teacher_tp_size):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg_obj,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_indices[i],
            )
            engine = ray.remote(SGLangEngine).options(
                num_cpus=0.2,
                num_gpus=0.2,
                scheduling_strategy=scheduling_strategy,
                runtime_env={"env_vars": build_noset_visible_devices_env_vars()},
            ).remote(
                args,
                rank=i,
                worker_type="teacher",
                base_gpu_id=int(gpu_ids[i]),
                sglang_overrides={},
                num_gpus_per_engine=args.opd_teacher_tp_size,
            )
            self._engines.append(engine)

        # Initialize engines (blocking so teacher is ready before training starts)
        ray.get([engine.init.remote() for engine in self._engines])
        logger.info(
            f"TeacherManager ready: model={args.opd_teacher_model_path}, "
            f"tp_size={args.opd_teacher_tp_size}, loss_type={self.loss_type}"
        )

    def score(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
    """
    Score student rollout tokens with the teacher model via SGLang HTTP /generate endpoint.
    
    Args:
        token_ids:      LongTensor [B, T]  -- student-generated token ids
        attention_mask: BoolTensor [B, T]  -- 1 for real tokens, 0 for pad
    
    Returns:
        dict: {"teacher_log_probs": Tensor [B, T]}
    """
    with torch.no_grad():
        # 使用 rank-0 引擎（TP > 1 时内部通信）
        engine = self._engines[0]
        
        # 1. 将 token_ids 转为 SGLang 可接受的格式
        # SGLang /generate 支持 input_ids 直接传入
        input_ids_list = token_ids.cpu().tolist()  # [B, T]
        
        # 2. 构造请求 payload
        payload = {
            "input_ids": input_ids_list,
            "sampling_params": {
                "temperature": 0.0,           # 贪婪解码
                "max_new_tokens": 0,          # ⚠️ 关键：不生成新 token，只计算 logprobs
                "return_logprob": True,       # 要求返回 logprobs
            },
            "return_logprob": True,           # 兼容不同 SGLang 版本
        }
        
        # 3. 通过 HTTP 调用 SGLang 的 /generate 端点
        server_host = engine.server_host
        server_port = engine.server_port
        url = f"http://{server_host}:{server_port}/generate"
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Teacher score HTTP request failed: {e}")
            raise
        
        # 4. 解析 SGLang 返回结果
        # SGLang 返回格式示例:
        # {
        #   "text": ["..."],
        #   "logprobs": [[{"token_id": 123, "logprob": -0.5}, ...], ...],
        #   "input_token_logprobs": [[-0.5, -0.3, ...], ...]  # 不同版本字段不同
        # }
        
        # 尝试多个可能的字段名
        if "input_token_logprobs" in result:
            # SGLang 0.3.x+ 返回格式
            teacher_log_probs_list = result["input_token_logprobs"]  # [B, T]
        elif "logprobs" in result:
            # 兼容旧版本
            teacher_log_probs_list = self._extract_logprobs_from_logprobs_field(
                result["logprobs"], token_ids
            )
        else:
            raise ValueError(f"Unexpected SGLang response format: {result.keys()}")
        
        # 5. 转换为 Tensor
        teacher_log_probs = torch.tensor(
            teacher_log_probs_list, 
            dtype=torch.float32
        )  # [B, T]
        
        return {"teacher_log_probs": teacher_log_probs}


    def _extract_logprobs_from_logprobs_field(
        self, 
        logprobs_list: list, 
        token_ids: torch.Tensor
    ) -> list[list[float]]:
        """
        从 SGLang 的 logprobs 字段提取每个 token 的 logprob。
        
        SGLang logprobs 格式:
        [
            [{"token_id": 101, "logprob": -0.1}, {"token_id": 2023, "logprob": -0.3}, ...],
            ...
        ]
        """
        result = []
        for batch_idx, token_logprobs in enumerate(logprobs_list):
            batch_logprobs = []
            for pos_idx, token_info in enumerate(token_logprobs):
                # 提取当前 token 的 logprob
                if isinstance(token_info, dict):
                    batch_logprobs.append(token_info.get("logprob", 0.0))
                else:
                    # 如果返回的是单个数值
                    batch_logprobs.append(float(token_info))
            result.append(batch_logprobs)
        return result


    def offload(self):
        """Offload teacher weights to CPU to free GPU memory during student training."""
        ray.get([engine.release_memory_occupation.remote() for engine in self._engines])

    def onload(self):
        """Reload teacher weights to GPU before scoring."""
        ray.get([engine.resume_memory_occupation.remote() for engine in self._engines])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_teacher_manager(args, pg) -> TeacherManager:
    """
    Instantiate TeacherManager on the teacher placement group.

    Called from train_opd.py after create_opd_placement_groups().
    The returned handle is a Ray actor ref; call .score.remote() on it.

    Args:
        args: parsed arguments (must include opd_teacher_* fields)
        pg:  teacher placement group
    """
    if pg is None:
        raise ValueError(
            "No 'teacher' placement group found. "
        )

    pg_obj, bundle_indices, _ = pg

    # Pin the TeacherManager head actor to the first bundle of the teacher group.
    # The actor itself spawns per-rank engine sub-actors inside __init__.
    server = TeacherManager.options(
        num_cpus=1,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg_obj,
            placement_group_bundle_index=bundle_indices[0],
        ),
    ).remote(args, pg)

    logger.info(
        f"TeacherManager actor scheduled on teacher placement group "
        f"({args.opd_teacher_tp_size} GPU(s))."
    )
    return server


# ---------------------------------------------------------------------------
# Data merge helper
# ---------------------------------------------------------------------------


def merge_teacher_signal(rollout_data: dict, teacher_output: dict) -> dict:
    """
    Merge teacher scoring output into rollout_data in-place.

    rollout_data must already be materialized (not a Ray object ref).
    The merged dict is passed to actor_model.train() via ray.put().
    """
    rollout_data.update(teacher_output)
    return rollout_data