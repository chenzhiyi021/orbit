# Unified post-training subspace experiments

本目录实现 [实施协议 v1](PROTOCOL.md)：同一份 Orbit 源码、同一个 **M6 Bridge/MCore/SGLang 依赖锚点**，分别运行 M6 OPD、Official-CoT SFT 和 math verifier RL。不同协议的数据、初始化、温度、schedule 和预算明确分开；这些实验不是纯 objective 对照。

分支从 `main@3346b7f` 直接增加一条 commit。研究代码在 `orbit/research/posttrain/`；主流程只增加显式 profile 接入点。原入口的默认行为保留。这里不包含历史日志、模型权重、训练数据、银行流水或集群专用路径。

## 当前验收状态

[baseline_parity_report.json](baseline_parity_report.json) / [说明](baseline_parity_report.md) 区分 CPU 回归、GPU 迁移、完整训练与正式评测。

- 已执行 CPU 数学、旧实现 fixture、batch/reducer、SFT shift、Adam/resume/freeze 等回归。`tests/posttrain/fixtures/` 保存 audit 原函数体及其源码哈希，M6 的100个真实 applied LR 值独立保留。
- **GPU fixed-real-batch 迁移、RL TP2/DP2 原生 optimizer 恢复和短程训练尚未验收。** CPU 模拟不能代替它们。
- SFT 与 RL 的数据身份模板保持 `state=unresolved`。未填齐 hash、长度、模板、split 或 ground truth 时，启动前报错；不会静默猜值或过滤样本。
- 没有新增100/300步训练，也没有新的正式分数。RL-r4 的历史成功运行不在本次证据范围内。

## 四个 profile

| profile | 目标 / 数据 | P × G / 更新数 | LR | 初始化 / 并行 |
|---|---|---|---|---|
| `m6_mixed_r2_reference` | Qwen3-1.7B ← Qwen3-4B-Instruct-2507；封存 mixed B_target | 256×1；100事务，每rollout一次update | peak3.2e-4，warmup10，cosine到0.1峰值 | r2/alpha2；name-seeded FP32 Kaiming A、B0；TP1/DP8，micro1×累积32，复制式Adam |
| `sft_omr_cot` | 官方 OpenMathReasoning CoT；主SFT | B256；100事务 | warmup10+cosine；**LoRA LR待有限Dev校准** | r2候选；初始化、并行与长度须明确补录 |
| `sft_teacher_matched` | 与M6共同teacher/prompt的冻结回答；辅助控制 | B256；100事务 | 独立SFT配置，待补录 | 不进入第一批必跑面板 |
| `rl_math_grpo_r4` | math-only verifier；真实 expected_answer | 64×4；300事务，每rollout一次update | **2e-4 constant，warmup0** | r4/alpha4；Xavier normal A、B0；TP2/DP2、动态microbatch、distributed optimizer |

全部使用标准 fused shared-A LoRA：QKV共享A、gate/up共享A，o/down独立。Qwen3-1.7B有112个fused模块；r2独立参数1,835,008，r4预期3,670,016。HF导出的196个逻辑projection重复A键不计为独立参数。运行时 registry 逐项核对base/embedding/head/norm冻结与独立参数数目。

SFT原成功流程不在 M6/RL audit 中。[provenance.json](provenance.json) 记录 czy 的原始 FullFT launcher 来源，未将它的旧分数或 `5e-6 constant` 直接归给新LoRA。主SFT按本轮决定使用warmup+cosine。

## 数学与 mask 契约

**M6**：behavior、teacher prefill、old learner、current learner都T=.7；top-p1/top-k-1。old learner是update前Megatron no_grad重算值。`A=.1*stopgrad(logq-logp_old)`，不center、不whiten；`r=exp(logp-stopgrad(logp))`前向1但有梯度。保留原clip helper语义，scalar loss仍为masked `-A` 平均。TIS、OPSM、ICEPOP、额外KL、entropy、async和多update均关闭。

teacher HTTP 的 `temperature=0,max_new_tokens=0` 只是prefill请求形式。`teacher.py` 和只对teacher子进程启用的 bootstrap 将**实际prefill logprob温度**设为.7，并写入运行证据。teacher/client不能各自随意换温度。

M6 prompt/response/context cap为4096/8192/12288；保留达到8192上限的真实token前缀，不添加假EOS。init_seed22、training/data_seed42、sample_seed=42+canonical_row分别记录。第1次optimizer事务LR=0，moments/step仍前进；第11次达到峰值，第100次applied LR为3.208772090925022e-5。D0/D1方向标为未定义，不拿零更新造SVD basis。

**RL**：最后boxed→mathd或SymPy verifier，0/1 reward；按group_index在DP分发前减组均值，除样本std（G−1）+1e-6。序列advantage广播到所有response token。全对/全错组为0但保留token分母；不会动态重采样。ratio=current learner / cached old learner，clip[.8,1.2]；无KL、entropy或TIS。2e-4是本轮选择；原代码默认5e-6只留在audit字段，注释5e-5不采用。

**SFT**：raw-logit NLL温度恒1；完整response监督，从第一个response预测位置到自然EOS；prompt和padding不监督。已截断回答不补EOS。数据准备显式保留或剥除think标签，**内容不因去标签丢弃**。序列边界使用Orbit的独立packed sequence布局，禁止未经声明跨样本拼接。

共同 reducer 为 `sum(mask*loss) / sum_i max(valid_tokens_i,1)`。零advantage组与零mask样本不同。主训练要求每条至少一个有效token；reference测试保留全mask=0时分母+1的旧行为。梯度先全局token归约、全局L2 clip1，再Adam（.9/.999、eps1e-8、WD0）。

## 环境

研究 profile 使用 [environment.lock](environment.lock) 中的M6锚点：Python3.12、torch2.11.0+cu130、Bridge `9d4c97c6`、MCore `980211ae`、SGLang `da862537`。完整参考包版本来自audit的环境manifest，**不是新GPU运行的认证**。

激活已准备好的M6环境，将本分支放到`PYTHONPATH`首位。研究入口使用这套环境；根目录原有的release `uv.lock`保留给原release环境，不能将其视为本研究栈。不要通过整包merge或一次`uv sync`把zh依赖覆盖进M6。新run启动时会导出全部实际package版本、三套源码的HEAD/diff/文件hash及本branch源码hash。

在工作区外准备 `environment_roots.json`，填实际源码目录：

```json
{
  "megatron-bridge": "/path/to/Megatron-Bridge",
  "megatron-core": "/path/to/Megatron-LM",
  "sglang": "/path/to/sglang"
}
```

以下命令均从仓库根目录、已经获得的计算资源中运行。集群申请、GPU绑定和bank计费属于调度层；这里不硬编码任何用户目录或偷偷提交作业。

## 数据准备与启动

1. 复制并填写 `sft_data_manifest.json` 或 `rl_data_manifest.json` 到实验输出目录。M6需提供 `identity=m6_frozen_b_target` 的同结构manifest及原始B_target；其数据sha必须为 `bfebe57eb0b3bccafcfa7efed981ed946f6af6192db363c661f19c46f78a1e0d`。
2. 填入source revision/subset/split、tokenizer/template hash、长度和EOS政策。SFT额外需要trace作者/字段/think处理与按问题隔离的Train/Dev/Eval证据；RL需要独立权威ground-truth JSONL及sha。
3. RL当前入口读取JSONL，每行 `question, answer, source_question`。独立label文件每行 `question, expected_answer`；缺失/不匹配在预检报错，合法答案`"0"`不会被当成缺失。由原parquet显式转换/审计后填hash，不能使用混合OPD文件里的占位答案。
4. SFT入口读取预分词的 `source_id,prompt_ids,response_ids,ended_with_eos,truncated,response_mask`。使用下面的准备命令；输入每行 `source_id,prompt_messages,response`，由已确认的Official-CoT处理流程提供，不在这里另加正确性筛选或去重。

```bash
PYTHONNOUSERSITE=1 python -m orbit.research.posttrain.prepare_sft \
  --input selected_cot.jsonl --manifest sft_data_manifest.json --output sft_tokens.jsonl
```

准备器保留所有行；超长时按显式政策报错或截断，输出processing审计和完整答案覆盖率。将输出hash填回manifest。SFT还需填写 `sft_decision.example.json` 的LR/初始化/init_seed/并行/microbatch；B256、100步、warmup10/cosine已锁定。一个有限Dev候选不等于已经锁定最优LR。

M6 teacher单独启动在同一分配资源内（M6参考中与学生colocate）。`CUDA_VISIBLE_DEVICES`只传入已分配给作业的设备：

```bash
PYTHONNOUSERSITE=1 python -m orbit.research.posttrain.teacher \
  --model "$TEACHER" --port 31001 --audit "$RUN_ROOT/teacher_temperature.json"
```

先生成计划；不带`--execute`不会启动Ray或训练：

```bash
PYTHONNOUSERSITE=1 python -m orbit.research.posttrain.launch m6_mixed_r2_reference \
  --data-manifest "$M6_MANIFEST" --model "$BASE" --output "$RUN_ROOT/m6_plan" \
  --teacher "$TEACHER" --teacher-url http://127.0.0.1:31001 \
  --teacher-audit "$RUN_ROOT/teacher_temperature.json"
```

短程GPU验收增加 `--execute --qualification-updates 3 --environment-roots environment_roots.json`，并使用全新输出目录。它保留原100/300步scheduler horizon，只提前结束，明确标为qualification。SFT增加 `--sft-decision chosen_sft.json`；RL替换profile和data manifest即可，不需要teacher参数。

**固定真实batch迁移**：从两个独立旧/新执行路径导出同一batch的观测数组，再运行比较器。固定batch须带同一batch/policy hash、温度、mask、raw/scaled advantage、scalar loss、梯度、Adam m/v/step、applied LR、实际D与源码hash。`migration_tolerances.json`预先列出容差；失败后不通过放宽阈值换取通过。这个比较器不冒充完整GPU运行 harness，也不会自行声称旧run已经复现。

```bash
python -m orbit.research.posttrain.qualify compare \
  --reference old_fixed_batch.pt --candidate new_fixed_batch.pt \
  --tolerances examples/posttrain_subspace/migration_tolerances.json --output fixed_batch_comparison.json
```

**当前已停用两道启动门槛**：`--gpu-acceptance` 验收文件和 M6 依赖锚点（Bridge/MCore/SGLang commit、torch、Python）检查。锚点里的 Bridge `9d4c97c6`、SGLang `da862537` 在上游和实验室远程仓库都取不到，GPU 验收工具也尚未实现。`--environment-roots` 仍可传入，环境会写进 run 目录的 `environment.lock`（`verified` 标明是否与锚点一致），但不阻止启动。因此现在的 run 不具备原协议的 GPU 迁移认证；恢复这两道门槛前，结果须注明此点。

## 保存、恢复与第一轮干预

完整保存点：M6 0/10/20/40/60/80/100；RL 0/20/50/100/150/200/250/300；SFT 0/20/40/60/80/100。M6另保留原15个轻量几何点。0点在`checkpoints/initial_state/`；其后完整adapter目录仍用Orbit零基rollout编号，例如20事务对应`iter_0000019/adapter`。

每个完整checkpoint含effective adapter、FP32 master、每rank optimizer/moments、scheduler、Python/NumPy/Torch/CUDA/TP RNG、数据身份/游标和policy version。distributed optimizer的外部tensor状态一并保存。`POSTTRAIN_COMPLETE.json`在所有rank和数据游标写齐后生成；它列出实际文件hash。adapter-only导出可用于推理，**不能充当完整resume**。

从共同step20分别启动Free-resume与Freeze-A-resume，使用相同`--resume`、`--data-cursor`及总100步profile；后者额外加`--freeze-factor right`。完全相同A0从头冻结也用同一profile，fresh启动加该flag。冻结发生在optimizer复制/归约梯度后、clip/step前，冻结参数及moments逐项保持不变，另一因子的状态不重置；前向输入梯度继续传播。`left`是Freeze-B，留作后续明确对照。

[panel.json](panel.json) 只列A阶段有效基线及B阶段最小M6面板，不自动展开组合网格。SFT teacher-matched、恒定LR控制、更多rank/seed和Haar来源仍按协议分阶段推进，未作为已验收recipe开放。

后验reset-A和初始空间投影只作解释：

```bash
python -m orbit.research.posttrain.interventions \
  --initial initial_state/posttrain_state_rank0.pt \
  --endpoint iter_0000099/adapter/posttrain_state_rank0.pt \
  --coordinates effective --output interpretation
```

分析用真实dense D核查固定A禁区泄漏，提供完整线性forward重构测试；不会用leading overlap替代真实D。`endpoint_basis`拒绝从rank2终点填零空间凑出top16/top256。

## 正式评测与产物

M6/SFT只在100、RL只在300的正式终点评测。训练checkpoint保存和全量评测独立。填写 `eval_manifest.example.json`，固定数据hash、prompt模板、seed、采样参数、checkpoint格式/merge政策，以及现有数学评测器的完整argv和源码hash。当前没有猜测未确认的正式task manifest。

```bash
python -m orbit.research.posttrain.evaluate --run "$RUN" \
  --manifest math_eval_manifest.json --completed-updates 100 \
  --checkpoint "$ADAPTER" --output "$EVAL_OUTPUT"
```

评测器必须输出`prompt_results.jsonl`，逐题包含generation、judge输出、score、response_tokens和truncated，校验后才写`COMPLETE.json`。HF adapter保留shared-A导出校验；是否merge及其GPU forward容差属于所选eval manifest。SFT同源held-out NLL、M6固定bank KL和RL reward/有效组比例是诊断，不替代正式task分数。

新run导出 `requested_protocol.json`、hook之后的`effective_protocol.json`、真实argv、环境lock、源码hash、每rank参数registry、逐步applied LR/grad norm、逐prompt rollout、长度/截断和teacher请求计时。原始壁钟与scheduler/GPU-hours分别记账；teacher请求延迟之和不是teacher GPU占用时间。

## 实现与回归入口

| 模块 | 责任 |
|---|---|
| `protocol.py` / profiles | 原协议字段、拒绝未命名科学变更、真实LR时序 |
| `data.py`, `prepare_sft.py`, `batch.py` | 数据身份、权威label、offline CoT、response边界、共同观测schema |
| `teacher.py`, `rollout.py` | teacher prefill温度、fresh采样、verifier、版本审计 |
| `objectives.py` | 生产M6/RL token loss；共用Orbit token reducer；SFT T=1 |
| `runtime.py`, `factors.py`, `constraints.py` | native初始化、registry、parity gate、精确冻结、endpoint核查 |
| `checkpoint.py` | FP32/有效参数、完整optimizer/RNG、严格恢复与完成标记 |
| `launch.py`, `qualify.py`, `evaluate.py` | 可审阅计划、独立验收、endpoint评测 |

```bash
PYTHONPATH=. python -m orbit.research.posttrain.qualify cpu \
  --output /path/to/new/cpu_parity_report.json
```

旧M6入口映射与函数来源见[provenance](provenance.json)、[fixture来源](../../tests/posttrain/fixtures/provenance.json)。完整历史run仍在原工作区；新branch未覆盖它。保留旧函数体作为fixture，新实现的测试直接调用生产函数，并读取实际core的response切片/reducer，不用另写的理想目标替代生产路径。
