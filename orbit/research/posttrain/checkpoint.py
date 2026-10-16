"""Strict rank-local research state alongside the ordinary native adapter export."""

from __future__ import annotations
import json
import random
from pathlib import Path
import torch

from .constraints import _iter_optimizer_leaves, _leaf_parameter_pairs
from .protocol import fingerprint


def capture_rng():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
        from megatron.core import tensor_parallel

        state["tracker"] = {key: value.get_state().clone() if isinstance(value, torch.Generator) else value.clone() for key, value in tensor_parallel.get_cuda_rng_tracker().get_states().items()}
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
        from megatron.core import tensor_parallel

        tensor_parallel.get_cuda_rng_tracker().set_states(state["tracker"])


def _master_pairs(model, optimizer):
    names = {id(p): f"chunk{i}.{name}" for i, chunk in enumerate(model) for name, p in chunk.named_parameters()}
    pairs = {}
    for leaf in _iter_optimizer_leaves(optimizer):
        for parameter, master in _leaf_parameter_pairs(leaf):
            if id(parameter) in names and parameter.requires_grad:
                pairs[names[id(parameter)]] = (parameter, master)
    return pairs


def save_state(path, model, optimizer, scheduler, *, completed_updates, protocol, data_hash, rank=0):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    pairs = _master_pairs(model, optimizer)
    state = {
        "schema_version": 1,
        "rank": rank,
        "completed_updates": completed_updates,
        "freeze_factor": getattr(getattr(optimizer, "_posttrain_context", (None,))[0], "posttrain_freeze_factor", "none"),
        "protocol_hash": fingerprint(protocol),
        "data_hash": data_hash,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "master_parameters": {n: p.detach().cpu().clone() for n, (_, p) in pairs.items()},
        "effective_parameters": {f"chunk{i}.{n}": p.detach().cpu().clone() for i, chunk in enumerate(model) for n, p in chunk.named_parameters() if p.requires_grad},
        "rng": capture_rng(),
        "next_prompt_index": completed_updates * protocol["batch"]["prompts"],
        "policy_version": completed_updates + 1,
        "adapter_scale": protocol["adapter"]["alpha"] / protocol["adapter"]["rank"] if protocol["adapter"]["method"] == "lora" else None,
    }
    # DistributedOptimizer may keep FP32 moments in an external collective file.
    external = any(hasattr(leaf, "model_param_group_index_map") for leaf in _iter_optimizer_leaves(optimizer))
    state["external_optimizer"] = external
    if external:
        if not callable(getattr(optimizer, "save_parameter_state", None)):
            raise RuntimeError("Distributed optimizer does not expose complete parameter-state save")
        optimizer.save_parameter_state(str(path / f"posttrain_optimizer_rank{rank}.pt"))
    temp = path / f"posttrain_state_rank{rank}.pt.partial"
    torch.save(state, temp)
    temp.replace(path / f"posttrain_state_rank{rank}.pt")
    return state


def load_state(path, model, optimizer, scheduler, *, protocol, data_hash, rank=0, freeze_factor="none"):
    file = Path(path) / f"posttrain_state_rank{rank}.pt"
    if not file.is_file():
        raise ValueError(f"Full resume state is missing: {file}; adapter-only restart is forbidden")
    state = torch.load(file, map_location="cpu", weights_only=False)
    if state["protocol_hash"] != fingerprint(protocol) or state["data_hash"] != data_hash or state["rank"] != rank:
        raise ValueError("Resume protocol/data/rank identity mismatch")
    if state.get("freeze_factor", "none") not in {"none", freeze_factor}:
        raise ValueError("Resuming a frozen run cannot silently unfreeze or switch factor")
    pairs = _master_pairs(model, optimizer)
    if set(pairs) != set(state["master_parameters"]):
        raise ValueError("Optimizer parameter layout changed across resume")
    optimizer.load_state_dict(state["optimizer"])
    if state["external_optimizer"]:
        optimizer.load_parameter_state(str(Path(path) / f"posttrain_optimizer_rank{rank}.pt"))
    scheduler.load_state_dict(state["scheduler"])
    with torch.no_grad():
        for i, chunk in enumerate(model):
            for name, parameter in chunk.named_parameters():
                key = f"chunk{i}.{name}"
                if parameter.requires_grad:
                    parameter.copy_(state["effective_parameters"][key].to(parameter))
        for name, (parameter, master) in pairs.items():
            expected = state["master_parameters"][name]
            master.copy_(expected.to(master.device))
            # TP/DP sharded masters are copied to model by Megatron itself.
            if parameter.shape == master.shape:
                parameter.copy_(master.to(parameter.dtype))
        for leaf in _iter_optimizer_leaves(optimizer):
            copy_back = getattr(leaf, "_copy_main_params_to_model_params", None)
            if callable(copy_back):
                copy_back()
    return state  # RNG is restored immediately before the first training batch.


def mark_complete(args, rollout_id):
    from .protocol import digest

    path = Path(args.save) / f"iter_{rollout_id:07d}" / "adapter"
    ranks = args.posttrain_protocol["parallel"]["tp"] * args.posttrain_protocol["parallel"]["dp"]
    files = [path / f"posttrain_state_rank{rank}.pt" for rank in range(ranks)]
    cursor = Path(args.save) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
    files.append(cursor)
    if any(not p.is_file() for p in files):
        raise RuntimeError("Checkpoint is missing a rank state or the data cursor")
    if args.posttrain_protocol["optimizer"]["distributed"] and len(list(path.glob("posttrain_optimizer_rank*.pt"))) < args.posttrain_protocol["parallel"]["tp"]:
        raise RuntimeError("Distributed optimizer parameter state is incomplete")
    files = [p for p in path.iterdir() if p.is_file() and not p.name.endswith(".partial") and not p.name.startswith("POSTTRAIN_COMPLETE")] + [cursor]
    value = {"completed_updates": rollout_id + 1, "protocol_hash": fingerprint(args.posttrain_protocol), "data_cursor": str(cursor.resolve()), "files": {str(p.resolve()): digest(p) for p in files}}
    temporary = path / "POSTTRAIN_COMPLETE.json.partial"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path / "POSTTRAIN_COMPLETE.json")
