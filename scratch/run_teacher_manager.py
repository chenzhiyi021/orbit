import sys

import ray
import torch
from ray.util.placement_group import placement_group

from orbit.utils.arguments import parse_args


def main():
    args = parse_args()

    print("=== Debug: Key arguments ===")
    print(f"hf_checkpoint: {args.hf_checkpoint}")
    print(f"rollout_batch_size: {args.rollout_batch_size}")
    print(f"num_rollout: {args.num_rollout}")
    print(f"opd_teacher_model_path: {args.opd_teacher_model_path}")
    print(f"opd_teacher_tp_size: {args.opd_teacher_tp_size}")
    print(f"hidden_size: {getattr(args, 'hidden_size', 'NOT SET')}")
    print(f"num_layers: {getattr(args, 'num_layers', 'NOT SET')}")
    print("==============================")

    ray.init(num_gpus=1)

    pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
    ray.get(pg.ready())
    print(f"Placement group created with bundles: {pg}")

    from orbit.ray.teacher import TeacherManager
    manager = TeacherManager.remote(args, (pg, [0], [0]))

    # NOTE: __ray_ready__ is not a real, documented method we can rely on --
    # use num_engines() (defined on TeacherManager itself) instead, both to
    # block until __init__ finishes and to get a meaningful real value back.
    num_engines = ray.get(manager.num_engines.remote())
    print(f"TeacherManager initialized successfully! num_engines={num_engines}")

    # --- Now actually exercise score() ---
    #
    # We need real, in-vocab token ids for the loaded tokenizer/model, or
    # SGLang's /generate endpoint may reject the request or behave oddly.
    # Using the tokenizer tied to the same checkpoint guarantees the ids
    # are valid for this model, which is more reliable than hand-picking
    # small integers (e.g. [1, 2, 3]) that may map to special/reserved
    # tokens depending on the tokenizer.
    from transformers import AutoTokenizer

    print("\n=== Testing score() ===")
    tokenizer = AutoTokenizer.from_pretrained(args.opd_teacher_model_path)
    text = "The capital of France is"
    encoded = tokenizer(text)["input_ids"]
    print(f"Prompt: {text!r}")
    print(f"Token ids: {encoded}")

    token_ids = torch.tensor([encoded])
    attention_mask = torch.ones_like(token_ids)

    result = ray.get(manager.score.remote(token_ids, attention_mask))

    print(f"score() returned keys: {list(result.keys())}")
    if "teacher_log_probs" in result:
        log_probs = result["teacher_log_probs"]
        print(f"teacher_log_probs shape: {log_probs.shape}")
        print(f"teacher_log_probs values: {log_probs}")
        print(f"teacher_log_probs dtype: {log_probs.dtype}")

        # Sanity checks -- log-probs of real tokens should be finite and
        # <= 0 (log of a probability in (0, 1]). If these fail, either the
        # response parsing is wrong or the /generate call isn't doing what
        # we assume (see NOTE in TeacherManager.score docstring).
        all_finite = torch.isfinite(log_probs).all().item()
        all_non_positive = (log_probs <= 0).all().item()
        print(f"All finite: {all_finite}")
        print(f"All <= 0 (valid log-probs): {all_non_positive}")

        if not all_finite or not all_non_positive:
            print("WARNING: log-probs look invalid -- check response parsing "
                  "in TeacherManager.score() / max_new_tokens=0 assumption.")
    else:
        print(f"WARNING: unexpected result format: {result}")

    print("\n=== score() test complete ===")


if __name__ == "__main__":
    main()