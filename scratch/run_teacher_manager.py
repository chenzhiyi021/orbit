import sys

import ray
import torch
from ray.util.placement_group import placement_group

from orbit.utils.arguments import parse_args


def setup(args):
    """One-time setup shared by both tests: placement group + TeacherManager.

    Both test_score_single() and test_merge_teacher_signal_structure() need
    a live TeacherManager. Building it once here (rather than each test
    calling ray.init()/placement_group() itself) avoids the "ray.init called
    twice" hang and avoids paying the multi-second model-load cost twice.
    """
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

    return manager


def test_score_single(manager, args):
    """Exercise score() with a single real prompt and sanity-check the result."""
    from transformers import AutoTokenizer

    print("\n=== Testing score() (single prompt) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.opd_teacher_model_path)
    text = "The capital of France is"
    encoded = tokenizer(text)["input_ids"]
    print(f"Prompt: {text!r}")
    print(f"Token ids: {encoded}")

    token_ids = torch.tensor([encoded])
    attention_mask = torch.ones_like(token_ids)

    result = ray.get(manager.score.remote(token_ids, attention_mask))

    print(f"score() returned keys: {list(result.keys())}")
    if "teacher_log_probs" not in result:
        print(f"WARNING: unexpected result format: {result}")
        return

    log_probs = result["teacher_log_probs"]
    print(f"teacher_log_probs shape: {log_probs.shape}")
    print(f"teacher_log_probs values: {log_probs}")
    print(f"teacher_log_probs dtype: {log_probs.dtype}")

    # Position 0 is expected to be nan (no preceding context for the first
    # token) -- check finiteness/sign only on the rest, since asserting it
    # on position 0 would always (correctly) fail.
    rest = log_probs[1:] if log_probs.dim() == 1 else log_probs[:, 1:]
    all_finite = torch.isfinite(rest).all().item()
    all_non_positive = (rest <= 0).all().item()
    print(f"position 0 is nan (expected): {torch.isnan(log_probs[0, 0]).item()}")
    print(f"All finite (excluding position 0): {all_finite}")
    print(f"All <= 0 (excluding position 0): {all_non_positive}")

    if not all_finite or not all_non_positive:
        print("WARNING: log-probs look invalid beyond position 0 -- check "
              "response parsing in TeacherManager.score().")

    print("=== score() single-prompt test complete ===")


def test_merge_teacher_signal_structure(manager, args):
    """Verify the shape/structure train_student() assumes for
    rollout_data["teacher_log_probs"], using merge_teacher_signal() the
    same way train_opd.py does -- not just score()'s raw return value.
    """
    from orbit.ray.teacher import merge_teacher_signal
    from transformers import AutoTokenizer

    print("\n=== Testing merge_teacher_signal() + multi-sequence batch shape ===")

    tokenizer = AutoTokenizer.from_pretrained(args.opd_teacher_model_path)

    # Two prompts of DIFFERENT lengths -- this is the case that matters for
    # train_student()'s masking logic, since rollout_data in real training
    # holds a *batch* of sequences, not a single one, and they are not
    # guaranteed to be the same length before padding.
    prompts = ["The capital of France is", "Hi"]
    encoded_list = [tokenizer(p)["input_ids"] for p in prompts]
    print(f"Prompt lengths: {[len(e) for e in encoded_list]}")

    # score() as currently written takes a single [B, T] tensor, which
    # requires equal-length sequences. To test unequal lengths we call it
    # once per sequence here -- this also mirrors how score() would need to
    # be extended if it doesn't already handle ragged batches internally.
    per_sample_log_probs = []
    for encoded in encoded_list:
        token_ids = torch.tensor([encoded])
        attention_mask = torch.ones_like(token_ids)
        result = ray.get(manager.score.remote(token_ids, attention_mask))
        log_probs = result["teacher_log_probs"]
        print(f"  len={len(encoded)} -> teacher_log_probs shape={log_probs.shape}, "
              f"first_val={log_probs[0].item()}")
        # score() returns a [1, T] or [T] tensor for a single sequence --
        # squeeze batch dim if present so per_sample_log_probs holds 1D
        # tensors, matching what train_student() iterates over.
        per_sample_log_probs.append(log_probs.squeeze(0) if log_probs.dim() > 1 else log_probs)

    # Build a minimal rollout_data dict the way train_opd.py would, then run
    # it through merge_teacher_signal() to confirm the key name and shape
    # train_student() will actually see.
    rollout_data = {
        "tokens": encoded_list,
        "loss_masks": [[1] * len(e) for e in encoded_list],
    }
    teacher_output = {"teacher_log_probs": per_sample_log_probs}
    merged = merge_teacher_signal(rollout_data, teacher_output)

    print(f"\nmerged keys: {list(merged.keys())}")
    print(f"merged['teacher_log_probs'] type: {type(merged['teacher_log_probs'])}")
    print(f"merged['teacher_log_probs'] length: {len(merged['teacher_log_probs'])}")
    for i, lp in enumerate(merged["teacher_log_probs"]):
        print(f"  seq {i}: type={type(lp)}, shape={getattr(lp, 'shape', 'N/A')}, "
              f"first_val={lp[0].item() if hasattr(lp, '__getitem__') else lp}, "
              f"is_first_nan={torch.isnan(lp[0]).item() if hasattr(lp, '__getitem__') else 'N/A'}")

    # This is exactly the assumption train_student()'s masking code makes:
    # rollout_data["teacher_log_probs"] is a list, one 1D tensor per
    # sequence, and the first element of each is nan. Confirm or refute it
    # here before trusting that masking code.
    print("\n--- Assumption check for train_student() masking logic ---")
    is_list = isinstance(merged["teacher_log_probs"], list)
    print(f"Is a list (not a single stacked tensor): {is_list}")
    if is_list:
        all_first_nan = all(
            torch.isnan(lp[0]).item() for lp in merged["teacher_log_probs"] if len(lp) > 0
        )
        print(f"All sequences have nan at position 0: {all_first_nan}")
        all_indexable = all(hasattr(lp, "__setitem__") for lp in merged["teacher_log_probs"])
        print(f"All sequences support item assignment (lp[0] = 0.0 will work): {all_indexable}")

    print("=== merge_teacher_signal structure test complete ===")

def test_score_batch(manager, args):
    """Test whether score() really supports batched inputs."""

    from transformers import AutoTokenizer
    import torch

    print("\n=== Testing score() with batched inputs ===")

    tokenizer = AutoTokenizer.from_pretrained(
        args.opd_teacher_model_path
    )

    prompts = [
        "The capital of France is",
        "Hi",
    ]

    enc = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )

    token_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print("prompts:")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p}")

    print("\ninput_ids shape:", token_ids.shape)
    print(token_ids)

    print("\nattention_mask shape:", attention_mask.shape)
    print(attention_mask)

    result = ray.get(
        manager.score.remote(
            token_ids,
            attention_mask,
        )
    )

    print("\nreturned keys:", list(result.keys()))

    if "teacher_log_probs" not in result:
        print("teacher_log_probs missing!")
        print(result)
        return

    log_probs = result["teacher_log_probs"]

    print("\nteacher_log_probs type:")
    print(type(log_probs))

    #
    # Case 1
    # Tensor[B,T]
    #
    if torch.is_tensor(log_probs):

        print("Tensor output detected")

        print("shape:", log_probs.shape)
        print(log_probs)

        assert log_probs.ndim == 2, (
            f"Expected [B,T], got {log_probs.shape}"
        )

        assert log_probs.shape[0] == len(prompts), (
            f"Expected batch={len(prompts)}, "
            f"got {log_probs.shape[0]}"
        )

        print("\nPer sample:")

        for i in range(log_probs.shape[0]):

            lp = log_probs[i]

            print(
                f"sample {i}: "
                f"first_is_nan={torch.isnan(lp[0]).item()}, "
                f"finite_after0={torch.isfinite(lp[1:]).all().item()}"
            )

    #
    # Case 2
    # List[Tensor]
    #
    elif isinstance(log_probs, list):

        print("List output detected")

        print("batch size:", len(log_probs))

        assert len(log_probs) == len(prompts)

        for i, lp in enumerate(log_probs):

            print(
                f"\nsample {i}"
            )

            print("type:", type(lp))

            if torch.is_tensor(lp):

                print("shape:", lp.shape)

                print(lp)

                print(
                    "first_is_nan:",
                    torch.isnan(lp[0]).item()
                )

                if len(lp) > 1:
                    print(
                        "finite_after0:",
                        torch.isfinite(lp[1:]).all().item()
                    )

            else:

                print("unexpected element type")
                print(lp)

    else:

        print("\nUnexpected output type")
        print(type(log_probs))
        print(log_probs)

    print("\n=== score() batch test complete ===")

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

    # Single ray.init() for the whole script -- both tests share one
    # TeacherManager built by setup() below, instead of each test calling
    # ray.init()/placement_group() and racing to create a second cluster.
    ray.init(num_gpus=1, ignore_reinit_error=True)

    manager = setup(args)

    test_score_single(manager, args)
    test_score_batch(manager, args)
    test_merge_teacher_signal_structure(manager, args)


if __name__ == "__main__":
    main()