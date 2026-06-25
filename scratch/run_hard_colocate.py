"""
Two-engine colocate stress test: start a first SGLang-backed TeacherManager
("simulated rollout", mem_fraction_static=0.60) on a GPU, then start a
second one ("teacher", mem_fraction_static=0.1) on the SAME GPU/bundle.

Purpose: isolate whether two SGLang instances sharing one GPU hang during
init regardless of mem_fraction_static value -- i.e. whether the colocate
hang seen in train_opd.py is really a memory-fraction conflict, or
something else (port collision, CUDA context contention, etc).

Run with:
    bash scratch/run_hard_colocate.sh
"""

import sys
import time

import ray
from ray.util.placement_group import placement_group

from orbit.utils.arguments import parse_args


def main():
    args1 = parse_args()
    args1.colocate = True
    args1.sglang_mem_fraction_static = 0.60

    # Re-parse for an independent copy rather than copy.copy(args1), so any
    # accidental shared mutable state (e.g. a list default) can't leak
    # between the two and confound the test.
    args2 = parse_args()
    args2.colocate = True
    args2.sglang_mem_fraction_static = 0.1

    print("=== Debug: Key arguments ===")
    print(f"hf_checkpoint: {args1.hf_checkpoint}")
    print(f"opd_teacher_model_path: {args1.opd_teacher_model_path}")
    print(f"opd_teacher_tp_size: {args1.opd_teacher_tp_size}")
    print(f"args1.sglang_mem_fraction_static: {args1.sglang_mem_fraction_static}")
    print(f"args2.sglang_mem_fraction_static: {args2.sglang_mem_fraction_static}")
    print("==============================")

    ray.init(num_gpus=1, ignore_reinit_error=True)
    pg = placement_group([{"GPU": 1, "CPU": 4}], strategy="PACK")
    ray.get(pg.ready())
    print(f"Placement group ready: {pg}")

    from orbit.ray.teacher import TeacherManager

    print("\nStarting first (simulated rollout) engine, mem_fraction_static=0.60...")
    t0 = time.time()
    manager1 = TeacherManager.remote(args1, (pg, [0], [0]))
    n1 = ray.get(manager1.num_engines.remote(), timeout=120)
    print(f"First engine ready in {time.time()-t0:.1f}s, num_engines={n1}")

    print("\nStarting second (teacher) engine on the SAME GPU/bundle, mem_fraction_static=0.1...")
    print("(this is where train_opd.py was observed to hang)")
    t1 = time.time()
    manager2 = TeacherManager.remote(args2, (pg, [0], [0]))
    try:
        # Explicit timeout so this test gives a definitive answer (success
        # or a clear GetTimeoutError) instead of hanging indefinitely like
        # the original train_opd.py run did.
        n2 = ray.get(manager2.num_engines.remote(), timeout=60)
        print(f"Second engine ready in {time.time()-t1:.1f}s, num_engines={n2}")
        print("\n=== RESULT: both engines started successfully on one GPU. ===")
    except ray.exceptions.GetTimeoutError:
        print(f"\n=== RESULT: second engine did NOT become ready within 60s. ===")
        print("This confirms two SGLang instances colliding on one GPU during "
              "init, independent of train_opd.py's own code -- the issue is "
              "in sharing a GPU between two SGLang servers, not in OPD-specific "
              "logic. Next step: inspect ray worker logs for manager2's PID "
              "to see what it's actually blocked on (likely memory allocation "
              "or a CUDA-level resource, not a Python-level deadlock).")
        raise


if __name__ == "__main__":
    main()