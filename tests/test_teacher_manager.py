"""
GPU test for TeacherManager.__init__: verifies engine count and that
init actually blocks until the engine is ready.

Run with:
    TEACHER_TEST_MODEL_PATH=/path/to/small/model \
        pytest orbit/tests/test_teacher_manager_init.py -v
"""

import os
from argparse import Namespace

import pytest
import ray
from ray.util.placement_group import placement_group

MODEL_PATH = os.environ.get("TEACHER_TEST_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="module")
def ray_cluster():
    ray.init(num_gpus=1, num_cpus=4)
    yield
    ray.shutdown()


def test_init_creates_one_engine_and_blocks_until_ready(ray_cluster):
    from orbit.ray.teacher import TeacherManager

    pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
    ray.get(pg.ready())
    args = Namespace(opd_teacher_model_path=MODEL_PATH, opd_teacher_tp_size=1)

    manager = TeacherManager.remote(args, (pg, [0], [0]))
    try:
        # If __init__ accidentally became non-blocking, this would hang/timeout
        # instead of returning, since num_engines() can't run until __init__ exits.
        assert ray.get(manager.num_engines.remote(), timeout=60) == 1
    finally:
        ray.kill(manager)
        ray.util.remove_placement_group(pg)