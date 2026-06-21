import ray
from ray.util.placement_group import placement_group
from argparse import Namespace

ray.init(num_gpus=1)

pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
ray.get(pg.ready())

print(f"Placement group created with bundles: {pg}")

args = Namespace(
    opd_teacher_model_path="/mnt/L202500431/model/qwen2.5-0.5b-instruct",
    opd_teacher_tp_size=1,
    hf_checkpoint="/mnt/L202500431/model/qwen2.5-0.5b-instruct",
    num_gpus_per_node = 1,
    sglang_dp_size = 1,
    env_report = "none",
)

from orbit.ray.teacher import TeacherManager
manager = TeacherManager.remote(args, (pg, [0], [0]))

ray.get(manager.__ray_ready__.remote())
print("TeacherManager initialized successfully!")