import sys

import ray
from ray.util.placement_group import placement_group

from orbit.utils.arguments import parse_args

ray.init(num_gpus=1)

pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
ray.get(pg.ready())

print(f"Placement group created with bundles: {pg}")

sys.argv += [
    "--hf-checkpoint", "/mnt/L202500431/model/qwen2.5-0.5b-instruct",
    "--rollout-batch-size", "1",
]

args = parse_args()
args.opd_teacher_model_path = "/mnt/L202500431/model/qwen2.5-0.5b-instruct"
args.opd_teacher_tp_size = 1

from orbit.ray.teacher import TeacherManager
manager = TeacherManager.remote(args, (pg, [0], [0]))

ray.get(manager.__ray_ready__.remote())
print("TeacherManager initialized successfully!")