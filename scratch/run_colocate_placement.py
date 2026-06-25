import ray
from orbit.ray.placement_group import create_opd_placement_groups
from argparse import Namespace

ray.init(num_gpus=1, ignore_reinit_error=True)

args = Namespace(
    use_critic=False,
    colocate=True,
    debug_train_only=False,
    debug_rollout_only=False,
    actor_num_nodes=1,
    actor_num_gpus_per_node=1,
    opd_teacher_num_gpus=1,
    rollout_num_gpus=1,
)
pgs = create_opd_placement_groups(args)
print(pgs["actor"])
print(pgs["teacher"])
print(pgs["rollout"])