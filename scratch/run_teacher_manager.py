import sys
import ray
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
    
    ray.get(manager.__ray_ready__.remote())
    print("TeacherManager initialized successfully!")

if __name__ == "__main__":
    main()