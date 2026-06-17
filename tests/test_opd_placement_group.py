"""
Tests for create_opd_placement_groups in orbit/ray/placement_group.py.

These tests only verify GPU-index bookkeeping (offsets, slicing) and do
NOT require real GPUs. We monkeypatch _create_placement_group so no Ray
placement group is actually created.

Run with:
    pytest tests/test_opd_placement_group.py -v
"""

from argparse import Namespace
from unittest.mock import patch

import pytest


def make_args(**overrides):
    """Minimal args Namespace covering every field create_opd_placement_groups reads."""
    defaults = dict(
        use_critic=False,
        colocate=False,
        debug_train_only=False,
        debug_rollout_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        opd_teacher_num_gpus=1,
        rollout_num_gpus=1,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def fake_create_placement_group(num_gpus):
    """Stand-in for _create_placement_group: returns deterministic fake handles.

    pg is a sentinel string (not a real Ray object) so we can assert identity
    across the returned dict without needing a Ray cluster.
    bundle_indices and gpu_ids are simple ranges, which makes the offset
    arithmetic easy to verify in assertions.
    """
    pg = f"fake_pg_{num_gpus}"
    bundle_indices = list(range(num_gpus))
    gpu_ids = list(range(num_gpus))
    return pg, bundle_indices, gpu_ids


@pytest.fixture(autouse=True)
def patch_create_placement_group():
    with patch(
        "orbit.ray.placement_group._create_placement_group",
        side_effect=fake_create_placement_group,
    ):
        yield


class TestNormalPath:
    def test_gpu_layout_actor_teacher_rollout(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(actor_num_gpus_per_node=2, opd_teacher_num_gpus=1, rollout_num_gpus=1)
        pgs = create_opd_placement_groups(args)

        # total = 2 (actor) + 1 (teacher) + 1 (rollout) = 4
        actor_pg, actor_bundles, actor_gpus = pgs["actor"]
        teacher_pg, teacher_bundles, teacher_gpus = pgs["teacher"]
        rollout_pg, rollout_bundles, rollout_gpus = pgs["rollout"]

        assert actor_pg == teacher_pg == rollout_pg == "fake_pg_4"

        assert actor_bundles == [0, 1]
        assert teacher_bundles == [2]
        assert rollout_bundles == [3]

        assert actor_gpus == [0, 1]
        assert teacher_gpus == [2]
        assert rollout_gpus == [3]

    def test_no_overlap_no_gap_across_segments(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(actor_num_gpus_per_node=4, opd_teacher_num_gpus=2, rollout_num_gpus=3)
        pgs = create_opd_placement_groups(args)

        all_indices = (
            pgs["actor"][1] + pgs["teacher"][1] + pgs["rollout"][1]
        )
        assert sorted(all_indices) == list(range(9))  # 4 + 2 + 3, contiguous, no dup

    def test_multi_node_actor(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(actor_num_nodes=2, actor_num_gpus_per_node=4, opd_teacher_num_gpus=1, rollout_num_gpus=1)
        pgs = create_opd_placement_groups(args)

        assert len(pgs["actor"][1]) == 8  # 2 nodes * 4 gpus
        assert pgs["teacher"][1] == [8]
        assert pgs["rollout"][1] == [9]


class TestGuardrails:
    def test_critic_raises(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(use_critic=True)
        with pytest.raises(NotImplementedError, match="Critic"):
            create_opd_placement_groups(args)

    def test_colocate_raises(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(colocate=True)
        with pytest.raises(NotImplementedError, match="Colocate"):
            create_opd_placement_groups(args)


class TestDebugModes:
    def test_debug_train_only_skips_teacher_and_rollout(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(debug_train_only=True, actor_num_gpus_per_node=2)
        pgs = create_opd_placement_groups(args)

        assert pgs["teacher"] is None
        assert pgs["rollout"] is None
        actor_pg, actor_bundles, actor_gpus = pgs["actor"]
        assert actor_bundles == [0, 1]

    def test_debug_rollout_only_skips_actor_and_teacher(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(debug_rollout_only=True, rollout_num_gpus=3)
        pgs = create_opd_placement_groups(args)

        assert pgs["actor"] is None
        assert pgs["teacher"] is None
        rollout_pg, rollout_bundles, rollout_gpus = pgs["rollout"]
        assert rollout_bundles == [0, 1, 2]