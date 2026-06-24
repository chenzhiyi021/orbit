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


class TestColocate:
    """colocate=True means all three roles share every bundle -- there is
    no partial-colocate mode (e.g. actor+teacher sharing while rollout is
    separate). This is the intended single-GPU debugging configuration.
    """

    def test_single_gpu_all_roles_share_same_bundle(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(colocate=True, actor_num_nodes=1, actor_num_gpus_per_node=1)
        pgs = create_opd_placement_groups(args)

        # total GPUs allocated = actor_gpus only (1), not actor+teacher+rollout
        actor_pg, actor_bundles, actor_gpus = pgs["actor"]
        teacher_pg, teacher_bundles, teacher_gpus = pgs["teacher"]
        rollout_pg, rollout_bundles, rollout_gpus = pgs["rollout"]

        assert actor_pg == teacher_pg == rollout_pg == "fake_pg_1"
        assert actor_bundles == teacher_bundles == rollout_bundles == [0]
        assert actor_gpus == teacher_gpus == rollout_gpus == [0]

    def test_multi_gpu_all_roles_share_full_set_not_a_slice(self):
        """With actor_gpus=3, all three roles should each get bundles
        [0, 1, 2] -- the FULL set, not e.g. actor getting [0,1,2] while
        teacher/rollout get an empty slice from a None:None index.
        """
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(colocate=True, actor_num_nodes=1, actor_num_gpus_per_node=3)
        pgs = create_opd_placement_groups(args)

        actor_bundles = pgs["actor"][1]
        teacher_bundles = pgs["teacher"][1]
        rollout_bundles = pgs["rollout"][1]

        assert actor_bundles == [0, 1, 2]
        assert teacher_bundles == [0, 1, 2]
        assert rollout_bundles == [0, 1, 2]

    def test_colocate_ignores_teacher_and_rollout_gpu_counts(self):
        """opd_teacher_num_gpus and rollout_num_gpus must NOT add extra GPUs
        on top of actor_gpus in colocate mode -- only actor_gpus sizes the
        placement group.
        """
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=2,
            opd_teacher_num_gpus=5,   # should be ignored
            rollout_num_gpus=7,        # should be ignored
        )
        pgs = create_opd_placement_groups(args)

        # If these counts were NOT ignored, total would be 2+5+7=14.
        assert pgs["actor"][0] == "fake_pg_2"


class TestGuardrails:
    def test_critic_raises(self):
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(use_critic=True)
        with pytest.raises(NotImplementedError, match="Critic"):
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

    def test_debug_train_only_takes_priority_over_colocate(self):
        """If both debug_train_only and colocate are set, debug_train_only
        wins -- this documents current branch ordering (debug flags are
        checked before colocate), it is not asserting that combination is
        a sanctioned use case.
        """
        from orbit.ray.placement_group import create_opd_placement_groups

        args = make_args(debug_train_only=True, colocate=True, actor_num_gpus_per_node=2)
        pgs = create_opd_placement_groups(args)

        assert pgs["teacher"] is None
        assert pgs["rollout"] is None