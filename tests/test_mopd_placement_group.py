"""
Tests for MOPD extensions in orbit/ray/placement_group.py:
  - _parse_mopd_teacher_configs
  - create_opd_placement_groups (teachers dict)

No real GPUs or Ray cluster needed: _create_placement_group is monkeypatched.

Run with:
    pytest tests/test_mopd_placement_group.py -v
"""

import json
from argparse import Namespace
from unittest.mock import patch

import pytest


def make_args(**overrides):
    defaults = dict(
        use_critic=False,
        colocate=False,
        debug_train_only=False,
        debug_rollout_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        opd_teacher_num_gpus=1,
        opd_teacher_tp_size=1,
        rollout_num_gpus=1,
        mopd_teacher_configs=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def fake_pg(num_gpus):
    return f"fake_pg_{num_gpus}", list(range(num_gpus)), list(range(num_gpus))


@pytest.fixture(autouse=True)
def patch_pg():
    with patch("orbit.ray.placement_group._create_placement_group", side_effect=fake_pg):
        yield


# ── _parse_mopd_teacher_configs ───────────────────────────────────────────────

class TestParseMopdTeacherConfigs:
    def _parse(self, **kwargs):
        from orbit.ray.placement_group import _parse_mopd_teacher_configs
        return _parse_mopd_teacher_configs(make_args(**kwargs))

    def test_no_configs_returns_only_general(self):
        result = self._parse(opd_teacher_num_gpus=2)
        assert result == [{"name": "general", "num_gpus": 2, "tp_size": 1}]

    def test_specialised_teachers_appended_after_general(self):
        cfg = json.dumps([{"name": "math", "path": "/ckpt/math", "domains": ["math"], "num_gpus": 1}])
        result = self._parse(mopd_teacher_configs=cfg)
        assert result[0]["name"] == "general"
        assert result[1]["name"] == "math"

    def test_missing_num_gpus_falls_back_to_general(self):
        # specialised entry has no num_gpus → should inherit opd_teacher_num_gpus
        cfg = json.dumps([{"name": "code", "path": "/ckpt/code", "domains": ["code"]}])
        result = self._parse(opd_teacher_num_gpus=3, mopd_teacher_configs=cfg)
        assert result[1]["num_gpus"] == 3

    def test_explicit_num_gpus_not_overridden(self):
        cfg = json.dumps([{"name": "code", "path": "/ckpt/code", "domains": ["code"], "num_gpus": 4}])
        result = self._parse(opd_teacher_num_gpus=1, mopd_teacher_configs=cfg)
        assert result[1]["num_gpus"] == 4


# ── create_opd_placement_groups — teachers dict ───────────────────────────────

class TestMopdPlacementGroups:
    def test_single_teacher_returns_general_key(self):
        from orbit.ray.placement_group import create_opd_placement_groups
        pgs = create_opd_placement_groups(make_args())
        assert "teachers" in pgs
        assert "general" in pgs["teachers"]

    def test_single_teacher_gpu_layout(self):
        # actor=2, general=1, rollout=1 → total 4, slices [0,1] / [2] / [3]
        from orbit.ray.placement_group import create_opd_placement_groups
        pgs = create_opd_placement_groups(make_args(actor_num_gpus_per_node=2))
        assert pgs["actor"][1] == [0, 1]
        assert pgs["teachers"]["general"][1] == [2]
        assert pgs["rollout"][1] == [3]

    def test_two_teachers_gpu_layout(self):
        # actor=2, general=1, math=2, rollout=1 → total 6
        cfg = json.dumps([{"name": "math", "path": "/ckpt", "domains": ["math"], "num_gpus": 2}])
        from orbit.ray.placement_group import create_opd_placement_groups
        pgs = create_opd_placement_groups(make_args(actor_num_gpus_per_node=2, mopd_teacher_configs=cfg))
        assert pgs["actor"][1] == [0, 1]
        assert pgs["teachers"]["general"][1] == [2]
        assert pgs["teachers"]["math"][1] == [3, 4]
        assert pgs["rollout"][1] == [5]

    def test_no_overlap_no_gap(self):
        cfg = json.dumps([{"name": "math", "path": "/ckpt", "domains": ["math"], "num_gpus": 2}])
        from orbit.ray.placement_group import create_opd_placement_groups
        pgs = create_opd_placement_groups(make_args(actor_num_gpus_per_node=2, mopd_teacher_configs=cfg))
        all_indices = (
            pgs["actor"][1]
            + pgs["teachers"]["general"][1]
            + pgs["teachers"]["math"][1]
            + pgs["rollout"][1]
        )
        assert sorted(all_indices) == list(range(6))

    def test_colocate_all_teachers_share_same_pg(self):
        cfg = json.dumps([{"name": "math", "path": "/ckpt", "domains": ["math"]}])
        from orbit.ray.placement_group import create_opd_placement_groups
        pgs = create_opd_placement_groups(
            make_args(colocate=True, actor_num_gpus_per_node=1, mopd_teacher_configs=cfg)
        )
        assert pgs["teachers"]["general"][1] == [0]
        assert pgs["teachers"]["math"][1] == [0]
        assert pgs["rollout"][1] == [0]

    def test_debug_train_only_teachers_is_none(self):
        from orbit.ray.placement_group import create_opd_placement_groups
        pgs = create_opd_placement_groups(make_args(debug_train_only=True))
        assert pgs["teachers"] is None

    def test_debug_rollout_only_teachers_is_none(self):
        from orbit.ray.placement_group import create_opd_placement_groups
        pgs = create_opd_placement_groups(make_args(debug_rollout_only=True))
        assert pgs["teachers"] is None
