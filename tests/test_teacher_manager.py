"""
Tests for TeacherManager / create_teacher_manager in orbit/ray/teacher.py.

TeacherManager spawns real Ray actors and SGLang engines, so a full
integration test needs real GPUs. These tests instead verify the
GPU-bookkeeping logic (num_engines computation, score() routing) using
mocks, and are meant to run on CPU-only CI / dev boxes.

A separate manual/GPU smoke test (not pytest) should be used to verify
the actual SGLang HTTP call against a real model -- see
examples/opd/run-teacher-smoke-test.sh.

Run with:
    pytest orbit/tests/test_teacher_manager.py -v
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def make_args(**overrides):
    defaults = dict(
        opd_teacher_tp_size=1,
        opd_teacher_model_path="/fake/path",
        opd_loss_type="sampled_token",
        opd_topk_k=32,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestNumEnginesComputation:
    """num_engines = total_gpus // tp_size; this is pure arithmetic and
    should not require spinning up real Ray actors. We patch ray.remote
    and ray.get so __init__ doesn't actually try to launch engines.
    """

    @patch("orbit.ray.teacher.ray")
    def test_single_gpu_tp1(self, mock_ray):
        from orbit.ray.teacher import TeacherManager

        mock_ray.get.return_value = [None]  # init.remote() results, unused
        mock_ray.remote.return_value.options.return_value.remote.return_value = MagicMock()

        args = make_args(opd_teacher_tp_size=1)
        pg = ("fake_pg", [0], [0])

        # Call __init__ directly (bypassing @ray.remote decorator) so we can
        # inspect self._engines without an actual Ray cluster.
        manager = TeacherManager.__new__(TeacherManager)
        TeacherManager.__init__(manager, args, pg)

        assert len(manager._engines) == 1

    @patch("orbit.ray.teacher.ray")
    def test_multi_gpu_tp2_yields_half_engines(self, mock_ray):
        from orbit.ray.teacher import TeacherManager

        mock_ray.get.return_value = [None, None]
        mock_ray.remote.return_value.options.return_value.remote.return_value = MagicMock()

        args = make_args(opd_teacher_tp_size=2)
        pg = ("fake_pg", [0, 1, 2, 3], [0, 1, 2, 3])  # 4 gpus, tp_size=2 -> 2 engines

        manager = TeacherManager.__new__(TeacherManager)
        TeacherManager.__init__(manager, args, pg)

        assert len(manager._engines) == 2

    @patch("orbit.ray.teacher.ray")
    def test_uneven_gpu_count_floors(self, mock_ray):
        """5 gpus // tp_size=2 -> 2 full engines, 1 gpu unused (documents current behavior)."""
        from orbit.ray.teacher import TeacherManager

        mock_ray.get.return_value = [None, None]
        mock_ray.remote.return_value.options.return_value.remote.return_value = MagicMock()

        args = make_args(opd_teacher_tp_size=2)
        pg = ("fake_pg", [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

        manager = TeacherManager.__new__(TeacherManager)
        TeacherManager.__init__(manager, args, pg)

        assert len(manager._engines) == 2  # not 2.5; floor division


class TestCreateTeacherManagerFactory:
    def test_raises_on_none_pg(self):
        from orbit.ray.teacher import create_teacher_manager

        args = make_args()
        with pytest.raises(ValueError, match="teacher"):
            create_teacher_manager(args, None)

    @patch("orbit.ray.teacher.PlacementGroupSchedulingStrategy")
    @patch("orbit.ray.teacher.TeacherManager")
    def test_options_called_with_first_bundle_index(self, mock_cls, mock_strategy):
        from orbit.ray.teacher import create_teacher_manager

        mock_cls.options.return_value.remote.return_value = MagicMock()
        args = make_args()
        pg = ("fake_pg_obj", [3, 4, 5], [3, 4, 5])

        create_teacher_manager(args, pg)

        mock_strategy.assert_called_once_with(
            placement_group="fake_pg_obj",
            placement_group_bundle_index=3,  # bundle_indices[0]
        )


class TestScoreRouting:
    """score() should branch on self.loss_type. We bypass the HTTP call by
    mocking the engine and directly testing the post-processing logic that
    does NOT depend on the network (sampled_token gather is local torch ops
    in some implementations -- adjust per actual TeacherManager.score body).

    NOTE: the current score() implementation calls requests.post directly.
    For unit testing we mock `requests.post` rather than re-implementing
    HTTP. If score() is refactored to separate "build payload" / "parse
    response" from the network call, these tests should be tightened to
    avoid mocking requests entirely.
    """

    def _make_manager_with_mocked_engine(self, loss_type="sampled_token"):
        from orbit.ray.teacher import TeacherManager

        manager = TeacherManager.__new__(TeacherManager)
        manager.loss_type = loss_type
        manager.topk_k = 32
        fake_engine = MagicMock()
        fake_engine.server_host = "127.0.0.1"
        fake_engine.server_port = 30000
        manager._engines = [fake_engine]
        return manager

    @patch("orbit.ray.teacher.requests.post")
    def test_sampled_token_returns_teacher_log_probs_key(self, mock_post):
        manager = self._make_manager_with_mocked_engine("sampled_token")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "input_token_logprobs": [[-0.1, -0.2, -0.3]]
        }
        mock_post.return_value = mock_response

        token_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        out = manager.score(token_ids, attention_mask)

        assert "teacher_log_probs" in out
        assert out["teacher_log_probs"].shape == (1, 3)

    @patch("orbit.ray.teacher.requests.post")
    def test_unexpected_response_format_raises(self, mock_post):
        manager = self._make_manager_with_mocked_engine("sampled_token")

        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected_key": []}
        mock_post.return_value = mock_response

        token_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])

        with pytest.raises(ValueError, match="Unexpected SGLang response format"):
            manager.score(token_ids, attention_mask)


class TestMergeTeacherSignal:
    def test_merges_keys_into_rollout_data(self):
        from orbit.ray.teacher import merge_teacher_signal

        rollout_data = {"token_ids": torch.tensor([[1, 2]]), "rewards": [1.0]}
        teacher_output = {"teacher_log_probs": torch.tensor([[-0.1, -0.2]])}

        merged = merge_teacher_signal(rollout_data, teacher_output)

        assert "teacher_log_probs" in merged
        assert "rewards" in merged  # original keys preserved
        assert merged is rollout_data  # in-place, returns same dict