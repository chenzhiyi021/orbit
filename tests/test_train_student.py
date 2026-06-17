"""
Tests for _is_opd_batch / train_student routing in
orbit/backends/megatron_utils/actor.py.

These tests target the OPD routing decision in MegatronTrainRayActor.train()
and the train_student() control flow. They mock out everything Megatron-
specific (model, optimizer, data iterator, train()) so they can run without
GPUs or a real Megatron init.

Run with:
    pytest orbit/tests/test_train_student.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestIsOpdBatch:
    def test_true_when_teacher_log_probs_present(self):
        from orbit.backends.megatron_utils.actor import _is_opd_batch

        rollout_data = {"tokens": [[1, 2, 3]], "teacher_log_probs": torch.tensor([[-0.1]])}
        assert _is_opd_batch(rollout_data) is True

    def test_true_when_topk_logits_present(self):
        from orbit.backends.megatron_utils.actor import _is_opd_batch

        rollout_data = {"topk_logits": torch.zeros(1, 1, 32)}
        assert _is_opd_batch(rollout_data) is True

    def test_true_when_teacher_logits_present(self):
        from orbit.backends.megatron_utils.actor import _is_opd_batch

        rollout_data = {"teacher_logits": torch.zeros(1, 1, 100)}
        assert _is_opd_batch(rollout_data) is True

    def test_false_for_plain_rl_batch(self):
        from orbit.backends.megatron_utils.actor import _is_opd_batch

        rollout_data = {"tokens": [[1, 2, 3]], "rewards": [1.0]}
        assert _is_opd_batch(rollout_data) is False

    def test_false_for_empty_dict(self):
        from orbit.backends.megatron_utils.actor import _is_opd_batch

        assert _is_opd_batch({}) is False


class TestTrainRoutesToStudentPath:
    """train() should call train_student() instead of train_actor() when
    _is_opd_batch(rollout_data) is True, and should not touch train_critic.
    """

    def _make_actor_stub(self):
        """Build a MegatronTrainRayActor-like object with train(), train_critic(),
        train_actor(), train_student() all mocked, so we test only the routing
        branch inside train() without running any real Megatron code.
        """
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
        obj.args = MagicMock(offload_train=False, debug_rollout_only=False)
        obj.role = "actor"
        obj.train_critic = MagicMock(name="train_critic")
        obj.train_actor = MagicMock(name="train_actor")
        obj.train_student = MagicMock(name="train_student")
        return obj

    @patch("orbit.backends.megatron_utils.actor.get_rollout_data")
    def test_opd_batch_routes_to_train_student(self, mock_get_rollout_data):
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = self._make_actor_stub()
        mock_get_rollout_data.return_value = {"teacher_log_probs": torch.tensor([[-0.1]])}

        MegatronTrainRayActor.train(obj, rollout_id=5, rollout_data_ref=MagicMock())

        obj.train_student.assert_called_once()
        obj.train_actor.assert_not_called()
        obj.train_critic.assert_not_called()

    @patch("orbit.backends.megatron_utils.actor.get_rollout_data")
    def test_rl_batch_routes_to_train_actor(self, mock_get_rollout_data):
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = self._make_actor_stub()
        mock_get_rollout_data.return_value = {"rewards": [1.0]}

        MegatronTrainRayActor.train(obj, rollout_id=5, rollout_data_ref=MagicMock())

        obj.train_actor.assert_called_once()
        obj.train_student.assert_not_called()
        obj.train_critic.assert_not_called()

    @patch("orbit.backends.megatron_utils.actor.get_rollout_data")
    def test_critic_role_routes_to_train_critic_even_with_teacher_keys(self, mock_get_rollout_data):
        """If role == 'critic', critic path takes priority regardless of
        teacher keys (OPD has no critic, but guard against ordering bugs).
        """
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = self._make_actor_stub()
        obj.role = "critic"
        mock_get_rollout_data.return_value = {"teacher_log_probs": torch.tensor([[-0.1]])}

        MegatronTrainRayActor.train(obj, rollout_id=5, rollout_data_ref=MagicMock())

        obj.train_critic.assert_called_once()
        obj.train_student.assert_not_called()


class TestTrainStudentSetsLossType:
    """train_student() must set args.loss_type to 'opd_<opd_loss_type>'
    before calling the Megatron train() function, since that string is
    how model.py routes to the OPD loss path.
    """

    def _make_actor_stub(self, opd_loss_type="sampled_token", opd_rl_coef=0.0):
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = MegatronTrainRayActor.__new__(MegatronTrainRayActor)
        obj.args = MagicMock(
            opd_loss_type=opd_loss_type,
            opd_rl_coef=opd_rl_coef,
            ref_update_interval=None,
        )
        obj.model = MagicMock()
        obj.optimizer = MagicMock()
        obj.opt_param_scheduler = MagicMock()
        obj.prof = MagicMock()
        obj._switch_model = MagicMock()
        obj.compute_log_prob = MagicMock(return_value={"log_probs": torch.tensor([[-0.1]])})
        obj.compute_ref_log_probs = MagicMock(return_value=None)
        obj.model_state_manager = MagicMock(backup_tags=[])
        return obj

    @patch("orbit.backends.megatron_utils.actor.train")
    @patch("orbit.backends.megatron_utils.actor.log_rollout_data")
    @patch("orbit.backends.megatron_utils.actor.log_perf_data")
    @patch("orbit.backends.megatron_utils.actor.get_data_iterator")
    @patch("orbit.backends.megatron_utils.actor.train_dump_utils")
    @patch("orbit.backends.megatron_utils.actor.should_backup_actor_after_train", return_value=False)
    @patch("orbit.backends.megatron_utils.actor.is_megatron_main_rank", return_value=True)
    def test_loss_type_set_to_opd_prefixed_string(
        self,
        mock_is_main_rank,
        mock_should_backup,
        mock_dump_utils,
        mock_get_data_iterator,
        mock_log_perf,
        mock_log_rollout,
        mock_train_fn,
    ):
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = self._make_actor_stub(opd_loss_type="topk")
        mock_get_data_iterator.return_value = ([MagicMock()], [1])

        rollout_data = {"topk_logits": torch.zeros(1, 1, 32)}
        MegatronTrainRayActor.train_student(obj, rollout_id=3, rollout_data=rollout_data)

        assert obj.args.loss_type == "opd_topk"
        mock_train_fn.assert_called_once()

    @patch("orbit.backends.megatron_utils.actor.train")
    @patch("orbit.backends.megatron_utils.actor.log_rollout_data")
    @patch("orbit.backends.megatron_utils.actor.log_perf_data")
    @patch("orbit.backends.megatron_utils.actor.get_data_iterator")
    @patch("orbit.backends.megatron_utils.actor.train_dump_utils")
    @patch("orbit.backends.megatron_utils.actor.should_backup_actor_after_train", return_value=False)
    @patch("orbit.backends.megatron_utils.actor.is_megatron_main_rank", return_value=True)
    def test_ref_log_probs_skipped_when_opd_rl_coef_zero(
        self,
        mock_is_main_rank,
        mock_should_backup,
        mock_dump_utils,
        mock_get_data_iterator,
        mock_log_perf,
        mock_log_rollout,
        mock_train_fn,
    ):
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = self._make_actor_stub(opd_loss_type="sampled_token", opd_rl_coef=0.0)
        mock_get_data_iterator.return_value = ([MagicMock()], [1])

        rollout_data = {"teacher_log_probs": torch.tensor([[-0.1]])}
        MegatronTrainRayActor.train_student(obj, rollout_id=3, rollout_data=rollout_data)

        obj.compute_ref_log_probs.assert_not_called()

    @patch("orbit.backends.megatron_utils.actor.train")
    @patch("orbit.backends.megatron_utils.actor.log_rollout_data")
    @patch("orbit.backends.megatron_utils.actor.log_perf_data")
    @patch("orbit.backends.megatron_utils.actor.get_data_iterator")
    @patch("orbit.backends.megatron_utils.actor.train_dump_utils")
    @patch("orbit.backends.megatron_utils.actor.should_backup_actor_after_train", return_value=False)
    @patch("orbit.backends.megatron_utils.actor.is_megatron_main_rank", return_value=True)
    def test_ref_log_probs_computed_when_opd_rl_coef_positive(
        self,
        mock_is_main_rank,
        mock_should_backup,
        mock_dump_utils,
        mock_get_data_iterator,
        mock_log_perf,
        mock_log_rollout,
        mock_train_fn,
    ):
        from orbit.backends.megatron_utils.actor import MegatronTrainRayActor

        obj = self._make_actor_stub(opd_loss_type="sampled_token", opd_rl_coef=0.5)
        mock_get_data_iterator.return_value = ([MagicMock()], [1])
        obj.compute_ref_log_probs.return_value = {"ref_log_probs": torch.tensor([[-0.2]])}

        rollout_data = {"teacher_log_probs": torch.tensor([[-0.1]])}
        MegatronTrainRayActor.train_student(obj, rollout_id=3, rollout_data=rollout_data)

        obj.compute_ref_log_probs.assert_called_once()