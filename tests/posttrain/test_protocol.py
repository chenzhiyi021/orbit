from __future__ import annotations
import ast
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from orbit.research.posttrain import objectives as obj
from orbit.research.posttrain.protocol import load_profile, validate, applied_lr, resolve_sft
from orbit.research.posttrain.data import validate_token_row, supervised_pairs, verify_manifest
from orbit.research.posttrain.factors import initial_factors, independent_parameters, endpoint_reconstruction, validate_shared_export
from orbit.research.posttrain.constraints import install_lora_factor_constraint
from orbit.research.posttrain.checkpoint import save_state, load_state, restore_rng
from orbit.research.posttrain.runtime import event

OBSERVATIONS = {}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def source_functions(path, names, extra=None):
    tree = ast.parse(path.read_text())
    nodes = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            n.decorator_list = []
            nodes.append(n)
    scope = {"torch": torch, "math": math, "hashlib": __import__("hashlib"), "_LOG_RATIO_EXP_CLAMP": 20.0, "PEAKS": {2: 3.2e-4, 4: 2.4e-4, 8: 1.6e-4, 16: 1.08e-4}, **(extra or {})}
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    return scope


LEGACY = source_functions(HERE / "fixtures/m6_legacy.py", ["_safe_clamp_log_ratio", "_safe_exp_neg_ppo_kl", "compute_policy_loss", "opd_mopd_advantages", "seed", "initial_factors", "lr_at"])


class ProfileTests(unittest.TestCase):
    def test_m6_full_actual_lr_trace(self):
        p = load_profile("m6_mixed_r2_reference")
        validate(p)
        observed = json.loads((HERE / "fixtures/m6_applied_lr.json").read_text())
        for i, actual in enumerate(observed):
            self.assertEqual(applied_lr(p, i), actual)
            self.assertAlmostEqual(applied_lr(p, i), LEGACY["lr_at"](i, 3.2e-4), places=17)
        self.assertEqual(applied_lr(p, 0), 0.0)
        self.assertEqual(applied_lr(p, 10), 3.2e-4)

    def test_m6_rejects_uncertified_modes(self):
        for key, value in [("async_rollout", True), ("tis", True), ("score_centering", True), ("updates_per_rollout", 2)]:
            p = load_profile("m6_mixed_r2_reference")
            p[key] = value
            with self.assertRaises(ValueError):
                validate(p)

    def test_train_seed_does_not_replace_init_seed(self):
        a, _ = initial_factors((7, 9), 2, "decoder.layers.0.self_attention.linear_qkv.weight")
        torch.manual_seed(44)
        a2, _ = initial_factors((7, 9), 2, "decoder.layers.0.self_attention.linear_qkv.weight")
        self.assertTrue(torch.equal(a, a2))
        reference = LEGACY["initial_factors"]((7, 9), {"basis": "Kaiming-A", "rank": 2}, None, "decoder.layers.0.self_attention.linear_qkv.weight", "cpu")
        self.assertTrue(torch.equal(a, reference["A"]))

    def test_rl_scientific_decision_is_explicit(self):
        p = load_profile("rl_math_grpo_r4")
        validate(p)
        self.assertEqual((p["steps"], p["optimizer"]["lr"]), (300, 2e-4))
        self.assertEqual([applied_lr(p, i) for i in range(300)], [2e-4] * 300)
        self.assertEqual(p["group_std"], "sample")
        self.assertEqual(p["audit_code_default_lr"], 5e-6)

    def test_sft_unresolved_configuration_fails(self):
        p = load_profile("sft_omr_cot")
        self.assertEqual((p["batch"]["global_sequences"], p["steps"]), (256, 100))
        with self.assertRaises(ValueError):
            validate(p)

    def test_sft_cosine_resolution(self):
        p = resolve_sft("sft_omr_cot", {"optimizer": {"lr": 1e-4, "distributed": False}, "adapter": {"initialization": "name_seeded_kaiming"}, "parallel": {"tp": 1, "dp": 8}, "batch": {"micro_sequences": 1}, "init_seed": 42})
        validate(p)
        self.assertEqual(p["optimizer"]["min_lr"], 1e-5)

    def test_incomplete_manifest_fails_before_training(self):
        with self.assertRaises(ValueError):
            verify_manifest(ROOT / "examples/posttrain_subspace/rl_data_manifest.json", load_profile("rl_math_grpo_r4"))


class ObjectiveTests(unittest.TestCase):
    def test_enumerated_rkl_and_baselines(self):
        for vocab in [2, 3, 7, 11]:
            for temperature in [0.35, 0.7, 1.0, 2.0]:
                for seed in [1, 19]:
                    g = torch.Generator().manual_seed(seed + vocab)
                    raw = torch.randn(vocab, generator=g, dtype=torch.float64)
                    q = torch.log_softmax(torch.randn(vocab, generator=g, dtype=torch.float64) / temperature, -1).detach()
                    for scale in [1.0, 0.1]:
                        z = raw.clone().requires_grad_()
                        lp = (z / temperature).log_softmax(-1)
                        p = lp.exp().detach()
                        adv = obj.m6_advantages(lp.detach(), q, scale)
                        loss, clip, ratio = obj.policy_tokens(lp, lp.detach(), adv, force_one=True, clip_low=1e6, clip_high=1e6)
                        grad = torch.autograd.grad((p * loss).sum(), z)[0]
                        zl = raw.clone().requires_grad_()
                        lpl = (zl / temperature).log_softmax(-1)
                        al = LEGACY["opd_mopd_advantages"]({"teacher_log_probs": [q]}, [lpl.detach()], [vocab])[0] * scale
                        old_loss, _ = LEGACY["compute_policy_loss"](lpl.detach() - lpl, al, 1e6, 1e6, None)
                        old_gradient = torch.autograd.grad((p * old_loss).sum(), zl)[0]
                        torch.testing.assert_close(grad, old_gradient, atol=0, rtol=0)
                        zr = raw.clone().requires_grad_()
                        lpr = (zr / temperature).log_softmax(-1)
                        refgrad = torch.autograd.grad(scale * (lpr.exp() * (lpr - q)).sum(), zr)[0]
                        OBSERVATIONS["rkl_max_abs_gradient_error"] = max(OBSERVATIONS.get("rkl_max_abs_gradient_error", 0.0), float((grad - refgrad).abs().max()))
                        torch.testing.assert_close(grad, refgrad, atol=1e-7, rtol=2e-6)
                        self.assertTrue(torch.equal(ratio, torch.ones_like(ratio)))
                        self.assertFalse(clip.any())
                        for baseline in [-3.5, 0.0, 1.75, float((p * (q - lp.detach())).sum())]:
                            zb = raw.clone().requires_grad_()
                            lb = (zb / temperature).log_softmax(-1)
                            loss, _, _ = obj.policy_tokens(lb, lb.detach(), adv - scale * baseline, force_one=True, clip_low=1e6, clip_high=1e6)
                            gb = torch.autograd.grad((p * loss).sum(), zb)[0]
                            torch.testing.assert_close(gb, refgrad, atol=2e-7, rtol=3e-6)
                        ze = raw.clone().requires_grad_()
                        le = (ze / temperature).log_softmax(-1)
                        ae = obj.m6_advantages(le.detach(), le.detach(), scale)
                        leq, _, _ = obj.policy_tokens(le, le.detach(), ae, force_one=True)
                        self.assertTrue(torch.equal(torch.autograd.grad((p * leq).sum(), ze)[0], torch.zeros_like(ze)))

    def test_literal_loss_and_legacy_gradient(self):
        z = torch.tensor([0.4, -0.7, 0.3], requires_grad=True)
        lp = z.log_softmax(0)
        old = lp.detach()
        teacher = torch.tensor([-0.2, -1.1, -1.6])
        adv = obj.m6_advantages(old, teacher)
        actual, _, ratio = obj.policy_tokens(lp, old, adv, force_one=True)
        legacy_adv = LEGACY["opd_mopd_advantages"]({"teacher_log_probs": [teacher]}, [old], [3])[0] * 0.1
        expected, _ = LEGACY["compute_policy_loss"](lp.detach() - lp, legacy_adv, 0.2, 0.2, None)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual, -adv))
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), z, retain_graph=True)[0], torch.autograd.grad(expected.sum(), z)[0], atol=0, rtol=0)
        self.assertFalse(adv.requires_grad)
        self.assertTrue(actual.requires_grad)

    def test_detach_contract_rejected(self):
        t = torch.randn(3, requires_grad=True)
        with self.assertRaises(ValueError):
            obj.m6_advantages(t.detach(), t)
        with self.assertRaises(ValueError):
            obj.policy_tokens(t, t, torch.zeros_like(t))

    def test_rl_groups_sample_std_and_interleaving(self):
        actual = obj.group_advantages([0, 1, 1, 0, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1])
        a = 0.5 / (math.sqrt(1 / 3) + 1e-6)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor([-a, a, a, -a, -a, a, a, -a]))
        self.assertEqual(obj.group_advantages([0, 0, 0, 0, 1, 1, 1, 1], [0] * 4 + [1] * 4), [0.0] * 8)
        self.assertEqual(obj.group_advantages([1], [0]), [0.0])

    def test_rl_zero_advantage_still_counts_tokens(self):
        values = [torch.tensor([1.0, 2.0]), torch.zeros(3)]
        masks = [torch.ones(2), torch.ones(3)]
        num, den = obj.token_terms(values, masks)
        self.assertEqual(den.item(), 5)
        self.assertEqual(num.item(), 3)

    def test_actual_core_reducer_partition_invariance(self):
        extra = {"get_parallel_state": lambda: SimpleNamespace(cp=SimpleNamespace(size=1))}
        core = source_functions(ROOT / "orbit/backends/training_utils/cp_utils.py", ["get_sum_of_sample_mean"], extra)
        x = torch.arange(10, dtype=torch.float32, requires_grad=True)
        lengths = [2, 3, 1, 4]
        masks = [torch.ones(2), torch.tensor([1.0, 0.0, 1.0]), torch.zeros(1), torch.ones(4)]
        get = core["get_sum_of_sample_mean"]
        full = get(lengths, lengths, masks, True)(x) / sum(m.sum().clamp_min(1) for m in masks)
        pieces = x.split(lengths)
        a_num, a_den = obj.token_terms([pieces[0], pieces[2]], [masks[0], masks[2]])
        b_num, b_den = obj.token_terms([pieces[1], pieces[3]], [masks[1], masks[3]])
        split = (a_num + b_num) / (a_den + b_den)
        torch.testing.assert_close(full, split)
        torch.testing.assert_close(torch.autograd.grad(full, x, retain_graph=True)[0], torch.autograd.grad(split, x)[0])

    def test_production_sft_first_token_and_temperature(self):
        core = source_functions(ROOT / "orbit/backends/training_utils/loss.py", ["get_responses"], {"get_parallel_state": lambda: SimpleNamespace(cp=SimpleNamespace(size=1))})
        args = SimpleNamespace(posttrain_protocol={"loss_temperature": 1.0}, rollout_temperature=0.7, qkv_format="thd")
        logits = torch.arange(30, dtype=torch.float32).reshape(1, 6, 5)
        tokens = torch.tensor([1, 2, 3, 4, 0, 1])
        out = list(core["get_responses"](logits, args=args, unconcat_tokens=[tokens], total_lengths=[6], response_lengths=[3]))
        self.assertTrue(torch.equal(out[0][0], logits[0, 2:5]))
        self.assertTrue(torch.equal(out[0][1], tokens[3:]))
        expected = torch.nn.functional.cross_entropy(logits[0, 2:5], tokens[3:], reduction="none")
        torch.testing.assert_close(obj.raw_sft_nll(*out[0]), expected)

    def test_production_packed_response_boundaries(self):
        core = source_functions(ROOT / "orbit/backends/training_utils/loss.py", ["get_responses"], {"get_parallel_state": lambda: SimpleNamespace(cp=SimpleNamespace(size=1))})
        args = SimpleNamespace(posttrain_protocol={"loss_temperature": 1.0}, rollout_temperature=1.0, qkv_format="thd")
        logits = torch.arange(40, dtype=torch.float32).reshape(1, 8, 5)
        out = list(core["get_responses"](logits, args=args, unconcat_tokens=[torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2, 3, 4])], total_lengths=[3, 5], response_lengths=[1, 2]))
        self.assertTrue(torch.equal(out[0][0], logits[0, 1:2]))
        self.assertTrue(torch.equal(out[1][0], logits[0, 5:7]))

    def test_sft_eos_truncation_and_no_cross_sample_shift(self):
        policy = {"max_prompt_tokens": 5, "max_response_tokens": 5, "max_total_tokens": 8, "truncation_policy": "prevalidated_no_truncation"}
        row = {"prompt_ids": [1, 2], "response_ids": [3, 4, 0], "ended_with_eos": True}
        self.assertEqual(supervised_pairs(row), ([1, 2, 3], [3, 4, 0]))
        self.assertEqual(validate_token_row(row, policy)[1], [1, 1, 1])
        row["ended_with_eos"] = False
        with self.assertRaises(ValueError):
            validate_token_row(row, policy)
        row["truncated"] = True
        policy["truncation_policy"] = "explicit_right_truncation"
        self.assertEqual(validate_token_row(row, policy)[0], [1, 2, 3, 4, 0])
        other = {"prompt_ids": [8], "response_ids": [9], "ended_with_eos": True}
        self.assertEqual(supervised_pairs(other), ([0], [9]))

    def test_observation_switch_does_not_change_gradient(self):
        reference = None
        for observe in [False, True]:
            z = torch.tensor([0.1, 0.9], requires_grad=True)
            with tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(posttrain_run_dir=tmp, posttrain_observe=observe)
                before = torch.get_rng_state().clone()
                event(args, "probe", value=3)
                loss = obj.policy_tokens(z.log_softmax(0), z.log_softmax(0).detach(), torch.tensor([1.0, -1.0]), force_one=True)[0].sum()
                grad = torch.autograd.grad(loss, z)[0]
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                if observe:
                    self.assertIsNotNone(reference)
                    self.assertTrue(torch.equal(grad, reference))
                else:
                    reference = grad


class TinyLoRA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.randn(3, 4), requires_grad=False)
        self.adapter = torch.nn.Module()
        self.adapter.linear_in = torch.nn.Linear(4, 2, bias=False)
        self.adapter.linear_out = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.adapter.linear_out.weight.zero_()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.base) + self.adapter.linear_out(self.adapter.linear_in(x))


class OptimizerWrapper:
    def __init__(self, model):
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.0, betas=(0.9, 0.999), eps=1e-8)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, v):
        self.optimizer.load_state_dict(v)

    def prepare_grads(self):
        return False

    def step_with_ready_grads(self):
        self.optimizer.step()
        return True

    def step(self):
        self.prepare_grads()
        torch.nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]], 1.0)
        self.step_with_ready_grads()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)


class Scheduler:
    def __init__(self, optimizer, p):
        self.optimizer = optimizer
        self.p = p
        self.u = 0
        self.apply()

    def apply(self):
        for g in self.optimizer.optimizer.param_groups:
            g["lr"] = applied_lr(self.p, self.u)

    def state_dict(self):
        return {"u": self.u}

    def load_state_dict(self, state):
        self.u = state["u"]
        self.apply()

    def step(self):
        self.u += 1
        self.apply()


def update(model, opt, scheduler, x):
    opt.zero_grad()
    model(x).square().mean().backward()
    opt.step()
    scheduler.step()


class OptimizerTests(unittest.TestCase):
    def test_count_and_shared_export(self):
        self.assertEqual(independent_parameters(2), 1835008)
        self.assertEqual(independent_parameters(4), 3670016)
        state = {}
        for i in range(28):
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                prefix = "self_attn" if projection in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp"
                state[f"base_model.model.model.layers.{i}.{prefix}.{projection}.lora_A.weight"] = torch.ones(2, 4)
        self.assertEqual(validate_shared_export(state)["independent_fused_a_count"], 112)
        state["base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight"] = torch.zeros(2, 4)
        with self.assertRaises(ValueError):
            validate_shared_export(state)

    def test_zero_lr_advances_moments_without_movement(self):
        torch.manual_seed(1)
        model = TinyLoRA()
        opt = OptimizerWrapper(model)
        p = load_profile("m6_mixed_r2_reference")
        scheduler = Scheduler(opt, p)
        a = model.adapter.linear_in.weight
        b = model.adapter.linear_out.weight
        a0 = a.detach().clone()
        x = torch.randn(5, 4)
        update(model, opt, scheduler, x)
        self.assertTrue(torch.equal(a, a0))
        self.assertTrue(torch.equal(b, torch.zeros_like(b)))
        self.assertEqual(float(opt.optimizer.state[b]["step"]), 1.0)
        self.assertGreater(float(opt.optimizer.state[b]["exp_avg"].abs().sum()), 0.0)
        update(model, opt, scheduler, x)
        self.assertGreater(float(b.abs().sum()), 0.0)
        self.assertTrue(torch.equal(a, a0))
        update(model, opt, scheduler, x)
        self.assertFalse(torch.equal(a, a0))

    def test_complete_resume_equals_continuation(self):
        torch.manual_seed(2)
        p = load_profile("m6_mixed_r2_reference")
        model = TinyLoRA()
        initial = copy.deepcopy(model)
        opt = OptimizerWrapper(model)
        sch = Scheduler(opt, p)
        x = torch.randn(5, 4)
        for _ in range(3):
            update(model, opt, sch, x)
        with tempfile.TemporaryDirectory() as tmp:
            save_state(tmp, [model], opt, sch, completed_updates=3, protocol=p, data_hash="fixed")
            update(model, opt, sch, x)
            expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
            opt2 = OptimizerWrapper(initial)
            sch2 = Scheduler(opt2, p)
            state = load_state(tmp, [initial], opt2, sch2, protocol=p, data_hash="fixed")
            restore_rng(state["rng"])
            update(initial, opt2, sch2, x)
            for k, v in initial.state_dict().items():
                self.assertTrue(torch.equal(v, expected[k]), k)
            self.assertEqual(sch2.u, 4)
            with self.assertRaises(ValueError):
                load_state(tmp, [initial], opt2, sch2, protocol=p, data_hash="different")

    def test_exact_freeze_preserves_other_moments_and_input_gradient(self):
        torch.manual_seed(3)
        p = load_profile("m6_mixed_r2_reference")
        model = TinyLoRA()
        opt = OptimizerWrapper(model)
        sch = Scheduler(opt, p)
        x = torch.randn(5, 4)
        for _ in range(3):
            update(model, opt, sch, x)
        a = model.adapter.linear_in.weight
        b = model.adapter.linear_out.weight
        a0 = a.detach().clone()
        astate = copy.deepcopy(opt.optimizer.state[a])
        bstep = float(opt.optimizer.state[b]["step"])
        install_lora_factor_constraint([model], opt, factor="right")
        x = x.requires_grad_()
        update(model, opt, sch, x)
        self.assertTrue(torch.equal(a, a0))
        self.assertIsNone(a.grad)
        for k, v in astate.items():
            self.assertTrue(torch.equal(v, opt.optimizer.state[a][k]))
        self.assertEqual(float(opt.optimizer.state[b]["step"]), bstep + 1)
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_endpoint_actual_delta_leakage_and_full_forward(self):
        torch.manual_seed(4)
        a = torch.randn(2, 7)
        b = torch.randn(5, 2)
        base = torch.randn(5, 7)
        d = b @ a
        result = endpoint_reconstruction(a, b, d)
        self.assertLess(result["fixed_a_leakage_relative"], 1e-7)
        x = torch.randn(3, 7)
        torch.testing.assert_close(torch.nn.functional.linear(x, base + d), torch.nn.functional.linear(x, base) + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b), atol=2e-6, rtol=1e-5)
        bad = d + torch.randn_like(d) * 0.1
        self.assertGreater(endpoint_reconstruction(a, b, bad)["fixed_a_leakage_relative"], 1e-3)


class MixedMasterOptimizer(OptimizerWrapper):
    def __init__(self, model):
        parameters = [p for p in model.parameters() if p.requires_grad]
        self.float16_groups = [parameters]
        self.fp32_from_float16_groups = [[p.detach().float().clone().requires_grad_() for p in parameters]]
        self.optimizer = torch.optim.Adam(self.fp32_from_float16_groups[0], lr=0.0, betas=(0.9, 0.999), eps=1e-8)

    def prepare_grads(self):
        for p, master in zip(self.float16_groups[0], self.fp32_from_float16_groups[0], strict=True):
            master.grad = None if p.grad is None else p.grad.float().clone()
        return False

    def _copy_main_params_to_model_params(self):
        with torch.no_grad():
            for p, master in zip(self.float16_groups[0], self.fp32_from_float16_groups[0], strict=True):
                p.copy_(master.to(p.dtype))

    def step_with_ready_grads(self):
        self.optimizer.step()
        self._copy_main_params_to_model_params()
        return True

    def zero_grad(self):
        super().zero_grad()
        for p in self.float16_groups[0]:
            p.grad = None


class AdditionalContractTests(unittest.TestCase):
    def test_fp32_masters_survive_bf16_resume(self):
        torch.manual_seed(57)
        p = load_profile("m6_mixed_r2_reference")
        model = TinyLoRA().to(torch.bfloat16)
        initial = copy.deepcopy(model)
        optimizer = MixedMasterOptimizer(model)
        scheduler = Scheduler(optimizer, p)
        x = torch.randn(5, 4, dtype=torch.bfloat16)
        for _ in range(3):
            update(model, optimizer, scheduler, x)
        masters = [m.detach().clone() for m in optimizer.fp32_from_float16_groups[0]]
        self.assertTrue(any(not torch.equal(m, m.bfloat16().float()) for m in masters))
        with tempfile.TemporaryDirectory() as tmp:
            save_state(tmp, [model], optimizer, scheduler, completed_updates=3, protocol=p, data_hash="bank")
            other = MixedMasterOptimizer(initial)
            other_scheduler = Scheduler(other, p)
            state = load_state(tmp, [initial], other, other_scheduler, protocol=p, data_hash="bank")
            for m, expected in zip(other.fp32_from_float16_groups[0], masters, strict=True):
                self.assertEqual(m.dtype, torch.float32)
                self.assertTrue(torch.equal(m, expected))
            restore_rng(state["rng"])
            update(model, optimizer, scheduler, x)
            restore_rng(state["rng"])
            update(initial, other, other_scheduler, x)
            for m, expected in zip(other.fp32_from_float16_groups[0], optimizer.fp32_from_float16_groups[0], strict=True):
                self.assertTrue(torch.equal(m, expected))

    def test_restore_rng_keeps_te_tracker_generators(self):
        generator = torch.Generator().manual_seed(1)
        saved = generator.get_state().clone()
        expected = torch.rand(1, generator=generator)
        tracker = SimpleNamespace(states={"model-parallel-rng": generator})
        tracker.get_states = lambda: tracker.states
        tracker.set_states = lambda states: setattr(tracker, "states", states)
        fake = SimpleNamespace(get_cuda_rng_tracker=lambda: tracker)
        state = {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": [], "tracker": {"model-parallel-rng": saved}}
        with mock.patch.dict(sys.modules, {"megatron.core": SimpleNamespace(tensor_parallel=fake)}), mock.patch.object(torch.cuda, "set_rng_state_all"):
            restore_rng(state)
        self.assertIs(tracker.states["model-parallel-rng"], generator)
        self.assertTrue(torch.equal(torch.rand(1, generator=generator), expected))

    def test_effective_motion_matches_dense_delta(self):
        from orbit.research.posttrain.factors import effective_motion_squared, endpoint_basis

        torch.manual_seed(1)
        a, b = torch.randn(2, 7).double(), torch.randn(5, 2).double()
        anew, bnew = a + torch.randn_like(a) * 1e-6, b + torch.randn_like(b) * 1e-6
        actual = effective_motion_squared(a, b, anew, bnew)
        expected = (bnew @ anew - b @ a).square().sum()
        torch.testing.assert_close(actual, expected, atol=1e-20, rtol=1e-7)
        self.assertEqual(float(effective_motion_squared(a, b, a, b)), 0.0)
        with self.assertRaises(ValueError):
            endpoint_basis(b @ a, 3)
        with self.assertRaises(ValueError):
            endpoint_basis(torch.zeros(5, 7), 1)

    def test_teacher_prefill_temperature_and_no_grad(self):
        from orbit.research.posttrain.teacher import install_prefill_temperature

        class Metadata:
            @classmethod
            def from_forward_batch(cls, batch):
                return SimpleNamespace(extend_return_logprob=True, temp_scaled_logprobs=False)

        module = SimpleNamespace(LogitsMetadata=Metadata)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "audit.json"
            install_prefill_temperature(module, 0.7, file, "/fixed_teacher")
            result = Metadata.from_forward_batch(SimpleNamespace(batch_size=2, input_ids=torch.zeros(3, dtype=torch.int)))
            self.assertTrue(result.temp_scaled_logprobs)
            self.assertFalse(result.temperature.requires_grad)
            torch.testing.assert_close(result.temperature, torch.full((2, 1), 0.7))
            self.assertEqual(json.loads(file.read_text())["temperature"], 0.7)
            with self.assertRaises(RuntimeError):
                install_prefill_temperature(module, 1.0, file, "/fixed_teacher")

    def test_invalid_ground_truth_fails_but_zero_is_valid(self):
        from orbit.research.posttrain.protocol import digest
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = "fixed template"
            config = root / "tokenizer_config.json"
            config.write_text(json.dumps({"chat_template": template}))
            data = root / "data.jsonl"
            truth = root / "truth.jsonl"
            truth.write_text(json.dumps({"question": "2-2", "expected_answer": "0"}) + "\n")
            row = {"question": "2-2, boxed please", "source_question": "2-2", "answer": "0"}
            data.write_text(json.dumps(row) + "\n")
            manifest = {
                "schema_version": 1,
                "state": "ready",
                "identity": "verified_math_labels",
                "row_count": 1,
                "data": {"path": str(data), "sha256": digest(data)},
                "source": {"dataset": "math", "revision": "revision", "split": "train", "provenance_sha256": "a" * 64},
                "tokenizer": {"path": str(root), "files": [{"path": str(config), "sha256": digest(config)}], "chat_template_sha256": hashlib.sha256(json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
                "processing": {"prompt_format": "explicit", "assistant_format": "explicit", "eos_policy": "natural", "truncation_policy": "prevalidated_no_truncation", "packing_policy": "isolated_sequences", "max_prompt_tokens": 10, "max_response_tokens": 8192, "max_total_tokens": 8202},
                "splits": {key: "a" * 64 for key in ["train_question_ids_sha256", "dev_question_ids_sha256", "eval_question_ids_sha256", "question_disjoint_audit_sha256"]},
                "ground_truth": {"path": str(truth), "sha256": digest(truth), "source_revision": "revision"},
            }
            file = root / "manifest.json"
            file.write_text(json.dumps(manifest))
            verify_manifest(file, load_profile("rl_math_grpo_r4"))
            row["answer"] = "1"
            data.write_text(json.dumps(row) + "\n")
            manifest["data"]["sha256"] = digest(data)
            file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "ground truth"):
                verify_manifest(file, load_profile("rl_math_grpo_r4"))

    def test_sft_preparation_never_adds_fake_eos_to_truncation(self):
        from orbit.research.posttrain.prepare_sft import tokenize_row

        class Tokenizer:
            eos_token = "~"
            eos_token_id = ord("~")

            def apply_chat_template(self, messages, **kwargs):
                return "user:" + messages[0]["content"] + "|assistant:"

            def encode(self, value, **kwargs):
                return [ord(c) for c in value]

        policy = {"max_prompt_tokens": 100, "max_response_tokens": 100, "max_total_tokens": 200, "truncation_policy": "prevalidated_no_truncation"}
        row = {"source_id": "one", "prompt_messages": [{"role": "user", "content": "q"}], "response": "<think>reason</think>answer"}
        result = tokenize_row(row, Tokenizer(), policy, "strip_literal_tags_keep_content")
        self.assertEqual("".join(map(chr, result["response_ids"])), "reasonanswer~")
        self.assertTrue(result["ended_with_eos"])
        row["truncated"] = True
        result = tokenize_row(row, Tokenizer(), policy, "preserve")
        self.assertFalse(result["ended_with_eos"])
        self.assertEqual("".join(map(chr, result["response_ids"])), row["response"])
        policy["max_response_tokens"] = 2
        with self.assertRaises(ValueError):
            tokenize_row(row, Tokenizer(), policy, "preserve")
        policy["truncation_policy"] = "explicit_right_truncation"
        self.assertEqual(len(tokenize_row(row, Tokenizer(), policy, "preserve")["response_ids"]), 2)

    def test_nonfinite_policy_stops(self):
        with self.assertRaises(ValueError):
            obj.policy_tokens(torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
