import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from cts.train.stage1_warmup import (
    _picard_fixed_point,
    _sampled_secant_contraction_loss,
    _trajectory_secant_contraction_loss,
    fixed_point_surrogate_loss,
    language_model_preservation_loss,
    run_stage1_demo_step,
    wproj_teacher_forcing_loss,
)
from cts.types import NuVector


def test_fixed_point_loss_scalar():
    loss, _ = run_stage1_demo_step(lr=1e-2)
    assert loss == loss  # finite
    assert loss >= 0.0


def test_surrogate_backward():
    from cts.backbone.mock_tiny import MockTinyBackbone

    bb = MockTinyBackbone(hidden=32, num_layers=8)
    nu = NuVector(nu_temp=1.0)
    w_g = torch.randn(19, 32) * 0.02
    z = torch.randn(4, 32)
    loss = fixed_point_surrogate_loss(bb, "x", z, nu, w_g=w_g)
    loss.backward()
    assert bb.mix.weight.grad is not None


class _ScaledIdentityDEQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def encode_context(self, text):
        return torch.zeros(1, 3)

    def deq_step(self, z, context, module_weights, extra):
        return self.scale * z


class _CountingScaledIdentityDEQ(_ScaledIdentityDEQ):
    def __init__(self):
        super().__init__()
        self.deq_calls = 0

    def deq_step(self, z, context, module_weights, extra):
        self.deq_calls += 1
        return super().deq_step(z, context, module_weights, extra)


def test_stage1_fixed_point_term_is_squared_l2_sum_not_mean():
    bb = _ScaledIdentityDEQ()
    z = torch.ones(2, 3)
    loss = fixed_point_surrogate_loss(
        bb,
        "problem",
        z,
        NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 3),
        lambda_lm=0.0,
        lambda_wproj=0.0,
        solve_equilibrium=False,
    )

    assert torch.equal(loss.detach(), torch.tensor(6.0))
    loss.backward()
    assert torch.equal(bb.scale.grad, torch.tensor(-12.0))


def test_zero_contraction_lambda_preserves_loss_gradient_and_phi_count() -> None:
    first = _CountingScaledIdentityDEQ()
    second = _CountingScaledIdentityDEQ()
    first.scale.data.fill_(0.5)
    second.load_state_dict(first.state_dict())
    z = torch.ones(2, 3)
    w_g = torch.randn(19, 3)
    common = dict(
        parent_text="problem",
        z=z,
        nu=NuVector(nu_temp=1.0),
        w_g=w_g,
        lambda_lm=0.0,
        lambda_wproj=0.0,
        solve_equilibrium=False,
    )

    baseline = fixed_point_surrogate_loss(first, **common)
    explicit_zero = fixed_point_surrogate_loss(
        second,
        diagnostic_contraction_lambda=0.0,
        diagnostic_contraction_mode="trajectory",
        **common,
    )
    baseline.backward()
    explicit_zero.backward()

    assert torch.equal(baseline.detach(), explicit_zero.detach())
    assert torch.equal(first.scale.grad, second.scale.grad)
    assert first.deq_calls == second.deq_calls == 1


class _PromptRoutingProbe(_ScaledIdentityDEQ):
    def root_fwdpass(
        self,
        context,
        *,
        K: int,
        output_dtype=torch.float32,
    ):
        row = torch.tensor([1.0, 2.0, 3.0], dtype=output_dtype)
        return row.expand(K, -1).clone()

    def deq_step(self, z, context, module_weights, extra):
        module_axis = torch.arange(
            module_weights.numel(),
            device=module_weights.device,
            dtype=module_weights.dtype,
        )
        routed_gate = (module_weights.reshape(-1) * module_axis).sum()
        return self.scale * z + routed_gate


def test_stage1_diagnostic_can_freeze_routing_support_without_cutting_wg_grad():
    bb = _PromptRoutingProbe()
    w_g = torch.randn(19, 3, requires_grad=True)
    metrics = {}
    loss = fixed_point_surrogate_loss(
        bb,
        "problem",
        torch.randn(2, 3),
        NuVector(nu_temp=1.0),
        w_g=w_g,
        extra={"routing_support_override": [7, 1, 4]},
        lambda_lm=0.0,
        lambda_wproj=0.0,
        solve_equilibrium=False,
        metrics_out=metrics,
    )
    loss.backward()

    assert metrics["root_initialization"] == "prompt_full_stack"
    assert metrics["routing_module_ids"] == [1, 4, 7]
    assert metrics["routing_support_fixed"] is True
    assert metrics["routing_entropy"] > 0.0
    assert w_g.grad is not None
    assert w_g.grad.abs().sum().item() > 0.0


def test_stage1_diagnostic_can_freeze_complete_routing_weights() -> None:
    bb = _PromptRoutingProbe()
    w_g = torch.randn(19, 3, requires_grad=True)
    cache = {}
    metrics = {}
    loss = fixed_point_surrogate_loss(
        bb,
        "problem",
        torch.randn(2, 3),
        NuVector(nu_temp=1.0),
        w_g=w_g,
        extra={"routing_support_override": [7, 1, 4]},
        lambda_lm=0.0,
        solve_equilibrium=False,
        diagnostic_routing_weights_cache=cache,
        metrics_out=metrics,
    )
    loss.backward()
    assert metrics["routing_weights_frozen"] is True
    assert w_g.grad is None
    assert bb.scale.grad is not None
    assert "module_weights" in cache


def test_stage1_diagnostic_reuses_first_broyden_target() -> None:
    bb = _ScaledIdentityDEQ()
    bb.scale.data.fill_(0.5)
    cache = {}
    common = dict(
        parent_text="single problem",
        z=torch.randn(2, 3),
        nu=NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 3),
        extra={"routing_support_override": [1, 4, 7]},
        solve_equilibrium=True,
        equilibrium_max_iter=4,
        lambda_lm=0.0,
        diagnostic_z_target_cache=cache,
    )
    first_metrics = {}
    fixed_point_surrogate_loss(bb, metrics_out=first_metrics, **common)
    first_target = cache["z_target"].clone()

    bb.scale.data.fill_(0.9)
    second_metrics = {}
    fixed_point_surrogate_loss(bb, metrics_out=second_metrics, **common)

    assert torch.equal(cache["z_target"], first_target)
    assert first_metrics["z_target_source"] == "broyden_current_parameters"
    assert second_metrics["z_target_source"] == "diagnostic_first_solve_reuse"


def test_stage1_picard_solver_builds_a_converged_detached_target() -> None:
    bb = _ScaledIdentityDEQ()
    bb.scale.data.fill_(0.5)
    metrics = {}
    loss = fixed_point_surrogate_loss(
        bb,
        "problem",
        torch.ones(2, 3),
        NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 3),
        solve_equilibrium=True,
        equilibrium_solver="picard",
        equilibrium_tol=1e-3,
        equilibrium_max_iter=20,
        lambda_lm=0.0,
        metrics_out=metrics,
    )

    assert torch.isfinite(loss)
    assert metrics["equilibrium_solver"] == "picard"
    assert metrics["z_target_source"] == "picard_current_parameters"
    assert metrics["equilibrium_converged"] is True
    assert metrics["equilibrium_iterations"] <= 20
    assert metrics["equilibrium_residual_history"][-1] < 1e-3
    assert (
        metrics["equilibrium_residual_history"][-1]
        < metrics["equilibrium_residual_history"][0]
    )


def test_stage1_picard_solver_fails_closed_on_non_finite_map() -> None:
    def non_finite_phi(state: torch.Tensor) -> torch.Tensor:
        return torch.full_like(state, float("inf"))

    z0 = torch.ones(2, 3)
    target, info = _picard_fixed_point(
        non_finite_phi,
        z0,
        tol=1e-3,
        max_iter=20,
    )

    assert torch.equal(target, z0)
    assert info.converged is False
    assert info.iterations == 1
    assert info.phi_eval_count == 1
    assert info.failure_reason == "non_finite_residual"


def test_picard_diagnostic_selects_worst_stable_secant() -> None:
    diagnostics = {}

    def scaled_identity(state: torch.Tensor) -> torch.Tensor:
        return 0.95 * state

    _, info = _picard_fixed_point(
        scaled_identity,
        torch.ones(2, 3),
        tol=1e-5,
        max_iter=80,
        diagnostics_out=diagnostics,
    )

    assert info.converged is False
    assert diagnostics["selector"] == "worst_consecutive_picard_secant_v1"
    assert abs(diagnostics["worst_secant_gain"] - 0.95) < 1e-5
    assert diagnostics["worst_secant_iteration"] >= 5
    assert diagnostics["anchor_teacher_residual"] >= 1e-2
    assert diagnostics["probe_a"].shape == (2, 3)
    assert diagnostics["probe_b"].shape == (2, 3)


def test_endpoint_secant_contraction_loss_value_and_gradient() -> None:
    scale = nn.Parameter(torch.tensor(0.95))
    anchor = torch.zeros(2, 3)
    target = torch.ones(2, 3)

    def phi(state: torch.Tensor) -> torch.Tensor:
        return scale * state

    mapped_target = phi(target)
    loss, gain, details = _sampled_secant_contraction_loss(
        phi,
        anchor,
        target,
        mapped_target,
        target_gain=0.90,
    )

    assert torch.allclose(gain.detach(), torch.tensor(0.95), atol=1e-6)
    assert torch.allclose(loss.detach(), torch.tensor(0.015), atol=1e-6)
    assert details["scale_n"] == 6
    assert details["input_chord_relative"] > 0.02
    loss.backward()
    assert torch.allclose(scale.grad, torch.tensor(0.6), atol=1e-5)


def test_endpoint_secant_contraction_loss_is_zero_below_target() -> None:
    scale = nn.Parameter(torch.tensor(0.80))
    anchor = torch.zeros(2, 3)
    target = torch.ones(2, 3)

    def phi(state: torch.Tensor) -> torch.Tensor:
        return scale * state

    loss, gain, _ = _sampled_secant_contraction_loss(
        phi,
        anchor,
        target,
        phi(target),
        target_gain=0.90,
    )
    loss.backward()

    assert float(gain.detach()) < 0.90
    assert torch.equal(loss.detach(), torch.tensor(0.0))
    assert torch.equal(scale.grad, torch.tensor(0.0))


def test_trajectory_secant_contraction_loss_value_gradient_and_two_phi() -> None:
    scale = nn.Parameter(torch.tensor(0.95))
    calls = 0

    def phi(state: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return scale * state

    loss, gain, details = _trajectory_secant_contraction_loss(
        phi,
        torch.zeros(2, 3),
        torch.ones(2, 3),
        target_gain=0.90,
    )

    assert calls == 2
    assert torch.allclose(gain.detach(), torch.tensor(0.95), atol=1e-6)
    assert torch.allclose(loss.detach(), torch.tensor(0.015), atol=1e-6)
    assert details["scale_n"] == 6
    loss.backward()
    assert torch.allclose(scale.grad, torch.tensor(0.6), atol=1e-5)


def test_endpoint_secant_contraction_rejects_tiny_chord() -> None:
    target = torch.ones(2, 3)
    anchor = target + 1e-4

    try:
        _sampled_secant_contraction_loss(
            lambda state: 0.95 * state,
            anchor,
            target,
            0.95 * target,
            target_gain=0.90,
        )
    except RuntimeError as exc:
        assert "BF16-safe" in str(exc)
    else:
        raise AssertionError("a numerically tiny secant chord was accepted")


def test_stage1_endpoint_contraction_rejects_nonfinite_lambda() -> None:
    bb = _ScaledIdentityDEQ()
    try:
        fixed_point_surrogate_loss(
            bb,
            "problem",
            torch.ones(2, 3),
            NuVector(nu_temp=1.0),
            w_g=torch.randn(19, 3),
            lambda_lm=0.0,
            diagnostic_contraction_lambda=float("nan"),
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("a NaN contraction lambda was accepted")


def test_stage1_reuses_first_converged_picard_target_bitwise() -> None:
    bb = _CountingScaledIdentityDEQ()
    bb.scale.data.fill_(0.5)
    cache = {}
    common = dict(
        parent_text="single problem",
        z=torch.ones(2, 3),
        nu=NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 3),
        extra={"routing_support_override": [1, 4, 7]},
        solve_equilibrium=True,
        equilibrium_solver="picard",
        equilibrium_tol=1e-3,
        equilibrium_max_iter=20,
        lambda_lm=0.0,
        diagnostic_z_target_cache=cache,
    )
    first_metrics = {}
    fixed_point_surrogate_loss(bb, metrics_out=first_metrics, **common)
    first_target = cache["z_target"].clone()
    calls_after_first = bb.deq_calls

    second_metrics = {}
    fixed_point_surrogate_loss(bb, metrics_out=second_metrics, **common)

    assert torch.equal(cache["z_target"], first_target)
    assert first_metrics["z_target_source"] == "picard_current_parameters"
    assert second_metrics["z_target_source"] == "diagnostic_first_solve_reuse"
    assert "contraction_probe_a" not in cache
    # A reused target performs only the one differentiable Phi(z_target)
    # evaluation needed for the loss; it does not rerun the Picard teacher.
    assert bb.deq_calls == calls_after_first + 1


def test_stage1_endpoint_contraction_cache_adds_exactly_one_phi() -> None:
    bb = _CountingScaledIdentityDEQ()
    bb.scale.data.fill_(0.5)
    target_cache = {}
    routing_cache = {}
    common = dict(
        parent_text="single problem",
        z=torch.ones(2, 3),
        nu=NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 3),
        extra={"routing_support_override": [1, 4, 7]},
        solve_equilibrium=True,
        equilibrium_solver="picard",
        equilibrium_tol=1e-3,
        equilibrium_max_iter=20,
        lambda_lm=0.0,
        diagnostic_z_target_cache=target_cache,
        diagnostic_routing_weights_cache=routing_cache,
        diagnostic_contraction_lambda=1.0,
        diagnostic_contraction_target_gain=0.90,
    )
    first_metrics = {}
    fixed_point_surrogate_loss(bb, metrics_out=first_metrics, **common)
    calls_after_first = bb.deq_calls

    second_metrics = {}
    fixed_point_surrogate_loss(bb, metrics_out=second_metrics, **common)

    assert "contraction_probe_a" in target_cache
    assert second_metrics["contraction_extra_phi_evals"] == 1
    assert second_metrics["teacher_anchor_iteration"] >= 5
    # Reused target: Phi(z_target) plus exactly one Phi(anchor).
    assert bb.deq_calls == calls_after_first + 2

    trajectory_common = dict(common)
    trajectory_common["diagnostic_contraction_mode"] = "trajectory"
    calls_before_trajectory = bb.deq_calls
    trajectory_metrics = {}
    fixed_point_surrogate_loss(
        bb, metrics_out=trajectory_metrics, **trajectory_common
    )
    assert trajectory_metrics["contraction_mode"] == "trajectory"
    assert trajectory_metrics["contraction_extra_phi_evals"] == 2
    assert trajectory_metrics["contraction_trajectory_gain"] is not None
    assert trajectory_metrics["contraction_endpoint_gain"] is None
    # Reused target: Phi(z_target) plus Phi(probe_a) and Phi(probe_b).
    assert bb.deq_calls == calls_before_trajectory + 3

    broken_cache = dict(target_cache)
    del broken_cache["teacher_anchor_residual"]
    broken_common = dict(common)
    broken_common["diagnostic_z_target_cache"] = broken_cache
    try:
        fixed_point_surrogate_loss(bb, metrics_out={}, **broken_common)
    except RuntimeError as exc:
        assert "cache is incomplete" in str(exc)
    else:
        raise AssertionError("a partial contraction cache was accepted")


def test_stage1_endpoint_contraction_refuses_nonconverged_teacher() -> None:
    bb = _ScaledIdentityDEQ()
    bb.scale.data.fill_(1.1)

    try:
        fixed_point_surrogate_loss(
            bb,
            "single problem",
            torch.ones(2, 3),
            NuVector(nu_temp=1.0),
            w_g=torch.randn(19, 3),
            extra={"routing_support_override": [1, 4, 7]},
            solve_equilibrium=True,
            equilibrium_solver="picard",
            equilibrium_tol=1e-3,
            equilibrium_max_iter=3,
            lambda_lm=0.0,
            diagnostic_z_target_cache={},
            diagnostic_routing_weights_cache={},
            diagnostic_contraction_lambda=1.0,
        )
    except RuntimeError as exc:
        assert "non-converged" in str(exc)
    else:
        raise AssertionError("a nonconverged contraction teacher was accepted")


def test_stage1_local_gain_proxy_detects_sampled_expansion() -> None:
    bb = _ScaledIdentityDEQ()
    bb.scale.data.fill_(1.2)
    metrics = {}
    fixed_point_surrogate_loss(
        bb,
        "problem",
        torch.randn(2, 3),
        NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 3),
        solve_equilibrium=False,
        lambda_lm=0.0,
        diagnostic_local_gain_directions=4,
        diagnostic_local_gain_epsilon=1e-2,
        metrics_out=metrics,
    )
    assert metrics["local_gain_proxy_directions"] == 4
    assert abs(metrics["local_gain_proxy_median"] - 1.2) < 1e-4
    assert metrics["local_gain_proxy_max"] > 1.0
    assert abs(metrics["local_gain_proxy_residual_direction"] - 1.2) < 1e-4


def test_stage1_picard_diagnostic_reports_contractive_trajectory() -> None:
    bb = _ScaledIdentityDEQ()
    bb.scale.data.fill_(0.5)
    metrics = {}
    fixed_point_surrogate_loss(
        bb,
        "problem",
        torch.randn(2, 3),
        NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 3),
        solve_equilibrium=False,
        lambda_lm=0.0,
        diagnostic_picard_steps=6,
        metrics_out=metrics,
    )
    assert metrics["picard_steps"] == 6
    assert metrics["picard_final_residual"] < metrics["picard_residuals"][0]
    assert len(metrics["picard_secant_gains"]) == 5
    assert abs(metrics["picard_worst_secant_gain"] - 0.5) < 1e-6
    assert abs(metrics["picard_worst_stable_secant_gain"] - 0.5) < 1e-6


class _TinyTokenizer:
    eos_token_id = 31
    unk_token_id = 0
    chat_template = None

    def convert_tokens_to_ids(self, token):
        return 31 if token == "<turn|>" else self.unk_token_id

    def __call__(self, text, **kwargs):
        limit = int(kwargs.get("max_length", 64))
        ids = [1 + (ord(ch) % 30) for ch in str(text)][:limit] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _TinyTextModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.ple = nn.Linear(hidden, 4, bias=False)
        self.per_layer_input_scale = 2.0 ** -0.5
        self.lora_probe = nn.Linear(hidden, hidden, bias=False)
        self.forward_calls = 0
        self.last_forward_call = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        return torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            2,
            4,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

    def project_per_layer_inputs(self, inputs_embeds, identity):
        projected = self.ple(inputs_embeds).unsqueeze(2).expand(-1, -1, 2, -1)
        if identity is None:
            return projected
        return (projected + identity) * self.per_layer_input_scale

    def forward(
        self,
        *,
        inputs_embeds,
        per_layer_inputs=None,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        self.forward_calls += 1
        self.last_forward_call = {
            "inputs_shape": tuple(inputs_embeds.shape),
            "per_layer_inputs_shape": tuple(per_layer_inputs.shape),
            "attention_mask_shape": tuple(attention_mask.shape),
            "use_cache": use_cache,
            "extra_keys": tuple(sorted(kwargs)),
        }
        # A tiny causal dependency: supervised target positions must depend on
        # the masked W_proj prefix or its gradient would correctly be zero.
        hidden = inputs_embeds.cumsum(dim=1)
        hidden = hidden + 0.1 * self.lora_probe(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyCore(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8):
        super().__init__()
        self.language_model = _TinyTextModel(vocab, hidden)
        self.forward_calls = 0

    def forward(
        self,
        *,
        input_ids,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        self.forward_calls += 1
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        per_layer_inputs = self.language_model.get_per_layer_inputs(
            input_ids, inputs_embeds
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )


class _RecordingLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_input_shape = None

    def forward(self, inputs):
        self.last_input_shape = tuple(inputs.shape)
        return super().forward(inputs)


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 8):
        super().__init__()
        self.model = _TinyCore(vocab, hidden)
        self.lm_head = _RecordingLinear(hidden, vocab, bias=False)
        self.forward_calls = 0

    def forward(self, *, input_ids=None, inputs_embeds=None, labels, **kwargs):
        self.forward_calls += 1
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        logits = self.lm_head(inputs_embeds)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss)


class _TinyWProjBackbone(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.w_proj = nn.Linear(hidden, hidden, bias=False)
        self.cg = _TinyCausalLM(hidden=hidden)


def test_wproj_teacher_forcing_loss_reaches_projection_gradient():
    torch.manual_seed(7)
    bb = _TinyWProjBackbone()
    z = torch.randn(4, 8)
    tokenizer = _TinyTokenizer()
    target_text = "\\boxed{4}"
    loss = wproj_teacher_forcing_loss(
        bb,
        z,
        parent_text="2+2?",
        target_text=target_text,
        tokenizer=tokenizer,
        max_target_tokens=32,
    )
    text_model = bb.cg.model.language_model
    assert bb.cg.forward_calls == 0
    assert text_model.forward_calls == 1
    assert text_model.last_forward_call["inputs_shape"][:2] == (
        text_model.last_forward_call["per_layer_inputs_shape"][:2]
    )
    assert text_model.last_forward_call["inputs_shape"][:2] == (
        text_model.last_forward_call["attention_mask_shape"]
    )
    assert text_model.last_forward_call["use_cache"] is False
    assert "labels" not in text_model.last_forward_call["extra_keys"]
    content_tokens = tokenizer(
        target_text, add_special_tokens=False
    )["input_ids"].shape[1]
    expected_target_tokens = content_tokens + 1  # canonical <turn|> stop
    assert bb.cg.lm_head.last_input_shape[:2] == (1, expected_target_tokens)
    assert expected_target_tokens < text_model.last_forward_call["inputs_shape"][1]
    loss.backward()
    assert torch.isfinite(loss)
    assert bb.w_proj.weight.grad is not None
    assert bb.w_proj.weight.grad.abs().sum().item() > 0
    assert bb.cg.lm_head.weight.grad is not None
    assert bb.cg.lm_head.weight.grad.abs().sum().item() > 0
    assert text_model.lora_probe.weight.grad is not None
    assert text_model.lora_probe.weight.grad.abs().sum().item() > 0


def test_language_model_preservation_uses_solution_only_labels():
    torch.manual_seed(11)
    bb = _TinyWProjBackbone()
    loss = language_model_preservation_loss(
        bb,
        parent_text="2+2?",
        target_text="Reasoning. \\boxed{4}",
        tokenizer=_TinyTokenizer(),
        max_target_tokens=32,
    )
    assert bb.cg.forward_calls == 0
    assert bb.cg.model.forward_calls == 1
    assert bb.cg.model.language_model.forward_calls == 1
    loss.backward()
    assert torch.isfinite(loss)
    assert bb.cg.lm_head.weight.grad is not None


class _DropoutDEQBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.9)
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.dropout_flags: list[bool] = []

    def encode_context(self, text):
        return torch.zeros(1, 8)

    def deq_step(self, z, context, module_weights, extra):
        self.dropout_flags.append(self.dropout.training)
        return self.scale * self.dropout(z)


def test_stage1_broyden_and_phantom_phi_disable_then_restore_dropout():
    torch.manual_seed(19)
    bb = _DropoutDEQBackbone().train()
    z = torch.randn(2, 8)
    loss = fixed_point_surrogate_loss(
        bb,
        "problem",
        z,
        NuVector(nu_temp=1.0),
        w_g=torch.randn(19, 8),
        solve_equilibrium=True,
        equilibrium_max_iter=4,
        lambda_lm=0.0,
    )
    loss.backward()
    assert bb.dropout_flags and not any(bb.dropout_flags)
    assert bb.dropout.training is True
