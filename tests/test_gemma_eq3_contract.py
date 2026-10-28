from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn as nn

from cts.deq.gemma_latent_forward import (
    GemmaEq3Context,
    _build_explicit_eq3_masks,
    _direct_anchor_shared_kv,
    _validate_shared_kv_abi,
    encode_prompt_context,
    parallel_sparse_module_forward,
    parallel_sparse_module_forward_batch,
    prepare_eq3_module_prefix_cache,
    prepare_eq3_inputs,
    prepare_eq3_static_inputs,
)


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.store_full_length_kv = False
        self.is_kv_shared_layer = False


class _FakeLayer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.self_attn = _FakeAttention()
        self.forward_calls = 0

    def forward(
        self,
        hidden,
        per_layer_input,
        **kwargs,
    ):
        self.forward_calls += 1
        return (hidden + self.scale * per_layer_input,)


class _FakeLM(nn.Module):
    def __init__(self, hidden: int = 4, layers: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList(
            [_FakeLayer(float(i + 1) / 10.0) for i in range(layers)]
        )
        self.norm = nn.Identity()
        self.config = SimpleNamespace(
            num_hidden_layers=layers,
            layer_types=["full"] * layers,
        )
        self.unique_layer_types = {"full"}
        self.project_calls = 0

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        base = input_ids.to(inputs_embeds.dtype).unsqueeze(-1).unsqueeze(-1)
        return base.expand(
            input_ids.shape[0],
            input_ids.shape[1],
            len(self.layers),
            inputs_embeds.shape[-1],
        ) / 100.0

    def project_per_layer_inputs(self, inputs_embeds, identity):
        self.project_calls += 1
        projected = inputs_embeds.unsqueeze(2).expand(
            *inputs_embeds.shape[:2], len(self.layers), inputs_embeds.shape[-1]
        )
        return projected if identity is None else projected + identity

    def build_causal_masks_for_cts(self, hidden, attention_mask, position_ids):
        seq = hidden.shape[1]
        return {
            "full": torch.zeros(
                hidden.shape[0], 1, seq, seq,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        }

    def rotary_emb(self, hidden, position_ids, layer_type):
        shape = (*position_ids.shape, hidden.shape[-1])
        return torch.ones(shape, device=hidden.device), torch.zeros(
            shape, device=hidden.device
        )


def _layers_for_module(module: int):
    return [int(module)]


def test_eq3_static_cache_preserves_s0_history_z_order_and_values():
    torch.manual_seed(1)
    lm = _FakeLM()
    ids = torch.tensor([[2, 3]])
    context = encode_prompt_context(lm, ids).with_history(
        torch.full((1, 2, 4), 7.0)
    )
    z = torch.full((2, 4), 11.0)

    reference = prepare_eq3_inputs(lm, z, context)
    static = prepare_eq3_static_inputs(lm, z, context)
    cached = prepare_eq3_inputs(lm, z, context, static_inputs=static)

    assert torch.equal(reference.hidden_states, cached.hidden_states)
    assert torch.equal(reference.per_layer_inputs, cached.per_layer_inputs)
    assert torch.equal(reference.position_ids, cached.position_ids)
    assert torch.equal(cached.hidden_states[:, :2], context.prompt_embeds)
    assert torch.equal(cached.hidden_states[:, 2:4], context.history_latents)
    assert torch.equal(cached.hidden_states[:, 4:], z.unsqueeze(0))


def test_eq3_static_cache_reprojects_only_dynamic_z_per_phi():
    lm = _FakeLM()
    context = encode_prompt_context(lm, torch.tensor([[2, 3]])).with_history(
        torch.ones(1, 2, 4)
    )
    z = torch.ones(2, 4)
    static = prepare_eq3_static_inputs(lm, z, context)
    calls_after_static = lm.project_calls

    prepare_eq3_inputs(lm, z, context, static_inputs=static)
    prepare_eq3_inputs(lm, z + 1, context, static_inputs=static)

    assert lm.project_calls - calls_after_static == 2


def test_sparse_w_batch_matches_independent_scalar_outputs_and_gradients():
    torch.manual_seed(2)
    lm_batch = _FakeLM()
    lm_scalar = _FakeLM()
    lm_scalar.load_state_dict(lm_batch.state_dict())
    ids = torch.tensor([[2, 3]])
    base_batch = encode_prompt_context(lm_batch, ids)
    base_scalar = encode_prompt_context(lm_scalar, ids)
    history = torch.randn(2, 1, 2, 4)
    z_batch = torch.randn(2, 2, 4, requires_grad=True)
    z_scalar = z_batch.detach().clone().requires_grad_(True)
    alpha = torch.tensor(
        [[0.7, 0.2, 0.1, 0.0], [0.0, 0.1, 0.3, 0.6]]
    )

    batch_context = base_batch.with_history(history)
    batch_static = prepare_eq3_static_inputs(lm_batch, z_batch, batch_context)
    out_batch = parallel_sparse_module_forward_batch(
        lm_batch,
        z_batch,
        batch_context,
        alpha,
        _layers_for_module,
        top_k=2,
        static_inputs=batch_static,
    )

    rows = []
    for i in range(2):
        row_context = base_scalar.with_history(history[i])
        row_static = prepare_eq3_static_inputs(
            lm_scalar, z_scalar[i], row_context
        )
        rows.append(
            parallel_sparse_module_forward(
                lm_scalar,
                z_scalar[i],
                row_context,
                alpha[i],
                _layers_for_module,
                top_k=2,
                static_inputs=row_static,
            )
        )
    out_scalar = torch.stack(rows)

    assert torch.allclose(out_batch, out_scalar, atol=1e-6, rtol=1e-6)
    out_batch.sum().backward()
    out_scalar.sum().backward()
    assert torch.allclose(z_batch.grad, z_scalar.grad, atol=1e-6, rtol=1e-6)
    for batch_layer, scalar_layer in zip(lm_batch.layers, lm_scalar.layers):
        assert batch_layer.scale.grad is not None
        assert scalar_layer.scale.grad is not None
        assert torch.allclose(
            batch_layer.scale.grad,
            scalar_layer.scale.grad,
            atol=1e-6,
            rtol=1e-6,
        )


def test_fixed_routing_support_matches_dynamic_without_topk_or_nonzero(monkeypatch):
    """The Phi hot path reuses one ordered support for every W branch."""
    torch.manual_seed(22)
    lm_dynamic = _FakeLM()
    lm_fixed = _FakeLM()
    lm_fixed.load_state_dict(lm_dynamic.state_dict())
    ids = torch.tensor([[2, 3]])
    context_dynamic = encode_prompt_context(lm_dynamic, ids)
    context_fixed = encode_prompt_context(lm_fixed, ids)
    z_dynamic = torch.randn(2, 2, 4, requires_grad=True)
    z_fixed = z_dynamic.detach().clone().requires_grad_(True)
    # Both branches have exactly the same ordered top-2 support.
    alpha = torch.tensor(
        [[0.7, 0.2, 0.1, 0.0], [0.6, 0.3, 0.1, 0.0]]
    )
    out_dynamic = parallel_sparse_module_forward_batch(
        lm_dynamic,
        z_dynamic,
        context_dynamic,
        alpha,
        _layers_for_module,
        top_k=2,
        static_inputs=prepare_eq3_static_inputs(
            lm_dynamic, z_dynamic, context_dynamic
        ),
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("fixed routing Phi recomputed dynamic support")

    monkeypatch.setattr(torch, "topk", _unexpected)
    monkeypatch.setattr(torch, "nonzero", _unexpected)
    out_fixed = parallel_sparse_module_forward_batch(
        lm_fixed,
        z_fixed,
        context_fixed,
        alpha,
        _layers_for_module,
        top_k=2,
        static_inputs=prepare_eq3_static_inputs(lm_fixed, z_fixed, context_fixed),
        module_ids=[0, 1],
        fixed_module_ids=torch.tensor([0, 1]),
    )

    assert torch.allclose(out_fixed, out_dynamic, atol=1e-6, rtol=1e-6)
    out_dynamic.sum().backward()
    out_fixed.sum().backward()
    assert torch.allclose(z_fixed.grad, z_dynamic.grad, atol=1e-6, rtol=1e-6)
    for index, (dynamic_layer, fixed_layer) in enumerate(
        zip(lm_dynamic.layers, lm_fixed.layers)
    ):
        if index in {0, 1}:
            assert dynamic_layer.scale.grad is not None
            assert fixed_layer.scale.grad is not None
            assert torch.allclose(
                fixed_layer.scale.grad,
                dynamic_layer.scale.grad,
                atol=1e-6,
                rtol=1e-6,
            )
        else:
            assert dynamic_layer.scale.grad is None
            assert fixed_layer.scale.grad is None


def test_eq3_static_cache_rejects_shape_drift_and_reuses_static_objects():
    lm = _FakeLM()
    context = encode_prompt_context(lm, torch.tensor([[2, 3]]))
    z = torch.ones(2, 4)
    static = prepare_eq3_static_inputs(lm, z, context)
    cached = prepare_eq3_inputs(lm, z, context, static_inputs=static)

    assert cached.causal_masks is static.causal_masks
    assert cached.position_embeddings is static.position_embeddings
    try:
        prepare_eq3_inputs(lm, torch.ones(3, 4), context, static_inputs=static)
    except ValueError as exc:
        assert "shape changed" in str(exc)
    else:
        raise AssertionError("Eq.3 static cache accepted a different K")


class _AnchorAttention(nn.Module):
    def __init__(self, layer_type: str):
        super().__init__()
        self.store_full_length_kv = True
        self.layer_type = layer_type
        self.head_dim = 2
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()


class _AnchorLayer(nn.Module):
    def __init__(self, layer_type: str):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.self_attn = _AnchorAttention(layer_type)


class _AnchorLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [_AnchorLayer("sliding_attention"), _AnchorLayer("full_attention")]
        )
        self.config = SimpleNamespace(num_hidden_layers=2)


def test_shared_kv_mapping_uses_anchor_layer_indices_not_attention_names():
    lm = _AnchorLM()
    hidden = torch.randn(1, 3, 4)
    cos = torch.ones(1, 3, 2)
    sin = torch.zeros_like(cos)
    shared = _direct_anchor_shared_kv(
        lm,
        hidden,
        {
            "sliding_attention": (cos, sin),
            "full_attention": (cos, sin),
        },
    )
    assert set(shared) == {0, 1}
    assert "sliding_attention" not in shared


def test_shared_kv_consumer_abi_must_match_integer_anchor_indices():
    lm = _AnchorLM()
    lm.layers[1].self_attn.store_full_length_kv = False
    lm.layers[1].self_attn.is_kv_shared_layer = True
    lm.layers[1].self_attn.kv_shared_layer_index = 0
    _validate_shared_kv_abi(lm)

    lm.layers[1].self_attn.kv_shared_layer_index = "sliding_attention"
    try:
        _validate_shared_kv_abi(lm)
    except RuntimeError as exc:
        assert "expected integer" in str(exc)
    else:
        raise AssertionError("string shared-KV ABI was silently accepted")


class _CacheAttention(nn.Module):
    def __init__(self, hidden: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = "full"
        self.is_kv_shared_layer = False
        self.store_full_length_kv = False
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)


class _CacheLayer(nn.Module):
    def __init__(self, hidden: int, layer_idx: int):
        super().__init__()
        self.self_attn = _CacheAttention(hidden, layer_idx)
        self.ple_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        hidden,
        per_layer_input,
        *,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        attn = self.self_attn
        q = attn.q(hidden).unsqueeze(1)
        k = attn.k(hidden).unsqueeze(1)
        v = attn.v(hidden).unsqueeze(1)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, attn.layer_idx)
        scores = torch.matmul(q, k.transpose(-1, -2)) / hidden.shape[-1] ** 0.5
        scores = scores + attention_mask
        mixed = torch.matmul(torch.softmax(scores, dim=-1), v).squeeze(1)
        return (hidden + attn.o(mixed) + self.ple_scale * per_layer_input,)


class _CacheLM(nn.Module):
    def __init__(self, hidden: int = 4, layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList(
            [_CacheLayer(hidden, i) for i in range(layers)]
        )
        self.norm = nn.Identity()
        self.config = SimpleNamespace(
            num_hidden_layers=layers,
            layer_types=["full"] * layers,
        )
        self.unique_layer_types = {"full"}

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        return torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            len(self.layers),
            inputs_embeds.shape[-1],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

    def project_per_layer_inputs(self, inputs_embeds, identity):
        projected = inputs_embeds.unsqueeze(2).expand(
            *inputs_embeds.shape[:2], len(self.layers), inputs_embeds.shape[-1]
        )
        return projected if identity is None else projected + identity

    def build_causal_masks_for_cts(self, hidden, attention_mask, position_ids):
        seq = hidden.shape[1]
        mask = torch.full(
            (hidden.shape[0], 1, seq, seq),
            float("-inf"),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return {"full": torch.triu(mask, diagonal=1)}

    def rotary_emb(self, hidden, position_ids, layer_type):
        shape = (*position_ids.shape, hidden.shape[-1])
        return torch.ones(shape, device=hidden.device), torch.zeros(
            shape, device=hidden.device
        )


def test_module_prefix_suffix_cache_matches_full_output_and_gradients():
    torch.manual_seed(23)
    lm_full = _CacheLM()
    lm_cached = _CacheLM()
    lm_cached.load_state_dict(lm_full.state_dict())
    lm_full.eval()
    lm_cached.eval()
    ids = torch.tensor([[2, 3, 4]])
    context_full = encode_prompt_context(lm_full, ids)
    context_cached = encode_prompt_context(lm_cached, ids)
    z_full = torch.randn(2, 4, requires_grad=True)
    z_cached = z_full.detach().clone().requires_grad_(True)
    alpha = torch.tensor([1.0])

    static_full = prepare_eq3_static_inputs(
        lm_full, z_full, context_full, cache_shared_kv=False
    )
    static_cached = prepare_eq3_static_inputs(
        lm_cached, z_cached, context_cached, cache_shared_kv=False
    )
    static_cached = prepare_eq3_module_prefix_cache(
        lm_cached,
        static_cached,
        [0],
        lambda module: [0, 1],
    )
    assert static_cached.suffix_cache_reason == "ready"
    assert set(static_cached.module_prefix_kv[0]) == {0, 1}

    out_full = parallel_sparse_module_forward(
        lm_full,
        z_full,
        context_full,
        alpha,
        lambda module: [0, 1],
        top_k=1,
        static_inputs=static_full,
        module_ids=[0],
    )
    out_cached = parallel_sparse_module_forward(
        lm_cached,
        z_cached,
        context_cached,
        alpha,
        lambda module: [0, 1],
        top_k=1,
        static_inputs=static_cached,
        module_ids=[0],
    )
    torch.testing.assert_close(out_cached, out_full, atol=1e-6, rtol=1e-6)
    out_full.square().sum().backward()
    out_cached.square().sum().backward()
    torch.testing.assert_close(z_cached.grad, z_full.grad, atol=1e-6, rtol=1e-6)
    for p_full, p_cached in zip(lm_full.parameters(), lm_cached.parameters()):
        if p_full.grad is None and p_cached.grad is None:
            continue
        assert p_full.grad is not None and p_cached.grad is not None
        torch.testing.assert_close(
            p_cached.grad, p_full.grad, atol=1e-6, rtol=1e-6
        )


def test_suffix_cache_training_requires_explicit_deterministic_opt_in():
    lm = _CacheLM().train()
    context = encode_prompt_context(lm, torch.tensor([[2, 3, 4]]))
    z = torch.randn(2, 4)
    static = prepare_eq3_static_inputs(
        lm, z, context, cache_shared_kv=False
    )

    disabled = prepare_eq3_module_prefix_cache(
        lm, static, [0], lambda module: [0, 1]
    )
    assert disabled.suffix_cache_reason == "disabled_in_training"

    lm.active_dropout = nn.Dropout(0.1)
    active = prepare_eq3_module_prefix_cache(
        lm,
        static,
        [0],
        lambda module: [0, 1],
        allow_deterministic_training=True,
    )
    assert active.suffix_cache_reason == "active_dropout_in_training"

    lm.active_dropout.train(False)
    ready = prepare_eq3_module_prefix_cache(
        lm,
        static,
        [0],
        lambda module: [0, 1],
        allow_deterministic_training=True,
    )
    assert ready.suffix_cache_reason == "ready"


def test_suffix_cache_fails_closed_when_backend_has_no_explicit_mask():
    lm = _FakeLM(layers=1)
    lm.eval()
    context = encode_prompt_context(lm, torch.tensor([[2, 3]]))
    z = torch.ones(2, 4)
    static = prepare_eq3_static_inputs(
        lm, z, context, cache_shared_kv=False
    )
    static = replace(static, causal_masks={"full": None})

    cached = prepare_eq3_module_prefix_cache(
        lm, static, [0], _layers_for_module
    )

    assert cached.module_prefix_kv is None
    assert cached.suffix_cache_reason == "unsupported_mask:TypeError"


def test_suffix_cache_fails_closed_for_non_4d_or_short_masks():
    lm = _FakeLM(layers=1)
    lm.eval()
    context = encode_prompt_context(lm, torch.tensor([[2, 3]]))
    z = torch.ones(2, 4)
    static = prepare_eq3_static_inputs(
        lm, z, context, cache_shared_kv=False
    )

    for bad_mask, reason in (
        (torch.zeros(1, 4, 4), "unsupported_mask:TypeError"),
        (torch.zeros(1, 1, 1, 4, 4), "unsupported_mask:TypeError"),
        (torch.zeros(1, 1, 3, 3), "unsupported_mask:ValueError"),
    ):
        cached = prepare_eq3_module_prefix_cache(
            lm,
            replace(static, causal_masks={"full": bad_mask}),
            [0],
            _layers_for_module,
        )
        assert cached.module_prefix_kv is None
        assert cached.suffix_cache_reason == reason


def test_explicit_full_and_sliding_masks_preserve_absolute_causality():
    lm = _FakeLM(layers=2)
    lm.config.layer_types = ["sliding_attention", "full_attention"]
    lm.config.sliding_window = 2
    lm.unique_layer_types = {"sliding_attention", "full_attention"}
    hidden = torch.zeros(1, 4, 4)
    valid = torch.tensor([[1, 1, 1, 1]])
    positions = torch.tensor([[0, 1, 2, 3]])

    masks = _build_explicit_eq3_masks(lm, hidden, valid, positions)

    full = masks["full_attention"][0, 0]
    sliding = masks["sliding_attention"][0, 0]
    assert full.shape == (4, 4)
    assert full[3, 0] == 0 and full[0, 1] < 0
    # Window 2 means query position 3 sees keys 2 and 3, but not 1.
    assert sliding[3, 2] == 0 and sliding[3, 3] == 0
    assert sliding[3, 1] < 0


def test_shared_only_module_skips_all_prefix_query_forwards():
    lm = _FakeLM(layers=1)
    lm.eval()
    lm.layers[0].self_attn.is_kv_shared_layer = True
    context = encode_prompt_context(lm, torch.tensor([[2, 3]]))
    z = torch.ones(2, 4)
    static = prepare_eq3_static_inputs(
        lm, z, context, cache_shared_kv=False
    )
    # A shared-only module consumes anchor K/V; it must not execute prefix
    # queries because their outputs cannot affect any suffix query.
    dummy = torch.zeros(1, 1, 2, 1)
    static = replace(static, shared_prefix_kv={0: (dummy, dummy)})

    cached = prepare_eq3_module_prefix_cache(
        lm, static, [0], _layers_for_module
    )

    assert cached.suffix_cache_reason == "ready"
    assert cached.module_prefix_kv == {0: {}}
    assert lm.layers[0].forward_calls == 0


class _SharedConsumerAttention(nn.Module):
    def __init__(self, hidden: int, anchor_idx: int):
        super().__init__()
        self.store_full_length_kv = False
        self.is_kv_shared_layer = True
        self.kv_shared_layer_index = int(anchor_idx)
        self.layer_type = "full"
        self.head_dim = 2
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _SharedConsumerLayer(nn.Module):
    def __init__(self, hidden: int, anchor_idx: int):
        super().__init__()
        self.self_attn = _SharedConsumerAttention(hidden, anchor_idx)
        self.ple_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        hidden,
        per_layer_input,
        *,
        attention_mask,
        shared_kv_states,
        **kwargs,
    ):
        attn = self.self_attn
        key, value = shared_kv_states[attn.kv_shared_layer_index]
        batch, queries, width = hidden.shape
        query = attn.q_proj(hidden).view(
            batch, queries, -1, attn.head_dim
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (
            attn.head_dim ** 0.5
        )
        scores = scores + attention_mask
        mixed = torch.matmul(torch.softmax(scores, dim=-1), value)
        mixed = mixed.transpose(1, 2).reshape(batch, queries, width)
        return (hidden + attn.o_proj(mixed) + self.ple_scale * per_layer_input,)


class _SharedConsumerLM(nn.Module):
    def __init__(self, hidden: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList(
            [_AnchorLayer("full"), _SharedConsumerLayer(hidden, 0)]
        )
        self.norm = nn.Identity()
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["full", "full"],
        )
        self.unique_layer_types = {"full"}

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        return torch.zeros(
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            len(self.layers),
            inputs_embeds.shape[-1],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

    def project_per_layer_inputs(self, inputs_embeds, identity):
        projected = inputs_embeds.unsqueeze(2).expand(
            *inputs_embeds.shape[:2], len(self.layers), inputs_embeds.shape[-1]
        )
        return projected if identity is None else projected + identity

    def build_causal_masks_for_cts(self, hidden, attention_mask, position_ids):
        seq = hidden.shape[1]
        mask = torch.full(
            (hidden.shape[0], 1, seq, seq),
            float("-inf"),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return {"full": torch.triu(mask, diagonal=1)}

    def rotary_emb(self, hidden, position_ids, layer_type):
        shape = (*position_ids.shape, 2)
        return torch.ones(shape, device=hidden.device), torch.zeros(
            shape, device=hidden.device
        )


def test_shared_consumer_suffix_cache_matches_full_output_and_z_jacobian():
    """Exercise the real integer anchor->shared-consumer data flow."""
    torch.manual_seed(29)
    lm_full = _SharedConsumerLM().eval()
    lm_cached = _SharedConsumerLM().eval()
    lm_cached.load_state_dict(lm_full.state_dict())
    context_full = encode_prompt_context(lm_full, torch.tensor([[2, 3, 4]]))
    context_cached = encode_prompt_context(lm_cached, torch.tensor([[2, 3, 4]]))
    z_full = torch.randn(2, 4, requires_grad=True)
    z_cached = z_full.detach().clone().requires_grad_(True)
    alpha = torch.tensor([1.0])
    module_layers = lambda module: [1]

    static_full = prepare_eq3_static_inputs(
        lm_full, z_full, context_full, cache_shared_kv=False
    )
    static_cached = prepare_eq3_static_inputs(
        lm_cached, z_cached, context_cached, cache_shared_kv=True
    )
    static_cached = prepare_eq3_module_prefix_cache(
        lm_cached, static_cached, [0], module_layers
    )
    assert static_cached.suffix_cache_reason == "ready"
    assert static_cached.module_prefix_kv == {0: {}}

    out_full = parallel_sparse_module_forward(
        lm_full,
        z_full,
        context_full,
        alpha,
        module_layers,
        top_k=1,
        static_inputs=static_full,
        module_ids=[0],
    )
    out_cached = parallel_sparse_module_forward(
        lm_cached,
        z_cached,
        context_cached,
        alpha,
        module_layers,
        top_k=1,
        static_inputs=static_cached,
        module_ids=[0],
    )
    torch.testing.assert_close(out_cached, out_full, atol=1e-6, rtol=1e-6)
    out_full.square().sum().backward()
    out_cached.square().sum().backward()
    torch.testing.assert_close(z_cached.grad, z_full.grad, atol=1e-6, rtol=1e-6)
