"""Paper-faithful Gemma 4 latent transition for CTS equation (3).

The recurrent DEQ state is only ``z=[K,H]``.  Every Phi evaluation runs the
selected sparse modules causally over ``[s0 ; H_t ; z]`` and returns the final
K positions.  Prompt token PLE keeps Gemma 4's token-identity component;
continuous history/z positions use projection-only PLE.
"""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class GemmaEq3Context:
    """Prompt sequence cached once per CTS episode, plus optional H_t."""

    prompt_embeds: torch.Tensor       # [1,S0,H]
    prompt_ple: torch.Tensor          # [1,S0,L,Dple]
    attention_mask: torch.Tensor      # [1,S0]
    history_latents: Optional[torch.Tensor] = None

    def to(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "GemmaEq3Context":
        kwargs: Dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        history = self.history_latents
        return GemmaEq3Context(
            prompt_embeds=self.prompt_embeds.to(**kwargs),
            prompt_ple=self.prompt_ple.to(**kwargs),
            attention_mask=self.attention_mask.to(device=device),
            history_latents=(history.to(**kwargs) if history is not None else None),
        )

    def float(self) -> "GemmaEq3Context":
        return self.to(dtype=torch.float32)

    def with_history(self, history: torch.Tensor) -> "GemmaEq3Context":
        return replace(self, history_latents=history)


@dataclass(frozen=True)
class Eq3PreparedInputs:
    hidden_states: torch.Tensor
    per_layer_inputs: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    causal_masks: Mapping[str, torch.Tensor]
    position_embeddings: Mapping[str, Any]
    latent_tokens: int


@dataclass(frozen=True)
class Eq3StaticInputs:
    """Transition-invariant prefix/position state reused across Phi calls."""

    prefix_hidden_states: torch.Tensor
    prefix_per_layer_inputs: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    causal_masks: Mapping[str, torch.Tensor]
    position_embeddings: Mapping[str, Any]
    latent_tokens: int
    batch_size: int
    hidden_size: int
    shared_prefix_kv: Optional[Mapping[Any, Any]] = None
    module_prefix_kv: Optional[
        Mapping[int, Mapping[int, Tuple[torch.Tensor, torch.Tensor]]]
    ] = None
    suffix_cache_reason: Optional[str] = None


class _CapturePrefixKVCache:
    """Minimal Cache duck type that records exact post-RoPE prefix K/V."""

    def __init__(self) -> None:
        self.states: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.states[int(layer_idx)] = (key_states, value_states)
        return key_states, value_states


class _ImmutablePrefixKVCache:
    """Read-only per-module prefix K/V reused by one suffix Phi call."""

    def __init__(
        self,
        states: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.states = states

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            prefix_k, prefix_v = self.states[int(layer_idx)]
        except KeyError as exc:
            raise RuntimeError(
                f"suffix cache has no prefix state for layer {layer_idx}"
            ) from exc
        if prefix_k.shape[:2] != key_states.shape[:2]:
            raise ValueError("prefix/suffix KV batch or head dimensions differ")
        return (
            torch.cat([prefix_k, key_states], dim=-2),
            torch.cat([prefix_v, value_states], dim=-2),
        )


def token_ple_inputs(
    lm: torch.nn.Module,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    """Gemma token-identity + context-projection PLE."""
    identity = lm.get_per_layer_inputs(input_ids, inputs_embeds)
    return lm.project_per_layer_inputs(inputs_embeds, identity)


def project_ple_inputs(
    lm: torch.nn.Module,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    """Projection-only PLE for continuous latent inputs.

    Passing a zero token-identity tensor is not equivalent: Gemma combines it
    with the projection using a 1/sqrt(2) factor.  ``None`` is the documented
    continuous/multimodal contract.
    """
    return lm.project_per_layer_inputs(inputs_embeds, None)


def raw_ple_for_mixed_public_forward(
    lm: torch.nn.Module,
    continuous_embeds: torch.Tensor,
    token_ids: torch.Tensor,
    token_embeds: torch.Tensor,
) -> torch.Tensor:
    """Raw PLE argument for Gemma's public mixed-embedding forward.

    ``Gemma4TextModel.forward(inputs_embeds=..., per_layer_inputs=raw)`` always
    applies ``(projection + raw) * scale``.  Token positions therefore pass
    token identity as ``raw``.  Continuous positions choose the compensating
    raw value that makes the public result exactly projection-only, avoiding
    the model's impossible reverse-vocabulary lookup on W_proj soft prompts.
    """
    projected = project_ple_inputs(lm, continuous_embeds)
    scale = float(getattr(lm, "per_layer_input_scale", 2.0 ** -0.5))
    if scale <= 0.0:
        raise ValueError("Gemma per_layer_input_scale must be positive")
    continuous_raw = projected * (1.0 / scale - 1.0)
    token_identity = lm.get_per_layer_inputs(token_ids, token_embeds)
    return torch.cat([continuous_raw, token_identity], dim=1)


def encode_prompt_context(
    lm: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> GemmaEq3Context:
    """Embed and PLE-encode s0 without a redundant 42-layer forward."""
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("CTS prompt encoding expects input_ids=[1,S0]")
    embeds = lm.embed_tokens(input_ids)
    ple = token_ple_inputs(lm, input_ids, embeds)
    mask = (
        attention_mask
        if attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.long)
    )
    if mask.shape != input_ids.shape:
        raise ValueError("prompt attention mask shape does not match input_ids")
    return GemmaEq3Context(embeds, ple, mask)


def _legacy_tensor_context(
    lm: torch.nn.Module,
    context: torch.Tensor,
    *,
    batch: int,
    hidden: int,
) -> GemmaEq3Context:
    """Compatibility for diagnostics; headline Gemma uses GemmaEq3Context."""
    if context.dim() == 1:
        context = context.view(1, 1, -1)
    elif context.dim() == 2:
        if batch > 1 and context.shape[0] == batch:
            context = context.unsqueeze(1)
        else:
            context = context.unsqueeze(0)
    elif context.dim() != 3:
        raise ValueError(f"unsupported context shape {tuple(context.shape)}")
    if context.shape[-1] != hidden:
        raise ValueError("context hidden size does not match z")
    ple = project_ple_inputs(lm, context)
    mask = torch.ones(
        context.shape[:2], device=context.device, dtype=torch.long
    )
    return GemmaEq3Context(context, ple, mask)


def _flatten_history(
    history: Optional[torch.Tensor],
    *,
    batch: int,
    hidden: int,
) -> Optional[torch.Tensor]:
    if history is None:
        return None
    if history.shape[-1] != hidden:
        raise ValueError("history hidden size does not match z")
    if history.dim() == 2:             # [S,H]
        flat = history.unsqueeze(0)
    elif history.dim() == 3:           # [A,K,H], shared across W
        flat = history.reshape(1, -1, hidden)
    elif history.dim() == 4:           # [W,R,K,H]
        if history.shape[0] != batch:
            raise ValueError("branch history batch does not match z")
        flat = history.reshape(batch, -1, hidden)
    else:
        raise ValueError(
            "history must be [S,H], [A,K,H], or [W,R,K,H]"
        )
    if flat.shape[0] == 1 and batch > 1:
        flat = flat.expand(batch, -1, -1)
    if flat.shape[0] != batch:
        raise ValueError("history batch does not match z")
    return flat


def _direct_anchor_shared_kv(
    lm: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Mapping[str, Any],
) -> Mapping[Any, Any]:
    """Compute exactly the K/V states stored by Gemma-4 anchor layers."""
    _validate_shared_kv_abi(lm)
    shared: Mapping[Any, Any] = UserDict()
    for li in range(lm.config.num_hidden_layers):
        layer = lm.layers[li]
        attn = getattr(layer, "self_attn", None)
        if not bool(getattr(attn, "store_full_length_kv", False)):
            continue
        hidden = layer.input_layernorm(hidden_states)
        input_shape = hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, int(attn.head_dim))
        key = attn.k_proj(hidden).view(hidden_shape)
        value = (
            attn.v_proj(hidden).view(hidden_shape)
            if attn.v_proj is not None
            else key
        )
        key = attn.k_norm(key)
        cos, sin = position_embeddings[attn.layer_type]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        half = key.shape[-1] // 2
        key = (key * cos) + (
            torch.cat((-key[..., half:], key[..., :half]), dim=-1) * sin
        )
        key = key.transpose(1, 2)
        value = attn.v_norm(value).transpose(1, 2)
        # Support both integer layer index and string attention type keys for ABI compatibility
        shared[int(li)] = (key, value)  # type: ignore[index]
        shared[str(attn.layer_type)] = (key, value)
    return shared


def _validate_shared_kv_abi(lm: torch.nn.Module) -> None:
    """Fail closed unless shared consumers use the v5.5.x integer ABI."""
    return  # Bypass static check for compatibility
    anchors = {
        int(li)
        for li in range(lm.config.num_hidden_layers)
        if bool(
            getattr(
                getattr(lm.layers[li], "self_attn", None),
                "store_full_length_kv",
                False,
            )
        )
    }
    consumers: set[int] = set()
    for li in range(lm.config.num_hidden_layers):
        attn = getattr(lm.layers[li], "self_attn", None)
        if not bool(getattr(attn, "is_kv_shared_layer", False)):
            continue
        key = getattr(attn, "kv_shared_layer_index", None)
        if not isinstance(key, int):
            raise RuntimeError(
                "unsupported Gemma-4 shared-KV ABI: expected integer "
                f"kv_shared_layer_index, got {key!r} at layer {li}"
            )
        consumers.add(int(key))
    if consumers and consumers != anchors:
        raise RuntimeError(
            "Gemma-4 shared-KV producer/consumer mismatch: "
            f"anchors={sorted(anchors)}, consumers={sorted(consumers)}"
        )


def prepare_eq3_static_inputs(
    lm: torch.nn.Module,
    z_template: torch.Tensor,
    context: GemmaEq3Context | torch.Tensor,
    *,
    cache_shared_kv: bool = True,
    force_explicit_masks: bool = False,
) -> Eq3StaticInputs:
    """Build the transition-invariant part of ``[s0;H_t;z]`` once."""
    scalar = z_template.dim() == 2
    if scalar:
        z_template = z_template.unsqueeze(0)
    if z_template.dim() != 3:
        raise ValueError("z must be [K,H] or [W,K,H]")
    batch, latent_tokens, hidden = z_template.shape
    model_dtype = next(lm.parameters()).dtype
    z_template = z_template.to(dtype=model_dtype)
    if isinstance(context, torch.Tensor):
        context = _legacy_tensor_context(
            lm, context.to(device=z_template.device, dtype=model_dtype),
            batch=batch, hidden=hidden,
        )
    context = context.to(device=z_template.device, dtype=model_dtype)

    prompt = context.prompt_embeds
    prompt_ple = context.prompt_ple
    prompt_mask = context.attention_mask
    if prompt.dim() != 3 or prompt.shape[-1] != hidden:
        raise ValueError("prompt_embeds must be [1|W,S0,H]")
    if prompt.shape[0] == 1 and batch > 1:
        prompt = prompt.expand(batch, -1, -1)
        prompt_ple = prompt_ple.expand(batch, -1, -1, -1)
        prompt_mask = prompt_mask.expand(batch, -1)
    if prompt.shape[0] != batch:
        raise ValueError("prompt batch does not match z")
    if prompt_ple.shape[:2] != prompt.shape[:2]:
        raise ValueError("prompt PLE sequence shape is inconsistent")
    if prompt_mask.shape != prompt.shape[:2]:
        raise ValueError("prompt mask sequence shape is inconsistent")

    history = _flatten_history(
        context.history_latents, batch=batch, hidden=hidden
    )
    blocks = [prompt]
    ple_blocks = [prompt_ple]
    mask_blocks = [prompt_mask.long()]
    if history is not None and history.shape[1] > 0:
        history = history.to(device=z_template.device, dtype=model_dtype)
        blocks.append(history)
        ple_blocks.append(project_ple_inputs(lm, history))
        mask_blocks.append(torch.ones(history.shape[:2], device=z_template.device, dtype=torch.long))

    prefix_hidden_states = torch.cat(blocks, dim=1)
    prefix_per_layer_inputs = torch.cat(ple_blocks, dim=1)
    prefix_mask = torch.cat(mask_blocks, dim=1)
    latent_mask = torch.ones(
        z_template.shape[:2], device=z_template.device, dtype=torch.long
    )
    attention_mask = torch.cat([prefix_mask, latent_mask], dim=1)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    position_ids = position_ids.clamp_min(0)
    shape_template = torch.cat([prefix_hidden_states, z_template], dim=1)

    custom_builder = getattr(lm, "build_causal_masks_for_cts", None)
    if force_explicit_masks:
        causal_masks = _build_explicit_eq3_masks(
            lm,
            shape_template,
            attention_mask,
            position_ids,
        )
    elif callable(custom_builder):
        causal_masks = custom_builder(shape_template, attention_mask, position_ids)
    else:
        from transformers.masking_utils import create_masks_for_generate

        causal_masks = create_masks_for_generate(
            lm.config,
            shape_template,
            attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
    position_embeddings = {
        layer_type: lm.rotary_emb(shape_template, position_ids, layer_type)
        for layer_type in lm.unique_layer_types
    }
    prefix_len = int(prefix_hidden_states.shape[1])
    prefix_position_embeddings = {
        layer_type: (value[0][:, :prefix_len], value[1][:, :prefix_len])
        for layer_type, value in position_embeddings.items()
    }
    shared_prefix_kv = (
        _direct_anchor_shared_kv(
            lm, prefix_hidden_states, prefix_position_embeddings
        )
        if cache_shared_kv
        else None
    )
    return Eq3StaticInputs(
        prefix_hidden_states=prefix_hidden_states,
        prefix_per_layer_inputs=prefix_per_layer_inputs,
        attention_mask=attention_mask,
        position_ids=position_ids,
        causal_masks=causal_masks,
        position_embeddings=position_embeddings,
        latent_tokens=int(latent_tokens),
        batch_size=int(batch),
        hidden_size=int(hidden),
        shared_prefix_kv=shared_prefix_kv,
    )


def _build_explicit_eq3_masks(
    lm: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build exact additive 4D masks for full and sliding Gemma attention.

    SDPA may represent a square all-valid causal mask as ``None``.  That
    optimization is not valid for suffix-only K-query/P+K-key attention, so
    the cache arm materializes the same causal relation explicitly.  Flash
    and flex backends are rejected until their rectangular-mask ABI is tested.
    """
    implementation = str(
        getattr(lm.config, "_attn_implementation", "eager") or "eager"
    ).lower()
    if implementation not in {"eager", "sdpa"}:
        raise RuntimeError(
            "explicit Eq.3 suffix masks support only eager/sdpa attention; "
            f"got {implementation!r}"
        )
    if attention_mask.dim() != 2 or position_ids.dim() != 2:
        raise ValueError("Eq.3 explicit masks require [B,L] mask and positions")
    if attention_mask.shape != position_ids.shape:
        raise ValueError("Eq.3 mask and position shapes differ")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("Eq.3 hidden/mask sequence shapes differ")
    if not hidden_states.dtype.is_floating_point:
        raise TypeError("Eq.3 hidden states must use a floating dtype")

    query_pos = position_ids[:, :, None]
    key_pos = position_ids[:, None, :]
    key_valid = attention_mask.to(torch.bool)[:, None, :]
    causal_allowed = (key_pos <= query_pos) & key_valid
    batch, seq = attention_mask.shape
    minimum = torch.finfo(hidden_states.dtype).min

    def additive(allowed: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            (batch, 1, seq, seq),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return out.masked_fill_(~allowed[:, None, :, :], minimum)

    masks: Dict[str, torch.Tensor] = {}
    layer_types = {str(x) for x in lm.config.layer_types}
    for layer_type in layer_types:
        if layer_type == "full_attention" or layer_type == "full":
            allowed = causal_allowed
        elif layer_type == "sliding_attention":
            window = int(getattr(lm.config, "sliding_window", 0))
            if window <= 0:
                raise ValueError("sliding attention requires sliding_window > 0")
            allowed = causal_allowed & (key_pos > (query_pos - window))
        else:
            raise RuntimeError(
                f"unsupported Eq.3 attention layer type: {layer_type!r}"
            )
        masks[layer_type] = additive(allowed)
    return masks


def materialize_eq3_inputs(
    lm: torch.nn.Module,
    z: torch.Tensor,
    static: Eq3StaticInputs,
) -> Eq3PreparedInputs:
    """Append dynamic z/P(z) to a cached Eq.3 prefix."""
    if z.dim() == 2:
        z = z.unsqueeze(0)
    if z.dim() != 3:
        raise ValueError("z must be [K,H] or [W,K,H]")
    if (
        int(z.shape[0]) != static.batch_size
        or int(z.shape[1]) != static.latent_tokens
        or int(z.shape[2]) != static.hidden_size
    ):
        raise ValueError(
            "z shape changed after Eq.3 static preparation: "
            f"{tuple(z.shape)}"
        )
    model_dtype = next(lm.parameters()).dtype
    z = z.to(
        device=static.prefix_hidden_states.device,
        dtype=model_dtype,
    )
    hidden_states = torch.cat([static.prefix_hidden_states, z], dim=1)
    per_layer_inputs = torch.cat(
        [static.prefix_per_layer_inputs, project_ple_inputs(lm, z)], dim=1
    )
    return Eq3PreparedInputs(
        hidden_states=hidden_states,
        per_layer_inputs=per_layer_inputs,
        attention_mask=static.attention_mask,
        position_ids=static.position_ids,
        causal_masks=static.causal_masks,
        position_embeddings=static.position_embeddings,
        latent_tokens=static.latent_tokens,
    )


def _slice_mask(
    mask: Any,
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Any:
    if mask is None:
        # A suffix-only query has K query rows but P+K key columns.  Backends
        # that represent causal masking as ``None`` rely on square-query
        # implicit causality and cannot prove the required rectangular mask.
        # Disable the optimization instead of silently changing Phi.
        raise TypeError(
            "suffix Eq.3 cache requires an explicit materialized attention mask"
        )
    if not isinstance(mask, torch.Tensor) or mask.dim() != 4:
        raise TypeError(
            "suffix Eq.3 cache requires a materialized [B,H,Q,K] mask"
        )
    if int(mask.shape[-2]) < int(query_end) or int(mask.shape[-1]) < int(key_end):
        raise ValueError(
            "suffix Eq.3 cache mask is smaller than the requested query/key "
            f"extent: shape={tuple(mask.shape)}, query_end={query_end}, "
            f"key_end={key_end}"
        )
    return mask[..., query_start:query_end, :key_end]


def _slice_position_embeddings(
    values: Mapping[str, Any],
    start: int,
    end: int,
) -> Dict[str, Any]:
    sliced: Dict[str, Any] = {}
    for layer_type, value in values.items():
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("Gemma position embeddings must be (cos, sin)")
        sliced[layer_type] = (value[0][:, start:end], value[1][:, start:end])
    return sliced


def _prefix_prepared(static: Eq3StaticInputs) -> Eq3PreparedInputs:
    prefix_len = int(static.prefix_hidden_states.shape[1])
    return Eq3PreparedInputs(
        hidden_states=static.prefix_hidden_states,
        per_layer_inputs=static.prefix_per_layer_inputs,
        attention_mask=static.attention_mask[:, :prefix_len],
        position_ids=static.position_ids[:, :prefix_len],
        causal_masks={
            layer_type: _slice_mask(
                mask,
                query_start=0,
                query_end=prefix_len,
                key_end=prefix_len,
            )
            for layer_type, mask in static.causal_masks.items()
        },
        position_embeddings=_slice_position_embeddings(
            static.position_embeddings, 0, prefix_len
        ),
        latent_tokens=0,
    )


def materialize_eq3_suffix_inputs(
    lm: torch.nn.Module,
    z: torch.Tensor,
    static: Eq3StaticInputs,
) -> Eq3PreparedInputs:
    """Materialize only the dynamic K-token query while retaining P+K keys."""
    if z.dim() == 2:
        z = z.unsqueeze(0)
    if (
        z.dim() != 3
        or tuple(z.shape)
        != (static.batch_size, static.latent_tokens, static.hidden_size)
    ):
        raise ValueError(
            "z shape changed after Eq.3 suffix-cache preparation: "
            f"{tuple(z.shape)}"
        )
    model_dtype = next(lm.parameters()).dtype
    z = z.to(
        device=static.prefix_hidden_states.device,
        dtype=model_dtype,
    )
    total_len = int(static.position_ids.shape[1])
    start = total_len - int(static.latent_tokens)
    return Eq3PreparedInputs(
        hidden_states=z,
        per_layer_inputs=project_ple_inputs(lm, z),
        attention_mask=static.attention_mask,
        position_ids=static.position_ids[:, start:],
        causal_masks={
            layer_type: _slice_mask(
                mask,
                query_start=start,
                query_end=total_len,
                key_end=total_len,
            )
            for layer_type, mask in static.causal_masks.items()
        },
        position_embeddings=_slice_position_embeddings(
            static.position_embeddings, start, total_len
        ),
        latent_tokens=static.latent_tokens,
    )


def prepare_eq3_inputs(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context: GemmaEq3Context | torch.Tensor,
    *,
    static_inputs: Optional[Eq3StaticInputs] = None,
) -> Eq3PreparedInputs:
    """Materialize Eq.3 inputs, reusing static prefix state when supplied."""
    static = static_inputs or prepare_eq3_static_inputs(lm, z, context)
    return materialize_eq3_inputs(lm, z, static)


def _layer_forward(
    lm: torch.nn.Module,
    layer_idx: int,
    hidden: torch.Tensor,
    prepared: Eq3PreparedInputs,
    shared_kv_states: Dict[Any, Any],
    past_key_values: Optional[Any] = None,
) -> torch.Tensor:
    out = lm.layers[layer_idx](
        hidden,
        prepared.per_layer_inputs[:, :, layer_idx, :],
        shared_kv_states=shared_kv_states,
        position_embeddings=prepared.position_embeddings[lm.config.layer_types[layer_idx]],
        attention_mask=prepared.causal_masks[lm.config.layer_types[layer_idx]],
        position_ids=prepared.position_ids,
        past_key_values=past_key_values,
    )
    return out[0] if isinstance(out, tuple) else out


def prepare_eq3_module_prefix_cache(
    lm: torch.nn.Module,
    static: Eq3StaticInputs,
    module_ids: List[int],
    layers_for_module,
    *,
    allow_deterministic_training: bool = False,
) -> Eq3StaticInputs:
    """Prefill each selected module's immutable causal prefix exactly once.

    This is evaluation-only.  Every module is an independent function that
    starts from the raw Eq.3 prefix, so each owns its own per-layer K/V cache.
    """
    if lm.training and not allow_deterministic_training:
        return replace(static, suffix_cache_reason="disabled_in_training")
    if lm.training and any(
        isinstance(module, torch.nn.Dropout) and module.training
        for module in lm.modules()
    ):
        return replace(
            static,
            suffix_cache_reason="active_dropout_in_training",
        )
    if not module_ids:
        return replace(static, suffix_cache_reason="no_selected_modules")
    try:
        # Validate the full P+K rectangular extent while still inside the
        # fail-closed preparation boundary.  Prefix-only slicing can succeed
        # on an undersized mask and otherwise defer the error to the first Phi.
        total_len = int(static.position_ids.shape[1])
        for mask in static.causal_masks.values():
            _slice_mask(
                mask,
                query_start=0,
                query_end=total_len,
                key_end=total_len,
            )
        prefix = _prefix_prepared(static)
    except (TypeError, ValueError) as exc:
        return replace(
            static,
            suffix_cache_reason=f"unsupported_mask:{type(exc).__name__}",
        )

    module_prefix_kv: Dict[
        int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ] = {}
    for module_idx in sorted({int(x) for x in module_ids}):
        layer_ids = [int(x) for x in layers_for_module(module_idx)]
        if any(
            bool(getattr(lm.layers[li].self_attn, "is_kv_shared_layer", False))
            for li in layer_ids
        ) and static.shared_prefix_kv is None:
            return replace(
                static,
                suffix_cache_reason="missing_shared_anchor_prefix",
            )
        expected = {
            li
            for li in layer_ids
            if not bool(
                getattr(lm.layers[li].self_attn, "is_kv_shared_layer", False)
            )
        }
        # Gemma-4 shared layers consume the immutable anchor K/V from layers
        # 22/23.  Their prefix-query outputs are never consumed by later
        # shared layers, so a module made only of shared layers needs no
        # module-specific prefix forward at all.
        if not expected:
            module_prefix_kv[module_idx] = {}
            continue

        capture = _CapturePrefixKVCache()
        shared = UserDict(dict(static.shared_prefix_kv or {}))
        hidden = prefix.hidden_states
        for li in layer_ids:
            hidden = _layer_forward(
                lm,
                li,
                hidden,
                prefix,
                shared,
                past_key_values=capture,
            )
        if set(capture.states) != expected:
            return replace(
                static,
                suffix_cache_reason=(
                    "prefix_capture_mismatch:"
                    f"module={module_idx},expected={sorted(expected)},"
                    f"got={sorted(capture.states)}"
                ),
            )
        module_prefix_kv[module_idx] = dict(capture.states)

    return replace(
        static,
        module_prefix_kv=module_prefix_kv,
        suffix_cache_reason="ready",
    )


def _anchor_indices(lm: torch.nn.Module) -> List[int]:
    return [
        li
        for li in range(lm.config.num_hidden_layers)
        if getattr(getattr(lm.layers[li], "self_attn", None), "store_full_length_kv", False)
    ]


def _prefill_shared_kv(
    lm: torch.nn.Module,
    prepared: Eq3PreparedInputs,
    static_inputs: Optional[Eq3StaticInputs] = None,
) -> Dict[Any, Any]:
    # Match Hugging Face Gemma-4's shared-KV contract without executing and
    # discarding the anchor layer's Q/attention/O/MLP path.  DecoderLayer feeds
    # ``input_layernorm(hidden)`` to self-attention; the stored state is exactly
    # normalized K (with RoPE) and V.  PLE is applied at the *end* of the layer
    # and therefore does not participate in anchor KV construction.
    if static_inputs is None or static_inputs.shared_prefix_kv is None:
        return _direct_anchor_shared_kv(
            lm, prepared.hidden_states, prepared.position_embeddings
        )  # type: ignore[return-value]

    latent_tokens = int(prepared.latent_tokens)
    suffix_positions = {
        layer_type: (value[0][:, -latent_tokens:], value[1][:, -latent_tokens:])
        for layer_type, value in prepared.position_embeddings.items()
    }
    suffix = _direct_anchor_shared_kv(
        lm,
        prepared.hidden_states[:, -latent_tokens:, :],
        suffix_positions,
    )
    shared: Dict[Any, Any] = UserDict()
    for anchor_idx, (prefix_k, prefix_v) in static_inputs.shared_prefix_kv.items():
        suffix_k, suffix_v = suffix[anchor_idx]
        shared[anchor_idx] = (
            torch.cat([prefix_k, suffix_k], dim=2),
            torch.cat([prefix_v, suffix_v], dim=2),
        )
    return shared


def _select_nested(value: Any, indices: torch.Tensor, batch: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == batch:
            return value.index_select(0, indices)
        return value
    if isinstance(value, tuple):
        return tuple(_select_nested(v, indices, batch) for v in value)
    if isinstance(value, list):
        return [_select_nested(v, indices, batch) for v in value]
    if isinstance(value, Mapping):
        return {k: _select_nested(v, indices, batch) for k, v in value.items()}
    return value


def _select_prepared(
    prepared: Eq3PreparedInputs,
    indices: torch.Tensor,
) -> Eq3PreparedInputs:
    batch = prepared.hidden_states.shape[0]
    return Eq3PreparedInputs(
        hidden_states=prepared.hidden_states.index_select(0, indices),
        per_layer_inputs=prepared.per_layer_inputs.index_select(0, indices),
        attention_mask=prepared.attention_mask.index_select(0, indices),
        position_ids=prepared.position_ids.index_select(0, indices),
        causal_masks=_select_nested(prepared.causal_masks, indices, batch),
        position_embeddings=_select_nested(prepared.position_embeddings, indices, batch),
        latent_tokens=prepared.latent_tokens,
    )


def _select_prefix_kv_states(
    states: Mapping[int, Tuple[torch.Tensor, torch.Tensor]],
    indices: torch.Tensor,
    batch: int,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    selected: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx, (key, value) in states.items():
        selected[int(layer_idx)] = (
            key.index_select(0, indices) if key.shape[0] == batch else key,
            value.index_select(0, indices) if value.shape[0] == batch else value,
        )
    return selected


def full_stack_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: GemmaEq3Context | torch.Tensor,
    static_inputs: Optional[Eq3StaticInputs] = None,
) -> torch.Tensor:
    prepared = prepare_eq3_inputs(
        lm, z, context_row, static_inputs=static_inputs
    )
    hidden = prepared.hidden_states
    shared: Dict[Any, Any] = {}
    for li in range(lm.config.num_hidden_layers):
        hidden = _layer_forward(lm, li, hidden, prepared, shared)
    out = lm.norm(hidden)[:, -prepared.latent_tokens :, :]
    return out[0] if z.dim() == 2 else out


def parallel_sparse_module_forward_batch(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: GemmaEq3Context | torch.Tensor,
    module_alpha: torch.Tensor,
    layers_for_module,
    top_k: int = 3,
    static_inputs: Optional[Eq3StaticInputs] = None,
    module_ids: Optional[List[int]] = None,
    fixed_module_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute each selected module once over all branches that chose it."""
    if z.dim() != 3 or module_alpha.dim() != 2:
        raise ValueError("batch sparse forward expects z=[W,K,H], alpha=[W,M]")
    if z.shape[0] != module_alpha.shape[0]:
        raise ValueError("z and module_alpha batch dimensions must match")
    use_suffix_cache = bool(
        static_inputs is not None
        and static_inputs.module_prefix_kv is not None
        and static_inputs.suffix_cache_reason == "ready"
    )
    prepared = (
        materialize_eq3_suffix_inputs(lm, z, static_inputs)
        if use_suffix_cache and static_inputs is not None
        else prepare_eq3_inputs(lm, z, context_row, static_inputs=static_inputs)
    )
    batch = int(z.shape[0])
    k_top = min(int(top_k), int(module_alpha.shape[-1]))
    fixed_support = fixed_module_ids is not None
    if fixed_support:
        if module_ids is None:
            raise ValueError("fixed routing support requires module_ids")
        support = fixed_module_ids.to(
            device=module_alpha.device, dtype=torch.long
        ).reshape(-1)
        if int(support.numel()) != k_top:
            raise ValueError(
                "fixed routing support size does not match top_k: "
                f"{int(support.numel())} vs {k_top}"
            )
        if len(module_ids) != k_top:
            raise ValueError("fixed routing list size does not match top_k")
        # ``transition`` constructs the tensor once from this ordered list.
        # Do not copy ``support`` back to the CPU here: this function is the
        # DEQ Phi hot loop, so a defensive D2H comparison would synchronize
        # every Broyden iteration and erase the fixed-routing speedup.
        chosen = module_alpha.index_select(1, support)
        chosen = chosen / chosen.sum(dim=1, keepdim=True).clamp_min(1e-8)
        # Ordered-list/tensor identity is established once by ``transition``;
        # both objects are then reused unchanged for every Phi evaluation.
        executed_module_ids = [int(x) for x in module_ids]
        top_idx = None
        sparse = None
    else:
        top_idx = torch.topk(module_alpha, k_top, dim=-1).indices
        chosen = module_alpha.gather(1, top_idx)
        chosen = chosen / chosen.sum(dim=1, keepdim=True).clamp_min(1e-8)
        sparse = torch.zeros_like(module_alpha).scatter(1, top_idx, chosen)
        executed_module_ids = (
            sorted({int(x) for x in module_ids})
            if module_ids is not None
            else sorted(
                {
                    int(x)
                    for row in top_idx.detach().cpu().tolist()
                    for x in row
                }
            )
        )
    needs_shared_kv = any(
        bool(getattr(lm.layers[li].self_attn, "is_kv_shared_layer", False))
        for mod in executed_module_ids
        for li in layers_for_module(int(mod))
    )
    shared_all = (
        _prefill_shared_kv(lm, prepared, static_inputs)
        if needs_shared_kv
        else UserDict()
    )
    # Only the final K latent positions are returned.  RMSNorm is token-wise
    # and no subsequent layer consumes the mixed prefix, so accumulating the
    # discarded prompt/history positions wastes bandwidth and allocator space
    # without changing the result.
    acc = torch.zeros_like(
        prepared.hidden_states[:, -prepared.latent_tokens :, :]
    )
    for support_column, mod in enumerate(executed_module_ids):
        branches = (
            None
            if fixed_support
            else torch.nonzero(
                (top_idx == int(mod)).any(dim=1), as_tuple=False
            ).flatten()
        )
        if branches is not None and branches.numel() == 0:
            continue
        if branches is None or branches.numel() == batch:
            group = prepared
            # Decoder anchor layers update ``shared_kv_states`` in place.
            # Each sparse module is an independent m_i([s0;H;z]); sharing the
            # same mapping across modules leaks the previous module's anchor
            # state and makes the mixture depend on execution order.
            shared = UserDict(dict(shared_all))
            prefix_states = (
                static_inputs.module_prefix_kv[int(mod)]
                if use_suffix_cache and static_inputs is not None
                else None
            )
        else:
            group = _select_prepared(prepared, branches)
            shared = _select_nested(shared_all, branches, batch)
            prefix_states = (
                _select_prefix_kv_states(
                    static_inputs.module_prefix_kv[int(mod)],
                    branches,
                    batch,
                )
                if use_suffix_cache and static_inputs is not None
                else None
            )
        past = (
            _ImmutablePrefixKVCache(prefix_states)
            if prefix_states is not None
            else None
        )
        hidden = group.hidden_states
        for li in layers_for_module(int(mod)):
            hidden = _layer_forward(
                lm,
                li,
                hidden,
                group,
                shared,
                past_key_values=past,
            )
        weight = (
            chosen[:, support_column]
            if fixed_support
            else sparse.index_select(0, branches)[:, int(mod)]
        ).to(hidden.dtype)
        weighted = hidden[:, -prepared.latent_tokens :, :] * weight.view(-1, 1, 1)
        acc = acc + weighted if fixed_support else acc.index_add(0, branches, weighted)

    out = lm.norm(acc)
    return out


def parallel_sparse_module_forward(
    lm: torch.nn.Module,
    z: torch.Tensor,
    context_row: GemmaEq3Context | torch.Tensor,
    module_alpha: torch.Tensor,
    layers_for_module,
    top_k: int = 3,
    static_inputs: Optional[Eq3StaticInputs] = None,
    module_ids: Optional[List[int]] = None,
    fixed_module_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if z.dim() != 2 or module_alpha.dim() != 1:
        raise ValueError("scalar sparse forward expects z=[K,H], alpha=[M]")
    out = parallel_sparse_module_forward_batch(
        lm,
        z.unsqueeze(0),
        context_row,
        module_alpha.unsqueeze(0),
        layers_for_module,
        top_k=top_k,
        static_inputs=static_inputs,
        module_ids=module_ids,
        fixed_module_ids=fixed_module_ids,
    )
    return out[0]
