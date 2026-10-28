"""Gemma 4 E4B backbone for CTS (paper §4.3, §5, §6).

Key components:
- encode_context: text -> cached canonical prompt embeddings + token PLE
- deq_step: configurable DEQ map (blend/parallel/full)
- routing_proj: W_g [19, H] for sparse module routing (Eq. 3)
- w_proj: Wproj [d, d_model] learned projection (§4.3)
- decode_from_z_star: Wproj soft-prompt prefix -> frozen AR pass (§4.3)

Paper §4.3: "Terminal z* is projected via Wproj in R^{d x d_model} as a
soft-prompt prefix into the frozen Gemma 4 decoder for one autoregressive
pass (~1.2 s)."

Paper §6: Wproj is trained jointly in Stage 1.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from cts.backbone.protocol import BaseCTSBackbone
from cts.deq.gemma_latent_forward import (
    GemmaEq3Context,
    Eq3StaticInputs,
    encode_prompt_context,
    full_stack_forward,
    parallel_sparse_module_forward,
    parallel_sparse_module_forward_batch,
    prepare_eq3_module_prefix_cache,
    prepare_eq3_static_inputs,
    raw_ple_for_mixed_public_forward,
)
from cts.model.module_partition import layers_for_module


class GemmaCTSBackbone(BaseCTSBackbone, nn.Module):
    def __init__(self, cg_model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.cg = cg_model
        self.tokenizer = tokenizer
        cfg = cg_model.config.get_text_config()
        self._hidden = int(cfg.hidden_size)
        self._num_layers = int(cfg.num_hidden_layers)
        # Vision/audio towers may be CPU-offloaded.  CTS parameters must live
        # beside the text decoder, not whichever multimodal parameter happens
        # to be returned first by ``parameters()``.
        text_weight = cg_model.model.language_model.embed_tokens.weight
        dev = text_weight.device
        dt = text_weight.dtype
        from torch.nn.utils.parametrizations import spectral_norm
        class RoutingProjModule(nn.Module):
            def __init__(self, in_features, out_features, device=None, dtype=None):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(in_features, out_features, device=device, dtype=dtype) * 0.02)
            def forward(self, x=None):
                return self.weight

        self.routing_proj = RoutingProjModule(19, self._hidden, device=dev, dtype=dt)
        spectral_norm(self.routing_proj, name="weight")

        self._blend = nn.Linear(self._hidden, self._hidden, bias=True).to(device=dev)
        nn.init.normal_(self._blend.weight, std=0.02)
        nn.init.zeros_(self._blend.bias)

        # Paper §4.3: Wproj in R^{d x d_model} — projects z* [K, d] into
        # embedding space for soft-prompt prefix decoding.
        # Trained jointly in Stage 1.
        self.w_proj = nn.Linear(self._hidden, self._hidden, bias=False).to(device=dev, dtype=dt)
        nn.init.normal_(self.w_proj.weight, std=0.02)
        spectral_norm(self.w_proj, name="weight")

        # Sigmoid(gamma_param) damping parameter, initialized to 0.0 (sigmoid(0) = 0.5)
        self.gamma_param = nn.Parameter(torch.tensor(0.0, device=dev, dtype=dt))

        self.deq_map_mode = os.environ.get("CTS_DEQ_MAP_MODE", "parallel")
        # Opt-in until the target GPU/Transformers build passes exact cache
        # equivalence.  Once enabled, evaluation Phi processes only K latent
        # queries after one per-module causal prefix prefill.
        self.eq3_suffix_cache_enabled = (
            os.environ.get("CTS_EQ3_SUFFIX_CACHE", "0") == "1"
        )
        self.eq3_suffix_cache_train_enabled = (
            os.environ.get("CTS_EQ3_SUFFIX_CACHE_TRAIN", "0") == "1"
        )
        self.eq3_explicit_masks_enabled = (
            os.environ.get("CTS_EQ3_EXPLICIT_MASKS", "0") == "1"
        )
        self.last_decode_stats: Dict[str, Any] = {}

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        return self._hidden

    @property
    def supports_batched_deq(self) -> bool:
        return self.deq_map_mode in ("blend", "parallel", "paper")

    def _device(self) -> torch.device:
        return self._lm().embed_tokens.weight.device

    def routing_matrix(self) -> torch.Tensor:
        return self.routing_proj()

    def prompt_input_ids(self, parent_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Render the -it prompt contract exactly once for CTS and W_proj."""
        device = self._device()
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None) is not None
        ):
            raw = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": parent_text}],
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            input_ids = raw if isinstance(raw, torch.Tensor) else raw["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.dim() != 2 or input_ids.shape[0] != 1:
                raise ValueError(
                    "CTS canonical prompt must render as one [1,S] sequence"
                )
            input_ids = input_ids.to(device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            enc = self.tokenizer(
                parent_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            attention_mask = (
                attention_mask.to(device)
                if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.long)
            )
        return input_ids, attention_mask

    def encode_context(self, parent_text: str) -> GemmaEq3Context:
        input_ids, attention_mask = self.prompt_input_ids(parent_text)
        return encode_prompt_context(self._lm(), input_ids, attention_mask)

    def root_fwdpass(
        self,
        context: GemmaEq3Context,
        *,
        K: int,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build the deterministic prompt-conditioned root warm start.

        Algorithm 1 starts with ``z*_0 = FwdPass(s0)``.  The sparse Eq. (3)
        solve still determines the final root fixed point, but its initial
        iterate and fixed routing support must not come from prompt-agnostic
        Gaussian noise.  A zero continuous suffix passed once through the
        canonical full Gemma stack gives every latent slot causal access to
        the prompt while preserving the existing PLE/RoPE contract.

        This warm start is intentionally detached: Stage-1 trains the sparse
        Eq. (3) operator and routing weights, not a second 42-layer graph.
        """
        if not isinstance(context, GemmaEq3Context):
            raise TypeError("Gemma root FwdPass requires GemmaEq3Context")
        latent_tokens = int(K)
        if latent_tokens <= 0:
            raise ValueError("root FwdPass K must be positive")
        lm = self._lm()
        model_dtype = lm.embed_tokens.weight.dtype
        zero_suffix = torch.zeros(
            latent_tokens,
            self._hidden,
            device=self._device(),
            dtype=model_dtype,
        )
        # Stage-1 calls this no-grad warm start inside a wider autocast region.
        # Autocast's weight cache must be disabled here: otherwise the first
        # FP32 LoRA -> BF16 casts are cached without autograd metadata and the
        # later phantom-gradient Phi reuses detached weights, silently giving
        # every LoRA parameter a zero/None gradient.
        root_autocast = (
            torch.amp.autocast(
                "cuda",
                dtype=model_dtype,
                cache_enabled=False,
            )
            if self._device().type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), root_autocast:
            root = full_stack_forward(lm, zero_suffix, context)
        if root.shape != (latent_tokens, self._hidden):
            raise RuntimeError(
                "Gemma root FwdPass returned an invalid shape: "
                f"{tuple(root.shape)}"
            )
        if not torch.isfinite(root).all():
            raise RuntimeError("Gemma root FwdPass returned non-finite values")
        return root.detach().to(dtype=output_dtype)

    def with_latent_history(
        self,
        context: GemmaEq3Context,
        history: torch.Tensor,
    ) -> GemmaEq3Context:
        if not isinstance(context, GemmaEq3Context):
            raise TypeError("Gemma paper path requires GemmaEq3Context")
        return context.with_history(history)

    def prepare_deq_context(
        self,
        z_template: torch.Tensor,
        context: GemmaEq3Context | torch.Tensor,
        module_ids: Optional[List[int]] = None,
    ) -> Optional[Eq3StaticInputs]:
        """Cache transition-invariant Eq.3 prefix/position work once."""
        if self.deq_map_mode not in {"parallel", "paper", "full"}:
            return None
        needs_shared_kv = True
        if module_ids is not None:
            needs_shared_kv = any(
                bool(
                    getattr(
                        self._lm().layers[li].self_attn,
                        "is_kv_shared_layer",
                        False,
                    )
                )
                for mod in module_ids
                for li in layers_for_module(int(mod))
            )
        suffix_allowed = bool(
            self.eq3_suffix_cache_enabled
            and (not self.training or self.eq3_suffix_cache_train_enabled)
        )
        static = prepare_eq3_static_inputs(
            self._lm(),
            z_template,
            context,
            # LoRA dropout is active in Stage-1.  Reusing one stochastic
            # prefix sample while recomputing the z suffix would define a
            # different Phi from the uncached training operator.  Evaluation
            # and Stage-2 run in eval mode and may safely reuse exact prefix KV.
            cache_shared_kv=(
                needs_shared_kv
                and (not self.training or self.eq3_suffix_cache_train_enabled)
            ),
            # Suffix-only attention requires a rectangular K-query/P+K-key
            # causal mask.  Force the exact additive representation for the
            # cache arm; the standalone flag provides the mask-only A/B arm.
            force_explicit_masks=(
                not self.training
                and (
                    self.eq3_explicit_masks_enabled
                    or suffix_allowed
                )
            ),
        )
        if (
            suffix_allowed
            and module_ids is not None
        ):
            static = prepare_eq3_module_prefix_cache(
                self._lm(),
                static,
                module_ids,
                layers_for_module,
                allow_deterministic_training=self.eq3_suffix_cache_train_enabled,
            )
        return static

    @staticmethod
    def _pooled_context(
        context: GemmaEq3Context | torch.Tensor,
        *,
        batch: int,
    ) -> torch.Tensor:
        if isinstance(context, GemmaEq3Context):
            prompt = context.prompt_embeds.float()
            mask = context.attention_mask.to(prompt.device).float().unsqueeze(-1)
            pooled = (prompt * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            history = context.history_latents
            if history is not None:
                if history.dim() == 4:
                    hp = history.float().mean(dim=(1, 2))
                    if pooled.shape[0] == 1 and batch > 1:
                        pooled = pooled.expand(batch, -1)
                    pooled = 0.5 * (pooled + hp)
                else:
                    pooled = 0.5 * (pooled + history.float().reshape(-1, history.shape[-1]).mean(dim=0, keepdim=True))
            return pooled
        ctx = context.float()
        if ctx.dim() == 1:
            return ctx.unsqueeze(0)
        if ctx.dim() == 2:
            return ctx.mean(dim=0, keepdim=True)
        return ctx.mean(dim=1)

    def _lm(self) -> nn.Module:
        return self.cg.model.language_model

    def deq_step(
        self,
        z: torch.Tensor,
        context: GemmaEq3Context | torch.Tensor,
        module_weights: torch.Tensor,
        extra: Dict[str, Any],
    ) -> torch.Tensor:
        mode = extra.get("deq_map_mode", self.deq_map_mode)
        top_k = int(extra.get("top_k", 3))
        static_inputs = extra.get("eq3_static_inputs")
        module_ids = extra.get("routing_module_ids")
        fixed_module_ids = extra.get("routing_module_ids_tensor")
        ctx = context
        gamma = torch.sigmoid(self.gamma_param)

        if mode in ("parallel", "paper"):
            lm = self._lm()
            if z.dim() == 3:
                out = parallel_sparse_module_forward_batch(
                    lm, z, ctx, module_weights, layers_for_module, top_k=top_k,
                    static_inputs=static_inputs,
                    module_ids=module_ids,
                    fixed_module_ids=fixed_module_ids,
                )
            else:
                out = parallel_sparse_module_forward(
                    lm, z, ctx, module_weights, layers_for_module, top_k=top_k,
                    static_inputs=static_inputs,
                    module_ids=module_ids,
                    fixed_module_ids=fixed_module_ids,
                )
            out = out.to(dtype=z.dtype)
            out = 0.92 * out
            g = gamma.to(dtype=out.dtype, device=out.device)
            return (g * out + (1.0 - g) * z).to(dtype=z.dtype)

        if mode == "full":
            out = full_stack_forward(
                self._lm(), z, ctx, static_inputs=static_inputs
            )
            out = out.to(dtype=z.dtype)
            out = 0.92 * out
            g = gamma.to(dtype=out.dtype, device=out.device)
            return (g * out + (1.0 - g) * z).to(dtype=z.dtype)

        zf = z.float()
        pooled = self._pooled_context(
            ctx, batch=(int(zf.shape[0]) if zf.dim() == 3 else 1)
        ).to(device=zf.device, dtype=torch.float32)
        if zf.dim() == 3:
            ctxf = pooled
            if ctxf.shape[0] == 1:
                ctxf = ctxf.expand(zf.shape[0], -1)
            ctxf = ctxf.unsqueeze(1)
            ctxf = ctxf.expand(zf.shape[0], zf.shape[1], -1)
        else:
            ctxf = pooled[:1].expand(zf.shape[0], -1)
        gate = float(module_weights.sum().item() / max(module_weights.numel(), 1))
        gate = max(0.25, min(1.5, gate))
        h = zf + ctxf
        delta = self._blend(h)
        mixed = 0.82 * zf + 0.18 * gate * torch.tanh(delta)
        out = mixed.to(dtype=z.dtype)
        out = 0.92 * out
        g = gamma.to(dtype=out.dtype, device=out.device)
        return (g * out + (1.0 - g) * z).to(dtype=z.dtype)

    def decode_from_z_star(
        self,
        z_star: torch.Tensor,
        *,
        max_new_tokens: int = 64,
        problem_text: Optional[str] = None,
        phase_profiler: Optional[Any] = None,
        profile_scope: str = "decode",
        eos_check_interval: int = 4,
    ) -> str:
        """Paper §4.3: Wproj soft-prompt prefix -> frozen Gemma AR pass.

        z_star [K, d] -> Wproj -> [K, d_model] soft-prompt prefix
        -> frozen Gemma autoregressive decoding -> answer text

        Because final decoding uses W=1, KV-cache collapses to O(L).
        Peak VRAM stays below 18.0 GB (paper §4.3).

        ``problem_text``: optional. When supplied, the original problem
        text is tokenized and concatenated *after* the soft-prompt prefix
        so the frozen Gemma has explicit textual context in addition to
        the latent prefix. This restores paper §4.3 intent ("the soft
        prompt augments, not replaces, the problem context") and avoids
        the failure mode where a compute-limited Wproj produces random
        non-sequitur tokens like 'Cultura' / 'LinearLayout' on AIME
        prompts when no textual context is available. Backwards-
        compatible: callers that don't pass ``problem_text`` get the
        previous soft-prompt-only behaviour unchanged.
        """
        z_star = z_star.detach()
        if max_new_tokens <= 0:
            return ""
        if int(eos_check_interval) <= 0:
            raise ValueError("eos_check_interval must be >= 1")
        lm = self._lm()
        device = self._device()
        dt = lm.embed_tokens.weight.dtype

        prepare_scope = (
            phase_profiler.span(f"{profile_scope}.prepare")
            if phase_profiler is not None
            else nullcontext()
        )
        with prepare_scope:
            # Project z* through Wproj: [K, d] -> [K, d_model]
            z_projected = self.w_proj(
                z_star.to(device=device, dtype=self.w_proj.weight.dtype)
            ).to(dtype=dt)
            prefix_embeds = z_projected.unsqueeze(0)  # [1, K, d_model]

            if problem_text:
                tid_text, _ = self.prompt_input_ids(problem_text)
            else:
                bos = getattr(self.tokenizer, "bos_token_id", 2)
                tid_text = torch.tensor(
                    [[int(bos)]], device=device, dtype=torch.long
                )
            txt_embeds = lm.embed_tokens(tid_text).to(dtype=dt)
            # The public Gemma forward always combines a projection with the raw
            # PLE argument. Supply exact compensation for the continuous block.
            raw_per_layer_inputs = raw_ple_for_mixed_public_forward(
                lm, prefix_embeds, tid_text, txt_embeds
            )
            prefix_embeds = torch.cat([prefix_embeds, txt_embeds], dim=1)
            prefix_len = prefix_embeds.shape[1]

            eos_raw = getattr(
                getattr(self.cg, "generation_config", None),
                "eos_token_id",
                None,
            )
            if eos_raw is None:
                eos_raw = getattr(self.tokenizer, "eos_token_id", None)
            if eos_raw is None:
                try:
                    eos_raw = self.cg.config.get_text_config().eos_token_id
                except Exception:
                    eos_raw = 1
            if isinstance(eos_raw, (list, tuple, set)):
                eos_ids = {int(x) for x in eos_raw}
            else:
                eos_ids = {int(eos_raw)}

        ids: List[int] = []
        # Per-call local state avoids cross-contamination if two diagnostic
        # callers decode concurrently.  Publish one immutable snapshot only
        # after completion (or on the exception path).
        decode_stats: Dict[str, Any] = {
            "schema_version": 2,
            "generated_token_ids": [],
            "generated_token_count": 0,
            "decoded_char_count": None,
            "stop_reason": None,
            "max_new_tokens": int(max_new_tokens),
            "soft_prompt_tokens": int(z_projected.shape[0]),
            "text_prompt_tokens": int(tid_text.shape[1]),
            "prefix_tokens": int(prefix_len),
            "prefill_input_tokens": int(prefix_len),
            "problem_text_present": bool(problem_text),
            "eos_token_ids": sorted(eos_ids),
            "eos_check_interval": int(eos_check_interval),
            "host_eos_checks": 0,
            "decoder_forward_tokens": 0,
            "discarded_after_eos": 0,
            "cache_class": None,
            "kv_cache_requested": True,
            "kv_cache_prefill_present": False,
            "kv_cache_type": None,
            "kv_cache_reuse_calls": 0,
            "observed_eos_token_id": None,
            "observed_eos_index": None,
            "eos_reached": False,
            "eos_emitted": False,
            "cap_reached": False,
        }
        try:
            with torch.inference_mode():
                attn_mask = torch.ones(1, prefix_len, device=device, dtype=torch.long)
                prefill_scope = (
                    phase_profiler.span(f"{profile_scope}.prefill")
                    if phase_profiler is not None
                    else nullcontext()
                )
                with prefill_scope:
                    out = lm(
                        inputs_embeds=prefix_embeds,
                        per_layer_inputs=raw_per_layer_inputs,
                        attention_mask=attn_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values
                    decode_stats["cache_class"] = type(past).__name__
                    decode_stats["kv_cache_prefill_present"] = past is not None
                    decode_stats["kv_cache_type"] = type(past).__name__
                    h = out.last_hidden_state[:, -1, :]
                    logits = self.cg.lm_head(h)
                    # Keep the greedy feedback token on device.  The old loop
                    # called ``.item()`` and rebuilt a CUDA tensor every token,
                    # forcing a full GPU/CPU synchronization on the AR hot
                    # path.  The first token is checked immediately so an
                    # early EOS remains just as cheap as before.
                    next_token = logits.argmax(dim=-1).reshape(1, 1)
                    next_id = int(next_token.item())
                    decode_stats["host_eos_checks"] += 1
                    decode_stats["decoder_forward_tokens"] += 1
                    ids.append(next_id)

                autoregressive_scope = (
                    phase_profiler.span(f"{profile_scope}.autoregressive")
                    if phase_profiler is not None
                    else nullcontext()
                )
                with autoregressive_scope:
                    if next_id in eos_ids:
                        decode_stats["stop_reason"] = "eos"
                        decode_stats["observed_eos_token_id"] = int(next_id)
                        decode_stats["observed_eos_index"] = 0
                    pending: List[torch.Tensor] = []
                    for step_idx in range(max_new_tokens - 1):
                        if decode_stats["stop_reason"] == "eos":
                            break
                        if past is not None:
                            decode_stats["kv_cache_reuse_calls"] += 1
                        out = lm(
                            input_ids=next_token,
                            past_key_values=past,
                            use_cache=True,
                            return_dict=True,
                        )
                        past = out.past_key_values
                        h = out.last_hidden_state[:, -1, :]
                        logits = self.cg.lm_head(h)
                        next_token = logits.argmax(dim=-1).reshape(1, 1)
                        pending.append(next_token)
                        decode_stats["decoder_forward_tokens"] += 1

                        flush = (
                            len(pending) >= int(eos_check_interval)
                            or step_idx + 1 == max_new_tokens - 1
                        )
                        if not flush:
                            continue
                        chunk = (
                            torch.cat(pending, dim=1)
                            .reshape(-1)
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        decode_stats["host_eos_checks"] += 1
                        pending.clear()
                        eos_pos = next(
                            (i for i, token_id in enumerate(chunk) if token_id in eos_ids),
                            None,
                        )
                        if eos_pos is None:
                            ids.extend(int(token_id) for token_id in chunk)
                            continue
                        decode_stats["observed_eos_token_id"] = int(
                            chunk[eos_pos]
                        )
                        decode_stats["observed_eos_index"] = int(
                            len(ids) + eos_pos
                        )
                        ids.extend(int(token_id) for token_id in chunk[: eos_pos + 1])
                        decode_stats["discarded_after_eos"] += int(
                            len(chunk) - eos_pos - 1
                        )
                        decode_stats["stop_reason"] = "eos"
            if decode_stats["stop_reason"] is None:
                decode_stats["stop_reason"] = (
                    "eos" if ids and ids[-1] in eos_ids else "max_new_tokens"
                )
            decode_stats["generated_token_ids"] = list(ids)
            decode_stats["generated_token_count"] = len(ids)
            decode_stats["eos_emitted"] = bool(
                ids and ids[-1] in eos_ids
            )
            decode_stats["eos_reached"] = bool(decode_stats["eos_emitted"])
            decode_stats["cap_reached"] = bool(
                decode_stats["stop_reason"] == "max_new_tokens"
                and len(ids) >= int(max_new_tokens)
            )
            detokenize_scope = (
                phase_profiler.span(f"{profile_scope}.detokenize")
                if phase_profiler is not None
                else nullcontext()
            )
            with detokenize_scope:
                decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            decode_stats["decoded_char_count"] = len(decoded)
            self.last_decode_stats = dict(decode_stats)
            self.last_decode_stats["generated_token_ids"] = list(ids)
            return decoded
        except Exception as exc:
            decode_stats["stop_reason"] = "error"
            decode_stats["error_type"] = type(exc).__name__
            decode_stats["generated_token_ids"] = list(ids)
            decode_stats["generated_token_count"] = len(ids)
            self.last_decode_stats = dict(decode_stats)
            self.last_decode_stats["generated_token_ids"] = list(ids)
            raise RuntimeError(
                "CTS W_proj autoregressive decode failed; refusing the "
                "historical one-token lm_head fallback"
            ) from exc
