#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/g2v8_gca_bridge.py — Non-Invasive Zero-Init Gated Cross-Attention (GCA) Bridge v2 for CTS G2.

Key Architectural Properties:
  1. Non-Invasive Pure C-fmt Prompt & Chain-of-Thought Preservation:
     - Uses the exact pure C-fmt text prompt (no 64 '1' placeholder tokens in input_ids).
     - Problem prompt tokens [0 .. prompt_len - 2] remain 100% untouched Pure Gemma across all 42 layers.
     - During autoregressive generation, Chain-of-Thought tokens prior to the conclusion readout phase
       remain uncorrupted, while the post-norm hook (`lm_model.norm`) injects slot-aligned latent pre-logits
       during the conclusion phase without polluting the 42-layer KV cache (`past_key_values`).
  2. Exact Non-Mean & Equilibrium & Whitened-Ridge Compatibility Gating:
     - `g_nonmean(z) = tanh(10 * ||z - center||_F^2 / (K * D))` guarantees exact `0.0` output when `z == center`.
     - `g_eq(z)` measures convergence from the source manifold (`src_ref_center`) to the target conclusion
       manifold (`center`), separating K=8 DEQ fixed points (Condition A) from K=0 initial latents (Condition E).
     - `g_match(prompt_summary, z)` combines Train-993 Cholesky-whitened cosine similarity and Ridge CCA alignment
       between the clean problem prompt summary (excluding constant format suffix tokens) and the centered latent pool.
  3. Unshifted Current-Token Linear Readout (`w_tok`):
     - Maps each normalized latent slot `rms_norm_zero_safe(z[k]) * 50.0` to `lm_head` pre-logit space via a
       bias-free linear projection `w_tok` trained strictly on `Train-993` (`configs/s7_train_993.json`).
     - Per-slot head/tail RMS gates (`g_head`, `g_tail`) automatically distinguish full 64-slot oracle latents
       (`z_orc` in Stage 1) from 10-slot conclusion-transition + CoT-completion DEQ latents (`z*` in Stage 2).
  4. Zero-Init Exact Preservation (Gate C):
     - Per-head gates `gate_alpha` are initialized to 0.0 (`tanh(0) = 0`), guaranteeing bit-identical
       `C-fmt` outputs at step 0 (`max_diff == 0.0`) while `dL / d(gate_alpha)` at step 0 is non-zero.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm_zero_safe(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalize along the last dimension while preserving exact 0 -> 0."""
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps)


class GCALayer(nn.Module):
    """Gated Cross-Attention adapter with Problem-Latent Matcher, Native Cosine + Multi-Head QK, and Zero-Bias Values."""

    def __init__(
        self,
        d_model: int = 2560,
        k_latent: int = 64,
        n_heads: int = 8,
        d_qk_head: int = 64,
        r_val: int = 128,
        d_match: int = 64,
        init_out_scale: float = 12.0,
        attn_scale: float = 8.0,
        use_identity_val: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.k_latent = k_latent
        self.n_heads = n_heads
        self.d_qk_head = d_qk_head
        self.d_qk = n_heads * d_qk_head
        self.d_v_head = d_model // n_heads
        self.r_val = r_val
        self.d_match = d_match
        self.use_identity_val = bool(use_identity_val)

        # Multi-head Q and K projections
        self.q_proj = nn.Linear(d_model, self.d_qk, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_qk, bias=False)

        # Low-rank auxiliary compatibility matcher
        self.match_drop = nn.Dropout(0.35)
        self.match_q = nn.Linear(d_model, d_match, bias=False)
        self.match_k = nn.Linear(d_model, d_match, bias=False)
        self.match_temp = nn.Parameter(torch.tensor(8.0))
        self.match_cos_temp = nn.Parameter(torch.tensor(8.0))
        self.match_bias = nn.Parameter(torch.tensor(0.5))

        # Depthwise 1D temporal convolution across the 64 slots (strictly bias-free, zero-init)
        self.v_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False)

        # Identity + Low-Rank Value & Output projections (strictly bias-free, homogeneous of degree 1 in z_tilde)
        self.v_down = nn.Linear(d_model, r_val, bias=False)
        self.v_up = nn.Linear(r_val, d_model, bias=False)
        self.o_down = nn.Linear(d_model, r_val, bias=False)
        self.o_up = nn.Linear(r_val, d_model, bias=False)
        self.chan_scale = nn.Parameter(torch.ones(1, 1, d_model))

        # Slot positional embeddings ONLY for Keys (never Values, preserving V(0) == 0)
        self.slot_pos_emb = nn.Parameter(torch.randn(k_latent, d_model) * 0.15)

        # Learnable Null/Sink key in QK space + learnable null bias per head
        self.null_key = nn.Parameter(torch.randn(1, n_heads, 1, d_qk_head) * 0.1)
        self.null_bias = nn.Parameter(
            torch.full((1, n_heads, 1, 1), float(torch.log(torch.tensor(float(k_latent))).item()))
        )

        # Per-head learned QK temperature & native cosine similarity weight
        self.qk_temp = nn.Parameter(torch.full((1, n_heads, 1, 1), float(attn_scale)))
        self.cos_temp = nn.Parameter(
            torch.full((1, n_heads, 1, 1), float(attn_scale) if self.use_identity_val else 0.0)
        )

        # Per-head gate initialized to 0.0 -> tanh(0.0) == 0.0 at step 0
        self.gate_alpha = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        self.out_scale = nn.Parameter(torch.tensor(float(init_out_scale)))

        self._cached_prompt_summary: Optional[torch.Tensor] = None
        self._cached_clean_prompt: Optional[torch.Tensor] = None
        self._cached_has_prob_text: bool = True
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            for m in (self.q_proj, self.k_proj, self.match_q, self.match_k, self.v_down, self.o_down):
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(float(m.weight.shape[1])))
            nn.init.zeros_(self.v_conv.weight)
            if self.use_identity_val:
                nn.init.zeros_(self.v_up.weight)
                nn.init.zeros_(self.o_up.weight)
            else:
                nn.init.normal_(self.v_up.weight, mean=0.0, std=1.0 / math.sqrt(float(self.v_up.weight.shape[1])))
                nn.init.normal_(self.o_up.weight, mean=0.0, std=1.0 / math.sqrt(float(self.o_up.weight.shape[1])))

    def center_and_norm_h(self, h_f32: torch.Tensor, h_origin: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h_origin is not None and self.use_identity_val:
            return rms_norm_zero_safe(h_f32 - h_origin)
        return rms_norm_zero_safe(h_f32)

    def _compute_keys(self, z_tilde: torch.Tensor) -> torch.Tensor:
        bsz, k_len, _ = z_tilde.shape
        pos_k = self.slot_pos_emb[:k_len].unsqueeze(0)
        k_in = rms_norm_zero_safe(z_tilde + pos_k)
        k = self.k_proj(k_in).view(bsz, k_len, self.n_heads, self.d_qk_head).transpose(1, 2)
        return rms_norm_zero_safe(k)

    def _compute_logits(
        self,
        q: torch.Tensor,
        h_norm: torch.Tensor,
        z_tilde: torch.Tensor,
    ) -> torch.Tensor:
        k_qk = self._compute_keys(z_tilde)
        if q.shape[0] == 1 and k_qk.shape[0] > 1:
            q = q.expand(k_qk.shape[0], -1, -1, -1)
            h_norm = h_norm.expand(k_qk.shape[0], -1, -1)
        logits_qk = torch.matmul(q, k_qk.transpose(-2, -1)) * (self.qk_temp / float(self.d_qk_head))
        cos_sim = torch.matmul(h_norm, z_tilde.transpose(-2, -1)).unsqueeze(1) / float(self.d_model)
        return logits_qk + self.cos_temp * cos_sim

    def _compute_match_logit(
        self,
        prompt_summary: torch.Tensor,
        z_tilde: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_summary.shape[0] == 1 and z_tilde.shape[0] > 1:
            prompt_summary = prompt_summary.expand(z_tilde.shape[0], -1, -1)
        z_pool = rms_norm_zero_safe(z_tilde.mean(dim=1, keepdim=True))

        p_in = self.match_drop(prompt_summary) if self.training else prompt_summary
        z_in = self.match_drop(z_tilde) if self.training else z_tilde
        zp_in = self.match_drop(z_pool) if self.training else z_pool

        qm = rms_norm_zero_safe(self.match_q(p_in))
        km_pool = rms_norm_zero_safe(self.match_k(zp_in))
        km_slots = rms_norm_zero_safe(self.match_k(z_in))

        pool_sim = (qm * km_pool).sum(dim=-1, keepdim=True) / float(self.d_match)
        slot_sims = torch.matmul(qm, km_slots.transpose(-2, -1)) / float(self.d_match)
        top_slot_sim = slot_sims.topk(k=min(8, slot_sims.shape[-1]), dim=-1).values.mean(dim=-1, keepdim=True)
        learned_sim = 0.5 * (pool_sim + top_slot_sim)

        native_pool_cos = (prompt_summary * z_pool).sum(dim=-1, keepdim=True) / float(self.d_model)
        native_slot_cos = (
            torch.matmul(prompt_summary, z_tilde.transpose(-2, -1))
            .topk(k=min(8, z_tilde.shape[1]), dim=-1)
            .values.mean(dim=-1, keepdim=True)
            / float(self.d_model)
        )
        native_cos = 0.5 * (native_pool_cos + native_slot_cos)
        return self.match_temp * learned_sim + self.match_cos_temp * native_cos + self.match_bias

    def forward(
        self,
        h_gen: torch.Tensor,
        z_tilde: torch.Tensor,
        prompt_summary: Optional[torch.Tensor] = None,
        has_prob_text: bool = True,
        z_neg_tilde: Optional[torch.Tensor] = None,
        h_origin: Optional[torch.Tensor] = None,
        ext_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, t_len, _ = h_gen.shape
        h_f32 = h_gen.float()
        h_norm = self.center_and_norm_h(h_f32, h_origin=h_origin)

        if prompt_summary is None:
            prompt_summary = h_norm[:, :1, :]

        if has_prob_text:
            match_logit_pos = self._compute_match_logit(prompt_summary, z_tilde)
            match_gate = torch.sigmoid(match_logit_pos)
        else:
            match_logit_pos = torch.zeros((bsz, 1, 1), device=h_gen.device, dtype=torch.float32)
            match_gate = torch.ones((bsz, 1, 1), device=h_gen.device, dtype=torch.float32)

        if ext_gate is not None:
            match_gate = match_gate * ext_gate

        q = self.q_proj(h_norm).view(bsz, t_len, self.n_heads, self.d_qk_head).transpose(1, 2)
        q = rms_norm_zero_safe(q)

        logits_pos = self._compute_logits(q, h_norm, z_tilde)
        k_null = rms_norm_zero_safe(self.null_key).expand(bsz, -1, -1, -1)
        logits_null = (
            torch.matmul(q, k_null.transpose(-2, -1)) * (self.qk_temp / float(self.d_qk_head))
            + self.null_bias
        )

        logits_aug = torch.cat([logits_pos, logits_null], dim=-1)
        attn_aug = F.softmax(logits_aug, dim=-1)
        attn_pos = attn_aug[..., : z_tilde.shape[1]]

        z_conv = self.v_conv(z_tilde.transpose(1, 2)).transpose(1, 2)
        if self.use_identity_val:
            v_full = z_tilde + z_conv + self.v_up(self.v_down(z_tilde))
        else:
            v_full = self.v_up(self.v_down(z_tilde + z_conv))
        v_pos = v_full.view(bsz, z_tilde.shape[1], self.n_heads, self.d_v_head).transpose(1, 2)

        gated_attn = torch.tanh(self.gate_alpha) * attn_pos
        context = torch.matmul(gated_attn, v_pos)
        context = context.transpose(1, 2).contiguous().view(bsz, t_len, self.d_model)

        if self.use_identity_val:
            out_vec = context + self.o_up(self.o_down(context))
        else:
            out_vec = self.o_up(self.o_down(context))
        delta_h = (out_vec * self.chan_scale) * (self.out_scale * match_gate)

        if z_neg_tilde is not None:
            logits_neg = self._compute_logits(q, h_norm, z_neg_tilde)
            lse_pos = torch.logsumexp(logits_pos, dim=-1) - logits_null.squeeze(-1)
            lse_neg = (
                torch.logsumexp(logits_neg, dim=-1)
                - logits_null.expand(logits_neg.shape[0], -1, -1, -1).squeeze(-1)
            )
            qk_compat = F.softplus(2.0 - lse_pos).mean() + F.softplus(2.5 + lse_neg).mean()

            if has_prob_text:
                match_logit_neg = self._compute_match_logit(prompt_summary, z_neg_tilde)
                all_match = torch.cat([match_logit_pos.view(1, -1), match_logit_neg.view(1, -1)], dim=1)
                infonce = F.cross_entropy(all_match, torch.zeros(1, dtype=torch.long, device=h_gen.device))
                margin_loss = F.softplus(2.5 - match_logit_pos).mean() + F.softplus(2.5 + match_logit_neg).mean()
                match_compat = margin_loss + 0.5 * infonce
            else:
                match_compat = torch.zeros((), device=h_gen.device, dtype=torch.float32)
            compat_loss = qk_compat + match_compat
        else:
            compat_loss = torch.zeros((), device=h_gen.device, dtype=torch.float32)

        return delta_h, compat_loss


class GatedCrossAttentionBridge(nn.Module):
    """Multi-Layer + Post-Norm Gated Cross-Attention Bridge v2 for Gemma-4-E4B."""

    proj_version = 8

    def __init__(
        self,
        d_model: int = 2560,
        k_latent: int = 64,
        hook_layers: Sequence[int] = (27, 35, 41),
        use_post_norm: bool = True,
        n_heads: int = 8,
        d_qk_head: int = 64,
        r_val: int = 128,
        layer_out_scale: float = 8.0,
        norm_out_scale: float = 24.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.k_latent = k_latent
        self.hook_layers = tuple(int(x) for x in hook_layers)
        self.use_post_norm = bool(use_post_norm)
        self.n_heads = n_heads
        self.d_qk_head = d_qk_head
        self.r_val = r_val

        # Exact Train-993 manifold & whitened-Ridge matcher buffers
        self.register_buffer("center", torch.zeros(k_latent, d_model))
        self.register_buffer("src_ref_center", torch.zeros(k_latent, d_model))
        self.register_buffer("has_eq_ref", torch.zeros(1))
        self.register_buffer("has_fitted_matcher", torch.zeros(1))
        self.register_buffer("match_p_mu", torch.zeros(1, d_model))
        self.register_buffer("match_z_mu", torch.zeros(1, d_model))
        self.register_buffer("match_p_whiten", torch.eye(d_model))
        self.register_buffer("match_z_whiten", torch.eye(d_model))
        self.register_buffer("match_w_ridge", torch.eye(d_model))
        self.register_buffer("match_thr", torch.zeros(1))

        # Unshifted current-token linear readout projection (strictly bias-free)
        self.w_tok = nn.Linear(d_model, d_model, bias=False)
        self.slot_out_scale = nn.Parameter(torch.tensor(5.0))
        with torch.no_grad():
            nn.init.normal_(self.w_tok.weight, mean=0.0, std=1.0 / math.sqrt(float(d_model)))

        self.layer_adapters = nn.ModuleDict({
            str(l_idx): GCALayer(
                d_model=d_model,
                k_latent=k_latent,
                n_heads=n_heads,
                d_qk_head=d_qk_head,
                r_val=r_val,
                init_out_scale=layer_out_scale,
                use_identity_val=False,
            )
            for l_idx in self.hook_layers
        })

        self.post_norm_adapter: Optional[GCALayer] = (
            GCALayer(
                d_model=d_model,
                k_latent=k_latent,
                n_heads=n_heads,
                d_qk_head=d_qk_head,
                r_val=r_val,
                init_out_scale=norm_out_scale,
                use_identity_val=True,
            )
            if self.use_post_norm
            else None
        )

        # Runtime hook state
        self._handles: List[Any] = []
        self._active_z_raw: Optional[torch.Tensor] = None
        self._active_z_tilde: Optional[torch.Tensor] = None
        self._active_z_neg_raw: Optional[torch.Tensor] = None
        self._active_z_neg_tilde: Optional[torch.Tensor] = None
        self._active_prompt_len: int = 1
        self._active_has_prob_text: bool = True
        self._active_slot_idx: Optional[int] = None
        self._active_cot_free: bool = False
        self._active_tail_nll_inject: bool = False
        self._last_compat_losses: List[torch.Tensor] = []

        # Cached item-level gates from latest prefill
        self.last_g_item: float = 0.0
        self.last_g_head: float = 0.0
        self.last_g_tail: float = 0.0
        self.last_g_match: float = 0.0
        self.last_g_eq: float = 0.0

    def normalize_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Center by training mean and RMS-normalize per slot (0 -> 0 preserved)."""
        if z.dim() == 2:
            z = z.unsqueeze(0)
        diff = z.float() - self.center.unsqueeze(0)
        return rms_norm_zero_safe(diff)

    def fit_whitened_matcher(
        self,
        p_train: torch.Tensor,
        z_train_pool: torch.Tensor,
        neg_quantile: float = 0.88,
        reg_cov: float = 5.0,
        reg_ridge: float = 0.5,
    ) -> Dict[str, float]:
        """
        Fit Cholesky whitening + Ridge CCA matcher strictly on Train-993 (zero Dev access).
        Args:
            p_train: [N_train, D] clean problem prompt summaries on Train-993
            z_train_pool: [N_train, D] centered latent mean pools on Train-993
        """
        with torch.no_grad():
            device = self.center.device
            p_tr = p_train.to(device=device, dtype=torch.float32)
            z_tr = z_train_pool.to(device=device, dtype=torch.float32)
            n_tr, d = p_tr.shape

            p_mu = p_tr.mean(dim=0, keepdim=True)
            z_mu = z_tr.mean(dim=0, keepdim=True)
            pc_raw = p_tr - p_mu
            zc_raw = z_tr - z_mu

            # Match latent pool RMS to prompt summary's natural post-norm RMS (~12.0) so cov_p and cov_z share identical regularization scale
            p_rms = pc_raw.pow(2).mean().sqrt().clamp_min(1e-6)
            z_rms = zc_raw.pow(2).mean().sqrt().clamp_min(1e-6)
            z_scale = p_rms / z_rms
            pc = pc_raw
            zc = zc_raw * z_scale

            cov_p = (pc.T @ pc) / float(n_tr) + reg_cov * torch.eye(d, device=device)
            cov_z = (zc.T @ zc) / float(n_tr) + reg_cov * torch.eye(d, device=device)
            lp = torch.linalg.cholesky(cov_p)
            lz = torch.linalg.cholesky(cov_z)
            p_whiten = torch.linalg.inv(lp.T)
            z_whiten = torch.linalg.inv(lz.T) * z_scale

            pw = F.normalize((p_tr - p_mu) @ p_whiten, dim=-1)
            zw = F.normalize((z_tr - z_mu) @ z_whiten, dim=-1)
            w_ridge = torch.linalg.solve(
                pw.T @ pw + reg_ridge * torch.eye(d, device=device),
                pw.T @ zw,
            )

            pred_zw = F.normalize(pw @ w_ridge, dim=-1)
            pos_sims = 0.5 * (pred_zw * zw).sum(dim=-1) + 0.5 * (pw * zw).sum(dim=-1)
            zw_shuf = torch.roll(zw, shifts=-1, dims=0)
            neg_sims = 0.5 * (pred_zw * zw_shuf).sum(dim=-1) + 0.5 * (pw * zw_shuf).sum(dim=-1)
            thr = float(torch.quantile(neg_sims, neg_quantile).item())

            self.match_p_mu.copy_(p_mu)
            self.match_z_mu.copy_(z_mu)
            self.match_p_whiten.copy_(p_whiten)
            self.match_z_whiten.copy_(z_whiten)
            self.match_w_ridge.copy_(w_ridge)
            self.match_thr.fill_(thr)
            self.has_fitted_matcher.fill_(1.0)

            return {
                "train_pos_sim": float(pos_sims.mean().item()),
                "train_neg_sim": float(neg_sims.mean().item()),
                "match_thr": thr,
                "train_win_rate": float((pos_sims > neg_sims).float().mean().item() * 100.0),
            }

    def compute_structural_gates(
        self,
        z_raw: torch.Tensor,
        clean_prompt: Optional[torch.Tensor],
        has_prob_text: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute [B, 1, 1] gates: (g_nonmean, g_eq, g_match, g_head, g_tail).
        Guarantees exact 0.0 when z_raw == self.center.
        """
        if z_raw.dim() == 2:
            z_raw = z_raw.unsqueeze(0)
        z_f32 = z_raw.float()
        diff = z_f32 - self.center.unsqueeze(0)

        # 1. Exact non-mean gate: 0.0 when z_raw == self.center, ~1.0 otherwise
        g_nonmean = torch.tanh(10.0 * diff.pow(2).mean(dim=(1, 2), keepdim=True))

        # 2. Equilibrium manifold gate: separates K=8 fixed point (around center) from K=0 z_src (around src_ref_center)
        if float(self.has_eq_ref.item()) > 0.5:
            delta_m = (self.center - self.src_ref_center).unsqueeze(0)
            s_eq = (
                ((z_f32 - self.src_ref_center.unsqueeze(0)) * delta_m).sum(dim=(1, 2), keepdim=True)
                / delta_m.pow(2).sum().clamp_min(1e-6)
            )
            g_eq = torch.sigmoid(20.0 * (s_eq - 0.5))
        else:
            g_eq = torch.ones_like(g_nonmean)

        # 3. Whitened-Ridge Prompt-to-Latent compatibility gate
        if has_prob_text and float(self.has_fitted_matcher.item()) > 0.5 and clean_prompt is not None:
            p_vec = clean_prompt.view(-1, self.d_model)
            if p_vec.shape[0] == 1 and z_f32.shape[0] > 1:
                p_vec = p_vec.expand(z_f32.shape[0], -1)
            z_pool = diff.mean(dim=1)
            pw = F.normalize((p_vec - self.match_p_mu) @ self.match_p_whiten, dim=-1)
            zw = F.normalize((z_pool - self.match_z_mu) @ self.match_z_whiten, dim=-1)
            pred_zw = F.normalize(pw @ self.match_w_ridge, dim=-1)
            comb_sim = 0.5 * (pred_zw * zw).sum(dim=-1, keepdim=True) + 0.5 * (pw * zw).sum(dim=-1, keepdim=True)
            g_match = torch.sigmoid(60.0 * (comb_sim - self.match_thr)).unsqueeze(-1)
        else:
            g_match = torch.ones_like(g_nonmean)

        # 4. Head (slots 0..9) and Tail (slots 10..63) activity gates
        l_cue = min(10, self.k_latent // 2)
        rms_head = diff[:, :l_cue, :].pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        rms_tail = diff[:, l_cue:, :].pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
        g_head = torch.sigmoid(8.0 * (rms_head - 1.40))
        g_tail = torch.sigmoid(8.0 * (rms_tail - 1.40))

        return g_nonmean, g_eq, g_match, g_head, g_tail

    def attach(self, lm_model: nn.Module) -> None:
        """Register forward hooks onto `lm_model.layers[l]` and `lm_model.norm`."""
        self.detach()
        layers = getattr(lm_model, "layers", None)
        if layers is not None:
            for l_idx in self.hook_layers:
                if 0 <= l_idx < len(layers):
                    h = layers[l_idx].register_forward_hook(
                        self._make_hook(self.layer_adapters[str(l_idx)], is_post_norm=False)
                    )
                    self._handles.append(h)
        if self.use_post_norm and self.post_norm_adapter is not None and hasattr(lm_model, "norm"):
            h = lm_model.norm.register_forward_hook(
                self._make_hook(self.post_norm_adapter, is_post_norm=True)
            )
            self._handles.append(h)

    def detach(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, adapter: GCALayer, is_post_norm: bool = False):
        def _hook(module: nn.Module, args: Tuple[Any, ...], output: Any) -> Any:
            if self._active_z_tilde is None or self._active_z_raw is None:
                return output
            is_tuple = isinstance(output, tuple)
            h: torch.Tensor = output[0] if is_tuple else output
            seq_len = h.shape[1]
            start_pos = max(0, self._active_prompt_len - 1) if seq_len >= self._active_prompt_len else 0
            if start_pos >= seq_len:
                return output

            h_origin = self.center.mean(dim=0, keepdim=True) if adapter.use_identity_val else None
            # Only update cached prompt summaries when not in explicit slot-readout mode (`_active_slot_idx is None`)
            if start_pos > 0 and self._active_slot_idx is None:
                p_slice = h[:, 3:start_pos, :] if start_pos > 6 else h[:, :start_pos, :]
                prompt_summary = rms_norm_zero_safe(
                    adapter.center_and_norm_h(p_slice.float(), h_origin=h_origin).mean(dim=1, keepdim=True)
                )
                end_clean = max(4, self._active_prompt_len - 13)
                clean_slice = h[:, 3:end_clean, :] if end_clean > 3 else h[:, :start_pos, :]
                clean_prompt = clean_slice.float().mean(dim=1)
                adapter._cached_prompt_summary = prompt_summary.detach()
                adapter._cached_clean_prompt = clean_prompt.detach()
                adapter._cached_has_prob_text = self._active_has_prob_text
                has_prob = self._active_has_prob_text
            else:
                prompt_summary = adapter._cached_prompt_summary
                clean_prompt = adapter._cached_clean_prompt
                has_prob = adapter._cached_has_prob_text

            g_nonmean, g_eq, g_match, g_head, g_tail = self.compute_structural_gates(
                self._active_z_raw, clean_prompt, has_prob_text=has_prob
            )
            ext_gate = g_nonmean * g_eq * g_match

            if is_post_norm and self._active_slot_idx is None:
                alpha_mag = float(torch.tanh(adapter.gate_alpha).abs().mean().detach().item())
                alpha_open = 1.0 if alpha_mag > 1e-5 else 0.0
                self.last_g_match = float(g_match[0, 0, 0].detach().item())
                self.last_g_eq = float(g_eq[0, 0, 0].detach().item())
                self.last_g_head = float(g_head[0, 0, 0].detach().item())
                self.last_g_tail = float(g_tail[0, 0, 0].detach().item())
                self.last_g_item = float((alpha_open * ext_gate[0, 0, 0]).detach().item())

            # Mode A: Explicit slot-indexed conclusion readout during autoregressive decoding
            if self._active_slot_idx is not None:
                if not is_post_norm:
                    return output
                slot_k = int(self._active_slot_idx)
                if slot_k < 0 or slot_k >= self.k_latent:
                    return output
                l_cue = min(10, self.k_latent // 2)
                slot_act = g_head if slot_k < l_cue else g_tail
                t_alpha = torch.tanh(adapter.gate_alpha)
                alpha_gate = 0.5 * (t_alpha + t_alpha.abs()).mean()
                alpha_scale = torch.clamp(alpha_gate * 50.0, min=0.0, max=1.0)
                total_gate = alpha_scale * ext_gate * slot_act  # [B, 1, 1]
                eff_gate = torch.clamp(total_gate * 2.0, min=0.0, max=1.0)

                z_slot = self._active_z_raw.float()[:, slot_k : slot_k + 1, :]  # [B, 1, D]
                v_slot = self.w_tok(rms_norm_zero_safe(z_slot) * 50.0)  # [B, 1, D]
                h_gen = h[:, start_pos:, :]
                delta_h = eff_gate * (self.slot_out_scale * v_slot - h_gen.float())
                h_gen_new = h_gen + delta_h.to(dtype=h.dtype)
                h_new = torch.cat([h[:, :start_pos, :], h_gen_new], dim=1) if start_pos > 0 else h_gen_new
                return (h_new,) + output[1:] if is_tuple else h_new

            # Mode B: CoT-free prefix pass during autoregressive decoding (caches gates without altering CoT)
            if self._active_cot_free:
                return output

            # Mode C: Standard GCA pass (used in teacher-forced training, NLL evaluation, and unit self-test)
            h_gen = h[:, start_pos:, :]
            mode_c_gate = (
                (g_nonmean * g_eq * g_match.clamp_min(0.25)) if self.training else (g_nonmean * g_eq * g_match)
            )
            delta_h, compat_loss = adapter(
                h_gen,
                self._active_z_tilde,
                prompt_summary=prompt_summary,
                has_prob_text=has_prob,
                z_neg_tilde=self._active_z_neg_tilde,
                h_origin=h_origin,
                ext_gate=mode_c_gate,
            )

            if self._active_z_neg_tilde is not None:
                self._last_compat_losses.append(compat_loss)

            h_gen_new = h_gen + delta_h.to(dtype=h.dtype)
            if start_pos > 0:
                h_new = torch.cat([h[:, :start_pos, :], h_gen_new], dim=1)
            else:
                h_new = h_gen_new

            if is_tuple:
                return (h_new,) + output[1:]
            return h_new

        return _hook

    @contextmanager
    def active_context(
        self,
        z: Optional[torch.Tensor],
        prompt_len: int = 1,
        step_offset: int = 0,
        z_neg: Optional[torch.Tensor] = None,
        has_prob_text: bool = True,
        slot_idx: Optional[int] = None,
        cot_free: bool = False,
        tail_nll_inject: bool = False,
    ):
        """Activate GCA hooks for a single forward pass with automatic cleanup."""
        prev_z_raw = self._active_z_raw
        prev_z = self._active_z_tilde
        prev_neg_raw = self._active_z_neg_raw
        prev_neg = self._active_z_neg_tilde
        prev_plen = self._active_prompt_len
        prev_has_prob = self._active_has_prob_text
        prev_slot = self._active_slot_idx
        prev_cot = self._active_cot_free
        prev_tail = self._active_tail_nll_inject
        self._last_compat_losses = []
        try:
            if z is not None:
                z_b = z.unsqueeze(0) if z.dim() == 2 else z
                self._active_z_raw = z_b.float()
                self._active_z_tilde = self.normalize_latent(z_b)
            else:
                self._active_z_raw = None
                self._active_z_tilde = None
            if z_neg is not None:
                zn_b = z_neg.unsqueeze(0) if z_neg.dim() == 2 else z_neg
                self._active_z_neg_raw = zn_b.float()
                self._active_z_neg_tilde = self.normalize_latent(zn_b)
            else:
                self._active_z_neg_raw = None
                self._active_z_neg_tilde = None
            self._active_prompt_len = int(prompt_len)
            self._active_has_prob_text = bool(has_prob_text)
            self._active_slot_idx = slot_idx
            self._active_cot_free = bool(cot_free)
            self._active_tail_nll_inject = bool(tail_nll_inject)
            yield self
        finally:
            self._active_z_raw = prev_z_raw
            self._active_z_tilde = prev_z
            self._active_z_neg_raw = prev_neg_raw
            self._active_z_neg_tilde = prev_neg
            self._active_prompt_len = prev_plen
            self._active_has_prob_text = prev_has_prob
            self._active_slot_idx = prev_slot
            self._active_cot_free = prev_cot
            self._active_tail_nll_inject = prev_tail

    def get_compat_loss(self) -> torch.Tensor:
        if not self._last_compat_losses:
            return torch.zeros((), device=self.center.device, dtype=torch.float32)
        return torch.stack(self._last_compat_losses).mean()

    def get_gate_summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        with torch.no_grad():
            for k, m in self.layer_adapters.items():
                out[f"L{k}_tanh_abs_mean"] = float(torch.tanh(m.gate_alpha).abs().mean().item())
                out[f"L{k}_out_scale"] = float(m.out_scale.item())
            if self.post_norm_adapter is not None:
                out["norm_tanh_abs_mean"] = float(torch.tanh(self.post_norm_adapter.gate_alpha).abs().mean().item())
                out["norm_out_scale"] = float(self.post_norm_adapter.out_scale.item())
        return out


def build_pure_cfmt_prompt(
    tokenizer: Any, prob_text: str, fmt_instr: str = "Put your final answer within \\boxed{}."
) -> List[int]:
    """Build pure C-fmt prompt tokens (NO 64 '1' placeholder tokens)."""
    content = (prob_text + "\n\n" + fmt_instr) if prob_text else fmt_instr
    messages = [{"role": "user", "content": content}]
    templated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(templated, add_special_tokens=False)


def prepare_cfmt_item_batch(
    tokenizer: Any,
    prob_text: str,
    sol_text: str,
    fmt_instr: str = "Put your final answer within \\boxed{}.",
    max_seq_len: int = 1536,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build pure C-fmt input_ids, targets (-100 on prompt), and prompt_len."""
    prompt_tokens = build_pure_cfmt_prompt(tokenizer, prob_text, fmt_instr=fmt_instr)
    prompt_len = len(prompt_tokens)
    sol_tokens = tokenizer.encode(sol_text, add_special_tokens=False)
    if not sol_tokens or sol_tokens[-1] != tokenizer.eos_token_id:
        sol_tokens.append(tokenizer.eos_token_id)

    full_tokens = prompt_tokens + sol_tokens
    if len(full_tokens) > max_seq_len:
        full_tokens = full_tokens[:max_seq_len]

    input_ids = torch.tensor(full_tokens, dtype=torch.long)
    targets = input_ids.clone()
    targets[:prompt_len] = -100
    return input_ids, targets, prompt_len


def _has_boxed_seq(token_ids: Sequence[int]) -> bool:
    """Check whether token_ids contains the 'boxed{' token pair (57296, 236782)."""
    for i in range(len(token_ids) - 1):
        if token_ids[i] == 57296 and token_ids[i + 1] == 236782:
            return True
    return False


def generate_greedy_kv_gca(
    lm_model: Any,
    lm_head: Any,
    embed_tokens: Any,
    bridge: Optional[GatedCrossAttentionBridge],
    input_ids: torch.Tensor,
    z: Optional[torch.Tensor],
    eos_set: Sequence[int] = (1, 106),
    max_new_tokens: int = 512,
    cfmt_cot_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    KV-cached greedy generation with non-invasive GCA bridge hooks.
    1. Prefills `input_ids` with `cot_free=True` to evaluate `bridge`'s structural & compatibility gates
       on the untouched Pure Gemma prompt hidden states.
    2. Preserves Pure Gemma's Chain-of-Thought until the conclusion readout phase triggers (at EOS or tail budget),
       then steps sequentially through the active latent slots `slot_idx = 0, 1, ...` via the `lm_model.norm` hook
       and `lm_head.argmax(dim=-1)`.
    """
    eos_lookup = set(int(x) for x in eos_set)
    prompt_len = int(input_ids.shape[1])
    device = input_ids.device

    with torch.inference_mode():
        inputs_embeds = embed_tokens(input_ids)
        per_layer_inputs = (
            lm_model.get_per_layer_inputs(input_ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
        )
        kwargs_init = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}

        if bridge is not None and z is not None:
            with bridge.active_context(z=z, prompt_len=prompt_len, cot_free=True):
                out = lm_model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True, **kwargs_init)
            g_item = bridge.last_g_item
            g_head = bridge.last_g_head
            g_tail = bridge.last_g_tail
        else:
            out = lm_model(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True, **kwargs_init)
            g_item, g_head, g_tail = 0.0, 0.0, 0.0

        bridge_active = (bridge is not None) and (z is not None) and (g_item > 0.5) and (g_head > 0.5)
        full_slot_mode = bridge_active and (g_tail > 0.5)
        cue_slot_mode = bridge_active and (not full_slot_mode)

        # Fast-path when unperturbed CoT prefix tokens are available (verified bit-identical to step-by-step CoT)
        if cfmt_cot_ids is not None and max_new_tokens >= 64:
            cot_list = [int(x) for x in cfmt_cot_ids]
            if not bridge_active:
                return cot_list[:max_new_tokens]
            if cue_slot_mode and _has_boxed_seq(cot_list):
                return cot_list[:max_new_tokens]

            # Determine exact CoT prefix before the conclusion readout window
            while cot_list and cot_list[-1] in eos_lookup:
                cot_list.pop()
            k_window = 64 if full_slot_mode else 34
            prefix_cot = cot_list[: max(0, max_new_tokens - k_window)]

            full_prefix = torch.cat(
                [input_ids, torch.tensor([prefix_cot], dtype=torch.long, device=device)], dim=1
            ) if prefix_cot else input_ids
            f_emb = embed_tokens(full_prefix)
            f_pli = (
                lm_model.get_per_layer_inputs(full_prefix, None)
                if hasattr(lm_model, "get_per_layer_inputs")
                else None
            )
            f_kw = {"per_layer_inputs": f_pli} if f_pli is not None else {}

            # Run prefill + slot 0 on the final token of full_prefix
            with bridge.active_context(z=z, prompt_len=full_prefix.shape[1], slot_idx=0):
                out_s0 = lm_model(inputs_embeds=f_emb, use_cache=True, return_dict=True, **f_kw)
            past = out_s0.past_key_values
            next_token = lm_head(out_s0.last_hidden_state[:, -1:, :]).argmax(dim=-1)
            gen_ids = list(prefix_cot) + [int(next_token.item())]

            n_active_slots = bridge.k_latent if full_slot_mode else min(10, bridge.k_latent // 2)
            max_tail_steps = bridge.k_latent if full_slot_mode else 34
            seen_box_in_tail = False

            for t_step in range(1, max_tail_steps):
                s_idx = t_step if t_step < n_active_slots else None
                t_emb = embed_tokens(next_token)
                t_pli = (
                    lm_model.get_per_layer_inputs(next_token, None)
                    if hasattr(lm_model, "get_per_layer_inputs")
                    else None
                )
                t_kw = {"per_layer_inputs": t_pli} if t_pli is not None else {}
                with bridge.active_context(z=z, prompt_len=1, slot_idx=s_idx, cot_free=(s_idx is None)):
                    out_t = lm_model(
                        inputs_embeds=t_emb, past_key_values=past, use_cache=True, return_dict=True, **t_kw
                    )
                past = out_t.past_key_values
                next_token = lm_head(out_t.last_hidden_state[:, -1:, :]).argmax(dim=-1)
                tid = int(next_token.item())
                gen_ids.append(tid)
                if tid == 236782:
                    seen_box_in_tail = True
                if tid in eos_lookup and (s_idx is None or seen_box_in_tail):
                    break
            return gen_ids

        # Standard step-by-step generation from token 0
        gen_ids = []
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        next_token = logits.argmax(dim=-1)
        next_id = int(next_token.item())
        gen_ids.append(next_id)
        if next_id in eos_lookup and not bridge_active:
            return gen_ids
        past = out.past_key_values

        k_window = 64 if full_slot_mode else 34
        n_active_slots = (bridge.k_latent if full_slot_mode else min(10, bridge.k_latent // 2)) if bridge else 0
        slot_ptr = -1

        for step_idx in range(1, max_new_tokens):
            if bridge_active and slot_ptr < 0:
                if full_slot_mode and step_idx >= max(1, max_new_tokens - k_window):
                    slot_ptr = 0
                elif cue_slot_mode and (not _has_boxed_seq(gen_ids)) and step_idx >= max(1, max_new_tokens - k_window):
                    slot_ptr = 0

            s_idx = slot_ptr if (0 <= slot_ptr < n_active_slots) else None
            tok_embed = embed_tokens(next_token)
            tok_per_layer = (
                lm_model.get_per_layer_inputs(next_token, None)
                if hasattr(lm_model, "get_per_layer_inputs")
                else None
            )
            kwargs_step = {"per_layer_inputs": tok_per_layer} if tok_per_layer is not None else {}

            if bridge is not None and z is not None:
                with bridge.active_context(z=z, prompt_len=1, slot_idx=s_idx, cot_free=(s_idx is None)):
                    out = lm_model(
                        inputs_embeds=tok_embed, past_key_values=past, use_cache=True, return_dict=True, **kwargs_step
                    )
            else:
                out = lm_model(
                    inputs_embeds=tok_embed, past_key_values=past, use_cache=True, return_dict=True, **kwargs_step
                )

            logits = lm_head(out.last_hidden_state[:, -1:, :])
            cand_token = logits.argmax(dim=-1)
            cand_id = int(cand_token.item())

            # If premature EOS occurs before conclusion slot readout has run, trigger slot_ptr = 0 on this step
            if (
                cand_id in eos_lookup
                and bridge_active
                and slot_ptr < 0
                and (full_slot_mode or not _has_boxed_seq(gen_ids))
            ):
                slot_ptr = 0
                with bridge.active_context(z=z, prompt_len=1, slot_idx=0, cot_free=False):
                    out = lm_model(
                        inputs_embeds=tok_embed, past_key_values=past, use_cache=True, return_dict=True, **kwargs_step
                    )
                logits = lm_head(out.last_hidden_state[:, -1:, :])
                cand_token = logits.argmax(dim=-1)
                cand_id = int(cand_token.item())
                slot_ptr = 1
            elif slot_ptr >= 0:
                slot_ptr += 1

            next_token = cand_token
            next_id = cand_id
            gen_ids.append(next_id)
            past = out.past_key_values
            if next_id in eos_lookup and (slot_ptr < 0 or slot_ptr > n_active_slots):
                break

    return gen_ids


def _self_test() -> None:
    """Self-audit unit test verifying:
      1. Zero-init exact identity (alpha == 0 -> max_diff == 0.0)
      2. Mean-latent exact identity (z == center -> max_diff == 0.0 even when alpha != 0)
      3. Non-zero gradient wrt gate_alpha at step 0
      4. Teacher-forced vs Prefill+Decode step equivalence
    """
    torch.manual_seed(42)
    d_model = 64
    k_latent = 8
    vocab = 100

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            return self.lin(x)

    class MockLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, d_model)
            self.layers = nn.ModuleList([MockLayer() for _ in range(4)])
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab, bias=False)

        def forward(self, inputs_embeds, **kwargs):
            h = inputs_embeds
            for l in self.layers:
                h = l(h)
            h = self.norm(h)

            class Out:
                pass

            o = Out()
            o.last_hidden_state = h
            return o

    lm = MockLM().eval()
    for p in lm.parameters():
        p.requires_grad = False

    bridge = GatedCrossAttentionBridge(
        d_model=d_model,
        k_latent=k_latent,
        hook_layers=(1, 3),
        use_post_norm=True,
        n_heads=4,
        d_qk_head=8,
        r_val=16,
    )
    bridge.center.copy_(torch.randn(k_latent, d_model))
    bridge.attach(lm)
    n_params = sum(p.numel() for p in bridge.parameters())

    ids = torch.randint(0, vocab, (1, 16))
    prompt_len = 5
    z = bridge.center.unsqueeze(0) + torch.randn(1, k_latent, d_model) * 3.0
    z_neg = bridge.center.unsqueeze(0) + torch.randn(4, k_latent, d_model) * 3.0

    with torch.no_grad():
        base_h = lm(lm.embed_tokens(ids)).last_hidden_state
        with bridge.active_context(z=z, prompt_len=prompt_len):
            zero_init_h = lm(lm.embed_tokens(ids)).last_hidden_state
    diff_zero_init = float((base_h - zero_init_h).abs().max().item())
    assert diff_zero_init == 0.0, f"Zero-init parity failed: {diff_zero_init}"

    # Check step-0 gradient wrt gate_alpha (including multi-negative compatibility loss)
    bridge.train()
    with bridge.active_context(z=z, prompt_len=prompt_len, z_neg=z_neg):
        out_h = lm(lm.embed_tokens(ids)).last_hidden_state
        logits = lm.lm_head(out_h[:, :-1, :])
        loss = F.cross_entropy(logits.reshape(-1, vocab), ids[:, 1:].reshape(-1)) + 0.1 * bridge.get_compat_loss()
    loss.backward()
    grad_norm_alpha = sum(
        float(m.gate_alpha.grad.abs().sum().item())
        for m in list(bridge.layer_adapters.values()) + ([bridge.post_norm_adapter] if bridge.post_norm_adapter else [])
    )
    assert grad_norm_alpha > 0.0, "Step-0 gradient wrt gate_alpha must be non-zero!"

    # Open gates and check mean-latent (z == center) exact identity
    bridge.eval()
    with torch.no_grad():
        for m in list(bridge.layer_adapters.values()) + ([bridge.post_norm_adapter] if bridge.post_norm_adapter else []):
            m.gate_alpha.fill_(0.5)
        with bridge.active_context(z=bridge.center.unsqueeze(0), prompt_len=prompt_len):
            mean_lat_h = lm(lm.embed_tokens(ids)).last_hidden_state
        diff_mean_lat = float((base_h - mean_lat_h).abs().max().item())
        assert diff_mean_lat == 0.0, f"Mean-latent zero-bias property failed: {diff_mean_lat}"

        # Check token-by-token equivalence
        with bridge.active_context(z=z, prompt_len=prompt_len):
            full_h = lm(lm.embed_tokens(ids)).last_hidden_state
        with bridge.active_context(z=z, prompt_len=prompt_len):
            prefill_h = lm(lm.embed_tokens(ids[:, :prompt_len])).last_hidden_state
        step_hs = [prefill_h]
        for s_idx in range(1, ids.shape[1] - prompt_len + 1):
            tok_id = ids[:, prompt_len + s_idx - 1 : prompt_len + s_idx]
            with bridge.active_context(z=z, prompt_len=1):
                sh = lm(lm.embed_tokens(tok_id)).last_hidden_state
            step_hs.append(sh)
        cat_h = torch.cat(step_hs, dim=1)
        diff_step = float((full_h - cat_h).abs().max().item())
        assert diff_step < 5e-4, f"Step equivalence failed: {diff_step}"

    print(
        f"[SELF-TEST PASS] params={n_params} | Zero-init diff=0.0 | Mean-latent diff=0.0 | "
        f"Step-0 alpha grad={grad_norm_alpha:.4e} | Step equiv f32 diff={diff_step:.2e}"
    )


if __name__ == "__main__":
    _self_test()
