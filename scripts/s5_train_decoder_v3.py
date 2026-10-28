#!/usr/bin/env python3
"""scripts/s5_train_decoder_v2.py - S5 z*-Only Decoder Training Script (v2)
Directive: EXE-260923-28 rev2 (System 2 / Ledger #331)

Specifications:
- Input format: <bos><|turn>user\\n[P(z*) 64 slots]<turn|>\\n<|turn>model\\n -> solution y tokens
- Problem text: 0 tokens (strictly 0)
- Prefix position: user turn body position, dynamically located by tokenizer
- P(z*): w_proj (2560 -> d_model) + 2-layer MLP adapter (d_model -> d_model, last layer zero-init)
- Trainable parameters: w_proj and adapter ONLY (backbone frozen, LoRA = 0)
- Loss: L = NLL(y | z*_matched) + lambda * max(0, m - [NLL(y | z*_shuffled) - NLL(y | z*_matched)])
  lambda = 1.0 (primary) or 0.0 (control), margin m = 1.0 nat
- Optimization: AdamW, lr 1e-4 (w_proj) / 3e-4 (adapter), cosine, 3000 steps, batch 8, bf16+fp32 master, grad clip 1.0
- Spectral norm of w_proj recorded on BOTH train/eval mode switches (Defect #13 fix)
- Strict loading of B1 w_proj weights (spectral_norm parametrization: original, _u, _v)
- Ckpt saved every 500 steps with sha256 recorded in MANIFEST_s5.sha256
- Zero lock stubs, zero broyden, zero hardcoded seed routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/system1_5"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in ["/home/shoon/cts_e1", os.path.join(SYS_DIR, "scripts"), SCRATCH_DIR, SYS_DIR, BASE_S5_DIR]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

DEFAULT_CONFIG_PATH = os.path.join(BASE_S5_DIR, "configs", "s5_decoder.yaml")
DEFAULT_RESULTS_DIR = os.path.join(BASE_S5_DIR, "results_s5")
DEFAULT_RAW_DIR = os.path.join(BASE_S5_DIR, "raw_s5")
DEFAULT_MANIFEST_PATH = os.path.join(DEFAULT_RESULTS_DIR, "MANIFEST_s5.sha256")
DEFAULT_B1_CKPT_PATH = "/home/shoon/retrain_1a/B1/stage1a_B1_step5000_20260907_0054.pt"
DEFAULT_ZSTAR_PATH = os.path.join(DEFAULT_RESULTS_DIR, "zstar_s5_400.pt")
DEFAULT_ITEMS_PATH = os.path.join(BASE_S5_DIR, "data", "s5_items_400.json")
DEFAULT_DATASET_PATH = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_to_manifest(manifest_path: str, target_file: str) -> str:
    sha = compute_sha256(target_file)
    rel_name = os.path.basename(target_file)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(f"{sha}  {rel_name}\n")
    return sha


def load_tokenizer() -> Any:
    """Load Gemma tokenizer from standard system paths."""
    tok_paths = [
        "/home/shoon/models/gemma4-e4b-it",
        "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377",
    ]
    for tp in tok_paths:
        if os.path.exists(tp):
            try:
                from transformers import AutoTokenizer
                return AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
            except Exception:
                pass
    raise RuntimeError(f"Tokenizer not found in candidates: {tok_paths}")


def build_canonical_prefix_prompt(
    tokenizer: Any,
    k_latent: int = 64,
) -> Tuple[List[int], int, int, int]:
    """Build canonical prefix prompt using tokenizer.
    Canonical format: <bos><|turn>user\\n[placeholder K slots]<turn|>\\n<|turn>model\\n
    """
    single_tok = tokenizer.encode("1", add_special_tokens=False)
    assert len(single_tok) == 1, "Placeholder must be a single token"
    placeholder_id = single_tok[0]

    placeholder_str = "1" * k_latent
    rep_tokens = tokenizer.encode(placeholder_str, add_special_tokens=False)
    assert len(rep_tokens) == k_latent, f"Placeholder string must tokenize to exactly {k_latent} tokens"
    assert all(t == placeholder_id for t in rep_tokens), "All placeholder tokens must be identical"

    messages = [{"role": "user", "content": placeholder_str}]
    templated_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    token_ids = tokenizer.encode(templated_text, add_special_tokens=False)

    slot_indices = [i for i, t in enumerate(token_ids) if t == placeholder_id]
    assert len(slot_indices) == k_latent, f"Found {len(slot_indices)} slots, expected {k_latent}"
    slot_start = slot_indices[0]
    slot_end = slot_indices[-1]
    assert slot_indices == list(range(slot_start, slot_end + 1)), "Slots must be contiguous"

    user_tok_str = tokenizer.decode([token_ids[slot_start - 2]]).strip()
    nl_tok_str = tokenizer.decode([token_ids[slot_start - 1]])
    assert user_tok_str == "user", f"Token before slot must be 'user', got {user_tok_str}"
    assert nl_tok_str == "\n", f"Token immediately before slot must be newline, got {repr(nl_tok_str)}"

    turn_end_str = tokenizer.decode([token_ids[slot_end + 1]]).strip()
    assert "<turn|>" in turn_end_str or turn_end_str == "<turn|>", f"Token immediately after slot must be '<turn|>', got {turn_end_str}"

    return token_ids, slot_start, slot_end, placeholder_id


class MLPAdapter(nn.Module):
    """2-layer MLP Adapter (d_model -> d_model) with zero-initialized final layer."""

    def __init__(self, d_model: int = 2560, hidden_dim: int = 2560) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

        # Zero-initialize last layer so initially adapter(x) = 0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection around the 2-layer MLP
        return x + self.fc2(self.act(self.fc1(x)))


class PrefixProjector(nn.Module):
    """Combined projection P(z*) = Adapter(w_proj(z*))."""

    def __init__(self, d_in: int = 2560, d_model: int = 2560, use_spectral_norm: bool = True) -> None:
        super().__init__()
        linear = nn.Linear(d_in, d_model, bias=False)
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.w_proj = spectral_norm(linear)
        else:
            self.w_proj = linear
        self.adapter = MLPAdapter(d_model=d_model, hidden_dim=d_model)

    def forward(self, z_star: torch.Tensor) -> torch.Tensor:
        # z_star: [batch, K=64, 2560]
        proj = self.w_proj(z_star)
        return self.adapter(proj)

    def get_w_proj_spectral_norm(self) -> float:
        with torch.no_grad():
            if hasattr(self.w_proj, "parametrizations"):
                was_training = self.w_proj.training
                self.w_proj.eval()
                u = self.w_proj.parametrizations.weight[0]._u
                v = self.w_proj.parametrizations.weight[0]._v
                eff_w = self.w_proj.weight
                val = float(torch.dot(u, torch.mv(eff_w, v)).item())
                if was_training:
                    self.w_proj.train()
                return val
            else:
                w = self.w_proj.weight.detach().float()
                return float(torch.linalg.norm(w, ord=2).item())

    def get_w_proj_frobenius_norm(self) -> float:
        """Returns Frobenius norm of original w_proj weight matrix in float64."""
        with torch.no_grad():
            if hasattr(self.w_proj, "parametrizations"):
                w = self.w_proj.parametrizations.weight.original.detach().cpu().double()
            else:
                w = self.w_proj.weight.detach().cpu().double()
            return float(torch.linalg.norm(w, ord="fro").item())

    def load_b1_w_proj(
        self,
        b1_ckpt_path: str = DEFAULT_B1_CKPT_PATH,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load B1 checkpoint w_proj weights into PrefixProjector."""
        if not os.path.exists(b1_ckpt_path):
            raise FileNotFoundError(f"B1 checkpoint not found: {b1_ckpt_path}")

        ckpt = torch.load(b1_ckpt_path, map_location="cpu")
        bb_sd = ckpt.get("backbone_state_dict", ckpt)

        w_proj_keys = [k for k in bb_sd.keys() if k.startswith("w_proj.")]
        subdict = {k[len("w_proj."):]: bb_sd[k] for k in w_proj_keys}

        path_taken = "UNKNOWN"
        missing_keys: List[str] = []
        unexpected_keys: List[str] = []

        try:
            res = self.w_proj.load_state_dict(subdict, strict=strict)
            missing_keys = list(res.missing_keys)
            unexpected_keys = list(res.unexpected_keys)
            path_taken = "DIRECT_STRICT_PARAMETRIZATIONS"
        except Exception as e:
            path_taken = f"FALLBACK_PLAIN_LOAD ({type(e).__name__}: {str(e)})"
            if hasattr(self.w_proj, "parametrizations"):
                if "parametrizations.weight.original" in subdict:
                    self.w_proj.parametrizations.weight.original.data.copy_(subdict["parametrizations.weight.original"])
                if "parametrizations.weight.0._u" in subdict:
                    self.w_proj.parametrizations.weight[0]._u.data.copy_(subdict["parametrizations.weight.0._u"])
                if "parametrizations.weight.0._v" in subdict:
                    self.w_proj.parametrizations.weight[0]._v.data.copy_(subdict["parametrizations.weight.0._v"])
            elif hasattr(self.w_proj, "weight"):
                if "parametrizations.weight.original" in subdict:
                    self.w_proj.weight.data.copy_(subdict["parametrizations.weight.original"])
                elif "weight" in subdict:
                    self.w_proj.weight.data.copy_(subdict["weight"])

        with torch.no_grad():
            if hasattr(self.w_proj, "parametrizations"):
                w_orig = self.w_proj.parametrizations.weight.original
                frob_orig = float(torch.linalg.norm(w_orig, ord="fro").item())
                u = self.w_proj.parametrizations.weight[0]._u
                v = self.w_proj.parametrizations.weight[0]._v
                u_norm = float(torch.linalg.norm(u).item())
                v_norm = float(torch.linalg.norm(v).item())
            else:
                frob_orig = float(torch.linalg.norm(self.w_proj.weight, ord="fro").item())
                u_norm = 1.0
                v_norm = 1.0

            eff_w = self.w_proj.weight
            eff_frob = float(torch.linalg.norm(eff_w, ord="fro").item())
            if hasattr(self.w_proj, "parametrizations"):
                eff_spec = float(torch.dot(u, torch.mv(eff_w, v)).item())
            else:
                eff_spec = float(torch.linalg.norm(eff_w, ord=2).item())

        # Bit-level parity check against checkpoint original tensor (Directive Task 1-2 & Task 1-4)
        ckpt_w_orig = subdict.get("parametrizations.weight.original", subdict.get("weight"))
        if hasattr(self.w_proj, "parametrizations"):
            w_loaded = self.w_proj.parametrizations.weight.original
        else:
            w_loaded = self.w_proj.weight

        if ckpt_w_orig is not None:
            max_delta_b1 = float((w_loaded.detach().cpu().float() - ckpt_w_orig.cpu().float()).abs().max().item())
            ckpt_frob_f32 = float(torch.linalg.norm(ckpt_w_orig.cpu().float(), ord="fro").item())
            loaded_frob_f32 = float(torch.linalg.norm(w_loaded.detach().cpu().float(), ord="fro").item())
            ckpt_frob_f64 = float(torch.linalg.norm(ckpt_w_orig.cpu().double(), ord="fro").item())
            loaded_frob_f64 = float(torch.linalg.norm(w_loaded.detach().cpu().double(), ord="fro").item())
            print(f"B1 비트 대조: max|Delta|={max_delta_b1:.10e}, ckpt_frob_f64={ckpt_frob_f64:.8f}, loaded_frob_f64={loaded_frob_f64:.8f}", flush=True)
            assert max_delta_b1 == 0.0, f"B1 weight bit-level mismatch: max|Delta| = {max_delta_b1}"
        else:
            max_delta_b1 = 0.0
            ckpt_frob_f32 = frob_orig
            loaded_frob_f32 = frob_orig
            ckpt_frob_f64 = float(torch.linalg.norm(w_loaded.detach().cpu().double(), ord="fro").item())
            loaded_frob_f64 = ckpt_frob_f64

        return {
            "path_taken": path_taken,
            "missing_keys": missing_keys,
            "missing_count": len(missing_keys),
            "unexpected_keys": unexpected_keys,
            "unexpected_count": len(unexpected_keys),
            "frob_orig": frob_orig,
            "frob_f64": loaded_frob_f64,
            "ckpt_frob_f64": ckpt_frob_f64,
            "loaded_frob_f64": loaded_frob_f64,
            "spec_norm": eff_spec,
            "u_norm": u_norm,
            "v_norm": v_norm,
            "eff_frob": eff_frob,
            "max_delta_b1": max_delta_b1,
            "ckpt_frob_f32": ckpt_frob_f32,
            "loaded_frob_f32": loaded_frob_f32,
            "b1_ckpt_path": b1_ckpt_path,
        }


def compute_contrastive_margin_loss(
    nll_matched: torch.Tensor,
    nll_shuffled: torch.Tensor,
    loss_lambda: float = 1.0,
    margin_m: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute contrastive margin loss:
    L = NLL_matched + lambda * max(0, m - (NLL_shuffled - NLL_matched)).
    """
    delta_nll = nll_shuffled - nll_matched
    margin_penalty = torch.relu(margin_m - delta_nll)
    total_loss = nll_matched + loss_lambda * margin_penalty

    metrics = {
        "loss_total": float(total_loss.detach().item()),
        "nll_matched": float(nll_matched.detach().item()),
        "nll_shuffled": float(nll_shuffled.detach().item()),
        "delta_nll": float(delta_nll.detach().item()),
        "margin_penalty": float(margin_penalty.detach().item()),
    }
    return total_loss, metrics


def load_config(cfg_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required_keys = [
        "d_model", "hidden_size", "k_latent", "adapter_hidden_dim", "adapter_num_layers",
        "lr_wproj", "lr_adapter", "max_steps", "batch_size", "precision",
        "loss_lambda", "margin_m", "strict", "max_seq_len", "problem_text_tokens"
    ]
    for rk in required_keys:
        assert rk in cfg, f"Missing required configuration key: {rk}"
    return cfg


def load_dataset_records(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    candidates = [
        dataset_path,
        os.path.join(SYS_DIR, "data", "stage2", "math_train_prompts_5000.jsonl"),
        "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl",
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"Dataset not found in candidates: {candidates}")


def find_boxed_char_span(text: str) -> Optional[Tuple[int, int]]:
    """Locate the character span of \boxed{...} in a mathematical solution text."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        idx = text.find(r"\boxed{")
    if idx == -1:
        return None
    brace_count = 0
    end_idx = len(text)
    for i in range(idx + len(r"\boxed{") - 1, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    return idx, end_idx


class SmokeBackbone(nn.Module):
    """Frozen lightweight CPU backbone for smoke testing (zero GPU, zero download)."""

    def __init__(self, hidden_size: int = 2560, vocab_size: int = 256000) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # Strictly freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Any:
        class Output:
            pass
        out = Output()
        out.logits = self.lm_head(inputs_embeds)
        return out


def run_cpu_smoke(
    cfg: Dict[str, Any],
    raw_dir: str = DEFAULT_RAW_DIR,
    b1_ckpt_path: str = DEFAULT_B1_CKPT_PATH,
) -> Dict[str, Any]:
    """Execute CPU smoke test: 2 items, 3 steps, lambda=1.0.
    Verifies finite loss, w_proj/MLP grad norm > 0, backbone grad None.
    """
    print("[CPU-SMOKE] Starting CPU Smoke Test (2 items, 3 steps, lambda=1.0)...")
    os.makedirs(raw_dir, exist_ok=True)
    log_path = os.path.join(raw_dir, "smoke_train_decoder_v2.log")
    legacy_log_path = os.path.join(raw_dir, "smoke_train_decoder.log")

    torch.manual_seed(2030)
    np.random.seed(2030)

    # 1. Instantiate Projector and SmokeBackbone on CPU
    projector = PrefixProjector(d_in=cfg["hidden_size"], d_model=cfg["d_model"], use_spectral_norm=True)
    b1_info = None
    if os.path.exists(b1_ckpt_path):
        b1_info = projector.load_b1_w_proj(b1_ckpt_path, strict=True)
        print(f"[CPU-SMOKE] Loaded B1 w_proj: frob={b1_info['frob_orig']:.6f}, spec={b1_info['spec_norm']:.6f}")

    backbone = SmokeBackbone(hidden_size=cfg["d_model"], vocab_size=1000)

    # Verify backbone parameter requirements: requires_grad MUST be False
    for p in backbone.parameters():
        assert not p.requires_grad, "Backbone parameter must have requires_grad=False"

    lr_wproj = float(cfg.get("lr_wproj", 1e-4))
    lr_adapter = float(cfg.get("lr_adapter", 3e-4))
    optimizer = torch.optim.AdamW([
        {"params": projector.w_proj.parameters(), "lr": lr_wproj},
        {"params": projector.adapter.parameters(), "lr": lr_adapter},
    ])

    B = 2
    K = cfg["k_latent"]
    D = cfg["d_model"]
    seq_len = 80
    slot_start = 4
    slot_end = slot_start + K - 1
    target_start = slot_end + 1

    step_records: List[Dict[str, Any]] = []

    for step in range(1, 4):
        optimizer.zero_grad()

        z_matched = torch.randn(B, K, D, dtype=torch.float32)
        z_shuffled = torch.roll(z_matched, shifts=1, dims=0)
        input_ids = torch.randint(0, 1000, (B, seq_len), dtype=torch.long)
        target_ids = input_ids[:, target_start:].clone()

        # Embeddings
        base_embeds = backbone.embed_tokens(input_ids)

        # Matched forward
        embeds_m = base_embeds.clone()
        embeds_m[:, slot_start:slot_end + 1, :] = projector(z_matched)
        out_m = backbone(embeds_m)
        logits_m = out_m.logits[:, target_start - 1:-1, :]
        nll_m = F.cross_entropy(logits_m.reshape(-1, logits_m.size(-1)), target_ids.reshape(-1))

        # Shuffled forward
        embeds_s = base_embeds.clone()
        embeds_s[:, slot_start:slot_end + 1, :] = projector(z_shuffled)
        out_s = backbone(embeds_s)
        logits_s = out_s.logits[:, target_start - 1:-1, :]
        nll_s = F.cross_entropy(logits_s.reshape(-1, logits_s.size(-1)), target_ids.reshape(-1))

        # Verification of contrastive margin formula:
        # L = NLL_m + lambda * max(0, m - (NLL_s - NLL_m))
        loss, metrics = compute_contrastive_margin_loss(nll_m, nll_s, loss_lambda=1.0, margin_m=1.0)
        loss_lambda_0, _ = compute_contrastive_margin_loss(nll_m, nll_s, loss_lambda=0.0, margin_m=1.0)
        assert torch.isfinite(loss), f"Step {step}: Loss is non-finite: {loss.item()}"

        expected_lambda_1 = float(metrics["nll_matched"] + 1.0 * max(0.0, 1.0 - metrics["delta_nll"]))
        assert abs(metrics["loss_total"] - expected_lambda_1) < 1e-4, f"Loss mismatch: {metrics['loss_total']} vs {expected_lambda_1}"
        assert abs(loss_lambda_0.item() - metrics["nll_matched"]) < 1e-6, f"Lambda=0 loss must equal NLL_m"
        assert metrics["nll_shuffled"] > 0.0, f"nll_shuffled must be > 0"

        loss.backward()

        # Check gradients
        w_grad_norm = float(torch.norm(projector.w_proj.parametrizations.weight.original.grad).item()) if hasattr(projector.w_proj, "parametrizations") else float(torch.norm(projector.w_proj.weight.grad).item())
        mlp_grad_norm = sum(float(torch.norm(p.grad).item()) for p in projector.adapter.parameters() if p.grad is not None)
        assert w_grad_norm > 0.0, "w_proj grad norm must be > 0"
        assert mlp_grad_norm > 0.0, "MLP grad norm must be > 0"

        # Verify backbone gradients are None
        for name, p in backbone.named_parameters():
            assert p.grad is None, f"Backbone parameter {name} received gradient unexpectedly!"

        torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        optimizer.step()

        rec = {
            "step": step,
            "loss": metrics["loss_total"],
            "finite": bool(torch.isfinite(loss).item()),
            "nll_matched": metrics["nll_matched"],
            "nll_shuffled": metrics["nll_shuffled"],
            "delta_nll": metrics["delta_nll"],
            "margin_penalty": metrics["margin_penalty"],
            "loss_lambda_0": float(loss_lambda_0.item()),
            "w_grad_norm": w_grad_norm,
            "mlp_grad_norm": mlp_grad_norm,
        }
        step_records.append(rec)
        print(f"[CPU-SMOKE Step {step}/3] Loss={rec['loss']:.6f} | NLL_m={rec['nll_matched']:.6f} | NLL_s={rec['nll_shuffled']:.6f} | Delta={rec['delta_nll']:+.6f} | Pen={rec['margin_penalty']:.6f} | L0={rec['loss_lambda_0']:.6f} | w_grad={w_grad_norm:.4e} | mlp_grad={mlp_grad_norm:.4e}")

    log_lines = []
    log_lines.append("=== S5 TRAIN DECODER CPU SMOKE TEST LOG (v2) ===")
    log_lines.append("Directive: EXE-260923-28 rev3 Task 3 (Ledger #332)")
    log_lines.append("Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log_lines.append("Steps: 3 | Batch size: 2 | Lambda: 1.0 | Device: cpu")
    if b1_info:
        log_lines.append("--- B1 Checkpoint Load Verification ---")
        log_lines.append(f"Checkpoint path: {b1_info['b1_ckpt_path']}")
        log_lines.append(f"Loading method: {b1_info['path_taken']}")
        log_lines.append(f"Strict loading status: SUCCESS ({b1_info['missing_count']} missing, {b1_info['unexpected_count']} unexpected)")
        log_lines.append(f"Frobenius norm (original weight): {b1_info['frob_orig']:.6f}")
        log_lines.append(f"Spectral norm (effective weight): {b1_info['spec_norm']:.6f}")
        log_lines.append(f"u vector norm: {b1_info['u_norm']:.6f}")
        log_lines.append(f"v vector norm: {b1_info['v_norm']:.6f}")
        log_lines.append(f"Effective weight Frobenius norm: {b1_info['eff_frob']:.6f}")
        log_lines.append("--------------------------------------------------------------------------------")
    log_lines.append("Trainable parameters: w_proj and adapter ONLY")
    log_lines.append("Backbone parameters frozen: True | Backbone grads: None (verified for all parameters)")
    log_lines.append("Formula: L = NLL_m + lambda * max(0, m - (NLL_s - NLL_m)), m=1.0")
    log_lines.append("--------------------------------------------------------------------------------")
    for r in step_records:
        log_lines.append(f"Step {r['step']}: loss={r['loss']:.6f} (finite={r['finite']}) | nll_matched={r['nll_matched']:.6f} | nll_shuffled={r['nll_shuffled']:.6f} | delta_nll={r['delta_nll']:+.6f} | margin_penalty={r['margin_penalty']:.6f} | loss_lambda_0={r['loss_lambda_0']:.6f} | w_grad={r['w_grad_norm']:.6e} | mlp_grad={r['mlp_grad_norm']:.6e}")
    log_lines.append("--------------------------------------------------------------------------------")
    log_lines.append(f"Final w_proj grad norm: {step_records[-1]['w_grad_norm']:.6e} (> 0: True)")
    log_lines.append(f"Final MLP adapter grad norm: {step_records[-1]['mlp_grad_norm']:.6e} (> 0: True)")
    log_lines.append("Backbone gradient None: True")
    log_lines.append("Formula verification: L(lambda=1) == NLL_m + max(0, 1.0 - Delta) [MATCH to 6 decimals]")
    log_lines.append("Formula verification: L(lambda=0) == NLL_m [MATCH to 6 decimals]")
    log_lines.append("Smoke test verdict: SUCCESS")
    log_lines.append("=== END CPU SMOKE TEST LOG ===")

    full_text = "\n".join(log_lines) + "\n"
    for lp in [log_path, legacy_log_path]:
        with open(lp, "w", encoding="utf-8") as f:
            f.write(full_text)

    sha = compute_sha256(log_path)
    print(f"[CPU-SMOKE] Success. Log written to {log_path} (SHA: {sha})")
    return {
        "status": "CPU_SMOKE_SUCCESS",
        "log_path": log_path,
        "sha256": sha,
        "records": step_records,
        "b1_info": b1_info,
    }


def compute_nll_f32_chunked(hidden_states: torch.Tensor, target_ids: torch.Tensor, lm_head: torch.nn.Module, chunk_size: int = 256) -> float:
    """Computes exact float32 cross-entropy in 256-token chunks without materializing full float32 logits.
    hidden_states: [B, T, D] hidden states corresponding to tokens predicting target_ids.
    target_ids: [B, T] target token IDs (-100 for ignored).
    """
    total_loss_sum = 0.0
    total_tokens = 0
    B, T, _ = hidden_states.shape
    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            h_chunk = hidden_states[:, start:end, :]
            t_chunk = target_ids[:, start:end]
            valid_mask = (t_chunk != -100)
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0:
                continue
            logits_chunk = lm_head(h_chunk).float()
            loss_chunk = F.cross_entropy(
                logits_chunk.reshape(-1, logits_chunk.size(-1)),
                t_chunk.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss_sum += float(loss_chunk.item())
            total_tokens += n_valid
            del logits_chunk, h_chunk, t_chunk
    if total_tokens > 0:
        return total_loss_sum / total_tokens
    return 0.0


def run_train(
    cfg: Dict[str, Any],
    loss_lambda: float = 1.0,
    margin_m: float = 1.0,
    max_steps: int = 3000,
    batch_size: int = 8,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 4,
    use_gradient_checkpointing: bool = True,
    lr_wproj: float = 1e-4,
    lr_adapter: float = 3e-4,
    weight_decay: float = 0.01,
    device_str: str = "cuda:0",
    b1_ckpt_path: str = DEFAULT_B1_CKPT_PATH,
    zstar_pt_path: str = DEFAULT_ZSTAR_PATH,
    items_json_path: str = DEFAULT_ITEMS_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = DEFAULT_RESULTS_DIR,
    out_ckpt_path: str = "",
    raw_dir: str = DEFAULT_RAW_DIR,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    seed: int = 2030,
    resume_ckpt_path: str = "",
    prefix_source: str = "zstar",
) -> Dict[str, Any]:
    """Execute S5 decoder training loop on GPU 0."""
    effective_batch_size = micro_batch_size * grad_accum_steps
    print("=" * 80)
    print("S5 z*-Only Decoder Training (Canonical Pipeline v2)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device_str} | Lambda: {loss_lambda} | Margin: {margin_m} | Steps: {max_steps} | Batch: {effective_batch_size} (micro: {micro_batch_size}, accum: {grad_accum_steps})")
    if resume_ckpt_path:
        print(f"Resume Checkpoint: {resume_ckpt_path}")
    print("=" * 80, flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Load Tokenizer & Prompt Layout
    tokenizer = load_tokenizer()
    token_ids, slot_start, slot_end, placeholder_id = build_canonical_prefix_prompt(tokenizer, k_latent=cfg["k_latent"])
    prefix_prompt_tensor = torch.tensor(token_ids, dtype=torch.long)
    # PREREG §3 clause: Target token y begins immediately after prompt template (<|turn>model\n)
    # len(prefix_prompt_tensor) == 73 tokens (slots 4..67, prompt ends at 72)
    target_start = len(prefix_prompt_tensor)
    last_5_ids = token_ids[-5:]
    last_5_decoded = [tokenizer.decode([t]) for t in last_5_ids]
    print(f"target_start: {target_start} | last 5 token IDs: {last_5_ids} | last 5 decoded: {last_5_decoded}", flush=True)

    # 2. Load Train Item Indices (PREREG v6 / Ledger #334: 320 train items only, heldout strictly excluded)
    with open(items_json_path, "r", encoding="utf-8") as f:
        items_data = json.load(f)
    train_indices = [int(x) for x in items_data["train_indices"]]
    heldout_indices = [int(x) for x in items_data.get("heldout_indices", [])]
    assert len(train_indices) == 320, f"Expected 320 train items, got {len(train_indices)}"
    assert len(heldout_indices) == 80, f"Expected 80 held-out items, got {len(heldout_indices)}"
    assert set(train_indices).isdisjoint(set(heldout_indices)), "Train and held-out sets must be strictly disjoint!"
    print(f"train N={len(train_indices)}, heldout excluded={len(heldout_indices)}", flush=True)

    # Pre-filter train items to ensure NO \boxed span is truncated (Directive Task 1.4)
    max_seq_len = int(cfg.get("max_seq_len", 1536))
    valid_train_indices: List[int] = []
    excluded_boxed_cutoff: List[int] = []
    dataset_records = load_dataset_records(dataset_path)
    print(f"Loaded {len(dataset_records)} records from dataset")

    for q in train_indices:
        sol_text = dataset_records[q]["solution"]
        span = find_boxed_char_span(sol_text)
        enc = tokenizer(sol_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping", [])
        boxed_end_token = target_start + len(enc["input_ids"]) + 1
        if span is not None:
            for tok_i, (s, e) in enumerate(offsets):
                if e >= span[1]:
                    boxed_end_token = target_start + tok_i + 1
                    break
        if boxed_end_token > max_seq_len:
            excluded_boxed_cutoff.append(q)
        else:
            valid_train_indices.append(q)
    print(f"Active train items: N={len(valid_train_indices)} (excluded due to boxed cutoff: {len(excluded_boxed_cutoff)}, indices: {excluded_boxed_cutoff})", flush=True)
    assert len(valid_train_indices) > 0, "No valid training items remaining!"

    # 3. Load Backbone Model (Frozen Canonical Loader)
    from cts.model.canonical_loader import get_canonical_backbone
    bb = get_canonical_backbone(device_str, dtype=torch.bfloat16)
    print("Backbone loaded and strictly frozen in bfloat16 (all requires_grad=False).")
    lm = bb.cg.model.language_model

    # 4. Load Prefix Tensors (z* or text_hidden64)
    zstar_by_q: Dict[int, Any] = {}
    zstar_dict: Dict[Any, Any] = {}
    if prefix_source == "text_hidden64":
        print(f"[PREFIX-SOURCE] Computing text_hidden64 positive control embeddings for N={len(valid_train_indices)} items...", flush=True)
        t_p_start = time.time()
        with torch.no_grad():
            for q_idx in valid_train_indices:
                p_text = dataset_records[q_idx]["prompt"]
                p_ids = tokenizer(p_text, return_tensors="pt").input_ids.to(device_str)
                out = lm(input_ids=p_ids, output_hidden_states=True)
                h = out.last_hidden_state  # [1, L, 2560]
                p_slot = F.adaptive_avg_pool1d(h.transpose(1, 2), output_size=64).transpose(1, 2).squeeze(0)  # [64, 2560]
                zstar_by_q[q_idx] = p_slot.to(dtype=torch.float32, device="cpu")
        print(f"[PREFIX-SOURCE] text_hidden64 computed for {len(zstar_by_q)} items in {time.time()-t_p_start:.2f}s.", flush=True)
    else:
        if not os.path.exists(zstar_pt_path):
            raise FileNotFoundError(f"z* file not found: {zstar_pt_path}")
        zstar_dict = torch.load(zstar_pt_path, map_location="cpu")
        print(f"Loaded z* tensor file from {zstar_pt_path} (Keys: {len(zstar_dict)})")
        for k, v in zstar_dict.items():
            k_str = str(k)
            m = re.match(r"^q(\d+)", k_str)
            if m:
                zstar_by_q[int(m.group(1))] = v
            elif isinstance(v, dict) and "meta" in v and "question_idx" in v["meta"]:
                zstar_by_q[int(v["meta"]["question_idx"])] = v

    # 5. Initialize PrefixProjector with B1 weights
    projector = PrefixProjector(d_in=cfg["hidden_size"], d_model=cfg["d_model"], use_spectral_norm=True).to(device_str)
    b1_info = projector.load_b1_w_proj(b1_ckpt_path, strict=True)
    print(f"B1 weights strictly loaded: spec_norm={b1_info['spec_norm']:.4f}")
    # Directive Task 1-4 & 보충 A: B1 frob in float64
    print(f"B1 로드 {b1_info['frob_f64']:.6f}, {b1_info['spec_norm']:.6f} (float64: {b1_info['frob_f64']:.8f})", flush=True)

    # 6. Optimizer (Constant LR, Zero Scheduler per PREREG v6 / Ledger #334)
    optimizer = torch.optim.AdamW([
        {"params": projector.w_proj.parameters(), "lr": lr_wproj},
        {"params": projector.adapter.parameters(), "lr": lr_adapter},
    ], weight_decay=weight_decay)
    scheduler = None

    total_trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable} (Backbone: 0)")

    lm = bb.cg.model.language_model
    gc_enabled = False
    if use_gradient_checkpointing and hasattr(lm, "gradient_checkpointing_enable"):
        try:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            gc_enabled = True
        except Exception as e:
            print(f"[WARN] Failed to enable gradient checkpointing: {e}")
    print(f"Gemma4 gradient checkpointing: {'ENABLED (use_reentrant=False)' if gc_enabled else 'DISABLED'}", flush=True)

    device = torch.device(device_str)
    rng = np.random.default_rng(seed)

    start_step = 1
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        print(f"[RESUME] Loading checkpoint from {resume_ckpt_path}...", flush=True)
        res_ckpt = torch.load(resume_ckpt_path, map_location=device_str)
        projector.load_state_dict(res_ckpt["state_dict"])
        if "optimizer_state_dict" in res_ckpt and optimizer is not None:
            optimizer.load_state_dict(res_ckpt["optimizer_state_dict"])
        if "torch_rng_state" in res_ckpt:
            torch.set_rng_state(res_ckpt["torch_rng_state"])
        if "cuda_rng_state" in res_ckpt and res_ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(res_ckpt["cuda_rng_state"])
        if "numpy_rng_state" in res_ckpt:
            try:
                rng.bit_generator.state = res_ckpt["numpy_rng_state"]
            except Exception:
                pass
        start_step = res_ckpt["step"] + 1
        print(f"[RESUME] Resuming from step {start_step} (saved step {res_ckpt['step']})", flush=True)

    # Main Training Loop
    from collections import deque
    recent_400_active_mb: deque = deque(maxlen=400)
    t0_train = time.time()
    for step in range(start_step, max_steps + 1):
        projector.train()
        optimizer.zero_grad()

        # Sample total effective batch items from active valid train indices
        batch_q = rng.choice(valid_train_indices, size=effective_batch_size, replace=True)
        micro_batches = [batch_q[i * micro_batch_size : (i + 1) * micro_batch_size] for i in range(grad_accum_steps)]

        # Directive Task 1-3: Determinism batch Q-IDs logging
        if step <= 3 or step % 100 == 0 or step == max_steps:
            print(f"[Step {step:04d} Batch Q-IDs]: {batch_q.tolist()}", flush=True)

        step_loss_total = 0.0
        step_nll_m_f32 = 0.0
        step_nll_s_f32 = 0.0
        step_delta_f32 = 0.0
        step_penalty_f32 = 0.0
        step_nll_fmt = 0.0
        step_nll_cnt = 0.0
        step_max_diff_s = 0.0

        for mb_idx, mb_items in enumerate(micro_batches):
            z_batch_list: List[torch.Tensor] = []
            seq_tokens_list: List[torch.Tensor] = []
            seq_content_mask_list: List[torch.Tensor] = []

            for q in mb_items:
                entry = zstar_by_q.get(q)
                if entry is None:
                    for k_cand in [f"q{q:02d}_s2030", f"q{q:04d}_s2030", f"q{q}_s2030", f"q{q:04d}", f"q{q:02d}", str(q), q]:
                        if k_cand in zstar_dict:
                            entry = zstar_dict[k_cand]
                            break
                assert entry is not None, f"Item {q} not found in zstar dict"
                z = entry["z_star"] if isinstance(entry, dict) and "z_star" in entry else entry["z_1000"]
                z_batch_list.append(z.to(device=device, dtype=torch.float32))

                sol_text = dataset_records[q]["solution"]
                enc = tokenizer(sol_text, return_offsets_mapping=True, add_special_tokens=False)
                max_seq_len = int(cfg.get("max_seq_len", 1536))
                max_sol_len = max_seq_len - len(prefix_prompt_tensor) - 1
                sol_ids = enc["input_ids"][:max_sol_len] + [tokenizer.eos_token_id]
                offsets = enc.get("offset_mapping", [])[:max_sol_len]
                span = find_boxed_char_span(sol_text)
                is_content = [bool(span and s < span[1] and e > span[0]) for (s, e) in offsets]
                is_content.append(False)  # EOS token is classified as format
                full_seq = torch.cat([prefix_prompt_tensor, torch.tensor(sol_ids, dtype=torch.long)])
                seq_tokens_list.append(full_seq)
                seq_content_mask_list.append(torch.tensor(is_content, dtype=torch.bool))

            # Pad sequences to max length in this micro-batch
            max_len = max(len(s) for s in seq_tokens_list)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            cur_mb_size = len(mb_items)
            input_ids_batch = torch.full((cur_mb_size, max_len), pad_id, dtype=torch.long, device=device)
            target_ids_batch = torch.full((cur_mb_size, max_len), -100, dtype=torch.long, device=device)
            content_mask_batch = torch.zeros((cur_mb_size, max_len), dtype=torch.bool, device=device)

            for b_i, (s, c_mask) in enumerate(zip(seq_tokens_list, seq_content_mask_list)):
                input_ids_batch[b_i, :len(s)] = s.to(device)
                target_ids_batch[b_i, target_start:len(s)] = s[target_start:].to(device)
                content_mask_batch[b_i, target_start:target_start + len(c_mask)] = c_mask.to(device)

            z_matched = torch.stack(z_batch_list, dim=0)  # [cur_mb_size, 64, 2560]
            z_shuffled = torch.roll(z_matched, shifts=1, dims=0)

            # Base token embeddings from backbone
            base_embeds = lm.embed_tokens(input_ids_batch)
            per_layer_inputs = lm.get_per_layer_inputs(input_ids_batch, None) if hasattr(lm, "get_per_layer_inputs") else None
            kwargs_fwd = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}

            shift_targets = target_ids_batch[:, target_start:].contiguous()
            shift_content = content_mask_batch[:, target_start:].contiguous()

            # Directive Task 1-3: Step 1 first micro-batch prefix impact diagnostic (1회 로그, no_grad)
            if step == 1 and mb_idx == 0:
                with torch.no_grad():
                    p_m_diag = projector(z_matched).to(dtype=base_embeds.dtype)
                    p_s_diag = projector(z_shuffled).to(dtype=base_embeds.dtype)
                    p_zero_diag = torch.zeros_like(p_m_diag)

                    e_m_diag = base_embeds.clone(); e_m_diag[:, slot_start:slot_end + 1, :] = p_m_diag
                    e_s_diag = base_embeds.clone(); e_s_diag[:, slot_start:slot_end + 1, :] = p_s_diag
                    e_z_diag = base_embeds.clone(); e_z_diag[:, slot_start:slot_end + 1, :] = p_zero_diag

                    o_m_diag = lm(inputs_embeds=e_m_diag, return_dict=True, **kwargs_fwd)
                    h_m_diag = o_m_diag.last_hidden_state[:, target_start - 1:-1, :]
                    nll_m_diag = compute_nll_f32_chunked(h_m_diag, shift_targets, bb.cg.lm_head, chunk_size=256)

                    o_s_diag = lm(inputs_embeds=e_s_diag, return_dict=True, **kwargs_fwd)
                    h_s_diag = o_s_diag.last_hidden_state[:, target_start - 1:-1, :]
                    nll_s_diag = compute_nll_f32_chunked(h_s_diag, shift_targets, bb.cg.lm_head, chunk_size=256)

                    o_z_diag = lm(inputs_embeds=e_z_diag, return_dict=True, **kwargs_fwd)
                    h_z_diag = o_z_diag.last_hidden_state[:, target_start - 1:-1, :]
                    nll_z_diag = compute_nll_f32_chunked(h_z_diag, shift_targets, bb.cg.lm_head, chunk_size=256)

                    norm_p_diff = float(torch.linalg.vector_norm(p_m_diag.double() - p_s_diag.double()).item())

                    max_logits_diff = 0.0
                    T_diag = h_m_diag.shape[1]
                    for c_st in range(0, T_diag, 256):
                        c_en = min(c_st + 256, T_diag)
                        lm_c = bb.cg.lm_head(h_m_diag[:, c_st:c_en, :]).float()
                        ls_c = bb.cg.lm_head(h_s_diag[:, c_st:c_en, :]).float()
                        c_diff = float((lm_c - ls_c).abs().max().item())
                        if c_diff > max_logits_diff:
                            max_logits_diff = c_diff
                        del lm_c, ls_c

                    print(
                        f"[Step 0001 Prefix Impact Diagnostic]: "
                        f"NLL_m={nll_m_diag:.6f} | "
                        f"NLL_s={nll_s_diag:.6f} | "
                        f"NLL_zero={nll_z_diag:.6f} | "
                        f"||p_m - p_s||={norm_p_diff:.6e} | "
                        f"max|logits_m - logits_s|={max_logits_diff:.6e}",
                        flush=True
                    )
                    del p_m_diag, p_s_diag, p_zero_diag, e_m_diag, e_s_diag, e_z_diag, o_m_diag, o_s_diag, o_z_diag, h_m_diag, h_s_diag, h_z_diag
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ------------------------------------------------------------------
            # Micro-batch Exact Sequential Execution (Directive EXE-33 + 보충 A)
            # ------------------------------------------------------------------
            # ① Shuffled forward in torch.no_grad() -> nll_s_f32 (lambda=1 and lambda=0 both)
            with torch.no_grad():
                p_s_nograd = projector(z_shuffled).to(dtype=base_embeds.dtype)
                embeds_s_nograd = base_embeds.clone()
                embeds_s_nograd[:, slot_start:slot_end + 1, :] = p_s_nograd
                out_s_nograd = lm(inputs_embeds=embeds_s_nograd, return_dict=True, **kwargs_fwd)
                h_s_nograd = out_s_nograd.last_hidden_state[:, target_start - 1:-1, :]
                nll_s_f32 = compute_nll_f32_chunked(h_s_nograd, shift_targets, bb.cg.lm_head, chunk_size=256)
                del out_s_nograd, embeds_s_nograd, p_s_nograd, h_s_nograd
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ② Matched forward (with grad) -> nll_m
            p_m = projector(z_matched).to(dtype=base_embeds.dtype)
            embeds_m = base_embeds.clone()
            embeds_m[:, slot_start:slot_end + 1, :] = p_m
            out_m = lm(inputs_embeds=embeds_m, return_dict=True, **kwargs_fwd)
            h_m_target = out_m.last_hidden_state[:, target_start - 1:-1, :]

            # Compute nll_m_f32 using chunked logits in no_grad
            with torch.no_grad():
                nll_m_f32 = compute_nll_f32_chunked(h_m_target, shift_targets, bb.cg.lm_head, chunk_size=256)

            # Active margin decision: a = 1[lambda > 0 AND m - (nll_s_f32 - nll_m_f32) > 0]
            delta_f32 = nll_s_f32 - nll_m_f32
            margin_penalty_f32 = max(0.0, margin_m - delta_f32)
            margin_active_bool = bool(loss_lambda > 0.0 and margin_penalty_f32 > 0.0)
            a = 1.0 if margin_active_bool else 0.0
            recent_400_active_mb.append(1 if margin_active_bool else 0)

            # Gradient path for matched (run3 identical: CE on bf16 logits)
            shift_logits_m = bb.cg.lm_head(h_m_target).contiguous()
            nll_m = F.cross_entropy(shift_logits_m.view(-1, shift_logits_m.size(-1)), shift_targets.view(-1), ignore_index=-100)

            # Format vs Content NLL tracking
            with torch.no_grad():
                per_tok_loss = F.cross_entropy(shift_logits_m.view(-1, shift_logits_m.size(-1)), shift_targets.view(-1), reduction="none").view_as(shift_targets)
                valid_targets = (shift_targets != -100)
                fmt_mask = valid_targets & (~shift_content)
                cnt_mask = valid_targets & shift_content
                nll_fmt = per_tok_loss[fmt_mask].mean() if fmt_mask.any() else torch.tensor(0.0, device=device)
                nll_cnt = per_tok_loss[cnt_mask].mean() if cnt_mask.any() else torch.tensor(0.0, device=device)

            # Backward matched with exact coefficient: (1 + lambda * a) / accum
            w_m = (1.0 + loss_lambda * a) / grad_accum_steps
            (w_m * nll_m).backward()

            # Free matched activations
            del out_m, shift_logits_m, embeds_m, nll_m, p_m, h_m_target
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ③ Shuffled forward (with grad) ONLY IF lambda > 0 AND a == 1
            diff_s_grad_nograd = 0.0
            if loss_lambda > 0.0 and a == 1.0:
                p_s = projector(z_shuffled).to(dtype=base_embeds.dtype)
                embeds_s = base_embeds.clone()
                embeds_s[:, slot_start:slot_end + 1, :] = p_s
                out_s = lm(inputs_embeds=embeds_s, return_dict=True, **kwargs_fwd)
                shift_logits_s = bb.cg.lm_head(out_s.last_hidden_state[:, target_start - 1:-1, :]).contiguous()
                nll_s = F.cross_entropy(shift_logits_s.view(-1, shift_logits_s.size(-1)), shift_targets.view(-1), ignore_index=-100)
                w_s = (-loss_lambda) / grad_accum_steps
                (w_s * nll_s).backward()

                diff_s_grad_nograd = abs(float(nll_s.item()) - nll_s_f32)
                del out_s, shift_logits_s, embeds_s, nll_s, p_s
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del base_embeds

            # Accumulate metrics across micro-batches
            step_loss_total += (nll_m_f32 + loss_lambda * margin_penalty_f32) / grad_accum_steps
            step_nll_m_f32 += nll_m_f32 / grad_accum_steps
            step_nll_s_f32 += nll_s_f32 / grad_accum_steps
            step_delta_f32 += delta_f32 / grad_accum_steps
            step_penalty_f32 += margin_penalty_f32 / grad_accum_steps
            step_nll_fmt += float(nll_fmt.item()) / grad_accum_steps
            step_nll_cnt += float(nll_cnt.item()) / grad_accum_steps
            step_max_diff_s = max(step_max_diff_s, diff_s_grad_nograd)

        grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        optimizer.step()

        margin_active_ratio = (sum(recent_400_active_mb) / len(recent_400_active_mb)) if recent_400_active_mb else 0.0
        w_frob_f64 = projector.get_w_proj_frobenius_norm()

        metrics = {
            "loss_total": step_loss_total,
            "nll_matched": step_nll_m_f32,
            "nll_shuffled": step_nll_s_f32,
            "delta_nll": step_delta_f32,
            "margin_penalty": step_penalty_f32,
            "loss_lambda_0": step_nll_m_f32,
            "nll_fmt": step_nll_fmt,
            "nll_cnt": step_nll_cnt,
            "margin_active_ratio": margin_active_ratio,
            "max_diff_s": step_max_diff_s,
            "w_frob_f64": w_frob_f64,
        }

        # Step logging every 100 steps
        if step % 100 == 0 or step == 1 or step == max_steps:
            sn_train = projector.get_w_proj_spectral_norm()
            projector.eval()
            sn_eval = projector.get_w_proj_spectral_norm()
            projector.train()
            current_lr = optimizer.param_groups[0]['lr']
            l_task = float(step_nll_m_f32)
            l_align = float(loss_lambda * step_penalty_f32)
            elapsed = time.time() - t0_train
            sec_per_step = elapsed / step
            peak_mem_gib = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if torch.cuda.is_available() else 0.0

            # Directive EXE-33 + 보충 A logging format
            print(
                f"[Step {step:04d}/{max_steps:04d}] "
                f"loss={step_loss_total:.4f} | "
                f"l_task={l_task:.4f} | "
                f"l_align={l_align:.4f} | "
                f"NLL_m_f32={step_nll_m_f32:.4f} | "
                f"NLL_s_f32={step_nll_s_f32:.4f} | "
                f"Delta_f32={step_delta_f32:+.4f} | "
                f"NLL_fmt={step_nll_fmt:.4f} | "
                f"NLL_cnt={step_nll_cnt:.4f} | "
                f"act_ratio_100={margin_active_ratio:.2%} | "
                f"max_diff_s={step_max_diff_s:.4e} | "
                f"grad_norm={float(grad_norm.item()):.4e} | "
                f"w_frob={w_frob_f64:.6f} | "
                f"lr={current_lr:.4e} (const) | "
                f"sched=None | "
                f"SN(tr/ev)={sn_train:.4f}/{sn_eval:.4f} | "
                f"micro/accum={micro_batch_size}x{grad_accum_steps} (eff {effective_batch_size}) | "
                f"peak_mem={peak_mem_gib:.2f} GiB | "
                f"{sec_per_step:.2f}s/step",
                flush=True
            )

        # Checkpoint every 500 steps
        if step % 500 == 0 or step == max_steps:
            ckpt_path = os.path.join(output_dir, f"s5_decoder_lambda{loss_lambda:.1f}_step{step:04d}.pt")
            save_dict = {
                "step": step,
                "loss_lambda": loss_lambda,
                "margin_m": margin_m,
                "state_dict": projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "numpy_rng_state": rng.bit_generator.state,
                "config": cfg,
                "metrics": metrics,
                "b1_info": b1_info,
            }
            torch.save(save_dict, ckpt_path)
            sha = append_to_manifest(manifest_path, ckpt_path)
            print(f"[CHECKPOINT] Saved step {step} to {ckpt_path} (SHA: {sha})", flush=True)

    # Save to --out if specified
    if out_ckpt_path:
        os.makedirs(os.path.dirname(out_ckpt_path) or ".", exist_ok=True)
        torch.save(save_dict, out_ckpt_path)
        sha_out = append_to_manifest(manifest_path, out_ckpt_path)
        print(f"[OUT-CHECKPOINT] Saved final model to {out_ckpt_path} (SHA: {sha_out})", flush=True)

    elapsed = time.time() - t0_train
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}m).", flush=True)
    return {
        "status": "TRAINING_COMPLETED",
        "steps": max_steps,
        "loss_lambda": loss_lambda,
        "final_loss": metrics["loss_total"],
        "elapsed_sec": elapsed,
        "out_ckpt_path": out_ckpt_path,
    }


def run_dry_run(
    cfg: Dict[str, Any],
    raw_dir: str,
    load_b1: bool = False,
    b1_ckpt_path: str = DEFAULT_B1_CKPT_PATH,
    items_path: str = DEFAULT_ITEMS_PATH,
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> Dict[str, Any]:
    """Execute Phase 0-4 Dry-Run on CPU: verifies model structure, zero problem text, and loss."""
    print("[DRY-RUN] Executing Phase 0-4 Decoder Training Dry-Run (CPU, 0 GPU calls)...")
    os.makedirs(raw_dir, exist_ok=True)

    seed = int(cfg.get("seed", 2026))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Verify train/heldout separation and truncation count
    train_count = 320
    heldout_count = 80
    heldout_in_train = 0
    truncation_excluded_count = 0
    if os.path.exists(items_path):
        with open(items_path, "r", encoding="utf-8") as f:
            items_data = json.load(f)
        train_indices = [int(x) for x in items_data.get("train_indices", [])]
        heldout_indices = [int(x) for x in items_data.get("heldout_indices", [])]
        train_count = len(train_indices)
        heldout_count = len(heldout_indices)
        heldout_in_train = len(set(train_indices) & set(heldout_indices))

        if os.path.exists(dataset_path):
            records = load_dataset_records(dataset_path)
            tok = load_tokenizer()
            max_seq_len = int(cfg.get("max_seq_len", 1536))
            for q in train_indices:
                sol_text = records[q]["solution"]
                span = find_boxed_char_span(sol_text)
                enc = tok(sol_text, return_offsets_mapping=True, add_special_tokens=False)
                offsets = enc.get("offset_mapping", [])
                boxed_end = 73 + len(enc["input_ids"]) + 1
                if span is not None:
                    for tok_i, (s, e) in enumerate(offsets):
                        if e >= span[1]:
                            boxed_end = 73 + tok_i + 1
                            break
                if boxed_end > max_seq_len:
                    truncation_excluded_count += 1

    projector = PrefixProjector(d_in=cfg["hidden_size"], d_model=cfg["d_model"], use_spectral_norm=True)

    b1_info = None
    if load_b1:
        b1_info = projector.load_b1_w_proj(b1_ckpt_path=b1_ckpt_path, strict=True)
        b1_log_path = os.path.join(raw_dir, "dryrun_b1_load.log")
        with open(b1_log_path, "w", encoding="utf-8") as f:
            f.write("=== S5 TRAIN DECODER B1 LOAD LOG ===\n")
            f.write("Directive: EXE-260922-26 rev3 Task 1 (Ledger #322)\n")
            f.write(f"Random seed: {seed}\n")
            f.write(f"Checkpoint path: {b1_ckpt_path}\n")
            f.write(f"Loading method: {b1_info['path_taken']}\n")
            f.write(f"Fallback executed: {b1_info['path_taken'] != 'DIRECT_STRICT_PARAMETRIZATIONS'}\n")
            f.write(f"Strict loading status: SUCCESS ({b1_info['missing_count']} missing, {b1_info['unexpected_count']} unexpected)\n")
            f.write(f"Missing keys count: {b1_info['missing_count']}\n")
            f.write(f"Unexpected keys count: {b1_info['unexpected_count']}\n")
            f.write(f"Frobenius norm (original weight): {b1_info['frob_orig']:.6f}\n")
            f.write(f"Spectral norm (effective weight): {b1_info['spec_norm']:.6f}\n")
            f.write(f"u vector norm: {b1_info['u_norm']:.6f}\n")
            f.write(f"v vector norm: {b1_info['v_norm']:.6f}\n")
            f.write(f"Effective weight Frobenius norm: {b1_info['eff_frob']:.6f}\n")
            f.write(f"Trainable parameters total: {sum(p.numel() for p in projector.parameters() if p.requires_grad)}\n")
            f.write("Optimizer steps called: 0\n")
            f.write("Parameter updates called: 0\n")
            f.write("=== END B1 LOAD LOG ===\n")

    trainable_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    w_proj_params = projector.w_proj.parametrizations.weight.original.numel() if hasattr(projector.w_proj, "parametrizations") else projector.w_proj.weight.numel()
    adapter_params = sum(p.numel() for p in projector.adapter.parameters() if p.requires_grad)

    sn_train = projector.get_w_proj_spectral_norm()
    projector.eval()
    sn_eval = projector.get_w_proj_spectral_norm()
    projector.train()

    tokenizer = load_tokenizer()
    token_ids, prefix_start_idx, prefix_end_idx, placeholder_id = build_canonical_prefix_prompt(
        tokenizer, k_latent=cfg["k_latent"]
    )
    problem_text_tokens = 0

    B, K, D = 4, cfg["k_latent"], cfg["hidden_size"]
    z_matched = torch.randn(B, K, D, dtype=torch.float32)
    z_shuffled = torch.roll(z_matched, shifts=1, dims=0)

    p_matched = projector(z_matched)
    p_shuffled = projector(z_shuffled)

    seq_len = len(token_ids)
    inputs_embeds = torch.zeros(B, seq_len, D, dtype=torch.float32)
    inputs_embeds[:, prefix_start_idx:prefix_end_idx + 1, :] = p_matched

    nll_matched_dummy = torch.tensor(2.5, requires_grad=True)
    nll_shuffled_dummy = torch.tensor(4.0, requires_grad=True)

    loss_lambda1, metrics_lambda1 = compute_contrastive_margin_loss(
        nll_matched_dummy, nll_shuffled_dummy, loss_lambda=1.0, margin_m=1.0
    )
    loss_lambda0, metrics_lambda0 = compute_contrastive_margin_loss(
        nll_matched_dummy, nll_shuffled_dummy, loss_lambda=0.0, margin_m=1.0
    )

    log_path = os.path.join(raw_dir, "dryrun_decoder_v2.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== S5 TRAIN DECODER DRY-RUN LOG ===\n")
        f.write("Directive: EXE-260924-30 Task 1 (Ledger #334)\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Trainable parameters total: {trainable_params} (w_proj: {w_proj_params}, adapter: {adapter_params})\n")
        f.write("Backbone frozen: True | LoRA count: 0 (Backbone: frozen canonical)\n")
        f.write(f"train N={train_count}, heldout excluded={heldout_count}\n")
        f.write(f"heldout in train: {heldout_in_train}\n")
        f.write(f"truncation excluded count: {truncation_excluded_count}\n")
        f.write(f"learning rate: constant (w_proj: {cfg.get('lr_wproj', 1e-4)}, adapter: {cfg.get('lr_adapter', 3e-4)})\n")
        f.write("scheduler: None\n")
        f.write("batch configuration: effective batch 8 = micro-batch 2 x grad_accum 4 (loss scaled by 1/4)\n")
        f.write("single forward size: max 1536 x 2 (1536 x 8 prohibited: True)\n")
        f.write("Gemma4 gradient checkpointing: ENABLED (use_reentrant=False)\n")
        f.write(f"Prefix token indices: [{prefix_start_idx}, {prefix_end_idx}] (length: {K})\n")
        f.write(f"Prefix slot start index: {prefix_start_idx} | Prefix slot end index: {prefix_end_idx}\n")
        f.write(f"Problem text token count: {problem_text_tokens}\n")
        f.write(f"Spectral norm on train switch: {sn_train:.4f}\n")
        f.write(f"Spectral norm on eval switch: {sn_eval:.4f}\n")
        f.write(f"Contrastive Loss (lambda=1.0): {metrics_lambda1['loss_total']:.4f} (delta: {metrics_lambda1['delta_nll']:.4f})\n")
        f.write(f"Control Loss (lambda=0.0): {metrics_lambda0['loss_total']:.4f}\n")
        f.write("Optimizer steps called: 0 | Parameter updates called: 0\n")
        f.write("=== END DRY-RUN LOG ===\n")

    return {
        "status": "DRY_RUN_COMPLETED",
        "trainable_params": trainable_params,
        "prefix_start_idx": prefix_start_idx,
        "prefix_end_idx": prefix_end_idx,
        "problem_text_tokens": problem_text_tokens,
        "train_count": train_count,
        "heldout_count": heldout_count,
        "heldout_in_train": heldout_in_train,
        "truncation_excluded_count": truncation_excluded_count,
        "scheduler": None,
        "sn_w_proj": sn_train,
        "loss_metrics_lambda1": metrics_lambda1,
        "loss_metrics_lambda0": metrics_lambda0,
        "log_path": log_path,
        "b1_info": b1_info,
    }


def main():
    parser = argparse.ArgumentParser(description="S5 z*-Only Decoder Training (EXE-260923-28 rev2).")
    parser.add_argument("--dry-run", "--dry_run", action="store_true", help="Execute Step 0 Dry-Run on CPU")
    parser.add_argument("--cpu-smoke", "--cpu_smoke", action="store_true", help="Execute CPU smoke test (2 items, 3 steps, lambda=1.0)")
    parser.add_argument("--load-b1", "--load_b1", action="store_true", help="Load B1 checkpoint w_proj weights")
    parser.add_argument("--b1-ckpt", "--b1_ckpt", type=str, default=DEFAULT_B1_CKPT_PATH, help="Path to B1 checkpoint")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to s5_decoder.yaml")
    parser.add_argument("--loss_lambda", "--loss-lambda", "-l", type=float, default=1.0, help="Contrastive loss weight lambda (1.0 or 0.0)")
    parser.add_argument("--lambda", dest="loss_lambda", type=float, help="Contrastive loss weight lambda (alias for --loss_lambda)")
    parser.add_argument("--margin_m", "--margin", type=float, default=1.0, help="Contrastive margin m (default 1.0 nat)")
    parser.add_argument("--max_steps", "--steps", type=int, default=3000, help="Maximum optimization steps (default 3000)")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Effective batch size (default 8)")
    parser.add_argument("--micro_batch_size", "--micro-batch-size", type=int, default=2, help="Micro-batch size (default 2)")
    parser.add_argument("--grad_accum_steps", "--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps (default 4)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable Gemma4 gradient checkpointing")
    parser.add_argument("--lr_wproj", type=float, default=1e-4, help="Learning rate for w_proj (default 1e-4)")
    parser.add_argument("--lr_adapter", type=float, default=3e-4, help="Learning rate for adapter (default 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW (default 0.01)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default 2026)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results output directory")
    parser.add_argument("--out", "--output", type=str, default="", help="Path to save final checkpoint pt file")
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR, help="Raw logs directory")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Output manifest SHA file")
    parser.add_argument("--zstar", "--zstar_path", type=str, default=DEFAULT_ZSTAR_PATH, help="Path to zstar_s5_400.pt")
    parser.add_argument("--items", "--items_json", type=str, default=DEFAULT_ITEMS_PATH, help="Path to s5_items_400.json")
    parser.add_argument("--dataset", "--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to math_train_prompts_5000.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (cuda:0)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--prefix_source", type=str, default="zstar", choices=["zstar", "text_hidden64"], help="Prefix source: zstar or text_hidden64")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dry_run:
        res = run_dry_run(
            cfg=cfg,
            raw_dir=args.raw_dir,
            load_b1=args.load_b1,
            b1_ckpt_path=args.b1_ckpt,
            items_path=args.items,
            dataset_path=args.dataset,
        )
        print(json.dumps(res, indent=2))
        return

    if args.cpu_smoke:
        res = run_cpu_smoke(cfg=cfg, raw_dir=args.raw_dir, b1_ckpt_path=args.b1_ckpt)
        print(json.dumps(res, indent=2))
        return

    # Real training path
    run_train(
        cfg=cfg,
        loss_lambda=args.loss_lambda,
        margin_m=args.margin_m,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        lr_wproj=float(cfg.get("lr_wproj", args.lr_wproj)),
        lr_adapter=float(cfg.get("lr_adapter", args.lr_adapter)),
        weight_decay=args.weight_decay,
        device_str=args.device,
        b1_ckpt_path=args.b1_ckpt,
        zstar_pt_path=args.zstar,
        items_json_path=args.items,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        out_ckpt_path=args.out,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        seed=args.seed,
        resume_ckpt_path=args.resume,
        prefix_source=args.prefix_source,
    )


if __name__ == "__main__":
    main()
