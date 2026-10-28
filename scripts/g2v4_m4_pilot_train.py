#!/usr/bin/env python3
"""
scripts/g2v4_m4_pilot_train.py - copy of g2v3_m3_full_train.py (36e0f18a) + M4 pilot options (CC, 2026-10-11):
  --text_drop p : per item, with prob p the problem text is removed from the prompt (placeholders only), so the decoder must read the latent.
  --init_decoder / --init_deq : warm start from M3 checkpoints. --seed : torch + drop RNG seed.
  --proj_v2 : PrefixProjectorV2 (scripts/g2v4_proj_v2.py, centred + RMS-normalised in/out, v3, CC 11:4x).
  --lm_lora_rank r (>0) : decoder-side LoRA on the (otherwise frozen) LM q/v/o, fp32, saved in decoder ckpt as lm_lora_state_dict (v2, CC 10:2x).
Original header follows.

scripts/g2v3_m3_pilot_train.py - G2 3rd Attempt (G2v3 / M3 / N8) Joint Pilot Training Script
Directive: TASK-0071 (GRAPH Node N8 / proposals/G2v3_PLAN_20261009.md / Principal Autonomous Execution)

Specifications:
- Objective: Joint training of DEQ transition map Phi_theta (LoRA, routing_proj, gamma) and
  Decoder PrefixProjector using math solution NLL (lambda=0) via phantom gradient.
- DEQ Fixed-Point Phantom Gradient:
    z* (detached) -> 1-step differentiable transition Phi_theta(z*) -> PrefixProjector -> LM CE loss
    Auxiliary IFT residual loss: lambda_ift * ||Phi_theta(z*) - z*||^2 / ||z*||^2
- Contrast Condition E''' Baseline:
    z_src (detached) -> 1-step differentiable transition Phi_theta(z_src) -> PrefixProjector -> LM CE loss
- Pure Gemma Language Model: Pure Gemma-4-e4b Base (completely frozen)
- Precision: bfloat16 / float32 for DEQ transitions, float32 accumulation.
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Default snapshots & paths
DEFAULT_BASE_REV = "58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
DEFAULT_PURE_GEMMA_SNAPSHOT = f"/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/{DEFAULT_BASE_REV}"
DEFAULT_FROZEN_BACKBONE_CKPT = "/home/shoon/checkpoints/m1_final_frozen.pt"
DEFAULT_DATASET_PATH = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"


# =========================================================================
# PrefixProjector & MLPAdapter (Canonical Definitions)
# =========================================================================

class MLPAdapter(nn.Module):
    """2-layer MLP Adapter (d_model -> d_model) with zero-initialized final layer."""

    def __init__(self, d_model: int = 2560, hidden_dim: int = 2560) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class PrefixProjector(nn.Module):
    """Combined projection P(z) = Adapter(w_proj(z))."""

    def __init__(self, d_in: int = 2560, d_model: int = 2560, use_spectral_norm: bool = True) -> None:
        super().__init__()
        linear = nn.Linear(d_in, d_model, bias=False)
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            self.w_proj = spectral_norm(linear)
        else:
            self.w_proj = linear
        self.adapter = MLPAdapter(d_model=d_model, hidden_dim=d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = self.w_proj(z)
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


# =========================================================================
# Dataset and Prompt Preparation Helpers
# =========================================================================

def load_dataset_records(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    candidates = [
        dataset_path,
        os.path.join(SYS_DIR, "data", "stage2", "math_train_prompts_5000.jsonl"),
        "data/stage2/math_train_prompts_5000.jsonl",
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"Dataset not found in candidates: {candidates}")


def load_tokenizer() -> Any:
    """Load Gemma tokenizer from standard system paths with chat_template fallback."""
    tok_paths = [
        "/home/shoon/models/gemma4-e4b-it",
        "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377",
    ]
    for tp in tok_paths:
        if os.path.exists(tp):
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
                if getattr(tok, "chat_template", None) is not None:
                    return tok
            except Exception:
                pass

    for tp in tok_paths:
        if os.path.exists(tp):
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
            if getattr(tok, "chat_template", None) is None:
                tok.chat_template = (
                    "{{ bos_token }}{% for message in messages %}"
                    "{% if (message['role'] == 'user') %}{{'<|turn>user\n' + message['content'] + '<turn|>\n'}}"
                    "{% elif (message['role'] == 'model') %}{{'<|turn>model\n' + message['content'] + '<turn|>\n'}}"
                    "{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<|turn>model\n'}}{% endif %}"
                )
            return tok
    raise RuntimeError(f"Tokenizer not found in candidates: {tok_paths}")


def build_canonical_prefix_prompt(
    tokenizer: Any,
    prob_text: str = "",
    k_latent: int = 64,
) -> Tuple[List[int], int, int, int]:
    """Build canonical prefix prompt with problem text using tokenizer (v3.5/v3.6 §3 format)."""
    single_tok = tokenizer.encode("1", add_special_tokens=False)
    assert len(single_tok) == 1, "Placeholder must be a single token"
    placeholder_id = single_tok[0]

    placeholder_str = "1" * k_latent
    user_content = placeholder_str + "\n\n" + prob_text if prob_text else placeholder_str
    messages = [{"role": "user", "content": user_content}]
    templated_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    token_ids = tokenizer.encode(templated_text, add_special_tokens=False)

    slot_start = 4
    slot_end = slot_start + k_latent - 1
    assert token_ids[:4] == [2, 105, 2364, 107], f"Expected prefix [2, 105, 2364, 107], got {token_ids[:4]}"
    assert all(t == placeholder_id for t in token_ids[slot_start:slot_end + 1]), f"Expected {k_latent} placeholders in [{slot_start}..{slot_end}]"

    user_tok_str = tokenizer.decode([token_ids[slot_start - 2]]).strip()
    nl_tok_str = tokenizer.decode([token_ids[slot_start - 1]])
    assert user_tok_str == "user", f"Token before slot must be 'user', got {user_tok_str}"
    assert nl_tok_str == "\n", f"Token immediately before slot must be newline, got {repr(nl_tok_str)}"

    if prob_text:
        assert token_ids[68] == 108, f"Token immediately after slots must be newline (108), got {token_ids[68]}"

    return token_ids, slot_start, slot_end, placeholder_id


def prepare_item_batch(
    tokenizer: Any,
    prob_text: str,
    sol_text: str,
    max_seq_len: int = 1536,
    k_latent: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build input_ids and target labels with -100 masking on user prompt tokens."""
    prompt_tokens, s_st, s_en, _ = build_canonical_prefix_prompt(tokenizer, prob_text, k_latent=k_latent)
    prompt_len = len(prompt_tokens)

    sol_tokens = tokenizer.encode(sol_text, add_special_tokens=False)
    if not sol_tokens or sol_tokens[-1] != tokenizer.eos_token_id:
        sol_tokens.append(tokenizer.eos_token_id)

    full_tokens = prompt_tokens + sol_tokens
    if len(full_tokens) > max_seq_len:
        full_tokens = full_tokens[:max_seq_len]

    input_ids = torch.tensor(full_tokens, dtype=torch.long)
    targets = input_ids.clone()
    # Mask out prompt tokens so loss is only calculated on solution tokens
    targets[:prompt_len] = -100

    return input_ids, targets, s_st, s_en


# =========================================================================
# Synthetic Smoke Modules for Fast Local / CPU Testing
# =========================================================================

class SmokeBackbone(nn.Module):
    """Minimal lightweight mock of CTS backbone for dry-run testing."""
    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.hidden_size = hidden_size
        self.deq_map_mode = "paper"
        self.gamma_param = nn.Parameter(torch.tensor(0.5))
        self.lora_A = nn.Linear(hidden_size, 8, bias=False)
        self.lora_B = nn.Linear(8, hidden_size, bias=False)
        self.routing_proj = nn.Linear(hidden_size, 4, bias=False)

    def deq_step(self, z, context, module_weights, extra):
        delta = self.lora_B(self.lora_A(z.float()))
        gamma = torch.sigmoid(self.gamma_param)
        return gamma * (z.float() + 0.01 * delta) + (1.0 - gamma) * z.float()

    def encode_context(self, text):
        return torch.randn(1, 10, self.hidden_size)

    def routing_matrix(self):
        return self.routing_proj.weight.T


class SmokeLM(nn.Module):
    """Minimal lightweight mock of Gemma LM."""
    def __init__(self, hidden_size: int = 2560, vocab_size: int = 1000):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, inputs_embeds=None, return_dict=True, **kwargs):
        logits = self.lm_head(inputs_embeds)
        class MockOut:
            pass
        out = MockOut()
        out.logits = logits
        out.last_hidden_state = inputs_embeds
        return out


# =========================================================================
# Main Pilot Training Pipeline
# =========================================================================

def run_pilot(args: argparse.Namespace) -> Dict[str, Any]:
    print("=" * 80)
    print(f"CTS G2v3 M3 Pilot Training - Mode [{args.mode}]")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device} | Max Steps: {args.max_steps} | Batch Size: {args.batch_size}")
    print("=" * 80, flush=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    drop_rng = np.random.default_rng(args.seed)
    n_drop = 0

    # 1. Smoke mode branch
    if args.smoke:
        print("[SMOKE] Running synthetic in-memory smoke verification...", flush=True)
        backbone = SmokeBackbone(hidden_size=args.hidden_size).to(device)
        lm_model = SmokeLM(hidden_size=args.hidden_size, vocab_size=args.vocab_size).to(device)
        projector = PrefixProjector(d_in=args.hidden_size, d_model=args.hidden_size, use_spectral_norm=False).to(device)

        # Freeze base LM
        for p in lm_model.parameters():
            p.requires_grad = False

        opt_params = [
            {"params": projector.parameters(), "lr": args.lr_projector},
            {"params": [backbone.lora_A.weight, backbone.lora_B.weight, backbone.gamma_param], "lr": args.lr_deq},
        ]
        optimizer = torch.optim.AdamW(opt_params)

        telemetry = []
        for step in range(args.max_steps):
            t0 = time.time()
            z_star = torch.randn(args.batch_size, 64, args.hidden_size, device=device)
            z_src = torch.randn(args.batch_size, 64, args.hidden_size, device=device)

            optimizer.zero_grad()
            # 1-step transition
            z_cur = z_star if args.mode == "A" else z_src
            phi_z = backbone.deq_step(z_cur, None, None, {})
            alpha = 0.25
            g0 = 0.25
            z_next = (1.0 - alpha) * z_cur + alpha * (phi_z + g0 * z_src)

            # IFT loss
            l_ift = ((z_next - z_cur) ** 2).mean() if args.mode == "A" else torch.tensor(0.0, device=device)

            # Prefix & LM loss
            prefix = projector(z_next)
            seq_len = 128
            fake_embeds = torch.randn(args.batch_size, seq_len, args.hidden_size, device=device)
            fake_embeds[:, 4:68, :] = prefix
            out = lm_model(inputs_embeds=fake_embeds)
            fake_targets = torch.randint(0, args.vocab_size, (args.batch_size, seq_len), device=device)
            fake_targets[:, :68] = -100
            l_task = F.cross_entropy(out.logits.view(-1, args.vocab_size), fake_targets.view(-1), ignore_index=-100)

            total_loss = l_task + (args.lambda_ift * l_ift if args.mode == "A" else 0.0)
            total_loss.backward()
            optimizer.step()

            step_time = time.time() - t0
            telemetry.append({
                "step": step,
                "loss": float(total_loss.item()),
                "l_task": float(l_task.item()),
                "l_ift": float(l_ift.item()),
                "step_time_s": step_time,
            })
            if step % 5 == 0 or step == args.max_steps - 1:
                print(f"[SMOKE Step {step:3d}] Loss: {total_loss.item():.4f} | Task: {l_task.item():.4f} | IFT: {l_ift.item():.6f} ({step_time*1000:.1f}ms)", flush=True)

        print("[SMOKE] Verification completed successfully.", flush=True)
        return {"status": "SUCCESS", "telemetry": telemetry}

    # 2. Real Model Loading
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cts.backbone.gemma_adapter import GemmaCTSBackbone
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.train.lora_compat import apply_paper_lora
    from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

    print(f"[LOAD] Loading Pure Gemma-4-e4b Base from {args.base_model}...", flush=True)
    pure_gemma, _ = load_gemma4_e4b(
        model_id=args.base_model,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        offload_vision_audio=True,
    )
    tokenizer = load_tokenizer()
    for p in pure_gemma.parameters():
        p.requires_grad = False
    pure_gemma.eval()

    if hasattr(pure_gemma, "model") and hasattr(pure_gemma.model, "language_model"):
        lm_model = pure_gemma.model.language_model
        lm_head = pure_gemma.lm_head
        embed_tokens = pure_gemma.model.language_model.embed_tokens
    else:
        lm_model = pure_gemma
        lm_head = getattr(pure_gemma, "lm_head", None)
        embed_tokens = getattr(pure_gemma, "embed_tokens", None)

    lm_lora_params = []
    if args.lm_lora_rank > 0:
        from cts.train.lora_compat import replace_target_linears
        n_new, _ = replace_target_linears(lm_model, ("q_proj", "v_proj", "o_proj"), rank=args.lm_lora_rank, alpha=2 * args.lm_lora_rank, dropout=0.0)
        for n_, m_ in lm_model.named_modules():
            if n_.endswith("lora_A") or n_.endswith("lora_B"):
                m_.to(dtype=torch.float32)
        for n_, p_ in lm_model.named_parameters():
            if "lora_" in n_:
                p_.requires_grad = True
                lm_lora_params.append(p_)
        print(f"[LM-LORA] wrapped {n_new} linears, {sum(p.numel() for p in lm_lora_params):,} trainable weights", flush=True)

    print(f"[LOAD] Loading CTS DEQ Backbone with LoRA in bfloat16...", flush=True)
    m_base, tok_base = load_gemma4_e4b(
        model_id=args.base_model,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        offload_vision_audio=True,
    )
    bb = GemmaCTSBackbone(m_base, tok_base)
    apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0, require_match=False, verbose=False)

    ckpt = torch.load(args.backbone_ckpt, map_location="cpu")
    sd = ckpt.get("backbone_state_dict", ckpt)
    new_sd = {("routing_proj.parametrizations.weight.original" if k == "routing_proj" else k): v for k, v in sd.items()}
    bb.load_state_dict(new_sd, strict=True)
    if args.init_deq:
        _q = torch.load(args.init_deq, map_location="cpu")
        _r = bb.load_state_dict(_q.get("state_dict", _q), strict=False)
        if _r.unexpected_keys:
            raise RuntimeError(f"[FATAL] unexpected DEQ keys n={len(_r.unexpected_keys)}")
        print(f"[INIT] DEQ <- {args.init_deq} ({len(_q.get('state_dict', _q))} tensors, unexpected 0)", flush=True)
    bb.to(device=device, dtype=torch.bfloat16)

    # Enable gradients strictly on DEQ LoRA, routing_proj, and gamma
    trainable_deq_params = []
    for name, param in bb.named_parameters():
        if any(target in name for target in ["lora_A", "lora_B", "routing_proj", "gamma_param"]):
            param.requires_grad = True
            trainable_deq_params.append(param)
        else:
            param.requires_grad = False
    print(f"[PARAMS] Trainable DEQ parameters: {len(trainable_deq_params)} tensors ({sum(p.numel() for p in trainable_deq_params):,} weights)", flush=True)

    # 3. PrefixProjector Initialization
    if args.proj_v2:
        from g2v4_proj_v2 import PrefixProjectorV2
        with torch.no_grad():
            _e = embed_tokens(torch.arange(1000, 3000, device=device).unsqueeze(0)).float()
            _rms = float(_e.pow(2).mean(-1).sqrt().mean())
        projector = PrefixProjectorV2(d_model=args.hidden_size, out_rms=_rms).to(device=device, dtype=torch.float32)
        print(f"[PROJ-V2] out_rms init {_rms:.4f}", flush=True)
    else:
        projector = PrefixProjector(d_in=args.hidden_size, d_model=args.hidden_size, use_spectral_norm=True).to(device=device, dtype=torch.float32)
    if args.init_decoder:
        _d = torch.load(args.init_decoder, map_location="cpu")
        projector.load_state_dict(_d.get("state_dict", _d), strict=True)
        print(f"[INIT] decoder <- {args.init_decoder}", flush=True)
    trainable_proj_params = list(projector.parameters())
    print(f"[PARAMS] Trainable Projector parameters: {len(trainable_proj_params)} tensors ({sum(p.numel() for p in trainable_proj_params):,} weights)", flush=True)

    # 4. Optimizer Setup
    opt_groups = [
        {"params": [projector.w_proj.parametrizations.weight.original if hasattr(projector.w_proj, "parametrizations") else projector.w_proj.weight], "lr": args.lr_projector, "weight_decay": 0.01},
        {"params": list(projector.adapter.parameters()) + ([projector.out_rms] if args.proj_v2 else []), "lr": args.lr_adapter, "weight_decay": 0.01},
        {"params": trainable_deq_params, "lr": args.lr_deq, "weight_decay": 0.001},
    ]
    if lm_lora_params:
        opt_groups.append({"params": lm_lora_params, "lr": args.lr_lm_lora, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(opt_groups)

    # 5. Load Tensors & Items
    print(f"[DATA] Loading items and cached latent tensors...", flush=True)
    with open(args.items, "r", encoding="utf-8") as f:
        item_indices = json.load(f)
    if isinstance(item_indices, dict):
        item_indices = item_indices.get("indices", list(item_indices.keys()))
    item_indices = [int(x) for x in item_indices][:args.max_items]

    dataset_records = load_dataset_records(args.dataset)
    zstar_dict = torch.load(args.zstar, map_location="cpu") if args.zstar and os.path.exists(args.zstar) else {}
    zsrc_dict = torch.load(args.zsrc, map_location="cpu") if args.zsrc and os.path.exists(args.zsrc) else {}

    def extract_tensor(d: Dict[str, Any], q: int) -> Optional[torch.Tensor]:
        entry = None
        for k_cand in [
            q, str(q),
            f"q{q:04d}_a0.25", f"q{q:04d}", f"q{q:02d}",
            f"q{q:04d}_s2030", f"q{q}_s2030", f"q{q:02d}_s2030"
        ]:
            if k_cand in d:
                entry = d[k_cand]
                break
        if entry is None:
            for k, v in d.items():
                m = re.match(r"^q(\d+)", str(k))
                if m and int(m.group(1)) == q:
                    entry = v
                    break
        if entry is None:
            return None

        if isinstance(entry, torch.Tensor):
            return entry
        elif isinstance(entry, dict):
            for k in ["z_star", "z_src", "z_1000", "z_lowest"]:
                if k in entry and isinstance(entry[k], torch.Tensor):
                    return entry[k]
        return None

    # Filter available items
    valid_items = []
    for q in item_indices:
        zs = extract_tensor(zstar_dict, q)
        zr = extract_tensor(zsrc_dict, q)
        if (args.mode == "A" and zs is not None and zr is not None) or (args.mode == "E" and zr is not None):
            valid_items.append(q)
    print(f"[DATA] Filtered {len(valid_items)} / {len(item_indices)} valid items with required latent tensors.", flush=True)

    if not valid_items:
        # Print sample keys from dicts for debugging
        print(f"[DEBUG] zstar keys sample: {list(zstar_dict.keys())[:5] if zstar_dict else 'EMPTY'}")
        print(f"[DEBUG] zsrc keys sample: {list(zsrc_dict.keys())[:5] if zsrc_dict else 'EMPTY'}")
        raise RuntimeError("No valid items found with required latent tensors!")

    if args.proj_v2:
        with torch.no_grad():
            _src = zstar_dict if args.mode == "A" else zsrc_dict
            _c = torch.stack([extract_tensor(_src, q).float().reshape(64, -1) for q in valid_items]).mean(0)
            projector.center.copy_(_c.to(device))
        print(f"[PROJ-V2] center = mean of {len(valid_items)} training start latents ({'z*' if args.mode == 'A' else 'z_src'}), norm/slot {float(_c.norm(dim=-1).mean()):.1f}", flush=True)

    print(f"[CACHE] Pre-encoding prompt contexts for {len(valid_items)} items...", flush=True)
    ctx_by_q: Dict[int, Any] = {}
    with torch.no_grad():
        for q in valid_items:
            rec = dataset_records[q]
            p_text = rec["prompt"] if "prompt" in rec else rec.get("problem", "")
            c = bb.encode_context(p_text)
            if hasattr(c, "to"):
                c = c.to(device=device)
            ctx_by_q[q] = c
    print(f"[CACHE] Contexts pre-encoded.", flush=True)

    # 6. Training Loop
    telemetry = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    alpha = 0.25
    g0 = 0.25

    print(f"[TRAIN] Starting pilot training loop for {args.max_steps} steps...", flush=True)
    bb.train()
    projector.train()

    step = 0
    item_ptr = 0
    while step < args.max_steps:
        t0 = time.time()
        optimizer.zero_grad()

        # Build micro-batch
        batch_qids = []
        for _ in range(args.batch_size):
            batch_qids.append(valid_items[item_ptr % len(valid_items)])
            item_ptr += 1

        batch_loss = torch.tensor(0.0, device=device)
        batch_task_loss = torch.tensor(0.0, device=device)
        batch_ift_loss = torch.tensor(0.0, device=device)

        for q in batch_qids:
            rec = dataset_records[q]
            p_text = rec["prompt"] if "prompt" in rec else rec.get("problem", "")
            s_text = rec["solution"]

            if args.text_drop > 0 and drop_rng.random() < args.text_drop:
                n_drop += 1
                input_ids, targets, s_st, s_en = prepare_item_batch(tokenizer, "", s_text)
            else:
                input_ids, targets, s_st, s_en = prepare_item_batch(tokenizer, p_text, s_text)
            input_ids = input_ids.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)

            zr = extract_tensor(zsrc_dict, q).to(device=device, dtype=torch.float32)
            if zr.dim() == 2:
                zr = zr.unsqueeze(0)

            if args.mode == "A":
                zs = extract_tensor(zstar_dict, q).to(device=device, dtype=torch.float32)
                if zs.dim() == 2:
                    zs = zs.unsqueeze(0)
                z_cur = zs.detach()
            else:
                z_cur = zr.detach()

            # Differentiable DEQ transition step
            w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
            f_weights = sparse_module_weights(routing_weights(z_cur.float(), w_g, 1.0), 3)
            sel_ids = torch.sort(torch.topk(f_weights.reshape(-1), 3).indices.to(dtype=torch.long)).values
            r_ids = [int(x) for x in sel_ids.detach().cpu().tolist()]
            ex = {
                "top_k": 3,
                "deq_map_mode": bb.deq_map_mode,
                "routing_module_ids": r_ids,
                "routing_module_ids_tensor": sel_ids,
            }
            ctx = ctx_by_q[q]
            prepare_hook = getattr(bb, "prepare_deq_context", None)
            if callable(prepare_hook):
                ex["eq3_static_inputs"] = prepare_hook(z_cur, ctx, r_ids)

            phi_z = bb.deq_step(z_cur, ctx, f_weights, ex).float()
            z_next = (1.0 - alpha) * z_cur + alpha * (phi_z + g0 * zr)

            if args.mode == "A":
                res = z_next - z_cur
                l_ift = (res.square().sum()) / (z_cur.square().sum().clamp_min(1e-6))
            else:
                l_ift = torch.tensor(0.0, device=device)

            # Prefix projection & LM CE loss
            prefix = projector(z_next).to(dtype=embed_tokens.weight.dtype)
            base_embeds = embed_tokens(input_ids)
            base_embeds[:, s_st:s_en + 1, :] = prefix

            per_layer_inputs = lm_model.get_per_layer_inputs(input_ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kwargs_fwd = {"per_layer_inputs": per_layer_inputs} if per_layer_inputs is not None else {}

            out = lm_model(inputs_embeds=base_embeds, return_dict=True, **kwargs_fwd)
            h_target = out.last_hidden_state[:, :-1, :]
            shift_logits = lm_head(h_target).contiguous()
            shift_targets = targets[:, 1:].contiguous()
            l_task = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=-100)

            item_loss = (l_task + (args.lambda_ift * l_ift if args.mode == "A" else 0.0)) / args.batch_size
            item_loss.backward()

            batch_loss += item_loss.detach()
            batch_task_loss += (l_task.detach() / args.batch_size)
            batch_ift_loss += (l_ift.detach() / args.batch_size)

        optimizer.step()
        step_time = time.time() - t0

        vram_mb = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        telemetry.append({
            "step": step,
            "loss": float(batch_loss.item()),
            "l_task": float(batch_task_loss.item()),
            "l_ift": float(batch_ift_loss.item()),
            "step_time_s": step_time,
            "vram_mb": vram_mb,
        })

        if step % 5 == 0 or step == args.max_steps - 1:
            print(f"[Step {step:3d}] Loss: {batch_loss.item():.4f} | Task: {batch_task_loss.item():.4f} | IFT: {batch_ift_loss.item():.6f} | VRAM: {vram_mb:.1f}MB ({step_time:.2f}s)", flush=True)

        step += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 7. Save Checkpoint & Manifest
    save_dict = {
        "mode": args.mode,
        "step": args.max_steps,
        "projector_state_dict": projector.state_dict(),
        "deq_lora_state_dict": {k: v.cpu() for k, v in bb.state_dict().items() if any(t in k for t in ["lora_A", "lora_B", "routing_proj", "gamma_param"])},
        "telemetry": telemetry,
    }
    torch.save(save_dict, args.out)
    print(f"[SAVE] Checkpoint saved to {args.out} ({os.path.getsize(args.out):,} bytes)", flush=True)

    if getattr(args, "decoder_out", None):
        os.makedirs(os.path.dirname(args.decoder_out) or ".", exist_ok=True)
        _dec = {"state_dict": projector.state_dict(), "step": args.max_steps, "proj_version": 2 if args.proj_v2 else 1}
        if lm_lora_params:
            _dec["lm_lora_rank"] = args.lm_lora_rank
            _dec["lm_lora_state_dict"] = {k: v.detach().cpu() for k, v in lm_model.state_dict().items() if "lora_" in k}
        torch.save(_dec, args.decoder_out)
        print(f"[SAVE] Canonical Decoder checkpoint saved to {args.decoder_out}", flush=True)

    if getattr(args, "deq_out", None):
        os.makedirs(os.path.dirname(args.deq_out) or ".", exist_ok=True)
        torch.save({"state_dict": save_dict["deq_lora_state_dict"], "step": args.max_steps}, args.deq_out)
        print(f"[SAVE] DEQ backbone checkpoint saved to {args.deq_out}", flush=True)

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump({
                "mode": args.mode,
                "steps": args.max_steps,
                "batch_size": args.batch_size,
                "final_task_loss": telemetry[-1]["l_task"] if telemetry else None,
                "text_drop": args.text_drop, "n_text_dropped": n_drop, "seed": args.seed,
                "init_decoder": args.init_decoder, "init_deq": args.init_deq,
                "final_ift_loss": telemetry[-1]["l_ift"] if telemetry else None,
                "avg_step_time_s": float(np.mean([t["step_time_s"] for t in telemetry])),
                "peak_vram_mb": max([t["vram_mb"] for t in telemetry]) if telemetry else 0.0,
            }, f, indent=2)
        print(f"[MANIFEST] Manifest saved to {args.manifest}", flush=True)

    print("=" * 80)
    print(f"CTS G2v3 M3 Full Joint Training Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80, flush=True)
    return {"status": "SUCCESS", "telemetry": telemetry}


def main():
    parser = argparse.ArgumentParser(description="CTS G2v3 M3 Full Joint Training Script")
    parser.add_argument("--mode", type=str, choices=["A", "E"], default="A", help="Mode: A (M3 fixed-point phantom gradient) or E (E''' 1-step contrast)")
    parser.add_argument("--items", type=str, default="configs/s7_train_993.json", help="Path to json list of item indices")
    parser.add_argument("--max_items", type=int, default=1000, help="Max items to load")
    parser.add_argument("--max_steps", type=int, default=1500, help="Number of optimization steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="Learning rate for w_proj")
    parser.add_argument("--lr_adapter", type=float, default=3e-4, help="Learning rate for adapter")
    parser.add_argument("--lr_deq", type=float, default=3e-5, help="Learning rate for DEQ parameters")
    parser.add_argument("--lambda_ift", type=float, default=0.1, help="Weight for auxiliary IFT residual loss")
    parser.add_argument("--base_model", type=str, default=DEFAULT_PURE_GEMMA_SNAPSHOT, help="Pure Gemma snapshot path")
    parser.add_argument("--backbone_ckpt", type=str, default=DEFAULT_FROZEN_BACKBONE_CKPT, help="Frozen CTS backbone checkpoint path")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to math prompts dataset jsonl")
    parser.add_argument("--zstar", type=str, default="results_s7/g2v2/zstar_zsrcinj_train1000.pt", help="Path to zstar tensor pt")
    parser.add_argument("--zsrc", type=str, default="results_s7/zsrc_train1000_P4.pt", help="Path to zsrc tensor pt")
    parser.add_argument("--out", type=str, default="results_s7/g2v3/full_model_A.pt", help="Output checkpoint path")
    parser.add_argument("--decoder_out", type=str, default="results_s7/g2v3/decoder_A_v36.pt", help="Canonical decoder checkpoint path")
    parser.add_argument("--deq_out", type=str, default="results_s7/g2v3/deq_backbone_A.pt", help="DEQ backbone checkpoint path")
    parser.add_argument("--manifest", type=str, default="results_s7/g2v3/full_manifest_A.json", help="Output manifest path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device string")
    parser.add_argument("--smoke", action="store_true", help="Run fast in-memory smoke test")
    parser.add_argument("--text_drop", type=float, default=0.0, help="M4: prob of removing problem text per item")
    parser.add_argument("--init_decoder", type=str, default="", help="M4: warm-start decoder ckpt")
    parser.add_argument("--init_deq", type=str, default="", help="M4: warm-start DEQ ckpt")
    parser.add_argument("--seed", type=int, default=2041, help="M4: seed")
    parser.add_argument("--proj_v2", action="store_true", help="use PrefixProjectorV2")
    parser.add_argument("--lm_lora_rank", type=int, default=0, help="decoder-side LM LoRA rank (0 = off)")
    parser.add_argument("--lr_lm_lora", type=float, default=1e-4, help="LR for decoder-side LM LoRA")
    parser.add_argument("--hidden_size", type=int, default=2560, help="Hidden size (smoke only)")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocab size (smoke only)")

    args = parser.parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()

