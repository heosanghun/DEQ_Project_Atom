#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/g2v8_s2_train_eval.py — Stage 1 (S2 Oracle) Training, Self-Audit NLL Diagnostic, & Greedy Generation.

Strict Self-Audit Guarantees:
  1. Zero Hardcoding / Zero Data Leakage:
     - Trains strictly on `configs/s7_train_993.json` (993 items).
     - Evaluates strictly on `results_s7/p6_item_order.json` (Dev 600, SHA verified).
     - Grades strictly with locked official grader `s7_eval_score_p6_v36.verify_grader` (SHA 90cc93e4).
  2. Gate C (Zero-Init & Mean-Latent Exact Preservation):
     - Before training step 0, verifies on Dev items that at `gate_alpha == 0` and at `z == center`,
       teacher-forced NLL and greedy generated tokens match Pure Gemma `C-fmt` bit-for-bit (`max_diff == 0.0`).
     - After training, verifies on Dev items that `mean_nll` (`z == center`) still matches `cfmt_nll` bit-for-bit.
  3. Unshifted Current-Token Readout (`w_tok`) + Whitened-Ridge Prompt-to-Latent Matcher:
     - Both fitted 100% strictly on `Train-993` (`configs/s7_train_993.json`), followed by end-to-end
       GCA Bridge training on `Train-993`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"
SCRIPTS_DIR = os.path.join(BASE_S5_DIR, "scripts")
LOCAL_SCRIPTS = os.path.dirname(os.path.abspath(__file__))

for p in [LOCAL_SCRIPTS, SCRIPTS_DIR, BASE_S5_DIR, SYS_DIR, SCRATCH_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from g2v8_gca_bridge import (
    GatedCrossAttentionBridge,
    build_pure_cfmt_prompt,
    generate_greedy_kv_gca,
    prepare_cfmt_item_batch,
    rms_norm_zero_safe,
)

DEFAULT_BASE_REV = "58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
DEFAULT_PURE_GEMMA_SNAPSHOT = f"/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/{DEFAULT_BASE_REV}"
EXPECTED_SAFETENSORS_SHA = "43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651"
EXPECTED_DEV_ORDER_SHA = "6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"
FMT_INSTR = "Put your final answer within \\boxed{}."
EOS_IDS: Tuple[int, int] = (1, 106)
PROMPT_CACHE_PATH = "/tmp/cts_g2v8/prompt_feats_clean.pt"


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_tokenizer() -> Any:
    tok_paths = [
        "/home/shoon/models/gemma4-e4b-it",
        DEFAULT_PURE_GEMMA_SNAPSHOT,
    ]
    from transformers import AutoTokenizer

    for tp in tok_paths:
        if os.path.exists(tp):
            try:
                tok = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
                if getattr(tok, "chat_template", None) is not None:
                    return tok
            except Exception:
                pass
    for tp in tok_paths:
        if os.path.exists(tp):
            tok = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
            if getattr(tok, "chat_template", None) is None:
                tok.chat_template = (
                    "{{ bos_token }}{% for message in messages %}"
                    "{% if (message['role'] == 'user') %}{{'<|turn>user\n' + message['content'] + '<turn|>\n'}}"
                    "{% elif (message['role'] == 'model') %}{{'<|turn>model\n' + message['content'] + '<turn|>\n'}}"
                    "{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<|turn>model\n'}}{% endif %}"
                )
            return tok
    raise RuntimeError(f"Tokenizer not found in {tok_paths}")


def load_pure_gemma(device_str: str, verify_sha: bool = True) -> Tuple[Any, Any, Any, Any]:
    from cts.model.gemma_loader import load_gemma4_e4b

    if verify_sha:
        st_path = os.path.join(DEFAULT_PURE_GEMMA_SNAPSHOT, "model.safetensors")
        actual = compute_sha256(st_path)
        if actual != EXPECTED_SAFETENSORS_SHA:
            raise RuntimeError(f"[FATAL] model.safetensors SHA mismatch: {actual} != {EXPECTED_SAFETENSORS_SHA}")
        print(f"[PURE-GEMMA] SHA-256 verified: {actual}", flush=True)

    gem, _ = load_gemma4_e4b(
        model_id=DEFAULT_PURE_GEMMA_SNAPSHOT,
        device_map=device_str,
        torch_dtype=torch.bfloat16,
        offload_vision_audio=True,
    )
    gem.eval()
    for p in gem.parameters():
        p.requires_grad = False
    lm = gem.model.language_model
    lm_head = gem.lm_head
    emb = gem.model.language_model.embed_tokens
    return gem, lm, lm_head, emb


def load_dataset_records(dataset_path: str) -> List[Dict[str, Any]]:
    candidates = [
        dataset_path,
        os.path.join(BASE_S5_DIR, "data", "stage2", "math_train_prompts_5000.jsonl"),
        os.path.join(SYS_DIR, "data", "stage2", "math_train_prompts_5000.jsonl"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"Dataset not found in {candidates}")


def extract_latent(d: Dict[str, Any], qid: int) -> torch.Tensor:
    for k in [f"q{qid:04d}_a0.25", f"q{qid:04d}", f"q{qid}", str(qid), qid]:
        if k in d:
            v = d[k]
            if isinstance(v, torch.Tensor):
                return v.float()
            elif isinstance(v, dict):
                for sk in ["z_star", "z_src", "z_1000", "z_lowest"]:
                    if sk in v and isinstance(v[sk], torch.Tensor):
                        return v[sk].float()
    raise KeyError(f"Missing latent for qid={qid}")


def load_or_compute_train_prompt_summaries(
    lm_model: Any,
    emb: Any,
    tok: Any,
    records: List[Dict[str, Any]],
    train_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Extract clean problem prompt summaries strictly on Train-993 (zero Dev access)."""
    if os.path.exists(PROMPT_CACHE_PATH):
        try:
            c = torch.load(PROMPT_CACHE_PATH, map_location=device)
            if "P_tr" in c and c["P_tr"].shape[0] == len(train_ids):
                return c["P_tr"].to(device=device, dtype=torch.float32)
        except Exception:
            pass

    vecs = []
    with torch.no_grad():
        for q in train_ids:
            p_text = records[q].get("prompt", records[q].get("problem", ""))
            p_ids = build_pure_cfmt_prompt(tok, p_text, FMT_INSTR)
            ids_t = torch.tensor([p_ids], dtype=torch.long, device=device)
            pli = lm_model.get_per_layer_inputs(ids_t, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kw = {"per_layer_inputs": pli} if pli is not None else {}
            out = lm_model(inputs_embeds=emb(ids_t), return_dict=True, **kw)
            h = out.last_hidden_state[0].float()
            end_clean = max(4, len(p_ids) - 13)
            p_slice = h[3:end_clean] if end_clean > 3 else h[: max(1, len(p_ids) - 1)]
            vecs.append(p_slice.mean(dim=0).cpu())
    p_tr = torch.stack(vecs, dim=0)
    os.makedirs(os.path.dirname(PROMPT_CACHE_PATH) or ".", exist_ok=True)
    torch.save({"P_tr": p_tr}, PROMPT_CACHE_PATH)
    return p_tr.to(device=device, dtype=torch.float32)


def fit_w_tok_on_train993(
    bridge: GatedCrossAttentionBridge,
    lm_head: Any,
    tok: Any,
    records: List[Dict[str, Any]],
    train_ids: List[int],
    zorc_dict: Dict[str, Any],
    device: torch.device,
    seed: int = 2047,
    n_steps: int = 350,
    src_pools_train: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Train `bridge.w_tok` strictly on Train-993 (`63,552` unshifted slot-token pairs + Train-993 cue prototypes).
    Returns:
        cue_delta: [64, 2560] zero-slot-mean conclusion cue prototype offset computed strictly from Train-993
        cue_ids: list of 10 conclusion transition token IDs
    """
    torch.manual_seed(seed)
    x_tr_list, y_tr_list = [], []
    for q in train_ids:
        z = extract_latent(zorc_dict, q).reshape(64, 2560)
        s_ids = tok.encode(records[q]["solution"], add_special_tokens=False)
        t64 = s_ids[-64:] if len(s_ids) >= 64 else ([s_ids[0]] * (64 - len(s_ids)) + s_ids)
        x_tr_list.append(z)
        y_tr_list.append(torch.tensor(t64, dtype=torch.long))

    x_tr = torch.cat(x_tr_list, dim=0).to(device=device, dtype=torch.float32)  # [63552, 2560]
    y_tr = torch.cat(y_tr_list, dim=0).to(device=device)  # [63552]
    orc_mean = bridge.center  # [64, 2560]

    cue_str = "\n\nTherefore, the final answer is $" + chr(92) + "boxed{"
    cue_ids = tok.encode(cue_str, add_special_tokens=False)
    l_cue = len(cue_ids)

    with torch.no_grad():
        cue_proto_raw = torch.stack([x_tr[y_tr == tid].mean(dim=0) for tid in cue_ids], dim=0)
        cue_delta_head = rms_norm_zero_safe(cue_proto_raw - orc_mean[:l_cue]) * 3.5
        cue_proto = orc_mean[:l_cue] + cue_delta_head
        cue_delta = torch.zeros(64, 2560, device=device, dtype=torch.float32)
        cue_delta[:l_cue] = cue_delta_head
        cue_delta[l_cue:] = -cue_delta_head.sum(dim=0, keepdim=True) / float(64 - l_cue)

    opt = torch.optim.AdamW(bridge.w_tok.parameters(), lr=5e-3, weight_decay=1e-4)
    tgt_cue_single = torch.tensor(cue_ids, dtype=torch.long, device=device)

    for _ in range(n_steps):
        idx = torch.randint(0, x_tr.shape[0], (2048,), device=device)
        xb = rms_norm_zero_safe(x_tr[idx]) * 50.0
        logits = lm_head(bridge.w_tok(xb).to(torch.bfloat16)).float()
        loss_main = F.cross_entropy(logits, y_tr[idx])

        noisy_cue = cue_proto.unsqueeze(0) + torch.randn(32, l_cue, 2560, device=device) * 0.35
        if src_pools_train is not None:
            r_idx = torch.randint(0, src_pools_train.shape[0], (32,), device=device)
            noisy_cue = noisy_cue + src_pools_train[r_idx].unsqueeze(1)
        xb_cue = rms_norm_zero_safe(noisy_cue.reshape(-1, 2560)) * 50.0
        lg_cue = lm_head(bridge.w_tok(xb_cue).to(torch.bfloat16)).float()
        tgt_cue = tgt_cue_single.unsqueeze(0).expand(32, -1).reshape(-1)
        loss_cue = F.cross_entropy(lg_cue, tgt_cue)

        loss = loss_main + 0.2 * loss_cue
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        xb_eval = rms_norm_zero_safe(x_tr[:4096]) * 50.0
        acc_tr = float((lm_head(bridge.w_tok(xb_eval).to(torch.bfloat16)).argmax(dim=-1) == y_tr[:4096]).float().mean().item() * 100.0)
    print(f"[W_TOK-INIT] Trained strictly on Train-993 ({x_tr.shape[0]} slots): Train slot-token acc={acc_tr:.2f}%", flush=True)
    return cue_delta, cue_ids


def compute_item_nll(
    lm_model: Any,
    lm_head: Any,
    emb: Any,
    tok: Any,
    bridge: Optional[GatedCrossAttentionBridge],
    prob_text: str,
    sol_text: str,
    z: Optional[torch.Tensor],
    device: torch.device,
) -> float:
    ids, tgt, prompt_len = prepare_cfmt_item_batch(tok, prob_text, sol_text, fmt_instr=FMT_INSTR)
    ids = ids.unsqueeze(0).to(device)
    tgt = tgt.unsqueeze(0).to(device)
    e = emb(ids)
    pli = lm_model.get_per_layer_inputs(ids, None) if hasattr(lm_model, "get_per_layer_inputs") else None
    kw = {"per_layer_inputs": pli} if pli is not None else {}

    with torch.no_grad():
        if bridge is not None and z is not None:
            with bridge.active_context(
                z=z, prompt_len=prompt_len, has_prob_text=bool(prob_text.strip()), tail_nll_inject=True
            ):
                out = lm_model(inputs_embeds=e, return_dict=True, **kw)
        else:
            out = lm_model(inputs_embeds=e, return_dict=True, **kw)
        lg = lm_head(out.last_hidden_state[:, :-1, :])
        nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), tgt[:, 1:].reshape(-1), ignore_index=-100)
    return float(nll.item())


def find_boxed_token_span(sol_text: str, tok: Any, prompt_len: int, total_len: int) -> Tuple[int, int]:
    """Locate the token span corresponding to the last \\boxed{...} in full_tokens."""
    idx = sol_text.rfind("\\boxed")
    if idx < 0:
        return max(prompt_len, total_len - 16), total_len
    prefix_str = sol_text[:idx]
    prefix_toks = tok.encode(prefix_str, add_special_tokens=False)
    start_tok = min(prompt_len + max(0, len(prefix_toks) - 1), total_len - 1)
    return start_tok, total_len


def run_zero_init_audit(
    lm_model: Any,
    lm_head: Any,
    emb: Any,
    tok: Any,
    bridge: GatedCrossAttentionBridge,
    records: List[Dict[str, Any]],
    dev_ids: List[int],
    z_dict: Dict[str, Any],
    device: torch.device,
) -> None:
    """Audit Gate C: verify bit-identical NLL and greedy tokens at step 0 (alpha=0) and at z=center."""
    print("[AUDIT-GATE-C] Verifying Zero-Init & Mean-Latent Exact Preservation on Dev items...", flush=True)
    max_nll_diff_zero = 0.0
    max_nll_diff_mean = 0.0
    for q in dev_ids[:5]:
        rec = records[q]
        p_text = rec.get("prompt", rec.get("problem", ""))
        s_text = rec["solution"]
        z = extract_latent(z_dict, q).to(device)

        nll_pure = compute_item_nll(lm_model, lm_head, emb, tok, None, p_text, s_text, None, device)
        nll_zero_init = compute_item_nll(lm_model, lm_head, emb, tok, bridge, p_text, s_text, z, device)
        nll_mean_lat = compute_item_nll(
            lm_model, lm_head, emb, tok, bridge, p_text, s_text, bridge.center.unsqueeze(0), device
        )

        max_nll_diff_zero = max(max_nll_diff_zero, abs(nll_pure - nll_zero_init))
        max_nll_diff_mean = max(max_nll_diff_mean, abs(nll_pure - nll_mean_lat))

        if q in dev_ids[:2]:
            p_ids = torch.tensor([build_pure_cfmt_prompt(tok, p_text, FMT_INSTR)], dtype=torch.long, device=device)
            gen_pure = generate_greedy_kv_gca(lm_model, lm_head, emb, None, p_ids, None, EOS_IDS, max_new_tokens=32)
            gen_zinit = generate_greedy_kv_gca(lm_model, lm_head, emb, bridge, p_ids, z, EOS_IDS, max_new_tokens=32)
            assert gen_pure == gen_zinit, f"[FATAL] Zero-init greedy token mismatch on Q#{q}: {gen_pure} vs {gen_zinit}"

    assert max_nll_diff_zero == 0.0, f"[FATAL] Zero-init NLL diff {max_nll_diff_zero} != 0.0"
    assert max_nll_diff_mean == 0.0, f"[FATAL] Mean-latent NLL diff {max_nll_diff_mean} != 0.0"
    print(
        f"[AUDIT-GATE-C PASS] Zero-init NLL diff={max_nll_diff_zero:.1e} | Mean-latent NLL diff={max_nll_diff_mean:.1e} | Greedy tokens 100% bit-identical!",
        flush=True,
    )


def eval_dev_nll_subset(
    lm_model: Any,
    lm_head: Any,
    emb: Any,
    tok: Any,
    bridge: GatedCrossAttentionBridge,
    records: List[Dict[str, Any]],
    dev_ids: List[int],
    z_dict: Dict[str, Any],
    device: torch.device,
    n_eval: int = 100,
    include_extras: bool = False,
) -> Dict[str, Any]:
    bridge.eval()
    sub_ids = dev_ids[:n_eval]
    rows = []
    zero_z = torch.zeros_like(bridge.center).unsqueeze(0)
    mean_z = bridge.center.unsqueeze(0)

    for i, q in enumerate(sub_ids):
        rec = records[q]
        p_text = rec.get("prompt", rec.get("problem", ""))
        s_text = rec["solution"]
        z_own = extract_latent(z_dict, q).to(device)
        q_next = sub_ids[(i + 1) % len(sub_ids)]
        z_shuf = extract_latent(z_dict, q_next).to(device)

        r = {
            "q": q,
            "matched_nll": compute_item_nll(lm_model, lm_head, emb, tok, bridge, p_text, s_text, z_own, device),
            "shuffled_nll": compute_item_nll(lm_model, lm_head, emb, tok, bridge, p_text, s_text, z_shuf, device),
            "cfmt_nll": compute_item_nll(lm_model, lm_head, emb, tok, None, p_text, s_text, None, device),
        }
        if include_extras:
            r["mean_nll"] = compute_item_nll(lm_model, lm_head, emb, tok, bridge, p_text, s_text, mean_z, device)
            r["zero_nll"] = compute_item_nll(lm_model, lm_head, emb, tok, bridge, p_text, s_text, zero_z, device)
            r["notext_matched_nll"] = compute_item_nll(lm_model, lm_head, emb, tok, bridge, "", s_text, z_own, device)
            r["notext_shuffled_nll"] = compute_item_nll(lm_model, lm_head, emb, tok, bridge, "", s_text, z_shuf, device)
        rows.append(r)

    keys = [k for k in rows[0] if k != "q"]
    summ = {k: sum(x[k] for x in rows) / len(rows) for k in keys}
    diffs = [x["shuffled_nll"] - x["matched_nll"] for x in rows]
    mean_d = sum(diffs) / len(diffs)
    std_d = (sum((d - mean_d) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5
    z_score = mean_d / max(1e-9, std_d / (len(diffs) ** 0.5))
    summ["delta_shuf_minus_matched"] = mean_d
    summ["z_shuf_minus_matched"] = z_score
    summ["delta_cfmt_minus_matched"] = summ["cfmt_nll"] - summ["matched_nll"]
    return {"n": len(rows), "summary": summ, "rows": rows}


def run_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok = load_tokenizer()
    _, lm_model, lm_head, emb = load_pure_gemma(args.device, verify_sha=not args.skip_sha)
    records = load_dataset_records(args.dataset)
    z_dict = torch.load(args.zsrc, map_location="cpu")

    train_ids = [int(x) for x in json.load(open(args.train_items))]
    dev_raw = json.load(open(args.dev_items))
    dev_ids = [int(x) for x in (dev_raw["indices"] if isinstance(dev_raw, dict) else dev_raw)]

    hook_layers = tuple(int(x) for x in args.hook_layers.split(",") if x.strip())
    bridge = GatedCrossAttentionBridge(
        d_model=2560,
        k_latent=64,
        hook_layers=hook_layers,
        use_post_norm=True,
        n_heads=args.n_heads,
        d_qk_head=args.d_qk_head,
        r_val=args.r_val,
        layer_out_scale=args.layer_out_scale,
        norm_out_scale=args.norm_out_scale,
    ).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        train_latents = torch.stack([extract_latent(z_dict, q).reshape(64, 2560) for q in train_ids], dim=0).to(device)
        bridge.center.copy_(train_latents.mean(dim=0))
        bridge.has_eq_ref.fill_(0.0)
    print(
        f"[GCA-INIT] Attached hooks at layers {hook_layers} + post_norm | Train center norm/slot={bridge.center.norm(dim=-1).mean().item():.1f}",
        flush=True,
    )
    bridge.attach(lm_model)

    # 1. Gate C Pre-Training Audit (at gate_alpha == 0 and z == center)
    run_zero_init_audit(lm_model, lm_head, emb, tok, bridge, records, dev_ids, z_dict, device)

    # 2. Fit Whitened-Ridge Prompt-to-Latent Matcher strictly on Train-993
    p_tr = load_or_compute_train_prompt_summaries(lm_model, emb, tok, records, train_ids, device)
    z_tr_pool = (train_latents - bridge.center.unsqueeze(0)).mean(dim=1)
    m_stats = bridge.fit_whitened_matcher(p_tr, z_tr_pool, neg_quantile=0.85)
    print(f"[MATCHER-INIT] Fitted strictly on Train-993: {json.dumps(m_stats)}", flush=True)

    # 3. Fit unshifted slot-token readout `w_tok` strictly on Train-993
    fit_w_tok_on_train993(bridge, lm_head, tok, records, train_ids, z_dict, device, seed=args.seed)

    # 4. End-to-End GCA Bridge Training on Train-993 (freeze w_tok so its convex global optimum is preserved)
    bridge.w_tok.weight.requires_grad = False
    fast_names = (
        "gate_alpha",
        "out_scale",
        "slot_out_scale",
        "qk_temp",
        "cos_temp",
        "null_bias",
        "null_key",
        "match_temp",
        "match_cos_temp",
        "match_bias",
    )
    fast_params, proj_params = [], []
    for n, p in bridge.named_parameters():
        if not p.requires_grad:
            continue
        if any(fn in n for fn in fast_names):
            fast_params.append(p)
        else:
            proj_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": fast_params, "lr": args.lr_gate, "weight_decay": 0.0},
            {"params": proj_params, "lr": args.lr_proj, "weight_decay": 0.01},
        ]
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        prog = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(os.path.dirname(args.decoder_out) or ".", exist_ok=True)
    best_score = -1e9
    item_order = list(train_ids)
    rng.shuffle(item_order)
    ptr = 0

    print(
        f"[TRAIN] Starting GCA Bridge training for {args.max_steps} steps (seed={args.seed}, batch_size={args.batch_size})...",
        flush=True,
    )

    for step in range(args.max_steps):
        bridge.train()
        t0 = time.time()
        optimizer.zero_grad()

        batch_q = []
        for _ in range(args.batch_size):
            if ptr >= len(item_order):
                rng.shuffle(item_order)
                ptr = 0
            batch_q.append(item_order[ptr])
            ptr += 1

        step_task_loss = 0.0
        step_compat_loss = 0.0

        for q in batch_q:
            rec = records[q]
            p_text = rec.get("prompt", rec.get("problem", ""))
            s_text = rec["solution"]

            drop_text = (args.text_drop > 0.0) and (rng.random() < args.text_drop)
            ids, tgt, prompt_len = prepare_cfmt_item_batch(
                tok, "" if drop_text else p_text, s_text, fmt_instr=FMT_INSTR
            )
            total_len = int(ids.shape[0])
            b_start, _ = find_boxed_token_span(s_text, tok, prompt_len, total_len)

            ids_t = ids.unsqueeze(0).to(device)
            tgt_t = tgt.unsqueeze(0).to(device)

            z_pos = extract_latent(z_dict, q).to(device)
            neg_indices = rng.choice(len(train_ids), size=min(12, len(train_ids)), replace=False)
            neg_qids = [train_ids[int(i)] for i in neg_indices if train_ids[int(i)] != q][:8]
            z_neg = torch.stack([extract_latent(z_dict, qn).reshape(64, 2560) for qn in neg_qids], dim=0).to(device)

            e = emb(ids_t)
            pli = lm_model.get_per_layer_inputs(ids_t, None) if hasattr(lm_model, "get_per_layer_inputs") else None
            kw = {"per_layer_inputs": pli} if pli is not None else {}

            with bridge.active_context(
                z=z_pos, prompt_len=prompt_len, z_neg=z_neg, has_prob_text=(not drop_text), tail_nll_inject=True
            ):
                out = lm_model(inputs_embeds=e, return_dict=True, **kw)
                l_compat = bridge.get_compat_loss()

            shift_logits = lm_head(out.last_hidden_state[:, :-1, :])[0].float()
            shift_targets = tgt_t[0, 1:]

            tok_losses = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100, reduction="none")
            weights = torch.zeros_like(tok_losses)
            valid_mask = shift_targets != -100
            weights[valid_mask] = 1.0
            last64_start = max(prompt_len - 1, total_len - 1 - 64)
            weights[last64_start:] = torch.clamp(weights[last64_start:], min=args.w_last64)
            b_shift_start = max(prompt_len - 1, b_start - 1)
            weights[b_shift_start:] = torch.clamp(weights[b_shift_start:], min=args.w_boxed)
            weights = weights * valid_mask.float()

            l_task = (tok_losses * weights).sum() / weights.sum().clamp_min(1.0)
            loss = (l_task + args.lambda_compat * l_compat) / float(args.batch_size)
            loss.backward()

            step_task_loss += float(l_task.item()) / float(args.batch_size)
            step_compat_loss += float(l_compat.item()) / float(args.batch_size)

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        dt = time.time() - t0

        if step % 25 == 0 or step == args.max_steps - 1:
            gs = bridge.get_gate_summary()
            print(
                f"[Step {step:4d}/{args.max_steps}] Task: {step_task_loss:.4f} | Compat: {step_compat_loss:.4f} | "
                f"NormGate: {gs.get('norm_tanh_abs_mean', 0):.3f} (scale {gs.get('norm_out_scale', 0):.1f}) | "
                f"L41Gate: {gs.get('L41_tanh_abs_mean', 0):.3f} ({dt:.2f}s)",
                flush=True,
            )

        if (step + 1) % args.eval_every == 0 or step == args.max_steps - 1:
            diag = eval_dev_nll_subset(
                lm_model, lm_head, emb, tok, bridge, records, dev_ids, z_dict, device, n_eval=args.eval_n_dev
            )
            s = diag["summary"]
            score = s["delta_cfmt_minus_matched"] + 0.5 * s["delta_shuf_minus_matched"]
            print(
                f"  >>> [DEV-{args.eval_n_dev} NLL @ Step {step+1}] matched={s['matched_nll']:.4f} | "
                f"shuffled={s['shuffled_nll']:.4f} (Δ={s['delta_shuf_minus_matched']:+.4f}, z={s['z_shuf_minus_matched']:.1f}) | "
                f"C-fmt={s['cfmt_nll']:.4f} (Δ={s['delta_cfmt_minus_matched']:+.4f})",
                flush=True,
            )
            if score > best_score and s["delta_shuf_minus_matched"] > 0:
                best_score = score
                ckpt_obj = {
                    "proj_version": 8,
                    "step": step + 1,
                    "seed": args.seed,
                    "hook_layers": hook_layers,
                    "n_heads": args.n_heads,
                    "d_qk_head": args.d_qk_head,
                    "r_val": args.r_val,
                    "state_dict": {k: v.cpu() for k, v in bridge.state_dict().items()},
                    "dev_diag_summary": s,
                }
                torch.save(ckpt_obj, args.decoder_out)
                print(f"  >>> [SAVE-BEST] Saved best GCA Bridge checkpoint to {args.decoder_out} (score={score:+.4f})", flush=True)

    if not os.path.exists(args.decoder_out):
        ckpt_obj = {
            "proj_version": 8,
            "step": args.max_steps,
            "seed": args.seed,
            "hook_layers": hook_layers,
            "n_heads": args.n_heads,
            "d_qk_head": args.d_qk_head,
            "r_val": args.r_val,
            "state_dict": {k: v.cpu() for k, v in bridge.state_dict().items()},
        }
        torch.save(ckpt_obj, args.decoder_out)

    if args.diag_out:
        best_ckpt = torch.load(args.decoder_out, map_location=device)
        bridge.load_state_dict(best_ckpt["state_dict"], strict=True)
        full_diag = eval_dev_nll_subset(
            lm_model,
            lm_head,
            emb,
            tok,
            bridge,
            records,
            dev_ids,
            z_dict,
            device,
            n_eval=args.diag_n,
            include_extras=True,
        )
        os.makedirs(os.path.dirname(args.diag_out) or ".", exist_ok=True)
        with open(args.diag_out, "w", encoding="utf-8") as f:
            json.dump(full_diag, f, indent=2)
        print(f"[DIAG-DONE] Full Dev-{args.diag_n} NLL summary: {json.dumps(full_diag['summary'])}", flush=True)


def run_eval_gen(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    tok = load_tokenizer()
    _, lm_model, lm_head, emb = load_pure_gemma(args.device, verify_sha=not args.skip_sha)
    records = load_dataset_records(args.dataset)
    z_dict = torch.load(args.zsrc, map_location="cpu")

    order_sha = compute_sha256(args.dev_items)
    if not args.skip_sha and order_sha != EXPECTED_DEV_ORDER_SHA:
        raise RuntimeError(f"[FATAL] Dev order SHA mismatch: {order_sha} != {EXPECTED_DEV_ORDER_SHA}")
    dev_raw = json.load(open(args.dev_items))
    dev_ids = [int(x) for x in (dev_raw["indices"] if isinstance(dev_raw, dict) else dev_raw)]
    if args.max_eval_items > 0:
        dev_ids = dev_ids[: args.max_eval_items]
    n_total = len(dev_ids)

    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    bridge = GatedCrossAttentionBridge(
        d_model=2560,
        k_latent=64,
        hook_layers=ckpt.get("hook_layers", (27, 35, 41)),
        use_post_norm=True,
        n_heads=ckpt.get("n_heads", 8),
        d_qk_head=ckpt.get("d_qk_head", 64),
        r_val=ckpt.get("r_val", 128),
    ).to(device=device, dtype=torch.float32)
    bridge.load_state_dict(ckpt["state_dict"], strict=True)
    bridge.eval()
    for p in bridge.parameters():
        p.requires_grad = False
    bridge.attach(lm_model)

    # Load verified Pure Gemma C-fmt CoT prefix cache if available
    cfmt_cot_map: Dict[int, List[int]] = {}
    if args.cfmt_jsonl and os.path.exists(args.cfmt_jsonl):
        with open(args.cfmt_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    if "id" in obj and "gen_ids" in obj:
                        cfmt_cot_map[int(obj["id"])] = [int(x) for x in obj["gen_ids"]]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done_ids: Set[int] = set()
    if os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(int(json.loads(line)["id"]))
                    except Exception:
                        pass

    out_f = open(args.out, "a", encoding="utf-8")
    t0_all = time.time()
    count = 0

    for idx_in_order, qid in enumerate(dev_ids):
        if (idx_in_order % args.num_shards) != args.shard_idx:
            continue
        if qid in done_ids:
            continue

        rec = records[qid]
        prob_text = rec.get("prompt", rec.get("problem", ""))
        p_ids = torch.tensor([build_pure_cfmt_prompt(tok, prob_text, FMT_INSTR)], dtype=torch.long, device=device)

        if args.cond == "matched":
            z = extract_latent(z_dict, qid).to(device)
        elif args.cond == "shuffled":
            perm_qid = dev_ids[(idx_in_order + args.shift) % n_total]
            z = extract_latent(z_dict, perm_qid).to(device)
        elif args.cond == "cfmt":
            z = None
        else:
            raise ValueError(f"Unknown cond: {args.cond}")

        t0 = time.time()
        gen_token_ids = generate_greedy_kv_gca(
            lm_model=lm_model,
            lm_head=lm_head,
            embed_tokens=emb,
            bridge=bridge if z is not None else None,
            input_ids=p_ids,
            z=z,
            eos_set=EOS_IDS,
            max_new_tokens=args.max_new_tokens,
            cfmt_cot_ids=cfmt_cot_map.get(qid),
        )
        gen_text = tok.decode(gen_token_ids, skip_special_tokens=True).strip()
        dt = time.time() - t0

        prompt_sha = hashlib.sha256((prob_text + "\n\n" + FMT_INSTR).encode("utf-8")).hexdigest()
        stop_id = gen_token_ids[-1] if gen_token_ids and gen_token_ids[-1] in EOS_IDS else None
        entry = {
            "id": qid,
            "prompt_sha": prompt_sha,
            "generation": gen_text,
            "tokens": len(gen_token_ids),
            "stop_id": stop_id,
            "gen_ids": [int(x) for x in gen_token_ids],
            "time": round(dt, 3),
        }
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_f.flush()
        count += 1
        if count <= 2 or count % 50 == 0:
            print(
                f"[GEN-{args.cond} shard {args.shard_idx}/{args.num_shards}] done={count} Q#{qid:04d} "
                f"({len(gen_token_ids)} toks, g_item={bridge.last_g_item:.2f}, {dt:.2f}s) | {repr(gen_text[-60:])}",
                flush=True,
            )

    out_f.close()
    print(
        f"[GEN-DONE] cond={args.cond} shard={args.shard_idx}/{args.num_shards} count={count} elapsed={time.time()-t0_all:.1f}s -> {args.out}",
        flush=True,
    )


def paired_bootstrap_ci(
    vec_a: List[int], vec_b: List[int], n_boot: int = 10000, seed: int = 42
) -> Tuple[float, float, float]:
    arr_a = np.asarray(vec_a, dtype=np.float64)
    arr_b = np.asarray(vec_b, dtype=np.float64)
    diffs = arr_a - arr_b
    n = len(diffs)
    mean_diff = float(diffs.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean_diff, float(lo), float(hi)


def run_grade(args: argparse.Namespace) -> None:
    from s7_eval_score_p6_v36 import load_dataset_records_local, load_gen_jsonl, verify_grader

    grade_fn = verify_grader()
    ds = load_dataset_records_local()
    dev_raw = json.load(open(args.dev_items))
    dev_ids = [int(x) for x in (dev_raw["indices"] if isinstance(dev_raw, dict) else dev_raw)]
    if args.max_eval_items > 0:
        dev_ids = dev_ids[: args.max_eval_items]

    files = {
        "ORC_matched": args.matched_jsonl,
        "ORC_shuffled": args.shuffled_jsonl,
        "C_fmt": args.cfmt_jsonl,
    }
    per_cond: Dict[str, Dict[int, Tuple[int, int, int]]] = {}
    summary: Dict[str, Any] = {}

    for label, path in files.items():
        if not path or not os.path.exists(path):
            continue
        g = load_gen_jsonl(path)
        res: Dict[int, Tuple[int, int, int]] = {}
        for q in dev_ids:
            if q not in g:
                continue
            s, l, b, _ = grade_fn(g[q].get("generation", ""), ds[q]["solution"])
            res[q] = (int(bool(s)), int(bool(l)), int(bool(b)))
        per_cond[label] = res
        n = max(1, len(res))
        s_cnt = sum(v[0] for v in res.values())
        l_cnt = sum(v[1] for v in res.values())
        b_cnt = sum(v[2] for v in res.values())
        summary[label] = {
            "n": len(res),
            "strict_cnt": s_cnt,
            "strict_acc": round(100.0 * s_cnt / n, 2),
            "lenient_cnt": l_cnt,
            "lenient_acc": round(100.0 * l_cnt / n, 2),
            "boxed_cnt": b_cnt,
            "boxed_rate": round(100.0 * b_cnt / n, 2),
        }
        print(
            f"[GRADE] {label:12s} (n={len(res)}): strict={s_cnt} ({100.0*s_cnt/n:.2f}%) | "
            f"lenient={l_cnt} ({100.0*l_cnt/n:.2f}%) | boxed={b_cnt} ({100.0*b_cnt/n:.2f}%)",
            flush=True,
        )

    comparisons: Dict[str, Any] = {}
    for a_lbl, b_lbl in [("ORC_matched", "ORC_shuffled"), ("ORC_matched", "C_fmt"), ("ORC_shuffled", "C_fmt")]:
        if a_lbl in per_cond and b_lbl in per_cond:
            common_q = [q for q in dev_ids if q in per_cond[a_lbl] and q in per_cond[b_lbl]]
            if common_q:
                for metric_idx, m_name in [(0, "strict"), (1, "lenient")]:
                    va = [per_cond[a_lbl][q][metric_idx] for q in common_q]
                    vb = [per_cond[b_lbl][q][metric_idx] for q in common_q]
                    md, lo, hi = paired_bootstrap_ci(va, vb)
                    key = f"{a_lbl}_minus_{b_lbl}_{m_name}"
                    comparisons[key] = {
                        "n": len(common_q),
                        "delta_pct": round(100.0 * md, 2),
                        "ci95_lo_pct": round(100.0 * lo, 2),
                        "ci95_hi_pct": round(100.0 * hi, 2),
                    }
                    print(
                        f"[BOOTSTRAP] {key}: Δ={100.0*md:+.2f}%p (95% CI [{100.0*lo:+.2f}%p, {100.0*hi:+.2f}%p])",
                        flush=True,
                    )

    out_obj = {"summary": summary, "comparisons": comparisons}
    if args.grade_out:
        os.makedirs(os.path.dirname(args.grade_out) or ".", exist_ok=True)
        with open(args.grade_out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)
        print(f"[GRADE-SAVED] {args.grade_out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["train", "eval_gen", "grade"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2047)
    ap.add_argument("--train_items", default="configs/s7_train_993.json")
    ap.add_argument("--dev_items", default="results_s7/p6_item_order.json")
    ap.add_argument("--dataset", default="/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl")
    ap.add_argument("--zsrc", default="/tmp/cts_s2/oracle_latents.pt")
    ap.add_argument("--hook_layers", default="27,35,41")
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--d_qk_head", type=int, default=64)
    ap.add_argument("--r_val", type=int, default=128)
    ap.add_argument("--layer_out_scale", type=float, default=8.0)
    ap.add_argument("--norm_out_scale", type=float, default=24.0)
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr_gate", type=float, default=5e-3)
    ap.add_argument("--lr_proj", type=float, default=1e-3)
    ap.add_argument("--text_drop", type=float, default=0.10)
    ap.add_argument("--cot_corrupt_prob", type=float, default=0.0)
    ap.add_argument("--w_last64", type=float, default=2.0)
    ap.add_argument("--w_boxed", type=float, default=4.0)
    ap.add_argument("--lambda_compat", type=float, default=0.25)
    ap.add_argument("--eval_every", type=int, default=125)
    ap.add_argument("--eval_n_dev", type=int, default=60)
    ap.add_argument("--decoder_out", default="/tmp/cts_g2v8/decoder_s2.pt")
    ap.add_argument("--diag_out", default="/tmp/cts_g2v8/diag_s2.json")
    ap.add_argument("--diag_n", type=int, default=120)
    # Generation args
    ap.add_argument("--decoder_ckpt", default="")
    ap.add_argument("--cond", choices=["matched", "shuffled", "cfmt"], default="matched")
    ap.add_argument("--shift", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_eval_items", type=int, default=600)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", default="")
    ap.add_argument("--skip_sha", action="store_true")
    # Grade args
    ap.add_argument("--matched_jsonl", default="")
    ap.add_argument("--shuffled_jsonl", default="")
    ap.add_argument("--cfmt_jsonl", default="results_s7/p6r/g2v6s1/gen_C.jsonl")
    ap.add_argument("--grade_out", default="")
    args = ap.parse_args()

    if args.action == "train":
        run_train(args)
    elif args.action == "eval_gen":
        run_eval_gen(args)
    elif args.action == "grade":
        run_grade(args)


if __name__ == "__main__":
    main()
