#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/g2v11_x8a_train.py - X8-A: joint training of the CTS DEQ (trainable LoRA/routing/gamma, as in M3) with the
cross-attention latent bridge under the X5 readout objective (CC 2026-10-23). Plan: shared/proposals/X8_READABLE_LATENT_DEQ_20261021.md §1-2, §7.

Per training item q (train 993 only):
  z_src  = cached root warm start (results_s7/zsrc_train1000_P4.pt, field z_src)            [64, 2560], fixed
  z_0    = cached canonical fixed point z*' (results_s7/g2v2/zstar_zsrcinj_train1000.pt)      [64, 2560], start (detached)
  z_{k+1} = (1 - alpha) z_k + alpha (Phi_theta(z_k) + g0 * z_src),  k = 0..K_unroll-1  (M2' zsrc-injection form, alpha=g0=0.25)
  z_out  = z_{K_unroll}  -> bridge (LatentXAttnBridge rev2) -> frozen pure Gemma -> X5 readout loss
           (style-matched own generation + \\boxed{gold} answer span weighted answer_weight, gen_budget, exact span policy of g2v10 rev3)
  L = L_read + lambda_fp * L_fp,  L_fp = ||z_out - z_{K_unroll-1}||^2 / ||z_{K_unroll-1}||^2 (relative fixed-point residual, G1 guard)
Gradients flow into the bridge AND into the DEQ trainable parameters (lora_A/B, routing_proj, gamma_param) through the K_unroll
unrolled steps ("phantom gradient", same parameter set as scripts/g2v3_m3_full_train.py). The pure LM is frozen. The DEQ backbone's
other weights are frozen (m1_final_frozen.pt). Center of the bridge = mean of the cached z*' training latents (fixed; recorded).
Outputs: bridge ckpt (+center), DEQ trainable state dict, manifest with target counts, residual telemetry and input SHA256.
No dev/test data is read. Items without a \\boxed gold or without an own generation are skipped and counted (as g2v10)."""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
for p in ["/home/shoon/cts_e1", "/home/shoon/cts_e1/scratch"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
from g2v3_m3_full_train import load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT, DEFAULT_FROZEN_BACKBONE_CKPT
from g2v3_m3_eval_latents_v2 import extract_tensor
from g2v9_xattn import LatentXAttnBridge
from g2v9_train import build_prompt_ids, DATA
from g2v10_train import gold_answer, build_target_text, sha256, BOX, CONNECTOR
from cts.model.gemma_loader import load_gemma4_e4b
from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.train.lora_compat import apply_paper_lora
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

TRAINABLE_KEYS = ("lora_A", "lora_B", "routing_proj", "gamma_param")


def deq_unroll(bb, z0, z_src, ctx, w_g, k_unroll, alpha, g0, top_k=3):
    """K differentiable M2'-form steps from z0. Returns (z_out, z_prev, residuals list)."""
    z_cur = z0; z_prev = z0; res = []
    for _ in range(k_unroll):
        f_weights = sparse_module_weights(routing_weights(z_cur.float(), w_g, 1.0), top_k)
        sel_ids = torch.sort(torch.topk(f_weights.reshape(-1), top_k).indices.to(dtype=torch.long)).values
        r_ids = [int(x) for x in sel_ids.detach().cpu().tolist()]
        ex = {"top_k": top_k, "deq_map_mode": bb.deq_map_mode, "routing_module_ids": r_ids, "routing_module_ids_tensor": sel_ids}
        hook = getattr(bb, "prepare_deq_context", None)
        if callable(hook):
            ex["eq3_static_inputs"] = hook(z_cur, ctx, r_ids)
        phi = bb.deq_step(z_cur, ctx, f_weights, ex).float()
        z_next = (1.0 - alpha) * z_cur + alpha * (phi + g0 * z_src)
        res.append(float(((z_next - z_cur).norm() / z_cur.norm().clamp_min(1e-6)).detach()))
        z_prev, z_cur = z_cur, z_next
    return z_cur, z_prev, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="configs/s7_train_993.json"); ap.add_argument("--dev_order", default="results_s7/p6_item_order.json")
    ap.add_argument("--zstar", default="results_s7/g2v2/zstar_zsrcinj_train1000.pt"); ap.add_argument("--zsrc", default="results_s7/zsrc_train1000_P4.pt")
    ap.add_argument("--targets", required=True); ap.add_argument("--backbone_ckpt", default=DEFAULT_FROZEN_BACKBONE_CKPT)
    ap.add_argument("--out_bridge", required=True); ap.add_argument("--out_deq", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--layers", default="10,20,30,40"); ap.add_argument("--max_steps", type=int, default=1500); ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lr_deq", type=float, default=3e-5)
    ap.add_argument("--k_unroll", type=int, default=2); ap.add_argument("--alpha", type=float, default=0.25); ap.add_argument("--g0", type=float, default=0.25)
    ap.add_argument("--lambda_fp", type=float, default=0.1); ap.add_argument("--answer_weight", type=float, default=5.0); ap.add_argument("--gen_budget", type=int, default=480)
    ap.add_argument("--max_seq_len", type=int, default=1536); ap.add_argument("--seed", type=int, default=2047); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_items", type=int, default=0, help="smoke: limit items"); a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = torch.device(a.device)
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    j = json.load(open(a.items)); items_all = [int(x) for x in (j["indices"] if isinstance(j, dict) else j)]
    if a.max_items: items_all = items_all[: a.max_items]
    assert len(set(items_all)) == len(items_all), "duplicate ids"
    jd = json.load(open(a.dev_order)); dev_ids = set(int(x) for x in (jd["indices"] if isinstance(jd, dict) else jd))
    assert not (set(items_all) & dev_ids), "training items overlap dev order"
    # targets (same policy as g2v10 rev3)
    own, seen_raw, n_dup, n_empty, n_rows = {}, set(), 0, 0, 0
    for l in open(a.targets, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l); q = int(r["id"]); n_rows += 1; g = (r.get("generation") or "").strip()
        if not g: n_empty += 1
        if q in seen_raw: n_dup += 1; continue
        seen_raw.add(q)
        if g: own[q] = (g, r.get("stop_id") is not None)
    tok = load_tokenizer(); eos = tok.eos_token_id
    T, C = {}, {"replaced": 0, "appended": 0, "skip_no_gold": 0, "skip_no_target": 0, "budget_truncated": 0, "skip_answer_truncated": 0, "finished_own": 0}
    for q in items_all:
        if q not in own: C["skip_no_target"] += 1; continue
        g = gold_answer(recs[q]["solution"])
        if g is None: C["skip_no_gold"] += 1; continue
        own_txt, finished = own[q]; C["finished_own"] += int(finished)
        pre, box, post, mode = build_target_text(own_txt, g, finished)
        ids_pre = tok.encode(pre, add_special_tokens=False); ids_box = tok.encode(box, add_special_tokens=False); ids_post = tok.encode(post, add_special_tokens=False) + [eos]
        if len(ids_pre) + len(ids_box) + len(ids_post) > a.gen_budget:
            keep = a.gen_budget - len(ids_box) - len(ids_post)
            if mode == "replaced":
                pre_txt = tok.decode(ids_pre[:max(1, keep - 16)], skip_special_tokens=True).rstrip() + CONNECTOR
                ids_pre = tok.encode(pre_txt, add_special_tokens=False); mode = "appended"; ids_post = tok.encode(".", add_special_tokens=False) + [eos]
                keep = a.gen_budget - len(ids_box) - len(ids_post); ids_pre = ids_pre[:max(1, keep)]
            else:
                conn = tok.encode(CONNECTOR, add_special_tokens=False); ids_pre = ids_pre[:max(1, keep - len(conn))] + conn
            C["budget_truncated"] += 1
        s_ids = ids_pre + ids_box + ids_post
        if len(s_ids) > a.gen_budget or len(ids_box) + len(ids_post) + 1 > a.gen_budget: C["skip_answer_truncated"] += 1; continue
        C[mode] += 1
        T[q] = (s_ids, [1.0] * len(ids_pre) + [a.answer_weight] * len(ids_box) + [1.0] * len(ids_post), [False] * len(ids_pre) + [True] * len(ids_box) + [False] * len(ids_post))
    zd_star = torch.load(a.zstar, map_location="cpu"); zd_src = torch.load(a.zsrc, map_location="cpu")
    Z0, ZS = {}, {}
    for q in list(T):
        zs, zr = extract_tensor(zd_star, q), extract_tensor(zd_src, q)
        if zs is None or zr is None: del T[q]; C["skip_no_target"] += 1; continue
        Z0[q] = zs.float().reshape(64, 2560); ZS[q] = zr.float().reshape(64, 2560)
    items = [q for q in items_all if q in T]
    print(f"[DATA] items {len(items)} of {len(items_all)} | targets rows dup {n_dup} empty {n_empty} | {C} | answer_weight {a.answer_weight} gen_budget {a.gen_budget} k_unroll {a.k_unroll} lambda_fp {a.lambda_fp}", flush=True)
    # frozen pure LM (readout)
    gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True); gem.eval()
    for p in gem.parameters(): p.requires_grad = False
    lm, head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
    # CTS DEQ backbone with trainable LoRA/routing/gamma (as M3)
    m_base, tok_base = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True)
    bb = GemmaCTSBackbone(m_base, tok_base)
    apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0, require_match=False, verbose=False)
    ck = torch.load(a.backbone_ckpt, map_location="cpu"); sd = ck.get("backbone_state_dict", ck)
    bb.load_state_dict({("routing_proj.parametrizations.weight.original" if k == "routing_proj" else k): v for k, v in sd.items()}, strict=True)
    bb.to(device=dev, dtype=torch.bfloat16)
    deq_params = []
    for name, p in bb.named_parameters():
        p.requires_grad = any(t in name for t in TRAINABLE_KEYS)
        if p.requires_grad: deq_params.append(p)
    w_g_fn = lambda: bb.routing_matrix().to(device=dev, dtype=torch.float32)
    print(f"[DEQ] trainable tensors {len(deq_params)} ({sum(p.numel() for p in deq_params):,} weights), map_mode {bb.deq_map_mode}", flush=True)
    ctx_by_q = {}
    with torch.no_grad():
        for q in items:
            c = bb.encode_context(recs[q]["prompt"]); ctx_by_q[q] = c.to(device=dev) if hasattr(c, "to") else c
    print(f"[CACHE] contexts {len(ctx_by_q)}", flush=True)
    layers = [int(x) for x in a.layers.split(",")]
    br = LatentXAttnBridge(layers).to(dev).float()
    with torch.no_grad():
        br.center.copy_(torch.stack([Z0[q] for q in items]).mean(0).to(dev))
    br.attach(lm); br.train(); bb.train()
    opt = torch.optim.AdamW([{"params": br.adapters.parameters(), "lr": a.lr, "weight_decay": 0.01}, {"params": deq_params, "lr": a.lr_deq, "weight_decay": 0.001}])
    print(f"[BRIDGE] layers {layers}, params {br.n_params():,}, center norm/slot {float(br.center.norm(dim=-1).mean()):.1f}", flush=True)
    tel, ptr, n_updates, n_visits, n_skip, contrib, t0 = [], 0, 0, 0, 0, set(), time.time()
    for step in range(a.max_steps):
        opt.zero_grad(set_to_none=True); bl = 0.0; bl_fp = 0.0; ans_sum = 0.0; ans_n = 0; any_grad = False; res_all = []
        for _ in range(a.batch_size):
            q = items[ptr % len(items)]; ptr += 1; n_visits += 1
            p_ids = build_prompt_ids(tok, recs[q]["prompt"]); s_ids, s_w, s_ans = T[q]
            ids = (p_ids + s_ids)[: a.max_seq_len]; wts = ([0.0] * len(p_ids) + s_w)[: a.max_seq_len]; ans = ([False] * len(p_ids) + s_ans)[: a.max_seq_len]
            if len(p_ids) + len(s_ids) > a.max_seq_len and (sum(ans) < sum(s_ans) or ids[-1] != eos): n_skip += 1; continue
            # DEQ unroll (differentiable) from cached z*' with cached z_src injection
            z0 = Z0[q].to(dev); zs = ZS[q].to(dev)
            z_out, z_prev, res = deq_unroll(bb, z0, zs, ctx_by_q[q], w_g_fn(), a.k_unroll, a.alpha, a.g0)
            l_fp = (z_out - z_prev).square().sum() / z_prev.detach().square().sum().clamp_min(1e-6)
            res_all.extend(res)
            br.set_latent(z_out)  # gradient flows through the bridge into the DEQ
            x = torch.tensor(ids, device=dev).unsqueeze(0); w = torch.tensor(wts[1:], device=dev, dtype=torch.float32); am = torch.tensor(ans[1:], device=dev)
            pli = lm.get_per_layer_inputs(x, None) if hasattr(lm, "get_per_layer_inputs") else None
            out = lm(inputs_embeds=emb(x), return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
            lg = head(out.last_hidden_state[:, :-1, :]).float()
            ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
            sup = w > 0
            if not bool(torch.isfinite(ce[sup]).all()): raise RuntimeError(f"non-finite CE step {step} item {q}")
            l_read = (ce[sup] * w[sup]).sum() / w[sup].sum()
            loss = (l_read + a.lambda_fp * l_fp) / a.batch_size
            if not bool(torch.isfinite(loss)): raise RuntimeError(f"non-finite loss step {step} item {q}")
            loss.backward(); any_grad = True; contrib.add(q)
            bl += float(loss.detach()); bl_fp += float(l_fp.detach()) / a.batch_size
            if bool(am.any()): ans_sum += float(ce.detach()[am].mean()); ans_n += 1
        if any_grad:
            torch.nn.utils.clip_grad_norm_(list(br.adapters.parameters()) + deq_params, 1.0); opt.step(); n_updates += 1
        bl_ans = ans_sum / ans_n if ans_n else float("nan")
        tel.append({"step": step, "loss": bl, "ans_ce": bl_ans, "l_fp": bl_fp, "res_mean": float(np.mean(res_all)) if res_all else None})
        if step % 25 == 0 or step == a.max_steps - 1:
            gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]
            print(f"[Step {step:4d}] loss {bl:.4f} ans_ce {bl_ans:.4f} l_fp {bl_fp:.5f} res {np.mean(res_all) if res_all else float('nan'):.4f} gates {gates} ({time.time()-t0:.0f}s)", flush=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    br.set_latent(None); br.detach()
    torch.save({"bridge_state": br.state_dict(), "layers": layers, "k_latent": 64, "d_model": 2560, "zfile": a.zstar, "seed": a.seed, "steps": a.max_steps, "x8a": True}, a.out_bridge)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in bb.state_dict().items() if any(t in k for t in TRAINABLE_KEYS)}, "k_unroll": a.k_unroll, "alpha": a.alpha, "g0": a.g0, "seed": a.seed, "steps": a.max_steps}, a.out_deq)
    json.dump({"items": a.items, "zstar": a.zstar, "zstar_sha256": sha256(a.zstar), "zsrc": a.zsrc, "zsrc_sha256": sha256(a.zsrc), "targets": a.targets, "targets_sha256": sha256(a.targets),
               "backbone_ckpt": a.backbone_ckpt, "backbone_sha256": sha256(a.backbone_ckpt), "seed": a.seed, "steps": a.max_steps, "batch_size": a.batch_size, "lr": a.lr, "lr_deq": a.lr_deq,
               "k_unroll": a.k_unroll, "alpha": a.alpha, "g0": a.g0, "lambda_fp": a.lambda_fp, "answer_weight": a.answer_weight, "gen_budget": a.gen_budget, "layers": layers,
               "deq_trainable_weights": int(sum(p.numel() for p in deq_params)), "bridge_params": br.n_params(), "n_items_trained": len(items), "target_counts": C, "targets_rows": n_rows,
               "n_updates": n_updates, "n_sample_visits": n_visits, "n_samples_skipped": n_skip, "n_unique_ids_contributing_grad": len(contrib),
               "final_loss_last200": float(np.mean([t["loss"] for t in tel[-200:]])), "final_ans_ce_last200": float(np.nanmean([t["ans_ce"] for t in tel[-200:]])),
               "final_lfp_last200": float(np.mean([t["l_fp"] for t in tel[-200:]])), "final_res_last200": float(np.mean([t["res_mean"] for t in tel[-200:] if t["res_mean"] is not None])),
               "telemetry": tel}, open(a.manifest, "w"), indent=1)
    print(f"[SAVE] {a.out_bridge} {a.out_deq} updates {n_updates} skipped {n_skip}", flush=True)


if __name__ == "__main__":
    main()
