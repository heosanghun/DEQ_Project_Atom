#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/g2v5_diag.py - copy of g2v4_diag_iterate.py (85b4e94f) for R1x (CC 2026-10-12, CTO CTS-432): both start points (z*'0 and z_src) for BOTH modes,
plus zero-latent and no-injection NLL. Original doc: G2 4th-attempt root-cause diagnostic (CC, 2026-10-11). NOT a gate; dev600 items only (already seen).

Questions:
  Q1 (item specificity): does the trained decoder use the item's own latent? NLL(own latent) vs NLL(next item's latent) vs NLL(mean latent).
  Q2 (iteration): M3 applied the learned map Phi_theta ONCE. If Phi_theta is iterated k times (z_{k+1} = (1-a) z_k + a (Phi_theta(z_k) + g0 z_src)),
      does solution NLL improve with k, and does the start point (M2' fixed point vs z_src) still matter after iteration?

Output: per-item JSON + summary print. Teacher-forced solution NLL (mean CE over solution tokens), same prompt format as training (v3.6 §3).
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/shoon/cts_e1_s5")
sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import PrefixProjector, prepare_item_batch, load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT, DEFAULT_FROZEN_BACKBONE_CKPT  # noqa: E402
from g2v3_m3_eval_latents_v2 import extract_tensor  # noqa: E402

KS = (1, 2, 4, 8, 16)  # overridable by --ks
ALPHA, G0 = 0.25, 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "E"], required=True)
    ap.add_argument("--items", default="results_s7/p6_item_order.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--deq_ckpt", required=True)
    ap.add_argument("--decoder_ckpt", required=True)
    ap.add_argument("--zstar", default="results_s7/g2v2/zstar_zsrcinj_dev600.pt")
    ap.add_argument("--zsrc", default="results_s7/zsrc_heldout600.pt")
    ap.add_argument("--dataset", default="/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--notext", action="store_true", help="also NLL with problem text removed (own / next item latent)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    global KS
    KS = tuple(int(x) for x in a.ks.split(","))

    from cts.backbone.gemma_adapter import GemmaCTSBackbone
    from cts.model.gemma_loader import load_gemma4_e4b
    from cts.train.lora_compat import apply_paper_lora
    from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

    _j = json.load(open(a.items)); q_all = [int(x) for x in (_j["indices"] if isinstance(_j, dict) else _j)]
    qs = q_all[: a.n]
    recs = {}
    for idx, line in enumerate(open(a.dataset, encoding="utf-8")):
        if line.strip():
            it = json.loads(line)
            recs[int(it.get("question_idx", it.get("idx", it.get("q_idx", idx))))] = it
    zstar_d = torch.load(a.zstar, map_location="cpu")
    zsrc_d = torch.load(a.zsrc, map_location="cpu")

    gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True)
    gem.eval()
    for p in gem.parameters():
        p.requires_grad = False
    lm, lm_head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
    tok = load_tokenizer()
    dsd = torch.load(a.decoder_ckpt, map_location="cpu")
    if isinstance(dsd, dict) and dsd.get("lm_lora_rank", 0) > 0:
        from cts.train.lora_compat import replace_target_linears
        r_ = int(dsd["lm_lora_rank"])
        replace_target_linears(lm, ("q_proj", "v_proj", "o_proj"), rank=r_, alpha=2 * r_, dropout=0.0)
        for n_, m_ in lm.named_modules():
            if n_.endswith("lora_A") or n_.endswith("lora_B"):
                m_.to(device=dev, dtype=torch.float32)
        res_ = lm.load_state_dict(dsd["lm_lora_state_dict"], strict=False)
        miss = [k for k in res_.missing_keys if "lora_" in k]
        if res_.unexpected_keys or miss:
            raise RuntimeError(f"[FATAL] LM LoRA load: unexpected {len(res_.unexpected_keys)} missing-lora {len(miss)}")
        print(f"[LM-LORA] loaded rank {r_}, {len(dsd['lm_lora_state_dict'])} tensors", flush=True)

    m_base, tok_base = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True)
    bb = GemmaCTSBackbone(m_base, tok_base)
    apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0, require_match=False, verbose=False)
    sd = torch.load(DEFAULT_FROZEN_BACKBONE_CKPT, map_location="cpu")
    sd = sd.get("backbone_state_dict", sd)
    bb.load_state_dict({("routing_proj.parametrizations.weight.original" if k == "routing_proj" else k): v for k, v in sd.items()}, strict=True)
    deq = torch.load(a.deq_ckpt, map_location="cpu")
    res = bb.load_state_dict(deq.get("state_dict", deq), strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"[FATAL] unexpected DEQ keys n={len(res.unexpected_keys)}")
    print(f"[LOAD-GUARD] applied {len(deq.get('state_dict', deq))} DEQ tensors, unexpected 0", flush=True)
    bb.to(device=dev, dtype=torch.bfloat16).eval()
    for p in bb.parameters():
        p.requires_grad = False
    w_g = bb.routing_matrix().to(device=dev, dtype=torch.float32)

    if isinstance(dsd, dict) and dsd.get("proj_version", 1) == 2:
        from g2v4_proj_v2 import PrefixProjectorV2
        proj = PrefixProjectorV2(d_model=2560).to(device=dev, dtype=torch.float32)
        print("[PROJ-V2] decoder uses PrefixProjectorV2", flush=True)
    else:
        proj = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device=dev, dtype=torch.float32)
    proj.load_state_dict(dsd.get("state_dict", dsd), strict=True)
    proj.eval()

    def step(z, zr, ctx):
        fw = sparse_module_weights(routing_weights(z.float(), w_g, 1.0), 3)
        sel = torch.sort(torch.topk(fw.reshape(-1), 3).indices.to(dtype=torch.long)).values
        ids = [int(x) for x in sel.cpu().tolist()]
        ex = {"top_k": 3, "deq_map_mode": bb.deq_map_mode, "routing_module_ids": ids, "routing_module_ids_tensor": sel}
        hook = getattr(bb, "prepare_deq_context", None)
        if callable(hook):
            ex["eq3_static_inputs"] = hook(z, ctx, ids)
        phi = bb.deq_step(z, ctx, fw, ex).float()
        return (1.0 - ALPHA) * z + ALPHA * (phi + G0 * zr)

    def nll(q, z, text=True):
        r = recs[q]
        ids, tgt, s0, s1 = prepare_item_batch(tok, r.get("prompt", r.get("problem", "")) if text else "", r["solution"])
        ids, tgt = ids.unsqueeze(0).to(dev), tgt.unsqueeze(0).to(dev)
        e = emb(ids)
        if z is not None:  # z None = no-injection (placeholder token embeddings stay in the 64 slots)
            e[:, s0:s1 + 1, :] = proj(z).to(dtype=e.dtype)
        pli = lm.get_per_layer_inputs(ids, None) if hasattr(lm, "get_per_layer_inputs") else None
        out = lm(inputs_embeds=e, return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
        lg = lm_head(out.last_hidden_state[:, :-1, :])
        return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), tgt[:, 1:].reshape(-1), ignore_index=-100))

    def get(d, q):
        t = extract_tensor(d, q)
        if t is None:
            raise RuntimeError(f"[FATAL] missing latent q{q}")
        t = t.to(device=dev, dtype=torch.float32)
        return t.unsqueeze(0) if t.dim() == 2 else t

    starts = ["zstar", "zsrc"] if a.mode == "A" else ["zsrc", "zstar"]  # first = the matched (trained) condition
    rows, lat1 = [], {}
    t0 = time.time()
    with torch.no_grad():
        for i, q in enumerate(qs):
            r = recs[q]
            ctx = bb.encode_context(r.get("prompt", r.get("problem", "")))
            ctx = ctx.to(device=dev) if hasattr(ctx, "to") else ctx
            zr = get(zsrc_d, q)
            row = {"q": q}
            for s in starts:
                z = get(zstar_d, q) if s == "zstar" else zr
                for k in range(1, max(KS) + 1):
                    zn = step(z, zr, ctx)
                    rk = float((zn - z).double().norm() / z.double().norm().clamp_min(1e-12))
                    z = zn
                    if k in KS:
                        row[f"{s}_k{k}_nll"] = nll(q, z)
                        row[f"{s}_k{k}_res"] = rk
                    if k == 1 and s == starts[0]:
                        lat1[q] = z.detach()
            rows.append(row)
            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(qs)}] {time.time()-t0:.0f}s", flush=True)
        # Q1: own vs next item's latent vs mean latent (k=1, first start = the condition actually evaluated)
        mean_lat = torch.stack([lat1[q] for q in qs]).mean(0)
        for i, q in enumerate(qs):
            rows[i]["shift1_nll"] = nll(q, lat1[qs[(i + 1) % len(qs)]])
            rows[i]["mean_nll"] = nll(q, mean_lat)
            rows[i]["zero_nll"] = nll(q, torch.zeros_like(lat1[q]))
            rows[i]["noinj_nll"] = nll(q, None)
            if a.notext:
                rows[i]["notext_own_nll"] = nll(q, lat1[q], text=False)
                rows[i]["notext_shift1_nll"] = nll(q, lat1[qs[(i + 1) % len(qs)]], text=False)

    keys = [k for k in rows[0] if k != "q"]
    summ = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    json.dump({"mode": a.mode, "n": len(rows), "summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    for k in keys:
        print(f"[SUM] {a.mode} {k} {summ[k]:.4f}")
    print(f"[DONE] {time.time()-t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
