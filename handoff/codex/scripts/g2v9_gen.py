#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/g2v9_gen.py - X3-min greedy generation with the latent bridge (CC 2026-10-19).
--cond matched : item's own latent; shuffled : latent of the next item in the order (shift 1); none : no injection (== C-fmt).
Prompt = pure C-fmt prompt (problem + FMT_INSTR). Greedy, KV cache, max 512 new tokens, stop ids (1, 106) - same as v36g.
Output jsonl rows: id, generation, tokens, stop_id (grader-compatible)."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import torch
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from g2v3_m3_eval_latents_v2 import extract_tensor
from g2v9_xattn import LatentXAttnBridge
from g2v9_train import build_prompt_ids, DATA
from cts.model.gemma_loader import load_gemma4_e4b
EOS = {1, 106}


def greedy(lm, head, emb, ids, max_new=512):
    x = torch.tensor(ids, device=emb.weight.device).unsqueeze(0)
    pli = lm.get_per_layer_inputs(x, None) if hasattr(lm, "get_per_layer_inputs") else None
    out = lm(inputs_embeds=emb(x), use_cache=True, return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
    nxt = head(out.last_hidden_state[:, -1:, :]).argmax(-1); gen = [int(nxt)]
    past = out.past_key_values
    while gen[-1] not in EOS and len(gen) < max_new:
        tp = lm.get_per_layer_inputs(nxt, None) if hasattr(lm, "get_per_layer_inputs") else None
        out = lm(inputs_embeds=emb(nxt), past_key_values=past, use_cache=True, return_dict=True, **({"per_layer_inputs": tp} if tp is not None else {}))
        nxt = head(out.last_hidden_state[:, -1:, :]).argmax(-1); gen.append(int(nxt)); past = out.past_key_values
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--zfile", required=True)
    ap.add_argument("--cond", choices=["matched", "shuffled", "none"], required=True)
    ap.add_argument("--items", default="results_s7/p6_item_order.json"); ap.add_argument("--order_sha", default="6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c")
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shift", type=int, default=1)
    a = ap.parse_args()
    assert hashlib.sha256(open(a.items, "rb").read()).hexdigest() == a.order_sha, "order SHA mismatch"
    j = json.load(open(a.items)); order = [int(x) for x in (j["indices"] if isinstance(j, dict) else j)]
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    ck = torch.load(a.ckpt, map_location="cpu")
    zd = torch.load(a.zfile, map_location="cpu")
    gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True); gem.eval()
    for p in gem.parameters():
        p.requires_grad = False
    lm, head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
    tok = load_tokenizer()
    br = LatentXAttnBridge(ck["layers"]).to(a.device).float(); br.load_state_dict(ck["bridge_state"], strict=True); br.eval()
    if a.cond != "none":
        br.attach(lm)
    print(f"[BRIDGE-LOADED] layers {ck['layers']} cond {a.cond} gates {[round(float(torch.tanh(ad.gate)),3) for ad in br.adapters.values()]}", flush=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try: done.add(int(json.loads(l)["id"]))
            except Exception: pass
    f = open(a.out, "a", encoding="utf-8"); n = len(order); t0 = time.time(); k = 0
    with torch.inference_mode():
        for i, q in enumerate(order):
            if q in done: continue
            zq = order[(i + a.shift) % n] if a.cond == "shuffled" else q
            br.set_latent(None if a.cond == "none" else extract_tensor(zd, zq).float().reshape(64, 2560).to(a.device))
            ids = build_prompt_ids(tok, recs[q]["prompt"])
            g = greedy(lm, head, emb, ids)
            txt = tok.decode(g, skip_special_tokens=True).strip()
            f.write(json.dumps({"id": q, "latent_id": zq, "generation": txt, "tokens": len(g), "stop_id": g[-1] if g[-1] in EOS else None}, ensure_ascii=False) + "\n"); f.flush(); k += 1
            if k % 50 == 0: print(f"[{k}/{n-len(done)}] {time.time()-t0:.0f}s", flush=True)
    f.close(); br.detach()
    print(f"[DONE] {a.out} rows_new {k} sha {hashlib.sha256(open(a.out,'rb').read()).hexdigest()[:16]}", flush=True)


if __name__ == "__main__":
    main()
