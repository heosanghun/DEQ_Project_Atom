#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/g2v9_train.py - X3-min bridge training (CC 2026-10-19). Frozen pure Gemma-4-e4b (bf16) + LatentXAttnBridge (fp32).
Prompt = pure C-fmt chat prompt (problem text + FMT_INSTR, no placeholder slots); with prob. --text_drop the problem text is dropped
(instruction kept). Loss = CE on gold solution tokens. Latents are precomputed per item (--zfile, keys q{idx:04d}[_a0.25] / fields z_star or z_src);
center = per-slot mean of the training latents. Train items only (configs/s7_train_993.json). Saves bridge state + center + config (small)."""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from g2v3_m3_eval_latents_v2 import extract_tensor
from g2v9_xattn import LatentXAttnBridge
from cts.model.gemma_loader import load_gemma4_e4b

FMT_INSTR = "Put your final answer within \\boxed{}."
DATA = "/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl"


def build_prompt_ids(tok, prob_text: str):
    content = (prob_text + "\n\n" + FMT_INSTR) if prob_text else FMT_INSTR
    text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zfile", required=True); ap.add_argument("--items", default="configs/s7_train_993.json")
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--layers", default="10,20,30,40"); ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--text_drop", type=float, default=0.5); ap.add_argument("--seed", type=int, default=2047)
    ap.add_argument("--latent_drop", type=float, default=0.0)  # rev3: prob. of training with NO latent (bridge identity) so 'no latent' stays pure Gemma
    ap.add_argument("--max_seq_len", type=int, default=1536); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    j = json.load(open(a.items)); items = [int(x) for x in (j["indices"] if isinstance(j, dict) else j)]
    zd = torch.load(a.zfile, map_location="cpu")
    Z = {q: extract_tensor(zd, q).float().reshape(64, 2560) for q in items}
    print(f"[DATA] items {len(items)} latents {len(Z)}", flush=True)
    gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True)
    gem.eval()
    for p in gem.parameters():
        p.requires_grad = False
    lm, head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
    tok = load_tokenizer()
    layers = [int(x) for x in a.layers.split(",")]
    n_layers = len(lm.layers); assert max(layers) < n_layers, (layers, n_layers)
    br = LatentXAttnBridge(layers).to(dev).float()
    with torch.no_grad():
        br.center.copy_(torch.stack([Z[q] for q in items]).mean(0).to(dev))
    br.attach(lm); br.train()
    opt = torch.optim.AdamW(br.adapters.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"[BRIDGE] layers {layers} of {n_layers}, params {br.n_params():,}, center norm/slot {float(br.center.norm(dim=-1).mean()):.1f}", flush=True)
    tel, ptr, n_drop, n_ldrop, t0 = [], 0, 0, 0, time.time()
    for step in range(a.max_steps):
        opt.zero_grad(); bl = 0.0
        for _ in range(a.batch_size):
            q = items[ptr % len(items)]; ptr += 1; r = recs[q]
            drop = rng.random() < a.text_drop; n_drop += int(drop)
            p_ids = build_prompt_ids(tok, "" if drop else r["prompt"])
            s_ids = tok.encode(r["solution"], add_special_tokens=False)
            if not s_ids or s_ids[-1] != tok.eos_token_id:
                s_ids.append(tok.eos_token_id)
            ids = (p_ids + s_ids)[: a.max_seq_len]
            x = torch.tensor(ids, device=dev).unsqueeze(0); tgt = x.clone(); tgt[:, :len(p_ids)] = -100
            ldrop = rng.random() < a.latent_drop; n_ldrop += int(ldrop)
            br.set_latent(None if ldrop else Z[q].to(dev))
            pli = lm.get_per_layer_inputs(x, None) if hasattr(lm, "get_per_layer_inputs") else None
            out = lm(inputs_embeds=emb(x), return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
            lg = head(out.last_hidden_state[:, :-1, :]).float()
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=-100) / a.batch_size
            if loss.requires_grad:  # rev3 fix: with --latent_drop the bridge is identity on dropped samples -> no grad path (LM frozen); skip backward
                loss.backward()
            bl += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(br.adapters.parameters(), 1.0); opt.step()
        tel.append({"step": step, "loss": bl})
        if step % 50 == 0 or step == a.max_steps - 1:
            gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]
            print(f"[Step {step:4d}] loss {bl:.4f} gates {gates} ({time.time()-t0:.0f}s)", flush=True)
    br.set_latent(None); br.detach()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"bridge_state": br.state_dict(), "layers": layers, "k_latent": 64, "d_model": 2560, "zfile": a.zfile, "seed": a.seed, "steps": a.max_steps}, a.out)
    json.dump({"zfile": a.zfile, "seed": a.seed, "steps": a.max_steps, "batch_size": a.batch_size, "lr": a.lr, "text_drop": a.text_drop,
               "n_text_dropped": n_drop, "latent_drop": a.latent_drop, "n_latent_dropped": n_ldrop, "layers": layers, "params": br.n_params(), "final_loss_last200": float(np.mean([t["loss"] for t in tel[-200:]])),
               "telemetry": tel}, open(a.manifest, "w"), indent=1)
    print(f"[SAVE] {a.out} ({os.path.getsize(a.out):,} bytes)", flush=True)


if __name__ == "__main__":
    main()
