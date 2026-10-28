#!/usr/bin/env python3
"""scripts/g2v5_forensic.py - seed-2047 forensic (CC 2026-10-13, CTO CTS-433). Reads EXISTING checkpoints/latents only; no parameter update.
Per model: (1) training telemetry (l_task early/late, spikes, l_ift), (2) projector weights (out_rms, ||w_proj - I||/||I||, adapter norms, center norm),
(3) DEQ LoRA learned size (sum ||lora_B||_F, gamma), (4) projector output geometry on its own dev600 latents (item-specific energy share),
(5) post-hoc gradients at the final checkpoint on 40 dev items (teacher forcing, problem text included): ||dL/d projector params||, ||dL/dz|| for the
matched latent, and the share of dL/dz that is item-specific. Prints one JSON line per model."""
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import prepare_item_batch, load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from g2v3_m3_eval_latents_v2 import extract_tensor
from g2v4_proj_v2 import PrefixProjectorV2
from cts.model.gemma_loader import load_gemma4_e4b
W, R = "/tmp/cts_r1b", "results_s7/g2v4m7rep"
def paths(t):
    if t in ("Ah", "Eh"):
        return f"{R}/model_{t}.pt", f"{R}/decoder_{t}.pt", f"{R}/dev600_latent_{'A' if t == 'Ah' else 'E'}.pt"
    return f"{W}/model_{t}.pt", f"{W}/decoder_{t}.pt", f"{W}/lat_{t}.pt"
tags = sys.argv[1].split(",")
dev = torch.device("cuda:0")
qs = [int(x) for x in json.load(open("results_s7/p6_item_order.json"))["indices"]]
recs = [json.loads(l) for l in open("/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl", encoding="utf-8") if l.strip()]
gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map="cuda:0", torch_dtype=torch.bfloat16, offload_vision_audio=True); gem.eval()
for p in gem.parameters(): p.requires_grad = False
lm, head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
tok = load_tokenizer()
def share(X):
    X = X.double().reshape(X.shape[0], -1); mu = X.mean(0, keepdim=True); return float((X - mu).pow(2).sum() / X.pow(2).sum())
for t in tags:
    mp, dp, lp = paths(t)
    m = torch.load(mp, map_location="cpu", weights_only=False)
    tel = m.get("telemetry", [])
    lt = [x["l_task"] for x in tel]; li = [x["l_ift"] for x in tel]
    med = sorted(lt)[len(lt) // 2] if lt else 0
    out = {"tag": t, "steps": len(lt),
           "l_task_first200": sum(lt[:200]) / max(1, len(lt[:200])), "l_task_last200": sum(lt[-200:]) / max(1, len(lt[-200:])),
           "l_task_spikes_gt2xmed": sum(x > 2 * med for x in lt), "l_ift_last200": sum(li[-200:]) / max(1, len(li[-200:]))}
    dq = m.get("deq_lora_state_dict", {})
    out["deq_loraB_fro_sum"] = float(sum(v.float().norm() for k, v in dq.items() if "lora_B" in k))
    g = [float(v.float().reshape(-1)[0]) for k, v in dq.items() if "gamma" in k]
    out["deq_gamma"] = g[:3]
    d = torch.load(dp, map_location="cpu", weights_only=False); sd = d.get("state_dict", d)
    proj = PrefixProjectorV2(d_model=2560).to(dev).float(); proj.load_state_dict(sd, strict=True); proj.eval()
    I = torch.eye(2560, device=dev)
    out["out_rms"] = float(proj.out_rms); out["w_proj_dev_from_I"] = float((proj.w_proj.weight - I).norm() / I.norm())
    out["adapter_fc1_fro"] = float(proj.adapter.fc1.weight.norm()); out["adapter_fc2_fro"] = float(proj.adapter.fc2.weight.norm())
    out["center_norm_per_slot"] = float(proj.center.norm(dim=-1).mean())
    L = torch.load(lp, map_location="cpu", weights_only=False)
    Z = torch.stack([extract_tensor(L, q).float().reshape(64, 2560) for q in qs]).to(dev)
    with torch.no_grad():
        Y = torch.cat([proj(Z[i:i + 50]) for i in range(0, 600, 50)])
    out["proj_in_item_share"] = share(Z - proj.center); out["proj_out_item_share"] = share(Y)
    # gradients on 40 dev items
    gp, gz, gz_item = 0.0, [], []
    proj.train(False)
    for p in proj.parameters(): p.requires_grad = True
    proj.zero_grad()
    GZ = []
    for q in qs[:40]:
        r = recs[q]; ids, tgt, s0, s1 = prepare_item_batch(tok, r["prompt"], r["solution"]); ids, tgt = ids.unsqueeze(0).to(dev), tgt.unsqueeze(0).to(dev)
        z = extract_tensor(L, q).float().reshape(1, 64, 2560).to(dev).requires_grad_(True)
        e = emb(ids).detach().clone(); e[:, s0:s1 + 1, :] = proj(z).to(e.dtype)
        pli = lm.get_per_layer_inputs(ids, None) if hasattr(lm, "get_per_layer_inputs") else None
        o = lm(inputs_embeds=e, return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
        lg = head(o.last_hidden_state[:, :-1, :]).float()
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=-100) / 40
        loss.backward()
        GZ.append(z.grad.detach().reshape(-1).double().cpu() * 40)
    out["grad_proj_fro"] = float(sum(p.grad.float().norm() ** 2 for p in proj.parameters() if p.grad is not None) ** 0.5)
    G = torch.stack(GZ); out["grad_z_norm_mean"] = float(G.norm(dim=1).mean())
    mu = G.mean(0, keepdim=True); out["grad_z_item_share"] = float((G - mu).pow(2).sum() / G.pow(2).sum())
    print("[FOR] " + json.dumps(out), flush=True)
    del m, d, L, Z, Y, proj; torch.cuda.empty_cache()
