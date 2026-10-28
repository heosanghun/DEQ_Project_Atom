#!/usr/bin/env python3
# scripts/g2v4_path_probe.py - CC 2026-10-11 diagnostic: where does item information vanish on the latent -> LM path?
# P1: projector output item-specific energy share (dev latents) vs input; norm vs real token embeddings.
# P2: slot readability: pure LM, problem text REMOVED, 64 slots filled with (a) placeholder '1' embeddings, (b) the problem's own first-64 token embeddings,
#     (c) the next item's token embeddings. If (b) << (a),(c) the slots are readable by the LM.
import json, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import PrefixProjector, prepare_item_batch, load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from g2v3_m3_eval_latents_v2 import extract_tensor
from cts.model.gemma_loader import load_gemma4_e4b
dev = torch.device("cuda:0")
qs = [int(x) for x in json.load(open("results_s7/p6_item_order.json"))["indices"]][:120]
recs = [json.loads(l) for l in open("/home/shoon/cts_e1_s5/data/stage2/math_train_prompts_5000.jsonl", encoding="utf-8") if l.strip()]
zs = torch.load("results_s7/zsrc_heldout600.pt", map_location="cpu")
Z = torch.stack([extract_tensor(zs, q).float().reshape(64, 2560) for q in qs]).to(dev)
gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map="cuda:0", torch_dtype=torch.bfloat16, offload_vision_audio=True); gem.eval()
lm, head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
tok = load_tokenizer()
def share(X):
    X = X.double(); mu = X.mean(0, keepdim=True); return float((X - mu).pow(2).sum() / X.pow(2).sum())
with torch.no_grad():
    te = emb(torch.tensor(tok.encode(recs[qs[0]]["prompt"], add_special_tokens=False)[:64], device=dev).unsqueeze(0)).float()
    print(f"[P1] input z_src: item share {share(Z):.3f}, mean slot norm {float(Z.norm(dim=-1).mean()):.1f} | real token embedding mean norm {float(te.norm(dim=-1).mean()):.2f}")
    for name, ck in (("M3_E", "results_s7/g2v2/decoder_Em3_v36.pt"), ("cap_E", "results_s7/g2v4cap/decoder_E.pt"), ("lora_E", "results_s7/g2v4lora/decoder_E.pt")):
        pr = PrefixProjector(2560, 2560, True).to(dev).float(); d = torch.load(ck, map_location="cpu"); pr.load_state_dict(d.get("state_dict", d), strict=True); pr.eval()
        Y = pr(Z); W = pr.w_proj(Z)
        print(f"[P1] {name}: after w_proj item share {share(W):.3f} norm {float(W.norm(dim=-1).mean()):.1f} | after adapter item share {share(Y):.3f} norm {float(Y.norm(dim=-1).mean()):.1f} | sigma(w_proj) {pr.get_w_proj_spectral_norm():.3f}")
    def nll(ids, tgt, s0, fill):
        e = emb(ids); e[:, s0:s0 + 64, :] = fill.to(e.dtype)
        pli = lm.get_per_layer_inputs(ids, None) if hasattr(lm, "get_per_layer_inputs") else None
        out = lm(inputs_embeds=e, return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
        lg = head(out.last_hidden_state[:, :-1, :]); return float(F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), tgt[:, 1:].reshape(-1), ignore_index=-100))
    def textemb(q):
        t = tok.encode(recs[q]["prompt"], add_special_tokens=False)[:64]; t = t + [tok.encode("1", add_special_tokens=False)[0]] * (64 - len(t))
        return emb(torch.tensor(t, device=dev).unsqueeze(0))
    A = B = C = 0.0; n = 60
    for i, q in enumerate(qs[:n]):
        ids, tgt, s0, s1 = prepare_item_batch(tok, "", recs[q]["solution"]); ids, tgt = ids.unsqueeze(0).to(dev), tgt.unsqueeze(0).to(dev)
        A += nll(ids, tgt, s0, emb(ids)[:, s0:s0 + 64, :]); B += nll(ids, tgt, s0, textemb(q)); C += nll(ids, tgt, s0, textemb(qs[(i + 1) % n]))
    print(f"[P2] no text, slots = placeholder {A/n:.4f} | own problem token embeddings {B/n:.4f} | next problem token embeddings {C/n:.4f}")
