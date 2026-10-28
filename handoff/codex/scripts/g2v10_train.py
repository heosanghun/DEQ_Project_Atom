#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/g2v10_train.py - X5 bridge training with STYLE-MATCHED targets + answer-span weighting (CC 2026-10-20, rev2).
Plan: shared/proposals/X_INTERFACE_PLAN_20261019.md ("X5 설계" + 10/20 정정). Same mechanism as X3-min rev2 (g2v9_xattn.LatentXAttnBridge,
frozen pure Gemma). Differences from g2v9_train.py:
  * targets are the pure model's OWN C-fmt generations on the training items (--targets jsonl: id, generation, tokens, stop_id),
    not the gold solutions, so the bridge has no new style to imitate (hypothesis under test: style-imitation harm);
  * the gold final answer is written into the target as \\boxed{gold}: the generation's last complete \\boxed{...} is replaced, or, if
    absent, "\\n\\nTherefore, the final answer is \\boxed{gold}." is appended; the tokens of the \\boxed{gold} segment get --answer_weight.
rev2 (after Codex REPORT-C003 D1-D5) + rev3 (REPORT-C003b R1-R4: supervised-only indexing, hard <=gen_budget guarantee, raw-row dup/empty counts, contributing-id count):
  D1 exact answer span: pre / box / post are tokenised SEPARATELY and concatenated, so the answer span is exact by construction
     (policy recorded in the manifest; boundary tokenisation may differ slightly from whole-string tokenisation).
  D2 validity: items with no retained target tokens, with the answer span truncated by --max_seq_len, or with no EOS are SKIPPED and
     counted; non-finite supervised loss raises.
  D3 generation budget: if the answer would end beyond --gen_budget tokens (eval max_new_tokens=512, margin kept), the OWN text before
     the box is truncated to the first tokens so that the whole target fits (counted: n_budget_truncated); own generations that hit the
     512 limit (stop_id null) are treated as unfinished and get the appended answer sentence.
  D4 integrity: duplicate/missing/empty target ids are counted, --items must be disjoint from --dev_order (dev 600), all counts + input
     SHA256 go to the manifest.
  D5 ans_ce is averaged over answer-present microbatches; optimizer.zero_grad(set_to_none=True); n_updates recorded.
Train items only. Saves bridge state + config (small)."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/shoon/cts_e1_s5"); sys.path.insert(0, "/home/shoon/cts_e1_s5/scripts")
from g2v3_m3_full_train import load_tokenizer, DEFAULT_PURE_GEMMA_SNAPSHOT
from g2v3_m3_eval_latents_v2 import extract_tensor
from g2v9_xattn import LatentXAttnBridge
from g2v9_train import build_prompt_ids, DATA
from cts.model.gemma_loader import load_gemma4_e4b

BOX = "\\boxed{"
CONNECTOR = "\n\nTherefore, the final answer is "


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def last_boxed_span(text: str):
    """(start, end) of the last COMPLETE \\boxed{...} in text (end exclusive, includes closing brace), or None."""
    i = text.rfind(BOX)
    while i != -1:
        j, depth = i + len(BOX), 1
        while j < len(text) and depth > 0:
            if text[j] == "{": depth += 1
            elif text[j] == "}": depth -= 1
            j += 1
        if depth == 0:
            return (i, j)
        i = text.rfind(BOX, 0, i)  # last one unterminated -> try the previous complete one
    return None


def gold_answer(solution: str):
    sp = last_boxed_span(solution)
    if sp is None:
        return None
    g = solution[sp[0] + len(BOX): sp[1] - 1].strip()
    return g or None


def build_target_text(own: str, gold: str, finished: bool):
    """Return (pre, box, post, mode). Only a FINISHED own generation with a complete box gets its box replaced in place;
    otherwise the answer sentence is appended after the (possibly truncated) own text."""
    box = BOX + gold + "}"
    sp = last_boxed_span(own) if finished else None
    if sp is not None:
        return own[:sp[0]], box, own[sp[1]:], "replaced"
    return own.rstrip() + CONNECTOR, box, ".", "appended"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zfile", required=True); ap.add_argument("--items", default="configs/s7_train_993.json")
    ap.add_argument("--dev_order", default="results_s7/p6_item_order.json", help="dev 600 order; training items must be disjoint")
    ap.add_argument("--targets", required=True, help="jsonl of the pure model's own generations on the training items (id, generation, tokens, stop_id)")
    ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--layers", default="10,20,30,40"); ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--text_drop", type=float, default=0.0); ap.add_argument("--latent_drop", type=float, default=0.0)
    ap.add_argument("--answer_weight", type=float, default=5.0); ap.add_argument("--seed", type=int, default=2047)
    ap.add_argument("--gen_budget", type=int, default=480, help="target (generated-token) length limit so the answer fits eval max_new_tokens=512")
    ap.add_argument("--max_seq_len", type=int, default=1536); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    j = json.load(open(a.items)); items_all = [int(x) for x in (j["indices"] if isinstance(j, dict) else j)]
    assert len(set(items_all)) == len(items_all), "duplicate ids in --items"
    jd = json.load(open(a.dev_order)); dev_ids = set(int(x) for x in (jd["indices"] if isinstance(jd, dict) else jd))
    assert not (set(items_all) & dev_ids), "training items overlap dev order"
    # --- targets: integrity (D4)
    own, seen_raw, n_dup, n_empty, n_rows = {}, set(), 0, 0, 0
    for l in open(a.targets, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l); q = int(r["id"]); n_rows += 1
        g = (r.get("generation") or "").strip()
        if not g:
            n_empty += 1  # counted on RAW rows, independent of duplicates
        if q in seen_raw:
            n_dup += 1; continue  # duplicate RAW row (any later row of an id); the FIRST row of an id is the only candidate
        seen_raw.add(q)
        if g:
            own[q] = (g, r.get("stop_id") is not None)
    n_missing = sum(1 for q in items_all if q not in own)
    assert n_rows - n_dup == len(seen_raw), "row accounting"
    tok = load_tokenizer()
    eos = tok.eos_token_id
    # --- build token targets (D1 exact span, D2 validity, D3 budget)
    T, C = {}, {"replaced": 0, "appended": 0, "skip_no_gold": 0, "skip_no_target": 0, "budget_truncated": 0, "skip_answer_truncated": 0, "finished_own": 0}
    for q in items_all:
        if q not in own:
            C["skip_no_target"] += 1; continue
        g = gold_answer(recs[q]["solution"])
        if g is None:
            C["skip_no_gold"] += 1; continue
        own_txt, finished = own[q]; C["finished_own"] += int(finished)
        pre, box, post, mode = build_target_text(own_txt, g, finished)
        ids_pre = tok.encode(pre, add_special_tokens=False)
        ids_box = tok.encode(box, add_special_tokens=False)
        ids_post = tok.encode(post, add_special_tokens=False) + [eos]
        if len(ids_pre) + len(ids_box) + len(ids_post) > a.gen_budget:  # D3: keep the answer inside the eval generation budget
            keep = a.gen_budget - len(ids_box) - len(ids_post)
            if mode == "replaced":  # fall back to appended form on the truncated own text
                pre_txt = tok.decode(ids_pre[:max(1, keep - 16)], skip_special_tokens=True).rstrip() + CONNECTOR
                ids_pre = tok.encode(pre_txt, add_special_tokens=False); mode = "appended"
                ids_post = tok.encode(".", add_special_tokens=False) + [eos]
                keep = a.gen_budget - len(ids_box) - len(ids_post)
                ids_pre = ids_pre[:max(1, keep)]
            else:
                conn = tok.encode(CONNECTOR, add_special_tokens=False)
                ids_pre = ids_pre[:max(1, keep - len(conn))] + conn  # final length re-checked below (R2)
            C["budget_truncated"] += 1
        s_ids = ids_pre + ids_box + ids_post
        if len(s_ids) > a.gen_budget or len(ids_box) + len(ids_post) + 1 > a.gen_budget:  # R2: hard guarantee; a box too long to fit is skipped
            C["skip_answer_truncated"] += 1; continue
        C[mode] += 1
        s_w = [1.0] * len(ids_pre) + [a.answer_weight] * len(ids_box) + [1.0] * len(ids_post)
        s_ans = [False] * len(ids_pre) + [True] * len(ids_box) + [False] * len(ids_post)
        T[q] = (s_ids, s_w, s_ans)
    items = [q for q in items_all if q in T]
    zd = torch.load(a.zfile, map_location="cpu")
    Z = {q: extract_tensor(zd, q).float().reshape(64, 2560) for q in items}
    print(f"[DATA] items {len(items)} of {len(items_all)} | targets rows dup {n_dup} empty {n_empty} missing {n_missing} | {C} | answer_weight {a.answer_weight} gen_budget {a.gen_budget}", flush=True)
    gem, _ = load_gemma4_e4b(model_id=DEFAULT_PURE_GEMMA_SNAPSHOT, device_map=a.device, torch_dtype=torch.bfloat16, offload_vision_audio=True)
    gem.eval()
    for p in gem.parameters():
        p.requires_grad = False
    lm, head, emb = gem.model.language_model, gem.lm_head, gem.model.language_model.embed_tokens
    layers = [int(x) for x in a.layers.split(",")]
    n_layers = len(lm.layers); assert max(layers) < n_layers, (layers, n_layers)
    br = LatentXAttnBridge(layers).to(dev).float()
    with torch.no_grad():
        br.center.copy_(torch.stack([Z[q] for q in items]).mean(0).to(dev))
    br.attach(lm); br.train()
    opt = torch.optim.AdamW(br.adapters.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"[BRIDGE] layers {layers} of {n_layers}, params {br.n_params():,}, center norm/slot {float(br.center.norm(dim=-1).mean()):.1f}", flush=True)
    tel, ptr, n_drop, n_ldrop, n_updates, n_seq_trunc, n_ans_cut, t0 = [], 0, 0, 0, 0, 0, 0, time.time()
    contrib_ids, n_visits = set(), 0
    for step in range(a.max_steps):
        opt.zero_grad(set_to_none=True); bl = 0.0; ans_sum = 0.0; ans_n = 0; any_grad = False
        for _ in range(a.batch_size):
            q = items[ptr % len(items)]; ptr += 1; r = recs[q]; n_visits += 1
            drop = rng.random() < a.text_drop; n_drop += int(drop)
            p_ids = build_prompt_ids(tok, "" if drop else r["prompt"])
            s_ids, s_w, s_ans = T[q]
            full_len = len(p_ids) + len(s_ids)
            ids = (p_ids + s_ids)[: a.max_seq_len]
            wts = ([0.0] * len(p_ids) + s_w)[: a.max_seq_len]
            ans = ([False] * len(p_ids) + s_ans)[: a.max_seq_len]
            if full_len > a.max_seq_len:
                n_seq_trunc += 1
                if sum(ans) < sum(s_ans) or ids[-1] != eos:  # D2: answer or EOS lost -> skip this sample
                    n_ans_cut += 1; continue
            if not any(w > 0 for w in wts[1:]):
                n_ans_cut += 1; continue
            x = torch.tensor(ids, device=dev).unsqueeze(0)
            w = torch.tensor(wts[1:], device=dev, dtype=torch.float32)  # weight of predicting token t+1
            am = torch.tensor(ans[1:], device=dev)
            ldrop = rng.random() < a.latent_drop; n_ldrop += int(ldrop)
            br.set_latent(None if ldrop else Z[q].to(dev))
            pli = lm.get_per_layer_inputs(x, None) if hasattr(lm, "get_per_layer_inputs") else None
            out = lm(inputs_embeds=emb(x), return_dict=True, **({"per_layer_inputs": pli} if pli is not None else {}))
            lg = head(out.last_hidden_state[:, :-1, :]).float()
            ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
            sup = w > 0
            if not bool(torch.isfinite(ce[sup]).all()):
                raise RuntimeError(f"non-finite supervised loss at step {step} item {q}")
            loss = (ce[sup] * w[sup]).sum() / w[sup].sum() / a.batch_size  # R1: only supervised positions enter the scalar
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite weighted loss at step {step} item {q}")
            if loss.requires_grad:
                loss.backward(); any_grad = True; contrib_ids.add(q)
            bl += float(loss.detach())
            if bool(am.any()):
                ans_sum += float(ce.detach()[am].mean()); ans_n += 1
        if any_grad:
            torch.nn.utils.clip_grad_norm_(br.adapters.parameters(), 1.0); opt.step(); n_updates += 1
        bl_ans = ans_sum / ans_n if ans_n else float("nan")
        tel.append({"step": step, "loss": bl, "ans_ce": bl_ans, "ans_n": ans_n})
        if step % 50 == 0 or step == a.max_steps - 1:
            gates = [round(float(torch.tanh(ad.gate)), 3) for ad in br.adapters.values()]
            print(f"[Step {step:4d}] loss {bl:.4f} ans_ce {bl_ans:.4f} (n {ans_n}) gates {gates} ({time.time()-t0:.0f}s)", flush=True)
    br.set_latent(None); br.detach()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"bridge_state": br.state_dict(), "layers": layers, "k_latent": 64, "d_model": 2560, "zfile": a.zfile, "seed": a.seed, "steps": a.max_steps,
                "targets": a.targets, "answer_weight": a.answer_weight, "gen_budget": a.gen_budget}, a.out)
    ans_vals = [t["ans_ce"] for t in tel[-200:] if t["ans_n"]]
    json.dump({"zfile": a.zfile, "zfile_sha256": sha256(a.zfile), "targets": a.targets, "targets_sha256": sha256(a.targets), "items": a.items, "items_sha256": sha256(a.items),
               "dev_order": a.dev_order, "dev_order_sha256": sha256(a.dev_order), "tokenizer": getattr(tok, "name_or_path", str(type(tok).__name__)), "eos_token_id": eos,
               "seed": a.seed, "steps": a.max_steps, "batch_size": a.batch_size, "lr": a.lr, "text_drop": a.text_drop, "n_text_dropped": n_drop,
               "latent_drop": a.latent_drop, "n_latent_dropped": n_ldrop, "answer_weight": a.answer_weight, "gen_budget": a.gen_budget, "max_seq_len": a.max_seq_len,
               "answer_span_policy": "pre/box/post tokenised separately and concatenated (exact span)",
               "n_items_trained": len(items), "n_items_listed": len(items_all), "targets_dup_rows": n_dup, "targets_empty_rows": n_empty, "targets_missing_ids": n_missing,
               "target_counts": C, "targets_rows": n_rows, "count_policy": "dup/empty counted on raw rows; first row of an id is the only candidate", "n_updates": n_updates, "n_sample_visits": n_visits, "n_seq_truncated": n_seq_trunc, "n_samples_skipped_answer_or_eos_cut": n_ans_cut, "n_unique_ids_contributing_grad": len(contrib_ids),
               "layers": layers, "params": br.n_params(), "final_loss_last200": float(np.mean([t["loss"] for t in tel[-200:]])),
               "final_ans_ce_last200": float(np.mean(ans_vals)) if ans_vals else None, "telemetry": tel}, open(a.manifest, "w"), indent=1)
    print(f"[SAVE] {a.out} ({os.path.getsize(a.out):,} bytes) updates {n_updates} seq_trunc {n_seq_trunc} skipped {n_ans_cut}", flush=True)


if __name__ == "__main__":
    main()
