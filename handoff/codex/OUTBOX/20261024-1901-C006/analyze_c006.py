import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\AI\cts")
sys.path.insert(0, str(ROOT / "scripts"))
import s5_eval_grader_v2 as grader

INBOX = ROOT / "handoff/codex/INBOX/20261021-0945-x7-results"
OUTBOX = ROOT / "handoff/codex/OUTBOX/20261024-1901-C006"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 1. SHA256 verification
sha_lines = (INBOX / "SHA256SUMS_full.txt").read_text(encoding="utf-8").splitlines()
sha_checks = {}
for line in sha_lines:
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if len(parts) == 2:
        exp_sha, raw_name = parts[0], parts[1].lstrip("*")
        p = INBOX / raw_name
        if p.exists():
            act_sha = sha256_file(p)
            sha_checks[raw_name] = {
                "expected": exp_sha,
                "actual": act_sha,
                "exists": True,
                "match": act_sha == exp_sha,
            }
        else:
            sha_checks[raw_name] = {
                "expected": exp_sha,
                "actual": None,
                "exists": False,
                "match": None,
            }

grader_sha_scripts = sha256_file(ROOT / "scripts/s5_eval_grader_v2.py")
grader_sha_codex = sha256_file(ROOT / "handoff/codex/scripts/s5_eval_grader_v2.py")
wrapper_sha = sha256_file(ROOT / "handoff/codex/scripts/codex_grade_local.py")
dev_order_path = ROOT / "handoff/codex/data/p6_item_order.json"
dev_order_sha = sha256_file(dev_order_path)
train_items_path = ROOT / "configs/s7_train_993.json"
train_items_sha = sha256_file(train_items_path)

dev_order_obj = json.loads(dev_order_path.read_text(encoding="utf-8"))
dev_ids = dev_order_obj["indices"] if isinstance(dev_order_obj, dict) else dev_order_obj
train_obj = json.loads(train_items_path.read_text(encoding="utf-8"))
train_ids = train_obj["indices"] if isinstance(train_obj, dict) and "indices" in train_obj else (
    train_obj["ids"] if isinstance(train_obj, dict) and "ids" in train_obj else train_obj
)

# Gold answers from dev600_records.jsonl
gold_map = {}
for line in (ROOT / "handoff/codex/data/dev600_records.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        gold_map[int(r["id"])] = grader.extract_answer(str(r.get("solution", "")))

# Load items
D_srv = json.loads((INBOX / "x7_items.json").read_text(encoding="utf-8"))
D_loc = json.loads((OUTBOX / "x7_items_local.json").read_text(encoding="utf-8"))

s0spec_local = json.loads((INBOX / "s0spec_local.json").read_text(encoding="utf-8"))
gen_files = {lbl: ROOT / rel[0] for lbl, rel in s0spec_local.items()}

gen_records = {}
for lbl, gpath in gen_files.items():
    recs = {}
    for line in gpath.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            qid = int(r["id"])
            text = r.get("generation", r.get("generated", ""))
            toks = int(r.get("tokens", r.get("gen_len", 0)))
            eos = (r.get("stop_id") is not None) if ("stop_id" in r or "generation" in r) else bool(r.get("stopped_by_eos", False))
            recs[qid] = {"text": text, "tokens": toks, "eos": eos}
    gen_records[lbl] = recs

items = sorted(D_srv["C_fmt"].keys(), key=int)
n = len(items)

ALL_LABELS = [
    "C_fmt",
    "X7A47", "X7E47", "X7A49", "X7E49", "X7A50", "X7E50",
    "shuf_X7A47", "shuf_X7E47", "shuf_X7A49", "shuf_X7E49", "shuf_X7A50", "shuf_X7E50",
]


def per_condition_table(D):
    out = {}
    for lbl in ALL_LABELS:
        if lbl not in D:
            continue
        strict_cnt = sum(int(D[lbl][k][0]) for k in items)
        lenient_cnt = sum(int(D[lbl][k][1]) for k in items)
        boxed_cnt = sum(int(D[lbl][k][2]) for k in items)
        lens = [gen_records[lbl][int(k)]["tokens"] for k in items]
        eos_cnt = sum(1 for k in items if gen_records[lbl][int(k)]["eos"])
        sl = sorted(lens)
        len_mean = round(sum(lens) / n, 1)
        p50 = sl[n // 2]
        p95 = sl[int(0.95 * n)]
        eos_pct = round(eos_cnt / n * 100.0, 2)
        out[lbl] = {
            "n": n,
            "boxed_cnt": boxed_cnt,
            "boxed_pct": round(boxed_cnt / n * 100.0, 2),
            "lenient_cnt": lenient_cnt,
            "lenient_pct": round(lenient_cnt / n * 100.0, 2),
            "strict_cnt": strict_cnt,
            "strict_pct": round(strict_cnt / n * 100.0, 2),
            "strict_given_boxed_pct": round(strict_cnt / max(1, boxed_cnt) * 100.0, 2),
            "len_mean": len_mean,
            "len_p50": p50,
            "len_p95": p95,
            "eos_cnt": eos_cnt,
            "eos_pct": eos_pct,
        }
    return out


_IX_CACHE = {}


def get_ix_mat(n_items, B=10000, seed=2026):
    key = (n_items, B, seed)
    if key not in _IX_CACHE:
        rng = random.Random(seed)
        _IX_CACHE[key] = np.array(
            [[rng.randrange(n_items) for _ in range(n_items)] for _ in range(B)],
            dtype=np.int32,
        )
    return _IX_CACHE[key]


def boot_exact(D, pairs, idx, B=10000, seed=2026, sub=None):
    ks = sub if sub is not None else items
    m = len(ks)
    X = np.array([[int(D[a][k][idx]) for k in ks] for a, _ in pairs], dtype=np.int64)
    Y = np.array([[int(D[b][k][idx]) for k in ks] for _, b in pairs], dtype=np.int64)
    diffs = (X - Y).sum(axis=0)
    n_pairs = len(pairs)
    obs = float(diffs.sum()) / (m * n_pairs) * 100.0
    ix_mat = get_ix_mat(m, B, seed)
    sample_sums = diffs[ix_mat].sum(axis=1)
    bs = np.sort(sample_sums.astype(np.float64) / (m * n_pairs) * 100.0)
    lo = float(bs[int(0.025 * B)])
    hi = float(bs[int(0.975 * B) - 1])
    per = [
        round((int(X[p].sum()) - int(Y[p].sum())) / m * 100.0, 2)
        for p in range(n_pairs)
    ]
    return {
        "n_items": m,
        "mean": round(obs, 2),
        "ci_lo": round(lo, 2),
        "ci_hi": round(hi, 2),
        "per_seed": per,
        "raw_mean": obs,
        "raw_lo": lo,
        "raw_hi": hi,
    }


def run_comparisons(D):
    A = ["X7A47", "X7A49", "X7A50"]
    E = ["X7E47", "X7E49", "X7E50"]
    both_6 = [k for k in items if all(D[t][k][2] for t in A + E)]
    comp_specs = [
        ("StageAE_1_A_minus_E", list(zip(A, E)), None),
        ("StageAE_2_A_matched_minus_shuffled", [(t, "shuf_" + t) for t in A], None),
        ("StageAE_ref_E_matched_minus_shuffled", [(t, "shuf_" + t) for t in E], None),
        ("StageAE_ref_A_minus_C_fmt", [(t, "C_fmt") for t in A], None),
        ("StageAE_ref_E_minus_C_fmt", [(t, "C_fmt") for t in E], None),
        ("StageAE_ref_A_shuffled_minus_C_fmt", [("shuf_" + t, "C_fmt") for t in A], None),
        ("StageAE_ref_E_shuffled_minus_C_fmt", [("shuf_" + t, "C_fmt") for t in E], None),
        ("StageAE_1_both_boxed_6cond_A_minus_E", list(zip(A, E)), both_6),
        ("seed47_A_minus_E", [("X7A47", "X7E47")], None),
        ("seed49_A_minus_E", [("X7A49", "X7E49")], None),
        ("seed50_A_minus_E", [("X7A50", "X7E50")], None),
        ("seed47_A_matched_minus_shuffled", [("X7A47", "shuf_X7A47")], None),
        ("seed49_A_matched_minus_shuffled", [("X7A49", "shuf_X7A49")], None),
        ("seed50_A_matched_minus_shuffled", [("X7A50", "shuf_X7A50")], None),
        ("seed47_E_matched_minus_shuffled", [("X7E47", "shuf_X7E47")], None),
        ("seed49_E_matched_minus_shuffled", [("X7E49", "shuf_X7E49")], None),
        ("seed50_E_matched_minus_shuffled", [("X7E50", "shuf_X7E50")], None),
    ]
    res = {}
    for name, pairs, sub in comp_specs:
        res[name] = {
            "lenient": boot_exact(D, pairs, 1, sub=sub),
            "strict": boot_exact(D, pairs, 0, sub=sub),
            "boxed": boot_exact(D, pairs, 2, sub=sub),
        }
    for s in ["47", "49", "50"]:
        a_tag, e_tag = f"X7A{s}", f"X7E{s}"
        sub_s = [k for k in items if D[a_tag][k][2] and D[e_tag][k][2]]
        res[f"seed{s}_both_boxed_pair_A_minus_E"] = {
            "lenient": boot_exact(D, [(a_tag, e_tag)], 1, sub=sub_s),
            "strict": boot_exact(D, [(a_tag, e_tag)], 0, sub=sub_s),
        }
    return res


discrepancies = []
for lbl in ALL_LABELS:
    for k in items:
        s_val = D_srv[lbl][k]
        l_val = D_loc[lbl][k]
        if s_val[:3] != l_val[:3]:
            qid = int(k)
            txt = gen_records[lbl][qid]["text"]
            pred = grader.extract_answer(txt)
            discrepancies.append(
                {
                    "label": lbl,
                    "id": qid,
                    "server_flags": s_val[:3],
                    "local_flags": l_val[:3],
                    "gold": gold_map.get(qid),
                    "pred_extracted": pred,
                }
            )

manifest_audit = {}
for tag in ["X7A47", "X7A49", "X7A50", "X7E47", "X7E49", "X7E50"]:
    m = json.loads((INBOX / f"manifest_{tag}.json").read_text(encoding="utf-8"))
    tel = m["telemetry"]
    last200_loss = sum(x["loss"] for x in tel[-200:]) / 200.0
    last200_ans_ce = sum(x["ans_ce"] for x in tel[-200:]) / 200.0
    first50_loss = sum(x["loss"] for x in tel[:50]) / 50.0
    first50_ans_ce = sum(x["ans_ce"] for x in tel[:50]) / 50.0
    manifest_audit[tag] = {
        "zfile": m["zfile"],
        "zfile_sha256": m["zfile_sha256"],
        "targets_sha256": m["targets_sha256"],
        "items_sha256": m["items_sha256"],
        "dev_order_sha256": m["dev_order_sha256"],
        "seed": m["seed"],
        "steps": m["steps"],
        "batch_size": m["batch_size"],
        "lr": m["lr"],
        "answer_weight": m["answer_weight"],
        "gen_budget": m["gen_budget"],
        "max_seq_len": m["max_seq_len"],
        "n_items_trained": m["n_items_trained"],
        "n_items_listed": m["n_items_listed"],
        "target_counts": m["target_counts"],
        "n_updates": m["n_updates"],
        "n_sample_visits": m["n_sample_visits"],
        "n_seq_truncated": m["n_seq_truncated"],
        "n_samples_skipped_answer_or_eos_cut": m["n_samples_skipped_answer_or_eos_cut"],
        "n_unique_ids_contributing_grad": m["n_unique_ids_contributing_grad"],
        "params": m["params"],
        "telemetry_len": len(tel),
        "step0": tel[0],
        "step_last": tel[-1],
        "first50_loss": first50_loss,
        "first50_ans_ce": first50_ans_ce,
        "final_loss_last200_manifest": m["final_loss_last200"],
        "final_loss_last200_recalc": last200_loss,
        "final_loss_diff": abs(m["final_loss_last200"] - last200_loss),
        "final_ans_ce_last200_manifest": m["final_ans_ce_last200"],
        "final_ans_ce_last200_recalc": last200_ans_ce,
        "final_ans_ce_diff": abs(m["final_ans_ce_last200"] - last200_ans_ce),
    }

run_srv = subprocess.run(
    [sys.executable, "-B", str(INBOX / "x7_analyze.py"), str(INBOX / "x7_items.json")],
    capture_output=True,
    text=True,
    check=True,
)
(OUTBOX / "x7_analyze_server_utf8.txt").write_text(run_srv.stdout, encoding="utf-8")

run_loc = subprocess.run(
    [sys.executable, "-B", str(INBOX / "x7_analyze.py"), str(OUTBOX / "x7_items_local.json")],
    capture_output=True,
    text=True,
    check=True,
)
(OUTBOX / "x7_analyze_local_utf8.txt").write_text(run_loc.stdout, encoding="utf-8")

cc_txt = (INBOX / "cc_x7_analysis.txt").read_text(encoding="utf-8")

summary = {
    "sha_checks": sha_checks,
    "all_26_sha_match": all(v["match"] for v in sha_checks.values()),
    "grader_sha_scripts": grader_sha_scripts,
    "grader_sha_codex": grader_sha_codex,
    "wrapper_sha": wrapper_sha,
    "dev_order_sha": dev_order_sha,
    "train_items_sha": train_items_sha,
    "dev_ids_count": len(dev_ids),
    "train_ids_count": len(train_ids),
    "overlap_train_dev": len(set(dev_ids).intersection(set(train_ids))),
    "cc_x7_analysis_exact_match": (run_srv.stdout.replace("\r\n", "\n").strip() == cc_txt.replace("\r\n", "\n").strip()),
    "per_condition_server": per_condition_table(D_srv),
    "per_condition_local": per_condition_table(D_loc),
    "comparisons_server": run_comparisons(D_srv),
    "comparisons_local": run_comparisons(D_loc),
    "discrepancies_count": len(discrepancies),
    "discrepancies": discrepancies,
    "manifest_audit": manifest_audit,
}

(OUTBOX / "c006_analysis.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("C006 analysis complete.")
print("All 26 SHA match:", summary["all_26_sha_match"])
print("CC exact match:", summary["cc_x7_analysis_exact_match"])
print("Discrepancies (13 labels):", summary["discrepancies_count"])
for d in discrepancies:
    print("  diff:", d)
