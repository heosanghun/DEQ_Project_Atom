import hashlib
import json
import pathlib
import random
import statistics
import subprocess
import sys

ROOT = pathlib.Path(r"D:\AI\cts")
sys.path.insert(0, str(ROOT / "handoff" / "codex" / "scripts"))
import s5_eval_grader_v2 as g2

INBOX = ROOT / "handoff" / "codex" / "INBOX" / "20261023-1950-x8a-results"
X7_INBOX = ROOT / "handoff" / "codex" / "INBOX" / "20261021-0945-x7-results"
OUTBOX = ROOT / "handoff" / "codex" / "OUTBOX" / "20261024-1833-C009"

EXPECTED_GRADER_SHA = "90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13"
EXPECTED_P6_SHA = "6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"
FLAG_IDX = {"strict": 0, "lenient": 1, "boxed": 2}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: pathlib.Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bootstrap_diff(a_dict, b_dict, keys, flag="lenient", B=10000, seed=2026):
    idx = FLAG_IDX[flag]
    n = len(keys)
    diffs = [a_dict[k][idx] - b_dict[k][idx] for k in keys]
    obs = sum(diffs) / n * 100.0
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        bs.append(s / n * 100.0)
    bs.sort()
    lo = bs[int(0.025 * B)]
    hi = bs[int(0.975 * B) - 1]
    return {"obs": obs, "lo": lo, "hi": hi, "n": n}


def bootstrap_pooled3(pairs, keys, flag="lenient", B=10000, seed=2026):
    idx = FLAG_IDX[flag]
    n = len(keys)
    per_seed_diffs = [[a[k][idx] - b[k][idx] for k in keys] for a, b in pairs]
    per_seed_obs = [sum(d) / n * 100.0 for d in per_seed_diffs]
    obs = sum(per_seed_obs) / len(pairs)
    rng = random.Random(seed)
    bs = []
    m_pairs = len(pairs)
    for _ in range(B):
        ix = [rng.randrange(n) for _ in range(n)]
        total = 0.0
        for d in per_seed_diffs:
            total += sum(d[i] for i in ix) / n * 100.0
        bs.append(total / m_pairs)
    bs.sort()
    lo = bs[int(0.025 * B)]
    hi = bs[int(0.975 * B) - 1]
    return {
        "obs": obs,
        "lo": lo,
        "hi": hi,
        "n": n,
        "per_seed": per_seed_obs,
        "signs": sum(1 for x in per_seed_obs if x > 0),
    }


def summarize_condition(item_dict, gen_rows, keys):
    n = len(keys)
    s_cnt = sum(item_dict[k][0] for k in keys)
    l_cnt = sum(item_dict[k][1] for k in keys)
    b_cnt = sum(item_dict[k][2] for k in keys)
    lens = [item_dict[k][3] for k in keys]
    eos_cnt = sum(item_dict[k][4] for k in keys)
    lens_sorted = sorted(lens)
    p50 = statistics.median(lens_sorted)
    p95 = lens_sorted[int(0.95 * (n - 1))]
    gen_eos_cnt = sum(1 for r in gen_rows if r.get("stopped_by_eos", False))
    gen_lens = [int(r.get("gen_len", 0)) for r in gen_rows]
    return {
        "n": n,
        "strict_cnt": s_cnt,
        "strict_pct": 100.0 * s_cnt / n,
        "lenient_cnt": l_cnt,
        "lenient_pct": 100.0 * l_cnt / n,
        "boxed_cnt": b_cnt,
        "boxed_pct": 100.0 * b_cnt / n,
        "len_mean": sum(lens) / n,
        "len_p50": p50,
        "len_p95": p95,
        "eos_cnt": eos_cnt,
        "eos_pct": 100.0 * eos_cnt / n,
        "gen_eos_cnt": gen_eos_cnt,
        "gen_len_mean": sum(gen_lens) / len(gen_lens) if gen_lens else 0.0,
    }


def main():
    grader1 = ROOT / "scripts" / "s5_eval_grader_v2.py"
    grader2 = ROOT / "handoff" / "codex" / "scripts" / "s5_eval_grader_v2.py"
    p6_1 = ROOT / "handoff" / "codex" / "data" / "p6_item_order.json"

    sha_checks = {
        "scripts/s5_eval_grader_v2.py": sha256_file(grader1),
        "handoff/codex/scripts/s5_eval_grader_v2.py": sha256_file(grader2),
        "handoff/codex/data/p6_item_order.json": sha256_file(p6_1),
        "configs/s7_train_993.json": sha256_file(ROOT / "configs" / "s7_train_993.json"),
        "handoff/codex/scripts/codex_grade_local.py": sha256_file(
            ROOT / "handoff" / "codex" / "scripts" / "codex_grade_local.py"
        ),
        "handoff/codex/scripts/g2v11_x8a_train.py": sha256_file(
            ROOT / "handoff" / "codex" / "scripts" / "g2v11_x8a_train.py"
        ),
        "handoff/codex/scripts/g2v11_x8a_latents.py": sha256_file(
            ROOT / "handoff" / "codex" / "scripts" / "g2v11_x8a_latents.py"
        ),
        "handoff/codex/scripts/cc_x8a.sh": sha256_file(
            ROOT / "handoff" / "codex" / "scripts" / "cc_x8a.sh"
        ),
    }
    assert sha_checks["scripts/s5_eval_grader_v2.py"] == EXPECTED_GRADER_SHA
    assert sha_checks["handoff/codex/scripts/s5_eval_grader_v2.py"] == EXPECTED_GRADER_SHA
    assert sha_checks["handoff/codex/data/p6_item_order.json"] == EXPECTED_P6_SHA

    # Verify SHA256SUMS_full.txt for all files present in INBOX
    sha_full_lines = (INBOX / "SHA256SUMS_full.txt").read_text(encoding="utf-8").splitlines()
    inbox_sha_verification = []
    for line in sha_full_lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        expected_h, remote_path = parts
        fname = pathlib.Path(remote_path).name
        local_candidate = INBOX / fname
        if local_candidate.exists():
            actual_h = sha256_file(local_candidate)
            inbox_sha_verification.append(
                {
                    "file": fname,
                    "remote_path": remote_path,
                    "expected": expected_h,
                    "actual": actual_h,
                    "match": expected_h == actual_h,
                }
            )
        else:
            inbox_sha_verification.append(
                {
                    "file": fname,
                    "remote_path": remote_path,
                    "expected": expected_h,
                    "actual": None,
                    "match": None,
                }
            )

    # Load gold answers
    dev_recs = load_jsonl(ROOT / "handoff" / "codex" / "data" / "dev600_records.jsonl")
    gold_by_id = {
        str(r["id"]): str(
            r.get("final_answer")
            if r.get("final_answer") is not None
            else g2.extract_answer(r.get("solution", ""))[0]
        )
        for r in dev_recs
    }

    # Load server items and local regraded items
    server_x8a = json.loads((INBOX / "x8a_items.json").read_text(encoding="utf-8"))
    server_x7 = json.loads((X7_INBOX / "x7_items.json").read_text(encoding="utf-8"))
    local_x8a = json.loads((OUTBOX / "x8a_items_local.json").read_text(encoding="utf-8"))

    D_srv = dict(server_x8a)
    for s in ("47", "49", "50"):
        D_srv[f"X7E{s}"] = server_x7[f"X7E{s}"]
        D_srv[f"shuf_X7E{s}"] = server_x7[f"shuf_X7E{s}"]

    D_loc = dict(local_x8a)
    for s in ("47", "49", "50"):
        D_loc[f"X7E{s}"] = server_x7[f"X7E{s}"]
        D_loc[f"shuf_X7E{s}"] = server_x7[f"shuf_X7E{s}"]

    raw_p6 = json.loads(p6_1.read_text(encoding="utf-8"))
    p6_list = (
        raw_p6.get("item_ids", raw_p6.get("indices", raw_p6))
        if isinstance(raw_p6, dict)
        else raw_p6
    )
    p6_ids = [str(x) for x in p6_list]
    keys = sorted(D_srv["C_fmt"].keys(), key=int)
    assert sorted(p6_ids, key=int) == keys and len(keys) == 600

    gen_map = {
        "C_fmt": INBOX / "gen_C_fmt.jsonl",
        "X8A47": INBOX / "gen_X8A47.jsonl",
        "X8A49": INBOX / "gen_X8A49.jsonl",
        "X8A50": INBOX / "gen_X8A50.jsonl",
        "shuf_X8A47": INBOX / "gen_shuf_X8A47.jsonl",
        "shuf_X8A49": INBOX / "gen_shuf_X8A49.jsonl",
        "shuf_X8A50": INBOX / "gen_shuf_X8A50.jsonl",
        "X7E47": X7_INBOX / "gen_X7E47.jsonl",
        "X7E49": X7_INBOX / "gen_X7E49.jsonl",
        "X7E50": X7_INBOX / "gen_X7E50.jsonl",
        "shuf_X7E47": X7_INBOX / "gen_shuf_X7E47.jsonl",
        "shuf_X7E49": X7_INBOX / "gen_shuf_X7E49.jsonl",
        "shuf_X7E50": X7_INBOX / "gen_shuf_X7E50.jsonl",
    }
    gen_rows_by_label = {lbl: load_jsonl(gpath) for lbl, gpath in gen_map.items()}
    gen_by_label_id = {
        lbl: {str(r["id"]): r for r in rows} for lbl, rows in gen_rows_by_label.items()
    }

    # Compare server vs local flags on all 7 labels in x8a_items
    labels_x8a = [
        "C_fmt",
        "X8A47",
        "X8A49",
        "X8A50",
        "shuf_X8A47",
        "shuf_X8A49",
        "shuf_X8A50",
    ]
    flag_mismatches = []
    total_flags = 0
    for lbl in labels_x8a:
        for k in keys:
            for flg, idx in FLAG_IDX.items():
                total_flags += 1
                sv = server_x8a[lbl][k][idx]
                lv = local_x8a[lbl][k][idx]
                if sv != lv:
                    txt = gen_by_label_id[lbl][k].get("text", "")
                    pred, src = g2.extract_answer(txt)
                    flag_mismatches.append(
                        {
                            "label": lbl,
                            "item_id": int(k),
                            "flag": flg,
                            "server": sv,
                            "local": lv,
                            "gold": gold_by_id.get(k),
                            "extracted_pred": pred,
                            "extract_src": src,
                        }
                    )

    cond_summary_srv = {}
    cond_summary_loc = {}
    for lbl, grows in gen_rows_by_label.items():
        cond_summary_srv[lbl] = summarize_condition(D_srv[lbl], grows, keys)
        cond_summary_loc[lbl] = summarize_condition(D_loc[lbl], grows, keys)

    # Bootstrap comparisons (server and local)
    def run_all_bootstraps(D):
        res = {}
        pairs_AE = [(D[f"X8A{s}"], D[f"X7E{s}"]) for s in ("47", "49", "50")]
        pairs_A_ms = [(D[f"X8A{s}"], D[f"shuf_X8A{s}"]) for s in ("47", "49", "50")]
        pairs_A_C = [(D[f"X8A{s}"], D["C_fmt"]) for s in ("47", "49", "50")]
        pairs_Ashuf_C = [(D[f"shuf_X8A{s}"], D["C_fmt"]) for s in ("47", "49", "50")]
        pairs_E_C = [(D[f"X7E{s}"], D["C_fmt"]) for s in ("47", "49", "50")]
        pairs_E_ms = [(D[f"X7E{s}"], D[f"shuf_X7E{s}"]) for s in ("47", "49", "50")]
        pairs_shuf_AE = [(D[f"shuf_X8A{s}"], D[f"shuf_X7E{s}"]) for s in ("47", "49", "50")]

        for name, pairs in [
            ("X8A_minus_E_pooled", pairs_AE),
            ("X8A_matched_minus_shuf_pooled", pairs_A_ms),
            ("X8A_minus_C_pooled", pairs_A_C),
            ("X8A_shuf_minus_C_pooled", pairs_Ashuf_C),
            ("X7E_minus_C_pooled", pairs_E_C),
            ("X7E_matched_minus_shuf_pooled", pairs_E_ms),
            ("X8A_shuf_minus_X7E_shuf_pooled", pairs_shuf_AE),
        ]:
            res[name] = {
                flg: bootstrap_pooled3(pairs, keys, flg)
                for flg in ("lenient", "boxed", "strict")
            }

        per_seed = {}
        for s in ("47", "49", "50"):
            per_seed[s] = {
                "X8A_minus_E": {
                    flg: bootstrap_diff(D[f"X8A{s}"], D[f"X7E{s}"], keys, flg)
                    for flg in ("lenient", "boxed", "strict")
                },
                "X8A_matched_minus_shuf": {
                    flg: bootstrap_diff(D[f"X8A{s}"], D[f"shuf_X8A{s}"], keys, flg)
                    for flg in ("lenient", "boxed", "strict")
                },
                "X8A_minus_C": {
                    flg: bootstrap_diff(D[f"X8A{s}"], D["C_fmt"], keys, flg)
                    for flg in ("lenient", "boxed", "strict")
                },
                "X8A_shuf_minus_C": {
                    flg: bootstrap_diff(D[f"shuf_X8A{s}"], D["C_fmt"], keys, flg)
                    for flg in ("lenient", "boxed", "strict")
                },
                "X7E_minus_C": {
                    flg: bootstrap_diff(D[f"X7E{s}"], D["C_fmt"], keys, flg)
                    for flg in ("lenient", "boxed", "strict")
                },
            }
            bb_AE = [k for k in keys if D[f"X8A{s}"][k][2] and D[f"X7E{s}"][k][2]]
            bb_Ams = [
                k for k in keys if D[f"X8A{s}"][k][2] and D[f"shuf_X8A{s}"][k][2]
            ]
            per_seed[s]["both_boxed_AE_lenient"] = bootstrap_diff(
                D[f"X8A{s}"], D[f"X7E{s}"], bb_AE, "lenient"
            )
            per_seed[s]["both_boxed_Ams_lenient"] = bootstrap_diff(
                D[f"X8A{s}"], D[f"shuf_X8A{s}"], bb_Ams, "lenient"
            )
        res["per_seed"] = per_seed
        return res

    boot_srv = run_all_bootstraps(D_srv)
    boot_loc = run_all_bootstraps(D_loc)

    # Audit manifests and dev latent residual files
    manifests_audit = {}
    for s, seed_num in [("47", 2047), ("49", 2049), ("50", 2050)]:
        m = json.loads((INBOX / f"manifest_X8A{s}.json").read_text(encoding="utf-8"))
        tel = m["telemetry"]
        last200 = tel[-200:]
        re_loss = sum(x["loss"] for x in last200) / len(last200)
        re_ans_ce = sum(x["ans_ce"] for x in last200) / len(last200)
        re_lfp = sum(x["l_fp"] for x in last200) / len(last200)
        re_res = sum(x["res_mean"] for x in last200) / len(last200)

        lat = json.loads((INBOX / f"latdev_X8A{s}.pt.res.json").read_text(encoding="utf-8"))
        rbi = lat["res_by_item"]
        step0_vals = [v[0] for v in rbi.values()]
        step1_vals = [v[1] for v in rbi.values()]
        n_contracted = sum(1 for v in rbi.values() if v[1] < v[0])

        tlog_lines = (INBOX / f"train_X8A{s}.log").read_text(encoding="utf-8").splitlines()
        step1499_line = [ln for ln in tlog_lines if "[Step 1499]" in ln][0]
        warn_lines = [ln for ln in tlog_lines if "UserWarning" in ln or "gates =" in ln]

        manifests_audit[s] = {
            "seed": m["seed"],
            "expected_seed": seed_num,
            "steps": m["steps"],
            "n_updates": m["n_updates"],
            "n_sample_visits": m["n_sample_visits"],
            "n_samples_skipped": m["n_samples_skipped"],
            "n_items_trained": m["n_items_trained"],
            "n_unique_ids_contributing_grad": m["n_unique_ids_contributing_grad"],
            "deq_trainable_weights": m["deq_trainable_weights"],
            "bridge_params": m["bridge_params"],
            "k_unroll": m["k_unroll"],
            "alpha": m["alpha"],
            "g0": m["g0"],
            "lambda_fp": m["lambda_fp"],
            "lr": m["lr"],
            "lr_deq": m["lr_deq"],
            "target_counts": m["target_counts"],
            "telemetry_len": len(tel),
            "step0": tel[0],
            "step1499": tel[-1],
            "manifest_final_loss_last200": m["final_loss_last200"],
            "recalc_final_loss_last200": re_loss,
            "manifest_final_ans_ce_last200": m["final_ans_ce_last200"],
            "recalc_final_ans_ce_last200": re_ans_ce,
            "manifest_final_lfp_last200": m["final_lfp_last200"],
            "recalc_final_lfp_last200": re_lfp,
            "manifest_final_res_last200": m["final_res_last200"],
            "recalc_final_res_last200": re_res,
            "latdev_n_items": lat["n_items"],
            "latdev_n_missing": lat["n_missing"],
            "latdev_res_step0_mean": sum(step0_vals) / len(step0_vals),
            "latdev_res_step0_max": max(step0_vals),
            "latdev_res_last_mean_recorded": lat["res_last_mean"],
            "latdev_res_last_mean_recalc": sum(step1_vals) / len(step1_vals),
            "latdev_res_last_max_recorded": lat["res_last_max"],
            "latdev_res_last_max_recalc": max(step1_vals),
            "latdev_n_step1_lt_step0": n_contracted,
            "train_log_step1499_line": step1499_line,
            "train_log_warn_lines": warn_lines,
        }

    # Capture UTF-8 output of x8a_analyze.py on server and local items
    srv_txt = subprocess.check_output(
        [
            sys.executable,
            "-B",
            str(INBOX / "x8a_analyze.py"),
            str(INBOX / "x8a_items.json"),
            str(X7_INBOX / "x7_items.json"),
        ],
        text=True,
        encoding="utf-8",
    )
    loc_txt = subprocess.check_output(
        [
            sys.executable,
            "-B",
            str(INBOX / "x8a_analyze.py"),
            str(OUTBOX / "x8a_items_local.json"),
            str(X7_INBOX / "x7_items.json"),
        ],
        text=True,
        encoding="utf-8",
    )
    (OUTBOX / "x8a_analyze_server_utf8.txt").write_text(srv_txt, encoding="utf-8")
    (OUTBOX / "x8a_analyze_local_utf8.txt").write_text(loc_txt, encoding="utf-8")

    cc_orig_txt = (INBOX / "cc_x8a_analysis.txt").read_text(encoding="utf-8")

    out_data = {
        "sha_checks": sha_checks,
        "inbox_sha_verification": inbox_sha_verification,
        "cc_analysis_exact_match": srv_txt.strip() == cc_orig_txt.strip(),
        "flag_comparison": {
            "total_flags": total_flags,
            "matched_flags": total_flags - len(flag_mismatches),
            "mismatch_count": len(flag_mismatches),
            "mismatches": flag_mismatches,
        },
        "condition_summary_server": cond_summary_srv,
        "condition_summary_local": cond_summary_loc,
        "bootstrap_server": boot_srv,
        "bootstrap_local": boot_loc,
        "manifests_audit": manifests_audit,
    }
    (OUTBOX / "c009_analysis.json").write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("C009 analysis complete. Mismatches:", len(flag_mismatches))
    print("CC analysis exact match:", srv_txt.strip() == cc_orig_txt.strip())


if __name__ == "__main__":
    main()
