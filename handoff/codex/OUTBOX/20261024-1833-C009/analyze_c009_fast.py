import hashlib
import json
import pathlib
import random
import statistics
import sys
import numpy as np

ROOT = pathlib.Path(r"D:\AI\cts")
sys.path.insert(0, str(ROOT / "handoff" / "codex" / "scripts"))
import s5_eval_grader_v2 as g2

INBOX = ROOT / "handoff" / "codex" / "INBOX" / "20261023-1950-x8a-results"
X7_INBOX = ROOT / "handoff" / "codex" / "INBOX" / "20261021-0945-x7-results"
OUTBOX = ROOT / "handoff" / "codex" / "OUTBOX" / "20261024-1833-C009"

EXPECTED_GRADER_SHA = "90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13"
EXPECTED_P6_SHA = "6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"
FLAG_IDX = {"strict": 0, "lenient": 1, "boxed": 2}

_IX_CACHE = {}


def get_ix_mat(n: int, B: int = 10000, seed: int = 2026) -> np.ndarray:
    key = (n, B, seed)
    if key not in _IX_CACHE:
        rng = random.Random(seed)
        _IX_CACHE[key] = np.array(
            [[rng.randrange(n) for _ in range(n)] for _ in range(B)], dtype=np.int32
        )
    return _IX_CACHE[key]


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
    diffs = np.array([a_dict[k][idx] - b_dict[k][idx] for k in keys], dtype=np.int64)
    obs = float(diffs.sum()) / n * 100.0
    ix_mat = get_ix_mat(n, B, seed)
    sums = diffs[ix_mat].sum(axis=1)
    bs = np.sort(sums.astype(np.float64) / n * 100.0)
    lo = float(bs[int(0.025 * B)])
    hi = float(bs[int(0.975 * B) - 1])
    return {"obs": obs, "lo": lo, "hi": hi, "n": n}


def bootstrap_pooled3(pairs, keys, flag="lenient", B=10000, seed=2026):
    idx = FLAG_IDX[flag]
    n = len(keys)
    m_pairs = len(pairs)
    per_seed_diffs = [
        np.array([a[k][idx] - b[k][idx] for k in keys], dtype=np.int64)
        for a, b in pairs
    ]
    per_seed_obs = [float(d.sum()) / n * 100.0 for d in per_seed_diffs]
    obs = sum(per_seed_obs) / m_pairs
    ix_mat = get_ix_mat(n, B, seed)
    # Match st.mean(sum(x[i] - y[i] for i in ix) / n for x, y in zip(X, Y)) * 100
    means_per_seed = [(d[ix_mat].sum(axis=1).astype(np.float64) / n) for d in per_seed_diffs]
    bs = np.sort((sum(means_per_seed) / m_pairs) * 100.0)
    lo = float(bs[int(0.025 * B)])
    hi = float(bs[int(0.975 * B) - 1])
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
    gen_by_id = {str(r["id"]): r for r in gen_rows}
    lens = [
        item_dict[k][3]
        if len(item_dict[k]) > 3
        else int(gen_by_id[k].get("tokens", gen_by_id[k].get("gen_len", 0)))
        for k in keys
    ]
    eos_cnt = sum(
        item_dict[k][4]
        if len(item_dict[k]) > 4
        else int(
            gen_by_id[k].get("stop_id") is not None
            or bool(gen_by_id[k].get("stopped_by_eos", False))
        )
        for k in keys
    )
    lens_sorted = sorted(lens)
    p50 = statistics.median(lens_sorted)
    p95 = lens_sorted[int(0.95 * (n - 1))]
    gen_eos_cnt = sum(
        1
        for r in gen_rows
        if (r.get("stop_id") is not None or bool(r.get("stopped_by_eos", False)))
    )
    gen_lens = [int(r.get("tokens", r.get("gen_len", 0))) for r in gen_rows]
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


def format_cc_style_report(D, keys):
    n = len(keys)
    def acc(t, i, sub=None):
        ks = sub if sub is not None else keys
        return sum(D[t][k][i] for k in ks) / max(1, len(ks)) * 100.0

    lines = []
    lines.append(f"== 0) per condition (n={n}): boxed / lenient / strict / len / eos")
    for t in [
        "C_fmt",
        "X8A47", "X8A49", "X8A50",
        "shuf_X8A47", "shuf_X8A49", "shuf_X8A50",
        "X7E47", "X7E49", "X7E50",
        "shuf_X7E47", "shuf_X7E49", "shuf_X7E50",
    ]:
        if t not in D:
            continue
        v = [D[t][k] for k in keys]
        ln = sum(x[3] for x in v) / n if len(v[0]) > 3 else 0.0
        eo = sum(x[4] for x in v) / n * 100.0 if len(v[0]) > 4 else 0.0
        lines.append(
            f"  {t:12s} boxed {acc(t,2):5.1f}%  lenient {acc(t,1):5.2f}%  strict {acc(t,0):5.2f}%  len {ln:5.1f}  eos {eo:4.1f}%"
        )

    def rep(name, pairs_names, sub=None):
        ks = sub if sub is not None else keys
        lines.append(f"== {name}" + ("" if sub is None else f" (subset n={len(sub)})"))
        pairs_dicts = [(D[a], D[b]) for a, b in pairs_names]
        for idx, flg, lab in ((1, "lenient", "lenient (primary)"), (2, "boxed", "boxed (aux)      "), (0, "strict", "strict (aux)     ")):
            r = bootstrap_pooled3(pairs_dicts, ks, flg)
            per = [round(acc(a, idx, ks) - acc(b, idx, ks), 2) for a, b in pairs_names]
            lines.append(
                f"  {lab} mean {r['obs']:+.2f}%p  95% CI [{r['lo']:+.2f}, {r['hi']:+.2f}]  per-seed {per} (+seeds {r['signs']}/{len(per)})"
            )

    A = ["X8A47", "X8A49", "X8A50"]
    E = ["X7E47", "X7E49", "X7E50"]
    rep("CTO (1): X8-A - E (rule: 3 seeds same sign AND pooled lenient CI lower > 0)", list(zip(A, E)))
    rep("CTO (2): X8-A matched - shuffled (rule: lenient CI lower > 0)", [(t, "shuf_" + t) for t in A])
    rep("Aux: X8-A matched - C_fmt (bridge cost check)", [(t, "C_fmt") for t in A])
    rep("Aux: X8-A shuffled - C_fmt (harm)", [("shuf_" + t, "C_fmt") for t in A])
    rep("Ref: E matched - C_fmt", [(t, "C_fmt") for t in E])
    for s in ("47", "49", "50"):
        bb = [k for k in keys if D[f"X8A{s}"][k][2] and D[f"X7E{s}"][k][2]]
        rep(f"Ref both-boxed seed {s}: X8A{s} - X7E{s}", [(f"X8A{s}", f"X7E{s}")], sub=bb)
    return "\n".join(lines) + "\n"


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
            else g2.extract_answer(str(r.get("solution", "")))
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

    spec_x8a = json.loads((INBOX / "s0spec_local.json").read_text(encoding="utf-8"))
    spec_x7 = json.loads((X7_INBOX / "s0spec_local.json").read_text(encoding="utf-8"))
    gen_map = {lbl: ROOT / paths[0] for lbl, paths in spec_x8a.items()}
    for s in ("47", "49", "50"):
        gen_map[f"X7E{s}"] = ROOT / spec_x7[f"X7E{s}"][0]
        gen_map[f"shuf_X7E{s}"] = ROOT / spec_x7[f"shuf_X7E{s}"][0]

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
                    r_gen = gen_by_label_id[lbl][k]
                    txt = str(r_gen.get("generated", r_gen.get("generation", r_gen.get("text", ""))))
                    pred = g2.extract_answer(txt)
                    flag_mismatches.append(
                        {
                            "label": lbl,
                            "item_id": int(k),
                            "flag": flg,
                            "server": sv,
                            "local": lv,
                            "gold": gold_by_id.get(k),
                            "extracted_pred": pred,
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
            per_seed[s]["both_boxed_AE_strict"] = bootstrap_diff(
                D[f"X8A{s}"], D[f"X7E{s}"], bb_AE, "strict"
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

    srv_txt = format_cc_style_report(D_srv, keys)
    loc_txt = format_cc_style_report(D_loc, keys)
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
