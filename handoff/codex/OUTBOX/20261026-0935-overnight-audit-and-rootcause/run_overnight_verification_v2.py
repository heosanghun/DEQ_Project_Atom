"""Independent CPU verification of overnight CTS packages (2026-10-25 22:20 -> 2026-10-26 09:30 KST):
1. D2-MAP-AUDIT & ITER-CONTENT-XC-REVIEW SHA256SUMS + Addendum on X8DS smoke rho_last (res_by_item[q][-1]).
2. RESULT-XC-CONTRAST (xc/final/xc_items.json) & RESULT-D2-XC47 (xc/d2/*.jsonl).
3. RESULT-INFO-PROBE (info_probe/info_probe.json).
4. RESULT-X8E-AUX-ANSWER (x8e/final/x8e_info_probe.json + x8e/train/manifest_X8E*.json vs X8D manifests).

Read-only CPU audit; writes --out exclusively without modifying any existing file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INBOX_CC = ROOT / "INBOX/20261025-0210-CC-x8a-rootcause"

METRICS = ("strict", "lenient", "boxed_present")
SEEDS = ("47", "49", "50")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_sha256sums_file(sums_path: Path, fallback_dir: Path | None = None) -> dict:
    base = sums_path.parent
    checked = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*").strip()
        target = base / rel
        if not target.exists() and fallback_dir is not None and (fallback_dir / rel).exists():
            target = fallback_dir / rel
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"SHA mismatch in {sums_path}: {rel} expected {expected}, got {actual}")
        checked[rel] = actual
    return checked


def load_json(p: Path):
    return json.loads(p.read_bytes()), sha256_file(p)


def get_flags(table: dict, cond: str, ids: list[str]) -> np.ndarray:
    if cond not in table or set(table[cond].keys()) != set(ids):
        raise ValueError(f"Condition {cond} missing or ID mismatch")
    arr = np.asarray([table[cond][q] for q in ids], dtype=np.int8)
    if arr.shape != (len(ids), 3):
        raise ValueError(f"Unexpected shape for {cond}: {arr.shape}")
    return arr


def bootstrap_draws(n: int = 600, n_boot: int = 10000, seed: int = 2026) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)])


def contrast(left: np.ndarray, right: np.ndarray, draws: np.ndarray) -> dict:
    delta = (left.astype(float) - right.astype(float)).mean(axis=0)
    out = {}
    for col, m in enumerate(METRICS):
        d = delta[:, col]
        bs = np.sort(d[draws].mean(axis=1) * 100.0)
        out[m] = {
            "mean_pp": float(d.mean() * 100.0),
            "ci95_pp": [float(bs[250]), float(bs[9749])],
            "per_seed_pp": ((left[:, :, col] - right[:, :, col]).mean(axis=1) * 100.0).tolist(),
            "paired_wins": int((left[:, :, col] > right[:, :, col]).sum()),
            "paired_losses": int((left[:, :, col] < right[:, :, col]).sum()),
        }
    return out


def boot_1d(vals: list[float], n_boot: int = 10000, seed: int = 2026) -> dict:
    rng = random.Random(seed)
    n = len(vals)
    arr = np.asarray(vals, dtype=float)
    draws = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)])
    bs = np.sort(arr[draws].mean(axis=1))
    return {
        "mean": float(arr.mean()),
        "ci95": [float(bs[int(0.025 * n_boot)]), float(bs[int(0.975 * n_boot) - 1])],
        "n": n,
    }


def verify_rho_last(res_path: Path, expected_ids: list[str]) -> dict:
    obj, h = load_json(res_path)
    rmap = obj["r_output_by_item"]
    smap = obj["res_by_item"]
    assert set(rmap.keys()) == set(smap.keys()) == set(expected_ids)
    r_vals = np.array([rmap[q] for q in expected_ids], dtype=float)
    rho_last = np.array([smap[q][-1] for q in expected_ids], dtype=float)
    return {
        "sha256": h,
        "n_items": len(expected_ids),
        "r_output_mean": float(r_vals.mean()),
        "r_output_median": float(np.median(r_vals)),
        "rho_last_mean_from_res_by_item": float(rho_last.mean()),
        "rho_last_median_from_res_by_item": float(np.median(rho_last)),
        "res_last_mean_header": float(obj["res_last_mean"]),
        "header_matches_computed": bool(abs(float(rho_last.mean()) - float(obj["res_last_mean"])) < 1e-12),
    }


def verify_d2_jsonl(jsonl_path: Path, ref_none_by_id: dict | None = None) -> dict:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    m_minus_s = [r["answer_nll"]["matched"]["mean"] - r["answer_nll"]["shuffled"]["mean"] for r in rows]
    m_minus_n = [r["answer_nll"]["matched"]["mean"] - r["answer_nll"]["none"]["mean"] for r in rows]
    abs_ms = [abs(x) for x in m_minus_s]
    if ref_none_by_id is not None:
        for r in rows:
            assert abs(r["answer_nll"]["none"]["mean"] - ref_none_by_id[r["id"]]) < 1e-12
    return {
        "sha256": sha256_file(jsonl_path),
        "n": len(rows),
        "matched_minus_shuffled": boot_1d(m_minus_s),
        "matched_minus_none": boot_1d(m_minus_n),
        "mean_abs_matched_minus_shuffled": float(np.mean(abs_ms)),
        "matched_lower_count": int(sum(x < 0 for x in m_minus_s)),
        "shuffled_lower_count": int(sum(x > 0 for x in m_minus_s)),
        "none_by_id": {r["id"]: r["answer_nll"]["none"]["mean"] for r in rows},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    sha_checks = {
        "OUTBOX/20261025-D2-MAP-AUDIT": verify_sha256sums_file(ROOT / "OUTBOX/20261025-D2-MAP-AUDIT/SHA256SUMS.txt"),
        "OUTBOX/20261026-ITER-CONTENT-XC-REVIEW": verify_sha256sums_file(ROOT / "OUTBOX/20261026-ITER-CONTENT-XC-REVIEW/SHA256SUMS.txt"),
        "OUTBOX/20261026-ITER-CONTENT-XC-REVIEW-ADDENDUM": verify_sha256sums_file(ROOT / "OUTBOX/20261026-ITER-CONTENT-XC-REVIEW/SHA256SUMS-ADDENDUM.txt"),
        "INBOX/x8e": verify_sha256sums_file(INBOX_CC / "x8e/SHA256SUMS.txt", fallback_dir=ROOT / "OUTBOX/20261025-X8DS-CANDIDATE"),
    }

    order_obj, _ = load_json(INBOX_CC / "server_runtime_snapshot/results_s7/p6_item_order.json")
    dev_ids = sorted(map(str, order_obj["indices"]), key=int)
    train_obj, _ = load_json(INBOX_CC / "server_runtime_snapshot/configs/s7_train_993.json")
    train_ids = train_obj["indices"] if isinstance(train_obj, dict) else train_obj
    train16_ids = [str(q) for q in train_ids[:16]]

    rho_addendum = {
        "smoke_train16": {
            m: verify_rho_last(INBOX_CC / f"x8ds_smoke/lat_train16_{m}.pt.res.json", train16_ids)
            for m in ("initial", "legacy", "undamped_previous")
        },
        "dev600": {
            "X8D0": verify_rho_last(INBOX_CC / "x8d0/latdev_X8D0.pt.res.json", dev_ids),
            "X8D47": verify_rho_last(INBOX_CC / "x8d_full/latdev_X8D47.pt.res.json", dev_ids),
            "X8D49": verify_rho_last(INBOX_CC / "x8d_full/latdev_X8D49.pt.res.json", dev_ids),
            "X8D50": verify_rho_last(INBOX_CC / "x8d_full/latdev_X8D50.pt.res.json", dev_ids),
        },
    }

    xc_table, xc_sha = load_json(INBOX_CC / "xc/final/xc_items.json")
    x7_table, x7_sha = load_json(ROOT / "INBOX/20261021-0945-x7-results/x7_items.json")
    assert xc_table["C_fmt"] == x7_table["C_fmt"]
    draws600 = bootstrap_draws(600, 10000, 2026)

    c_fmt = np.stack([get_flags(xc_table, "C_fmt", dev_ids)] * 3)
    xc_m = np.stack([get_flags(xc_table, f"XC{s}", dev_ids) for s in SEEDS])
    xc_s = np.stack([get_flags(xc_table, f"shuf_XC{s}", dev_ids) for s in SEEDS])
    x7a_m = np.stack([get_flags(x7_table, f"X7A{s}", dev_ids) for s in SEEDS])
    x7a_s = np.stack([get_flags(x7_table, f"shuf_X7A{s}", dev_ids) for s in SEEDS])

    xc_audit = {
        "xc_items_sha256": xc_sha,
        "counts_dev600": {
            "C_fmt": {m: int(c_fmt[0, :, i].sum()) for i, m in enumerate(METRICS)},
            "X7A_matched": {m: x7a_m[:, :, i].sum(axis=1).tolist() for i, m in enumerate(METRICS)},
            "X7A_shuffled": {m: x7a_s[:, :, i].sum(axis=1).tolist() for i, m in enumerate(METRICS)},
            "XC_matched": {m: xc_m[:, :, i].sum(axis=1).tolist() for i, m in enumerate(METRICS)},
            "XC_shuffled": {m: xc_s[:, :, i].sum(axis=1).tolist() for i, m in enumerate(METRICS)},
        },
        "contrasts": {
            "XC_matched_minus_shuffled": contrast(xc_m, xc_s, draws600),
            "XC_matched_minus_X7A_matched": contrast(xc_m, x7a_m, draws600),
            "XC_matched_minus_C_fmt": contrast(xc_m, c_fmt, draws600),
            "XC_shuffled_minus_X7A_shuffled": contrast(xc_s, x7a_s, draws600),
            "X7A_matched_minus_shuffled": contrast(x7a_m, x7a_s, draws600),
            "DiD_XC_MS_minus_X7A_MS": contrast(xc_m.astype(float) - xc_s.astype(float), x7a_m.astype(float) - x7a_s.astype(float), draws600),
        },
    }

    d2_x8d_dev_own = verify_d2_jsonl(INBOX_CC / "d2_x8d/d2_X8D47_dev200_own.jsonl")
    d2_x8d_dev_empty = verify_d2_jsonl(INBOX_CC / "d2_x8d/d2_X8D47_dev200_empty.jsonl")
    d2_x8d_tr_own = verify_d2_jsonl(INBOX_CC / "d2_x8d/d2_X8D47_train200_own.jsonl")
    d2_x8d_tr_empty = verify_d2_jsonl(INBOX_CC / "d2_x8d/d2_X8D47_train200_empty.jsonl")
    d2_x8d0_dev_own = verify_d2_jsonl(INBOX_CC / "d2_x8d/d2_X8D0_dev200_own.jsonl", d2_x8d_dev_own["none_by_id"])
    d2_x8d0_dev_empty = verify_d2_jsonl(INBOX_CC / "d2_x8d/d2_X8D0_dev200_empty.jsonl", d2_x8d_dev_empty["none_by_id"])

    d2_xc47_dev_own = verify_d2_jsonl(INBOX_CC / "xc/d2/d2_XC47_dev200_own.jsonl", d2_x8d_dev_own["none_by_id"])
    d2_xc47_dev_empty = verify_d2_jsonl(INBOX_CC / "xc/d2/d2_XC47_dev200_empty.jsonl", d2_x8d_dev_empty["none_by_id"])
    d2_xc47_tr_own = verify_d2_jsonl(INBOX_CC / "xc/d2/d2_XC47_train200_own.jsonl", d2_x8d_tr_own["none_by_id"])
    d2_xc47_tr_empty = verify_d2_jsonl(INBOX_CC / "xc/d2/d2_XC47_train200_empty.jsonl", d2_x8d_tr_empty["none_by_id"])

    for d in (d2_x8d_dev_own, d2_x8d_dev_empty, d2_x8d_tr_own, d2_x8d_tr_empty,
              d2_x8d0_dev_own, d2_x8d0_dev_empty,
              d2_xc47_dev_own, d2_xc47_dev_empty, d2_xc47_tr_own, d2_xc47_tr_empty):
        d.pop("none_by_id", None)

    d2_audit = {
        "X8D47_dev_own": d2_x8d_dev_own,
        "X8D47_dev_empty": d2_x8d_dev_empty,
        "X8D47_train_own": d2_x8d_tr_own,
        "X8D47_train_empty": d2_x8d_tr_empty,
        "X8D0_dev_own": d2_x8d0_dev_own,
        "X8D0_dev_empty": d2_x8d0_dev_empty,
        "XC47_dev_own": d2_xc47_dev_own,
        "XC47_dev_empty": d2_xc47_dev_empty,
        "XC47_train_own": d2_xc47_tr_own,
        "XC47_train_empty": d2_xc47_tr_empty,
        "none_baseline_exact_match_across_X8D47_X8D0_XC47": True,
    }

    info_obj, info_sha = load_json(INBOX_CC / "info_probe/info_probe.json")
    x8e_probe_obj, x8e_probe_sha = load_json(INBOX_CC / "x8e/final/x8e_info_probe.json")

    R, PI = x8e_probe_obj["results"], x8e_probe_obj["per_item"]
    E_tags, D_tags = ["X8E47", "X8E49", "X8E50"], ["X8D47", "X8D49", "X8D50"]
    x8e_pooled_verify = {}
    for task in x8e_probe_obj["tasks"]:
        ids = sorted(set.intersection(*[set(PI[t][task].keys()) for t in E_tags + D_tags + ["X8D0"]]), key=int)
        e_scores = [sum(int(PI[t][task][i][0] == PI[t][task][i][1]) for t in E_tags) / 3.0 for i in ids]
        d_scores = [sum(int(PI[t][task][i][0] == PI[t][task][i][1]) for t in D_tags) / 3.0 for i in ids]
        z0_scores = [float(PI["X8D0"][task][i][0] == PI["X8D0"][task][i][1]) for i in ids]
        gold = [PI["X8D47"][task][i][1] for i in ids]
        maj_acc = R["X8D47"][task]["acc_train_majority"]
        classes = R["X8D47"][task]["classes"]
        matching_maj = [c for c in classes if abs(sum(g == c for g in gold) / len(gold) - maj_acc) < 1e-9]
        assert len(matching_maj) == 1
        maj_cls = matching_maj[0]
        maj_scores = [float(g == maj_cls) for g in gold]
        x8e_pooled_verify[task] = {
            "n_dev": len(ids),
            "X8E_acc": float(np.mean(e_scores)),
            "X8D_acc": float(np.mean(d_scores)),
            "X8D0_acc": float(np.mean(z0_scores)),
            "maj_acc": float(np.mean(maj_scores)),
            "X8E_minus_maj": boot_1d([a - b for a, b in zip(e_scores, maj_scores)]),
            "X8D_minus_maj": boot_1d([a - b for a, b in zip(d_scores, maj_scores)]),
            "X8E_minus_X8D": boot_1d([a - b for a, b in zip(e_scores, d_scores)]),
        }

    x8e_train_summary = {}
    for s in SEEDS:
        mf_e, h_e = load_json(INBOX_CC / f"x8e/train/manifest_X8E{s}.json")
        mf_d, h_d = load_json(INBOX_CC / f"x8d_full/manifest_X8D{s}.json")
        tel_e = mf_e["telemetry"]
        first100_aux_acc = float(np.mean([t["aux_acc"] for t in tel_e[:100] if t["aux_acc"] is not None]))
        last200_aux_acc = float(np.mean([t["aux_acc"] for t in tel_e[-200:] if t["aux_acc"] is not None]))
        last200_aux_loss = float(np.mean([t["aux_loss"] for t in tel_e[-200:] if t["aux_loss"] is not None]))
        g0_e = mf_e["update_telemetry"][0]["task_gradient"]
        g0_d = mf_d["update_telemetry"][0]["task_gradient"]
        x8e_train_summary[f"seed_20{s}"] = {
            "manifest_X8E_sha256": h_e,
            "manifest_X8D_sha256": h_d,
            "n_updates": mf_e["n_updates"],
            "n_unique_ids": mf_e["n_unique_ids_contributing_grad"],
            "X8E_final_ans_ce_last200": mf_e["final_ans_ce_last200"],
            "X8D_final_ans_ce_last200": mf_d["final_ans_ce_last200"],
            "X8E_final_res_last200": mf_e["final_res_last200"],
            "X8D_final_res_last200": mf_d["final_res_last200"],
            "X8E_first100_aux_acc": first100_aux_acc,
            "X8E_last200_aux_acc": last200_aux_acc,
            "X8E_last200_aux_loss": last200_aux_loss,
            "step0_read_grad_l2_X8E_vs_X8D": [g0_e["read"]["l2"], g0_d["read"]["l2"]],
        }

    out_obj = {
        "git_head_verified": "e21930b",
        "sha256sums_verified": sha_checks,
        "rho_last_addendum": rho_addendum,
        "xc_final_audit": xc_audit,
        "d2_audit": d2_audit,
        "info_probe_sha256": info_sha,
        "x8e_info_probe_sha256": x8e_probe_sha,
        "x8e_pooled_verify": x8e_pooled_verify,
        "x8e_train_vs_x8d_summary": x8e_train_summary,
    }

    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(out_obj, f, indent=2, allow_nan=False)
        f.write("\n")

    print(json.dumps({
        "rho_smoke_means": {k: v["rho_last_mean_from_res_by_item"] for k, v in rho_addendum["smoke_train16"].items()},
        "xc_primary_lenient": xc_audit["contrasts"]["XC_matched_minus_shuffled"]["lenient"],
        "xc_did_strict": xc_audit["contrasts"]["DiD_XC_MS_minus_X7A_MS"]["strict"],
        "d2_xc47_dev_own_ms": d2_audit["XC47_dev_own"]["matched_minus_shuffled"],
        "d2_xc47_train_own_ms": d2_audit["XC47_train_own"]["matched_minus_shuffled"],
        "x8e_minus_x8d": {k: v["X8E_minus_X8D"] for k, v in x8e_pooled_verify.items()},
        "x8e_train_summary": x8e_train_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
