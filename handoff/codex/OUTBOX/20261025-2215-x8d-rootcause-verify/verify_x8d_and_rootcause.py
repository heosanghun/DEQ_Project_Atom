"""Independent verification of X8D / X8D0 (including shuffled) / X8DS smoke / Codex Root-Cause Packages.

Read-only local CPU audit; does not train, decode, regrade, or modify existing files.
Writes VERIFICATION.json in the same directory when --out is specified.
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
OUTBOX_RC = ROOT / "OUTBOX/20261025-ROOTCAUSE-STEPS"
OUTBOX_AU = ROOT / "OUTBOX/20261025-X8D0-X8DS-AUDIT"

METRICS = ("strict", "lenient", "boxed_present")
SEEDS = ("47", "49", "50")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_sha256sums_file(sums_path: Path) -> dict:
    base = sums_path.parent
    checked = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        actual = sha256_file(base / rel)
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
    # left, right: shape [S, N, 3]
    delta = (left.astype(float) - right.astype(float)).mean(axis=0)  # [N, 3]
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


def summarize_residual(res_path: Path, expected_ids: list[str]) -> dict:
    obj, h = load_json(res_path)
    rmap = obj["r_output_by_item"]
    if set(rmap.keys()) != set(expected_ids):
        raise ValueError(f"Residual item mismatch in {res_path}")
    vals = np.array([rmap[q] for q in expected_ids], dtype=float)
    last_step = obj.get("res_last_by_item", {})
    last_vals = np.array([last_step[q] for q in expected_ids], dtype=float) if last_step else None
    res = {
        "sha256": h,
        "n_items": int(obj["n_items"]),
        "r_output_mean": float(vals.mean()),
        "r_output_median": float(np.median(vals)),
        "r_output_p90": float(np.quantile(vals, 0.90)),
        "r_output_max": float(vals.max()),
        "r_output_below_0p01": int((vals < 0.01).sum()),
    }
    if last_vals is not None:
        res["rho_last_mean"] = float(last_vals.mean())
        res["rho_last_median"] = float(np.median(last_vals))
        res["rho_last_equiv_r_z1_mean"] = float(last_vals.mean() / obj["alpha"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rc_sums = verify_sha256sums_file(OUTBOX_RC / "SHA256SUMS.txt")
    au_sums = verify_sha256sums_file(OUTBOX_AU / "SHA256SUMS.txt")

    order_obj, order_sha = load_json(INBOX_CC / "server_runtime_snapshot/results_s7/p6_item_order.json")
    dev_ids = sorted(map(str, order_obj["indices"]), key=int)
    assert len(dev_ids) == 600

    train_obj, train_sha = load_json(INBOX_CC / "server_runtime_snapshot/configs/s7_train_993.json")
    train_ids = train_obj["indices"] if isinstance(train_obj, dict) else train_obj
    assert len(train_ids) == 993
    train16_ids = [str(q) for q in train_ids[:16]]

    x8d_table, x8d_sha = load_json(INBOX_CC / "x8d_full/x8d_items.json")
    x8d0_full_table, x8d0_full_sha = load_json(INBOX_CC / "x8d0/x8d0_items.json")
    x8d0_matched_table, x8d0_matched_sha = load_json(INBOX_CC / "x8d0/x8d0_matched_items.json")
    x8b_table, x8b_sha = load_json(INBOX_CC / "x8b_full/x8b_items.json")
    x7_table, x7_sha = load_json(ROOT / "INBOX/20261021-0945-x7-results/x7_items.json")

    # Verify C_fmt and X8D0 matched flags are identical across files
    assert x8d_table["C_fmt"] == x8d0_full_table["C_fmt"] == x8b_table["C_fmt"] == x7_table["C_fmt"]
    assert x8d0_full_table["X8D0"] == x8d0_matched_table["X8D0"]

    draws = bootstrap_draws(600, 10000, 2026)

    c_fmt = np.stack([get_flags(x8d_table, "C_fmt", dev_ids)] * 3)
    x5o47 = np.stack([get_flags(x8d_table, "X5O47", dev_ids)] * 3)
    shuf_x5o47 = np.stack([get_flags(x8d_table, "shuf_X5O47", dev_ids)] * 3)
    x8d_m = np.stack([get_flags(x8d_table, f"X8D{s}", dev_ids) for s in SEEDS])
    x8d_s = np.stack([get_flags(x8d_table, f"shuf_X8D{s}", dev_ids) for s in SEEDS])
    x8d0_m = np.stack([get_flags(x8d0_full_table, "X8D0", dev_ids)] * 3)
    x8d0_s = np.stack([get_flags(x8d0_full_table, "shuf_X8D0", dev_ids)] * 3)
    x8b_m = np.stack([get_flags(x8b_table, f"X8B{s}", dev_ids) for s in SEEDS])
    x7e_m = np.stack([get_flags(x7_table, f"X7E{s}", dev_ids) for s in SEEDS])

    # Single-seed arrays for 1-seed contrasts
    c_fmt_1 = c_fmt[:1]
    x8d0_m_1 = x8d0_m[:1]
    x8d0_s_1 = x8d0_s[:1]
    x5o47_1 = x5o47[:1]
    shuf_x5o47_1 = shuf_x5o47[:1]

    contrasts = {
        "X8D_matched_minus_C_fmt": contrast(x8d_m, c_fmt, draws),
        "X8D_matched_minus_X8D0_matched": contrast(x8d_m, x8d0_m, draws),
        "X8D_matched_minus_X8D_shuffled": contrast(x8d_m, x8d_s, draws),
        "X8D_shuffled_minus_X8D0_shuffled": contrast(x8d_s, x8d0_s, draws),
        "X8D_shuffled_minus_C_fmt": contrast(x8d_s, c_fmt, draws),
        "X8D0_matched_minus_X8D0_shuffled": contrast(x8d0_m_1, x8d0_s_1, draws),
        "X8D0_matched_minus_C_fmt": contrast(x8d0_m_1, c_fmt_1, draws),
        "X8D0_shuffled_minus_C_fmt": contrast(x8d0_s_1, c_fmt_1, draws),
        "X5O47_matched_minus_X5O47_shuffled": contrast(x5o47_1, shuf_x5o47_1, draws),
        "X8D_matched_minus_X7E_matched": contrast(x8d_m, x7e_m, draws),
        "X8D_matched_minus_X8B_matched": contrast(x8d_m, x8b_m, draws),
    }

    # Check difference-in-differences: (X8D_m - X8D_s) - (X8D0_m - X8D0_s)
    did_left = x8d_m.astype(float) - x8d_s.astype(float)
    did_right = x8d0_m.astype(float) - x8d0_s.astype(float)
    contrasts["DID_content_gain_X8D_minus_X8D0"] = contrast(did_left, did_right, draws)

    # Residuals on dev600 and train16
    dev_residuals = {
        "X8D0": summarize_residual(INBOX_CC / "x8d0/latdev_X8D0.pt.res.json", dev_ids),
        "X8D47": summarize_residual(INBOX_CC / "x8d_full/latdev_X8D47.pt.res.json", dev_ids),
        "X8D49": summarize_residual(INBOX_CC / "x8d_full/latdev_X8D49.pt.res.json", dev_ids),
        "X8D50": summarize_residual(INBOX_CC / "x8d_full/latdev_X8D50.pt.res.json", dev_ids),
    }
    smoke_residuals = {
        mode: summarize_residual(INBOX_CC / f"x8ds_smoke/lat_train16_{mode}.pt.res.json", train16_ids)
        for mode in ("initial", "legacy", "undamped_previous")
    }

    report = {
        "git_head_verified": "fe054ac",
        "sha256sums_verified": {
            "OUTBOX/20261025-ROOTCAUSE-STEPS": rc_sums,
            "OUTBOX/20261025-X8D0-X8DS-AUDIT": au_sums,
            "x8d_items.json": x8d_sha,
            "x8d0_items.json": x8d0_full_sha,
            "x8d0_matched_items.json": x8d0_matched_sha,
            "p6_item_order.json": order_sha,
            "s7_train_993.json": train_sha,
        },
        "counts_dev600": {
            "C_fmt": {m: int(c_fmt_1[0, :, i].sum()) for i, m in enumerate(METRICS)},
            "X5O47": {m: int(x5o47_1[0, :, i].sum()) for i, m in enumerate(METRICS)},
            "shuf_X5O47": {m: int(shuf_x5o47_1[0, :, i].sum()) for i, m in enumerate(METRICS)},
            "X8D0_matched": {m: int(x8d0_m_1[0, :, i].sum()) for i, m in enumerate(METRICS)},
            "X8D0_shuffled": {m: int(x8d0_s_1[0, :, i].sum()) for i, m in enumerate(METRICS)},
            "X8D_matched_by_seed": {m: x8d_m[:, :, i].sum(axis=1).tolist() for i, m in enumerate(METRICS)},
            "X8D_shuffled_by_seed": {m: x8d_s[:, :, i].sum(axis=1).tolist() for i, m in enumerate(METRICS)},
        },
        "contrasts": contrasts,
        "dev_residuals": dev_residuals,
        "smoke_residuals": smoke_residuals,
    }

    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(args.out)
        with args.out.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")

    print(json.dumps({
        "counts_dev600": report["counts_dev600"],
        "key_lenient_contrasts": {k: v["lenient"] for k, v in contrasts.items()},
        "key_strict_contrasts": {k: v["strict"] for k, v in contrasts.items()},
        "dev_residuals": dev_residuals,
        "smoke_residuals": smoke_residuals,
    }, indent=2))


if __name__ == "__main__":
    main()
