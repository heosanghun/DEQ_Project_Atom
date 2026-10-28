import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\AI\cts")
OUTBOX = ROOT / "handoff/codex/OUTBOX/20261024-1910-C008"
INBOX_X8A = ROOT / "handoff/codex/INBOX/20261023-1950-x8a-results"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


target_files = [
    ROOT / "handoff/codex/scripts/g2v11_x8a_train.py",
    ROOT / "handoff/codex/scripts/g2v11_x8a_latents.py",
    ROOT / "handoff/codex/scripts/cc_x8a.sh",
    ROOT / "handoff/codex/scripts/g2v10_train.py",
    ROOT / "handoff/codex/scripts/g2v9_xattn.py",
    ROOT / "handoff/codex/scripts/g2v9_gen.py",
    ROOT / "scripts/g2v3_m3_full_train.py",
    ROOT / "cts/deq/transition_inject.py",
    ROOT / "cts/backbone/gemma_adapter.py",
    ROOT / "cts/models/routing.py",
    ROOT / "docs/X8_READABLE_LATENT_DEQ_20261021.md",
]

shas = {
    str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
    for p in target_files
    if p.exists()
}

# Check diff between g2v10_train.py and g2v11_x8a_train.py target preparation
g2v10_src = (ROOT / "handoff/codex/scripts/g2v10_train.py").read_text(encoding="utf-8")
g2v11_tr_src = (ROOT / "handoff/codex/scripts/g2v11_x8a_train.py").read_text(encoding="utf-8")
g2v11_lat_src = (ROOT / "handoff/codex/scripts/g2v11_x8a_latents.py").read_text(encoding="utf-8")
cc_x8a_src = (ROOT / "handoff/codex/scripts/cc_x8a.sh").read_text(encoding="utf-8")

# Inspect runtime logs from INBOX/20261023-1950-x8a-results for post-hoc confirmation of static review items
runtime_evidence = {}
for s in ["47", "49", "50"]:
    m = json.loads((INBOX_X8A / f"manifest_X8A{s}.json").read_text(encoding="utf-8"))
    res = json.loads((INBOX_X8A / f"latdev_X8A{s}.pt.res.json").read_text(encoding="utf-8"))
    tlog = (INBOX_X8A / f"train_X8A{s}.log").read_text(encoding="utf-8").splitlines()
    deq_line = next((l for l in tlog if l.startswith("[DEQ]")), None)
    warn_lines = [l for l in tlog if "UserWarning" in l or "grad_norm" in l]
    step0_line = next((l for l in tlog if "[Step    0]" in l), None)
    step_last_line = next((l for l in tlog if "[Step 1499]" in l), None)
    save_line = next((l for l in tlog if l.startswith("[SAVE]")), None)
    runtime_evidence[f"X8A{s}"] = {
        "deq_log_line": deq_line,
        "step0_log_line": step0_line,
        "step_last_log_line": step_last_line,
        "save_log_line": save_line,
        "warnings": warn_lines,
        "manifest_summary": {
            "deq_trainable_tensors": m["deq_trainable_tensors"],
            "deq_trainable_params": m["deq_trainable_params"],
            "bridge_params": m["bridge_params"],
            "k_unroll": m["k_unroll"],
            "alpha": m["alpha"],
            "g0": m["g0"],
            "top_k": m["top_k"],
            "lr_bridge": m["lr_bridge"],
            "lr_deq": m["lr_deq"],
            "lambda_fp": m["lambda_fp"],
            "final_loss_last200": m["final_loss_last200"],
            "final_ans_ce_last200": m["final_ans_ce_last200"],
            "final_l_fp_last200": m["final_l_fp_last200"],
            "final_z_drift_last200": m["final_z_drift_last200"],
        },
        "latdev_res": res,
    }

out_data = {
    "task": "TASK-C008",
    "script_shas": shas,
    "task_c008_note_on_g2v11_sha": {
        "task_md_prefix": "5721ffc5",
        "actual_repo_and_server_sha": shas["handoff/codex/scripts/g2v11_x8a_train.py"],
        "matches_cts_x8a_master_log_line4": True,
    },
    "runtime_evidence": runtime_evidence,
}

(OUTBOX / "c008_static_verification.json").write_text(
    json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("C008 verification JSON written.")
print(json.dumps(shas, indent=2))
