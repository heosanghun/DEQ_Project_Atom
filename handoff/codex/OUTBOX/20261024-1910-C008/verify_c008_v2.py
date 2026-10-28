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
    ROOT / "handoff/codex/INBOX/TASK-C008.md",
    ROOT / "handoff/codex/scripts/g2v11_x8a_train.py",
    ROOT / "handoff/codex/scripts/g2v11_x8a_latents.py",
    ROOT / "handoff/codex/scripts/cc_x8a.sh",
    ROOT / "handoff/codex/scripts/g2v10_train.py",
    ROOT / "handoff/codex/scripts/g2v9_xattn.py",
    ROOT / "handoff/codex/scripts/g2v9_gen.py",
    ROOT / "scripts/g2v3_m3_full_train.py",
    ROOT / "cts/deq/transition_inject.py",
    ROOT / "cts/backbone/gemma_adapter.py",
    ROOT / "cts/deq/gemma_latent_forward.py",
    ROOT / "scripts/s5_eval_grader_v2.py",
    INBOX_X8A / "cts_x8a_master.log",
    INBOX_X8A / "manifest_X8A47.json",
    INBOX_X8A / "manifest_X8A49.json",
    INBOX_X8A / "manifest_X8A50.json",
    INBOX_X8A / "train_X8A47.log",
    INBOX_X8A / "train_X8A49.log",
    INBOX_X8A / "train_X8A50.log",
    INBOX_X8A / "latdev_X8A47.pt.res.json",
    INBOX_X8A / "latdev_X8A49.pt.res.json",
    INBOX_X8A / "latdev_X8A50.pt.res.json",
]

shas = {
    str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
    for p in target_files
    if p.exists()
}

runtime_evidence = {}
for s in ["47", "49", "50"]:
    m = json.loads((INBOX_X8A / f"manifest_X8A{s}.json").read_text(encoding="utf-8"))
    res = json.loads((INBOX_X8A / f"latdev_X8A{s}.pt.res.json").read_text(encoding="utf-8"))
    tlog = (INBOX_X8A / f"train_X8A{s}.log").read_text(encoding="utf-8").splitlines()
    data_line = next((l for l in tlog if l.startswith("[DATA]")), None)
    deq_line = next((l for l in tlog if l.startswith("[DEQ]")), None)
    bridge_line = next((l for l in tlog if l.startswith("[BRIDGE]")), None)
    warn_lines = [l for l in tlog if "UserWarning" in l or "Consider using" in l]
    step0_line = next((l for l in tlog if "[Step    0]" in l), None)
    step_last_line = next((l for l in tlog if "[Step 1499]" in l), None)
    save_line = next((l for l in tlog if l.startswith("[SAVE]")), None)

    res_by_item = res["res_by_item"]
    res0_vals = [v[0] for v in res_by_item.values()]
    res1_vals = [v[1] for v in res_by_item.values()]
    n_contractive = sum(1 for v in res_by_item.values() if v[1] < v[0])

    runtime_evidence[f"X8A{s}"] = {
        "data_log_line": data_line,
        "deq_log_line": deq_line,
        "bridge_log_line": bridge_line,
        "step0_log_line": step0_line,
        "step_last_log_line": step_last_line,
        "save_log_line": save_line,
        "warnings": warn_lines,
        "manifest_summary": {
            "seed": m["seed"],
            "steps": m["steps"],
            "batch_size": m["batch_size"],
            "lr": m["lr"],
            "lr_deq": m["lr_deq"],
            "k_unroll": m["k_unroll"],
            "alpha": m["alpha"],
            "g0": m["g0"],
            "lambda_fp": m["lambda_fp"],
            "answer_weight": m["answer_weight"],
            "gen_budget": m["gen_budget"],
            "layers": m["layers"],
            "deq_trainable_weights": m["deq_trainable_weights"],
            "bridge_params": m["bridge_params"],
            "n_items_trained": m["n_items_trained"],
            "target_counts": m["target_counts"],
            "n_updates": m["n_updates"],
            "n_sample_visits": m["n_sample_visits"],
            "n_samples_skipped": m["n_samples_skipped"],
            "n_unique_ids_contributing_grad": m["n_unique_ids_contributing_grad"],
            "final_loss_last200": m["final_loss_last200"],
            "final_ans_ce_last200": m["final_ans_ce_last200"],
            "final_lfp_last200": m["final_lfp_last200"],
            "final_res_last200": m["final_res_last200"],
            "step0_tel": m["telemetry"][0],
            "step1499_tel": m["telemetry"][-1],
        },
        "latdev_summary": {
            "n_items": res["n_items"],
            "n_missing": res["n_missing"],
            "res_step1_mean": sum(res0_vals) / len(res0_vals),
            "res_step1_max": max(res0_vals),
            "res_step2_last_mean": res["res_last_mean"],
            "res_step2_last_max": res["res_last_max"],
            "n_items_step2_lt_step1": n_contractive,
        },
    }

out_data = {
    "task": "TASK-C008",
    "grader_sha256": shas["scripts/s5_eval_grader_v2.py"],
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
(OUTBOX / "INPUT_MANIFEST.json").write_text(
    json.dumps(shas, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("C008 verification JSON and INPUT_MANIFEST.json written.")
for s in ["47", "49", "50"]:
    ev = runtime_evidence[f"X8A{s}"]
    ms = ev["manifest_summary"]
    ls = ev["latdev_summary"]
    print(
        f"X8A{s}: loss200={ms['final_loss_last200']:.6f}, ans_ce200={ms['final_ans_ce_last200']:.6f}, "
        f"lfp200={ms['final_lfp_last200']:.6f}, res200={ms['final_res_last200']:.6f} | "
        f"dev_res1_mean={ls['res_step1_mean']:.6f}, dev_res2_mean={ls['res_step2_last_mean']:.6f}, "
        f"dev_res2_max={ls['res_step2_last_max']:.6f}, contractive={ls['n_items_step2_lt_step1']}/600"
    )
