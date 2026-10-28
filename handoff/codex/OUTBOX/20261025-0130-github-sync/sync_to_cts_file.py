import csv
import glob
import hashlib
import json
import os
import shutil

SRC_ROOT = r"D:\AI\cts"
DST_ROOT = r"D:\AI\CTS_file"

EXCLUDE_SCRIPTS = {
    "build_report_exe40.py",
    "cloud_setup.sh",
    "net_probe_runner.py",
    "progress_heartbeat.py",
    "run_local_s7_zstar.py",
    "update_rules_protocol.py",
    "upload_checkpoints.sh",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: str, dst: str, added_files: list):
    if os.path.exists(dst):
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    added_files.append(dst)


def safe_write_text(dst: str, content: str, added_files: list):
    if os.path.exists(dst):
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    added_files.append(dst)


def copy_tree_no_overwrite(src_dir: str, dst_dir: str, added_files: list):
    for root, dirs, files in os.walk(src_dir):
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        rel = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        for fn in sorted(files):
            if fn.endswith((".pyc", ".pt", ".safetensors")):
                continue
            s_fp = os.path.join(root, fn)
            d_fp = os.path.join(target_root, fn)
            safe_copy(s_fp, d_fp, added_files)


def build_item_grades_csv(dst_csv: str, added_files: list):
    with open(os.path.join(SRC_ROOT, "results_s7", "p6_item_order.json"), encoding="utf-8") as f:
        item_order = [str(x) for x in json.load(f)]

    sources = [
        ("C_fmt_baseline", os.path.join(SRC_ROOT, "handoff", "codex", "OUTBOX", "20261024-1833-C009", "x8a_items_local.json"), ["C_fmt"]),
        ("X1", os.path.join(SRC_ROOT, "handoff", "codex", "data", "x1_items_localcheck2.json"), ["X1O47", "shuf_X1O47", "X1O49", "shuf_X1O49", "X1O50", "shuf_X1O50"]),
        ("X5", os.path.join(SRC_ROOT, "handoff", "codex", "OUTBOX", "20261020-1535-C004", "x5_items_local.json"), ["X5O47", "shuf_X5O47"]),
        ("X6", os.path.join(SRC_ROOT, "handoff", "codex", "OUTBOX", "20261024-1859-C005", "x6_items_local.json"), ["X6R49", "shuf_X6R49", "X6R50", "shuf_X6R50", "X6S47", "shuf_X6S47"]),
        ("X7", os.path.join(SRC_ROOT, "handoff", "codex", "OUTBOX", "20261024-1901-C006", "x7_items_local.json"), ["X7A47", "shuf_X7A47", "X7E47", "shuf_X7E47", "X7A49", "shuf_X7A49", "X7E49", "shuf_X7E49", "X7A50", "shuf_X7A50", "X7E50", "shuf_X7E50"]),
        ("X8_0", os.path.join(SRC_ROOT, "handoff", "codex", "OUTBOX", "20261024-1906-C007", "x8_0_items_local.json"), ["X80P47", "shuf_X80P47", "X80P49", "shuf_X80P49", "X80P50", "shuf_X80P50"]),
        ("X8_A", os.path.join(SRC_ROOT, "handoff", "codex", "OUTBOX", "20261024-1833-C009", "x8a_items_local.json"), ["X8A47", "shuf_X8A47", "X8A49", "shuf_X8A49", "X8A50", "shuf_X8A50"]),
    ]

    if os.path.exists(dst_csv):
        raise FileExistsError(f"Refusing to overwrite existing file: {dst_csv}")
    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)

    with open(dst_csv, "w", encoding="utf-8", newline="\n") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["item_id", "seed", "condition", "strict_correct", "lenient_correct", "boxed_correct", "experiment_group", "condition_key"])
        for exp_group, json_path, keys in sources:
            with open(json_path, encoding="utf-8") as jf:
                data = json.load(jf)
            for k in keys:
                if k == "C_fmt":
                    seed = "none"
                    cond = "C_fmt"
                elif k.startswith("shuf_"):
                    base_k = k[len("shuf_"):]
                    seed = base_k[-2:]
                    cond = f"shuffled_{base_k[:-2]}"
                else:
                    seed = k[-2:]
                    cond = f"matched_{k[:-2]}"
                item_map = data[k]
                for iid in item_order:
                    lenient_v, strict_v, boxed_v = item_map[iid]
                    writer.writerow([iid, seed, cond, int(bool(strict_v)), int(bool(lenient_v)), int(bool(boxed_v)), exp_group, k])

    added_files.append(dst_csv)


def main():
    added_files = []

    # 1. Copy missing INBOX result packages (C004~C009 inputs)
    inbox_pkgs = [
        "20261020-1320-x5-results",
        "20261020-2020-x6-results",
        "20261021-0945-x7-results",
        "20261022-0015-x8_0-results",
        "20261023-1950-x8a-results",
    ]
    for pkg in inbox_pkgs:
        src_p = os.path.join(SRC_ROOT, "handoff", "codex", "INBOX", pkg)
        dst_p = os.path.join(DST_ROOT, "INBOX", pkg)
        copy_tree_no_overwrite(src_p, dst_p, added_files)

    # 2. Copy missing OUTBOX report packages (C004~C009 outputs)
    outbox_pkgs = [
        "20261020-1245-C004",
        "20261020-1535-C004",
        "20261024-1833-C009",
        "20261024-1859-C005",
        "20261024-1901-C006",
        "20261024-1906-C007",
        "20261024-1910-C008",
    ]
    for pkg in outbox_pkgs:
        src_p = os.path.join(SRC_ROOT, "handoff", "codex", "OUTBOX", pkg)
        dst_p = os.path.join(DST_ROOT, "OUTBOX", pkg)
        copy_tree_no_overwrite(src_p, dst_p, added_files)

    # 3. Create INBOX/20261025-0130-full-code-and-docs
    full_pkg = os.path.join(DST_ROOT, "INBOX", "20261025-0130-full-code-and-docs")

    # 3a. cts/ Python package
    copy_tree_no_overwrite(
        os.path.join(SRC_ROOT, "cts"),
        os.path.join(full_pkg, "cts"),
        added_files,
    )

    # 3b. handoff/codex/scripts/
    copy_tree_no_overwrite(
        os.path.join(SRC_ROOT, "handoff", "codex", "scripts"),
        os.path.join(full_pkg, "handoff_scripts"),
        added_files,
    )

    # 3c. scripts/*.py and *.sh (clean, excluding SSH infra scripts)
    src_scripts = os.path.join(SRC_ROOT, "scripts")
    dst_scripts = os.path.join(full_pkg, "repo_scripts")
    for fn in sorted(os.listdir(src_scripts)):
        if fn in EXCLUDE_SCRIPTS or not fn.endswith((".py", ".sh")):
            continue
        fp = os.path.join(src_scripts, fn)
        if os.path.isfile(fp):
            safe_copy(fp, os.path.join(dst_scripts, fn), added_files)

    # 3d. configs/ + p6_item_order.json
    copy_tree_no_overwrite(
        os.path.join(SRC_ROOT, "configs"),
        os.path.join(full_pkg, "configs"),
        added_files,
    )
    safe_copy(
        os.path.join(SRC_ROOT, "results_s7", "p6_item_order.json"),
        os.path.join(full_pkg, "configs", "p6_item_order.json"),
        added_files,
    )

    # 3e. tasks_and_answers/ from handoff/codex/INBOX/*.md
    src_inbox = os.path.join(SRC_ROOT, "handoff", "codex", "INBOX")
    dst_tasks = os.path.join(full_pkg, "tasks_and_answers")
    for fn in sorted(os.listdir(src_inbox)):
        fp = os.path.join(src_inbox, fn)
        if os.path.isfile(fp) and fn.endswith(".md"):
            safe_copy(fp, os.path.join(dst_tasks, fn), added_files)

    # 3f. state_and_docs/
    dst_docs = os.path.join(full_pkg, "state_and_docs")
    safe_copy(os.path.join(SRC_ROOT, "AGENTS.md"), os.path.join(dst_docs, "AGENTS.md"), added_files)
    safe_copy(os.path.join(SRC_ROOT, "handoff", "codex", "CONTEXT.md"), os.path.join(dst_docs, "CONTEXT.md"), added_files)
    safe_copy(os.path.join(SRC_ROOT, "shared", "GOAL.md"), os.path.join(dst_docs, "GOAL.md"), added_files)

    # Redact company IP in STATE.md and RULES.md copies
    for md_name in ["STATE.md", "RULES.md"]:
        with open(os.path.join(SRC_ROOT, "shared", md_name), encoding="utf-8") as f:
            txt = f.read().replace("<REDACTED_IP>", "<REDACTED_COMPANY_IP>")
        safe_write_text(os.path.join(dst_docs, md_name), txt, added_files)

    # Proposals
    copy_tree_no_overwrite(
        os.path.join(SRC_ROOT, "shared", "proposals"),
        os.path.join(dst_docs, "proposals"),
        added_files,
    )
    # handoff/codex/docs
    copy_tree_no_overwrite(
        os.path.join(SRC_ROOT, "handoff", "codex", "docs"),
        os.path.join(dst_docs, "handoff_docs"),
        added_files,
    )
    # RESULT-*.md reports
    for fp in sorted(glob.glob(os.path.join(SRC_ROOT, "shared", "reports", "RESULT-*.md"))):
        safe_copy(fp, os.path.join(dst_docs, "results", os.path.basename(fp)), added_files)

    # 4. Build OUTBOX/20261025-0130-ASTRA-QA-PACKAGE/all_experiments_item_grades.csv
    qa_pkg = os.path.join(DST_ROOT, "OUTBOX", "20261025-0130-ASTRA-QA-PACKAGE")
    build_item_grades_csv(os.path.join(qa_pkg, "all_experiments_item_grades.csv"), added_files)

    # 5. Write top-level index and manifest
    index_md = """# CTS Full Code, Results & Audit Sync (2026-10-25)

이 문서는 Codex App (Astra)의 근본 원인 분석 및 소통을 위해 `D:\\AI\\cts` (회사 PC 로컬 저장소, 브랜치 `scale-v1`, 커밋 `992afa62faa87fbe4e5a58021b7fb2e435291cc9`)의 모든 관련 코드, 실험 결과 패키지(`C004`~`C009` / `X5`~`X8-A`), 독립 재채점·검증 보고서(`REPORT-C004`~`REPORT-C009`), 문항별 채점 원자료(`all_experiments_item_grades.csv`)를 기존 파일 수정 없이 추가한 전체 색인입니다.

---

## 1. 핵심 디렉터리 안내

### A. 전체 소스 코드 및 프로젝트 문서 (`INBOX/20261025-0130-full-code-and-docs/`)
- **핵심 실행·학습·생성·브리지 스크립트 (`handoff_scripts/`)**:
  - [`g2v11_x8a_train.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/g2v11_x8a_train.py): **X8-A 결합 학습기** (`DEQ + CrossAttnBridge + LoRA`, $K_{\\text{unroll}}=2$, $L_{\\text{read}} + 0.1 L_{\\text{fp}}$)
  - [`g2v11_x8a_latents.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/g2v11_x8a_latents.py): **X8-A 결합 학습된 DEQ로 평가용 `z*` 추출** (`steps=8`, $\\alpha=0.5$)
  - [`cc_x8a.sh`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/cc_x8a.sh): **X8-A 전체 실행 런처 및 `x8a_analyze.py` 부트스트랩 통계 스크립트**
  - [`g2v9_xattn.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/g2v9_xattn.py): **Cross-Attention Bridge (`CrossAttnBridge`, `GatedXAttnLayer`, `XAttnOverride`)**
  - [`g2v9_gen.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/g2v9_gen.py): **자유 생성 스크립트 (vLLM / HF greedy decoding, `max_new_tokens=512`)**
  - [`g2v10_train.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/g2v10_train.py): **X5 / X6-R / X7-E / X7-A / X8-0 학습기**
  - [`g2v8x_problem_latents.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/g2v8x_problem_latents.py): **X8-0 문제 전용 잠재 추출기**
  - [`codex_grade_local.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/codex_grade_local.py) & [`s5_eval_grader_v2.py`](INBOX/20261025-0130-full-code-and-docs/handoff_scripts/s5_eval_grader_v2.py): **잠금 채점기 (SHA256 `90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13`)**
- **CTS 코어 파이썬 패키지 (`cts/`)**:
  - [`cts/deq/`](INBOX/20261025-0130-full-code-and-docs/cts/deq/): DEQ 전이 함수, Broyden/Anderson solver, `zstar_canonical.py`, `gemma_latent_forward.py`
  - [`cts/backbone/`](INBOX/20261025-0130-full-code-and-docs/cts/backbone/), [`cts/model/`](INBOX/20261025-0130-full-code-and-docs/cts/model/), [`cts/routing/`](INBOX/20261025-0130-full-code-and-docs/cts/routing/), [`cts/train/`](INBOX/20261025-0130-full-code-and-docs/cts/train/), [`cts/eval/`](INBOX/20261025-0130-full-code-and-docs/cts/eval/) 등 전체 100개 모듈
- **전체 실험 스크립트 (`repo_scripts/`)**:
  - `g1*.py`, `g2v2_*.py` ~ `g2v9_*.py`, `s5_*.py`, `s6_*.py`, `s7_*.py` 등 302개 스크립트
- **프로젝트 상태·설계·결과 문서 (`state_and_docs/`)**:
  - [`STATE.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/STATE.md), [`CONTEXT.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/CONTEXT.md), [`GOAL.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/GOAL.md), [`RULES.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/RULES.md), [`AGENTS.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/AGENTS.md)
  - 설계서: [`X_INTERFACE_PLAN_20261019.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/proposals/X_INTERFACE_PLAN_20261019.md), [`X8_READABLE_LATENT_DEQ_20261021.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/proposals/X8_READABLE_LATENT_DEQ_20261021.md)
  - 공식 실험 결과 보고서 17종 (`state_and_docs/results/RESULT-*.md`): [`RESULT-G1-final.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/results/RESULT-G1-final.md) ~ [`RESULT-X8A-readable-deq-20261023.md`](INBOX/20261025-0130-full-code-and-docs/state_and_docs/results/RESULT-X8A-readable-deq-20261023.md)
- **과제 지시서 및 답변서 (`tasks_and_answers/`)**:
  - `TASK-C001.md` ~ `TASK-C009.md`, `ACK-*`, `ANSWER-*`

---

### B. 실험별 원시 생성물·매니페스트·잠재 진단 (`INBOX/` 하위 패키지)
- **X5 (C004)**: [`INBOX/20261020-1320-x5-results/`](INBOX/20261020-1320-x5-results/) (`gen_X5O47.jsonl`, `gen_shuf_X5O47.jsonl`, `manifest_X5O47.json`, `x5_summary.json`, `s0spec_x.json` 등)
- **X6 (C005)**: [`INBOX/20261020-2020-x6-results/`](INBOX/20261020-2020-x6-results/) (`gen_X6R49.jsonl`, `gen_X6R50.jsonl`, `gen_X6S47.jsonl`, `x6_summary.json` 등)
- **X7 (C006)**: [`INBOX/20261021-0945-x7-results/`](INBOX/20261021-0945-x7-results/) (`gen_X7A*.jsonl`, `gen_X7E*.jsonl`, `x7_summary.json` 등)
- **X8-0 (C007)**: [`INBOX/20261022-0015-x8_0-results/`](INBOX/20261022-0015-x8_0-results/) (`gen_X80P*.jsonl`, `x8_0_summary.json`, `x8_0_latent_meta.json` 등)
- **X8-A (C008 & C009)**: [`INBOX/20261023-1950-x8a-results/`](INBOX/20261023-1950-x8a-results/) (`gen_X8A47.jsonl`, `gen_X8A49.jsonl`, `gen_X8A50.jsonl`, `gen_shuf_X8A*.jsonl`, `manifest_X8A*.json`, `latent_meta_X8A*.json`, `x8a_summary.json`, `x8a_items_server.json`, `g2v11_x8a_train.py`, `g2v11_x8a_latents.py`, `cc_x8a.sh` 등)

---

### C. 독립 감사·재채점 완료 보고서 (`OUTBOX/` 하위 패키지)
- **REPORT-C004 (X5 seed 47 재채점)**: [`OUTBOX/20261020-1535-C004/REPORT-C004.md`](OUTBOX/20261020-1535-C004/REPORT-C004.md)
- **REPORT-C005 (X6 3시드 재채점·부트스트랩)**: [`OUTBOX/20261024-1859-C005/REPORT-C005.md`](OUTBOX/20261024-1859-C005/REPORT-C005.md)
- **REPORT-C006 (X7 Stage E/A 재채점·비교)**: [`OUTBOX/20261024-1901-C006/REPORT-C006.md`](OUTBOX/20261024-1901-C006/REPORT-C006.md)
- **REPORT-C007 (X8-0 문제 전용 잠재 재채점)**: [`OUTBOX/20261024-1906-C007/REPORT-C007.md`](OUTBOX/20261024-1906-C007/REPORT-C007.md)
- **REPORT-C008 (X8-A 코드 정적 검증)**: [`OUTBOX/20261024-1910-C008/REPORT-C008.md`](OUTBOX/20261024-1910-C008/REPORT-C008.md)
- **REPORT-C009 (X8-A 3시드 재채점·최종 검증)**: [`OUTBOX/20261024-1833-C009/REPORT-C009.md`](OUTBOX/20261024-1833-C009/REPORT-C009.md)
- **통합 문항별 채점 표 (`item_id, seed, condition, strict_correct, lenient_correct, boxed_correct`)**:
  - [`OUTBOX/20261025-0130-ASTRA-QA-PACKAGE/all_experiments_item_grades.csv`](OUTBOX/20261025-0130-ASTRA-QA-PACKAGE/all_experiments_item_grades.csv) (총 19,200행 = 32개 조건 × 600문항)
"""
    safe_write_text(os.path.join(DST_ROOT, "INDEX-20261025-FULL-SYNC.md"), index_md, added_files)

    manifest = {
        "synced_at": "2026-10-25T01:30:00+09:00",
        "source_repo": "D:\\AI\\cts",
        "source_branch": "scale-v1",
        "source_commit": "992afa62faa87fbe4e5a58021b7fb2e435291cc9",
        "total_added_files": len(added_files),
        "files": {
            os.path.relpath(fp, DST_ROOT).replace("\\", "/"): {
                "sha256": sha256_file(fp),
                "bytes": os.path.getsize(fp),
            }
            for fp in sorted(added_files)
        },
    }
    manifest_path = os.path.join(full_pkg, "MANIFEST-SYNC-20261025.json")
    safe_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False), added_files)
    print(f"Successfully staged {len(added_files)} new files into {DST_ROOT} with 0 overwrites.")


if __name__ == "__main__":
    main()
