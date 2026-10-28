import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\AI\cts")
INBOX = ROOT / "handoff/codex/INBOX/20261022-0015-x8_0-results"
OUTBOX = ROOT / "handoff/codex/OUTBOX/20261024-1906-C007"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


input_manifest = {
    "task": "TASK-C007",
    "inbox_dir": str(INBOX.relative_to(ROOT)).replace("\\", "/"),
    "files": {},
}

for line in (INBOX / "SHA256SUMS_full.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if len(parts) == 2:
        exp_sha, raw_name = parts[0], parts[1].lstrip("*")
        p = INBOX / raw_name
        act_sha = sha256_file(p) if p.exists() else None
        input_manifest["files"][raw_name] = {
            "expected_sha256": exp_sha,
            "actual_sha256": act_sha,
            "exists_locally": p.exists(),
            "match": (act_sha == exp_sha) if act_sha else None,
        }

extra_inputs = [
    ROOT / "scripts/s5_eval_grader_v2.py",
    ROOT / "handoff/codex/scripts/s5_eval_grader_v2.py",
    ROOT / "handoff/codex/scripts/codex_grade_local.py",
    ROOT / "handoff/codex/data/p6_item_order.json",
    ROOT / "handoff/codex/data/dev600_records.jsonl",
    ROOT / "configs/s7_train_993.json",
]
input_manifest["extra_inputs"] = {
    str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
    for p in extra_inputs
    if p.exists()
}

(OUTBOX / "INPUT_MANIFEST.json").write_text(
    json.dumps(input_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

output_files = [
    "REPORT-C007.md",
    "x8_0_items_local.json",
    "c007_analysis.json",
    "x8_0_analyze_server_utf8.txt",
    "x8_0_analyze_local_utf8.txt",
    "analyze_c007.py",
    "make_manifests_c007.py",
    "INPUT_MANIFEST.json",
]
output_manifest = {
    "task": "TASK-C007",
    "outbox_dir": str(OUTBOX.relative_to(ROOT)).replace("\\", "/"),
    "files": {
        fname: sha256_file(OUTBOX / fname)
        for fname in output_files
        if (OUTBOX / fname).exists()
    },
}
(OUTBOX / "OUTPUT_MANIFEST.json").write_text(
    json.dumps(output_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("C007 manifests written:", json.dumps(output_manifest["files"], indent=2))
