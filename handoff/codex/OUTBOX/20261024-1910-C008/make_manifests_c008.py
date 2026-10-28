import hashlib
import json
from pathlib import Path

OUTBOX = Path(r"D:\AI\cts\handoff\codex\OUTBOX\20261024-1910-C008")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out_files = [
    "REPORT-C008.md",
    "c008_static_verification.json",
    "verify_c008.py",
    "verify_c008_v2.py",
    "INPUT_MANIFEST.json",
]

manifest = {f: sha256_file(OUTBOX / f) for f in out_files if (OUTBOX / f).exists()}
(OUTBOX / "OUTPUT_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
