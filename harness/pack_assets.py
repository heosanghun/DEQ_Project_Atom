import os
import sys
import json
import hashlib

def sha256_file(p: str) -> str:
    if not os.path.exists(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()

def main():
    print("Starting N9 pack_assets...")
    
    # Generate MANIFEST_CTS_M1M6.json containing hashes of critical files
    files_to_pack = [
        "/home/shoon/cts_e1/SELFCHECK_LEDGER.jsonl",
        "/home/shoon/cts_e1/harness/state.json",
        "/home/shoon/cts_e1/harness/state_hash_chain.log",
        "/home/shoon/cts_e1/harness/vram_results.json"
    ]
    
    manifest = {
        "manifest_version": "1.0",
        "timestamp": os.popen("date").read().strip(),
        "files": {}
    }
    
    for f in files_to_pack:
        if os.path.exists(f):
            manifest["files"][f] = {
                "size": os.path.getsize(f),
                "mtime": os.path.getmtime(f),
                "sha256": sha256_file(f)
            }
            
    out_path = "/home/shoon/cts_e1/MANIFEST_CTS_M1M6.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print("Manifest generated")

if __name__ == "__main__":
    main()
