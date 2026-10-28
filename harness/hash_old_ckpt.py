import hashlib
import os
import sys

def main():
    p = "/home/shoon/workspace/Cognitive-Tree-Search/artifacts/stage1_last.pt"
    if not os.path.exists(p):
        print(f"File not found: {p}")
        sys.exit(1)
        
    size = os.path.getsize(p)
    mtime = os.path.getmtime(p)
    
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha = h.hexdigest().lower()
    
    out_path = "/home/shoon/cts_e1/old_ckpt_manifest.txt"
    manifest_content = f"path: {p}\nsize: {size}\nmtime: {mtime}\nsha256: {sha}\n"
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(manifest_content)
        
    print(manifest_content)

if __name__ == "__main__":
    main()
