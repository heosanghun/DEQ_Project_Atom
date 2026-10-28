import json
import time
import hashlib

def main():
    f = "/home/shoon/cts_e1/harness/state.json"
    chain_log = "/home/shoon/cts_e1/harness/state_hash_chain.log"
    
    with open(f, "r", encoding="utf-8") as jf:
        s = json.load(jf)
        
    s["nodes"]["N1"] = {
        "status": "PENDING",
        "detail": "Retrying N1 with corrected 600s mtime tolerance",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    with open(f, "w", encoding="utf-8") as jf:
        json.dump(s, jf, indent=2, sort_keys=True)
        
    h = hashlib.sha256(open(f, "rb").read()).hexdigest().lower()
    
    prev = ""
    with open(chain_log, "r", encoding="utf-8") as clf:
        lines = clf.readlines()
        if lines:
            prev = lines[-1].split("self:")[-1].strip()
            
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(chain_log, "a", encoding="utf-8") as clf:
        clf.write(f"[{ts}] | node:N1 -> PENDING (auto retry) | prev:{prev} | self:{h}\n")
        
    print(f"Reset N1 to PENDING. self:{h}")

if __name__ == "__main__":
    main()
