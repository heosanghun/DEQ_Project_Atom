import json
import os

def main():
    log_path = "/home/shoon/cts_e1/m1_matrix_logs/m1_matrix_cell_0.92_1000.log"
    out_path = "/home/shoon/cts_e1/m1_d1_residuals_092_1000.json"
    
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    records = []
    for line in lines:
        if line.strip():
            # Format: Sample 00: Converged=True, Residual=0.004855
            parts = line.strip().split(":")
            sid_str = parts[0].replace("Sample", "").strip()
            sample_id = int(sid_str)
            
            rest = parts[1]
            conv_str = rest.split(",")[0].split("=")[1].strip()
            converged = (conv_str == "True")
            
            res_str = rest.split("Residual=")[1].strip()
            residual = float(res_str)
            
            records.append({
                "sample_id": sample_id,
                "converged": converged,
                "final_residual": residual
            })
            
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(records, out, indent=2)
        
    print(f"Extracted {len(records)} records to {out_path}")

if __name__ == "__main__":
    main()
