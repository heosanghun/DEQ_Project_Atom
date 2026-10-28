import os
import sys
import gc
import json
import torch

sys.path.insert(0, "/home/shoon/cts_e1")

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.model.gemma_loader import load_gemma4_e4b
from cts.types import NuVector, RuntimeBudgetState

def measure_vram_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def reset_peak():
    torch.cuda.reset_peak_memory_stats()

def main():
    device = torch.device("cuda:0")
    print("Loading Gemma 4 E4B for VRAM profiling...")
    model, tok = load_gemma4_e4b(device_map="cuda:0", torch_dtype=torch.bfloat16)
    backbone = GemmaCTSBackbone(model, tok)
    backbone.eval()
    
    depths = [1, 15, 35, 70, 100]
    cts_results = {}
    nu = NuVector(nu_tol=5e-3, nu_temp=1.0, nu_expl=1.0)
    
    parent_z = None
    current_depth = 0
    
    for target_depth in sorted(depths):
        while current_depth < target_depth:
            budget = RuntimeBudgetState()
            reset_peak()
            
            # Radix prefix sharing & node deallocation
            with torch.no_grad():
                r = transition(
                    f"Depth {current_depth + 1} problem: What is {current_depth + 1} * 7?",
                    current_depth,
                    nu,
                    budget,
                    backbone,
                    K=8,
                    d=backbone.hidden_size,
                    broyden_max_iter=20,
                    broyden_tol_min=1e-2,
                    broyden_tol_max=5e-2,
                    tau_flops_budget=1e20,
                    parent_z_star=parent_z,
                )
            
            if r.z_star_child is not None:
                parent_z = r.z_star_child.detach().clone()
            current_depth += 1
            
            del r
            gc.collect()
            torch.cuda.empty_cache()
            
        peak_mb = measure_vram_mb()
        peak_gb = peak_mb / 1024.0
        cts_results[target_depth] = round(peak_gb, 2)
        print(f"VRAM profile depth {target_depth}: {peak_gb:.2f} GB")
        
    out_path = "/home/shoon/cts_e1/harness/vram_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cts_results, f, indent=2)
    print(f"Profiling complete. Saved to {out_path}")

if __name__ == "__main__":
    main()
