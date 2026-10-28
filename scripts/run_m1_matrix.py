import os
import sys
import torch
import json
import time
from pathlib import Path

# Add repo root to path
sys.path.append("/home/shoon/cts_e1")

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition, _routing_sparse
from cts.types import NuVector, RuntimeBudgetState
from cts.model.gemma_loader import load_gemma4_e4b
from cts.deq.root_bootstrap import solve_root_z_star

def run_matrix():
    device = "cuda:0"
    jsonl_path = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"
    logs_dir = "/home/shoon/cts_e1/m1_matrix_logs"
    results_json = "/home/shoon/cts_e1/m1_matrix_results.json"
    
    os.makedirs(logs_dir, exist_ok=True)
    
    dataset = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    test_samples = dataset[:32]
    
    print("[M1-MATRIX] Loading backbone model (Base)...")
    base_snapshot = '/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377'
    mid = base_snapshot
    assert "58c4d7b5" in str(mid), f"[BACKBONE LOCK VIOLATION] expected base snapshot (58c4d7b5), got {mid}"
    print(f"[BACKBONE-EVIDENCE] loading from: {mid}", flush=True)
    model, tok = load_gemma4_e4b(model_id=mid, device_map=device, torch_dtype=torch.float32)
    bb = GemmaCTSBackbone(model, tok)
    bb.eval()
    
    ckpt_path = "/home/shoon/retrain/ckpt/stage1_last.pt"
    if not os.path.exists(ckpt_path):
         print(f"Error: checkpoint {ckpt_path} not found.")
         sys.exit(1)
         
    print(f"[M1-MATRIX] Loading checkpoints from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("backbone_state_dict", ckpt)
    new_sd = {}
    for k, v in sd.items():
        if k == "routing_proj":
            new_sd["routing_proj.parametrizations.weight.original"] = v
        else:
            new_sd[k] = v
    bb.load_state_dict(new_sd, strict=False)
    
    nu = NuVector(nu_act=1.0e-3, nu_expl=1.0e-3, nu_tol=5e-3)  # Explicitly enforce tol=5e-3 (‖F(z)-z‖/‖z‖)
    
    # Precalculate z0
    print("[M1-MATRIX] Precalculating z0...")
    z0_list = []
    ctx_list = []
    prompts_list = []
    for idx, row in enumerate(test_samples):
        prompt = row.get("prompt")
        prompts_list.append(prompt)
        with torch.no_grad():
            ctx = bb.encode_context(prompt)
        ctx = ctx.to(device=device, dtype=torch.float32)
        ctx_list.append(ctx)
        
        root_result = solve_root_z_star(
            prompt,
            backbone=bb,
            context=ctx,
            K=64,
            d=2560,
            broyden_max_iter=30,
            broyden_tol_min=1e-4,
            broyden_tol_max=1e-2,
            top_k=3,
            tau_flops_budget=1e14,
            seed=2026 + idx,
            device=device,
            require_converged=True,
        )
        z0 = root_result.z_star_child.detach().clone().to(dtype=torch.float32)
        z0_list.append(z0)
        
    # Grid search configs: Task S1 (3000, 4000)
    configs = [
        ("0.92", 3000),
        ("0.92", 4000)
    ]
    
    original_deq_step = bb.deq_step
    gamma = torch.sigmoid(bb.gamma_param).to(device=device, dtype=torch.float32)
    
    def make_patched_deq_step(target_scale: float):
        def patched_deq_step(z, context, module_weights, extra):
            ds_0 = original_deq_step(z, context, module_weights, extra)
            ratio = target_scale / 0.92
            ds_new = ratio * (ds_0 - (1.0 - gamma) * z) + (1.0 - gamma) * z
            return ds_new.to(dtype=z.dtype)
        return patched_deq_step

    results_data = {}
    
    for c_str, max_iter in configs:
        target_c = float(c_str)
        print(f"\n--- Running Cell: c={target_c}, max_iter={max_iter} ---")
        
        # Apply patch if c is not 0.92 (original)
        if target_c != 0.92:
            bb.deq_step = make_patched_deq_step(target_c)
        else:
            bb.deq_step = original_deq_step
            
        converged_count = 0
        total_count = 32
        
        log_file_path = os.path.join(logs_dir, f"m1_matrix_cell_base_{c_str}_{max_iter}.log")
        with open(log_file_path, "w", encoding="utf-8") as lf:
            for idx in range(32):
                prompt = prompts_list[idx]
                ctx = ctx_list[idx]
                z0 = z0_list[idx]
                
                budget = RuntimeBudgetState()
                with torch.no_grad():
                    tr = transition(
                        prompt,
                        0,
                        nu,
                        budget,
                        bb,
                        K=64,
                        d=2560,
                        broyden_max_iter=max_iter,
                        broyden_tol_min=1e-4,
                        broyden_tol_max=1e-2,
                        tau_flops_budget=1e14,
                        max_decode_tokens=64,
                        parent_z_star=z0,
                    )
                converged = tr.solver_stats.get("converged", False)
                res_norm = tr.solver_stats.get("residual_norm", 1e9)
                
                lf.write(f"Sample {idx:02d}: Converged={converged}, Residual={res_norm:.6f}\n")
                lf.flush()
                if converged:
                    converged_count += 1
                    
        print(f"Cell c={target_c}, max_iter={max_iter}: {converged_count}/32 Converged", flush=True)
        
        # Update results JSON
        if c_str not in results_data:
            results_data[c_str] = {}
        results_data[c_str][str(max_iter)] = {
            "conv_count": converged_count,
            "total_count": total_count,
            "conv_rate": round(converged_count / total_count, 4)
        }
        
    # Write summary JSON
    with open(results_json, "w", encoding="utf-8") as rf:
        json.dump(results_data, rf, indent=2)
        
    print(f"\n[M1-MATRIX] Saved matrix summary to {results_json}")

if __name__ == "__main__":
    run_matrix()
