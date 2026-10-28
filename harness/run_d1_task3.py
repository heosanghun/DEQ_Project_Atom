import os
import sys
import torch
import json
import time

sys.path.append("/home/shoon/cts_e1")

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.types import NuVector, RuntimeBudgetState
from cts.model.gemma_loader import load_gemma4_e4b
from cts.deq.root_bootstrap import solve_root_z_star

def main():
    device = "cuda:0"
    jsonl_path = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"
    logs_dir = "/home/shoon/cts_e1/m1_matrix_logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    dataset = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    test_samples = dataset[:32]
    
    print("[TASK-3] Loading backbone model...", flush=True)
    mid = "/home/shoon/System2.5_deq/hf_cache/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
    model, tok = load_gemma4_e4b(model_id=mid, device_map=device, torch_dtype=torch.float32)
    bb = GemmaCTSBackbone(model, tok)
    bb.eval()
    
    ckpt_path = "/home/shoon/workspace/Cognitive-Tree-Search/artifacts/stage1_last.pt"
    print(f"[TASK-3] Loading OLD checkpoint from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("backbone_state_dict", ckpt)
    new_sd = {}
    for k, v in sd.items():
        if k == "routing_proj":
            new_sd["routing_proj.parametrizations.weight.original"] = v
        else:
            new_sd[k] = v
    bb.load_state_dict(new_sd, strict=False)
    
    nu = NuVector(nu_act=1.0e-3, nu_expl=1.0e-3, nu_tol=5e-3)
    
    print("[TASK-3] Precalculating z0 with old ckpt...", flush=True)
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
        
    iters = [600, 1000]
    for max_iter in iters:
        print(f"\n--- Running Task 3 (Old Ckpt): c=0.92, max_iter={max_iter} ---", flush=True)
        converged_count = 0
        log_file_path = os.path.join(logs_dir, f"m1_matrix_cell_oldckpt_0.92_{max_iter}.log")
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
                    
        print(f"Old Ckpt Cell c=0.92, max_iter={max_iter}: {converged_count}/32 Converged", flush=True)

if __name__ == "__main__":
    main()
