import os
import sys
import json
import torch
from pathlib import Path

sys.path.append('/home/shoon/cts_e1')

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.types import NuVector, RuntimeBudgetState
from cts.model.gemma_loader import load_gemma4_e4b
from cts.deq.root_bootstrap import solve_root_z_star
from cts.deq.broyden_forward import _relative_residual, _limited_memory_broyden_batch
from cts.routing.sparse_moe_ref import routing_weights
from cts.deq.hybrid_solver import solve_hybrid

def _routing_sparse(z, w_g, temp, top_k):
    alpha = routing_weights(z, w_g, temp)
    vals, idxs = torch.topk(alpha, top_k, dim=-1)
    mask = torch.zeros_like(alpha).scatter_(-1, idxs, 1.0)
    normed = (alpha * mask) / (alpha * mask).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return normed

def main():
    device = 'cuda:0'
    jsonl_path = '/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl'
    results_dir = '/home/shoon/cts_e1/results'
    traj_dir = '/home/shoon/cts_e1/trajectories'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    
    # 1. Load dataset (first 32 samples)
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    test_samples = dataset[:32]
                    
    # 2. Load Base Backbone
    base_snapshot = '/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377'
    assert '58c4d7b5' in str(base_snapshot), f"[BACKBONE LOCK VIOLATION] expected base snapshot (58c4d7b5), got {base_snapshot}"
    print(f'[BACKBONE-EVIDENCE] loading from: {base_snapshot}', flush=True)
    model, tok = load_gemma4_e4b(model_id=base_snapshot, device_map=device, torch_dtype=torch.float32)
    bb = GemmaCTSBackbone(model, tok)
    bb.eval()
    
    # Load checkpoint
    ckpt_path = '/home/shoon/retrain/ckpt/stage1_last.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt.get('backbone_state_dict', ckpt)
    new_sd = {}
    for k, v in sd.items():
        if k == 'routing_proj':
            new_sd['routing_proj.parametrizations.weight.original'] = v
        else:
            new_sd[k] = v
    bb.load_state_dict(new_sd, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}", flush=True)
    
    # Apply c=0.92 scaling
    original_deq_step = bb.deq_step
    def scaled_deq_step(z, context, module_weights, extra):
        return 0.92 * original_deq_step(z, context, module_weights, extra)
    bb.deq_step = scaled_deq_step
    
    nu = NuVector(nu_act=1.0e-3, nu_expl=1.0e-3, nu_tol=5e-3)
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    routing_mode = "sparse"
    
    print("\n--- Precalculating root z0 for 32 samples ---", flush=True)
    ctx_list = []
    z0_list = []
    prompts_list = []
    for idx, row in enumerate(test_samples):
        prompt = row.get('prompt')
        prompts_list.append(prompt)
        with torch.no_grad():
            ctx = bb.encode_context(prompt).to(device=device, dtype=torch.float32)
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
        
    # ==========================================
    # Task V3: Hybrid Solves for budget 1000 & 2000
    # ==========================================
    budgets = [1000, 2000]
    for b in budgets:
        log_file = os.path.join(results_dir, f"m1_matrix_cell_hybrid_2x{b}_0.92.log")
        print(f"\n======================================================", flush=True)
        print(f"Task V3: 32 samples Hybrid evaluation (budget_each={b}, max_total=2x{b}, tol=5e-3)", flush=True)
        print(f"Log path: {log_file}", flush=True)
        print(f"======================================================", flush=True)
        
        passed_count = 0
        log_lines = []
        
        for idx in range(32):
            ctx = ctx_list[idx]
            z0 = z0_list[idx]
            
            def phi(zz):
                alpha = routing_weights(zz, w_g, nu.nu_temp)
                mw = alpha if routing_mode == "dense" else _routing_sparse(zz, w_g, nu.nu_temp, 3)
                return bb.deq_step(zz, ctx, mw, {'top_k': 3})
                
            with torch.no_grad():
                final_z, h_info = solve_hybrid(
                    phi=phi,
                    z0=z0,
                    budget_total=b,
                    tol=5e-3,
                    memory_limit=16,
                    is_root=False,
                )
                
            is_conv = h_info.converged
            if is_conv:
                passed_count += 1
                
            line = (
                f"Sample {idx:02d}: Converged={is_conv}, Residual={h_info.residual_norm:.6f}, "
                f"Solver={h_info.selected_solver}, BroydenIters={h_info.broyden_iters}, "
                f"SimpleIters={h_info.simple_iters}, TotalEvals={h_info.total_evals}"
            )
            log_lines.append(line)
            print(line, flush=True)
            
        with open(log_file, "w", encoding="utf-8") as f:
            for l in log_lines:
                f.write(l + "\n")
                
        print(f"\n[Task V3 Result budget=2x{b}] {passed_count}/32 ({passed_count/32*100:.2f}%) passed.", flush=True)

    # ==========================================
    # Task V5: Broyden Upper Limit Probe (#05, #27 max_iter=8000)
    # ==========================================
    print(f"\n======================================================", flush=True)
    print("Task V5: Broyden Upper Limit Probe for #05 and #27 (max_iter=8000)", flush=True)
    print(f"======================================================", flush=True)
    
    target_v5_indices = [5, 27]
    for idx in target_v5_indices:
        prompt = prompts_list[idx]
        ctx = ctx_list[idx]
        z0 = z0_list[idx]
        
        print(f"\n[Task V5] Running Broyden max_iter=8000 on Sample {idx:02d}...", flush=True)
        
        def phi(zz):
            alpha = routing_weights(zz, w_g, nu.nu_temp)
            mw = alpha if routing_mode == "dense" else _routing_sparse(zz, w_g, nu.nu_temp, 3)
            return bb.deq_step(zz, ctx, mw, {'top_k': 3})
            
        with torch.no_grad():
            z_res, infos = _limited_memory_broyden_batch(
                phi=phi,
                z0_batch=z0.unsqueeze(0) if z0.dim() == 2 else z0,
                tol=5e-3,
                max_iter=8000,
                parent_inv_jacobian=None,
                fp32_buffer=True,
                memory_limit=16,
                is_root=False,
            )
        info = infos[0]
        residuals = info.all_residuals
        
        csv_path = os.path.join(traj_dir, f"trajectory_broyden_sample_{idx:02d}_8000.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("iteration,rel_residual\n")
            for it_i, r_val in enumerate(residuals, 1):
                f.write(f"{it_i},{r_val:.8f}\n")
                
        # Calculate checkpoints at 6000 and 8000
        best_6000 = min(residuals[:6000]) if len(residuals) >= 6000 else min(residuals)
        best_6000_it = residuals.index(best_6000) + 1
        conv_6000 = best_6000 < 5e-3
        
        best_8000 = min(residuals)
        best_8000_it = residuals.index(best_8000) + 1
        conv_8000 = best_8000 < 5e-3
        
        print(f"[Task V5 Summary Sample {idx:02d}] CSV: {csv_path}", flush=True)
        print(f"  Total iters run: {len(residuals)}", flush=True)
        print(f"  @ 6000 iters: Converged={conv_6000}, Best_Rel={best_6000:.8f} (@ iter {best_6000_it})", flush=True)
        print(f"  @ 8000 iters: Converged={conv_8000}, Best_Rel={best_8000:.8f} (@ iter {best_8000_it})", flush=True)
        print(f"  Final (iter {len(residuals)}): Residual={residuals[-1]:.8f}", flush=True)

if __name__ == "__main__":
    main()
