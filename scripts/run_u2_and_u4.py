import os
import sys
import json
import torch

sys.path.append('/home/shoon/cts_e1')

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.types import NuVector, RuntimeBudgetState
from cts.model.gemma_loader import load_gemma4_e4b
from cts.deq.root_bootstrap import solve_root_z_star
from cts.deq.broyden_forward import _lm_apply, _lm_apply_transpose, _relative_residual
from cts.routing.sparse_moe_ref import routing_weights, sparse_module_weights

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
    
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    test_samples = dataset[:32]
    
    print('[U2-U4] Loading backbone model (Base)...', flush=True)
    base_snapshot = '/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377'
    assert '58c4d7b5' in str(base_snapshot), f"[BACKBONE LOCK VIOLATION] expected base snapshot (58c4d7b5), got {base_snapshot}"
    print(f'[BACKBONE-EVIDENCE] loading from: {base_snapshot}', flush=True)
    model, tok = load_gemma4_e4b(model_id=base_snapshot, device_map=device, torch_dtype=torch.float32)
    bb = GemmaCTSBackbone(model, tok)
    bb.eval()
    
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
    
    nu = NuVector(nu_act=1.0e-3, nu_expl=1.0e-3, nu_tol=5e-3)
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    routing_mode = "sparse"
    
    # 1. Precalculate z0 and context for 32 samples
    print('[U2-U4] Precalculating root z0 for 32 samples...', flush=True)
    z0_list = []
    ctx_list = []
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
        
    print('\n======================================================', flush=True)
    print('Task U2: 32 samples evaluation using trajectory dump solver (max_iter=1000, tol=5e-3)', flush=True)
    print('======================================================', flush=True)
    
    log_file_path = os.path.join(results_dir, 'm1_matrix_cell_simple_0.92_1000.log')
    simple_results = []
    
    with open(log_file_path, 'w', encoding='utf-8') as lf:
        for idx in range(32):
            prompt = prompts_list[idx]
            ctx = ctx_list[idx]
            z0 = z0_list[idx]
            
            def phi(zz):
                alpha = routing_weights(zz, w_g, nu.nu_temp)
                mw = alpha if routing_mode == "dense" else _routing_sparse(zz, w_g, nu.nu_temp, 3)
                return bb.deq_step(zz, ctx, mw, {'top_k': 3})
            
            z = z0.detach().reshape(1, -1).clone()
            batch = 1
            n = z.shape[1]
            
            def G(vec):
                shaped = vec.view(batch, 64, 2560)
                return phi(shaped).reshape(batch, -1)
            
            scale = 0.1
            work_capacity = 1000
            u_buf = torch.empty(batch, work_capacity, n, device=device, dtype=torch.float32)
            v_buf = torch.empty_like(u_buf)
            work_rank = 0
            
            with torch.no_grad():
                phi_z = G(z)
                residual = z - phi_z
                
                converged = False
                final_res = float('inf')
                
                for eval_idx in range(1000):
                    rel_norm, abs_norm = _relative_residual(residual, z, phi_z, dim=1)
                    rel_val = float(rel_norm.item())
                    final_res = rel_val
                    
                    if rel_val < 0.005 or not torch.isfinite(rel_norm):
                        converged = (rel_val < 0.005)
                        break
                        
                    step = -_lm_apply(residual, u_buf, v_buf, work_rank, scale)
                    z_new = z + step
                    phi_new = G(z_new)
                    residual_new = z_new - phi_new
                    
                    s = z_new - z
                    y = residual_new - residual
                    hy = _lm_apply(y, u_buf, v_buf, work_rank, scale)
                    ht_s = _lm_apply_transpose(s, u_buf, v_buf, work_rank, scale)
                    denom = torch.sum(s * hy, dim=1)
                    scale_denom = (torch.linalg.vector_norm(s, dim=1) * torch.linalg.vector_norm(hy, dim=1)).clamp_min(1e-20)
                    curvature = denom.abs() / scale_denom
                    valid = torch.isfinite(denom) & (curvature > 1e-7)
                    if valid.item() and work_rank < work_capacity:
                        u_new = s - hy
                        v_new = ht_s / denom
                        u_buf[:, work_rank] = u_new
                        v_buf[:, work_rank] = v_new
                        work_rank += 1
                        
                    z = z_new
                    phi_z = phi_new
                    residual = residual_new
                
            line = f"Sample {idx:02d}: Converged={converged}, Residual={final_res:.6f}"
            print(line, flush=True)
            lf.write(line + '\n')
            lf.flush()
            simple_results.append((idx, converged, final_res))
            
    num_passed = sum(1 for _, c, _ in simple_results if c)
    print(f"\n[Task U2 Summary] Simple Solver 1000 iters: {num_passed}/32 passed", flush=True)
    
    print('\n======================================================', flush=True)
    print('Task U4: Broyden 1000 iters per-iteration residual dump for #05, #27, #30', flush=True)
    print('======================================================', flush=True)
    
    for target_idx in [5, 27, 30]:
        prompt = prompts_list[target_idx]
        z0 = z0_list[target_idx]
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
                broyden_max_iter=1000,
                broyden_tol_min=1e-4,
                broyden_tol_max=1e-2,
                tau_flops_budget=1e14,
                max_decode_tokens=64,
                parent_z_star=z0,
            )
        residuals = tr.solver_stats.get('all_residuals', [])
        csv_file = os.path.join(traj_dir, f'trajectory_broyden_sample_{target_idx:02d}.csv')
        with open(csv_file, 'w', encoding='utf-8') as cf:
            cf.write("iteration,rel_residual\n")
            for it_num, r_val in enumerate(residuals, start=1):
                cf.write(f"{it_num},{r_val:.8f}\n")
        print(f"[Task U4 Dumped] Sample {target_idx:02d}: {len(residuals)} iterations written to {csv_file}", flush=True)

if __name__ == '__main__':
    main()
