import os
import sys
import json
import torch
from pathlib import Path

# Repo path
sys.path.append('/home/shoon/cts_e1')

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition, _routing_sparse
from cts.types import NuVector, RuntimeBudgetState
from cts.model.gemma_loader import load_gemma4_e4b
from cts.deq.root_bootstrap import solve_root_z_star
from cts.deq.broyden_forward import _relative_residual, _limited_memory_broyden_batch
from cts.routing.sparse_moe_ref import routing_weights
from cts.deq.hybrid_solver import solve_hybrid

def run_w2_and_w3():
    device = 'cuda:0'
    jsonl_path = '/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl'
    results_dir = '/home/shoon/cts_e1/results'
    traj_dir = '/home/shoon/cts_e1/trajectories'
    official_log_path = '/home/shoon/cts_e1/m1_matrix_logs/m1_matrix_cell_base_0.92_1000.log'
    
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
    
    # Matching official scaling contract: Base model is already calibrated at c=0.92
    original_deq_step = bb.deq_step
    gamma = torch.sigmoid(bb.gamma_param).to(device=device, dtype=torch.float32)
    
    def make_patched_deq_step(target_scale: float):
        def patched_deq_step(z, context, module_weights, extra):
            ds_0 = original_deq_step(z, context, module_weights, extra)
            ratio = target_scale / 0.92
            ds_new = ratio * (ds_0 - (1.0 - gamma) * z) + (1.0 - gamma) * z
            return ds_new.to(dtype=z.dtype)
        return patched_deq_step

    target_c = 0.92
    if target_c != 0.92:
        bb.deq_step = make_patched_deq_step(target_c)
    else:
        bb.deq_step = original_deq_step  # c=0.92: maintain original deq_step exactly
        
    nu = NuVector(nu_act=1.0e-3, nu_expl=1.0e-3, nu_tol=5e-3)
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    routing_mode = "sparse"
    
    print("\n--- Precalculating root z0 and seed alignment for 32 samples ---", flush=True)
    ctx_list = []
    z0_list = []
    pz_list = []
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
        pz = root_result.z_star_child.detach().clone().to(device=device, dtype=torch.float32)
        pz_list.append(pz)
        
        # Seed calculation aligned with official transition: 2026 + branch*31 + len(prompt)%997
        seed_val = 2026 + 0 * 31 + (len(prompt) % 997)
        if idx == 0:
            print(f"[SEED-EVIDENCE] sample=0 seed={seed_val} prompt_len={len(prompt)}", flush=True)
            
        gen = torch.Generator(device=device)
        gen.manual_seed(seed_val)
        noise_sigma = 0.02
        epsilon = torch.randn(64, 2560, device=device, dtype=torch.float32, generator=gen) * noise_sigma * nu.nu_expl
        z0 = pz + epsilon
        z0_list.append(z0)

    # Read official 1000 log for 1:1 verification
    official_data = {}
    if os.path.exists(official_log_path):
        with open(official_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Sample "):
                    parts = line.split(":")
                    s_idx = int(parts[0].replace("Sample ", ""))
                    # Sample 00: Converged=True, Residual=0.004828
                    rest = parts[1].strip()
                    c_str = rest.split(",")[0].split("=")[1].strip()
                    r_str = rest.split(",")[1].split("=")[1].strip()
                    official_data[s_idx] = {
                        "converged": (c_str == "True"),
                        "residual": float(r_str)
                    }
        print(f"Loaded {len(official_data)} samples from official log: {official_log_path}", flush=True)
    else:
        print(f"[WARNING] Official log not found at {official_log_path}", flush=True)

    # =========================================================================
    # Task W2-2 (1): Broyden 단독 @1000 재현 게이트 (32표본)
    # =========================================================================
    print(f"\n======================================================", flush=True)
    print("Task W2-2 (1): Broyden Solo @1000 Replication Gate (32 samples)", flush=True)
    print(f"======================================================", flush=True)
    
    w2_broyden_results = []
    broyden_passed = 0
    mismatch_count = 0
    
    print(f"{'Sample':<8} | {'Harness Conv':<13} | {'Official Conv':<14} | {'Harness Res':<12} | {'Official Res':<13} | {'Diff':<10} | {'Status'}")
    print("-" * 85)
    
    for idx in range(32):
        ctx = ctx_list[idx]
        z0 = z0_list[idx]
        
        def phi(zz):
            mw = _routing_sparse(zz, w_g, nu.nu_temp, 3)
            return bb.deq_step(zz, ctx, mw, {'top_k': 3})
            
        with torch.no_grad():
            z_batch, infos = _limited_memory_broyden_batch(
                phi=phi,
                z0_batch=z0.unsqueeze(0),
                tol=5e-3,
                max_iter=1000,
                parent_inv_jacobian=None,
                fp32_buffer=True,
                memory_limit=16,
                is_root=True,
            )
            
        info = infos[0]
        h_conv = info.converged
        h_res = float(info.residual_norm)
        if h_conv:
            broyden_passed += 1
            
        off_conv = official_data.get(idx, {}).get("converged", None)
        off_res = official_data.get(idx, {}).get("residual", None)
        
        diff_str = "N/A"
        match_status = "OK"
        if off_res is not None:
            res_diff = abs(h_res - off_res)
            diff_str = f"{res_diff:.6f}"
            if h_conv != off_conv or res_diff > 1e-4:
                match_status = "MISMATCH"
                mismatch_count += 1
                
        print(f"Sample {idx:02d} | {str(h_conv):<13} | {str(off_conv):<14} | {h_res:<12.6f} | {off_res:<13.6f} | {diff_str:<10} | {match_status}")
        w2_broyden_results.append({
            "idx": idx,
            "converged": h_conv,
            "residual": h_res,
            "iterations": info.iterations
        })

    print("-" * 85)
    print(f"[Task W2-2 (1) Summary] Broyden@1000: {broyden_passed}/32 passed. (Official: 18/32). Mismatches: {mismatch_count}", flush=True)
    assert broyden_passed == 18, f"[GATE FAILURE] Expected 18/32, got {broyden_passed}/32"
    assert mismatch_count == 0, f"[GATE FAILURE] Found {mismatch_count} mismatches with official records"
    print(">>> [GATE PASS] Broyden@1000 Replication Gate PASSED with 100% 4-decimal precision! <<<\n", flush=True)

    # =========================================================================
    # Task W2-2 (2): Full-Memory Fallback 단독 @1000 (32표본)
    # =========================================================================
    print(f"\n======================================================", flush=True)
    print("Task W2-2 (2): Full-Memory Fallback Solo @1000 (32 samples)", flush=True)
    print(f"======================================================", flush=True)
    
    fallback_passed = 0
    for idx in range(32):
        ctx = ctx_list[idx]
        z0 = z0_list[idx]
        
        def phi(zz):
            mw = _routing_sparse(zz, w_g, nu.nu_temp, 3)
            return bb.deq_step(zz, ctx, mw, {'top_k': 3})
            
        # Full memory solver from z0
        z = z0.detach().reshape(1, -1).clone().float()
        n = int(z.shape[1])
        
        def G(vec: torch.Tensor) -> torch.Tensor:
            shaped = vec.to(z0.dtype).view(1, 64, 2560)
            return phi(shaped).reshape(1, -1).float()
            
        scale = 0.1
        work_capacity = 1000
        u_buf = torch.empty(1, work_capacity, n, device=device, dtype=torch.float32)
        v_buf = torch.empty_like(u_buf)
        work_rank = 0
        
        phi_z = G(z)
        residual = z - phi_z
        best_simple_res = float("inf")
        simple_converged = False
        simple_used_iters = 0
        
        with torch.no_grad():
            for s_idx in range(1, 1001):
                simple_used_iters = s_idx
                rel_norm, abs_norm = _relative_residual(residual, z, phi_z, dim=1)
                rel_val = float(rel_norm.item())
                if rel_val < best_simple_res:
                    best_simple_res = rel_val
                if rel_val < 5e-3:
                    simple_converged = True
                    break
                if not torch.isfinite(rel_norm):
                    break
                from cts.deq.broyden_forward import _lm_apply, _lm_apply_transpose
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
                if bool(valid.item()) and work_rank < work_capacity:
                    u_buf[:, work_rank] = s - hy
                    v_buf[:, work_rank] = ht_s / denom
                    work_rank += 1
                z, phi_z, residual = z_new, phi_new, residual_new
                
        if simple_converged:
            fallback_passed += 1
        print(f"Sample {idx:02d}: Converged={simple_converged}, BestRes={best_simple_res:.6f}, Iters={simple_used_iters}", flush=True)

    print(f"\n[Task W2-2 (2) Result] Full-Memory Fallback@1000: {fallback_passed}/32 ({fallback_passed/32*100:.2f}%) passed.", flush=True)

    # =========================================================================
    # Task W3: Official transition() path High-budget probe for #05 and #27 (@6000, @8000)
    # =========================================================================
    print(f"\n======================================================", flush=True)
    print("Task W3: Official transition() High-Budget Probe on #05 and #27 (max_iter=8000)", flush=True)
    print(f"======================================================", flush=True)
    
    target_samples = [5, 27]
    official_3000_check = {5: 0.005569, 27: 0.005369}
    
    for s_idx in target_samples:
        prompt = prompts_list[s_idx]
        pz = pz_list[s_idx]
        print(f"\n[Task W3] Running Official transition() with max_iter=8000 on Sample {s_idx:02d}...", flush=True)
        torch.cuda.empty_cache()
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
                broyden_max_iter=8000,
                broyden_tol_min=1e-4,
                broyden_tol_max=1e-2,
                tau_flops_budget=1e14,
                max_decode_tokens=64,
                parent_z_star=pz,
            )
            
        stats = tr.solver_stats
        converged = stats.get("converged", False)
        final_res = stats.get("residual_norm", float("inf"))
        all_res = stats.get("all_residuals", [])
        
        # Check alignment at 3000 iters
        res_at_3000 = all_res[2999] if len(all_res) >= 3000 else float("inf")
        min_res_3000 = min(all_res[:3000]) if len(all_res) >= 3000 else float("inf")
        expected_3000 = official_3000_check[s_idx]
        
        # Check stats at 6000 and 8000
        min_res_6000 = min(all_res[:6000]) if len(all_res) >= 6000 else float("inf")
        iter_at_min_6000 = all_res.index(min_res_6000) + 1 if len(all_res) >= 6000 else -1
        conv_6000 = (min_res_6000 < 5e-3)
        
        min_res_8000 = min(all_res[:8000]) if len(all_res) >= 8000 else float("inf")
        iter_at_min_8000 = all_res.index(min_res_8000) + 1 if len(all_res) >= 8000 else -1
        conv_8000 = (min_res_8000 < 5e-3)
        
        csv_path = os.path.join(traj_dir, f"trajectory_w3_sample_{s_idx:02d}_8000.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("iteration,rel_residual\n")
            for it_idx, r_val in enumerate(all_res, 1):
                f.write(f"{it_idx},{r_val:.8f}\n")
                
        print(f"[Task W3 Summary Sample {s_idx:02d}] CSV: {csv_path}", flush=True)
        print(f"  Total iters run: {len(all_res)}", flush=True)
        print(f"  @ 3000 iters: Current_Rel={res_at_3000:.8f}, Min_Rel={min_res_3000:.8f} (Expected min ~ {expected_3000})", flush=True)
        print(f"  @ 6000 iters: Converged={conv_6000}, Best_Rel={min_res_6000:.8f} (@ iter {iter_at_min_6000})", flush=True)
        print(f"  @ 8000 iters: Converged={conv_8000}, Best_Rel={min_res_8000:.8f} (@ iter {iter_at_min_8000})", flush=True)
        print(f"  Final (iter {len(all_res)}): Residual={final_res:.8f}", flush=True)

if __name__ == '__main__':
    run_w2_and_w3()
