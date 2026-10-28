import os
import sys
import torch
import json
import csv

sys.path.append('/home/shoon/cts_e1')

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import _routing_sparse
from cts.types import NuVector
from cts.model.gemma_loader import load_gemma4_e4b
from cts.deq.root_bootstrap import solve_root_z_star
from cts.deq.broyden_forward import _relative_residual, _lm_apply, _lm_apply_transpose

def run_trajectory(bb, prompt, idx, device, out_dir):
    with torch.no_grad():
        ctx = bb.encode_context(prompt).to(device=device, dtype=torch.float32)
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
        
        nu = NuVector(nu_act=1.0e-3, nu_expl=1.0e-3, nu_tol=5e-3)
        w_g = bb.routing_matrix().to(device=device, dtype=torch.float32) if hasattr(bb, 'routing_matrix') else torch.randn(19, 2560, device=device, dtype=torch.float32) * 0.02
        
        def phi(zz):
            mw = _routing_sparse(zz, w_g, nu.nu_temp, 3)
            return bb.deq_step(zz, ctx, mw, {'top_k': 3})
            
        z = z0.detach().reshape(1, -1).clone()
        batch = 1
        n = z.shape[1]
        
        def G(vec):
            shaped = vec.view(batch, 64, 2560)
            return phi(shaped).reshape(batch, -1)
            
        scale = 0.1
        work_capacity = 2000
        u_buf = torch.empty(batch, work_capacity, n, device=device, dtype=torch.float32)
        v_buf = torch.empty_like(u_buf)
        work_rank = 0
        
        phi_z = G(z)
        residual = z - phi_z
        
        csv_path = os.path.join(out_dir, f'trajectory_sample_{idx:02d}.csv')
        records = []
        
        for eval_idx in range(2000):
            rel_norm, abs_norm = _relative_residual(residual, z, phi_z, dim=1)
            rel_val = float(rel_norm.item())
            records.append((eval_idx + 1, rel_val))
            
            if rel_val < 0.005 or not torch.isfinite(rel_norm):
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
            
        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['iteration', 'rel_residual'])
            for it, r in records:
                writer.writerow([it, f'{r:.8f}'])
                
        print(f'Sample {idx:02d}: iters={len(records)}, first_res={records[0][1]:.6f}, final_res={records[-1][1]:.6f}, min_res={min(r for _, r in records):.6f}', flush=True)

def main():
    device = 'cuda:0'
    jsonl_path = '/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl'
    out_dir = '/home/shoon/cts_e1/trajectories'
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    base_snapshot = '/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377'
    mid = base_snapshot
    assert "58c4d7b5" in str(mid), f"[BACKBONE LOCK VIOLATION] expected base snapshot (58c4d7b5), got {mid}"
    print(f'[BACKBONE-EVIDENCE] loading from: {mid}', flush=True)
    model, tok = load_gemma4_e4b(model_id=mid, device_map=device, torch_dtype=torch.float32)
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
    
    # 7 non-converged samples at 2000 iters
    for idx in [5, 7, 9, 16, 20, 27, 30]:
        run_trajectory(bb, dataset[idx]['prompt'], idx, device, out_dir)

if __name__ == '__main__':
    main()
