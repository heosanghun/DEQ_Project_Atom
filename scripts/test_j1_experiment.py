import os
import sys
import torch
import json

sys.path.append('/home/shoon/cts_e1')

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.deq.transition import transition
from cts.types import NuVector, RuntimeBudgetState
from cts.model.gemma_loader import load_gemma4_e4b
from cts.deq.root_bootstrap import solve_root_z_star

def main():
    device = 'cuda:0'
    jsonl_path = '/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl'
    
    dataset = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    test_samples = dataset[:32]
    
    print('[J1-TEST] Loading backbone model (Base)...', flush=True)
    base_snapshot = '/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377'
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
    
    # 1. Precalculate z0 for 32 samples exactly as in run_m1_matrix.py
    print('[J1-TEST] Precalculating z0 for 32 samples...', flush=True)
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
    print('Experiment A: Single sample #16 in cell evaluator (max_iter=600)', flush=True)
    print('======================================================', flush=True)
    budget_a = RuntimeBudgetState()
    with torch.no_grad():
        tr_a = transition(
            prompts_list[16],
            0,
            nu,
            budget_a,
            bb,
            K=64,
            d=2560,
            broyden_max_iter=600,
            broyden_tol_min=1e-4,
            broyden_tol_max=1e-2,
            tau_flops_budget=1e14,
            max_decode_tokens=64,
            parent_z_star=z0_list[16],
        )
    conv_a = tr_a.solver_stats.get('converged', False)
    res_a = tr_a.solver_stats.get('residual_norm', 1e9)
    iters_a = tr_a.solver_stats.get('iterations', 0)
    print(f'[Experiment A Result] Sample 16: Converged={conv_a}, Residual={res_a:.6f}, iters={iters_a}', flush=True)

    print('\n======================================================', flush=True)
    print('Experiment B: 32-sample sequential evaluation in cell evaluator (logging Sample #16)', flush=True)
    print('======================================================', flush=True)
    # We run full 32 samples loop as in run_m1_matrix.py
    for idx in range(32):
        budget_b = RuntimeBudgetState()
        with torch.no_grad():
            tr_b = transition(
                prompts_list[idx],
                0,
                nu,
                budget_b,
                bb,
                K=64,
                d=2560,
                broyden_max_iter=600,
                broyden_tol_min=1e-4,
                broyden_tol_max=1e-2,
                tau_flops_budget=1e14,
                max_decode_tokens=64,
                parent_z_star=z0_list[idx],
            )
        conv_b = tr_b.solver_stats.get('converged', False)
        res_b = tr_b.solver_stats.get('residual_norm', 1e9)
        iters_b = tr_b.solver_stats.get('iterations', 0)
        if idx == 16:
            print(f'[Experiment B Result] Sample 16: Converged={conv_b}, Residual={res_b:.6f}, iters={iters_b}', flush=True)

if __name__ == '__main__':
    main()
