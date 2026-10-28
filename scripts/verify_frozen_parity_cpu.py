#!/usr/bin/env python3
"""
scripts/verify_frozen_parity_cpu.py - Pre-run Bitwise Parity Verification (Ledger #380 Point 2(c), Ledger #381 Point 2)
Directive:
  Verify on 1 control item with stable mask that frozen vs dynamic trajectories
  from step 301 to 1000 are bitwise identical: max |Delta| = 0.0 across all 700 steps.
  Compliance with Ledger #381 Rules:
    (a) Strictly torch.no_grad() everywhere, zero computation graph retention between steps.
    (b) Model in eval() mode.
    (c) Periodic memory checks.
  Device: CPU only, 8 threads.
"""

import os
import gc
import sys
import time
import json
import torch
import torch.nn.functional as F

torch.set_num_threads(8)

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from harness_A_v1 import get_model, get_dataset
from cts.routing.sparse_moe_ref import routing_weights
from cts.deq.transition_inject import (
    compute_broadcast_context_c,
    compute_c_hat_0,
    InjectedTransitionStep,
)

class FrozenInjectedTransitionStep:
    """Transition step with frozen routing mask."""
    def __init__(
        self,
        backbone,
        context,
        w_g,
        c_hat_0,
        frozen_top3_ids,
        g0=0.25,
        alpha=0.25,
        top_k=3,
    ):
        self.backbone = backbone
        self.context = context
        self.w_g = w_g
        self.c_hat_0 = c_hat_0
        self.frozen_top3_ids = sorted([int(x) for x in frozen_top3_ids])
        self.g0 = float(g0)
        self.alpha = float(alpha)
        self.top_k = top_k
        self.const_inj = self.g0 * self.c_hat_0

        device = w_g.device
        self.sel_ids = torch.tensor(self.frozen_top3_ids, dtype=torch.long, device=device)
        self.r_ids = self.frozen_top3_ids

    @torch.no_grad()
    def __call__(self, z_cur: torch.Tensor) -> torch.Tensor:
        device = z_cur.device
        # 1. Routing weights with frozen mask
        alpha_r = routing_weights(z_cur.float(), self.w_g.to(device), 1.0)
        
        # Mask out non-selected modules
        idx = self.sel_ids.expand(alpha_r.shape[:-1] + (len(self.frozen_top3_ids),))
        m = torch.zeros_like(alpha_r)
        m.scatter_(-1, idx, 1.0)
        w = alpha_r * m
        f_weights = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-8))

        ex = {
            "top_k": self.top_k,
            "deq_map_mode": self.backbone.deq_map_mode,
            "routing_module_ids": self.r_ids,
            "routing_module_ids_tensor": self.sel_ids,
        }

        # 2. Context hook
        prepare_hook = getattr(self.backbone, "prepare_deq_context", None)
        if callable(prepare_hook):
            ex["eq3_static_inputs"] = prepare_hook(z_cur, self.context, self.r_ids)

        # 3. Base Phi step (no_grad guaranteed)
        phi_z = self.backbone.deq_step(z_cur, self.context, f_weights, ex).float()

        # 4. Constant injection and damping
        inj = self.const_inj.to(device=device, dtype=torch.float32)
        injected = phi_z + inj
        z_next = (1.0 - self.alpha) * z_cur.float() + self.alpha * injected
        return z_next


@torch.no_grad()
def test_item_parity(bb, dataset, qid, w_g):
    prompt_text = dataset[qid]["prompt"]
    ctx = bb.encode_context(prompt_text)
    if hasattr(ctx, "to"):
        ctx = ctx.to(device="cpu")

    k_latent = 64
    d = bb.hidden_size
    c_x = compute_broadcast_context_c(ctx, K=k_latent, device="cpu", dtype=torch.float32)

    torch.manual_seed(2030)
    z_src = bb.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device="cpu")
    z0 = (z_src + torch.randn(k_latent, d, device="cpu", dtype=torch.float32) * 0.02).clone()
    c_hat_0 = compute_c_hat_0(c_x, z0).to(device="cpu", dtype=torch.float32)

    dyn_step = InjectedTransitionStep(
        backbone=bb,
        context=ctx,
        w_g=w_g,
        c_hat_0=c_hat_0,
        g0=0.25,
        alpha=0.25,
        top_k=3,
        routing_mode="sparse",
    )

    # 1. Run dynamic steps 1 to 300 under no_grad
    print(f"Running steps 1 to 300 for Q#{qid:04d}...", flush=True)
    t0_300 = time.time()
    z = z0.clone()
    for step in range(1, 301):
        z = dyn_step(z)

    print(f"Steps 1~300 done in {time.time() - t0_300:.2f}s", flush=True)

    # Extract mask at step 300
    alpha_300 = routing_weights(z.float(), w_g, 1.0)
    top3_ids_300 = sorted(torch.topk(alpha_300.squeeze(0), 3).indices.tolist())
    print(f"Q#{qid}: Step 300 Top-3 Modules = {top3_ids_300}", flush=True)

    # Now run from 301 to 1000 in two parallel branches:
    # Branch A: Dynamic routing (check if top3 stays unchanged)
    # Branch B: Frozen mask routing
    frozen_step = FrozenInjectedTransitionStep(
        backbone=bb,
        context=ctx,
        w_g=w_g,
        c_hat_0=c_hat_0,
        frozen_top3_ids=top3_ids_300,
        g0=0.25,
        alpha=0.25,
        top_k=3,
    )

    z_dyn = z.clone()
    z_froz = z.clone()

    max_delta_overall = 0.0
    switches_detected = 0

    print(f"Running steps 301 to 1000 (700 steps)...", flush=True)
    t0_700 = time.time()

    # Test 700 steps (steps 301 to 1000) under strict no_grad
    for step in range(301, 1001):
        z_dyn_next = dyn_step(z_dyn)
        z_froz_next = frozen_step(z_froz)

        # Check if dynamic routing switched
        dyn_r_ids = sorted(dyn_step.last_r_ids)
        if dyn_r_ids != top3_ids_300:
            switches_detected += 1

        delta = torch.max(torch.abs(z_dyn_next - z_froz_next)).item()
        if delta > max_delta_overall:
            max_delta_overall = delta

        z_dyn = z_dyn_next
        z_froz = z_froz_next

        if (step - 300) % 100 == 0 or step == 1000:
            print(f"  Step {step:4d}/1000: max|Delta| = {delta:.10e} | Dyn top-3 = {dyn_r_ids} | Switches = {switches_detected}", flush=True)

    print(f"Steps 301~1000 done in {time.time() - t0_700:.2f}s", flush=True)
    print(f"Q#{qid} Parity Summary:", flush=True)
    print(f"  Total steps tested: 700 (steps 301 to 1000)", flush=True)
    print(f"  Routing switches in dynamic: {switches_detected}", flush=True)
    print(f"  Max absolute difference across all 700 steps: {max_delta_overall:.10e}", flush=True)
    return max_delta_overall, switches_detected


def main():
    print("=" * 80)
    print("PRE-RUN BITWISE PARITY VERIFICATION: DYNAMIC VS FROZEN (LEDGER #380 POINT 2(c), #381)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Device: cpu | Threads: 8 | Alpha: 0.25 | g0: 0.25 | K: 300 freeze")
    print("Safety Rules: torch.no_grad() active, eval() mode, 0 graph retention")
    print("=" * 80, flush=True)

    t0 = time.time()
    with torch.no_grad():
        bb = get_model("cpu")
        bb.eval()
        dataset = get_dataset()
        w_g = bb.routing_matrix().to(device="cpu", dtype=torch.float32)
        print(f"Model loaded in {time.time() - t0:.2f}s", flush=True)

        # Load control 20 items
        with open("/home/shoon/cts_e1_s5/configs/s6_control_20.json") as f:
            control_20 = json.load(f)

        # Test control items until we find a stable one
        verified_item = None
        for qid in control_20:
            print(f"\n--- Testing Candidate Control Item Q#{qid:04d} ---", flush=True)
            max_delta, switches = test_item_parity(bb, dataset, qid, w_g)
            if switches == 0:
                print(f"\n=> SUCCESS: Candidate Q#{qid} has 0 routing switches in steps 301~1000.", flush=True)
                print(f"=> BITWISE PARITY VERIFIED: max|Delta| = {max_delta:.10e}", flush=True)
                assert max_delta == 0.0, f"Non-zero difference: {max_delta}"
                verified_item = qid
                break
            else:
                print(f"=> Item Q#{qid} experienced {switches} switches, trying next candidate...", flush=True)

        assert verified_item is not None, "Could not find a mask-stable control item"
        print("\n" + "=" * 80, flush=True)
        print(f"VERIFICATION CONFIRMED: Control Q#{verified_item} passes bitwise parity test (max|Delta| = 0.0)", flush=True)
        print("Frozen mask implementation is bit-identical to dynamic routing across all 700 steps (301~1000).", flush=True)
        print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
