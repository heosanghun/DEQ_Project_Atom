#!/usr/bin/env python3
"""local_L1_zstar_trace.py — PREREG_LOCAL_L1 (exploratory, non-confirmatory) z* trace on local RTX 4090.

Canonical init/transition (cts/deq/zstar_canonical.py + transition_inject.InjectedTransitionStep), FP32 weights
with PLE on CPU (scripts/local_loader_fp32_ple_cpu.py), local Stage-1 adapter (artifacts/stage1_last.lora_only.pt).
Per item it records the full rho trajectory, the top-3 routing ids used at every step, the lowest-residual
iterate z_{k*} (tensor saved as-is), the final iterate z_1000 and a freshly evaluated r_1000.

Held-out seal: refuses any item listed in configs/s7_new_heldout_520.json or configs/s5_heldout_80.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from local_loader_fp32_ple_cpu import load_gemma4_fp32_ple_cpu  # noqa: E402
from cts.backbone.gemma_adapter import GemmaCTSBackbone  # noqa: E402
from cts.train.lora_compat import apply_paper_lora  # noqa: E402
from cts.deq.zstar_canonical import load_canonical_config  # noqa: E402
from cts.deq.transition_inject import (  # noqa: E402
    InjectedTransitionStep,
    compute_broadcast_context_c,
    compute_c_hat_0,
)

HUB = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-e4b")
SNAPSHOT_SUFFIX = "snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"
CKPT = os.path.join(ROOT, "artifacts", "stage1_last.lora_only.pt")
PROMPTS = os.path.join(ROOT, "data", "stage2", "math_train_prompts_5000.jsonl")
SEALED = [os.path.join(ROOT, "configs", "s7_new_heldout_520.json"), os.path.join(ROOT, "configs", "s5_heldout_80.json")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_indices(path):
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, dict):
        for k in ("indices", "items", "heldout_indices", "train_indices"):
            if k in d:
                return [int(x) for x in d[k]]
        raise ValueError(f"no index list in {path}")
    return [int(x) for x in d]


def sealed_set():
    s = set()
    for p in SEALED:
        if os.path.exists(p):
            s |= set(load_indices(p))
    return s


def snapshot_dir():
    cand = [os.path.join(ROOT, SNAPSHOT_SUFFIX), os.path.join(HUB, SNAPSHOT_SUFFIX)]
    for c in cand:
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    raise FileNotFoundError("gemma-4-e4b snapshot 58c4d7b5 not found")


def load_bf16_store(snap, device):
    """Weights kept in their stored dtype (bf16) on the GPU; see install_fp32_compute. Same layout as local_loader_fp32_ple_cpu.load_gemma4_fp32_ple_cpu (PLE on CPU memmap, contiguous fp32
    GPU storage) but each tensor is moved to the GPU in its stored dtype and upcast there, so the CPU never
    holds an fp32 copy (Windows commit limit)."""
    import numpy as np
    import torch.nn as nn
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration
    from local_loader_fp32_ple_cpu import CPUPLEWrapper, ensure_ple_binary, DEFAULT_PLE_BIN

    st_path = os.path.join(snap, "model.safetensors")
    ensure_ple_binary(st_path, DEFAULT_PLE_BIN)
    cfg = AutoConfig.from_pretrained(snap)
    with init_empty_weights():
        model = Gemma4ForConditionalGeneration(cfg)
    model.model.vision_tower = None
    model.model.audio_tower = None
    model.model.embed_vision = None
    model.model.embed_audio = None
    ple_key = "model.language_model.embed_tokens_per_layer.weight"
    keys = [k for k in sorted(model.state_dict().keys()) if k not in ("lm_head.weight", ple_key)]
    with safe_open(st_path, framework="pt", device="cpu") as f:
        st_keys = set(f.keys())
        shapes = [(k, f.get_slice(k).get_shape()) for k in keys if k in st_keys]
        mm = np.memmap(DEFAULT_PLE_BIN, dtype="float32", mode="r", shape=(262144, 10752))
        model.model.language_model.embed_tokens_per_layer.weight = nn.Parameter(torch.from_numpy(mm), requires_grad=False)
        for k, shape in shapes:
            dst = f.get_tensor(k).to(device)  # stored dtype (bf16) kept on GPU; upcast per forward (exact)
            if k == "model.language_model.embed_tokens.weight":
                dst = dst.float()  # model code infers its compute dtype from this tensor -> fp32 everywhere
            *parents, leaf = k.split(".")
            mod = model
            for p in parents:
                mod = getattr(mod, p)
            if leaf in mod._buffers:
                mod._buffers[leaf] = dst
            else:
                mod._parameters[leaf] = nn.Parameter(dst, requires_grad=False)
    torch.cuda.synchronize()
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model.model.language_model.embed_tokens_per_layer = CPUPLEWrapper(
        model.model.language_model.embed_tokens_per_layer, target_device=device)
    model.eval()
    return model, AutoTokenizer.from_pretrained(snap)


def install_fp32_compute(model):
    """Every module that directly owns bf16 parameters/buffers gets them upcast to fp32 for the duration of
    its forward and restored afterwards. bf16 -> fp32 is exact, so the arithmetic equals an fp32-loaded
    model built from the same bf16 checkpoint (the server upcasts the same hub weights to fp32)."""
    n = 0
    for mod in model.modules():
        own = [k for k, v in mod._parameters.items() if v is not None and v.dtype == torch.bfloat16]
        ownb = [k for k, v in mod._buffers.items() if v is not None and v.dtype == torch.bfloat16]
        if not own and not ownb:
            continue

        def pre(m, args, _own=own, _ownb=ownb):
            m._l1_saved = {}
            for k in _own:
                m._l1_saved[k] = m._parameters[k]
                m._parameters[k] = torch.nn.Parameter(m._parameters[k].float(), requires_grad=False)
            for k in _ownb:
                m._l1_saved["b:" + k] = m._buffers[k]
                m._buffers[k] = m._buffers[k].float()

        def post(m, args, out):
            for k, v in m._l1_saved.items():
                if k.startswith("b:"):
                    m._buffers[k[2:]] = v
                else:
                    m._parameters[k] = v
            m._l1_saved = {}

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        n += 1
    return n


def build_backbone(device):
    m, tok = load_bf16_store(snapshot_dir(), device)
    bb = GemmaCTSBackbone(m, tok)
    apply_paper_lora(bb, rank=8, target_modules=("q_proj", "v_proj", "o_proj"), dropout=0.0,
                     require_match=False, verbose=False)
    for name, p in bb.named_parameters():
        if any(t in name for t in ("lora_", "routing_proj", "w_proj", "gamma_param", "_blend")):
            p.data = p.data.to(device=device, dtype=torch.float32)
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck.get("backbone_state_dict", ck)
    new_sd = {}
    for k, v in sd.items():
        if k == "routing_proj":
            new_sd["routing_proj.parametrizations.weight.original"] = v
        elif k.endswith(".lora_A.weight"):
            new_sd[k[: -len(".weight")] + ".parametrizations.weight.original"] = v
        else:
            new_sd[k] = v
    res = bb.load_state_dict(new_sd, strict=False)
    lm = bb.cg.model.language_model
    dts = (next(lm.parameters()).dtype, lm.embed_tokens.weight.dtype)
    assert dts == (torch.float32, torch.float32), dts
    n_hooks = install_fp32_compute(bb)
    bad = [n for n, p in bb.named_parameters() if p.device.type == "cpu" and "embed_tokens_per_layer" not in n]
    assert not bad, f"params left on CPU: {bad[:5]}"
    print(f"[L1] fp32-compute hooks on {n_hooks} modules", flush=True)
    bb.eval()
    w_g = bb.routing_matrix().to(device=device, dtype=torch.float32)
    return bb, w_g, len(res.missing_keys), len(res.unexpected_keys)


@torch.no_grad()
def trace_item(bb, w_g, prompt, cfg, device, max_nfe):
    g0, alpha, seed = float(cfg["g0"]), float(cfg["alpha"]), int(cfg["seed"])
    k_latent, sigma = int(cfg.get("k_latent", 64)), float(cfg.get("sigma_noise", 0.020))
    ctx = bb.encode_context(prompt)
    if hasattr(ctx, "to"):
        ctx = ctx.to(device=device)
    z_src = bb.root_fwdpass(ctx, K=k_latent, output_dtype=torch.float32).to(device=device)
    c_x = compute_broadcast_context_c(ctx, K=k_latent, device=device, dtype=torch.float32)
    torch.manual_seed(seed)
    z0 = (z_src + torch.randn(k_latent, bb.hidden_size, device=device, dtype=torch.float32) * sigma).clone()
    c_hat_0 = compute_c_hat_0(c_x, z0).to(device=device, dtype=torch.float32)
    step = InjectedTransitionStep(backbone=bb, context=ctx, w_g=w_g, c_hat_0=c_hat_0, g0=g0, alpha=alpha,
                                  top_k=int(cfg.get("top_k", 3)), routing_mode=str(cfg.get("routing_mode", "sparse")))
    z = z0
    rho, top3 = [], []
    best_r, best_k, best_z = float("inf"), -1, None
    for k in range(max_nfe + 1):  # k = 0..max_nfe ; residual r_k evaluated at z_k
        z_next = step(z)
        diff = (z_next.double() - z.double()).norm().item()
        nz = z.double().norm().item()
        r_k = diff / (alpha * max(nz, 1e-12))
        top3.append(list(step.last_r_ids))
        if k < max_nfe:
            rho.append(alpha * r_k)
        if r_k < best_r:
            best_r, best_k, best_z = r_k, k, z.detach().clone()
        if k == max_nfe:
            r_final = r_k  # fresh evaluation at z_1000
            z_final = z.detach().clone()
            break
        z = z_next
    switches_301 = sum(1 for i in range(301, max_nfe + 1) if top3[i] != top3[i - 1])
    return {
        "z_src": z_src.detach().cpu(), "z_final": z_final.cpu(), "z_lowest": best_z.cpu(),
        "k_star": best_k, "r_lowest": best_r, "r_final_fresh": r_final,
        "r_last_log": rho[-1] / alpha, "rho_traj": rho, "top3_traj": top3,
        "top3_final": top3[max_nfe], "top3_lowest": top3[best_k],
        "switches_301_1000": switches_301, "norm_final": z_final.norm().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(ROOT, "configs", "s7_new_train_680.json"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_nfe", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results_local", "L1_trace.pt"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    items = load_indices(a.items)[: a.n]
    bad = sorted(set(items) & sealed_set())
    if bad:
        raise SystemExit(f"HELD-OUT SEAL: refusing sealed items {bad[:10]}")
    cfg = load_canonical_config(os.path.join(ROOT, "configs", "zstar_canonical.yaml"))
    cfg.update({"alpha": 0.25, "seed": 2030, "sigma_noise": 0.020})
    prompts = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    partial = a.out.replace(".pt", ".partial.pt")
    out = torch.load(partial) if os.path.exists(partial) else {}
    meta = {"started": datetime.now().isoformat(timespec="seconds"), "ckpt": CKPT, "ckpt_sha256": sha256(CKPT),
            "items_file": a.items, "n": len(items), "cfg": cfg, "max_nfe": a.max_nfe, "torch": torch.__version__,
            "zstar_canonical_sha256": sha256(os.path.join(ROOT, "cts", "deq", "zstar_canonical.py")),
            "weights_storage": "bf16 on GPU, exact fp32 upcast per module forward; heads/LoRA fp32", "transition_inject_sha256": sha256(os.path.join(ROOT, "cts", "deq", "transition_inject.py"))}
    print("[L1] meta", json.dumps({k: v for k, v in meta.items() if k != "cfg"}), flush=True)

    t0 = time.time()
    bb, w_g, n_miss, n_unexp = build_backbone(a.device)
    print(f"[L1] model ready {time.time()-t0:.1f}s | missing={n_miss} unexpected={n_unexp} | "
          f"vram={torch.cuda.memory_allocated()/2**20:.0f}MiB", flush=True)
    meta.update({"missing_keys": n_miss, "unexpected_keys": n_unexp})

    for i, q in enumerate(items, 1):
        key = f"q{q:04d}"
        if key in out:
            continue
        t1 = time.time()
        rec = trace_item(bb, w_g, prompts[q]["prompt"], cfg, a.device, a.max_nfe)
        rec["wall_s"] = time.time() - t1
        out[key] = rec
        torch.save(out, partial)
        print(f"[L1 {i:03d}/{len(items)}] Q#{q:04d} r_last(log)={rec['r_last_log']:.3e} r_1000={rec['r_final_fresh']:.3e} "
              f"r_min={rec['r_lowest']:.3e} k*={rec['k_star']} sw301={rec['switches_301_1000']} "
              f"top3_eq={rec['top3_final']==rec['top3_lowest']} peak={torch.cuda.max_memory_allocated()/2**20:.0f}MiB "
              f"{rec['wall_s']:.1f}s", flush=True)
    out["_meta"] = meta
    torch.save(out, a.out)
    print(f"[L1] done {len(items)} items in {time.time()-t0:.1f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
