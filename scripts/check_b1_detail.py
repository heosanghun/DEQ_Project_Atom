import torch
from scripts.s5_train_decoder_v2 import PrefixProjector

ckpt_path = "/home/shoon/retrain_1a/B1/stage1a_B1_step5000_20260907_0054.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
bb = ckpt.get("backbone_state_dict", ckpt)

print("=== CHECKPOINT TENSORS ===")
for k, v in bb.items():
    if "w_proj" in k:
        if v.ndim >= 2:
            norm_native = torch.linalg.norm(v, ord="fro").item()
            norm_f32 = torch.linalg.norm(v.float(), ord="fro").item()
        else:
            norm_native = torch.linalg.norm(v).item()
            norm_f32 = torch.linalg.norm(v.float()).item()
        print(f"Key: {k} | Shape: {v.shape} | Dtype: {v.dtype} | Native Fro: {norm_native:.8f} | F32 Fro: {norm_f32:.8f}")

# Compare with PrefixProjector loaded on CPU vs GPU
print("\n=== PREFIX PROJECTOR LOAD (CPU) ===")
proj_cpu = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True)
info_cpu = proj_cpu.load_b1_w_proj(ckpt_path, strict=True)
orig_cpu = proj_cpu.w_proj.parametrizations.weight.original
print(f"CPU orig shape={orig_cpu.shape}, dtype={orig_cpu.dtype}")
print(f"CPU orig frob (native)={torch.linalg.norm(orig_cpu, ord='fro').item():.8f}")
print(f"CPU orig frob (float32)={torch.linalg.norm(orig_cpu.float(), ord='fro').item():.8f}")
print(f"CPU b1_info frob_orig={info_cpu['frob_orig']:.8f}, spec_norm={info_cpu['spec_norm']:.8f}")

if torch.cuda.is_available():
    print("\n=== PREFIX PROJECTOR LOAD (GPU cuda:0) ===")
    proj_gpu = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to("cuda:0")
    info_gpu = proj_gpu.load_b1_w_proj(ckpt_path, strict=True)
    orig_gpu = proj_gpu.w_proj.parametrizations.weight.original
    print(f"GPU orig shape={orig_gpu.shape}, dtype={orig_gpu.dtype}")
    print(f"GPU orig frob (native)={torch.linalg.norm(orig_gpu, ord='fro').item():.8f}")
    print(f"GPU orig frob (float32)={torch.linalg.norm(orig_gpu.float(), ord='fro').item():.8f}")
    print(f"GPU b1_info frob_orig={info_gpu['frob_orig']:.8f}, spec_norm={info_gpu['spec_norm']:.8f}")

    # Bit-level comparison
    raw_tensor = bb["w_proj.parametrizations.weight.original"]
    diff = (orig_gpu.cpu().float() - raw_tensor.float()).abs().max().item()
    print(f"Bit-level comparison max|Delta| vs raw ckpt: {diff:.10e}")
    print(f"CPU orig frob (float64)={torch.linalg.norm(orig_cpu.double(), ord='fro').item():.8f}")
    print(f"GPU orig frob (float64)={torch.linalg.norm(orig_gpu.double(), ord='fro').item():.8f}")
