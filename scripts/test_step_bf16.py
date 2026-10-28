import sys
import torch
import torch.nn.functional as F

sys.path.append("/home/shoon/cts_e1/scratch")
sys.path.append("/home/shoon/cts_e1_s5")
sys.path.append("/home/shoon/cts_e1_s5/scripts")
from harness_A_v1 import get_model
from s5_train_decoder_v2 import PrefixProjector

device = "cuda:0"
bb = get_model(device)
bb.to(device=torch.device(device), dtype=torch.bfloat16)
bb.eval()
for p in bb.parameters():
    p.requires_grad = False

b1_path = "/home/shoon/retrain_1a/B1/stage1a_B1_step5000_20260907_0054.pt"
projector = PrefixProjector(d_in=2560, d_model=2560, use_spectral_norm=True).to(device)
b1_info = projector.load_b1_w_proj(b1_path, strict=True)
print("B1 loaded:", b1_info["frob_orig"], b1_info["spec_norm"])

lm = bb.cg.model.language_model
input_ids = torch.randint(0, 1000, (8, 400), device=device)
z_m = torch.randn(8, 64, 2560, device=device, dtype=torch.float32)
z_s = torch.roll(z_m, 1, 0)
target_start = 73

per_layer = lm.get_per_layer_inputs(input_ids, None)
base_embeds = lm.embed_tokens(input_ids)

torch.cuda.reset_peak_memory_stats()

# Matched pass
p_m = projector(z_m).to(dtype=base_embeds.dtype)
embeds_m = base_embeds.clone()
embeds_m[:, 4:68, :] = p_m
out_m = lm(inputs_embeds=embeds_m, per_layer_inputs=per_layer, return_dict=True)
target_hidden_m = out_m.last_hidden_state[:, target_start - 1:-1, :]
logits_m = bb.cg.lm_head(target_hidden_m)
loss_m = logits_m.float().mean()
loss_m.backward()

mem_m = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Matched pass backward OK! Peak mem: {mem_m:.2f} GiB")

del out_m, target_hidden_m, logits_m, embeds_m, p_m
torch.cuda.empty_cache()

# Shuffled pass
p_s = projector(z_s).to(dtype=base_embeds.dtype)
embeds_s = base_embeds.clone()
embeds_s[:, 4:68, :] = p_s
out_s = lm(inputs_embeds=embeds_s, per_layer_inputs=per_layer, return_dict=True)
target_hidden_s = out_s.last_hidden_state[:, target_start - 1:-1, :]
logits_s = bb.cg.lm_head(target_hidden_s)
loss_s = logits_s.float().mean()
(-loss_s).backward()

mem_s = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Shuffled pass backward OK! Peak mem: {mem_s:.2f} GiB")

w_grad = projector.w_proj.parametrizations.weight.original.grad
print(f"Gradient norm on w_proj: {w_grad.norm().item():.4f}")
print("ALL TESTS PASSED WITH BFLOAT16 BACKBONE!")
