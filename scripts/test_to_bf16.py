import sys
import torch

sys.path.append("/home/shoon/cts_e1/scratch")
from harness_A_v1 import get_model

device = "cuda:0"
bb = get_model(device)
print("Loaded in:", bb.cg.model.language_model.embed_tokens.weight.dtype)
bb.to(device=torch.device(device), dtype=torch.bfloat16)
print("Converted to:", bb.cg.model.language_model.embed_tokens.weight.dtype)
print("Allocated mem:", torch.cuda.memory_allocated() / (1024**3), "GiB")
