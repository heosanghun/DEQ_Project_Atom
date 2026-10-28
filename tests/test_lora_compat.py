from __future__ import annotations

import torch
import torch.nn as nn

from cts.train.lora_compat import LoraLinear


def test_fp32_lora_master_weights_accept_bf16_activations():
    base = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    layer = LoraLinear(base, rank=2, alpha=4, dropout=0.0)
    layer.lora_A.weight.data = layer.lora_A.weight.data.float()
    layer.lora_B.weight.data = layer.lora_B.weight.data.float()
    x = torch.randn(3, 4, dtype=torch.bfloat16)

    out = layer(x)
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert layer.lora_A.weight.grad is not None
    assert layer.lora_A.weight.grad.dtype == torch.float32
    assert layer.lora_B.weight.grad is not None
    assert layer.lora_B.weight.grad.dtype == torch.float32
