"""cts/model/canonical_loader.py - Canonical Backbone Loader for S5
Directive: EXE-260924-30 (System 2 / Ledger #334 Task 1.5)

Ensures identical backbone model configuration across all S5 training and evaluation scripts:
- Loads canonical GemmaCTSBackbone from harness_A_v1 (including LoRA rank 8 q/v/o)
- Loads frozen Stage-1a B1 checkpoint strictly
- Converts to specified dtype (default: torch.bfloat16)
- Sets bb.eval() and requires_grad=False on all parameters (Backbone: 0 trainable params)
- Zero duplicated implementation between s5_train_decoder_v2.py and s5_eval_gate_v2.py.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def get_canonical_backbone(
    device_str: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load canonical GemmaCTSBackbone with frozen LoRA from harness_A_v1 and convert to specified dtype."""
    from harness_A_v1 import get_model

    bb = get_model(device_str)
    if dtype != torch.float32:
        bb.to(device=torch.device(device_str), dtype=dtype)
    bb.eval()
    for p in bb.parameters():
        p.requires_grad = False
    return bb
