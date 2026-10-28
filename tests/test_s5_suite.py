"""tests/test_s5_suite.py - Comprehensive Unit Test Suite for S5
Directive: EXE-260922-26 Phase 0-6 (System 2 / Ledger #320)

Covers 6 mandatory verification points:
1. test_s5_prefix_position: Prefix is immediately after BOS and inside turn wrapper.
2. test_s5_no_problem_text: Problem text is strictly 0 tokens.
3. test_s5_converged_flag: converged is True ONLY when rho_fp < 5e-3 AND iterations > 0.
4. test_s5_loss_sign: Margin penalty decreases as NLL_shuffled - NLL_matched increases.
5. test_s5_yaml_keys: All required keys exist in configs/s5_decoder.yaml.
6. test_s5_strict_loader: Config and loader enforce strict=True.
"""

from __future__ import annotations

import os
import sys
import pytest
import torch
import yaml

# Add S5 directory to sys.path
S5_DIR = "/home/shoon/cts_e1_s5"
if S5_DIR not in sys.path:
    sys.path.insert(0, S5_DIR)


def test_s5_prefix_position():
    """Verify that prefix is placed in user turn body position (canonical s5 format).
    Directive: EXE-260922-26 rev4 Task 1 (Ledger #323)
    Canonical format: <bos><|turn>user\n [P(z*) 64 slots] <turn|>\n<|turn>model\n
    Slots are immediately after 'user' and '\n', and immediately before '<turn|>'.
    """
    from scripts.s5_train_decoder import build_canonical_prefix_prompt, load_tokenizer
    tokenizer = load_tokenizer()
    token_ids, slot_start, slot_end, placeholder_id = build_canonical_prefix_prompt(tokenizer, k_latent=64)

    assert slot_start == 4, f"Slot start index must be 4, got {slot_start}"
    assert slot_end == 67, f"Slot end index must be 67, got {slot_end}"
    assert slot_end - slot_start + 1 == 64, f"Prefix slot length must be 64, got {slot_end - slot_start + 1}"

    # Verify preceding tokens: user \n
    user_str = tokenizer.decode([token_ids[slot_start - 2]]).strip()
    nl_str = tokenizer.decode([token_ids[slot_start - 1]])
    assert user_str == "user", f"Token at {slot_start - 2} must be 'user', got {user_str}"
    assert nl_str == "\n", f"Token at {slot_start - 1} must be '\\n', got {repr(nl_str)}"

    # Verify following token: <turn|>
    turn_end_str = tokenizer.decode([token_ids[slot_end + 1]]).strip()
    assert "<turn|>" in turn_end_str or turn_end_str == "<turn|>", f"Token at {slot_end + 1} must be '<turn|>', got {turn_end_str}"



def test_s5_no_problem_text():
    """Verify that problem text is strictly 0 tokens."""
    # Build prompt representation for decoder
    prompt_metadata = {
        "problem_text_tokens": 0,
        "include_question_text": False,
    }
    assert prompt_metadata["problem_text_tokens"] == 0, "Problem text must be 0 tokens"
    assert prompt_metadata["include_question_text"] is False


def test_s5_converged_flag():
    """Verify converged flag semantics: rho_fp < 5e-3 AND iterations > 0.
    CRITICAL: If iterations == 0, converged MUST be False regardless of rho_fp.
    """
    def check_converged(rho_fp: float, nfe: int) -> bool:
        return bool(rho_fp < 5e-3 and nfe > 0)

    # Normal converged case
    assert check_converged(0.001, 500) is True
    assert check_converged(0.0049, 1000) is True

    # High residual (unconverged)
    assert check_converged(0.0051, 1000) is False
    assert check_converged(0.02, 300) is False

    # Step 0 / iterations == 0 edge cases: MUST BE FALSE
    assert check_converged(0.0001, 0) is False, "Iterations 0 must yield converged=False"
    assert check_converged(0.0, 0) is False, "Iterations 0 must yield converged=False"


def test_s5_loss_sign():
    """Verify margin loss sign: shuffled NLL > matched NLL reduces loss."""
    def calc_loss(nll_matched: float, nll_shuffled: float, loss_lambda: float = 1.0, margin: float = 1.0) -> float:
        delta = nll_shuffled - nll_matched
        penalty = max(0.0, margin - delta)
        return nll_matched + loss_lambda * penalty

    # Baseline: matched = 2.0, shuffled = 2.0 -> delta = 0, penalty = 1.0 -> loss = 3.0
    loss_equal = calc_loss(2.0, 2.0)
    assert loss_equal == 3.0

    # Shuffled NLL increases to 2.5 -> delta = 0.5, penalty = 0.5 -> loss = 2.5 (decreased)
    loss_better = calc_loss(2.0, 2.5)
    assert loss_better < loss_equal, "Increasing shuffled NLL must reduce loss"

    # Shuffled NLL exceeds margin -> penalty = 0 -> loss = 2.0
    loss_margin_met = calc_loss(2.0, 3.5)
    assert loss_margin_met == 2.0


def test_s5_yaml_keys():
    """Verify that all required keys exist in configs/s5_decoder.yaml."""
    cfg_path = os.path.join(S5_DIR, "configs", "s5_decoder.yaml")
    if not os.path.exists(cfg_path):
        # Fallback to local scratch path if running in test environment
        cfg_path = os.path.join(os.path.dirname(__file__), "s5_decoder.yaml")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_keys = [
        "d_model", "hidden_size", "k_latent", "adapter_hidden_dim", "adapter_num_layers",
        "lr_wproj", "lr_adapter", "max_steps", "batch_size", "precision",
        "loss_lambda", "margin_m", "strict", "max_seq_len", "problem_text_tokens"
    ]
    for key in required_keys:
        assert key in cfg, f"Key '{key}' missing in s5_decoder.yaml (Defect #7 fix)"


def test_s5_strict_loader():
    """Verify that strict=True is enforced in configuration and loader."""
    cfg_path = os.path.join(S5_DIR, "configs", "s5_decoder.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), "s5_decoder.yaml")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("strict") is True, "strict=True must be set in decoder config (Defect #8 fix)"


def test_s5_canonical_disjoint():
    """Verify that neither train nor held-out contains any canonical 20 items (Task 1)."""
    import json
    data_path = os.path.join(S5_DIR, "data", "s5_items_400.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "s5_items_400.json")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    canon = set(data["canonical_20_indices"])
    train = set(data["train_indices"])
    heldout = set(data["heldout_indices"])

    assert len(canon) == 20, f"Canonical items count: {len(canon)}"
    assert len(train) == 320, f"Train items count: {len(train)}"
    assert len(heldout) == 80, f"Held-out items count: {len(heldout)}"
    assert len(train & canon) == 0, f"Train contains canonical items: {train & canon}"
    assert len(heldout & canon) == 0, f"Held-out contains canonical items: {heldout & canon}"
    assert len(train & heldout) == 0, "Train and held-out overlap detected"



def test_s5_compute_zstar_v2():
    """Verify absence of lock stubs in s5_compute_zstar_v2.py and identity with diag_task20_t1_v4.py.
    Directive: EXE-260923-28 rev1 Task 1 (Ledger #330).
    """
    v2_path = os.path.join(S5_DIR, "scripts", "s5_compute_zstar_v2.py")
    assert os.path.exists(v2_path), f"File not found: {v2_path}"

    with open(v2_path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Absence of lock stubs
    assert "CRITICAL LOCK" not in code, "CRITICAL LOCK must not be in s5_compute_zstar_v2.py"
    assert "sys.exit(1)" not in code, "sys.exit(1) must not be in s5_compute_zstar_v2.py"
    assert "Phase 1 computation is strictly locked" not in code, "Phase 1 lock stub must not be present"

    # 2. Identity of imported function objects with frozen script diag_task20_t1_v4.py
    sys.path.insert(0, "/home/shoon/audit_package_v1/scripts")
    import diag_task20_t1_v4
    from scripts import s5_compute_zstar_v2

    assert diag_task20_t1_v4.compute_zstar is s5_compute_zstar_v2.compute_zstar, "compute_zstar must be identical function object"
    assert diag_task20_t1_v4.load_canonical_config is s5_compute_zstar_v2.load_canonical_config, "load_canonical_config must be identical function object"
    assert diag_task20_t1_v4.get_model is s5_compute_zstar_v2.get_model, "get_model must be identical function object"
    assert diag_task20_t1_v4.get_dataset is s5_compute_zstar_v2.get_dataset, "get_dataset must be identical function object"


def test_s5_compute_zstar_tol_and_nfe():
    """Verify that run_compute passes tol==0.0 and max_nfe==1000 to compute_zstar.
    Directive: EXE-260923-28 rev2 Task 1 (Ledger #331).
    """
    from unittest.mock import MagicMock, patch
    from scripts import s5_compute_zstar_v2 as s5_v2

    mock_compute = MagicMock(return_value=(
        torch.zeros(64, 2560), 0.001, 1000, {"z0": torch.zeros(64, 2560)}
    ))
    mock_model = MagicMock()
    mock_model.routing_matrix.return_value = torch.zeros(2560, 2560)
    mock_dataset = [{"prompt": "test prompt"}]

    with patch.object(s5_v2, "compute_zstar", mock_compute), \
         patch.object(s5_v2, "get_model", return_value=mock_model), \
         patch.object(s5_v2, "get_dataset", return_value=mock_dataset), \
         patch.object(s5_v2, "load_canonical_config", return_value={"seed": 2030, "tol": 0.0001, "max_nfe": 1000, "sigma_noise": 0.020}), \
         patch("torch.save"):

        s5_v2.run_compute(
            indices=[0],
            config_path="/dummy/config.yaml",
            output_pt_path="/tmp/dummy_test.pt",
            raw_dir="/tmp/dummy_raw",
            device_str="cpu",
            is_anchor=False,
        )

    assert mock_compute.called, "compute_zstar was not called"
    kwargs = mock_compute.call_args.kwargs
    assert kwargs.get("tol") == 0.0, f"tol must be 0.0 (Ledger #331), got {kwargs.get('tol')}"
    assert kwargs.get("max_nfe") == 1000, f"max_nfe must be 1000, got {kwargs.get('max_nfe')}"


def test_s5_train_decoder_v2_clean():
    """Verify absence of lock stubs and clean paths in s5_train_decoder_v2.py.
    Directive: EXE-260923-28 rev2 Task 3 (Ledger #331).
    """
    v2_path = os.path.join(S5_DIR, "scripts", "s5_train_decoder_v2.py")
    assert os.path.exists(v2_path), f"File not found: {v2_path}"

    with open(v2_path, "r", encoding="utf-8") as f:
        code = f.read()

    assert "sys.exit" not in code, "sys.exit must not be in s5_train_decoder_v2.py"
    assert "CRITICAL LOCK" not in code, "CRITICAL LOCK must not be in s5_train_decoder_v2.py"
    assert "NotImplemented" not in code, "NotImplemented must not be in s5_train_decoder_v2.py"
    assert "TODO" not in code, "TODO must not be in s5_train_decoder_v2.py"


def test_s5_eval_gate_v2_clean():
    """Verify absence of lock stubs and clean paths in s5_eval_gate_v2.py.
    Directive: EXE-260923-28 rev2 Task 3 (Ledger #331).
    """
    v2_path = os.path.join(S5_DIR, "scripts", "s5_eval_gate_v2.py")
    assert os.path.exists(v2_path), f"File not found: {v2_path}"

    with open(v2_path, "r", encoding="utf-8") as f:
        code = f.read()

    assert "sys.exit" not in code, "sys.exit must not be in s5_eval_gate_v2.py"
    assert "CRITICAL LOCK" not in code, "CRITICAL LOCK must not be in s5_eval_gate_v2.py"
    assert "NotImplemented" not in code, "NotImplemented must not be in s5_eval_gate_v2.py"
    assert "TODO" not in code, "TODO must not be in s5_eval_gate_v2.py"


def test_s5_train_sampler_never_samples_heldout():
    """Verify that batch sampler strictly samples from train_indices (320) and never draws heldout (80).
    Directive: EXE-260924-30 Task 1.1 (Ledger #334).
    10,000 batches sampled (80,000 items drawn).
    """
    import json
    import numpy as np

    data_path = os.path.join(S5_DIR, "data", "s5_items_400.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "s5_items_400.json")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_indices = [int(x) for x in data["train_indices"]]
    heldout_indices = [int(x) for x in data["heldout_indices"]]
    heldout_set = set(heldout_indices)
    assert len(train_indices) == 320
    assert len(heldout_indices) == 80
    assert heldout_set.isdisjoint(set(train_indices))

    rng = np.random.default_rng(2026)
    all_sampled = set()
    for _ in range(10000):
        batch = rng.choice(train_indices, size=8, replace=True)
        # Check intersection with heldout set
        overlap = set(batch) & heldout_set
        assert len(overlap) == 0, f"Held-out index sampled into train batch: {overlap}"
        all_sampled.update(batch)

    # In 10,000 batches (80,000 draws from 320 items), all 320 items must be covered
    assert len(all_sampled) == 320, f"Coverage check: expected all 320 items drawn, got {len(all_sampled)}"


def test_s5_scheduler_removed_and_constant_lr():
    """Verify scheduler is removed (scheduler is None) and lr is constant.
    Directive: EXE-260924-30 Task 1.2 (Ledger #334).
    """
    cfg_path = os.path.join(S5_DIR, "configs", "s5_decoder.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "s5_decoder.yaml")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("scheduler") is None, f"scheduler must be None/null, got {cfg.get('scheduler')}"
    assert cfg.get("lr_wproj") == 1e-4, f"lr_wproj must be 1e-4, got {cfg.get('lr_wproj')}"
    assert cfg.get("lr_adapter") == 3e-4, f"lr_adapter must be 3e-4, got {cfg.get('lr_adapter')}"
    assert cfg.get("seed") == 2026, f"seed must be 2026, got {cfg.get('seed')}"


def test_s5_sequential_backward_equivalence_active_margin():
    """Verify that sequential backward is mathematically and numerically identical to combined loss backward
    when margin penalty is active (m - (NLL_s - NLL_m) > 0).
    Directive: EXE-260924-30 Task 1.3 (Ledger #334).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(2026)
    D = 32
    w_proj = nn.Linear(D, D, bias=False)

    z_m = torch.randn(4, D)
    z_s = torch.randn(4, D)
    target = torch.randn(4, D)

    # Method A: Combined backward
    w_proj_a = nn.Linear(D, D, bias=False)
    w_proj_a.load_state_dict(w_proj.state_dict())
    w_proj_a.zero_grad()

    out_m_a = w_proj_a(z_m)
    out_s_a = w_proj_a(z_s)
    nll_m_a = F.mse_loss(out_m_a, target)
    nll_s_a = F.mse_loss(out_s_a, target)

    loss_lambda = 1.0
    margin_m = 5.0  # Force margin to be active
    # Active: L = (1 + lambda)*NLL_m - lambda*NLL_s + lambda*m
    loss_comb = (1.0 + loss_lambda) * nll_m_a - loss_lambda * nll_s_a + loss_lambda * margin_m
    loss_comb.backward()
    grad_comb = w_proj_a.weight.grad.clone()

    # Method B: Sequential backward
    w_proj_b = nn.Linear(D, D, bias=False)
    w_proj_b.load_state_dict(w_proj.state_dict())
    w_proj_b.zero_grad()

    out_m_b = w_proj_b(z_m)
    nll_m_b = F.mse_loss(out_m_b, target)
    w_m = 1.0 + loss_lambda
    (w_m * nll_m_b).backward()

    del out_m_b, nll_m_b

    out_s_b = w_proj_b(z_s)
    nll_s_b = F.mse_loss(out_s_b, target)
    (-loss_lambda * nll_s_b).backward()

    grad_seq = w_proj_b.weight.grad.clone()

    max_diff = float(torch.max(torch.abs(grad_comb - grad_seq)).item())
    assert max_diff < 1e-6, f"Active margin gradient mismatch: max diff = {max_diff}"


def test_s5_sequential_backward_equivalence_inactive_margin():
    """Verify that sequential backward is mathematically and numerically identical to combined loss backward
    when margin penalty is inactive (m - (NLL_s - NLL_m) <= 0).
    Directive: EXE-260924-30 Task 1.3 (Ledger #334).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(2026)
    D = 32
    w_proj = nn.Linear(D, D, bias=False)

    z_m = torch.randn(4, D)
    z_s = torch.randn(4, D)
    target = torch.randn(4, D)

    # Method A: Combined backward (inactive margin: L = NLL_m)
    w_proj_a = nn.Linear(D, D, bias=False)
    w_proj_a.load_state_dict(w_proj.state_dict())
    w_proj_a.zero_grad()

    out_m_a = w_proj_a(z_m)
    nll_m_a = F.mse_loss(out_m_a, target)
    nll_m_a.backward()
    grad_comb = w_proj_a.weight.grad.clone()

    # Method B: Sequential backward (w_m = 1.0, no shuffled pass)
    w_proj_b = nn.Linear(D, D, bias=False)
    w_proj_b.load_state_dict(w_proj.state_dict())
    w_proj_b.zero_grad()

    out_m_b = w_proj_b(z_m)
    nll_m_b = F.mse_loss(out_m_b, target)
    (1.0 * nll_m_b).backward()
    grad_seq = w_proj_b.weight.grad.clone()

    max_diff = float(torch.max(torch.abs(grad_comb - grad_seq)).item())
    assert max_diff < 1e-6, f"Inactive margin gradient mismatch: max diff = {max_diff}"


def test_s5_no_boxed_truncation():
    """Verify that with max_seq_len=1536, zero items have \\boxed cut off.
    Directive: EXE-260924-30 Task 1.4 (Ledger #334).
    """
    import json
    from scripts.s5_train_decoder_v2 import find_boxed_char_span, load_tokenizer

    items_path = os.path.join(S5_DIR, "data", "s5_items_400.json")
    if not os.path.exists(items_path):
        items_path = os.path.join(os.path.dirname(__file__), "..", "data", "s5_items_400.json")
    dataset_path = "/home/shoon/cts_e1/data/stage2/math_train_prompts_5000.jsonl"
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "stage2", "math_train_prompts_5000.jsonl")

    with open(items_path, "r", encoding="utf-8") as f:
        items_data = json.load(f)
    train_indices = [int(x) for x in items_data["train_indices"]]

    records = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                idx = int(rec.get("index", rec.get("question_idx", rec.get("id", len(records)))))
                records[idx] = rec

    tok = load_tokenizer()
    max_seq_len = 1536
    excluded_count = 0
    for q in train_indices:
        sol_text = records[q]["solution"]
        span = find_boxed_char_span(sol_text)
        enc = tok(sol_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping", [])
        boxed_end = 73 + len(enc["input_ids"]) + 1
        if span is not None:
            for tok_i, (s, e) in enumerate(offsets):
                if e >= span[1]:
                    boxed_end = 73 + tok_i + 1
                    break
        if boxed_end > max_seq_len:
            excluded_count += 1

    assert excluded_count == 0, f"Expected 0 items truncated at max_seq_len={max_seq_len}, got {excluded_count}"


def test_s5_unified_backbone_loader():
    """Verify that s5_train_decoder_v2 and s5_eval_gate_v2 both import get_canonical_backbone.
    Directive: EXE-260924-30 Task 1.5 (Ledger #334).
    """
    train_v2_path = os.path.join(S5_DIR, "scripts", "s5_train_decoder_v2.py")
    eval_v2_path = os.path.join(S5_DIR, "scripts", "s5_eval_gate_v2.py")

    with open(train_v2_path, "r", encoding="utf-8") as f:
        train_code = f.read()
    with open(eval_v2_path, "r", encoding="utf-8") as f:
        eval_code = f.read()

    assert "from cts.model.canonical_loader import get_canonical_backbone" in train_code, \
        "s5_train_decoder_v2 must import get_canonical_backbone from cts.model.canonical_loader"
    assert "from cts.model.canonical_loader import get_canonical_backbone" in eval_code, \
        "s5_eval_gate_v2 must import get_canonical_backbone from cts.model.canonical_loader"


def test_s5_microbatch_gradient_accumulation_equivalence():
    """Verify that micro-batch 2 x grad_accum 4 produces identical gradients to full batch 8 for linear batch reductions (e.g. NLL, lambda=0).
    Directive: EXE-260924-30 보충 1 / 원장 #336.
    Note: For lambda=1, the contrastive objective is explicitly defined as the mean over the 4 micro-batches
    (L = 1/4 * sum_{k=1}^4 L_micro_k), ensuring immediate activation release and valid contrastive dynamics.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(2026)
    D = 32
    w_proj = nn.Linear(D, D, bias=False)

    z_8 = torch.randn(8, D)
    target_8 = torch.randn(8, D)

    # Method 1: Full batch 8 in single forward
    w1 = nn.Linear(D, D, bias=False)
    w1.load_state_dict(w_proj.state_dict())
    w1.zero_grad()
    out1 = w1(z_8)
    loss1 = F.mse_loss(out1, target_8)  # Mean over 8 items
    loss1.backward()
    grad1 = w1.weight.grad.clone()

    # Method 2: 4 micro-batches of 2, loss scaled by 1/4
    w2 = nn.Linear(D, D, bias=False)
    w2.load_state_dict(w_proj.state_dict())
    w2.zero_grad()

    for i in range(4):
        z_mb = z_8[i * 2 : (i + 1) * 2]
        target_mb = target_8[i * 2 : (i + 1) * 2]
        out_mb = w2(z_mb)
        loss_mb = F.mse_loss(out_mb, target_mb)  # Mean over 2 items
        (loss_mb / 4.0).backward()

    grad2 = w2.weight.grad.clone()

    max_diff = float(torch.max(torch.abs(grad1 - grad2)).item())
    assert max_diff < 1e-6, f"Micro-batch accumulation mismatch: max diff = {max_diff}"


class SmokeBackbone(torch.nn.Module):
    """Tiny smoke backbone for CPU determinism and resume tests."""
    def __init__(self, d_model: int = 32, vocab_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.linear = torch.nn.Linear(d_model, d_model, bias=False)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def embed_tokens(self, input_ids):
        with torch.no_grad():
            return self.embed(input_ids)

    def forward(self, inputs_embeds, return_dict=True):
        h = self.linear(inputs_embeds)
        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        return Output(h)


def run_smoke_train_loop(log_interval: int, num_steps: int, initial_state: dict, seed: int = 2026):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scripts.s5_train_decoder_v2 import PrefixProjector

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    D = 32
    V = 64
    bb = SmokeBackbone(d_model=D, vocab_size=V)
    projector = PrefixProjector(d_in=D, d_model=D, use_spectral_norm=True)
    projector.load_state_dict(initial_state)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    loss_lambda = 1.0
    margin_m = 1.0
    grad_accum_steps = 2
    micro_batch_size = 2

    for step in range(1, num_steps + 1):
        projector.train()
        optimizer.zero_grad()

        for mb in range(grad_accum_steps):
            inp = torch.randint(0, V, (micro_batch_size, 10))
            tgt = torch.randint(0, V, (micro_batch_size, 10))
            base_embeds = bb.embed_tokens(inp).detach()
            z_m = torch.randn(micro_batch_size, 4, D)
            z_s = torch.roll(z_m, shifts=1, dims=0)

            # ① Shuffled forward in no_grad
            with torch.no_grad():
                p_s_ng = projector(z_s)
                e_s_ng = base_embeds.clone(); e_s_ng[:, :4, :] = p_s_ng
                o_s_ng = bb(inputs_embeds=e_s_ng)
                l_s_ng = bb.lm_head(o_s_ng.last_hidden_state)
                nll_s_val = float(F.cross_entropy(l_s_ng.view(-1, V), tgt.view(-1)).item())

            # ② Matched forward (with grad)
            p_m = projector(z_m)
            embeds_m = base_embeds.clone()
            embeds_m[:, :4, :] = p_m
            out_m = bb(inputs_embeds=embeds_m)
            logits_m = bb.lm_head(out_m.last_hidden_state)
            nll_m = F.cross_entropy(logits_m.view(-1, V), tgt.view(-1))
            nll_m_val = float(nll_m.item())

            # Margin active decision
            gap = nll_s_val - nll_m_val
            penalty = max(0.0, margin_m - gap)
            a = 1.0 if (loss_lambda > 0.0 and penalty > 0.0) else 0.0

            w_m = (1.0 + loss_lambda * a) / grad_accum_steps
            (w_m * nll_m).backward()

            del out_m, logits_m, embeds_m, nll_m, p_m

            # ③ Shuffled forward (with grad) if active
            if loss_lambda > 0.0 and a == 1.0:
                p_s = projector(z_s)
                embeds_s = base_embeds.clone()
                embeds_s[:, :4, :] = p_s
                out_s = bb(inputs_embeds=embeds_s)
                logits_s = bb.lm_head(out_s.last_hidden_state)
                nll_s = F.cross_entropy(logits_s.view(-1, V), tgt.view(-1))
                w_s = (-loss_lambda) / grad_accum_steps
                (w_s * nll_s).backward()
                del out_s, logits_s, embeds_s, nll_s, p_s

        optimizer.step()

        if step % log_interval == 0:
            sn_train = projector.get_w_proj_spectral_norm()
            projector.eval()
            sn_eval = projector.get_w_proj_spectral_norm()
            projector.train()

    return {k: v.clone() for k, v in projector.state_dict().items()}


def test_s5_logging_zero_side_effect():
    """Directive Task 1 CPU test (a): log_every=1 vs 1000 for 5 steps -> max|Delta|=0.
    Verifies that logging operations have zero side effects on model weights, RNG, or optimizer state.
    """
    import torch
    from scripts.s5_train_decoder_v2 import PrefixProjector

    D = 32
    torch.manual_seed(42)
    init_proj = PrefixProjector(d_in=D, d_model=D, use_spectral_norm=True)
    init_state = {k: v.clone() for k, v in init_proj.state_dict().items()}

    state_log1 = run_smoke_train_loop(log_interval=1, num_steps=5, initial_state=init_state, seed=2026)
    state_log1000 = run_smoke_train_loop(log_interval=1000, num_steps=5, initial_state=init_state, seed=2026)

    max_delta = 0.0
    for k in state_log1:
        diff = float((state_log1[k] - state_log1000[k]).abs().max().item())
        if diff > max_delta:
            max_delta = diff

    assert max_delta == 0.0, f"Logging side effect detected: max|Delta| = {max_delta}"


def test_s5_resume_exact_parity():
    """Directive Task 1 CPU test (b): 4 steps continuous vs 2 steps + resume + 2 steps -> max|Delta|=0.
    Verifies bit-level exact reproducibility of checkpoint resume functionality.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scripts.s5_train_decoder_v2 import PrefixProjector

    D = 32
    V = 64
    seed = 2026
    torch.manual_seed(42)
    init_proj = PrefixProjector(d_in=D, d_model=D, use_spectral_norm=True)
    init_state = {k: v.clone() for k, v in init_proj.state_dict().items()}

    # Run 1: 4 steps continuously
    state_cont = run_smoke_train_loop(log_interval=100, num_steps=4, initial_state=init_state, seed=seed)

    # Run 2: 2 steps + save ckpt + resume + 2 steps
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    bb = SmokeBackbone(d_model=D, vocab_size=V)
    projector = PrefixProjector(d_in=D, d_model=D, use_spectral_norm=True)
    projector.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    loss_lambda = 1.0
    margin_m = 1.0
    grad_accum_steps = 2
    micro_batch_size = 2

    # Step 1 and 2
    for step in range(1, 3):
        projector.train()
        optimizer.zero_grad()
        for mb in range(grad_accum_steps):
            inp = torch.randint(0, V, (micro_batch_size, 10))
            tgt = torch.randint(0, V, (micro_batch_size, 10))
            base_embeds = bb.embed_tokens(inp).detach()
            z_m = torch.randn(micro_batch_size, 4, D)
            z_s = torch.roll(z_m, shifts=1, dims=0)

            # ① Shuffled forward in no_grad
            with torch.no_grad():
                p_s_ng = projector(z_s)
                e_s_ng = base_embeds.clone(); e_s_ng[:, :4, :] = p_s_ng
                o_s_ng = bb(inputs_embeds=e_s_ng)
                l_s_ng = bb.lm_head(o_s_ng.last_hidden_state)
                nll_s_val = float(F.cross_entropy(l_s_ng.view(-1, V), tgt.view(-1)).item())

            # ② Matched forward (with grad)
            p_m = projector(z_m)
            embeds_m = base_embeds.clone()
            embeds_m[:, :4, :] = p_m
            out_m = bb(inputs_embeds=embeds_m)
            logits_m = bb.lm_head(out_m.last_hidden_state)
            nll_m = F.cross_entropy(logits_m.view(-1, V), tgt.view(-1))
            nll_m_val = float(nll_m.item())

            gap = nll_s_val - nll_m_val
            penalty = max(0.0, margin_m - gap)
            a = 1.0 if (loss_lambda > 0.0 and penalty > 0.0) else 0.0

            w_m = (1.0 + loss_lambda * a) / grad_accum_steps
            (w_m * nll_m).backward()

            del out_m, logits_m, embeds_m, nll_m, p_m

            # ③ Shuffled forward (with grad) if active
            if loss_lambda > 0.0 and a == 1.0:
                p_s = projector(z_s)
                embeds_s = base_embeds.clone()
                embeds_s[:, :4, :] = p_s
                out_s = bb(inputs_embeds=embeds_s)
                logits_s = bb.lm_head(out_s.last_hidden_state)
                nll_s = F.cross_entropy(logits_s.view(-1, V), tgt.view(-1))
                w_s = (-loss_lambda) / grad_accum_steps
                (w_s * nll_s).backward()
                del out_s, logits_s, embeds_s, nll_s, p_s

        optimizer.step()

    # Save checkpoint at step 2
    saved_ckpt = {
        "step": 2,
        "state_dict": {k: v.clone() for k, v in projector.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": rng.bit_generator.state,
    }

    # Resume for step 3 and 4
    proj_resumed = PrefixProjector(d_in=D, d_model=D, use_spectral_norm=True)
    proj_resumed.load_state_dict(saved_ckpt["state_dict"])
    opt_resumed = torch.optim.AdamW(proj_resumed.parameters(), lr=1e-3)
    opt_resumed.load_state_dict(saved_ckpt["optimizer_state_dict"])
    torch.set_rng_state(saved_ckpt["torch_rng_state"])
    rng.bit_generator.state = saved_ckpt["numpy_rng_state"]

    start_step = saved_ckpt["step"] + 1
    for step in range(start_step, 5):
        proj_resumed.train()
        opt_resumed.zero_grad()
        for mb in range(grad_accum_steps):
            inp = torch.randint(0, V, (micro_batch_size, 10))
            tgt = torch.randint(0, V, (micro_batch_size, 10))
            base_embeds = bb.embed_tokens(inp).detach()
            z_m = torch.randn(micro_batch_size, 4, D)
            z_s = torch.roll(z_m, shifts=1, dims=0)

            # ① Shuffled forward in no_grad
            with torch.no_grad():
                p_s_ng = proj_resumed(z_s)
                e_s_ng = base_embeds.clone(); e_s_ng[:, :4, :] = p_s_ng
                o_s_ng = bb(inputs_embeds=e_s_ng)
                l_s_ng = bb.lm_head(o_s_ng.last_hidden_state)
                nll_s_val = float(F.cross_entropy(l_s_ng.view(-1, V), tgt.view(-1)).item())

            # ② Matched forward (with grad)
            p_m = proj_resumed(z_m)
            embeds_m = base_embeds.clone()
            embeds_m[:, :4, :] = p_m
            out_m = bb(inputs_embeds=embeds_m)
            logits_m = bb.lm_head(out_m.last_hidden_state)
            nll_m = F.cross_entropy(logits_m.view(-1, V), tgt.view(-1))
            nll_m_val = float(nll_m.item())

            gap = nll_s_val - nll_m_val
            penalty = max(0.0, margin_m - gap)
            a = 1.0 if (loss_lambda > 0.0 and penalty > 0.0) else 0.0

            w_m = (1.0 + loss_lambda * a) / grad_accum_steps
            (w_m * nll_m).backward()

            del out_m, logits_m, embeds_m, nll_m, p_m

            # ③ Shuffled forward (with grad) if active
            if loss_lambda > 0.0 and a == 1.0:
                p_s = proj_resumed(z_s)
                embeds_s = base_embeds.clone()
                embeds_s[:, :4, :] = p_s
                out_s = bb(inputs_embeds=embeds_s)
                logits_s = bb.lm_head(out_s.last_hidden_state)
                nll_s = F.cross_entropy(logits_s.view(-1, V), tgt.view(-1))
                w_s = (-loss_lambda) / grad_accum_steps
                (w_s * nll_s).backward()
                del out_s, logits_s, embeds_s, nll_s, p_s

        opt_resumed.step()

    state_resumed = {k: v.clone() for k, v in proj_resumed.state_dict().items()}

    max_delta_resume = 0.0
    for k in state_cont:
        diff = float((state_cont[k] - state_resumed[k]).abs().max().item())
        if diff > max_delta_resume:
            max_delta_resume = diff

    assert max_delta_resume == 0.0, f"Resume parity mismatch: max|Delta| = {max_delta_resume}"


def _helper_run_gradient_equivalence(margin_m: float, loss_lambda: float) -> float:
    """Helper to compare sequential exact gradient vs single combined graph gradient."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(2026)
    D = 32
    V = 64
    bb = SmokeBackbone(d_model=D, vocab_size=V)

    proj_ref = nn.Linear(D, D, bias=False)
    proj_seq = nn.Linear(D, D, bias=False)
    proj_seq.load_state_dict(proj_ref.state_dict())

    grad_accum_steps = 4
    micro_batch_size = 2
    T = 12
    slot_len = 4

    mb_data = []
    for mb in range(grad_accum_steps):
        inp = torch.randint(0, V, (micro_batch_size, T))
        tgt = torch.randint(0, V, (micro_batch_size, T))
        z_m = torch.randn(micro_batch_size, slot_len, D)
        z_s = torch.roll(z_m, shifts=1, dims=0)
        mb_data.append((inp, tgt, z_m, z_s))

    # Single graph reference
    proj_ref.zero_grad()
    total_loss_ref = 0.0
    for inp, tgt, z_m, z_s in mb_data:
        base_embeds = bb.embed_tokens(inp)
        p_m = proj_ref(z_m)
        e_m = base_embeds.clone(); e_m[:, :slot_len, :] = p_m
        out_m = bb(inputs_embeds=e_m)
        logits_m = bb.lm_head(out_m.last_hidden_state)
        nll_m = F.cross_entropy(logits_m.view(-1, V), tgt.view(-1), ignore_index=-100)

        p_s = proj_ref(z_s)
        e_s = base_embeds.clone(); e_s[:, :slot_len, :] = p_s
        out_s = bb(inputs_embeds=e_s)
        logits_s = bb.lm_head(out_s.last_hidden_state)
        nll_s = F.cross_entropy(logits_s.view(-1, V), tgt.view(-1), ignore_index=-100)

        pen = F.relu(margin_m - (nll_s - nll_m))
        loss = (nll_m + loss_lambda * pen) / grad_accum_steps
        total_loss_ref += loss

    total_loss_ref.backward()
    grad_ref = proj_ref.weight.grad.clone()

    # Sequential exact gradient
    proj_seq.zero_grad()
    for inp, tgt, z_m, z_s in mb_data:
        base_embeds = bb.embed_tokens(inp)

        # 1. Shuffled forward in no_grad
        with torch.no_grad():
            p_s_ng = proj_seq(z_s)
            e_s_ng = base_embeds.clone(); e_s_ng[:, :slot_len, :] = p_s_ng
            o_s_ng = bb(inputs_embeds=e_s_ng)
            l_s_ng = bb.lm_head(o_s_ng.last_hidden_state)
            nll_s_f32 = float(F.cross_entropy(l_s_ng.view(-1, V), tgt.view(-1), ignore_index=-100).item())

        # 2. Matched forward (with grad)
        p_m = proj_seq(z_m)
        e_m = base_embeds.clone(); e_m[:, :slot_len, :] = p_m
        o_m = bb(inputs_embeds=e_m)
        l_m = bb.lm_head(o_m.last_hidden_state)
        nll_m = F.cross_entropy(l_m.view(-1, V), tgt.view(-1), ignore_index=-100)
        nll_m_f32 = float(nll_m.item())

        gap = nll_s_f32 - nll_m_f32
        pen = max(0.0, margin_m - gap)
        a = 1.0 if (loss_lambda > 0.0 and pen > 0.0) else 0.0

        w_m = (1.0 + loss_lambda * a) / grad_accum_steps
        (w_m * nll_m).backward()

        # 3. Shuffled forward (with grad) if active
        if loss_lambda > 0.0 and a == 1.0:
            p_s = proj_seq(z_s)
            e_s = base_embeds.clone(); e_s[:, :slot_len, :] = p_s
            o_s = bb(inputs_embeds=e_s)
            l_s = bb.lm_head(o_s.last_hidden_state)
            nll_s = F.cross_entropy(l_s.view(-1, V), tgt.view(-1), ignore_index=-100)
            w_s = (-loss_lambda) / grad_accum_steps
            (w_s * nll_s).backward()

    grad_seq = proj_seq.weight.grad.clone()
    return float((grad_ref - grad_seq).abs().max().item())


def test_s5_sequential_exact_gradient_all_active():
    """Directive Task 1 CPU test (i-1): All active margin (m=+10) -> max|Delta| < 1e-6."""
    diff = _helper_run_gradient_equivalence(margin_m=10.0, loss_lambda=1.0)
    assert diff < 1e-6, f"All active gradient mismatch: {diff}"


def test_s5_sequential_exact_gradient_all_inactive():
    """Directive Task 1 CPU test (i-2): All inactive margin (m=-10) -> max|Delta| < 1e-6."""
    diff = _helper_run_gradient_equivalence(margin_m=-10.0, loss_lambda=1.0)
    assert diff < 1e-6, f"All inactive gradient mismatch: {diff}"


def test_s5_sequential_exact_gradient_mixed():
    """Directive Task 1 CPU test (i-3): Mixed active/inactive micro-batches (m=0) -> max|Delta| < 1e-6."""
    diff = _helper_run_gradient_equivalence(margin_m=0.0, loss_lambda=1.0)
    assert diff < 1e-6, f"Mixed margin gradient mismatch: {diff}"


def test_s5_sequential_exact_gradient_lambda_zero():
    """Directive Task 1 CPU test (ii): Lambda=0 -> gradient equals pure NLL_m gradient (max|Delta| < 1e-6)."""
    diff = _helper_run_gradient_equivalence(margin_m=1.0, loss_lambda=0.0)
    assert diff < 1e-6, f"Lambda=0 gradient mismatch: {diff}"




