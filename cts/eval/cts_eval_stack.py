"""Strict shared CTS evaluation stack loader."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.policy.meta_policy import MetaPolicy
from cts.train.checkpoint_contracts import validate_stage2_final_contract
from cts.train.lora_compat import apply_paper_lora


def _load_torch(path: Path) -> Any:
    """Safe checkpoint load; never fall back to arbitrary pickle execution."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_family_id(value: object) -> str:
    text = str(value or "").rstrip("/\\")
    return text.replace("\\", "/").split("/")[-1].lower()


def load_cts_backbone_with_stage1(
    model: torch.nn.Module,
    tok: Any,
    *,
    cfg: Dict[str, Any],
    stage1_ckpt: Path | str = Path("artifacts/stage1_last.pt"),
) -> Any:
    """Load a complete Stage-1 contract into the matching backbone."""
    model_name = type(model).__name__.lower()
    config_name = type(model.config).__name__.lower()
    is_qwen = "qwen" in model_name or "qwen" in config_name
    if is_qwen:
        from cts.backbone.qwen_adapter import QwenCTSBackbone

        bb = QwenCTSBackbone(model, tok)
    else:
        bb = GemmaCTSBackbone(model, tok)

    s1 = Path(stage1_ckpt)
    if not s1.is_file():
        raise FileNotFoundError(
            f"headline CTS requires a validated Stage-1 checkpoint: {s1}"
        )
    ck = _load_torch(s1)
    if not isinstance(ck, dict):
        raise RuntimeError("Stage-1 checkpoint payload must be a mapping")
    sd = ck.get("backbone_state_dict")
    if not isinstance(sd, dict):
        raise RuntimeError("Stage-1 checkpoint has no backbone_state_dict")

    expected_model_id = str(
        getattr(model.config, "_name_or_path", "")
        or cfg.get("gemma_model_id", "")
    )
    from cts.train.stage2_ppo_train import validate_stage1_accuracy_contract

    contract = validate_stage1_accuracy_contract(
        ck,
        expected_model_id=expected_model_id,
    )
    is_gemma_ckpt = any("language_model" in k for k in sd)
    if is_qwen and is_gemma_ckpt:
        raise RuntimeError("Qwen evaluation cannot load a Gemma Stage-1 checkpoint")
    if not any(
        k.endswith("lora_A.weight") or k.endswith("lora_B.weight") for k in sd
    ):
        raise RuntimeError("validated Stage-1 checkpoint has no LoRA tensors")
    apply_paper_lora(
        bb,
        rank=int(contract["lora_rank"]),
        target_modules=tuple(contract["lora_targets"]),
        dropout=float(contract["lora_dropout"]),
        require_match=True,
        verbose=True,
    )
    bb.load_state_dict(sd, strict=True)
    bb.eval()
    return bb


def load_stage2_heads(
    *,
    meta_policy: MetaPolicy,
    critic: NeuroCritic,
    device: str,
    stage2_ckpt: Path | str = Path("artifacts/stage2_meta_value.pt"),
    stage1_ckpt: Path | str = Path("artifacts/stage1_last.pt"),
    expected_model_id: Optional[str] = None,
) -> Tuple[bool, bool]:
    """Load only a complete, provenance-matched Stage-2 contract."""
    s2 = Path(stage2_ckpt)
    if not s2.is_file():
        raise FileNotFoundError(
            f"headline CTS requires a validated Stage-2 checkpoint: {s2}"
        )
    ck = _load_torch(s2)
    s1 = Path(stage1_ckpt)
    if not s1.is_file():
        raise FileNotFoundError(f"Stage-2 provenance requires Stage-1: {s1}")
    if not expected_model_id:
        raise ValueError("expected_model_id is required for headline Stage-2")
    validate_stage2_final_contract(
        ck,
        expected_stage1_sha256=_sha256_file(s1),
        expected_model_id=expected_model_id,
    )

    meta_state = ck.get("meta_policy_state_dict")
    critic_state = ck.get("critic_state_dict")
    if not isinstance(meta_state, dict) or not isinstance(critic_state, dict):
        raise RuntimeError("Stage-2 checkpoint is missing canonical trained heads")
    meta_policy.load_state_dict(meta_state, strict=True)
    critic.load_state_dict(critic_state, strict=True)
    map_dev = torch.device(device)
    meta_policy.to(map_dev).eval()
    critic.to(map_dev).eval()
    print(
        f"  [ckpt] stage2={s2} (meta_policy=True, critic=True)",
        flush=True,
    )
    return True, True


def eval_tau_and_timeout(cfg: Dict[str, Any]) -> Tuple[float, float]:
    tau_budget = float(cfg.get("tau_flops_budget", 1e14))
    cap_raw = os.environ.get("CTS_EVAL_TAU_CAP")
    eval_tau = (
        min(tau_budget, float(cap_raw)) if cap_raw is not None else tau_budget
    )
    episode_timeout_s = float(os.environ.get("CTS_EVAL_EPISODE_TIMEOUT", "0"))
    return eval_tau, episode_timeout_s
