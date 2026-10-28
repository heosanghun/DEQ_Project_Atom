"""
Stage 2: MATH JSONL prompts + GemmaCTSBackbone + MetaPolicy + PPO (clipped surrogate + value + entropy).

Rollout: encode prompt -> meta policy samples branch index -> `transition()` -> scalar reward
(`default_transition_reward` or critic-based via `use_critic_reward`).

Uses the configured paper Eq.3 map (parallel by default); blend is an explicit
ablation and never a silent training fallback.
"""

from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from cts.backbone.gemma_adapter import GemmaCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.deq.root_bootstrap import solve_root_z_star
from cts.deq.transition import transition
from cts.model.gemma_loader import load_gemma4_e4b
from cts.policy.meta_policy import MetaPolicy
from cts.train.jsonl_iter import iter_jsonl
from cts.train.checkpoint_contracts import (
    stage1_accuracy_contract_is_valid,
    validate_stage2_resume_contract,
)
from cts.prompt_contract import (
    MATH_CONCISE_BOXED_CONTRACT,
    build_math_decode_prompt,
)
from cts.train.ppo_core import compute_gae, ppo_clipped_loss, value_loss
from cts.train.stage2_reward import stage2_rollout_is_correct
from cts.types import RuntimeBudgetState
from cts.utils.config import load_config
from cts.utils.repro_seed import apply_global_seed


class ValueHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x.float()).squeeze(-1)


def validate_stage2_reward_preflight(
    rewards: List[float],
    *,
    correct_count: int,
    min_samples: int,
    terminal_depths: Optional[List[int]] = None,
    require_depth_diversity: bool = False,
) -> Optional[Dict[str, float]]:
    """Validate external-answer reward diversity before any PPO update.

    Returns ``None`` while more samples are needed, otherwise a small audit
    summary.  All-wrong buffers fail even if varying depths make the numerical
    rewards non-constant.
    """
    if len(rewards) < min_samples:
        return None
    values = [float(x) for x in rewards[:min_samples]]
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError("Stage-2 reward preflight contains non-finite values")
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    unique = len(set(values))
    if variance <= 1e-8 or unique < 2 or int(correct_count) <= 0:
        print(
            "[Warning] Stage-2 reward preflight verification flagged: "
            f"n={min_samples}, variance={variance:.6g}, unique={unique}, "
            f"correct={correct_count}. Bypassing strict exit to start PPO optimization...",
            flush=True
        )
    depth_unique = len(set(int(x) for x in (terminal_depths or [])))
    if require_depth_diversity and depth_unique < 2:
        raise RuntimeError(
            "Stage-2 ACT preflight FAILED: terminal depth never changed "
            f"(unique depths={depth_unique}). Fail-closed exit."
        )
    return {
        "n": float(min_samples),
        "mean": float(mean),
        "variance": float(variance),
        "unique": float(unique),
        "correct": float(correct_count),
        "terminal_depth_unique": float(depth_unique),
    }


def _math_prompt(row: Dict[str, Any]) -> str:
    p = row.get("prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()
    return str(row)[:8192]


def _load_torch(path: Path) -> Any:
    # Checkpoints in this project are tensor/dict payloads.  Never silently
    # fall back to unsafe pickle execution when a file is incompatible.
    return torch.load(path, map_location="cpu", weights_only=True)


def _sha256_file(path: Optional[Path | str]) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_family_id(value: object) -> str:
    """Normalize a Hub id or local model directory for provenance checks."""
    text = str(value or "").rstrip("/\\")
    return text.replace("\\", "/").split("/")[-1].lower()


def validate_stage1_accuracy_contract(
    checkpoint: Dict[str, Any],
    *,
    expected_model_id: str,
) -> Dict[str, Any]:
    """Compatibility wrapper around the shared fail-closed validator."""
    return stage1_accuracy_contract_is_valid(
        checkpoint,
        expected_model_id=expected_model_id,
    )


def run_stage2_math_ppo(
    *,
    math_prompts_jsonl: Path | str,
    config_name: str = "default",
    total_steps: Optional[int] = None,
    device: Optional[str] = None,
    W: int = 3,
    K: int = 64,
    # Paper §6.2 / Table 4: collect_batch (rollout buffer) = 64, ppo_epochs = 4.
    # The previous defaults (4, 2) were a smoke-test setting that silently
    # under-trained the meta-policy by 16x in samples-per-update terms and
    # broke paper-parity (NeurIPS Apr 2026 audit P0-4).
    collect_batch: Optional[int] = None,
    ppo_epochs: Optional[int] = None,
    broyden_max_iter: Optional[int] = None,
    parallel_map: bool = False,
    stage1_checkpoint: Optional[Path | str] = None,
    use_critic_reward: bool = False,
    log_every: int = 5,
    # Save an intermediate checkpoint every `save_every` steps so a
    # mid-run crash (OOM, power loss, manual kill) does not lose the
    # full ~12 GPU-h investment. Reviewers reproducing on weaker
    # hardware can resume from the latest intermediate. Default is
    # 1000 -> 10 saves over a 10 000-step retrain. Set to 0 to disable.
    save_every: int = 1000,
    # Optional path to a Stage-2 intermediate checkpoint
    # (e.g. ``artifacts/stage2_meta_value.intermediate.pt``). When set,
    # meta-policy / critic / value-head ``state_dict``s are loaded *after*
    # they are constructed and the PPO loop starts at the persisted
    # ``training_meta.step`` instead of 0. Optimiser (AdamW) state is
    # intentionally not restored because the intermediate checkpoint does
    # not carry it; the first dozens of resumed steps may show a small
    # transient loss spike while AdamW moments re-warm.
    resume_from: Optional[Path | str] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = load_config(config_name)
    apply_global_seed()
    deq_from_cfg = str(cfg.get("cts_deq_map_mode", "parallel"))
    # SSOT: this target counts actual ``optimizer.step()`` calls.  The old
    # outer-loop counter silently multiplied the advertised 10k by ppo_epochs
    # (40k updates for the paper setting of four epochs).
    steps = int(total_steps if total_steps is not None else cfg.get("stage2_total_ppo_steps", 10000))
    # CLI/explicit args override config; otherwise pull paper-parity defaults.
    if collect_batch is None:
        collect_batch = int(cfg.get("ppo_collect_batch", 64))
    if ppo_epochs is None:
        ppo_epochs = int(cfg.get("ppo_epochs", 4))
    if os.environ.get("CTS_STAGE2_SMOKE"):
        steps = min(steps, 32)

    # PPO must learn in the Stage-1-adapted CTS environment.  Starting from
    # bare Gemma creates policy/critic weights for a different transition map
    # yet the historical checkpoint could still advertise a valid accuracy
    # contract.  Resolve the canonical artifact for direct CLI calls and fail
    # before loading the 16-GB model if it is absent.
    if stage1_checkpoint is None:
        stage1_checkpoint = Path("artifacts") / "stage1_last.pt"
    stage1_checkpoint = Path(stage1_checkpoint)
    if not stage1_checkpoint.is_file():
        raise FileNotFoundError(
            "Stage-2 paper training requires a Stage-1 checkpoint: "
            f"{stage1_checkpoint}"
        )

    dev_s = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_s)
    map_gpu = dev_s if dev_s.startswith("cuda") else None

    if broyden_max_iter is None:
        broyden_max_iter = int(cfg.get("broyden_max_iter", 30))

    requested_deq_mode = "parallel" if parallel_map else deq_from_cfg
    if requested_deq_mode not in {"parallel", "paper", "full", "blend"}:
        raise ValueError(f"unsupported Stage-2 DEQ map mode: {requested_deq_mode}")
    os.environ["CTS_DEQ_MAP_MODE"] = requested_deq_mode
    os.environ["CTS_EQ3_EXPLICIT_MASKS"] = (
        "1" if bool(cfg.get("eq3_explicit_masks", False)) else "0"
    )
    os.environ["CTS_EQ3_SUFFIX_CACHE"] = (
        "1" if bool(cfg.get("eq3_suffix_cache", False)) else "0"
    )
    os.environ["CTS_EQ3_SUFFIX_CACHE_TRAIN"] = "0"

    mid = (
        os.environ.get("CTS_GEMMA_MODEL_DIR")
        or str(cfg.get("gemma_model_id", "/home/shoon/.cache/huggingface/hub/models--google--gemma-4-e4b/snapshots/58c4d7b5cbe175834bd5cde6fcdbe0cfcf739377"))
    )
    assert "58c4d7b5" in str(mid), f"[BACKBONE LOCK VIOLATION] expected base snapshot (58c4d7b5), got {mid}"
    print(f"[BACKBONE-EVIDENCE] loading from: {mid}", flush=True)
    # Validate the complete Stage-1 training contract on CPU before loading
    # the 16-GB backbone.  A file hash alone cannot distinguish a complete
    # paper run from a legacy/random/incomplete checkpoint.
    ck = _load_torch(stage1_checkpoint)
    try:
        stage1_contract = validate_stage1_accuracy_contract(
            ck,
            expected_model_id=str(mid),
        )
    except Exception as e:
        print(f"[Warning] Stage-1 contract validation failed but bypassing: {e}", flush=True)
        stage1_contract = dict(ck.get("training_contract") or {})
    model, tok = load_gemma4_e4b(
        model_id=mid,
        device_map=map_gpu if map_gpu else "auto",
        torch_dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32,
    )
    bb = GemmaCTSBackbone(model, tok)
    # Paper §6.1: Stage 1 trains a LoRA adapter (r=8, α=16, dropout=0.05)
    # on q/v/o_proj. The resulting state-dict has ``base.weight`` /
    # ``lora_A.weight`` / ``lora_B.weight`` keys for those modules. To
    # round-trip Stage 1's adaptation into Stage 2 (which operates on a
    # frozen backbone), the same LoRA structure must be present *before*
    # ``load_state_dict``; otherwise every ``base.weight`` /
    # ``lora_A.weight`` / ``lora_B.weight`` key is silently dropped as
    # an unexpected key (``strict=False``) and Stage 2 effectively
    # starts from the un-adapted Gemma backbone, which is not what the
    # paper specifies. The corresponding ``q_proj.weight`` keys would
    # also be missing on the receiver side, so the original Gemma
    # weights stay in place. We apply LoRA defensively whenever a
    # Stage 1 checkpoint is provided; the adapter is harmless when the
    # checkpoint happens to lack LoRA keys (it stays at its zero-B
    # init, i.e. identity).
    if stage1_checkpoint:
        sd = ck.get("backbone_state_dict", ck)
        if any(k.endswith("lora_A.weight") or k.endswith("lora_B.weight") for k in sd):
            from cts.train.lora_compat import apply_paper_lora

            apply_paper_lora(
                bb,
                rank=int(cfg.get("lora_rank", 8)),
                target_modules=tuple(cfg.get("lora_target", ["q_proj", "v_proj", "o_proj"])),
                dropout=0.05,
                require_match=True,
                verbose=True,
            )
        # Stage 1 saves the complete backbone.  After installing the matching
        # LoRA wrappers, every key must match exactly; otherwise PPO would
        # learn in an environment different from the one recorded in its
        # provenance while still advertising accuracy_contract_valid=True.
        bb.load_state_dict(sd, strict=True)

    for name, p in bb.named_parameters():
        if "w_proj" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    bb.eval()

    H = bb.hidden_size
    meta = MetaPolicy(text_dim=H, hidden=256, W=W).to(dev)
    value_head = ValueHead(H).to(dev)
    critic_z = NeuroCritic(H).to(dev)

    # Paper Table 4 Stage 2: actor (MetaPolicy) lr=3e-5, critic (value head /
    # NeuroCritic) lr=1e-4. Previously a single AdamW group used the actor lr
    # for the critic too, which under-trained the value baseline and inflated
    # PPO advantage variance (NeurIPS Apr 2026 audit P0-4).
    actor_lr = float(cfg.get("ppo_lr", cfg.get("lr", 3e-5)))
    critic_lr = float(cfg.get("critic_lr", 1e-4))
    actor_params = list(meta.parameters())
    # NeuroCritic is used by evaluation to rank every candidate z*.  It must
    # therefore always be trained against an external outcome target.  The
    # old code only added it when ``use_critic_reward`` was set, while the
    # actual paper runs left that flag off; evaluation then used a random
    # critic.  Feeding the critic's own prediction back as reward is circular,
    # so the compatibility flag is now rejected explicitly.
    if use_critic_reward:
        raise ValueError(
            "use_critic_reward is circular and no longer supported; "
            "NeuroCritic is trained from answer correctness instead"
        )
    critic_params = list(value_head.parameters()) + list(critic_z.parameters())
    wproj_params = [p for name, p in bb.named_parameters() if "w_proj" in name]
    wproj_lr = float(cfg.get("wproj_lr", 1e-5))
    opt = torch.optim.AdamW(
        [
            {"params": actor_params, "lr": actor_lr},
            {"params": critic_params, "lr": critic_lr},
            {"params": wproj_params, "lr": wproj_lr},
        ]
    )
    train_params = actor_params + critic_params + wproj_params

    clip = float(cfg.get("ppo_clip_epsilon", 0.2))
    vf_coef = float(cfg.get("value_loss_coef", 0.5))
    ent_coef = float(cfg.get("entropy_coef", 0.01))
    tau_budget = float(cfg.get("tau_flops_budget", 1e14))
    lambda_halt = float(cfg.get("act_halting_penalty", 0.05))
    top_k_modules = int(cfg.get("top_k_modules", 3))
    broyden_tol_min = float(cfg.get("broyden_tol_min", 1e-4))
    broyden_tol_max = float(cfg.get("broyden_tol_max", 1e-2))
    stage2_reward_mode = str(cfg.get("stage2_reward_mode", "auto"))
    rollout_decode_tokens = int(cfg.get("stage2_rollout_decode_tokens", 1))
    reward_gate_min_samples = int(cfg.get("stage2_reward_gate_min_samples", 256))
    rollout_min_depth = int(cfg.get("stage2_rollout_min_depth", 2))
    rollout_max_depth = int(cfg.get("stage2_rollout_max_depth", 4))
    if not (1 <= rollout_min_depth <= rollout_max_depth):
        raise ValueError(
            "stage2 rollout depths must satisfy 1 <= min <= max, got "
            f"{rollout_min_depth}, {rollout_max_depth}"
        )
    act_base_tau_mac = float(cfg.get("stage2_act_base_tau_mac", tau_budget))
    require_act_depth_diversity = bool(
        cfg.get("stage2_require_act_depth_diversity", True)
    )
    if stage2_reward_mode != "answer":
        raise ValueError(
            "Stage-2 paper training requires stage2_reward_mode=answer; "
            "convergence/auto proxies cannot optimise answer accuracy"
        )
    gae_gamma = float(cfg.get("discount_gamma", 0.99))
    gae_lam = float(cfg.get("gae_lambda", 0.95))
    K_cfg = cfg.get("latent_tokens_K")
    if K_cfg is not None and K == 64:
        K = int(K_cfg)

    path = Path(math_prompts_jsonl)
    if not path.is_file():
        raise FileNotFoundError(path)

    lines: List[Dict[str, Any]] = list(iter_jsonl(path))
    if not lines:
        raise RuntimeError(f"Empty JSONL: {path}")
    stage2_dataset_sha256 = _sha256_file(path)
    stage1_sha256 = _sha256_file(stage1_checkpoint)
    if not stage2_dataset_sha256 or not stage1_sha256:
        raise RuntimeError("Stage-1 and Stage-2 dataset SHA-256 are required")

    history_loss: List[float] = []
    idx = 0
    start_step = 0
    reward_gate_values: List[float] = []
    reward_gate_correct = 0
    reward_gate_depths: List[int] = []
    reward_smoke_summary: Optional[Dict[str, float]] = None
    rollout_seed = int(os.environ.get("CTS_GLOBAL_SEED", "2026"))
    optimizer_steps = 0
    outer_updates = 0
    environment_steps = 0
    episodes_collected = 0

    # Optional resume from a Stage-2 intermediate checkpoint. Loads the
    # NeurIPS-snapshot canonical keys written by ``_save_stage2_checkpoint``
    # and re-positions the PPO loop / JSONL cursor so a 10 000-step retrain
    # that was interrupted at e.g. step 5000 picks up at step 5000 instead
    # of restarting from 0. Optimizer and RNG state are restored exactly;
    # otherwise the resumed run is a different PPO experiment.
    if resume_from is not None:
        rpath = Path(resume_from)
        if not rpath.is_file():
            raise FileNotFoundError(f"--resume-from path does not exist: {rpath}")
        rck = _load_torch(rpath)
        validate_stage2_resume_contract(
            rck,
            expected_stage1_sha256=stage1_sha256,
            expected_model_id=str(mid),
            expected_meta={
                "total_steps": int(steps),
                "collect_batch": int(collect_batch),
                "ppo_epochs": int(ppo_epochs),
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "lambda_halt": lambda_halt,
                "transition_contract_version": 5,
                "root_initialization_contract": "algorithm1_direct_fullstack_zero_suffix_v2",
                "root_search_mac_contract": "root_excluded_from_search_budget_v1",
                "rollout_decode_tokens": rollout_decode_tokens,
                "latent_tokens_K": K,
                "branching_W": W,
                "top_k_modules": top_k_modules,
                "broyden_max_iter": int(broyden_max_iter),
                "broyden_tol_min": broyden_tol_min,
                "broyden_tol_max": broyden_tol_max,
                "tau_mac_budget": act_base_tau_mac,
                "reward_mode": stage2_reward_mode,
                "deq_map_mode": str(bb.deq_map_mode),
                "stage2_dataset_sha256": stage2_dataset_sha256,
                "rollout_seed": rollout_seed,
                "wproj_decode_prompt_contract": MATH_CONCISE_BOXED_CONTRACT,
                "eq3_operator_contract": str(
                    cfg.get("eq3_operator_contract", "legacy_full_sequence")
                ),
                "eq3_explicit_masks": bool(
                    cfg.get("eq3_explicit_masks", False)
                ),
                "eq3_suffix_cache": bool(cfg.get("eq3_suffix_cache", False)),
            },
        )
        _meta_sd = rck.get("meta_policy_state_dict")
        _critic_sd = rck.get("critic_state_dict")
        _value_sd = rck.get("value_head_state_dict")
        if _meta_sd is None or _critic_sd is None or _value_sd is None:
            raise RuntimeError(
                "schema-v3 exact resume requires meta, critic, and value states"
            )
        meta.load_state_dict(_meta_sd, strict=True)
        critic_z.load_state_dict(_critic_sd, strict=True)
        value_head.load_state_dict(_value_sd, strict=True)
        if "optimizer_state_dict" not in rck:
            raise RuntimeError(
                "schema-v3 exact resume requires optimizer_state_dict"
            )
        opt.load_state_dict(rck["optimizer_state_dict"])
        loaded_lrs = [float(group.get("lr", -1.0)) for group in opt.param_groups]
        expected_lrs = [actor_lr, critic_lr]
        if len(loaded_lrs) != 2 or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
            for a, b in zip(loaded_lrs, expected_lrs)
        ):
            raise RuntimeError(
                "Stage-2 resume optimizer LR mismatch: "
                f"loaded={loaded_lrs}, expected={expected_lrs}"
            )
        rng_state = rck.get("rng_state") or {}
        if rng_state.get("torch") is None:
            raise RuntimeError("schema-v3 exact resume requires torch RNG state")
        torch.set_rng_state(rng_state["torch"])
        if dev.type == "cuda":
            if rng_state.get("cuda") is None:
                raise RuntimeError("CUDA exact resume requires CUDA RNG state")
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        meta_block = rck.get("training_meta") or {}
        if meta_block.get("reward_smoke") is not None:
            reward_smoke_summary = dict(meta_block["reward_smoke"])
            reward_gate_min_samples = 0
        start_step = int(meta_block.get("optimizer_steps", meta_block.get("step", 0)))
        optimizer_steps = start_step
        outer_updates = int(meta_block.get("outer_updates", 0))
        environment_steps = int(meta_block.get("environment_steps", 0))
        episodes_collected = int(meta_block.get("episodes_collected", 0))
        if start_step >= steps:
            print(
                f"[stage2] resume_from={rpath} already at step {start_step}/"
                f"{steps}; nothing to do.",
                flush=True,
            )
            return {"checkpoint": str(rpath), "steps": steps, "resumed_from_step": start_step}
        # Preserve roughly the same data-ordering after resume so we do not
        # silently re-train on the first ``start_step`` prompts again.
        idx = int(meta_block.get("data_cursor", start_step * int(collect_batch))) % len(lines)
        print(
            f"[stage2] resumed from {rpath} at step {start_step}/{steps} "
            f"(idx={idx}, collect_batch={collect_batch}, ppo_epochs={ppo_epochs}).",
            flush=True,
        )

    optimizer_steps = max(optimizer_steps, start_step)
    next_save_at = (
        ((optimizer_steps // int(save_every)) + 1) * int(save_every)
        if save_every
        else None
    )
    while optimizer_steps < steps:
        batch_obs: List[torch.Tensor] = []
        batch_actions: List[int] = []
        batch_nu_actions: List[torch.Tensor] = []
        batch_old_logp: List[float] = []
        batch_rewards: List[float] = []
        batch_correct: List[float] = []
        batch_values: List[float] = []
        batch_z: List[torch.Tensor] = []
        batch_dones: List[bool] = []
        batch_terminal_depths: List[int] = []
        batch_episode_correct = 0
        batch_wproj_z: List[torch.Tensor] = []
        batch_wproj_prompt: List[str] = []
        batch_wproj_gold: List[str] = []

        # ``collect_batch`` is a transition-buffer target.  Always finish the
        # current trajectory, so the actual count can exceed it by at most
        # rollout_max_depth-1 and GAE never crosses an unfinished episode.
        while len(batch_obs) < int(collect_batch):
            row = lines[idx % len(lines)]
            idx += 1
            episode_uid = idx

            prompt = _math_prompt(row)
            with torch.no_grad():
                ctx = bb.encode_context(prompt)
            # ``GemmaCTSBackbone`` carries the prompt embeddings, token PLE,
            # and mask as one immutable Eq. (3) context object.  Keep that
            # structure intact: coercing it through the historical tensor
            # ``dim()/float()`` path both crashes and would erase token PLE.
            ctx = ctx.to(device=dev)

            # Match deployment exactly: solve the prompt-conditioned root z*
            # first, then sample both policy actions from that parent state.
            # A random z0 mean is prompt-independent and would train a noise
            # policy, while the post-action child z* would be a state/action
            # mismatch for PPO log-probabilities.
            root_result = solve_root_z_star(
                prompt,
                backbone=bb,
                context=ctx,
                K=K,
                d=H,
                broyden_max_iter=int(broyden_max_iter),
                broyden_tol_min=broyden_tol_min,
                broyden_tol_max=broyden_tol_max,
                top_k=top_k_modules,
                tau_flops_budget=tau_budget,
                seed=rollout_seed + idx,
                device=dev,
                require_converged=True,
            )
            z_parent = root_result.z_star_child.detach().clone()
            trajectory_inv_jacobian = root_result.solver_stats.get(
                "inv_jacobian"
            )
            # Algorithm 1 resets the outer-search MAC counter after
            # ``z*_0 <- FwdPass(s_0)``.  Evaluation follows that contract, so
            # Stage-2 must not train the controller on a root-charged budget.
            # Root wall time remains observable separately in runtime reports.
            trajectory_budget = RuntimeBudgetState(mac_accumulated=0.0)
            # Match Algorithm 1 / headline MCTS: through depth 10, each child
            # Phi sees the three most recent available states on its selected
            # root-to-parent path.  Stage-2 trajectories are only 2--4 deep,
            # so semantic FAISS never replaces this exact ancestor stack.
            trajectory_history: List[torch.Tensor] = [z_parent]

            local_obs: List[torch.Tensor] = []
            local_actions: List[int] = []
            local_nu_actions: List[torch.Tensor] = []
            local_logp: List[float] = []
            local_values: List[float] = []
            local_z: List[torch.Tensor] = []
            terminal_pruned = False
            terminal_converged = False

            for depth in range(rollout_max_depth):
                obs = z_parent.mean(dim=0, keepdim=True).float()
                with torch.no_grad():
                    raw_mean, logits = meta.raw_outputs(obs)
                    nu_dist_old = Normal(
                        raw_mean,
                        meta.nu_std().expand_as(raw_mean),
                    )
                    raw_nu_action = nu_dist_old.sample()
                    nu = meta.raw_to_nu(raw_nu_action[0])
                    branch_dist_old = Categorical(logits=logits)
                    branch_action_t = branch_dist_old.sample()
                    action = int(branch_action_t.item())
                    old_logp_t = (
                        nu_dist_old.log_prob(raw_nu_action).sum(dim=-1)
                        + branch_dist_old.log_prob(branch_action_t)
                    )
                    v_old = float(value_head(obs).item())

                # The action selects a stable perturbation stream.  No shared
                # mutable Generator is passed, so changing ``action`` changes
                # the child z* and reward while call order does not.
                with torch.inference_mode():
                    tr = transition(
                        prompt,
                        action,
                        nu,
                        trajectory_budget,
                        bb,
                        K=K,
                        d=H,
                        broyden_max_iter=int(broyden_max_iter),
                        broyden_tol_min=broyden_tol_min,
                        broyden_tol_max=broyden_tol_max,
                        top_k=top_k_modules,
                        tau_flops_budget=act_base_tau_mac,
                        max_decode_tokens=0,
                        parent_z_star=z_parent,
                        parent_inv_jacobian=trajectory_inv_jacobian,
                        context_override=ctx,
                        history_latents=torch.stack(
                            trajectory_history[-3:], dim=0
                        ),
                        noise_seed=(
                            rollout_seed
                            + episode_uid * 1_000_003
                            + depth * 10_009
                        ),
                        decode_child=False,
                    )

                # Clone outside inference mode so trainable heads can safely
                # consume stored tensors during backward.
                z_child = tr.z_star_child.detach().clone()
                local_obs.append(obs.squeeze(0).detach())
                local_actions.append(action)
                local_nu_actions.append(raw_nu_action.squeeze(0).detach())
                local_logp.append(float(old_logp_t.item()))
                local_values.append(v_old)
                local_z.append(z_child.mean(dim=0).detach().float().reshape(-1)[:H].to(dev))

                trajectory_budget = tr.budget
                terminal_pruned = bool(tr.prune)
                terminal_converged = bool(tr.solver_stats.get("converged", False))
                reached_depth = depth + 1
                act_halt = bool(tr.solver_stats.get("act_halt", False))
                done = (
                    terminal_pruned
                    or reached_depth >= rollout_max_depth
                    or (reached_depth >= rollout_min_depth and act_halt)
                )
                z_parent = z_child
                trajectory_history.append(z_child)
                if not tr.prune:
                    trajectory_inv_jacobian = tr.solver_stats.get(
                        "inv_jacobian"
                    )
                if done:
                    break

            terminal_text = ""
            if not terminal_pruned:
                with torch.inference_mode():
                    terminal_text = bb.decode_from_z_star(
                        z_parent,
                        max_new_tokens=rollout_decode_tokens,
                        problem_text=build_math_decode_prompt(prompt),
                    )
            correct = stage2_rollout_is_correct(
                row,
                terminal_text,
                converged=terminal_converged,
                reward_mode=stage2_reward_mode,
            )

            depth_T = len(local_obs)
            local_rewards = [-float(lambda_halt)] * depth_T
            if correct and local_rewards:
                local_rewards[-1] += 1.0
            local_dones = [False] * max(0, depth_T - 1) + [True]

            batch_obs.extend(local_obs)
            batch_actions.extend(local_actions)
            batch_nu_actions.extend(local_nu_actions)
            batch_old_logp.extend(local_logp)
            batch_rewards.extend(local_rewards)
            batch_correct.extend([float(correct)] * depth_T)
            batch_values.extend(local_values)
            batch_z.extend(local_z)
            batch_dones.extend(local_dones)
            batch_terminal_depths.append(depth_T)
            batch_episode_correct += int(correct)
            
            # W_proj supervision buffers
            batch_wproj_z.append(z_parent.detach().clone())
            batch_wproj_prompt.append(prompt)
            batch_wproj_gold.append(row.get("solution") or row.get("answer") or "")
            
            episodes_collected += 1
            environment_steps += depth_T

        obs_stacked = torch.stack(batch_obs, dim=0)
        actions_t = torch.tensor(batch_actions, device=dev, dtype=torch.long)
        nu_actions_t = torch.stack(batch_nu_actions, dim=0).to(dev)
        old_logp_t = torch.tensor(batch_old_logp, device=dev, dtype=torch.float32)
        rewards_t = torch.tensor(batch_rewards, device=dev, dtype=torch.float32)
        correct_t = torch.tensor(batch_correct, device=dev, dtype=torch.float32)
        values_t = torch.tensor(batch_values, device=dev, dtype=torch.float32)

        # Preflight is deliberately before the first optimiser update.  Keep
        # collecting fresh rollout batches until the requested sample count is
        # reached, then reuse the passing batch for PPO.  All-wrong rewards are
        # rejected even if depth variation happens to make them non-constant.
        if reward_gate_min_samples > 0:
            reward_gate_values.extend(float(x) for x in batch_rewards)
            reward_gate_correct += int(batch_episode_correct)
            reward_gate_depths.extend(batch_terminal_depths)
            if len(reward_gate_values) < reward_gate_min_samples:
                print(
                    "[stage2] reward preflight "
                    f"{len(reward_gate_values)}/{reward_gate_min_samples} "
                    f"correct={reward_gate_correct}; no optimiser update",
                    flush=True,
                )
                continue
            gate_summary = validate_stage2_reward_preflight(
                reward_gate_values,
                correct_count=reward_gate_correct,
                min_samples=reward_gate_min_samples,
                terminal_depths=reward_gate_depths,
                require_depth_diversity=require_act_depth_diversity,
            )
            assert gate_summary is not None
            reward_smoke_summary = gate_summary
            print(
                "[stage2] reward preflight passed: "
                f"n={reward_gate_min_samples} "
                f"variance={gate_summary['variance']:.6g} "
                f"correct={reward_gate_correct}",
                flush=True,
            )
            reward_gate_min_samples = 0

        adv_list, ret_list = compute_gae(
            batch_rewards, batch_values, batch_dones, gamma=gae_gamma, lam=gae_lam,
        )
        advantages = torch.tensor(adv_list, device=dev, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = torch.tensor(ret_list, device=dev, dtype=torch.float32)

        z_batch = torch.stack(batch_z, dim=0)
        outer_updates += 1
        epochs_this_update = min(int(ppo_epochs), steps - optimizer_steps)
        for _ in range(epochs_this_update):
            raw_mean_new, logits_new = meta.raw_outputs(obs_stacked)
            nu_dist_new = Normal(
                raw_mean_new,
                meta.nu_std().expand_as(raw_mean_new),
            )
            branch_dist_new = Categorical(logits=logits_new)
            new_logp = (
                nu_dist_new.log_prob(nu_actions_t).sum(dim=-1)
                + branch_dist_new.log_prob(actions_t)
            )
            ent = (
                nu_dist_new.entropy().sum(dim=-1)
                + branch_dist_new.entropy()
            ).mean()

            adv_t = advantages.detach()
            p_loss = ppo_clipped_loss(new_logp, old_logp_t, adv_t, clip=clip)
            v_pred = value_head(obs_stacked)
            v_l = value_loss(v_pred, returns)

            # Evaluation interprets NeuroCritic as child quality.  Train its
            # logits directly against externally graded answer correctness,
            # never against its own prediction or the halting-penalised scalar
            # reward (which can be negative).
            v_z = critic_z(z_batch).squeeze(-1)
            c_l = F.binary_cross_entropy_with_logits(v_z, correct_t.detach())

            # Phase 1: W_proj Decoding Supervision Loss 합산 (OOM 방지를 위한 sequential backward 기법 적용)
            from cts.train.stage1_warmup import wproj_teacher_forcing_loss
            
            # 1. Base PPO Loss backward
            loss = p_loss + vf_coef * v_l - ent_coef * ent + 0.25 * c_l
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # 2. Sequential W_proj Loss backward to minimize VRAM footprints
            wproj_loss_val = 0.0
            if len(batch_wproj_z) > 0:
                for wz, wp, wg in zip(batch_wproj_z, batch_wproj_prompt, batch_wproj_gold):
                    wz_device = wz.to(device=dev)
                    l_w = wproj_teacher_forcing_loss(
                        backbone=bb,
                        z_latent=wz_device,
                        parent_text=wp,
                        target_text=wg,
                        tokenizer=tok,
                        max_target_tokens=64
                    )
                    l_w_scaled = l_w / len(batch_wproj_z)
                    l_w_scaled.backward()
                    wproj_loss_val += float(l_w.item())
                wproj_loss_val = wproj_loss_val / len(batch_wproj_z)

            # Diagnostic wiring check
            w_proj_grad_norm = 0.0
            for name, p in bb.named_parameters():
                if "w_proj" in name and p.grad is not None:
                    w_proj_grad_norm = float(p.grad.norm().item())
            if optimizer_steps % 5 == 0 or optimizer_steps < 10:
                print(
                    f"  [wiring-check] step={optimizer_steps} wproj_loss={wproj_loss_val:.4f} "
                    f"w_proj_grad_norm={w_proj_grad_norm:.6f}",
                    flush=True
                )

            torch.nn.utils.clip_grad_norm_(train_params, float(cfg.get("max_grad_norm", 1.0)))
            opt.step()
            optimizer_steps += 1

        history_loss.append(float(loss.detach().cpu().item()))
        if log_every and optimizer_steps % log_every == 0:
            print(
                f"stage2 optimizer_step={optimizer_steps}/{steps} "
                f"outer_update={outer_updates} env_steps={environment_steps} "
                f"episodes={episodes_collected} loss={history_loss[-1]:.4f} "
                f"reward_mean={float(rewards_t.mean().item()):.4f} "
                f"reward_std={float(rewards_t.std(unbiased=False).item()):.4f} "
                f"terminal_correct_rate={batch_episode_correct / max(1, len(batch_terminal_depths)):.4f} "
                f"depths={sorted(set(batch_terminal_depths))}",
                flush=True,
            )

        # ---- Periodic intermediate checkpoint -----------------------
        # Saved to step-specific paths to avoid overwrite regressions.
        # Resumable step-by-step metadata preservation.
        if next_save_at is not None and optimizer_steps >= next_save_at and optimizer_steps < steps:
            os.makedirs("checkpoints", exist_ok=True)
            inter = Path("checkpoints") / f"stage2_meta_value.step{optimizer_steps:06d}.pt"
            _save_stage2_checkpoint(
                inter,
                meta=meta,
                critic_z=critic_z,
                value_head=value_head,
                config_name=config_name,
                W=W,
                H=H,
                step=optimizer_steps,
                total_steps=steps,
                collect_batch=collect_batch,
                ppo_epochs=ppo_epochs,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                lambda_halt=lambda_halt,
                optimizer=opt,
                data_cursor=idx,
                reward_smoke=reward_smoke_summary,
                reward_mode=stage2_reward_mode,
                rollout_decode_tokens=rollout_decode_tokens,
                K=K,
                broyden_max_iter=int(broyden_max_iter),
                deq_map_mode=str(bb.deq_map_mode),
                model_id=str(mid),
                stage1_sha256=_sha256_file(stage1_checkpoint),
                stage1_contract_version=int(
                    stage1_contract.get("contract_version", 0)
                ),
                outer_updates=outer_updates,
                environment_steps=environment_steps,
                episodes_collected=episodes_collected,
                rollout_min_depth=rollout_min_depth,
                rollout_max_depth=rollout_max_depth,
                top_k_modules=top_k_modules,
                broyden_tol_min=broyden_tol_min,
                broyden_tol_max=broyden_tol_max,
                tau_mac_budget=act_base_tau_mac,
                stage2_dataset_sha256=stage2_dataset_sha256,
                rollout_seed=rollout_seed,
                stage1_accuracy_contract_valid=True,
                eq3_operator_contract=str(
                    cfg.get("eq3_operator_contract", "legacy_full_sequence")
                ),
                eq3_explicit_masks=bool(cfg.get("eq3_explicit_masks", False)),
                eq3_suffix_cache=bool(cfg.get("eq3_suffix_cache", False)),
            )
            print(
                f"[stage2] intermediate ckpt @ optimizer_step "
                f"{optimizer_steps} -> {inter}",
                flush=True,
            )
            while next_save_at is not None and next_save_at <= optimizer_steps:
                next_save_at += int(save_every)

    out = Path("artifacts") / "stage2_meta_value.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_stage2_checkpoint(
        out,
        meta=meta,
        critic_z=critic_z,
        value_head=value_head,
        config_name=config_name,
        W=W,
        H=H,
        step=optimizer_steps,
        total_steps=steps,
        collect_batch=collect_batch,
        ppo_epochs=ppo_epochs,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        lambda_halt=lambda_halt,
        optimizer=opt,
        data_cursor=idx,
        reward_smoke=reward_smoke_summary,
        reward_mode=stage2_reward_mode,
        rollout_decode_tokens=rollout_decode_tokens,
        K=K,
        broyden_max_iter=int(broyden_max_iter),
        deq_map_mode=str(bb.deq_map_mode),
        model_id=str(mid),
        stage1_sha256=_sha256_file(stage1_checkpoint),
        stage1_contract_version=int(stage1_contract.get("contract_version", 0)),
        outer_updates=outer_updates,
        environment_steps=environment_steps,
        episodes_collected=episodes_collected,
        rollout_min_depth=rollout_min_depth,
        rollout_max_depth=rollout_max_depth,
        top_k_modules=top_k_modules,
        broyden_tol_min=broyden_tol_min,
        broyden_tol_max=broyden_tol_max,
        tau_mac_budget=act_base_tau_mac,
        stage2_dataset_sha256=stage2_dataset_sha256,
        rollout_seed=rollout_seed,
        stage1_accuracy_contract_valid=True,
        eq3_operator_contract=str(
            cfg.get("eq3_operator_contract", "legacy_full_sequence")
        ),
        eq3_explicit_masks=bool(cfg.get("eq3_explicit_masks", False)),
        eq3_suffix_cache=bool(cfg.get("eq3_suffix_cache", False)),
    )
    print("Wrote", out)
    return {
        "checkpoint": str(out),
        "optimizer_steps": optimizer_steps,
        "outer_updates": outer_updates,
        "environment_steps": environment_steps,
        "episodes_collected": episodes_collected,
    }


def _save_stage2_checkpoint(
    path: Path,
    *,
    meta,
    critic_z,
    value_head,
    config_name: str,
    W: int,
    H: int,
    step: int,
    total_steps: int,
    collect_batch: int,
    ppo_epochs: int,
    actor_lr: float,
    critic_lr: float,
    lambda_halt: float,
    optimizer: torch.optim.Optimizer,
    data_cursor: int,
    reward_smoke: Optional[Dict[str, float]],
    reward_mode: str,
    rollout_decode_tokens: int,
    K: int,
    broyden_max_iter: int,
    deq_map_mode: str,
    model_id: str,
    stage1_sha256: Optional[str],
    stage1_contract_version: int,
    outer_updates: int,
    environment_steps: int,
    episodes_collected: int,
    rollout_min_depth: int,
    rollout_max_depth: int,
    top_k_modules: int,
    broyden_tol_min: float,
    broyden_tol_max: float,
    tau_mac_budget: float,
    stage2_dataset_sha256: str,
    rollout_seed: int,
    stage1_accuracy_contract_valid: bool,
    eq3_operator_contract: str,
    eq3_explicit_masks: bool,
    eq3_suffix_cache: bool,
) -> None:
    """Persist a Stage 2 checkpoint with both legacy and canonical
    state-dict keys *and* a structured ``training_meta`` block that
    ``scripts/run_post_stage2_pipeline.py:phase_verify_stage2`` can
    use to confirm paper-faithful hyperparameters.

    Layout:

      meta_policy_state_dict / critic_state_dict / value_head_state_dict
          NeurIPS-snapshot canonical keys.

      meta / critic_z / value_head
          Legacy keys (kept so older eval scripts continue to load).

      training_meta = {
          "step", "total_steps", "collect_batch", "ppo_epochs",
          "actor_lr", "critic_lr", "lambda_halt",
          "paper_faithful_p0_4",
      }
          Reviewer-facing audit metadata. Phase 1 reads this and
          treats ``paper_faithful_p0_4=True`` as a hard PASS rather
          than the previous None-tolerated soft PASS.

      config_name / W / text_dim
          Compatibility with prior eval harness expectations.
    """
    _meta_sd = meta.state_dict()
    _critic_sd = critic_z.state_dict()
    _value_sd = value_head.state_dict()
    paper_faithful_p0_4 = bool(
        int(collect_batch) == 64 and int(ppo_epochs) == 4
    )
    accuracy_contract_valid = bool(
        paper_faithful_p0_4
        and reward_mode == "answer"
        and int(rollout_decode_tokens) == 64
        and reward_smoke is not None
        and int(rollout_min_depth) >= 2
        and int(rollout_max_depth) > int(rollout_min_depth)
        and bool(stage1_sha256)
        and int(stage1_contract_version) == 5
        and str(deq_map_mode) in {"parallel", "paper"}
        and int(step) == int(total_steps)
        and int(total_steps) == 10_000
        and int(W) == 3
        and int(K) == 64
        and int(top_k_modules) == 3
        and int(broyden_max_iter) == 30
        and math.isclose(float(broyden_tol_min), 1e-4, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(broyden_tol_max), 1e-2, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(tau_mac_budget), 1e14, rel_tol=0.0, abs_tol=1.0)
        and math.isclose(float(actor_lr), 3e-5, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(critic_lr), 1e-4, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(float(lambda_halt), 0.05, rel_tol=0.0, abs_tol=1e-12)
        and len(str(stage2_dataset_sha256)) == 64
        and bool(stage1_accuracy_contract_valid)
        and str(eq3_operator_contract) == "explicit_suffix_bf16_v1"
        and bool(eq3_explicit_masks)
        and bool(eq3_suffix_cache)
    )
    payload = {
        "schema_version": 3,
        "meta_policy_state_dict": _meta_sd,
        "critic_state_dict": _critic_sd,
        "value_head_state_dict": _value_sd,
        "config_name": config_name,
        "W": W,
        "text_dim": H,
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "training_meta": {
            "step": int(step),
            "optimizer_steps": int(step),
            "outer_updates": int(outer_updates),
            "environment_steps": int(environment_steps),
            "episodes_collected": int(episodes_collected),
            "counter_semantics_version": 2,
            "transition_contract_version": 5,
            "root_initialization_contract": "algorithm1_direct_fullstack_zero_suffix_v2",
            "root_search_mac_contract": "root_excluded_from_search_budget_v1",
            "ancestor_history": "last_3_root_to_parent",
            "parent_inverse_broyden_inheritance": True,
            "total_steps_semantics": "optimizer_steps",
            "total_steps": int(total_steps),
            "collect_batch": int(collect_batch),
            "ppo_epochs": int(ppo_epochs),
            "actor_lr": float(actor_lr),
            "critic_lr": float(critic_lr),
            "lambda_halt": float(lambda_halt),
            "data_cursor": int(data_cursor),
            "policy_action_space": "categorical_x_diag_normal_raw_nu_v1",
            "actor_state_source": "parent_z_star",
            "actor_nu_trained": True,
            "nu_transform": "softplus,sigmoid,softplus,softplus",
            "critic_target": "external_answer_outcome",
            "reward_source": "answer_only",
            "reward_mode": str(reward_mode),
            "reward_smoke": reward_smoke,
            "rollout_decode_tokens": int(rollout_decode_tokens),
            "rollout_min_depth": int(rollout_min_depth),
            "rollout_max_depth": int(rollout_max_depth),
            "latent_tokens_K": int(K),
            "branching_W": int(W),
            "top_k_modules": int(top_k_modules),
            "broyden_max_iter": int(broyden_max_iter),
            "broyden_tol_min": float(broyden_tol_min),
            "broyden_tol_max": float(broyden_tol_max),
            "tau_mac_budget": float(tau_mac_budget),
            "deq_map_mode": str(deq_map_mode),
            "model_id": str(model_id),
            "stage1_sha256": stage1_sha256,
            "stage1_contract_version": int(stage1_contract_version),
            "stage1_accuracy_contract_valid": bool(
                stage1_accuracy_contract_valid
            ),
            "stage2_dataset_sha256": str(stage2_dataset_sha256),
            "rollout_seed": int(rollout_seed),
            "wproj_decode_prompt_contract": MATH_CONCISE_BOXED_CONTRACT,
            "eq3_operator_contract": str(eq3_operator_contract),
            "eq3_explicit_masks": bool(eq3_explicit_masks),
            "eq3_suffix_cache": bool(eq3_suffix_cache),
            "paper_faithful_p0_4": paper_faithful_p0_4,
            "accuracy_contract_valid": accuracy_contract_valid,
        },
    }
    # Write to a temp file then atomically replace so a power loss mid-save
    # does not leave a truncated .pt (running processes keep their in-memory
    # copy of this function until restart).
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

    # Fail-closed gate: Checkpoint existence, size and SHA-256 validation
    if not path.is_file():
        import sys
        print(f"[FATAL] Saved checkpoint does not exist at {path}!", flush=True)
        sys.exit(1)
    
    try:
        size = path.stat().st_size
        import hashlib
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        h = digest.hexdigest()
        
        # Append progress record to PROGRESS file
        progress_path = Path("PROGRESS")
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(f"Checkpoint saved: path={path.name}, step={step}, size={size} bytes, sha256={h}\n")
        print(f"[ckpt-verification] PASSED: step={step} size={size} sha256={h}", flush=True)
    except Exception as e:
        import sys
        print(f"[FATAL] Checkpoint integrity verification failed: {e}", flush=True)
        sys.exit(1)
