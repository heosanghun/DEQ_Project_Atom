"""Algorithm 1: Cognitive Tree Search (CTS) — Single Episode.

Paper-aligned full MCTS episode loop with:
  - PUCT selection across the full tree (Eq. 2)
  - MetaPolicy nu sampling (§4.1)
  - W parallel DEQ transitions with parent z* noise (line 6)
  - Neuro-Critic Q evaluation (line 12)
  - FAISS registration (line 12)
  - Tree backpropagation (line 14)
  - ACT halting (line 15)
  - Best-trajectory decoding via Wproj (line 18)
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from cts.backbone.protocol import BaseCTSBackbone
from cts.critic.neuro_critic import NeuroCritic
from cts.eval.mac_accounting import linear_layer_mac_per_item
from cts.eval.isoflop_matcher import logical_transition_mac_from_stats
from cts.deq.root_bootstrap import solve_root_z_star
from cts.deq.transition import transition, transition_batch
from cts.latent.faiss_context import LatentContextWindow
from cts.mcts.hybrid_kv import HybridKVManager, hybrid_transition_decision
from cts.mcts.puct import PUCTVariant, select_action
from cts.mcts.tree import SearchTree
from cts.policy.meta_policy import MetaPolicy
from cts.perf.phase_profiler import CudaPhaseProfiler
from cts.types import (
    NuConfigMode,
    NuVector,
    RuntimeBudgetState,
    TransitionResult,
    TreeNode,
)


@dataclass
class CtsEpisodeResult:
    """Output of a single CTS episode."""
    answer: str
    best_z_star: Optional[torch.Tensor]
    tree: SearchTree
    total_mac: float
    total_iterations: int
    stats: Dict[str, Any] = field(default_factory=dict)


def _pool_z_star(z: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Mean-pool z* [K, d] -> [d] for MetaPolicy / Critic input."""
    if z is None:
        return None
    return z.detach().float().mean(dim=0)


def _store_inverse_state(state: Any) -> Any:
    """Keep dense test states as-is; offload Gemma low-rank factors to CPU."""
    if state is None:
        return None
    if hasattr(state, "for_tree_storage"):
        return state.for_tree_storage()
    if hasattr(state, "detach"):
        return state.detach()
    return state


def _apply_nu_expl_override(nu: NuVector, override: Optional[float]) -> NuVector:
    if override is None:
        return nu
    return NuVector(
        nu_val=nu.nu_val,
        nu_expl=float(override),
        nu_tol=nu.nu_tol,
        nu_temp=nu.nu_temp,
        nu_act=nu.nu_act,
    )


def _backpropagate(tree: SearchTree, node_id: int, q_values: List[float]) -> None:
    """Backpropagate mean Q from children up to root (paper line 14)."""
    node = tree.nodes[node_id]
    if not q_values:
        return
    mean_q = sum(q_values) / len(q_values)

    cur_id: Optional[int] = node_id
    while cur_id is not None:
        cur = tree.nodes[cur_id]
        old_n = cur.mcts_N
        cur.mcts_N = old_n + 1
        if cur.parent_id is not None:
            parent = tree.nodes[cur.parent_id]
            child_idx = parent.children_ids.index(cur_id) if cur_id in parent.children_ids else 0
            if child_idx < len(parent.mcts_Q):
                old_q = parent.mcts_Q[child_idx]
                # ``mcts_N`` is the edge-value sample count.  A child is
                # initialized to one visit when its critic value is installed;
                # descendant rollouts then update the exact running mean.
                # The historical max(1, old_n) counted two samples in Q while
                # recording one visit and increasingly overweighted recent
                # rollouts.
                old_visits = old_n
                parent.mcts_Q[child_idx] = (old_q * old_visits + mean_q) / (old_visits + 1)
        cur_id = cur.parent_id


def _select_leaf(
    tree: SearchTree,
    nu_expl: float,
    variant: PUCTVariant = "paper",
    *,
    nu_temp: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """PUCT tree traversal: select a leaf node for expansion (paper line 3).

    ``nu_temp`` is retained only for API compatibility.  Paper Eq. 3 defines
    it as the sparse-module routing temperature, not a PUCT/Gumbel
    temperature.  Exploration in PUCT is controlled solely by ``nu_expl``;
    branch diversity comes from the explicit latent proposals.

    ``generator`` is an optional ``torch.Generator`` used to draw the Gumbel
    samples deterministically; pass ``selection_seed`` from
    ``cts_full_episode`` to make per-(seed, problem) exploration
    reproducible.
    """
    from cts.mcts.puct import puct_score

    cur = 0
    while tree.nodes[cur].children_ids:
        node = tree.nodes[cur]
        active_children = [
            (idx, cid)
            for idx, cid in enumerate(node.children_ids)
            if not tree.nodes[cid].is_terminal
        ]
        if not active_children:
            node.is_terminal = True
            return cur
        n_parent = max(1, node.mcts_N)

        best_score = float("-inf")
        best_child = active_children[0][1]
        for idx, cid in active_children:
            child = tree.nodes[cid]
            prior = node.mcts_prior[idx] if idx < len(node.mcts_prior) else 1.0 / len(active_children)
            q = node.mcts_Q[idx] if idx < len(node.mcts_Q) else 0.0
            n_sa = child.mcts_N

            score = puct_score(variant, nu_expl, prior, n_parent, n_sa, q)

            if score > best_score:
                best_score = score
                best_child = cid
        cur = best_child
    return cur


def cts_full_episode(
    prompt: str,
    *,
    decode_prompt: Optional[str] = None,
    backbone: BaseCTSBackbone,
    meta_policy: MetaPolicy,
    critic: NeuroCritic,
    W: int = 3,
    K: int = 64,
    tau_budget: float = 1e14,
    broyden_max_iter: int = 30,
    broyden_tol_min: float = 1e-4,
    broyden_tol_max: float = 1e-2,
    top_k: int = 3,
    puct_variant: PUCTVariant = "paper",
    faiss_context: Optional[LatentContextWindow] = None,
    hybrid_kv_manager: Optional[HybridKVManager] = None,
    max_decode_tokens: int = 64,
    routing_mode: str = "sparse",
    noise_sigma: float = 0.02,
    device: Optional[torch.device] = None,
    wall_clock_budget_s: Optional[float] = None,
    z0_seed: Optional[int] = None,
    selection_seed: Optional[int] = None,
    nu_config_mode: Optional[NuConfigMode] = None,
    nu_trace: Optional[List[NuVector]] = None,
    nu_expl_override: Optional[float] = None,
    k_override: Optional[int] = None,
    w_override: Optional[int] = None,
    require_root_convergence: bool = False,
    batch_siblings: bool = True,
    strict_decode: bool = True,
    max_depth: Optional[int] = None,
    act_override: Optional[str] = None,
    solver: str = "broyden",
    latch_warmup: Optional[int] = None,
) -> CtsEpisodeResult:
    """Algorithm 1: Cognitive Tree Search — Single Episode.

    Require: Prompt s0, budget tau, W, f_theta, pi_phi, V_psi, FAISS F
    Ensure: Decoded answer y_hat

    ``nu_config_mode``: paper Table 5 nu-component Pareto switch.
        ``None`` or ``"4nu"`` keeps every meta-policy output live (CTS-4nu).
        ``"2nu_fast"`` keeps {expl, temp} active and freezes {tol, act} at
        the Stage 1 converged means; this is the canonical CTS-2nu fast
        mode used in Table 5. Other valid modes: ``"3nu_no_act"``,
        ``"2nu_expl_tol"``, ``"1nu"``. Mode switching requires NO
        retraining (paper §7.5 footnote).

    ``nu_trace``: optional caller-owned list. When provided, every NuVector
        sampled by the meta-policy (after any ``nu_config_mode`` override)
        is appended to it. This is the persistence hook used by
        ``cts/eval/nu_stats.py`` to reproduce the paper Table 19 per-domain
        ν statistics. ``None`` (the default) keeps the historical zero-overhead
        behaviour so the existing 308 tests stay green.

    ``nu_expl_override``: when set, replaces ``nu.nu_expl`` after the
        meta-policy sample and ``nu_config_mode`` mask (paper Table 2
        UCB1 bandit baseline: gradient-free adaptive nu_expl).

    ``k_override``: paper Table 13 (MCTS top-K children sensitivity sweep).
        When provided, overrides the per-leaf children-expansion count
        (otherwise equal to ``W``) without retraining. ``None`` (default)
        preserves the historical behaviour where ``W`` controls both the
        branching factor and the per-step simulation budget. NOTE: this is
        a DIFFERENT "K" than the latent-token count ``K`` (paper §4.2,
        README "Configuration" K=64); the latter is the bottleneck width
        of z*_t and is unaffected.

    ``w_override``: paper Table 15 (per-step MCTS simulation budget scaling
        sweep). When provided, caps the outer PUCT loop at exactly
        ``w_override`` iterations regardless of ``tau_budget`` / ACT halting.
        ``None`` (default) preserves the historical compute-budget halting
        behaviour. NOTE: this is a DIFFERENT "W" than the branching factor
        ``W`` (paper §4.1, README "Configuration" W=3); the latter is the
        children-per-leaf parallelism and is unaffected unless
        ``k_override`` is set.
    """
    if device is None:
        if hasattr(backbone, "parameters"):
            device = next(backbone.parameters()).device
        else:
            device = torch.device("cpu")
    device = torch.device(device)
    phase_profiler = CudaPhaseProfiler.from_env(device)
    if (
        phase_profiler is not None
        and wall_clock_budget_s is not None
        and float(wall_clock_budget_s) > 0.0
    ):
        raise ValueError(
            "CTS phase profiling requires MAC/simulation-controlled halting; "
            "disable the wall-clock budget"
        )

    d = backbone.hidden_size
    meta_policy_mac_per_call = linear_layer_mac_per_item(meta_policy)
    critic_mac_per_state = linear_layer_mac_per_item(critic)
    policy_mac_accumulated = 0.0
    critic_mac_accumulated = 0.0

    # Sweep overrides (paper Table 13 / Table 15). These default to ``None`` so
    # existing call sites are byte-identical; pinned by
    # ``tests/test_sweep_K_W_lambda.py::test_cts_full_episode_accepts_k_and_w_override``.
    W_eff = int(k_override) if k_override is not None else int(W)
    if W_eff < 1:
        raise ValueError(f"k_override must be >= 1 (got {k_override})")
    sim_cap: Optional[int] = None
    if w_override is not None:
        sim_cap = int(w_override)
        if sim_cap < 1:
            raise ValueError(f"w_override must be >= 1 (got {w_override})")

    import time as _time
    _start_t = _time.time()
    _deadline = (_start_t + wall_clock_budget_s) if wall_clock_budget_s and wall_clock_budget_s > 0 else None

    # Line 1: z*_0 <- FwdPass(s0); B0 <- 0.1*I; init T; MAC <- 0
    context_scope = (
        phase_profiler.span("episode.context_encode")
        if phase_profiler is not None
        else nullcontext()
    )
    with context_scope:
        with torch.no_grad():
            context_0 = backbone.encode_context(prompt)
    # ``transition`` historically re-encoded the (eventually one-token)
    # child text for every branch.  Preserve the original problem grounding
    # and pay the full-backbone context cost once per episode instead.
    if isinstance(context_0, torch.Tensor):
        if context_0.dim() == 1:
            context_0 = context_0.unsqueeze(0)
        context_0 = context_0.to(device=device, dtype=torch.float32)
    elif hasattr(context_0, "to"):
        context_0 = context_0.to(device=device)

    # Bootstrap a prompt-conditioned root fixed point before MetaPolicy acts.
    # This same helper is used by Stage-2 collection, eliminating the former
    # train/eval state mismatch (text context in training, random z0 in eval).
    _root_seed = int(z0_seed) if z0_seed is not None else 2026
    root_scope = (
        phase_profiler.span("root.total")
        if phase_profiler is not None
        else nullcontext()
    )
    with root_scope:
        root_result = solve_root_z_star(
            prompt,
            backbone=backbone,
            context=context_0,
            K=K,
            d=d,
            broyden_max_iter=broyden_max_iter,
            broyden_tol_min=broyden_tol_min,
            broyden_tol_max=broyden_tol_max,
            top_k=top_k,
            tau_flops_budget=tau_budget,
            seed=_root_seed,
            device=device,
            require_converged=require_root_convergence,
            phase_profiler=phase_profiler,
            profile_scope="root.transition",
        )
    root_wall_s = _time.time() - _start_t
    transition_wall_s = 0.0
    critic_wall_s = 0.0
    search_start_t = _time.time()
    _select_g: Optional[torch.Generator] = None
    if selection_seed is not None:
        _select_g = torch.Generator(device="cpu").manual_seed(int(selection_seed))

    tree = SearchTree()
    root_id = tree.new_node(
        prompt,
        root_result.z_star_child,
        depth=0,
        parent_id=None,
        W=W_eff,
        solver_stats=root_result.solver_stats,
    )
    root_inv_jac = root_result.solver_stats.get("inv_jacobian")
    if root_inv_jac is not None:
        tree.nodes[root_id].inv_jacobian_state = _store_inverse_state(
            root_inv_jac
        )

    root_logical_mac = logical_transition_mac_from_stats(
        root_result.solver_stats
    )
    # Algorithm 1 initializes z*_0/B0 and then starts the outer-search MAC
    # counter at zero.  Charging root FwdPass/solve against tau prematurely
    # removes simulations and changes the method.  Keep both accounting views:
    # ``mac_accumulated`` controls the paper search budget, while
    # ``total_executed_mac`` below includes root work for honest cost reports.
    mac_accumulated = 0.0
    # Algorithm-1 root FwdPass is not an Eq. (3) transition and therefore has
    # no routing-LUT identity.  Establish the MAC-v2 provenance from the first
    # actual child transition instead of comparing it with a synthetic root.
    mac_contract_valid = True
    mac_lut_schema_version: Optional[int] = None
    mac_lut_sha256: Optional[str] = None
    faiss_retrieval_calls: int = 0
    total_iterations = int(root_result.solver_stats.get("iterations", 0))
    sim_count = 0

    if faiss_context is not None:
        faiss_context.reset()
        if root_result.z_star_child is not None:
            faiss_context.add(root_result.z_star_child, node_id=root_id)

    # Algorithm 1 selects with the previously available controller state,
    # then evaluates pi(z_s) exactly once for the selected leaf.  The old loop
    # selected a default leaf, sampled pi, selected a *different* leaf, and
    # sampled pi again while charging one policy pass.
    selection_nu_expl = 1.0

    # Line 2: while MAC < tau do
    while mac_accumulated < tau_budget:
        if _deadline is not None and _time.time() > _deadline:
            break
        # Paper Table 15 sim-budget cap: when ``w_override`` is set, halt the
        # outer PUCT loop after exactly ``sim_cap`` iterations regardless of
        # the analytic FLOP budget. ``None`` preserves the historical
        # tau_budget / ACT halting path used by the integrated baselines.
        if sim_cap is not None and sim_count >= sim_cap:
            break
        sim_count += 1
        # Line 3: s <- PUCT(T, V_psi, nu_expl)
        leaf_id = _select_leaf(
            tree, nu_expl=selection_nu_expl, variant=puct_variant,
            nu_temp=0.0, generator=_select_g,
        )
        leaf = tree.nodes[leaf_id]
        if max_depth is not None and leaf.depth >= max_depth:
            leaf.is_terminal = True
            if leaf_id == root_id:
                break
            continue
        if leaf.is_terminal:
            if leaf_id == root_id:
                break
            continue

        # Line 4: nu_A <- pi_phi(z*_s)
        z_star_s = leaf.z_star
        z_pooled = _pool_z_star(z_star_s)
        if z_pooled is None:
            z_pooled = torch.zeros(d, device=device, dtype=torch.float32)

        policy_scope = (
            phase_profiler.span("search.policy")
            if phase_profiler is not None
            else nullcontext()
        )
        with policy_scope:
            with torch.no_grad():
                nu, priors = meta_policy(z_pooled.to(device))
        if nu_config_mode is not None:
            nu = nu.apply_config(nu_config_mode)
        nu = _apply_nu_expl_override(nu, nu_expl_override)
        selection_nu_expl = float(nu.nu_expl)
        if nu_trace is not None:
            nu_trace.append(nu)

        # Line 5: charge the actual Linear geometry.  The historical 2e11
        # constant was ~9,800x larger than this implementation and distorted
        # ACT/tree depth while the placeholder Eq.3 LUT erred the other way.
        mac_accumulated += meta_policy_mac_per_call
        policy_mac_accumulated += meta_policy_mac_per_call

        # Line 6: t <- depth(s); {z_tilde_w} <- z*_s + epsilon_w
        t = leaf.depth
        # Algorithm 1: through depth 10, H_t is the exact root-to-selected-
        # leaf stack.  At t>10 it is replaced by per-branch semantic FAISS
        # retrieval over the entire *past* tree snapshot.
        if t <= 10:
            _path_history = tree.ancestor_latents(leaf_id, include_self=True)
            # Equation (3) fixes H_t in R^{3 x K x d}.  Before semantic
            # retrieval activates, AncestorStack supplies the three most
            # recent available states in chronological order.
            shallow_history = (
                _path_history[-3:] if _path_history is not None else None
            )
        else:
            shallow_history = None

        # Line 7-11: for w = 1,...,W in parallel (W_eff honours k_override)
        child_q_values: List[float] = []
        # Paper Remark 2: thread the parent's converged inverse Jacobian (if any)
        # into each child solve as the warm start.
        leaf_inv_jac = getattr(leaf, "inv_jacobian_state", None)
        # Paper §7.7: query the Hybrid KV manager for a per-leaf decision. The
        # decision call is recorded on the result regardless of whether the
        # KV-reuse fast path is wired into the active backbone, so reviewers
        # can audit that the §7.7 policy actually fires for shallow nodes.
        kv_use, kv_cached = (
            hybrid_transition_decision(leaf.depth, leaf_id, hybrid_kv_manager, backbone, leaf.text_state)
            if hybrid_kv_manager is not None
            else (False, None)
        )
        branch_seed = (
            _root_seed
            + sim_count * 1_000_003
            + leaf_id * 10_009
            + t * 101
        )
        common_kwargs = dict(
            K=K,
            d=d,
            top_k=top_k,
            broyden_max_iter=broyden_max_iter,
            broyden_tol_min=broyden_tol_min,
            broyden_tol_max=broyden_tol_max,
            tau_flops_budget=tau_budget,
            routing_mode=routing_mode,
            max_decode_tokens=1,
            faiss_context=faiss_context,
            faiss_retrieve=(t > 10),
            faiss_depth=t,
            parent_z_star=z_star_s,
            noise_sigma=noise_sigma,
            parent_inv_jacobian=leaf_inv_jac,
            context_override=context_0,
            history_latents=shallow_history,
            defer_faiss_add=True,
            decode_child=False,
            noise_seed=branch_seed,
            solver=solver,
            latch_warmup=latch_warmup,
        )
        if phase_profiler is not None:
            common_kwargs.update(
                phase_profiler=phase_profiler,
                profile_scope="search.transition",
            )
        starting_budget = RuntimeBudgetState(mac_accumulated=mac_accumulated)
        transition_t0 = _time.time()
        can_batch = bool(
            batch_siblings
            and W_eff > 1
            and getattr(backbone, "supports_batched_deq", False)
        )
        # Search is inference-only.  ``eval()`` does not disable autograd;
        # without this guard every Broyden Phi retained a full Gemma graph,
        # multiplying VRAM and latency by the iteration count.
        transition_scope = (
            phase_profiler.span("search.transition.total")
            if phase_profiler is not None
            else nullcontext()
        )
        with transition_scope:
            with torch.no_grad():
                if can_batch:
                    branch_results = transition_batch(
                        leaf.text_state,
                        nu,
                        starting_budget,
                        backbone,
                        W=W_eff,
                        **common_kwargs,
                    )
                else:
                    branch_results = [
                        transition(
                            leaf.text_state,
                            w,
                            nu,
                            starting_budget,
                            backbone,
                            **common_kwargs,
                        )
                        for w in range(W_eff)
                    ]
        # Once a node has been expanded its inverse-Jacobian state is never
        # needed again: future PUCT visits descend through its children.  A
        # rank-16 K=64,H=2560 state is ~20 MiB, so retaining expanded nodes can
        # consume many GiB on a large tree.
        leaf.inv_jacobian_state = None
        transition_wall_s += _time.time() - transition_t0

        # One batched critic call replaces W scalar ``.item()`` synchronises.
        pooled_children = []
        for r in branch_results:
            pooled = _pool_z_star(r.z_star_child)
            pooled_children.append(
                pooled if pooled is not None else torch.zeros(d, device=device)
            )
        critic_t0 = _time.time()
        critic_scope = (
            phase_profiler.span("search.critic")
            if phase_profiler is not None
            else nullcontext()
        )
        with critic_scope:
            with torch.no_grad():
                q_batch = critic(torch.stack(pooled_children).to(device))
                q_values = q_batch.reshape(-1).detach().float().cpu().tolist()
        critic_wall_s += _time.time() - critic_t0
        critic_step_mac = critic_mac_per_state * float(len(pooled_children))
        mac_accumulated += critic_step_mac
        critic_mac_accumulated += critic_step_mac

        successful_children = 0
        faiss_commit_latents: List[torch.Tensor] = []
        faiss_commit_ids: List[int] = []
        for r, q_raw in zip(branch_results, q_values):
            if getattr(r, "faiss_retrieved", None) is not None:
                faiss_retrieval_calls += 1
            total_iterations += int(r.solver_stats.get("iterations", 0))
            step_mac = logical_transition_mac_from_stats(r.solver_stats)
            mac_accumulated += step_mac
            branch_schema = int(
                r.solver_stats.get("mac_lut_schema_version", 0)
            )
            branch_sha = str(r.solver_stats.get("mac_lut_sha256", ""))
            if mac_lut_schema_version is None:
                mac_lut_schema_version = branch_schema
                mac_lut_sha256 = branch_sha
            mac_contract_valid = bool(
                mac_contract_valid
                and r.solver_stats.get("mac_headline_eligible", False)
                and branch_schema == mac_lut_schema_version
                and branch_sha == mac_lut_sha256
            )
            q_w = 0.0 if r.prune else float(q_raw)
            child_q_values.append(q_w)

            if r.prune:
                terminal_id = tree.new_node(
                    leaf.text_state,
                    None,
                    depth=t + 1,
                    parent_id=leaf_id,
                    W=W_eff,
                    solver_stats=r.solver_stats,
                )
                tree.nodes[terminal_id].is_terminal = True
                tree.nodes[terminal_id].mcts_N = 1
                continue

            z_child = r.z_star_child
            child_id = tree.new_node(
                r.child_text or leaf.text_state,
                z_child,
                depth=t + 1,
                parent_id=leaf_id,
                W=W_eff,
                solver_stats=r.solver_stats,
            )
            if faiss_context is not None:
                faiss_commit_latents.append(z_child)
                faiss_commit_ids.append(child_id)
            inv_jac_child = r.solver_stats.get("inv_jacobian")
            if inv_jac_child is not None:
                tree.nodes[child_id].inv_jacobian_state = _store_inverse_state(
                    inv_jac_child
                )
            # The critic value written into the parent edge is the first value
            # sample for this child.
            tree.nodes[child_id].mcts_N = 1
            successful_children += 1

        if faiss_context is not None and faiss_commit_latents:
            # One commit after all sibling queries/solves preserves the
            # parallel expansion snapshot and avoids W separate D2H copies.
            faiss_scope = (
                phase_profiler.span("search.faiss_commit")
                if phase_profiler is not None
                else nullcontext()
            )
            with faiss_scope:
                faiss_context.add_batch(
                    torch.stack(faiss_commit_latents), node_ids=faiss_commit_ids
                )

        if successful_children == 0:
            leaf.is_terminal = True

        # Update priors on the expanded node (W_eff honours k_override)
        tree.nodes[leaf_id].mcts_prior = list(priors) if len(priors) == W_eff else [1.0 / W_eff] * W_eff
        tree.nodes[leaf_id].mcts_Q = child_q_values[:W_eff]

        # Line 14: BackProp(T, {Q_w})
        _backpropagate(tree, leaf_id, child_q_values)

        # Line 15-16: if MAC >= tau * nu_act then break
        if act_override != "disable":
            if mac_accumulated >= tau_budget * nu.nu_act:
                break

    search_wall_s = _time.time() - search_start_t

    # Line 18: y_hat <- Decode(W_proj @ z*_best)
    best_id = 0
    best_q = float("-inf")
    candidate_ids: List[int] = []
    candidate_latents: List[torch.Tensor] = []
    for node in tree.nodes:
        if node.z_star is not None and node.depth > 0 and not node.is_terminal:
            z_p = _pool_z_star(node.z_star)
            if z_p is not None:
                candidate_ids.append(node.node_id)
                candidate_latents.append(z_p)
    if candidate_latents:
        final_critic_scope = (
            phase_profiler.span("search.final_critic")
            if phase_profiler is not None
            else nullcontext()
        )
        with final_critic_scope:
            with torch.no_grad():
                final_values = critic(torch.stack(candidate_latents).to(device))
            values_host = final_values.reshape(-1).detach().float().cpu().tolist()
        best_pos = max(range(len(values_host)), key=values_host.__getitem__)
        best_id = candidate_ids[best_pos]
        best_q = float(values_host[best_pos])
    post_search_critic_mac = (
        critic_mac_per_state * float(len(candidate_latents))
    )

    best_z = tree.nodes[best_id].z_star

    answer = ""
    decode_t0 = _time.time()
    if best_z is not None and hasattr(backbone, "decode_from_z_star"):
        try:
            # Keep search conditioned on the raw problem while allowing the
            # terminal decoder to use a separately frozen answer-format
            # contract. This avoids changing the MetaPolicy/Eq.3 state
            # distribution merely to make the final response concise.
            # The soft-prompt prefix is composed with the problem context,
            # avoiding the
            # failure mode where a compute-limited W_proj produces random
            # non-sequitur tokens (e.g. 'Cultura', 'LinearLayout') on
            # math/AIME prompts because the soft-prompt alone lacks the
            # textual grounding the model needs to emit a numeric answer.
            # Headline evaluation is strict: a backbone that violates this
            # prompt-grounded decoder contract must fail rather than silently
            # switching to the old promptless path.
            decode_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_decode_tokens,
                "problem_text": (
                    decode_prompt if decode_prompt is not None else prompt
                ),
            }
            if phase_profiler is not None:
                decode_signature = inspect.signature(backbone.decode_from_z_star)
                if "phase_profiler" in decode_signature.parameters:
                    decode_kwargs.update(
                        phase_profiler=phase_profiler,
                        profile_scope="decode",
                    )
            decode_scope = (
                phase_profiler.span("decode.total")
                if phase_profiler is not None
                else nullcontext()
            )
            with decode_scope:
                answer = backbone.decode_from_z_star(best_z, **decode_kwargs)
        except Exception:
            if strict_decode:
                raise
            answer = tree.nodes[best_id].text_state
    decode_wall_s = _time.time() - decode_t0

    stats: Dict[str, Any] = {
        "tree_size": len(tree.nodes),
        "max_depth": max(n.depth for n in tree.nodes),
        "best_node_id": best_id,
        "best_q": best_q,
        # Sweep-driver provenance (paper Tables 13 / 15). When neither
        # override is set, ``k_override_used`` falls back to ``W`` so
        # downstream aggregation always sees the *effective* branching
        # factor; ``w_override_used`` stays ``None`` to disambiguate the
        # uncapped (FLOP-budget halting) path.
        "k_override_used": W_eff,
        "w_override_used": sim_count if sim_cap is not None else None,
        "sim_count": sim_count,
        "logical_mac_accumulated": float(mac_accumulated),
        "root_logical_mac": float(root_logical_mac),
        "total_executed_mac": float(root_logical_mac + mac_accumulated),
        "meta_policy_mac_accumulated": float(policy_mac_accumulated),
        "critic_mac_accumulated": float(critic_mac_accumulated),
        "post_search_critic_mac": float(post_search_critic_mac),
        "mac_unit": "MAC",
        "mac_headline_eligible": bool(mac_contract_valid),
        "mac_lut_schema_version": int(mac_lut_schema_version or 0),
        "mac_lut_sha256": str(mac_lut_sha256 or ""),
        "root_iterations": int(root_result.solver_stats.get("iterations", 0)),
        "root_converged": bool(root_result.solver_stats.get("converged", False)),
        "root_residual_norm": float(
            root_result.solver_stats.get("residual_norm", float("nan"))
        ),
        "root_absolute_residual_norm": float(
            root_result.solver_stats.get(
                "absolute_residual_norm", float("nan")
            )
        ),
        "root_failure_reason": root_result.solver_stats.get("failure_reason"),
        "root_routing_module_ids": list(
            root_result.solver_stats.get("routing_module_ids", [])
        ),
        "root_initialization": str(
            root_result.solver_stats.get("root_initialization", "unknown")
        ),
        "context_encode_calls": 1,
        "intermediate_decode_calls": 0,
        "faiss_retrieval_calls": int(faiss_retrieval_calls),
        "act_override": act_override,
        "solver": solver,
        "decode_prompt_separate": bool(decode_prompt is not None),
        "batch_siblings_requested": bool(batch_siblings and W_eff > 1),
        "batch_siblings_used": bool(
            batch_siblings
            and W_eff > 1
            and getattr(backbone, "supports_batched_deq", False)
        ),
        "timing_s": {
            "root": float(root_wall_s),
            "search": float(search_wall_s),
            "transitions": float(transition_wall_s),
            "critic": float(critic_wall_s),
            "decode": float(decode_wall_s),
            "total": float(_time.time() - _start_t),
        },
    }
    if hybrid_kv_manager is not None:
        stats["hybrid_kv"] = hybrid_kv_manager.report()
    if phase_profiler is not None:
        stats["phase_profile"] = phase_profiler.finish()
    return CtsEpisodeResult(
        answer=answer,
        best_z_star=best_z,
        tree=tree,
        total_mac=mac_accumulated,
        total_iterations=total_iterations,
        stats=stats,
    )
