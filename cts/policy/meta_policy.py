"""Meta-policy: nu vector + branch priors (paper §4.1).

Paper §4.1: "Meta-policy pi_phi: a separate 2-layer MLP outputting
nu = [nu_expl, nu_tol, nu_temp, nu_act] in R^4"

V(z*) is output by Neuro-Critic V_psi, NOT part of nu.
Input: mean-pooled z*_s from the selected MCTS node.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from cts.types import NuVector


class MetaPolicy(nn.Module):
    """Paper §4.1: 2-layer MLP outputting nu in R^4 + branch priors.

    nu = [nu_expl, nu_tol, nu_temp, nu_act]  (4 outputs, NOT 5)
    V(z*) is handled by separate NeuroCritic, not part of nu.
    """

    def __init__(self, text_dim: int = 64, hidden: int = 256, W: int = 3) -> None:
        super().__init__()
        self.W = W
        self.enc = nn.Linear(text_dim, hidden)
        self.act = nn.ReLU()
        self.head_nu = nn.Linear(hidden, 4)
        self.head_prior = nn.Linear(hidden, W)
        # Stage-2 PPO treats the four raw nu controls as a continuous action.
        # The previous trainer converted ``head_nu`` to Python floats before
        # constructing its policy loss, so this head never received a single
        # gradient during the advertised 10k PPO steps.  A learned diagonal
        # log-standard-deviation makes the raw-space Normal policy explicit;
        # deterministic evaluation continues to use the mean below.
        self.nu_log_std = nn.Parameter(torch.full((4,), -1.0))

        # Start the continuous controller at the documented Stage-1 means.
        # In particular, nu_act is a bounded budget fraction; the historical
        # softplus(raw)+0.5 mapping started around 1.19, so the outer
        # ``MAC < tau`` guard always fired before ACT and the action had no
        # causal effect.  Zero weights retain a state-independent safe start
        # while PPO learns state dependence through this head.
        with torch.no_grad():
            self.head_nu.weight.zero_()
            self.head_nu.bias.copy_(
                torch.tensor(
                    [
                        -0.4327521296,  # inv_softplus(1.0 - 0.5)
                        -0.0202027073,  # logit((5e-3 - 1e-4) / 9.9e-3)
                        -0.4327521296,  # inv_softplus(1.0 - 0.5)
                        1.2656663733,   # logit(0.78)
                    ],
                    device=self.head_nu.bias.device,
                    dtype=self.head_nu.bias.dtype,
                )
            )

    def nu_std(self) -> torch.Tensor:
        """Numerically bounded raw-nu policy standard deviation."""
        return self.nu_log_std.clamp(min=-5.0, max=1.0).exp()

    def raw_outputs(
        self, z_star_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable ``(raw_nu_mean, branch_logits)`` batches.

        Both outputs keep their batch dimension: ``[B, 4]`` and ``[B, W]``.
        Stage-2 PPO must use this method so the continuous controls remain in
        the autograd graph.  :meth:`logits_and_nu` is the single-state,
        deterministic inference adapter.
        """
        if z_star_pooled.dim() == 1:
            z_star_pooled = z_star_pooled.unsqueeze(0)
        if z_star_pooled.dim() != 2:
            raise ValueError(
                "z_star_pooled must have shape [d] or [batch, d], "
                f"got {tuple(z_star_pooled.shape)}"
            )
        h = self.act(self.enc(z_star_pooled))
        return self.head_nu(h), self.head_prior(h)

    @staticmethod
    def raw_to_nu(raw: torch.Tensor) -> NuVector:
        """Map one raw four-vector to the runtime ``NuVector`` contract."""
        if raw.dim() == 2 and raw.shape[0] == 1:
            raw = raw.squeeze(0)
        if raw.shape != (4,):
            raise ValueError(f"raw nu action must have shape [4], got {tuple(raw.shape)}")
        tol_min, tol_max = 1e-4, 1e-2
        return NuVector(
            nu_expl=float(torch.nn.functional.softplus(raw[0]).item()) + 0.5,
            nu_tol=float(
                (tol_min + (tol_max - tol_min) * torch.sigmoid(raw[1])).item()
            ),
            nu_temp=float(torch.nn.functional.softplus(raw[2]).item()) + 0.5,
            nu_act=float(torch.sigmoid(raw[3]).item()),
        )

    def logits_and_nu(
        self, z_star_pooled: torch.Tensor
    ) -> Tuple[NuVector, torch.Tensor]:
        """Returns (nu vector, branch logits tensor) for PPO log-prob.

        Args:
            z_star_pooled: mean-pooled z* from selected node [batch, d] or [d].
        """
        raw, logits = self.raw_outputs(z_star_pooled)
        if raw.shape[0] != 1:
            raise ValueError(
                "logits_and_nu is a single-state inference API; "
                "use raw_outputs for batched PPO updates"
            )
        return self.raw_to_nu(raw[0]), logits[0]

    def forward(
        self, z_star_pooled: torch.Tensor
    ) -> Tuple[NuVector, List[float]]:
        nu, logits = self.logits_and_nu(z_star_pooled)
        p = torch.softmax(logits, dim=-1).tolist()
        return nu, p
