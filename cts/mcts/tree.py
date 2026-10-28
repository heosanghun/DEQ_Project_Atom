"""Minimal MCTS tree (expand / attach children)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from cts.types import TreeNode

from cts.types import TreeNode


@dataclass
class SearchTree:
    nodes: List[TreeNode] = field(default_factory=list)

    def new_node(self, text: str, z_star, depth: int, parent_id: Optional[int], W: int = 3, solver_stats: Optional[Dict[str, Any]] = None) -> int:
        nid = len(self.nodes)
        self.nodes.append(
            TreeNode(
                text_state=text,
                z_star=z_star,
                depth=depth,
                parent_id=parent_id,
                node_id=nid,
                mcts_W=W,
                solver_stats=solver_stats,
            )
        )
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(nid)
        return nid

    def root(self) -> TreeNode:
        return self.nodes[0]

    def ancestor_ids(self, node_id: int, *, include_self: bool = True) -> List[int]:
        """Return the selected branch path in root-to-leaf order.

        Algorithm 1 uses ``AncestorStack(s)`` through depth 10.  Deriving the
        stack from parent links prevents unrelated siblings from leaking into
        the shallow context merely because they were inserted more recently.
        """
        if node_id < 0 or node_id >= len(self.nodes):
            raise IndexError(f"invalid tree node id {node_id}")
        current: Optional[int] = node_id if include_self else self.nodes[node_id].parent_id
        path: List[int] = []
        seen = set()
        while current is not None:
            if current in seen:
                raise RuntimeError("cycle detected in SearchTree parent links")
            seen.add(current)
            path.append(current)
            current = self.nodes[current].parent_id
        path.reverse()
        return path

    def ancestor_latents(
        self, node_id: int, *, include_self: bool = True
    ) -> Optional[torch.Tensor]:
        """Stack available branch latents as ``[A,K,H]`` root-to-leaf."""
        path = self.ancestor_ids(node_id, include_self=include_self)
        missing = [nid for nid in path if self.nodes[nid].z_star is None]
        if missing:
            raise ValueError(
                f"selected ancestor path contains terminal/missing latents: {missing}"
            )
        latents = [self.nodes[nid].z_star for nid in path]
        if not latents:
            return None
        first_shape = tuple(latents[0].shape)
        if any(tuple(z.shape) != first_shape for z in latents):
            raise ValueError("ancestor latent shapes are inconsistent")
        return torch.stack(latents, dim=0)
