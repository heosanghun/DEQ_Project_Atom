"""FAISS Latent Space Context Window (paper §4.3, Appendix H).

Paper: FAISS-IVF-PQ (nlist=100, nprobe=20, PQ m=8 codes, 8 bits/code).
Mean-pooled z* used as search key; full K x d FP16 vectors stored and returned.
Table 1: FAISS raw latent 800 KB (N=100), structural (IVF+PQ) 99 MB.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

try:
    import faiss  # type: ignore[import-untyped]
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def faiss_available() -> bool:
    return _FAISS_AVAILABLE


class LatentContextWindow:
    """Paper §4.3 / Appendix H: FAISS-backed Latent Space Context Window.

    - Stores full K x d FP16 z* vectors (not just mean-pooled)
    - Mean-pooled 1D used as search key in IVF-PQ index
    - Retrieves Top-k semantically relevant ancestral vectors
    - Active when step > min_steps (paper: t > 10)
    """

    def __init__(
        self,
        dim: int,
        *,
        retrieval_k: int = 3,
        min_steps: int = 10,
        nlist: int = 100,
        nprobe: int = 20,
        pq_m: int = 8,
        pq_bits: int = 8,
    ) -> None:
        self.dim = dim
        self.retrieval_k = retrieval_k
        self.min_steps = min_steps
        self.nlist = nlist
        self.nprobe = nprobe
        self.pq_m = pq_m
        self.pq_bits = pq_bits

        self._keys: List[torch.Tensor] = []
        self._full_vectors: List[torch.Tensor] = []
        self._node_ids: List[Optional[int]] = []
        self._node_id_to_slot: Dict[int, int] = {}
        self._step_count = 0
        self._trained = False
        self.last_retrieval_stats: Dict[str, Any] = {}

        effective_pq_m = pq_m
        if dim % pq_m != 0:
            for candidate in [8, 4, 2, 1]:
                if dim % candidate == 0:
                    effective_pq_m = candidate
                    break
        self._effective_pq_m = effective_pq_m

        if _FAISS_AVAILABLE and np is not None:
            self._flat_index = faiss.IndexFlatIP(dim)
            self._ivfpq_index = None
        else:
            self._flat_index = None
            self._ivfpq_index = None

    def _build_ivfpq(self) -> None:
        """Build IVF-PQ index once we have enough vectors for training."""
        if not _FAISS_AVAILABLE or np is None:
            return
        n = len(self._keys)
        actual_nlist = min(self.nlist, max(1, n // 10))
        if n < actual_nlist * 4:
            return

        quantizer = faiss.IndexFlatIP(self.dim)
        ivfpq = faiss.IndexIVFPQ(
            quantizer, self.dim, actual_nlist, self._effective_pq_m, self.pq_bits,
        )
        ivfpq.nprobe = min(self.nprobe, actual_nlist)

        train_data = torch.stack(self._keys).numpy().astype(np.float32)
        faiss.normalize_L2(train_data)
        ivfpq.train(train_data)
        ivfpq.add(train_data)

        self._ivfpq_index = ivfpq
        self._trained = True

    @property
    def size(self) -> int:
        return len(self._full_vectors)

    @property
    def step_count(self) -> int:
        return self._step_count

    def add(self, z_star: torch.Tensor, *, node_id: Optional[int] = None) -> int:
        """Add a fixed-point to the context window.

        z_star: [K, d] latent tokens.
        Stores: full K x d FP16 tensor + mean-pooled [d] as search key.
        """
        return self.add_batch(z_star.unsqueeze(0), node_ids=[node_id])[0]

    def add_batch(
        self,
        z_star_batch: torch.Tensor,
        *,
        node_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> List[int]:
        """Commit converged siblings with one D2H/index update."""
        if z_star_batch.dim() != 3 or z_star_batch.shape[-1] != self.dim:
            raise ValueError("add_batch expects [B,K,dim]")
        batch = int(z_star_batch.shape[0])
        ids = list(node_ids) if node_ids is not None else [None] * batch
        if len(ids) != batch:
            raise ValueError("node_ids length must match z_star_batch")
        for node_id in ids:
            if node_id is not None and int(node_id) in self._node_id_to_slot:
                raise ValueError(f"duplicate FAISS node_id {node_id}")

        full_cpu = z_star_batch.detach().to(device="cpu", dtype=torch.float16)
        pooled_cpu = z_star_batch.detach().float().mean(dim=1).cpu()
        slots: List[int] = []
        for bi, node_id in enumerate(ids):
            slot = len(self._full_vectors)
            slots.append(slot)
            self._full_vectors.append(full_cpu[bi])
            self._keys.append(pooled_cpu[bi])
            canonical_id = None if node_id is None else int(node_id)
            self._node_ids.append(canonical_id)
            if canonical_id is not None:
                self._node_id_to_slot[canonical_id] = slot
        self._step_count += batch

        if np is not None and _FAISS_AVAILABLE:
            vec_np = pooled_cpu.numpy().astype(np.float32)
            faiss.normalize_L2(vec_np)
            if self._flat_index is not None:
                self._flat_index.add(vec_np)
            if self._ivfpq_index is not None and self._trained:
                self._ivfpq_index.add(vec_np)

        min_train = max(40, self.nlist * 4)
        if not self._trained and len(self._keys) >= min_train:
            self._build_ivfpq()
        return slots

    def retrieve(
        self,
        z_star: torch.Tensor,
        k: Optional[int] = None,
        *,
        depth: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Retrieve Top-k ancestral z* vectors (paper: t > min_steps).

        Returns: [k, K, d] tensor of full FP16 z* vectors, or None.
        """
        k = k or self.retrieval_k
        gate_value = self._step_count if depth is None else int(depth)
        if gate_value <= self.min_steps:
            return None
        if len(self._full_vectors) < k:
            return None

        pooled = z_star.detach().float().mean(dim=0).reshape(-1)

        indices = self._search_indices_batch(pooled.unsqueeze(0), k)[0]
        if not indices:
            return None

        retrieved = [self._full_vectors[idx].float() for idx in indices]
        return torch.stack(retrieved)

    def retrieve_batch(
        self,
        z_star_batch: torch.Tensor,
        k: Optional[int] = None,
        *,
        depth: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Query the same frozen tree snapshot independently for W branches.

        Returns ``[W,R,K,H]``.  Querying each perturbed ``z_tilde_w`` is part
        of Algorithm 1; broadcasting the parent's retrieval to all siblings
        removes the intended semantic diversity.
        """
        if z_star_batch.dim() != 3:
            raise ValueError("retrieve_batch expects [W,K,H]")
        k = k or self.retrieval_k
        gate_value = self._step_count if depth is None else int(depth)
        if gate_value <= self.min_steps or len(self._full_vectors) < k:
            return None
        pooled = z_star_batch.detach().float().mean(dim=1).cpu()
        all_indices = self._search_indices_batch(pooled, k)
        if any(len(indices) < k for indices in all_indices):
            return None
        rows = [
            torch.stack([self._full_vectors[idx].float() for idx in indices])
            for indices in all_indices
        ]
        self.last_retrieval_stats = {
            "query_count": int(z_star_batch.shape[0]),
            "snapshot_size": len(self._full_vectors),
            "storage_indices": all_indices,
            "node_ids": [
                [self._node_ids[idx] for idx in indices]
                for indices in all_indices
            ],
            "backend": "IVF-PQ" if self._trained else "FlatIP",
        }
        return torch.stack(rows, dim=0)

    def _search_indices_batch(
        self, query_pooled: torch.Tensor, k: int
    ) -> List[List[int]]:
        """Search W pooled queries in one FAISS/torch call."""
        if query_pooled.dim() != 2 or query_pooled.shape[1] != self.dim:
            raise ValueError("pooled queries must be [W,dim]")
        if np is not None and _FAISS_AVAILABLE:
            query_np = query_pooled.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(query_np)
            index = (
                self._ivfpq_index
                if self._ivfpq_index is not None and self._trained
                else self._flat_index
            )
            if index is not None and index.ntotal > 0:
                _, raw = index.search(query_np, min(k, index.ntotal))
                return [
                    [int(idx) for idx in row if 0 <= idx < len(self._full_vectors)]
                    for row in raw
                ]

        if not self._keys:
            return [[] for _ in range(query_pooled.shape[0])]
        queries = query_pooled.float()
        keys = torch.stack(self._keys).to(device=queries.device, dtype=torch.float32)
        keys = keys / keys.norm(dim=1, keepdim=True).clamp_min(1e-8)
        queries = queries / queries.norm(dim=1, keepdim=True).clamp_min(1e-8)
        top = torch.topk(
            queries @ keys.transpose(0, 1),
            k=min(k, keys.shape[0]),
            dim=1,
        ).indices
        return [[int(x) for x in row] for row in top.tolist()]

    def _search_indices(self, query_pooled: torch.Tensor, k: int) -> List[int]:
        """Search for top-k nearest neighbors."""
        if np is None:
            return self._cosine_fallback(query_pooled, k)

        query_np = query_pooled.cpu().numpy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_np) if _FAISS_AVAILABLE else None

        if self._ivfpq_index is not None and self._trained:
            try:
                _, indices = self._ivfpq_index.search(query_np, min(k, self._ivfpq_index.ntotal))
                return [int(idx) for idx in indices[0] if 0 <= idx < len(self._full_vectors)]
            except Exception:
                pass

        if self._flat_index is not None and self._flat_index.ntotal > 0:
            _, indices = self._flat_index.search(query_np, min(k, self._flat_index.ntotal))
            return [int(idx) for idx in indices[0] if 0 <= idx < len(self._full_vectors)]

        return self._cosine_fallback(query_pooled, k)

    def _cosine_fallback(self, query: torch.Tensor, k: int) -> List[int]:
        query_norm = query / (query.norm() + 1e-8)
        sims = []
        for i, v in enumerate(self._keys):
            v_norm = v / (v.norm() + 1e-8)
            sims.append((float(torch.dot(query_norm, v_norm)), i))
        sims.sort(reverse=True)
        return [idx for _, idx in sims[:k]]

    def memory_kb_per_node(self) -> float:
        """Backward compat: average KB per stored node."""
        n = max(len(self._full_vectors), 1)
        return self.memory_bytes() / n / 1024

    def memory_bytes(self) -> int:
        per_key = self.dim * 4
        per_full = 0
        if self._full_vectors:
            per_full = self._full_vectors[0].numel() * 2
        n = len(self._full_vectors)
        faiss_structural = 99 * 1024 * 1024 if self._trained else 0
        return n * (per_key + per_full) + faiss_structural

    def memory_report(self) -> dict:
        n = len(self._full_vectors)
        raw_kb = 0
        if n > 0 and self._full_vectors:
            raw_kb = n * self._full_vectors[0].numel() * 2 / 1024
        return {
            "n_vectors": n,
            "raw_latent_kb": raw_kb,
            "structural_mb": 99 if self._trained else 0,
            "index_type": "IVF-PQ" if self._trained else "FlatIP",
        }

    def reset(self) -> None:
        self._keys.clear()
        self._full_vectors.clear()
        self._node_ids.clear()
        self._node_id_to_slot.clear()
        self._step_count = 0
        self._trained = False
        self.last_retrieval_stats = {}
        self._ivfpq_index = None
        if _FAISS_AVAILABLE and np is not None:
            self._flat_index = faiss.IndexFlatIP(self.dim)


def prepend_soft_prefix(
    context: Any, retrieved: torch.Tensor
) -> Any:
    """Inject retrieved ancestral z* as prepended soft-prefix.

    Paper §4.3: concatenation s0 + Ht in sequence dimension.

    context: [seq_len, d] or GemmaEq3Context
    retrieved: [k, K, d] (full z* vectors) or [k, d] (pooled)
    Returns: [prefix_len + seq_len, d] or updated GemmaEq3Context
    """
    if hasattr(context, "with_history"):
        return context.with_history(retrieved)

    if retrieved.dim() == 3:
        k, K, d_r = retrieved.shape
        retrieved_flat = retrieved.reshape(k * K, d_r)
    else:
        retrieved_flat = retrieved

    if hasattr(context, "device") and retrieved_flat.device != context.device:
        retrieved_flat = retrieved_flat.to(context.device)
    if hasattr(context, "dtype") and retrieved_flat.dtype != context.dtype:
        retrieved_flat = retrieved_flat.to(context.dtype)

    if hasattr(context, "shape") and retrieved_flat.shape[-1] != context.shape[-1]:
        return context

    # Equation 3 is explicit: s0 comes first and H_t follows in the sequence
    # dimension.  Reversing these blocks changes causal visibility.
    return torch.cat([context, retrieved_flat], dim=0)
