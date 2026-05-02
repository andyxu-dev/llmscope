"""Memory profiling utilities for LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from llmscope.core.events import KVCacheSnapshot


@dataclass
class MemoryProfile:
    weights_bytes: int
    kv_bytes: int
    # None when running on CPU or CUDA stats unavailable
    activations_bytes: Optional[int]
    allocated_bytes: Optional[int]
    on_gpu: bool

    def weights_mb(self) -> float:
        return self.weights_bytes / 1e6

    def kv_mb(self) -> float:
        return self.kv_bytes / 1e6

    def activations_mb(self) -> Optional[float]:
        if self.activations_bytes is None:
            return None
        return self.activations_bytes / 1e6


class MemoryProfiler:
    """Compute memory attribution using the residual method."""

    def profile(
        self,
        model: nn.Module,
        latest_snapshot: Optional[KVCacheSnapshot] = None,
    ) -> MemoryProfile:
        weights = int(sum(p.nelement() * p.element_size() for p in model.parameters()))
        kv = latest_snapshot.total_bytes if latest_snapshot is not None else 0

        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated())
            activations = max(0, allocated - weights - kv)
            return MemoryProfile(
                weights_bytes=weights,
                kv_bytes=kv,
                activations_bytes=activations,
                allocated_bytes=allocated,
                on_gpu=True,
            )

        return MemoryProfile(
            weights_bytes=weights,
            kv_bytes=kv,
            activations_bytes=None,
            allocated_bytes=None,
            on_gpu=False,
        )
