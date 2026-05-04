"""Abstract base class for architecture-specific model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

PastKeyValues = tuple[tuple[Any, Any], ...]


class ArchitectureAdapter(ABC):
    """Encapsulates architecture-specific details for hook attachment and KV extraction.
    """

    @abstractmethod
    def get_hook_module(self, model: nn.Module) -> nn.Module:
        """Return the submodule to attach the forward hook to."""

    @abstractmethod
    def extract_past_key_values(self, output: Any) -> PastKeyValues | None:
        """Extract a tuple-of-(key, value) pairs from the module's forward output."""

    @abstractmethod
    def get_model_info(self, model: nn.Module) -> dict[str, Any]:
        """Return model metadata: num_layers, num_heads, hidden_dim, num_kv_heads."""


def normalize_past_key_values(raw: Any) -> PastKeyValues | None:
    """Convert all KV cache formats to a uniform tuple-of-(key, value) pairs.

    Handles three formats across transformers versions:
    - tuple-of-tuples  (≤4.35, still used by some models)
    - DynamicCache with .key_cache / .value_cache lists  (4.36–4.x)
    - DynamicCache with .layers list of DynamicLayer     (5.x)
    """
    if raw is None:
        return None
    if isinstance(raw, tuple):
        return raw

    # transformers 4.36–4.x: DynamicCache.key_cache / .value_cache
    key_cache: Any = getattr(raw, "key_cache", None)
    value_cache: Any = getattr(raw, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return tuple(zip(key_cache, value_cache))

    # transformers 5.x: DynamicCache.layers -> list[DynamicLayer] with .keys / .values
    layers: Any = getattr(raw, "layers", None)
    if layers is not None and len(layers) > 0:
        pairs = []
        for layer in layers:
            k: Any = getattr(layer, "keys", None)
            v: Any = getattr(layer, "values", None)
            if k is not None and v is not None:
                pairs.append((k, v))
        if pairs:
            return tuple(pairs)

    return None
