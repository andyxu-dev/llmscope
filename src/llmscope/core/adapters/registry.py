"""Auto-detect architecture and return the appropriate adapter."""

from __future__ import annotations

import torch.nn as nn

from .base import ArchitectureAdapter
from .causal_lm import CausalLMAdapter
from .generic import GenericAdapter
from .gpt2 import GPT2Adapter

_REGISTRY: dict[str, type[ArchitectureAdapter]] = {
    "gpt2": GPT2Adapter,
    "llama": CausalLMAdapter,
    "mistral": CausalLMAdapter,
    "qwen2": CausalLMAdapter,
}


def get_adapter(model: nn.Module) -> ArchitectureAdapter:
    """Return an adapter matched to the model's architecture, or GenericAdapter."""
    model_type: str = getattr(getattr(model, "config", None), "model_type", "")
    adapter_cls = _REGISTRY.get(model_type, GenericAdapter)
    return adapter_cls()
