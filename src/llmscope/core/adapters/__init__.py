"""Architecture adapters for llmscope hook attachment."""

from .base import ArchitectureAdapter, PastKeyValues, normalize_past_key_values
from .causal_lm import CausalLMAdapter
from .generic import GenericAdapter
from .gpt2 import GPT2Adapter
from .registry import get_adapter

__all__ = [
    "ArchitectureAdapter",
    "PastKeyValues",
    "normalize_past_key_values",
    "CausalLMAdapter",
    "GenericAdapter",
    "GPT2Adapter",
    "get_adapter",
]
