"""Adapter for Llama/Qwen/Mistral-style causal LM models.

These models share a common structure: the transformer backbone is at
model.model, and the KV cache is in past_key_values on the backbone output.

Experimental: tested with fake config objects only, not with real checkpoints.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import ArchitectureAdapter, PastKeyValues, normalize_past_key_values


class CausalLMAdapter(ArchitectureAdapter):
    """Adapter for HuggingFace CausalLM models whose backbone lives at model.model.

    Covers Llama, Qwen2, Mistral, and similar architectures.
    Falls back to the top-level model if model.model is absent.
    """

    def get_hook_module(self, model: nn.Module) -> nn.Module:
        backbone = getattr(model, "model", None)
        if backbone is None or not isinstance(backbone, nn.Module):
            # Graceful fallback: hook the wrapper itself.
            return model
        return backbone

    def extract_past_key_values(self, output: Any) -> PastKeyValues | None:
        raw = getattr(output, "past_key_values", None)
        return normalize_past_key_values(raw)

    def get_model_info(self, model: nn.Module) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return {"architecture": "unknown"}
        return {
            "architecture": getattr(cfg, "model_type", "unknown"),
            "num_layers": int(getattr(cfg, "num_hidden_layers", 0)),
            "num_heads": int(getattr(cfg, "num_attention_heads", 0)),
            "hidden_dim": int(getattr(cfg, "hidden_size", 0)),
            # Llama/Mistral may use GQA; fall back to num_attention_heads.
            "num_kv_heads": int(
                getattr(
                    cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0)
                )
            ),
        }
