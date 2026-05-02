"""Fallback adapter for unknown / unsupported architectures."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import ArchitectureAdapter, PastKeyValues, normalize_past_key_values


class GenericAdapter(ArchitectureAdapter):
    """Best-effort adapter: hooks the top-level model and reads past_key_values."""

    def get_hook_module(self, model: nn.Module) -> nn.Module:
        return model

    def extract_past_key_values(self, output: Any) -> PastKeyValues | None:
        raw = getattr(output, "past_key_values", None)
        return normalize_past_key_values(raw)

    def get_model_info(self, model: nn.Module) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return {"architecture": "unknown"}
        return {
            "architecture": getattr(cfg, "model_type", "unknown"),
            "num_layers": int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))),
            "num_heads": int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))),
            "hidden_dim": int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0))),
            "num_kv_heads": int(
                getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 0))
            ),
        }
