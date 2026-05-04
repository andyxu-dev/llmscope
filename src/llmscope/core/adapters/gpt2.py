"""Architecture adapter for GPT-2 family models."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import ArchitectureAdapter, PastKeyValues, normalize_past_key_values


class GPT2Adapter(ArchitectureAdapter):
    """Adapter for GPT2LMHeadModel / GPT2Model."""

    def get_hook_module(self, model: nn.Module) -> nn.Module:
        # model.transformer is the GPT2Model backbone
        return model.transformer  # type: ignore[return-value]

    def extract_past_key_values(self, output: Any) -> PastKeyValues | None:
        raw = getattr(output, "past_key_values", None)
        return normalize_past_key_values(raw)

    def get_model_info(self, model: nn.Module) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        return {
            "architecture": "gpt2",
            "num_layers": int(getattr(cfg, "n_layer", 0)),
            "num_heads": int(getattr(cfg, "n_head", 0)),
            "hidden_dim": int(getattr(cfg, "n_embd", 0)),
            "num_kv_heads": int(getattr(cfg, "n_head", 0)),
        }
