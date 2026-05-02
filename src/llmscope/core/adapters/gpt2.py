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
        cfg = model.config  # type: ignore[union-attr]
        return {
            "architecture": "gpt2",
            "num_layers": int(cfg.n_layer),
            "num_heads": int(cfg.n_head),
            "hidden_dim": int(cfg.n_embd),
            "num_kv_heads": int(cfg.n_head),  # GPT-2 has no GQA
        }
