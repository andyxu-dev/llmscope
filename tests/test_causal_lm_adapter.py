"""Tests for CausalLMAdapter and updated adapter registry.

All tests use lightweight fake nn.Module objects — no real model downloads.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from llmscope.core.adapters import (
    CausalLMAdapter,
    GPT2Adapter,
    GenericAdapter,
    get_adapter,
)


# ── minimal fake model helpers ────────────────────────────────────────────────

class _FakeBackbone(nn.Module):
    """Minimal backbone that lives at model.model."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> Any:
        return SimpleNamespace(past_key_values=None)


class _FakeCausalLM(nn.Module):
    """Wraps a backbone at self.model, like LlamaForCausalLM."""

    def __init__(self, model_type: str = "llama") -> None:
        super().__init__()
        self.model = _FakeBackbone()
        self.config = SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=4,
            num_attention_heads=8,
            hidden_size=256,
            num_key_value_heads=4,
        )

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)


class _FakeGPT2(nn.Module):
    """Minimal GPT-2-style model with a transformer backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(4, 4, bias=False)
        self.config = SimpleNamespace(model_type="gpt2")

    def forward(self, x: torch.Tensor) -> Any:
        return SimpleNamespace(past_key_values=None)


class _FakeUnknown(nn.Module):
    """Model with an unrecognised model_type."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="bert")

    def forward(self, x: torch.Tensor) -> Any:
        return SimpleNamespace(past_key_values=None)


class _FakeNoModel(nn.Module):
    """CausalLM-style model that is missing the .model attribute entirely."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama")

    def forward(self, x: torch.Tensor) -> Any:
        return SimpleNamespace(past_key_values=None)


# ── registry dispatch ─────────────────────────────────────────────────────────

def test_registry_gpt2_returns_gpt2_adapter() -> None:
    model = _FakeGPT2()
    adapter = get_adapter(model)
    assert isinstance(adapter, GPT2Adapter)


def test_registry_llama_returns_causal_lm_adapter() -> None:
    model = _FakeCausalLM("llama")
    adapter = get_adapter(model)
    assert isinstance(adapter, CausalLMAdapter)


def test_registry_qwen2_returns_causal_lm_adapter() -> None:
    model = _FakeCausalLM("qwen2")
    adapter = get_adapter(model)
    assert isinstance(adapter, CausalLMAdapter)


def test_registry_mistral_returns_causal_lm_adapter() -> None:
    model = _FakeCausalLM("mistral")
    adapter = get_adapter(model)
    assert isinstance(adapter, CausalLMAdapter)


def test_registry_unknown_returns_generic_adapter() -> None:
    model = _FakeUnknown()
    adapter = get_adapter(model)
    assert isinstance(adapter, GenericAdapter)


def test_registry_no_config_returns_generic_adapter() -> None:
    model = nn.Linear(4, 4)  # no .config attribute
    adapter = get_adapter(model)
    assert isinstance(adapter, GenericAdapter)


# ── CausalLMAdapter.get_hook_module() ─────────────────────────────────────────

def test_causal_lm_hook_module_is_model_dot_model() -> None:
    model = _FakeCausalLM("llama")
    adapter = CausalLMAdapter()
    hook_module = adapter.get_hook_module(model)
    assert hook_module is model.model


def test_causal_lm_hook_module_falls_back_when_model_missing() -> None:
    model = _FakeNoModel()
    adapter = CausalLMAdapter()
    # Should return the top-level model, not raise
    hook_module = adapter.get_hook_module(model)
    assert hook_module is model


def test_causal_lm_hook_module_falls_back_when_model_is_not_nn_module() -> None:
    """model.model exists but is not an nn.Module (e.g., a plain string).

    PyTorch's __setattr__ prevents assigning non-Modules to registered
    submodule slots, so we bypass it with object.__setattr__ to put the
    value directly into __dict__, where Python finds it before __getattr__.
    """
    model = _FakeCausalLM("llama")
    object.__setattr__(model, "model", "not a module")
    adapter = CausalLMAdapter()
    hook_module = adapter.get_hook_module(model)
    assert hook_module is model


# ── CausalLMAdapter.extract_past_key_values() ────────────────────────────────

def test_causal_lm_extract_returns_none_when_absent() -> None:
    adapter = CausalLMAdapter()
    output = SimpleNamespace()  # no past_key_values attribute
    assert adapter.extract_past_key_values(output) is None


def test_causal_lm_extract_returns_none_when_explicitly_none() -> None:
    adapter = CausalLMAdapter()
    output = SimpleNamespace(past_key_values=None)
    assert adapter.extract_past_key_values(output) is None


def test_causal_lm_extract_tuple_of_tuples() -> None:
    """Handles the legacy tuple-of-tuples KV format."""
    adapter = CausalLMAdapter()
    k = torch.zeros(1, 4, 10, 32)
    v = torch.zeros(1, 4, 10, 32)
    pkv = ((k, v), (k, v))  # 2 layers
    output = SimpleNamespace(past_key_values=pkv)
    result = adapter.extract_past_key_values(output)
    assert result is not None
    assert len(result) == 2
    assert result[0][0] is k


# ── CausalLMAdapter.get_model_info() ─────────────────────────────────────────

def test_causal_lm_model_info_llama() -> None:
    model = _FakeCausalLM("llama")
    adapter = CausalLMAdapter()
    info = adapter.get_model_info(model)
    assert info["architecture"] == "llama"
    assert info["num_layers"] == 4
    assert info["num_heads"] == 8
    assert info["hidden_dim"] == 256
    assert info["num_kv_heads"] == 4  # GQA


def test_causal_lm_model_info_no_config() -> None:
    model = nn.Linear(4, 4)  # no .config
    adapter = CausalLMAdapter()
    info = adapter.get_model_info(model)
    assert info["architecture"] == "unknown"


def test_causal_lm_model_info_missing_kv_heads_falls_back_to_num_heads() -> None:
    """If num_key_value_heads is absent, fall back to num_attention_heads."""
    model = _FakeCausalLM("mistral")
    del model.config.num_key_value_heads  # simulate absent attribute
    adapter = CausalLMAdapter()
    info = adapter.get_model_info(model)
    assert info["num_kv_heads"] == info["num_heads"]
