"""Tests for HookManager attach/detach lifecycle."""

from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

from llmscope.core.adapters.registry import get_adapter
from llmscope.core.hooks import HookManager


def _tiny_gpt2() -> GPT2LMHeadModel:
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=100, n_positions=64, n_ctx=64)
    return GPT2LMHeadModel(cfg)


def test_hook_manager_starts_detached() -> None:
    model = _tiny_gpt2()
    adapter = get_adapter(model)
    manager = HookManager(model, adapter, lambda pkv, step: None)
    assert not manager.is_attached


def test_hook_manager_attach_detach() -> None:
    model = _tiny_gpt2()
    adapter = get_adapter(model)
    manager = HookManager(model, adapter, lambda pkv, step: None)

    manager.attach()
    assert manager.is_attached

    manager.detach()
    assert not manager.is_attached


def test_hook_manager_attach_idempotent() -> None:
    model = _tiny_gpt2()
    adapter = get_adapter(model)
    manager = HookManager(model, adapter, lambda pkv, step: None)

    manager.attach()
    manager.attach()  # second call should not add duplicate hooks
    assert manager.is_attached
    # Only one handle should be registered
    assert len(manager._handles) == 1

    manager.detach()
    assert not manager.is_attached
