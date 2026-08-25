"""Optional integration test for a real Hugging Face Llama checkpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from llmscope import Tracer
from llmscope.core.adapters import CausalLMAdapter, get_adapter

CHECKPOINT = "SimpleStories/SimpleStories-1.25M"
PROMPT_LENGTH = 6
MAX_NEW_TOKENS = 4


def _repo_hf_cache() -> Path:
    return Path(__file__).resolve().parents[1] / ".hf_home" / "hub"


def _forward_hook_count(module: Any) -> int:
    return len(getattr(module, "_forward_hooks", {}))


@pytest.mark.integration
def test_real_llama_checkpoint_traces_grouped_query_kv_cache() -> None:
    if os.environ.get("LLMSCOPE_RUN_HF_INTEGRATION") != "1":
        pytest.skip("set LLMSCOPE_RUN_HF_INTEGRATION=1 to run Hugging Face test")

    from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

    try:
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            cache_dir=_repo_hf_cache(),
        ).eval()
    except OSError as exc:
        pytest.skip(f"could not load {CHECKPOINT}: {exc}")

    cfg = model.config
    adapter = get_adapter(model)
    hook_module = adapter.get_hook_module(model)
    before_hooks = _forward_hook_count(hook_module)

    assert isinstance(adapter, CausalLMAdapter)
    assert hook_module is model.model
    assert cfg.num_key_value_heads < cfg.num_attention_heads

    input_ids = torch.tensor([[1, 10, 20, 30, 40, 50]])
    head_dim = int(
        getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    )

    with torch.no_grad():
        output = model(input_ids, use_cache=True)
    raw_cache = output.past_key_values
    assert type(raw_cache).__name__ == "DynamicCache"
    assert hasattr(raw_cache, "layers")

    first_layer = raw_cache.layers[0]
    assert tuple(first_layer.keys.shape) == (
        1,
        cfg.num_key_value_heads,
        PROMPT_LENGTH,
        head_dim,
    )
    assert tuple(first_layer.values.shape) == tuple(first_layer.keys.shape)

    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )

    assert not tracer.is_active
    assert tracer._hook_manager is None
    assert _forward_hook_count(hook_module) == before_hooks

    snapshots = tracer.kv_snapshots
    assert len(snapshots) == MAX_NEW_TOKENS
    assert [len(snapshot.per_layer) for snapshot in snapshots] == [
        cfg.num_hidden_layers
    ] * MAX_NEW_TOKENS
    assert [snapshot.total_tokens for snapshot in snapshots] == [
        PROMPT_LENGTH + step for step in range(MAX_NEW_TOKENS)
    ]
    bytes_by_step = [snapshot.total_bytes for snapshot in snapshots]
    assert all(
        current > previous
        for previous, current in zip(bytes_by_step, bytes_by_step[1:])
    )

    first = snapshots[0].per_layer[0]
    last = snapshots[-1].per_layer[0]
    assert first.k_shape == (1, cfg.num_key_value_heads, PROMPT_LENGTH, head_dim)
    assert first.v_shape == first.k_shape
    assert last.k_shape == (
        1,
        cfg.num_key_value_heads,
        PROMPT_LENGTH + MAX_NEW_TOKENS - 1,
        head_dim,
    )
    assert last.v_shape == last.k_shape
