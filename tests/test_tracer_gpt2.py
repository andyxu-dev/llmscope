"""End-to-end smoke test: instrument GPT-2, generate tokens, assert KV events."""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

from llmscope import Tracer

N_LAYER = 2
N_HEAD = 2
N_EMBD = 64
VOCAB = 100
MAX_NEW_TOKENS = 10
PROMPT_LEN = 5


def _tiny_gpt2() -> GPT2LMHeadModel:
    cfg = GPT2Config(
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        vocab_size=VOCAB,
        n_positions=128,
        n_ctx=128,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def test_kv_snapshots_captured() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (1, PROMPT_LEN))

    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

    snapshots = tracer.kv_snapshots
    assert len(snapshots) > 0, "No KV snapshots were captured"


def test_kv_token_count_grows() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (1, PROMPT_LEN))

    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

    token_counts = [s.total_tokens for s in tracer.kv_snapshots]
    assert all(
        b >= a for a, b in zip(token_counts, token_counts[1:])
    ), f"Token counts not monotonically non-decreasing: {token_counts}"


def test_kv_snapshot_has_correct_layer_count() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (1, PROMPT_LEN))

    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

    for snapshot in tracer.kv_snapshots:
        assert len(snapshot.per_layer) == N_LAYER, (
            f"Expected {N_LAYER} layers, got {len(snapshot.per_layer)}"
        )


def test_kv_snapshot_step_index_monotonic() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (1, PROMPT_LEN))

    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

    steps = [s.step_index for s in tracer.kv_snapshots]
    assert steps == sorted(steps), f"step_index not monotonic: {steps}"


def test_sample_every_n_steps() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (1, PROMPT_LEN))

    from llmscope import Config

    with Tracer(model, config=Config(sample_every_n_steps=2)) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

    full_tracer = Tracer(model)
    with full_tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

    assert len(tracer.kv_snapshots) < len(full_tracer.kv_snapshots)


def test_summary_after_capture() -> None:
    model = _tiny_gpt2()
    input_ids = torch.randint(0, VOCAB, (1, PROMPT_LEN))

    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

    out = tracer.summary()
    assert "KV snapshots" in out
    assert "Weights" in out
