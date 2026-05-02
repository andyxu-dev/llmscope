"""Unit tests for Tracer state machine (no generate, no GPU required)."""

from __future__ import annotations

from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

from llmscope import Config, Tracer


def _tiny_gpt2() -> GPT2LMHeadModel:
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=100, n_positions=64, n_ctx=64)
    return GPT2LMHeadModel(cfg)


def test_tracer_starts_inactive() -> None:
    model = _tiny_gpt2()
    tracer = Tracer(model)
    assert not tracer.is_active
    assert tracer.kv_snapshots == []
    assert tracer.memory_events == []


def test_tracer_start_stop() -> None:
    model = _tiny_gpt2()
    tracer = Tracer(model)

    tracer.start()
    assert tracer.is_active
    assert tracer._session_id is not None

    tracer.stop()
    assert not tracer.is_active


def test_tracer_context_manager() -> None:
    model = _tiny_gpt2()
    with Tracer(model) as tracer:
        assert tracer.is_active
    assert not tracer.is_active


def test_tracer_start_idempotent() -> None:
    model = _tiny_gpt2()
    tracer = Tracer(model)
    tracer.start()
    sid = tracer._session_id
    tracer.start()  # must not create a new session or add duplicate hooks
    assert tracer._session_id == sid
    assert tracer.is_active
    tracer.stop()


def test_tracer_stop_idempotent() -> None:
    model = _tiny_gpt2()
    tracer = Tracer(model)
    tracer.stop()  # calling stop before start must not raise
    assert not tracer.is_active


def test_tracer_custom_config() -> None:
    model = _tiny_gpt2()
    cfg = Config(sample_every_n_steps=2, ring_buffer_size=50)
    tracer = Tracer(model, config=cfg)
    assert tracer._config.sample_every_n_steps == 2
    assert tracer._config.ring_buffer_size == 50


def test_tracer_summary_before_capture() -> None:
    model = _tiny_gpt2()
    tracer = Tracer(model)
    out = tracer.summary()
    assert "not started" in out
    assert "No KV snapshots" in out
