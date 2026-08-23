"""Tests for loading saved JSONL traces."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

from llmscope import TraceLoadError, Tracer, TraceSession
from llmscope.analysis.kv_cache import KVCacheAnalyzer
from llmscope.core.events import KVCacheSnapshot, LayerKVStats, MemoryEvent

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SID = "session-test"


def _layer_stats(tokens: int = 5) -> LayerKVStats:
    return LayerKVStats(
        layer_idx=0,
        k_shape=(1, 2, tokens, 32),
        v_shape=(1, 2, tokens, 32),
        k_dtype="float32",
        v_dtype="float32",
        k_bytes=1_280,
        v_bytes=1_280,
        k_min=-0.1,
        k_max=0.2,
        k_mean=0.01,
        k_std=0.03,
        v_min=-0.2,
        v_max=0.3,
        v_mean=0.02,
        v_std=0.04,
    )


def _snapshot(step: int, tokens: int = 5, session_id: str = _SID) -> KVCacheSnapshot:
    layer = _layer_stats(tokens)
    return KVCacheSnapshot(
        session_id=session_id,
        timestamp=_TS,
        step_index=step,
        per_layer=[layer],
        total_bytes=layer.k_bytes + layer.v_bytes,
        total_tokens=tokens,
    )


def _memory_event(kv_bytes: int = 2_560, session_id: str = _SID) -> MemoryEvent:
    return MemoryEvent(
        session_id=session_id,
        timestamp=_TS,
        allocated_bytes=0,
        reserved_bytes=0,
        peak_allocated_bytes=0,
        breakdown={"weights": 10_000, "kv_cache": kv_bytes, "activations": 0},
    )


def _write_events(path: Path, *events: object) -> None:
    lines = [event.to_jsonl() for event in events]
    path.write_text("".join(lines), encoding="utf-8")


def test_trace_session_round_trip_from_tracer_save(tmp_path: Path) -> None:
    torch.manual_seed(42)
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=100,
        n_positions=64,
        n_ctx=64,
    )
    model = GPT2LMHeadModel(cfg).eval()
    input_ids = torch.randint(0, 100, (1, 3))

    trace_path = tmp_path / "trace.jsonl"
    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=3, use_cache=True)
    tracer.save(trace_path)

    trace = TraceSession.load(trace_path)

    assert trace.session_id == tracer.kv_snapshots[0].session_id
    assert trace.kv_snapshots == tracer.kv_snapshots
    assert trace.memory_events == tracer.memory_events
    assert trace.events == [*tracer.kv_snapshots, *tracer.memory_events]
    assert KVCacheAnalyzer().analyze(trace.kv_snapshots).growth_curve


def test_trace_session_restores_typed_events_and_preserves_order(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    first_mem = _memory_event(kv_bytes=1_000)
    first_snapshot = _snapshot(step=0, tokens=5)
    second_snapshot = _snapshot(step=1, tokens=6)
    _write_events(trace_path, first_mem, first_snapshot, second_snapshot)

    trace = TraceSession.load(trace_path)

    assert trace.session_id == _SID
    assert trace.events == [first_mem, first_snapshot, second_snapshot]
    assert trace.kv_snapshots == [first_snapshot, second_snapshot]
    assert trace.memory_events == [first_mem]
    assert isinstance(trace.kv_snapshots[0], KVCacheSnapshot)
    assert isinstance(trace.memory_events[0], MemoryEvent)
    assert trace.kv_snapshots[1].total_tokens == 6
    assert trace.memory_events[0].breakdown["kv_cache"] == 1_000


def test_trace_session_load_empty_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "empty.jsonl"
    trace_path.write_text("", encoding="utf-8")

    trace = TraceSession.load(trace_path)

    assert trace.session_id is None
    assert trace.events == []
    assert trace.kv_snapshots == []
    assert trace.memory_events == []


def test_trace_session_malformed_json_fails(tmp_path: Path) -> None:
    trace_path = tmp_path / "bad.jsonl"
    trace_path.write_text("{not json\n", encoding="utf-8")

    with pytest.raises(TraceLoadError, match="Malformed JSON at line 1"):
        TraceSession.load(trace_path)


def test_trace_session_non_object_json_fails(tmp_path: Path) -> None:
    trace_path = tmp_path / "bad.jsonl"
    trace_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TraceLoadError, match="line 1 must be a JSON object"):
        TraceSession.load(trace_path)


def test_trace_session_unknown_event_type_fails(tmp_path: Path) -> None:
    trace_path = tmp_path / "unknown.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "session_id": _SID,
                "timestamp": _TS.isoformat(),
                "event_type": "mystery_event",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TraceLoadError, match="Unsupported event_type 'mystery_event'"):
        TraceSession.load(trace_path)


def test_trace_session_unsupported_event_type_fails(tmp_path: Path) -> None:
    trace_path = tmp_path / "unsupported.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "session_id": _SID,
                "timestamp": _TS.isoformat(),
                "event_type": "model_metadata",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TraceLoadError, match="Unsupported event_type 'model_metadata'"):
        TraceSession.load(trace_path)


def test_trace_session_inconsistent_session_ids_fail(tmp_path: Path) -> None:
    trace_path = tmp_path / "mixed.jsonl"
    _write_events(
        trace_path,
        _snapshot(step=0, session_id="session-a"),
        _memory_event(session_id="session-b"),
    )

    with pytest.raises(TraceLoadError, match="Inconsistent session_id at line 2"):
        TraceSession.load(trace_path)
