"""Tests for llmscope event schemas and JSON round-trip serialization."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llmscope.core.events import (
    AttentionPattern,
    ForwardPassEvent,
    InferenceSession,
    KVCacheSnapshot,
    LayerKVStats,
    MemoryEvent,
    ModelMetadata,
    QuantizationWhatIf,
    trace_event_adapter,
)

_TS = datetime.now(tz=timezone.utc)
_SID = "session-test"


def _make_model_metadata() -> ModelMetadata:
    return ModelMetadata(
        session_id=_SID,
        timestamp=_TS,
        model_name="gpt-test",
        architecture="decoder-only",
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        num_kv_heads=12,
        dtype="float16",
        device="cpu",
    )


def _make_layer_stats() -> LayerKVStats:
    return LayerKVStats(
        layer_idx=0,
        k_shape=(1, 12, 10, 64),
        v_shape=(1, 12, 10, 64),
        k_dtype="float16",
        v_dtype="float16",
        k_bytes=15360,
        v_bytes=15360,
        k_min=-0.1,
        k_max=0.1,
        k_mean=0.0,
        k_std=0.01,
        v_min=-0.1,
        v_max=0.1,
        v_mean=0.0,
        v_std=0.01,
    )


def test_event_roundtrip() -> None:
    meta = _make_model_metadata()

    events = [
        meta,
        InferenceSession(
            session_id=_SID,
            timestamp=_TS,
            model_metadata=meta,
            prompt_length=10,
            generation_config={"max_new_tokens": 50},
        ),
        ForwardPassEvent(
            session_id=_SID,
            timestamp=_TS,
            step_index=0,
            input_token_count=10,
            total_token_count=10,
        ),
        KVCacheSnapshot(
            session_id=_SID,
            timestamp=_TS,
            step_index=0,
            per_layer=[_make_layer_stats()],
            total_bytes=30720,
            total_tokens=10,
        ),
        AttentionPattern(
            session_id=_SID,
            timestamp=_TS,
            layer_idx=0,
            head_idx=0,
            pattern_type="full",
            sparsity_ratio=0.0,
            top_k_attended_indices=[0, 1, 2],
            attention_weights_summary=[[1.0, 0.0], [0.0, 1.0]],
        ),
        MemoryEvent(
            session_id=_SID,
            timestamp=_TS,
            allocated_bytes=1024,
            reserved_bytes=2048,
            peak_allocated_bytes=4096,
            breakdown={"weights": 512, "kv_cache": 512, "activations": 0},
        ),
        QuantizationWhatIf(
            session_id=_SID,
            timestamp=_TS,
            target_bits=8,
            target_components=["kv_cache", "weights"],
            estimated_savings_bytes=102400,
            estimated_quality_impact="minor",
        ),
    ]

    for event in events:
        jsonl = event.to_jsonl()
        assert jsonl.endswith("\n")
        data = json.loads(jsonl)
        assert data["event_type"] == event.event_type
        assert data["session_id"] == _SID
        # Verify it round-trips through the discriminated union adapter
        reparsed = trace_event_adapter.validate_python(data)
        assert type(reparsed) is type(event)


def test_layer_kv_stats_validation() -> None:
    with pytest.raises(ValidationError) as exc:
        LayerKVStats(
            layer_idx=-1,
            k_shape=(1, 0, 10),
            v_shape=(1, 12, 10),
            k_dtype="float16",
            v_dtype="float16",
            k_bytes=-1,
            v_bytes=1024,
            k_min=0.0,
            k_max=0.0,
            k_mean=0.0,
            k_std=0.0,
            v_min=0.0,
            v_max=0.0,
            v_mean=0.0,
            v_std=0.0,
        )
    msg = str(exc.value)
    assert "layer_idx" in msg or "shapes must be tuples" in msg


def test_attention_pattern_validation() -> None:
    with pytest.raises(ValidationError) as exc:
        AttentionPattern(
            session_id=_SID,
            timestamp=_TS,
            layer_idx=0,
            head_idx=0,
            pattern_type="sparse",
            sparsity_ratio=1.5,
            top_k_attended_indices=[-1],
            attention_weights_summary=[[1.0]],
        )
    msg = str(exc.value)
    assert "sparsity_ratio" in msg or "top_k_attended_indices" in msg


def test_memory_event_validation() -> None:
    with pytest.raises(ValidationError) as exc:
        MemoryEvent(
            session_id=_SID,
            timestamp=_TS,
            allocated_bytes=-100,
            reserved_bytes=0,
            peak_allocated_bytes=0,
            breakdown={"weights": -10},
        )
    msg = str(exc.value)
    assert "memory bytes" in msg or "breakdown values" in msg
