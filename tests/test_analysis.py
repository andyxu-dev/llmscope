"""Tests for KVCacheAnalyzer, MemoryProfiler, and tracer.dashboard()."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from llmscope.analysis.kv_cache import KVCacheAnalyzer
from llmscope.analysis.memory import MemoryProfiler
from llmscope.core.events import KVCacheSnapshot, LayerKVStats

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_layer_stats(
    layer_idx: int,
    tokens: int,
    k_max: float = 1.0,
    k_std: float = 0.5,
) -> LayerKVStats:
    shape = (1, 2, tokens, 32)
    nbytes = 1 * 2 * tokens * 32 * 2  # float16
    return LayerKVStats(
        layer_idx=layer_idx,
        k_shape=shape,
        v_shape=shape,
        k_dtype="float16",
        v_dtype="float16",
        k_bytes=nbytes,
        v_bytes=nbytes,
        k_min=-k_max,
        k_max=k_max,
        k_mean=0.0,
        k_std=k_std,
        v_min=-1.0,
        v_max=1.0,
        v_mean=0.0,
        v_std=0.5,
    )


def _make_snapshot(step: int, tokens: int, **layer_kwargs: float) -> KVCacheSnapshot:
    layers = [_make_layer_stats(i, tokens, **layer_kwargs) for i in range(2)]
    total = sum(layer.k_bytes + layer.v_bytes for layer in layers)
    return KVCacheSnapshot(
        session_id="test",
        timestamp=datetime.now(tz=timezone.utc),
        step_index=step,
        per_layer=layers,
        total_bytes=total,
        total_tokens=tokens,
    )


# ── KVCacheAnalyzer ───────────────────────────────────────────────────────────

def test_kv_analyzer_empty_snapshots() -> None:
    result = KVCacheAnalyzer().analyze([])
    assert result.growth_curve == []
    assert result.latest_per_layer == []
    assert result.outlier_risk == "unknown"


def test_kv_analyzer_growth_curve() -> None:
    snapshots = [_make_snapshot(i, tokens=5 + i) for i in range(4)]
    result = KVCacheAnalyzer().analyze(snapshots)

    assert len(result.growth_curve) == 4
    for i, pt in enumerate(result.growth_curve):
        assert pt.step_index == i
        assert pt.total_tokens == 5 + i


def test_kv_analyzer_latest_per_layer() -> None:
    snapshots = [_make_snapshot(0, 5), _make_snapshot(1, 10)]
    result = KVCacheAnalyzer().analyze(snapshots)

    # latest snapshot has tokens=10
    assert len(result.latest_per_layer) == 2
    for lb in result.latest_per_layer:
        assert lb.total_bytes > 0


def test_kv_analyzer_outlier_risk_high() -> None:
    # k_max=20, k_std=1 → ratio=20 > 10 → high
    snap = _make_snapshot(0, 5, k_max=20.0, k_std=1.0)
    result = KVCacheAnalyzer().analyze([snap])
    assert result.outlier_risk == "high"


def test_kv_analyzer_outlier_risk_low() -> None:
    # k_max=2, k_std=1 → ratio=2 ≤ 10 → low
    snap = _make_snapshot(0, 5, k_max=2.0, k_std=1.0)
    result = KVCacheAnalyzer().analyze([snap])
    assert result.outlier_risk == "low"


def test_kv_analyzer_outlier_risk_unknown_zero_std() -> None:
    snap = _make_snapshot(0, 5, k_max=1.0, k_std=0.0)
    result = KVCacheAnalyzer().analyze([snap])
    assert result.outlier_risk == "unknown"


def test_kv_analyzer_outlier_risk_unknown_missing_stats() -> None:
    snap = SimpleNamespace(
        step_index=0,
        total_tokens=5,
        total_bytes=2,
        per_layer=[SimpleNamespace(layer_idx=0, k_bytes=1, v_bytes=1)],
    )

    result = KVCacheAnalyzer().analyze([snap])  # type: ignore[list-item]

    assert result.outlier_risk == "unknown"


# ── MemoryProfiler ────────────────────────────────────────────────────────────

class _TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)  # 4*4 float32 = 64 bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_memory_profiler_weights_bytes() -> None:
    model = _TinyLinear()
    profiler = MemoryProfiler()
    result = profiler.profile(model)

    expected = 4 * 4 * 4  # 4x4 matrix, float32 (4 bytes each)
    assert result.weights_bytes == expected


def test_memory_profiler_no_snapshot_kv_zero() -> None:
    model = _TinyLinear()
    result = MemoryProfiler().profile(model, latest_snapshot=None)
    assert result.kv_bytes == 0


def test_memory_profiler_kv_from_snapshot() -> None:
    model = _TinyLinear()
    snap = _make_snapshot(0, 5)
    result = MemoryProfiler().profile(model, latest_snapshot=snap)
    assert result.kv_bytes == snap.total_bytes


def test_memory_profiler_cpu_activations_none() -> None:
    model = _TinyLinear()
    result = MemoryProfiler().profile(model)
    # On CPU the allocator is not tracked
    if not torch.cuda.is_available():
        assert result.activations_bytes is None
        assert result.on_gpu is False


def test_memory_profiler_cuda_branch_with_mocked_allocator() -> None:
    model = _TinyLinear()
    snap = _make_snapshot(0, 5)
    allocated = model.fc.weight.nelement() * model.fc.weight.element_size()
    allocated += snap.total_bytes + 1234

    with patch("llmscope.analysis.memory.torch.cuda.is_available", return_value=True):
        with patch(
            "llmscope.analysis.memory.torch.cuda.memory_allocated",
            return_value=allocated,
        ):
            result = MemoryProfiler().profile(model, latest_snapshot=snap)

    assert result.on_gpu is True
    assert result.allocated_bytes == allocated
    assert result.activations_bytes == 1234


def test_memory_profiler_mb_helpers() -> None:
    model = _TinyLinear()
    result = MemoryProfiler().profile(model)
    assert result.weights_mb() == result.weights_bytes / 1e6
    if result.activations_bytes is None:
        assert result.activations_mb() is None
    else:
        assert result.activations_mb() == result.activations_bytes / 1e6


# ── tracer.dashboard() ────────────────────────────────────────────────────────


def test_dashboard_load_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    from llmscope.dashboard.app import _load_jsonl

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"event_type": "kv_cache_snapshot", "total_bytes": 10}),
                "{not json",
                "[]",
                json.dumps({"event_type": "memory_event", "breakdown": {}}),
                json.dumps({"event_type": "unknown"}),
            ]
        ),
        encoding="utf-8",
    )

    kv_events, mem_events = _load_jsonl(trace_path)

    assert len(kv_events) == 1
    assert len(mem_events) == 1


def test_dashboard_helpers_tolerate_missing_fields() -> None:
    from llmscope.dashboard.app import _layer_breakdown_series, _memory_values

    layers, k_mb, v_mb = _layer_breakdown_series(
        [{"layer_idx": 0, "k_bytes": "bad"}, {"v_bytes": 2_000_000}, []]
    )
    allocated, weights_mb, kv_mb, act_mb, on_gpu = _memory_values(
        {"allocated_bytes": None, "breakdown": None}
    )

    assert layers == [0, 1]
    assert k_mb == [0.0, 0.0]
    assert v_mb == [0.0, 2.0]
    assert allocated == 0.0
    assert weights_mb == 0.0
    assert kv_mb == 0.0
    assert act_mb == 0.0
    assert on_gpu is False


def test_dashboard_no_streamlit_raises() -> None:
    """ImportError with helpful message when streamlit is absent."""
    from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

    from llmscope import Tracer

    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=100,
        n_positions=64,
        n_ctx=64,
    )
    model = GPT2LMHeadModel(cfg)
    tracer = Tracer(model)

    real_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> object:
        if name == "streamlit":
            raise ImportError("No module named streamlit")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="streamlit"):
            tracer.dashboard()


def test_dashboard_launches_popen() -> None:
    """dashboard() writes a JSONL file and calls subprocess.Popen with streamlit run."""
    from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

    from llmscope import Tracer

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

    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=3, use_cache=True)

    fake_proc = MagicMock()
    with patch.dict("sys.modules", {"streamlit": MagicMock()}):
        with patch(
            "llmscope.core.tracer.subprocess.Popen",
            return_value=fake_proc,
        ) as mock_popen:
            proc = tracer.dashboard(port=9999)

    assert proc is fake_proc
    call_args = mock_popen.call_args
    cmd = call_args[0][0]
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert "9999" in cmd

    # Verify the JSONL temp file was written with at least one event
    env_passed = call_args[1]["env"]
    trace_path = env_passed.get("LLMSCOPE_TRACE_PATH")
    assert trace_path is not None
    assert Path(trace_path).exists()
    lines = Path(trace_path).read_text().strip().splitlines()
    assert len(lines) > 0
    obj = json.loads(lines[0])
    assert "event_type" in obj
