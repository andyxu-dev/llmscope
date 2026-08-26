"""Tests for the what-if KV cache memory estimator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmscope.analysis.what_if import (
    _BYTES_PER_ELEMENT,
    WhatIfEstimator,
    estimate_kv_memory,
)
from llmscope.core.events import KVCacheSnapshot, LayerKVStats

# ── shared fixtures ────────────────────────────────────────────────────────────

def _estimator(
    num_layers: int = 4, num_heads: int = 8, head_dim: int = 64
) -> WhatIfEstimator:
    return WhatIfEstimator(
        num_layers=num_layers, num_heads=num_heads, head_dim=head_dim
    )


def _make_snapshot(
    num_layers: int = 4, num_heads: int = 8, head_dim: int = 64, seq: int = 10
) -> KVCacheSnapshot:
    shape = (1, num_heads, seq, head_dim)
    nbytes = num_heads * seq * head_dim * 2  # fp16
    layers = [
        LayerKVStats(
            layer_idx=i,
            k_shape=shape,
            v_shape=shape,
            k_dtype="float16",
            v_dtype="float16",
            k_bytes=nbytes,
            v_bytes=nbytes,
            k_min=-1.0, k_max=1.0, k_mean=0.0, k_std=0.5,
            v_min=-1.0, v_max=1.0, v_mean=0.0, v_std=0.5,
        )
        for i in range(num_layers)
    ]
    total = sum(layer.k_bytes + layer.v_bytes for layer in layers)
    return KVCacheSnapshot(
        session_id="test",
        timestamp=datetime.now(tz=timezone.utc),
        step_index=0,
        per_layer=layers,
        total_bytes=total,
        total_tokens=seq,
    )


# ── dtype byte sizes ───────────────────────────────────────────────────────────

def test_bytes_per_element_table() -> None:
    assert _BYTES_PER_ELEMENT["fp32"] == 4.0
    assert _BYTES_PER_ELEMENT["fp16"] == 2.0
    assert _BYTES_PER_ELEMENT["bf16"] == 2.0
    assert _BYTES_PER_ELEMENT["int8"] == 1.0
    assert _BYTES_PER_ELEMENT["int4"] == 0.5


# ── fp16 estimate ──────────────────────────────────────────────────────────────

def test_fp16_estimate_correct() -> None:
    # KV = 2 × layers × heads × head_dim × seq × batch × bytes
    # 2 × 4 × 8 × 64 × 100 × 1 × 2 = 819 200
    est = _estimator().estimate(sequence_length=100, dtype="fp16")
    expected = 2 * 4 * 8 * 64 * 100 * 1 * 2
    assert est.total_bytes == expected


def test_fp16_baseline_is_none() -> None:
    est = _estimator().estimate(sequence_length=100, dtype="fp16")
    assert est.fp16_baseline_bytes is None
    assert est.savings_bytes is None
    assert est.savings_mb is None
    assert est.compression_ratio is None


# ── int8 estimate ──────────────────────────────────────────────────────────────

def test_int8_estimate_half_of_fp16() -> None:
    fp16 = _estimator().estimate(sequence_length=100, dtype="fp16")
    int8 = _estimator().estimate(sequence_length=100, dtype="int8")
    assert int8.total_bytes == fp16.total_bytes // 2


def test_int8_savings_equals_half_fp16() -> None:
    est = _estimator().estimate(sequence_length=100, dtype="int8")
    assert est.savings_bytes == est.fp16_baseline_bytes // 2
    assert est.compression_ratio == pytest.approx(2.0)


# ── int4 estimate ──────────────────────────────────────────────────────────────

def test_int4_estimate_quarter_of_fp16() -> None:
    fp16 = _estimator().estimate(sequence_length=100, dtype="fp16")
    int4 = _estimator().estimate(sequence_length=100, dtype="int4")
    assert int4.total_bytes == fp16.total_bytes // 4


def test_int4_compression_ratio_four() -> None:
    est = _estimator().estimate(sequence_length=100, dtype="int4")
    assert est.compression_ratio == pytest.approx(4.0)


def test_int4_savings_three_quarters_fp16() -> None:
    est = _estimator().estimate(sequence_length=100, dtype="int4")
    assert est.savings_bytes == pytest.approx(est.fp16_baseline_bytes * 3 / 4)


# ── fp32 estimate ──────────────────────────────────────────────────────────────

def test_fp32_costs_more_than_fp16() -> None:
    est = _estimator().estimate(sequence_length=100, dtype="fp32")
    assert est.savings_bytes is not None and est.savings_bytes < 0
    assert est.compression_ratio == pytest.approx(0.5)


# ── bf16 same size as fp16 ─────────────────────────────────────────────────────

def test_bf16_same_bytes_as_fp16() -> None:
    fp16 = _estimator().estimate(sequence_length=100, dtype="fp16")
    bf16 = _estimator().estimate(sequence_length=100, dtype="bf16")
    # bf16 and fp16 both use 2 bytes/element; bf16 baseline should equal bf16 total
    assert bf16.total_bytes == fp16.total_bytes
    assert bf16.fp16_baseline_bytes == bf16.total_bytes
    assert bf16.savings_bytes == 0


# ── scaling with sequence length ───────────────────────────────────────────────

def test_doubling_seq_len_doubles_bytes() -> None:
    est_100 = _estimator().estimate(sequence_length=100)
    est_200 = _estimator().estimate(sequence_length=200)
    assert est_200.total_bytes == 2 * est_100.total_bytes


def test_sequence_length_stored_correctly() -> None:
    est = _estimator().estimate(sequence_length=512)
    assert est.sequence_length == 512


# ── scaling with batch size ────────────────────────────────────────────────────

def test_doubling_batch_doubles_bytes() -> None:
    est_b1 = _estimator().estimate(sequence_length=100, batch_size=1)
    est_b2 = _estimator().estimate(sequence_length=100, batch_size=2)
    assert est_b2.total_bytes == 2 * est_b1.total_bytes


def test_batch_size_stored_correctly() -> None:
    est = _estimator().estimate(sequence_length=100, batch_size=4)
    assert est.batch_size == 4


# ── standalone estimate_kv_memory() ───────────────────────────────────────────

def test_standalone_function_matches_class() -> None:
    via_class = _estimator(num_layers=2, num_heads=4, head_dim=32).estimate(
        sequence_length=50, batch_size=2, dtype="int8"
    )
    via_func = estimate_kv_memory(
        num_layers=2, num_heads=4, head_dim=32,
        sequence_length=50, batch_size=2, dtype="int8",
    )
    assert via_func.total_bytes == via_class.total_bytes


# ── WhatIfEstimator.from_snapshot() ───────────────────────────────────────────

def test_from_snapshot_derives_dims_correctly() -> None:
    snap = _make_snapshot(num_layers=4, num_heads=8, head_dim=64, seq=10)
    est = WhatIfEstimator.from_snapshot(snap)
    assert est.num_layers == 4
    assert est.num_heads == 8
    assert est.head_dim == 64


def test_from_snapshot_estimate_matches_manual() -> None:
    snap = _make_snapshot(num_layers=2, num_heads=4, head_dim=32, seq=20)
    est = WhatIfEstimator.from_snapshot(snap)
    result = est.estimate(sequence_length=20, dtype="fp16")
    expected = 2 * 2 * 4 * 32 * 20 * 1 * 2
    assert result.total_bytes == expected


# ── error / safety handling ────────────────────────────────────────────────────

def test_unknown_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        _estimator().estimate(sequence_length=100, dtype="float7")


def test_zero_sequence_length_raises() -> None:
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        _estimator().estimate(sequence_length=0)


def test_negative_sequence_length_raises() -> None:
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        _estimator().estimate(sequence_length=-1)


def test_zero_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        _estimator().estimate(sequence_length=100, batch_size=0)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"num_layers": 0}, "num_layers"),
        ({"num_heads": 0}, "num_heads"),
        ({"head_dim": 0}, "head_dim"),
    ],
)
def test_invalid_model_dimension_raises(
    kwargs: dict[str, int],
    field: str,
) -> None:
    params = {"num_layers": 4, "num_heads": 8, "head_dim": 64}
    params.update(kwargs)

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        WhatIfEstimator(**params)


def test_missing_sequence_length_raises_value_error() -> None:
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        _estimator().estimate(sequence_length=None)  # type: ignore[arg-type]


def test_missing_dtype_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        _estimator().estimate(sequence_length=100, dtype=None)  # type: ignore[arg-type]


def test_from_snapshot_empty_per_layer_raises() -> None:
    from types import SimpleNamespace
    snap = SimpleNamespace(per_layer=[])
    with pytest.raises(ValueError, match="per_layer is empty"):
        WhatIfEstimator.from_snapshot(snap)


def test_from_snapshot_bad_k_shape_raises() -> None:
    from types import SimpleNamespace
    snap = SimpleNamespace(
        per_layer=[SimpleNamespace(k_shape=(1, 8))]  # too few dims
    )
    with pytest.raises(ValueError, match="k_shape"):
        WhatIfEstimator.from_snapshot(snap)


def test_dtype_case_insensitive() -> None:
    est_lower = _estimator().estimate(sequence_length=100, dtype="fp16")
    est_upper = _estimator().estimate(sequence_length=100, dtype="FP16")
    assert est_lower.total_bytes == est_upper.total_bytes


def test_dtype_aliases_normalize_to_canonical_names() -> None:
    est = _estimator().estimate(sequence_length=100, dtype="float16")

    assert est.dtype == "fp16"


# ── mb helper ─────────────────────────────────────────────────────────────────

def test_total_mb_helper() -> None:
    est = _estimator().estimate(sequence_length=100, dtype="fp16")
    assert est.total_mb == est.total_bytes / 1e6
