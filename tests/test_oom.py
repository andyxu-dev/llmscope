"""Tests for conservative OOM headroom estimates."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmscope.analysis import OOMAnalyzer
from llmscope.core.events import KVCacheSnapshot, MemoryEvent

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _snapshots(tokens: list[int], bytes_: list[int]) -> list[KVCacheSnapshot]:
    return [
        KVCacheSnapshot(
            session_id="test",
            timestamp=_TS,
            step_index=step,
            per_layer=[],
            total_bytes=total_bytes,
            total_tokens=total_tokens,
        )
        for step, (total_tokens, total_bytes) in enumerate(zip(tokens, bytes_))
    ]


def _memory(
    allocated: int = 7_000,
    reserved: int = 8_000,
    peak: int = 8_500,
) -> MemoryEvent:
    return MemoryEvent(
        session_id="test",
        timestamp=_TS,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        peak_allocated_bytes=peak,
        breakdown={"weights": 4_000, "kv_cache": 1_000, "activations": 2_000},
    )


def test_basic_linear_growth_estimates_bytes_per_token() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12], [1_000, 1_100, 1_200]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.status == "available"
    assert result.kv_growth_bytes_per_token == 100
    assert result.current_tokens == 12
    assert result.current_kv_bytes == 1_200


def test_sampled_snapshots_estimate_per_token_growth() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.kv_growth_bytes_per_token == 100


def test_irregular_valid_intervals_estimate_per_token_growth() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 13, 20], [1_000, 1_300, 2_000]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.kv_growth_bytes_per_token == 100


def test_median_slope_reduces_outlier_sensitivity() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12, 13], [1_000, 1_100, 1_200, 2_200]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.kv_growth_bytes_per_token == 100


def test_headroom_floor_and_estimated_max_tokens() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 20], [1_000, 1_100, 2_000]),
        memory_events=[_memory(allocated=7_050)],
        device_capacity_bytes=10_000,
    )

    assert result.estimated_headroom_bytes == 2_950
    assert result.estimated_additional_tokens == 29
    assert result.estimated_max_tokens == 49


def test_safety_margin_reduces_headroom() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        safety_margin_bytes=1_000,
    )

    assert result.estimated_headroom_bytes == 2_000
    assert result.estimated_additional_tokens == 20
    assert result.estimated_max_tokens == 40


def test_allocation_exactly_at_effective_limit_reports_zero_headroom() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=9_000)],
        device_capacity_bytes=10_000,
        safety_margin_bytes=1_000,
    )

    assert result.status == "at_or_over_limit"
    assert result.estimated_headroom_bytes == 0
    assert result.estimated_additional_tokens == 0
    assert result.estimated_max_tokens == 20


def test_allocation_above_effective_limit_reports_zero_headroom() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=9_500)],
        device_capacity_bytes=10_000,
        safety_margin_bytes=1_000,
    )

    assert result.status == "at_or_over_limit"
    assert result.estimated_headroom_bytes == 0
    assert result.estimated_additional_tokens == 0


def test_zero_estimated_token_headroom_is_available_when_below_limit() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 20], [1_000, 1_100, 2_000]),
        memory_events=[_memory(allocated=9_950)],
        device_capacity_bytes=10_000,
    )

    assert result.status == "available"
    assert result.estimated_headroom_bytes == 50
    assert result.estimated_additional_tokens == 0
    assert result.estimated_max_tokens == 20


def test_current_reserved_bytes_is_diagnostic_context_not_headroom_input() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000, reserved=9_900)],
        device_capacity_bytes=10_000,
    )

    assert result.current_reserved_bytes == 9_900
    assert result.estimated_headroom_bytes == 3_000


def test_cpu_allocator_telemetry_is_unavailable() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12], [1_000, 1_100, 1_200]),
        memory_events=[_memory(allocated=0, reserved=0, peak=0)],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert "CPU traces" in result.reason
    assert result.estimated_additional_tokens is None


def test_peak_only_allocator_telemetry_is_unavailable() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12], [1_000, 1_100, 1_200]),
        memory_events=[_memory(allocated=0, reserved=0, peak=8_000)],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert result.estimated_additional_tokens is None


def test_missing_memory_events_is_unavailable() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12], [1_000, 1_100, 1_200]),
        memory_events=[],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert result.current_allocated_bytes is None


def test_zero_snapshots_is_unavailable() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=[],
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert result.current_tokens is None
    assert result.kv_growth_bytes_per_token is None


def test_single_snapshot_is_unavailable() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10], [1_000]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert result.current_tokens == 10
    assert result.kv_growth_bytes_per_token is None


def test_duplicate_token_counts_are_not_usable_slopes() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 10, 10], [1_000, 1_100, 1_200]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert result.kv_growth_bytes_per_token is None


def test_decreasing_byte_samples_are_not_usable_slopes() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12], [1_000, 900, 800]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert result.kv_growth_bytes_per_token is None


def test_zero_kv_growth_is_unavailable() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12], [1_000, 1_000, 1_000]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.status == "unavailable"
    assert result.kv_growth_bytes_per_token is None


def test_invalid_decreasing_sample_is_skipped_when_valid_slopes_remain() -> None:
    result = OOMAnalyzer().estimate(
        snapshots=_snapshots([10, 11, 12, 13], [1_000, 900, 1_000, 1_100]),
        memory_events=[_memory()],
        device_capacity_bytes=10_000,
    )

    assert result.status == "available"
    assert result.kv_growth_bytes_per_token == 100


def test_invalid_device_capacity_raises() -> None:
    with pytest.raises(ValueError, match="device_capacity_bytes"):
        OOMAnalyzer().estimate(
            snapshots=_snapshots([10, 11], [1_000, 1_100]),
            memory_events=[_memory()],
            device_capacity_bytes=0,
        )


@pytest.mark.parametrize("safety_margin_bytes", [-1, 10_000, 12_000])
def test_invalid_safety_margin_raises(safety_margin_bytes: int) -> None:
    with pytest.raises(ValueError, match="safety_margin_bytes"):
        OOMAnalyzer().estimate(
            snapshots=_snapshots([10, 11], [1_000, 1_100]),
            memory_events=[_memory()],
            device_capacity_bytes=10_000,
            safety_margin_bytes=safety_margin_bytes,
        )
