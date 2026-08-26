"""Tests for conservative OOM headroom estimates."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmscope.analysis import OOMAnalyzer
from llmscope.core.events import KVCacheSnapshot, LayerKVStats, MemoryEvent

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


def _precision_snapshots(
    tokens: list[int],
    bytes_: list[int],
    *,
    dtype: str = "float16",
) -> list[KVCacheSnapshot]:
    return [
        KVCacheSnapshot(
            session_id="test",
            timestamp=_TS,
            step_index=step,
            per_layer=[
                LayerKVStats(
                    layer_idx=0,
                    k_shape=(1, 1, total_tokens, 1),
                    v_shape=(1, 1, total_tokens, 1),
                    k_dtype=dtype,
                    v_dtype=dtype,
                    k_bytes=total_bytes // 2,
                    v_bytes=total_bytes - (total_bytes // 2),
                    k_min=-1.0,
                    k_max=1.0,
                    k_mean=0.0,
                    k_std=0.5,
                    v_min=-1.0,
                    v_max=1.0,
                    v_mean=0.0,
                    v_std=0.5,
                )
            ],
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


def test_current_dtype_precision_scenario_matches_baseline_oom_estimate() -> None:
    snapshots = _precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000])
    memory_events = [_memory(allocated=7_000)]
    baseline = OOMAnalyzer().estimate(
        snapshots=snapshots,
        memory_events=memory_events,
        device_capacity_bytes=10_000,
        safety_margin_bytes=1_000,
    )

    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=snapshots,
        memory_events=memory_events,
        device_capacity_bytes=10_000,
        target_dtypes=("fp16",),
        safety_margin_bytes=1_000,
    )[0]

    assert scenario.status == baseline.status
    assert scenario.current_dtype == "fp16"
    assert scenario.target_ratio == 1.0
    assert scenario.projected_current_kv_bytes == baseline.current_kv_bytes
    assert scenario.projected_current_allocated_bytes == (
        baseline.current_allocated_bytes
    )
    assert scenario.projected_kv_growth_bytes_per_token == (
        baseline.kv_growth_bytes_per_token
    )
    assert scenario.projected_headroom_bytes == baseline.estimated_headroom_bytes
    assert scenario.estimated_additional_tokens == (
        baseline.estimated_additional_tokens
    )
    assert scenario.estimated_max_tokens == baseline.estimated_max_tokens


def test_fp16_to_int8_precision_scenario_matches_expected_example_math() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
        safety_margin_bytes=1_000,
    )[0]

    assert scenario.status == "available"
    assert scenario.current_dtype == "fp16"
    assert scenario.target_dtype == "int8"
    assert scenario.target_ratio == 0.5
    assert scenario.projected_current_kv_bytes == 1_000
    assert scenario.projected_current_allocated_bytes == 6_000
    assert scenario.projected_kv_growth_bytes_per_token == 50
    assert scenario.projected_headroom_bytes == 3_000
    assert scenario.estimated_additional_tokens == 60
    assert scenario.estimated_max_tokens == 80


def test_int8_projection_subtracts_only_kv_memory_difference() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.current_allocated_bytes == 7_000
    assert scenario.current_kv_bytes == 2_000
    assert scenario.projected_current_kv_bytes == 1_000
    assert scenario.projected_current_allocated_bytes == 6_000


def test_fp16_to_int4_uses_theoretical_packed_ratio() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int4",),
    )[0]

    assert scenario.target_ratio == 0.25
    assert scenario.projected_current_kv_bytes == 500
    assert scenario.projected_kv_growth_bytes_per_token == 25
    assert any("INT4" in item for item in scenario.assumptions)


def test_int4_projected_current_kv_bytes_rounds_up_fractional_bytes() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_002]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int4",),
    )[0]

    assert scenario.target_ratio == 0.25
    assert scenario.projected_current_kv_bytes == 501


def test_fp32_to_fp16_precision_scenario() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots(
            [10, 15, 20], [2_000, 3_000, 4_000], dtype="float32"
        ),
        memory_events=[_memory(allocated=8_000)],
        device_capacity_bytes=12_000,
        target_dtypes=("fp16",),
    )[0]

    assert scenario.current_dtype == "fp32"
    assert scenario.target_ratio == 0.5
    assert scenario.projected_current_kv_bytes == 2_000
    assert scenario.projected_kv_growth_bytes_per_token == 100
    assert scenario.projected_current_allocated_bytes == 6_000


def test_bf16_to_int8_precision_scenario() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots(
            [10, 15, 20], [1_000, 1_500, 2_000], dtype="bfloat16"
        ),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.current_dtype == "bf16"
    assert scenario.target_ratio == 0.5
    assert scenario.projected_current_kv_bytes == 1_000


def test_precision_scenario_safety_margin_reduces_projected_headroom() -> None:
    no_margin = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]
    with_margin = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
        safety_margin_bytes=1_000,
    )[0]

    assert no_margin.projected_headroom_bytes == 4_000
    assert with_margin.projected_headroom_bytes == 3_000


def test_precision_scenario_uses_sampled_snapshot_growth_per_token() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.baseline_kv_growth_bytes_per_token == 100
    assert scenario.projected_kv_growth_bytes_per_token == 50


def test_precision_scenario_cpu_trace_is_unavailable() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=0, reserved=0, peak=0)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "unavailable"
    assert "baseline OOM estimate unavailable" in scenario.reason
    assert scenario.estimated_additional_tokens is None


def test_precision_scenario_insufficient_snapshots_is_unavailable() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([20], [2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "unavailable"
    assert "baseline OOM estimate unavailable" in scenario.reason


def test_precision_scenario_missing_memory_events_is_unavailable() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "unavailable"
    assert scenario.current_allocated_bytes is None


def test_precision_scenario_unsupported_target_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        OOMAnalyzer().compare_kv_precision(
            snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
            memory_events=[_memory(allocated=7_000)],
            device_capacity_bytes=10_000,
            target_dtypes=("float7",),
        )


def test_precision_scenario_unsupported_current_dtype_is_unavailable() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots(
            [10, 15, 20], [1_000, 1_500, 2_000], dtype="float7"
        ),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "unavailable"
    assert "unsupported current KV dtype" in scenario.reason


def test_precision_scenario_mixed_current_dtypes_are_unavailable() -> None:
    snapshots = _precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000])
    snapshots[-1].per_layer[0].v_dtype = "float32"

    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=snapshots,
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "unavailable"
    assert "mixed current KV dtypes" in scenario.reason


def test_precision_scenario_dtype_change_across_snapshots_is_unavailable() -> None:
    snapshots = _precision_snapshots(
        [10, 15, 20], [2_000, 3_000, 4_000], dtype="float32"
    )
    snapshots[-1].per_layer[0].k_dtype = "float16"
    snapshots[-1].per_layer[0].v_dtype = "float16"

    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=snapshots,
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "unavailable"
    assert "KV dtype changed across snapshots" in scenario.reason


def test_precision_scenario_consistent_dtype_aliases_are_accepted() -> None:
    snapshots = _precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000])
    snapshots[0].per_layer[0].k_dtype = "float16"
    snapshots[0].per_layer[0].v_dtype = "float16"
    snapshots[1].per_layer[0].k_dtype = "torch.float16"
    snapshots[1].per_layer[0].v_dtype = "torch.float16"
    snapshots[2].per_layer[0].k_dtype = "fp16"
    snapshots[2].per_layer[0].v_dtype = "fp16"

    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=snapshots,
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "available"
    assert scenario.current_dtype == "fp16"
    assert scenario.projected_current_kv_bytes == 1_000


def test_precision_scenario_kv_larger_than_allocated_is_unavailable() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=1_500)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.status == "unavailable"
    assert "current KV bytes exceed current allocated bytes" in scenario.reason
    assert scenario.estimated_additional_tokens is None


def test_precision_scenario_current_dtype_alias_preserves_baseline_values() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("float16",),
    )[0]

    assert scenario.target_dtype == "fp16"
    assert scenario.target_ratio == 1.0
    assert scenario.projected_current_allocated_bytes == 7_000
    assert scenario.projected_kv_growth_bytes_per_token == 100


def test_precision_scenario_additional_tokens_use_floor() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=6_950)],
        device_capacity_bytes=10_000,
        target_dtypes=("fp16",),
        safety_margin_bytes=1_000,
    )[0]

    assert scenario.projected_headroom_bytes == 2_050
    assert scenario.estimated_additional_tokens == 20
    assert scenario.estimated_max_tokens == 40


def test_precision_scenario_uses_plain_typed_events_without_gpu() -> None:
    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=_precision_snapshots([10, 15, 20], [1_000, 1_500, 2_000]),
        memory_events=[_memory(allocated=7_000)],
        device_capacity_bytes=10_000,
        target_dtypes=("int8",),
    )[0]

    assert scenario.is_available


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
