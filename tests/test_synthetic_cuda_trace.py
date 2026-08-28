"""Tests for the synthetic CUDA-style demonstration trace."""

from __future__ import annotations

from pathlib import Path

from examples import generate_synthetic_cuda_trace as synthetic
from llmscope import TraceSession
from llmscope.analysis import OOMAnalyzer

CAPACITY_BYTES = 24 * 1024**3
SAFETY_MARGIN_BYTES = 1024**3
EFFECTIVE_LIMIT_BYTES = CAPACITY_BYTES - SAFETY_MARGIN_BYTES
KV_BYTES_PER_TOKEN = (
    synthetic.BATCH_SIZE
    * synthetic.NUM_LAYERS
    * synthetic.NUM_KV_HEADS
    * synthetic.HEAD_DIM
    * 2
    * synthetic.BYTES_PER_ELEMENT
)
LATEST_TOKENS = synthetic.SEQUENCE_LENGTHS[-1]
LATEST_KV_BYTES = KV_BYTES_PER_TOKEN * LATEST_TOKENS
LATEST_ALLOCATED_BYTES = synthetic.NON_KV_ALLOCATED_BYTES + LATEST_KV_BYTES
BASELINE_HEADROOM_BYTES = EFFECTIVE_LIMIT_BYTES - LATEST_ALLOCATED_BYTES


def _load_generated_trace(tmp_path: Path) -> TraceSession:
    trace_path = tmp_path / "synthetic_cuda_trace.jsonl"
    synthetic.generate_trace(trace_path)
    return TraceSession.load(trace_path)


def test_synthetic_generator_run_twice_produces_identical_file_bytes(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    synthetic.generate_trace(first_path)
    synthetic.generate_trace(second_path)

    assert second_path.read_bytes() == first_path.read_bytes()


def test_committed_synthetic_trace_loads_with_trace_session() -> None:
    trace = TraceSession.load(synthetic.TRACE_PATH)

    assert trace.session_id == synthetic.SYNTHETIC_SESSION_ID


def test_synthetic_trace_has_expected_snapshot_count(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)

    assert len(trace.kv_snapshots) == len(synthetic.SEQUENCE_LENGTHS)


def test_synthetic_trace_has_expected_memory_event_count(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)

    assert len(trace.memory_events) == len(synthetic.SEQUENCE_LENGTHS)


def test_synthetic_trace_uses_single_consistent_session_id(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)

    assert {event.session_id for event in trace.events} == {
        synthetic.SYNTHETIC_SESSION_ID
    }


def test_synthetic_kv_shapes_match_documented_dimensions(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)

    for snapshot in trace.kv_snapshots:
        for layer in snapshot.per_layer:
            assert layer.k_shape == (
                synthetic.BATCH_SIZE,
                synthetic.NUM_KV_HEADS,
                snapshot.total_tokens,
                synthetic.HEAD_DIM,
            )
            assert layer.v_shape == layer.k_shape


def test_synthetic_per_layer_bytes_sum_to_total_bytes(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)

    for snapshot in trace.kv_snapshots:
        per_layer_total = sum(
            layer.k_bytes + layer.v_bytes for layer in snapshot.per_layer
        )
        assert per_layer_total == snapshot.total_bytes


def test_synthetic_total_bytes_match_kv_formula(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)

    for snapshot in trace.kv_snapshots:
        assert snapshot.total_bytes == (
            synthetic.BATCH_SIZE
            * snapshot.total_tokens
            * synthetic.NUM_LAYERS
            * synthetic.NUM_KV_HEADS
            * synthetic.HEAD_DIM
            * 2
            * synthetic.BYTES_PER_ELEMENT
        )


def test_synthetic_kv_growth_bytes_per_token_is_deterministic(
    tmp_path: Path,
) -> None:
    trace = _load_generated_trace(tmp_path)

    estimate = OOMAnalyzer().estimate(
        snapshots=trace.kv_snapshots,
        memory_events=trace.memory_events,
        device_capacity_bytes=CAPACITY_BYTES,
        safety_margin_bytes=SAFETY_MARGIN_BYTES,
    )

    assert estimate.kv_growth_bytes_per_token == KV_BYTES_PER_TOKEN


def test_synthetic_memory_allocated_equals_non_kv_plus_kv(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)

    for snapshot, memory_event in zip(trace.kv_snapshots, trace.memory_events):
        assert memory_event.allocated_bytes == (
            synthetic.NON_KV_ALLOCATED_BYTES + snapshot.total_bytes
        )


def test_synthetic_oom_estimate_is_available_for_24_gib_with_1_gib_margin(
    tmp_path: Path,
) -> None:
    trace = _load_generated_trace(tmp_path)

    estimate = OOMAnalyzer().estimate(
        snapshots=trace.kv_snapshots,
        memory_events=trace.memory_events,
        device_capacity_bytes=CAPACITY_BYTES,
        safety_margin_bytes=SAFETY_MARGIN_BYTES,
    )

    assert estimate.status == "available"


def test_synthetic_baseline_additional_tokens_match_expected_math(
    tmp_path: Path,
) -> None:
    trace = _load_generated_trace(tmp_path)

    estimate = OOMAnalyzer().estimate(
        snapshots=trace.kv_snapshots,
        memory_events=trace.memory_events,
        device_capacity_bytes=CAPACITY_BYTES,
        safety_margin_bytes=SAFETY_MARGIN_BYTES,
    )

    assert estimate.current_tokens == LATEST_TOKENS
    assert estimate.current_kv_bytes == LATEST_KV_BYTES
    assert estimate.current_allocated_bytes == LATEST_ALLOCATED_BYTES
    assert estimate.estimated_headroom_bytes == BASELINE_HEADROOM_BYTES
    assert estimate.estimated_additional_tokens == 11_520
    assert estimate.estimated_max_tokens == 16_384


def test_synthetic_fp16_precision_scenario_equals_baseline(tmp_path: Path) -> None:
    trace = _load_generated_trace(tmp_path)
    baseline = OOMAnalyzer().estimate(
        snapshots=trace.kv_snapshots,
        memory_events=trace.memory_events,
        device_capacity_bytes=CAPACITY_BYTES,
        safety_margin_bytes=SAFETY_MARGIN_BYTES,
    )

    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=trace.kv_snapshots,
        memory_events=trace.memory_events,
        device_capacity_bytes=CAPACITY_BYTES,
        safety_margin_bytes=SAFETY_MARGIN_BYTES,
        target_dtypes=("fp16",),
    )[0]

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


def test_synthetic_int8_precision_scenario_matches_expected_math(
    tmp_path: Path,
) -> None:
    trace = _load_generated_trace(tmp_path)

    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=trace.kv_snapshots,
        memory_events=trace.memory_events,
        device_capacity_bytes=CAPACITY_BYTES,
        safety_margin_bytes=SAFETY_MARGIN_BYTES,
        target_dtypes=("int8",),
    )[0]

    assert scenario.projected_current_kv_bytes == 796_917_760
    assert scenario.projected_current_allocated_bytes == 20_124_270_592
    assert scenario.projected_kv_growth_bytes_per_token == 163_840
    assert scenario.projected_headroom_bytes == 4_571_791_360
    assert scenario.estimated_additional_tokens == 27_904
    assert scenario.estimated_max_tokens == 32_768


def test_synthetic_int4_precision_scenario_matches_expected_math(
    tmp_path: Path,
) -> None:
    trace = _load_generated_trace(tmp_path)

    scenario = OOMAnalyzer().compare_kv_precision(
        snapshots=trace.kv_snapshots,
        memory_events=trace.memory_events,
        device_capacity_bytes=CAPACITY_BYTES,
        safety_margin_bytes=SAFETY_MARGIN_BYTES,
        target_dtypes=("int4",),
    )[0]

    assert scenario.projected_current_kv_bytes == 398_458_880
    assert scenario.projected_current_allocated_bytes == 19_725_811_712
    assert scenario.projected_kv_growth_bytes_per_token == 81_920
    assert scenario.projected_headroom_bytes == 4_970_250_240
    assert scenario.estimated_additional_tokens == 60_672
    assert scenario.estimated_max_tokens == 65_536
