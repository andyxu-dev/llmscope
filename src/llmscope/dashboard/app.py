"""Streamlit dashboard: reads a JSONL trace file and visualises KV cache and memory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

from llmscope import TraceLoadError, TraceSession
from llmscope.analysis import KVPrecisionOOMScenario, OOMAnalyzer, OOMEstimate
from llmscope.analysis.what_if import normalize_kv_dtype
from llmscope.core.events import KVCacheSnapshot, MemoryEvent

# Allow running as `streamlit run app.py --` with LLMSCOPE_TRACE_PATH env var
TRACE_PATH_ENV = "LLMSCOPE_TRACE_PATH"
_DEFAULT_GPU_CAPACITY_GIB = 24.0
_OOM_UNAVAILABLE_MESSAGE = (
    "OOM headroom estimate unavailable: this trace does not contain CUDA allocator "
    "telemetry."
)
_OOM_CAVEAT = (
    "OOM headroom is analytical, not a guaranteed prediction. It assumes non-KV "
    "allocated memory remains approximately constant. Real behavior can differ "
    "because of temporary CUDA allocations, fragmentation, allocator behavior, "
    "other GPU processes, and model behavior. INT8/INT4 scenarios model "
    "theoretical KV storage size only; actual quantized-cache implementations may "
    "include scale metadata, packing/alignment overhead, kernels, and temporary "
    "memory."
)


def _load_trace(path: Path) -> TraceSession:
    """Load a typed TraceSession for dashboard analysis."""
    return TraceSession.load(path)


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kv_snapshots_dicts, memory_event_dicts) from a JSONL file."""
    trace = _load_trace(path)
    kv = [event.model_dump(mode="json") for event in trace.kv_snapshots]
    mem = [event.model_dump(mode="json") for event in trace.memory_events]
    return kv, mem


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _layer_breakdown_series(
    per_layer: Any,
) -> tuple[list[Any], list[float], list[float]]:
    if not isinstance(per_layer, list):
        return [], [], []

    layer_indices = []
    k_bytes_mb = []
    v_bytes_mb = []
    for i, layer in enumerate(per_layer):
        if not isinstance(layer, dict):
            continue
        layer_indices.append(layer.get("layer_idx", i))
        k_bytes_mb.append(_number(layer.get("k_bytes")) / 1e6)
        v_bytes_mb.append(_number(layer.get("v_bytes")) / 1e6)
    return layer_indices, k_bytes_mb, v_bytes_mb


def _memory_values(mem: dict[str, Any]) -> tuple[float, float, float, float, bool]:
    allocated = _number(mem.get("allocated_bytes"))
    breakdown = mem.get("breakdown", {})
    if not isinstance(breakdown, dict):
        breakdown = {}

    weights_mb = _number(breakdown.get("weights")) / 1e6
    kv_mb = _number(breakdown.get("kv_cache")) / 1e6
    act_mb = _number(breakdown.get("activations")) / 1e6
    return allocated, weights_mb, kv_mb, act_mb, allocated > 0


def _format_bytes(value: int | float | None) -> str:
    """Format byte counts with binary units for dashboard display."""
    if value is None:
        return "N/A"

    numeric = float(value)
    if numeric == 0:
        return "0 B"

    units = ("B", "KiB", "MiB", "GiB", "TiB")
    abs_value = abs(numeric)
    unit_index = 0
    while abs_value >= 1024 and unit_index < len(units) - 1:
        numeric /= 1024
        abs_value /= 1024
        unit_index += 1

    if unit_index == 0:
        if numeric.is_integer():
            return f"{int(numeric)} B"
        return f"{numeric:.2f} B"
    return f"{numeric:.2f} {units[unit_index]}"


def _gib_to_bytes(value_gib: float) -> int:
    """Convert a GiB UI value to bytes."""
    if value_gib < 0:
        raise ValueError("GiB value must be non-negative")
    return int(value_gib * 1024**3)


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,}"


def _prepare_oom_summary_rows(
    *,
    snapshots: Sequence[KVCacheSnapshot],
    memory_events: Sequence[MemoryEvent],
    device_capacity_bytes: int,
    safety_margin_bytes: int,
) -> tuple[OOMEstimate, list[dict[str, str]]]:
    """Prepare display rows for the OOM headroom estimate."""
    estimate = OOMAnalyzer().estimate(
        snapshots=snapshots,
        memory_events=memory_events,
        device_capacity_bytes=device_capacity_bytes,
        safety_margin_bytes=safety_margin_bytes,
    )
    if estimate.status == "unavailable":
        return estimate, []

    rows = [
        {
            "Metric": "Current sequence length",
            "Estimate": _format_tokens(estimate.current_tokens),
        },
        {
            "Metric": "Current KV cache size",
            "Estimate": _format_bytes(estimate.current_kv_bytes),
        },
        {
            "Metric": "Observed KV growth per token",
            "Estimate": f"{_format_bytes(estimate.kv_growth_bytes_per_token)} / token",
        },
        {
            "Metric": "Current allocated GPU memory",
            "Estimate": _format_bytes(estimate.current_allocated_bytes),
        },
        {
            "Metric": "Current reserved GPU memory",
            "Estimate": _format_bytes(estimate.current_reserved_bytes),
        },
        {
            "Metric": "Device capacity assumption",
            "Estimate": _format_bytes(device_capacity_bytes),
        },
        {
            "Metric": "Safety margin",
            "Estimate": _format_bytes(safety_margin_bytes),
        },
        {
            "Metric": "Estimated remaining memory headroom",
            "Estimate": _format_bytes(estimate.estimated_headroom_bytes),
        },
        {
            "Metric": "Estimated additional tokens",
            "Estimate": _format_tokens(estimate.estimated_additional_tokens),
        },
        {
            "Metric": "Estimated maximum sequence length",
            "Estimate": _format_tokens(estimate.estimated_max_tokens),
        },
    ]
    return estimate, rows


def _current_dtype_for_precision_targets(
    snapshots: Sequence[KVCacheSnapshot],
) -> str | None:
    if not snapshots or not snapshots[-1].per_layer:
        return None

    dtypes = set()
    try:
        for layer in snapshots[-1].per_layer:
            dtypes.add(normalize_kv_dtype(layer.k_dtype))
            dtypes.add(normalize_kv_dtype(layer.v_dtype))
    except ValueError:
        return None

    if len(dtypes) != 1:
        return None
    return dtypes.pop()


def _select_precision_target_dtypes(current_dtype: str | None) -> tuple[str, ...]:
    """Select useful precision targets while avoiding duplicate rows."""
    candidates = [current_dtype, "fp16", "int8", "int4"]
    selected: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        canonical = normalize_kv_dtype(candidate)
        if canonical not in selected:
            selected.append(canonical)
    return tuple(selected)


def _prepare_precision_scenario_rows(
    *,
    snapshots: Sequence[KVCacheSnapshot],
    memory_events: Sequence[MemoryEvent],
    device_capacity_bytes: int,
    safety_margin_bytes: int,
) -> list[dict[str, str]]:
    """Prepare display rows for analytical KV precision OOM scenarios."""
    current_dtype = _current_dtype_for_precision_targets(snapshots)
    target_dtypes = _select_precision_target_dtypes(current_dtype)
    scenarios = OOMAnalyzer().compare_kv_precision(
        snapshots=snapshots,
        memory_events=memory_events,
        device_capacity_bytes=device_capacity_bytes,
        safety_margin_bytes=safety_margin_bytes,
        target_dtypes=target_dtypes,
    )
    return [_precision_scenario_row(scenario) for scenario in scenarios]


def _precision_scenario_row(scenario: KVPrecisionOOMScenario) -> dict[str, str]:
    is_current = scenario.target_dtype == scenario.current_dtype
    dtype_label = scenario.target_dtype
    if is_current:
        dtype_label = f"{dtype_label} (current)"

    if scenario.target_dtype in {"int8", "int4"} and not is_current:
        note = "analytical storage-size scenario"
    elif is_current:
        note = "current precision"
    else:
        note = "analytical storage-size scenario"

    return {
        "Target KV dtype": dtype_label,
        "Projected current KV": _format_bytes(scenario.projected_current_kv_bytes),
        "Projected allocated GPU memory": _format_bytes(
            scenario.projected_current_allocated_bytes
        ),
        "Projected KV growth/token": (
            f"{_format_bytes(scenario.projected_kv_growth_bytes_per_token)} / token"
        ),
        "Projected headroom": _format_bytes(scenario.projected_headroom_bytes),
        "Estimated additional tokens": _format_tokens(
            scenario.estimated_additional_tokens
        ),
        "Estimated max tokens": _format_tokens(scenario.estimated_max_tokens),
        "Status": scenario.status,
        "Notes": note,
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="llmscope dashboard", layout="wide")
    st.title("llmscope — KV Cache & Memory Inspector")

    trace_path_str = os.environ.get(TRACE_PATH_ENV)
    if not trace_path_str:
        st.error(
            f"No trace file specified. Set the `{TRACE_PATH_ENV}` environment variable "
            "or launch via `tracer.dashboard()`."
        )
        return

    trace_path = Path(trace_path_str)
    if not trace_path.exists():
        st.error(f"Trace file not found: `{trace_path}`")
        return

    try:
        trace = _load_trace(trace_path)
    except TraceLoadError as exc:
        st.error(f"Could not load trace: {exc}")
        return
    kv_events = [event.model_dump(mode="json") for event in trace.kv_snapshots]
    mem_events = [event.model_dump(mode="json") for event in trace.memory_events]

    # ── Summary row ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("KV snapshots", len(kv_events))

    latest_tokens: Optional[int] = None
    if kv_events:
        latest_tokens = kv_events[-1].get("total_tokens")
    col2.metric(
        "Generated tokens (latest)",
        latest_tokens if latest_tokens is not None else "N/A",
    )

    latest_kv_mb: Optional[str] = None
    if kv_events:
        total_bytes = _number(kv_events[-1].get("total_bytes"))
        latest_kv_mb = f"{total_bytes / 1e6:.3f} MB"
    col3.metric("KV cache size (latest)", latest_kv_mb or "N/A")

    st.divider()

    # ── KV growth chart ────────────────────────────────────────────────────────
    st.subheader("KV Cache Growth")
    if kv_events:
        import plotly.graph_objects as go
        steps = [e.get("step_index", i) for i, e in enumerate(kv_events)]
        bytes_mb = [_number(e.get("total_bytes")) / 1e6 for e in kv_events]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=steps, y=bytes_mb, mode="lines+markers", name="KV bytes (MB)")
        )
        fig.update_layout(
            xaxis_title="Step index",
            yaxis_title="KV cache (MB)",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No KV cache snapshots in this trace.")

    # ── Per-layer breakdown ────────────────────────────────────────────────────
    st.subheader("Per-layer KV Breakdown (latest snapshot)")
    if kv_events:
        per_layer = kv_events[-1].get("per_layer", [])
        if per_layer:
            import plotly.graph_objects as go
            layer_indices, k_bytes_mb, v_bytes_mb = _layer_breakdown_series(per_layer)

            if layer_indices:
                fig2 = go.Figure(
                    data=[
                        go.Bar(name="K bytes", x=layer_indices, y=k_bytes_mb),
                        go.Bar(name="V bytes", x=layer_indices, y=v_bytes_mb),
                    ]
                )
                fig2.update_layout(
                    barmode="stack",
                    xaxis_title="Layer",
                    yaxis_title="MB",
                    height=300,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No per-layer data available.")
        else:
            st.info("No per-layer data available.")
    else:
        st.info("No KV cache snapshots in this trace.")

    # ── Memory attribution ─────────────────────────────────────────────────────
    st.subheader("Memory Attribution (latest)")
    if mem_events:
        allocated, weights_mb, kv_mb, act_mb, on_gpu = _memory_values(mem_events[-1])

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Weights", f"{weights_mb:.2f} MB")
        mc2.metric("KV cache", f"{kv_mb:.3f} MB")
        if on_gpu:
            mc3.metric("Activations (residual)", f"{act_mb:.2f} MB")
            mc4.metric("Total allocated", f"{allocated / 1e6:.2f} MB")
        else:
            mc3.metric("Activations (residual)", "N/A — CPU")
            mc4.metric("Total allocated", "N/A — CPU")
    else:
        st.info("No memory events in this trace.")

    # ── OOM headroom estimate ─────────────────────────────────────────────────
    st.subheader("OOM Headroom Estimate")
    oc1, oc2 = st.columns(2)
    capacity_gib = oc1.number_input(
        "GPU capacity assumption (GiB)",
        min_value=0.01,
        value=_DEFAULT_GPU_CAPACITY_GIB,
        step=1.0,
        help=(
            "User-provided assumption; saved traces do not store total GPU "
            "capacity."
        ),
    )
    margin_gib = oc2.number_input(
        "Safety margin (GiB)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="User-provided capacity reserve for a more conservative estimate.",
    )

    try:
        device_capacity_bytes = _gib_to_bytes(float(capacity_gib))
        safety_margin_bytes = _gib_to_bytes(float(margin_gib))
        oom_estimate, oom_rows = _prepare_oom_summary_rows(
            snapshots=trace.kv_snapshots,
            memory_events=trace.memory_events,
            device_capacity_bytes=device_capacity_bytes,
            safety_margin_bytes=safety_margin_bytes,
        )
    except ValueError as exc:
        st.warning(f"Invalid OOM assumptions: {exc}")
    else:
        if oom_estimate.status == "unavailable":
            st.info(_OOM_UNAVAILABLE_MESSAGE)
        else:
            if oom_estimate.status == "at_or_over_limit":
                st.warning(
                    "Current allocated GPU memory is at or above the effective "
                    "capacity assumption."
                )
            st.table(oom_rows)
            st.caption(_OOM_CAVEAT)

            scenario_rows = _prepare_precision_scenario_rows(
                snapshots=trace.kv_snapshots,
                memory_events=trace.memory_events,
                device_capacity_bytes=device_capacity_bytes,
                safety_margin_bytes=safety_margin_bytes,
            )
            st.subheader("KV Precision OOM Scenarios")
            st.caption(
                "These scenarios model KV storage size only. They do not enable "
                "KV quantization or guarantee model/runtime support."
            )
            st.table(scenario_rows)

    # ── What-if estimator ──────────────────────────────────────────────────────
    st.subheader("What-if: KV Cache Memory by dtype")
    if kv_events:
        _render_what_if(kv_events[-1])
    else:
        st.info("No KV cache snapshots — cannot estimate.")


def _render_what_if(latest_kv: dict[str, Any]) -> None:
    """Show estimated KV cache memory for fp16/int8/int4 at the current seq length."""
    import streamlit as st

    from llmscope.analysis.what_if import WhatIfEstimator

    per_layer = latest_kv.get("per_layer", [])
    total_tokens = int(_number(latest_kv.get("total_tokens")))

    if not isinstance(per_layer, list) or not per_layer or total_tokens == 0:
        st.info("Insufficient data for what-if estimate.")
        return

    # Derive model dimensions from the snapshot dict (not the Pydantic model).
    try:
        num_layers = len(per_layer)
        first = per_layer[0]
        k_shape = first.get("k_shape") if isinstance(first, dict) else None
        if not k_shape or len(k_shape) < 4:
            raise ValueError(f"Unexpected k_shape: {k_shape}")
        num_heads = int(k_shape[1])
        head_dim = int(k_shape[3])
        estimator = WhatIfEstimator(
            num_layers=num_layers, num_heads=num_heads, head_dim=head_dim
        )
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        st.warning(f"Could not derive model dimensions from snapshot: {exc}")
        return

    rows = []
    for dtype in ("fp16", "int8", "int4"):
        est = estimator.estimate(sequence_length=total_tokens, dtype=dtype)
        savings = est.savings_mb
        rows.append(
            {
                "dtype": dtype,
                "KV cache (MB)": f"{est.total_mb:.3f}",
                "vs fp16": "baseline" if savings is None else f"{savings:+.3f} MB",
                "compression": "1.0×" if est.compression_ratio is None
                else f"{est.compression_ratio:.1f}×",
            }
        )

    st.table(rows)


if __name__ == "__main__":
    main()
