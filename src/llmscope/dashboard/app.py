"""Streamlit dashboard: reads a JSONL trace file and visualises KV cache and memory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

# Allow running as `streamlit run app.py --` with LLMSCOPE_TRACE_PATH env var
TRACE_PATH_ENV = "LLMSCOPE_TRACE_PATH"


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (kv_snapshots_dicts, memory_event_dicts) from a JSONL file."""
    kv: list[dict[str, Any]] = []
    mem: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            et = obj.get("event_type")
            if et == "kv_cache_snapshot":
                kv.append(obj)
            elif et == "memory_event":
                mem.append(obj)
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

    kv_events, mem_events = _load_jsonl(trace_path)

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
        import plotly.graph_objects as go  # type: ignore[import-untyped]

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
            import plotly.graph_objects as go  # type: ignore[import-untyped]

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
