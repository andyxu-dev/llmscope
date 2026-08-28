"""Generate deterministic synthetic CUDA-style demonstration telemetry.

This file contains synthetic demonstration data. It was not captured from a real
GPU run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from llmscope.core.events import KVCacheSnapshot, LayerKVStats, MemoryEvent

TRACE_PATH = Path(__file__).parent / "synthetic_cuda_trace.jsonl"

SYNTHETIC_SESSION_ID = "synthetic-cuda-demo-v1"
BASE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)

BATCH_SIZE = 1
NUM_LAYERS = 40
NUM_KV_HEADS = 16
HEAD_DIM = 128
DTYPE = "float16"
BYTES_PER_ELEMENT = 2
SEQUENCE_LENGTHS = (4096, 4352, 4608, 4864)

WEIGHTS_BYTES = 14 * 1024**3
ACTIVATIONS_BYTES = 4 * 1024**3
NON_KV_ALLOCATED_BYTES = WEIGHTS_BYTES + ACTIVATIONS_BYTES
RESERVED_PADDING_BYTES = 1024**3
PEAK_PADDING_BYTES = 256 * 1024**2

SYNTHETIC_DISCLOSURE = (
    "This file contains synthetic demonstration data. It was not captured from "
    "a real GPU run."
)


def kv_bytes_for_sequence_length(sequence_length: int) -> int:
    """Return total synthetic KV bytes for the configured sequence length."""
    return (
        BATCH_SIZE
        * sequence_length
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * 2
        * BYTES_PER_ELEMENT
    )


def _layer_stats(layer_idx: int, sequence_length: int) -> LayerKVStats:
    tensor_bytes = (
        BATCH_SIZE * sequence_length * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_ELEMENT
    )
    stat_offset = layer_idx / 1000
    return LayerKVStats(
        layer_idx=layer_idx,
        k_shape=(BATCH_SIZE, NUM_KV_HEADS, sequence_length, HEAD_DIM),
        v_shape=(BATCH_SIZE, NUM_KV_HEADS, sequence_length, HEAD_DIM),
        k_dtype=DTYPE,
        v_dtype=DTYPE,
        k_bytes=tensor_bytes,
        v_bytes=tensor_bytes,
        k_min=-0.50 - stat_offset,
        k_max=0.50 + stat_offset,
        k_mean=stat_offset,
        k_std=0.125 + stat_offset,
        v_min=-0.45 - stat_offset,
        v_max=0.45 + stat_offset,
        v_mean=-stat_offset,
        v_std=0.100 + stat_offset,
    )


def _snapshot(step_index: int, sequence_length: int) -> KVCacheSnapshot:
    per_layer = [
        _layer_stats(layer_idx, sequence_length) for layer_idx in range(NUM_LAYERS)
    ]
    total_bytes = sum(layer.k_bytes + layer.v_bytes for layer in per_layer)
    return KVCacheSnapshot(
        session_id=SYNTHETIC_SESSION_ID,
        timestamp=BASE_TIMESTAMP + timedelta(seconds=step_index * 2),
        step_index=step_index,
        per_layer=per_layer,
        total_bytes=total_bytes,
        total_tokens=sequence_length,
    )


def _memory_event(step_index: int, kv_bytes: int) -> MemoryEvent:
    allocated_bytes = NON_KV_ALLOCATED_BYTES + kv_bytes
    return MemoryEvent(
        session_id=SYNTHETIC_SESSION_ID,
        timestamp=BASE_TIMESTAMP + timedelta(seconds=(step_index * 2) + 1),
        allocated_bytes=allocated_bytes,
        reserved_bytes=allocated_bytes + RESERVED_PADDING_BYTES,
        peak_allocated_bytes=allocated_bytes + PEAK_PADDING_BYTES,
        breakdown={
            "weights": WEIGHTS_BYTES,
            "kv_cache": kv_bytes,
            "activations": ACTIVATIONS_BYTES,
        },
    )


def build_events() -> list[KVCacheSnapshot | MemoryEvent]:
    """Build the deterministic synthetic trace events in file order."""
    events: list[KVCacheSnapshot | MemoryEvent] = []
    for step_index, sequence_length in enumerate(SEQUENCE_LENGTHS):
        snapshot = _snapshot(step_index, sequence_length)
        events.append(snapshot)
        events.append(_memory_event(step_index, snapshot.total_bytes))
    return events


def generate_trace(path: Path = TRACE_PATH) -> None:
    """Write the deterministic synthetic JSONL trace."""
    with path.open("w", encoding="utf-8") as output:
        for event in build_events():
            output.write(event.to_jsonl())


def main() -> None:
    generate_trace()
    print(SYNTHETIC_DISCLOSURE)
    print(f"Synthetic CUDA-style trace saved to {TRACE_PATH}")
    print()
    print("To view it in the dashboard, run:")
    print(f"  export LLMSCOPE_TRACE_PATH={TRACE_PATH}")
    print("  .venv/bin/python -m streamlit run src/llmscope/dashboard/app.py")
    print("Use GPU capacity assumption = 24 GiB and safety margin = 1 GiB.")


if __name__ == "__main__":
    main()
