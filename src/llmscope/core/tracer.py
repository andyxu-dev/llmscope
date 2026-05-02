"""Tracer: the main public interface for instrumenting LLM inference."""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .adapters.base import PastKeyValues
from .adapters.registry import get_adapter
from .config import Config
from .events import KVCacheSnapshot, LayerKVStats, MemoryEvent


def _weights_bytes(model: nn.Module) -> int:
    return sum(p.nelement() * p.element_size() for p in model.parameters())


class Tracer:
    """Instrument an nn.Module to capture KV cache and memory events during generate().

    Usage (context manager)::

        with Tracer(model) as tracer:
            model.generate(input_ids, max_new_tokens=50)
        print(tracer.summary())

    Usage (explicit)::

        tracer = Tracer(model)
        tracer.start()
        # ... multiple cells ...
        tracer.stop()
    """

    def __init__(self, model: nn.Module, config: Config | None = None) -> None:
        self._model = model
        self._config = config or Config()
        self._session_id: str | None = None
        self._is_active = False
        self._hook_manager: Any = None  # HookManager, imported lazily to avoid circular
        self._kv_snapshots: deque[KVCacheSnapshot] = deque(
            maxlen=self._config.ring_buffer_size
        )
        self._memory_events: deque[MemoryEvent] = deque(
            maxlen=self._config.ring_buffer_size
        )
        self._adapter = get_adapter(model)
        self._weights_bytes = _weights_bytes(model)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "Tracer":
        """Attach hooks and begin recording. Idempotent: safe to call twice."""
        if self._is_active:
            return self
        from .hooks import HookManager

        self._session_id = str(uuid.uuid4())
        self._kv_snapshots.clear()
        self._memory_events.clear()
        self._hook_manager = HookManager(
            model=self._model,
            adapter=self._adapter,
            on_snapshot=self._on_snapshot,
            sample_every_n_steps=self._config.sample_every_n_steps,
        )
        self._hook_manager.attach()
        self._is_active = True
        return self

    def stop(self) -> None:
        """Detach hooks and stop recording. Idempotent."""
        if not self._is_active:
            return
        if self._hook_manager is not None:
            self._hook_manager.detach()
            self._hook_manager = None
        self._is_active = False

    def __enter__(self) -> "Tracer":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def kv_snapshots(self) -> list[KVCacheSnapshot]:
        return list(self._kv_snapshots)

    @property
    def memory_events(self) -> list[MemoryEvent]:
        return list(self._memory_events)

    # ------------------------------------------------------------------
    # Internal hook callback
    # ------------------------------------------------------------------

    def _on_snapshot(self, past_key_values: PastKeyValues, step_index: int) -> None:
        assert self._session_id is not None
        now = datetime.now(tz=timezone.utc)
        per_layer: list[LayerKVStats] = []
        total_bytes = 0

        for layer_idx, (k, v) in enumerate(past_key_values):
            k_bytes = int(k.nelement() * k.element_size())
            v_bytes = int(v.nelement() * v.element_size())
            total_bytes += k_bytes + v_bytes

            # Cast to float32 for stats — avoids bfloat16 numpy issues
            k_f = k.detach().float().cpu()
            v_f = v.detach().float().cpu()

            per_layer.append(
                LayerKVStats(
                    layer_idx=layer_idx,
                    k_shape=tuple(k.shape),
                    v_shape=tuple(v.shape),
                    k_dtype=str(k.dtype).split(".")[-1],
                    v_dtype=str(v.dtype).split(".")[-1],
                    k_bytes=k_bytes,
                    v_bytes=v_bytes,
                    k_min=float(k_f.min()),
                    k_max=float(k_f.max()),
                    k_mean=float(k_f.mean()),
                    k_std=float(k_f.std()),
                    v_min=float(v_f.min()),
                    v_max=float(v_f.max()),
                    v_mean=float(v_f.mean()),
                    v_std=float(v_f.std()),
                )
            )

        # Sequence length is dim 2 in (batch, heads, seq, head_dim)
        total_tokens = int(per_layer[0].k_shape[2]) if per_layer else 0

        snapshot = KVCacheSnapshot(
            session_id=self._session_id,
            timestamp=now,
            step_index=step_index,
            per_layer=per_layer,
            total_bytes=total_bytes,
            total_tokens=total_tokens,
        )
        self._kv_snapshots.append(snapshot)
        self._memory_events.append(self._build_memory_event(now, total_bytes))

    def _build_memory_event(self, now: datetime, kv_bytes: int) -> MemoryEvent:
        assert self._session_id is not None
        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            peak = int(torch.cuda.max_memory_allocated())
        else:
            allocated = reserved = peak = 0

        activations = max(0, allocated - self._weights_bytes - kv_bytes)
        return MemoryEvent(
            session_id=self._session_id,
            timestamp=now,
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            peak_allocated_bytes=peak,
            breakdown={
                "weights": self._weights_bytes,
                "kv_cache": kv_bytes,
                "activations": activations,
            },
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def summary(self, tokenizer: Any = None) -> str:
        """Return a human-readable summary of the current trace."""
        lines = [f"llmscope Tracer  session={self._session_id or 'not started'}"]
        lines.append(f"Weights : {self._weights_bytes / 1e6:.2f} MB")
        if self._kv_snapshots:
            first = self._kv_snapshots[0]
            last = self._kv_snapshots[-1]
            lines.append(f"KV snapshots : {len(self._kv_snapshots)} captured")
            lines.append(
                f"KV tokens    : {first.total_tokens} → {last.total_tokens}"
                f"  ({last.total_bytes / 1e6:.3f} MB latest)"
            )
        else:
            lines.append("No KV snapshots captured yet.")
        if self._memory_events:
            mem = self._memory_events[-1]
            if mem.allocated_bytes > 0:
                lines.append("Memory (latest):")
                for k, v in mem.breakdown.items():
                    lines.append(f"  {k:12s}: {v / 1e6:.2f} MB")
            else:
                lines.append("Memory : N/A (CPU mode — no CUDA allocator)")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write all captured events to a JSONL file."""
        out = Path(path)
        with out.open("w", encoding="utf-8") as f:
            for snapshot in self._kv_snapshots:
                f.write(snapshot.to_jsonl())
            for event in self._memory_events:
                f.write(event.to_jsonl())

    @classmethod
    def load(cls, path: str | Path) -> "Tracer":
        """Reconstruct a Tracer from a saved JSONL file (replay mode)."""
        raise NotImplementedError("Replay mode coming in v0.2")

    def dashboard(self, port: int = 8501) -> subprocess.Popen[bytes]:
        """Save current trace to a temp JSONL file and launch the Streamlit dashboard.

        Returns the Popen handle so the caller can wait() or terminate() if needed.
        Raises ImportError if streamlit is not installed.
        """
        import os
        import tempfile

        try:
            import streamlit  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "streamlit is required to run the dashboard. "
                'Install it with: pip install "llmscope[dashboard]"'
            ) from exc

        app_path = Path(__file__).parent.parent / "dashboard" / "app.py"

        tmp = tempfile.NamedTemporaryFile(
            suffix=".jsonl", prefix="llmscope_", delete=False
        )
        tmp.close()
        self.save(tmp.name)

        env = {**os.environ, "LLMSCOPE_TRACE_PATH": tmp.name}
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(port),
            ],
            env=env,
        )
        return proc

    def serve(self, port: int = 8000) -> None:
        """Start the FastAPI server for remote access (available after Week 3)."""
        raise NotImplementedError("FastAPI server coming in Week 3")
