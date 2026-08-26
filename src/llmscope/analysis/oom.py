"""Conservative OOM headroom estimates from observed KV-cache growth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Literal, Sequence

from llmscope.core.events import KVCacheSnapshot, MemoryEvent

OOMStatus = Literal["available", "unavailable", "at_or_over_limit"]

_ASSUMPTIONS = (
    "Estimate only; real CUDA OOM behavior can differ.",
    "Assumes non-KV allocated memory remains approximately constant.",
    "Uses current allocated bytes, not reserved bytes, as the primary headroom input.",
    "Requires caller-provided device_capacity_bytes; traces do not store GPU capacity.",
)


@dataclass(frozen=True)
class OOMEstimate:
    """Estimated token headroom before exhausting a caller-provided memory limit."""

    status: OOMStatus
    reason: str
    current_tokens: int | None
    current_kv_bytes: int | None
    kv_growth_bytes_per_token: float | None
    current_allocated_bytes: int | None
    current_reserved_bytes: int | None
    device_capacity_bytes: int
    safety_margin_bytes: int
    estimated_headroom_bytes: int | None
    estimated_additional_tokens: int | None
    estimated_max_tokens: int | None
    assumptions: tuple[str, ...] = _ASSUMPTIONS

    @property
    def is_available(self) -> bool:
        """Whether an estimate could be computed from the supplied trace data."""
        return self.status == "available"


class OOMAnalyzer:
    """Estimate remaining token headroom from observed KV growth and CUDA memory."""

    def estimate(
        self,
        *,
        snapshots: Sequence[KVCacheSnapshot],
        memory_events: Sequence[MemoryEvent],
        device_capacity_bytes: int,
        safety_margin_bytes: int = 0,
    ) -> OOMEstimate:
        """Return a conservative, explainable token-headroom estimate.

        The estimate uses observed KV-cache growth and the latest meaningful CUDA
        allocator event. It does not predict temporary workspaces, fragmentation,
        changing activations, other processes, or model-specific generation shifts.
        """
        _validate_capacity_args(device_capacity_bytes, safety_margin_bytes)

        current = snapshots[-1] if snapshots else None
        memory_event = _latest_meaningful_cuda_event(memory_events)

        if memory_event is None:
            return _unavailable(
                reason=(
                    "no CUDA allocator memory event available; CPU traces record "
                    "zero allocator metrics"
                ),
                current=current,
                device_capacity_bytes=device_capacity_bytes,
                safety_margin_bytes=safety_margin_bytes,
            )

        growth = _median_kv_growth_bytes_per_token(snapshots)
        if growth is None:
            return _unavailable(
                reason="at least two usable KV snapshots with positive growth required",
                current=current,
                memory_event=memory_event,
                device_capacity_bytes=device_capacity_bytes,
                safety_margin_bytes=safety_margin_bytes,
            )

        effective_limit = device_capacity_bytes - safety_margin_bytes
        headroom = max(0, effective_limit - memory_event.allocated_bytes)
        additional_tokens = math.floor(headroom / growth)
        estimated_max_tokens = (current.total_tokens if current is not None else 0)
        estimated_max_tokens += additional_tokens

        if memory_event.allocated_bytes >= effective_limit:
            return OOMEstimate(
                status="at_or_over_limit",
                reason="current allocated bytes are at or above the effective limit",
                current_tokens=current.total_tokens if current is not None else None,
                current_kv_bytes=current.total_bytes if current is not None else None,
                kv_growth_bytes_per_token=growth,
                current_allocated_bytes=memory_event.allocated_bytes,
                current_reserved_bytes=memory_event.reserved_bytes,
                device_capacity_bytes=device_capacity_bytes,
                safety_margin_bytes=safety_margin_bytes,
                estimated_headroom_bytes=0,
                estimated_additional_tokens=0,
                estimated_max_tokens=current.total_tokens if current is not None else 0,
            )

        return OOMEstimate(
            status="available",
            reason="estimate available",
            current_tokens=current.total_tokens if current is not None else None,
            current_kv_bytes=current.total_bytes if current is not None else None,
            kv_growth_bytes_per_token=growth,
            current_allocated_bytes=memory_event.allocated_bytes,
            current_reserved_bytes=memory_event.reserved_bytes,
            device_capacity_bytes=device_capacity_bytes,
            safety_margin_bytes=safety_margin_bytes,
            estimated_headroom_bytes=headroom,
            estimated_additional_tokens=additional_tokens,
            estimated_max_tokens=estimated_max_tokens,
        )


def _validate_capacity_args(
    device_capacity_bytes: int,
    safety_margin_bytes: int,
) -> None:
    if (
        not isinstance(device_capacity_bytes, int)
        or isinstance(device_capacity_bytes, bool)
        or device_capacity_bytes <= 0
    ):
        raise ValueError("device_capacity_bytes must be a positive integer")
    if (
        not isinstance(safety_margin_bytes, int)
        or isinstance(safety_margin_bytes, bool)
        or safety_margin_bytes < 0
    ):
        raise ValueError("safety_margin_bytes must be a non-negative integer")
    if safety_margin_bytes >= device_capacity_bytes:
        raise ValueError("safety_margin_bytes must be less than device_capacity_bytes")


def _latest_meaningful_cuda_event(
    memory_events: Sequence[MemoryEvent],
) -> MemoryEvent | None:
    for event in reversed(memory_events):
        if event.allocated_bytes > 0 or event.reserved_bytes > 0:
            return event
    return None


def _median_kv_growth_bytes_per_token(
    snapshots: Sequence[KVCacheSnapshot],
) -> float | None:
    slopes: list[float] = []
    for previous, current in zip(snapshots, snapshots[1:]):
        delta_tokens = current.total_tokens - previous.total_tokens
        delta_bytes = current.total_bytes - previous.total_bytes
        if delta_tokens <= 0 or delta_bytes <= 0:
            continue
        slopes.append(delta_bytes / delta_tokens)

    if not slopes:
        return None

    growth = float(median(slopes))
    if growth <= 0:
        return None
    return growth


def _unavailable(
    *,
    reason: str,
    current: KVCacheSnapshot | None,
    device_capacity_bytes: int,
    safety_margin_bytes: int,
    memory_event: MemoryEvent | None = None,
) -> OOMEstimate:
    return OOMEstimate(
        status="unavailable",
        reason=reason,
        current_tokens=current.total_tokens if current is not None else None,
        current_kv_bytes=current.total_bytes if current is not None else None,
        kv_growth_bytes_per_token=None,
        current_allocated_bytes=(
            memory_event.allocated_bytes if memory_event is not None else None
        ),
        current_reserved_bytes=(
            memory_event.reserved_bytes if memory_event is not None else None
        ),
        device_capacity_bytes=device_capacity_bytes,
        safety_margin_bytes=safety_margin_bytes,
        estimated_headroom_bytes=None,
        estimated_additional_tokens=None,
        estimated_max_tokens=None,
    )
