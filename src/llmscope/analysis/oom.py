"""Conservative OOM headroom estimates from observed KV-cache growth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Literal, Sequence

from llmscope.analysis.what_if import bytes_per_element, normalize_kv_dtype
from llmscope.core.events import KVCacheSnapshot, MemoryEvent

OOMStatus = Literal["available", "unavailable", "at_or_over_limit"]

_ASSUMPTIONS = (
    "Estimate only; real CUDA OOM behavior can differ.",
    "Assumes non-KV allocated memory remains approximately constant.",
    "Uses current allocated bytes, not reserved bytes, as the primary headroom input.",
    "Requires caller-provided device_capacity_bytes; traces do not store GPU capacity.",
)

_PRECISION_ASSUMPTIONS = _ASSUMPTIONS + (
    "Analytical KV precision scenario only; it does not enable KV quantization.",
    "Replaces only observed KV bytes; non-KV allocated memory is not scaled.",
    "INT4 is modeled as a theoretical packed 0.5-byte-per-element estimate.",
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


@dataclass(frozen=True)
class KVPrecisionOOMScenario:
    """Projected token headroom for an analytical KV precision scenario."""

    target_dtype: str
    current_dtype: str | None
    status: OOMStatus
    reason: str
    target_ratio: float | None
    current_tokens: int | None
    current_kv_bytes: int | None
    current_allocated_bytes: int | None
    current_reserved_bytes: int | None
    device_capacity_bytes: int
    safety_margin_bytes: int
    baseline_kv_growth_bytes_per_token: float | None
    baseline_headroom_bytes: int | None
    baseline_estimated_additional_tokens: int | None
    baseline_estimated_max_tokens: int | None
    projected_current_kv_bytes: int | None
    projected_kv_growth_bytes_per_token: float | None
    projected_current_allocated_bytes: int | None
    projected_headroom_bytes: int | None
    estimated_additional_tokens: int | None
    estimated_max_tokens: int | None
    assumptions: tuple[str, ...] = _PRECISION_ASSUMPTIONS
    warnings: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        """Whether this precision scenario could be computed."""
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

    def compare_kv_precision(
        self,
        *,
        snapshots: Sequence[KVCacheSnapshot],
        memory_events: Sequence[MemoryEvent],
        device_capacity_bytes: int,
        target_dtypes: Sequence[str] = ("fp16", "int8", "int4"),
        safety_margin_bytes: int = 0,
    ) -> tuple[KVPrecisionOOMScenario, ...]:
        """Project OOM headroom for analytical KV-cache precision scenarios.

        This does not quantize KV caches or assert runtime support for any target
        dtype. It replaces only the observed KV component of current allocated
        memory, then scales future KV growth by the same byte-width ratio.
        """
        _validate_capacity_args(device_capacity_bytes, safety_margin_bytes)
        targets = _normalize_target_dtypes(target_dtypes)

        current = snapshots[-1] if snapshots else None
        current_dtype, dtype_reason = _uniform_growth_snapshot_kv_dtype(snapshots)
        baseline = self.estimate(
            snapshots=snapshots,
            memory_events=memory_events,
            device_capacity_bytes=device_capacity_bytes,
            safety_margin_bytes=safety_margin_bytes,
        )

        if current_dtype is None:
            return tuple(
                _unavailable_precision_scenario(
                    target_dtype=target,
                    current_dtype=None,
                    reason=dtype_reason,
                    baseline=baseline,
                    current=current,
                    device_capacity_bytes=device_capacity_bytes,
                    safety_margin_bytes=safety_margin_bytes,
                )
                for target in targets
            )

        if baseline.status == "unavailable":
            return tuple(
                _unavailable_precision_scenario(
                    target_dtype=target,
                    current_dtype=current_dtype,
                    reason=f"baseline OOM estimate unavailable: {baseline.reason}",
                    baseline=baseline,
                    current=current,
                    device_capacity_bytes=device_capacity_bytes,
                    safety_margin_bytes=safety_margin_bytes,
                )
                for target in targets
            )

        if (
            current is not None
            and baseline.current_allocated_bytes is not None
            and current.total_bytes > baseline.current_allocated_bytes
        ):
            return tuple(
                _unavailable_precision_scenario(
                    target_dtype=target,
                    current_dtype=current_dtype,
                    reason=(
                        "invalid trace telemetry: current KV bytes exceed current "
                        "allocated bytes"
                    ),
                    baseline=baseline,
                    current=current,
                    device_capacity_bytes=device_capacity_bytes,
                    safety_margin_bytes=safety_margin_bytes,
                )
                for target in targets
            )

        return tuple(
            _project_precision_scenario(
                target_dtype=target,
                current_dtype=current_dtype,
                baseline=baseline,
                current=current,
                device_capacity_bytes=device_capacity_bytes,
                safety_margin_bytes=safety_margin_bytes,
            )
            for target in targets
        )


def _normalize_target_dtypes(target_dtypes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(target_dtypes, str) or not target_dtypes:
        raise ValueError("target_dtypes must be a non-empty sequence of dtype strings")
    return tuple(normalize_kv_dtype(dtype) for dtype in target_dtypes)


def _uniform_growth_snapshot_kv_dtype(
    snapshots: Sequence[KVCacheSnapshot],
) -> tuple[str | None, str]:
    if not snapshots:
        return None, "no KV snapshots available"

    growth_pair_start_indices = {
        index
        for index, (previous, current) in enumerate(zip(snapshots, snapshots[1:]))
        if (
            current.total_tokens - previous.total_tokens > 0
            and current.total_bytes - previous.total_bytes > 0
        )
    }
    growth_snapshot_indices = growth_pair_start_indices | {
        index + 1 for index in growth_pair_start_indices
    }
    snapshots_to_check = (
        [snapshots[index] for index in sorted(growth_snapshot_indices)]
        if growth_snapshot_indices
        else [snapshots[-1]]
    )

    dtypes = []
    for snapshot in snapshots_to_check:
        dtype, reason = _uniform_snapshot_kv_dtype(snapshot)
        if dtype is None:
            return None, reason
        dtypes.append(dtype)

    unique_dtypes = set(dtypes)
    if len(unique_dtypes) != 1:
        return None, f"KV dtype changed across snapshots: {sorted(unique_dtypes)}"

    return dtypes[0], "dtype available"


def _uniform_snapshot_kv_dtype(snapshot: KVCacheSnapshot) -> tuple[str | None, str]:
    if not snapshot.per_layer:
        return None, "snapshot per-layer dtype metadata required"

    snapshot_dtypes: set[str] = set()
    for layer in snapshot.per_layer:
        try:
            snapshot_dtypes.add(normalize_kv_dtype(layer.k_dtype))
            snapshot_dtypes.add(normalize_kv_dtype(layer.v_dtype))
        except ValueError as exc:
            return None, f"unsupported current KV dtype: {exc}"

    if len(snapshot_dtypes) != 1:
        return (
            None,
            f"mixed current KV dtypes are unsupported: {sorted(snapshot_dtypes)}",
        )

    return snapshot_dtypes.pop(), "dtype available"


def _project_precision_scenario(
    *,
    target_dtype: str,
    current_dtype: str,
    baseline: OOMEstimate,
    current: KVCacheSnapshot | None,
    device_capacity_bytes: int,
    safety_margin_bytes: int,
) -> KVPrecisionOOMScenario:
    current_bpe = bytes_per_element(current_dtype)
    target_bpe = bytes_per_element(target_dtype)
    target_ratio = target_bpe / current_bpe

    current_tokens = baseline.current_tokens or 0
    current_kv_bytes = baseline.current_kv_bytes or 0
    current_allocated_bytes = baseline.current_allocated_bytes or 0
    baseline_growth = baseline.kv_growth_bytes_per_token or 0.0

    projected_current_kv_bytes = math.ceil(current_kv_bytes * target_ratio)
    projected_growth = baseline_growth * target_ratio
    projected_allocated = (
        current_allocated_bytes - current_kv_bytes + projected_current_kv_bytes
    )
    effective_limit = device_capacity_bytes - safety_margin_bytes
    projected_headroom = max(0, effective_limit - projected_allocated)

    if projected_allocated >= effective_limit:
        status: OOMStatus = "at_or_over_limit"
        additional_tokens = 0
        estimated_max_tokens = current_tokens
        reason = "projected allocated bytes are at or above the effective limit"
    else:
        status = "available"
        additional_tokens = math.floor(projected_headroom / projected_growth)
        estimated_max_tokens = current_tokens + additional_tokens
        reason = "precision scenario estimate available"

    return KVPrecisionOOMScenario(
        target_dtype=target_dtype,
        current_dtype=current_dtype,
        status=status,
        reason=reason,
        target_ratio=target_ratio,
        current_tokens=current.total_tokens if current is not None else None,
        current_kv_bytes=baseline.current_kv_bytes,
        current_allocated_bytes=baseline.current_allocated_bytes,
        current_reserved_bytes=baseline.current_reserved_bytes,
        device_capacity_bytes=device_capacity_bytes,
        safety_margin_bytes=safety_margin_bytes,
        baseline_kv_growth_bytes_per_token=baseline.kv_growth_bytes_per_token,
        baseline_headroom_bytes=baseline.estimated_headroom_bytes,
        baseline_estimated_additional_tokens=baseline.estimated_additional_tokens,
        baseline_estimated_max_tokens=baseline.estimated_max_tokens,
        projected_current_kv_bytes=projected_current_kv_bytes,
        projected_kv_growth_bytes_per_token=projected_growth,
        projected_current_allocated_bytes=projected_allocated,
        projected_headroom_bytes=projected_headroom,
        estimated_additional_tokens=additional_tokens,
        estimated_max_tokens=estimated_max_tokens,
    )


def _unavailable_precision_scenario(
    *,
    target_dtype: str,
    current_dtype: str | None,
    reason: str,
    baseline: OOMEstimate,
    current: KVCacheSnapshot | None,
    device_capacity_bytes: int,
    safety_margin_bytes: int,
) -> KVPrecisionOOMScenario:
    return KVPrecisionOOMScenario(
        target_dtype=target_dtype,
        current_dtype=current_dtype,
        status="unavailable",
        reason=reason,
        target_ratio=(
            bytes_per_element(target_dtype) / bytes_per_element(current_dtype)
            if current_dtype is not None
            else None
        ),
        current_tokens=current.total_tokens if current is not None else None,
        current_kv_bytes=baseline.current_kv_bytes,
        current_allocated_bytes=baseline.current_allocated_bytes,
        current_reserved_bytes=baseline.current_reserved_bytes,
        device_capacity_bytes=device_capacity_bytes,
        safety_margin_bytes=safety_margin_bytes,
        baseline_kv_growth_bytes_per_token=baseline.kv_growth_bytes_per_token,
        baseline_headroom_bytes=baseline.estimated_headroom_bytes,
        baseline_estimated_additional_tokens=baseline.estimated_additional_tokens,
        baseline_estimated_max_tokens=baseline.estimated_max_tokens,
        projected_current_kv_bytes=None,
        projected_kv_growth_bytes_per_token=None,
        projected_current_allocated_bytes=None,
        projected_headroom_bytes=None,
        estimated_additional_tokens=None,
        estimated_max_tokens=None,
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
