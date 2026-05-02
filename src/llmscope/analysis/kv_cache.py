"""KV cache analysis helpers for transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from llmscope.core.events import KVCacheSnapshot


@dataclass
class KVGrowthPoint:
    step_index: int
    total_tokens: int
    total_bytes: int


@dataclass
class LayerBreakdown:
    layer_idx: int
    k_bytes: int
    v_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.k_bytes + self.v_bytes


@dataclass
class KVAnalysisResult:
    growth_curve: List[KVGrowthPoint]
    latest_per_layer: List[LayerBreakdown]
    # "low", "high", or "unknown"
    outlier_risk: str = "unknown"
    outlier_risk_reason: str = ""


class KVCacheAnalyzer:
    """Analyze KV cache snapshots from a Tracer session."""

    def analyze(self, snapshots: List[KVCacheSnapshot]) -> KVAnalysisResult:
        """Compute growth curve, per-layer breakdown, and outlier risk."""
        if not snapshots:
            return KVAnalysisResult(
                growth_curve=[],
                latest_per_layer=[],
                outlier_risk="unknown",
                outlier_risk_reason="no snapshots",
            )

        growth_curve = [
            KVGrowthPoint(
                step_index=s.step_index,
                total_tokens=s.total_tokens,
                total_bytes=s.total_bytes,
            )
            for s in snapshots
        ]

        latest = snapshots[-1]
        latest_per_layer = [
            LayerBreakdown(
                layer_idx=layer.layer_idx,
                k_bytes=layer.k_bytes,
                v_bytes=layer.v_bytes,
            )
            for layer in latest.per_layer
        ]

        outlier_risk, reason = _compute_outlier_risk(latest)
        return KVAnalysisResult(
            growth_curve=growth_curve,
            latest_per_layer=latest_per_layer,
            outlier_risk=outlier_risk,
            outlier_risk_reason=reason,
        )


def _compute_outlier_risk(snapshot: KVCacheSnapshot) -> tuple[str, str]:
    """Return ("high"/"low"/"unknown", reason) based on k_max/k_std ratio."""
    ratios: List[float] = []
    for layer in snapshot.per_layer:
        k_max = getattr(layer, "k_max", None)
        k_std = getattr(layer, "k_std", None)
        try:
            k_max_float = float(k_max)
            k_std_float = float(k_std)
        except (TypeError, ValueError):
            continue
        if k_std_float > 0:
            ratios.append(k_max_float / k_std_float)

    if not ratios:
        return "unknown", "k_max/k_std unavailable, zero, or no layers"

    max_ratio = max(ratios)
    if max_ratio > 10:
        return "high", f"k_max/k_std={max_ratio:.1f} > 10"
    return "low", f"k_max/k_std={max_ratio:.1f} ≤ 10"
