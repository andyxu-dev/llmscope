"""What-if KV cache memory estimator.

Answers the question: "If I change sequence length, batch size, or dtype,
how much KV cache memory would I need?"

Two entry points:
  - estimate_kv_memory()   — standalone function, easiest to use
  - WhatIfEstimator        — class wrapping a snapshot for repeated queries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Bytes per element for each supported dtype.
# int4 uses 0.5 bytes/element (two values packed per byte).
_BYTES_PER_ELEMENT: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

_DTYPE_ALIASES: dict[str, str] = {
    "float32": "fp32",
    "torch.float32": "fp32",
    "fp32": "fp32",
    "float16": "fp16",
    "torch.float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
    "bf16": "bf16",
    "int8": "int8",
    "torch.int8": "int8",
    "int4": "int4",
}

# fp16 is the reference baseline for savings calculations.
_BASELINE_DTYPE = "fp16"


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive; expected an integer, got {value!r}")
    return value


def normalize_kv_dtype(dtype: str) -> str:
    """Return the canonical analytical dtype name used by what-if estimators."""
    if not isinstance(dtype, str):
        raise ValueError(
            f"dtype must be one of {sorted(_BYTES_PER_ELEMENT)}, got {dtype!r}"
        )

    normalized = dtype.strip().lower()
    canonical = _DTYPE_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unknown dtype '{dtype}'. Supported: {sorted(_BYTES_PER_ELEMENT)}"
        )
    return canonical


def bytes_per_element(dtype: str) -> float:
    """Return analytical bytes per KV element for a supported dtype."""
    return _BYTES_PER_ELEMENT[normalize_kv_dtype(dtype)]


@dataclass
class KVMemoryEstimate:
    """Result of a single what-if KV memory query."""

    num_layers: int
    num_heads: int
    head_dim: int
    batch_size: int
    sequence_length: int
    dtype: str

    # 2 × num_layers × num_heads × head_dim × seq_len × batch × bytes_per_element
    total_bytes: int

    # Bytes that would be used at the fp16 baseline for the same dimensions.
    # None when dtype is already fp16 (savings would be zero by definition).
    fp16_baseline_bytes: Optional[int]

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1e6

    @property
    def savings_bytes(self) -> Optional[int]:
        """Bytes saved vs. fp16 baseline. Negative means dtype costs more than fp16."""
        if self.fp16_baseline_bytes is None:
            return None
        return self.fp16_baseline_bytes - self.total_bytes

    @property
    def savings_mb(self) -> Optional[float]:
        s = self.savings_bytes
        return s / 1e6 if s is not None else None

    @property
    def compression_ratio(self) -> Optional[float]:
        """fp16_bytes / dtype_bytes. > 1 means this dtype is smaller than fp16."""
        if self.fp16_baseline_bytes is None or self.total_bytes == 0:
            return None
        return self.fp16_baseline_bytes / self.total_bytes


@dataclass
class WhatIfEstimator:
    """Estimate KV cache memory for varying sequence lengths, batch sizes, and dtypes.

    Construct from explicit model dimensions or from a live KVCacheSnapshot.
    """

    num_layers: int
    num_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        self.num_layers = _require_positive_int("num_layers", self.num_layers)
        self.num_heads = _require_positive_int("num_heads", self.num_heads)
        self.head_dim = _require_positive_int("head_dim", self.head_dim)

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "WhatIfEstimator":
        """Build from the latest KVCacheSnapshot captured by a Tracer.

        Derives num_layers from len(snapshot.per_layer).
        Derives num_heads and head_dim from the first layer's k_shape:
          k_shape = (batch, num_heads, seq_len, head_dim)
        """
        per_layer = getattr(snapshot, "per_layer", None)
        if not per_layer:
            raise ValueError(
                "snapshot.per_layer is empty — cannot derive model dimensions"
            )

        num_layers = len(per_layer)
        first = per_layer[0]
        k_shape = getattr(first, "k_shape", None)
        if k_shape is None or len(k_shape) < 4:
            raise ValueError(
                "Expected k_shape with 4 dims (batch, heads, seq, head_dim),"
                f" got: {k_shape}"
            )
        num_heads = int(k_shape[1])
        head_dim = int(k_shape[3])
        return cls(num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)

    def estimate(
        self,
        sequence_length: int,
        batch_size: int = 1,
        dtype: str = "fp16",
    ) -> KVMemoryEstimate:
        """Return a KVMemoryEstimate for the given configuration.

        Raises ValueError for unknown dtype, non-positive dimensions.
        """
        dtype = normalize_kv_dtype(dtype)
        sequence_length = _require_positive_int("sequence_length", sequence_length)
        batch_size = _require_positive_int("batch_size", batch_size)

        bpe = bytes_per_element(dtype)
        # KV cache = 2 tensors (K and V) per layer
        elements = (
            2
            * self.num_layers
            * self.num_heads
            * self.head_dim
            * sequence_length
            * batch_size
        )
        total_bytes = int(elements * bpe)

        if dtype == _BASELINE_DTYPE:
            fp16_baseline_bytes = None
        else:
            fp16_bpe = _BYTES_PER_ELEMENT[_BASELINE_DTYPE]
            fp16_baseline_bytes = int(elements * fp16_bpe)

        return KVMemoryEstimate(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dtype=dtype,
            total_bytes=total_bytes,
            fp16_baseline_bytes=fp16_baseline_bytes,
        )


def estimate_kv_memory(
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
    batch_size: int = 1,
    dtype: str = "fp16",
) -> KVMemoryEstimate:
    """Standalone helper — no class instantiation needed.

    Example::

        est = estimate_kv_memory(
            num_layers=32, num_heads=32, head_dim=128,
            sequence_length=2048, dtype="int8"
        )
        print(f"{est.total_mb:.1f} MB  (saves {est.savings_mb:.1f} MB vs fp16)")
    """
    return WhatIfEstimator(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
    ).estimate(sequence_length=sequence_length, batch_size=batch_size, dtype=dtype)
