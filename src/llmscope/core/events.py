"""Event schemas for LLM inference internals."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Tuple, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator


class Event(BaseModel):
    """Base event with common session and timestamp fields."""

    session_id: str
    timestamp: datetime

    def to_jsonl(self) -> str:
        """Serialize the event to a single JSONL line."""
        return self.model_dump_json() + "\n"


class ModelMetadata(Event):
    """Model configuration emitted once at session start."""

    event_type: Literal["model_metadata"] = "model_metadata"
    model_name: str
    architecture: str
    num_layers: int
    num_heads: int
    hidden_dim: int
    num_kv_heads: int
    dtype: str
    device: str

    @field_validator("num_layers", "num_heads", "hidden_dim", "num_kv_heads")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be a positive integer")
        return value


class InferenceSession(Event):
    """Emitted once per generate() call with prompt and config details."""

    event_type: Literal["inference_session"] = "inference_session"
    model_metadata: ModelMetadata
    prompt_length: int
    generation_config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt_length")
    @classmethod
    def _non_negative_prompt_length(cls, value: int) -> int:
        if value < 0:
            raise ValueError("prompt_length must be non-negative")
        return value


class ForwardPassEvent(Event):
    """Emitted for each forward pass during generation."""

    event_type: Literal["forward_pass"] = "forward_pass"
    step_index: int
    input_token_count: int
    total_token_count: int

    @field_validator("step_index", "input_token_count", "total_token_count")
    @classmethod
    def _non_negative_ints(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value


class LayerKVStats(BaseModel):
    """Per-layer KV cache statistics."""

    layer_idx: int
    k_shape: Tuple[int, ...]
    v_shape: Tuple[int, ...]
    k_dtype: str
    v_dtype: str
    k_bytes: int
    v_bytes: int
    k_min: float
    k_max: float
    k_mean: float
    k_std: float
    v_min: float
    v_max: float
    v_mean: float
    v_std: float

    @field_validator("layer_idx")
    @classmethod
    def _non_negative_layer_idx(cls, value: int) -> int:
        if value < 0:
            raise ValueError("layer_idx must be non-negative")
        return value

    @field_validator("k_shape", "v_shape")
    @classmethod
    def _validate_shape(cls, value: Tuple[int, ...]) -> Tuple[int, ...]:
        if not value or any(item < 1 for item in value):
            raise ValueError("shapes must be tuples of positive ints")
        return value

    @field_validator("k_bytes", "v_bytes")
    @classmethod
    def _non_negative_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("byte counts must be non-negative")
        return value


class KVCacheSnapshot(Event):
    """Per-step snapshot of KV cache across all layers."""

    event_type: Literal["kv_cache_snapshot"] = "kv_cache_snapshot"
    step_index: int
    per_layer: List[LayerKVStats]
    total_bytes: int
    total_tokens: int

    @field_validator("total_bytes", "total_tokens", "step_index")
    @classmethod
    def _non_negative_totals(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value


class AttentionPattern(Event):
    """Attention pattern metadata (capture_attention_weights=True only)."""

    event_type: Literal["attention_pattern"] = "attention_pattern"
    layer_idx: int
    head_idx: int
    pattern_type: Literal["full", "sparse", "sliding"]
    sparsity_ratio: float
    top_k_attended_indices: List[int]
    attention_weights_summary: List[List[float]]

    @field_validator("layer_idx", "head_idx")
    @classmethod
    def _non_negative_indices(cls, value: int) -> int:
        if value < 0:
            raise ValueError("indices must be non-negative")
        return value

    @field_validator("sparsity_ratio")
    @classmethod
    def _validate_sparsity_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("sparsity_ratio must be between 0 and 1")
        return value

    @field_validator("top_k_attended_indices")
    @classmethod
    def _validate_topk(cls, value: List[int]) -> List[int]:
        if any(item < 0 for item in value):
            raise ValueError(
                "top_k_attended_indices must contain non-negative integers"
            )
        return value

    @field_validator("attention_weights_summary")
    @classmethod
    def _validate_attention_summary(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("attention_weights_summary must be a non-empty 2D array")
        if len(value) > 64 or any(len(row) > 64 for row in value):
            raise ValueError("attention_weights_summary must be at most 64x64")
        row_length = len(value[0])
        if row_length == 0 or any(len(row) != row_length for row in value):
            raise ValueError(
                "attention_weights_summary rows must be non-empty and equal length"
            )
        return value


class MemoryEvent(Event):
    """Memory breakdown snapshot (weights / kv_cache / activations)."""

    event_type: Literal["memory_event"] = "memory_event"
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    breakdown: Dict[str, int]

    @field_validator("allocated_bytes", "reserved_bytes", "peak_allocated_bytes")
    @classmethod
    def _non_negative_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("memory bytes must be non-negative")
        return value

    @field_validator("breakdown")
    @classmethod
    def _validate_breakdown(cls, value: Dict[str, int]) -> Dict[str, int]:
        for item in value.values():
            if item < 0:
                raise ValueError("breakdown values must be non-negative")
        return value


class QuantizationWhatIf(Event):
    """Estimated impact of applying a quantization configuration."""

    event_type: Literal["quantization_what_if"] = "quantization_what_if"
    target_bits: int
    target_components: List[Literal["kv_cache", "weights", "activations"]]
    estimated_savings_bytes: int
    estimated_quality_impact: str

    @field_validator("target_bits", "estimated_savings_bytes")
    @classmethod
    def _validate_non_negative_ints(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value


TraceEvent = Annotated[
    Union[
        ModelMetadata,
        InferenceSession,
        ForwardPassEvent,
        KVCacheSnapshot,
        AttentionPattern,
        MemoryEvent,
        QuantizationWhatIf,
    ],
    Field(discriminator="event_type"),
]

# Pre-built TypeAdapter for validating/deserializing TraceEvent from JSON
trace_event_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)
