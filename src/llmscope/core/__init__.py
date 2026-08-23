"""Core tracing and hook primitives for llmscope."""

from .adapters import ArchitectureAdapter, GenericAdapter, GPT2Adapter, get_adapter
from .config import Config
from .events import (
    AttentionPattern,
    ForwardPassEvent,
    InferenceSession,
    KVCacheSnapshot,
    LayerKVStats,
    MemoryEvent,
    ModelMetadata,
    QuantizationWhatIf,
    TraceEvent,
    trace_event_adapter,
)
from .hooks import HookManager
from .trace_session import TraceLoadError, TraceSession
from .tracer import Tracer

__all__ = [
    "ArchitectureAdapter",
    "GPT2Adapter",
    "GenericAdapter",
    "get_adapter",
    "Config",
    "AttentionPattern",
    "ForwardPassEvent",
    "InferenceSession",
    "KVCacheSnapshot",
    "LayerKVStats",
    "MemoryEvent",
    "ModelMetadata",
    "QuantizationWhatIf",
    "TraceEvent",
    "trace_event_adapter",
    "HookManager",
    "TraceLoadError",
    "TraceSession",
    "Tracer",
]
