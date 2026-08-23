"""Read-only access to saved llmscope JSONL traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .events import KVCacheSnapshot, MemoryEvent, trace_event_adapter

SupportedTraceEvent = KVCacheSnapshot | MemoryEvent

_SUPPORTED_EVENT_TYPES = frozenset({"kv_cache_snapshot", "memory_event"})


class TraceLoadError(ValueError):
    """Raised when a saved trace cannot be parsed or validated."""


@dataclass(frozen=True)
class TraceSession:
    """Immutable view of a saved llmscope trace.

    Loading a trace does not require a model instance or hook installation. The
    loaded snapshots and memory events are reconstructed as the same typed event
    models produced by live tracing.
    """

    session_id: str | None
    _events: tuple[SupportedTraceEvent, ...] = ()

    @classmethod
    def load(cls, path: str | Path) -> "TraceSession":
        """Load a JSONL trace saved by ``Tracer.save()``."""
        events: list[SupportedTraceEvent] = []
        session_id: str | None = None

        trace_path = Path(path)
        with trace_path.open(encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                raw_event = _parse_json_line(line, line_number)
                event_type = raw_event.get("event_type")
                if event_type not in _SUPPORTED_EVENT_TYPES:
                    raise TraceLoadError(
                        f"Unsupported event_type {event_type!r} at line "
                        f"{line_number}; supported event types are: "
                        f"{', '.join(sorted(_SUPPORTED_EVENT_TYPES))}"
                    )

                event = _validate_event(raw_event, line_number)
                if session_id is None:
                    session_id = event.session_id
                elif event.session_id != session_id:
                    raise TraceLoadError(
                        f"Inconsistent session_id at line {line_number}: "
                        f"expected {session_id!r}, got {event.session_id!r}"
                    )
                events.append(event)

        return cls(session_id=session_id, _events=tuple(events))

    @property
    def events(self) -> list[SupportedTraceEvent]:
        """All supported events in file order."""
        return list(self._events)

    @property
    def kv_snapshots(self) -> list[KVCacheSnapshot]:
        """KV-cache snapshots in file order."""
        return [event for event in self._events if isinstance(event, KVCacheSnapshot)]

    @property
    def memory_events(self) -> list[MemoryEvent]:
        """Memory events in file order."""
        return [event for event in self._events if isinstance(event, MemoryEvent)]


def _parse_json_line(line: str, line_number: int) -> dict[str, Any]:
    try:
        raw_event = json.loads(line)
    except json.JSONDecodeError as exc:
        raise TraceLoadError(
            f"Malformed JSON at line {line_number}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw_event, dict):
        raise TraceLoadError(f"Trace line {line_number} must be a JSON object")
    return raw_event


def _validate_event(raw_event: dict[str, Any], line_number: int) -> SupportedTraceEvent:
    try:
        event = trace_event_adapter.validate_python(raw_event)
    except ValidationError as exc:
        raise TraceLoadError(
            f"Invalid trace event at line {line_number}: {exc}"
        ) from exc

    if isinstance(event, (KVCacheSnapshot, MemoryEvent)):
        return event
    raise TraceLoadError(
        f"Unsupported event_type {event.event_type!r} at line {line_number}; "
        f"supported event types are: {', '.join(sorted(_SUPPORTED_EVENT_TYPES))}"
    )
