"""Regression tests for the bundled demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from examples import demo


def _stable_trace_payload(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row.pop("session_id", None)
        row.pop("timestamp", None)
    return rows


def test_demo_trace_is_deterministic_except_runtime_metadata(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    trace_path = tmp_path / "demo_trace.jsonl"
    monkeypatch.setattr(demo, "TRACE_PATH", trace_path)

    demo.main()
    first = _stable_trace_payload(trace_path)

    demo.main()
    second = _stable_trace_payload(trace_path)

    assert second == first
