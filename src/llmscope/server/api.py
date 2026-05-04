"""REST API endpoints for llmscope."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")


@router.get("/health")
def health_check() -> dict[str, Any]:
    """Return a simple health check response."""
    return {"status": "ok", "service": "llmscope"}


@router.post("/trace")
def create_trace(payload: dict[str, Any]) -> dict[str, Any]:
    """Receive a trace request payload and return a placeholder response."""
    raise HTTPException(status_code=501, detail="Trace endpoint is not implemented yet")
