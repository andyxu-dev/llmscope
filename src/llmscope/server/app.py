"""FastAPI application factory for llmscope."""

from __future__ import annotations

from fastapi import FastAPI

from .api import router

app = FastAPI(
    title="llmscope",
    description="API server for visualizing LLM inference internals.",
    version="0.1.0",
)

app.include_router(router)
