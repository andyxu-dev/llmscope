"""Configuration for the llmscope Tracer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Config(BaseModel):
    """Tracer configuration controlling capture behavior and memory usage."""

    capture_attention_weights: bool = False
    sample_every_n_steps: int = Field(default=1, ge=1)
    ring_buffer_size: int = Field(default=1000, ge=1)
