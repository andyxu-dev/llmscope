"""Minimal llmscope demo — no downloads, CPU-only, tiny random-weight GPT-2."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore[import-untyped]

from llmscope import Tracer

TRACE_PATH = Path(__file__).parent / "demo_trace.jsonl"
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)

    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=100,
        n_positions=64,
    )
    model = GPT2LMHeadModel(cfg).eval()

    input_ids = torch.randint(0, 100, (1, 5))

    print("Running model.generate() under llmscope.Tracer …")
    with Tracer(model) as tracer:
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=10, use_cache=True)

    print(tracer.summary())

    tracer.save(str(TRACE_PATH))
    print(f"Trace saved to {TRACE_PATH}")
    print()
    print("To view the dashboard, run:")
    print(f"  export LLMSCOPE_TRACE_PATH={TRACE_PATH}")
    print("  python -m streamlit run src/llmscope/dashboard/app.py")


if __name__ == "__main__":
    main()
