# llmscope

[![CI](https://img.shields.io/github/actions/workflow/status/llmscope/llmscope/ci.yml)](https://github.com/llmscope/llmscope/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

**llmscope** is a lightweight Python library for inspecting LLM inference internals — specifically how KV cache grows token-by-token and how memory is attributed across weights, cache, and activations.

It is **not** a generic observability platform. The focus is narrow and intentional: give ML engineers a clear, step-by-step view of what happens inside a transformer's KV cache during `model.generate()`.

---

## What it does

```
┌─────────────────────────────────────────────────────────┐
│  model.generate(input_ids, max_new_tokens=50)           │
│                          │                              │
│              hook on model backbone                     │
│                          │                              │
│          KVCacheSnapshot per decoding step              │
│   (layer shapes, byte counts, per-layer float stats)   │
│                          │                              │
│     ┌────────────────────┼──────────────────┐          │
│     ↓                    ↓                  ↓          │
│  JSONL file         Tracer.summary()   Streamlit UI     │
└─────────────────────────────────────────────────────────┘
```

**Single hook, no per-layer overhead.** The hook attaches once to the model backbone (`model.transformer` for GPT-2, `model.model` for Llama-style), capturing the atomic `past_key_values` snapshot after every decoding step.

---

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmscope import Tracer, Config

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
input_ids = tokenizer("Hello, world", return_tensors="pt").input_ids

# Context manager — hooks attach/detach automatically
with Tracer(model, config=Config(sample_every_n_steps=2)) as tracer:
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=50, use_cache=True)

print(tracer.summary())
# llmscope Tracer  session=<uuid>
# Weights : 548.09 MB
# KV snapshots : 26 captured
# KV tokens    : 5 → 54  (0.084 MB latest)
# Memory : N/A (CPU mode — no CUDA allocator)

tracer.save("run.jsonl")   # all snapshots + memory events as JSONL
```

### Explicit start/stop (Colab cross-cell)

```python
tracer = Tracer(model)
tracer.start()

# ... other notebook cells ...

tracer.stop()
```

### Streamlit dashboard

```python
# Launches a local Streamlit server; returns a Popen handle
proc = tracer.dashboard(port=8501)
```

Requires the `dashboard` extra:

```bash
pip install "llmscope[dashboard]"
```

The dashboard shows:
- KV cache size (MB) over decoding steps
- Per-layer K/V byte breakdown for the latest snapshot
- Memory attribution: weights / KV cache / activations (residual), with "N/A" on CPU

---

## Installation

Not yet on PyPI (targeting v0.1 release). Install from source:

```bash
git clone https://github.com/llmscope/llmscope
cd llmscope
pip install -e "."                   # core only
pip install -e ".[dashboard]"        # + Streamlit
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, transformers ≥ 4.35.

---

## Run the demo

No model download required — uses a tiny random-weight GPT-2:

```bash
python examples/demo.py
```

This generates a trace, prints a summary, saves `examples/demo_trace.jsonl`, and prints the exact command to open the Streamlit dashboard.

---

## Config options

```python
from llmscope import Config

Config(
    sample_every_n_steps=2,   # capture every 2nd decoding step (default: 1)
    ring_buffer_size=500,     # max snapshots kept in memory (default: 1000)
    capture_attention_weights=False,  # not yet implemented
)
```

---

## Analysis layer

```python
from llmscope.analysis import KVCacheAnalyzer, MemoryProfiler

# KV growth curve + outlier risk
result = KVCacheAnalyzer().analyze(tracer.kv_snapshots)
# result.growth_curve        → list of (step, tokens, bytes) points
# result.latest_per_layer    → per-layer byte breakdown
# result.outlier_risk        → "low" | "high" | "unknown"
#   high = k_max/k_std > 10 (signals potential attention sink / outlier dim)

# Memory attribution
profile = MemoryProfiler().profile(model, latest_snapshot=tracer.kv_snapshots[-1])
# profile.weights_bytes      → sum of all parameter bytes
# profile.kv_bytes           → from latest KVCacheSnapshot
# profile.activations_bytes  → allocated - weights - kv  (None on CPU)
```

---

## Architecture decisions

| Decision | Choice | Rationale |
|---|---|---|
| Hook level | Model backbone, not per-attention-layer | Atomic `past_key_values`; no cross-layer sync complexity |
| Frontend | Streamlit imports core directly | No FastAPI indirection in v0.1 |
| Storage | In-memory `deque` ring buffer | Low-overhead hot path; `save()` for persistence |
| Memory | Residual method | `activations = total_allocated − weights − kv`; CPU → N/A |
| Cache format | Handles all three transformers KV formats | tuple-of-tuples (≤4.35), `.key_cache/.value_cache` (4.36–4.x), `.layers[i].keys/.values` (5.x) |

---


## Roadmap

- [ ] Llama / Qwen / Mistral / Phi-3 adapters
- [ ] What-if KV memory estimator (quantization impact lookup table)
- [ ] FastAPI + SSE for real-time streaming dashboards
- [ ] Colab demo notebook
- [ ] PyPI release (v0.1)

---

## Contributing

Issues and PRs welcome. Run the test suite with:

```bash
pip install -e ".[dev]"
pytest
```
