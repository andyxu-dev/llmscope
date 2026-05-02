# llmscope — Handoff Summary

## Overall Goal

Open-source Python library + Streamlit visualization tool that lets LLM developers "see inside" a model during inference. Target: PyPI v0.1 in 4 weeks, solo dev + AI assistant, no personal GPU (Colab T4 / CPU daily dev).

**Definition of shipped:** `pip install llmscope` works on PyPI + README + Colab demo notebook.

---

## Week Milestones

| Week | Goal | Status |
|------|------|--------|
| 1 | Skeleton + event schema + GPT-2 hook end-to-end (CPU) | **DONE** |
| 2 | Streamlit dashboard + memory attribution | next |
| 3 | Expand to Llama/Qwen + what-if estimator + FastAPI SSE | |
| 4 | Tests + PyPI packaging + Colab demo + README | |

---

## Current Status (end of Week 1)

**All 21 tests pass.** `python -m pytest` from `.venv` (Python 3.11, uv-managed).

```
21 passed in 2.86s  |  overall coverage 83%  |  core paths 87–97%
```

The Week 1 acceptance test passes:
- Instrument a 2-layer GPT-2 (random weights, no download)
- `model.generate(input_ids, max_new_tokens=10)`
- Assert KV snapshots captured, token counts monotonically grow, layer count == `n_layer`

---

## What Was Built (files changed / created)

### New files
| File | What it does |
|------|-------------|
| `src/llmscope/core/config.py` | `Config` pydantic model (`capture_attention_weights`, `sample_every_n_steps`, `ring_buffer_size`) |
| `src/llmscope/core/adapters/base.py` | `ArchitectureAdapter` ABC + `normalize_past_key_values()` (handles tuple, DynamicCache 4.x, DynamicCache 5.x) |
| `src/llmscope/core/adapters/gpt2.py` | `GPT2Adapter`: hooks `model.transformer`, extracts `.past_key_values` |
| `src/llmscope/core/adapters/generic.py` | `GenericAdapter`: fallback for unknown architectures, hooks top-level model |
| `src/llmscope/core/adapters/registry.py` | `get_adapter(model)` auto-detects via `model.config.model_type` |
| `src/llmscope/core/adapters/__init__.py` | Package exports |
| `tests/test_tracer_gpt2.py` | End-to-end GPT-2 smoke tests (6 tests) |

### Rewritten files
| File | Key changes |
|------|-------------|
| `src/llmscope/core/events.py` | Removed Pydantic v2-incompatible `Field(..., const=True)`; added `trace_event_adapter: TypeAdapter[TraceEvent]` for discriminated union deserialization; added `step_index` to `KVCacheSnapshot` |
| `src/llmscope/core/hooks.py` | Full `HookManager` (was all stubs): `attach()`/`detach()`/`is_attached`; single hook on adapter's module; `sample_every_n_steps` skip logic |
| `src/llmscope/core/tracer.py` | Full `Tracer` (was all stubs): `start()`/`stop()`/context manager; idempotent (safe to call twice); `deque` ring buffer; `_on_snapshot` callback builds `KVCacheSnapshot` + `MemoryEvent`; `summary()`/`save()` |
| `src/llmscope/core/__init__.py` | Fixed wrong imports (`AttentionEvent`/`KVCacheEvent` → correct names) |
| `src/llmscope/__init__.py` | Now exports `Tracer`, `Config`, `__version__` |
| `pyproject.toml` | Moved `fastapi`/`uvicorn`/`typer` to `[server]` optional; added `numpy`; fixed hatchling src-layout (`packages = ["src/llmscope"]`); `mypy strict = true` |
| `.github/workflows/ci.yml` | Quoted Python version strings (`"3.9"`, `"3.10"`, `"3.11"`); `pip install ".[server]"`; added `httpx` |
| `tests/test_events.py` | Fixed `TraceEvent.model_validate()` → `trace_event_adapter.validate_python()` |
| `tests/test_hooks.py` | Replaced stub test with real attach/detach/idempotency tests |
| `tests/test_tracer.py` | Replaced `ModelTracer` stub tests with `Tracer` state machine tests |

---

## Known Issues / Gotchas

1. **transformers version sensitivity** — `normalize_past_key_values()` in `adapters/base.py` handles three historical cache formats. If transformers changes again, this is the first place to check. The installed env uses transformers 5.7 (DynamicCache with `.layers[i].keys/.values`).

2. **`mypy strict` not yet enforced in CI** — CI runs `mypy` but there are likely remaining strict violations in the `analysis/` stubs (all `Any`-typed). Fix before Week 4 packaging.

3. **`analysis/` layer is all stubs** — `KVCacheAnalyzer`, `MemoryProfiler`, `QuantizationEstimator` in `src/llmscope/analysis/` are `raise NotImplementedError`. Coverage 0%. These are Week 2–3 work.

4. **`cli.py` is untouched** — The `llmscope` CLI entry point is still a placeholder. Coverage 0%.

5. **`Tracer.dashboard()` and `Tracer.serve()`** raise `NotImplementedError` — expected, Week 2 and Week 3 respectively.

6. **Local venv** — created at `.venv/` (Python 3.11, uv). Run tests with `.venv/bin/python -m pytest`. The system Python on this Mac is 3.14 (Homebrew) with no packages; always use `.venv`.

---

## Next Steps (Week 2)

### Priority 1: Streamlit dashboard

Architecture decision (already locked): Streamlit imports `llmscope.core` directly — no FastAPI in the middle. `tracer.dashboard()` saves state to a temp JSONL, then `subprocess.Popen(["streamlit", "run", ...])`.

Files to create/fill in:
- `src/llmscope/dashboard/app.py` — Streamlit app: KV growth line chart, memory three-way bar, token count display
- `src/llmscope/dashboard/__init__.py`
- `src/llmscope/core/tracer.py:dashboard()` — implement (currently raises `NotImplementedError`)

### Priority 2: Memory attribution (`analysis/memory.py`)

Implement `MemoryProfiler` using residual method (already designed in tracer but not exposed via analysis layer):
- `weights_bytes`: `sum(p.nelement() * p.element_size())`
- `kv_bytes`: from latest `KVCacheSnapshot.total_bytes`
- `activations`: `total_allocated - weights - kv` (clamped to 0; show N/A on CPU)

### Priority 3: `analysis/kv_cache.py`

Implement `KVCacheAnalyzer.analyze(snapshots)`:
- KV growth curve (tokens → bytes per step)
- Per-layer byte breakdown
- Outlier detection: `k_max / k_std > 10` → flag high outlier risk (used by what-if estimator)

---

## Public API (locked)

```python
from llmscope import Tracer, Config

# Context manager
with Tracer(model, config=Config(sample_every_n_steps=2)) as tracer:
    model.generate(input_ids, max_new_tokens=50)

tracer.summary()            # prints KV growth + memory breakdown
tracer.save("run.jsonl")    # explicit write, all events as JSONL

# Explicit start/stop (Colab cross-cell)
tracer = Tracer(model)
tracer.start()
# ... other cells ...
tracer.stop()

tracer.dashboard()          # Week 2: launches Streamlit
tracer.serve()              # Week 3: launches FastAPI
```

## Architecture Decisions (locked, don't re-debate)

- Hook level: **model backbone** (`model.transformer` for GPT-2, `model.model` for Llama etc.), not per-attention-layer — gives atomic per-step `past_key_values` without synchronization complexity
- Frontend: **Streamlit imports core directly** (no FastAPI in the middle for v0.1)
- Storage: **in-memory `deque` ring buffer** (hot path); explicit `.save()` for JSONL
- Memory attribution: **residual method** (`total - weights - kv`); CPU → show N/A
- Quantization estimator: **lookup table** from papers (LLM.int8, GPTQ/AWQ, KIVI); auto-uprank risk if `k_max/k_std > 10`
- `tokenizer` is NOT in `Tracer.__init__` — optional arg to `summary(tokenizer=...)`
