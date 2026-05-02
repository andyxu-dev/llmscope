"""Hook management for instrumenting PyTorch model attention layers."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .adapters.base import ArchitectureAdapter, PastKeyValues

SnapshotCallback = Callable[[PastKeyValues, int], None]


class HookManager:
    """Attach and remove forward hooks on the adapter-specified module."""

    def __init__(
        self,
        model: nn.Module,
        adapter: ArchitectureAdapter,
        on_snapshot: SnapshotCallback,
        sample_every_n_steps: int = 1,
    ) -> None:
        self._model = model
        self._adapter = adapter
        self._on_snapshot = on_snapshot
        self._sample_every = sample_every_n_steps
        self._step_count = 0
        self._handles: list[Any] = []  # torch RemovableHandle

    @property
    def is_attached(self) -> bool:
        return len(self._handles) > 0

    def attach(self) -> None:
        """Register the forward hook on the adapter's hook module."""
        if self.is_attached:
            return
        hook_module = self._adapter.get_hook_module(self._model)
        handle = hook_module.register_forward_hook(self._hook)
        self._handles.append(handle)

    def detach(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset_step_count(self) -> None:
        self._step_count = 0

    def _hook(self, module: nn.Module, input: Any, output: Any) -> None:
        step_index = self._step_count
        self._step_count += 1

        if step_index % self._sample_every != 0:
            return

        pkv = self._adapter.extract_past_key_values(output)
        if pkv is not None and len(pkv) > 0:
            self._on_snapshot(pkv, step_index)
