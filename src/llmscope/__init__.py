"""llmscope — visualize LLM inference internals."""

from .core.config import Config
from .core.tracer import Tracer

__version__ = "0.1.0"

__all__ = ["Config", "Tracer", "__version__"]
