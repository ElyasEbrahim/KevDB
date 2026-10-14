"""Decision models as DuckDB SQL functions, with open-source models as first-class backends."""

from . import calibration
from .backends import JEV_MODEL, jev_backend, llm_backend, open_backend
from .client import JevAPIError, JevAuthError, JevClient
from .engine import Backend, Meter
from .open_models import OPEN_MODELS, OpenModel, OpenModelClient
from .openrouter import OpenRouterClient
from .pipeline import Task, run_task
from .questions import Question
from .routes import ROUTES, Route
from .sql import register, register_cascade

__all__ = [
    "Backend", "JEV_MODEL", "JevAPIError", "JevAuthError", "JevClient", "Meter", "OPEN_MODELS", "OpenModel",
    "OpenModelClient", "OpenRouterClient", "Question", "ROUTES", "Route", "Task", "calibration", "jev_backend",
    "llm_backend", "open_backend", "register", "register_cascade", "run_task",
]
