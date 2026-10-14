"""The three kinds of backend used in the experiments. Keys are read from the environment (call load_dotenv first)."""

from __future__ import annotations

from typing import Any

from .client import JevClient

from .engine import Backend
from .open_models import OPEN_MODELS, OpenModelClient
from .openrouter import OpenRouterClient
from .routes import ROUTES

JEV_MODEL = "jev-1.13.0"  # pinned to the patch version so results are reproducible


def jev_backend(**kwargs: Any) -> Backend:
    """Jev always goes through the official API. Throughput stops increasing above a concurrency of 16."""
    return Backend("jev", lambda: JevClient(provider="typesafe", model=JEV_MODEL), **kwargs)


def llm_backend(route: str, **kwargs: Any) -> Backend:
    return Backend(route, lambda: OpenRouterClient(ROUTES[route]), **kwargs)


def open_backend(name: str, **kwargs: Any) -> Backend:
    """Open, Jev-compatible models (kev-4b, clef, laya, ...; see open_models.py). Local ones need their server running first."""
    return Backend(name, lambda: OpenModelClient(OPEN_MODELS[name]), **kwargs)
