"""HTTP client for the decision-model API: POST /v1/systemone, with retries on rate limiting and overload."""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import Any

import httpx

@dataclass(frozen=True)
class Provider:
    base_url: str
    # None: a local server that needs no key
    env_key: str | None
    default_model: str


# Both expose the same /systemone endpoint and differ only in URL, key and model ID.
# Real tasks always use the official typesafe API: it has half the latency and no rate limit (see experiments/table_match/route_speed.py).
# opencode is a third-party platform, used only for comparison. Models are pinned to the patch version so results are reproducible.
PROVIDERS = {
    "typesafe": Provider("https://api.typesafe.ai/v1", "TYPESAFE_API_KEY", "jev-1.13.0"),
    "opencode": Provider("https://opencode.ai/zen/v1", "OPENCODE_API_KEY", "jev-1.13"),
}
# Jev list price: $42 per billion input tokens; output is free
USD_PER_INPUT_TOKEN = 42 / 1e9
# 429 (rate limited) and 529 (overloaded) are retried with backoff as the docs require; 5xx is retried too
RETRY_STATUS = {429, 500, 502, 503, 504, 529}


def detect_provider() -> str:
    """With no provider given, use the one that has a key in the environment. Only the two Jev providers are considered: other modules also register LLM and open-model endpoints in PROVIDERS."""
    for name in ("typesafe", "opencode"):
        if os.environ.get(PROVIDERS[name].env_key):
            return name
    return "typesafe"


class JevAPIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class JevAuthError(JevAPIError):
    """The key is missing or invalid; retrying is pointless and the whole task should stop."""


class JevQuotaError(JevAPIError):
    """Quota exhausted: the server asks to wait longer than max_wait, so no new requests should be sent."""

    def __init__(self, status: int, message: str, retry_after: float) -> None:
        super().__init__(status, message)
        self.retry_after = retry_after


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
        max_retries: int = 5,
        backoff: float = 0.5,
        max_wait: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.max_wait = max_wait
        self.provider = provider or detect_provider()
        if self.provider not in PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}, expected one of {tuple(PROVIDERS)}")
        config = PROVIDERS[self.provider]
        api_key = api_key or (os.environ.get(config.env_key) if config.env_key else None)
        if not api_key and config.env_key:
            raise JevAuthError(401, f"no API key found; set {config.env_key} in .env or in the environment")
        self.model = model or config.default_model
        self._api_key = api_key
        self.max_retries = max_retries
        self.backoff = backoff
        self._http = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "JevClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()

    @property
    def cache_id(self) -> str:
        """The part of the cache key that says who answered."""
        return self.model

    @property
    def prices(self) -> tuple[float, float]:
        """USD price per input token and per output token."""
        return (0.0 if self.model.endswith("-free") else USD_PER_INPUT_TOKEN), 0.0

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """One state plus several questions in one request. Returns the raw response {model, answers, usage}."""
        return await self._post("/systemone", {"state": state, "model": self.model, "questions": questions})

    async def _post(self, path: str, body: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                resp = await self._http.post(path, json=body, headers=headers)
            except httpx.TransportError as e:
                if last:
                    raise JevAPIError(0, f"network error: {e}") from e
                await self._sleep(attempt, None)
                continue
            if resp.status_code == 200:
                return resp.json()
            # 402 means insufficient balance: like an invalid key, every further request would fail, so stop the whole task
            if resp.status_code in (401, 402, 403):
                raise JevAuthError(resp.status_code, resp.text[:300])
            retry_after = _seconds(resp.headers.get("retry-after"))
            if retry_after > self.max_wait:
                raise JevQuotaError(resp.status_code, resp.text[:300], retry_after)
            if resp.status_code in RETRY_STATUS and not last:
                await self._sleep(attempt, retry_after)
                continue
            raise JevAPIError(resp.status_code, resp.text[:300])
        raise AssertionError("unreachable")

    async def _sleep(self, attempt: int, retry_after: float | None) -> None:
        delay = retry_after or self.backoff * 2**attempt * (1 + random.random() * 0.25)
        await asyncio.sleep(delay)


def _seconds(header: str | None) -> float:
    try:
        return float(header) if header else 0.0
    except ValueError:
        return 0.0
