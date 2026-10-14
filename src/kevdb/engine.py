"""A model backend: concurrency, caching and metering.

DuckDB calls SQL functions synchronously on its own threads, while model requests are asynchronous and concurrent, so the
backend runs its own background event loop: a SQL function hands in a batch of requests and waits for the results. The
SQLite connection used by the cache is only ever used on that loop's thread.

Cache key = (namespace + model and its configuration, state, questions). Each key is paid for once;
to time a cold run, use a fresh namespace.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cache import Cache, cache_key
from .client import JevAPIError, JevAuthError, JevClient

Item = tuple[Any, dict[str, dict[str, Any]]]  # (state, questions)


@dataclass
class Meter:
    """Usage within one metering interval (usually one query). A request that appears several times in the interval counts once."""

    calls: int = 0  # distinct requests
    fresh: int = 0  # of which actually sent this time and answered
    input_tokens: int = 0
    output_tokens: int = 0
    # Cost of these requests. Cache hits count at what they originally cost, so this equals the cost of a cold run
    cost: float = 0.0
    # Duration of each request in seconds, retries included; cache hits use their original duration
    latencies: list[float] = field(default_factory=list)
    seen: set[str] = field(default_factory=set, repr=False)
    # Requests that have no answer yet. Failures are not cached; a successful retry in the same interval removes them from here
    failed: set[str] = field(default_factory=set, repr=False)

    @property
    def errors(self) -> int:
        return len(self.failed)

    def latency(self, quantile: float) -> float | None:
        ordered = sorted(self.latencies)
        return ordered[min(int(len(ordered) * quantile), len(ordered) - 1)] if ordered else None


class Backend:
    def __init__(
        self,
        name: str,
        make_client: Callable[[], JevClient],
        *,
        concurrency: int = 16,
        cache_path: str | Path | None = ".jevcache.sqlite",
        namespace: str = "",
    ) -> None:
        self.name = name
        self.namespace = namespace
        self.meter = Meter()
        # New spending by this backend in the whole process (cache hits excluded), to keep an eye on total cost
        self.spent = 0.0
        self._inflight: dict[str, asyncio.Future] = {}
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name=f"backend-{name}", daemon=True).start()

        async def start() -> None:
            self._client = make_client()
            self._cache = Cache(cache_path) if cache_path else None
            self._gate = asyncio.Semaphore(concurrency)

        self._call(start())

    def _call(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @property
    def model(self) -> str:
        return self._client.cache_id

    @property
    def client(self) -> JevClient:
        return self._client

    def reset(self) -> Meter:
        """Start a new metering interval and return the usage of the previous one."""
        old, self.meter = self.meter, Meter()
        return old

    def ask_many(self, items: list[Item]) -> list[dict[str, Any]]:
        """Synchronous interface, callable from any thread. Results line up with items; failed requests return {"_error": ...}."""
        return self._call(self._ask_many(items))

    def close(self) -> None:
        async def stop() -> None:
            await self._client.__aexit__()
            if self._cache:
                self._cache.close()

        self._call(stop())
        self._loop.call_soon_threadsafe(self._loop.stop)

    async def _ask_many(self, items: list[Item]) -> list[dict[str, Any]]:
        keys = [cache_key(self.namespace + self._client.cache_id, state, questions) for state, questions in items]
        unique = dict(zip(keys, items))
        answers = dict(zip(unique, await asyncio.gather(*(self._one(k, *item) for k, item in unique.items()))))
        meter = self.meter
        for key, resp in answers.items():
            if key in meter.seen and key not in meter.failed:
                continue
            if key not in meter.seen:
                meter.seen.add(key)
                meter.calls += 1
            if "_error" in resp:
                meter.failed.add(key)
                continue
            meter.failed.discard(key)
            meter.fresh += not resp["_cached"]
            meter.input_tokens += resp["usage"]["input_tokens"]
            meter.output_tokens += resp["usage"].get("output_tokens", 0)
            meter.cost += resp["usage"]["cost"]
            meter.latencies.append(resp["_latency"])
        return [answers[k] for k in keys]

    async def _one(self, key: str, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        hit = self._cache.get(key) if self._cache else None
        if hit is not None:
            return {**hit, "_cached": True}
        if key in self._inflight:  # another batch is already asking this request; wait for its result
            return {**await asyncio.shield(self._inflight[key]), "_cached": True}
        waiting = self._inflight[key] = self._loop.create_future()
        try:
            async with self._gate:
                asked = time.perf_counter()
                try:
                    resp = await self._client.ask(state, questions)
                except JevAuthError:
                    raise  # invalid key or insufficient balance: every further request would fail, so stop the whole task
                except JevAPIError as e:
                    resp = {"_error": str(e)}
                latency = time.perf_counter() - asked
            if "_error" not in resp:
                usage = resp["usage"]
                if usage.get("cost") is None:
                    usd_in, usd_out = self._client.prices
                    usage["cost"] = usage["input_tokens"] * usd_in + usage.get("output_tokens", 0) * usd_out
                resp["_latency"] = latency
                self.spent += usage["cost"]
                if self._cache:
                    self._cache.put(key, resp)
            waiting.set_result(resp)
            return {**resp, "_cached": False}
        except BaseException as e:
            waiting.set_exception(e)
            waiting.exception()  # mark the exception as retrieved, so an unobserved failure does not log a warning
            raise
        finally:
            del self._inflight[key]
