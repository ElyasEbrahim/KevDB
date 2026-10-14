"""Batch pipeline: each row -> state -> one request with all questions -> flattened columns -> confidence routing."""

from __future__ import annotations

import asyncio
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cache import Cache, cache_key
from .client import JevAPIError, JevAuthError, JevClient, JevQuotaError
from .io import Row
from .questions import Question


@dataclass
class Task:
    name: str
    input: str
    questions: dict[str, Question]
    # Only the fields the decision needs are sent to the model (per the docs, a large irrelevant state lowers accuracy)
    state_fields: list[str]
    id_field: str = "id"
    records_path: str | None = None
    # When both are empty: the provider that has a key configured, with its default model
    provider: str | None = None
    model: str | None = None
    concurrency: int = 16
    # Each field is truncated to this many characters, so a single huge row cannot overflow the context
    max_field_chars: int = 4000

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        raw["questions"] = {name: Question.from_dict(q) for name, q in raw["questions"].items()}
        return cls(**raw)

    def state_of(self, row: Row) -> dict[str, Any]:
        state = {}
        for name in self.state_fields:
            value = row.get(name)
            if value in (None, ""):
                continue
            state[name] = value[: self.max_field_chars] if isinstance(value, str) else value
        return state


@dataclass
class Stats:
    rows: int = 0
    cached: int = 0
    errors: int = 0
    review: int = 0
    # Rows not processed because the quota ran out; re-running the same command resumes from the cache
    pending: int = 0
    quota_retry_after: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    # USD price per input token and per output token, from client.prices
    prices: tuple[float, float] = (0.0, 0.0)
    # Duration of each new request in seconds, retries included, cache hits excluded
    latencies: list[float] = field(default_factory=list)
    models: set[str] = field(default_factory=set)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * self.prices[0] + self.output_tokens * self.prices[1]

    def latency(self, quantile: float) -> float | None:
        ordered = sorted(self.latencies)
        return ordered[min(int(len(ordered) * quantile), len(ordered) - 1)] if ordered else None


def flatten(name: str, q: Question, answer: dict[str, Any]) -> tuple[Row, float]:
    """Flatten one question's answer into output columns and return the confidence used for routing."""
    if q.type == "noul":
        p = answer["noul"]
        # noul returns only P(yes) and no confidence field; distance from 0.5 stands in for it
        conf = max(p, 1 - p)
        return {name: p >= q.threshold, f"{name}_p": round(p, 4)}, conf
    conf = answer["confidence"]
    if q.type == "choice":
        return {name: answer["choice"], f"{name}_conf": round(conf, 4)}, conf
    return {name: round(answer["score"], 2), f"{name}_conf": round(conf, 4)}, conf


async def run_task(
    task: Task,
    rows: list[Row],
    client: JevClient,
    cache: Cache | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Row], Stats]:
    """Return (rows with the new columns, stats). Rows with _review=True need a human or a stronger model;
    rows with _pending=True were not processed because the quota ran out."""
    payloads = {name: q.payload() for name, q in task.questions.items()}
    stats = Stats(rows=len(rows), prices=client.prices)
    gate = asyncio.Semaphore(task.concurrency)
    done = 0
    started = time.perf_counter()

    async def one(row: Row) -> Row:
        nonlocal done
        state = task.state_of(row)
        key = cache_key(client.cache_id, state, payloads)
        out: Row = {task.id_field: row.get(task.id_field)}
        reasons: list[str] = []
        response = cache.get(key) if cache else None
        if response is not None:
            stats.cached += 1
        else:
            try:
                async with gate:
                    if stats.quota_retry_after is not None:
                        raise JevQuotaError(429, "quota exhausted", stats.quota_retry_after)
                    asked = time.perf_counter()
                    try:
                        response = await client.ask(state, payloads)
                    finally:
                        stats.latencies.append(time.perf_counter() - asked)
            except JevAuthError:
                raise
            except JevQuotaError as e:
                stats.quota_retry_after = e.retry_after
                stats.pending += 1
                return {task.id_field: row.get(task.id_field), "_pending": True}
            except JevAPIError as e:
                stats.errors += 1
                out["_error"] = str(e)
                reasons.append("error")
            else:
                stats.input_tokens += response["usage"]["input_tokens"]
                stats.output_tokens += response["usage"].get("output_tokens", 0)
                if cache:
                    cache.put(key, response)
        if response is not None:
            stats.models.add(response["model"])
            for name, q in task.questions.items():
                columns, conf = flatten(name, q, response["answers"][name])
                out.update(columns)
                if conf < q.review_below:
                    reasons.append(name)
        out["_review"] = bool(reasons)
        out["_review_reasons"] = ",".join(reasons)
        stats.review += bool(reasons)
        done += 1
        if on_progress:
            on_progress(done, len(rows))
        return out

    async with asyncio.TaskGroup() as tg:
        jobs = [tg.create_task(one(row)) for row in rows]
    stats.seconds = time.perf_counter() - started
    return [job.result() for job in jobs], stats
