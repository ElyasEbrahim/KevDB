"""Offline fake backend: responses have the same structure as the real API, but answers are derived from a hash and carry **no meaning at all**.

Use it only to exercise the pipeline and write tests without a key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

MOCK_MODEL = "mock-jev"


def _unit(*parts: str) -> float:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _distribution(seed: str, keys: list[str]) -> dict[str, float]:
    weights = [_unit(seed, k) ** 4 + 1e-6 for k in keys]
    total = sum(weights)
    return {k: w / total for k, w in zip(keys, weights)}


def _answer(seed: str, q: dict[str, Any]) -> dict[str, Any]:
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(_unit(seed), 4)}
    if q["type"] == "choice":
        probs = _distribution(seed, list(q["criteria"]))
        choice = max(probs, key=probs.__getitem__)
        return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": round(probs[choice], 4)}
    levels = [str(i) for i in range(len(q["criteria"]))]
    probs = _distribution(seed, levels)
    return {
        "type": "score",
        "score": round(sum(int(k) * p for k, p in probs.items()), 4),
        "legend": dict(zip(levels, q["criteria"])),
        "probabilities": probs,
        "confidence": round(max(probs.values()), 4),
    }


def _handle(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    state = json.dumps(body["state"], ensure_ascii=False, sort_keys=True)
    answers = {name: _answer(f"{state}|{name}", q) for name, q in body["questions"].items()}
    usage = {"input_tokens": len(request.content) // 4, "output_tokens": 0}
    return httpx.Response(200, json={"model": MOCK_MODEL, "answers": answers, "usage": usage})


def mock_transport() -> httpx.MockTransport:
    return httpx.MockTransport(_handle)
