"""LLM baseline: a small generative model answers the same state and the same questions, for comparison with Jev.

The interface matches JevClient (ask returns {model, answers, usage}), so the pipeline, cache and evaluation are reused as is.
An LLM gives answers but no calibrated probabilities, so confidence is always recorded as 1.0: confidence routing does nothing for it.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .client import JevAPIError, JevClient

# OpenCode Zen list prices, USD per million tokens (input, output), checked 2026-10-02 at https://opencode.ai/docs/zen/
LLM_PRICES = {
    "deepseek-v4.1-flash": (0.30, 1.20),
    "glm-5.3-flash": (0.15, 0.50),
    "claude-opus-5-5": (4.00, 20.00),  # checked 2026-10-05
}

SYSTEM = (
    "You label data. You are given a JSON object with `state` and `questions`. "
    "Answer every question about the state, following each question's instructions and criteria exactly. "
    "Reply with one JSON object and nothing else, mapping each question name to its answer: "
    'for type "choice", one of the option keys in `criteria` (a string); '
    'for type "score", the 0-based index of the best matching level in `criteria` (an integer); '
    'for type "noul", true or false.'
)


class LLMClient(JevClient):
    PROMPT_VERSION = "chat-v1"

    def __init__(self, *args: Any, reasoning: str | None = "none", **kwargs: Any) -> None:
        """reasoning: passed as reasoning_effort. The default "none" turns thinking off (the usual setting for classification;
        measured: from ~13 s / ~1,000 output tokens down to ~2 s / ~30 per row). None omits the field and uses the model default (thinking on)."""
        super().__init__(*args, **kwargs)
        self.reasoning = reasoning

    @property
    def cache_id(self) -> str:
        return f"{self.model}#{self.PROMPT_VERSION}#{self.reasoning or 'default'}"

    @property
    def prices(self) -> tuple[float, float]:
        usd_in, usd_out = LLM_PRICES.get(self.model, (0.0, 0.0))
        return usd_in / 1e6, usd_out / 1e6

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        prompt = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
        if self.model.startswith("claude-"):
            return await self._ask_messages(prompt, questions)
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        }
        if self.reasoning:
            body["reasoning_effort"] = self.reasoning
        resp = await self._post("/chat/completions", body)
        content = resp["choices"][0]["message"].get("content") or ""
        usage = resp.get("usage", {})
        return {
            "model": self.model,
            "answers": parse_answers(content, questions),
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)},
        }


    async def _ask_messages(self, prompt: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Opus goes through the platform's /messages endpoint: the request body and the auth header differ.
        Opus 5.5 always thinks adaptively and cannot turn it off; thinking counts against max_tokens, so a small budget can leave no room for the answer."""
        resp = await self._post(
            "/messages",
            {"model": self.model, "max_tokens": 2000, "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]},
            headers={"x-api-key": self._api_key, "anthropic-version": "2023-06-01"},
        )
        content = "".join(block.get("text", "") for block in resp["content"] if block.get("type") == "text")
        usage = resp.get("usage", {})
        return {
            "model": self.model,
            "answers": parse_answers(content, questions),
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)},
        }


def _level_index(value: Any, levels: list[str]) -> int | None:
    """Score questions expect the level index, but LLMs often reply with the level name ("Low") or the full description.
    deepseek-v4.1-flash with thinking off did this on 58% of rows, so any reply that matches exactly one level is accepted."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text.isdigit():
            value = int(text)
        else:
            hits = [i for i, level in enumerate(levels) if level.lower() == text or level.lower().startswith(text + ":")]
            value = hits[0] if len(hits) == 1 else None
    return value if type(value) is int and 0 <= value < len(levels) else None


def parse_answers(content: str, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Parse an LLM's text reply and validate it into Jev's answer structure; malformed replies raise, and the row counts as failed."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    try:
        raw = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        raw = None
    if not isinstance(raw, dict):
        raise JevAPIError(200, f"LLM did not return a JSON object: {content[:120]!r}")
    answers: dict[str, dict[str, Any]] = {}
    for name, q in questions.items():
        value = raw.get(name)
        if q["type"] == "choice" and value in q["criteria"]:
            answers[name] = {"type": "choice", "choice": value, "probabilities": {value: 1.0}, "confidence": 1.0}
        elif q["type"] == "score" and (level := _level_index(value, q["criteria"])) is not None:
            answers[name] = {"type": "score", "score": float(level), "probabilities": {str(level): 1.0}, "confidence": 1.0}
        elif q["type"] == "noul" and isinstance(value, bool):
            answers[name] = {"type": "noul", "noul": float(value)}
        else:
            raise JevAPIError(200, f"invalid LLM answer for {name!r}: {value!r}")
    return answers
