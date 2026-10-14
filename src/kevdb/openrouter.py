"""Call LLMs through OpenRouter. The prompt and answer parsing are those of the kevdb LLM baseline (chat-v1),
so results differ from the earlier DeepSeek and Opus runs through OpenCode only in the route, not in the questions.

Three differences from the baseline:
- the provider is pinned in the request, and a response from any other provider counts as a failure;
- cost comes from usage.cost in the response (what was actually charged), not from a price list;
- with thinking off and a single yes/no question, P(yes) is read from the output probabilities and stored in the answer's `p_true`.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from .client import PROVIDERS, JevAPIError, Provider
from .llm import SYSTEM, LLMClient, parse_answers

from .routes import Route

PROVIDERS.setdefault("openrouter", Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-6-luna"))


def p_true(logprobs: dict[str, Any] | None) -> float | None:
    """At the position of the first true/false in the answer JSON: the share of "true" in the combined probability of true and false."""
    for token in (logprobs or {}).get("content") or []:
        if token["token"].strip().lower() not in ("true", "false"):
            continue
        mass = {"true": 0.0, "false": 0.0}
        for alt in token.get("top_logprobs") or [token]:
            word = alt["token"].strip().lower()
            if word in mass:
                mass[word] += math.exp(alt["logprob"])
        total = mass["true"] + mass["false"]
        return mass["true"] / total if total else None
    return None


class OpenRouterClient(LLMClient):
    def __init__(self, route: Route, api_key: str | None = None, *, body_retries: int = 2, **kwargs: Any) -> None:
        kwargs.setdefault("timeout", 120.0)
        super().__init__(api_key, provider="openrouter", model=route.model, reasoning=None, **kwargs)
        self.route = route
        self.body_retries = body_retries
        # Can be lowered temporarily: OpenRouter reserves cost per in-flight request based on max_tokens and rejects requests (402) when a low balance cannot cover the total
        self.max_tokens = route.max_tokens

    @property
    def cache_id(self) -> str:
        r = self.route
        return f"openrouter:{r.model}@{r.provider}#{self.PROMPT_VERSION}#{json.dumps(r.reasoning, sort_keys=True)}"

    @property
    def prices(self) -> tuple[float, float]:
        return 0.0, 0.0  # cost is taken from usage.cost

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        route = self.route
        body: dict[str, Any] = {
            "model": route.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"state": state, "questions": questions}, ensure_ascii=False)},
            ],
            "provider": {"order": [route.provider], "allow_fallbacks": False},
        }
        if route.temperature is not None:
            body["temperature"] = route.temperature
        if route.reasoning is not None:
            body["reasoning"] = route.reasoning
        if route.logprobs:
            body.update(logprobs=True, top_logprobs=5)
        for attempt in range(self.body_retries + 1):
            resp = await self._post("/chat/completions", body)
            # When a provider fails, OpenRouter may still return 200 with the error in the body
            if resp.get("choices") and not resp.get("error"):
                break
            if attempt == self.body_retries:
                raise JevAPIError(200, f"no answer in the response: {json.dumps(resp, ensure_ascii=False)[:200]}")
            await asyncio.sleep(1 + attempt)
        if resp.get("provider") != route.provider:
            raise JevAPIError(200, f"expected {route.provider} to answer, got {resp.get('provider')!r}")
        choice = resp["choices"][0]
        answers = parse_answers(choice["message"].get("content") or "", questions)
        noul = [name for name, q in questions.items() if q["type"] == "noul"]
        if route.logprobs and len(questions) == 1 and noul:
            answers[noul[0]]["p_true"] = p_true(choice.get("logprobs"))
        usage = resp.get("usage") or {}
        return {
            "model": f"{route.model}@{route.provider}",
            "answers": answers,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                "cost": usage.get("cost"),
            },
        }
