"""Pinned routes on OpenRouter.

One model ID is served by ten to forty providers with different quantisation, routed by price by default (6 requests landed on 5 providers).
So every route pins its provider and forbids fallback, which makes results reproducible.

The `logprobs` column was tested provider by provider on 2026-10-07: providers listed as supporting it do not always return it (Novita and NextBit do not).
With thinking on, the probability of the answer token collapses to 0 or 1, so probabilities are read only on routes with thinking off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OFF: dict[str, Any] = {"enabled": False}
LOW: dict[str, Any] = {"effort": "low"}


@dataclass(frozen=True)
class Route:
    name: str
    model: str
    provider: str
    # None: send no reasoning field and use the model's default thinking
    reasoning: dict[str, Any] | None = field(default_factory=lambda: dict(OFF))
    # Whether this provider really returns output probabilities with thinking off
    logprobs: bool = False
    # None: send no temperature (Anthropic rejects a temperature while thinking is on)
    temperature: float | None = 0

    @property
    def thinking(self) -> bool:
        return self.reasoning is None or (self.reasoning.get("enabled") is not False and self.reasoning.get("effort") != "none")

    @property
    def max_tokens(self) -> int:
        # Thinking counts against max_tokens; too small a budget can leave no room for the answer.
        # It started at 2000: in the qualification test low-effort thinking sometimes used over a thousand tokens on one question and 1%-3% of answers came back empty, so it was raised
        return 8000 if self.thinking else 1000  # with thinking off this still leaves room to answer 21 questions at once


ROUTES = {
    r.name: r
    for r in (
        Route("v4pro.off", "deepseek/deepseek-v4-pro", "Parasail", OFF, logprobs=True),
        Route("v4pro.low", "deepseek/deepseek-v4-pro", "Parasail", LOW),
        Route("v41flash.off", "deepseek/deepseek-v4.1-flash", "Together", OFF, logprobs=True),
        Route("v41flash.low", "deepseek/deepseek-v4.1-flash", "Together", LOW),
        Route("qwen38flash.off", "qwen/qwen3.8-flash", "Alibaba", OFF, logprobs=True),
        Route("qwen38flash.low", "qwen/qwen3.8-flash", "Alibaba", LOW),
        # OpenAI returns no output probabilities; its "thinking off" is effort=none
        Route("luna.off", "openai/gpt-6-luna", "OpenAI", {"effort": "none"}),
        Route("luna.low", "openai/gpt-6-luna", "OpenAI", LOW),
        # The stronger upper-tier model with its default settings (adaptive thinking), as when it served as the reference on product matching
        Route("opus", "anthropic/claude-opus-5.5", "Anthropic", None, temperature=None),
    )
}
