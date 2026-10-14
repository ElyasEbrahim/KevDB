"""Open, Jev-compatible models: they accept the same request (state plus typed questions) and also return probabilities.

The request body is exactly Jev's, so JevClient is reused with three differences:
- endpoint: kev, laya and clef-flash are servers started locally (scripts/open_models.sh installs and starts them);
  names containing @cloudflare are Clef hosted on Cloudflare Workers AI, which is paid and was not used in phase 6;
- the cache name includes the weight version: a local server's request body only says `kev-latest`, which hides the weights behind it;
- cost: local models are free; hosted ones cost the list price times the input tokens they report.

Versions and prices were checked on 2026-10-09 and 10-10. Workers AI has no pinnable version, so the name records only the month.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .client import PROVIDERS, JevAPIError, JevAuthError, JevClient, Provider

# Ports match scripts/open_models.sh. The two Kev models use different ports, so a request for 4B can never reach the 9B server.
# laya defaults to 8000, which clashes with many dev servers, so it moved to 8011
PROVIDERS.setdefault("kev-4b", Provider("http://127.0.0.1:8009/v1", None, "kev-latest"))
PROVIDERS.setdefault("kev-9b", Provider("http://127.0.0.1:8010/v1", None, "kev-latest"))
PROVIDERS.setdefault("laya", Provider("http://127.0.0.1:8011/v1", None, "laya"))
PROVIDERS.setdefault("clef-flash", Provider("http://127.0.0.1:8012/v1", None, "clef-flash"))
PROVIDERS.setdefault("laya-multilingual", Provider("http://127.0.0.1:8013/v1", None, "laya-multilingual"))
PROVIDERS.setdefault("kev-0.8b", Provider("http://127.0.0.1:8014/v1", None, "kev-latest"))
PROVIDERS.setdefault("cloudflare", Provider("https://api.cloudflare.com/client/v4", "CLOUDFLARE_API_TOKEN", "clef"))


@dataclass(frozen=True)
class OpenModel:
    provider: str
    # model field in the request body
    model: str
    # name used in the cache key
    version: str
    # USD per million input tokens; output is free
    usd_per_m_input: float = 0.0
    path: str = "/systemone"


OPEN_MODELS = {
    "kev-4b": OpenModel("kev-4b", "kev-latest", "jaredpalmer/kev-4b@v1.0"),
    "kev-9b": OpenModel("kev-9b", "kev-latest", "jaredpalmer/kev-9b@v1.0"),
    # Added in phase 7: the smallest one; its author reports it is clearly weaker than 4B on unseen data, but about 5x faster
    "kev-0.8b": OpenModel("kev-0.8b", "kev-latest", "jaredpalmer/kev-0.8b@v1.0"),
    # Both laya versions were behind Jev on all three datasets in phase 6 and were dropped; not installed by default. Kept only to reproduce that run
    "laya": OpenModel("laya", "laya", "convaiinnovations/laya#laya-0.4.1"),
    # Multilingual weights from the same repository (subfolder multilingual)
    "laya-multilingual": OpenModel("laya-multilingual", "laya-multilingual", "convaiinnovations/laya/multilingual#laya-0.4.1"),
    # Weights and the bundled loading code are pinned to this commit; served locally by scripts/clef_server.py
    "clef-flash": OpenModel("clef-flash", "clef-flash", "Cloudflare/clef-flash@8b2e5fd17c09"),
    # Prices from https://developers.cloudflare.com/workers-ai/platform/pricing/
    "clef@cloudflare": OpenModel("cloudflare", "clef", "@cf/cloudflare/clef#2026-10", 0.24, "/accounts/{account}/ai/run/@cf/cloudflare/clef"),
    "clef-flash@cloudflare": OpenModel("cloudflare", "clef-flash", "@cf/cloudflare/clef-flash#2026-10", 0.09, "/accounts/{account}/ai/run/@cf/cloudflare/clef-flash"),
}


class OpenModelClient(JevClient):
    def __init__(self, spec: OpenModel, api_key: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("timeout", 120.0)  # under high concurrency a local model queues requests
        super().__init__(api_key, provider=spec.provider, model=spec.model, **kwargs)
        self.spec = spec
        self._path = spec.path
        if "{account}" in spec.path:
            account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            if not account:
                raise JevAuthError(401, "no Cloudflare account ID found; set CLOUDFLARE_ACCOUNT_ID in .env or in the environment")
            self._path = spec.path.format(account=account)

    @property
    def cache_id(self) -> str:
        return self.spec.version

    @property
    def prices(self) -> tuple[float, float]:
        return self.spec.usd_per_m_input / 1e6, 0.0

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        resp = await self._post(self._path, {"state": state, "model": self.model, "questions": questions})
        if "answers" not in resp and isinstance(resp.get("result"), dict):
            resp = resp["result"]  # Workers AI wraps the answer in result
        answers = resp.get("answers") or {}
        # Incomplete answers or probabilities that are not probabilities count as failures and are not cached, so a bad answer is never replayed
        for name, q in questions.items():
            answer = answers.get(name)
            p = answer.get("noul") if isinstance(answer, dict) else None
            if not isinstance(answer, dict) or (q["type"] == "noul" and not (isinstance(p, (int, float)) and 0 <= p <= 1)):
                raise JevAPIError(200, f"no usable answer for question {name!r}: {str(answer)[:120]}")
        usage = resp.get("usage") or {}
        return {
            "model": resp.get("model", self.model),
            "answers": answers,
            "usage": {
                "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                # The server's own report that part of this request was truncated (laya does not fail beyond its window; it only says so here)
                "truncated": bool(usage.get("truncated") or usage.get("truncated_questions") or usage.get("state_tokens_dropped")),
            },
        }
