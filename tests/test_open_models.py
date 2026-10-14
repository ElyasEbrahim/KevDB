import asyncio
import json

import httpx
import pytest

from kevdb import JevAPIError
from kevdb.client import JevAuthError, detect_provider

from kevdb import OPEN_MODELS, Backend, OpenModelClient

Q = {"q": {"type": "noul", "instructions": "Is it urgent?"}}
ANSWER = {"model": "served", "answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 100, "output_tokens": 0}}


def ask(name, handler, questions=Q, **kwargs):
    async def go():
        async with OpenModelClient(OPEN_MODELS[name], transport=httpx.MockTransport(handler), max_retries=0, **kwargs) as client:
            return await client.ask({"a": 1}, questions)

    return asyncio.run(go())


def test_local_server_needs_no_key_and_gets_the_same_request_as_jev(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=ANSWER)

    resp = ask("kev-4b", handler)
    assert resp["answers"]["q"]["noul"] == 0.7 and resp["usage"] == {"input_tokens": 100, "output_tokens": 0, "truncated": False}
    assert str(seen[0].url) == "http://127.0.0.1:8009/v1/systemone" and "authorization" not in seen[0].headers
    assert json.loads(seen[0].content) == {"state": {"a": 1}, "model": "kev-latest", "questions": Q}


def test_cache_name_carries_the_weights_version_and_local_models_cost_nothing(tmp_path):
    handler = lambda request: httpx.Response(200, json=ANSWER)
    b = Backend("kev-4b", lambda: OpenModelClient(OPEN_MODELS["kev-4b"], transport=httpx.MockTransport(handler)), cache_path=tmp_path / "c.sqlite")
    assert b.model == "jaredpalmer/kev-4b@v1.0"
    b.ask_many([("a", Q)])
    assert (b.meter.fresh, b.meter.input_tokens, b.meter.cost) == (1, 100, 0.0)
    b.close()


def test_cloudflare_path_envelope_and_price(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"success": True, "result": {**ANSWER, "usage": {"prompt_tokens": 1_000_000}}})

    make = lambda: OpenModelClient(OPEN_MODELS["clef@cloudflare"], "token", transport=httpx.MockTransport(handler))
    b = Backend("clef@cloudflare", make, cache_path=tmp_path / "c.sqlite")
    assert b.ask_many([("a", Q)])[0]["answers"]["q"]["noul"] == 0.7
    assert str(seen[0].url) == "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
    assert seen[0].headers["authorization"] == "Bearer token" and json.loads(seen[0].content)["model"] == "clef"
    assert b.meter.cost == pytest.approx(0.24)
    b.close()


def test_truncation_reported_by_the_server_is_kept():
    usage = {"input_tokens": 600, "output_tokens": 0, "truncated": False, "truncated_questions": ["q"], "state_tokens_dropped": 0}
    assert ask("laya", lambda request: httpx.Response(200, json={**ANSWER, "usage": usage}))["usage"]["truncated"] is True


def test_cloudflare_without_credentials_stops_early(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(JevAuthError, match="CLOUDFLARE_API_TOKEN"):
        OpenModelClient(OPEN_MODELS["clef@cloudflare"])
    with pytest.raises(JevAuthError, match="CLOUDFLARE_ACCOUNT_ID"):
        OpenModelClient(OPEN_MODELS["clef@cloudflare"], "token")


@pytest.mark.parametrize("answers", [{}, {"q": {"type": "noul"}}, {"q": {"type": "noul", "noul": 1.7}}, {"q": 0.7}])
def test_unusable_answer_is_a_failure(answers):
    with pytest.raises(JevAPIError, match="no usable answer"):
        ask("laya", lambda request: httpx.Response(200, json={**ANSWER, "answers": answers}))


def test_detect_provider_only_looks_at_jev_providers(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert detect_provider() == "typesafe"
