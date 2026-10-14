import asyncio
import json
import math

import httpx
import pytest

from kevdb import JevAPIError

from kevdb import ROUTES, OpenRouterClient
from kevdb.openrouter import p_true

Q = {"same": {"type": "noul", "instructions": "Same product?"}}


def tok(token, alts):
    return {"token": token, "logprob": math.log(alts[token]), "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in alts.items()]}


LOGPROBS = {"content": [tok('{"', {'{"': 1.0}), tok("same", {"same": 1.0}), tok(" true", {" true": 0.6, " false": 0.2, "true": 0.1, " maybe": 0.1})]}


def reply(content='{"same": true}', provider="Parasail", **extra):
    return {"provider": provider, "choices": [{"message": {"content": content}, "logprobs": LOGPROBS}],
            "usage": {"prompt_tokens": 400, "completion_tokens": 6, "cost": 0.00021, "completion_tokens_details": {"reasoning_tokens": 0}}, **extra}


def ask(route, handler, questions=Q):
    async def go():
        async with OpenRouterClient(ROUTES[route], "k", transport=httpx.MockTransport(handler), max_retries=0) as client:
            return await client.ask({"a": 1}, questions)

    return asyncio.run(go())


def test_p_true_is_the_share_of_true_among_true_and_false():
    assert p_true(LOGPROBS) == pytest.approx(0.7 / 0.9)
    assert p_true(None) is None and p_true({"content": [tok("yes", {"yes": 1.0})]}) is None


def test_request_pins_the_provider_and_reads_probability_and_real_cost():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply())

    resp = ask("v4pro.off", handler)
    body = seen[0]
    assert body["provider"] == {"order": ["Parasail"], "allow_fallbacks": False}
    assert body["reasoning"] == {"enabled": False} and body["logprobs"] is True and body["temperature"] == 0
    assert json.loads(body["messages"][1]["content"]) == {"state": {"a": 1}, "questions": Q}
    assert resp["answers"]["same"]["noul"] == 1.0 and resp["answers"]["same"]["p_true"] == pytest.approx(0.7 / 0.9)
    assert resp["usage"] == {"input_tokens": 400, "output_tokens": 6, "reasoning_tokens": 0, "cost": 0.00021}

    # Routes with thinking on must not request output probabilities: the answer's probability would only be 0 or 1
    seen.clear()
    resp = ask("v4pro.low", handler)
    assert "logprobs" not in seen[0] and seen[0]["reasoning"] == {"effort": "low"} and seen[0]["max_tokens"] == 8000
    assert "p_true" not in resp["answers"]["same"]

    # Routes using the model's default thinking send neither reasoning nor temperature
    seen.clear()
    ask("opus", lambda request: (seen.append(json.loads(request.content)), httpx.Response(200, json=reply(provider="Anthropic")))[1])
    assert "reasoning" not in seen[0] and "temperature" not in seen[0] and seen[0]["provider"]["order"] == ["Anthropic"]


def test_wrong_provider_or_empty_body_is_an_error():
    with pytest.raises(JevAPIError, match="Parasail"):
        ask("v4pro.off", lambda request: httpx.Response(200, json=reply(provider="Novita")))

    # A 200 with an error in the body is retried; if it persists, it raises
    replies = [{"error": {"message": "upstream"}}, reply()]
    assert ask("v4pro.off", lambda request: httpx.Response(200, json=replies.pop(0)))["answers"]["same"]["noul"] == 1.0
    with pytest.raises(JevAPIError, match="no answer"):
        ask("luna.off", lambda request: httpx.Response(200, json={"error": {"message": "upstream"}}))
