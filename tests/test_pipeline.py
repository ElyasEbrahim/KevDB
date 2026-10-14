import asyncio
import json

import httpx
import pytest

from kevdb import JevAuthError, JevClient, Question, Task, run_task
from kevdb.cache import Cache
from kevdb.mock import mock_transport
from kevdb.pipeline import flatten

QUESTIONS = {
    "kind": Question("choice", "What kind?", {"a": "A", "b": "B"}, review_below=0.9),
    "depth": Question("score", "How deep?", ["none", "some", "lots"]),
    "urgent": Question("noul", "Is it urgent?"),
}
ROWS = [{"id": i, "text": f"row {i}", "secret": "do not send"} for i in range(8)]


def make_task(**overrides):
    return Task(name="t", input="unused", questions=QUESTIONS, state_fields=["text"], **overrides)


def client_with(handler, **kwargs):
    return JevClient("k", transport=httpx.MockTransport(handler), backoff=0.001, **kwargs)


def fixed_response(request):
    answers = {
        "kind": {"type": "choice", "choice": "a", "probabilities": {"a": 0.95, "b": 0.05}, "confidence": 0.95},
        "depth": {"type": "score", "score": 1.4, "legend": {}, "probabilities": {}, "confidence": 0.7},
        "urgent": {"type": "noul", "noul": 0.2},
    }
    return httpx.Response(200, json={"model": "jev-test", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 0}})


def run(task, client, cache=None, rows=ROWS):
    async def go():
        async with client:
            return await run_task(task, rows, client, cache)

    return asyncio.run(go())


def test_question_validation():
    with pytest.raises(ValueError):
        Question("choice", "x", {"only": "one"})
    with pytest.raises(ValueError):
        Question("score", "x", ["a"] * 11)
    with pytest.raises(ValueError):
        Question("essay", "x")
    assert Question("noul", "x", review_below=0.8).payload() == {"type": "noul", "instructions": "x"}


def test_flatten_noul_confidence_is_distance_from_half():
    columns, conf = flatten("urgent", QUESTIONS["urgent"], {"type": "noul", "noul": 0.2})
    assert columns == {"urgent": False, "urgent_p": 0.2}
    assert conf == pytest.approx(0.8)


def test_only_state_fields_are_sent_and_answers_become_columns():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return fixed_response(request)

    results, stats = run(make_task(), client_with(handler))
    assert all(body["state"] == {"text": body["state"]["text"]} for body in seen)
    assert set(seen[0]["questions"]["kind"]) == {"type", "instructions", "criteria"}
    assert [r["id"] for r in results] == list(range(8))
    assert results[0] == {
        "id": 0, "kind": "a", "kind_conf": 0.95, "depth": 1.4, "depth_conf": 0.7,
        "urgent": False, "urgent_p": 0.2, "_review": False, "_review_reasons": "",
    }
    assert stats.input_tokens == 800 and stats.review == 0 and stats.models == {"jev-test"}


def test_low_confidence_rows_go_to_review():
    results, stats = run(make_task(), JevClient("k", transport=mock_transport()))
    flagged = [r for r in results if r["_review"]]
    assert stats.review == len(flagged) > 0
    assert all(r["kind_conf"] < 0.9 and r["_review_reasons"] == "kind" for r in flagged)
    assert all(r["kind_conf"] >= 0.9 for r in results if not r["_review"])


def test_cache_makes_second_run_free(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return fixed_response(request)

    cache = Cache(tmp_path / "c.sqlite")
    first, _ = run(make_task(), client_with(handler), cache)
    second, stats = run(make_task(), client_with(handler), cache)
    assert calls == 8 and stats.cached == 8 and stats.input_tokens == 0
    assert first == second


def test_retries_rate_limit_then_succeeds():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, text="slow down") if attempts < 3 else fixed_response(request)

    results, stats = run(make_task(), client_with(handler), rows=ROWS[:1])
    assert attempts == 3 and stats.errors == 0 and results[0]["kind"] == "a"


def test_failed_row_is_flagged_not_fatal():
    def handler(request):
        if json.loads(request.content)["state"]["text"] == "row 3":
            return httpx.Response(422, text="bad row")
        return fixed_response(request)

    results, stats = run(make_task(), client_with(handler))
    assert stats.errors == 1 and stats.review == 1
    assert results[3]["_review_reasons"] == "error" and "422" in results[3]["_error"]
    assert "kind" not in results[3] and results[4]["kind"] == "a"


def test_provider_picks_endpoint_model_and_price(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "oc_test")
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers["authorization"], json.loads(request.content)["model"]))
        return fixed_response(request)

    _, stats = run(make_task(), JevClient(model="jev-1.13-free", transport=httpx.MockTransport(handler)), rows=ROWS[:1])
    assert seen == [("https://opencode.ai/zen/v1/systemone", "Bearer oc_test", "jev-1.13-free")]
    assert stats.input_tokens == 100 and stats.cost_usd == 0

    # With both keys configured, the official API is the default and the model is pinned to the patch version
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test")
    client = JevClient(transport=httpx.MockTransport(handler))
    assert (client.provider, client.model) == ("typesafe", "jev-1.13.0")
    asyncio.run(client.__aexit__())

    _, stats = run(make_task(), client_with(lambda r: seen.append(str(r.url)) or fixed_response(r), provider="typesafe"), rows=ROWS[:1])
    assert seen[-1] == "https://api.typesafe.ai/v1/systemone" and stats.cost_usd > 0


def test_quota_exhausted_stops_early_and_keeps_finished_rows():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls > 3:
            return httpx.Response(429, text="FreeUsageLimitError", headers={"retry-after": "36000"})
        return fixed_response(request)

    results, stats = run(make_task(concurrency=1), client_with(handler))
    done = [r for r in results if not r.get("_pending")]
    # After the 4th request hits the quota there are no more retries and no requests for the remaining rows
    assert calls == 4 and len(done) == 3 and stats.pending == 5
    assert stats.quota_retry_after == 36000 and stats.errors == 0
    assert all(r == {"id": r["id"], "_pending": True} for r in results[3:])


def test_bad_key_aborts_whole_run():
    with pytest.raises(ExceptionGroup) as info:
        run(make_task(), client_with(lambda request: httpx.Response(401, text="invalid key")))
    assert info.group_contains(JevAuthError)
