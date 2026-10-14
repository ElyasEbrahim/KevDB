import asyncio
import json

import httpx
import pytest

from kevdb import JevAPIError, Question, Task, run_task
from kevdb.llm import LLMClient, parse_answers

QUESTIONS = {
    "kind": Question("choice", "What kind?", {"a": "A", "b": "B"}, review_below=0.9),
    "depth": Question("score", "How deep?", ["none", "some", "lots"]),
    "urgent": Question("noul", "Is it urgent?"),
}
PAYLOADS = {name: q.payload() for name, q in QUESTIONS.items()}


def test_parse_accepts_json_wrapped_in_prose_and_rejects_bad_values():
    answers = parse_answers('Sure!\n```json\n{"kind": "b", "depth": 2, "urgent": false}\n```', PAYLOADS)
    assert answers["kind"]["choice"] == "b" and answers["kind"]["confidence"] == 1.0
    assert answers["depth"]["score"] == 2.0 and answers["urgent"]["noul"] == 0.0
    # A score answer given as the level name or the full description is accepted if it matches exactly one level
    levels = {"depth": {"type": "score", "instructions": "x", "criteria": ["None: nothing.", "Low: a bit.", "High: lots."]}}
    for value, expected in (('"Low"', 1.0), ('"high"', 2.0), ('"None: nothing."', 0.0), ('"2"', 2.0)):
        assert parse_answers(f'{{"depth": {value}}}', levels)["depth"]["score"] == expected
    with pytest.raises(JevAPIError):
        parse_answers('{"depth": "Medium"}', levels)
    for bad in ('{"kind": "c", "depth": 2, "urgent": false}', '{"kind": "a", "depth": 3, "urgent": false}',
                '{"kind": "a", "depth": 1, "urgent": "no"}', '{"kind": "a", "depth": 1}', "I think it is a."):
        with pytest.raises(JevAPIError):
            parse_answers(bad, PAYLOADS)


def test_llm_runs_through_the_same_pipeline():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((str(request.url), body["model"], json.loads(body["messages"][1]["content"])))
        text = body["messages"][1]["content"]
        content = "not json" if "row 1" in text else '{"kind": "a", "depth": 1, "urgent": true}'
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 20},
        })

    task = Task(name="t", input="unused", questions=QUESTIONS, state_fields=["text"])
    rows = [{"id": i, "text": f"row {i}"} for i in range(3)]

    async def go():
        async with LLMClient("k", provider="opencode", model="deepseek-v4.1-flash", transport=httpx.MockTransport(handler)) as client:
            return await run_task(task, rows, client)

    results, stats = asyncio.run(go())
    url, model, sent = seen[0]
    assert url == "https://opencode.ai/zen/v1/chat/completions" and model == "deepseek-v4.1-flash"
    assert sent == {"state": {"text": "row 0"}, "questions": PAYLOADS}
    # An LLM has no confidence: well-formed rows never go to review, malformed ones count as failures
    assert results[0] == {"id": 0, "kind": "a", "kind_conf": 1.0, "depth": 1.0, "depth_conf": 1.0,
                          "urgent": True, "urgent_p": 1.0, "_review": False, "_review_reasons": ""}
    assert results[1]["_review_reasons"] == "error" and stats.errors == 1
    assert (stats.input_tokens, stats.output_tokens) == (1000, 40)
    assert stats.cost_usd == pytest.approx(1000 * 0.30 / 1e6 + 40 * 1.20 / 1e6)
    assert len(stats.latencies) == 3
