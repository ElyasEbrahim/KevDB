import httpx

from kevdb import JevClient
from kevdb.mock import mock_transport

from kevdb import Backend

Q = {"q": {"type": "noul", "instructions": "Is it urgent?"}}


def backend(tmp_path, transport=None, **kwargs):
    make = lambda: JevClient("k", provider="typesafe", transport=transport or mock_transport(), max_retries=0)
    return Backend("jev", make, cache_path=tmp_path / "cache.sqlite", **kwargs)


def test_same_request_is_asked_and_counted_once(tmp_path):
    b = backend(tmp_path)
    out = b.ask_many([("a", Q), ("b", Q), ("a", Q)])
    assert out[0]["answers"] == out[2]["answers"] != out[1]["answers"]
    assert (b.meter.calls, b.meter.fresh, len(b.meter.latencies)) == (2, 2, 2)
    # Asking again in the same interval does not count twice
    b.ask_many([("a", Q)])
    assert b.meter.calls == 2
    cold = b.reset()
    assert cold.cost > 0 and cold.cost == b.spent

    # Replaying in a new interval sends no request, but the cost is what it was originally (the cost of a cold run)
    b.ask_many([("a", Q), ("b", Q)])
    assert (b.meter.calls, b.meter.fresh, b.meter.cost) == (2, 0, cold.cost)
    assert b.spent == cold.cost

    # A new namespace means a cold run
    b.reset()
    b.namespace = "run2:"
    b.ask_many([("a", Q)])
    assert b.meter.fresh == 1
    b.close()


def test_failed_request_is_reported_not_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500 if len(calls) == 1 else 200, json={"model": "m", "answers": {"q": {"type": "noul", "noul": 0.7}},
                                                                    "usage": {"input_tokens": 100, "output_tokens": 0}})

    b = backend(tmp_path, httpx.MockTransport(handler))
    assert "_error" in b.ask_many([("a", Q)])[0] and (b.meter.calls, b.meter.errors, b.meter.cost) == (1, 1, 0.0)
    # Retrying in the same interval: once it succeeds it no longer counts as failed, its usage is recorded, and the request is not counted twice
    assert b.ask_many([("a", Q)])[0]["answers"]["q"]["noul"] == 0.7
    assert (b.meter.calls, b.meter.errors, b.meter.fresh) == (1, 0, 1) and b.meter.cost > 0
    b.close()
