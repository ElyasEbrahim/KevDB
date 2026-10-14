"""Register a model backend as DuckDB SQL functions.

    <prefix>_noul(state, instructions)             -> DOUBLE   P(yes). An LLM only answers yes or no, so it returns 1.0 or 0.0
    <prefix>_noulc(state, instructions, criteria)  -> DOUBLE   the same, with criteria: a JSON object {"true": what counts as yes, "false": what counts as no}
    <prefix>_choice(state, instructions, options)  -> VARCHAR  options is a JSON object: option name -> description
    <prefix>_score(state, instructions, levels)    -> DOUBLE   levels is a JSON array from low to high; returns the level counted from 0
    <prefix>_nouls(state, instructions)            -> DOUBLE[] instructions is a JSON array; several yes/no questions in one request, P(yes) for each in order

One block of a semantic join (one left record against its k candidates, all in one request; the candidates are found first with plain SQL):

    <prefix>_match(left, candidates, instructions, criteria) -> DOUBLE[]  k yes/no questions, one candidate in each, P(yes) for each candidate in order
    <prefix>_pick(left, candidates, instructions, none)      -> STRUCT(pos INTEGER, confidence DOUBLE)
                                                            one multiple-choice question that picks the matching candidate: pos is its position in candidates (from 1), 0 if none matches

left is one record (a JSON object) and candidates is a JSON array; the question calls the left record `record_a`, and _match questions call the candidate `record_b`.
none describes the "none of them" option. _pick picks at most one candidate, so it only suits joins where a left record matches at most one right record.
Both functions always save requests and tokens; that they answer like pairwise questions has only been verified on record matching (phases 8 and 9 in experiments/database/README.md).

state is a piece of text; if it is a JSON object (built with json_object(...)), it is sent to the model as a state with named fields.
Functions run per Arrow batch, and identical inputs within a batch are asked once. Rows that fail return NULL and are counted in the backend's meter.

A cascade (register_cascade) asks the first backend and passes the rows it is unsure about to the second:
a yes/no probability inside (lo, hi) counts as unsure; a choice confidence below min_conf counts as unsure.
Score questions are not cascaded: Jev returns a continuous probability-weighted score, and swapping in an LLM's integer level makes no sense.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import duckdb
import pyarrow as pa
from duckdb.sqltypes import DOUBLE, INTEGER, VARCHAR

from .engine import Backend, Item

Q = "q"  # each request asks one question, under a fixed name


def _state(text: str) -> Any:
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def _items(kind: str, states: pa.ChunkedArray, instructions: pa.ChunkedArray, criteria: pa.ChunkedArray | None) -> list[Item | None]:
    """Turn a batch of arguments into requests; rows with any NULL argument send no request."""
    crits = criteria.to_pylist() if criteria is not None else [""] * len(states)
    items: list[Item | None] = []
    for state, instr, crit in zip(states.to_pylist(), instructions.to_pylist(), crits):
        if state is None or instr is None or crit is None:
            items.append(None)
            continue
        question = {"type": kind, "instructions": instr}
        if criteria is not None:
            question["criteria"] = json.loads(crit)
        items.append((_state(state), {Q: question}))
    return items


def _ask(backend: Backend, items: list[Item | None]) -> list[dict[str, Any] | None]:
    """Return each row's answer to its question; NULL rows and failed rows give None."""
    asked = [i for i, item in enumerate(items) if item is not None]
    out: list[dict[str, Any] | None] = [None] * len(items)
    for i, resp in zip(asked, backend.ask_many([items[i] for i in asked])):
        out[i] = None if "_error" in resp else resp["answers"][Q]
    return out


def _column(answers: list[dict[str, Any] | None], field: str, kind: pa.DataType) -> pa.Array:
    return pa.array([None if a is None else a[field] for a in answers], type=kind)


def register(con: duckdb.DuckDBPyConnection, prefix: str, backend: Backend) -> None:
    def noul(states: pa.ChunkedArray, instructions: pa.ChunkedArray) -> pa.Array:
        answers = _ask(backend, _items("noul", states, instructions, None))
        return _column(answers, "noul", pa.float64())

    def noulc(states: pa.ChunkedArray, instructions: pa.ChunkedArray, criteria: pa.ChunkedArray) -> pa.Array:
        answers = _ask(backend, _items("noul", states, instructions, criteria))
        return _column(answers, "noul", pa.float64())

    def choice(states: pa.ChunkedArray, instructions: pa.ChunkedArray, options: pa.ChunkedArray) -> pa.Array:
        answers = _ask(backend, _items("choice", states, instructions, options))
        return _column(answers, "choice", pa.string())

    def score(states: pa.ChunkedArray, instructions: pa.ChunkedArray, levels: pa.ChunkedArray) -> pa.Array:
        answers = _ask(backend, _items("score", states, instructions, levels))
        return _column(answers, "score", pa.float64())

    def nouls(states: pa.ChunkedArray, instructions: pa.ChunkedArray) -> pa.Array:
        # Measured: Jev answering 21 questions in one request agrees with asking them one by one on 99.5% of judgements, same latency, 7.5x fewer input tokens
        rows: list[list[str] | None] = [None if s is None or i is None else json.loads(i) for s, i in zip(states.to_pylist(), instructions.to_pylist())]
        asked = [n for n, r in enumerate(rows) if r is not None]
        texts = states.to_pylist()
        items = [(_state(texts[n]), {f"q{k}": {"type": "noul", "instructions": instr} for k, instr in enumerate(rows[n])}) for n in asked]
        out: list[list[float] | None] = [None] * len(rows)
        for n, resp in zip(asked, backend.ask_many(items)):
            if "_error" not in resp:
                out[n] = [resp["answers"][f"q{k}"]["noul"] for k in range(len(rows[n]))]
        return pa.array(out, type=pa.list_(pa.float64()))

    def blocks(lefts: pa.ChunkedArray, candidates: pa.ChunkedArray, *rest: pa.ChunkedArray) -> list[tuple | None]:
        """Each row is (left record, candidate list, other arguments...); rows with any NULL argument are None."""
        rows = zip(lefts.to_pylist(), candidates.to_pylist(), *(r.to_pylist() for r in rest))
        return [None if None in row else (_state(row[0]), json.loads(row[1]), *row[2:]) for row in rows]

    def match(lefts: pa.ChunkedArray, candidates: pa.ChunkedArray, instructions: pa.ChunkedArray, criteria: pa.ChunkedArray) -> pa.Array:
        # Phase 8: with each candidate in its own question, Jev's judgements match pairwise asking and do not depend on candidate order; 1/k of the requests, about 40% of the input tokens
        # Phase 9: this holds only for record matching. For another predicate (do two reviews have opposite sentiment?) moving the candidate into the question changes the answer, so compare against pairwise asking on a sample first
        rows = blocks(lefts, candidates, instructions, criteria)
        asked = [n for n, r in enumerate(rows) if r is not None and r[1]]
        items = [({"record_a": rows[n][0]}, {f"c{i + 1}": {"type": "noul", "instructions": {"question": rows[n][2], "record_b": b}, "criteria": json.loads(rows[n][3])}
                                             for i, b in enumerate(rows[n][1])}) for n in asked]
        out: list[list[float] | None] = [None if r is None else [] for r in rows]
        for n, resp in zip(asked, backend.ask_many(items)):
            out[n] = None if "_error" in resp else [resp["answers"][f"c{i + 1}"]["noul"] for i in range(len(rows[n][1]))]
        return pa.array(out, type=pa.list_(pa.float64()))

    def pick(lefts: pa.ChunkedArray, candidates: pa.ChunkedArray, instructions: pa.ChunkedArray, none: pa.ChunkedArray) -> pa.Array:
        # Phase 8: on nearly one-to-one joins Jev's multiple choice is more accurate than pairwise asking and does not depend on candidate order; the local Kev-4B does, so do not use it there
        rows = blocks(lefts, candidates, instructions, none)
        asked = [n for n, r in enumerate(rows) if r is not None and r[1]]
        items = [({"record_a": rows[n][0]}, {"match": {"type": "choice", "instructions": rows[n][2],
                                                       "criteria": {**{f"c{i + 1}": b for i, b in enumerate(rows[n][1])}, "none": rows[n][3]}}}) for n in asked]
        out: list[dict | None] = [None if r is None else {"pos": 0, "confidence": None} for r in rows]
        for n, resp in zip(asked, backend.ask_many(items)):
            a = None if "_error" in resp else resp["answers"]["match"]
            out[n] = a and {"pos": 0 if a["choice"] == "none" else int(a["choice"][1:]), "confidence": a.get("confidence")}
        return pa.array(out, type=pa.struct([("pos", pa.int32()), ("confidence", pa.float64())]))

    _create(con, f"{prefix}_noul", noul, 2, DOUBLE)
    _create(con, f"{prefix}_noulc", noulc, 3, DOUBLE)
    _create(con, f"{prefix}_choice", choice, 3, VARCHAR)
    _create(con, f"{prefix}_score", score, 3, DOUBLE)
    _create(con, f"{prefix}_nouls", nouls, 2, con.list_type(DOUBLE))
    _create(con, f"{prefix}_match", match, 4, con.list_type(DOUBLE))
    _create(con, f"{prefix}_pick", pick, 4, con.struct_type({"pos": INTEGER, "confidence": DOUBLE}))


def register_cascade(
    con: duckdb.DuckDBPyConnection, prefix: str, first: Backend, second: Backend,
    *, lo: float = 0.1, hi: float = 0.9, min_conf: float = 0.9,
) -> None:
    def escalate(items: list[Item | None], answers: list[dict[str, Any] | None], unsure: Callable[[dict[str, Any]], bool]) -> None:
        """Rows where the first backend was unsure or failed get the second backend's answer instead."""
        again = [i for i, (item, a) in enumerate(zip(items, answers)) if item is not None and (a is None or unsure(a))]
        for i, a in zip(again, _ask(second, [items[i] for i in again])):
            answers[i] = a

    def noul(states: pa.ChunkedArray, instructions: pa.ChunkedArray) -> pa.Array:
        items = _items("noul", states, instructions, None)
        answers = _ask(first, items)
        escalate(items, answers, lambda a: lo < a["noul"] < hi)
        return _column(answers, "noul", pa.float64())

    def choice(states: pa.ChunkedArray, instructions: pa.ChunkedArray, options: pa.ChunkedArray) -> pa.Array:
        items = _items("choice", states, instructions, options)
        answers = _ask(first, items)
        escalate(items, answers, lambda a: a["confidence"] < min_conf)
        return _column(answers, "choice", pa.string())

    _create(con, f"{prefix}_noul", noul, 2, DOUBLE)
    _create(con, f"{prefix}_choice", choice, 3, VARCHAR)


def _create(con: duckdb.DuckDBPyConnection, name: str, fn: Callable[..., pa.Array], n_args: int, returns: Any) -> None:
    # null_handling="special": NULL is passed to the function too, which decides to return NULL; otherwise the rows of a batch would not line up
    con.create_function(name, fn, [VARCHAR] * n_args, returns, type="arrow", null_handling="special")
