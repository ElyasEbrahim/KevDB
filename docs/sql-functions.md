# SQL functions

`kevdb.register(con, prefix, backend)` adds a family of scalar functions to a DuckDB connection. The prefix is yours to
choose (`kev`, `jev`, `llm`, ...), so several backends can live side by side in one query.

## Backends

```python
from kevdb import jev_backend, llm_backend, open_backend, Backend

open_backend("kev-4b")       # a local open model; start its server first (docs/open-models.md)
jev_backend()                # the hosted Jev API, pinned to jev-1.13.0; needs TYPESAFE_API_KEY
llm_backend("luna.low")      # a generative LLM through OpenRouter with a pinned provider; needs OPENROUTER_API_KEY
```

Every backend takes `concurrency` (default 16) and `cache_path` (default `.jevcache.sqlite`; `None` disables the cache).
Keys are read from the environment; `kevdb.cli.load_dotenv()` loads a `.env` file.

## Functions

| Function | Returns | Notes |
| --- | --- | --- |
| `<p>_noul(state, question)` | `DOUBLE` P(yes) | an LLM backend returns 1.0 or 0.0 |
| `<p>_noulc(state, question, criteria)` | `DOUBLE` P(yes) | `criteria` is a JSON object `{"true": "...", "false": "..."}` |
| `<p>_choice(state, question, options)` | `VARCHAR` | `options` is a JSON object, option name → description |
| `<p>_score(state, question, levels)` | `DOUBLE` | `levels` is a JSON array from low to high; returns the probability-weighted level, from 0 |
| `<p>_nouls(state, questions)` | `DOUBLE[]` | several yes/no questions in one request, P(yes) for each in order |
| `<p>_match(left, candidates, question, criteria)` | `DOUBLE[]` | one block of a semantic join: one yes/no question per candidate in a single request |
| `<p>_pick(left, candidates, question, none)` | `STRUCT(pos INTEGER, confidence DOUBLE)` | the same block as one multiple-choice question; `pos` counts from 1, 0 means none matches |

`state` is text; a JSON object (built with `json_object(...)`) is sent as a state with named fields. In `_match` and
`_pick`, `left` is a JSON object, `candidates` a JSON array, and the question refers to the left record as `record_a`
(and, in `_match`, to the candidate as `record_b`).

### Execution

- Functions run per Arrow batch. Identical inputs within a batch are asked once; answers are cached in SQLite, so a
  repeated query costs nothing.
- Rows with any `NULL` argument send no request and return `NULL`. Failed requests return `NULL` and are counted in the
  backend's meter (`backend.meter`, reset with `backend.reset()`).
- An invalid key or an exhausted balance stops the query instead of failing every row.

### Example

```python
import duckdb
from kevdb import open_backend, register

con = duckdb.connect()
register(con, "kev", open_backend("kev-4b"))
con.sql("""
    SELECT id,
           kev_choice(json_object('text', body), 'What is this ticket about?',
                      '{"billing": "payments, invoices, refunds", "bug": "something does not work", "other": "anything else"}') AS topic,
           kev_noul(json_object('text', body), 'Is the customer angry?') AS p_angry
    FROM tickets
""")
```

## Semantic joins

Block first with ordinary SQL (full-text search, a shared key, a cheap filter), then judge each left record's candidates
in one request:

```sql
WITH blocks AS (
    SELECT l.id, json_object('name', l.name, 'price', l.price) AS left_rec,
           to_json(list(json_object('name', r.name, 'price', r.price) ORDER BY r.score DESC)) AS cands,
           list(r.id ORDER BY r.score DESC) AS right_ids
    FROM left_table l JOIN candidates r ON r.left_id = l.id
    GROUP BY ALL
)
SELECT id, right_ids, kev_match(left_rec, cands, 'Do `record_a` and `record_b` describe the same product?', '{}') AS p
FROM blocks;
```

What the experiments found ([phases 8 and 9](../experiments/database/README.md#phase-8-semantic-joins-one-request-per-pair-or-one-request-per-left-record)):

- On record matching, `_match` gives the same answers as asking pair by pair, with a tenth of the requests on Jev.
- `_pick` is only for joins where a left record matches at most one record, and only on Jev; Kev-4B's multiple choice
  depends on candidate order. On Kev, keep each left record's highest `_match` probability at ≥ 0.5 instead.
- For predicates other than record matching (for example "opposite sentiment"), moving the second record into the
  question can change the answer. Compare against pairwise asking on a sample before packing.

## Cascades

```python
from kevdb import register_cascade

register_cascade(con, "cas", open_backend("kev-4b"), llm_backend("luna.low"), lo=0.1, hi=0.9, min_conf=0.9)
```

`cas_noul` and `cas_choice` ask the first backend and pass rows it is unsure about (a probability inside `(lo, hi)`, or a
choice confidence below `min_conf`) to the second. Score questions are not cascaded. Cascades only paid off on record
matching in the experiments; check how concentrated the first backend's errors are in its unsure rows before using one.

## Calibration tables

Open models rank well but often sit on the wrong probability scale. Calibration parameters can be stored in the database
like statistics and applied in SQL:

```python
from kevdb import calibration

calibration.install(con, "kev", kev.model)                       # table semantic_calibration and the macros below
calibration.fit(con, kev.model, question, criteria, "products", p, y)   # p: raw probabilities, y: 0/1 human labels
con.sql("SELECT count(*) FROM pairs WHERE kev_noulc_cal(state, $q, $c, 'products') >= 0.5",
        params={"q": question, "c": criteria})
```

| Macro | Returns |
| --- | --- |
| `<p>_noul_cal(state, question, scope)` | calibrated `<p>_noul(...)` |
| `<p>_noulc_cal(state, question, criteria, scope)` | calibrated `<p>_noulc(...)` |
| `calibrated(p, model, predicate, scope)` | any probability column, calibrated |
| `predicate_key(question, criteria)` | the predicate's key in the table |

Parameters are looked up by (model, predicate, scope): the given scope first, then scope `''` (the predicate's default),
otherwise the probability is returned unchanged. The question can be a column, so each predicate in a query uses its own
parameters. Changing a row of `semantic_calibration` takes effect on the next query.

Fit on human labels only. In the experiments 100–200 labels per predicate were enough where calibration helped at all; it
fixes the scale, not the ranking.
