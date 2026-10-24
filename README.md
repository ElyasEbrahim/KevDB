# KevDB

**Semantic SQL in DuckDB, powered by open-source decision models.**

[![CI](https://github.com/ElyasEbrahim/KevDB/actions/workflows/ci.yml/badge.svg)](https://github.com/ElyasEbrahim/KevDB/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

[Chinese introduction](README.zh-CN.md) · New here? [docs/overview.md](docs/overview.md) lists every experiment: what it does, on which data, with what result.

A *decision model* does not generate text. It reads a record (the *state*) and answers typed questions about it with
probabilities: yes/no (`noul`), pick one option (`choice`), or rate on a scale (`score`). That makes it a natural
database operator: one cheap, deterministic call per row, with a probability you can threshold, calibrate and aggregate.

The best-known decision model, **Jev** from TypeSafe AI, is closed and API-only. Several open-weight models now accept
the same request format and run on a laptop: **Kev** (0.8B, 4B, 9B), **Clef-flash** and **Laya**. KevDB plugs them
into DuckDB as SQL functions and measures, under pre-registered criteria, how close they come to Jev and what a local
model can do that a hosted API cannot.

```sql
SELECT id, review
FROM reviews
WHERE kev_noul(json_object('review', review), 'Is the movie review in `review` positive?') >= 0.5;
```

## Key findings

All results come from experiments whose success criteria were written down before any data was collected
(see [experiments/database/protocol](experiments/database/protocol)). Hardware: one MacBook Pro, M3 Max, 64 GB.

| Question | Answer |
| --- | --- |
| Can an open model replace Jev out of the box? | **No, not yet.** Kev-4B, Kev-9B and Clef-flash match Jev on movie reviews but fall short on restaurant-aspect questions; on product matching only Kev-9B matches. ([Phase 6](experiments/database/README.md#phase-6-can-open-models-replace-jev-as-the-default-backend)) |
| Where is the gap? | **Calibration, not ranking.** AUC differs by only 0.002–0.030. After recalibrating with 200 human labels, all three are within 1.4 points of Jev. |
| What can only a local model do? | **Reuse what it has already read.** On long documents (about 5,800 tokens) that are asked one question after another, the server's prefix cache makes every later question **43× faster on Kev-4B and 23× faster on Kev-0.8B**, with bit-identical answers. Kev-0.8B overtakes the hosted Jev API (one request at a time) from the third question. ([Phase 7](experiments/database/README.md#phase-7-what-does-a-local-model-inside-duckdb-do-that-a-cloud-api-cannot)) |
| How good is a local model as a semantic-SQL backend? | Kev-4B scores **0.803** on the 10 SemBench movie queries (Jev 0.765, a cheap LLM 0.745) at no marginal cost, but takes 8.7× longer than Jev. Kev-0.8B is only 2× slower than Jev but scores 0.664. |
| Can calibration live in the database? | **Yes.** Per-predicate calibration parameters stored in a table and applied by SQL macros lift Kev-0.8B from 66–86% to 91–98% accuracy on record matching. Calibration fixes the scale, not the ranking: on two of three new matching datasets both models stay behind Jev. |
| How should a semantic join ask its questions? | For record matching on Jev, put a left record's 10 candidates into one request: 1/10 of the requests and time, 38% of the tokens, no loss in F1. On Kev-4B the same packing is only 1.7–1.9× faster, and its multiple-choice form is biased towards later options. A one-to-one constraint alone lifts Kev-4B from 85.6% to 93.7% F1 without any labels. ([Phase 8](experiments/database/README.md#phase-8-semantic-joins-one-request-per-pair-or-one-request-per-left-record)) |
| Does packing transfer to other join predicates? | **No.** On "do these two reviews have opposite sentiment?" packing still saves 67–74% of the tokens, but Jev's F1 drops from 73.3% to 65.3% and Kev-4B's by 15 points. Moving the second record from the state into the question changes the answer. ([Phase 9](experiments/database/README.md#phase-9-does-packing-carry-over-to-joins-other-than-record-matching)) |
| Can fine-tuning close the gap? | **Partly.** Fine-tuned on 6,000 public, human-labelled pairs from three other datasets, Kev-4B matches Jev's zero-shot quality on **2 of 4** unseen matching datasets without forgetting; it stays behind on the hardest product set, and on a publication task asked with a question it never saw in training it does not improve. Kev-0.8B matches on 1 of 4. With in-domain labels Kev-4B's F1 is above Jev's on all four. ([Phase 10](experiments/database/README.md#phase-10-does-fine-tuning-on-public-matching-data-bring-kev-to-jevs-zero-shot-level)) |
| Why did it not help on the new question? | **The training data were of one kind.** Rewording the product question changes nothing. Adding another publication dataset asked with the same question, or 1,156 pairs from three other domains with their own questions, brings Kev-4B to Jev's level on dblp-acm (F1 0.962 and 0.945, Jev 0.973) without forgetting. ([Phase 11](experiments/database/README.md#phase-11-why-did-fine-tuning-not-help-kev-4b-on-dblp-acm)) |
| How many of your own labels does it need? | **About 300.** Measured on records no label contains, Kev-4B is on par with Jev on all four matching datasets with 300 labels of the dataset itself, from the released checkpoint or from phase 10's. On product data like its public training data it needs none, on the publication question about 100. On the hardest product set its F1 stays below Jev's. ([Phase 12](experiments/database/README.md#phase-12-how-many-of-your-own-labels-does-kev-4b-need)) |
| Do these verdicts survive another training seed? | **Mostly.** With two more seeds, three of the four verdicts that sat near a line hold. Kev-4B on amazon-google turns from below Jev to on par, at the edge. Kev-0.8B's slight forgetting on movie reviews does not recur. ([Phase 13](experiments/database/README.md#phase-13-are-phase-10s-close-verdicts-stable-when-only-the-training-seed-changes)) |
| Is Kev-0.8B's forgetting real? | **It belongs to one training draw.** Retrained with phase 10's seed 1, the 2-point drop on movie reviews comes back unchanged; with seeds 2 and 3 it does not. The padding fix of phase 10 changes no result. ([Phase 14](experiments/database/README.md#phase-14-is-kev-08bs-drop-on-movie-reviews-a-property-of-seed-1-or-of-training-without-padding)) |

Results with the hosted Jev API alone (phases 0–5 and the earlier studies) are summarised in
[experiments/README.md](experiments/README.md). In short: as a SQL function Jev is the cheapest and fastest per-row
probability source we measured (3× cheaper and 5× faster than a cheap LLM), cascades to a stronger LLM pay off only
for record matching, and the two biggest wins (rewriting pairwise joins and packing several questions into one
request) do not depend on which model you use.

## Quick start

Requires Python 3.12+. The library and the offline test suite need no API key and no model.

```bash
git clone https://github.com/ElyasEbrahim/KevDB.git
cd KevDB
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

### Run an open model locally (Apple Silicon)

```bash
./scripts/open_models.sh setup          # Kev-4B, Kev-9B and Clef-flash, about 49 GB, all inside ./open_models
./scripts/open_models.sh serve kev-4b   # serves on 127.0.0.1:8009 inside a macOS sandbox with no internet access
```

Everything is installed into `open_models/` inside the repository: nothing touches the system Python, shell
configuration or user caches. See [docs/open-models.md](docs/open-models.md) for all models, ports and versions.

### Query it from SQL

```python
import duckdb
from kevdb import open_backend, register

con = duckdb.connect()
register(con, "kev", open_backend("kev-4b"))   # creates kev_noul, kev_choice, kev_score, kev_match, ...

con.sql("""
    SELECT review,
           kev_noul(json_object('review', review), 'Is the movie review in `review` positive?') AS p_positive
    FROM (VALUES ('A joyless, overlong mess.'), ('Funny, warm and beautifully shot.')) AS t(review)
""").show()
```

The same functions run on the hosted Jev API (`jev_backend()`, needs `TYPESAFE_API_KEY` in `.env`) or on a
generative LLM through OpenRouter (`llm_backend("luna.low")`). Answers are cached in SQLite, identical inputs within a
batch are asked once, and failed rows return `NULL`.

## SQL functions

`register(con, prefix, backend)` adds these functions under the prefix you choose:

| Function | Returns |
| --- | --- |
| `<prefix>_noul(state, question)` | P(yes) |
| `<prefix>_noulc(state, question, criteria)` | P(yes), with JSON criteria for what counts as yes and no |
| `<prefix>_choice(state, question, options)` | the chosen option |
| `<prefix>_score(state, question, levels)` | probability-weighted level, counted from 0 |
| `<prefix>_nouls(state, questions)` | several yes/no questions in one request, P(yes) for each |
| `<prefix>_match(left, candidates, question, criteria)` | one block of a semantic join: P(match) for every candidate, in one request |
| `<prefix>_pick(left, candidates, question, none)` | the same block as one multiple-choice question: position of the chosen candidate and its confidence |

Calibration tables, cascades and the full reference are in [docs/sql-functions.md](docs/sql-functions.md).

## Supported models

| Model | Runs | Pinned version | Notes |
| --- | --- | --- | --- |
| Kev-4B, Kev-9B, Kev-0.8B | locally, MLX | code `kev-1.0`, weights `jaredpalmer/kev-*@v1.0` | Kev-4B is the most practical local backend |
| Clef-flash | locally, PyTorch on the Mac GPU | `Cloudflare/clef-flash@8b2e5fd1` | 9.4B; slow on a Mac |
| Laya (English, multilingual) | locally | `laya` 0.4.1 | dropped after phase 6; not installed by default |
| Jev | TypeSafe API | `jev-1.13.0` | closed reference model |
| LLMs (GPT 6 Luna, DeepSeek, Qwen, ...) | OpenRouter, provider pinned per request | see [src/kevdb/routes.py](src/kevdb/routes.py) | generative baselines |

## Repository layout

```text
KevDB/
├── src/kevdb/          the library: decision-model clients, DuckDB SQL functions, calibration, batch pipeline, CLI
├── tests/              offline tests; they never call a model or a paid API
├── scripts/            install and serve the open models locally
├── experiments/        every study, each with its own README, scripts and data-download script
│   ├── database/       phases 0–10: decision models inside DuckDB (open models from phase 6 on)
│   ├── table_match/    end-to-end matching of two raw tables
│   ├── entity_match/   entity matching on the Magellan benchmarks
│   ├── classify/       text classification on AG News and Banking77
│   ├── query_plan/     choosing PostgreSQL query plans (a negative result)
│   └── posts_triage/   the first batch-labelling task
├── docs/               user documentation
└── pyproject.toml
```

Downloaded datasets (`experiments/*/data`), model weights (`open_models/`), run outputs (`out/`) and the answer cache
(`.jevcache.sqlite`) stay local and are not part of the repository.

## Reproducing the experiments

Each experiment folder has a README with the exact commands. Scripts that call a paid API say so at the top, and every
answer is cached, so re-running a finished step costs nothing. The offline tests and all `report` steps replay cached
answers only. Install the analysis dependencies with:

```bash
.venv/bin/pip install -e ".[dev,experiments]"
```

## Disclaimer

Jev and TypeSafe belong to TypeSafe AI. Kev, Clef and Laya are independent projects by their respective authors.
This repository is not affiliated with any of them and ships none of their weights. No output of Jev was used to train,
fine-tune or calibrate any open model; all calibration uses human labels that come with public benchmarks.

## License

[MIT](LICENSE). Datasets are downloaded from their original sources and keep their own licenses.
If you use this work, please cite it as described in [CITATION.cff](CITATION.cff).
