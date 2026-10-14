# Experiments

Every study in this repository, with its question, main result and folder. Each folder has a README with full results,
limitations and the exact commands to reproduce them.

## Open models (the focus of this repository)

| Phase | Question | Result | Where |
| --- | --- | --- | --- |
| 6 | Can open, Jev-compatible models (Kev-4B, Kev-9B, Clef-flash, Laya) replace Jev as a backend out of the box? | No. Movie reviews on par, restaurant aspects behind, product matching on par only for Kev-9B. After recalibration with 200 human labels, within 1.4 points | [database](database/README.md#phase-6-can-open-models-replace-jev-as-the-default-backend) |
| 7 | What does a local model inside DuckDB do that a cloud API cannot? | Reusing already-read text: 43× (Kev-4B) and 23× (Kev-0.8B) faster on long documents asked several questions, identical answers. Not for short-text tables; calibration tables in SQL fix the scale but not the ranking | [database](database/README.md#phase-7-what-does-a-local-model-inside-duckdb-do-that-a-cloud-api-cannot) |
| 8 | Semantic joins: one request per pair or per left record? | Packing works on Jev (1/10 of the time, no F1 loss) but only gives Kev-4B 1.7–1.9×; Kev-4B's multiple choice is biased by position; a one-to-one constraint lifts Kev-4B from 85.6% to 93.7% F1 | [database](database/README.md#phase-8-semantic-joins-one-request-per-pair-or-one-request-per-left-record) |
| 9 | Does packing carry over to other join predicates? | No: savings remain, but answers change (Jev F1 73.3% → 65.3%, Kev-4B −15 points) | [database](database/README.md#phase-9-does-packing-carry-over-to-joins-other-than-record-matching) |
| 10 | Can fine-tuning Kev on public matching data reach Jev's zero-shot level? | Designed, not run | [database/protocol/p10_plan.md](database/protocol/p10_plan.md) |

## Jev and generative LLMs

| Study | Question | Result | Where |
| --- | --- | --- | --- |
| Phases 0–5 | Jev as SQL functions in DuckDB: quality, cost, cascades, aggregation, ordering, packing | SemBench mean quality 0.765 vs 0.745 for a cheap LLM, 3× cheaper and 5× faster; cascades only work for record matching; the biggest wins (join rewriting, packing) do not depend on the model | [database](database/README.md) |
| Table matching | Two raw tables end to end: blocking, judging, merging | Jev on par with a small LLM (F1 90.2% vs 91.0%) at a fifth of the cost; Jev plus a flagship LLM on its unsure pairs matches the flagship's quality at 19% of its cost | [table_match](table_match/README.md) |
| Entity matching | Magellan benchmarks against human labels | Zero-shot close to the supervised Ditto on product data, about 5× cheaper than a small LLM | [entity_match](entity_match/README.md) |
| Classification | AG News and Banking77 against human labels | On par with or slightly above a small LLM, below supervised models | [classify](classify/README.md) |
| Query plans | Choosing PostgreSQL plans for the JOB benchmark | Negative: neither Jev nor an LLM beats the default optimizer | [query_plan](query_plan/README.md) |
| Posts triage | The first batch-labelling task with the CLI | Accuracy level with small LLMs, 4–6× cheaper, with usable confidence | [posts_triage](posts_triage/README.md) |

## Ground rules

- **Criteria first.** Each phase of the database experiments wrote its question, metrics, thresholds and stopping rules
  into [database/protocol](database/protocol/) before any data was collected. Anything added after seeing results is
  labelled post hoc.
- **No tuning on test data.** Questions are first drafts and were never edited against the labels; thresholds come from
  validation splits.
- **No Jev outputs for training.** No answer of Jev was used to train, fine-tune, calibrate or pick thresholds for any
  open model (TypeSafe customer agreement, section 2.3(b)). Calibration uses human labels only.
- **Pinned versions.** Jev is pinned to `jev-1.13.0`, open models to tagged weights and commits, LLM routes to a fixed
  provider.

## Running them

Install the analysis dependencies and, where needed, download the data with each folder's `fetch.sh`:

```bash
.venv/bin/pip install -e ".[dev,experiments]"
./experiments/database/fetch.sh
```

All scripts run from the repository root. Answers are cached in `.jevcache.sqlite` and run outputs go to `out/`, both
kept local. Scripts that call a paid API say so in their docstring; every `report` step and analysis script replays
cached answers only. The open models need `./scripts/open_models.sh setup` (Apple Silicon, about 49 GB).

## Data sources

| Data | Used by | Source and license |
| --- | --- | --- |
| SemBench movie scenario and Rotten Tomatoes reviews | database | SemBench (Apache-2.0 / MIT) |
| MAMS-ACSA restaurant reviews | database | Apache-2.0 |
| GoEmotions | database (downloaded, unused) | Apache-2.0 |
| Magellan Abt-Buy, Walmart-Amazon, Amazon-Google, DBLP-ACM | entity_match, database | the version curated in the Ditto repository (Apache-2.0) |
| Abt-Buy and DBLP-Scholar raw tables | table_match, database | Leipzig University database group benchmarks |
| AG News, Banking77 | classify | their original repositories |
| JOB / IMDb | query_plan | CWI-hosted snapshot and the JOB queries |
| jev-hub X posts | posts_triage | jev-hub |

None of these datasets is committed; each folder's `fetch.sh` or README downloads them from the original source.
