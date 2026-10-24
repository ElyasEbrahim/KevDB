# Overview: what was tested, on which data, with what result

This page is the entry point for readers new to the project. Every row names what an experiment does, the data it used
and its main result; the linked sections have the full numbers, limitations and the commands to reproduce them. The
same material with figures is in [the report](../experiments/database/report/Jev-databases-report.pdf).

## Background

A **decision model** does not write text. It reads a piece of data and answers typed questions about it with a
probability for each answer:

```text
data      two product records: "electrolux oxygen 3 canister hepa h12 filter el012a, $19.99"
                               "electrolux homecare #el012a electro h12 hepa filter"
question  Do the two records describe the same product?
answer    yes 0.93, no 0.07          (human label: yes)
```

**Jev** (TypeSafe AI) is a closed decision model behind a paid API. **Kev**, **Clef-flash** and **Laya** are open models
that accept the same request and run on a laptop. KevDB turns either kind into DuckDB SQL functions, so that a query can
ask a question about every row.

Measures used below:

| Measure | Meaning |
| --- | --- |
| Accuracy | share of answers that are right |
| F1 | for matching tasks: combines how many reported matches are right and how many true matches are found; 100 is perfect |
| AUC | how well the probabilities rank yes above no; 1 is perfect, 0.5 is chance |
| Brier | how far the probabilities are from the true answers; lower is better |
| On par | compared with Jev's zero-shot answers: accuracy at most 2 points lower, AUC at most 0.02 lower, Brier at most 0.02 higher (fixed before the open-model experiments) |

## 1. Jev as a data-processing tool (1–5 October)

Each task uses public data with known answers and compares Jev with a small LLM (DeepSeek V4.1 Flash, thinking off).

| Task | What it does | Data | Result |
| --- | --- | --- | --- |
| [Post tagging](../experiments/posts_triage/README.md) | asks four questions about each post: type, substance, first-hand, critical of Jev | 982 X posts about Jev; reference answers for 161, labelled by an LLM, not by people | accuracy 88%, 93% and 97% on the three category questions; the 4-level score is exact 75% of the time, within one level 99%; close to small LLMs at 1/4 to 1/14 of their cost |
| [Entity matching](../experiments/entity_match/README.md) | asks for each pair of records "same product (or paper)?" | four Magellan benchmarks, 8,731 test pairs with human labels | F1 69.5–97.3 without training; above the trained Ditto on 2 of 4, below on 2 |
| [Table matching](../experiments/table_match/README.md) | full-text search proposes 5 candidates per record, Jev judges each pair, unsure pairs can go to a stronger model or a person | Abt-Buy product tables, 1,081 × 1,092 records, 1,097 human-matched pairs | F1 90.2% (small LLM 91.0%) at 1/5 of the cost; sending Jev's unsure 12% to Opus gives 93.7% at about 1/5 of Opus's cost |
| [Classification](../experiments/classify/README.md) | one multiple-choice question per text | AG News (7,600 news, 4 topics), Banking77 (3,080 messages, 77 intents) | 89.0% and 79.9%: level with the small LLM, about 5–14 points below trained classifiers |
| [Query plans](../experiments/query_plan/README.md) | asks whether a candidate PostgreSQL plan will run faster than the default | JOB benchmark: IMDb database, 113 queries, measured run times | AUC 0.52: no useful signal; the small LLM did no better |

## 2. Jev inside DuckDB (phases 0–5, 7 October)

| Task | What it does | Data | Result |
| --- | --- | --- | --- |
| [Semantic SQL](../experiments/database/README.md#phase-1-the-10-queries-of-sembenchs-movie-scenario) | runs SemBench's 10 movie queries with Jev as a SQL function | 2,000 Rotten Tomatoes reviews, SemBench's reference answers and scorer | mean quality 0.765 (small LLM 0.745); 1/3 of the cost and 1/5 of the time |
| [Cascades](../experiments/database/README.md#phase-5-a-truly-stronger-upper-model) | sends the rows Jev is least sure about to a stronger model | product matching, movie reviews, restaurant sentences, human labels | helps on record matching only (F1 0.884 to 0.929); elsewhere Jev's errors are confident ones |
| [Estimating a share](../experiments/database/README.md#phase-3-semantic-aggregation) | corrects Jev's estimate of the share of positive reviews with 200 human labels | 15,388 reviews, 81.3% positive | 2.5× fewer labels than sampling alone; the target was 3× |
| [Packing](../experiments/database/README.md#phase-4-selectivity-ordering-packing) | puts the 21 questions about one sentence into one request | 1,000 MAMS restaurant sentences, 21 human-labelled predicates | cost per 1,000 sentences $0.284 to $0.038, 99.5% of answers unchanged |

## 3. Open decision models as the backend (phases 6–9, 9–13 October)

| Task | What it does | Data | Result |
| --- | --- | --- | --- |
| [Unchanged](../experiments/database/README.md#phase-6-can-open-models-replace-jev-as-the-default-backend) | sends the requests Jev answered to five open models | 5,000 movie reviews, 1,000 restaurant sentences × 21 predicates, 2,049 product pairs | on par on movie reviews (3 models), below on restaurants (all), on products only Kev-9B |
| [Local cache](../experiments/database/README.md#phase-7-what-does-a-local-model-inside-duckdb-do-that-a-cloud-api-cannot) | asks the same documents several questions, reusing what the server has read | 40 long documents of about 5,800 tokens; movie and restaurant tables | 43× (Kev-4B) and 23× (Kev-0.8B) faster on long documents, identical answers; 1.7–1.9× on short texts |
| [Semantic joins](../experiments/database/README.md#phase-8-semantic-joins-one-request-per-pair-or-one-request-per-left-record) | asks a record's 10 candidates in one request instead of ten | Abt-Buy and DBLP-Scholar raw tables with full human mappings; 8,128 review pairs | on Jev 10–11% of the time at the same F1; on Kev-4B only about half the time; for "opposite sentiment" F1 drops 8 points |

## 4. Fine-tuning Kev for record matching (phases 10–14, 16–23 October)

| Task | What it does | Data | Result |
| --- | --- | --- | --- |
| [Public labels](../experiments/database/README.md#phase-10-does-fine-tuning-on-public-matching-data-bring-kev-to-jevs-zero-shot-level) | trains Kev on three Magellan data sets, tests on the fourth | 2,000 human-labelled pairs per data set; full test splits | on par on 2 of 4 (Kev-4B) and 1 of 4 (Kev-0.8B); no change on the publication data set |
| [Why not publications](../experiments/database/README.md#phase-11-why-did-fine-tuning-not-help-kev-4b-on-dblp-acm) | adds one ingredient at a time to the training data | product pairs plus rewordings, three other domains, or DBLP-GoogleScholar | F1 0.742 to 0.945 and 0.962 with more kinds of data; rewording does nothing |
| [Own labels](../experiments/database/README.md#phase-12-how-many-of-your-own-labels-does-kev-4b-need) | trains on 100, 300 or 1,000 labels of the data set itself | records split so that no test record is in any label | on par on all four data sets at 300 labels; none needed on two |
| [Seeds](../experiments/database/README.md#phase-13-are-phase-10s-close-verdicts-stable-when-only-the-training-seed-changes) | retrains close cases with other random seeds | the same data as phase 10 | 3 of 4 verdicts hold; [Kev-0.8B's drop on movie reviews](../experiments/database/README.md#phase-14-is-kev-08bs-drop-on-movie-reviews-a-property-of-seed-1-or-of-training-without-padding) belongs to one seed |

## Limits that apply throughout

- Public benchmarks, which the models may have seen in training.
- One laptop (M3 Max, 64 GB): speeds do not describe GPU servers, and the largest open models were not run.
- Fine-tuning covers record matching only, and most checkpoints were trained once.
- LLM baselines in section 1 ran through a third-party platform, so their times are not strictly comparable.

Success criteria for every phase were written down before its data were collected: see
[experiments/database/protocol](../experiments/database/protocol/README.md).
