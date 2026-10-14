# Phase 7 execution plan: does a local decision model inside DuckDB offer anything a cloud API cannot?

Finalised 2026-10-11. Revised on 2026-10-12 at the owner's decisions: use as much memory as possible, add Kev-0.8B, add
Jev's reference on long documents (see the end of section 2). Status: scripts and rehearsal done, the owner said "go",
and Kev-4B's collection started on 2026-10-12 at 15:47.

**The owner's decision of 2026-10-12, 19:15: run Kev-0.8B on a later instruction, everything else as planned.** Wherever
this plan says "the two models run one after the other", this time only Kev-4B ran, followed by Jev's long-document
reference and Kev-4B's analysis; the Kev-0.8B arm stays exactly as frozen (data, questions, settings from the capacity
probe, bars, scripts) and runs unchanged on instruction: `P7_NO_REPORT=1 ./experiments/database/p7_run.sh --model
kev-0.8b`, then `P7_MODEL=kev-0.8b python experiments/database/p7_report.py`. One deviation follows: Kev-4B's analysis
precedes Kev-0.8B's collection.

This file is the runbook: the order of steps, the acceptance check of each step and how to handle mishaps are all fixed
before the run, so no decisions are improvised midway. The original record of the decision criteria is
[p7.txt](p7.txt) (written before collection, with script checksums); where the two differ, p7.txt prevails.

*Translated from the Chinese original for the public release.*

## 1 Questions

**Main question**: without a cloud API, does a decision model running locally inside DuckDB offer a "wonder", and of
which kind?

A wonder will not come from speed (Kev-4B serves 8–13 requests per second locally, the Jev API 52) nor from zero-shot
quality (phase 6: it cannot replace Jev as is). So the test covers three things that only a local model can do, or that
only pay off locally, each with a bar fixed beforehand:

| | Test | What counts as a wonder |
| --- | --- | --- |
| A | semantic-SQL backend on the 10 queries of SemBench's movie scenario | mean quality not below Jev (0.765) and total time not above the cheap LLM (1,647 s) |
| B | the server caches the text it has read; the same rows asked another question | long documents: later questions at least 3× faster, answers unchanged, the cache within the memory budget. Whole short-text tables: at least 2× faster, answers unchanged, both whole tables within the budget |
| C | per-predicate calibration parameters fitted on 200 human labels, stored in a table, applied in SQL | on three matching datasets Kev never answered, on par with Jev after calibration on all three, with at least one behind before |

Two properties are stated without a test: no marginal cost per request, and the data never leaves the machine (the
server runs in a sandbox with no network).

## 2 Models and settings

| | What | How obtained |
| --- | --- | --- |
| Kev-4B (subject) | identical to phase 6: code `kev-1.0`, weights `jaredpalmer/kev-4b@v1.0`, MLX, bfloat16, factory temperature 2.406 | local, `./scripts/open_models.sh serve kev-4b`, sandboxed |
| Kev-0.8B (subject, added 2026-10-12) | the same code, weights `jaredpalmer/kev-0.8b@v1.0` (bf75a6a8), base `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2`, MLX, bfloat16, factory temperature 2.351 | local, port 8014, sandboxed; `./scripts/open_models.sh setup kev-0.8b` downloads 1.84 GB |
| Jev (long-document reference, added 2026-10-12) | `jev-1.13.0`, official API, answering the three questions on the 40 long documents only | paid, about $0.03; comparison column only |
| Jev, cheap LLM, published systems (otherwise) | cached results of earlier phases | comparison columns only, nothing new asked |

The cache is built into Kev's server and is controlled only through environment variables, without changing its code:
`KEV_PREFIX_CACHE` is how many texts to keep (factory default 4), `KEV_PREFIX_MAX_TOKENS` the cap on total cached tokens
(factory default 65,536).

**The owner's decisions of 2026-10-12 (before any evaluation data was asked): go; use as much memory as possible; do both
optional additions.**

"As much memory as possible" means the cache's memory budget is no longer a fixed 24 GB but measured per model with
`experiments/database/p7_capacity.py` on made-up rows: blocks of 25 texts are added to the cache until the system's
available memory falls below 4 GB, the server exceeds 75% of physical memory, swap grows by more than 1 GB, cache hits get
25% slower or the server reports running out of memory; the last block before stopping counts as healthy, and the
budget is 90% of the cache's size at that point.

| | Per cached text | Most healthy entries | Why it stopped | Budget |
| --- | --- | --- | --- | --- |
| Kev-4B | 57.8 MB | 675 | swap started to grow | 34.3 GB, 600 entries |
| Kev-0.8B | 22.0 MB | 2,125 | the server reached 75% of physical memory | 41.2 GB, 1,900 entries |

Three changes follow (the three bars and both decision definitions are unchanged):
- B3's budget takes each model's number from the table above.
- The SQL step's budget run opens the cache to the budget: 600 entries for Kev-4B, 1,900 for Kev-0.8B (previously 300).
- The two short-text datasets of B use the whole table if it fits the budget together with 20 warm-up rows, otherwise as
  many leading rows of the fixed order as fit (previously 300 rows each).

## 3 Data and questions

| Part | Data | Rows | Questions |
| --- | --- | --- | --- |
| A | the 2,000-row version of SemBench's movie scenario | 10 queries plus 3 rewritten joins, 16,940 requests | phase 1's questions and SQL, unchanged |
| A′ | same | only the 10 per-review queries (Q1–Q4, Q8, Q5r–Q7r, Q9, Q10), 4,716 requests | same |
| B movie | the distinct review texts of that table (1,864), ordered by hash | Kev-4B the first 580; Kev-0.8B the whole table of 1,864 | phase 1's four questions: clearly positive, positive, sentiment (choice), how much liked (score) |
| B mams | phase 4's fixed order of restaurant sentences (3,949) | Kev-4B the first 580; Kev-0.8B the first 1,880 | the first three of the 21 predicates |
| B long | movies in the Rotten Tomatoes source data that are not in SemBench: one movie's reviews, one per line in id order, joined into a document of at most 24,000 characters; 20 documents where most reviews are positive and 20 where they are not | 40 | phase 1's three questions rewritten from "one review" to "a group of reviews"; first draft, not edited |
| B strata | the first 12 movies of each stratum, prefixes of 1,500 / 3,000 / 6,000 / 12,000 characters | 24 each | as long |
| B Jev reference | the same 40 long documents | 120 requests | as long |
| C | abt-buy, amazon-google, dblp-acm (never answered by Kev), walmart-amazon (answered by Kev-4B in phase 6) | per dataset: a fixed 1,000 validation pairs as the label pool, and the whole test split (1,916 / 2,293 / 2,473 / 2,049 pairs) | the original questions asked of Jev |
| C replay (Kev-4B only) | phase 6's 5,000 movie reviews and 1,000 restaurant sentences × 21 questions | no model asked; replayed from a copy of the cache | original questions |

A probability ≥ 0.5 counts as yes. No threshold, question or temperature is tuned. Data, questions, analysis and bars are
identical for both models, and each gets its own verdict.

## 4 Metrics and decision criteria

### 4.1 A: semantic-SQL benchmark

- Scoring uses SemBench's own scoring code, with mean quality computed as in phase 1. The server runs at factory
  settings, every query cold, concurrency 16, one run (answers are deterministic).
- **Qualifying bar A1**: mean quality over the 10 queries not below the cheap LLM's 0.7451, and no query computed with
  unanswered rows.
- **Wonder bar A2**: mean quality not below Jev's 0.7648, and total time not above the cheap LLM's 1,646.9 s.
- Expectation beforehand: by phase 6's throughput Kev-4B needs 2,400–2,900 s, so A2 will probably fail on time; Kev-0.8B
  is much faster but its self-reported quality is much lower, so it will probably fail on quality.
- A′ is a reading only: with the cache opened to the memory budget, how much faster each of the 10 per-review queries is,
  whether the result sets are identical and how many hits there are, with the total compared to Jev's 90.5 s. Kev-0.8B's
  budget holds all 1,864 reviews these queries touch, so its A′ is the "whole table cached" case.

### 4.2 B: prefix cache

Three arms: cache off (control), whole-table cache (open enough for every row of the dataset), factory setting (4
entries). Each run is "one dataset × one arm" against a freshly started server: warm-up on made-up rows, then the same
rows asked each question once, at concurrency 1. Off and whole-table run 3 times each in the order off/on, on/off,
off/on; the factory setting once; the long-document strata once each.

- **Primary metric**: speed-up from question 2 on = (time of questions 2 to last with the cache off) ÷ (the same with the
  whole-table cache), each question taking the median of its 3 passes.
- **Mechanism check**: with the whole-table cache, question 1 misses on every row and later questions hit on every row;
  with the cache off nothing hits. Read from the server's own counters. Datasets that do not match are reported as
  failed.
- **Answers unchanged**: the share of identical judgements between cache on and off, from question 2 on, over all 3
  passes.
- **Memory model**: the server process's memory is read after each block of rows, fitting "memory per cached text +
  memory per token".
- **Bars**
  - B1: on long documents, at least 3× faster from question 2 on.
  - B2: at least 99.5% identical judgements (Jev's own repeat agreement).
  - B3: the cache fits this model's memory budget (table in section 2).
  - Long documents are a wonder if B1, B2 and B3 all hold (40 documents).
  - Whole short-text tables are a wonder if both short-text datasets are at least 2× faster, both meet B2, and both whole
    tables (1,864 movie reviews, 3,949 restaurant sentences) fit the budget.
- Expectation beforehand (from the probes and rehearsal on made-up rows): B1 passes by a wide margin; short texts get
  about 1.4–2.0×; Kev-0.8B can hold the whole movie table but neither model can hold the whole restaurant table, so the
  "both whole tables" verdict probably fails for both models. The movie table alone is reported separately.
- Also reported: the extra cost of the pass that builds the cache, the effect of the factory setting, the time to pack
  several questions into one request, the speed-up as a function of length, and throughput with cache hits against the
  Jev API.
- Are the long-document answers meaningful (reference reading)? "Mostly positive?" AUC at least 0.8 and a rank correlation
  of at least 0.5 between score and share of positive reviews count as meaningful; otherwise the speed conclusion must
  carry that caveat.
- **Jev's long-document reference** (comparison column only, no bar): Jev's time per document and question, and the total
  time and cost of one pass of the three questions, side by side with local cache-off and cache-hit times; Jev's answers
  get the same reference readings. A cloud API has no "already read" text to reuse, so every question re-sends the whole
  document. Concurrency 1 measures single-request time; the API supports concurrency, which goes into the limitations.

### 4.3 C: calibration parameters in the database

- Parameter table `semantic_calibration(model, predicate, scope, a, b, …)`; `kev_noulc_cal(state, question, criteria,
  scope)` looks up the parameters in the database and calibrates, returning the probability unchanged if none exist.
- Three steps per dataset, all in SQL: ask the label pool once → fit on the human labels of the pool's first 200 pairs
  and store → one query over the test split returning the raw probability, the probability calibrated with this table's
  parameters, and the one calibrated with walmart-amazon's.
- The decision rule is phase 6's: accuracy at most 2 points below Jev, AUC at most 0.02 below, Brier at most 0.02 above;
  all three mean on par with Jev. Point estimates decide, with 2,000-resample bootstrap intervals reported.
- **Bar C**: on par with Jev after calibration on all three new datasets, with at least one behind before. If all are on
  par raw already, report "calibration not needed", not a wonder.
- Also reported: whether one question's parameters work on another table; the effect of 25 / 50 / 100 / 200 / 500
  labels; whether SQL and Python compute the same values; whether Kev-4B's answers on walmart-amazon match phase 6's.
- The replay part covers data already seen; it only illustrates per-predicate parameters on 22 predicates and is not a new
  test. Kev-0.8B has no phase 6 data and no replay part.

## 5 Steps

The whole run is one command:

```bash
./experiments/database/p7_all.sh
```

The main log is `out/database/p7/all.log`, each model's log `out/database/p7/<model>/run.log`.

| Step | What | Acceptance |
| --- | --- | --- |
| 1 | Kev-4B: all four steps of `experiments/database/p7_run.sh`, no analysis | inventory complete |
| 2 | Kev-0.8B: as above | as above |
| 3 | Jev's long-document reference: `experiments/database/p7_jev_long.py run` | all 120 requests answered |
| 4 | one analysis per model: `out/database/p7/<model>/report.txt` and `results.json` | |

The four steps per model (`experiments/database/p7_run.sh`):

| Step | What | Acceptance |
| --- | --- | --- |
| 0 | pre-flight checks: port free, system tools present, offline tests pass, sleep prevented | nothing starts if any fails |
| sql | start server (factory settings) → 13 queries → stop server | the command ends normally |
| budget | start server (cache at budget) → the 10 per-review queries → stop server | as above |
| cache | 29 runs in a fixed order, each with its own server start and stop | every run writes a record |
| calibration | start server (factory settings) → four matching datasets → stop server → (Kev-4B) replay phase 6's cache | as above |
| inventory | are files and tables complete, how many requests failed | |

After the command finishes:
1. Check the inventory and read both `report.txt` files.
2. Add a section to `experiments/database/README.md`: conclusions, tables, limitations, deviations, reproduction commands.
3. Append an execution record to the end of p7.txt: start and end times, failures, deviations, cost.
4. Report to the owner, conclusion first, then the evidence.

## 6 Rules fixed in advance

| # | Situation | Handling | Record |
| --- | --- | --- | --- |
| 1 | pre-flight checks fail | do not start; fix first | not a deviation |
| 2 | a step's server does not start, exits during collection, or the command fails | redo the whole step once; if it still fails keep what exists and move on | deviation; report "not completed" with the reason |
| 3 | sql / budget step resumed after an interruption | finished queries are not rerun; the interrupted query is partly cached when rerun, so its time is not cold: afterwards run that query cold once under a new label and use that time | deviation |
| 4 | a cache-step run is interrupted | discard that run and redo it from scratch on a freshly started server (records are written only at the end of a run) | deviation |
| 5 | single requests get no answer | sql, calibration: rerun the same statement at most 4 rounds, re-asking only the failures; cache: at most 3 re-ask rounds within the block, time included. Still missing: in A the query counts as computed with unanswered rows; in B as a different judgement; in C as wrong for accuracy and 0.5 otherwise | listed in the report |
| 6 | more than 5% of a C dataset's requests unanswered | that dataset is "behind"; no patching | stated in the report |
| 7 | the mechanism check fails (hit counts do not match) | B for that dataset is reported as failed; no rerun with changed settings | stated in the report |
| 8 | the whole-table cache runs out of memory (the server clears its cache and retries) | recorded as is (the record has this counter); that run's hit counts will not match, handled by rule 7 | stated in the report |
| 9 | passes differ in time by more than 15% | reported as is and flagged; the median is still used | flagged in the report |
| 10 | the computer sleeps or the script is interrupted | run the same command again; parts that have results are not rerun | deviation |
| 11 | a bug in the collection code that does not affect requests, answers or times | fix and resume | deviation, with new checksums |
| 12 | a bug that affects requests, answers or times | delete the affected part, fix, redo that part from scratch | deviation, with new checksums |
| 13 | the analysis script fails | collected results are unaffected; fix the analysis and rerun it | deviation; criteria unchanged |
| 14 | unexpected results | report as they are | no "rescue" experiments; anything added later is marked post hoc |
| 15 | Jev's reference cannot run (key, quota or network) | skip it, continue; the report leaves that column empty to fill in later | deviation |
| 16 | one model's pre-flight check fails | the whole command stops there and later models do not run; fix it and rerun the same command | not a deviation |

Not done during the run:
- No looking at any answers; only counts of requests, failures, time, hits and memory. The analysis runs after both
  models and Jev's reference are collected.
- No changes to questions, thresholds, temperatures, cache settings or criteria; no added arms, no removed data.
- No paid API except the approved Jev long-document reference.
- No use of Jev's outputs to train, fine-tune, calibrate or pick thresholds for any open model (TypeSafe customer
  agreement, section 2.3(b)); calibration uses human labels only.

## 7 Time and resources

Estimated from phase 6's measured throughput and this phase's measurements on made-up rows; read them as ±30%.

| Step | Kev-4B | Kev-0.8B |
| --- | --- | --- |
| sql (16,940 requests) | about 43 min | about 10 min |
| budget (4,716 requests) | about 9 min | about 3 min |
| cache (29 runs) | about 2.5 h | about 1.9 h |
| calibration (12,731 requests, plus replay for Kev-4B) | about 47 min | about 11 min |
| Total | about 4.2 h | about 2.3 h |

Jev's long-document reference about 3 minutes, both analyses about 5 minutes. About 6.6 hours in total, 4.6–8.6 hours at
±30%. Checking and writing the README afterwards about 30 minutes.

Resources:
- One server at a time. With the whole-table cache the server uses 42–44 GB (both models), the peak of the run, set by
  "as much memory as possible"; otherwise 3–20 GB.
- Mains power and an open lid are needed. Avoid memory-hungry programs meanwhile: during the whole-table runs only a dozen
  or so GB stay free, and a few more GB used elsewhere start paging and skew the times (flagged in the report).
- Cost: only Jev's long-document reference, about $0.03; the rehearsal has spent $0.003.

## 8 Data and privacy

- Collected results go to `out/database/p7/` (not committed).
- Local model requests go to 127.0.0.1 only, without any key; the server runs in a sandbox with no internet access.
- Only Jev's long-document reference sends anything out: 40 documents built from public Rotten Tomatoes reviews, to
  TypeSafe's official API.
- Kev-0.8B's weights are downloaded into `open_models/` inside the repository; nothing is installed system-wide.

## 9 Quality assurance done before the run

All on made-up rows; no evaluation row has been asked of any model.

| Item | Result |
| --- | --- |
| Offline tests | 44 pass (11 of them new in this phase) |
| Criteria and scripts frozen | p7.txt with checksums; the 2026-10-12 revision appended with new checksums |
| Full rehearsal | 2026-10-11: the single-model run from an empty directory, 0 failed requests. After the 2026-10-12 revision: a full Kev-4B rehearsal from an empty directory again, then the final single command rehearsed with both models, Jev's reference and both analyses |
| Failure drills | killing the server during collection: restarted automatically and the run redone from scratch. Killed twice in a row: skipped as planned, the missing run marked in inventory and report; rerunning the same command fills it in |
| Capacity probe | once per model, see the table in section 2; Kev-4B's result agrees with the memory model (at most 601 entries within the budget) |
| Analysis script | every code path exercised on the rehearsal results; a second run produces byte-identical output |
| Mechanism check | in the rehearsal, the whole-table cache misses everything on question 1 and hits everything afterwards, nothing hits with the cache off; cached answers are identical to uncached ones |
| SQL form of calibration | calibrated probabilities computed in SQL differ by 0 from Python with the same parameters; tests cover select lists, WHERE, aggregates, GROUP BY, per-row predicates, caller tables with columns named like the parameter table's, and unanswered rows |
| Replay requests | all 26,000 request keys are in phase 6's cache (only keys counted, no answers read); the replay runs on a copy of the cache |
| Earlier phases unaffected | replays of the phase 1 and phase 6 reports are byte-identical to the archive |
| Environment | only the approved Kev-0.8B weights were downloaded; nothing installed; the server still runs sandboxed |

Three problems the rehearsals caught and fixed (none raised an error; each silently produced wrong numbers):
1. DuckDB's `least` / `greatest` skip NULL, so unanswered rows in the calibration macro became 0.999.
2. A column name passed into the calibration macro (e.g. `question`) was captured by the parameter table's own column of
   the same name inside the lookup subquery, so every row used one parameter set, and only in WHERE.
3. When merging several answer comparisons, the weighting denominator of probability differences was wrong (only matters
   with missing answers).

## 10 Outputs and definition of done

1. Under `out/database/p7/<model>/`: each query's results and records in `sql/`, the 29 runs' records and answers in
   `cache/`, `calibration/calibration.duckdb`, with an inventory showing everything present (or gaps recorded under
   section 6); Jev's long-document reference under `out/database/p7/jev/`.
2. One `report.txt` and `results.json` per model.
3. A new section in `experiments/database/README.md` answering whether a local model inside DuckDB offers a wonder, and
   which, with reproduction commands.
4. The execution record at the end of p7.txt: start and end times, deviations, cost.
5. A report to the owner.

## 11 Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| The criteria were written after the probes on made-up rows | B1 was already likely to pass | the probe numbers and the resulting expectations are recorded in p7.txt; the real uncertainties are unchanged answers, A and C |
| As much memory as possible | during whole-table runs little memory is left; other programs cause paging, slow timings or the server being killed | the budget is 90% of the probed healthy limit; each run records free memory at its start; medians of 3 passes, differences above 15% flagged; server exits handled by rules 2 and 4 |
| The long-document questions are new and untested | quality readings on long documents may be low | reference readings only, not part of the verdict; questions not edited |
| Long documents come close to the 8,192 tokens Kev was validated on | single requests may exceed it | a character margin was left; the report gives the longest request |
| Times depend on the machine's state | noise in the ratio between two arms | the two arms run interleaved 3 times; every run uses a freshly started server plus warm-up |
| A's references were measured days earlier | Jev's and the LLM's times depend on the network at the time | reference columns only; quality is directly comparable |
| Jev's long-document reference is measured at concurrency 1 only | the API supports concurrency, so real throughput is better than single-request time | listed in the limitations; per-request time and cost reported together |
| The matching data are public datasets | models may have seen them in training | listed in the limitations; affects raw and calibrated results alike |
| Calibration cannot change the AUC criterion in C's rule | if AUC is already behind raw, C must fail | known in advance; the report states which criterion failed |
| Kev-0.8B is clearly weaker | A and C may be far off the bar | report as is; its point is B: a whole table fits in its cache |

## 12 Start

The owner said go (2026-10-12). After the revision is frozen:
1. Check that every script's checksum matches the last set in p7.txt and record the start time at the end of p7.txt.
2. Run `./experiments/database/p7_all.sh` in the background (detached from the session, so quitting the app does not
   affect it).
3. Meanwhile, watch only the counts in the log.
4. When the command finishes, do the four things at the end of section 5.
