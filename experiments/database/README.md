# Decision models inside DuckDB (phases 0–10)

This folder holds the experiments behind `kevdb`'s SQL functions (`src/kevdb/`): decision models executed as DuckDB
functions for semantic predicates, and their probabilities used as "semantic statistics" by the database.

Phases 0–5 use the hosted Jev API (TypeSafe AI's decision model, which answers yes/no, multiple-choice and score
questions with probabilities) and generative LLMs. From phase 6 on the subject is the open, Jev-compatible models that
run on a laptop. Phases 0–5 ran on 2026-10-07, phase 6 on 2026-10-11, phase 7 on 2026-10-12/13, phases 8 and 9 on
2026-10-13, phase 10 from 2026-10-16 to 2026-10-20. Other experiments (entity matching, classification, query plans) are
listed in [experiments/README.md](../README.md).

Every phase wrote its question, metrics, thresholds and stopping rules into [protocol/](protocol/) before any data
was collected. Results added after seeing the data are marked as post hoc.

## Summary

**Open models (phases 6–14)**

- **No open model replaces Jev out of the box.** Of the five models that run on a 64 GB Mac, none passes the
  pre-registered criteria. Kev-4B, Kev-9B and Clef-flash match Jev on movie reviews, are all behind on restaurant
  aspects, and only Kev-9B matches on product matching. The gap is the probability scale: after recalibration with 200
  human labels the three are within 1.4 points of Jev. Both Laya models are far behind and were dropped (phase 6).
- **Inside DuckDB, only one local advantage holds by the pre-registered bars: asking long documents question after
  question.** Kev's server keeps the text it has read; asking the same documents another question is 43× faster on
  Kev-4B and 23× faster on Kev-0.8B, with bit-identical answers. Kev-4B's first pass is 12× slower than the Jev API and
  pays off only after a dozen questions; Kev-0.8B's first pass is only 2.2× slower, about 10× faster once cached, and
  beats unparallelised Jev from the third question.
  On short-text tables the cache is only 1.7–1.9× faster and costs a fixed 22–57 MB per cached text: Kev-0.8B can hold
  the whole 1,864-review table (then matching the Jev API's throughput); larger tables fit on neither model.
  As a semantic-SQL backend Kev-4B scores 0.803 on SemBench (Jev 0.765) at no marginal cost but 8.7× slower; Kev-0.8B is
  only 2× slower but fails the quality bar (0.664). Calibration parameters stored in a table and applied in SQL work
  and are essential for the small model (accuracy from 66–86% to 91–98%), but both models stay behind Jev on two new
  matching datasets after calibration (phase 7).
- **Do not ask a record-matching join pair by pair.** Putting a left record's 10 candidates into one request (one
  question per candidate) cuts Jev's requests and time to a tenth and input tokens to 38% with no loss in F1, and the
  judgements do not depend on candidate order. For nearly one-to-one joins one multiple-choice question raises F1 from
  90.7% to 94.0% at 22% of the tokens. Part of this is specific to Jev: the local Kev-4B is only 1.7–1.9× faster when
  packed, and its multiple choice is strongly biased towards later options (phase 8).
- **For other join predicates packing still saves money but can change the answer.** On "do these two reviews have
  opposite sentiment?" tokens and time drop to 26–33% and 2–11%, but F1 falls from 73.3% to 65.3%: moving the second
  record from the state into the question changes the answer, regardless of how many questions share a request. Verify
  per predicate before packing (phase 9).
- **Fine-tuning on public matching labels closes the gap on unseen product data, not on a new question.** Fine-tuned on
  6,000 human-labelled pairs from three other datasets, Kev-4B matches Jev's zero-shot quality on two of four held-out
  datasets (abt-buy, walmart-amazon) without forgetting; it stays behind on the hardest product set, and on dblp-acm,
  asked a publication question no training record contains, it does not improve at all. Kev-0.8B matches on one of four
  and forgets slightly on movie reviews. Trained with in-domain labels, Kev-4B's F1 is above Jev's on all four test sets.
  Neither model meets the pre-registered verdict, which needs all four (phase 10).
- **What blocked the new question was training data of one kind, not its wording.** Rewording the product question
  changes nothing. Adding 2,000 pairs of another publication dataset asked with the same question, or 1,156 pairs of
  beer, restaurants and songs with their own questions, brings Kev-4B to Jev's level on dblp-acm (F1 0.962 and 0.945,
  Jev 0.973) without forgetting (phase 11).
- **With 300 labels of your own data, Kev-4B is on par with Jev on all four matching datasets**, measured on records no
  label contains. On product data like its public training data it needs none, on the publication question about 100.
  The public-label stage matters only below that: at 300 labels the released checkpoint with the author's recipe does as
  well. On the hardest product set F1 stays below Jev's (0.701 against 0.744 at 1,000 labels) (phase 12).
- **Verdicts near a line depend on the training seed.** With two more seeds, three of phase 10's four close verdicts
  hold, but Kev-4B on amazon-google turns from below to on par (at the edge). Kev-0.8B's slight forgetting on movie
  reviews does not recur with the new seeds (phase 13). It belongs to one training draw: retrained with seed 1 through
  the padding launcher, it comes back unchanged, so the padding fix changes nothing and seed 1's record order and replay
  sample decide it (phase 14).

**Jev and LLMs (phases 0–5)**

- **It works.** Jev as a SQL function runs all 10 queries of SemBench's movie scenario with mean quality 0.765, not
  below a cheap LLM in the same framework (0.745) or three published systems (0.689–0.741).
- **Against a cheap LLM the advantage is 3× in price and 5× in speed**, not the hundreds advertised. Against the
  published systems it is 20–30×, under different conditions.
- **"Jev first, unsure rows to an LLM" (a cascade) only works for record matching.** It needs a clearly more accurate
  upper model and Jev's errors concentrated in the rows it is unsure about. Product matching satisfies both (also on a
  second dataset); on movie reviews even Opus is barely more accurate than Jev; on restaurant reviews the stronger model
  is more accurate, but Jev's probabilities cannot pick the rows to hand over.
- **Estimating counts and shares from Jev's probabilities plus a few human labels helps but misses the pre-set bar**:
  2.6× fewer labels (target 3×). An open model's output probabilities do just as well.
- **Ordering semantic predicates with Jev's statistics is not worth it.** Even a perfect order saves only 10–30% over
  the worst one, and packing the predicates into one request gets the same without statistics.
- **The two biggest wins do not depend on the model**: rewriting a pairwise semantic join as "judge each row once, then
  an ordinary join" (8,128 calls become 128), and packing several questions about the same row into one request.

## Setup and data

Run from the repository root:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,experiments]"
./experiments/database/fetch.sh      # SemBench, Rotten Tomatoes source data, MAMS, GoEmotions, two papers; about 520 MB in experiments/database/data and refs
cp .env.example .env                 # TYPESAFE_API_KEY and OPENROUTER_API_KEY, only for re-collecting
.venv/bin/python -m pytest -q        # offline tests, no model calls
```

The exact package versions used are in [requirements.txt](requirements.txt).

Jev always goes through the official API, pinned to `jev-1.13.0`. LLMs go through OpenRouter with the provider pinned
on every route and no fallback ([src/kevdb/routes.py](../../src/kevdb/routes.py)): unpinned, six requests to one model
landed on five providers.

## SQL functions

```python
import duckdb
from kevdb import jev_backend, register
from kevdb.cli import load_dotenv

load_dotenv()
con = duckdb.connect()
register(con, "jev", jev_backend())
con.sql("""
    SELECT count(*) FROM read_csv('experiments/database/data/sembench/files/movie/data/sf_2000/Reviews.csv')
    WHERE id = 'taken_3' AND jev_noul(json_object('review', reviewText), 'Is the movie review in `review` positive?') >= 0.5
""")
```

| Function | Returns |
| --- | --- |
| `jev_noul(state, question)` | P(yes) |
| `jev_noulc(state, question, criteria JSON)` | P(yes), with criteria for what counts as yes and no |
| `jev_choice(state, question, options JSON)` | the chosen option |
| `jev_score(state, question, levels JSON)` | probability-weighted level, counted from 0 |
| `jev_nouls(state, questions JSON array)` | several yes/no questions in one request, P(yes) for each |
| `jev_match(left, candidates JSON array, question, criteria JSON)` | one block of a semantic join: one yes/no question per candidate in a single request, P(yes) for each candidate in order. That it answers like pairwise asking has only been verified on record matching (phases 8 and 9) |
| `jev_pick(left, candidates JSON array, question, "none" description)` | the same request as one multiple-choice question: position of the chosen candidate (0 if none) and its confidence. Only for joins where a left record matches at most one record (phase 8) |

Functions run per Arrow batch: identical inputs within a batch are asked once, answers go to a SQLite cache, and
failed rows return NULL and are counted. `register_cascade` registers the cascade version. Swapping the backend gives
the LLM version (`llm_backend("luna.low")`) or a local model (`open_backend("kev-4b")`, server started first).

Calibration parameters can be stored in the database like statistics (`src/kevdb/calibration.py`, phase 7):

```python
from kevdb import calibration, open_backend, register

kev = open_backend("kev-4b")
register(con, "kev", kev)
calibration.install(con, "kev", kev.model)                              # creates semantic_calibration and the macros below
calibration.fit(con, kev.model, question, criteria, "table", p, y)      # p: raw model probabilities, y: human labels (human labels only)
con.sql("SELECT count(*) FROM pairs WHERE kev_noulc_cal(state, $q, $c, 'table') >= 0.5", params={"q": question, "c": criteria})
```

`kev_noul_cal(state, question, scope)` and `kev_noulc_cal(state, question, criteria, scope)` look up the parameters by
(model, predicate, scope) and calibrate, returning the probability unchanged if none exist; the question can be a
column, so every predicate in a query uses its own parameters. `calibrated(p, model, predicate, scope)` works on any
probability column. The full reference is in [docs/sql-functions.md](../../docs/sql-functions.md).

## Results

### Phase 0: which LLM takes over the rows Jev is unsure about?

The data are the 637 Abt-Buy product pairs Jev was unsure about. The rule, fixed before the run: the cheapest route
within 3 points of Opus.

| Route | Accuracy | Cost per 1,000 calls | Median latency |
| --- | --- | --- | --- |
| Opus 5.5 (reference, measured earlier) | 86.7% | | |
| **GPT 6 Luna, low reasoning** | **84.5%** | $0.068 | 2.6 s |
| DeepSeek V4.1 Flash, low reasoning | 82.1% | $0.448 | 1.0 s |
| GPT 6 Luna, reasoning off | 81.8% | $0.041 | 1.1 s |
| Qwen 3.8 Flash, low reasoning | 81.8% | $0.283 | 4.9 s |
| DeepSeek V4.1 Flash, reasoning off | 80.2% | $0.115 | 0.4 s |
| Qwen 3.8 Flash, reasoning off | 79.0% | $0.059 | 1.3 s |
| DeepSeek V4 Pro, reasoning off / low | 75.5% / 75.4% | $0.185 / $1.71 | 1.0 / 8.3 s |
| Jev (reference) | 74.9% | | |

Only GPT 6 Luna with low reasoning qualifies; "the LLM" below means this route. Throughput: Jev 51 requests per second
at concurrency 16; Luna 13 per second at concurrency 32.

### Phase 1: the 10 queries of SemBench's movie scenario

2,000 reviews. Jev ran cold three times (median taken; the three runs barely differ), the LLM and the cascade once each.
Scoring uses SemBench's own code.

| Query | Metric | Jev | LLM | Cascade | Four published systems |
| --- | --- | --- | --- | --- | --- |
| Q1 five positive reviews | F1 | 1.000 | 1.000 | 1.000 | 0.952–1.000 |
| Q2 five positive reviews of one movie | F1 | 0.600 | 0.600 | 0.600 | 0.923–1.000 |
| Q3 count positive reviews | relative error | 0.214 | 0.214 | 0.143 | 0.357–0.571 |
| Q4 share of positive reviews | relative error | 0.214 | 0.143 | 0.143 | 0.357–0.571 |
| Q5 ten pairs with the same sentiment | F1 | 0.900 | 0.800 | 0.800 | 0.316–1.000 |
| Q6 ten pairs with opposite sentiment | F1 | 0.600 | 0.400 | 0.400 | 0.632–0.850 |
| Q7 all pairs with opposite sentiment | F1 | 0.732 | 0.769 | 0.772 | 0.202–0.695 |
| Q8 grouped counts | relative error | 0.040 | 0.000 | 0.040 | 0.081–0.323 |
| Q9 score reviews | correlation | 0.795 | 0.763 | | 0.748–0.780 |
| Q10 rank movies | correlation | 0.412 | 0.421 | | 0.404–0.438 |

| All 10 queries | Mean quality | Total time | Total cost |
| --- | --- | --- | --- |
| Jev | 0.765 | 313 s | $0.28 |
| LLM (GPT 6 Luna, low reasoning, concurrency 32) | 0.745 | 1,647 s | $0.85 |
| Published: BigQuery / LOTUS / Palimpzest | 0.741 / 0.689 / 0.721 | 401 / 1,544 / 1,400 s | $5.47 / $5.94 / $8.23 |

Mean quality follows SemBench (relative error mapped to 1 / (1 + error)). ThalamusDB did not run Q9 and Q10; on the
first 8 queries it scores 0.825 and Jev 0.805.

- **Cascade**: with the earlier thresholds (probability 0.1–0.9 counts as unsure), 18–63% of rows are escalated; time
  and cost approach the LLM-only run with no gain in quality. On the reviews Jev is unsure about, the LLM agrees with the
  human labels 63% of the time and Jev 69%.
- **Rewritten joins** (Q5r–Q7r): judging the sentiment of the 128 reviews once and then joining raises Q7's F1 from
  0.73 to 0.79; all 10 queries then take 90 s and $0.07.
- **Top-k by probability** (Q1p, Q2p, Q5p, Q6p, added after seeing Q2 at 0.6): all four LIMIT queries reach F1 1.0.

### Phase 3: semantic aggregation

15,388 distinct reviews, 81.3% labelled positive by humans. Primary analysis and success criteria are in
[protocol/p3.txt](protocol/p3.txt).

Without labels: Jev threshold count 75.4% (5.9 points off), mean of Jev's probabilities 70.0% (11.3 off), LLM on every
row 72.9% (8.4 off).

With 200 human labels (2,000 random draws each):

| Method | Error (RMSE, points) | 95% interval coverage | Labels saved vs sampling only |
| --- | --- | --- | --- |
| Sampling only | 2.73 | 95.2% | 1.0× |
| Jev probability + difference estimator (primary) | 1.72 | 94.5% | 2.5× (theory 2.62) |
| Stratified by Jev probability | 1.38 | 95.5% | 3.9× |
| LLM on every row + difference estimator | 2.18 | 94.8% | 1.6× |

- **The primary bar on labels saved (≥3×) is not met**; the coverage bar (≥90%) is.
- Share positive per movie (87 movies): calibrating Jev's probabilities with 200 labels drawn from the whole table gives
  a mean error of 3.7 points per movie; Jev used directly 5.9; the same labels spread across the movies 17.7.
- Jev asked the same 2,000 reviews twice agrees with itself on 99.45%.
- On restaurant reviews the method barely helps: the regression estimator saves 1.3×, the difference estimator is
  worse than sampling only (0.87×). The gain depends on how well the model agrees with the labels, which vanishes when
  the question's wording and the labelling guidelines diverge.

**Attribution: the gain comes from having probabilities, not from Jev**

| Source (movie data, 5,001 rows) | Per-row agreement | AUC | Labels saved | Cost per 1,000 rows | Latency |
| --- | --- | --- | --- | --- | --- |
| Jev's probability | 91.5% | 0.972 | 2.5 | $0.013 | 0.29 s |
| DeepSeek V4.1 Flash output probability, reasoning off | 91.5% | 0.975 | 2.2–2.6 | $0.060 | 0.42 s |
| DeepSeek V4 Pro output probability | 89.3% | 0.965 | 1.8–2.2 | $0.100 | 0.92 s |
| Qwen 3.8 Flash output probability | 86.7% | 0.968 | 1.9–2.1 | $0.030 | 2.16 s |
| GPT 6 Luna, yes or no only | 89.6% | | 1.5–2.0 | $0.036 | 2.49 s |

On the 21 restaurant predicates the mean AUC is 0.871 for Jev and 0.863 for DeepSeek V4.1 Flash. Both datasets agree:
an open model's output probabilities with reasoning off are as useful as Jev's; Jev's advantage is price and speed.
With reasoning on, LLM output probabilities collapse to 0 or 1 and are useless.

### Phase 4: selectivity, ordering, packing

**Selectivity** (estimated from 50 sampled rows; median factor off the truth):

| Method | Movies, 87 subsets, two predicates | Restaurant reviews, 21 predicates |
| --- | --- | --- |
| Default guess of 1/3 | 2.67 | 3.36 |
| Jev on a sample | 1.13 | 2.10 |
| Human labels on a sample (ceiling) | 1.04 | 1.31 |
| Jev on a sample, vs Jev's own pass rate | | 1.19 |

The movie data passes the stopping bar (≤2); restaurant reviews do not: when wording and labelling guidelines diverge,
Jev estimates how many rows *it* will pass, not the selectivity under the human truth. The former is enough for an
optimizer's cost estimates, not for estimating result sizes.

**Is the LLM stronger than Jev?** (restaurant reviews, 1,000 sentences × 21 predicates, one question per request):
Jev 84.0%, LLM 84.1%. The 0.1-point gap is below the pre-set 2 points, so the "escalate to the LLM and order the
predicates" experiment was not run.

**Value of ordering when only Jev executes** (all conjunctions of 2 or 3 of the 21 predicates, 1,540 queries):

| Ratio to the hindsight-best order (median) | 2 predicates | 3 predicates |
| --- | --- | --- |
| Worst order | 1.11 | 1.27 |
| Ordered by pass rates Jev estimated from 50 rows | 1.00 | 1.00 |
| Packed into one request: requests / tokens | 0.89 / 0.97 | 0.89 / 1.04 |

Jev's sample does pick the right order, but best and worst differ by only 10–30%, and packing matches it without
statistics.

**Packing** (the 21 questions about one sentence in one request):

| Model | One by one | Packed | Agreement between forms | Request latency | Cost per 1,000 sentences |
| --- | --- | --- | --- | --- | --- |
| Jev | 84.0% | 84.1% | 99.5% | 0.29 → 0.29 s | $0.284 → $0.038 |
| GPT 6 Luna, low reasoning | 84.1% | 88.0% | 94.1% | 2.27 → 3.51 s | $0.801 → $0.220 |
| DeepSeek V4.1 Flash, reasoning off | 86.5% | 88.6% | 93.4% | 0.45 → 0.76 s | $1.329 → $0.477 |

- Packed, Jev's answers and latency do not change; the whole table (3,949 sentences × 21 questions) drops from 1,561 s
  to 76 s.
- Packing is not specific to Jev. LLMs answer 21 questions at once too, and gain 2–4 points of accuracy, overtaking Jev.
- Packed, Jev is still cheapest and fastest: $0.038 per 1,000 sentences, a sixth of Luna and a twelfth of DeepSeek.
- One SQL query with three predicates (3,949 sentences, cold): one by one 117 s, 6,177 requests, $0.085; packed with
  `jev_nouls` 78 s, 3,949 requests, $0.063.

**Cascade with the packed LLM as upper model** (a replay added afterwards): handing the 30% of rows Jev is least sure
about to the LLM recovers only 43% of the gap (random choice gives 30%; the pre-set bar was 60%). On this data Jev's
probabilities cannot find the rows the LLM would fix.

### Phase 5: a truly stronger upper model

The whole remaining balance ($11.24) went to Opus 5.5 as the upper model; questions and reading rules are in
[protocol/p5.txt](protocol/p5.txt).

| Data | Upper model's accuracy vs Jev | Cascade |
| --- | --- | --- |
| Product matching, Walmart-Amazon (positive control, 2,049 pairs) | 7.5 points higher on the 214 pairs Jev is unsure about (83.6% → 91.1%) | works: F1 from 0.884 to 0.929 (95% interval +0.016 to +0.074), escalating only 10.4% of rows |
| Movie reviews (1,851) | 1.0 point higher overall (95% interval −0.1 to +1.9) | nothing to gain: below the 2-point threshold; the labels cap every model |
| Restaurant reviews (546 sentences × 21 predicates, packed) | 3.3 points higher (+2.7 to +3.9) | fails: escalating the least certain 30% recovers 36% (28–46%); the bar is 60% |

The reason is visible without any upper model: the share of Jev's errors that falls in its least certain 30% of rows.
On six matching datasets it is 91–100%, on movie reviews 80%, on restaurant predicates 67%. On the last two Jev is
confidently wrong, because the question's wording and the labelling guidelines diverge. This curve takes a few hundred
labels to draw and tells in advance whether a predicate is worth cascading.

On restaurant reviews Opus (87.1%) is not more accurate than the cheaper GPT 6 Luna (87.7%) or DeepSeek V4.1 Flash
(88.5%).

### Phase 6: can open models replace Jev as the default backend?

After Jev's release a number of open, compatible models appeared: they take the same request and return probabilities
too, but their quality numbers are all self-reported. The five that run on a 64 GB Mac (M3 Max) were put in Jev's place
unchanged and asked the questions Jev had answered before, with states and questions unchanged to the letter. Criteria
are in [protocol/p6.txt](protocol/p6.txt), the execution plan in [protocol/p6_plan.md](protocol/p6_plan.md).

| Model | Pinned version | Size | How it runs |
| --- | --- | --- | --- |
| Kev-4B | code `kev-1.0`, weights `jaredpalmer/kev-4b@v1.0` | Qwen3.5-4B with an adapter and decision head | MLX |
| Kev-9B | `jaredpalmer/kev-9b@v1.0` | Qwen3.5-9B, same approach | MLX |
| Clef-flash | `Cloudflare/clef-flash@8b2e5fd1` | 9.4B, based on Qwen3.5-9B | PyTorch on the Mac GPU, with its bundled loading code |
| Laya English (dropped) | `laya` 0.4.1 | 421M, ModernBERT-large encoder | its own server |
| Laya multilingual (dropped) | same, multilingual weights | 322M, mmBERT-base encoder | same |

Data: 5,000 movie reviews (5,001 labelled rows), 1,000 restaurant sentences × 21 predicates (one request per question)
and the 2,049 pairs of the Walmart-Amazon product-matching test set. 29,249 requests per model, all answered. A
probability ≥ 0.5 counts as yes.

**Decision rule** (fixed before collection): on one dataset a model is on par with Jev if its accuracy is at most 2
points lower, its AUC at most 0.02 lower and its Brier score at most 0.02 higher; it can replace Jev only if it is on
par on all three datasets.

**Result: none of the five replaces Jev unchanged.**

Each cell is accuracy / AUC / Brier; ✓ on par with Jev, ✗ behind.

| Source | Movie reviews | Restaurant reviews (one by one) | Product matching | Replaces Jev |
| --- | --- | --- | --- | --- |
| Jev (cached, reference) | 91.5% / 0.972 / 0.071 | 84.0% / 0.871 / 0.124 | 97.9% / 0.987 / 0.019 | |
| Jev (asked again that day, control) | 91.6% / 0.972 / 0.071 | 84.0% / 0.871 / 0.124 | 97.9% / 0.985 / 0.019 | |
| Kev-4B | 90.7% / 0.966 / 0.072 ✓ | 82.9% / 0.850 / 0.126 ✗ | 93.9% / 0.981 / 0.054 ✗ | no |
| Kev-9B | 91.6% / 0.970 / 0.063 ✓ | 79.4% / 0.841 / 0.158 ✗ | 96.5% / 0.974 / 0.032 ✓ | no |
| Clef-flash | 90.4% / 0.967 / 0.077 ✓ | 81.9% / 0.843 / 0.145 ✗ | 95.8% / 0.984 / 0.042 ✗ | no |
| DeepSeek V4.1 Flash output probability (reference) | 91.5% / 0.975 / 0.068 ✓ | 86.5% / 0.863 / 0.119 ✓ | no data | |
| No-skill baseline (always the base rate) | 82.4% / 0.500 / 0.145 | 87.5% / 0.500 / 0.096 | 90.6% / 0.500 / 0.085 | |

- **Close calls.** Kev-4B misses on restaurant reviews only by AUC: 0.021 lower against a bar of 0.02 (95% interval
  0.015–0.028), not separable from the bar; it clearly fails on product matching (4.0 points lower, interval 3.0–5.0).
  Clef-flash is 2.1 points lower in accuracy on both restaurant reviews and product matching, right at the bar. Kev-9B is
  clearly behind on restaurant reviews (4.7 points lower) and on par on product matching, with an interval crossing the
  bar.
- **Bigger is not better.** On restaurant reviews Kev-9B is 3.5 points below Kev-4B; on product matching 2.6 points above.
- **Control group.** Asking Jev every question again that day agrees with the cache on 99.5–99.8% of judgements, with a
  mean probability difference of 0.004–0.006; with that day's Jev as the reference no verdict above changes. The five
  local models asked the first 200 reviews again gave identical answers.
- By the same criteria, the only open source on par with Jev is still the output probability of an open LLM with
  reasoning off, from phases 2 and 4.

**Calibration** (equal-mass ECE, lower is better). These are the first calibration numbers for these models not
measured by their authors.

| | Movie reviews | Restaurant reviews | Product matching |
| --- | --- | --- | --- |
| Jev | 0.116 | 0.148 | 0.026 |
| Kev-4B | 0.097 | 0.148 | 0.134 |
| Kev-9B | 0.089 | 0.207 | 0.062 |
| Clef-flash | 0.100 | 0.153 | 0.115 |

On movie and restaurant reviews the three models are broadly no worse calibrated than Jev (except Kev-9B on restaurant
reviews), but there Jev itself is off, because the questions and the labelling guidelines diverge. On the cleanly
labelled product matching the gap shows: Jev 0.026, the open models 0.062–0.134. They put many non-matching pairs in the
middle: 54% of Kev-4B's rows fall between 0.1 and 0.9 with a mean probability of 0.31, while only 10% of them truly match.

**After recalibration with 200 human labels** (accuracy / Brier on the remaining rows, mean of 200 draws):

| | Movie reviews | Restaurant reviews | Product matching |
| --- | --- | --- | --- |
| Jev | 92.8% / 0.050 | 89.0% / 0.079 | 97.9% / 0.018 |
| Kev-4B | 92.1% / 0.056 | 87.6% / 0.087 | 97.2% / 0.023 |
| Kev-9B | 92.8% / 0.052 | 88.0% / 0.088 | 97.3% / 0.022 |
| Clef-flash | 92.7% / 0.054 | 87.8% / 0.092 | 97.5% / 0.020 |

The three models come within 1.4 points of Jev. They lose on the probability scale, not on ranking: their AUC was only
0.002–0.030 lower to begin with.

**Other readings.**
- Packing: with all 21 questions in one request, Jev's and Kev's judgements barely change (99.5–99.9% agreement).
  Clef-flash changes 5.4% of its judgements, for the better: accuracy from 81.9% to 85.7%, AUC from 0.843 to 0.874, Brier
  from 0.145 to 0.108, packed not below Jev (84.1% / 0.871 / 0.124). The verdict uses one-by-one results as planned.
- Error concentration: on product matching 98%, 92% and 93% of Kev-4B's, Kev-9B's and Clef-flash's errors fall in their
  own least certain 30% of rows (Jev 95%). A cascade has the same precondition for them.
- Throughput on this machine (requests per second; in brackets, share of Jev's cold run):

| | Movie reviews | Restaurant reviews (one by one) | Product matching |
| --- | --- | --- | --- |
| Jev (official API, concurrency 16) | 52 | 53 | 38 |
| Kev-4B | 8.2 (16%) | 13.1 (25%) | 4.5 (12%) |
| Kev-9B | 4.9 (9%) | 7.8 (15%) | 2.5 (7%) |
| Clef-flash | 2.5 (5%) | 2.5 (5%) | 1.7 (4%) |

  By the pre-set 10% line, Kev-4B is fast enough for batch use; Kev-9B and Clef-flash are not. The whole collection took
  7 h 48 min, of which Clef-flash took 4 h 27 min.

**What it means for `kevdb`.** The default backend should still be Jev; without it, the safer substitute on current
evidence is an open LLM's output probabilities with reasoning off. Of the local models Kev-4B comes closest: on par on
movie reviews, at the bar on restaurant reviews, fast enough, but on record matching it needs calibration with about two
hundred labels first.

**Where they lose** (post-hoc breakdown, not in the pre-registered analysis; `p6_breakdown.py`).
- On product matching all three judge non-matching pairs as matches. Of 2,049 pairs 193 truly match; false matches:
  Jev 18, Kev-4B 104, Kev-9B 54, Clef-flash 72, while all three miss fewer true matches than Jev (15–21 against 26).
  Their mean probability on non-matching pairs is 0.09–0.17, Jev's 0.04. So calibrating the scale removes most of the gap.
- On restaurant reviews Kev-9B loses on "is there a negative (or positive) opinion about aspect X": it says yes for
  26.6% and 23.6%, against 8.7% and 7.6% truly, a false-alarm rate of 22% and 19% against 12–16% for Jev and Kev-4B. On
  "is aspect X mentioned" all three are level (79–80%).
- Clef-flash one by one scores 83.3% and 83.7% on the two opinion question types and, packed, 87.8% and 88.8%, above
  Jev's 85.2% and 87.0%: only with all 21 questions in view does it tell which aspect an opinion targets. Jev and Kev give
  the same results one by one and packed.

**Precision is not the cause** (post-hoc check; `p6_precision.py`). In the experiment Kev ran as its server does by
default, in half precision, using half the memory of full precision. Asking the first few hundred rows of each dataset
again in full precision (1,020 requests): Kev-4B changes 1 judgement, Kev-9B none, with a mean probability difference
of about 0.002. Clef-flash is released in half precision only. No model had a memory cap.

**Dropped: Laya.** Both versions are far behind Jev on all three datasets and were removed from the candidates after
the experiment (no longer installed or run by default). Their numbers (accuracy / AUC / Brier):

| | Movie reviews | Restaurant reviews (one by one) | Product matching | Accuracy after recalibration |
| --- | --- | --- | --- | --- |
| Laya English | 17.6% / 0.503 / 0.824 | 76.8% / 0.759 / 0.154 | 18.8% / 0.490 / 0.469 | 82.4% / 85.4% / 90.6% |
| Laya multilingual | 47.8% / 0.865 / 0.419 | 81.3% / 0.695 / 0.139 | 9.4% / 0.494 / 0.885 | 85.2% / 85.1% / 90.6% |

The English version answers "no" to every movie review (17.6% is the share of negatives); its own documentation
states that it ignores the text on yes/no questions without criteria. The multilingual version ranks movie reviews
(AUC 0.865), but its probabilities are low across the board, so thresholding at 0.5 gives only 47.8%. On product
matching both have an AUC of about 0.49, no discrimination at all, and recalibration only brings them back to the
no-skill baseline. Their strength is speed: 42–68 requests per second, on the order of the Jev API.

Cost: Jev control group $0.43, rehearsal before the run about $0.01; local models are free.

### Phase 7: what does a local model inside DuckDB do that a cloud API cannot?

Without any cloud API, a decision model that runs locally sits behind `kevdb`. Phase 6 showed that Kev-4B is slow (8–13
requests per second against the Jev API's 52) and below Jev in raw quality, so this phase tests three things only a
local model can do, or that only pay off locally, each with a bar fixed before collection. Criteria are in
[protocol/p7.txt](protocol/p7.txt), the execution plan in [protocol/p7_plan.md](protocol/p7_plan.md). The subjects are
Kev-4B (identical to phase 6) and the smaller Kev-0.8B (`jaredpalmer/kev-0.8b@v1.0`), both on MLX in bfloat16 with the
server in a sandbox, on an M3 Max with 64 GB. Data, questions, analysis and bars are identical for both models.

| | Test | Bar fixed beforehand | Kev-4B | Kev-0.8B |
| --- | --- | --- | --- | --- |
| A | Semantic-SQL backend on SemBench's 10 movie queries | qualifying: mean quality not below the cheap LLM (0.745). Wonder: not below Jev (0.765) and total time not above the cheap LLM (1,647 s) | qualifies (0.803), no wonder (2,725 s) | does not qualify (0.664; 613 s) |
| B | The server keeps the text it has read; ask the same long documents another question | later questions at least 3× faster, at least 99.5% identical judgements, cache within the memory budget | **wonder**: 43.3× faster, 100% identical, 9.1 GB | **wonder**: 22.9× faster, 100% identical, 3.4 GB |
| B | The same on whole short-text tables | at least 2× faster, identical judgements, both whole tables within the budget | fails: 1.76× and 1.90×; neither table fits | fails: 1.65× and 1.75×; the movie table fits, the restaurant table does not |
| C | Per-predicate calibration parameters fitted on 200 human labels, stored in a table, applied in SQL | on par with Jev after calibration on three matching datasets Kev never answered, with at least one behind before | fails: two still behind after calibration | fails: two still behind after calibration |

Two properties need no test: no marginal cost per request, and the data never leaves the machine (the server has no
network access).

**A: semantic-SQL benchmark.** Queries and questions identical to phase 1, server at factory settings, every query cold,
scored with SemBench's own code. 16,940 requests per model, all answered.

| Query | Metric | Kev-4B | Kev-0.8B | Jev | LLM | Time (s): Kev-4B | Kev-0.8B | Jev | LLM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 five positive reviews | F1 | 1.000 | 1.000 | 1.000 | 1.000 | 225 | 51 | 35 | 153 |
| Q2 five positive reviews of one movie | F1 | 0.600 | 0.800 | 0.600 | 0.600 | 14 | 3 | 2 | 11 |
| Q3 count positive reviews | relative error | 0.071 | 1.071 | 0.214 | 0.214 | 15 | 3 | 3 | 11 |
| Q4 share of positive reviews | relative error | 0.071 | 1.071 | 0.214 | 0.143 | 15 | 3 | 2 | 11 |
| Q5 ten pairs with the same sentiment | F1 | 1.000 | 1.000 | 0.900 | 0.800 | 346 | 77 | 38 | 197 |
| Q6 ten pairs with opposite sentiment | F1 | 1.000 | 0.500 | 0.600 | 0.400 | 350 | 76 | 38 | 217 |
| Q7 all pairs with opposite sentiment | F1 | 0.530 | 0.666 | 0.732 | 0.769 | 1,412 | 323 | 154 | 820 |
| Q8 grouped counts | relative error | 0.121 | 0.526 | 0.040 | 0.000 | 15 | 4 | 3 | 16 |
| Q9 score reviews | correlation | 0.742 | 0.700 | 0.795 | 0.763 | 22 | 5 | 3 | 13 |
| Q10 rank movies | correlation | 0.396 | 0.358 | 0.412 | 0.421 | 312 | 68 | 36 | 198 |

| All 10 queries | Mean quality | Total time | Total cost |
| --- | --- | --- | --- |
| Kev-4B (local) | 0.803 | 2,725 s | no marginal cost |
| Kev-0.8B (local) | 0.664 | 613 s | no marginal cost |
| Jev | 0.765 | 313 s | $0.28 |
| LLM (GPT 6 Luna, low reasoning) | 0.745 | 1,647 s | $0.85 |

- Kev-4B's 0.038 lead over Jev comes mainly from Q6 (1.0 vs 0.6), Q3 and Q4. LIMIT queries look at only 10 rows, so one
  row moves F1 by 0.1 and the gap is within that granularity; on Q7 (all pairs, no LIMIT) it is clearly lower. It takes
  8.7× Jev's time and 1.65× the cheap LLM's; the three pairwise queries (12,224 requests) account for 2,108 s.
- Kev-0.8B is 4.4× faster than Kev-4B, at 2× Jev's time and 0.37× the cheap LLM's, but fails the qualifying bar: on the
  two counting queries its relative error is 1.07 because it judges far too many reviews positive. This is a scale
  problem (see C).
- With the joins rewritten ("judge each review once, then an ordinary join"): Kev-4B 0.761 in 667 s; Kev-0.8B 0.725 in
  148 s; Jev 0.771 in 90 s; LLM 0.745 in 456 s.
- The same 10 queries with the server's cache opened to the memory budget. Kev-4B's budget of 600 entries cannot hold the
  whole table: 609 s, only 9% faster, 18% hit rate, no change on the two queries scanning all 1,864 reviews. Kev-0.8B's
  budget of 1,900 entries holds the whole table: 107 s, 1.39× faster, 60% hit rate, every query after the first 1.7–2.2×
  faster, close to Jev's 90 s in total. Both models' result sets are identical to the factory-setting runs.

**B: prefix cache.** Kev's server has this cache built in and keeps 4 texts by default; here the environment variable
`KEV_PREFIX_CACHE` opens it wide enough for a whole dataset. Each of three arms (cache off, whole-table cache, factory
setting) runs against a freshly started server, asking the same rows one question after another at concurrency 1; off
and on run 3 times each, interleaved, and medians are reported. The long documents join one movie's reviews into one
text (at most 24,000 characters, 40 documents, movies outside SemBench); the three questions are phase 1's, rewritten
for a group of reviews. The memory budget follows "use as much memory as possible": each model probed on made-up rows
the largest cache that causes no paging, and took 90% of it — Kev-4B 600 entries, 34.3 GB; Kev-0.8B 1,900 entries,
41.2 GB. That fixed the short-text sizes: Kev-4B takes the first 580 movie reviews and the first 580 restaurant
sentences; Kev-0.8B the whole movie table (1,864) and the first 1,880 restaurant sentences.

Time per row from question 2 on (seconds) and speed-up:

| Data | Tokens per request | Kev-4B off | whole-table cache | speed-up | Kev-0.8B off | whole-table cache | speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Movie reviews | 54–55 | 0.131 | 0.075 | 1.76 | 0.032 | 0.020 | 1.65 |
| Restaurant reviews | 56 | 0.116 | 0.061 | 1.90 | 0.030 | 0.017 | 1.75 |
| Long-document prefixes, 1,500 characters | 351 | 0.343 | 0.084 | 4.07 | 0.069 | 0.023 | 2.97 |
| 3,000 characters | 711 | 0.566 | 0.085 | 6.68 | 0.116 | 0.024 | 4.86 |
| 6,000 characters | 1,428 | 1.058 | 0.087 | 12.2 | 0.205 | 0.025 | 8.13 |
| 12,000 characters | 2,862 | 2.081 | 0.092 | 22.5 | 0.383 | 0.029 | 13.3 |
| Long documents (40) | 5,768 | 4.470 | 0.103 | 43.3 | 0.818 | 0.036 | 22.9 |

- **With a cache hit, time hardly depends on text length** (Kev-4B 0.06–0.10 s per row, Kev-0.8B 0.02–0.04 s), so the
  longer the text, the more is saved.
- **Answers do not change**: for both models and all 7 datasets, cached answers are bit-identical to the cache-off ones;
  repeated runs of an arm are bit-identical too.
- **The mechanism checks out**: with the whole-table cache, question 1 misses everywhere and every later question hits;
  with the cache off nothing hits. The factory setting (4 entries) never hits in this "whole batch, next question"
  pattern and is as slow as cache off.
- **Building the cache costs almost nothing**: question 1 is 0–5% slower than with the cache off (one Kev-4B stratum
  9.8%).
- **Memory is the hard limit.** Each cached text costs Kev-4B 56.9 MB plus 31.1 KB per token, and Kev-0.8B 21.7 MB plus
  11.7 KB per token (R² 0.9997 and 0.9999). The fixed part comes from the model architecture, however short the text.

  | Memory for a whole table's cache | Kev-4B (budget 34.3 GB) | Kev-0.8B (budget 41.2 GB) |
  | --- | --- | --- |
  | Movie reviews, 1,864 | 107 GB, does not fit | 40.7 GB, fits: the whole table was actually run, the server used 43.5 GB, no out-of-memory |
  | Restaurant reviews, 3,949 | 226 GB, does not fit | 86 GB, does not fit |
  | Long documents, 40 | 9.1 GB | 3.4 GB |

- **With the whole movie table cached** (Kev-0.8B), from question 2 on it answers 51 rows per second, level with the
  Jev API's 52 requests per second; 31 with the cache off. Kev-4B with cache hits does 13 rows per second.
- **Against packing**: when all questions are known in advance they can be packed into one request. Kev-4B packed takes
  233 s for movie reviews (4 questions), 97 s for restaurants (3) and 183 s for long documents (3); one by one with the
  cache 200, 139 and 176 s, without it 297, 202 and 524 s. Kev-0.8B packed 142, 73 and 37 s; cached one by one 166, 122
  and 34 s; uncached 236, 168 and 97 s. The cache makes questions added one at a time afterwards about as cheap as
  packing them in advance.
- **Timing noise**: Kev-4B's passes differ by at most 8%. Kev-0.8B's three main datasets differ by 20–30% between passes,
  above the pre-set 15%: its whole-table cache took 43.5 GB, and one pass started with only 43 GB of system memory free.
  Medians are used; per pass the speed-ups are 1.62–1.69 (movie), 1.75–2.08 (restaurants) and 19.6–22.9 (long
  documents), so the verdict is unaffected.
- The long-document answers are meaningful (reference reading, 40 documents): "are most reviews positive?" accuracy and
  AUC Kev-4B 87.5% and 0.965, Kev-0.8B 92.5% and 0.993; rank correlation of the score with the share of positive reviews
  0.960 and 0.932. The longest request has 6,520 tokens, within the 8,192 both models were validated on.

The same 40 long documents and three questions through the official Jev API (concurrency 1; the only paid step of this
phase):

| | Jev | Kev-4B off | Kev-4B cache hit | Kev-0.8B off | Kev-0.8B cache hit |
| --- | --- | --- | --- | --- | --- |
| Time per document and question (median) | 0.34–0.37 s | 4.2–4.5 s | 0.08–0.12 s | 0.78–0.85 s | 0.03–0.04 s |
| One question over 40 documents | 16 s, $0.010 | 166–180 s | 3.2–5.0 s | 31–34 s | 1.3–1.6 s |
| All three questions | 49 s, $0.031 | 524 s | 176 s | 97 s | 34 s |
| Accuracy / AUC / rank correlation | 95.0% / 0.995 / 0.958 | 87.5% / 0.965 / 0.960 | same | 92.5% / 0.993 / 0.932 | same |

A cloud API has no "already read" text to reuse: every question re-sends and re-bills the whole document.
- Kev-4B: with cache hits each request is 3–4.5× faster than Jev, but the first pass is 12× slower. Post-hoc arithmetic:
  the same documents must be asked 14 questions before the cumulative time drops below unparallelised Jev.
- Kev-0.8B: the first pass is only 2.2× slower than Jev and cache hits are about 10× faster. The three questions take
  34 s in total, already below Jev's 49 s (post-hoc arithmetic: level at question 2, ahead from question 3). On these
  three questions its answers are close to Jev's in quality.
- The Jev API supports concurrency; this measures single requests at concurrency 1, and Jev's total time shrinks with
  concurrency.

So this advantage applies when texts are long, the same texts are asked question after question, and the data must not
leave the machine or should not be billed per token.

**C: calibration parameters in the database.** The table `semantic_calibration` stores two numbers per (model,
predicate, scope), and `kev_noulc_cal(...)` looks them up and calibrates. Each dataset is fitted on a fixed set of 200
human-labelled validation pairs and evaluated on the test split; the rule is phase 6's (accuracy at most 2 points below
Jev, AUC at most 0.02 below, Brier at most 0.02 above). Each cell is accuracy / AUC / Brier / F1; ✓ on par with Jev.

| Test set | Jev | Kev-4B raw | + this table's parameters | + walmart-amazon's parameters |
| --- | --- | --- | --- | --- |
| abt-buy (new) | 98.0% / 0.996 / 0.014 / 0.910 | 96.2% / 0.993 / 0.034 / 0.844 ✓ | 97.0% / 0.993 / 0.021 / 0.853 ✓ | 97.1% / 0.993 / 0.021 / 0.870 ✓ |
| amazon-google (new) | 92.1% / 0.961 / 0.057 / 0.695 | 85.2% / 0.937 / 0.117 / 0.557 ✗ | 90.7% / 0.937 / 0.062 / 0.283 ✗ | 91.3% / 0.937 / 0.065 / 0.635 ✗ |
| dblp-acm (new, another question) | 99.0% / 0.999 / 0.009 / 0.973 | 92.0% / 0.985 / 0.045 / 0.742 ✗ | 92.2% / 0.985 / 0.045 / 0.750 ✗ | no parameters for this question, returned unchanged |
| walmart-amazon (seen in phase 6) | 97.9% / 0.987 / 0.019 / 0.884 | 93.9% / 0.981 / 0.054 / 0.733 ✗ | 97.4% / 0.981 / 0.022 / 0.856 ✓ | same |

| Test set | Jev | Kev-0.8B raw | + this table's parameters | + walmart-amazon's parameters |
| --- | --- | --- | --- | --- |
| abt-buy | 98.0% / 0.996 / 0.014 / 0.910 | 70.3% / 0.974 / 0.202 / 0.418 ✗ | 94.8% / 0.974 / 0.037 / 0.728 ✗ | 94.2% / 0.974 / 0.041 / 0.663 ✗ |
| amazon-google | 92.1% / 0.961 / 0.057 / 0.695 | 66.9% / 0.923 / 0.216 / 0.373 ✗ | 91.2% / 0.923 / 0.062 / 0.309 ✗ | 92.0% / 0.923 / 0.060 / 0.476 ✗ |
| dblp-acm (another question) | 99.0% / 0.999 / 0.009 / 0.973 | 85.7% / 0.995 / 0.150 / 0.716 ✗ | 97.9% / 0.995 / 0.017 / 0.944 ✓ | no parameters for this question, returned unchanged |
| walmart-amazon | 97.9% / 0.987 / 0.019 / 0.884 | 66.0% / 0.960 / 0.219 / 0.350 ✗ | 95.6% / 0.960 / 0.034 / 0.720 ✗ | same |

- Kev-4B: on abt-buy it is on par raw, and calibration helps a little. On amazon-google calibration adds 5.5 points of
  accuracy and halves Brier, but its AUC is 0.024 below Jev (bar 0.02, interval crossing it); calibration does not change
  ranking, so by the rule it stays behind. Thresholded at 0.5, recall also drops to 17.9% and F1 from 0.557 to 0.283
  (those parameters were fitted on only 15 matching labels). On dblp-acm calibration barely helps: its errors in paper
  matching are misses (recall 64%), not a scale problem. On walmart-amazon phase 6's finding is reproduced (93.9% →
  97.4%), and this run's answers are bit-identical to those collected in phase 6.
- Kev-0.8B: raw, it is unusable for matching; it calls most pairs a match (recall 96–100%, precision 21–56%), for 66–86%
  accuracy. Calibration on 200 labels adds 12–30 points, to 91–98%, and cuts Brier from about 0.2 to 0.02–0.06. On
  dblp-acm it is on par with Jev after calibration (97.9% vs 99.0%), above calibrated Kev-4B's 92.2%. The other three
  stay behind: abt-buy 3.2 points lower in accuracy, amazon-google 0.038 lower in AUC, walmart-amazon at the bar on all
  three metrics.
- So for the small model calibration is not optional but a precondition for use; for both models, however, calibration
  cannot lift data with insufficient ranking to Jev's level, and F1 at 0.5 is not guaranteed to improve.
- Labels needed: on the datasets that reach Jev after calibration, 100 labels suffice (on par in 98–100% of 200 draws);
  with 25 labels 39–69%.
- The calibrated probabilities computed in SQL are identical to those computed in Python with the same parameters.
- Replaying phase 6's data (Kev-4B, already seen, not a new test): movie reviews 90.7% → 92.4%, ECE 0.097 → 0.016;
  restaurant reviews with one parameter set per predicate 82.9% → 91.6%, with one set shared by all 87.6%. Per-predicate
  parameters are clearly better.

**What it means for `kevdb`.**
- The default backend should still be Jev.
- A local model is worth it for long texts asked question after question, or when data must not leave the machine. Then
  start the server with `KEV_PREFIX_CACHE` set to what fits in memory; the factory 4 entries do nothing for table scans.
  For quality choose Kev-4B; for speed Kev-0.8B, which beats unparallelised Jev on long documents from the third
  question, provided the questions suit it and it is calibrated on human labels first.
- It is not a "semantic index" for arbitrary tables: a fixed 22–57 MB per row makes it a hot cache of a few hundred to
  two thousand rows. Kev-0.8B holds the whole 1,864-review table and then matches the Jev API's throughput; larger
  tables do not fit.
- The calibration table and macros work for any backend that returns probabilities. They fix the scale, not the ranking.

Cost: Jev's long-document reference $0.031, its rehearsal $0.003; local models are free (electricity and hardware not
counted).

### Phase 8: semantic joins, one request per pair or one request per left record?

A semantic join first uses plain SQL to find k candidates for each left record, then asks the model. Earlier
([experiments/table_match/](../table_match/README.md)) each candidate pair was one request. This phase tests three ways
to put a left record's k candidates into one request, each with bars fixed before collection. Criteria are in
[protocol/p8.txt](protocol/p8.txt).

| Form | State | Questions |
| --- | --- | --- |
| pairwise (baseline) | both records | one yes/no question |
| Q | the left record | k yes/no questions, each containing one candidate. The JEVDB paper calls this the star layout; it reports no quality, token or order tests for it |
| S | the left record and k candidates | k yes/no questions, each pointing at one candidate in the state |
| C | the left record | one multiple-choice question with the k candidates plus "none of them". At most one per left record, so only run on the nearly one-to-one abt-buy |

The data are the two raw tables of the table-matching experiment with their complete human mapping; candidates are the
top 10 found then by full-text search, with the same question: abt-buy (1,081 × 1,092 products, 1,097 true matches) and
dblp-scholar (2,616 × 64,263 publications, 5,347 true matches, a left record often has several). Jev runs the whole
tables; the local Kev-4B a fixed subset of 400 left records. Candidates appear in three orders: hash-shuffled (primary),
search rank, and reversed rank.

**Bars**: quality = packed F1 at most 1.0 point below pairwise; cost = input tokens and time on the subset both at most
50% of pairwise; order = pairwise agreement between the orders at least 99.0% and F1 within 1.0 point. A form holds only
if all three pass on every dataset it applies to.

**Result: on Jev, Q and C hold and S does not; on Kev-4B none holds.**

Jev, whole tables, end-to-end precision / recall / F1 (in brackets the 95% interval of the F1 difference from pairwise):

| Form | abt-buy | dblp-scholar | Input tokens (pairwise = 100%) | Time | Agreement across orders (lowest) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| pairwise | 90.0% / 91.4% / 90.7% | 88.5% / 93.0% / 90.7% | 100% | 100% | | |
| Q | 95.8% / 87.7% / 91.6% (−0.2 to +2.0) | 91.4% / 90.7% / 91.1% (−0.0 to +0.7) | 38% | 10–11% | 99.77% / 99.80% | holds |
| S | 96.1% / 89.3% / 92.6% (+0.8 to +3.0) | 93.9% / 87.8% / 90.8% (−0.5 to +0.7) | 38% | 10–11% | 99.60% / 97.90% | fails: varies with order on dblp-scholar |
| C | 98.5% / 90.0% / 94.0% (+2.0 to +4.7) | not applicable | 22% | 10% | 99.75% | holds |

- **A tenth of the requests and a tenth of the time.** Jev's latency per request does not depend on the number of
  candidates: 0.31–0.34 s with 1, 5, 10, 20 or 40. On the subset, pairwise takes 83 s for 3,954 requests; packed, 8–9 s
  for 400.
- **Tokens drop to 38% (Q) and 22% (C).** Whole tables: abt-buy pairwise $0.245 and 210 s, Q $0.094 and 24 s, C $0.054 and
  22 s; dblp-scholar pairwise $0.507 and 517 s, Q $0.195 and 55 s. For the same pairwise request Jev bills about 300 more
  tokens than Kev-4B (539 vs 241, 463 vs 160), a fixed cost per request that packing spreads out.
- **No loss in quality.** Q agrees with pairwise on 98.6–98.9% of judgements; the F1 difference is not separable from
  zero. Packed, precision rises and recall dips slightly.
- **Candidates in the questions (Q) do not depend on order; candidates in the state (S) do.** Q's three orders agree on
  99.8% pairwise, which is Jev's own repeat agreement; S only 97.9% on dblp-scholar, with F1 between 88.4% and 90.1%.
  Adding candidates behaves the same: going from 5 to 40 candidates leaves 99.75–100% of the top-5 judgements unchanged
  for Q and 99.25–100% for S.
- **Jev's multiple choice has no position bias.** With shuffled order it is right on 91.9% when the true match is in the
  first 3 positions and 95.3% when in the last 3; F1 is 94.7%, 95.2% and 95.2% across the three orders. 81% of choices
  have confidence ≥ 0.9 and are right 99.1% of the time, the rest only 64%, which allows routing.
- **Cascading still works after packing.** In every form 95–99% of wrong judgements fall in the least certain 30% of
  candidate pairs.
- Control: that day's fresh pairwise answers agree with the cached ones of 5 October on 99.97% and 99.80%.

Kev-4B, subset (400 left records), F1 (in brackets the 95% interval of the difference from its own pairwise F1):

| Form | abt-buy | dblp-scholar | Input tokens | Time | Agreement across orders (lowest) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| pairwise | 85.6% | 88.5% | 100% | 100% | | |
| Q | 91.4% (+3.4 to +8.3) | 90.0% (−0.1 to +3.2) | 62% / 73% | 53% / 59% | 100% / 100% | fails: only the cost bar |
| S | 86.9% (−0.9 to +3.7) | 83.2% (−7.0 to −3.4) | 62% / 74% | 47% / 55% | 98.34% / 93.90% | fails |
| C | 87.9% (−1.1 to +5.5) | not applicable | 29% | 23% | 96.03% | fails: varies strongly with order |

- **Q's judgements do not depend on order at all, and quality beats pairwise** (on abt-buy precision rises from 77.9% to
  90.5%), but it is only 1.7–1.9× faster: a local model's time follows its tokens, with no fixed per-request part to save.
- **Kev-4B's multiple choice favours later options.** Ordered by search rank (true matches mostly early) F1 is only
  73.5%, reversed 92.3%, shuffled 87.9%; with the true match in the first 3 positions it is right 69.4% of the time, in
  the last 3 94.3%. None of its confidences reaches 0.9.

**Post-hoc analysis** (added after seeing the results, [p8_posthoc.py](p8_posthoc.py)):
- How much of the multiple choice's gain comes from the "a left record matches at most one" constraint? Applied directly
  to pairwise probabilities (keep each left record's highest candidate if at least 0.5), Jev on abt-buy goes from 90.7%
  to 92.8% against the multiple choice's 94.0%, still 1.2 points higher (interval 0.3–2.1). So about 60% of the multiple
  choice's 3.4-point gain comes from the constraint, which works for any model that returns probabilities.
- The constraint helps Kev-4B even more: pairwise from 85.6% to 93.7%, S from 86.9% to 95.7%, above its own multiple
  choice (87.9%) and level with Jev's multiple choice on the same subset (94.7%). Phase 6 found that Kev loses on scale,
  not ranking; here the one-to-one structure sidesteps the scale problem without any labels.
- Averaging Kev-4B's multiple-choice probabilities over several orders before choosing: 90.5–90.6%, between the worst
  and best single order, at 2–3× the requests. The yes/no form with the constraint above is better.

**SQL.** The two forms that hold became SQL functions (`jev_match`, `jev_pick`, see the table above). The whole abt-buy
join is one DuckDB statement ([p8_demo.py](p8_demo.py)): replayed from this phase's cache it produces exactly the
collection script's matches; asked fresh it takes 26.9 s, 1,081 requests and $0.054, F1 94.1%, 7 pairs different from the
collection run, which is Jev's own variation between two answers.

**What it means for `kevdb`.** After blocking, do not ask a semantic join pair by pair on Jev: use `jev_match` in
general and `jev_pick` for nearly one-to-one joins. Put the candidates in the questions, not all into the state. On the
local Kev-4B use `kev_match` and, for one-to-one joins, take each left record's highest probability in SQL; do not use
`kev_pick`. All of this holds for record matching only: for another predicate, packing can change the answer (phase 9).

Cost: Jev collection $1.214, two rehearsals $0.021, the fresh SQL run $0.054; local models are free. The whole collection
took 1 h 56 min, of which Kev-4B took 1 h 43 min.

### Phase 9: does packing carry over to joins other than record matching?

Both datasets of phase 8 are record matching. This phase applies the same three bars to another predicate: "do the two
reviews have the same sentiment?" and "...opposite sentiment?" from the three pairwise queries of SemBench's movie
scenario (Q5–Q7). Criteria are in [protocol/p9.txt](protocol/p9.txt).

The data are the 128 reviews of the movie those queries use (66 labelled positive, 62 negative) and all 8,128 pairs, with
the truth derived from the two reviews' labels as in SemBench's reference answers. The questions are phase 1's verbatim.
Pairwise means phase 1's requests, all asked again that day; packed means phase 8's Q: only the first review in the
state, one yes/no question per partner, inside the question. Each review is paired with every later review, k per
request, with k = 10, 40 or 127 (127 means one request per review), in the same three orders. Jev runs both questions;
the local Kev-4B only "opposite", k = 10 and two orders. The bars are phase 8's, with quality as F1 over all pairs.

**Result: no k holds; phase 8's conclusion does not carry over to this predicate. Savings remain, but Jev's answer to
"opposite" changes.**

Jev, F1 over all 8,128 pairs (in brackets the 95% interval of the difference from pairwise, resampling reviews):

| Form | "same" | "opposite" | Input tokens | Time | Median latency per request |
| --- | --- | --- | --- | --- | --- |
| pairwise | 77.1% | 73.3% | 100% | 100% (168 s) | 0.30 s |
| Q, k = 10 | 78.2% (−0.9 to +3.1) | 65.3% (−12.1 to −4.2) | 33% | 11% | 0.31 s |
| Q, k = 40 | 78.1% (−0.9 to +2.9) | 65.5% (−12.0 to −4.0) | 27% | 4% | 0.32–0.36 s |
| Q, k = 127 | 78.2% (−0.9 to +3.1) | 65.3% (−12.2 to −4.3) | 26% | 2–3% | 0.36–0.46 s |
| reference: phase 1's rewrite, 128 requests | 79.1% | 79.2% | 1.3% | | |

- **Savings remain and grow with the block size.** Jev accepts 127 questions in one request; median latency rises only
  from 0.30 s to 0.36–0.46 s, and all 8,128 pairs are answered in about 4 seconds.
- **On "opposite", packing costs 8 points of F1.** Recall drops from 65.8% to 52.7%, and only 87.5% of judgements agree
  with pairwise, against 98.6–98.9% for record matching in phase 8. "Same" does not drop and agrees on 92.6%.
- **Block size and order do not matter.** The three values of k give the same F1; across orders 98.1–98.7% of
  judgements agree, the same level as asking one request twice (98.1–98.8%).
- **The order bar fails literally, but that shows no order effect.** The 99.0% bar came from Jev's repeat agreement on
  earlier tasks (99.45–99.8%); on these two questions Jev agrees with itself on only 98.1–98.8%, below the bar. The
  verdict rests on the quality bar.
- **Kev-4B drops more**: "opposite" F1 from 53.0% to 38.0% (−20.7 to −9.6), tokens 71%, time 65%, 99.98% agreement across
  orders.
- SemBench's own scoring code gives the same F1 for Q7: pairwise 0.733, packed 0.653–0.655.
- Control: that day's pairwise answers agree with phase 1's cache on 98.6%; Kev-4B's 672 re-asked pairs agree with phase 7
  on 100%.

**Post-hoc analysis** (added after seeing the results, with two extra sets of questions; [p9_posthoc.py](p9_posthoc.py)):
- **The answer changes because the second review moves from the state into the question, not because of packing.** A
  request with a single partner (k = 1) scores 65.5% F1, agrees with k = 10 on 98.7% and with pairwise on only 87.8%. At
  k = 1 this form saves nothing (tokens and time 101% of pairwise).
- **Direction of the shift**: with the second review in the question, Jev says "opposite" less often (30.8% of pairs vs
  39.9% pairwise and 50.3% in truth); "same" barely moves (48.0% vs 47.6%). Its answers also agree less with what the two
  reviews' own sentiments imply (77.0% vs 87.5% pairwise).

**What it means for `kevdb`.** `jev_match` always saves requests and tokens; "answers like pairwise asking" holds for
record matching only, and another predicate must be verified first. Verification needs no labels: ask a few hundred
pairs both ways and do not pack if agreement falls short of asking the same request twice. Predicates like this one,
which decompose into per-row judgements, should be rewritten (phase 1): 128 requests, F1 79%, more accurate than any
pairwise form.

Cost: Jev collection $1.004, the probe and two rehearsals $0.066, post-hoc questions $0.235; local models are free. The
whole collection took 40 min, of which Kev-4B took 31 min.

### Phase 10: does fine-tuning on public matching data bring Kev to Jev's zero-shot level?

Phases 6 and 7 found the largest gap between the open models and Jev on record matching, partly in the ranking, which
calibration cannot change, and Kev's training data contain no record pairs. This phase asks: after Kev has seen
human-labelled matching pairs from other sources, is it as good as Jev's zero-shot answers on a matching dataset it has
never seen? Criteria: [protocol/p10.txt](protocol/p10.txt), frozen before any real pair was used, with the execution
record at its end; runbook: [protocol/p10_plan.md](protocol/p10_plan.md).

Four Magellan datasets, leave one out: abt-buy, amazon-google and walmart-amazon (products) and dblp-acm (publications).
For each held-out dataset D, the model is fine-tuned on 2,000 human-labelled pairs from the training split of each of
the other three (6,000 pairs, plus 2,000 replayed records of Kev's own training set against forgetting) and tested on
D's full test split ("loo-D"); a fifth arm trains on all four datasets as an in-domain reference. Kev-4B and Kev-0.8B
continue from their released v1.0 checkpoints with the settings of their own last fine-tuning stage (LoRA, learning rate
2e-5, one epoch, one seed), on one Mac. Each checkpoint gets one temperature, fitted on 200 validation labels per training
dataset and never on D. The line (T) is phase 6's rule against Jev's zero-shot answers on D: accuracy at most 2 points
below, AUC at most 0.02 below, Brier at most 0.02 above. The guard (G): accuracy on movie reviews, restaurant aspects
and Kev's own development set at most 2 points below the released checkpoint. A model passes only if all four held-out
datasets meet T with G met. Only the benchmarks' human labels were used; no output of Jev was used for training,
calibration or thresholds.

**Result: neither model reaches Jev's zero-shot level on all four unseen datasets. Kev-4B does on two (abt-buy,
walmart-amazon), Kev-0.8B on one (walmart-amazon).**

Accuracy / AUC / Brier / F1 on the held-out dataset's full test split (released = the v1.0 checkpoint as shipped,
fine-tuned = loo-D with its temperature; "n.s." = the interval of the difference contains the line):

| Held out | Jev, zero-shot | Kev-4B released | Kev-4B fine-tuned | Kev-0.8B released | Kev-0.8B fine-tuned |
| --- | --- | --- | --- | --- | --- |
| abt-buy | 98.0% / 0.996 / 0.014 / 0.910 | 96.2% / 0.993 / 0.034 / 0.844 | **97.9% / 0.994 / 0.017 / 0.902: on par** | 70.3% / 0.974 / 0.202 / 0.418 | 94.4% / 0.984 / 0.045 / 0.665: below (accuracy, Brier) |
| amazon-google | 92.1% / 0.961 / 0.057 / 0.695 | 85.2% / 0.937 / 0.117 / 0.557 | 89.9% / 0.951 / 0.085 / 0.644: below (accuracy −2.2, n.s.; Brier) | 66.9% / 0.923 / 0.216 / 0.373 | 91.6% / 0.940 / 0.065 / 0.635: below (AUC −0.021, n.s.) |
| walmart-amazon | 97.9% / 0.987 / 0.019 / 0.884 | 93.9% / 0.981 / 0.054 / 0.733 | **97.9% / 0.987 / 0.017 / 0.882: on par** | 66.0% / 0.960 / 0.219 / 0.350 | **97.0% / 0.973 / 0.024 / 0.840: on par** |
| dblp-acm | 99.0% / 0.999 / 0.009 / 0.973 | 92.0% / 0.985 / 0.045 / 0.742 | 92.2% / 0.990 / 0.060 / 0.742: below (accuracy, Brier) | 85.7% / 0.995 / 0.150 / 0.716 | 97.7% / 0.993 / 0.018 / 0.940: meets T, fails G |

- **Fine-tuning helps on every product dataset, for both models.** F1 rises by 0.06–0.15 for Kev-4B and 0.25–0.49 for
  Kev-0.8B, every interval above zero. Most of the gain is fewer false matches: Kev-4B on walmart-amazon goes from 104
  to 16 (Jev 18).
- **A new question does not transfer for Kev-4B.** dblp-acm is asked the publication question, which no training record
  contains. Kev-4B's F1 there is unchanged (0.742 before and after; difference −0.001, −0.013 to +0.011), and it still
  misses most of the matches it missed before (167, released 159, Jev 1). Kev-0.8B, starting much lower, rises from
  0.716 to 0.940 and meets T, but fails the guard.
- **Forgetting is absent for Kev-4B and slight for Kev-0.8B.** Kev-4B passes the guard on every checkpoint (largest drop
  1.3 points, on Kev's own development set). Kev-0.8B loses 2.0–2.4 points on movie reviews in every fine-tuned
  checkpoint (walmart-amazon passes at exactly −2.0, by one judgement of 1,001), while restaurant aspects gain 1.9–3.7
  points and its own development set stays within 0.5.
- **The temperature matters little.** The fitted temperatures are 1.3–1.8 (slightly softer), lowering Brier by at most
  0.009 against the raw probabilities. A temperature fitted the same way on the released checkpoint stays far from the
  fine-tuned Brier on the product datasets (Kev-4B walmart-amazon 0.045 against 0.017); on dblp-acm it is better than
  Kev-4B's fine-tuned checkpoint (0.051 against 0.060).
- **With in-domain labels, Kev-4B's F1 is above Jev's on all four test sets** (the all arm: abt-buy 0.926 vs 0.910,
  amazon-google 0.749 vs 0.695, walmart-amazon 0.895 vs 0.884, dblp-acm 0.977 vs 0.973), and above Ditto's published F1
  on abt-buy and walmart-amazon (0.893, 0.868). Kev-0.8B's all arm is level with Jev (0.712–0.982). Not like for like:
  56–93% of these test pairs have both records in the training split, and Ditto trains one model per dataset on all its
  training pairs.
- **End to end on phase 8's raw tables** (pair form, 400 left records, sources the checkpoint never saw), Kev-4B's
  fine-tuned checkpoints reach Jev's F1: abt-buy 91.7% vs 91.2% (difference −1.3 to +2.6), dblp-scholar 91.0% vs 90.4%
  (−0.5 to +1.7), up from 85.6% and 88.5%. With "at most one match per left record" the released Kev-4B already reached
  93.7%, and fine-tuning adds nothing (92.3%). Kev-0.8B's abt-buy checkpoint becomes too cautious (precision 100%, recall
  50%).
- **Candidate order**: training on yes/no pairs leaves the single-choice form's dependence on order as it was (Kev-4B
  released 70–93% right across the three orders, fine-tuned 69–89%), as expected.
- **Against the expectations written before the run**: Kev-4B met T on exactly the two datasets expected and missed the
  two marked open; the expected improvement on all four folds held on the three product folds but not on dblp-acm.
  Kev-0.8B improved strongly and met T on two, within the expected "at most two", but the expectation that the guard
  holds everywhere was wrong for it.

**What it means for `kevdb`.** Public labels are enough to make Kev-4B a substitute for Jev on product matching data it
has not seen, as long as the question is one it was trained on, and without losing its other abilities. They are not
enough for a matching task asked with a new question, nor for the hardest product set, where even Jev's F1 is 0.695.
What the training fixes is the released model's main weakness, calling non-matching pairs a match. Kev-0.8B needs
in-domain labels or calibration.

Cost: none (local training and serving; Jev's column comes from the answers cached on 2026-10-05). Machine time,
including the runs that were stopped and excluding the pause: Kev-0.8B about 12 h, Kev-4B about 30 h (five training runs
of 5–6 h, evaluation 4–45 min per checkpoint), spread over five days in phases started by the owner.

### Phase 11: why did fine-tuning not help Kev-4B on dblp-acm?

In phase 10, Kev-4B fine-tuned on 6,000 product pairs did not improve at all on dblp-acm, a publication dataset asked
with a question no training record contains (F1 0.742 before and after), and it stayed too strict: 167 missed matches,
Jev 1. This phase asks which missing ingredient explains that. Criteria: [protocol/p11.txt](protocol/p11.txt), frozen
before any phase 11 checkpoint existed, with the execution record at its end.

Each arm keeps phase 10's loo-dblp-acm training data (2,000 pairs each of abt-buy, amazon-google and walmart-amazon,
plus 2,000 replayed records) and its settings, and changes one thing:

- **words**: the same 6,000 pairs, each asked with one of five wordings of the product question that mean the same.
  Does the model learn the wording rather than the task?
- **domains**: plus every training pair of three other Magellan domains, each asked with its own question: beer (268
  pairs), restaurants (567) and songs (321).
- **pubs**: plus 2,000 pairs of DBLP-GoogleScholar, asked with exactly dblp-acm's publication question (12 pairs that
  share a record with dblp-acm's test split were dropped first).

The control is phase 10's loo-dblp-acm checkpoint, not retrained. Lines on dblp-acm's full test split (2,473 pairs): P1,
F1 above the released checkpoint; P2, F1 above the control (95% paired-bootstrap interval of the difference above zero);
T, on par with Jev by phase 6's rule; G, phase 10's guard. The added data come from the Ditto repository at the commit
phase 10 used. Only human labels were used.

**Result: the wording is not the problem. Two other ingredients each bring Kev-4B to Jev's level on dblp-acm: another
publication dataset asked with the same question (F1 0.962), and, against the expectation written before the run, three
other kinds of record asked with their own questions (0.945). Jev's F1 is 0.973.**

| On dblp-acm | Accuracy / AUC / Brier / F1 | False / missed matches | F1 minus control (95% CI) | P1 | P2 | T | G |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev, zero-shot | 99.0% / 0.999 / 0.009 / 0.973 | 24 / 1 | | | | | |
| Kev-4B released | 92.0% / 0.985 / 0.045 / 0.742 | 39 / 159 | | | | | |
| control (phase 10) | 92.2% / 0.990 / 0.060 / 0.742 | 26 / 167 | | | | below | met |
| words | 92.3% / 0.991 / 0.056 / 0.745 | 23 / 167 | +0.003 (−0.003 to +0.009) | no | no | below | met |
| domains | 97.9% / 0.995 / 0.018 / 0.945 | 47 / 4 | **+0.204** (+0.166 to +0.239) | yes | yes | **on par** | met |
| pubs | 98.6% / 0.996 / 0.014 / 0.962 | 23 / 11 | **+0.221** (+0.185 to +0.255) | yes | yes | **on par** | met |

- **Rewording one question does not teach a new one.** words cannot be told apart from the control: the same 167
  missed matches, F1 +0.003.
- **The publication question with publication records almost closes the gap.** pubs misses 11 matches instead of 167;
  its accuracy is 0.4 points below Jev's, and its F1 0.010 below (−0.021 to −0.000).
- **So does variety without any publication data.** domains adds 1,156 pairs of beer, restaurants and songs. It misses
  only 4 matches, but calls more non-matching pairs a match (47; control 26, Jev 24) and meets T by a smaller margin
  (accuracy −1.1 points, Brier +0.009).
- **No arm forgets.** The largest guard drop is 1.1 points (words, on movie reviews); every set passes.
- **End to end on phase 8's dblp-scholar table** (words and domains only, because pubs trains on the same source): F1
  91.5% and 91.0%, against 91.0% for the control, 88.5% for the released checkpoint and 90.4% for Jev. That task, asked
  with phase 8's own question, was already solved in phase 10.
- **Against the expectations written before the run**: pubs (P1, P2 and T met, far fewer missed matches) and words (P2
  not met, F1 within 0.02 of the control) came out as expected, and G held everywhere. domains was expected not to meet
  P2 and did.

**What it means for `kevdb`.** Phase 10's failure was not about how the product question is worded. Trained on one
question only, Kev-4B stays as strict on a new question as it was released. Adding the target question with similar
records, or a few other questions with other kinds of records, removes that strictness; 1,156 extra pairs were enough.
When fine-tuning Kev for your own matching data, mix in labelled data of more than one kind, and if you can, a dataset
asked with the question you will use.

Cost: none. Machine time 18.6 h in one go: training 5.0–6.1 h per arm, evaluation 23–36 min.

### Phase 12: how many of your own labels does Kev-4B need?

Phase 10 measured what public labels from other datasets do. A user usually has some labels of their own data as well.
This phase asks how many Kev-4B needs to reach Jev's zero-shot level on a matching dataset it has not seen, measured on
records that no label contains. Criteria: [protocol/p12.txt](protocol/p12.txt), frozen before any phase 12 checkpoint
existed, with the execution record at its end.

Each dataset's records are split in two halves by a hash of the record's text.
- **Test pairs:** both records on the test side, taken from the test split and from the 600 validation pairs Jev was
  asked; 632–766 pairs per dataset, 59–152 matches.
- **Labels:** training-split pairs whose two records are both on the label side, in budgets of 100, 300 and 1,000. The
  budgets are nested, and each one's last 20% fits the temperature.
- **Training:** the author's recipe for your own data (`--init_from`, 2 epochs, learning rate 2e-5, effective batch 8,
  nothing replayed).
- **Starting points:** main runs start from phase 10's loo-D checkpoint, which has never seen D, so at n = 0 it is
  phase 10's model. Reference runs start from the released checkpoint, at n = 300 and 1,000.
- **Line n\*:** the smallest budget at which the run is on par with Jev by phase 6's rule (accuracy, AUC and Brier;
  F1 is not part of the rule) and passes phase 10's guard.

**Result: no own labels for abt-buy and walmart-amazon, 100 for dblp-acm, 300 for amazon-google.** At 300 labels
either starting point is on par with Jev on all four datasets, without forgetting.

F1 on the test pairs ("par" = on par with Jev by the rule, "below" = not):

| Dataset | Jev | n = 0 | 100 | 300 | 1,000 | released, n = 0 | released + 300 | released + 1,000 | n\* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abt-buy | 0.959 | 0.950 par | 0.868 par | 0.943 par | 0.967 par | 0.861 below | 0.934 par | 0.950 par | **0** |
| walmart-amazon | 0.882 | 0.889 par | 0.908 par | 0.868 par | 0.913 par | 0.709 below | 0.842 par | 0.900 par | **0** |
| dblp-acm | 0.974 | 0.751 below | 0.956 par | 0.984 par | 0.993 par | 0.764 below | 0.974 par | 0.990 par | **100** |
| amazon-google | 0.744 | 0.663 below | 0.348 below | 0.646 par | 0.701 par | 0.603 below | 0.576 par | 0.667 par | **300** |

- **Product data like the public labels need none.** On abt-buy and walmart-amazon phase 10's checkpoint is already on
  par on records it never saw, and every budget keeps it there.
- **A new question needs about 100.** On dblp-acm, 80 trained labels take F1 from 0.751 to 0.956; with 300 it is
  above Jev's (0.984 against 0.974).
- **The hardest product set needs 300, and its F1 stays below Jev's.** On amazon-google, 100 labels first made the
  model stricter: it missed 50 of 66 matches (F1 0.348). With 300 it is on par by the rule's accuracy, AUC and Brier,
  but its F1 is still 0.098 lower (0.646 against 0.744), and 0.042 lower at 1,000.
- **The public-label stage pays only when you have few labels of your own.** At n = 0 phase 10's checkpoint is far
  ahead of the released one on the product datasets (F1 +0.06 to +0.18). At 300 and 1,000 labels the two starting
  points cannot be told apart: the largest difference is +0.069 (−0.000 to +0.138, amazon-google at 300).
- **No forgetting without replay.** Every run passes the guard; the largest drop is 1.5 points.
- **The small temperature slice does not decide anything.** Fitted on 20–200 held-out labels, the temperature hit the
  grid's lower bound (0.25) at n = 100 on abt-buy and walmart-amazon. Scoring every run with its starting point's
  temperature instead changes no verdict.
- **Against the expectations written before the run**:
  - **Held:** n\* = 0 on abt-buy and walmart-amazon, 100 to 300 on dblp-acm, and the guard holding everywhere.
  - **Missed:** amazon-google was expected to need 1,000 or more and needed 300. Phase 10's starting point was expected
    to stay clearly ahead at 300 labels, and is not ahead on any dataset.

**What it means for `kevdb`.** With no labels of your own, start from a checkpoint fine-tuned on public matching data.
It is already on par with Jev on product data like its training data, though not on a new kind of question. With about
300 labels of your own (240 to train on, 60 to fit the temperature), the author's recipe reaches Jev's level on all four
datasets from either starting point, in about 15–20 minutes of training on this Mac.

Cost: none. Machine time 14.1 h in one go: training 5–7, 14–20 and 47–66 min at 100, 300 and 1,000 labels, evaluation
about 12 min per run.

### Phase 13: are phase 10's close verdicts stable when only the training seed changes?

Phase 10 trained each checkpoint once, with seed 1. Several of its verdicts sit within a hair of a line, so another seed
might move them across it. The seed changes the order of the training records, the LoRA dropout draws and which 2,000
records of Kev's own training set are replayed. Criteria: [protocol/p13.txt](protocol/p13.txt), frozen before any phase 13
checkpoint existed, with the execution record at its end.

**Which arms.** A rule fixed before the run picks the close ones: every quantity that decides the verdict lies within a
quarter of the line's width of it. That is 0.5 points for accuracy and the guard, 0.005 for AUC and Brier. Three of
Kev-0.8B's arms meet it: amazon-google, walmart-amazon and dblp-acm. Kev-4B's amazon-google does not (its Brier was 0.008
past the line), but it was the nearest Kev-4B verdict and is reported apart.

**How.** Each arm is trained again with seeds 2 and 3, everything else as in phase 10, and judged exactly as phase 10
judged it. Scored this way, seed 1 reproduces phase 10's published numbers.

**Result: three of the four verdicts hold for all three seeds; Kev-4B on amazon-google does not. Phase 10 judged it
below Jev, and seeds 2 and 3 are on par.**

| Arm | Seed 1 (phase 10) | Seed 2 | Seed 3 | Over the seeds |
| --- | --- | --- | --- | --- |
| Kev-0.8B, walmart-amazon | on par | on par | on par | stable, on par |
| Kev-0.8B, amazon-google | below: AUC −0.021, guard | below: AUC −0.021 | below: AUC −0.020 | stable, below |
| Kev-0.8B, dblp-acm | below: guard | below: accuracy −2.4 | below: accuracy −2.3 | stable, below |
| Kev-4B, amazon-google (outside the rule) | below: accuracy −2.2, Brier +0.028 | on par (accuracy −1.4, Brier +0.020) | on par (accuracy −1.1, Brier +0.019) | **not stable** |

- **Kev-4B's amazon-google verdict depends on the seed.** Seeds 2 and 3 are on par, but at the edge: Brier +0.0199
  and +0.0192 against a line of +0.02. Phase 10's "2 of 4" for Kev-4B would be "3 of 4" with either of them.
- **The spread across seeds is enough to flip a verdict at the line.** Accuracy moves by up to 1.1 points, Brier by
  up to 0.009 and F1 by up to 0.027.
- **Kev-0.8B's verdicts hold, though one for another reason.** On dblp-acm, seed 1 fails only the guard; seeds 2 and
  3 pass the guard but fall 2.3–2.4 points short on accuracy.
- **Kev-0.8B's forgetting on movie reviews did not recur.** In phase 10, all five Kev-0.8B checkpoints (seed 1) were
  2.0–2.4 points below the released checkpoint on movie reviews. All six checkpoints here are 1.7 points below to 0.3
  above, and pass the guard. That is a systematic difference, not seed-to-seed noise. It has two possible sources:
  - seed 1 itself: its record order and replay sample are shared by every phase 10 arm;
  - the padding launcher of phase 10's rule-8 fix, which seeds 2 and 3 went through and Kev-0.8B's seed-1 runs predate.

  This phase cannot separate them, so phase 10's "Kev-0.8B forgets slightly" is not a robust finding.
- **Against the expectations written before the run**: walmart-amazon was expected to flip and held on par. On
  amazon-google and dblp-acm the guard did flip as expected, but the verdict stayed below for another reason. Kev-4B
  was expected to stay below and did not. The F1 spread stayed within the expected 0.03; the accuracy spread slightly
  exceeded the expected 1 point (1.1 points on two arms).

**What it means for `kevdb`.** A single fine-tuning run settles a verdict only when it is clear of the line by more
than about a point of accuracy or 0.01 of Brier. When a fine-tuned checkpoint lands near your bar, train another seed
before deciding.

Cost: none. Machine time 21.1 h in one go: Kev-0.8B 1.6–1.7 h of training per run, Kev-4B 5.1 h, evaluation 4–18 min.

### Phase 14: is Kev-0.8B's drop on movie reviews a property of seed 1 or of training without padding?

In phase 13, Kev-0.8B's seeds 2 and 3 were 1.3 points better than phase 10's seed 1 on the movie-review guard (95%
interval +0.4 to +2.1). They also differed in a second way: they went through the padding launcher of phase 10's rule-8
fix, and Kev-0.8B's seed-1 runs predate it. Under one seed, the order of the training records and the 2,000 replayed
records are fixed, and padding leaves them unchanged. So phase 14 trains seed 1 again through the launcher, on phase
13's three arms. Criteria: [protocol/p14.txt](protocol/p14.txt), frozen before any phase 14 checkpoint existed.

The reading uses the mean over the three arms of the movie-review accuracy, padded seed 1 minus phase 10's seed 1, with
one 95% interval from resampling the review texts. If the interval contains zero, the drop belongs to seed 1 (A). If it
lies above zero, the drop came with training without padding (B).

**Result: A. Padded seed 1 cannot be told apart from phase 10's seed 1 (−0.0 points, −0.1 to +0.1), so the drop belongs
to seed 1 itself, as expected before the run.**

| Movie-review guard (points against the released checkpoint) | amazon-google | walmart-amazon | dblp-acm |
| --- | --- | --- | --- |
| Seed 1, phase 10 (no padding) | −2.4 | −2.0 | −2.3 |
| Seed 1, padded (phase 14) | −2.4 | −2.1 | −2.2 |
| Seeds 2 and 3, padded (phase 13) | −1.7 / −0.8 | −1.7 / −0.3 | −1.6 / +0.3 |

- **The slight forgetting is real, but it belongs to one training draw.** With seed 1 it reproduces, about 2.2 points
  below the released checkpoint; with seeds 2 and 3 it does not. Which records are replayed and in what order decides
  it. The share of sentiment records among the replayed ones does not: seed 3 replays fewer than seed 1 and passes.
- **Padding leaves the results unchanged.** The padded seed-1 checkpoints reproduce the unpadded ones: the same fitted
  temperatures, accuracy within 0.25 points, AUC within 0.002, Brier within 0.002 and F1 within 0.009. This also
  confirms that phase 10's rule-8 fix did not change any Kev-4B result.
- **One verdict sat on a single judgement.** Kev-0.8B's walmart-amazon passed the guard in phase 10 at exactly −2.0.
  Padded seed 1 lands at −2.1, so this arm turns from on par to below; the other two arms keep their verdicts.

**What it means for `kevdb`.** The Kev-0.8B checkpoint you get from one fine-tuning run may lose about 2 points on an
unrelated task, or may not. Check the tasks you care about after training, and train another seed if one drops.

Cost: none. Machine time 5.2 h in one go: training 1.6–1.7 h per run, evaluation 4–5 min.

## Limitations

- Two datasets and one LLM as upper model. The published systems' times and costs were measured in 2025 with Gemini 2.5
  Flash; only their quality is directly comparable.
- The movie data's truth is the critic's overall verdict, not the sentiment of the text itself; the restaurant questions
  are first drafts written from category names and clearly diverge from the labelling guidelines (for example "service"
  vs "staff"). No question was edited against the truth, so accuracy on both datasets is low, equally for every model.
- LIMIT queries look at 5–10 rows, so one row moves F1 by 0.1–0.2.
- LLM timings depend on concurrency and the network at the time and were measured once.
- Selectivity and ordering were not wired into DuckDB's optimizer; they were computed by replaying collected judgements.
- All datasets are public, so the models may have seen them in training.
- GoEmotions was downloaded but not used: after the pre-check the ordering experiment was not run by the rules, and a
  third dataset was no longer needed.
- Phase 6: one machine, three datasets, one version per model. The questions are the first drafts written for Jev and
  were not adapted to any model, which hurts Laya English. Clef-flash ran on PyTorch's Mac GPU backend, not the NVIDIA
  setup it is officially tested on, so its speed here says nothing about a GPU server; Kev ran on MLX. Throughput was
  measured one request at a time, without batch APIs. The 27B Clef and Kev do not run on a 64 GB machine and were not
  tested.
- Phase 7: one machine, two models. The memory budget was measured in that machine's state at the time (other programs
  held about 20 GB) and must be re-measured elsewhere. The cache lives in the server's memory and is lost on restart; it
  is not stored in DuckDB. The cache experiment and Jev's long-document reference ran at concurrency 1; the Jev API
  supports concurrency, and concurrent throughput was not measured. The long documents are joined reviews, the questions
  new first drafts and fairly easy, and the quality reading covers only 40 documents; Kev-0.8B's closeness to Jev on these
  three questions says nothing about others. Kev-0.8B's whole-table cache took 43.5 GB and its three main datasets varied
  by 20–30% between passes; speed-ups are medians. The bars were written after the probe on made-up rows, when "long
  documents at least 3× faster" was already likely to pass. The SemBench references (Jev, LLM) were measured on 7 October.
  The validation and test splits of the matching data share a few identical pairs (label pool and test set overlap by 8,
  30 and 9 pairs); fitting uses only the first 200 pool pairs, so the effect is negligible. Electricity and hardware are
  not counted.
- Phase 8: both datasets are record matching and public for years; one version per model; multiple choice on one dataset
  only; candidates fixed at the top 10 of full-text search. Kev-4B ran only 400 left records, so its F1 intervals span
  ±2–3 points. Times measured once on one Mac; Jev's include the network. No LLM arm (the OpenRouter balance was used up),
  so generative models were not tested with these forms. "Jev bills about 300 more tokens per request" subtracts the two
  backends' own counts, whose tokenizers may differ. The 50% cost bar was set before measuring; Kev-4B's Q missed it at
  53% and 59%. The "at most one" results are post hoc.
- Phase 10: each arm was trained once with one seed and 2,000 pairs per dataset. Three of the four datasets are
  products, so the reading on a new question rests on one fold (dblp-acm). The benchmarks have been public for years and
  may be in the base models' training data, which affects released and fine-tuned checkpoints alike. Training ran in full
  precision on Apple's GPU, not with the released recipe's bf16 on NVIDIA GPUs. The guard covers three sets only. The all
  arm's test splits share most of their records with its training data. Kev-0.8B was trained without and Kev-4B with the
  shape padding of the rule-8 fix below, so the two models also differ in their random dropout draws. The recorded
  training time of Kev-4B's loo-abt-buy includes a 37.6-hour pause.
- Phase 11: one seed per arm and one held-out dataset. domains adds new kinds of record and new questions at once, so it
  cannot say which of the two helps. domains and pubs also add 1,156 and 2,000 training pairs, and no arm with as many
  extra product pairs was run. DBLP-GoogleScholar shares its DBLP side with dblp-acm: pairs with a record that occurs
  verbatim in dblp-acm's test split were dropped, but the same paper can appear written differently. The three new
  questions are first drafts.
- Phase 12: one seed per run; the test sets are small (59–152 matches), so F1 intervals span ±0.04–0.10, and n\* is
  only resolved to the budgets 0, 100, 300 and 1,000. The released starting point was run at 300 and 1,000 only. The split
  keeps records written identically on one side, but the same product written differently can fall on both. The test
  pairs include only the 600 validation pairs Jev was asked, because answering the rest would have needed paid requests.
- Phase 13: two more seeds per arm only, and four arms. Kev-0.8B's seeds 2 and 3 also went through the padding launcher,
  which its seed-1 runs predate, so a difference between them may come from either. Phase 10's five checkpoints share
  seed 1 and are therefore not independent draws.
- Phase 14: three arms and one model. It separates seed 1 from the padding, not the replay sample from the record order,
  which one seed fixes together.
- Phase 9: 128 reviews of one movie and one pair of opposite questions; the truth is the critics' overall verdict, not the
  text's sentiment, so no form reaches a high F1. Pairs are not independent, so intervals resample reviews. Kev-4B ran one
  question and one k. Times measured once. Packing with the second review kept in the state (phase 8's S) was not tested,
  so whether some packed form preserves this question's answers is unknown.

## Pre-registered criteria and deviations

The primary analysis, success criteria and stopping rules of each phase were written before the run and are kept in
[protocol/](protocol/) (with checksums of the scripts at the time). Deviations and post-hoc additions:

- In the qualification test, 1–3% of the answers of routes with reasoning on were cut to empty by the token limit; the
  limit was raised and those rows re-asked, with the selection rule unchanged.
- Phase 1 had a metering error: requests that failed twice in a row were not counted as failures, so 35 rows across 6 of
  the LLM and cascade results were computed without an answer. A replay check (`p1_movie.py verify`) was added and the rows
  re-asked; mean quality is unchanged to three decimals, and Jev's results are unaffected.
- Phase 3's stratified sample initially drew one label more or less than n; it was fixed to exactly n with an independent
  random stream, moving the primary simulation from 2.61 to 2.53 (theory 2.62).
- Added afterwards: phase 1's `Q*p` variants, the packing comparison (question D) and the cascade replay with the packed
  LLM as upper model. Question D's pre-written reading was "packed accuracy within 1 point of one by one"; the LLMs came
  out 2–4 points higher, which fails the literal reading but in the favourable direction.
- Phase 6 ran without deviation: in one go, no interruption, no failed requests. Changes to the subjects and analysis
  before the start are dated in protocol/p6.txt: the hosted Clef was not run (paid), Clef-flash was run locally instead,
  that day's Jev was added as a control group, and Laya multilingual was added the day before the start. After the
  experiment, Laya was dropped from the candidates (results kept; scripts no longer install or run it by default) and a
  post-hoc breakdown was added (`p6_breakdown.py`).
- Phase 7: at the start the owner decided to use as much memory as possible, to add Kev-0.8B and to add Jev's
  long-document reference; all were revised into protocol/p7.txt with checksums before collection (the cache budget
  changed from a fixed 24 GB to 90% of each model's probed limit, so the short-text rows went from 300 to 580). Four
  deviations during execution: the first start failed at once without sending a request (a log statement failed under
  another locale) and was restarted after the fix; midway the owner postponed Kev-0.8B, so Kev-4B's analysis preceded
  Kev-0.8B's collection instead of following all collection; the last line of the analysis script misstated the cost and
  the report was reissued with all other numbers unchanged; "from which question on it beats Jev" is post-hoc
  arithmetic. Both models' collections ran in one go without interruption or failed requests: Kev-4B on 12 October
  (4 h 8 min), Kev-0.8B on the owner's instruction on 13 October (1 h 54 min), with the settings as frozen.
- Phase 8 ran without deviation: in one go, no interruption, no failed requests. The background of protocol/p8.txt
  originally said JEVDB asks pairwise, which is wrong: its star layout puts candidates into one request as separate
  questions, which is Q here. This is corrected at the end of that file; no bar or verdict depends on it. Added
  afterwards: the `jev_match` function and its replay on both datasets, and the four analyses of `p8_posthoc.py`.
- Phase 9 ran without deviation: in one go, no interruption, no failed requests. The 99.0% order bar is higher than
  Jev's own repeat agreement on these two questions (98.1–98.8%), which only the post-hoc questions measured; this is
  recorded at the end of protocol/p9.txt, and the verdict rests on the quality bar. Added afterwards: the two extra sets
  of questions and the four analyses of `p9_posthoc.py`.
- Phase 10: on Apple's GPU the trainer's memory grew with every new input length (Kev-0.8B's 8,000-record runs peaked
  at 38.9–40.5 GiB; Kev-4B passed the run's 41 GiB guard after 2,160 records). Under rule 8 every later training run (all
  of Kev-4B) went through [p10_kev_train.py](p10_kev_train.py), which applies Kev's own eval-mode padding to multiples of
  64 tokens during training as well: memory then stayed at 28–32 GiB, and with dropout switched off the trained weights
  change only at rounding level (2.2e-4 of the update); with dropout on, only the random dropout draw changes. The memory
  probes behind this are [p10_memprobe.sh](p10_memprobe.sh). Interruptions, none of which changed a training record, a
  request or an answer: the owner stopped phase 1 during Kev-0.8B's all arm, which was retrained from the start (rule 7);
  the memory guard stopped Kev-4B's first loo-abt-buy run, retrained after the fix (rule 7); the owner paused Kev-4B's
  loo-abt-buy for 37.6 hours by freezing its processes and resumed them, so the same computation continued. Every arm of
  both models was collected with no unanswered request and no failed server. All of it is recorded in protocol/p10.txt.
- Phase 11 ran without deviation: in one go (18.6 h), no interruption, no unanswered request, no failed server. The
  domains result contradicts the expectation written before the run and is reported as measured.
- Phase 12 ran without deviation: in one go (14.1 h), no interruption, no unanswered request, no failed server. The
  owner's first "start" came while phase 11 was still running. A start queued for after phase 11 was stopped on the
  owner's word before it had started anything, and the rehearsal and run followed the owner's next instruction.
- Phase 13 ran without deviation: in one go (21.1 h), no interruption, no unanswered request, no failed server. Before
  the freeze, the layout of the analysis' reading lines was fixed after the rehearsal; no number or rule changed.
- Phase 14 ran without deviation: in one go (5.2 h), no interruption, no unanswered request, no failed server. Before
  its rehearsal there was a false start. The training script, copied from phase 13's, still tested phase 13's rehearsal
  variable, so the first rehearsal run loaded phase 10's real training file. It was stopped 35 s in, before any training
  step, with nothing saved or asked. The script was fixed, a test now checks every phase's rehearsal variable, and the
  rehearsal was repeated from empty folders. All of it is recorded in protocol/p14.txt.
- The follow-up's size was set by the remaining balance. OpenRouter reserves credit per in-flight request based on its
  output limit and rejects concurrent requests when the balance is low, so collection took three runs (lower output
  limit, less concurrency, waiting after rejections); no answer was analysed between them.

## Reproducing

All commands run from the repository root.

```bash
.venv/bin/python experiments/database/p0_qualify.py run && .venv/bin/python experiments/database/p0_qualify.py report
.venv/bin/python experiments/database/p0_speed.py jev luna.low
.venv/bin/python experiments/database/p1_movie.py run jev --runs 3
.venv/bin/python experiments/database/p1_movie.py run llm --llm luna.low --cache out/database/p1/llm-cache.sqlite
.venv/bin/python experiments/database/p1_movie.py run cascade --llm luna.low
.venv/bin/python experiments/database/p1_movie.py verify --llm luna.low && .venv/bin/python experiments/database/p1_movie.py report
for s in jev luna.low v41flash.off qwen38flash.off v4pro.off jev.repeat; do .venv/bin/python experiments/database/p2_collect.py run $s; done
.venv/bin/python experiments/database/p2_collect.py merge
.venv/bin/python experiments/database/p3_aggregates.py
.venv/bin/python experiments/database/p4_selectivity.py
for s in jev jev.packed luna.low luna.low.packed v41flash.off v41flash.off.packed; do .venv/bin/python experiments/database/p4_collect.py run $s; done
.venv/bin/python experiments/database/p4_collect.py merge
.venv/bin/python experiments/database/p4_mams.py && .venv/bin/python experiments/database/p4_ordering.py && .venv/bin/python experiments/database/p4_packing.py && .venv/bin/python experiments/database/p4_cascade.py
.venv/bin/python experiments/database/p4_demo.py
.venv/bin/python experiments/database/p5_strong.py collect && .venv/bin/python experiments/database/p5_strong.py report   # collect spends the OpenRouter balance down to about $0.15
./experiments/database/report/build.sh   # redraws the figures and builds the PDF report

# Phase 6: open compatible models. Only run on Apple Silicon Macs; .env needs TYPESAFE_API_KEY
./scripts/open_models.sh setup                     # environments and weights for Kev and Clef-flash, about 49 GB, all inside open_models/
./experiments/database/p6_run.sh --rehearsal       # optional: rehearse the whole run on a few dozen made-up rows
./experiments/database/p6_run.sh                   # Jev control group plus Kev-4B, Kev-9B and Clef-flash, about 7.5 h; control about $0.43, local models free
.venv/bin/python experiments/database/p6_breakdown.py   # post-hoc breakdown
# for the post-hoc precision check see the top of experiments/database/p6_precision.py (start a full-precision server first)
# the dropped Laya is not installed or run by default. To reproduce its rows:
./scripts/open_models.sh setup laya && ./experiments/database/p6_run.sh laya laya-multilingual

# Phase 7: local models inside DuckDB. Uses Kev-4B from phase 6; .env needs TYPESAFE_API_KEY (only for Jev's long-document reference)
# capacity probe, to redo on every machine: start the server in another terminal as below, then copy the printed budget into CAPACITY in experiments/database/p7_data.py
KEV_PREFIX_CACHE=100000 KEV_PREFIX_MAX_TOKENS=4000000 ./scripts/open_models.sh serve kev-4b
P7_MODEL=kev-4b .venv/bin/python experiments/database/p7_capacity.py
./experiments/database/p7_run.sh --rehearsal               # optional: rehearse on made-up rows, about 25 min
P7_NO_REPORT=1 ./experiments/database/p7_run.sh            # Kev-4B's four steps, about 4.1 h, free; peak memory about 43 GB
.venv/bin/python experiments/database/p7_jev_long.py run   # Jev's reference on 40 long documents, about $0.03
.venv/bin/python experiments/database/p7_report.py | tee out/database/p7/kev-4b/report.txt
# Kev-0.8B: the same four steps, about 1.9 h, peak memory about 44 GB; probe capacity as above with kev-0.8b
./scripts/open_models.sh setup kev-0.8b                    # downloads 1.84 GB
P7_NO_REPORT=1 ./experiments/database/p7_run.sh --model kev-0.8b && P7_MODEL=kev-0.8b .venv/bin/python experiments/database/p7_report.py | tee out/database/p7/kev-0.8b/report.txt
# both models plus Jev's reference in one go: ./experiments/database/p7_all.sh

# Phase 8: semantic-join forms. Uses the data and candidates of experiments/table_match/ (run its fetch.sh and pipeline.py block first) and Kev-4B from phase 6
./experiments/database/p8_run.sh --rehearsal               # optional: rehearse on made-up rows, about 4 min, about $0.02
./experiments/database/p8_run.sh                           # Jev about 13 min and $1.21, then Kev-4B about 1.7 h; results in out/database/p8/report.txt
.venv/bin/python experiments/database/p8_posthoc.py | tee out/database/p8/posthoc.txt   # post-hoc analysis, replay only
.venv/bin/python experiments/database/p8_demo.py replay    # the same join in DuckDB SQL, replayed from the cache, free
.venv/bin/python experiments/database/p8_demo.py fresh     # the abt-buy multiple-choice join asked afresh, about $0.05

# Phase 9: packing on SemBench's sentiment pairs. Uses phase 1's data and Kev-4B from phase 6; its pairwise reference comes from phase 7's cache
./experiments/database/p9_run.sh --rehearsal               # optional: rehearse on made-up reviews, about 2 min, about $0.03
./experiments/database/p9_run.sh                           # Jev about 9 min and $1.00, then Kev-4B about 31 min; results in out/database/p9/report.txt
.venv/bin/python experiments/database/p9_posthoc.py collect   # the two post-hoc sets of questions, about $0.23
.venv/bin/python experiments/database/p9_posthoc.py report | tee out/database/p9/posthoc.txt

# Phase 10: fine-tuning Kev on public matching data. Uses Kev-4B and Kev-0.8B from phases 6 and 7; no API key, free
./experiments/database/p10_fetch.sh                               # the four training splits and Kev's replay data, about 28 MB
./experiments/database/p10_run.sh --model kev-0.8b --rehearsal    # optional: the whole run on made-up rows, about 15 min
./experiments/database/p10_run.sh --model kev-0.8b                # about 10 h on an M3 Max with 64 GB; out/database/p10/kev-0.8b/report.txt
./experiments/database/p10_run.sh --model kev-4b                  # about 29 h; split into phases with --arms (protocol/p10_plan.md, section 6a)
# since the rule-8 fix every training run is padded (p10_train.sh calls p10_kev_train.py); the recorded Kev-0.8B run was not

# Phase 11: why fine-tuning did not help Kev-4B on dblp-acm. Uses Kev-4B and phase 10's data and results; no API key, free
./experiments/database/p11_fetch.sh                               # eight files of four more Magellan datasets, about 8.4 MB
./experiments/database/p11_run.sh --rehearsal                     # optional: the whole run on made-up rows, about 20 min
./experiments/database/p11_run.sh                                 # about 18.6 h on an M3 Max with 64 GB; out/database/p11/kev-4b/report.txt

# Phase 12: how many of your own labels Kev-4B needs. Uses Kev-4B, phase 10's data and its four loo checkpoints; no API key, free
./experiments/database/p12_run.sh --rehearsal                     # optional: the whole run on made-up rows, about 1 h
./experiments/database/p12_run.sh                                 # about 14 h on an M3 Max with 64 GB; out/database/p12/kev-4b/report.txt

# Phase 13: seeds 2 and 3 for phase 10's close arms. Uses phase 10's data and results; no API key, free
./experiments/database/p13_run.sh --rehearsal                     # optional: the whole run on made-up rows, about 20 min
./experiments/database/p13_run.sh                                 # about 21 h on an M3 Max with 64 GB; out/database/p13/report.txt

# Phase 14: Kev-0.8B's seed 1 again, through the padding launcher. Uses phase 10's data and the results of phases 10 and 13; no API key, free
./experiments/database/p14_run.sh --rehearsal                     # optional: the whole run on made-up rows, about 7 min
./experiments/database/p14_run.sh                                 # about 5.2 h on an M3 Max with 64 GB; out/database/p14/report.txt
```

The actual cost of running everything: Jev about $3.4 (plus $0.45 for phase 6, $0.03 for phase 7, $1.29 for phase 8,
$1.31 for phase 9; phases 10–14 used no paid service), OpenRouter $19.97 (the whole $20 top-up). Results go to `out/database/` (not committed), with a
`report.txt` or similarly named text file per phase.

## Report

The report covering phases 0–14 (part I: Jev on DuckDB, phases 0–5; part II: open decision models and fine-tuning,
phases 6–14) is [report/Jev-databases-report.pdf](report/Jev-databases-report.pdf), built from `report/report.en.md` with
`./experiments/database/report/build.sh` (needs pandoc and TeX Live).
