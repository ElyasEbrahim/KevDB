# Experiment: letting Jev choose among candidate query plans

Question: is it feasible to put Jev into database query optimization, and is it better than an ordinary LLM?

Method: PostgreSQL 17 with the JOB benchmark (IMDb data, 113 multi-join queries). Five sets of optimizer switches make the
database produce several different plans per query, and each plan's measured runtime is the ground truth. The model then
judges, for each "candidate plan vs default plan", whether the candidate will be faster; plans are chosen by its
judgements and the whole workload's total runtime is measured.

## Conclusion (ground truth collected 2026-10-02, Jev rerun through the official API on 2026-10-05)

It does not work zero-shot. Neither Jev nor the LLM beats the default optimizer. Jev used the official API, DeepSeek the
third-party platform OpenCode.

| Method | AUC | Queries switched | Total time | vs default | Clearly faster / slower | Latency per judgement | Cost of 256 questions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Default optimizer | | 0 | 90.7 s | 1.00× | | | |
| Hindsight optimum (ceiling) | | | 73.7 s | 0.81× | | | |
| Random candidate | | | 140.3 s | 1.55× | | | |
| Jev, plans with full estimates | 0.52 | 0 | 90.7 s | 1.00× | 0 / 0 | 0.42 s | $0.028 |
| Jev, row counts bucketed, costs removed | 0.44 | 7 | 89.3 s | 0.99× | 2 / 5 | 0.46 s | $0.021 |
| Jev, no numbers at all | 0.33 | 67 | 170.3 s | 1.88× | 3 / 61 | 0.35 s | $0.017 |
| DeepSeek V4.1 Flash, reasoning off | 0.50 | 0 | 90.7 s | 1.00× | 0 / 0 | 1.08 s | $0.19 |
| DeepSeek V4.1 Flash, low reasoning | 0.62 | 27 | 98.2 s | 1.08× | 8 / 14 | 4.72 s | $0.64 |
| Logistic regression (plan features, cross-validated by template) | 0.56 | 4 | 90.8 s | 1.00× | 0 / 3 | | |

An AUC of 0.5 is chance. The threshold is 0.5; at higher thresholds both Jev and the logistic regression fall back to the
default plan everywhere.

- The default optimizer is already good: for 82 of the 113 queries the default plan is the fastest, and the room to win is
  only 19%.
- A wrong choice costs far more than a right one gains: random choice is 55% slower, so without confidence a judgement must
  fall back to the default.
- Shown the estimates, Jev always says "not faster", which does nothing but does no harm; without the numbers its
  judgements are worse than chance.
- The LLM with reasoning has a little signal (right on 12 of the 38 times it says "faster", against a base rate of 13%),
  but the net effect is 8% slower.
- The judgements themselves take longer than they could save: the whole workload can save at most 17 seconds, while the
  256 Jev calls add up to about 110 seconds of latency. Such methods can only pay off when single queries run for minutes.

Limitations: one benchmark, one way of asking, five switch sets (of which "merge join off only" barely changes plans);
timings in Docker's VM on a Mac, median of 3 runs; no few-shot attempt with this database's own execution history. GLM 5.3
Flash with default reasoning needed one to two minutes per question and could not finish in time, so it is not included.

## Reproducing

```bash
./experiments/query_plan/fetch.sh                      # IMDb data (1.26 GB) and the JOB queries
./experiments/query_plan/load.sh                       # start the database (127.0.0.1:5434), create tables, load, index
.venv/bin/pip install -e ".[query_plan]"
.venv/bin/python experiments/query_plan/collect.py     # ground truth, about 30 minutes, no model calls → out/query_plan/truth.jsonl
.venv/bin/python experiments/query_plan/headroom.py    # room to win
.venv/bin/python experiments/query_plan/decide.py --rep full                               # Jev
.venv/bin/python experiments/query_plan/decide.py --rep full --llm deepseek-v4.1-flash     # LLM baseline
.venv/bin/python experiments/query_plan/learned.py     # logistic-regression reference
.venv/bin/python experiments/query_plan/evaluate.py    # evaluation
```

To remove the database completely (including its volume): `docker compose -p jev-query-plan down -v`.

## Additional exploration: semantic correlation and joint statistics (2026-10-06, no significant result, not in the report)

Idea: optimizers misestimate row counts because they assume columns are independent. Jev sees only table names, column
names and sample values and judges whether "knowing column A nearly determines column B"; joint statistics
(`CREATE STATISTICS`) are then built for the column pairs it flags, to see whether row estimates improve. The code is
`experiments/query_plan/corr.py`.

The truth is Goodman–Kruskal's λ(B|A), computed on a sample of at most 1 million rows per table. The 15 tables have 328
directed column pairs; 292 remain after removing those where A is near-unique, 27 of them strongly dependent (λ ≥ 0.5).

Ability: how well the judgements agree with the truth

| Source | AUC (λ ≥ 0.5) | Rank correlation |
| --- | --- | --- |
| Reference: existing distinct counts only (A many, B few) | 0.86 | 0.44 |
| Jev, with column names and sample values | 0.74 | 0.45 |
| Jev, column names only | 0.70 | 0.45 |
| DeepSeek V4.1 Flash, with column names and sample values | 0.57 | 0.14 |

Effect: the optimizer's row-estimate error factor on 181 counting queries of the form "A = x and B = y"

| Joint statistics built on | Count | Median error | p90 error | Share within 2× |
| --- | --- | --- | --- | --- |
| none | 0 | 3.07 | 222 | 42% |
| pairs Jev flags as dependent | 37 | 2.00 | 137 | 50% |
| the same number at random (3 draws) | 37 | 1.94–2.14 | 175–219 | 49–50% |
| the same number of truly most dependent pairs | 37 | 1.58 | 110 | 59% |
| all column pairs | 164 | 1.13 | 51 | 79% |

Conclusion:
- Jev has some signal and beats the small LLM, but not a simple rule that needs no model; statistics built on its picks
  do about as well as random picks.
- At this database's size, building statistics on every column pair is simplest and most accurate; `ANALYZE` only grows
  from 1.4 s to 10 s, so there is nothing to choose.
- Even with all of them built, the total runtime of the 113 JOB queries does not change (92.6 s vs 90.7 s). JOB's
  estimation errors come mainly from joins across tables, which single-table joint statistics cannot reach.

After the experiment the statistics and sample tables were dropped and every table was analysed again.
