# Experiment: end-to-end matching of two raw tables

Question: starting from two raw tables that have never been paired, and running the whole pipeline of blocking → model
judgement → merging, what do Jev and an LLM cost, how long do they take, and how good is the final result?

Unlike [entity_match/](../entity_match/README.md), which uses pairs prepared by others and only tests whether
judgements are right, this experiment starts from the raw tables and evaluates against the complete human mapping, so
true matches lost during blocking count against recall.

Calls: Jev always through the official API (`jev-1.13.0`, concurrency 16). The DeepSeek baseline was only available
through the third-party platform OpenCode (concurrency 8).

## Pipeline

1. **Blocking (free)**: SQLite full-text search finds, for each left record, the k most textually similar right records.
2. **Judgement**: one yes/no question per candidate pair, "are these the same entity?". The question is the first draft
   from `entity_match/`, not adapted to these datasets.
3. **Merge**: probabilities at or above the threshold count as matches; probabilities between 0.1 and 0.9 can go to a
   human or a stronger model.

## Results (2026-10-05)

### Abt-Buy: two retailers' product catalogues

1,081 × 1,092 records = 1.18 million possible pairs, 1,097 true matches.

The same candidates (top 5 per record, 5,397 pairs; blocking keeps 96.1% of the true matches):

| Scheme | Precision | Recall | F1 | Model cost | Model time | Pairs for humans |
| --- | --- | --- | --- | --- | --- | --- |
| Jev only | 90.8% | 89.7% | 90.2% | $0.12 | 1.8 min | 0 |
| DeepSeek V4.1 Flash only (reasoning off, via OpenCode) | 88.9% | 93.1% | 91.0% | $0.64 | 10.9 min | 0 |
| Jev first, the unsure 12% to DeepSeek | 90.2% | 93.1% | 91.6% | $0.20 | about 3.1 min | 0 |
| Jev first, the unsure 12% to a human | 100% | 95.2% | 97.5% | $0.12 | 1.8 min | 637 |

With a flagship LLM added (Opus 5.5 via OpenCode, default adaptive thinking, `experiments/table_match/strong.py`) and
confidence intervals (`experiments/table_match/stats.py`, 2,000 bootstrap resamples over Abt records):

| Scheme | F1 (95% interval) | Model cost |
| --- | --- | --- |
| Jev only | 90.2% (88.4–92.0) | $0.12 |
| DeepSeek V4.1 Flash only | 91.0% (89.4–92.5) | $0.64 |
| Opus 5.5 only (stratified estimate) | 93.7% (92.1–95.3) | about $15.4 |
| Jev first, unsure pairs to DeepSeek | 91.6% (90.1–93.1) | $0.20 |
| Jev first, unsure pairs to Opus (measured) | 93.7% (92.3–95.1) | $2.92 |

- F1 differences between schemes: Jev − DeepSeek = −0.7 (−1.8 to +0.3, not significant); Jev+Opus − Jev = +3.5 (+2.6 to
  +4.5); Jev+Opus − DeepSeek = +2.8 (+2.0 to +3.6).
- "Opus only" is a stratified estimate: all 637 pairs Jev is unsure about were measured, plus 250 of the 4,760 pairs Jev is
  sure about (Opus agreed with Jev on every one and all were right), scaled up proportionally.
- Accuracy on the 637 pairs Jev is unsure about: Jev 74.9%, DeepSeek 79.0%, Opus 86.7%.
- Opus unit prices: $0.0044 per unsure pair (110 output tokens on average, thinking included), about $0.0026 per confident
  pair. Opus for everything costs 125× Jev alone; the two-tier scheme costs 19% of Opus for everything.
- Opus's thinking cannot be turned off (`thinking.type=disabled` is rejected) and counts against `max_tokens`; with 300,
  some answers ran out of room after thinking, so the limit is 2,000.
- Jev is confident (probability ≤ 0.1 or ≥ 0.9) on 88.2% of pairs, with 99.8% accuracy there; the unsure 11.8% are 75%
  accurate. All 642 pairs with probability ≥ 0.9 are true matches.
- In the two-tier scheme, DeepSeek's share of the time is scaled from its measured rate.
- With the top 10 candidates per record (10,747 pairs, 98.6% of true matches kept), Jev reaches F1 90.7% for $0.25 in
  3.5 minutes; handing the 707 unsure pairs to a human gives F1 98.7%.

### DBLP-Scholar: two bibliographic databases

2,616 × 64,263 records = 168 million possible pairs, 5,347 true matches. Top 10 candidates per record, 26,160 pairs;
blocking keeps 97.2%.

| Scheme | Precision | Recall | F1 | Model cost | Time | Pairs for humans |
| --- | --- | --- | --- | --- | --- | --- |
| Jev only | 88.5% | 93.0% | 90.7% | $0.51 | 8.6 min | 0 |
| Jev first, the unsure 17% to a human | 98.2% | 96.8% | 97.5% | $0.51 | 8.6 min | 4,310 |

No LLM baseline was run on this dataset. At the unit price measured on Abt-Buy ($0.119 per 1,000 pairs), the same 26,160
pairs would cost about $3.1 with DeepSeek; this is an estimate, not a measurement.

## Effect of the call route on speed

`experiments/table_match/route_speed.py` sends the same requests to the official API and to OpenCode, with no retries
and no cache.

| Model and route | Median latency | 3,000 requests in a row (concurrency 16) | Abt-Buy, 5,397 pairs |
| --- | --- | --- | --- |
| Jev, official API | 0.31 s | all succeed, 33 per second | 1.8 min (concurrency 16) |
| Jev via OpenCode | 0.59 s | 1,000 succeed, the other 2,000 rejected (429) | 10.5 min (concurrency 8) |
| DeepSeek V4.1 Flash via OpenCode | 0.82 s | not tested; a short test of 400 requests gave 9–19 per second | 10.9 min (concurrency 8) |

- The rate limit comes from OpenCode, not from Jev: it lets about 1,000 requests through per window, roughly 10 per
  second.
- The official API handles 24.5 / 48.2 / 45.5 / 40.2 requests per second at concurrency 8 / 16 / 32 / 64 without
  rejections; above 16, latency rises and throughput stops growing.
- Both routes give the same judgements: on 1,011 repeated pairs the match decisions agree 100%, with a mean probability
  difference of 0.002.
- All real experiments therefore use the official API. Old Jev results collected through OpenCode are archived locally in
  `out/_archive_opencode/` and no longer used.

## Notes and limitations

- The blocking setup (Abt-Buy queried by product name only, spelling variants of model numbers merged into one word) was
  tuned against the truth; in practice it should be tuned on a small labelled sample.
- The "to a human" rows assume humans judge every pair they see correctly; they are an upper bound.
- DeepSeek was only tested through OpenCode, so its times include the platform's overhead and limits; this is not a
  like-for-like speed comparison with Jev on the official API.
- Both datasets are public benchmarks, so the models may have seen them in training.
- A single command could run at most 10 minutes, so long tasks were split with `--limit`; finished parts come from the
  cache.

## Reproducing

```bash
./experiments/table_match/fetch.sh
.venv/bin/python experiments/table_match/pipeline.py block abt-buy --k 10
.venv/bin/python experiments/table_match/pipeline.py judge abt-buy                                                # Jev, official API
.venv/bin/python experiments/table_match/pipeline.py judge abt-buy --k 5 --llm deepseek-v4.1-flash --concurrency 8   # baseline via OpenCode
.venv/bin/python experiments/table_match/pipeline.py eval abt-buy --k 5
.venv/bin/python experiments/table_match/route_speed.py --concurrency 8 16 32 64 && .venv/bin/python experiments/table_match/route_speed.py --report
```
