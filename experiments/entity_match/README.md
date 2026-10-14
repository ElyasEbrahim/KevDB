# Experiment: entity matching with Jev

Question: in the data-processing task best suited to a decision model, how good is Jev against human labels?

Entity matching decides whether two records describe the same entity (two product records the same product, two
bibliographic records the same paper). It was chosen because the question is naturally yes/no, the number of pairs grows
quadratically so the unit price decides feasibility, probabilities allow threshold tuning, and established benchmarks
with human labels exist.

Data: four datasets of the Magellan benchmark (the version curated in the Ditto repository). The question is a zero-shot
first draft that was not edited against the test sets; thresholds are chosen on validation splits only.

Calls: Jev through the official API (`jev-1.13.0`, concurrency 16). The DeepSeek baseline was only available through the
third-party platform OpenCode.

## Results (2026-10-05, full test sets, 8,731 pairs)

| Dataset | Test pairs | Jev precision | Jev recall | Jev F1 | F1 with threshold from validation | Ditto (supervised) | GPT-4 zero-shot (mean / best of 10 prompts) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Abt-Buy (products) | 1,916 | 88.5% | 93.7% | 91.0 | 90.2 | 89.33 | 93.87 / 95.78 |
| Walmart-Amazon (products) | 2,049 | 90.3% | 86.5% | 88.4 | 87.9 | 86.76 | 87.80 / 89.67 |
| Amazon-Google (software) | 2,293 | 57.4% | 88.0% | 69.5 | 67.5 | 75.58 | 72.27 / 76.38 |
| DBLP-ACM (papers) | 2,473 | 94.9% | 99.8% | 97.3 | 97.9 | 98.99 | 96.97 / 98.41 |

Ditto's numbers are from Li et al. 2020 (arXiv 2004.00584), trained on thousands of labelled pairs per dataset. GPT-4's are
from Peeters and Bizer 2025 (arXiv 2310.11244), who downsampled the test sets to about 1,250 pairs (keeping every match),
which raises the share of matches and inflates F1, so they are not strictly comparable with the numbers here.

Choosing the threshold on the validation split brought no consistent gain (three of four slightly lower); a fixed 0.5 is
enough.

Against a small LLM on the same 400 random pairs (DeepSeek V4.1 Flash, reasoning off, the same question, via OpenCode):

| Dataset | Jev F1 | DeepSeek F1 |
| --- | --- | --- |
| Abt-Buy | 93.1 | 93.1 |
| Walmart-Amazon | 87.0 | 79.5 |
| Amazon-Google | 66.0 | 61.0 |
| DBLP-ACM | 97.1 | 93.6 |

- Cost: Jev $0.020 per 1,000 pairs, DeepSeek $0.104, about 5×.
- Speed: Jev's median latency is 0.30–0.32 s; the four test sets (8,731 pairs) took about 3 minutes. DeepSeek via OpenCode
  takes 0.8–0.95 s per request, platform overhead included.
- Probabilities are trustworthy: Jev gives ≤ 0.1 or ≥ 0.9 on 74–94% of pairs, with 98.5–99.9% accuracy there.

Limitations: these benchmarks have been public for years and Jev may have seen them in training; the LLM baseline covers
one model and 400 pairs per dataset (only 34–66 matches, so F1 is uncertain by several points); no LLM with reasoning on
was tested.

How the open models do on the same data is in [phase 7 of the database experiments](../database/README.md#phase-7-what-does-a-local-model-inside-duckdb-do-that-a-cloud-api-cannot).

## Reproducing

```bash
./experiments/entity_match/fetch.sh                                                # validation and test splits, about 6 MB
.venv/bin/python experiments/entity_match/run.py abt-buy --split valid --sample 600
.venv/bin/python experiments/entity_match/run.py abt-buy --split test
.venv/bin/python experiments/entity_match/run.py abt-buy --split test --sample 400 --llm deepseek-v4.1-flash --concurrency 8
.venv/bin/python experiments/entity_match/evaluate.py jev-1.13.0
```
