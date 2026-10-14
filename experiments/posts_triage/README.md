# Experiment: batch-labelling social posts

The first real task for the `kevdb` batch pipeline (see [docs/batch-pipeline.md](../../docs/batch-pipeline.md)): clean
and label the 982 X posts about Jev collected by jev-hub, whose own tags come from keyword matching and are very noisy.
Each post gets four questions: what kind of post it is (`choice`), how much reusable substance it has (`score`), whether
the author reports first-hand work (`noul`), and whether the author criticises Jev itself (`noul`).

Files:

| File | Content |
| --- | --- |
| [posts_triage.toml](posts_triage.toml) | the task: input, state fields, questions, review thresholds |
| [sample_posts.jsonl](sample_posts.jsonl) | 12 made-up posts for trying the pipeline |
| [posts_gold.jsonl](posts_gold.jsonl) | gold labels for 161 posts (ids and labels only, no post text) |
| `data/posts.json` | the full dataset from jev-hub, downloaded locally, not committed |

## Results

On a random sample of 150 posts (official API, `jev-1.13.0`, question version v2, 2026-10-05):

| Question | Accuracy | Notes |
| --- | --- | --- |
| content_type | 88% (73% without alternative answers) | 95% accuracy on the 77% of rows with confidence ≥ 0.6 |
| substance | 75% exact | 99% within one level of the gold label |
| first_hand | 93% | 95% accuracy on the 92% of rows with confidence ≥ 0.6 |
| critical | 97% | precision 70%, recall 88% for "yes" |

The full 982 posts took 20.7 seconds and 912,064 input tokens, about $0.038, with 0 failures; 34% of rows went to review.

Caveat: the gold labels were produced blind by an LLM, not by humans; question version v2 was revised after looking at
Jev's errors on these labels, so the v2 numbers are optimistic.

### Against small LLMs

`--llm` lets a generative model answer exactly the same state, questions and option definitions (asked to return JSON),
with the rest of the pipeline unchanged. 161 gold rows, concurrency 8. Jev used the official API (2026-10-05); the three
LLMs were only available through OpenCode (2026-10-02), so the speed rows are not a like-for-like comparison:

| | jev-1.13.0 (official) | deepseek-v4.1-flash, reasoning off | glm-5.3-flash, reasoning low | glm-5.3-flash, default reasoning |
| --- | --- | --- | --- | --- |
| Total time | 8.2 s | 27.0 s | 39.3 s | 209.9 s |
| Request latency, median / p95 | 0.33 s / 0.67 s | 1.02 s / 2.40 s | 1.83 s / 2.92 s | 6.45 s / 28.49 s |
| Cost per 1,000 rows | $0.039 | $0.25 | $0.14 | $0.53 |
| content_type accuracy (strict) | 89% (73%) | 91% (79%) | 90% (75%) | 93% (79%) |
| first_hand accuracy | 93% | 96% | 92% | 97% |
| critical accuracy | 97% | 98% | 95% | 97% |
| substance within one level | 99% | 99% | 100% | 99% |
| Confidence / probability | yes, usable calibration | no | no | no |

Conclusion: on this task accuracy is level (150 samples; differences of a few points are within the error), Jev is 4–6×
cheaper, and its latency is a third to a twentieth of these LLMs' (LLM latency includes the third-party platform's
overhead). That is still far from the advertised "193× faster, 444× cheaper".

Beyond speed and price, Jev is simpler in three ways:
- Usable confidence routes uncertain rows to review; an LLM only gives answers.
- No output-parsing layer. With reasoning off, deepseek answered score questions with the level name instead of the
  requested index on 58% of rows, which needed extra tolerant parsing.
- No reasoning-level pitfalls. glm cannot turn reasoning off entirely; left at its default it is 5× slower and 4× more
  expensive.

Limitations: one task with short texts (≤ 386 characters); gold labels from an LLM, question v2 revised on Jev's errors;
timings measured once.

## Reproducing

```bash
.venv/bin/kevdb run experiments/posts_triage/posts_triage.toml --mock          # offline, meaningless answers, no key
.venv/bin/kevdb run experiments/posts_triage/posts_triage.toml --input experiments/posts_triage/data/posts.json --records-path items
.venv/bin/kevdb eval experiments/posts_triage/posts_triage.toml --pred out/posts_triage.posts.jsonl --gold experiments/posts_triage/posts_gold.jsonl
.venv/bin/kevdb run experiments/posts_triage/posts_triage.toml --input out/gold_todo.jsonl --llm deepseek-v4.1-flash
```
