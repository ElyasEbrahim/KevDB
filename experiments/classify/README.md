# Experiment: text classification against human labels

Question: labelling is the most common data-processing job. On public benchmarks with human labels, how accurate are
Jev's multiple-choice questions (`choice`), and how do they compare with a small LLM?

The earlier posts experiment ([posts_triage](../posts_triage/README.md)) used reference answers labelled by an LLM, which
limits how much it proves, hence this experiment.

Data: the AG News test set (news topics, 4 classes, 7,600 texts) and the Banking77 test set (banking customer-service
intents, 77 classes, 3,080 texts). The questions are zero-shot first drafts that give only the class names (plus one
sentence per class for AG News) and were not edited against the test sets.

Calls: Jev through the official API (`jev-1.13.0`, concurrency 16). The DeepSeek V4.1 Flash baseline went through the
third-party platform OpenCode (reasoning off, concurrency 8).

## Results (2026-10-05)

| Dataset | Classes | Jev accuracy (95% interval) | Same 500 texts: Jev / DeepSeek | Cost per 1,000: Jev / DeepSeek | Latency: Jev / DeepSeek |
| --- | --- | --- | --- | --- | --- |
| AG News (7,600) | 4 | 89.0% (88.3–89.7) | 88.8% / 88.8% | $0.019 / $0.092 | 0.30 s / 0.80 s |
| Banking77 (3,080) | 77 | 79.9% (78.5–81.4) | 79.8% / 76.4% | $0.043 / $0.275 | 0.33 s / 0.93 s |

- Differences on the same 500 texts: AG News 0.0 (interval −1.8 to +1.8), Banking77 +3.4 (interval +0.8 to +6.2).
- Confidence routing: accepting only results with confidence ≥ 0.9 passes 87% of AG News rows at 93.1% accuracy and 67% of
  Banking77 rows at 93.3%. The remaining rows are only 61.8% and 53.1% accurate, so confidence does separate easy from hard.
- Zero-shot Jev is clearly below specialised models trained on the full training data: Banking77 93.66% in the literature
  (Casanueva et al., 2020, arXiv 2003.04807), AG News usually reported at 94–95%.

Limitations: public benchmarks that the model may have seen in training; the LLM baseline is one small model on 500 texts
per dataset; no flagship LLM was tested.

## Reproducing

```bash
mkdir -p experiments/classify/data
curl -o experiments/classify/data/banking77_test.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
curl -o experiments/classify/data/agnews_test.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv
.venv/bin/python experiments/classify/run.py agnews
.venv/bin/python experiments/classify/run.py banking77
.venv/bin/python experiments/classify/run.py agnews --sample 500 --llm deepseek-v4.1-flash --concurrency 8
.venv/bin/python experiments/classify/run.py banking77 --sample 500 --llm deepseek-v4.1-flash --concurrency 8
.venv/bin/python experiments/classify/run.py --report
```
