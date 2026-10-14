# Phase 10 execution plan: does fine-tuning Kev on public record-matching data reach Jev's zero-shot level on unseen matching data?

Drafted 2026-10-13.
Status: the criteria are written ([p10.txt](p10.txt)) but not frozen; the download, data-building and training scripts
are written and rehearsed on made-up rows; the evaluation and analysis scripts are not written yet.
**The experiment starts only on the owner's instruction.** So far no real training record has been fed to a model, and
no fine-tuned checkpoint has answered any evaluation data.

This file is the runbook: what is done, in what order, the acceptance check of each step and how to handle mishaps.
p10.txt prevails on the criteria.

*Translated from the Chinese original for the public release.*

## 1 Question

**Main question**: once Kev has seen record-matching labels from other sources, does it match Jev's zero-shot quality on a
matching dataset it was never trained on?

Why ask this:
- Phases 6 and 7 found that the open models cannot replace Jev as they are, and the largest gap is record matching.
  Calibration fixes the scale, not the ranking; in phase 7 both models stayed behind Jev on two new datasets after
  calibration.
- Kev's training data contain no record pairs: the base training set is ten public classification datasets plus
  generated policy and rule examples, and later stages added dates, complaint texts, generated skill data and
  developer-tool data. This may be the cause, but it has not been tested.
- Kev ships its training code, and its author describes how to continue training from the released checkpoints. The
  author reports a 10-point gain from 5,219 in-domain labels and no visible gain from 400; those are in-domain gains, not
  the transfer asked about here.

So the design is leave-one-out: four matching datasets; each time fine-tune on the training splits of three and compare
with Jev on the test split of the fourth.

## 2 Models

| | What | How obtained |
| --- | --- | --- |
| Kev-4B (subject) | code `kev-1.0`, continued from `jaredpalmer/kev-4b@v1.0` (6cfce5c2), base `Qwen/Qwen3.5-4B-Base@1001bb4d` | local, already in `open_models/` |
| Kev-0.8B (subject) | the same code, continued from `jaredpalmer/kev-0.8b@v1.0` (bf75a6a8), base `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2` | same |
| Released checkpoints (control) | both v1.0 checkpoints as released | answers collected in phase 7, not asked again |
| Jev (comparison column) | `jev-1.13.0`, zero-shot answers on the four test splits from 2026-10-05 | cache; nothing new asked, free |

Data, settings, analysis and bars are identical for both models; each gets its own verdict and they can be started
separately.

## 3 Data

The four Magellan datasets in the version curated in the Ditto repository (Apache-2.0, pinned to commit 52985564). The
training splits are downloaded by `experiments/database/p10_fetch.sh` and checked one by one; the validation and test
splits were already present.

| Dataset | Question | Training split (pairs / matches) | Used | Test split (pairs / matches) |
| --- | --- | --- | --- | --- |
| abt-buy | products | 5,743 / 616 (10.7%) | 2,000 | 1,916 / 206 |
| amazon-google | products | 6,874 / 699 (10.2%) | 2,000 | 2,293 / 234 |
| walmart-amazon | products | 6,144 / 576 (9.4%) | 2,000 | 2,049 / 193 |
| dblp-acm | papers | 7,417 / 1,332 (18.0%) | 2,000 | 2,473 / 444 |

- Each dataset contributes the first 2,000 pairs of its training split ordered by `sha256("p10|<dataset>|<id>")`. The
  ratio of matches to non-matches stays as it is: no resampling, no augmentation, no rewriting.
- A training record is exactly the request sent at evaluation (the state holds both records; question and criteria are
  the first draft of `experiments/entity_match/run.py`, unchanged), plus the human label.
- **Why leave-one-out is necessary.** The splits were drawn at random over record pairs, so a dataset's test and training
  splits share many records: test pairs whose two records both occur in training are 56% on walmart-amazon and 93% on
  abt-buy, and 1–110 test pairs are identical to a training pair. No record is shared verbatim across datasets.
- Across all 26,178 training records the longest state is 311 tokens, within the training script's default of 384, so no
  record is dropped.
- Human labels for fitting the temperature: each dataset's fixed 200 validation pairs, the same 200 used to fit
  calibration parameters in phase 7.
- Replay data: the training part of Kev's own base training set decision-v7 (12,576 records), downloaded from its Hub
  mirror, with checksums matching the manifest in Kev's repository.

## 4 Arms and training

Six arms per model:

| Arm | Trained on | Training records | Evaluated on | Purpose |
| --- | --- | --- | --- | --- |
| released | no training | | all four | control |
| loo-abt-buy | the other three | 6,000 pairs + 2,000 replayed | abt-buy | main test |
| loo-amazon-google | the other three | same | amazon-google | main test |
| loo-walmart-amazon | the other three | same | walmart-amazon | main test |
| loo-dblp-acm | the three product datasets | same | dblp-acm | main test; the paper question never appears in training |
| all | all four | 8,000 pairs + 2,000 replayed | all four | reference: how far in-domain labels go |

Training settings (`experiments/database/p10_train.sh`, each arm trained once, no search):

| Setting | Value | Basis |
| --- | --- | --- |
| Starting point | the released v1.0 checkpoint (`--init_from`) | the author's method; retraining from the base would lose existing abilities |
| Learning rate, epochs | 2e-5, one epoch | the settings of the released checkpoint's last fine-tuning stage; the documentation recommends 2e-5 |
| Effective batch size | 8 (one record per forward pass, an update every 8) | same; more records per forward pass are slower on this machine |
| Replay | 2,000 records of decision-v7 | the author's method against forgetting; the released checkpoint replayed 2,000–4,000 per stage |
| Other | LoRA rank 16, `--p_none_pair 0.25`, seed 1, full-precision weights, gradient checkpointing on | as for the released checkpoint; gradient checkpointing affects only memory and speed |

Not done: no checkpoint selection on validation, no early stopping, no retraining with other seeds, no hyperparameter
tuning. A run that drops any record or produces a non-finite loss does not count.

**Temperature.** A fine-tuned checkpoint comes with temperature 1.0 (confirmed in the rehearsal; it does not inherit the
parent's temperature). Following the author's method, each checkpoint gets a temperature fitted on human labels outside
its training data: 200 validation labels per dataset the arm trained on (600 for a leave-one-out arm, 800 for all). Not a
single label of the held-out dataset is used. The temperature changes only Brier and calibration, not accuracy, AUC or F1.

## 5 Metrics and decision criteria

Evaluation is exactly as in phases 6–8: the checkpoint answers the full test split through Kev's own server (MLX,
sandboxed, concurrency 16), questions unchanged, a probability ≥ 0.5 counts as a match, no threshold tuning.

**Bar T (transfer, the main bar).** On held-out dataset D, the loo-D checkpoint with its fitted temperature is "on par with
Jev" by phase 6's rule: accuracy at most 2 points lower, AUC at most 0.02 lower, Brier at most 0.02 higher. Point
estimates decide, with 2,000-resample bootstrap intervals; intervals crossing the bar are reported as "not separable
from the bar". Only if all four held-out datasets are on par has "fine-tuning brought this model to Jev's zero-shot
level"; otherwise report how many of the four pass and which criterion each misses.

**Guard G (no forgetting).** Each fine-tuned checkpoint's accuracy on these three sets is at most 2 points below the
released checkpoint's:

| | Data | Requests |
| --- | --- | --- |
| g1 | phase 6's first 1,000 movie reviews, "positive?" | 1,000 |
| g2 | phase 6's first 100 restaurant sentences × 21 predicates, one request per question | 2,100 |
| g3 | the development set of decision-v7 (held-out questions from the sources Kev was trained on) | 1,204 |

A checkpoint that passes T but fails G is reported as "reaches Jev on matching but forgets other things" and does not
count towards the conclusion.

**A model's conclusion** = all four held-out datasets pass T, and all four checkpoints pass G.

Readings without a bar:
- How much each leave-one-out arm improves over the released checkpoint (differences and intervals of accuracy, AUC,
  Brier, F1). An F1 interval above zero counts as "improved". If T fails, this is the phase's conclusion.
- The three product folds (question seen in training) and the dblp-acm fold (question never seen) reported separately.
- Error types: false matches and misses, mean probability on non-matching pairs, share of probabilities in 0.1–0.9,
  calibration error, share of errors in the least certain 30%.
- The all arm on the four test splits, side by side with Jev and Ditto's published numbers (Abt-Buy 89.33, Walmart-Amazon
  86.76, Amazon-Google 75.58, DBLP-ACM 98.99; checked against table 5 of the paper). Not a like-for-like comparison: Ditto
  trains one model per dataset on all 5,743–7,417 pairs; here one model with 2,000 pairs per dataset. Both test splits
  contain many records seen in training.
- End to end: phase 8's two raw tables (subset of 400 left records, pairwise), abt-buy with the loo-abt-buy checkpoint and
  dblp-scholar with loo-dblp-acm; abt-buy also with "a left record matches at most one".
- Option order: phase 8's multiple-choice form on the first 100 left records of the abt-buy subset, three orders.
  Training contains yes/no questions only, so no change is expected; this reading is a baseline for later "train in order
  invariance".

**Expectations beforehand** (recorded in p10.txt; the results may overturn them): Kev-4B improves on all four folds, with
abt-buy and walmart-amazon passing T and amazon-google and dblp-acm uncertain; Kev-0.8B improves a lot but passes T on at
most two. Both pass G.

## 6 Steps

One command per model (`experiments/database/p10_run.sh --model <model>`, not yet written, see section 10), logging to
`out/database/p10/<model>/run.log`.

| Step | What | Acceptance |
| --- | --- | --- |
| 0 | pre-flight checks: port free, offline tests pass, checksums of training and replay data, disk and free memory, sleep prevented | nothing starts if any fails |
| 1 | `experiments/database/p10_fetch.sh`; `python experiments/database/p10_data.py build`; `python experiments/database/p10_data.py pools` | record counts and checksums of the five training files match those recorded at freezing |
| 2 | for each of the five arms: `experiments/database/p10_train.sh <model> <arm>` → start this checkpoint's server → ask (test split, temperature labels, the three guard sets, secondary readings) → stop the server | checkpoint complete; no dropped records in the training log; requests and failures per step |
| 3 | the released arm: replay from the caches of phases 6, 7 and 8, asking fresh only what is not cached | the replay sends no requests |
| 4 | inventory and analysis: `out/database/p10/<model>/report.txt` and `results.json` | |

Training and evaluation alternate (each arm is evaluated right after training), so if the run stops midway the finished
arms are complete. Only one training run or one server uses the GPU at a time.

After the command finishes: check the inventory and read the report; add a "Phase 10" section to
`experiments/database/README.md`; append an execution record to the end of p10.txt; report to the owner, conclusion
first.

## 7 Time and resources

Measured in the rehearsal (made-up record pairs, final settings; M3 Max 64 GB, PyTorch MPS; results in
`out/database/p10_rehearsal/feasibility.txt`):

| | Per training record | Peak process memory | Checkpoint size |
| --- | --- | --- | --- |
| Kev-0.8B | 0.8–1.0 s | under 9 GB | 62 MB |
| Kev-4B | 2.4–2.5 s | about 26 GB | 148 MB |

- Without gradient checkpointing Kev-4B grew to 84 GB within three minutes and was stopped; Kev-0.8B normally uses about
  10 GB but one record spiked it to 24 GB. So both models use it.
- Kev-4B with half-precision weights is only 8% faster and uses 17 GB; not adopted (the released checkpoint was trained
  in full precision), kept as a fallback if memory gets tight.
- Measured separately on the 24 longest records of decision-v7: Kev-4B still 26 GB, Kev-0.8B 9 GB; memory does not run
  away with longer records.
- Real training records average 186–207 tokens, slightly shorter than the rehearsal's (218), so the estimates below are
  slightly conservative.

Time with 2,000 pairs per dataset and 2,000 replayed records, read as ±30%:

| | Kev-0.8B | Kev-4B |
| --- | --- | --- |
| Training each of the four leave-one-out arms (8,000 records each) | about 1.8 h each | about 4.9 h each |
| Training the all arm (10,000 records) | about 2.3 h | about 6.1 h |
| Training in total | about 9.6 h | about 26 h |
| Evaluation (about 63,000 / 52,000 requests) | about 1 h | about 2.6 h |
| Total | about 10.5 h | about 28.5 h |

Resources:
- During training the process uses 9 GB (Kev-0.8B) or 26 GB (Kev-4B); evaluation as in earlier phases. A training process
  above 44 GB is stopped by the script and handled under section 8.
- Mains power and an open lid are needed. Other GPU-heavy programs slow training noticeably.
- Disk: the ten checkpoints take about 1.1 GB in `open_models/p10/` (not committed).
- Cost: zero. The Jev column comes entirely from the cache.

Alternatives (to be decided before the start and recorded in p10.txt):

| Option | Kev-0.8B | Kev-4B | Trade-off |
| --- | --- | --- | --- |
| Default: local, 2,000 pairs per dataset | about 10.5 h | about 28.5 h | long |
| Local, 1,000 pairs per dataset and 1,000 replayed | about 5.5 h | about 15.5 h | each leave-one-out arm has only 3,000 pairs, between the author's "no visible gain" (400) and "clear gain" (5,219); a failure would be hard to interpret |
| Kev-4B trained on a rented GPU | | roughly twenty minutes per training run | paid, needs the owner's own Modal account; public data and Kev's code have to be uploaded; training precision becomes the released recipe's bf16. The speed is scaled from the author's own numbers and the price is unverified; if chosen, check the price first |

## 8 Rules fixed in advance

| # | Situation | Handling | Record |
| --- | --- | --- | --- |
| 1 | pre-flight checks fail | do not start; fix first | not a deviation |
| 2 | a training run crashes or is stopped by the memory watchdog | retrain that arm from scratch once (adapter training cannot resume); if it fails again the arm is missing, continue with the next | deviation; a model missing a leave-one-out arm is reported "not completed" |
| 3 | the training log shows dropped records or a non-finite loss | the run does not count; handled by rule 2 | deviation |
| 4 | a server does not start or exits during collection | redo the whole step once; if it still fails keep what exists and continue | deviation; report "not completed" |
| 5 | single requests get no answer | at most 4 re-ask rounds; still missing ones count as wrong, probability 0.5 | listed in the report |
| 6 | more than 5% of a dataset's requests unanswered | that dataset is "behind"; no patching | stated in the report |
| 7 | the computer sleeps or the script is interrupted | run the same command again; trained arms are not retrained, asked requests come from the cache; the interrupted training restarts from scratch | deviation |
| 8 | a bug that does not affect training data, sent requests or recorded answers | fix and resume | deviation, with new checksums |
| 9 | a bug that affects training data, requests or answers | delete the affected part (including checkpoints trained on wrong data), fix, redo that part from scratch | deviation, with new checksums |
| 10 | unexpected results | report as they are | no "rescue" training; anything added later is marked post hoc |

Not done during the run:
- No looking at any answers; only counts, failures, time, memory and the losses the training script prints. The analysis
  runs after all arms of a model are collected.
- No changes to questions, the selection of training data, training settings, the temperature fitting, thresholds or
  bars; no added arms, no removed data.
- No paid API.
- No use of Jev's outputs to train, fine-tune, calibrate or pick thresholds for any open model (TypeSafe customer
  agreement, section 2.3(b)); every label is a human label from the benchmarks.

## 9 Data and privacy

- Training and serving run in the sandbox without network access; weights and data are already local.
- Training data, checkpoints and results are in `out/database/p10/train/`, `open_models/p10/` and `out/database/p10/`,
  none committed.
- Nothing is installed system-wide: the PyTorch used for training is already part of Kev's environment.
- Only two things were downloaded for this phase (done, about 28 MB together): the four training splits and the training
  part of Kev's decision-v7, the latter placed in `open_models/kev/evals/v7/decision-v7/`.

## 10 Done so far, and still to do before the start

Done (no real training record was used for training, and no fine-tuned checkpoint has answered evaluation data):

| Item | Result |
| --- | --- |
| Data checks | counts, match ratios, overlap with the test splits, cross-dataset overlap and token lengths of the four training splits, see section 3 |
| Published numbers | Ditto's four F1 scores and the 3:1:1 split ratio, checked against the source (arXiv 2004.00584 table 5, DeepMatcher's Datasets.md) |
| Three scripts | `experiments/database/p10_fetch.sh`, `p10_data.py`, `p10_train.sh`; 8 offline tests pass (`tests/test_p10.py`), one of them confirming that training records are identical to evaluation requests |
| Training rehearsal | each model trained once on 72 made-up record pairs plus 16 replayed records with the final settings; speed and memory in section 7 |
| Serving rehearsal | both fine-tuned checkpoints load in the server used in phases 6–8 and answer; their own temperature is 1.0 |
| Problems the rehearsal caught | without gradient checkpointing Kev-4B's memory exceeds physical memory; phase 7's label-pool ordering key differed from the first draft here (now copied from phase 7, with a test) |

Still to do before the start (free, without touching real data):

1. Write the evaluation script (start each checkpoint's server, ask, store, fit the temperature, the three guard sets,
   the secondary readings), the analysis script and the one-command `p10_run.sh`, with offline tests.
2. Rehearse the whole run on made-up rows from an empty directory, including failure drills (killing training midway,
   killing the server during collection) and the memory watchdog.
3. Append these scripts' checksums and the rehearsal results to p10.txt and freeze it. Only then can the run start.

## 11 Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Each arm is trained once with one seed | results near a bar cannot tell too little data from luck | report intervals; "not separable" when they cross the bar; no retraining afterwards |
| Only 2,000 pairs per dataset | if T fails, "all the data would have been enough" cannot be ruled out | listed as a limitation; the all arm gives the in-domain reference |
| Three of the four datasets are products | the dblp-acm fold changes both domain and question and is the hardest; the product folds may help each other | the two kinds of fold are reported separately |
| The held-out dataset's temperature is fitted on other data | the scale may not fit and Brier may fail | this is part of what is tested; raw-probability numbers are reported too |
| Benchmarks public for years | both Jev and the Qwen base may have seen them | listed as a limitation; affects before and after fine-tuning alike |
| Long training | an interruption costs at most about 6 hours | evaluate each arm right after training; start detached from the session; rules 2 and 7 |
| Local training in full precision, while the released recipe uses bf16 on a GPU | small numerical differences from the author's path | listed as a limitation; the evaluation path is identical to earlier phases |
| The guard checks only three sets | forgetting on other tasks goes unseen | listed as a limitation; g3 is Kev's own evaluation set |
| The evaluation script is not written yet | the evaluation part of the time estimate is scaled from earlier phases' throughput | update this section after the full rehearsal |

## 12 Open decisions

1. **Training size and place.** The default is local with 2,000 pairs per dataset, about 39 hours for both models. The
   other two options are in section 7.
2. **Order.** By default Kev-0.8B first (about 10.5 hours, one night), then Kev-4B (about 28.5 hours); either can also run
   alone. Kev-4B is the more likely to reach Jev's level and so the more important one; Kev-0.8B is cheap and suits a first
   pass of the whole process on real data.
3. **Whether to finish the three remaining items of section 10 now**, so that the run can start as soon as the owner says go.
