# Phase 6 execution plan: can open, Jev-compatible models replace Jev as the default backend?

Finalised 2026-10-10; Laya multilingual added on 2026-10-11 at the owner's decision. Status: environments, scripts and
rehearsal done, waiting for the owner's go.

This file is the runbook: the order of steps, the acceptance check of each step and how to handle mishaps are all fixed
before the run, so no decisions are improvised midway. The original record of the decision criteria is
[p6.txt](p6.txt) (written before collection, with script checksums); where the two differ, p6.txt prevails.

*Translated from the Chinese original for the public release.*

## 1 Questions

**Main question**: can open models that need no key replace Jev in quality as the default backend of `kevdb`?

**Secondary questions**:
- How well calibrated are their probabilities? So far only each project's self-reported numbers exist, with no
  independent test.
- How many rows per second do they process in batch on this machine?
- Do the answers change when the 21 questions are packed into one request?
- How much of the gap is left after recalibration with 200 human labels?

## 2 Arms

| Arm | What | Pinned version | How obtained | Role |
| --- | --- | --- | --- | --- |
| Jev (cached) | TypeSafe's `jev-1.13.0`, judgements collected on 5–7 October | `jev-1.13.0` | cache replay, free | main reference; every verdict is against it |
| Jev (asked today) | the same model, all questions asked again on the day of the experiment | `jev-1.13.0`, official API | new collection, about $0.43 | control group, see 4.4 |
| DeepSeek V4.1 Flash output probability | an open LLM's probability read with reasoning off | caches of phases 2 and 4 | cache replay | reference (movie and restaurant data only) |
| No-skill baseline | answers every question with its base rate | none | computed from the labels | reference, the floor of each metric |
| Kev-4B | adapter plus decision head on Qwen3.5-4B | code `kev-1.0`, weights `jaredpalmer/kev-4b@v1.0` (6cfce5c2) | local, MLX, bfloat16 | subject |
| Kev-9B | the same approach on Qwen3.5-9B | `jaredpalmer/kev-9b@v1.0` (db029f08) | local, MLX | subject |
| Laya English | ModernBERT-large encoder, English zero-shot | `laya` 0.4.1, weights 7b928d82 | local | subject |
| Laya multilingual | the same software with an mmBERT-base encoder, 100+ languages, zero-shot | `laya` 0.4.1, multilingual subfolder of the same repository, weights 7b928d82 | local | subject |
| Clef-flash | Cloudflare's decision model trained on Qwen3.5-9B | `Cloudflare/clef-flash@8b2e5fd1`, torch 2.11.0, transformers 5.10.2 (the tested versions in its model card) | local, PyTorch on the Mac GPU, released bfloat16 precision; loading and probabilities use the code bundled with its repository | subject |

Out of scope:
- Clef (27B). Its 55 GB of weights exceed this machine's recommended 48 GB GPU limit, and it officially supports NVIDIA
  only.
- Clef and Clef-flash hosted by Cloudflare. Paid; not approved by the owner.

## 3 Data and questions

Everything reuses the original rows and questions previously asked of Jev, with states and questions unchanged to the
letter; every arm receives identical requests.

| Data | Rows | Requests | Questions | Truth |
| --- | --- | --- | --- | --- |
| Movie reviews | 5,000 texts (5,001 labelled rows) | 5,000 | is this review positive? | the critic's overall verdict |
| Restaurant reviews, one by one | 1,000 sentences × 21 predicates | 21,000 | 7 aspects × (mentioned, negative, positive) | MAMS human labels |
| Restaurant reviews, packed | the same 1,000 sentences | 1,000 | 21 questions in one request | same |
| Product matching | Walmart-Amazon test set, 2,049 pairs | 2,049 | are the two records the same product? | the benchmark's labels |
| Movie reviews asked again | the first 200 reviews | 200 | as movie reviews | not needed |

29,249 requests per arm. A probability ≥ 0.5 counts as yes; no threshold, question or temperature is tuned for any model.

## 4 Metrics and decision criteria

### 4.1 Decision rule (frozen)

A model is "on par with Jev" on one dataset if, on the same rows, all of the following hold:

| | Metric | Relative to Jev (cached) |
| --- | --- | --- |
| A | accuracy | at most 2.0 points lower |
| B | AUC | at most 0.02 lower |
| C | Brier | at most 0.02 higher |

Only a model on par on all three datasets (movie reviews, restaurant one by one, product matching) "can replace Jev as
the default backend". Verdicts use point estimates, reported with 95% intervals from 2,000 bootstrap resamples; where the
interval contains the bar, it is marked "not separable from the bar". A dataset where more than 5% of requests have no
answer counts as failed for that model.

### 4.2 Readings reported separately, not part of the verdict

- **Calibration**: equal-mass ECE at most Jev's + 0.02 counts as "calibrated as well as Jev". Equal-width ECE, a
  reliability table and the number of rows in 0.1–0.9 are reported too, not relying on the 0.9–1.0 bin alone.
- **Practicality**: at least 10% of Jev's rows per second on this machine (Jev cold: movie 52, restaurants 53, products 38
  rows per second) counts as "fast enough for batch use".

### 4.3 Secondary analyses

Packed vs one by one; metrics after recalibration with 200 human labels; rows exceeding the model's window; DeepSeek's
output probability judged by the same rules; whether the conclusions change with today's Jev as the reference.

### 4.4 Control group

Jev (asked today) serves three purposes:
1. **Testing the collection process.** It goes through the same new scripts as the open models. If its agreement with the
   cached Jev on movie reviews is below 97%, the process or Jev itself has changed (phase 3 measured 99.45% repeat
   agreement); then the run stops before any open model is asked.
2. **A noise floor.** The difference between two runs of the same model a few days apart is what any other model's
   difference from Jev must exceed.
3. **Throughput and cost on the same day with the same process.**

In addition, every arm asks the first 200 movie reviews again after all other data, to see whether it answers the same
way twice.

### 4.5 Known issues left as they are

Laya's English model answers "no" to every yes/no question without criteria, as its own documentation states. The movie
and restaurant questions have no criteria, so Laya English is expected to fail on both. The questions are not changed
for it: the point is whether a drop-in replacement works. The documentation says the multilingual version lacks this
defect, which is why the owner added it as a subject before the start; it receives exactly the same requests.

Clef-flash runs at about 2.7 requests per second on this machine, around 5% of Jev, so it is expected to fail the
practicality reading. Its quality verdict is unaffected.

## 5 Steps

The whole run is one command:

```bash
./experiments/database/p6_run.sh
```

It does the following, logging to `out/database/p6/run.log`.

| Step | What | Acceptance |
| --- | --- | --- |
| 0 | pre-flight checks: five ports free, offline tests pass, system sleep prevented | nothing starts if any fails |
| 1 | Jev control group: 29,249 requests at concurrency 16 | agreement with the cached Jev ≥ 97% |
| 2 | Laya English: start server → concurrency probe (600 reviews outside the evaluation, 200 each at concurrency 1, 4, 16) → five datasets → stop server | the collection command ends normally |
| 3 | Laya multilingual: as above | as above |
| 4 | Kev-4B: as above | as above |
| 5 | Kev-9B: as above | as above |
| 6 | Clef-flash: as above | as above |
| 7 | inventory: rows and unanswered rows per arm and dataset | all files present |
| 8 | merge each arm's cache into the main cache | |
| 9 | analysis: `experiments/database/p6_report.py`, results in `out/database/p6/report.txt` and `results.json` | |

Local models run one at a time: running them together would compete for the GPU and spoil the throughput numbers. The
fast ones go first, so process problems surface early. Servers run in the macOS sandbox with no network access except
loopback.

After the script finishes:
1. Check the inventory and read `report.txt`.
2. Add a section to `experiments/database/README.md`: one-line conclusion, result tables, reproduction commands,
   limitations, deviations.
3. Append an execution record to the end of p6.txt: start and end time, failures per arm, deviations, cost.
4. Report to the owner, conclusion first, then the evidence.

## 6 Rules fixed in advance

Each of these situations has a fixed handling, so nothing needs to be asked midway. Only rules 1 and 3 stop the
experiment, and both apply before any open model has been asked.

| # | Situation | Handling | Record |
| --- | --- | --- | --- |
| 1 | pre-flight checks fail (port in use, tests fail) | do not start; fix first | not a deviation |
| 2 | the Jev control group cannot run (key, quota or network) | skip it, continue; the cached Jev stays the reference; run the control group later | deviation |
| 3 | control-group agreement with the cached Jev below 97% | stop and investigate the collection process | deviation |
| 4 | a model's server does not start | retry once; if it still fails, skip that model and continue | report "could not run" with the reason |
| 5 | a server exits during collection | stop the collection, restart the server, resume once from the cache; if it fails again keep what was collected and move on to the next model | deviation; throughput sums the new requests and time of every attempt |
| 6 | single requests get no answer | at most 4 re-ask rounds within the run; still missing ones count as wrong for accuracy and 0.5 for other metrics | listed in the report |
| 7 | more than 5% of a dataset's requests unanswered | that dataset fails; no patching | stated in the report |
| 8 | the computer sleeps or the script is interrupted | run the same command again; answered requests are not asked again | deviation |
| 9 | a bug in the collection code that does not affect sent requests or recorded answers | fix and resume | deviation, with new checksums |
| 10 | a bug in the collection code that affects sent requests or recorded answers | delete the affected arm's data, fix, rerun that arm from scratch | deviation, with new checksums |
| 11 | the analysis script fails | collected judgements are unaffected; fix the analysis and rerun it | deviation; criteria unchanged |
| 12 | unexpected results (all fail, all pass, one model absurdly bad) | report as they are | no "rescue" experiments; anything added later is marked post hoc |

Not done during the run:
- No looking at any model's answers; only counts of requests, failures and time. The analysis runs after everything is
  collected.
- No changes to questions, thresholds, temperatures, model versions or criteria.
- No added arms, no removed data.
- No use of Jev's outputs to train, fine-tune, calibrate or pick thresholds for any open model (TypeSafe customer
  agreement, section 2.3(b)); recalibration uses human labels only.

## 7 Time and resources

Times were measured on made-up rows (16–126 of each request type); read them as ±30%. Servers are serial, so more
concurrency does not speed them up.

| Arm | Probe | Movie | Restaurant one by one | Restaurant packed | Products | 200 asked again | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev (asked today) | none, fixed at 16 | 2 min | 7 min | under 1 min | 1 min | under 1 min | about 10 min |
| Laya English | under 1 min | 2 min | 9 min | 3 min | 1 min | under 1 min | about 15 min |
| Laya multilingual | under 1 min | 2 min | 6 min | 2 min | 1 min | under 1 min | about 10 min |
| Kev-4B | 1 min | 10 min | 22 min | 10 min | 7 min | under 1 min | about 50 min |
| Kev-9B | 2 min | 15 min | 35 min | 20 min | 13 min | 1 min | about 85 min |
| Clef-flash | 4 min | 31 min | 143 min | 67 min | 20 min | 1 min | about 4.4 h |

Analysis about 2 minutes; checking and writing the README about 30 minutes. About 7.7 hours in total, 5.5–10 hours at
±30%, most of it Clef-flash.

Resources:
- Only one model in memory at a time; the largest, Clef-flash, takes about 22 GB.
- Mains power and an open lid are needed. The computer can be used normally meanwhile, but nothing else heavy on the GPU,
  or the throughput numbers will be low.
- Cost: only the Jev control group, about $0.43.

## 8 Data and privacy

- Collected judgements go to `out/database/p6/`, and the caches are finally merged into `.jevcache.sqlite`. Neither is
  committed; Jev's judgements stay on this machine.
- The models' code, environments and weights are in `open_models/` (not committed, about 50 GB). Nothing is installed
  system-wide.
- Local model requests go to 127.0.0.1 only, without any key; the servers run in a sandbox with no internet access.
- Only the Jev control group sends anything out: the texts of three public datasets to TypeSafe's official API, as in
  every earlier phase.

## 9 Quality assurance done before the run

| Item | Result |
| --- | --- |
| Offline tests | 33 pass |
| Criteria and scripts frozen | p6.txt, every change dated with reason and checksum |
| Analysis script | every code path exercised on fake data; reproduces the earlier accuracy and AUC on the cached Jev and DeepSeek judgements |
| Shape check | the four request types sent to every model on made-up rows, 0 failures |
| Can Clef-flash run locally? | yes. Loaded on the Mac GPU at its released precision in 23 s; all four request types work |
| Full rehearsal | `./experiments/database/p6_run.sh --rehearsal` runs end to end: control group plus five local models, all files present, 0 failures; the local models asked the same rows again give identical answers |
| Failure drills | killing a server during collection: restarted automatically and resumed to completion. A server that does not start: retried once, then skipped and marked in the inventory |
| Control-group bar | tested with three fake cases: full agreement and slight disagreement pass, 10% flipped judgements fail |
| Environment isolation | snapshots of the home directory, cache directories, shell configuration and Homebrew list identical before and after installation |
| Sandbox | connecting to the internet from the sandbox fails immediately; every server answers normally inside it |

## 10 Outputs and definition of done

The experiment is complete when all of these exist:
1. Under `out/database/p6/`, judgement files and run records per arm and dataset, with an inventory showing everything
   present (or the reasons for gaps recorded under section 6).
2. `out/database/p6/report.txt` and `results.json`.
3. A new section in `experiments/database/README.md` stating which open model, if any, can be the default backend by the
   criteria fixed beforehand.
4. The execution record at the end of p6.txt: start and end time, deviations, total cost.
5. A report to the owner.

## 11 Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Clef-flash's code is written for NVIDIA and may be slow or unstable on a Mac | this arm takes long or does not finish | tried and timed locally before the run (section 7); if it cannot finish, rules 4 and 5 apply |
| Laya English's known defect | expected to fail on two datasets | report as is, citing its own documentation |
| Public datasets may be in these models' training data | every model's accuracy may be inflated | listed as a limitation; the effect goes the same way for every arm |
| The restaurant questions diverge from the labelling guidelines | on this dataset even Jev's accuracy and Brier are below the no-skill baseline | the verdict includes AUC, which a model that only says "no" cannot pass; the no-skill baseline is reported alongside |
| Throughput measured on one machine only | the numbers do not generalise to other hardware | the report names the machine; practicality is a reading only, not part of the verdict |
| Results near a bar | the point estimate passes but the interval crosses the bar | report the interval and mark "not separable from the bar" |
| The Jev API is unstable that day | the control group cannot run | rule 2 |

## 12 Start

After the owner says go:
1. Record the start time and the current script checksums at the end of p6.txt.
2. Run `./experiments/database/p6_run.sh` in the background.
3. Meanwhile, watch only the counts in the log.
4. When the script finishes, do the four things at the end of section 5.
