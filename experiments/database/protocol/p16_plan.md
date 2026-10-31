# Phase 16 plan: query-aware allocation of inference cost

Drafted 2026-10-30. Status (2026-10-31): approved by the owner with the recommended options; the criteria were written into `p16.txt`,
frozen at 03:49:28Z and run (899 cells, $0); the readings and one deviation are at the end of `p16.txt`. Original status line: draft
for the owner's approval. No arm has been run and nothing has been spent. After
approval, the criteria in sections 6–8 go into `p16.txt`, which is frozen (with checksums of scripts and inputs) before
any arm is run on real labels.

## 1 Question

A query computes a semantic column with a cheap model on every row (here Jev; Kev-4B as a check) and aggregates it.
The engine can buy a more reliable answer for some rows: a second, more expensive model, or a human label.

1. **Propagation.** How do the cheap model's row-level errors turn into the error of the final answer, depending on the
   query: one large group, many small groups, rare categories, weights, a join that reuses a record in many output rows,
   a threshold decision?
2. **Allocation.** If the engine chooses the rows that get the expensive tier from the query itself (each row's influence
   on the answer), is the final answer reliable against human labels at a lower cost than with choices that ignore the
   query or rely on the model's own confidence?
3. **Systematic bias and correlated errors.** Does the cheap model's error rate differ between groups? Does a second
   model share its errors, so that using it as the reference gives a confidently wrong answer? Does measuring the
   residual error on a small human pilot, instead of trusting confidence, change the allocation's cost or validity?

Not the contribution, used only as baselines: replacing the expensive model with a cheaper one, cascades that route
rows by confidence (Palimpzest/Abacus, LOTUS, BARGAIN). Everything sits at the model-invocation layer (which rows go to
which tier, and how the answer is computed from them); no change to any model server.

## 2 What earlier work already does

Checked in each paper's text on 2026-10-30 (not abstracts only).

| Work | Guarantee, relative to what | Query structure used to allocate |
| --- | --- | --- |
| LOTUS (Patel et al., PVLDB 18(11), 2025) | accuracy target with probability 1−δ, per operator, relative to the operator's gold algorithm (the expensive LLM) | per operator |
| BARGAIN (Zeighami, Shankar, Parameswaran, SIGMOD'26) | accuracy, precision or recall of per-row outputs relative to the oracle LLM | none (cascade threshold) |
| FDJ (same authors, arXiv 2512.05399) | join precision and recall relative to the LLM run on every pair | the join |
| Stretto (Sanmartino et al., arXiv 2602.04430 v3) | end-to-end precision and recall relative to a golden plan, or to human labels if given; error budget allocated across operators | whole plan, set-valued results only |
| Abacus in Palimpzest (Russo et al., arXiv 2505.14661) | none; sample-based quality estimates and Pareto cascades | operator choice |
| ABae, ABae-GroupBy (Kang et al., PVLDB 14, 2021) | intervals relative to the oracle predicate (an expensive DNN or human labellers); proxy used only to stratify; two stages | proxy-score strata; GROUP BY: allocation minimising the largest group MSE |
| BaS (Zhu, Jin, Mo, Kang, SIGMOD 2026) | intervals relative to the oracle, assumed to give the ground truth (may be a human) | semantic-join pairs; GROUP BY: minimises the largest relative group error |
| OLLA (Hui et al., DASFAA 2026) | intervals for the LLM's result on all rows; agreement with labels only measured | embedding strata, Neyman allocation |
| UQE (Dai et al., NeurIPS 2024) | unbiased for the LLM's judgements, assumed correct | embedding strata, importance sampling |
| ThalamusDB (Jo, Trummer, SIGMOD 2024) | deterministic bounds on the full-data result, assuming error-free models; three user labels per predicate | aggregates with joins |
| Evergreen (Lee et al., arXiv 2604.26180 v2) | anytime-valid confidence sequences for the operator's labels; family-wise α split across operators | threshold comparisons decided early |
| SampleClean (Wang et al., SIGMOD 2014) | unbiased for the result if all data were cleaned; corrects the dirty answer with a cleaned sample | uniform, or stratified by group |
| Kanagal, Li, Deshpande (SIGMOD 2011) | influence of input tuples on query results (conjunctive, threshold, top-k, aggregation), for explanations | not used to buy labels |
| CVOPT (Nguyen et al., arXiv 1909.02629) | group-by sampling without models: allocation optimising a norm of per-group CVs | groups |
| PPI, PPI++ (Angelopoulos et al., 2023) | valid against labels | none |
| Stratified PPI (Fisch et al., 2024) | valid against human labels; strata where the autorater's accuracy differs; allocation across strata | fixed strata, one estimand |
| Active inference (Zrnic, Candès, arXiv 2403.03208, 2024); robust sampling (Li, Zrnic, Candès, NeurIPS 2025) | valid against labels; π ∝ model uncertainty; for an M-estimator the optimal rule weights a point by its influence on one coordinate; robust version mixes with uniform | one coordinate |
| Confidence-driven inference (Gligorić et al., NAACL 2025) | valid against human labels; LLM confidence picks the items humans label | one estimand |
| OPAL (Ma, Candès, arXiv 2606.03211) | valid against labels; learns a labelling policy that minimises the variance of a functional of subgroup means | one functional; joint allocation over several estimands and overlapping groups listed as future work |
| MultiPPI (Cowen-Breen et al., arXiv 2603.27414); Active MultiPPI (Brawand et al., arXiv 2605.08429) | relative to the high-quality source (human or strong model); per-row choice of models and gold labels under one budget | one mean; Active MultiPPI warns that miscalibrated surrogates can mis-sample subgroups |
| DSL (Egami et al., NeurIPS 2023) | valid downstream inference with known gold-sampling probabilities, even with arbitrarily biased surrogates | — |
| Kim et al. (ICML 2025) | across 350+ LLMs, two models that are both wrong agree 60% of the time on one leaderboard; more accurate models are more correlated, also across providers | — |

**Conclusion of the check.** The working conclusion from the previous search is partly overturned:

- Allocating the expensive budget by GROUP BY with a guarantee exists: ABae-GroupBy and BaS do it, relative to an oracle
  that can be a human labeller. Influence-weighted labelling with validity against labels exists in statistics (active
  inference, OPAL). Correcting a cheap full-table answer with a labelled sample is SampleClean (2014). Threshold-aware
  early stopping exists (Evergreen), relative to the LLM's labels.
- Not found: weighting a record's budget by how many output rows reuse it through a join (BaS samples pairs; InferF,
  SIGMOD 2026, reuses computation, not labels; OPAL names two overlapping groups as future work); influences derived
  from the plan for weights, joins and HAVING in one engine, with one allocation for many outputs; the expensive model
  treated as a fallible tier whose shared errors are measured on the final answer against human labels; tests of
  group-dependent bias on grouped queries against human labels.

So this phase proposes no new estimator. It tests the engine-level combination (influences from the plan, a design-based
correction, residuals measured by a human pilot) against the strongest prior designs, on real model errors.

## 3 Data and models

Everything is cached from earlier phases; every row has a human label. All arms replay; no model is called.

| Population | Rows | Truth | Cheap model, on all rows | Other models cached |
| --- | --- | --- | --- | --- |
| D1 movie reviews (SemBench `sf_16000`, Rotten Tomatoes) | 15,388 reviews of 172 movies | the critic's own fresh/rotten verdict (`scoreSentiment`) | Jev P(positive) (phase 2) | GPT 6 Luna yes/no on all rows (phase 2) |
| D1-5k (fixed hash sample) | 5,001 | same | Jev; Kev-4B | Kev-9B, DeepSeek V4.1 Flash, DeepSeek V4 Pro, Qwen 3.8 Flash (phases 2, 6) |
| D1-1.85k (`sf_2000`) | 1,851 | same | Jev | Opus 5.5 (phase 5) |
| D2 restaurant sentences (MAMS) | 3,949 sentences × 21 predicates | MAMS aspect and sentiment annotations | Jev, 21 questions packed (phase 4) | — |
| D2-1k (fixed hash sample) | 1,000 × 21 | same | Jev; Kev-4B packed | Luna packed, DeepSeek V4.1 Flash packed (phases 4, 6) |
| D2-546 | 546 × 21 | same | Jev | Opus 5.5 packed (phase 5) |

Structure available to the engine: D1 has movie (4–503 reviews, median 30), critic (2,426, median 3), publication
(1,492, median 6), top critic (19% of rows), genre (20 genres, 21–9,015 reviews each; a movie has 1–6) and box office
(161 of 172 movies; `Movies.csv` repeats 15 of its 187 rows, so movies are deduplicated by id). D2 has 21 predicates
with prevalence 1.2% to 73%.

Measured cost per 1,000 rows: D1 Jev $0.013, Luna $0.036, DeepSeek V4.1 Flash $0.060, Opus $1.69. D2 (21 questions in
one request, per 1,000 sentences) Jev $0.038, Luna $0.220, DeepSeek $0.477, Opus $12.2. Human labels have no measured
price, so budgets are counted in labels and the three-tier analysis uses stated price ratios.

**Seen before writing this plan** (inventory, to be disclosed in `p16.txt`): on D1 Jev is right on 91.1% of rows, but
its threshold count puts the positive share at 75.4% against 81.3% (most errors are false negatives: 7.4% of rows,
against 1.5% false positives); Luna's share is 72.9%. For genres with at least 200 reviews, Jev's count is off by −1.7
(LGBTQ+) to −9.6 points (Horror), and it flips the fresh/rotten verdict (60% line) of 17 of 172 movies; 10 movies are
within 5 points of 60%. When Jev
is wrong, Luna is also wrong 80% of the time, the 5,001-row models 67–85%, Opus 64%. Only 8.7% of Jev's errors on D1 have
p ≤ 0.1 or ≥ 0.9. On D2, Jev's count is off by −10.8 to +40.3 points per predicate (negative:menu, 1.2% prevalence:
+9.2; mention:service: +40.3), and when Jev is wrong Luna is also wrong 21–89% of the time per predicate. Phases 3–5
results are known. No allocation arm or interval has been computed on any query.

## 4 Queries

| Query | SQL shape | Outputs | Where a row's influence comes from | Requirement h (95% half-width) | Budgets (labels); primary |
| --- | --- | --- | --- | --- | --- |
| Q1 large group (control) | share of positive reviews | 1 | 1/N, the same for every row | 1 point | 100, 200, 500, 1,000, 2,000; 500 |
| Q2 small groups | share positive per movie (the Tomatometer); per publication as a secondary | 172 | 1/n_g of its group | 10 points | 500, 1,000, 2,000, 4,000; 2,000 |
| Q3a weights | mean Tomatometer across movies | 1 | 1/(172·n_movie) | 1 point | as Q1; 500 |
| Q3b weights | box-office-weighted mean Tomatometer | 1 | box office / n_movie | 1 point | as Q1; 500 |
| Q4a join reuse | share positive per genre, via the movie→genre bridge | 20 | 1/n_genre for each genre of its movie (1–6 outputs) | 5 points | as Q2; 2,000 |
| Q4b join reuse, semi-synthetic | share positive over a fact table where review i appears k_i times (k_i heavy-tailed, lognormal σ = 1 and 2, fixed seed; one variant larger for top critics) | 1 | k_i / Σk | 1 point | as Q1; 500 |
| Q5 threshold | HAVING share ≥ 60%: which movies are fresh | 172 decisions | distance of the movie to the 60% line | band of ±5 points: either verdict accepted inside it | as Q2; 2,000 |
| Q6 rare categories | the 21 predicate shares on D2 | 21 | 1/N per predicate; one label answers all 21 | max(1 point, 25% of the share) | 100, 200, 400, 800 sentences; 400 |

From structure alone (constant residual), labelling in proportion to influence instead of uniformly would need about
6.5× fewer labels on Q2 and Q3, 7.2× on Q4a, 2.7× and 55× on Q4b (σ = 1, 2), but only 1.08× for the genre reuse on its
own: the real join reuses a review at most six times. Prior group-aware designs already capture Q2 and Q3a.

## 5 Method and arms

**The engine's procedure.** (1) The cheap model answers every row. (2) From the plan and the structured columns only, the
engine computes each row's influence a_ik on output k (table above). (3) It draws a sample with known inclusion
probabilities π_i, which minimise Σ_k Var_k / h_k²; for this estimator that gives π_i ∝ √(Σ_k a_ik² s_ik² / h_k²), with s_ik²
the expected squared residual (truth − cheap probability), capped at 1, mixed 80/20 with uniform so that no row has a
negligible chance. (4) The chosen rows get the expensive tier; every output gets a design-based estimate and interval.

**Estimator, the same for every arm** (so arms differ only in which rows are labelled): the cheap probability is
recalibrated by a working model fitted on the labelled rows (cross-fitted), and each output is the sum of influence ×
calibrated prediction over all rows plus the influence-weighted residuals of the labelled rows divided by π. The
interval is ±1.96 √V̂ with the Horvitz–Thompson variance of that residual sum. Two-stage arms combine the pilot's and
the main stage's estimates with fixed weights (the martingale form used by active inference), so they stay unbiased
although the main stage depends on the pilot.

**Arms**, at equal label budgets (the pilot counts towards the budget):

| Arm | Uses the query's structure | Residual s used to allocate | Corresponds to |
| --- | --- | --- | --- |
| U | no | none (uniform) | PPI; SampleClean |
| C | no | the model's confidence, √(p(1−p)) | active inference with a uniform mixture |
| G | group key only | 30% pilot, per proxy-score stratum | ABae (scalars), ABae-GroupBy minimax (groups); run both as published and with the common estimator |
| QU | yes | none | survey design: equal allocation per domain, sampling proportional to weight |
| QC | yes | the model's confidence | the active-inference/OPAL rule with influences from the plan |
| **QP** (primary) | yes | 30% pilot, mean squared residual per cell (cheap-score bin × the query's group key, shrunk towards the bin mean) | proposed |
| D (Q5) | yes, plus the threshold | as QP; after the pilot, h of each movie set from its distance to the 60% band | proposed, decision-aware |

References without labels: the cheap model alone (threshold count and mean probability), each second model alone, and a
cascade that sends the least confident 5%, 10% or 30% of rows to the second model (Palimpzest/LOTUS style). With labels:
labels only (no model), and a global calibration applied to every group (phase 3's method), which assumes the bias does
not depend on the group.

**Without human labels** (the guidance's example, a mix of models only): U and QP with a second model's answers in place
of human labels (D1: Luna; D1-5k: Kev-9B, DeepSeek; D1-1.85k: Opus; D2-1k: Luna, DeepSeek; D2-546: Opus). Their intervals
are valid for the second model's full-table answer; the question is how often they cover the human truth.

**Three tiers** (secondary): cheap model on all rows, a second model on a query-aware sample, human labels on a smaller
query-aware sample, combined so the answer stays valid against human labels. Compared with human-only QP at equal total
cost for human-label prices of 10, 100 and 1,000 second-model calls.

Secondary checks: Kev-4B as the cheap model (D1-5k, D2-1k); pilot share 15% and 50%; per-publication shares (Q2b).

## 6 Metrics

All against the human labels of the whole population. 2,000 replications per (query, arm, budget), seeds fixed.

- Error of each output; loss = mean over outputs of (error / h_k)², i.e. MSE for scalars.
- **Labels needed**: the budget at which an arm reaches QP's loss at the primary budget, by log–log interpolation on the
  budget grid (slope −1 beyond it). Reported as a ratio to QP.
- Coverage of the 95% intervals per output; mean over outputs and the worst decile of outputs; share of outputs with at
  least two labels (no valid interval otherwise).
- Q5: movies whose verdict is wrong outside the ±5-point band; share of movies decided by intervals with a family-wise
  95% level.
- Without human labels: coverage of the second model's own full answer and of the human truth.
- Propagation (no replications): per query and model, the model-only error per output, split into the part explained by
  the overall false-positive and false-negative rates and the remainder, compared with what independent errors would
  give.
- Cost: labels; model calls at the measured prices.

## 7 Systematic bias and correlated errors: what is measured

- Bias by group: for each grouping (movie, critic, publication, genre, top critic, review-length tertile, cheap-score
  bin; MAMS predicate), the between-group spread of the mean residual beyond chance (random-effects variance and a
  permutation test).
- Confident errors: share of errors with p ≤ 0.1 or ≥ 0.9, overall and by group.
- Shared errors: P(second model wrong | cheap model wrong) against its marginal error rate; each second model's
  final-answer error per query.
- Global versus per-group correction: the per-group bias left after the global calibration, and its coverage.
- Within-group correlation of residuals (intraclass correlation), which limits any method that borrows strength across
  rows of a group.

## 8 Lines and expectations (to be frozen in `p16.txt`)

- **Coverage line**: mean coverage ≥ 93% and worst decile ≥ 90% (nominal 95%). An arm below it is not reliable for that
  query.
- **Primary, per query shape** (Q2, Q3, Q4, Q5, Q6; Q1 is the control): query-aware allocation pays off if QP (D on Q5)
  meets the coverage line and needs at most 1/1.5 of the labels of the best prior arm (U, C, G) that also meets it. If
  no prior arm meets the coverage line and QP does, the result is reported as a gain in validity.
- **Bias-aware**: the pilot pays off if QP needs at most 1/1.2 of QC's labels, or QC misses the coverage line where QP
  meets it.
- **Second model as the reference**: unreliable against human labels if, at the largest budget, its intervals cover the
  truth in fewer than 80% of replications for at least a third of the outputs.
- **Three tiers**: the second model is worth adding if it lowers the total cost of reaching QP's loss by at least 1.2× at
  a price ratio of 100.

Expectations, written now: no gain on Q1 (influence is uniform); on Q2 and Q3a QP ≈ G (ABae-GroupBy already allocates
per group) but ≥ 3× fewer labels than U and C; gains over G only where the query carries more than a group key: Q3b,
Q4 (above all Q4b), Q5 (most movies are far from 60%) and Q6 (relative requirement for rare predicates); U and C
under-cover small movies and rare predicates; the global calibration under-covers the genres with the largest bias; on
D1 QP ≈ QC (Jev's errors are mostly unconfident), on D2 QP better (confident, systematic errors on the service
predicates); a second model as the reference misses the human truth on D1 and on the D2 predicates where both models
misread the question; a second tier helps only on D2.

## 9 Steps, time and cost

Each step starts on the owner's word; after each, a report and the time of the next.

1. Write `p16.txt` (sections 5–8 in full: estimator, cells, shrinkage, seeds, disclosure of section 3). About 1 hour.
2. Scripts and offline tests: `p16_data.py` (populations and influences), `p16_designs.py` (arms), `p16_estimate.py`,
   `p16_run.py`, `p16_report.py`, `tests/test_p16.py` (unbiasedness and variance on synthetic tables with known truth and
   planted group bias; budgets; determinism). About 3 hours.
3. Rehearsal on a synthetic population with D1's and D2's structure and simulated labels, so no real result is seen;
   append the freeze block (checksums of scripts and cached inputs) to `p16.txt`. About 30 minutes.
4. Run: replay only, under an hour on this Mac, detached from the session. $0.
5. Report: README section "Phase 16", one figure per question. About 2 hours.

No paid call, no model server and no GPU job. OpenRouter has $0.03 left (checked 2026-10-30); nothing here needs it.

## 10 Rules fixed in advance

- Human labels are the truth; a second model is never used as truth except in the arms that test exactly that.
- The engine sees only structured columns, the cheap model's answers and the labels it buys; strata and influences may
  not use any label outside the drawn sample.
- Bugs found after the freeze are fixed, recorded as deviations, and every cell is re-run (cheap). Analyses added after
  seeing results are labelled post hoc.
- Timestamps from `date`; long jobs detached; no shell script edited while it runs; nothing committed or pushed without
  the owner's word.

## 11 Risks and known limits

- With few labels in a group, normal intervals under-cover; this is measured, not hidden, and reported by labels per
  output.
- Truth is the critic's verdict, which the snippet does not always reveal; this caps every model (phase 5) but does not
  affect validity.
- Join reuse on real data is mild (1–6×); heavy reuse is semi-synthetic (real labels and answers, simulated
  multiplicities) and labelled as such.
- 15,388 and 3,949 rows, not 100 million: budgets are compared as shares of the table; design-based results scale, but
  the fixed pilot share is a choice.
- Second-model answers exist only on subsets (5,001, 1,851, 1,000, 546 rows), so those analyses run on the subsets.

## 12 Decisions for the owner

1. Approve the direction and scope above (three questions, eight queries on two datasets, replay only, $0).
2. Location: phase 16 of `experiments/database` (this file, `p16.txt`, `p16_*.py`, `out/database/p16/`), recommended; or
   a new folder `experiments/query_allocation/`.
3. After the run, if the method holds: put influence, allocation and estimator into `src/kevdb` as a small module with a
   DuckDB demo on cached answers (recommended), or keep them in the experiment scripts.
