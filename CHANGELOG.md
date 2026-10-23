# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Phase 10 of the database experiments, run and analysed: fine-tuning Kev-4B and Kev-0.8B on public, human-labelled
  record-matching pairs, leave one dataset out. Kev-4B reaches Jev's zero-shot level on 2 of 4 unseen datasets, Kev-0.8B
  on 1 of 4; results in `experiments/database/README.md`, protocol and execution record in
  `experiments/database/protocol/p10.txt`.
- `experiments/database/p10_eval.py` (serving, request sets, replay of earlier answers, checks, inventory),
  `p10_report.py` (temperature, lines, bootstrap), `p10_run.sh` (one command per model or phase, pre-flight checks,
  memory watchdog, resume), `p10_kev_train.py` (Kev's MPS shape padding applied during training) and `p10_memprobe.sh`
  (training memory probe), with offline tests.
- `kevdb.finetuned` and `scripts/serve_finetuned.sh`: serve a Kev run you fine-tuned yourself and use it as a backend. The
  cache name follows the weights, a server serving another run is refused, and a temperature fitted on your own labels is
  stored in the calibration table and applied in SQL. Documented in `docs/open-models.md`.
- Phase 11 of the database experiments, run and analysed: why fine-tuning did not help Kev-4B on dblp-acm. Rewording the
  product question changes nothing; another publication dataset asked with the same question, or three other domains
  with their own questions, brings Kev-4B to Jev's level there. Scripts `experiments/database/p11_*`, protocol and
  execution record in `experiments/database/protocol/p11.txt`.
- Phase 12 of the database experiments, run and analysed: how many of your own labels Kev-4B needs to reach Jev's
  zero-shot level on unseen records. None on product data like the public training data, about 100 for the
  publication question, 300 on all four datasets. Scripts `experiments/database/p12_*`, protocol and execution record in
  `experiments/database/protocol/p12.txt`.
- Phase 13 of the database experiments, run and analysed: are phase 10's close verdicts stable to the training seed? Three
  of four hold; Kev-4B on amazon-google turns on par with other seeds, and Kev-0.8B's forgetting on movie reviews does not
  recur. Scripts `experiments/database/p13_*`, protocol and execution record in `experiments/database/protocol/p13.txt`.
- Phase 14 of the database experiments, run and analysed: Kev-0.8B's drop on movie reviews belongs to phase 10's seed 1
  (it returns when that seed is retrained through the padding launcher), and the padding changes no result. Scripts
  `experiments/database/p14_*`, protocol and execution record in `experiments/database/protocol/p14.txt`.

### Changed

- `experiments/database/p10_train.sh` trains through `p10_kev_train.py`, which keeps a long training run's memory flat on
  Apple GPUs.

## [0.1.0] - 2026-10-14

First public release.

### Added

- `kevdb` package: clients for the decision-model request format (Jev API, local open models, OpenRouter LLMs),
  DuckDB SQL functions (`noul`, `noulc`, `choice`, `score`, `nouls`, `match`, `pick`), cascades, calibration tables
  and macros, a SQLite answer cache, and the `kevdb` batch-labelling CLI.
- Scripts to install and serve Kev, Clef-flash and Laya locally inside a macOS sandbox.
- Experiments: phases 0–9 of decision models inside DuckDB, entity matching, end-to-end table matching, text
  classification, query-plan selection and the first batch-labelling task, each with protocol, scripts and results.
- The protocol and execution plan for phase 10 (fine-tuning Kev on public matching data), not yet run.
