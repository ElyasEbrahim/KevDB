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
