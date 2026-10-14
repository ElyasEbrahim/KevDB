# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
