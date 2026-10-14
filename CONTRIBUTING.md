# Contributing to KevDB

Thanks for your interest. Bug reports, reproductions on other hardware and new experiments are all welcome.

## Development setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,experiments]"
.venv/bin/python -m pytest -q
```

The test suite is offline: it never calls a model, a local model server or a paid API. Keep it that way. New code
that talks to a backend should be tested against the mock transport in `src/kevdb/mock.py` or an `httpx.MockTransport`.

To run the open models locally, see [docs/open-models.md](docs/open-models.md).

## Ground rules for experiments

The experiments in this repository follow a few rules. Pull requests that touch `experiments/` should keep them.

1. **Write the criteria first.** Each phase states its question, metrics, thresholds and stopping rules in
   `experiments/database/protocol/` before any data is collected. Frozen protocols are never edited; corrections and
   execution notes are appended at the end with a date.
2. **Label post-hoc work as post-hoc.** Anything added after looking at the results is marked as such.
3. **Do not tune on the test data.** Questions are first drafts and are not edited against the labels; thresholds are
   chosen on validation splits only.
4. **No Jev outputs for training.** Never use Jev's answers to train, fine-tune, calibrate or pick thresholds for any
   other model (TypeSafe customer agreement, section 2.3(b)). Calibration uses human labels only.
5. **Report what happened.** Negative results are results. Failed requests, deviations and costs go into the README of
   the experiment.
6. **Say what costs money.** A script that calls a paid API must say so in its docstring, and cached answers must make
   a re-run free.

## Code style

- Python 3.12+, type hints on public functions, `from __future__ import annotations` where it helps.
- Comments explain *why*, not what. Write them in English.
- Keep paths relative to the file (`Path(__file__)`), so scripts work from any working directory.
- Do not commit datasets, model weights, caches, `.env` or anything under `out/`.

## Pull requests

1. Open an issue first for anything larger than a small fix.
2. Keep each pull request focused on one change and include a test when behaviour changes.
3. Make sure `pytest` passes locally; CI runs the same suite.
4. For new experiment results, include the protocol, the scripts and the README section in the same pull request.

