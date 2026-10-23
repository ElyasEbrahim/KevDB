# Running open decision models locally

KevDB talks to every backend through the same request format: a *state* plus typed questions, answered with
probabilities. The open models below accept that format and run on an Apple Silicon Mac. This page covers installing,
serving and choosing them.

## Supported models

| Name in `kevdb` | Model | Pinned version | Runtime | Port | Size on disk | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `kev-4b` | Kev-4B: adapter and decision head on Qwen3.5-4B | code `kev-1.0`, weights `jaredpalmer/kev-4b@v1.0` | MLX, bfloat16 | 8009 | part of the 49 GB setup | recommended local backend |
| `kev-9b` | Kev-9B: the same approach on Qwen3.5-9B | `jaredpalmer/kev-9b@v1.0` | MLX | 8010 | part of the setup | tested |
| `kev-0.8b` | Kev-0.8B: the same approach on Qwen3.5-0.8B | `jaredpalmer/kev-0.8b@v1.0` | MLX | 8014 | 1.84 GB extra | fastest; needs calibration |
| `clef-flash` | Cloudflare's Clef-flash, 9.4B on Qwen3.5-9B | `Cloudflare/clef-flash@8b2e5fd1` | PyTorch on the Mac GPU | 8012 | part of the setup | slow on a Mac |
| `laya`, `laya-multilingual` | Laya encoders (ModernBERT-large, mmBERT-base) | `laya` 0.4.1 | Laya's own server | 8011, 8013 | about 2 GB extra | dropped after phase 6 |
| `clef@cloudflare`, `clef-flash@cloudflare` | Clef hosted on Cloudflare Workers AI | month-stamped | remote, paid | | | supported, not used in the experiments |

The registry lives in [src/kevdb/open_models.py](../src/kevdb/open_models.py). The cache key of every answer includes the
pinned weight version, so answers from different weights never mix.

## Install

```bash
./scripts/open_models.sh setup            # Kev-4B, Kev-9B and Clef-flash, about 49 GB
./scripts/open_models.sh setup kev-0.8b   # add Kev-0.8B (1.84 GB; run the plain setup first)
./scripts/open_models.sh setup laya       # add Laya (about 2 GB; only to reproduce phase 6)
./scripts/open_models.sh size             # disk usage of each part
```

Everything goes into `open_models/` at the repository root (ignored by git): the `uv` package manager, the Python it
downloads, one virtual environment per model, the models' code and weights, and every cache. Nothing is installed
system-wide and no shell configuration changes. `rm -rf open_models` removes all of it.

Downloads resume after interruptions and large files are checked against their SHA-256. Versions are pinned: Kev's code
to the release tag `kev-1.0` and its weights to the Hub tag `v1.0`; Laya to 0.4.1 with
[scripts/requirements-laya.txt](../scripts/requirements-laya.txt); Clef-flash's weights and loading code to one commit with
[scripts/requirements-clef.txt](../scripts/requirements-clef.txt).

## Serve

```bash
./scripts/open_models.sh serve kev-4b     # Ctrl-C stops it
```

Servers listen on 127.0.0.1 only and run inside the macOS sandbox with every outgoing connection blocked except loopback,
since the models' code and dependencies are third-party. Weights are loaded offline, so a server can never silently
switch to a newer version. `OPEN_MODELS_NO_SANDBOX=1` turns the sandbox off.

Clef-flash ships no server of its own; [scripts/clef_server.py](../scripts/clef_server.py) wraps its bundled loading code
in the same `/v1/systemone` interface, changing no step of loading, encoding or probability computation.

### Prefix cache (Kev)

Kev's server can keep the text it has read and reuse it for the next question about the same state. The factory setting
keeps 4 texts, which does nothing for table scans. Open it up with environment variables:

```bash
KEV_PREFIX_CACHE=1900 KEV_PREFIX_MAX_TOKENS=4000000 ./scripts/open_models.sh serve kev-0.8b
```

Each cached text costs a fixed 56.9 MB (Kev-4B) or 21.7 MB (Kev-0.8B) plus 31.1 KB or 11.7 KB per token, so measure your
own machine's budget with `experiments/database/p7_capacity.py` before opening it wide. On a 64 GB M3 Max the healthy
limits were about 600 entries for Kev-4B and 1,900 for Kev-0.8B.

## Use from Python and SQL

```python
import duckdb
from kevdb import open_backend, register, calibration

con = duckdb.connect()
kev = open_backend("kev-4b")
register(con, "kev", kev)                       # kev_noul, kev_noulc, kev_choice, kev_score, kev_nouls, kev_match, kev_pick
calibration.install(con, "kev", kev.model)      # kev_noul_cal, kev_noulc_cal, calibrated(...)
```

See [sql-functions.md](sql-functions.md) for the functions and [batch-pipeline.md](batch-pipeline.md) for the CLI.

### A checkpoint you fine-tuned

A run directory written by `kev.train` (for example phase 10's `open_models/p10/kev-4b/all`) has its own server script and is
registered like any other backend:

```bash
./scripts/serve_finetuned.sh open_models/p10/kev-4b/all      # port 8016; a second run can listen on another port
```

```python
from kevdb.finetuned import calibrate, finetuned_backend

mine = finetuned_backend("open_models/p10/kev-4b/all")       # refuses a server that serves another run
register(con, "mine", mine)
calibration.install(con, "mine", mine.model)
# p: the run's answers to 100-200 held-out, human-labelled rows of this predicate; y: their labels.
# instructions and criteria are the same strings you pass to mine_noulc
calibrate(con, mine.model, instructions, criteria, "pairs", p, y)
```

It differs from the released models in three ways:

- The cache name carries a digest of the weights, so a retrained run never reuses an earlier run's answers.
- `kev.serve` silently serves another run when the one asked for has no `head.pt`, which is why `finetuned_backend` checks the
  run the server reports.
- A fine-tuned run is served at temperature 1.0. `calibrate` fits its temperature on held-out human labels (Kev's own fit, as in
  phases 10–12), and `mine_noulc_cal` applies it.

The SQL macro clips probabilities to [0.001, 0.999] before rescaling them, so `calibrate` fits on the same clip. A fine-tuned
matching model puts most answers beyond it, so the temperature can differ from the one the experiments report, which were
fitted on a clip of 0.00005. On phase 13's Kev-0.8B walmart-amazon checkpoint (seed 2), 81% of the pool answers lie beyond
the clip, and the temperature is 1.59 here against 1.82 in the experiment. Both lower the Brier score. On that checkpoint
served by `serve_finetuned.sh`, the SQL answers to 300 test pairs were identical to the experiment's, and a server serving
another run was refused (checked on 2026-10-23).

To train on a Mac, run `experiments/database/p10_kev_train.py kev.train …` instead of `python -m kev.train`. It applies phase 10's
fix that keeps memory flat. `experiments/database/p12_train.sh` follows the author's recipe for your own data.

## Which model to use

Measured on a MacBook Pro, M3 Max, 64 GB ([phases 6 and 7](../experiments/database/README.md)):

| | Quality vs Jev, raw | After calibration with 200 labels | Throughput (share of the Jev API) | Best for |
| --- | --- | --- | --- | --- |
| Kev-4B | on par on movie reviews; slightly behind on restaurant aspects; 4 points behind on product matching | within 1.4 points | 12–25% | the default local backend; long documents asked many questions |
| Kev-9B | on par on movie reviews and product matching; clearly behind on restaurant aspects | within 1.4 points | 7–15% | not better than Kev-4B overall |
| Kev-0.8B | far behind on matching (calls most pairs a match) | 91–98% on matching | about 4× Kev-4B | speed, but only with calibration |
| Clef-flash | at the bar on restaurants and products | within 1.4 points | 4–5% | NVIDIA servers, not Macs |

Practical guidance from the experiments:

- **Calibrate before trusting probabilities.** The open models rank well (AUC within 0.002–0.030 of Jev) but their
  probabilities sit on the wrong scale. A few hundred human labels per predicate and the calibration table fix that;
  never calibrate on another model's outputs.
- **Use the prefix cache for long texts.** Asking the same long documents several questions is 43× faster on Kev-4B and
  23× faster on Kev-0.8B once cached, with identical answers. On short rows the gain is under 2×.
- **For semantic joins, use `kev_match` plus a one-to-one constraint in SQL** (keep each left record's best candidate at
  ≥ 0.5). Do not use `kev_pick` on Kev-4B: its multiple choice favours later options.
- **Verify packing per predicate.** Packing several questions into one request changes answers for some predicates.
- **Fine-tune Kev-4B for product matching if you have public or own labels** ([phase 10](../experiments/database/README.md#phase-10-does-fine-tuning-on-public-matching-data-bring-kev-to-jevs-zero-shot-level)).
  6,000 human-labelled pairs from three other product and publication datasets brought it to Jev's zero-shot level on two
  of four unseen datasets, mostly by removing false matches, without forgetting its other tasks. Trained on product pairs
  only, it does not help for a matching task asked with a question that never appeared in training; mixing in labelled
  pairs of other kinds with their own questions, or a dataset asked with the same question, does
  ([phase 11](../experiments/database/README.md#phase-11-why-did-fine-tuning-not-help-kev-4b-on-dblp-acm)).
  Kev-0.8B's slight forgetting in phase 10 belongs to its seed: other seeds do not show it, the same seed does again
  ([phase 14](../experiments/database/README.md#phase-14-is-kev-08bs-drop-on-movie-reviews-a-property-of-seed-1-or-of-training-without-padding)).
  Check the tasks you care about after training. When a fine-tuned checkpoint lands near your
  bar, train another seed before deciding ([phase 13](../experiments/database/README.md#phase-13-are-phase-10s-close-verdicts-stable-when-only-the-training-seed-changes)).
- **With labels of your own data, about 300 are enough** ([phase 12](../experiments/database/README.md#phase-12-how-many-of-your-own-labels-does-kev-4b-need)).
  Fine-tuned on 240 of them, with 60 kept aside for the temperature, Kev-4B is on par with Jev on all four matching
  datasets, on records none of the labels contains. That takes 15–20 minutes of training on a Mac.

## Troubleshooting

- *"port … is in use"*: another server (or another model) holds the port; stop it first. Each model has its own port so a
  request can never reach the wrong model.
- *Connection refused from `kevdb`*: start the server first; local backends do not start servers themselves.
- *Memory pressure with a large prefix cache*: lower `KEV_PREFIX_CACHE`; read the process's real memory with
  `footprint -f bytes <pid>` (`ps` does not include GPU memory under MLX).
- *`kev.train`'s memory keeps growing on a Mac*: every new input length gets its own compiled GPU kernels. Train through
  `experiments/database/p10_kev_train.py kev.train ...`, which applies Kev's own padding to multiples of 64 tokens during
  training too; on phase 10's runs memory then stayed flat (Kev-4B 28–32 GiB instead of passing 41 GiB).
