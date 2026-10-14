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

## Troubleshooting

- *"port … is in use"*: another server (or another model) holds the port; stop it first. Each model has its own port so a
  request can never reach the wrong model.
- *Connection refused from `kevdb`*: start the server first; local backends do not start servers themselves.
- *Memory pressure with a large prefix cache*: lower `KEV_PREFIX_CACHE`; read the process's real memory with
  `footprint -f bytes <pid>` (`ps` does not include GPU memory under MLX).
