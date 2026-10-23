"""A Kev checkpoint you fine-tuned yourself, as a KevDB backend. Phase 10 (experiments/database/README.md) showed that a fine-tune on public
matching labels brings Kev-4B to Jev's zero-shot level on some product datasets; this module lets such a run answer SQL predicates.

    ./scripts/serve_finetuned.sh open_models/p10/kev-4b/all             # serve the run directory kev.train wrote, on port 8016

    from kevdb.finetuned import finetuned_backend
    backend = finetuned_backend("open_models/p10/kev-4b/all")           # checks first that the server really serves this run

Three things differ from the released checkpoints (open_models.py):
- identity: a local server's request body only says kev-latest, and kev.serve silently serves another run (runs/smoke) when the one asked
  for has no head.pt. The cache name therefore carries a digest of the weights, so a retrained run never inherits an earlier run's answers,
  and check_served() compares the run the server reports with this one;
- temperature: a fine-tuned run is served at temperature 1.0 (kev.train does not carry the released checkpoint's over). Fit one on held-out
  human labels with calibrate(); it is stored in the calibration table as a = 0, b = 1/T, and <prefix>_noul_cal applies it in SQL. Fit only
  on human labels: open models must not be calibrated on Jev's outputs (TypeSafe customer agreement, section 2.3(b)). The fit clips
  probabilities as the SQL macro does (calibration.CLIP, 0.001), so on a run whose answers mostly lie beyond that clip its temperature
  differs from the experiments' (fitted on 0.00005): 1.59 against 1.82 on phase 13's Kev-0.8B walmart-amazon checkpoint;
- port: 8016 by default, apart from the released models (8009-8014) and the experiments' evaluation server (8015).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import calibration
from .client import PROVIDERS, Provider
from .engine import Backend
from .open_models import OpenModel, OpenModelClient

PORT = 8016
WEIGHTS = ("head.pt", "adapter_model.safetensors")
# Kev's own temperature fit (kev/metrics.py, used for the released checkpoints and in phases 10-12): the minimum mean negative log
# likelihood over a 121-point log grid from 0.25 to 4. The grid keeps T finite when the labels are separated perfectly
GRID = np.exp(np.linspace(np.log(0.25), np.log(4), 121))


@dataclass(frozen=True)
class FineTuned(OpenModel):
    run: str = ""  # the run directory, absolute, as the server reports it
    port: int = PORT


def weights_digest(run: str | Path) -> str:
    """sha256 over head.pt, then adapter_model.safetensors; the first 16 hex digits (as the experiments' cache names from phase 10 on)."""
    h = hashlib.sha256()
    for name in WEIGHTS:
        with (Path(run) / name).open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()[:16]


def finetuned(run: str | Path, port: int = PORT) -> FineTuned:
    """The backend spec of a run directory served on localhost:<port>. Raises if the run is incomplete (the server would serve another one)."""
    path = Path(run).resolve()
    missing = [n for n in (*WEIGHTS, "training_config.json") if not (path / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is not a complete Kev run (missing {', '.join(missing)}); kev.serve would silently serve another run")
    provider = f"finetuned:{port}"
    PROVIDERS[provider] = Provider(f"http://127.0.0.1:{port}/v1", None, "kev-latest")
    return FineTuned(provider, "kev-latest", f"finetuned:{path.name}@{weights_digest(path)}", run=str(path), port=port)


def check_served(spec: FineTuned, timeout: float = 5.0) -> dict[str, Any]:
    """The server's model card, after checking that it serves this very run at temperature 1.0; raises RuntimeError otherwise."""
    with urllib.request.urlopen(f"http://127.0.0.1:{spec.port}/v1/models", timeout=timeout) as r:
        card = json.load(r)["models"][0]
    if card.get("run") != spec.run:
        raise RuntimeError(f"the server on port {spec.port} serves {card.get('run')!r}, not {spec.run!r}")
    if abs(float(card.get("temperature", -1)) - 1.0) > 1e-9:
        raise RuntimeError(f"the server applies temperature {card.get('temperature')}; a fine-tuned run is served at 1.0 and calibrated in SQL "
                           "(unset KEV_TEMPERATURE)")
    return card


def finetuned_backend(run: str | Path, port: int = PORT, name: str | None = None, **kwargs: Any) -> Backend:
    """A Backend for a fine-tuned run whose server is already running (scripts/serve_finetuned.sh); checks the server first."""
    spec = finetuned(run, port)
    check_served(spec)
    return Backend(name or f"finetuned-{Path(spec.run).name}", lambda: OpenModelClient(spec), **kwargs)


def fit_temperature(p: Any, y: Any) -> float:
    """Kev's temperature fit on human labels (p: the run's probability of yes as served, y: 0/1). Rows without an answer (NaN) are left out.
    Probabilities are clipped as the SQL macro clips them (calibration.CLIP), so the fitted T is the one the macro applies."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    keep = ~np.isnan(p)
    q = np.clip(p[keep], calibration.CLIP, 1 - calibration.CLIP)
    s = np.log(q / (1 - q))[None, :] / GRID[:, None]
    return float(GRID[int(np.argmin((np.logaddexp(0, s) - y[keep] * s).mean(1)))])


def calibrate(con: Any, model: str, instructions: str, criteria: str, scope: str, p: Any, y: Any) -> float:
    """Fit the temperature of one predicate on human-labelled rows and store it in the calibration table (a = 0, b = 1/T). model is
    backend.model. Call calibration.install(con, prefix, model) first. Returns T."""
    T = fit_temperature(p, y)
    keep = ~np.isnan(np.asarray(p, float))
    y = np.asarray(y)[keep]
    calibration.store(con, model, instructions, criteria, scope, 0.0, 1.0 / T, int(keep.sum()), int((y == 1).sum()))
    return T
