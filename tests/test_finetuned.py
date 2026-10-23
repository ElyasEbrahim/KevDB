"""A fine-tuned Kev run as a backend: its cache name follows its weights, an incomplete run or a server serving another run is refused, and the
temperature fitted on labels is the one the SQL macro applies. Calls no model."""

import io
import json

import duckdb
import numpy as np
import pytest

from kevdb import calibration
from kevdb import finetuned as F


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "kev-4b" / "mine"
    path.mkdir(parents=True)
    (path / "head.pt").write_bytes(b"head")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "training_config.json").write_text(json.dumps({"args": {"base": "Qwen/Qwen3.5-4B-Base", "init_from": "jaredpalmer/kev-4b@v1.0"}}))
    return path


def test_the_cache_name_follows_the_weights(run):
    spec = F.finetuned(run, port=8017)
    assert spec.version == f"finetuned:mine@{F.weights_digest(run)}" and spec.run == str(run.resolve()) and spec.port == 8017
    assert F.PROVIDERS[spec.provider].base_url == "http://127.0.0.1:8017/v1" and spec.model == "kev-latest"
    (run / "adapter_model.safetensors").write_bytes(b"retrained")
    assert F.finetuned(run, port=8017).version != spec.version


def test_an_incomplete_run_is_refused(run):
    (run / "head.pt").unlink()
    with pytest.raises(FileNotFoundError, match="head.pt"):
        F.finetuned(run)


def serving(monkeypatch, card):
    monkeypatch.setattr(F.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(json.dumps({"models": [card]}).encode()))


def test_a_server_serving_another_run_or_temperature_is_refused(run, monkeypatch):
    spec = F.finetuned(run)
    serving(monkeypatch, {"run": "runs/smoke", "temperature": 1.0})
    with pytest.raises(RuntimeError, match="runs/smoke"):
        F.check_served(spec)
    serving(monkeypatch, {"run": spec.run, "temperature": 2.4})
    with pytest.raises(RuntimeError, match="temperature"):
        F.check_served(spec)
    serving(monkeypatch, {"run": spec.run, "temperature": 1.0})
    assert F.check_served(spec)["run"] == spec.run


def test_the_fitted_temperature_is_recovered_and_applied_by_the_sql_macro():
    rng = np.random.default_rng(0)
    z = rng.logistic(size=40_000)  # few rows beyond the macro's clip (calibration.CLIP), which would bias the fit
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z / 2.0))).astype(int)  # the labels follow temperature 2
    p = 1 / (1 + np.exp(-z))
    assert abs(F.fit_temperature(p, y) - 2.0) < 0.1
    con = duckdb.connect()
    con.execute("CREATE MACRO t_noul(state, instructions) AS 0.5")
    con.execute("CREATE MACRO t_noulc(state, instructions, criteria) AS 0.5")
    calibration.install(con, "t", "finetuned:mine@0")
    T = F.calibrate(con, "finetuned:mine@0", "Same product?", "", "pairs", np.append(p[:4000], np.nan), np.append(y[:4000], 1))
    labels, b = con.execute("SELECT labels, b FROM semantic_calibration").fetchall()[0]
    assert labels == 4000 and b == pytest.approx(1 / T)
    for x in (0.01, 0.3, 0.9, 0.999):
        got = con.execute("SELECT calibrated(?, 'finetuned:mine@0', predicate_key('Same product?', ''), 'pairs')", [x]).fetchall()[0][0]
        assert got == pytest.approx(1 / (1 + np.exp(-np.log(x / (1 - x)) / T)))
