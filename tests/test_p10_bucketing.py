"""Phase 10's training launcher (experiments/database/p10_kev_train.py): it pads training rows on MPS to Kev's shape bucket exactly as Kev's own eval
path does, and changes nothing else. Kev itself is not installed in this environment, so a stand-in kev.model / kev.train is used; the padding tests need PyTorch and skip without it:
    open_models/kev/.venv/bin/python -m pytest -p no:cacheprovider tests/test_p10_bucketing.py"""

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(monkeypatch):
    torch = pytest.importorskip("torch")  # the repository's environment has no PyTorch; run these two with Kev's (open_models/kev/.venv)

    class DecisionModel:
        SHAPE_BUCKET = 64
        pad_id = 7

        def __init__(self, device, training):
            self.device, self.training = device, training

        def _pad_rows(self, rows):  # Kev's padding to the longest row (the eval-only bucketing left out, as in training)
            L = max(len(ids) for ids, _ in rows)
            ids = torch.full((len(rows), L), self.pad_id)
            pos = torch.zeros((len(rows), L), dtype=torch.long)
            att = torch.zeros((len(rows), L), dtype=torch.long)
            for i, (rid, rpos) in enumerate(rows):
                ids[i, : len(rid)] = torch.tensor(rid)
                pos[i, : len(rpos)] = torch.tensor(rpos)
                att[i, : len(rid)] = 1
            return ids, pos, att

    kev, model = types.ModuleType("kev"), types.ModuleType("kev.model")
    model.DecisionModel, kev.model = DecisionModel, model
    for name, module in (("kev", kev), ("kev.model", model)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("p10_kev_train", ROOT / "experiments" / "database" / "p10_kev_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return DecisionModel, module


ROWS = [([1, 2, 3], [0, 1, 2]), ([4] * 70, list(range(70)))]


def test_training_rows_on_mps_are_padded_to_the_bucket_with_kevs_pad_values(launcher):
    DecisionModel, _ = launcher
    ids, pos, att = DecisionModel("mps", True)._pad_rows(ROWS)
    assert ids.shape == pos.shape == att.shape == (2, 128)
    assert ids[0, :3].tolist() == [1, 2, 3] and set(ids[0, 3:].tolist()) == {7} and set(ids[1, 70:].tolist()) == {7}
    assert att[0].sum() == 3 and att[1].sum() == 70 and pos[1, 70:].abs().sum() == 0
    assert pos[1, :70].tolist() == list(range(70))


def test_nothing_changes_off_mps_or_outside_training(launcher):
    DecisionModel, module = launcher
    for device, training in (("cpu", True), ("mps", False), ("cuda", True)):
        assert DecisionModel(device, training)._pad_rows(ROWS)[0].shape == (2, 70)
    assert DecisionModel("mps", True)._pad_rows([([1] * 64, list(range(64)))])[0].shape == (1, 64)  # already a multiple: no padding


def test_the_training_wrapper_trains_through_the_launcher_and_the_watchdogs_still_find_it():
    text = (ROOT / "experiments" / "database" / "p10_train.sh").read_text()
    call = [line for line in text.splitlines() if line.startswith("sandbox-exec")]
    assert len(call) == 1 and 'experiments/database/p10_kev_train.py" kev.train ' in call[0] and "-m kev.train" not in call[0]
    # p10_run.sh's watchdog and pre-flight check, and the unattended chain's guard, look for this name in the process's command line
    run = (ROOT / "experiments" / "database" / "p10_run.sh").read_text()
    assert "pgrep -P \"$job\" -f 'kev\\.train'" in run and "pgrep -f 'kev\\.(serve|train)'" in run
    assert re.search(r"kev\.train", call[0].split("p10_kev_train.py")[1])
