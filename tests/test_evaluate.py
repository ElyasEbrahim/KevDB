import pytest

from kevdb import Question, Task
from kevdb.evaluate import evaluate, format_report, sample_for_gold

TASK = Task(
    name="t",
    input="unused",
    state_fields=["text"],
    questions={
        "kind": Question("choice", "What kind?", {"a": "A", "b": "B"}),
        "depth": Question("score", "How deep?", ["none", "some", "lots"]),
        "urgent": Question("noul", "Is it urgent?"),
    },
)


def pred(i, kind, conf, depth=1.0, p=0.9):
    return {"id": i, "kind": kind, "kind_conf": conf, "depth": depth, "depth_conf": 0.8, "urgent": p >= 0.5, "urgent_p": p}


def test_sample_is_reproducible_and_tops_up_rare_classes():
    rows = [pred(i, "a", 0.9) for i in range(95)] + [pred(i, "b", 0.9) for i in range(95, 100)]
    picked = sample_for_gold(rows, "id", "kind", n=20, min_per_class=4, seed=1)
    assert picked == sample_for_gold(rows, "id", "kind", n=20, min_per_class=4, seed=1)
    assert sum(r["uniform"] for r in picked) == 20
    assert sum(r["kind"] == "b" for r in picked) >= 4
    assert len({r["id"] for r in picked}) == len(picked)


def test_evaluate_accuracy_calibration_and_thresholds():
    preds = [pred(0, "a", 0.95), pred(1, "a", 0.95), pred(2, "b", 0.55), pred(3, "a", 0.55, depth=2.4, p=0.3)]
    gold = [
        {"id": 0, "kind": "a", "depth": 1, "urgent": True},
        {"id": 1, "kind": "a", "depth": 1, "urgent": True},
        {"id": 2, "kind": "a", "depth": 1, "urgent": True},
        {"id": 3, "kind": "a", "depth": 0, "urgent": True},
        # Rows added for rare classes count only towards per-class precision, not overall accuracy
        {"id": 2, "kind": "b", "depth": 1, "urgent": True, "uniform": False},
    ]
    report = evaluate(TASK, preds, gold)

    kind = report["kind"]
    assert kind["n"] == 4 and kind["accuracy"] == kind["accuracy_strict"] == 0.75
    assert [(b["range"], b["n"], b["accuracy"]) for b in kind["calibration"]] == [("0.0-0.6", 2, 0.5), ("0.9-1.0", 2, 1.0)]
    assert kind["thresholds"][0] == {"review_below": 0.5, "auto": 1.0, "auto_accuracy": 0.75}
    assert kind["thresholds"][-1] == {"review_below": 0.9, "auto": 0.5, "auto_accuracy": 1.0}
    assert kind["per_class"]["b"] == {"precision": 0.5, "predicted": 2, "recall": None, "actual": 0}
    assert kind["confusions"] == [(("a", "b"), 1)]

    assert report["depth"]["accuracy"] == 0.75 and report["depth"]["within_1"] == 0.75
    assert report["depth"]["mae"] == pytest.approx(0.6)
    assert report["urgent"]["accuracy"] == 0.75
    assert "[kind]" in format_report(report)


def test_alt_labels_count_as_correct_and_unlabeled_questions_are_skipped():
    preds = [pred(0, "b", 0.9), pred(1, "b", 0.9)]
    gold = [{"id": 0, "kind": "a", "kind_alt": ["b"]}, {"id": 1, "kind": "a", "urgent": False}]
    report = evaluate(TASK, preds, gold)
    assert report["kind"]["accuracy"] == 0.5 and report["kind"]["accuracy_strict"] == 0.0
    assert report["kind"]["per_class"]["b"]["precision"] == 0.5
    assert report["kind"]["confusions"] == [(("a", "b"), 1)]
    assert report["urgent"]["n"] == 1 and "depth" not in report
