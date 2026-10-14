"""Read and write .jsonl / .json / .csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def read_rows(path: str | Path, records_path: str | None = None) -> list[Row]:
    """records_path: the key of the record list inside a .json file; dotted paths such as "a.b" work, e.g. "items"."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in (records_path or "").split("."):
            if key:
                data = data[key]
        if not isinstance(data, list):
            raise ValueError(f"no record list found in {path}; set records_path in the task configuration")
        return data
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported input format: {suffix}")


def write_jsonl(path: str | Path, rows: list[Row]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[Row]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
