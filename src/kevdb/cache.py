"""SQLite answer cache: each (model, state, questions) is paid for once, and an interrupted task resumes where it stopped."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


def cache_key(model: str, state: Any, questions: dict[str, Any]) -> str:
    blob = json.dumps([model, state, questions], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    def __init__(self, path: str | Path = ".jevcache.sqlite") -> None:
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, response TEXT NOT NULL)")

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT response FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, response: dict[str, Any]) -> None:
        self._db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, json.dumps(response, ensure_ascii=False)))
        self._db.commit()

    def close(self) -> None:
        self._db.close()
