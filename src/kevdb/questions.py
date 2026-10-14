"""The three question types of the decision-model API: noul (yes/no), choice (pick one) and score (rate on levels).

Fields map one-to-one onto the request body at https://docs.typesafe.ai/api; review_below / threshold
are this project's routing policy and are never sent to the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TYPES = ("noul", "choice", "score")
MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = (2, 10)


@dataclass(frozen=True)
class Question:
    type: str
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None
    # Rows whose confidence is below this value go to the human review queue; 0 disables review
    review_below: float = 0.0
    # noul only: probability >= threshold counts as True
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"unknown question type {self.type!r}, expected one of {TYPES}")
        if not self.instructions.strip():
            raise ValueError("instructions must not be empty")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice criteria must be a dict with 2-{MAX_CHOICE_OPTIONS} options")
        elif self.type == "score":
            lo, hi = SCORE_LEVELS
            if not isinstance(self.criteria, list) or not lo <= len(self.criteria) <= hi:
                raise ValueError(f"score criteria must be a list of {lo}-{hi} level descriptions, from low to high")
        elif self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("noul criteria may be omitted, or be {true: ..., false: ...}")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Question":
        return cls(**raw)

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = self.criteria
        return body
