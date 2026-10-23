"""Small, JSON-compatible contracts independent of model and web frameworks."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1"
TERMINAL = frozenset({"completed", "partial", "failed", "cancelled", "timed_out"})
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("IDs must contain 1–64 letters, numbers, dots, dashes or underscores.")
    return value


def finite(value: Any, minimum: float, maximum: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    if not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return float(value)


def targets(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 8:
        raise ValueError("Provide between one and eight object phrases.")
    result = []
    for phrase in value:
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 120:
            raise ValueError("Each object phrase must contain 1–120 characters.")
        if any(ord(c) < 32 or ord(c) == 127 for c in phrase):
            raise ValueError("Object phrases cannot contain control characters.")
        phrase = phrase.strip().strip(".").strip()
        if not phrase:
            raise ValueError("An object phrase cannot be just punctuation.")
        if phrase not in result:
            result.append(phrase)
    if sum(map(len, result)) > 400:
        raise ValueError("Combined object phrases must be at most 400 characters.")
    return tuple(result)


@dataclass(frozen=True)
class Search:
    targets: tuple[str, ...]
    source_ids: tuple[str, ...]
    duration_seconds: float | None = 10
    timeout_seconds: float = 180

    @classmethod
    def parse(cls, data: Any, available: set[str]) -> Search:
        if not isinstance(data, dict) or set(data) - {
            "targets",
            "source_ids",
            "duration_seconds",
            "timeout_seconds",
        }:
            raise ValueError("Unsupported search fields.")
        phrases = targets(data.get("targets"))
        ids = data.get("source_ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 8:
            raise ValueError("Select between one and eight sources.")
        ids = tuple(identifier(v) for v in ids)
        if len(set(ids)) != len(ids) or not set(ids) <= available:
            raise ValueError("Source IDs must be unique and configured.")
        raw_duration = data.get("duration_seconds", 10)
        duration = (
            None if raw_duration is None else finite(raw_duration, 1, 120, "duration_seconds")
        )
        timeout = finite(data.get("timeout_seconds", 180), 5, 600, "timeout_seconds")
        if duration is not None and timeout < duration:
            raise ValueError("The timeout must include the capture duration.")
        return cls(phrases, ids, duration, timeout)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label or len(self.label) > 200:
            raise ValueError("Invalid detection label.")
        finite(self.score, 0, 1, "score")
        if len(self.box) != 4:
            raise ValueError("A box needs four normalized coordinates.")
        for value in self.box:
            finite(value, 0, 1, "box")
        x1, y1, x2, y2 = self.box
        if x1 >= x2 or y1 >= y2:
            raise ValueError("A box must have positive area.")

    def json(self) -> dict:
        return asdict(self)
