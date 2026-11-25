"""Small statistics helpers used in reports (no third-party dependencies)."""

from __future__ import annotations

import math
from typing import Iterable


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""

    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(successes: int, total: int) -> dict:
    low, high = wilson(successes, total)
    return {
        "n": total,
        "count": successes,
        "rate": round(successes / total, 4) if total else None,
        "ci95": [round(low, 4), round(high, 4)],
    }


def median(values: Iterable[float]) -> float | None:
    xs = sorted(values)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2
