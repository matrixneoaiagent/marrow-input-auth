"""One-class user model: robust Mahalanobis distance with a fixed threshold.

The enrolled user's *resting* sessions are used to learn a centre, a robust
scale per feature, and a shrunk covariance matrix.  A separate calibration
split (still owner-only) fixes the decision threshold *before* any impostor or
tired-owner data is scored.  The model never sees impostor data during
training or calibration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .features import FEATURE_NAMES, Window, _median, _quantile

PROFILE_SCHEMA = "marrow.profile/v1"


def _mad(values: Sequence[float], center: float | None = None) -> float:
    if center is None:
        center = _median(values)
    return _median([abs(v - center) for v in values])


def _clip(value: float, lo: float = -4.0, hi: float = 4.0) -> float:
    return max(lo, min(hi, value))


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse with partial pivoting.  Raises on singularity."""

    n = len(matrix)
    aug = [
        [float(matrix[i][j]) for j in range(n)]
        + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot = aug[pivot_row][col]
        if abs(pivot) < 1e-12:
            raise ValueError("singular covariance matrix")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        scale = 1.0 / aug[col][col]
        aug[col] = [v * scale for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


@dataclass
class Profile:
    features: list[str]
    center: list[float]
    scale: list[float]
    cov_inv: list[list[float]]
    shrinkage: float = 0.85
    threshold: float | None = None
    target_frr: float = 0.05
    calibration: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "schema": PROFILE_SCHEMA,
            "features": self.features,
            "center": self.center,
            "scale": self.scale,
            "cov_inv": self.cov_inv,
            "shrinkage": self.shrinkage,
            "threshold": self.threshold,
            "target_frr": self.target_frr,
            "calibration": self.calibration,
            "training": self.training,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Profile":
        if obj.get("schema") != PROFILE_SCHEMA:
            raise ValueError(f"not a Marrow profile (schema={obj.get('schema')!r})")
        return cls(
            features=list(obj["features"]),
            center=[float(v) for v in obj["center"]],
            scale=[float(v) for v in obj["scale"]],
            cov_inv=[[float(v) for v in row] for row in obj["cov_inv"]],
            shrinkage=float(obj.get("shrinkage", 0.85)),
            threshold=None if obj.get("threshold") is None else float(obj["threshold"]),
            target_frr=float(obj.get("target_frr", 0.05)),
            calibration=dict(obj.get("calibration", {})),
            training=dict(obj.get("training", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, features: dict[str, float | None]) -> tuple[float | None, dict]:
        """Return (distance, explanation).  Distance is None if too little data."""

        z: list[float] = []
        present: list[str] = []
        for name, center, scale in zip(self.features, self.center, self.scale):
            value = features.get(name)
            if value is None:
                z.append(0.0)
            else:
                z.append(_clip((float(value) - center) / scale))
                present.append(name)
        required = max(3, math.ceil(0.5 * len(self.features)))
        if len(present) < required:
            return None, {"reason": "insufficient features", "present": present}

        projected = [
            sum(self.cov_inv[i][j] * z[j] for j in range(len(z)))
            for i in range(len(z))
        ]
        squared = sum(zi * pi for zi, pi in zip(z, projected))
        distance = math.sqrt(max(0.0, squared))
        contributions = {
            name: (z[i] * projected[i] / squared if squared > 0 else 0.0)
            for i, name in enumerate(self.features)
        }
        top = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        return distance, {
            "present": present,
            "top_features": [(name, round(value, 4)) for name, value in top],
        }

    def accepts(self, distance: float | None) -> bool:
        if distance is None or self.threshold is None:
            return False
        return distance <= self.threshold


def fit_profile(
    windows: Iterable[Window],
    feature_names: Sequence[str] = FEATURE_NAMES,
    shrinkage: float = 0.85,
    min_coverage: float = 0.5,
) -> Profile:
    """Learn centre, scale and covariance from owner-only training windows."""

    rows = list(windows)
    if len(rows) < 5:
        raise ValueError(f"need at least 5 training windows, got {len(rows)}")

    kept: list[str] = []
    centers: list[float] = []
    scales: list[float] = []
    coverage: dict[str, float] = {}
    for name in feature_names:
        values = [row.features.get(name) for row in rows]
        present = [float(v) for v in values if v is not None]
        coverage[name] = len(present) / len(rows)
        if len(present) < max(3, min_coverage * len(rows)):
            continue
        center = _median(present)
        scale = 1.4826 * _mad(present, center)
        if scale < 1e-9:
            scale = float(
                math.sqrt(sum((v - center) ** 2 for v in present) / max(1, len(present) - 1))
            )
        if scale < 1e-9:
            continue  # constant feature: carries no identity information
        kept.append(name)
        centers.append(center)
        scales.append(scale)

    if len(kept) < 3:
        raise ValueError("fewer than 3 usable features in training data")

    z_rows: list[list[float]] = []
    for row in rows:
        z_row: list[float] = []
        for name, center, scale in zip(kept, centers, scales):
            value = row.features.get(name)
            z_row.append(0.0 if value is None else _clip((float(value) - center) / scale))
        z_rows.append(z_row)

    n = len(z_rows)
    dim = len(kept)
    cov = [[0.0] * dim for _ in range(dim)]
    for z_row in z_rows:
        for i in range(dim):
            for j in range(i, dim):
                cov[i][j] += z_row[i] * z_row[j]
    for i in range(dim):
        for j in range(i, dim):
            cov[i][j] /= max(1, n)
            cov[j][i] = cov[i][j]

    # Shrink toward the diagonal, then add a small ridge for numerical safety.
    mean_var = sum(cov[i][i] for i in range(dim)) / dim
    for i in range(dim):
        for j in range(dim):
            if i == j:
                cov[i][j] += 1e-3 * max(1.0, mean_var)
            else:
                cov[i][j] *= 1 - shrinkage
    try:
        cov_inv = _invert(cov)
    except ValueError:
        for i in range(dim):
            cov[i][i] += 0.1 * max(1.0, mean_var)
        cov_inv = _invert(cov)

    return Profile(
        features=kept,
        center=[round(v, 8) for v in centers],
        scale=[round(v, 8) for v in scales],
        cov_inv=[[round(v, 8) for v in row] for row in cov_inv],
        shrinkage=shrinkage,
        training={
            "n_windows": n,
            "n_recordings": len({row.recording_id for row in rows}),
            "coverage": {k: round(v, 3) for k, v in coverage.items()},
            "dropped_features": [name for name in feature_names if name not in kept],
        },
    )


def calibrate_profile(
    profile: Profile,
    windows: Iterable[Window],
    target_frr: float = 0.05,
) -> Profile:
    """Fix the threshold using owner-only calibration windows.

    Threshold = max(empirical (1-target) quantile, median + 3 robust sigma).
    No impostor data is consulted.  The threshold is frozen before test.
    """

    rows = list(windows)
    if not rows:
        raise ValueError("calibration needs at least one window")
    scores = [profile.score(row.features)[0] for row in rows]
    usable = [float(s) for s in scores if s is not None]
    if len(usable) < 3:
        raise ValueError("fewer than 3 scorable calibration windows")
    threshold = _quantile(usable, 1.0 - target_frr)
    profile.threshold = round(threshold, 8)
    profile.target_frr = target_frr
    profile.calibration = {
        "n_windows": len(usable),
        "target_frr": target_frr,
        "empirical_frr": round(sum(1 for s in usable if s > threshold) / len(usable), 4),
        "score_median": round(_median(usable), 4),
        "score_max": round(max(usable), 4),
        "threshold_rule": "empirical quantile at 1-target_frr on owner-only calibration windows",
    }
    return profile


def threshold_from_train(profile: Profile, windows: Iterable[Window], target_frr: float = 0.05) -> Profile:
    """Fallback for arbitrary recordings without a separate calibration split."""

    return calibrate_profile(profile, windows, target_frr=target_frr)
