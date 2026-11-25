from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from marrow.features import extract_windows
from marrow.model import Profile, calibrate_profile, fit_profile
from marrow.schema import load_recording

ROOT = Path(__file__).resolve().parents[1]
OWNER_DIR = ROOT / "data" / "fake" / "mouse_keyboard" / "owner_rested"


def owner_windows():
    windows = []
    for path in sorted(OWNER_DIR.glob("session-*.jsonl"))[:8]:
        windows.extend(extract_windows(load_recording(path)))
    return windows


def test_fit_calibrate_save_roundtrip(tmp_path: Path) -> None:
    windows = owner_windows()
    train, calib = windows[:20], windows[20:40]
    profile = fit_profile(train)
    assert len(profile.features) >= 6
    assert profile.threshold is None
    calibrate_profile(profile, calib, target_frr=0.15)
    assert profile.threshold is not None
    assert profile.calibration["n_windows"] == len(calib)
    assert profile.calibration["empirical_frr"] <= 0.15 + 1 / len(calib)

    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = Profile.load(path)
    assert loaded.features == profile.features
    assert loaded.threshold == pytest.approx(profile.threshold)
    assert loaded.center == profile.center


def test_owner_window_scores_lower_than_extreme_window() -> None:
    windows = owner_windows()
    profile = fit_profile(windows[:20])
    calibrate_profile(profile, windows[20:40], target_frr=0.15)
    base = windows[0]
    normal, _ = profile.score(base.features)
    assert normal is not None

    extreme_features = dict(base.features)
    for name in extreme_features:
        if extreme_features[name] is None:
            continue
        if name in {
            "path_efficiency",
            "preclick_slowdown",
            "pause_tail",
        }:
            extreme_features[name] = 1.0 if name != "preclick_slowdown" else 1.5
        elif name in {
            "pointer_speed_disp",
            "click_dwell_disp",
            "text_gap_disp",
            "insert_len_disp",
            "correction_size_disp",
        }:
            extreme_features[name] = 0.01
        else:
            extreme_features[name] = extreme_features[name] * 3.0
    extreme, _ = profile.score(extreme_features)
    assert extreme is not None
    assert extreme > normal
    assert extreme > float(profile.threshold)


def test_too_few_features_is_insufficient() -> None:
    windows = owner_windows()
    profile = fit_profile(windows[:20])
    calibrate_profile(profile, windows[20:40], target_frr=0.15)
    blank = {name: None for name in profile.features}
    distance, explanation = profile.score(blank)
    assert distance is None
    assert explanation["reason"] == "insufficient features"
    assert not profile.accepts(distance)


def test_constant_feature_is_dropped() -> None:
    windows = owner_windows()
    for window in windows:
        window.features["insert_len_disp"] = 0.0
    profile = fit_profile(windows[:20])
    assert "insert_len_disp" in profile.training["dropped_features"]


def test_calibration_uses_only_owner_calibration_windows() -> None:
    windows = owner_windows()
    profile = fit_profile(windows[:20])
    calib = windows[20:40]
    calibrate_profile(profile, calib, target_frr=0.15)
    assert profile.calibration["n_windows"] == len(calib)
    assert profile.training["n_windows"] == 20
    # The threshold is not None and the calibration summary is recorded before
    # any test window is scored.
    assert profile.calibration["target_frr"] == 0.15
    assert profile.calibration["empirical_frr"] <= 0.15 + 1 / len(calib)
