from __future__ import annotations

from pathlib import Path

from marrow.features import FEATURE_NAMES, extract_windows
from marrow.schema import SCHEMA, Event, Recording

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "fake" / "mouse_keyboard" / "owner_rested" / "session-00.jsonl"


def test_real_window_features_have_expected_ranges() -> None:
    from marrow.schema import load_recording

    windows = extract_windows(load_recording(SAMPLE))
    assert len(windows) == 4
    for window in windows:
        assert window.n_events > 30
        assert 0 < window.features["pointer_speed_med"] < 5
        assert 0 <= window.features["path_efficiency"] <= 1
        assert 0 < window.features["text_gap_med"] < 5
        assert 0 <= window.features["correction_rate"] < 2


def test_text_only_recording_has_no_pointer_features() -> None:
    events = [
        Event(
            t=500 + i * 900,
            type="text",
            source="dictation",
            kind="insert",
            text="a dictated chunk",
        )
        for i in range(25)
    ]
    rec = Recording(
        path=Path("text-only.jsonl"),
        header={
            "type": "recording",
            "schema": SCHEMA,
            "recording_id": "text-only",
            "style": "assistive_dictation",
            "persona": "owner_rested",
            "label": "owner",
            "duration_ms": 25000,
        },
        events=events,
    )
    windows = extract_windows(rec, window_ms=30000, min_events=1)
    assert len(windows) == 1
    features = windows[0].features
    assert features["pointer_speed_med"] is None
    assert features["path_efficiency"] is None
    assert features["text_gap_med"] is not None
    assert features["insert_len_disp"] is not None


def test_pointer_only_recording_has_no_text_features() -> None:
    events = [
        Event(t=500 + i * 20, type="pointer", x=0.2 + i * 0.001, y=0.5)
        for i in range(1000)
    ]
    rec = Recording(
        path=Path("pointer-only.jsonl"),
        header={
            "type": "recording",
            "schema": SCHEMA,
            "recording_id": "pointer-only",
            "style": "mouse_keyboard",
            "persona": "owner_rested",
            "label": "owner",
            "duration_ms": 25000,
        },
        events=events,
    )
    windows = extract_windows(rec, window_ms=30000, min_events=1)
    assert len(windows) == 1
    features = windows[0].features
    assert features["pointer_speed_med"] is not None
    assert features["text_gap_med"] is None
    assert features["correction_rate"] is None


def test_every_feature_is_documented() -> None:
    from marrow.features import FEATURE_DESCRIPTIONS

    assert set(FEATURE_NAMES) == set(FEATURE_DESCRIPTIONS)


def test_short_recording_produces_one_window() -> None:
    events = [Event(t=i * 100, type="pointer", x=0.1 + i * 0.01, y=0.5) for i in range(50)]
    rec = Recording(
        path=Path("short.jsonl"),
        header={
            "type": "recording",
            "schema": SCHEMA,
            "recording_id": "short",
            "duration_ms": 5000,
        },
        events=events,
    )
    windows = extract_windows(rec, window_ms=30000, min_events=1)
    assert len(windows) == 1
    assert windows[0].end_ms == 5000
