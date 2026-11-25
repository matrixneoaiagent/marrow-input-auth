from __future__ import annotations

import json
from pathlib import Path

import pytest

from marrow.schema import (
    SCHEMA,
    RecordingError,
    discover_recordings,
    load_recording,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fake"
SAMPLE = DATA / "mouse_keyboard" / "owner_rested" / "session-00.jsonl"


def header() -> dict:
    return {
        "type": "recording",
        "schema": SCHEMA,
        "recording_id": "test/recording/00",
        "capture": "synthetic",
        "style": "mouse_keyboard",
        "persona": "owner_rested",
        "split": "train",
        "label": "owner",
        "duration_ms": 2000,
    }


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_valid_recording_loads() -> None:
    rec = load_recording(SAMPLE)
    assert rec.style == "mouse_keyboard"
    assert rec.persona == "owner_rested"
    assert rec.split == "train"
    assert rec.label == "owner"
    assert rec.duration_ms == 60000
    counts = rec.counts()
    assert counts["pointer"] > 100
    assert counts["text_insert"] > 20


def test_discover_finds_every_shipped_recording() -> None:
    paths = discover_recordings([DATA])
    assert len(paths) == 92
    assert {p.suffix for p in paths} == {".jsonl"}


@pytest.mark.parametrize(
    "bad_event",
    [
        {"type": "pointer", "t": 10, "x": 1.2, "y": 0.5},
        {"type": "pointer", "t": 10, "x": 0.5, "y": -0.1},
        {"type": "button", "t": 10, "x": 0.5, "y": 0.5, "button": "left", "action": "press"},
        {"type": "text", "t": 10, "source": "keyboard", "kind": "insert", "text": "a"},
        {"type": "text", "t": 10, "source": "typed", "kind": "delete", "chars": 0},
        {"type": "text", "t": 10, "source": "typed", "kind": "insert", "text": ""},
        {"type": "mystery", "t": 10},
    ],
)
def test_bad_events_are_rejected(tmp_path: Path, bad_event: dict) -> None:
    path = write(tmp_path / "bad.jsonl", [header(), bad_event])
    with pytest.raises(RecordingError):
        load_recording(path)


def test_decreasing_timestamp_is_rejected(tmp_path: Path) -> None:
    path = write(
        tmp_path / "bad.jsonl",
        [
            header(),
            {"type": "pointer", "t": 100, "x": 0.5, "y": 0.5},
            {"type": "pointer", "t": 99, "x": 0.6, "y": 0.5},
        ],
    )
    with pytest.raises(RecordingError, match="non-decreasing"):
        load_recording(path)


def test_bad_header_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "bad.jsonl", [{"type": "recording", "schema": "other/v1"}])
    with pytest.raises(RecordingError):
        load_recording(path)


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(RecordingError):
        load_recording(path)


def test_empty_recording_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(RecordingError):
        load_recording(path)
