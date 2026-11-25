from __future__ import annotations

import hashlib
import json
from pathlib import Path

from marrow.generate import OWNER, generate_session
from marrow.pipeline import load_corpus
from marrow.schema import load_recordings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fake"


def test_demo_report_has_both_styles_and_fake_data(demo_report) -> None:
    report, out = demo_report
    assert report["data"]["capture"] == "synthetic"
    assert report["data"]["n_recordings"] == 92
    assert set(report["styles"]) == {"mouse_keyboard", "assistive_dictation"}
    assert report["target_frr"] == 0.15
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()
    assert (out / "profiles" / "mouse_keyboard_owner.json").exists()
    assert "synthetic" in (out / "report.md").read_text(encoding="utf-8").lower()


def test_demo_metrics_are_within_the_pre_registered_bounds(demo_report) -> None:
    report, _ = demo_report
    overall = report["overall"]["window"]
    assert overall["owner_frr"]["rate"] <= 0.20
    assert overall["stranger_far"]["rate"] <= 0.12
    assert overall["mimic_far"]["rate"] <= 0.05

    session = report["overall"]["session"]
    assert session["owner_frr"]["rate"] <= 0.05
    assert session["stranger_far"]["rate"] <= 0.05
    assert session["mimic_far"]["rate"] <= 0.05


def test_every_style_threshold_is_fixed_on_owner_calibration(demo_report) -> None:
    report, _ = demo_report
    for style, data in report["styles"].items():
        assert style in {"mouse_keyboard", "assistive_dictation"}
        assert data["threshold"] is not None
        assert data["calibration"]["n_windows"] == 20
        assert data["calibration"]["target_frr"] == 0.15
        assert data["calibration"]["empirical_frr"] <= 0.15 + 1e-9
        assert data["model"]["n_features"] >= 10


def test_swap_test_rejects_the_owner_everywhere(demo_report) -> None:
    report, _ = demo_report
    for style, swap in report["swap"].items():
        assert swap["style"] == style
        for key in ("trained_on_stranger", "trained_on_mimic"):
            block = swap["results"][key]
            assert block["owner_window_accept"]["rate"] == 0.0
            assert block["owner_session_accept"]["rate"] == 0.0
            assert block["self_window_accept"]["rate"] >= 0.75


def test_corpus_has_the_required_personas() -> None:
    corpus = load_corpus(DATA)
    expected = {"owner_rested", "owner_tired", "stranger", "mimic"}
    for style, recordings in corpus.items():
        personas = {r.persona for r in recordings}
        assert personas == expected, (style, personas)
        counts = {p: sum(1 for r in recordings if r.persona == p) for p in expected}
        assert counts == {"owner_rested": 20, "owner_tired": 6, "stranger": 10, "mimic": 10}
        labels = {r.label for r in recordings}
        assert "owner" in labels
        assert "impostor_stranger" in labels
        assert "impostor_mimic" in labels


def test_manifest_hashes_match_the_files() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture"] == "synthetic"
    assert manifest["generator_seed"] == 20261125
    assert len(manifest["files"]) == 92
    for item in manifest["files"][:5]:
        path = DATA / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["sha256"]


def test_generator_is_deterministic() -> None:
    params = OWNER["mouse_keyboard"]
    first = generate_session("mouse_keyboard", "owner_rested", params, 0, seed=20261125)
    second = generate_session("mouse_keyboard", "owner_rested", params, 0, seed=20261125)
    other = generate_session("mouse_keyboard", "owner_rested", params, 0, seed=7)
    assert first == second
    assert first != other
    assert first[0]["capture"] == "synthetic"


def test_all_shipped_recordings_load() -> None:
    recordings = load_recordings([DATA])
    assert len(recordings) == 92
    for rec in recordings:
        assert rec.events
        assert rec.duration_ms > 0
