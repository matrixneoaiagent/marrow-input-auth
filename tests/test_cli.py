from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from marrow import cli

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fake"
SAMPLE = DATA / "mouse_keyboard" / "owner_rested" / "session-00.jsonl"


def test_validate_command() -> None:
    assert cli.main(["validate", str(SAMPLE)]) == 0


def test_train_and_evaluate_commands(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    assert (
        cli.main(
            [
                "train",
                "--train",
                str(DATA / "mouse_keyboard" / "owner_rested"),
                "--calibration",
                str(DATA / "mouse_keyboard" / "owner_rested"),
                "--out",
                str(profile),
            ]
        )
        == 0
    )
    assert profile.exists()
    assert (
        cli.main(
            [
                "evaluate",
                "--profile",
                str(profile),
                "--recordings",
                str(SAMPLE),
            ]
        )
        == 0
    )


def test_generate_command(tmp_path: Path) -> None:
    out = tmp_path / "fake"
    assert cli.main(["generate", "--out", str(out), "--seed", "123"]) == 0
    assert (out / "manifest.json").exists()
    assert (out / "mouse_keyboard" / "owner_rested" / "session-00.jsonl").exists()


def test_python_m_marrow_demo_entry_point(tmp_path: Path) -> None:
    results = tmp_path / "results"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "marrow.demo",
            "--data",
            str(DATA),
            "--results",
            str(results),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (results / "report.md").exists()
    assert "OVERALL" in proc.stdout
