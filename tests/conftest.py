from __future__ import annotations

from pathlib import Path

import pytest

from marrow.pipeline import run_demo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fake"


@pytest.fixture(scope="session")
def demo_report(tmp_path_factory):
    out = tmp_path_factory.mktemp("marrow-results")
    report = run_demo(DATA, out, verbose=False)
    return report, out
