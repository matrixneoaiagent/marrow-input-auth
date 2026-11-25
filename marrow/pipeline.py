"""End-to-end training, calibration, evaluation and reporting."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, Window, extract_windows
from .generate import STYLES, generate_dataset
from .metrics import median, rate
from .model import Profile, calibrate_profile, fit_profile
from .schema import Recording, load_recordings

WINDOW_MS = 30_000
STRIDE_MS = 10_000
TARGET_FRR = 0.15


def load_corpus(data_root: str | Path) -> dict[str, list[Recording]]:
    data_root = Path(data_root)
    recordings = load_recordings([data_root])
    grouped: dict[str, list[Recording]] = {}
    for rec in recordings:
        grouped.setdefault(rec.style, []).append(rec)
    for style in grouped:
        grouped[style].sort(key=lambda r: (r.persona, r.recording_id))
    return grouped


def windows_by_style(corpus: dict[str, list[Recording]]) -> dict[str, list[Window]]:
    out: dict[str, list[Window]] = {}
    for style, recordings in corpus.items():
        windows: list[Window] = []
        for rec in recordings:
            windows.extend(extract_windows(rec, window_ms=WINDOW_MS, stride_ms=STRIDE_MS))
        out[style] = windows
    return out


def _select(windows: Iterable[Window], persona: str, split: str | None = None) -> list[Window]:
    return [
        w for w in windows
        if w.persona == persona and (split is None or w.split == split)
    ]


def train_owner_profile(
    windows: list[Window],
    target_frr: float = TARGET_FRR,
) -> tuple[Profile, list[Window], list[Window]]:
    train = _select(windows, "owner_rested", "train")
    calib = _select(windows, "owner_rested", "calibration")
    if not train or not calib:
        raise ValueError("owner_rested train/calibration windows not found")
    profile = fit_profile(train)
    calibrate_profile(profile, calib, target_frr=target_frr)
    return profile, train, calib


def score_windows(profile: Profile, windows: Iterable[Window]) -> list[dict]:
    rows: list[dict] = []
    for window in windows:
        distance, explanation = profile.score(window.features)
        accepted = distance is not None and profile.accepts(distance)
        rows.append(
            {
                "recording_id": window.recording_id,
                "style": window.style,
                "persona": window.persona,
                "label": window.label,
                "split": window.split,
                "start_ms": window.start_ms,
                "end_ms": window.end_ms,
                "distance": None if distance is None else round(distance, 4),
                "accepted": accepted,
                "scored": distance is not None,
                "n_events": window.n_events,
                "top_features": explanation.get("top_features", []),
            }
        )
    return rows


def _group_rate(rows: list[dict], direction: str) -> dict:
    usable = [r for r in rows if r["scored"]]
    n = len(usable)
    if direction == "accept":
        count = sum(1 for r in usable if r["accepted"])
        label = "accepted"
    else:
        count = sum(1 for r in usable if not r["accepted"])
        label = "rejected"
    out = rate(count, n)
    out["direction"] = label
    out["unscored"] = len(rows) - n
    return out


def _session_metrics(rows: list[dict], threshold: float, direction: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["recording_id"], []).append(row)
    decisions: list[bool] = []
    for group in grouped.values():
        usable = [r for r in group if r["scored"]]
        if not usable:
            continue
        # Session rule: average the highest 75% of window scores (drop the
        # single most owner-like window when at least four are available).
        # One lucky window must not open the session.
        distances = sorted((float(r["distance"]) for r in usable), reverse=True)
        keep = max(1, math.ceil(0.75 * len(distances)))
        distance = sum(distances[:keep]) / keep
        decisions.append(distance <= threshold)
    n = len(decisions)
    if direction == "accept":
        count = sum(1 for d in decisions if d)
        label = "accepted"
    else:
        count = sum(1 for d in decisions if not d)
        label = "rejected"
    out = rate(count, n)
    out["direction"] = label
    out["unscored_sessions"] = len(grouped) - n
    return out


def evaluate_style(profile: Profile, windows: list[Window]) -> dict:
    owner = score_windows(profile, _select(windows, "owner_tired", "test"))
    stranger = score_windows(profile, _select(windows, "stranger"))
    mimic = score_windows(profile, _select(windows, "mimic"))
    return {
        "threshold": profile.threshold,
        "calibration": profile.calibration,
        "model": {
            "features": profile.features,
            "n_features": len(profile.features),
            "n_train_windows": profile.training.get("n_windows"),
            "n_train_recordings": profile.training.get("n_recordings"),
            "dropped_features": profile.training.get("dropped_features", []),
            "coverage": profile.training.get("coverage", {}),
        },
        "window": {
            "owner_frr": _group_rate(owner, "reject"),
            "stranger_far": _group_rate(stranger, "accept"),
            "mimic_far": _group_rate(mimic, "accept"),
        },
        "session": {
            "owner_frr": _session_metrics(owner, float(profile.threshold), "reject"),
            "stranger_far": _session_metrics(stranger, float(profile.threshold), "accept"),
            "mimic_far": _session_metrics(mimic, float(profile.threshold), "accept"),
        },
        "rows": {"owner": owner, "stranger": stranger, "mimic": mimic},
    }


def _overall(style_reports: dict[str, dict]) -> dict:
    def pool(key: str) -> list[dict]:
        rows: list[dict] = []
        for report in style_reports.values():
            rows.extend(report["rows"][key])
        return rows

    def pool_sessions(key: str, direction: str) -> dict:
        counts = {"n": 0, "count": 0, "unscored": 0}
        for report in style_reports.values():
            block = report["session"][key]
            counts["n"] += block["n"]
            counts["count"] += block["count"]
            counts["unscored"] += block.get("unscored_sessions", 0)
        out = rate(counts["count"], counts["n"])
        out["direction"] = direction
        out["unscored_sessions"] = counts["unscored"]
        return out

    out = {
        "window": {
            "owner_frr": _group_rate(pool("owner"), "reject"),
            "stranger_far": _group_rate(pool("stranger"), "accept"),
            "mimic_far": _group_rate(pool("mimic"), "accept"),
        },
        "session": {
            "owner_frr": pool_sessions("owner_frr", "rejected"),
            "stranger_far": pool_sessions("stranger_far", "accepted"),
            "mimic_far": pool_sessions("mimic_far", "accepted"),
        },
    }
    return out


def run_swap_test(windows: list[Window], style: str) -> dict:
    """Train a profile on an impostor and try to open it as the owner."""

    out: dict[str, dict] = {}
    owner = _select(windows, "owner_tired", "test")
    for persona, key in (("stranger", "trained_on_stranger"), ("mimic", "trained_on_mimic")):
        impostor_windows = _select(windows, persona)
        if not impostor_windows:
            continue
        profile = fit_profile(impostor_windows)
        calibrate_profile(profile, impostor_windows, target_frr=TARGET_FRR)
        owner_rows = score_windows(profile, owner)
        impostor_rows = score_windows(profile, impostor_windows)
        out[key] = {
            "threshold": profile.threshold,
            "owner_window_accept": _group_rate(owner_rows, "accept"),
            "owner_session_accept": _session_metrics(owner_rows, float(profile.threshold), "accept"),
            "self_window_accept": _group_rate(impostor_rows, "accept"),
            "n_train_windows": len(impostor_windows),
        }
    return {"style": style, "results": out}


def feature_summary(windows: list[Window]) -> list[dict]:
    """Median feature values per persona, for the mimic-closeness table."""

    groups = {
        "owner": _select(windows, "owner_rested"),
        "mimic": _select(windows, "mimic"),
        "stranger": _select(windows, "stranger"),
    }
    rows: list[dict] = []
    for feature in FEATURE_NAMES:
        row: dict = {"feature": feature, "description": FEATURE_DESCRIPTIONS[feature]}
        for name, group in groups.items():
            values = [w.features.get(feature) for w in group]
            present = [float(v) for v in values if v is not None]
            row[name] = None if not present else round(float(median(present)), 5)
            row[f"{name}_n"] = len(present)
        rows.append(row)
    return rows


def write_markdown(report: dict, path: str | Path) -> None:
    def pct(block: dict) -> str:
        if block.get("rate") is None:
            return "n/a"
        return f"{100 * block['rate']:.1f}% ({block['count']}/{block['n']})"

    lines: list[str] = []
    lines.append("# Marrow prototype — measured results")
    lines.append("")
    lines.append(f"Generated: `{report['generated_at']}`  ")
    lines.append(f"Data: **synthetic only**, generator seed `{report['seed']}`  ")
    lines.append("Threshold: fixed on owner-only calibration windows before any test window was scored.  ")
    lines.append(f"Calibration target: {100 * report['target_frr']:.0f}% false-reject rate.")
    lines.append("")
    lines.append("## Window-level decision (primary)")
    lines.append("")
    lines.append("| Style | Owner FRR (tired) | Stranger FAR | Mimic FAR |")
    lines.append("|---|---:|---:|---:|")
    for style, sr in report["styles"].items():
        w = sr["window"]
        lines.append(
            f"| {style} | {pct(w['owner_frr'])} | {pct(w['stranger_far'])} | {pct(w['mimic_far'])} |"
        )
    ow = report["overall"]["window"]
    lines.append(
        f"| **overall** | **{pct(ow['owner_frr'])}** | **{pct(ow['stranger_far'])}** | **{pct(ow['mimic_far'])}** |"
    )
    lines.append("")
    lines.append("FRR = real owner rejected (doubted). FAR = impostor accepted.  ")
    lines.append("95% Wilson intervals are in `results/report.json`.")
    lines.append("")
    lines.append("## Session-level decision (upper-trimmed mean of window scores)")
    lines.append("")
    lines.append("| Style | Owner FRR (tired) | Stranger FAR | Mimic FAR |")
    lines.append("|---|---:|---:|---:|")
    for style, sr in report["styles"].items():
        s = sr["session"]
        lines.append(
            f"| {style} | {pct(s['owner_frr'])} | {pct(s['stranger_far'])} | {pct(s['mimic_far'])} |"
        )
    os_ = report["overall"]["session"]
    lines.append(
        f"| **overall** | **{pct(os_['owner_frr'])}** | **{pct(os_['stranger_far'])}** | **{pct(os_['mimic_far'])}** |"
    )
    lines.append("")
    lines.append("## Swap test: profile trained on an impostor")
    lines.append("")
    lines.append("| Style | Wrong profile trained on | Owner windows accepted | Owner sessions accepted |")
    lines.append("|---|---|---:|---:|")
    for style, swap in report["swap"].items():
        for key, label in (("trained_on_stranger", "stranger"), ("trained_on_mimic", "mimic")):
            block = swap["results"].get(key)
            if not block:
                continue
            lines.append(
                f"| {style} | {label} | {pct(block['owner_window_accept'])} | {pct(block['owner_session_accept'])} |"
            )
    lines.append("")
    lines.append("## Thresholds and model")
    lines.append("")
    lines.append("| Style | Threshold | Calibration windows | Calibration FRR | Features kept |")
    lines.append("|---|---:|---:|---:|---:|")
    for style, sr in report["styles"].items():
        cal = sr["calibration"]
        lines.append(
            f"| {style} | {sr['threshold']:.3f} | {cal.get('n_windows', 'n/a')} | "
            f"{100 * cal.get('empirical_frr', 0):.1f}% | {sr['model']['n_features']} |"
        )
    lines.append("")
    lines.append("## Mimic closeness (median feature values)")
    lines.append("")
    lines.append("The mimic is generated with the owner's medians for speed, pause, cadence, correction")
    lines.append("rate, dwell and chunk length, but not the owner's dispersion, tails, path shape or")
    lines.append("cross-event coupling.  Values below are medians over all windows.")
    lines.append("")
    header = "| Feature | " + " | ".join(report["styles"].keys()) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(report["styles"]))
    for feature in FEATURE_NAMES:
        cells: list[str] = []
        for style in report["styles"]:
            rows = {r["feature"]: r for r in report["styles"][style]["feature_summary"]}
            row = rows.get(feature, {})
            owner = row.get("owner")
            mimic = row.get("mimic")
            cells.append("n/a" if owner is None or mimic is None else f"{owner:g} / {mimic:g}")
        lines.append(f"| `{feature}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cell format: owner median / mimic median.")
    lines.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_demo(
    data_root: str | Path = "data/fake",
    results_dir: str | Path = "results",
    seed: int = 20261125,
    regenerate: bool = False,
    target_frr: float = TARGET_FRR,
    verbose: bool = True,
) -> dict:
    data_root = Path(data_root)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_root / "manifest.json"
    if regenerate or not manifest_path.exists():
        manifest = generate_dataset(data_root, seed=seed, force=True)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    corpus = load_corpus(data_root)
    windows = windows_by_style(corpus)
    style_reports: dict[str, dict] = {}
    swap_reports: dict[str, dict] = {}
    all_windows: list[Window] = []

    for style in STYLES:
        if style not in windows:
            raise RuntimeError(f"no recordings found for style {style!r}")
        style_windows = windows[style]
        all_windows.extend(style_windows)
        profile, train, calib = train_owner_profile(style_windows, target_frr=target_frr)
        profile.training["style"] = style
        profile.save(results_dir / "profiles" / f"{style}_owner.json")

        report = evaluate_style(profile, style_windows)
        report["feature_summary"] = feature_summary(style_windows)
        style_reports[style] = report
        swap_reports[style] = run_swap_test(style_windows, style)

        if verbose:
            w = report["window"]
            print(
                f"[{style}] threshold={profile.threshold:.3f}  "
                f"owner FRR={100 * w['owner_frr']['rate']:.1f}%  "
                f"stranger FAR={100 * w['stranger_far']['rate']:.1f}%  "
                f"mimic FAR={100 * w['mimic_far']['rate']:.1f}%"
            )

    overall = _overall(style_reports)
    report = {
        "schema": "marrow.report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": manifest.get("generator_seed", seed),
        "target_frr": target_frr,
        "window_ms": WINDOW_MS,
        "stride_ms": STRIDE_MS,
        "data": {
            "capture": "synthetic",
            "root": str(data_root),
            "n_recordings": sum(len(v) for v in corpus.values()),
            "n_windows": len(all_windows),
        },
        "styles": style_reports,
        "overall": overall,
        "swap": swap_reports,
    }
    (results_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # The markdown writer needs feature_summary inside each style; it is there.
    write_markdown(report, results_dir / "report.md")

    if verbose:
        print("")
        print("OVERALL window-level:")
        print(f"  owner FRR   {100 * overall['window']['owner_frr']['rate']:.1f}%")
        print(f"  stranger FAR {100 * overall['window']['stranger_far']['rate']:.1f}%")
        print(f"  mimic FAR    {100 * overall['window']['mimic_far']['rate']:.1f}%")
        print("")
        print("SWAP TEST (wrong profile -> owner):")
        for style, swap in swap_reports.items():
            for key, label in (("trained_on_stranger", "stranger"), ("trained_on_mimic", "mimic")):
                block = swap["results"].get(key)
                if block:
                    print(
                        f"  [{style}] trained on {label}: owner accepted "
                        f"{100 * block['owner_window_accept']['rate']:.1f}% of windows, "
                        f"{100 * block['owner_session_accept']['rate']:.1f}% of sessions"
                    )
        print("")
        print(f"Wrote {results_dir / 'report.json'}")
        print(f"Wrote {results_dir / 'report.md'}")
    return report
