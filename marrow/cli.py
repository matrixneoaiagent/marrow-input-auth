"""Command line entry points for Marrow.

Examples:
    python -m marrow demo
    python -m marrow generate --force
    python -m marrow train --train data/fake/mouse_keyboard/owner_rested \
        --calibration data/fake/mouse_keyboard/owner_rested --out /tmp/p.json
    python -m marrow evaluate --profile /tmp/p.json --recordings some.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, extract_all
from .generate import generate_dataset
from .model import Profile, calibrate_profile, fit_profile
from .pipeline import (
    TARGET_FRR,
    WINDOW_MS,
    STRIDE_MS,
    load_corpus,
    run_demo,
    run_swap_test,
    score_windows,
    windows_by_style,
)
from .schema import RecordingError, load_recordings


def _add_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-ms", type=int, default=WINDOW_MS, help="feature window length (default 30000)")
    parser.add_argument("--stride-ms", type=int, default=STRIDE_MS, help="window stride (default 10000)")
    parser.add_argument("--min-events", type=int, default=20, help="minimum events per scored window")


def cmd_generate(args: argparse.Namespace) -> int:
    manifest = generate_dataset(args.out, seed=args.seed, force=args.force)
    n = len(manifest["files"])
    print(f"Wrote {n} synthetic recordings under {args.out} (seed {manifest['generator_seed']})")
    for style in manifest["styles"]:
        count = sum(1 for f in manifest["files"] if f["style"] == style)
        print(f"  {style}: {count} sessions")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    run_demo(
        data_root=args.data,
        results_dir=args.results,
        seed=args.seed,
        regenerate=args.regenerate,
        target_frr=args.target_frr,
        verbose=True,
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    train_recs = load_recordings(args.train)
    train_windows = extract_all(
        train_recs, window_ms=args.window_ms, stride_ms=args.stride_ms, min_events=args.min_events
    )
    if args.calibration:
        calib_recs = load_recordings(args.calibration)
        calib_windows = extract_all(
            calib_recs, window_ms=args.window_ms, stride_ms=args.stride_ms, min_events=args.min_events
        )
    else:
        print("warning: no --calibration recordings supplied; calibrating on the training windows", file=sys.stderr)
        calib_windows = train_windows
    if not train_windows:
        print("error: no training windows", file=sys.stderr)
        return 2
    profile = fit_profile(train_windows)
    calibrate_profile(profile, calib_windows, target_frr=args.target_frr)
    profile.training["recordings"] = [str(r.path) for r in train_recs]
    profile.save(args.out)
    print(f"Wrote {args.out}")
    print(f"  features kept: {len(profile.features)} / {len(FEATURE_NAMES)}")
    for name in profile.features:
        print(f"    {name}: {FEATURE_DESCRIPTIONS[name]}")
    print(f"  training windows: {len(train_windows)}")
    print(f"  calibration windows: {len(calib_windows)}")
    print(f"  threshold: {profile.threshold:.3f}")
    print(f"  calibration FRR: {100 * profile.calibration['empirical_frr']:.1f}%")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    profile = Profile.load(args.profile)
    recs = load_recordings(args.recordings)
    windows = extract_all(
        recs, window_ms=args.window_ms, stride_ms=args.stride_ms, min_events=args.min_events
    )
    rows = score_windows(profile, windows)
    by_recording: dict[str, list[dict]] = {}
    for row in rows:
        by_recording.setdefault(row["recording_id"], []).append(row)

    print(f"profile: {args.profile}")
    print(f"threshold: {profile.threshold:.3f}   features: {len(profile.features)}")
    print("")
    print(f"{'recording':42s} {'windows':>8s} {'accepted':>9s} {'trim score':>11s}  verdict")
    verdicts: list[bool] = []
    for recording_id, group in sorted(by_recording.items()):
        usable = [r for r in group if r["scored"]]
        if not usable:
            print(f"{recording_id:42s} {len(group):8d} {'-':>9s} {'-':>13s}  insufficient")
            continue
        accepted = sum(1 for r in usable if r["accepted"])
        distances = sorted((float(r["distance"]) for r in usable), reverse=True)
        keep = max(1, math.ceil(0.75 * len(distances)))
        med = sum(distances[:keep]) / keep
        session_accept = med <= float(profile.threshold)
        verdicts.append(session_accept)
        print(
            f"{recording_id:42s} {len(usable):8d} {accepted:9d} {med:13.3f}  "
            f"{'ACCEPT' if session_accept else 'REJECT'}"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "profile": str(args.profile),
                    "threshold": profile.threshold,
                    "rows": rows,
                    "session_accept_rate": None
                    if not verdicts
                    else round(sum(verdicts) / len(verdicts), 4),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")
    return 0


def cmd_swap(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.data)
    windows = windows_by_style(corpus)
    failed = False
    for style in sorted(windows):
        result = run_swap_test(windows[style], style)
        print(f"[{style}]")
        for key, block in result["results"].items():
            owner = block["owner_window_accept"]
            print(
                f"  {key}: owner accepted {100 * owner['rate']:.1f}% of windows "
                f"({owner['count']}/{owner['n']}), "
                f"threshold {block['threshold']:.3f}, self-accept {100 * block['self_window_accept']['rate']:.1f}%"
            )
            if owner["rate"] > 0.10:
                failed = True
    return 1 if failed else 0


def cmd_validate(args: argparse.Namespace) -> int:
    ok = True
    for path in args.recordings:
        try:
            recs = load_recordings([path])
            for rec in recs:
                counts = rec.counts()
                print(
                    f"OK {rec.path}  style={rec.style} persona={rec.persona} "
                    f"split={rec.split} label={rec.label} "
                    f"pointer={counts['pointer']} buttons={counts['button']} "
                    f"inserts={counts['text_insert']} deletes={counts['text_delete']}"
                )
        except RecordingError as exc:
            ok = False
            print(f"FAIL {path}: {exc}")
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marrow",
        description="Marrow: recognise the enrolled user from input timing and dynamics.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="generate the fake corpus if needed, train, calibrate and evaluate")
    p.add_argument("--data", default="data/fake")
    p.add_argument("--results", default="results")
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--target-frr", type=float, default=TARGET_FRR)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("generate", help="write the deterministic synthetic corpus")
    p.add_argument("--out", default="data/fake")
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="train and calibrate a profile from recordings")
    p.add_argument("--train", nargs="+", required=True, help="recording files or directories")
    p.add_argument("--calibration", nargs="+", help="owner-only calibration files or directories")
    p.add_argument("--out", required=True, help="output profile JSON")
    p.add_argument("--target-frr", type=float, default=TARGET_FRR)
    _add_window_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score any recording with a profile")
    p.add_argument("--profile", required=True)
    p.add_argument("--recordings", nargs="+", required=True)
    p.add_argument("--json", help="optional JSON output path")
    _add_window_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("swap-test", help="train on an impostor and confirm the owner is rejected")
    p.add_argument("--data", default="data/fake")
    p.set_defaults(func=cmd_swap)

    p = sub.add_parser("validate", help="validate recording files against the schema")
    p.add_argument("recordings", nargs="+")
    p.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (RecordingError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
