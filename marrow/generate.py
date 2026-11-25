"""Deterministic synthetic recording generator.

The data is fake.  It is generated from a small motor/behaviour model so that
the repository contains no personal data and the demo is reproducible.  The
generator deliberately gives the mimic the owner's *summary statistics* (median
speed, pause, typing cadence, correction rate, click dwell, chunk length) while
leaving dispersion, heavy tails, path shape, and cross-event coupling different.
That is the adversary the prototype is meant to catch.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .schema import SCHEMA

STYLES = ("mouse_keyboard", "assistive_dictation")

STYLE_DESCRIPTIONS = {
    "mouse_keyboard": "standard mouse + physical keyboard; typed text and backspace corrections",
    "assistive_dictation": "slow head/eye pointer + dictation chunks, with on-screen-keyboard corrections",
}

WORDS = (
    "the quick brown fox jumps over lazy dog while morning light falls on a small desk "
    "and the quiet room fills with ordinary work notes about weather trains coffee books "
    "later today we will check the list and send a short reply before the meeting starts "
    "please review the draft and note any changes that need a second look soon"
).split()


@dataclass
class PersonaParams:
    pointer_speed: float          # screen widths / second
    pointer_speed_sigma: float
    pause_med: float              # seconds between movement segments
    pause_sigma: float
    pause_tail_prob: float
    path_curvature: float         # perpendicular bow, fraction of distance
    path_noise: float             # per-sample tremor, screen fractions
    ease: float                   # 0 = linear, 1 = minimum-jerk (strong end slowdown)
    click_prob: float
    click_dwell_med: float        # ms
    click_dwell_sigma: float
    text_mode: str                # "typed" or "dictation"
    block_pause_med: float        # seconds between text blocks / dictation chunks
    text_gap_med: float           # ms between typed characters
    text_gap_sigma: float
    burst_len_med: float          # chars per typed burst or dictation chunk
    burst_len_sigma: float
    insert_len_sigma: float       # dispersion of dictation chunk lengths
    correction_rate: float        # deleted chars / inserted chars
    correction_latency_med: float # ms
    correction_latency_sigma: float
    correction_size_med: float
    correction_size_sigma: float
    post_error_pause: float       # extra seconds after a correction
    onscreen_prob: float          # chance a dictation block is on-screen typing
    pointer_share: float          # fraction of blocks that are pointer blocks


OWNER = {
    "mouse_keyboard": PersonaParams(
        pointer_speed=0.38,
        pointer_speed_sigma=0.45,
        pause_med=0.65,
        pause_sigma=0.60,
        pause_tail_prob=0.10,
        path_curvature=0.10,
        path_noise=0.0022,
        ease=0.92,
        click_prob=0.80,
        click_dwell_med=95,
        click_dwell_sigma=0.32,
        text_mode="typed",
        block_pause_med=1.10,
        text_gap_med=140,
        text_gap_sigma=0.42,
        burst_len_med=8,
        burst_len_sigma=0.35,
        insert_len_sigma=0.0,
        correction_rate=0.070,
        correction_latency_med=900,
        correction_latency_sigma=0.50,
        correction_size_med=2.2,
        correction_size_sigma=0.50,
        post_error_pause=1.20,
        onscreen_prob=0.0,
        pointer_share=0.45,
    ),
    "assistive_dictation": PersonaParams(
        pointer_speed=0.18,
        pointer_speed_sigma=0.50,
        pause_med=1.20,
        pause_sigma=0.60,
        pause_tail_prob=0.10,
        path_curvature=0.12,
        path_noise=0.0020,
        ease=0.85,
        click_prob=0.98,
        click_dwell_med=380,
        click_dwell_sigma=0.30,
        text_mode="dictation",
        block_pause_med=1.70,
        text_gap_med=1700,
        text_gap_sigma=0.45,
        burst_len_med=16,
        burst_len_sigma=0.40,
        insert_len_sigma=0.45,
        correction_rate=0.200,
        correction_latency_med=2000,
        correction_latency_sigma=0.40,
        correction_size_med=6.5,
        correction_size_sigma=0.35,
        post_error_pause=1.80,
        onscreen_prob=0.15,
        pointer_share=0.55,
    ),
}

STRANGER = {
    "mouse_keyboard": PersonaParams(
        pointer_speed=0.85,
        pointer_speed_sigma=0.35,
        pause_med=0.30,
        pause_sigma=0.40,
        pause_tail_prob=0.03,
        path_curvature=0.04,
        path_noise=0.0008,
        ease=0.55,
        click_prob=0.75,
        click_dwell_med=65,
        click_dwell_sigma=0.25,
        text_mode="typed",
        block_pause_med=0.90,
        text_gap_med=200,
        text_gap_sigma=0.35,
        burst_len_med=5,
        burst_len_sigma=0.35,
        insert_len_sigma=0.0,
        correction_rate=0.015,
        correction_latency_med=500,
        correction_latency_sigma=0.40,
        correction_size_med=1.5,
        correction_size_sigma=0.40,
        post_error_pause=0.30,
        onscreen_prob=0.0,
        pointer_share=0.55,
    ),
    "assistive_dictation": PersonaParams(
        pointer_speed=0.50,
        pointer_speed_sigma=0.30,
        pause_med=0.55,
        pause_sigma=0.35,
        pause_tail_prob=0.02,
        path_curvature=0.02,
        path_noise=0.0006,
        ease=0.35,
        click_prob=0.75,
        click_dwell_med=90,
        click_dwell_sigma=0.15,
        text_mode="dictation",
        block_pause_med=0.80,
        text_gap_med=900,
        text_gap_sigma=0.25,
        burst_len_med=9,
        burst_len_sigma=0.25,
        insert_len_sigma=0.20,
        correction_rate=0.015,
        correction_latency_med=600,
        correction_latency_sigma=0.20,
        correction_size_med=4.0,
        correction_size_sigma=0.20,
        post_error_pause=0.20,
        onscreen_prob=0.25,
        pointer_share=0.50,
    ),
}

TIRED = {
    "mouse_keyboard": {
        "pointer_speed": 0.84, "pause_med": 1.25, "pause_tail_prob": 1.25,
        "block_pause_med": 1.10, "text_gap_med": 1.10, "burst_len_med": 0.97,
        "correction_rate": 1.30, "correction_latency_med": 1.12,
        "click_dwell_med": 1.05, "path_noise": 1.00,
    },
    "assistive_dictation": {
        "pointer_speed": 0.84, "pause_med": 1.25, "pause_tail_prob": 1.25,
        "block_pause_med": 1.10, "text_gap_med": 1.10, "burst_len_med": 0.97,
        "correction_rate": 1.30, "correction_latency_med": 1.12,
        "click_dwell_med": 1.05, "path_noise": 1.00,
    },
}


def mimic_from(owner: PersonaParams) -> PersonaParams:
    """Copy the owner's observable medians, not their dynamics."""

    speed_factor = 1.03 if owner.text_mode == "typed" else 1.30
    correction_factor = 0.95 if owner.text_mode == "typed" else 1.35
    return replace(
        owner,
        pointer_speed=owner.pointer_speed * speed_factor,
        pointer_speed_sigma=0.10,
        pause_med=owner.pause_med * 0.98,
        pause_sigma=0.10,
        pause_tail_prob=0.0,
        path_curvature=0.012,
        path_noise=0.0003,
        ease=0.15,
        click_dwell_med=owner.click_dwell_med * 1.04,
        click_dwell_sigma=0.05,
        text_gap_med=owner.text_gap_med * 1.02,
        text_gap_sigma=0.10,
        burst_len_med=owner.burst_len_med * 1.00,
        burst_len_sigma=0.05,
        insert_len_sigma=0.05,
        correction_rate=owner.correction_rate * correction_factor,
        correction_latency_med=owner.correction_latency_med * 1.02,
        correction_latency_sigma=0.04,
        correction_size_med=owner.correction_size_med,
        correction_size_sigma=0.03,
        post_error_pause=0.10,
    )


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    if sigma <= 0:
        return median
    return median * math.exp(rng.gauss(0.0, sigma))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _text_of_length(rng: random.Random, length: int) -> str:
    out: list[str] = []
    while sum(len(w) for w in out) + max(0, len(out) - 1) < length:
        out.append(rng.choice(WORDS))
    text = " ".join(out)
    return text[:length] if len(text) >= length else text.ljust(length, "x")


def _jitter_params(rng: random.Random, params: PersonaParams) -> PersonaParams:
    def j(value: float, sigma: float = 0.06) -> float:
        return value * math.exp(rng.gauss(0.0, sigma))

    return replace(
        params,
        pointer_speed=j(params.pointer_speed),
        pause_med=j(params.pause_med),
        click_dwell_med=j(params.click_dwell_med),
        block_pause_med=j(params.block_pause_med),
        text_gap_med=j(params.text_gap_med),
        burst_len_med=j(params.burst_len_med),
        correction_rate=j(params.correction_rate),
        correction_latency_med=j(params.correction_latency_med),
        correction_size_med=j(params.correction_size_med),
    )


def _tired_params(params: PersonaParams, style: str) -> PersonaParams:
    m = TIRED[style]
    return replace(
        params,
        pointer_speed=params.pointer_speed * m["pointer_speed"],
        pause_med=params.pause_med * m["pause_med"],
        pause_tail_prob=min(0.35, params.pause_tail_prob * m["pause_tail_prob"]),
        block_pause_med=params.block_pause_med * m["block_pause_med"],
        text_gap_med=params.text_gap_med * m["text_gap_med"],
        burst_len_med=params.burst_len_med * m["burst_len_med"],
        correction_rate=params.correction_rate * m["correction_rate"],
        correction_latency_med=params.correction_latency_med * m["correction_latency_med"],
        click_dwell_med=params.click_dwell_med * m["click_dwell_med"],
        path_noise=params.path_noise * m["path_noise"],
    )


def _pointer_block(
    rng: random.Random,
    params: PersonaParams,
    t: int,
    pos: tuple[float, float],
    duration_ms: int,
) -> tuple[int, tuple[float, float], list[dict]]:
    target = (rng.uniform(0.06, 0.94), rng.uniform(0.08, 0.92))
    dist = math.hypot(target[0] - pos[0], target[1] - pos[1])
    speed = max(0.01, _lognormal(rng, params.pointer_speed, params.pointer_speed_sigma))
    move_s = _clamp(dist / speed, 0.18, 9.0)
    n = max(4, int(round(move_s / 0.020)))
    mid_x = (pos[0] + target[0]) / 2
    mid_y = (pos[1] + target[1]) / 2
    if dist > 1e-9:
        perp_x = -(target[1] - pos[1]) / dist
        perp_y = (target[0] - pos[0]) / dist
    else:
        perp_x = perp_y = 0.0
    control = (
        mid_x + perp_x * params.path_curvature * dist,
        mid_y + perp_y * params.path_curvature * dist,
    )
    events: list[dict] = []
    last_t = t
    for i in range(1, n + 1):
        u = i / n
        s = (1 - params.ease) * u + params.ease * (10 * u**3 - 15 * u**4 + 6 * u**5)
        x = (1 - s) ** 2 * pos[0] + 2 * (1 - s) * s * control[0] + s**2 * target[0]
        y = (1 - s) ** 2 * pos[1] + 2 * (1 - s) * s * control[1] + s**2 * target[1]
        if i < n:
            x += rng.gauss(0.0, params.path_noise)
            y += rng.gauss(0.0, params.path_noise)
            x = _clamp(x, 0.0, 1.0)
            y = _clamp(y, 0.0, 1.0)
        tt = t + int(round(move_s * 1000 * u)) + rng.randint(-3, 3)
        tt = max(last_t, tt)
        last_t = tt
        events.append({"type": "pointer", "t": tt, "x": round(x, 5), "y": round(y, 5)})

    end_t = last_t
    if rng.random() < params.click_prob:
        down_t = last_t + rng.randint(0, 60)
        dwell = max(20, _lognormal(rng, params.click_dwell_med, params.click_dwell_sigma))
        up_t = down_t + int(round(dwell))
        for action, when in (("down", down_t), ("up", up_t)):
            events.append(
                {
                    "type": "button",
                    "t": when,
                    "x": round(target[0], 5),
                    "y": round(target[1], 5),
                    "button": "left",
                    "action": action,
                }
            )
        end_t = up_t

    pause = _lognormal(rng, params.pause_med, params.pause_sigma)
    if rng.random() < params.pause_tail_prob:
        pause *= rng.uniform(3.0, 6.0)
    return end_t + int(round(pause * 1000)), target, events


def _typed_block(
    rng: random.Random, params: PersonaParams, t: int
) -> tuple[int, list[dict]]:
    n_chars = int(_clamp(round(_lognormal(rng, params.burst_len_med, params.burst_len_sigma)), 3, 18))
    text = _text_of_length(rng, n_chars)
    events: list[dict] = []
    t_cursor = t + 100
    insert_times: list[int] = []
    for ch in text:
        gap = _lognormal(rng, params.text_gap_med, params.text_gap_sigma)
        if rng.random() < 0.25:
            gap *= 0.55
        if rng.random() < 0.04:
            gap *= 3.0
        t_cursor += int(round(_clamp(gap, 40, 1500)))
        insert_times.append(t_cursor)
        events.append(
            {
                "type": "text",
                "t": t_cursor,
                "source": "typed",
                "kind": "insert",
                "text": ch,
            }
        )

    correction_starts: list[int] = []
    for ti in insert_times:
        if rng.random() < params.correction_rate:
            latency = _lognormal(rng, params.correction_latency_med, params.correction_latency_sigma)
            start = ti + int(round(_clamp(latency, 120, 6000)))
            size = int(
                _clamp(
                    round(_lognormal(rng, params.correction_size_med, params.correction_size_sigma)),
                    1,
                    5,
                )
            )
            gap = rng.randint(90, 160)
            for j in range(size):
                events.append(
                    {
                        "type": "text",
                        "t": start + j * gap,
                        "source": "typed",
                        "kind": "delete",
                        "chars": 1,
                    }
                )
            correction_starts.append(start)

    last_t = max(e["t"] for e in events)
    pause = _lognormal(rng, params.block_pause_med, 0.35)
    end_t = last_t + int(round(pause * 1000))
    if correction_starts:
        end_t += int(round(params.post_error_pause * 1000))
    return end_t, events


def _dictation_block(
    rng: random.Random, params: PersonaParams, t: int
) -> tuple[int, list[dict]]:
    events: list[dict] = []
    if rng.random() < params.onscreen_prob:
        word = rng.choice([w for w in WORDS if 3 <= len(w) <= 8])
        t_cursor = t + 120
        for ch in word:
            t_cursor += rng.randint(260, 430)
            events.append(
                {
                    "type": "text",
                    "t": t_cursor,
                    "source": "onscreen",
                    "kind": "insert",
                    "text": ch,
                }
            )
        pause = _lognormal(rng, params.block_pause_med, 0.35)
        return t_cursor + int(round(pause * 1000)), events

    length = int(
        _clamp(
            round(_lognormal(rng, params.burst_len_med, params.insert_len_sigma)),
            4,
            40,
        )
    )
    text = _text_of_length(rng, length)
    t_insert = t + 120
    events.append(
        {
            "type": "text",
            "t": t_insert,
            "source": "dictation",
            "kind": "insert",
            "text": text,
        }
    )

    last_t = t_insert
    corrected = False
    p_corr = _clamp(params.correction_rate * length / max(1.0, params.correction_size_med), 0.0, 0.65)
    if rng.random() < p_corr:
        corrected = True
        latency = _lognormal(rng, params.correction_latency_med, params.correction_latency_sigma)
        start = t_insert + int(round(_clamp(latency, 300, 9000)))
        size = int(
            _clamp(
                round(_lognormal(rng, params.correction_size_med, params.correction_size_sigma)),
                1,
                max(1, length),
            )
        )
        if rng.random() < 0.5:
            events.append(
                {
                    "type": "text",
                    "t": start,
                    "source": "onscreen",
                    "kind": "delete",
                    "chars": size,
                }
            )
            last_t = start
        else:
            for j in range(size):
                events.append(
                    {
                        "type": "text",
                        "t": start + j * 330,
                        "source": "onscreen",
                        "kind": "delete",
                        "chars": 1,
                    }
                )
            last_t = start + (size - 1) * 330

    pause = _lognormal(rng, params.block_pause_med, 0.35)
    end_t = last_t + int(round(pause * 1000))
    if corrected:
        end_t += int(round(params.post_error_pause * 1000))
    return end_t, events


def generate_session(
    style: str,
    persona: str,
    params: PersonaParams,
    session_index: int,
    seed: int,
    duration_ms: int = 60_000,
    owner_train_sessions: int = 15,
) -> list[dict]:
    """Generate one session as a list of JSONL-ready objects."""

    rng = random.Random(f"marrow:{seed}:{style}:{persona}:{session_index}")
    params = _jitter_params(rng, params)
    split = "test"
    label = "owner"
    if persona == "owner_rested":
        split = "train" if session_index < owner_train_sessions else "calibration"
        label = "owner"
    elif persona == "owner_tired":
        label = "owner"
    elif persona == "stranger":
        label = "impostor_stranger"
    elif persona == "mimic":
        label = "impostor_mimic"

    header = {
        "type": "recording",
        "schema": SCHEMA,
        "recording_id": f"{style}/{persona}/{session_index:02d}",
        "capture": "synthetic",
        "style": style,
        "persona": persona,
        "session": session_index,
        "split": split,
        "label": label,
        "duration_ms": duration_ms,
        "screen": {"width_px": 1920, "height_px": 1080},
        "notes": STYLE_DESCRIPTIONS[style],
    }

    events: list[dict] = []
    t = 600 + rng.randint(0, 500)
    pos = (rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75))
    guard = 0
    pointer_next = rng.random() < params.pointer_share
    while t < duration_ms - 1200 and guard < 10_000:
        guard += 1
        if pointer_next:
            t, pos, block = _pointer_block(rng, params, t, pos, duration_ms)
        else:
            if params.text_mode == "typed":
                t, block = _typed_block(rng, params, t)
            else:
                t, block = _dictation_block(rng, params, t)
        events.extend(block)
        # People alternate between pointing and text far more often than an
        # independent coin would suggest; this also keeps every 20 s window
        # informative about both modalities.
        pointer_next = (not pointer_next) if rng.random() < 0.72 else pointer_next

    events.sort(key=lambda e: (e["t"], 0 if e["type"] == "pointer" else 1))
    # Clamp to the advertised session length; blocks may run slightly over.
    events = [e for e in events if e["t"] < duration_ms]
    return [header] + events


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(
    out_root: str | Path,
    seed: int = 20261125,
    force: bool = False,
    sessions_rested: int = 20,
    sessions_tired: int = 6,
    sessions_impostor: int = 10,
) -> dict:
    """Write the full fake corpus and a manifest.  Deterministic in ``seed``."""

    out_root = Path(out_root)
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((out_root / item["path"]).exists() for item in manifest.get("files", [])):
            return manifest

    files: list[dict] = []
    for style in STYLES:
        owner = OWNER[style]
        tired = _tired_params(owner, style)
        mimic = mimic_from(owner)
        stranger = STRANGER[style]
        plan = [
            ("owner_rested", owner, sessions_rested),
            ("owner_tired", tired, sessions_tired),
            ("stranger", stranger, sessions_impostor),
            ("mimic", mimic, sessions_impostor),
        ]
        for persona, params, count in plan:
            folder = out_root / style / persona
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                rows = generate_session(
                    style,
                    persona,
                    params,
                    index,
                    seed,
                    owner_train_sessions=max(1, int(sessions_rested * 0.75)),
                )
                path = folder / f"session-{index:02d}.jsonl"
                with path.open("w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                header = rows[0]
                files.append(
                    {
                        "path": str(path.relative_to(out_root)),
                        "style": style,
                        "persona": persona,
                        "session": index,
                        "split": header["split"],
                        "label": header["label"],
                        "events": len(rows) - 1,
                        "sha256": _sha256(path),
                    }
                )

    manifest = {
        "schema": "marrow.fake-data/v1",
        "generator_seed": seed,
        "capture": "synthetic",
        "styles": list(STYLES),
        "style_descriptions": STYLE_DESCRIPTIONS,
        "sessions": {
            "owner_rested": sessions_rested,
            "owner_tired": sessions_tired,
            "stranger": sessions_impostor,
            "mimic": sessions_impostor,
        },
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def persona_table() -> list[dict]:
    """Human-readable parameter summary used in docs and reports."""

    rows: list[dict] = []
    for style in STYLES:
        owner = OWNER[style]
        for name, params in (
            ("owner", owner),
            ("owner_tired", _tired_params(owner, style)),
            ("stranger", STRANGER[style]),
            ("mimic", mimic_from(owner)),
        ):
            rows.append({"style": style, "persona": name, **asdict(params)})
    return rows
