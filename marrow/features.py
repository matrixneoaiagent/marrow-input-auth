"""Device-neutral feature extraction from Marrow recordings.

Every feature is a summary of *how* input arrived, never of *what* was typed.
Positions are fractions of the screen, so the features do not depend on screen
size, operating system, or capture source.  The same code path handles physical
keyboards, on-screen keyboards, dictation, and pointer devices ranging from a
mouse to an eye tracker.

Features are computed over short windows (default 20 s).  A feature is ``None``
when the window lacks enough evidence for it; the model imputes a neutral value
and records which features were present.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterable

from .schema import Event, Recording

FEATURE_NAMES: tuple[str, ...] = (
    "pointer_speed_med",       # screen widths / second
    "pointer_speed_disp",      # IQR / median speed
    "path_efficiency",         # straight-line distance / travelled path
    "pause_tail",              # fraction of inter-burst pauses > 2x median
    "preclick_slowdown",       # end-of-move speed / burst median speed
    "click_dwell_disp",        # IQR / median button hold time
    "text_gap_med",            # seconds between text insertions
    "text_gap_disp",           # IQR / median insertion gap
    "correction_rate",         # deleted characters / inserted characters
    "correction_latency_med",  # seconds from insertion to first deletion
    "post_error_pause",        # seconds from a correction to the next insertion
    "insert_len_disp",         # IQR / median characters per insertion
    "correction_size_disp",    # IQR / median characters per correction
)

FEATURE_DESCRIPTIONS = {
    "pointer_speed_med": "median pointer speed while moving (screen widths/s)",
    "pointer_speed_disp": "relative spread (IQR/median) of pointer speed",
    "path_efficiency": "median straight-line/path-length ratio of movement bursts",
    "pause_tail": "fraction of between-burst pauses longer than twice the median",
    "preclick_slowdown": "speed in the last 150 ms before a click, relative to burst speed",
    "click_dwell_disp": "relative spread (IQR/median) of button hold durations",
    "text_gap_med": "median gap between text insertions (s)",
    "text_gap_disp": "relative spread (IQR/median) of insertion gaps",
    "correction_rate": "deleted characters per inserted character",
    "correction_latency_med": "median delay from insertion to correction (s)",
    "post_error_pause": "median pause after a correction (s)",
    "insert_len_disp": "relative spread (IQR/median) of characters per insertion",
    "correction_size_disp": "relative spread (IQR/median) of characters per correction",
}


def _quantile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("quantile of empty sequence")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def _dispersion(values: Iterable[float]) -> float | None:
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return None
    med = _median(xs)
    if med <= 0:
        return None
    return (_quantile(xs, 0.75) - _quantile(xs, 0.25)) / med


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return num / den


@dataclass
class Burst:
    start: int
    end: int
    points: list[Event]
    path_len: float
    displacement: float
    speeds: list[float]

    @property
    def speed_med(self) -> float | None:
        if not self.speeds:
            return None
        return _median(self.speeds)


def _pointer_bursts(moves: list[Event]) -> list[Burst]:
    if not moves:
        return []
    groups: list[list[Event]] = [[moves[0]]]
    for prev, cur in zip(moves, moves[1:]):
        if cur.t - prev.t > 250:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    bursts: list[Burst] = []
    for group in groups:
        if len(group) < 4:
            continue
        path_len = 0.0
        for a, b in zip(group, group[1:]):
            path_len += math.hypot(b.x - a.x, b.y - a.y)  # type: ignore[operator]
        displacement = math.hypot(
            group[-1].x - group[0].x,  # type: ignore[operator]
            group[-1].y - group[0].y,  # type: ignore[operator]
        )
        speeds: list[float] = []
        for a, b in zip(group, group[1:]):
            dt = (b.t - a.t) / 1000.0
            if dt <= 0 or dt > 0.25:
                continue
            dist = math.hypot(b.x - a.x, b.y - a.y)  # type: ignore[operator]
            speeds.append(dist / dt)
        bursts.append(
            Burst(
                start=group[0].t,
                end=group[-1].t,
                points=group,
                path_len=path_len,
                displacement=displacement,
                speeds=speeds,
            )
        )
    return bursts


def _pair_clicks(events: list[Event]) -> list[tuple[Event, Event]]:
    """Return (down, up) pairs.  Unpaired buttons are ignored."""

    open_buttons: dict[str, Event] = {}
    pairs: list[tuple[Event, Event]] = []
    for event in events:
        if event.type != "button":
            continue
        if event.action == "down":
            open_buttons[event.button or "left"] = event
        elif event.action == "up" and (event.button or "left") in open_buttons:
            down = open_buttons.pop(event.button or "left")
            pairs.append((down, event))
    return pairs


def _window_features(rec: Recording, start: int, end: int) -> dict[str, float | None]:
    f: dict[str, float | None] = {name: None for name in FEATURE_NAMES}

    # Pointer moves in and just before the window.  The context point lets us
    # measure speed at the window edge; it is never counted as an event.
    moves = [
        e for e in rec.events
        if e.type == "pointer" and start - 500 <= e.t < end
    ]
    bursts = _pointer_bursts(moves)
    bursts = [b for b in bursts if b.end >= start]

    speeds: list[float] = []
    for a, b in zip(moves, moves[1:]):
        if not (start <= b.t < end):
            continue
        dt = (b.t - a.t) / 1000.0
        if dt <= 0 or dt > 0.4:
            continue
        dist = math.hypot(b.x - a.x, b.y - a.y)  # type: ignore[operator]
        if dist > 0:
            speeds.append(dist / dt)
    if len(speeds) >= 5:
        f["pointer_speed_med"] = _median(speeds)
    if len(speeds) >= 8:
        f["pointer_speed_disp"] = _dispersion(speeds)

    efficiencies = [
        b.displacement / b.path_len
        for b in bursts
        if b.path_len > 1e-9
    ]
    efficiencies = [min(1.0, max(0.0, x)) for x in efficiencies]
    if efficiencies:
        f["path_efficiency"] = _median(efficiencies)

    if len(bursts) >= 3:
        gaps = [b.start - a.end for a, b in zip(bursts, bursts[1:]) if b.start > a.end]
        if len(gaps) >= 3:
            med_gap = _median(gaps)
            if med_gap > 0:
                f["pause_tail"] = sum(1 for g in gaps if g > 2 * med_gap) / len(gaps)

    clicks = _pair_clicks([e for e in rec.events if e.type == "button" and start <= e.t < end])
    if len(clicks) >= 3:
        dwells = [up.t - down.t for down, up in clicks if 10 <= up.t - down.t <= 5000]
        if len(dwells) >= 3:
            f["click_dwell_disp"] = _dispersion(dwells)

    preclick: list[float] = []
    for down, _up in clicks:
        burst = None
        for b in bursts:
            if b.end <= down.t and down.t - b.end <= 250:
                burst = b
                break
        if burst is None:
            continue
        burst_speed = burst.speed_med
        if burst_speed is None or burst_speed <= 0:
            continue
        tail_speeds: list[float] = []
        for a, b in zip(burst.points, burst.points[1:]):
            if b.t < down.t - 150 or b.t > down.t:
                continue
            dt = (b.t - a.t) / 1000.0
            if dt <= 0:
                continue
            tail_speeds.append(math.hypot(b.x - a.x, b.y - a.y) / dt)  # type: ignore[operator]
        if tail_speeds:
            preclick.append(_median(tail_speeds) / burst_speed)
    if preclick:
        f["preclick_slowdown"] = _median(preclick)

    # Text events.  Text is sparse, especially for dictation, so the text
    # features use a +/-15 s context around the 30 s window.  Pointer features
    # remain local to the window.  This is still OS-neutral: it is the same
    # person's nearby behaviour, just with enough events to estimate a rhythm.
    ctx = 15_000
    t_lo = max(0, start - ctx)
    t_hi = end + ctx
    inserts_ctx = [
        e for e in rec.events
        if e.type == "text" and e.kind == "insert" and t_lo <= e.t < t_hi
    ]
    deletes_ctx = [
        e for e in rec.events
        if e.type == "text" and e.kind == "delete" and t_lo <= e.t < t_hi
    ]

    # For dictation, most insertions are multi-character chunks; for typing
    # they are single characters.  Using multi-character insertions when they
    # exist keeps the rhythm estimate about chunk cadence rather than mixing in
    # one-character on-screen corrections.
    chunks = [e for e in inserts_ctx if len(e.text or "") > 1]
    rhythm = chunks if len(chunks) >= 2 else inserts_ctx
    gaps = [
        (b.t - a.t) / 1000.0
        for a, b in zip(rhythm, rhythm[1:])
        if 0 < b.t - a.t <= 6000
    ]
    if len(rhythm) >= 2 and gaps:
        f["text_gap_med"] = _median(gaps)
    if len(gaps) >= 3:
        f["text_gap_disp"] = _dispersion(gaps)

    inserted_chars = sum(len(e.text or "") for e in inserts_ctx)
    deleted_chars = sum(e.chars or 0 for e in deletes_ctx)
    if inserted_chars > 0:
        f["correction_rate"] = deleted_chars / inserted_chars

    runs: list[list[Event]] = []
    for event in deletes_ctx:
        if runs and event.t - runs[-1][-1].t <= 700:
            runs[-1].append(event)
        else:
            runs.append([event])

    if len(runs) >= 2:
        sizes = [sum(e.chars or 0 for e in run) for run in runs]
        f["correction_size_disp"] = _dispersion(sizes)

    latencies: list[float] = []
    post_pauses: list[float] = []
    for run in runs:
        prior = [e for e in inserts_ctx if e.t <= run[0].t]
        if prior and run[0].t - prior[-1].t <= 12_000:
            size = sum(e.chars or 0 for e in run)
            last = prior[-1]
            if len(last.text or "") > 1:
                corrected = last  # a dictation chunk: the whole chunk was just inserted
            else:
                corrected = prior[-min(max(1, size), len(prior))]
            latencies.append((run[0].t - corrected.t) / 1000.0)
        follow = [
            e for e in inserts_ctx
            if run[-1].t < e.t <= min(t_hi, run[-1].t + 10_000)
        ]
        if follow:
            post_pauses.append((follow[0].t - run[-1].t) / 1000.0)
    if latencies:
        f["correction_latency_med"] = _median(latencies)
    if post_pauses:
        f["post_error_pause"] = _median(post_pauses)

    lengths = [len(e.text or "") for e in (chunks if len(chunks) >= 3 else inserts_ctx)]
    if len(lengths) >= 3:
        f["insert_len_disp"] = _dispersion(lengths)

    # Round to keep profiles and reports readable without hiding structure.
    return {k: (None if v is None else round(float(v), 6)) for k, v in f.items()}


@dataclass
class Window:
    recording_id: str
    style: str
    persona: str
    split: str
    label: str
    start_ms: int
    end_ms: int
    n_events: int
    features: dict[str, float | None] = field(default_factory=dict)

    @property
    def present(self) -> list[str]:
        return [k for k, v in self.features.items() if v is not None]


def extract_windows(
    rec: Recording,
    window_ms: int = 30_000,
    stride_ms: int = 10_000,
    min_events: int = 20,
) -> list[Window]:
    """Split a recording into overlapping windows and extract features."""

    if window_ms <= 0 or stride_ms <= 0:
        raise ValueError("window_ms and stride_ms must be positive")
    duration = max(rec.duration_ms, int(rec.header.get("duration_ms", 0)))
    if duration <= 0:
        return []
    if duration <= window_ms:
        spans = [(0, duration)]
    else:
        spans = [
            (start, start + window_ms)
            for start in range(0, duration - window_ms + 1, stride_ms)
        ]
    out: list[Window] = []
    for start, end in spans:
        n_events = sum(1 for e in rec.events if start <= e.t < end)
        if n_events < min_events:
            continue
        features = _window_features(rec, start, end)
        if not any(v is not None for v in features.values()):
            continue
        out.append(
            Window(
                recording_id=rec.recording_id,
                style=rec.style,
                persona=rec.persona,
                split=rec.split,
                label=rec.label,
                start_ms=start,
                end_ms=end,
                n_events=n_events,
                features=features,
            )
        )
    return out


def extract_all(
    recordings: Iterable[Recording],
    window_ms: int = 30_000,
    stride_ms: int = 10_000,
    min_events: int = 20,
) -> list[Window]:
    windows: list[Window] = []
    for rec in recordings:
        windows.extend(extract_windows(rec, window_ms=window_ms, stride_ms=stride_ms, min_events=min_events))
    return windows
