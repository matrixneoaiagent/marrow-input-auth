"""Reading and validation for Marrow recording files.

The on-disk format is newline-delimited JSON (JSONL).  The first line is a
recording header; every following line is one input event.  See
``docs/FORMAT.md`` for the complete specification.

The loader is deliberately strict: a malformed recording should fail loudly
rather than silently producing plausible-looking features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "marrow.recording/v1"
TEXT_SOURCES = {"typed", "onscreen", "dictation"}
BUTTON_ACTIONS = {"down", "up"}


class RecordingError(ValueError):
    """Raised when a recording does not follow the Marrow schema."""


@dataclass
class Event:
    t: int
    type: str
    x: float | None = None
    y: float | None = None
    button: str | None = None
    action: str | None = None
    source: str | None = None
    kind: str | None = None
    text: str | None = None
    chars: int | None = None


@dataclass
class Recording:
    path: Path
    header: dict[str, Any]
    events: list[Event] = field(default_factory=list)

    @property
    def recording_id(self) -> str:
        return str(self.header.get("recording_id", self.path.stem))

    @property
    def style(self) -> str:
        return str(self.header.get("style", "unspecified"))

    @property
    def persona(self) -> str:
        return str(self.header.get("persona", "unspecified"))

    @property
    def split(self) -> str:
        return str(self.header.get("split", "test"))

    @property
    def label(self) -> str:
        return str(self.header.get("label", "unlabelled"))

    @property
    def duration_ms(self) -> int:
        declared = int(self.header.get("duration_ms", 0))
        if self.events:
            return max(declared, max(0, self.events[-1].t))
        return declared

    def counts(self) -> dict[str, int]:
        out = {"pointer": 0, "button": 0, "text_insert": 0, "text_delete": 0}
        for e in self.events:
            if e.type == "pointer":
                out["pointer"] += 1
            elif e.type == "button":
                out["button"] += 1
            elif e.type == "text" and e.kind == "insert":
                out["text_insert"] += 1
            elif e.type == "text" and e.kind == "delete":
                out["text_delete"] += 1
        return out


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordingError(message)


def _parse_header(obj: Any, path: Path, line_no: int) -> dict[str, Any]:
    _require(isinstance(obj, dict), f"{path}:{line_no}: header must be a JSON object")
    assert isinstance(obj, dict)
    _require(obj.get("type") == "recording", f"{path}:{line_no}: first line must be type='recording'")
    _require(obj.get("schema") == SCHEMA, f"{path}:{line_no}: schema must be {SCHEMA!r}")
    duration = obj.get("duration_ms")
    if duration is not None:
        _require(
            isinstance(duration, int) and duration >= 0,
            f"{path}:{line_no}: duration_ms must be a non-negative integer",
        )
    for key in ("recording_id", "style", "persona", "split", "label"):
        if key in obj:
            _require(isinstance(obj[key], str), f"{path}:{line_no}: {key} must be a string")
    screen = obj.get("screen", {})
    _require(isinstance(screen, dict), f"{path}:{line_no}: screen must be an object")
    return obj


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_event(obj: Any, path: Path, line_no: int, last_t: int) -> Event:
    where = f"{path}:{line_no}"
    _require(isinstance(obj, dict), f"{where}: event must be a JSON object")
    assert isinstance(obj, dict)
    _require(isinstance(obj.get("t"), int) and obj["t"] >= 0, f"{where}: t must be a non-negative integer (ms)")
    t = int(obj["t"])
    _require(t >= last_t, f"{where}: timestamps must be non-decreasing ({t} after {last_t})")

    etype = obj.get("type")
    if etype == "pointer":
        x, y = obj.get("x"), obj.get("y")
        _require(_number(x) and 0.0 <= float(x) <= 1.0, f"{where}: pointer x must be a fraction in [0,1]")
        _require(_number(y) and 0.0 <= float(y) <= 1.0, f"{where}: pointer y must be a fraction in [0,1]")
        return Event(t=t, type="pointer", x=float(x), y=float(y))

    if etype == "button":
        x, y = obj.get("x"), obj.get("y")
        _require(_number(x) and 0.0 <= float(x) <= 1.0, f"{where}: button x must be a fraction in [0,1]")
        _require(_number(y) and 0.0 <= float(y) <= 1.0, f"{where}: button y must be a fraction in [0,1]")
        _require(isinstance(obj.get("button"), str), f"{where}: button must be a string")
        _require(obj.get("action") in BUTTON_ACTIONS, f"{where}: action must be 'down' or 'up'")
        return Event(
            t=t,
            type="button",
            x=float(x),
            y=float(y),
            button=str(obj["button"]),
            action=str(obj["action"]),
        )

    if etype == "text":
        source = obj.get("source")
        kind = obj.get("kind")
        _require(source in TEXT_SOURCES, f"{where}: source must be one of {sorted(TEXT_SOURCES)}")
        _require(kind in {"insert", "delete"}, f"{where}: kind must be 'insert' or 'delete'")
        if kind == "insert":
            text = obj.get("text")
            _require(isinstance(text, str) and len(text) > 0, f"{where}: insert text must be a non-empty string")
            return Event(t=t, type="text", source=str(source), kind="insert", text=str(text))
        chars = obj.get("chars")
        _require(isinstance(chars, int) and not isinstance(chars, bool) and chars > 0, f"{where}: delete chars must be a positive integer")
        return Event(t=t, type="text", source=str(source), kind="delete", chars=int(chars))

    raise RecordingError(f"{where}: unknown event type {etype!r}")


def load_recording(path: str | Path) -> Recording:
    """Load and validate one JSONL recording."""

    path = Path(path)
    header: dict[str, Any] | None = None
    events: list[Event] = []
    last_t = 0
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RecordingError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if header is None:
                header = _parse_header(obj, path, line_no)
                continue
            event = _parse_event(obj, path, line_no, last_t)
            last_t = event.t
            events.append(event)
    if header is None:
        raise RecordingError(f"{path}: empty recording")
    return Recording(path=path, header=header, events=events)


def discover_recordings(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of .jsonl files."""

    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"recording path not found: {path}")
    # Preserve caller order while dropping duplicate paths.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_recordings(paths: Iterable[str | Path]) -> list[Recording]:
    return [load_recording(path) for path in discover_recordings(paths)]
