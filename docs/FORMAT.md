# Marrow recording format (`marrow.recording/v1`)

Marrow reads a plain newline-delimited JSON (JSONL) file. One file is one
recording. The format is deliberately small, OS-neutral, and easy to produce
from a browser, a desktop app, or a synthetic generator.

## File rules

- UTF-8 text, one JSON object per line.
- Blank lines are ignored.
- The first non-blank line is the **header**.
- Every following line is one **event**.
- Timestamps are integer milliseconds relative to the start of the recording
  (not an epoch timestamp), and must be non-decreasing.
- Coordinates are fractions of the screen in `[0.0, 1.0]`, never pixels. `x`
  is measured from the left edge and `y` from the top edge.
- Unknown event types, unknown text sources, negative/zero delete counts, empty
  insert text, out-of-range coordinates, and decreasing timestamps are rejected.

Validation is strict so that a malformed capture fails loudly instead of
silently producing plausible-looking features. Use:

```bash
python3 -m marrow validate path/to/recording.jsonl
```

## Header

```json
{
  "type": "recording",
  "schema": "marrow.recording/v1",
  "recording_id": "mouse_keyboard/owner_rested/00",
  "capture": "synthetic",
  "style": "mouse_keyboard",
  "persona": "owner_rested",
  "session": 0,
  "split": "train",
  "label": "owner",
  "duration_ms": 60000,
  "screen": {"width_px": 1920, "height_px": 1080},
  "notes": "optional free text"
}
```

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | Must be `"recording"` |
| `schema` | yes | Must be `"marrow.recording/v1"` |
| `recording_id` | recommended | Stable identifier; defaults to the filename stem |
| `capture` | recommended | `synthetic`, `browser`, `mac`, `windows`, … — not used for scoring |
| `style` | optional | Free label such as `mouse_keyboard` or `assistive_dictation` |
| `persona` | optional | Free label for the recorded person/session |
| `split` | optional | `train`, `calibration`, or `test` (metadata for this demo) |
| `label` | optional | `owner`, `impostor_stranger`, `impostor_mimic`, … |
| `duration_ms` | optional | Declared session length in ms |
| `screen` | optional | Capture metadata; features never use pixel dimensions |

The model does **not** use `style`, `persona`, `split`, `label`, `capture`, or
`screen` for scoring. They are for organisation and evaluation only.

## Events

All events have `type` and integer `t` (ms).

### Pointer movement

```json
{"type":"pointer","t":806,"x":0.51234,"y":0.44102}
```

Emit one event per pointer sample. The sample rate may vary; the feature code
uses timestamps, not event counts. A device that reports at 30 Hz and one that
reports at 125 Hz are both acceptable.

### Clicks

```json
{"type":"button","t":1300,"x":0.612,"y":0.380,"button":"left","action":"down"}
{"type":"button","t":1395,"x":0.612,"y":0.380,"button":"left","action":"up"}
```

`button` is a string (`left`, `right`, `switch`, …); `action` is `down` or `up`.
Dwell time is derived from the down/up pair. Switch access and dwell clicking
can emit the same events.

### Text insertion

```json
{"type":"text","t":1400,"source":"typed","kind":"insert","text":"h"}
{"type":"text","t":1530,"source":"dictation","kind":"insert","text":"a whole dictated chunk"}
{"type":"text","t":2200,"source":"onscreen","kind":"insert","text":"k"}
```

- `source` describes how the text arrived: `typed` (physical keyboard),
  `onscreen` (on-screen keyboard or selection), or `dictation` (speech-to-text
  or other software insertion).
- `kind` is `insert`; `text` is the non-empty inserted string. A physical
  keystroke normally has length 1; dictation may insert a whole phrase.
- The model uses the **length** and **timing** of the insertion, not the
  characters. A production capture layer can discard or hash the content.

### Text deletion / correction

```json
{"type":"text","t":2600,"source":"typed","kind":"delete","chars":1}
{"type":"text","t":3100,"source":"onscreen","kind":"delete","chars":6}
```

- `kind` is `delete`; `chars` is the positive number of characters removed.
- A backspace is `chars: 1`. Selecting a word and deleting it can be one event
  with `chars: 6`. Multiple on-screen backspaces can be separate events.
- Corrections are part of the behavioural signal: their timing, size, and the
  pause afterwards are used by the model.

## Complete small example

```json
{"type":"recording","schema":"marrow.recording/v1","recording_id":"example/one","capture":"synthetic","style":"mouse_keyboard","persona":"owner_rested","split":"train","label":"owner","duration_ms":4000}
{"type":"pointer","t":100,"x":0.20,"y":0.50}
{"type":"pointer","t":120,"x":0.21,"y":0.50}
{"type":"pointer","t":140,"x":0.23,"y":0.49}
{"type":"pointer","t":160,"x":0.26,"y":0.49}
{"type":"button","t":180,"x":0.26,"y":0.49,"button":"left","action":"down"}
{"type":"button","t":275,"x":0.26,"y":0.49,"button":"left","action":"up"}
{"type":"text","t":400,"source":"typed","kind":"insert","text":"h"}
{"type":"text","t":540,"source":"typed","kind":"insert","text":"e"}
{"type":"text","t":680,"source":"typed","kind":"insert","text":"l"}
{"type":"text","t":820,"source":"typed","kind":"insert","text":"l"}
{"type":"text","t":960,"source":"typed","kind":"insert","text":"o"}
{"type":"text","t":1100,"source":"typed","kind":"insert","text":"!"}
{"type":"text","t":1250,"source":"typed","kind":"delete","chars":1}
```

## Privacy and OS neutrality

- No key codes, scan codes, window titles, app names, or absolute screen pixels
  are part of the format.
- No audio or camera data is involved.
- Text content is optional for the model; only timing/length is used. If a
  capture adapter must store content, encrypt it and delete it after feature
  extraction.
- The same events can come from a browser, a native OS hook, a head mouse, an
  eye tracker, a switch, or a synthetic generator. The format does not encode a
  device assumption; the feature layer adapts to whichever modalities are
  present.
