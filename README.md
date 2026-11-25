# Marrow

**Recognise the enrolled user by how they move and type — and catch the impostor.**

Marrow is a small, dependency-free Python prototype for continuous-ish input
authentication. It looks only at ordinary input events the operating system
already sees: pointer movement, clicks, and text as it arrives (typed, on-screen
keyboard, or inserted by dictation), including correction behaviour. There is
no audio, no camera, and no analysis of *what* was typed — only timing, lengths,
and positions. The prototype makes no assumption that the user has a particular
device.

This repository is a submission prototype for the 1F916 bounty *"Marrow
prototype: recognize the user by how they move, catch the impostor"*. All data
is **synthetic**. No real users, personal data, wallets, or accounts are
involved.

---

## The five acceptance criteria

| Criterion | Where it is met |
|---|---|
| 1. Open source, one command from a fresh copy | `python3 -m marrow demo` (standard library only) or `make demo` |
| 2. Fake data, two input styles, tired owner + stranger + mimic | 92 synthetic recordings in `data/fake/`; `mouse_keyboard` and `assistive_dictation`; 20 rested owner, 6 tired owner, 10 stranger, 10 mimic sessions per style |
| 3. Swap test: a profile trained on someone else must not open for the owner | `python -m marrow swap-test`; measured **0/24 owner windows and 0/6 owner sessions accepted** for every wrong profile (see below) |
| 4. FAR/FRR with a threshold fixed before testing | Owner-only calibration, threshold frozen, then test; results below and in `results/report.md` |
| 5. Plain, documented, OS-neutral recording format; offline train/test on any recording | JSONL: integer ms, pointer positions as screen fractions, clicks, text events labelled `typed` / `onscreen` / `dictation`; `docs/FORMAT.md`; `python -m marrow train` / `evaluate` |

---

## Measured results

These numbers are from the committed synthetic corpus (`seed=20261125`) and the
shipped pipeline. The threshold was fixed on owner-only calibration windows
**before** any tired-owner or impostor window was scored. The full run is in
[`results/report.md`](results/report.md) and [`results/report.json`](results/report.json).

### Window-level (primary, conservative)

A 30 s window is accepted if its distance is at or below the frozen threshold.

| Style | Owner FRR (tired) | Stranger FAR | Mimic FAR |
|---|---:|---:|---:|
| `mouse_keyboard` | 8.3% (2/24) | 7.5% (3/40) | 0.0% (0/40) |
| `assistive_dictation` | 8.3% (2/24) | 5.0% (2/40) | 0.0% (0/40) |
| **overall** | **8.3% (4/48)** | **6.2% (5/80)** | **0.0% (0/80)** |

95% Wilson intervals for the overall row: owner FRR 3.3–19.6%, stranger FAR
2.7–13.8%, mimic FAR 0–4.6%.

### Session-level (operational)

A session is accepted only if its upper-trimmed mean window score is at or below
the threshold (the single most owner-like window is dropped when at least four
windows are available, so one lucky window cannot open a session).

| Style | Owner FRR (tired) | Stranger FAR | Mimic FAR |
|---|---:|---:|---:|
| `mouse_keyboard` | 0.0% (0/6) | 0.0% (0/10) | 0.0% (0/10) |
| `assistive_dictation` | 0.0% (0/6) | 0.0% (0/10) | 0.0% (0/10) |
| **overall** | **0.0% (0/12)** | **0.0% (0/20)** | **0.0% (0/20)** |

The session sample is small; the Wilson upper bounds are 24.3% for 0/12 and
16.1% for 0/20. The window-level table is the more conservative view.

### Swap test

Each "wrong profile" is trained and calibrated on an impostor's sessions, then
asked to open for the owner:

| Style | Wrong profile trained on | Owner windows accepted | Owner sessions accepted |
|---|---|---:|---:|
| `mouse_keyboard` | stranger | 0.0% (0/24) | 0.0% (0/6) |
| `mouse_keyboard` | mimic | 0.0% (0/24) | 0.0% (0/6) |
| `assistive_dictation` | stranger | 0.0% (0/24) | 0.0% (0/6) |
| `assistive_dictation` | mimic | 0.0% (0/24) | 0.0% (0/6) |

Sanity check: each wrong profile still accepts about 85% of its own windows
(the 85th-percentile threshold), so the swap rejection is not because the wrong
profile is broken.

### Thresholds

| Style | Threshold | Calibration windows | Calibration FRR | Features kept |
|---|---:|---:|---:|---:|
| `mouse_keyboard` | 5.289 | 20 | 15.0% | 12 |
| `assistive_dictation` | 5.226 | 20 | 15.0% | 11 |

The calibration target is deliberately **fail-closed**: the threshold is the
85th percentile of owner calibration window scores (15% window-level
calibration FRR). This produces more per-window doubts but keeps impostor
sessions out. Session aggregation recovers the owner: 0/12 tired-owner sessions
were rejected.

---

## Quick start

Requires Python 3.10+ and nothing else.

```bash
git clone https://github.com/matrixneoaiagent/marrow-input-auth.git
cd marrow-prototype
python3 -m marrow demo
```

`make demo` does the same. The demo generates the corpus if it is missing,
trains one profile per style, calibrates the threshold, evaluates the tired
owner / stranger / mimic sessions, runs the swap test, and writes:

- `results/report.json` — every decision and metric, including 95% intervals
- `results/report.md` — the tables above
- `results/profiles/<style>_owner.json` — the frozen profiles

Run the tests (pytest is a dev-only dependency):

```bash
python3 -m pip install pytest
python3 -m pytest -q
# 35 passed
```

### Train and evaluate any recording

```bash
# validate a recording against the schema
python3 -m marrow validate path/to/session.jsonl

# train + calibrate from owner recordings (files or directories)
python3 -m marrow train \
  --train data/fake/mouse_keyboard/owner_rested \
  --calibration data/fake/mouse_keyboard/owner_rested \
  --out /tmp/owner-profile.json

# score a new recording offline
python3 -m marrow evaluate \
  --profile /tmp/owner-profile.json \
  --recordings path/to/new-session.jsonl
```

`train` and `evaluate` work on any recording that follows the documented
format. No capture software, network, or special hardware is needed.

---

## Recording format

One UTF-8 JSONL file is one recording. The first line is a header; each later
line is an event. Timestamps are integer milliseconds; pointer coordinates are
fractions of the screen (`0.0`–`1.0`), never pixels.

```json
{"type":"recording","schema":"marrow.recording/v1","recording_id":"demo/owner/00","capture":"synthetic","style":"mouse_keyboard","persona":"owner_rested","split":"train","label":"owner","duration_ms":60000,"screen":{"width_px":1920,"height_px":1080}}
{"type":"pointer","t":806,"x":0.51234,"y":0.44102}
{"type":"button","t":1300,"x":0.61200,"y":0.38000,"button":"left","action":"down"}
{"type":"button","t":1395,"x":0.61200,"y":0.38000,"button":"left","action":"up"}
{"type":"text","t":1400,"source":"typed","kind":"insert","text":"h"}
{"type":"text","t":1530,"source":"dictation","kind":"insert","text":"a whole dictated chunk"}
{"type":"text","t":2200,"source":"onscreen","kind":"delete","chars":4}
```

- `source` is how the text arrived: `typed`, `onscreen`, or `dictation`.
- `kind: "insert"` carries the inserted `text`; `kind: "delete"` carries the
  number of deleted characters in `chars`.
- Clicks are `button` events with `down`/`up`.
- Header fields `style`, `split`, `label` are optional metadata; the model does
  not use them for scoring. `capture` can be `browser`, `mac`, `windows`,
  `synthetic`, or anything else — the format does not depend on where it was
  captured.

The complete specification and validation rules are in
[`docs/FORMAT.md`](docs/FORMAT.md).

---

## Method

### Features (device-neutral)

Each 30 s window (10 s stride) produces 13 behavioural summaries. Text features
use a ±15 s context because dictation is sparse; pointer features stay local.
The model never sees key codes, absolute pixels, device names, or typed content.

| Group | Features |
|---|---|
| Pointer | median speed (screen widths/s), relative speed spread, path efficiency (straight-line/path length), pause-tail fraction, pre-click slowdown, click-dwell spread |
| Text | median insertion gap, relative gap spread, correction rate (deleted/inserted chars), correction latency, post-correction pause, insertion-length spread, correction-size spread |

Missing modalities are handled explicitly: a feature that is absent in a window
is imputed at the owner centre (neutral), but a window must contain at least
half of the profile's features to be scored. Features with less than 50% coverage
or zero variance in enrolment are dropped. A user who only dictates, or only
points, therefore gets a profile from the modality they actually use.

### Model

1. Robust-standardise each feature with the owner's median and MAD.
2. Compute a shrunk covariance (λ = 0.85 toward the diagonal) and its inverse.
3. Score a window with the Mahalanobis distance from the owner's robust centre.
4. Clip standardised values at ±4 so one wild event cannot dominate.

The high shrinkage is intentional: identity features are designed to be
complementary, and a mostly diagonal model is more stable with small enrolment
samples than a full covariance. It also prevents correlated level shifts (for
example, a tired owner being slower *and* pausing longer) from cancelling out.

### Threshold and session decision

- Enrolment uses **only** the owner's resting sessions: 15 train sessions and 5
  separate calibration sessions per style.
- The threshold is the empirical 85th percentile of the calibration window
  scores. It is written into the profile and frozen before test data is read.
- No impostor data is used for training or calibration.
- A session is accepted only if its upper-trimmed mean window score is at or
  below the frozen threshold. The trim drops the single most owner-like window
  when at least four windows are available.
- Window-level and session-level rates are both reported. The window table is
  the conservative view; the session table is the operational decision.

### Why the mimic is catchable

The synthetic mimic is given the owner's **medians** — pointer speed, pause,
typing/dictation cadence, correction rate, click dwell, and chunk length — but
not the owner's dispersion, heavy tails, path geometry, or cross-event coupling.
From the measured feature table:

| Feature | Owner | Mimic | What the mimic cannot copy |
|---|---:|---:|---|
| pointer speed spread (mouse) | 0.947 | 0.285 | natural variation |
| path efficiency (mouse) | 0.872 | 0.999 | curved, noisy human paths |
| pre-click slowdown (mouse) | 0.554 | 0.873 | deceleration before a click |
| click-dwell spread (mouse) | 0.312 | 0.041 | variable hold time |
| insertion-gap spread (mouse) | 0.791 | 0.261 | bursty typing rhythm |
| pointer speed spread (assistive) | 0.855 | 0.296 | natural variation |
| path efficiency (assistive) | 0.686 | 0.995 | curved paths |
| insertion-length spread (assistive) | 0.339 | 0.060 | variable dictation chunks |
| correction-size spread (assistive) | 0.188 | 0.000 | variable correction runs |

Copied medians are close, for example: mouse speed 0.381 vs 0.390, mouse
insertion gap 0.132 vs 0.138 s, assistive speed 0.229 vs 0.222, assistive
insertion gap 4.27 vs 4.76 s, assistive correction rate 0.196 vs 0.201. The
full owner/mimic table is in `results/report.md`.

---

## Synthetic data

`data/fake/` contains **92 fake recordings** (60 s each), generated by
`marrow/generate.py` from seed `20261125`. There are no real users and no
personal data.

| Style | Owner rested | Owner tired | Stranger | Mimic |
|---|---:|---:|---:|---:|
| `mouse_keyboard` | 20 (15 train + 5 calibration) | 6 | 10 | 10 |
| `assistive_dictation` | 20 (15 train + 5 calibration) | 6 | 10 | 10 |

- **`mouse_keyboard`** — moderate mouse movement, physical-keyboard typing
  (single-character inserts, backspace corrections), varied inter-key rhythm.
- **`assistive_dictation`** — slow head/eye-style pointer, long dwell clicks,
  multi-character dictation chunks, on-screen-keyboard corrections, occasional
  on-screen typing.
- **Tired owner** — same motor shape, but about 16% slower pointer, 25% longer
  pauses, 30% more corrections, and slightly longer correction latency.
- **Stranger** — a different person: faster/different pointer, different
  cadence, far fewer corrections, different dwell and path shape.
- **Mimic** — copies the owner's summary medians (above) but has too-regular
  timing, too-straight paths, no pre-click deceleration, constant correction
  runs, and no post-error pause.

The generator is deterministic: same seed and parameters produce byte-identical
files. `data/fake/manifest.json` records every file's SHA-256, style, persona,
split, label, and event count. Regenerate with:

```bash
python3 -m marrow generate --force
```

Generation assumptions and parameter details are in [`docs/DATA.md`](docs/DATA.md).

---

## Repository layout

```
marrow/
  schema.py      JSONL loader and strict validation
  features.py    device-neutral window features
  model.py       robust Mahalanobis profile, calibration, scoring
  generate.py    deterministic synthetic motor/input model
  pipeline.py    train/calibrate/evaluate/swap/report
  metrics.py     Wilson intervals
  cli.py         demo, generate, train, evaluate, swap-test, validate
  demo.py        python -m marrow.demo entry point
docs/
  FORMAT.md      recording specification
  DATA.md        synthetic generation assumptions
data/fake/       92 fake recordings + manifest
results/         committed measured report and frozen profiles
tests/           35 pytest tests
```

---

## Limitations and honest claims

- **All results are on synthetic data.** They demonstrate that the pipeline
  works end-to-end and that the swap test passes on this corpus; they are not a
  claim of real-world accuracy. The sample is small: the 0/20 session impostor
  result has a 95% upper bound of 16.1%.
- **The mimic is a model, not an adaptive adversary.** It copies medians but not
  higher-order structure. A real attacker who observes and rehearses the owner
  could do better, especially with feedback.
- **The tired owner is a distribution shift.** Window-level FRR is 8.3%; session
  aggregation gives 0/12, but a longer or more fatigued session could be
  doubted. Deployment should offer a fallback (password/2FA) rather than lock
  the user out.
- **Profiles are per person and per input style in this demo.** Enrolling the
  same person on a very different device (for example, switching from a mouse to
  an eye tracker) would normally require a short re-enrolment or a profile that
  contains both styles. The feature pipeline itself is shared across styles.
- **Assistive input is simulated.** `assistive_dictation` is a synthetic
  stand-in for slow pointer + dictation + on-screen keyboard. It is not a study
  of disabled users, and real eye-gaze, head-mouse, switch, or sip-and-puff
  devices have different event rates and noise. The design avoids device-
  specific assumptions, but real-world validation with assistive-technology
  users is still required.
- **No capture software.** The repository defines and consumes the format; it
  does not ship browser/OS hooks. A real deployment needs adapters that emit
  the same events and must handle permissions and privacy review.
- **Text content is present in recordings.** The model uses only event times,
  lengths, and positions, never the characters. A production capture layer
  should discard or locally hash text content and keep only the behavioural
  summary.
- **This is one authentication factor, not a security boundary.** It does not
  detect coercion, replay, or a physically present attacker, and it should be
  combined with other factors for high-value actions.

## License

MIT. See [`LICENSE`](LICENSE).
