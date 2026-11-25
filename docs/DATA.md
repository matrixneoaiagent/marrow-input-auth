# Synthetic data generation

**Everything under `data/fake/` is fake.** It is produced by
`marrow/generate.py` from a deterministic motor/input model. There are no real
users, recordings, personal data, or captured text.

## Corpus shape

- Two input styles:
  - `mouse_keyboard`: ordinary mouse + physical keyboard, single-character
    typing, backspace corrections.
  - `assistive_dictation`: slow head/eye-style pointer, long dwell clicks,
    multi-character dictation chunks, on-screen-keyboard corrections, and some
    on-screen typing.
- Per style:
  - **Owner, rested:** 20 sessions (15 train, 5 calibration).
  - **Owner, tired:** 6 sessions (test only).
  - **Stranger:** 10 sessions (test only).
  - **Mimic:** 10 sessions (test only).
- Each session is 60 s and is written as one JSONL recording.
- Total: 92 recordings, 368 feature windows in the demo.
- Generator seed: `20261125`. Re-running the generator with the same seed
  produces byte-identical files.
- `data/fake/manifest.json` records the SHA-256, style, persona, split, label,
  session number, and event count for every file.

## Motor model

A session alternates between pointer blocks and text blocks. The next block
switches modality with probability 0.72, producing the natural mixing of
pointing and typing/dictation within a 30 s window.

### Pointer

- A target is chosen uniformly in the central 88% of the screen.
- Duration is `distance / speed`, with `speed` drawn from a lognormal
  distribution around the persona's median.
- The path is a quadratic Bézier curve with a perpendicular bow (curvature) and
  per-sample tremor.
- Position along the path uses a blend of linear motion and a minimum-jerk
  profile. `ease=1` means strong deceleration at the end (a human click);
  `ease=0.15` means almost linear motion (the mimic).
- A click occurs with a persona-specific probability. Button-down is followed
  by a lognormal dwell time, then button-up.
- The pause after a block is lognormal, with a small chance of a 3–6× "think"
  tail.

### Text

- **Typed:** 3–18 characters per burst. Inter-key intervals are lognormal around
  the persona median, with occasional fast digraphs and hesitations. Errors are
  inserted with a per-character probability; after a persona-specific latency,
  1–5 backspaces are emitted. If a burst ends with a correction, the next block
  starts after an extra post-error pause.
- **Dictation:** one multi-character chunk per block (4–40 characters), with a
  persona-specific correction probability derived from the target correction
  rate. A correction deletes part of the chunk, either as one selected-text
  deletion or as a run of on-screen backspaces. A corrected chunk adds a
  post-error pause before the next block.
- **On-screen typing:** with a persona-specific probability, a short word is
  inserted one character at a time at 260–430 ms intervals.
- The text content is random words from a small neutral vocabulary. The model
  never reads the content; only lengths and timings are used.

### Session variation

Each session applies independent 6% lognormal jitter to the persona's median
speed, pause, dwell, block pause, text gap, burst length, correction rate,
correction latency, and correction size. This gives the owner a realistic
day-to-day range during enrolment.

## Persona parameters

Selected generator parameters (medians before session jitter):

| Style | Persona | Speed (widths/s) | Speed σ | Pause (s) | Text gap | Correction rate | Dwell (ms) | Curvature | Ease | Post-error (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mouse | owner | 0.380 | 0.45 | 0.65 | 140 ms | 0.070 | 95 | 0.100 | 0.92 | 1.20 |
| mouse | tired | 0.319 | 0.45 | 0.81 | 154 ms | 0.091 | 100 | 0.100 | 0.92 | 1.20 |
| mouse | stranger | 0.850 | 0.35 | 0.30 | 200 ms | 0.015 | 65 | 0.040 | 0.55 | 0.30 |
| mouse | mimic | 0.391 | 0.10 | 0.64 | 143 ms | 0.067 | 99 | 0.012 | 0.15 | 0.10 |
| assistive | owner | 0.180 | 0.50 | 1.20 | 1.70 s | 0.200 | 380 | 0.120 | 0.85 | 1.80 |
| assistive | tired | 0.151 | 0.50 | 1.50 | 1.87 s | 0.260 | 399 | 0.120 | 0.85 | 1.80 |
| assistive | stranger | 0.500 | 0.30 | 0.55 | 0.90 s | 0.015 | 90 | 0.020 | 0.35 | 0.20 |
| assistive | mimic | 0.234 | 0.10 | 1.18 | 1.73 s | 0.270 | 395 | 0.012 | 0.15 | 0.10 |

`Text gap` is the median insertion gap for the style (single-character typing
for mouse; dictation chunk cadence for assistive). The correction rate is
deleted characters per inserted character. Tired multipliers are applied to the
owner parameters: speed ×0.84, pause ×1.25, text gap ×1.10, correction rate
×1.30, correction latency ×1.12, dwell ×1.05, with the same path shape.

## Mimic construction

`mimic_from(owner)` copies the owner's **observable medians**: pointer speed,
pause, text cadence, correction rate, correction latency, click dwell, and
chunk length. It then deliberately changes the higher-order structure:

| Property | Owner | Mimic |
|---|---|---|
| Speed dispersion σ | 0.45–0.50 | 0.10 |
| Pause dispersion σ | 0.60 | 0.10 |
| Pause heavy tail | 10% chance of 3–6× | none |
| Path curvature | 0.10–0.12 | 0.012 |
| Path tremor | 0.0020–0.0022 | 0.0003 |
| End-of-move ease | 0.85–0.92 | 0.15 (linear) |
| Click dwell dispersion | 0.30–0.32 | 0.05 |
| Typing/dictation gap dispersion | 0.42–0.45 | 0.10 |
| Correction latency dispersion | 0.40–0.50 | 0.04 |
| Correction size dispersion | 0.35–0.50 | 0.03 |
| Post-error pause | 1.2–1.8 s | 0.10 s |

This is the intended adversarial model: the mimic has watched the owner and
copies the obvious averages, but not the owner's variability, tails, curved
paths, deceleration, or correction-related pauses. It is a model of a mimic,
not a claim about every real attacker.

## What is not modelled

- Real human motor noise, handedness, fatigue over long sessions, or learning.
- Real browser/OS event sampling, coalescing, or coordinate quantisation.
- Real dictation errors, accents, or language effects.
- Real eye-gaze/head-mouse/switch hardware noise.
- Adaptive attackers who receive feedback and retrain.
- Multiple users sharing one account or a deliberate inside attacker.

The corpus is designed to exercise the full pipeline and the swap test, not to
estimate real-world error rates.
