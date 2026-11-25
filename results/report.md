# Marrow prototype — measured results

Generated: `2026-11-25T10:51:22+00:00`  
Data: **synthetic only**, generator seed `20261125`  
Threshold: fixed on owner-only calibration windows before any test window was scored.  
Calibration target: 15% false-reject rate.

## Window-level decision (primary)

| Style | Owner FRR (tired) | Stranger FAR | Mimic FAR |
|---|---:|---:|---:|
| mouse_keyboard | 8.3% (2/24) | 7.5% (3/40) | 0.0% (0/40) |
| assistive_dictation | 8.3% (2/24) | 5.0% (2/40) | 0.0% (0/40) |
| **overall** | **8.3% (4/48)** | **6.2% (5/80)** | **0.0% (0/80)** |

FRR = real owner rejected (doubted). FAR = impostor accepted.  
95% Wilson intervals are in `results/report.json`.

## Session-level decision (upper-trimmed mean of window scores)

| Style | Owner FRR (tired) | Stranger FAR | Mimic FAR |
|---|---:|---:|---:|
| mouse_keyboard | 0.0% (0/6) | 0.0% (0/10) | 0.0% (0/10) |
| assistive_dictation | 0.0% (0/6) | 0.0% (0/10) | 0.0% (0/10) |
| **overall** | **0.0% (0/12)** | **0.0% (0/20)** | **0.0% (0/20)** |

## Swap test: profile trained on an impostor

| Style | Wrong profile trained on | Owner windows accepted | Owner sessions accepted |
|---|---|---:|---:|
| mouse_keyboard | stranger | 0.0% (0/24) | 0.0% (0/6) |
| mouse_keyboard | mimic | 0.0% (0/24) | 0.0% (0/6) |
| assistive_dictation | stranger | 0.0% (0/24) | 0.0% (0/6) |
| assistive_dictation | mimic | 0.0% (0/24) | 0.0% (0/6) |

## Thresholds and model

| Style | Threshold | Calibration windows | Calibration FRR | Features kept |
|---|---:|---:|---:|---:|
| mouse_keyboard | 5.289 | 20 | 15.0% | 12 |
| assistive_dictation | 5.226 | 20 | 15.0% | 11 |

## Mimic closeness (median feature values)

The mimic is generated with the owner's medians for speed, pause, cadence, correction
rate, dwell and chunk length, but not the owner's dispersion, tails, path shape or
cross-event coupling.  Values below are medians over all windows.

| Feature | mouse_keyboard | assistive_dictation |
|---|---:|---:|
| `pointer_speed_med` | 0.38119 / 0.38998 | 0.22936 / 0.22231 |
| `pointer_speed_disp` | 0.94714 / 0.28495 | 0.85485 / 0.2959 |
| `path_efficiency` | 0.87193 / 0.9986 | 0.6863 / 0.9954 |
| `pause_tail` | 0 / 0 | 0 / 0.1 |
| `preclick_slowdown` | 0.55399 / 0.87253 | 0.59384 / 0.85743 |
| `click_dwell_disp` | 0.31169 / 0.04118 | 0.29914 / 0.0553 |
| `text_gap_med` | 0.132 / 0.138 | 4.2655 / 4.76 |
| `text_gap_disp` | 0.79145 / 0.26088 | 0.22353 / 0.21196 |
| `correction_rate` | 0.15029 / 0.12622 | 0.19633 / 0.20098 |
| `correction_latency_med` | 0.7135 / 0.69075 | 2.228 / 2.036 |
| `post_error_pause` | 3.985 / 3.1905 | 5.7255 / 5.40525 |
| `insert_len_disp` | 0 / 0 | 0.33929 / 0.05972 |
| `correction_size_disp` | 0.5 / 0 | 0.1875 / 0 |

Cell format: owner median / mimic median.

