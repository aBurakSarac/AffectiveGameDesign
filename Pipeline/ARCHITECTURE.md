# Pipeline — Architecture and Design Patterns

**Created:** 2026-04-28 | **Last updated:** 2026-04-28

---

## System Overview

Two independent pipelines share the `fer/` package. The live FER pipeline processes webcam
frames in real time and streams affect data to Unity. The annotation pipeline processes saved
session CSVs offline to produce labeled event spreadsheets.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        LIVE FER PIPELINE                            │
│                                                                     │
│  Webcam ──► fer/test_mp_hs.py ──► fer/face_detector.py  (MP)       │
│                     │         └──► fer/fusion.py         (HS + MP)  │
│                     │         └──► fer/two_gate_detector.py         │
│                     │         └──► fer/hud.py            (display)  │
│                     │                                               │
│                     ├──► utils/session_meta.py  (CSV row / frame)   │
│                     └──► utils/live_pipeline_sender.py  (TCP → Unity)│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     ANNOTATION PIPELINE                             │
│                                                                     │
│  Session CSV ──► fer/annotate_helper.py                             │
│                        │                                            │
│                        ├──► fer/annotation_io.py      (load CSV)    │
│                        └──► fer/annotation_algorithms.py            │
│                                 ├── v1: peak-window                 │
│                                 ├── v2: flood-fill                  │
│                                 └── v3: multi-channel vote           │
│                                                                     │
│  Output: logs/annotations/<session>_v1.csv | _v2.csv | _v3.xlsx    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Live FER Pipeline — Data Flow

```text
Webcam frame (BGR, ~30 fps)
    │
    ▼
fer/face_detector.py  [Strategy A]
    ├── compute_tension(blendshapes)       → mp_tension     [0, 1]
    ├── compute_face_valence(blendshapes)  → mp_valence     [-1, 1]
    ├── compute_au_velocities(blendshapes) → velocities, mp_startle_score
    └── compute_ctx_tag(blendshapes)       → mp_ctx_tag     (JOY|FEAR|STRESS|SAD|---)
    │
    ▼
HSEmotion model (EfficientNet-B0, called directly in test_mp_hs.py)
    └── 8 categorical emotions + valence/arousal
    │
    ▼
fer/fusion.py  [Strategy B / Pipeline-Filter]
    ├── compute_all_formulas(hs_*, mp_tension) → {F0…F6}  (7 formulas)
    ├── compute_composite_fear(hs_arousal, mp_tension) → composite_fear  [0, 1]
    └── compute_agreement(mp_ctx_tag, mp_tension, hs_dominant, hs_arousal, hs_emotions)
                                                → (agreement_tag, veto_tag)
    │
    ▼
fer/two_gate_detector.py  [State Machine + Ring Buffer]
    ├── Gate A: composite_fear derivative > 0.015 over 10-frame ring buffer
    ├── Gate B: composite_fear ≥ 0.30 sustained for ≥ 15 frames
    └── 30-frame refractory cooldown after EVENT_CONFIRMED
    │
    ├──► fer/hud.py  [Facade]                  → live overlay on frame
    ├──► utils/session_meta.py  [Observer]     → appends CSV row each frame
    └──► utils/live_pipeline_sender.py         → TCP socket to Unity
```

### Recommended formula

F2 = 0.40 × hs\_fear + 0.20 × hs\_surprise + 0.40 × mp\_tension

Selected based on 4.0× fear/anger discrimination ratio in S02 36-event benchmark.

---

## Annotation Pipeline — Data Flow

```text
Session CSV  (logs/sessions/<S_NN_label>/<id>.csv)
    │
    ▼
fer/annotation_io.py  [Repository]
    ├── load_csv()              → list of row dicts (all fields parsed, numeric)
    ├── load_rppg_csv()         → {algo: [(t_center, bpm_smoothed), ...]}
    └── recompute_composite_fear()  → updates rows in-place with named formula
    │
    ▼
fer/annotation_algorithms.py  [Strategy — 3 pluggable algorithms]

  ┌── v1: Peak-window ─────────────────────────────────────────────┐
  │   detect_peaks() → find_events() → get_window_indices()        │
  │   extract_event_v1() → run_v1()                                │
  │   Peak detection on startle + fear; fixed 6s window per peak   │
  └────────────────────────────────────────────────────────────────┘

  ┌── v2: Flood-fill ──────────────────────────────────────────────┐
  │   flood_fill_clusters() → _padded_window()                     │
  │   extract_event_v2() → run_v2()                                │
  │   Seed-frame flood fill; confidence scoring via padded window   │
  └────────────────────────────────────────────────────────────────┘

  ┌── v3: Multi-channel vote ──────────────────────────────────────┐
  │   build_channels() → detect_scene_cuts() → flood_fill_channel()│
  │   compute_vote_counts() → consolidate_vote_clusters()          │
  │   extract_event_v3() → run_v3()                                │
  │   7 independent signal channels; tier: CONFIRMED / CANDIDATE   │
  └────────────────────────────────────────────────────────────────┘
    │
    ▼
Output
    ├── v1 → logs/annotations/<id>_v1.csv   (HEADER_V1, 17 columns)
    ├── v2 → logs/annotations/<id>_v2.csv   (HEADER_V2, 20 columns)
    └── v3 → logs/annotations/<id>_v3.xlsx  (HEADER_V3, 40+ columns, freeze panes)
             or _v3.csv if openpyxl unavailable
```

### v3 signal channels

| Channel key | Source | Signal |
| ----------- | ------ | ------ |
| `DOM_FEAR` | HS emotion | `hs_dominant == "Fear"` |
| `DOM_SURP` | HS emotion | `hs_dominant == "Surprise"` |
| `FEAR_THR` | HS score | `hs_fear > 0.25` |
| `TENSION` | MP blendshapes | `mp_tension > 0.30` |
| `STARTLE` | MP velocity | `mp_startle_score > 3.0` |
| `AROUSAL` | HS valence-arousal | `hs_arousal > 0.50` |
| `CROSS_MODAL` | Fusion | `agreement_tag starts with AGREE_FEAR` |

---

## Module Dependency Map

Import direction flows downward. No upward imports exist.

```text
fer/blendshapes.py          fer/annotation_events.py
(no imports)                (no imports)
      │                           │
      ▼                           ▼
fer/face_detector.py     fer/annotation_io.py
      │                           │
      ▼                           ▼
fer/fusion.py            fer/annotation_algorithms.py
fer/hud.py                         │
fer/video_output.py                ▼
      │               fer/annotate_helper.py  ← entry point
      ▼
fer/test_mp_hs.py  ← entry point
  also imports:
    fer/two_gate_detector.py
    fer/improved_fear_detection.py
    utils/session_meta.py
    utils/post_session_analysis.py
    utils/live_pipeline_sender.py (optional, --live flag)
```

---

## Pattern Registry

| Module | Pattern | Role |
| ------ | ------- | ---- |
| `fer/blendshapes.py` | Constants | Blendshape lists, HUD dims, emotion labels — zero deps |
| `fer/face_detector.py` | Strategy A | MediaPipe 52-blendshape FER tool |
| `fer/fusion.py` | Strategy B / Pipeline-Filter | Pure math: 7 formulas, composite fear, agreement logic |
| `fer/two_gate_detector.py` | State Machine + Ring Buffer | Two-gate fear event detection |
| `fer/hud.py` | Facade | Live HUD + post-session 4-panel plot |
| `fer/video_output.py` | — | Phase 2 annotated video renderer (--render flag) |
| `fer/improved_fear_detection.py` | Utility | AU velocity tag, pure function, no state |
| `fer/test_mp_hs.py` | Facade / Orchestrator | Entry point; wires all strategies into frame loop |
| `fer/annotation_events.py` | Constants / Registry | Formulas, headers, channel defs — zero deps |
| `fer/annotation_io.py` | Repository | All CSV data access encapsulated, no algorithm code |
| `fer/annotation_algorithms.py` | Strategy | v1/v2/v3 pluggable algorithms |
| `fer/annotate_helper.py` | Command / Facade | Parses args; dispatches to selected algorithm |
| `utils/session_meta.py` | Observer target | Session state consumed by pipeline; drives CSV |
| `utils/live_pipeline_sender.py` | — | TCP bridge: Python → Unity per-frame F2 float |

---

## Key Thresholds

| Parameter | Value | Rationale |
| --------- | ----- | --------- |
| Active formula | F2 | 4.0× fear/anger discrimination in S02 benchmark |
| Gate A | derivative > 0.015 / 10 frames | Calibrated on 36-event S02 ground truth |
| Gate B | signal ≥ 0.30 / 15 frames | Sustained fear; avoids transient FP |
| Refractory | 30 frames | Prevents double-counting same event |
| Blur trigger | bbox\_area < baseline × 0.70 | FER signals unreliable below 70% face coverage |
| v3 vote threshold | FORMULA\_VOTE\_THR = 0.25 | At least 25% of formulas must agree |
| Startle velocity | STARTLE\_VELOCITY\_THRESHOLD = 3.0 | Gate for startle AU velocity events |
| Neutral tension max | NEUTRAL\_TENSION\_MAX = 0.15 | Used to detect genuine tension above baseline |

---

## Directory Layout

```text
Pipeline/
├── fer/                        — FER pipeline package
│   ├── test_mp_hs.py           — entry point [Facade]
│   ├── blendshapes.py          — constants [Constants]
│   ├── face_detector.py        — MediaPipe wrapper [Strategy A]
│   ├── fusion.py               — FER math [Strategy B / Pipeline-Filter]
│   ├── hud.py                  — live HUD + plot [Facade]
│   ├── video_output.py         — annotated video render
│   ├── two_gate_detector.py    — event detection [State Machine + Ring Buffer]
│   ├── improved_fear_detection.py  — AU velocity tag [Utility]
│   ├── annotate_helper.py      — annotation entry [Command / Facade]
│   ├── annotation_events.py    — constants + registries [Constants / Registry]
│   ├── annotation_io.py        — file I/O [Repository]
│   └── annotation_algorithms.py — v1/v2/v3 clustering [Strategy]
├── utils/                      — shared utilities
│   ├── session_meta.py         — session tracking [Observer target]
│   ├── post_session_analysis.py — post-session runner
│   └── live_pipeline_sender.py — Unity TCP bridge
├── analysis/                   — post-session analysis scripts
├── rppg/                       — rPPG (scope paused, preserved)
├── archive_legacy/             — superseded scripts (do not delete)
├── models/                     — downloaded model weights
├── logs/
│   ├── sessions/               — per-session CSVs
│   └── annotations/            — annotation output files
├── ARCHITECTURE.md             — this file
├── README.md                   — script inventory and run commands
└── requirements.txt            — pinned dependencies (Python 3.12, conda facade)
```
