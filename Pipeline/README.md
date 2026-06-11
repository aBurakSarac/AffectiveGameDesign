# Pipeline — Script Inventory

**Created:** 2026-04-28 | **Last updated:** 2026-05-02

## Active Entry Points

| Script | Purpose | Run command |
|--------|---------|-------------|
| `fer/test_mp_hs.py` | Primary FER pipeline (live analysis) | `python Pipeline/fer/test_mp_hs.py --video <path>` |
| `fer/annotate_helper.py` | Generate annotation template from session CSV | `python Pipeline/fer/annotate_helper.py --csv <path>` |
| `analysis/visualize_session.py` | Per-session signal plot | `python Pipeline/analysis/visualize_session.py <csv>` |
| `rppg/live_rppg.py` | rPPG live capture + analysis + GT comparison | `python Pipeline/rppg/live_rppg.py capture --label NAME` |

## Utility Modules (not run directly)

### fer/ — FER pipeline

| Module | Pattern | Role |
|--------|---------|------|
| `fer/blendshapes.py` | Constants | All blendshape lists, HUD constants, emotion labels |
| `fer/face_detector.py` | Strategy A | MediaPipe FaceLandmarker wrapper, tension/valence/velocity |
| `fer/fusion.py` | Strategy B / Pipeline-Filter | HSEmotion + MP composite fear formulas, agreement logic |
| `fer/hud.py` | Facade | Live HUD display, post-session 4-panel plot |
| `fer/video_output.py` | — | Phase 2 annotated video renderer (called with --render) |
| `fer/two_gate_detector.py` | State Machine / Ring Buffer | IDLE→ONSET→SUSTAINING→EVENT_CONFIRMED event detection |
| `fer/improved_fear_detection.py` | Utility | AU velocity tag computation |
| `fer/annotation_events.py` | Constants / Registry | Annotation constants, HEADER_V1/V2/V3, channel defs |
| `fer/annotation_io.py` | Repository | CSV load/parse, output dir, rPPG sidecar I/O |
| `fer/annotation_algorithms.py` | Strategy | v1 peak-window, v2 flood-fill, v3 multi-channel voting |

### utils/ — Shared utilities

| Module | Role |
|--------|------|
| `utils/session_meta.py` | Session tracking (Observer target) |
| `utils/post_session_analysis.py` | Post-session runner called by fer/test_mp_hs.py |
| `utils/live_pipeline_sender.py` | Unity TCP bridge for live affect data |

### analysis/ — Post-session analysis (run directly)

`visualize_session.py`, `visualize_explorer.py`, `benchmark_explorer.py`,
`benchmark_compare.py`, `formula_benchmark.py`

### rppg/ — rPPG pipeline (active)

| Module | Pattern | Role |
|--------|---------|------|
| `rppg/rppg_algorithms.py` | Strategy | CHROM/POS/GREEN/ICA/WAVELET algorithm registry + BPM estimation |
| `rppg/capture.py` | Observer | Live webcam recording + forehead ROI extraction |
| `rppg/analyzer.py` | Utility | File-I/O wrapper around `compute_bpm_timeseries` |
| `rppg/ground_truth.py` | Factory + Repository | Zepp ZIP / CSV GT loader, window alignment, accuracy metrics |
| `rppg/replay.py` | Template Method | rPPG HUD video renderer from session artefacts |
| `rppg/session.py` | Facade + Repository | Session folder CRUD + `RppgSession` orchestrator |
| `rppg/hud_constants.py` | Constants | HUD layout dimensions and colour palette |
| `rppg/smoke_test_rppg.py` | Script | Standalone CHROM feasibility test on any video file |

## Archived (archive_legacy/)

Scripts superseded by fer/ pipeline or used for one-off tasks.
Do not delete — referenced in session logs.

| Script | Reason archived |
|--------|----------------|
| `test_mediapipe.py` | Capability now in fer/face_detector.py |
| `test_hsemotion.py` | Capability now in fer/fusion.py |
| `test_deepface.py` | Tool not selected |
| `test_pyfeat.py` | Tool not selected |
| `test_fusion.py` | Superseded by fer/test_mp_hs.py |
| `patch_category_dropdown.py` | One-off UI fix, no longer needed |
| `check_null.py` | Ad-hoc debug utility |

## Directory Layout

```text
Pipeline/
├── fer/                    — FER pipeline package
├── utils/                  — Shared utilities
├── analysis/               — Post-session analysis scripts
├── rppg/                   — rPPG pipeline package
├── archive_legacy/         — Superseded scripts (keep for reference)
├── models/                 — Downloaded model weights (face_landmarker.task)
├── sessions/               — rPPG session artefacts (frames.csv, analysis.csv, replay.mp4)
├── logs/                   — FER session CSVs and outputs
│   ├── sessions/           — Per-session subdirectories
│   └── annotations/        — Annotation template outputs
├── README.md               — This file
└── requirements.txt        — Pinned dependencies
```
