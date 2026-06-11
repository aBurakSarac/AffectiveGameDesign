# HANDOFF — Merged FER + rPPG Fear-Detection HUD → offline mp4 renderer

**For:** Claude Code
**Visual source of truth:** `Merged Fear HUD.html` (in this project). Open it and scrub
the timeline — that HTML is the exact layout, hierarchy, and behavior to reproduce as a
video overlay. Ignore `Merged Fear HUD (Terminal).html`; it is an alternate style and is
**not** part of this task.

**Goal:** a Python script that **re-renders an already-recorded session** into a single
merged HUD **mp4**, combining the FER outputs (`fer/test_mp_hs.py`) and the rPPG outputs
(`rppg/live_rppg.py`) over the **raw recorded video**, laid out and colored like
`Merged Fear HUD.html`.

> The HTML is a hand-authored *teaching mock* with scripted numbers. Your job is to drive
> the **same layout** with the **real per-frame data already on disk** from a past session.

---

## 0. Hard requirements (read first)

1. **Offline re-render of existing sessions — not new live capture.** The script's only job
   is to take a **session that has already been recorded and analyzed** (raw video + the
   CSVs produced by `test_mp_hs.py` and `live_rppg.py`) and produce the merged HUD mp4.
   **Do not open a camera.** Decode the saved video frame-by-frame and overlay the HUD.
   This mirrors what `rppg/replay.py` already does for the rPPG-only HUD.
2. **Runtime input, not CLI arguments.** Don't require `argparse` flags. Prompt
   **interactively at run time** (stdin) for the inputs: the FER CSV, the rPPG session
   directory (or its `analysis.csv`/`frames.csv`), the raw video path, optional ground-truth
   file, and the output path. A `list`-style picker of available sessions (like
   `live_rppg.py list`) is ideal. The operator should run
   `python Pipeline/render_merged_hud.py` with no flags and be guided by prompts.
3. **Preserve, don't delete.** When you remove or replace pipeline behavior, **comment it
   out** with a dated marker (`# [COMMENTED OUT YYYY-MM-DD — reason]`) so it can be rolled
   back. Same convention already used in `Merged Fear HUD.html` / `hud-components.jsx`.
4. **Tweaks → constants.** The HTML has a live Tweaks panel; the mp4 renderer has no UI.
   Implement every tweak option as a module-level constant, **default to one**, and leave
   the other options **commented out** directly beneath it so they can be switched by
   editing the file. (List in §6.)

---

## 1. The two source pipelines — what they leave on disk

You are **consuming the artifacts** these scripts already produce. You do **not** re-run
capture; at most you may re-run the rPPG *analyze* step on saved frames if an `analysis.csv`
is missing.

### FER — `Pipeline/fer/test_mp_hs.py`
- When run (live or `--video PATH`) it writes two CSVs into `Pipeline/logs/`:
  - `<session_id>_mp_hs_compact_temp.csv`
  - `<session_id>_mp_hs_temp.csv` (**full — has `f0…f13` and all per-frame signals**) ← use this
- Each row is **one analyzed frame** with a timestamp, the MediaPipe + HSEmotion signals,
  and the precomputed formula columns.
- Formula math lives in `fer/fusion.py`:
  - `compute_composite_fear(hs_fear, hs_arousal, mp_tension)` → **F12**
    = `clamp((0.7·hs_fear + 0.3·hs_arousal)·(1 + mp_tension))`
  - `compute_all_formulas(...)` returns `f0…f13`; **`f12`** is the one you want.
    (You can read `f12` straight from the CSV, or recompute from `hs_fear/hs_arousal/mp_tension`.)

### rPPG — `Pipeline/rppg/live_rppg.py`  (+ `analyzer.py`, `replay.py`)
- A captured session lives in a **session directory** containing:
  - `raw_video.mp4` — the recording (this is the video you overlay).
  - `frames.csv` — per-frame ROI RGB means, bbox, `face_detected`, timestamps.
  - `analysis.csv` — **per-window** rows written by `RppgAnalyzer.save()`; columns include
    `algo`, `window_idx`, `t_center`, `bpm`, `bpm_smoothed`, `snr`. One row per (algorithm,
    window); windows step every `step_s`.
  - optional `gt_aligned.csv` — `t_center → gt_bpm`.
- Production config = **POS @ 30 s window / 3 s step** = `RPPG_CONFIG_POS_30S`
  (`rppg/evaluate_rppg.py`). Forehead ROI is `x ∈ [0.20w, 0.80w]` (`rppg/extractors.py`).
- `rppg/replay.py` **already** opens `raw_video.mp4`, loads `analysis.csv` via
  `_load_analysis`, indexes windows by `window_idx`, and draws a per-frame HUD. **Read it —
  it is the structural template for this task.**
- BPM normalization + augmentation in `rppg/evaluate_rppg.py`:
  - `compute_bpm_norm(bpm_at_frame, frame_timestamps, baseline_window_s=60.0)`
    → `clip((bpm − rolling_median_baseline)/baseline, 0, 1)` (60 s centered rolling median).
  - `enrich_rppg_formulas(model_df, bpm_norm, coeff=...)` → `fi_rppg = clip(fi·(1+coeff·bpm_norm),0,1)`.
- **F15** (report's `rppg_amp`) = `f12_rppg` with **coeff `c = 0.5`** on POS@30s:
  **F15 = F12·(1 + 0.5·bpm_norm)**.

> `test_rppg` in the brief = this `live_rppg` session (its saved `raw_video.mp4` +
> `frames.csv` + `analysis.csv`).

---

## 2. Offline orchestration (the part that needs design)

There is **no shared camera and no warmup** — both pipelines already ran. The renderer is a
**join + draw** loop over the saved raw video, exactly like `rppg/replay.py` but with the
merged panel.

Recommended structure for `Pipeline/render_merged_hud.py`:

```
prompt operator → pick session: raw_video.mp4, FER csv, rppg analysis.csv/frames.csv, [gt], output
load FER rows      → fer_by_ts   (per-frame: hs_*, mp_*, f12, latency_ms, timestamp)
load rppg windows  → pos_windows (per-window POS bpm + per-algo bpm/snr, keyed by t_center)
build bpm_at_frame → interpolate/step POS bpm onto the video timeline
bpm_norm_series    → compute_bpm_norm(bpm_at_frame, frame_timestamps, 60.0)   # whole session
open VideoCapture(raw_video.mp4)              # decode saved frames — NOT a camera
for frame_idx, frame in enumerate(video):
    t       = frame_idx / fps                 # or frames.csv timestamp
    fer     = fer_by_ts.nearest(t)            # join FER row by timestamp
    bpm     = pos_windows.at(t)               # POS window covering t (None before first window)
    bpm_n   = bpm_norm_series.at(t)           # 0 where no valid window yet
    f12     = fer.f12
    f15     = clamp(f12 * (1 + 0.5 * bpm_n))
    hud     = compose_merged_hud(frame, fer, bpm, algos_at(t), bpm_n, f12, f15, t, fer.latency_ms)
    writer.write(hud)                         # cv2.VideoWriter mp4 @ source fps
```

**Time alignment is the crux.** FER rows are per-frame; rPPG rows are per-window (every
`step_s`, centered at `t_center`, needing a full `window_s` of data). Join FER to video
frames by timestamp; map each frame to the POS window whose span covers it (reuse
`replay.py`'s `window_idx` indexing). The FER and rPPG recordings share the same wall clock
if they came from the same session — align on `session_start_utc` / timestamps, and expose a
small constant offset knob in case the two clocks differ (see §6).

**Reuse, don't fork:**
- Subclass / borrow `rppg/replay.py`'s `BaseReplayRenderer` (video open, `_open_writer`,
  `_load_analysis`, `_load_frame_bboxes`, the frame loop) and replace `draw_hud_panel` with
  the merged panel.
- Read FER `f12` and signals from the full CSV; recompute via `fer/fusion.py` only if a
  column is missing. Don't re-run `test_mp_hs.py`.

**Early frames (honest, not faked):** before the first valid POS window (~first `window_s`
seconds of the recording), there is no BPM. Render the Heart-rate card + F15 gauge in a muted
"no rPPG window yet — F15 = F12" state with `bpm_norm = 0` (so F15 = F12). Do not invent a BPM.

**Missing `analysis.csv`:** if a session has `frames.csv` but no `analysis.csv`, optionally
run `RppgAnalyzer` on the saved frames with `RPPG_CONFIG_POS_30S` to generate it, then
proceed. (Optional convenience — prompt before doing it.)

---

## 3. Layout (match `Merged Fear HUD.html`)

Canvas **1920×1080**. Left ≈ video, right = 768 px analysis panel. Top = telemetry bar.

```
┌──────────────────────────────────────────────┬───────────────────────────┐
│ TELEMETRY BAR  Time · Frame · Latency · fps                                │
├──────────────────────────────────────────────┼───────────────────────────┤
│                                                │  VERDICT (F15 drives it)  │
│   VIDEO (decoded frame from raw_video.mp4)     │   FEAR DETECTED / NO FEAR │
│   + forehead rPPG ROI box (green)              │   F12 gauge ── thr 0.70   │
│                                                │   F15 gauge ── thr 0.80   │
│                                                ├───────────────────────────┤
│                                                │ 1 PRIMARY · facial emotion│
│                                                │   Fear (hero, red)        │
│                                                │   Arousal (amber)         │
│                                                │   base = .7·fear+.3·arous │
│                                                │   7 other emotions (grid) │
│                                                ├───────────────────────────┤
│                                                │ 2 AMPLIFIERS              │
│                                                │   Facial tension ×(1+t)   │
│                                                │   Heart rate ×(1+.5·bn)   │
├────────────────────────────────────────────────┤   (other algos strip)   │
│  TRACE: F12 + F15 over time, both thresholds   ├───────────────────────────┤
│  (playhead at current time)                    │ 3 HOW THE SCORE IS BUILT  │
│                                                │   base→×tension→F12→×hr→F15│
└────────────────────────────────────────────────┴───────────────────────────┘
```

Because the whole session is known up front, the **trace can be drawn complete** (full F12
and F15 curves + both thresholds) with a **moving playhead** at the current frame's time. No
interactive scrubber/play button in the mp4.

---

## 4. Data → element map

| HUD element | Source column / call |
|---|---|
| **Time** | frame timestamp (`mm:ss.cc`) from video / `frames.csv` |
| **Frame** | frame index in `raw_video.mp4` |
| **Latency** | `latency_ms` (FER full CSV — recorded per-frame compute time) |
| **fps** | source video fps (constant) |
| **forehead ROI box** | `frames.csv` bbox / rPPG extractor coords `x∈[0.20w,0.80w]` |
| **Fear (hero)** | `hs_fear` (FER CSV) |
| **Arousal** | `hs_arousal` (FER CSV) |
| **base score** | `0.7·hs_fear + 0.3·hs_arousal` |
| **Dominant emotion** | `hs_dominant` / `hs_dominant_score` |
| **7 other emotions** | `hs_anger, hs_contempt, hs_disgust, hs_happiness, hs_neutral, hs_sadness, hs_surprise` |
| **Facial tension** | `mp_tension` → multiplier `×(1+mp_tension)` |
| **Valence / Smile / Startle** | `mp_face_valence` / `mp_smile_level` / `mp_startle_score` |
| **Heart rate (BPM)** | POS `bpm` / `bpm_smoothed` from `analysis.csv` (window covering t) |
| **rise (bpm_norm)** | `compute_bpm_norm(...)` over the session |
| **rest baseline** | 60 s rolling median baseline |
| **other algos chips** | CHROM, GREEN, ICA, WAVELET, CONSENSUS `bpm` from `analysis.csv` |
| **F12 gauge** | `f12` (FER CSV), threshold **0.70** |
| **F15 gauge** | `f12·(1+0.5·bpm_norm)`, threshold **0.80** |
| **Verdict state** | `F15 ≥ 0.80` → FEAR DETECTED, else NO FEAR |
| **Chain** | base → `×(1+mp_tension)` → F12 → `×(1+0.5·bpm_norm)` → F15 |

---

## 5. Design decisions baked into the HTML (carry them over)

- **Two scores shown together**, each with its **own threshold line** (F12 @ 0.70, F15 @
  0.80). F15 (production) drives the verdict; the pair shows how rPPG nudges F12 → F15.
- **F12 / F15 bars are neutral gray.** State is shown by the verdict banner + the red
  `OVER`/score text and the white threshold tick — not by bar color. (Fear=red and
  Arousal=amber are the only saturated signal colors.)
- **No `State: [STRESS]` tag.** `mp_ctx_tag` was removed from the HUD because it read like a
  verdict; surface only `mp_tension` (the value that feeds the formula). Keep `mp_ctx_tag`
  in the CSV, just don't draw it as a state.
- **Below-threshold = not fear, made obvious.** The canonical teaching case: high `hs_fear`
  but low `hs_arousal` and flat BPM → score stays under the line → `NO FEAR`. The gauge tick
  positions make this visible.
- **Primary > Amplifiers > Chain** ordering, with Fear and Tension the highlighted rows.

---

## 6. Tweaks → constants (default + commented alternatives)

Put these at the top of `render_merged_hud.py`. Default shown; alternatives commented.

```python
# Label style on signal rows
LABEL_MODE = "plain_tech"      # plain-language + technical subscript (default)
# LABEL_MODE = "plain"         # plain language only
# LABEL_MODE = "tech"          # technical names only (hs_fear, mp_tension, …)

SHOW_OTHER_ALGOS = True        # show CHROM/GREEN/ICA/WAVELET/CONSENSUS chips
# SHOW_OTHER_ALGOS = False     # POS only

RPPG_COEFF = 0.5               # c in F15 = F12·(1 + c·bpm_norm)  (production)
# RPPG_COEFF = 0.3
# RPPG_COEFF = 0.8

FER_RPPG_TIME_OFFSET_S = 0.0   # +/- shift if the FER and rPPG clocks differ for a session

THRESH_F12 = 0.70
THRESH_F15 = 0.80
POS_WINDOW_S = 30
POS_STEP_S = 3
BASELINE_WINDOW_S = 60
```

---

## 7. Constants & colors (from `hud.css`)

Match the dark theme. Key colors (oklch in CSS; hex approximations for cv2/PIL):

| Token | Use | Hex |
|---|---|---|
| bg | canvas | `#1b1b22` |
| surface | cards | `#262630` |
| ink / ink-2 / ink-3 / ink-4 | text + gray bars | `#f3f3f6 / #b8b8c0 / #87878f / #6a6a72` |
| danger (red) | Fear, verdict, OVER | `#e8584d` |
| arousal (amber) | Arousal | `#e0a83a` |
| tension (violet) | tension accent | `#b07fd6` |
| heart (red-pink) | heart-rate accent | `#e0726b` |
| clear (green) | NO-FEAR / ROI box | `#5ec8a8` |

Fonts: UI text = a clean sans (Plex Sans/Inter equiv.); **all numbers monospaced**,
tabular figures. Min on-screen text ~24 px at 1080p.

---

## 8. Acceptance checklist

- [ ] `python Pipeline/render_merged_hud.py` runs with **no flags**, prompts interactively to
      pick an **existing** session, and writes an mp4 — **no camera is opened**.
- [ ] Reads FER full CSV + rPPG `analysis.csv`/`frames.csv` + `raw_video.mp4` and **joins
      them on timestamp** (FER per-frame ↔ POS per-window).
- [ ] Per frame: F12 = `(0.7·hs_fear+0.3·hs_arousal)·(1+mp_tension)`, F15 = `F12·(1+0.5·bpm_norm)`.
- [ ] Verdict flips on **F15 ≥ 0.80**; F12 and F15 gauges show **separate** ticks (0.70 / 0.80).
- [ ] Before the first valid POS window: muted "no window yet" state, `bpm_norm = 0`, F15 = F12; no fake BPM.
- [ ] No `State: [STRESS]` tag drawn; `mp_tension` shown instead.
- [ ] Layout/hierarchy/colors match `Merged Fear HUD.html`.
- [ ] `test_mp_hs.py`, `live_rppg.py`, and `replay.py` still run standalone; removed code is commented, not deleted.
- [ ] Tweak options exist as constants with one default + commented alternatives.

---

## 9. Files to read before starting

- `Merged Fear HUD.html`, `hud.css`, `hud-components.jsx`, `hud-signals.js` (visual spec + math)
- `rppg/replay.py` (**closest template** — offline video + analysis.csv → HUD mp4) and `rppg/hud_constants.py`
- `fer/test_mp_hs.py` (CSV columns + `Session.pre_session_prompt`) and `fer/hud.py` (existing FER HUD)
- `fer/fusion.py` (`compute_composite_fear` = F12)
- `rppg/live_rppg.py`, `rppg/analyzer.py`, `rppg/extractors.py` (session dir layout, `analysis.csv` schema, ROI)
- `rppg/evaluate_rppg.py` (`compute_bpm_norm`, `enrich_rppg_formulas`, `RPPG_CONFIG_POS_30S`)
```
