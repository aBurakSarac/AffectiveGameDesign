# Claude Code brief — *La Façade Fissurée* study site

**Goal:** take the design-complete static site in `La Façade Fissurée.html` and make it real —
bind the actual session videos + analysis data, and replace the two stylized method demos with
real captured imagery. The design is the source of truth; **do not restyle** — only swap demo data
for real data and wire interactivity.

The site is a React-over-Babel single page. Files:
`La Façade Fissurée.html` (entry) · `hud.css` `site*.css` `method-anims.css` `site-report.css`
`site-appendix.css` (styles) · `hud-*.{js,jsx}` (the approved HUD) · `site-*.jsx` (sections) ·
`site-data.js` (all content/data).

---

## Task 1 — Bind the 7 real sessions to the HUD Player (highest priority)

Right now every session card in `SessionRail` drives the **same** demo timeline from
`hud-signals.js` (`window.HUD`). Make each card play its own pre-baked, per-frame data + raw video.

Source of truth per session lives in `Pipeline/presentation/<stem>/`:
- `merged_hud.mp4` — the rendered HUD video (reference only; we re-render live in the browser)
- `analysis.csv` — per-window rPPG + algorithm rows
- `meta.json` — `{ stem, subject, video_id, lighting, source_video, … }`

The 7 stems are already listed in `site-data.js → SESSIONS`
(`S06_Vid16, S02_Vid04, S08_Vid18, S04_Vid09, S05_Vid10, S02_Vid05, S10_Vid13`).

**Steps**
1. **Pre-compute per-frame state offline**, reusing the project's own logic. `render_merged_hud.py`
   already exposes `precompute_states(fer, pos_windows, per_win_algo, …)` and consumes `pos_windows`
   (POS @ 30 s / 3 s). Run it per session and dump a compact **per-frame JSON** (or a typed-array
   binary) with exactly the fields the HUD reads from a computed frame `f`:
   `t, isFear, F12, F15, fear, arousal, tension, bpm, bpm_norm, algorithm, state (IDLE/ONSET/EVENT_CONFIRMED), …`
   — match the keys produced by `window.HUD.computeFrame(t)` so the React components need no changes.
2. **Extract the raw player video** for each session (the `source_video` in `meta.json`, transcoded
   to web-friendly H.264 mp4) into e.g. `media/<stem>/video.mp4` + `media/<stem>/frames.json`.
3. Replace the `window.HUD` demo singleton with a **per-session loader**: on session select, fetch
   that session's `frames.json`, drive playback from the real video's `currentTime` (the `<video>`
   becomes the clock — keep the existing trace-as-scrubber wired to `video.currentTime`), and feed
   the nearest pre-computed frame into the existing components.
4. Swap the `<image-slot>` placeholder in `PlayerVideo` for the real `<video>` element, keeping the
   forehead-ROI overlay box on top. Keep `localStorage` playback-position persistence.
5. Keep the per-card thumbnails; optionally generate a real poster frame per session.

**Do not** change the math or the HUD layout — only the data feeding it. The site already states the
correct production config (POS @ 30 s / 3 s, F12 θ=0.70, F15 θ=0.80).

---

## Task 2 — Replace Method 02 (CLAHE) with real on/off screenshots

The CLAHE before/after wiper in `site-methods.jsx → ClaheDemo` currently uses a **stylized CSS face**.
Replace it with a **real pair of frames from a dark session**:

1. Pick a genuinely dark frame (a `dim` or `mixed` session — e.g. `S02_Vid04` or `S08_Vid18`).
2. Produce two PNGs of the **same frame**: `clahe_off.png` (raw face ROI) and `clahe_on.png` (after
   the production CLAHE — `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))` on the **L channel**
   of LAB, exactly as in the pipeline). Crop both to the analysis ROI; keep them pixel-aligned.
3. Drop them into `media/clahe/` and feed them into the existing wiper: the `.clahe-face` (raw) and
   `.clahe-face.after` (CLAHE) layers already implement the draggable `clip-path` reveal — just
   replace the two `.ff` gradient divs with `<img>`s. Keep the ROI box + RAW/CLAHE tags.
4. Recompute the two histograms from the actual pixel luminance of each frame (replace the hard-coded
   `HIST_BEFORE` / `HIST_AFTER` arrays) so the spread is real.
5. Remove the "stylized demo" hand-off note once real imagery is in.

*(Optional, same pattern: Method 01 detection bars and Method 04 sweep cells are illustrative —
bind them to the real detector log shares and the real benchmark scores if available. Method 01 must
keep the "missed by both" segment and the occlusion/cut note: recorded video has no guarantee a face
is present 100% of the time.)*

---

## Task 3 — Productionize the page

- **Precompile the JSX** (Babel CLI / esbuild) instead of the in-browser transformer; bundle the
  `site-*.jsx` files. Keep the same load order and the `window.*` exports between files.
- Pin/​vendor React 18.3.1 + ReactDOM locally for offline use.
- Lazy-load session video + JSON on demand (don't ship all 7 videos up front).
- Keep all entrance animations gated on the existing polling in-view hooks (they must survive
  reduced-motion and non-painting contexts — content is visible by default, animation is additive).
- Verify responsive behavior holds: HUD scales to one screen ≥920 px and stacks below; nav collapses
  to the hamburger <1000 px.

---

## Guardrails
- The visual system (dark telemetry, the HUD components, type, spacing) is **approved** — match it
  exactly; don't introduce new colors or fonts.
- Every number on the site is from the final report or labelled illustrative/bindable. When you bind
  real data, keep the labels honest.
- No video or biometric data leaves the device — preserve the fully-local property end to end.
