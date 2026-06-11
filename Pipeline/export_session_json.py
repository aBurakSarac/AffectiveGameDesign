"""export_session_json.py — Export per-frame HUD state as JSON for the website.

Reuses render_merged_hud.py's precompute_states() to produce a compact
per-frame JSON file for each presentation session, consumed by the
browser-side session-loader.js.

Usage:
    python Pipeline/export_session_json.py              # export all 7 sessions
    python Pipeline/export_session_json.py S06_Vid16    # export one session
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from render_merged_hud import (
    FrameState,
    build_frame_times,
    load_fer,
    load_pos_windows,
    precompute_states,
)
from rppg.replay import BaseReplayRenderer

PRES_DIR = _HERE / "presentation"
OUT_ROOT = _HERE.parent / "Website" / "media" / "sessions"

SESSIONS = [
    "S06_Vid16", "S02_Vid04", "S08_Vid18",
    "S04_Vid09", "S05_Vid10", "S02_Vid05", "S10_Vid13",
]


def _round(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    return round(float(v), d)


def _parse_bbox(s):
    if not s or pd.isna(s):
        return None
    try:
        parts = str(s).split(",")
        return [int(float(x)) for x in parts[:4]]
    except (ValueError, IndexError):
        return None


def _compute_hint(startle, mp_tension):
    if startle > 0.3:
        return "startle?", "alert"
    if mp_tension > 0.30:
        return "stress?", "warn"
    if mp_tension > 0.15:
        return "tension?", "warn"
    return "—", "idle"


def load_snr_map(analysis_csv):
    """Build {window_idx: {ALGO: snr}} from analysis.csv."""
    rows = BaseReplayRenderer._load_analysis(analysis_csv)
    snr_map = {}
    for r in rows:
        algo = str(r.get("algorithm", "")).upper()
        try:
            widx = int(float(r.get("window_idx", -1)))
            snr = float(r.get("snr", 0))
        except (TypeError, ValueError):
            continue
        snr_map.setdefault(widx, {})[algo] = round(snr, 1)
    return snr_map


def export_session(stem):
    pres = PRES_DIR / stem
    if not pres.exists():
        print(f"  SKIP {stem}: {pres} not found")
        return

    fer_csv = pres / "fer.csv"
    analysis_csv = pres / "analysis.csv"
    frames_csv = pres / "frames.csv"
    video_path = pres / "raw_video.mp4"

    if not all(p.exists() for p in [fer_csv, analysis_csv, frames_csv, video_path]):
        missing = [p.name for p in [fer_csv, analysis_csv, frames_csv, video_path] if not p.exists()]
        print(f"  SKIP {stem}: missing {missing}")
        return

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  {stem}: {vid_w}x{vid_h} @ {src_fps:.1f}fps, {n_frames} frames")

    fer = load_fer(fer_csv)
    pos_windows, per_win_algo = load_pos_windows(analysis_csv)
    frames_df = pd.read_csv(frames_csv, comment="#")
    frame_ts = build_frame_times(frames_df, n_frames, src_fps)

    states = precompute_states(fer, pos_windows, per_win_algo, frame_ts, src_fps)

    snr_map = load_snr_map(analysis_csv)

    bbox_by_frame = {}
    forehead_by_frame = {}
    for _, row in frames_df.iterrows():
        fi = int(row.get("frame_idx", -1))
        bbox_by_frame[fi] = _parse_bbox(row.get("bbox"))
        forehead_by_frame[fi] = _parse_bbox(row.get("roi_bbox"))

    duration = float(frame_ts[-1]) if len(frame_ts) > 0 else 0

    frames_out = []
    for s in states:
        hint, hintKind = _compute_hint(s.startle, s.mp_tension)

        algos_out = {}
        if s.algos:
            widx = None
            for w in pos_windows:
                if w["t_start"] <= s.t <= w["t_end"]:
                    widx = w["window_idx"]
                    break
            if widx is None:
                prior = [w for w in pos_windows if w["t_center"] <= s.t]
                if prior:
                    widx = prior[-1]["window_idx"]

            for algo, bpm_val in s.algos.items():
                snr_val = snr_map.get(widx, {}).get(algo, 0)
                algos_out[algo] = {"bpm": _round(bpm_val, 1), "snr": _round(snr_val, 1)}

        emotions_out = {k: _round(v, 3) for k, v in s.emotions.items()}

        roi = bbox_by_frame.get(s.idx)

        frame_obj = {
            "t": _round(s.t, 3),
            "frame": s.frame_num,
            "latency": _round(s.latency, 1),
            "fps": _round(s.fps, 1),
            "hs_fear": _round(s.hs_fear, 4),
            "hs_arousal": _round(s.hs_arousal, 4),
            "emotions": emotions_out,
            "dom": s.dom,
            "domScore": _round(s.dom_score, 3),
            "mp_tension": _round(s.mp_tension, 4),
            "valence": _round(s.valence, 3),
            "smile": _round(s.smile, 3),
            "startle": _round(s.startle, 2),
            "hint": hint,
            "hintKind": hintKind,
            "bpm": _round(s.bpm, 1) if s.has_bpm else 0,
            "bpm_norm": _round(s.bpm_norm, 4),
            "baseline": _round(s.baseline, 1),
            "algos": algos_out,
            "base": _round(s.base, 4),
            "mp_mult": _round(s.mp_mult, 4),
            "rppg_mult": _round(s.rppg_mult, 4),
            "F12": _round(s.f12, 4),
            "F15": _round(s.f15, 4),
            "isFear": s.is_fear,
        }
        if roi:
            frame_obj["roi"] = roi
        forehead = forehead_by_frame.get(s.idx)
        if forehead:
            frame_obj["foreheadRoi"] = forehead

        frames_out.append(frame_obj)

    output = {
        "stem": stem,
        "duration": _round(duration, 2),
        "fps": _round(src_fps, 2),
        "videoWidth": vid_w,
        "videoHeight": vid_h,
        "frameCount": n_frames,
        "frames": frames_out,
    }

    out_dir = OUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frames.json"

    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  → {out_path.relative_to(_HERE.parent)} ({size_mb:.1f} MB, {len(frames_out)} frames)")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else SESSIONS
    print(f"Exporting {len(targets)} session(s) to {OUT_ROOT}")
    for stem in targets:
        export_session(stem)
    print("Done.")


if __name__ == "__main__":
    main()
