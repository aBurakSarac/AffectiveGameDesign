"""render_merged_hud.py — offline merged FER + rPPG fear-detection HUD → mp4.

Re-renders an *already-recorded* session into a single 1920x1080 merged HUD mp4,
combining the FER outputs (``fer/test_mp_hs.py``) and the rPPG outputs
(``rppg/live_rppg.py``) over the **raw recorded video**, laid out and coloured
like ``Merged Fear HUD.html`` (the Claude Design handoff bundle).

This is NOT live capture.  It decodes a saved ``raw_video.mp4`` frame-by-frame
and overlays the HUD, exactly like ``rppg/replay.py`` does for the rPPG-only HUD
— it just draws the merged panel instead.  No camera is opened.

Run with NO flags; you are prompted interactively for the inputs::

    python Pipeline/render_merged_hud.py            # interactive picker
    python Pipeline/render_merged_hud.py list       # just list sessions

Formula chain (mirrors ``fer/fusion.py`` + ``rppg/evaluate_rppg.py``)::

    base = 0.7*hs_fear + 0.3*hs_arousal
    F12  = clamp( base * (1 + mp_tension) )                 # threshold 0.70
    bpm_norm = clip( (bpm - rolling_median_baseline)/baseline, 0, 1 )   # 60s median
    F15  = clamp( F12 * (1 + 0.5*bpm_norm) )                # threshold 0.80  (production)
    verdict: F15 >= 0.80  ->  FEAR DETECTED, else NO FEAR
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ── Allow running from anywhere: python Pipeline/render_merged_hud.py ───────
_HERE = Path(__file__).resolve().parent          # Pipeline/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fer.fusion import compute_composite_fear                       # F12  (noqa: E402)
from rppg.replay import BaseReplayRenderer                          # reused helpers (noqa: E402)
from rppg.analyzer import RppgAnalyzer, AnalysisConfig              # optional regen (noqa: E402)
from rppg.session import SessionRepository                          # session picker (noqa: E402)


# ===========================================================================
# 1. TWEAKS → CONSTANTS  (the HTML had a live Tweaks panel; the mp4 has no UI,
#    so every tweak is a module-level constant. Default first, alternatives
#    commented out directly beneath so they can be switched by editing here.)
# ===========================================================================

# Label style on signal rows.
LABEL_MODE = "plain_tech"      # plain-language name + technical subscript (default)
# LABEL_MODE = "plain"         # plain language only
# LABEL_MODE = "tech"          # technical names only (hs_fear, mp_tension, …)

SHOW_OTHER_ALGOS = True        # show CHROM/GREEN/ICA/WAVELET/CONSENSUS chips
# SHOW_OTHER_ALGOS = False     # POS only

RPPG_COEFF = 0.5               # c in F15 = F12*(1 + c*bpm_norm)  (production)
# RPPG_COEFF = 0.3
# RPPG_COEFF = 0.8

FER_RPPG_TIME_OFFSET_S = 0.0   # +/- shift if FER and rPPG clocks differ for a session

THRESH_F12 = 0.70
THRESH_F15 = 0.80
POS_WINDOW_S = 30
POS_STEP_S = 5                 # matches production RPPG_CONFIG_POS_30S (handoff §6 lists 3;
# POS_STEP_S = 3               #   the shipped sessions were analysed at 5s). Only used when
#                                regenerating a missing analysis.csv.
BASELINE_WINDOW_S = 60

# Video fit inside the left panel.
VIDEO_FIT = "contain"          # letterbox, full frame visible → ROI box maps exactly (default)
# VIDEO_FIT = "cover"          # fill+crop like the HTML image-slot (crops the frame)

# Output framerate.
OUTPUT_FPS_MODE = "source"     # write at the source video's reported fps (handoff acceptance)
# OUTPUT_FPS_MODE = "effective"  # write at the frames.csv median fps (real-time playback)


# ===========================================================================
# 2. COLOURS  (from hud.css — oklch in CSS; RGB approximations here for PIL.
#    PIL works in RGB; we flip to BGR only when handing frames to cv2.)
# ===========================================================================

C = {
    "bg":          (27, 27, 34),     # #1b1b22  canvas
    "telemetry":   (33, 33, 41),     # telemetry bar bg  (oklch 0.20)
    "panel":       (30, 30, 38),     # right panel bg    (oklch 0.185)
    "surface":     (38, 38, 48),     # #262630  cards
    "surface2":    (47, 47, 59),     # surface-2
    "video_bg":    (22, 22, 29),     # video area bg
    "line":        (58, 58, 71),     # borders
    "line_soft":   (47, 47, 59),     # soft borders
    "ink":         (243, 243, 246),  # text
    "ink2":        (184, 184, 192),
    "ink3":        (135, 135, 143),
    "ink4":        (106, 106, 114),
    "danger":      (232, 88, 77),    # #e8584d  Fear / verdict / OVER
    "danger_dim":  (158, 58, 51),
    "arousal":     (224, 168, 58),   # #e0a83a  Arousal
    "tension":     (176, 127, 214),  # #b07fd6  tension accent
    "heart":       (224, 114, 107),  # #e0726b  heart-rate accent
    "clear":       (94, 200, 168),   # #5ec8a8  NO-FEAR / ROI box
    "instrument":  (74, 144, 217),
}


def blend(fg: tuple, bg: tuple, a: float) -> tuple:
    """Pre-blend a translucent fg over a known solid bg (avoids per-frame alpha
    compositing — keeps the renderer fast).  a = fg opacity in [0,1]."""
    return tuple(int(round(fg[i] * a + bg[i] * (1 - a))) for i in range(3))


# Pre-blended tints used on known backgrounds.
HERO_BG       = blend(C["danger"], C["surface"], 0.07)    # Fear hero card bg
HERO_BORDER   = blend(C["danger"], C["surface"], 0.40)
FEAR_WIN_TINT = blend(C["danger"], C["surface"], 0.12)    # trace fear-window shading
TENSION_CHIP  = blend(C["tension"], C["surface2"], 0.18)
HEART_CHIP    = blend(C["heart"], C["surface2"], 0.18)
TENSION_BORDER = blend(C["tension"], C["surface2"], 0.45)
HEART_BORDER   = blend(C["heart"], C["surface2"], 0.45)
VERDICT_ICON_FEAR_BG  = blend(C["danger"], C["surface"], 0.16)
VERDICT_ICON_CLEAR_BG = blend(C["clear"], C["surface"], 0.14)


# ===========================================================================
# 3. FONTS  (UI = clean sans, all numbers monospaced/tabular — handoff §7.
#    IBM Plex isn't bundled; DejaVu Sans / DejaVu Sans Mono are the fallback.)
# ===========================================================================

_SANS_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
]
_SANS_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
]
_MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "C:/Windows/Fonts/consola.ttf",
]
_MONO_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "C:/Windows/Fonts/consolab.ttf",
]


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


_FONT_FILES = {
    ("sans", False): _first_existing(_SANS_CANDIDATES),
    ("sans", True):  _first_existing(_SANS_BOLD_CANDIDATES),
    ("mono", False): _first_existing(_MONO_CANDIDATES),
    ("mono", True):  _first_existing(_MONO_BOLD_CANDIDATES),
}
_FONT_CACHE: Dict[tuple, ImageFont.FreeTypeFont] = {}


def font(kind: str = "sans", size: int = 14, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (kind, size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _FONT_FILES.get((kind, bold)) or _FONT_FILES.get((kind, False))
    try:
        f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    except OSError:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


# ===========================================================================
# 4. DRAWING TOOLKIT  (thin PIL helpers shared by every panel)
# ===========================================================================

class Pen:
    """Wraps an ImageDraw on an RGB Image with convenience primitives."""

    def __init__(self, img: Image.Image):
        self.img = img
        self.d = ImageDraw.Draw(img)

    # -- shapes ----------------------------------------------------------
    def rect(self, box, fill=None, outline=None, width=1):
        self.d.rectangle(box, fill=fill, outline=outline, width=width)

    def rrect(self, box, radius, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(box, radius=radius, fill=fill,
                                 outline=outline, width=width)

    def hline(self, x1, x2, y, fill, width=1):
        self.d.line([(x1, y), (x2, y)], fill=fill, width=width)

    def vline(self, x, y1, y2, fill, width=1):
        self.d.line([(x, y1), (x, y2)], fill=fill, width=width)

    def dashed_hline(self, x1, x2, y, fill, dash=5, gap=5, width=1):
        x = x1
        while x < x2:
            self.d.line([(x, y), (min(x + dash, x2), y)], fill=fill, width=width)
            x += dash + gap

    def polyline(self, pts, fill, width=1):
        if len(pts) >= 2:
            self.d.line(pts, fill=fill, width=width, joint="curve")

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r],
                       fill=fill, outline=outline, width=width)

    # -- text ------------------------------------------------------------
    def text(self, x, y, s, f, fill, anchor="la", tracking=0.0):
        """Draw text. `tracking` (extra px between glyphs) only supports left
        horizontal anchors ('la','lm','ls')."""
        if tracking and tracking > 0:
            cx = x
            for ch in s:
                self.d.text((cx, y), ch, font=f, fill=fill, anchor="l" + anchor[1])
                cx += self.d.textlength(ch, font=f) + tracking
        else:
            self.d.text((x, y), s, font=f, fill=fill, anchor=anchor)

    def text_w(self, s, f, tracking=0.0):
        w = self.d.textlength(s, font=f)
        if tracking and len(s) > 1:
            w += tracking * (len(s) - 1)
        return w


def fmt(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def pct_clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ===========================================================================
# 5. DATA  — labels, schema helpers
# ===========================================================================

EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]
OTHER_EMOTIONS = [e for e in EMOTION_LABELS if e != "Fear"]
# Other rPPG algorithms shown as chips (POS is the production/headline one).
CHIP_ALGOS = ["CHROM", "GREEN", "ICA", "WAVELET", "CONSENSUS"]


@dataclass
class FrameState:
    """All per-frame values the HUD needs, already joined + computed."""
    idx: int
    t: float
    frame_num: int
    latency: float
    fps: float
    # facial emotion
    hs_fear: float
    hs_arousal: float
    emotions: Dict[str, float]
    dom: str
    dom_score: float
    # mediapipe amplifier
    mp_tension: float
    valence: float
    smile: float
    startle: float
    # rPPG amplifier
    has_bpm: bool
    bpm: float
    bpm_norm: float
    baseline: float
    algos: Dict[str, float]          # algo -> bpm  (empty if no window yet)
    # formula chain
    base: float
    mp_mult: float
    rppg_mult: float
    f12: float
    f15: float
    is_fear: bool


# ---------------------------------------------------------------------------
# Loading + joining the two pipelines' artefacts
# ---------------------------------------------------------------------------

def _num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def load_fer(fer_csv: Path) -> pd.DataFrame:
    """Load the FER full CSV (``<session>_mp_hs_temp.csv``).

    Reads columns by name and tolerates missing ones (defaults to 0 / "Neutral").
    We recompute F12 from hs_fear/hs_arousal/mp_tension via fer.fusion so the
    score is consistent even if the CSV predates the f12 column.
    """
    df = pd.read_csv(fer_csv, comment="#")
    out = pd.DataFrame()
    out["timestamp"] = _num(df.get("timestamp", pd.Series(range(len(df)))))
    out["frame"] = _num(df.get("frame", pd.Series(range(len(df))))).astype(int)
    out["hs_fear"] = _num(df.get("hs_fear", 0.0))
    out["hs_arousal"] = _num(df.get("hs_arousal", 0.0))
    out["mp_tension"] = _num(df.get("mp_tension", 0.0))
    out["hs_dominant"] = df.get("hs_dominant", pd.Series(["Neutral"] * len(df))).astype(str)
    out["hs_dominant_score"] = _num(df.get("hs_dominant_score", 0.0))
    out["mp_face_valence"] = _num(df.get("mp_face_valence", df.get("mp_valence", 0.0)))
    out["mp_smile_level"] = _num(df.get("mp_smile_level", df.get("mp_smile", 0.0)))
    out["mp_startle_score"] = _num(df.get("mp_startle_score", 0.0))
    out["latency_ms"] = _num(df.get("latency_ms", 0.0))
    for emo in EMOTION_LABELS:
        col = "hs_" + emo.lower()
        out[col] = _num(df.get(col, 0.0))
    # F12 — recompute (mirrors fer.fusion.compute_composite_fear); fall back to CSV.
    out["f12"] = [
        compute_composite_fear(fr, ar, te)
        for fr, ar, te in zip(out["hs_fear"], out["hs_arousal"], out["mp_tension"])
    ]
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def load_pos_windows(analysis_csv: Path) -> Tuple[List[dict], Dict[int, Dict[str, float]]]:
    """Return (pos_windows, per_window_algo_bpm).

    pos_windows: list of POS rows sorted by t_center (dicts with t_start/t_center/
                 t_end/bpm/bpm_smoothed/window_idx).
    per_window_algo_bpm: {window_idx: {ALGO: bpm_smoothed}} for the chips.
    """
    rows = BaseReplayRenderer._load_analysis(analysis_csv)   # reuse replay.py helper
    pos: List[dict] = []
    per_win: Dict[int, Dict[str, float]] = {}
    for r in rows:
        algo = str(r.get("algorithm", "")).upper()
        try:
            widx = int(float(r.get("window_idx", -1)))
            bpm_s = float(r.get("bpm_smoothed") or r.get("bpm"))
        except (TypeError, ValueError):
            continue
        per_win.setdefault(widx, {})[algo] = bpm_s
        if algo == "POS":
            try:
                pos.append({
                    "window_idx": widx,
                    "t_start": float(r["t_start"]),
                    "t_center": float(r["t_center"]),
                    "t_end": float(r["t_end"]),
                    "bpm": float(r["bpm"]),
                    "bpm_smoothed": bpm_s,
                })
            except (KeyError, ValueError):
                continue
    pos.sort(key=lambda w: w["t_center"])
    return pos, per_win


def rolling_baseline(bpm_at_frame: np.ndarray, frame_ts: np.ndarray,
                     baseline_window_s: float = 60.0) -> np.ndarray:
    """60 s centred rolling-median resting baseline (floored at 70 BPM)."""
    n = len(bpm_at_frame)
    if n == 0:
        return np.zeros(0)
    dt = float(np.median(np.diff(frame_ts))) if n > 1 else 0.033
    if dt <= 0:
        dt = 0.033
    win_frames = max(int(baseline_window_s / dt), 1)
    baseline = pd.Series(bpm_at_frame).rolling(
        win_frames, center=True, min_periods=max(win_frames // 4, 1)
    ).median().values
    return np.where(baseline > 30, baseline, 70.0)


def compute_bpm_norm(bpm_at_frame: np.ndarray, frame_ts: np.ndarray,
                     baseline_window_s: float = 60.0) -> np.ndarray:
    """Normalised BPM: fractional rise over a rolling-median resting baseline.

    Mirrors ``rppg/evaluate_rppg.py:compute_bpm_norm`` (re-implemented here to
    avoid importing that 100KB evaluation module — pure function, same maths:
    clip((bpm - 60s_rolling_median)/baseline, 0, 1), baseline floored at 70).
    """
    if len(bpm_at_frame) == 0:
        return np.zeros(0)
    safe_baseline = rolling_baseline(bpm_at_frame, frame_ts, baseline_window_s)
    bpm_norm = (bpm_at_frame - safe_baseline) / safe_baseline
    return np.clip(bpm_norm, 0.0, 1.0)


def build_frame_times(frames_df: Optional[pd.DataFrame], n_video: int,
                      src_fps: float) -> np.ndarray:
    """video frame index -> real timestamp (seconds).

    Prefers frames.csv timestamps (these captures are ~10fps wall-clock even
    though the mp4 is tagged 30fps, so i/fps would be wrong). Extrapolates
    beyond frames.csv using the source fps.
    """
    ts = np.array([i / max(src_fps, 1e-6) for i in range(n_video)], dtype=float)
    if frames_df is None or "timestamp" not in frames_df:
        return ts
    fi = frames_df["frame_idx"].astype(int).values
    ft = frames_df["timestamp"].astype(float).values
    for idx, t in zip(fi, ft):
        if 0 <= idx < n_video:
            ts[idx] = t
    # extrapolate tail (video longer than frames.csv) from last known cadence
    if len(ft) >= 2:
        last_idx = int(fi[-1])
        cad = float(np.median(np.diff(ft)))
        for i in range(last_idx + 1, n_video):
            ts[i] = ts[last_idx] + (i - last_idx) * cad
    return ts


def precompute_states(fer: pd.DataFrame, pos_windows: List[dict],
                      per_win_algo: Dict[int, Dict[str, float]],
                      frame_ts: np.ndarray, src_fps: float) -> List[FrameState]:
    """Join FER (per-frame) + rPPG (per-window) onto every video frame."""
    n = len(frame_ts)

    # ── FER join: nearest FER row by timestamp (clocks aligned; offset knob) ──
    fer_ts = fer["timestamp"].values
    lookup_t = frame_ts + FER_RPPG_TIME_OFFSET_S
    pos_idx = np.searchsorted(fer_ts, lookup_t)
    pos_idx = np.clip(pos_idx, 1, len(fer_ts) - 1) if len(fer_ts) > 1 else np.zeros(n, int)
    left = np.clip(pos_idx - 1, 0, len(fer_ts) - 1)
    right = np.clip(pos_idx, 0, len(fer_ts) - 1)
    choose_left = np.abs(lookup_t - fer_ts[left]) <= np.abs(fer_ts[right] - lookup_t)
    fer_row_idx = np.where(choose_left, left, right)

    # ── rPPG: interpolate POS bpm onto the video timeline ──
    if pos_windows:
        t_centers = np.array([w["t_center"] for w in pos_windows])
        bpms = np.array([w["bpm_smoothed"] for w in pos_windows])
        bpm_at_frame = np.interp(frame_ts, t_centers, bpms)
        first_center = float(t_centers[0])
        valid_mask = frame_ts >= first_center      # honest: no BPM before first window
    else:
        bpm_at_frame = np.full(n, 70.0)
        valid_mask = np.zeros(n, dtype=bool)

    bpm_norm_series = compute_bpm_norm(bpm_at_frame, frame_ts, BASELINE_WINDOW_S)
    baseline_series = rolling_baseline(bpm_at_frame, frame_ts, BASELINE_WINDOW_S)

    # map frame -> covering/nearest POS window_idx (for per-algo chips)
    def window_idx_at(t: float) -> Optional[int]:
        cover = [w for w in pos_windows if w["t_start"] <= t <= w["t_end"]]
        if cover:
            return min(cover, key=lambda w: abs(w["t_center"] - t))["window_idx"]
        prior = [w for w in pos_windows if w["t_center"] <= t]
        return prior[-1]["window_idx"] if prior else None

    states: List[FrameState] = []
    for i in range(n):
        fr = fer.iloc[int(fer_row_idx[i])]
        emotions = {e: float(fr["hs_" + e.lower()]) for e in EMOTION_LABELS}
        s = float(sum(emotions.values())) or 1.0
        emotions = {k: v / s for k, v in emotions.items()}       # normalise distribution
        hs_fear = emotions["Fear"]
        hs_arousal = pct_clamp(float(fr["hs_arousal"]))
        mp_tension = pct_clamp(float(fr["mp_tension"]))

        base = 0.7 * hs_fear + 0.3 * hs_arousal
        mp_mult = 1.0 + mp_tension
        f12 = pct_clamp(base * mp_mult)

        has_bpm = bool(valid_mask[i])
        bpm = float(bpm_at_frame[i]) if has_bpm else float("nan")
        bpm_norm = float(bpm_norm_series[i]) if has_bpm else 0.0
        rppg_mult = 1.0 + RPPG_COEFF * bpm_norm
        f15 = pct_clamp(f12 * rppg_mult)

        algos: Dict[str, float] = {}
        if has_bpm:
            widx = window_idx_at(frame_ts[i])
            if widx is not None:
                algos = dict(per_win_algo.get(widx, {}))

        dom = str(fr["hs_dominant"]) if str(fr["hs_dominant"]) in EMOTION_LABELS else \
            max(emotions, key=emotions.get)
        dom_score = emotions.get(dom, float(fr["hs_dominant_score"]))

        states.append(FrameState(
            idx=i, t=float(frame_ts[i]), frame_num=int(fr["frame"]),
            latency=float(fr["latency_ms"]), fps=src_fps,
            hs_fear=hs_fear, hs_arousal=hs_arousal, emotions=emotions,
            dom=dom, dom_score=dom_score,
            mp_tension=mp_tension, valence=float(fr["mp_face_valence"]),
            smile=pct_clamp(float(fr["mp_smile_level"])),
            startle=max(0.0, float(fr["mp_startle_score"])),
            has_bpm=has_bpm, bpm=bpm, bpm_norm=bpm_norm,
            baseline=float(baseline_series[i]) if has_bpm else 74.0, algos=algos,
            base=base, mp_mult=mp_mult, rppg_mult=rppg_mult,
            f12=f12, f15=f15, is_fear=f15 >= THRESH_F15,
        ))
    return states


# ===========================================================================
# 6. LAYOUT  — fixed 1920x1080 geometry (mirrors hud.css proportions)
# ===========================================================================

W, H = 1920, 1080
TELEMETRY_H = 64
PANEL_W = 768
LEFT_PAD = 22
LEFT_GAP = 18
TIMELINE_H = 196
RADIUS = 14
RADIUS_SM = 9

BODY_Y = TELEMETRY_H
BODY_H = H - TELEMETRY_H                       # 1016
LEFT_X1 = W - PANEL_W                          # 1152
RIGHT_X0 = LEFT_X1

# left column
LEFT_IN_X = LEFT_PAD                           # 22
LEFT_IN_W = LEFT_X1 - 2 * LEFT_PAD             # 1108
VIDEO_Y = BODY_Y + LEFT_PAD                    # 86
VIDEO_H = BODY_H - 2 * LEFT_PAD - LEFT_GAP - TIMELINE_H   # 758
VIDEO_BOX = (LEFT_IN_X, VIDEO_Y, LEFT_IN_X + LEFT_IN_W, VIDEO_Y + VIDEO_H)
TIMELINE_Y = VIDEO_Y + VIDEO_H + LEFT_GAP      # 862
TIMELINE_BOX = (LEFT_IN_X, TIMELINE_Y, LEFT_IN_X + LEFT_IN_W, TIMELINE_Y + TIMELINE_H)

# right column
RPAD_X, RPAD_Y, RGAP = 22, 13, 7
R_IN_X = RIGHT_X0 + 1 + RPAD_X
R_IN_W = PANEL_W - 1 - 2 * RPAD_X              # 723
R_IN_Y = BODY_Y + RPAD_Y                       # 77
R_IN_H = BODY_H - 2 * RPAD_Y                   # 990

# section heights (sum + 3 gaps == R_IN_H)
H_VERDICT = 240
H_PRIMARY = 320
H_AMP = 250
H_CHAIN = R_IN_H - (H_VERDICT + H_PRIMARY + H_AMP) - 3 * RGAP   # 159


# ===========================================================================
# 7. PANELS
# ===========================================================================

class HudRenderer:
    """Draws the full merged HUD onto a 1920x1080 RGB canvas, per frame."""

    def __init__(self, src_w: int, src_h: int, duration: float,
                 trace_t: np.ndarray, trace_f12: np.ndarray,
                 trace_f15: np.ndarray):
        self.src_w, self.src_h = src_w, src_h
        self.duration = max(duration, 1e-6)
        self.trace_t = trace_t
        self.trace_f12 = trace_f12
        self.trace_f15 = trace_f15
        # precompute fear windows (F15 >= threshold) for trace shading
        self.fear_windows = self._fear_windows(trace_t, trace_f15)

    @staticmethod
    def _fear_windows(t: np.ndarray, f15: np.ndarray) -> List[Tuple[float, float]]:
        out, start = [], None
        for i in range(len(t)):
            over = f15[i] >= THRESH_F15
            if over and start is None:
                start = t[i]
            elif not over and start is not None:
                out.append((start, t[i]))
                start = None
        if start is not None:
            out.append((start, t[-1]))
        return out

    # -- label helper (LABEL_MODE) --------------------------------------
    @staticmethod
    def _label_parts(plain: str, tech: str) -> Tuple[str, Optional[str]]:
        if LABEL_MODE == "tech":
            return tech, None
        if LABEL_MODE == "plain":
            return plain, None
        return plain, tech            # plain_tech

    # ===================================================================
    def render(self, frame_rgb: np.ndarray, st: FrameState,
               roi: Optional[Tuple[float, float, float, float]] = None) -> Image.Image:
        img = Image.new("RGB", (W, H), C["bg"])
        p = Pen(img)
        self._telemetry(p, st)
        self._video(img, p, frame_rgb, roi)
        self._timeline(p, st)
        self._verdict(p, st)
        self._primary(p, st)
        self._amplifiers(p, st)
        self._chain(p, st)
        return img

    # ── telemetry top bar ──────────────────────────────────────────────
    def _telemetry(self, p: Pen, st: FrameState):
        p.rect((0, 0, W, TELEMETRY_H), fill=C["telemetry"])
        p.hline(0, W, TELEMETRY_H - 1, C["line"])
        # brand: rec dot + title/sub
        p.circle(26 + 5, 32, 5, fill=C["danger"])
        p.text(44, 18, "Fear Analysis HUD", font("sans", 16, True), C["ink"])
        p.text(44, 39, "FER + rPPG · REPLAY", font("sans", 11), C["ink3"], tracking=1.6)

        # right-aligned telemetry groups
        mm = int(st.t // 60); ss = int(st.t % 60); cs = int((st.t % 1) * 100)
        groups = [
            ("TIME", f"{mm:02d}:{ss:02d}.{cs:02d}", C["ink"]),
            ("FRAME", f"{st.frame_num:05d}", C["ink"]),
            ("LATENCY", f"{round(st.latency)}ms",
             C["arousal"] if st.latency > 40 else C["ink"]),
            ("THROUGHPUT", f"{st.fps:.1f}fps", C["ink2"]),
        ]
        # measure to right-align the whole cluster
        kf, vf = font("sans", 10, True), font("mono", 19, True)
        widths = [max(p.text_w(k, kf, 1.0), p.text_w(v, vf)) for k, v, _ in groups]
        total = sum(widths) + 28 * (len(groups) - 1)
        x = W - 26 - total
        for i, (k, v, col) in enumerate(groups):
            p.text(x, 16, k, kf, C["ink4"], tracking=1.0)
            p.text(x, 30, v, vf, col)
            if i < len(groups) - 1:
                sep_x = x + widths[i] + 14
                p.vline(sep_x, 18, 46, C["line"])
            x += widths[i] + 28

    # ── video area + forehead ROI box ──────────────────────────────────
    def _video(self, img: Image.Image, p: Pen, frame_rgb: np.ndarray,
               roi: Optional[Tuple[float, float, float, float]] = None):
        x0, y0, x1, y1 = VIDEO_BOX
        p.rrect(VIDEO_BOX, RADIUS, fill=C["video_bg"])
        bw, bh = x1 - x0, y1 - y0
        vw, vh = self.src_w, self.src_h
        if VIDEO_FIT == "cover":
            scale = max(bw / vw, bh / vh)
        else:                                   # contain
            scale = min(bw / vw, bh / vh)
        nw, nh = int(round(vw * scale)), int(round(vh * scale))
        ox = x0 + (bw - nw) // 2
        oy = y0 + (bh - nh) // 2
        vid = Image.fromarray(frame_rgb).resize((nw, nh), Image.BILINEAR)
        if VIDEO_FIT == "cover":
            cx = max(0, (nw - bw) // 2); cy = max(0, (nh - bh) // 2)
            vid = vid.crop((cx, cy, cx + min(nw, bw), cy + min(nh, bh)))
            ox, oy = x0, y0
        img.paste(vid, (ox, oy))

        # forehead rPPG ROI box (green) — handoff §4: from frames.csv roi_bbox
        if roi:
            rx0, ry0, rx1, ry1 = roi
            dx0 = ox + rx0 * scale; dy0 = oy + ry0 * scale
            dx1 = ox + rx1 * scale; dy1 = oy + ry1 * scale
            # clamp into the video box
            dx0, dx1 = max(x0, dx0), min(x1, dx1)
            dy0, dy1 = max(y0, dy0), min(y1, dy1)
            if dx1 > dx0 and dy1 > dy0:
                p.rrect((dx0, dy0, dx1, dy1), 4, outline=C["clear"], width=2)
                p.text(dx0, dy0 - 20, "forehead · rPPG ROI",
                       font("mono", 13, True), C["clear"])

        # "analysis replay" badge top-right (always shown — mock's static caption)
        badge = "ANALYSIS REPLAY"
        bf = font("mono", 12)
        tw = p.text_w(badge, bf, 1.2)
        bx1 = x1 - 16; bx0 = bx1 - tw - 22
        p.rrect((bx0, y0 + 16, bx1, y0 + 16 + 28), 7,
                fill=C["video_bg"], outline=C["line_soft"], width=1)
        p.text(bx0 + 11, y0 + 16 + 8, badge, bf, C["ink3"], tracking=1.2)

        # re-stroke the rounded card border on top
        p.rrect(VIDEO_BOX, RADIUS, outline=C["line"], width=1)

    # ── trace timeline (full curves + moving playhead, no scrubber) ─────
    def _timeline(self, p: Pen, st: FrameState):
        x0, y0, x1, y1 = TIMELINE_BOX
        p.rrect(TIMELINE_BOX, RADIUS, fill=C["surface"], outline=C["line"], width=1)
        # head: title + legend
        p.text(x0 + 18, y0 + 12, "Fear score over time", font("sans", 13, True), C["ink"])
        lf = font("sans", 11)
        legend = [
            ("F15 (+ heart rate)", C["ink2"], 3),
            ("F12 (face only)", C["ink4"], 3),
            (f"thr {fmt(THRESH_F15)}", C["ink2"], 2),
            (f"thr {fmt(THRESH_F12)}", C["ink4"], 2),
        ]
        items = []
        for txt, col, th in legend:
            w = 14 + 6 + p.text_w(txt, lf)
            items.append((txt, col, th, w))
        lxs = x1 - 18 - (sum(w for *_, w in items) + 16 * (len(items) - 1))
        cx = lxs
        for txt, col, th, w in items:
            p.d.line([(cx, y0 + 18), (cx + 14, y0 + 18)], fill=col, width=th)
            p.text(cx + 20, y0 + 12, txt, lf, C["ink3"])
            cx += w + 16

        # plot area
        px0 = x0 + 18; px1 = x1 - 18
        py0 = y0 + 14 + 30; py1 = y1 - 14
        pad = 8

        def X(t):
            return px0 + (t / self.duration) * (px1 - px0)

        def Y(v):
            return py1 - pad - pct_clamp(v) * (py1 - py0 - 2 * pad)

        # fear-window shading
        for (a, b) in self.fear_windows:
            p.rect((X(a), py0, X(b), py1), fill=FEAR_WIN_TINT)
        # threshold lines
        p.dashed_hline(px0, px1, Y(THRESH_F12), C["ink4"], dash=4, gap=6, width=1)
        p.dashed_hline(px0, px1, Y(THRESH_F15), C["ink2"], dash=5, gap=5, width=1)
        # curves
        step = max(1, len(self.trace_t) // 1000)
        f12_pts = [(X(self.trace_t[i]), Y(self.trace_f12[i]))
                   for i in range(0, len(self.trace_t), step)]
        f15_pts = [(X(self.trace_t[i]), Y(self.trace_f15[i]))
                   for i in range(0, len(self.trace_t), step)]
        p.polyline(f12_pts, C["ink4"], width=2)
        p.polyline(f15_pts, C["ink2"], width=3)
        # playhead
        hx = X(st.t)
        p.vline(hx, py0, py1, blend(C["ink"], C["surface"], 0.5), width=1)
        p.circle(hx, Y(st.f15), 5, fill=C["ink"], outline=C["bg"], width=2)

    # ── verdict card (dual F12 / F15 gauges) ───────────────────────────
    def _verdict(self, p: Pen, st: FrameState):
        x0 = R_IN_X; x1 = R_IN_X + R_IN_W
        y0 = R_IN_Y; y1 = y0 + H_VERDICT
        fear = st.is_fear
        border = C["danger"] if fear else C["clear"]
        p.rrect((x0, y0, x1, y1), RADIUS, fill=C["surface"],
                outline=border, width=2 if fear else 1)

        # top row: icon · words · score
        icon_bg = VERDICT_ICON_FEAR_BG if fear else VERDICT_ICON_CLEAR_BG
        ix, iy = x0 + 18, y0 + 12
        p.rrect((ix, iy, ix + 56, iy + 56), RADIUS, fill=icon_bg)
        self._verdict_icon(p, ix + 28, iy + 28, fear, border)

        wx = ix + 56 + 18
        p.text(wx, y0 + 18, "FEAR DETECTED" if fear else "NO FEAR",
               font("sans", 30, True), border)
        p.text(wx, y0 + 56, "final decision · F15 (production)",
               font("sans", 12), C["ink3"])
        # score (right)
        sf = font("mono", 38, True)
        sval = fmt(st.f15)
        p.text(x1 - 18, y0 + 16, sval, sf, C["danger"] if fear else C["ink"], anchor="ra")
        p.text(x1 - 18, y0 + 60, "F15 SCORE", font("sans", 11), C["ink4"],
               anchor="ra", tracking=1.4)

        # dual gauges
        gy = y0 + 92
        self._gauge_row(p, x0 + 18, x1 - 18, gy, "F12", "face only",
                        st.f12, THRESH_F12)
        self._gauge_row(p, x0 + 18, x1 - 18, gy + 58, "F15", "+ heart rate",
                        st.f15, THRESH_F15)
        # scale
        scf = font("mono", 11)
        sy = gy + 58 + 44
        p.text(x0 + 18, sy, "0.00", scf, C["ink4"])
        p.text((x0 + x1) / 2, sy, "0.50", scf, C["ink4"], anchor="ma")
        p.text(x1 - 18, sy, "1.00", scf, C["ink4"], anchor="ra")

    @staticmethod
    def _verdict_icon(p: Pen, cx, cy, fear, col):
        if fear:   # warning triangle
            p.d.line([(cx, cy - 13), (cx + 13, cy + 11), (cx - 13, cy + 11), (cx, cy - 13)],
                     fill=col, width=3, joint="curve")
            p.vline(cx, cy - 4, cy + 3, col, width=3)
            p.circle(cx, cy + 8, 2, fill=col)
        else:      # check circle
            p.circle(cx, cy, 13, outline=col, width=3)
            p.d.line([(cx - 6, cy + 1), (cx - 2, cy + 6), (cx + 7, cy - 5)],
                     fill=col, width=3, joint="curve")

    def _gauge_row(self, p: Pen, x0, x1, y, tag, sub, score, thr):
        over = score >= thr
        fill_col = C["ink2"] if over else C["ink4"]
        # head line
        p.text(x0, y, tag, font("mono", 14, True), C["ink2"])
        tagw = p.text_w(tag, font("mono", 14, True))
        p.text(x0 + tagw + 9, y + 1, sub, font("sans", 12), C["ink4"])
        # right side: value, OVER/UNDER, needs thr
        numf = font("mono", 17, True)
        p.text(x1, y, fmt(score), numf, C["danger"] if over else C["ink"], anchor="ra")
        nw = p.text_w(fmt(score), numf)
        stf = font("mono", 11, True)
        state = "OVER" if over else "UNDER"
        p.text(x1 - nw - 12, y + 3, state, stf,
               C["danger"] if over else C["ink3"], anchor="ra")
        sw = p.text_w(state, stf)
        p.text(x1 - nw - 12 - sw - 12, y + 3, f"needs {fmt(thr)}",
               font("mono", 11), C["ink3"], anchor="ra")
        # track
        ty = y + 22
        p.rrect((x0, ty, x1, ty + 30), 8, fill=C["surface2"], outline=C["line_soft"], width=1)
        fw = pct_clamp(score) * (x1 - x0)
        if fw > 2:
            p.rrect((x0, ty, x0 + fw, ty + 30), 8, fill=fill_col)
        # threshold tick
        tx = x0 + pct_clamp(thr) * (x1 - x0)
        p.vline(tx, ty - 5, ty + 35, C["ink"], width=2)

    # ── primary signals (Fear hero + Arousal + emotion distribution) ────
    def _primary(self, p: Pen, st: FrameState):
        x0 = R_IN_X; x1 = R_IN_X + R_IN_W
        y0 = R_IN_Y + H_VERDICT + RGAP; y1 = y0 + H_PRIMARY
        self._section_frame(p, x0, x1, y0, y1, "1",
                            "Primary signal · facial emotion", "HSEmotion")
        bx0, bx1 = x0 + 16, x1 - 16
        y = y0 + 44

        # Fear hero
        hero_h = 56
        p.rrect((bx0, y, bx1, y + hero_h), 10, fill=HERO_BG, outline=HERO_BORDER, width=1)
        plain, tech = self._label_parts("Fear", "hs_fear")
        p.text(bx0 + 14, y + 8, plain, font("sans", 19, True), C["ink"])
        if tech:
            p.text(bx0 + 14 + p.text_w(plain, font("sans", 19, True)) + 8, y + 14,
                   tech, font("mono", 11), C["ink4"])
        p.text(bx1 - 14, y + 8, fmt(st.hs_fear), font("mono", 22, True),
               C["danger"], anchor="ra")
        self._bar(p, bx0 + 14, bx1 - 14, y + 36, 16, st.hs_fear, C["danger"])
        y += hero_h + 10

        # Arousal
        plain, tech = self._label_parts("Arousal", "hs_arousal")
        p.text(bx0, y, plain, font("sans", 19, True), C["ink"])
        if tech:
            p.text(bx0 + p.text_w(plain, font("sans", 19, True)) + 8, y + 6,
                   tech, font("mono", 11), C["ink4"])
        p.text(bx1, y, fmt(st.hs_arousal), font("mono", 22, True),
               C["arousal"], anchor="ra")
        self._bar(p, bx0, bx1, y + 28, 16, st.hs_arousal, C["arousal"])
        y += 56

        # base-note chip
        p.rrect((bx0, y, bx1, y + 32), RADIUS_SM, fill=C["surface2"])
        p.text(bx0 + 13, y + 8, "Base score", font("sans", 13), C["ink2"])
        seg = "= 0.7×Fear + 0.3×Arousal ="
        p.text(bx0 + 13 + p.text_w("Base score ", font("sans", 13)), y + 8,
               seg, font("mono", 13), C["ink3"])
        p.text(bx1 - 13, y + 8, fmt(st.base), font("mono", 14, True), C["ink"], anchor="ra")
        y += 42

        # emotion distribution
        p.text(bx0, y, f"FULL EMOTION DISTRIBUTION · DOMINANT: {st.dom.upper()} "
                       f"({fmt(st.dom_score)})", font("sans", 11), C["ink4"], tracking=0.6)
        y += 20
        col_gap = 22
        col_w = (bx1 - bx0 - col_gap) / 2
        rows = math.ceil(len(OTHER_EMOTIONS) / 2)
        for k, emo in enumerate(OTHER_EMOTIONS):
            r, c = k % rows, k // rows           # column-major to fill 2 columns evenly
            ex = bx0 + c * (col_w + col_gap)
            ey = y + r * 22
            is_dom = (emo == st.dom)
            name_col = C["ink"] if is_dom else C["ink3"]
            p.text(ex, ey, emo, font("sans", 13, is_dom), name_col)
            ebx0 = ex + 76; ebx1 = ex + col_w - 38
            self._bar(p, ebx0, ebx1, ey + 8, 6,
                      st.emotions[emo], C["ink2"] if is_dom else C["ink4"], bg=C["surface2"])
            p.text(ex + col_w, ey, fmt(st.emotions[emo]), font("mono", 11),
                   C["ink2"] if is_dom else C["ink4"], anchor="ra")

    # ── amplifiers (MediaPipe tension + rPPG heart rate) ───────────────
    def _amplifiers(self, p: Pen, st: FrameState):
        x0 = R_IN_X; x1 = R_IN_X + R_IN_W
        y0 = R_IN_Y + H_VERDICT + H_PRIMARY + 2 * RGAP; y1 = y0 + H_AMP
        self._section_frame(p, x0, x1, y0, y1, "2",
                            "Amplifiers · do the body & face agree?",
                            "multiply the base score")
        gap = 14
        cw = (x1 - 16 - (x0 + 16) - gap) / 2
        cy0 = y0 + 44; cy1 = y1 - 12
        ax0 = x0 + 16
        self._amp_tension(p, ax0, ax0 + cw, cy0, cy1, st)
        self._amp_heart(p, ax0 + cw + gap, ax0 + 2 * cw + gap, cy0, cy1, st)

    def _amp_card(self, p: Pen, x0, x1, y0, y1, border):
        p.rrect((x0, y0, x1, y1), 11, fill=C["surface2"], outline=border, width=1)

    def _amp_tension(self, p, x0, x1, y0, y1, st: FrameState):
        self._amp_card(p, x0, x1, y0, y1, TENSION_BORDER)
        ix = x0 + 14
        p.circle(ix + 4, y0 + 14, 4, fill=C["tension"])
        p.text(ix + 14, y0 + 8, "Facial tension", font("sans", 13, True), C["ink"])
        chip = f"×{fmt(st.mp_mult)}"
        cf = font("mono", 16, True)
        cw = p.text_w(chip, cf)
        p.rrect((x1 - 14 - cw - 14, y0 + 6, x1 - 14, y0 + 6 + 24), 7, fill=TENSION_CHIP)
        p.text(x1 - 14 - 7, y0 + 10, chip, cf, C["tension"], anchor="ra")
        sub = ("mp_tension · ×(1+mp_tension)" if LABEL_MODE == "tech"
               else "MediaPipe · brow / jaw / eye strain")
        p.text(ix, y0 + 34, sub, font("sans", 11), C["ink4"])
        # big number
        p.text(ix, y0 + 52, fmt(st.mp_tension), font("mono", 30, True), C["tension"])
        nw = p.text_w(fmt(st.mp_tension), font("mono", 30, True))
        p.text(ix + nw + 8, y0 + 70, "/ 1.00", font("sans", 13), C["ink3"])
        # mini-bar
        self._bar(p, ix, x1 - 14, y0 + 92, 8, st.mp_tension, C["tension"],
                  bg=C["bg"])
        # mp stats
        sy = y0 + 112
        stats = [("VALENCE", f"{'+' if st.valence >= 0 else ''}{fmt(st.valence)}",
                  C["danger"] if st.valence < 0 else C["clear"]),
                 ("SMILE", fmt(st.smile), C["ink"]),
                 ("STARTLE", f"{fmt(st.startle, 1)}/s", C["ink"])]
        sx = ix
        for k, v, col in stats:
            p.text(sx, sy, k, font("sans", 10), C["ink4"], tracking=0.5)
            p.text(sx, sy + 14, v, font("mono", 14, True), col)
            sx += 88

    def _amp_heart(self, p, x0, x1, y0, y1, st: FrameState):
        self._amp_card(p, x0, x1, y0, y1, HEART_BORDER)
        ix = x0 + 14
        p.circle(ix + 4, y0 + 14, 4, fill=C["heart"])
        p.text(ix + 14, y0 + 8, "Heart rate", font("sans", 13, True), C["ink"])
        chip = f"×{fmt(st.rppg_mult)}"
        cf = font("mono", 16, True)
        cw = p.text_w(chip, cf)
        p.rrect((x1 - 14 - cw - 14, y0 + 6, x1 - 14, y0 + 6 + 24), 7, fill=HEART_CHIP)
        p.text(x1 - 14 - 7, y0 + 10, chip, cf, C["heart"], anchor="ra")

        if not st.has_bpm:
            p.text(ix, y0 + 34, "no rPPG window yet — F15 = F12",
                   font("sans", 11), C["ink4"])
            p.text(ix, y0 + 52, "—", font("mono", 30, True), C["ink3"])
            p.text(ix + 28, y0 + 70, "BPM", font("sans", 13), C["ink3"])
            return

        sub = ("POS bpm · ×(1+0.5·bpm_norm)" if LABEL_MODE == "tech"
               else "rPPG (POS) · pulse from skin colour")
        p.text(ix, y0 + 34, sub, font("sans", 11), C["ink4"])
        p.text(ix, y0 + 52, f"{round(st.bpm)}", font("mono", 30, True), C["heart"])
        nw = p.text_w(f"{round(st.bpm)}", font("mono", 30, True))
        p.text(ix + nw + 8, y0 + 70, "BPM", font("sans", 13), C["ink3"])
        rise = ("rise bpm_norm " if LABEL_MODE == "tech" else "rise ") + fmt(st.bpm_norm)
        p.text(x1 - 14, y0 + 60, rise, font("mono", 12), C["ink3"], anchor="ra")
        # baseline→now strip
        sx0, sx1 = ix, x1 - 14
        ly = y0 + 100
        p.hline(sx0, sx1, ly, C["line"], width=2)
        bpos = sx0 + 0.08 * (sx1 - sx0)
        p.vline(bpos, ly - 8, ly + 8, C["ink4"], width=2)
        p.text(bpos, ly - 22, f"rest {round(st.baseline)}", font("mono", 10),
               C["ink4"], anchor="ma")
        nfrac = pct_clamp((st.bpm - 60) / 60)
        nfrac = min(0.96, max(0.08, nfrac))
        npos = sx0 + nfrac * (sx1 - sx0)
        p.vline(npos, ly - 8, ly + 8, C["heart"], width=2)
        p.text(npos, ly - 22, f"now {round(st.bpm)}", font("mono", 10),
               C["heart"], anchor="ma")
        # other algo chips
        if SHOW_OTHER_ALGOS and st.algos:
            cy = y0 + 122
            p.text(ix, cy, "OTHER rPPG ALGORITHMS (POS IS PRODUCTION)",
                   font("sans", 10), C["ink4"], tracking=0.4)
            cy += 16
            n = len(CHIP_ALGOS)
            chip_gap = 7
            chw = (sx1 - sx0 - chip_gap * (n - 1)) / n
            for i, algo in enumerate(CHIP_ALGOS):
                cx0 = sx0 + i * (chw + chip_gap)
                p.rrect((cx0, cy, cx0 + chw, cy + 34), 7,
                        fill=C["bg"], outline=C["line_soft"], width=1)
                p.text(cx0 + chw / 2, cy + 5, algo[:5], font("sans", 10),
                       C["ink4"], anchor="ma")
                bpm = st.algos.get(algo)
                txt = f"{round(bpm)}" if bpm else "—"
                p.text(cx0 + chw / 2, cy + 17, txt, font("mono", 14, True),
                       C["ink2"], anchor="ma")

    # ── formula chain ──────────────────────────────────────────────────
    def _chain(self, p: Pen, st: FrameState):
        x0 = R_IN_X; x1 = R_IN_X + R_IN_W
        y0 = R_IN_Y + H_VERDICT + H_PRIMARY + H_AMP + 3 * RGAP; y1 = y0 + H_CHAIN
        self._section_frame(p, x0, x1, y0, y1, "3", "How the score is built",
                            "F12 = base × (1+mp_tension) · F15 = F12 × (1+0.5·bpm_norm)")
        bx0, bx1 = x0 + 12, x1 - 12
        cy = y0 + 44
        op_w = 78
        node_w = (bx1 - bx0 - 2 * op_w) / 3
        # node helper
        def node(nx, cap, val, form, val_col):
            p.text(nx + node_w / 2, cy, cap, font("sans", 11), C["ink4"], anchor="ma")
            p.text(nx + node_w / 2, cy + 16, val, font("mono", 22, True), val_col, anchor="ma")
            p.text(nx + node_w / 2, cy + 46, form, font("mono", 11), C["ink4"], anchor="ma")

        def op(ox, mult, label, col):
            p.text(ox + op_w / 2, cy + 8, f"× {fmt(mult)}", font("mono", 15, True),
                   col, anchor="ma")
            p.text(ox + op_w / 2, cy + 28, "→", font("sans", 18), C["ink4"], anchor="ma")
            chip = label
            cf = font("mono", 11)
            cw = p.text_w(chip, cf)
            chip_bg = TENSION_CHIP if col == C["tension"] else HEART_CHIP
            p.rrect((ox + op_w / 2 - cw / 2 - 6, cy + 48, ox + op_w / 2 + cw / 2 + 6,
                     cy + 48 + 18), 5, fill=chip_bg)
            p.text(ox + op_w / 2, cy + 50, chip, cf, col, anchor="ma")

        nx = bx0
        node(nx, "Base", fmt(st.base), "fear + arousal", C["ink"])
        nx += node_w
        op(nx, st.mp_mult, "tension", C["tension"])
        nx += op_w
        node(nx, "F12 · face only", fmt(st.f12), f"≥ {fmt(THRESH_F12)}?", C["ink"])
        nx += node_w
        op(nx, st.rppg_mult, "heart", C["heart"])
        nx += op_w
        final_col = C["danger"] if st.is_fear else C["clear"]
        node(nx, "F15 · production", fmt(st.f15), f"≥ {fmt(THRESH_F15)}?", final_col)

    # ── shared section frame (rounded surface + sec-head) ──────────────
    def _section_frame(self, p: Pen, x0, x1, y0, y1, idx, label, note):
        p.rrect((x0, y0, x1, y1), RADIUS, fill=C["surface"], outline=C["line"], width=1)
        # sec-head
        p.rrect((x0 + 16, y0 + 8, x0 + 16 + 20, y0 + 8 + 20), 6, fill=C["ink3"])
        p.text(x0 + 16 + 10, y0 + 9, idx, font("mono", 11, True), C["bg"], anchor="ma")
        p.text(x0 + 16 + 30, y0 + 11, label.upper(), font("sans", 13, True),
               C["ink2"], tracking=0.6)
        p.text(x1 - 16, y0 + 13, note, font("sans", 11), C["ink4"], anchor="ra")
        p.hline(x0 + 12, x1 - 12, y0 + 36, C["line_soft"])

    @staticmethod
    def _bar(p: Pen, x0, x1, y, h, value, color, bg=None):
        bg = bg or C["surface2"]
        r = h / 2
        p.rrect((x0, y, x1, y + h), r, fill=bg, outline=C["line_soft"], width=1)
        fw = pct_clamp(value) * (x1 - x0)
        if fw > 2:
            p.rrect((x0, y, x0 + fw, y + h), r, fill=color)


# ===========================================================================
# 8. ROI lookup from frames.csv  (forehead box, original-frame coords)
# ===========================================================================

def load_roi_by_frame(frames_csv: Optional[Path]) -> Dict[int, Tuple[float, float, float, float]]:
    """{frame_idx: (x1,y1,x2,y2)} from frames.csv (prefers roi_bbox over bbox)."""
    out: Dict[int, Tuple[float, float, float, float]] = {}
    if not frames_csv or not Path(frames_csv).exists():
        return out
    bbox_map = BaseReplayRenderer._load_frame_bboxes(Path(frames_csv))  # reuse helper
    for idx, (bbox_str, _src) in bbox_map.items():
        if not bbox_str:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox_str.split(",")]
            out[idx] = (x1, y1, x2, y2)
        except (ValueError, AttributeError):
            continue
    return out


# ===========================================================================
# 9. RENDER LOOP
# ===========================================================================

def render_session(video_path: Path, fer_csv: Path, analysis_csv: Path,
                   frames_csv: Optional[Path], output_path: Path,
                   gt_csv: Optional[Path] = None,
                   frame_stride: int = 1, max_frames: Optional[int] = None) -> None:
    """Render the merged HUD mp4 for one recorded session.

    frame_stride: render every Nth frame (output fps is divided by N so playback
                  stays real-time). Use 2-3 for faster previews of long recordings.
    max_frames:   stop after this many *rendered* frames (None = whole video).
                  Handy for quick smoke renders.
    """
    frame_stride = max(1, int(frame_stride))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # load artefacts
    fer = load_fer(fer_csv)
    pos_windows, per_win_algo = load_pos_windows(analysis_csv)
    frames_df = (pd.read_csv(frames_csv, comment="#")
                 if frames_csv and Path(frames_csv).exists() else None)
    roi_map = load_roi_by_frame(frames_csv)

    frame_ts = build_frame_times(frames_df, max(n_frames, 1), src_fps)
    states = precompute_states(fer, pos_windows, per_win_algo, frame_ts, src_fps)
    duration = float(frame_ts[-1]) if len(frame_ts) else 1.0

    # trace arrays (full session)
    trace_t = np.array([s.t for s in states])
    trace_f12 = np.array([s.f12 for s in states])
    trace_f15 = np.array([s.f15 for s in states])

    # output fps — divide by stride so a strided render still plays back real-time
    if OUTPUT_FPS_MODE == "effective" and len(frame_ts) > 1:
        out_fps = 1.0 / float(np.median(np.diff(frame_ts)))
    else:
        out_fps = src_fps
    out_fps = max(1.0, out_fps / frame_stride)

    renderer = HudRenderer(src_w, src_h, duration, trace_t, trace_f12, trace_f15)
    writer = BaseReplayRenderer._open_writer(Path(output_path), out_fps, W, H)

    stride_note = f" stride={frame_stride}" if frame_stride > 1 else ""
    print(f"[MergedHUD] {video_path.name}: {n_frames} frames "
          f"({src_w}x{src_h} @ {src_fps:.1f} → output @ {out_fps:.1f} fps{stride_note})")
    print(f"[MergedHUD] FER rows={len(fer)}  POS windows={len(pos_windows)}  "
          f"duration={duration:.1f}s")
    if gt_csv and Path(gt_csv).exists():
        # The merged HUD design dropped the GT-HR element (see Merged Fear HUD.html);
        # GT is accepted as an input for parity with replay.py but is not drawn here.
        print(f"[MergedHUD] (ground truth {Path(gt_csv).name} loaded but not drawn "
              f"in this layout)")

    idx = 0          # decoded-frame index (video timeline)
    rendered = 0     # frames actually drawn + written
    try:
        while True:
            if not cap.grab():               # advance without decoding
                break
            if idx % frame_stride == 0:      # decode + draw only selected frames
                ret, frame_bgr = cap.retrieve()
                if not ret:
                    break
                st = states[idx] if idx < len(states) else states[-1]
                # this frame's forehead ROI (original-frame coords) for the video panel
                roi = roi_map.get(st.frame_num) or roi_map.get(idx)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = renderer.render(frame_rgb, st, roi)
                out_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)
                rendered += 1
                if rendered % 100 == 0:
                    p = 100 * idx / max(n_frames, 1)
                    print(f"  {idx}/{n_frames} ({p:.0f}%)  rendered={rendered}")
                if max_frames is not None and rendered >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
        writer.release()
    print(f"[MergedHUD] Done → {output_path}  ({rendered} frames)")


# ===========================================================================
# 10. INTERACTIVE PROMPTS  (no CLI flags — guided at run time, handoff §0.2)
# ===========================================================================

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def _list_sessions() -> List:
    repo = SessionRepository()
    return repo.list()


def _print_sessions(folders) -> None:
    print(f"\n{'#':>2}  {'ID':40s}  {'Video':5s}  {'Analysis':8s}  {'Frames':6s}  {'GT':3s}")
    print("-" * 78)
    for i, f in enumerate(folders):
        print(f"{i:>2}  {f.id:40s}  "
              f"{'yes' if f.raw_video else 'no':5s}  "
              f"{'yes' if f.analysis_csv else 'no':8s}  "
              f"{'yes' if f.frames_csv else 'no':6s}  "
              f"{'yes' if f.gt_csv else 'no':3s}")


def _find_fer_csvs() -> List[Path]:
    logs = _HERE / "logs"
    found = sorted(logs.rglob("*_mp_hs_temp.csv"))
    return found


def _maybe_regenerate_analysis(folder) -> Optional[Path]:
    """If a session has frames.csv but no analysis.csv, offer to generate it
    (POS @ 30s / step) with RppgAnalyzer — handoff §2 optional convenience."""
    if folder.analysis_csv:
        return folder.analysis_csv
    if not folder.frames_csv:
        return None
    ans = _ask("No analysis.csv. Generate one now with POS@30s? (y/N)", "N")
    if ans.lower() not in ("y", "yes"):
        return None
    cfg = AnalysisConfig(algorithm="pos", window_s=float(POS_WINDOW_S),
                         step_s=float(POS_STEP_S), bpm_min=60, bpm_max=180)
    out = folder.path / "analysis.csv"
    results = RppgAnalyzer().analyze(Path(folder.frames_csv), cfg)
    if not results:
        print("  Analysis produced no windows — aborting.")
        return None
    RppgAnalyzer().save(results, out)
    return out


def interactive_main() -> None:
    print("=" * 72)
    print("  Merged FER + rPPG Fear-Detection HUD — offline mp4 renderer")
    print("  (re-renders an existing session; no camera is opened)")
    print("=" * 72)

    folders = _list_sessions()
    use_session = False
    sel = None
    if folders:
        _print_sessions(folders)
        choice = _ask("\nPick a session # (or blank to enter paths manually)")
        if choice.isdigit() and int(choice) < len(folders):
            sel = folders[int(choice)]
            use_session = True

    if use_session:
        video_path = Path(sel.raw_video) if sel.raw_video else None
        analysis_csv = _maybe_regenerate_analysis(sel)
        frames_csv = Path(sel.frames_csv) if sel.frames_csv else None
        gt_csv = Path(sel.gt_csv) if sel.gt_csv else None
        default_out = str(sel.path / "merged_hud.mp4")
        if not video_path or not video_path.exists():
            print("  Session has no raw_video.mp4 — falling back to manual paths.")
            use_session = False

    if not use_session:
        video_path = Path(_ask("Raw video path (raw_video.mp4)"))
        analysis_csv = Path(_ask("rPPG analysis.csv path"))
        frames_in = _ask("frames.csv path (optional, for ROI box)")
        frames_csv = Path(frames_in) if frames_in else None
        gt_in = _ask("Ground-truth gt_aligned.csv (optional, not drawn)")
        gt_csv = Path(gt_in) if gt_in else None
        default_out = str(video_path.parent / "merged_hud.mp4")

    # FER CSV picker
    fer_csvs = _find_fer_csvs()
    fer_csv = None
    if fer_csvs:
        print("\nAvailable FER full CSVs (Pipeline/logs/**/*_mp_hs_temp.csv):")
        for i, f in enumerate(fer_csvs):
            print(f"  {i:>2}  {f.relative_to(_HERE)}")
        choice = _ask("Pick FER CSV # (or blank to type a path)")
        if choice.isdigit() and int(choice) < len(fer_csvs):
            fer_csv = fer_csvs[int(choice)]
    if fer_csv is None:
        fer_csv = Path(_ask("FER full CSV path (<session>_mp_hs_temp.csv)"))

    output_path = Path(_ask("Output mp4 path", default_out))

    # validate
    for label, pth in [("video", video_path), ("analysis.csv", analysis_csv),
                       ("FER csv", fer_csv)]:
        if not pth or not Path(pth).exists():
            print(f"\n[error] required input missing: {label} → {pth}")
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_session(video_path, fer_csv, analysis_csv, frames_csv, output_path, gt_csv)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        folders = _list_sessions()
        if not folders:
            print("No sessions found under Pipeline/sessions/.")
        else:
            _print_sessions(folders)
        return
    interactive_main()


if __name__ == "__main__":
    main()
