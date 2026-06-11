"""
La Facade Fissuree — Benchmark Explorer
========================================
Pattern: [Explorer / Script] — ad-hoc live tool for side-by-side comparison of ALL
detection methods in one frame loop; not imported by other modules.

Side-by-side comparison of ALL detection methods in a single live view.

LEFT PANEL  — MP + HS raw signals (same as test_fusion.py)
RIGHT PANEL — All 7 FER formulas (F0-F6) + all rPPG methods (CHROM/POS/GREEN + Docker)

rPPG algorithms (CHROM/POS/GREEN) run live in the same frame loop.
Docker rPPG results (pyVHR, rppg-toolbox) are pre-loaded from CSV and overlaid by timestamp.

Usage:
    # Live comparison, no Docker overlay:
    python benchmark_explorer.py --video path/to/video.mp4

    # With Docker rPPG results overlaid:
    python benchmark_explorer.py --video path/to/video.mp4 \\
        --pyvhr-csv logs/pyvhr_bpm.csv \\
        --toolbox-csv logs/toolbox_bpm.csv

    # Headless (saves CSV only):
    python benchmark_explorer.py --video path/to/video.mp4 --no-display --no-session

Docker CSV format (both --pyvhr-csv and --toolbox-csv):
    Must have columns: timestamp_s, bpm
    timestamp_s = window centre time in seconds from video start

Press 'q' to quit. Results saved to logs/ folder.
"""

# ── GPU CONFIGURATION — set before onnxruntime loads ──────────────────────────
USE_GPU = False
GPU_DEVICE_INDEX = 0
# ──────────────────────────────────────────────────────────────────────────────

import os
import onnxruntime as ort

_cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
if USE_GPU and _cuda_available:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_INDEX)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import argparse
import sys
import time
import urllib.request
import numpy as np
import cv2
import mediapipe as mp_lib
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from session_meta import Session
from post_session_analysis import post_session_analysis
from improved_fear_detection import get_velocity_tag
from rppg_algorithms import (compute_bpm_timeseries,
                              interpolate_motion_frames)

# ── Model path (MediaPipe) ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ── Blendshape groups ──────────────────────────────────────────────────────
VELOCITY_AUS = [
    "eyeWideLeft", "eyeWideRight", "browInnerUp",
    "jawOpen", "mouthPressLeft", "mouthPressRight",
]
STARTLE_VELOCITY_THRESHOLD = 3.0

KEY_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
    "mouthPressLeft", "mouthPressRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthSmileLeft", "mouthSmileRight", "jawOpen",
    "noseSneerLeft", "noseSneerRight",
    "cheekSquintLeft", "cheekSquintRight",
]

EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]

# ── rPPG forehead ROI ───────────────────────────────────────────────────────
FOREHEAD_Y_START = 0.15
FOREHEAD_Y_END   = 0.55
FOREHEAD_X_START = 0.20
FOREHEAD_X_END   = 0.80

EXPRESSION_ACTIVITY_THRESHOLD = 0.08

# ── rPPG window variants ────────────────────────────────────────────────────
WINDOW_SHORT  = 10.0   # shorter window — more temporal sensitivity, ~6 BPM resolution
STEP_SHORT    = 2.0

# ── Neutral-window gating ───────────────────────────────────────────────────
# A frame is considered "not neutral" (and gated out) when:
#   HS dominant emotion is not Neutral, OR MP tension exceeds this threshold.
NEUTRAL_TENSION_MAX = 0.15
EXPRESSION_AUS = [
    "browInnerUp", "browDownLeft", "browDownRight",
    "eyeWideLeft", "eyeWideRight",
    "jawOpen",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "cheekPuff",
]

# ── HUD layout ──────────────────────────────────────────────────────────────
LEFT_PANEL_W  = 400   # MP + HS raw signals
RIGHT_PANEL_W = 480   # Formulas + rPPG comparison
PANEL_BG      = (30, 30, 45)
DIVIDER_COLOR = (80, 80, 100)

# ── Formula colours ────────────────────────────────────────────────────────
# BGR
F_COLORS = {
    "F0": (0, 60, 220),    # red — broken
    "F1": (0, 200, 80),    # green
    "F2": (0, 255, 120),   # bright green — recommended
    "F3": (0, 200, 200),   # yellow
    "F4": (100, 100, 100), # grey — binary veto
    "F5": (180, 100, 0),   # teal
    "F6": (200, 200, 0),   # cyan
}

# ── rPPG algorithm colours ─────────────────────────────────────────────────
RPPG_COLORS = {
    "CHROM":     (0, 255, 200),
    "POS":       (255, 200, 0),
    "GREEN":     (0, 200, 50),
    "ICA":       (200, 150, 255),
    "WAVELET":   (255, 100, 200),
    "MULTI_ROI": (100, 220, 255),
    "pyVHR":     (200, 100, 255),
    "Toolbox":   (255, 150, 100),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return
    print(f"Downloading FaceLandmarker model to {MODEL_PATH} ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def load_docker_bpm_csv(path):
    """Load a Docker rPPG BPM CSV.

    Expected columns: timestamp_s, bpm
    Returns list of (timestamp_s: float, bpm: float) sorted ascending.
    Returns [] if path is None or file not found.
    """
    if path is None or not os.path.isfile(path):
        if path is not None:
            print(f"[WARN] Docker CSV not found: {path}")
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["timestamp_s"])
                b = float(row["bpm"])
                rows.append((t, b))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda x: x[0])
    print(f"  Loaded {len(rows)} BPM windows from {os.path.basename(path)}")
    return rows


def lookup_docker_bpm(bpm_series, timestamp_s, max_gap=15.0):
    """Return BPM from a pre-loaded series at the nearest timestamp.

    Returns None if the series is empty or the nearest window is further
    than max_gap seconds away.
    """
    if not bpm_series:
        return None
    best = min(bpm_series, key=lambda x: abs(x[0] - timestamp_s))
    if abs(best[0] - timestamp_s) > max_gap:
        return None
    return best[1]


# ══════════════════════════════════════════════════════════════════════════════
# FER SIGNAL FUNCTIONS  (from test_fusion.py)
# ══════════════════════════════════════════════════════════════════════════════

def compute_tension(blendshapes_dict):
    get = blendshapes_dict.get
    smile_level = max(get("mouthSmileLeft", 0.0), get("mouthSmileRight", 0.0))
    cheek_squint = max(get("cheekSquintLeft", 0.0), get("cheekSquintRight", 0.0))
    positive_signal = max(smile_level, cheek_squint * 0.8)
    frown_level = max(get("mouthFrownLeft", 0.0), get("mouthFrownRight", 0.0))
    press_level = max(get("mouthPressLeft", 0.0), get("mouthPressRight", 0.0))
    is_sad = frown_level > press_level and frown_level > 0.1
    brow_down = max(get("browDownLeft", 0.0), get("browDownRight", 0.0))
    is_concentrating = brow_down > 0.2 and press_level < 0.05

    pure_stress = {
        "mouthPressLeft": 1.5, "mouthPressRight": 1.5,
        "noseSneerLeft": 0.8, "noseSneerRight": 0.8,
        "mouthUpperUpLeft": 0.5, "mouthUpperUpRight": 0.5,
        "cheekPuff": 0.3,
    }
    stress_total = sum(get(n, 0.0) * w for n, w in pure_stress.items())
    stress_wsum = sum(pure_stress.values())

    ctx_multiplier = 0.2 if is_concentrating else (0.5 if is_sad else 1.0)
    brow_squint = {
        "browDownLeft": 0.7, "browDownRight": 0.7,
        "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
        "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4,
    }
    ctx_total = sum(get(n, 0.0) * w * ctx_multiplier for n, w in brow_squint.items())
    ctx_wsum = sum(brow_squint.values())

    smile_discount = max(0.0, 1.0 - positive_signal * 2.0)
    is_laughing = (get("jawOpen", 0.0) > 0.3 and get("browInnerUp", 0.0) > 0.5)
    brow_weight = 0.3 if is_laughing else 1.3
    fear_signals = {
        "browInnerUp": brow_weight, "eyeWideLeft": 1.5, "eyeWideRight": 1.5, "jawOpen": 0.6,
    }
    fear_total = sum(get(n, 0.0) * w * smile_discount for n, w in fear_signals.items())
    fear_wsum = sum(fear_signals.values())

    raw = (stress_total + ctx_total + fear_total) / (stress_wsum + ctx_wsum + fear_wsum)
    raw -= positive_signal * 0.15
    return max(0.0, min(1.0, raw * 2.5))


def compute_au_velocities(bs_dict, prev_bs_dict, delta_t):
    if prev_bs_dict is None or delta_t is None:
        return {au: 0.0 for au in VELOCITY_AUS}, 0.0
    dt = max(delta_t, 1.0 / 120.0)
    velocities = {au: max(0.0, (bs_dict.get(au, 0.0) - prev_bs_dict.get(au, 0.0)) / dt)
                  for au in VELOCITY_AUS}
    return velocities, max(velocities.values()) if velocities else 0.0


def compute_ctx_tag(bs_dict, smile_level):
    frown = max(bs_dict.get("mouthFrownLeft", 0), bs_dict.get("mouthFrownRight", 0))
    press = max(bs_dict.get("mouthPressLeft", 0), bs_dict.get("mouthPressRight", 0))
    brow_dn = max(bs_dict.get("browDownLeft", 0), bs_dict.get("browDownRight", 0))
    eye_wd  = max(bs_dict.get("eyeWideLeft", 0), bs_dict.get("eyeWideRight", 0))
    if smile_level > 0.3:
        return "JOY"
    elif eye_wd > 0.3 and bs_dict.get("browInnerUp", 0) > 0.2:
        return "FEAR"
    elif brow_dn > 0.2 and press < 0.05:
        return "CONC"
    elif frown > press and frown > 0.1:
        return "SAD"
    elif press > 0.15:
        return "STRESS"
    return "---"


def compute_expression_activity(bs_dict):
    total = sum(bs_dict.get(au, 0.0) for au in EXPRESSION_AUS)
    return total / len(EXPRESSION_AUS)


def compute_composite_fear(hs_arousal, mp_tension):
    """F0: 60% HS arousal + 40% MP tension (mirrors test_fusion.py)."""
    return min(1.0, max(0.0, 0.60 * hs_arousal + 0.40 * mp_tension))


def compute_agreement(mp_ctx_tag, _mp_tension, hs_dominant, hs_arousal, hs_emotions):
    """Agreement and veto tags between MP and HS readings (mirrors test_fusion.py)."""
    hs_fear      = hs_emotions.get("Fear", 0)
    hs_surprise  = hs_emotions.get("Surprise", 0)
    hs_anger     = hs_emotions.get("Anger", 0)
    hs_contempt  = hs_emotions.get("Contempt", 0)
    hs_happiness = hs_emotions.get("Happiness", 0)
    hs_sadness   = hs_emotions.get("Sadness", 0)

    if mp_ctx_tag == "JOY":
        if hs_dominant == "Happiness" or hs_happiness > 0.4:
            return "AGREE_JOY", "---"
        elif hs_dominant in ("Fear", "Anger") and hs_arousal > 0.5:
            return "VETO", f"MP:JOY/HS:{hs_dominant}"
    elif mp_ctx_tag == "FEAR":
        if hs_fear > 0.2 or hs_surprise > 0.3 or hs_arousal > 0.5:
            return "AGREE_FEAR", "---"
        elif hs_dominant == "Happiness" and hs_arousal < 0.3:
            return "VETO", "MP:FEAR/HS:Happy"
    elif mp_ctx_tag == "STRESS":
        if hs_anger > 0.2 or hs_contempt > 0.2 or hs_arousal > 0.4:
            return "AGREE_STRESS", "---"
        elif hs_dominant == "Happiness" and hs_arousal < 0.25:
            return "VETO", "MP:STRESS/HS:Happy"
    elif mp_ctx_tag == "SAD":
        if hs_sadness > 0.3:
            return "AGREE_SAD", "---"
        elif hs_dominant == "Happiness":
            return "VETO", "MP:SAD/HS:Happy"
    return "AMBIGUOUS", "---"


def extract_forehead_rgb(frame, face_bbox):
    x1, y1, x2, y2 = face_bbox
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return None
    roi = frame[
        max(0, y1 + int(FOREHEAD_Y_START * h)) : min(frame.shape[0], y1 + int(FOREHEAD_Y_END * h)),
        max(0, x1 + int(FOREHEAD_X_START * w)) : min(frame.shape[1], x1 + int(FOREHEAD_X_END * w)),
    ]
    if roi.size == 0:
        return None
    m = roi.mean(axis=(0, 1))
    return (m[2], m[1], m[0])  # BGR → RGB


def extract_cheek_rgbs(frame, face_bbox):
    """Extract left-cheek and right-cheek mean RGB for multi-ROI rPPG.

    ROI layout (fractions of face bbox):
      Left cheek:  y 45-75%, x 10-38%
      Right cheek: y 45-75%, x 62-90%

    Returns (left_rgb, right_rgb) tuples (R, G, B) or (None, None).
    """
    x1, y1, x2, y2 = face_bbox
    w, h = x2 - x1, y2 - y1
    if w < 20 or h < 20:
        return None, None

    def _roi_mean(ys, ye, xs, xe):
        r = frame[
            max(0, y1 + int(ys * h)):min(frame.shape[0], y1 + int(ye * h)),
            max(0, x1 + int(xs * w)):min(frame.shape[1], x1 + int(xe * w)),
        ]
        if r.size == 0:
            return None
        m = r.mean(axis=(0, 1))
        return (m[2], m[1], m[0])  # BGR → RGB

    return _roi_mean(0.45, 0.75, 0.10, 0.38), _roi_mean(0.45, 0.75, 0.62, 0.90)


# ══════════════════════════════════════════════════════════════════════════════
# FORMULA DEFINITIONS  (from formula_benchmark.py)
# ══════════════════════════════════════════════════════════════════════════════

def _clamp(v):
    return min(1.0, max(0.0, v))


def compute_all_formulas(hs_fear, hs_surprise, hs_arousal, hs_anger,
                         mp_tension, mp_startle_score):
    """Compute all 7 FER formulas from raw per-frame signals.

    Returns dict: formula_id → score (float in [0, 1]).
    """
    startle_norm = _clamp(mp_startle_score / 10.0)
    f0 = _clamp(0.60 * hs_arousal + 0.40 * mp_tension)
    f1 = _clamp(0.60 * hs_fear   + 0.40 * mp_tension)
    f2 = _clamp(0.40 * hs_fear   + 0.20 * hs_surprise + 0.40 * mp_tension)
    f3 = _clamp(0.50 * hs_fear   + 0.20 * hs_arousal  + 0.30 * mp_tension)
    f4 = _clamp(f0 * (1.0 if hs_anger <= hs_fear else 0.0))
    f5 = _clamp(0.50 * hs_fear   + 0.30 * mp_tension  + 0.20 * startle_norm)
    f6 = _clamp((0.60 * hs_fear  + 0.40 * mp_tension) * max(0.0, 1.0 - hs_anger))
    return {"F0": f0, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6}


# ══════════════════════════════════════════════════════════════════════════════
# HUD DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_left_panel(canvas, px_start, frame_h, mp_data, hs_data):
    """Left panel: raw MP + HS signals (same layout as test_fusion.py)."""
    pw = LEFT_PANEL_W
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas[0:frame_h, px_start:px_start + pw] = PANEL_BG

    y = 10
    px = px_start + 10
    bar_x = px_start + 165
    bar_max = 185

    # ── MP TRIGGERS ────────────────────────────────────────────────────────
    cv2.putText(canvas, "MP TRIGGERS", (px, y + 14), font, 0.55, (100, 200, 255), 2)
    y += 22
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 8

    if mp_data["face_detected"]:
        tension = mp_data["tension"]
        t_bar = int(min(tension, 1.0) * bar_max)
        cv2.putText(canvas, f"Tension: {tension:.2f}", (px, y + 12), font, 0.45, (0, 0, 255), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + t_bar, y + 14), (0, 0, 255), -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 20
        val = mp_data["face_valence"]
        v_color = (0, 200, 0) if val >= 0 else (0, 0, 255)
        v_bar = int(abs(val) * bar_max)
        cv2.putText(canvas, f"Valence: {val:+.2f}", (px, y + 12), font, 0.45, v_color, 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + v_bar, y + 14), v_color, -1)
        y += 20
        ctx = mp_data["ctx_tag"]
        ctx_colors = {"JOY": (0, 200, 0), "FEAR": (0, 0, 255), "CONC": (200, 200, 0),
                      "SAD": (200, 100, 0), "STRESS": (0, 50, 255), "---": (150, 150, 150)}
        cv2.putText(canvas, f"State: [{ctx}]", (px, y + 12), font, 0.50,
                    ctx_colors.get(ctx, (150, 150, 150)), 2)
        y += 20
        vel_tag = mp_data["velocity_tag"]
        ss = mp_data["startle_score"]
        vel_col = (0, 100, 255) if vel_tag == "STARTLE" else (100, 100, 100)
        cv2.putText(canvas, f"Startle: [{vel_tag}] {ss:.1f}/s", (px, y + 12), font, 0.40,
                    vel_col, 1)
        y += 18
    else:
        cv2.putText(canvas, "No face detected", (px, y + 12), font, 0.45, (100, 100, 100), 1)
        y += 20
    y += 5

    # ── HS TRIGGERS ────────────────────────────────────────────────────────
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 8
    crop_label = {"haar": "[H]", "mp": "[M]", "none": "[-]"}.get(
        hs_data.get("crop_source", "none"), "[-]")
    cv2.putText(canvas, f"HS TRIGGERS {crop_label}", (px, y + 14), font, 0.55, (0, 200, 255), 2)
    y += 22
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 8

    if hs_data["face_detected"]:
        arousal = hs_data["arousal"]
        a_bar = int(min(max(arousal, 0), 1.0) * bar_max)
        cv2.putText(canvas, f"Arousal: {arousal:.2f}", (px, y + 12), font, 0.45, (0, 0, 255), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + a_bar, y + 14), (0, 0, 255), -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 20
        dom = hs_data["dominant"]
        dom_score = hs_data["dominant_score"]
        cv2.putText(canvas, f"Dom: {dom} ({dom_score:.2f})", (px, y + 12), font, 0.45,
                    (255, 255, 255), 1)
        y += 20
        for emo, col in [("Fear", (0, 0, 255)), ("Surprise", (0, 180, 255)),
                          ("Anger", (0, 80, 220))]:
            score = hs_data["emotions"].get(emo, 0.0)
            e_bar = int(min(score, 1.0) * bar_max)
            cv2.putText(canvas, f"  {emo}: {score:.2f}", (px, y + 12), font, 0.38, col, 1)
            cv2.rectangle(canvas, (bar_x, y + 3), (bar_x + e_bar, y + 12), col, -1)
            cv2.rectangle(canvas, (bar_x, y + 3), (bar_x + bar_max, y + 12), (60, 60, 80), 1)
            y += 16
    else:
        cv2.putText(canvas, "No face detected", (px, y + 12), font, 0.45, (100, 100, 100), 1)
        y += 20
    y += 5

    # ── TIMESTAMP ──────────────────────────────────────────────────────────
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 8
    ts = mp_data.get("timestamp", 0.0)
    cv2.putText(canvas, f"t = {ts:.1f}s  frame {mp_data.get('frame', 0)}",
                (px, y + 12), font, 0.38, (140, 140, 140), 1)


def draw_right_panel(canvas, px_start, frame_h, formulas, rppg_live, rppg_docker, perf=None):
    """Right panel: all 7 formulas + all rPPG methods side by side."""
    pw = RIGHT_PANEL_W
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas[0:frame_h, px_start:px_start + pw] = PANEL_BG

    y = 10
    px = px_start + 10
    label_w = 155  # space for label text
    bar_x = px_start + label_w
    bar_max = pw - label_w - 20

    # ── FORMULA SCORES ─────────────────────────────────────────────────────
    cv2.putText(canvas, "FER FORMULAS", (px, y + 14), font, 0.55, (255, 100, 255), 2)
    y += 22
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 8

    labels = {
        "F0": "F0 AROUSAL  ",
        "F1": "F1 FEAR+TEN ",
        "F2": "F2 FEAR+SURP",
        "F3": "F3 HEDGED   ",
        "F4": "F4 HARDVETO ",
        "F5": "F5 STARTLE  ",
        "F6": "F6 SOFTVETO ",
    }
    notes = {
        "F0": "BROKEN",
        "F2": "RECOM.",
    }

    for fid, label in labels.items():
        score = formulas.get(fid, 0.0)
        col = F_COLORS.get(fid, (180, 180, 180))
        bar_w = int(min(score, 1.0) * bar_max)

        note = notes.get(fid, "")
        note_suffix = f" [{note}]" if note else ""
        cv2.putText(canvas, f"{label} {score:.2f}{note_suffix}",
                    (px, y + 12), font, 0.36, col, 1)
        cv2.rectangle(canvas, (bar_x, y + 3), (bar_x + bar_w, y + 13), col, -1)
        cv2.rectangle(canvas, (bar_x, y + 3), (bar_x + bar_max, y + 13), (60, 60, 80), 1)
        y += 18

    y += 8

    # ── rPPG COMPARISON ────────────────────────────────────────────────────
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 8
    cv2.putText(canvas, "rPPG METHODS", (px, y + 14), font, 0.55, (0, 255, 200), 2)
    y += 22
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 6

    bpm_max_display = 180.0
    n_col = rppg_live["n_collected"]
    n_neu = rppg_live.get("n_neutral", 0)

    def _bpm_row(label, bpm_val, col):
        nonlocal y
        if bpm_val is not None and bpm_val > 0:
            bpm_bar = int(min(bpm_val / bpm_max_display, 1.0) * bar_max)
            draw_col = col if 42 <= bpm_val <= 180 else (0, 0, 200)
            cv2.putText(canvas, f"{label} {bpm_val:.0f}",
                        (px, y + 11), font, 0.38, col, 1)
            cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bpm_bar, y + 12), draw_col, -1)
            cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 12), (60, 60, 80), 1)
        else:
            cv2.putText(canvas, f"{label} --",
                        (px, y + 11), font, 0.38, (80, 80, 80), 1)
        y += 15

    def _group_header(title, detail=""):
        nonlocal y
        cv2.putText(canvas, title, (px, y + 11), font, 0.38, (160, 160, 160), 1)
        if detail:
            cv2.putText(canvas, detail, (px + 130, y + 11), font, 0.33, (100, 100, 100), 1)
        y += 14

    # 30s window
    _group_header("30s window", f"({n_col}f)")
    for algo in ["CHROM", "POS", "GREEN", "ICA", "WAVELET"]:
        _bpm_row(f"  {algo:<8}", rppg_live["30s"].get(algo), RPPG_COLORS.get(algo, (180, 180, 180)))
    vals30 = [v for v in rppg_live["30s"].values() if v is not None and v > 0]
    if vals30:
        cons = float(np.median(vals30))
        _bpm_row("  CONSENS", cons, (255, 255, 255))
    y += 3

    # 10s window
    _group_header("10s window", "(~6 BPM res)")
    for algo in ["CHROM", "POS", "GREEN", "ICA", "WAVELET"]:
        _bpm_row(f"  {algo:<8}", rppg_live["10s"].get(algo), RPPG_COLORS.get(algo, (180, 180, 180)))
    y += 3

    # neutral-gated (Concern I — not definitive, face expression ≠ neutral during fear)
    _group_header("neutral-gated 30s", f"({n_neu}f neutral)")
    for algo in ["CHROM", "POS", "GREEN"]:
        _bpm_row(f"  {algo:<8}", rppg_live["gated"].get(algo), RPPG_COLORS[algo])
    y += 3

    # Multi-ROI (forehead + left cheek + right cheek, median-of-plausible)
    n_cl = rppg_live.get("n_cheekl", 0)
    n_cr = rppg_live.get("n_cheekr", 0)
    _group_header("multi-ROI 30s", f"(fh+{n_cl}cl+{n_cr}cr)")
    _bpm_row("  CHROM×3", rppg_live.get("multi_roi"), RPPG_COLORS["MULTI_ROI"])
    y += 3

    # Docker
    cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
    y += 5
    for label, key, ckey in [("pyVHR [D]", "pyvhr", "pyVHR"),
                               ("Toolbox[D]", "toolbox", "Toolbox")]:
        _bpm_row(f"  {label}", rppg_docker.get(key), RPPG_COLORS.get(ckey, (180, 180, 180)))

    # ── PERFORMANCE ────────────────────────────────────────────────────────
    if perf:
        y += 6
        cv2.line(canvas, (px_start + 5, y), (px_start + pw - 5, y), DIVIDER_COLOR, 1)
        y += 8
        cv2.putText(canvas, "PERFORMANCE", (px, y + 14), font, 0.55, (200, 200, 80), 2)
        y += 22

        fps   = perf.get("fps", 0.0)
        total = perf.get("total_ms", 0.0)
        mp_t  = perf.get("mp_ms", 0.0)
        hs_t  = perf.get("hs_ms", 0.0)

        fps_col   = (0, 220, 0) if fps >= 20 else (0, 140, 255) if fps >= 10 else (0, 0, 220)
        total_col = (0, 220, 0) if total < 50 else (0, 140, 255) if total < 100 else (0, 0, 220)

        cv2.putText(canvas, f"FPS (live):  {fps:5.1f}", (px, y + 12), font, 0.40, fps_col, 1)
        y += 16
        cv2.putText(canvas, f"Frame total: {total:5.1f} ms", (px, y + 12), font, 0.40, total_col, 1)
        y += 16
        cv2.putText(canvas, f"MediaPipe:   {mp_t:5.1f} ms", (px, y + 12), font, 0.40, (160, 200, 255), 1)
        y += 16
        cv2.putText(canvas, f"HSEmotion:   {hs_t:5.1f} ms", (px, y + 12), font, 0.40, (255, 200, 160), 1)
        y += 16
        rppg_t = perf.get("rppg_ms", 0.0)
        rppg_label = f"rPPG win:    {rppg_t:5.1f} ms" if rppg_t > 0 else "rPPG win:    -- ms"
        cv2.putText(canvas, rppg_label, (px, y + 12), font, 0.40, (160, 255, 200), 1)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark Explorer — all methods side by side")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. Omit for webcam.")
    parser.add_argument("--window", type=float, default=30.0,
                        help="rPPG BPM estimation window in seconds (default: 30)")
    parser.add_argument("--step", type=float, default=5.0,
                        help="rPPG window step in seconds (default: 5)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Max seconds to process (0 = full video)")
    parser.add_argument("--ref-video", type=str, default=None,
                        help="Reference video to display in sync below the analysed frame "
                             "(no analysis — visual comparison only)")
    parser.add_argument("--pyvhr-csv", type=str, default=None,
                        help="Docker pyVHR BPM CSV (columns: timestamp_s, bpm)")
    parser.add_argument("--toolbox-csv", type=str, default=None,
                        help="Docker rppg-toolbox BPM CSV (columns: timestamp_s, bpm)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (saves CSV only)")
    parser.add_argument("--no-session", action="store_true",
                        help="Skip session_meta interactive prompts")
    args = parser.parse_args()

    ensure_model()

    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ── Session ────────────────────────────────────────────────────────────
    session = Session("explorer", log_dir)
    if not args.no_session:
        session.pre_session_prompt()
    else:
        session.subject_id = "auto"
        session.session_label = "benchmark_explorer"

    # Build session subfolder + canonical filenames (subject + label in name)
    label_slug = session.session_label.replace(" ", "_") if session.session_label else "nosession"
    session_folder = f"{session.subject_id}_{label_slug}"
    session_dir = os.path.join(log_dir, "sessions", session_folder)
    os.makedirs(session_dir, exist_ok=True)
    stem = f"{session.session_id}_explorer_{session.subject_id}_{label_slug}"
    csv_path      = os.path.join(session_dir, stem + ".csv")
    rppg_csv_path = os.path.join(session_dir, stem + "_rppg.csv")

    session.device_used = "cpu"
    session.video_source = args.video if args.video else "webcam"

    # ── Load Docker BPM CSVs ───────────────────────────────────────────────
    print("\nLoading Docker rPPG results (if provided)...")
    pyvhr_series   = load_docker_bpm_csv(args.pyvhr_csv)
    toolbox_series = load_docker_bpm_csv(args.toolbox_csv)

    # ── MediaPipe ──────────────────────────────────────────────────────────
    print("Loading MediaPipe FaceLandmarker...")
    BaseOptions = mp_lib.tasks.BaseOptions
    FaceLandmarker = mp_lib.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_lib.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp_lib.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    print("MediaPipe loaded.")

    # ── HSEmotion ─────────────────────────────────────────────────────────
    print("Loading HSEmotion...")
    recognizer = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print("HSEmotion loaded.")

    # ── Video ──────────────────────────────────────────────────────────────
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"ERROR: Could not open {'file: ' + args.video if args.video else 'webcam'}.")
        return

    # Max width for the video portion of the canvas so panels stay on-screen.
    # Total canvas = VIDEO_MAX_W + LEFT_PANEL_W(400) + RIGHT_PANEL_W(480) = 1680px.
    VIDEO_MAX_W = 800

    if args.video:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(args.duration * video_fps) if args.duration > 0 else total_frames
        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_scale = min(1.0, VIDEO_MAX_W / native_w) if native_w > 0 else 1.0
        if video_scale < 1.0:
            scaled_w = int(native_w * video_scale)
            scaled_h = int(native_h * video_scale)
            print(f"Video: {args.video}  |  FPS: {video_fps:.1f}  |  Frames: {total_frames}")
            print(f"  Native: {native_w}x{native_h}  →  Display: {scaled_w}x{scaled_h} "
                  f"(scale={video_scale:.2f}, total canvas ~{scaled_w + LEFT_PANEL_W + RIGHT_PANEL_W}px wide)")
        else:
            print(f"Video: {args.video}  |  FPS: {video_fps:.1f}  |  Frames: {total_frames}")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_fps = 30.0
        total_frames = None
        max_frames = None
        video_scale = 1.0
        print("Webcam mode")

    TOTAL_PANEL_W = LEFT_PANEL_W + RIGHT_PANEL_W

    # ── Reference video (optional, display-only, frame-synced) ────────────
    cap_ref = None
    if args.ref_video:
        cap_ref = cv2.VideoCapture(args.ref_video)
        if cap_ref.isOpened():
            print(f"Reference video: {args.ref_video}")
        else:
            print(f"WARNING: Could not open reference video: {args.ref_video}")
            cap_ref = None

    # ── CSV setup ──────────────────────────────────────────────────────────
    formula_cols = ["f0", "f1", "f2", "f3", "f4", "f5", "f6"]
    rppg_cols = [
        "rppg_chrom_30s",  "rppg_pos_30s",  "rppg_green_30s",  "rppg_consensus_30s",
        "rppg_ica_30s",    "rppg_wavelet_30s", "rppg_multi_roi_30s",
        "rppg_chrom_10s",  "rppg_pos_10s",  "rppg_green_10s",
        "rppg_ica_10s",    "rppg_wavelet_10s",
        "rppg_chrom_gated","rppg_pos_gated","rppg_green_gated",
        "rppg_pyvhr_bpm",  "rppg_toolbox_bpm",
    ]
    csv_header = (
        ["frame", "timestamp",
         "hs_fear", "hs_surprise", "hs_arousal", "hs_anger", "hs_dominant", "hs_dominant_score",
         "mp_tension", "mp_startle_score", "mp_ctx_tag", "mp_velocity_tag",
         "composite_fear", "agreement_tag", "veto_tag",
         "hs_face_detected", "mp_face_detected", "is_neutral_frame"]
        + formula_cols + rppg_cols
    )
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    # ── rPPG state ─────────────────────────────────────────────────────────
    rppg_rgbs = []
    rppg_timestamps = []
    rppg_motion_flags  = []   # True = expression active (existing motion gate)
    rppg_neutral_flags = []   # True = face NOT neutral (for neutral-gating variant)
    live_bpm = {
        "30s":      {"CHROM": None, "POS": None, "GREEN": None, "ICA": None, "WAVELET": None},
        "10s":      {"CHROM": None, "POS": None, "GREEN": None, "ICA": None, "WAVELET": None},
        "gated":    {"CHROM": None, "POS": None, "GREEN": None},
        "multi_roi": None,   # SNR-weighted median across forehead + 2 cheeks
    }
    rppg_rgbs_cheekl: list = []   # left-cheek RGB buffer for multi-ROI
    rppg_rgbs_cheekr: list = []   # right-cheek RGB buffer for multi-ROI
    next_window_t       = args.window   # first 30s estimate
    next_window_short_t = WINDOW_SHORT  # first 10s estimate

    # ── Frame state ────────────────────────────────────────────────────────
    frame_count = 0
    prev_bs_dict = None
    prev_elapsed = None

    session.start()

    print(f"\nLogging to: {csv_path}")
    print("Press 'q' to quit.\n")

    # ── Performance tracking ───────────────────────────────────────────────
    _perf_wall_times: list = []   # rolling wall-clock timestamps (last 60 frames)
    dt_mp_ms    = 0.0
    dt_hs_ms    = 0.0
    dt_rppg_ms  = 0.0   # last rPPG window compute (fires every ~5s, shown as last measured)
    dt_total_ms = 0.0
    fps_live    = 0.0

    while (max_frames is None or frame_count < max_frames):
        t_loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        if video_scale < 1.0:
            frame = cv2.resize(frame, (int(frame.shape[1] * video_scale),
                                       int(frame.shape[0] * video_scale)),
                               interpolation=cv2.INTER_AREA)

        # Read one ref frame (frame-synced, no analysis)
        ref_frame = None
        if cap_ref is not None:
            ret_ref, ref_frame_raw = cap_ref.read()
            if ret_ref:
                # Scale to same width as analysed frame
                ref_h_raw, ref_w_raw = ref_frame_raw.shape[:2]
                target_w = frame.shape[1]
                ref_scale = target_w / ref_w_raw if ref_w_raw > 0 else 1.0
                ref_frame = cv2.resize(
                    ref_frame_raw,
                    (target_w, int(ref_h_raw * ref_scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                # Ref video ended — show grey placeholder
                ref_frame = np.full(
                    (frame.shape[0], frame.shape[1], 3), 40, dtype=np.uint8)
                cv2.putText(ref_frame, "Ref video ended", (10, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1)

        frame_count += 1
        elapsed = frame_count / video_fps

        # ── Defaults ───────────────────────────────────────────────────────
        mp_face_detected = False
        hs_face_detected = False
        tension = 0.0
        face_valence = 0.0
        smile_level = 0.0
        ctx_tag = "---"
        bs_dict = {}
        startle_score = 0.0
        velocity_tag = "---"
        mp_face_bbox = None

        arousal_hs = 0.0
        emotions_hs = {l: 0.0 for l in EMOTION_LABELS}
        dominant_hs = "Neutral"
        dominant_score_hs = 0.0
        hs_crop_source = "none"
        hs_face_bbox = None
        composite_fear = 0.0
        agreement_tag = ""
        veto_tag = "---"

        # ── MediaPipe ──────────────────────────────────────────────────────
        t0_mp = time.perf_counter()
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_count * 1000 / video_fps)
            mp_results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if mp_results.face_blendshapes and len(mp_results.face_blendshapes) > 0:
                mp_face_detected = True
                bs_dict = {bs.category_name: bs.score
                           for bs in mp_results.face_blendshapes[0]}
                tension = compute_tension(bs_dict)
                smile_level = max(bs_dict.get("mouthSmileLeft", 0),
                                  bs_dict.get("mouthSmileRight", 0))
                face_valence = (
                    (bs_dict.get("mouthSmileLeft", 0) + bs_dict.get("mouthSmileRight", 0)) / 2
                    - (bs_dict.get("browDownLeft", 0) + bs_dict.get("browDownRight", 0)) / 2
                )
                delta_t = (elapsed - prev_elapsed) if prev_elapsed is not None else None
                _, startle_score = compute_au_velocities(
                    bs_dict, prev_bs_dict, delta_t)
                velocity_tag = get_velocity_tag(startle_score, STARTLE_VELOCITY_THRESHOLD)
                ctx_tag = compute_ctx_tag(bs_dict, smile_level)

                if mp_results.face_landmarks and len(mp_results.face_landmarks) > 0:
                    fh_px, fw_px = frame.shape[:2]
                    lms = mp_results.face_landmarks[0]
                    xs = [lm.x * fw_px for lm in lms]
                    ys = [lm.y * fh_px for lm in lms]
                    mp_face_bbox = (int(min(xs)), int(min(ys)),
                                    int(max(xs)), int(max(ys)))
                prev_bs_dict = bs_dict.copy()
                prev_elapsed = elapsed
        except Exception as e:
            print(f"[MP error frame {frame_count}]: {e}")
        dt_mp_ms = (time.perf_counter() - t0_mp) * 1000

        # ── HSEmotion ──────────────────────────────────────────────────────
        t0_hs = time.perf_counter()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
            face_img = None
            if len(faces) > 0:
                fx, fy, fw_f, fh_f = max(faces, key=lambda f: f[2] * f[3])
                face_img = frame[fy:fy + fh_f, fx:fx + fw_f]
                hs_face_bbox = (fx, fy, fx + fw_f, fy + fh_f)
                hs_crop_source = "haar"
            elif mp_face_detected and mp_face_bbox:
                x1, y1, x2, y2 = mp_face_bbox
                pad = int(0.15 * max(x2 - x1, y2 - y1))
                x1c = max(0, x1 - pad); y1c = max(0, y1 - pad)
                x2c = min(frame.shape[1], x2 + pad)
                y2c = min(frame.shape[0], y2 + pad)
                face_img = frame[y1c:y2c, x1c:x2c]
                hs_face_bbox = (x1c, y1c, x2c, y2c)
                hs_crop_source = "mp"

            if face_img is not None and face_img.size > 0:
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)
                hs_face_detected = True
                arousal_hs = float(scores[-1])
                for i, label in enumerate(EMOTION_LABELS):
                    if i < len(scores) - 2:
                        emotions_hs[label] = float(scores[i])
                dominant_hs = emotion if emotion else max(emotions_hs, key=emotions_hs.get)
                dominant_score_hs = emotions_hs.get(dominant_hs, 0.0)
        except Exception as e:
            print(f"[HS error frame {frame_count}]: {e}")
        dt_hs_ms = (time.perf_counter() - t0_hs) * 1000

        hs_fear     = emotions_hs.get("Fear", 0.0)
        hs_surprise = emotions_hs.get("Surprise", 0.0)
        hs_anger    = emotions_hs.get("Anger", 0.0)

        # ── rPPG accumulation ──────────────────────────────────────────────
        rppg_bbox = hs_face_bbox if hs_face_bbox else mp_face_bbox
        is_neutral_frame = (dominant_hs == "Neutral") and (tension <= NEUTRAL_TENSION_MAX)
        if rppg_bbox is not None:
            rgb = extract_forehead_rgb(frame, rppg_bbox)
            if rgb is not None:
                expr_act = compute_expression_activity(bs_dict)
                rppg_rgbs.append(list(rgb))
                rppg_timestamps.append(elapsed)
                rppg_motion_flags.append(expr_act > EXPRESSION_ACTIVITY_THRESHOLD)
                rppg_neutral_flags.append(not is_neutral_frame)  # True = gate this frame
            # Multi-ROI: collect cheek signals (same motion flag as forehead)
            cl_rgb, cr_rgb = extract_cheek_rgbs(frame, rppg_bbox)
            if cl_rgb is not None:
                rppg_rgbs_cheekl.append(list(cl_rgb))
            if cr_rgb is not None:
                rppg_rgbs_cheekr.append(list(cr_rgb))

        def _bpm_last(rgb_in, ts_in, algo, win, stp):
            try:
                r = compute_bpm_timeseries(rgb_in, ts_in, fps=video_fps,
                                           algorithm=algo.lower(),
                                           window_s=win, step_s=stp)
                return r[-1]["bpm"] if r else None
            except Exception:
                return None

        # 30s window + gated update
        _rppg_frame_ms = 0.0
        if elapsed >= next_window_t and len(rppg_rgbs) >= int(args.window * video_fps * 0.5):
            t0_rppg = time.perf_counter()
            rgb_arr = np.array(rppg_rgbs)
            ts_arr  = np.array(rppg_timestamps)
            mf_arr  = np.array(rppg_motion_flags)
            nf_arr  = np.array(rppg_neutral_flags)
            rgb_clean  = interpolate_motion_frames(rgb_arr, mf_arr)
            rgb_gated  = interpolate_motion_frames(rgb_arr, mf_arr | nf_arr)
            for algo in ["CHROM", "POS", "GREEN", "ICA", "WAVELET"]:
                live_bpm["30s"][algo] = _bpm_last(rgb_clean, ts_arr, algo, args.window, args.step)
            for algo in ["CHROM", "POS", "GREEN"]:
                live_bpm["gated"][algo] = _bpm_last(rgb_gated, ts_arr, algo, args.window, args.step)
            # Multi-ROI: SNR-weighted median of CHROM across forehead + 2 cheeks
            _mr_bpms, _mr_snrs = [], []
            min_mr = int(args.window * video_fps * 0.5)
            for _roi_buf in [rppg_rgbs, rppg_rgbs_cheekl, rppg_rgbs_cheekr]:
                if len(_roi_buf) >= min_mr:
                    _roi_arr = interpolate_motion_frames(np.array(_roi_buf), mf_arr[:len(_roi_buf)])
                    _r = _bpm_last(_roi_arr, ts_arr[:len(_roi_buf)], "CHROM", args.window, args.step)
                    if _r is not None and 42 <= _r <= 180:
                        _mr_bpms.append(_r)
                        _mr_snrs.append(1.0)   # equal weight (full SNR weighting is a future step)
            live_bpm["multi_roi"] = float(np.median(_mr_bpms)) if _mr_bpms else None
            _rppg_frame_ms += (time.perf_counter() - t0_rppg) * 1000
            next_window_t += args.step

        # 10s window update
        if elapsed >= next_window_short_t and len(rppg_rgbs) >= int(WINDOW_SHORT * video_fps * 0.5):
            t0_rppg10 = time.perf_counter()
            rgb_arr   = np.array(rppg_rgbs)
            ts_arr    = np.array(rppg_timestamps)
            rgb_clean = interpolate_motion_frames(rgb_arr, np.array(rppg_motion_flags))
            for algo in ["CHROM", "POS", "GREEN", "ICA", "WAVELET"]:
                live_bpm["10s"][algo] = _bpm_last(rgb_clean, ts_arr, algo, WINDOW_SHORT, STEP_SHORT)
            _rppg_frame_ms += (time.perf_counter() - t0_rppg10) * 1000
            next_window_short_t += STEP_SHORT

        if _rppg_frame_ms > 0:
            dt_rppg_ms = _rppg_frame_ms   # keep last measured; 0ms frames don't overwrite

        # ── All 7 formulas ─────────────────────────────────────────────────
        formulas = compute_all_formulas(
            hs_fear, hs_surprise, arousal_hs, hs_anger, tension, startle_score)

        # ── Fusion tags (mirrors test_fusion.py) ───────────────────────────
        if mp_face_detected and hs_face_detected:
            composite_fear = compute_composite_fear(arousal_hs, tension)
            agreement_tag, veto_tag = compute_agreement(
                ctx_tag, tension, dominant_hs, arousal_hs, emotions_hs)

        # ── Docker BPM lookup ──────────────────────────────────────────────
        docker_pyvhr   = lookup_docker_bpm(pyvhr_series,   elapsed)
        docker_toolbox = lookup_docker_bpm(toolbox_series, elapsed)

        bpm30 = live_bpm["30s"]
        bpm10 = live_bpm["10s"]
        bpmg  = live_bpm["gated"]
        vals30 = [v for v in bpm30.values() if v is not None]
        consensus_bpm = float(np.median(vals30)) if vals30 else None

        def _f(v): return f"{v:.4f}" if v is not None else ""
        csv_writer.writerow([
            frame_count, f"{elapsed:.3f}",
            f"{hs_fear:.4f}", f"{hs_surprise:.4f}", f"{arousal_hs:.4f}", f"{hs_anger:.4f}",
            dominant_hs, f"{dominant_score_hs:.4f}",
            f"{tension:.4f}", f"{startle_score:.2f}",
            ctx_tag, velocity_tag,
            f"{composite_fear:.4f}", agreement_tag, veto_tag,
            int(hs_face_detected), int(mp_face_detected), int(is_neutral_frame),
            f"{formulas['F0']:.4f}", f"{formulas['F1']:.4f}", f"{formulas['F2']:.4f}",
            f"{formulas['F3']:.4f}", f"{formulas['F4']:.4f}", f"{formulas['F5']:.4f}",
            f"{formulas['F6']:.4f}",
            _f(bpm30.get("CHROM")), _f(bpm30.get("POS")), _f(bpm30.get("GREEN")),
            _f(consensus_bpm),
            _f(bpm30.get("ICA")), _f(bpm30.get("WAVELET")), _f(live_bpm["multi_roi"]),
            _f(bpm10.get("CHROM")), _f(bpm10.get("POS")), _f(bpm10.get("GREEN")),
            _f(bpm10.get("ICA")), _f(bpm10.get("WAVELET")),
            _f(bpmg.get("CHROM")), _f(bpmg.get("POS")), _f(bpmg.get("GREEN")),
            _f(docker_pyvhr), _f(docker_toolbox),
        ])

        # ── HUD ────────────────────────────────────────────────────────────
        if not args.no_display:
            frame_h, frame_w = frame.shape[:2]

            # If ref video is active, stack it below the analysed frame
            if ref_frame is not None:
                ref_h = ref_frame.shape[0]
                canvas_h = max(frame_h + ref_h, 720)
                canvas_w = frame_w + TOTAL_PANEL_W
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                canvas[:frame_h, :frame_w] = frame
                canvas[frame_h:frame_h + ref_h, :frame_w] = ref_frame
                # Divider line between the two videos
                canvas[frame_h:frame_h + 2, :frame_w] = (80, 80, 80)
                # Label
                cv2.putText(canvas, "REFERENCE", (6, frame_h + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            else:
                canvas_h = max(frame_h, 720)
                canvas_w = frame_w + TOTAL_PANEL_W
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                canvas[:frame_h, :frame_w] = frame

            mp_data = {
                "face_detected": mp_face_detected,
                "tension": tension, "face_valence": face_valence,
                "ctx_tag": ctx_tag, "velocity_tag": velocity_tag,
                "startle_score": startle_score,
                "timestamp": elapsed, "frame": frame_count,
            }
            hs_data = {
                "face_detected": hs_face_detected,
                "arousal": arousal_hs, "dominant": dominant_hs,
                "dominant_score": dominant_score_hs,
                "emotions": emotions_hs,
                "crop_source": hs_crop_source,
            }
            rppg_live_data = {
                "30s":        live_bpm["30s"],
                "10s":        live_bpm["10s"],
                "gated":      live_bpm["gated"],
                "multi_roi":  live_bpm["multi_roi"],
                "n_collected": len(rppg_rgbs),
                "n_neutral":  sum(1 for f in rppg_neutral_flags if not f),
                "n_cheekl":   len(rppg_rgbs_cheekl),
                "n_cheekr":   len(rppg_rgbs_cheekr),
            }
            rppg_docker_data = {
                "pyvhr":   docker_pyvhr,
                "toolbox": docker_toolbox,
            }

            dt_total_ms = (time.perf_counter() - t_loop_start) * 1000
            _perf_wall_times.append(time.perf_counter())
            if len(_perf_wall_times) > 60:
                _perf_wall_times.pop(0)
            if len(_perf_wall_times) >= 2:
                fps_live = (len(_perf_wall_times) - 1) / (_perf_wall_times[-1] - _perf_wall_times[0])

            perf_data = {
                "fps":      fps_live,
                "total_ms": dt_total_ms,
                "mp_ms":    dt_mp_ms,
                "hs_ms":    dt_hs_ms,
                "rppg_ms":  dt_rppg_ms,
            }
            draw_left_panel(canvas,  frame_w,              frame_h, mp_data, hs_data)
            draw_right_panel(canvas, frame_w + LEFT_PANEL_W, frame_h,
                             formulas, rppg_live_data, rppg_docker_data, perf_data)

            # ── Video timestamp bar (bottom of canvas) ──────────────────────
            cur_s  = int(elapsed)
            cur_mm, cur_ss = divmod(cur_s, 60)
            if total_frames and video_fps > 0:
                total_s  = int(total_frames / video_fps)
                tot_mm, tot_ss = divmod(total_s, 60)
                time_str = f"  {cur_mm:02d}:{cur_ss:02d} / {tot_mm:02d}:{tot_ss:02d}"
                bar_fill = int(frame_w * min(elapsed / (total_s or 1), 1.0))
                bar_y = canvas_h - 18
                canvas[bar_y:bar_y + 4, :frame_w] = (50, 50, 50)
                canvas[bar_y:bar_y + 4, :bar_fill] = (0, 200, 100)
            else:
                time_str = f"  {cur_mm:02d}:{cur_ss:02d}"
            cv2.putText(canvas, time_str, (4, canvas_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                        cv2.LINE_AA)

            cv2.imshow("Benchmark Explorer", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Teardown ───────────────────────────────────────────────────────────
    cap.release()
    if cap_ref is not None:
        cap_ref.release()
    csv_file.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    if session._monitor:
        session._monitor.stop()
        session._monitor.join(timeout=2)
    session.frame_count = frame_count

    print(f"\nDone. {frame_count} frames processed.")
    print(f"CSV saved: {csv_path}")

    # ── Final rPPG pass (full video) ───────────────────────────────────────
    if rppg_rgbs:
        print("\nRunning final rPPG pass (all algorithms, full video)...")
        rgb_arr   = np.array(rppg_rgbs)
        ts_arr    = np.array(rppg_timestamps)
        mf_arr    = np.array(rppg_motion_flags)
        rgb_clean = interpolate_motion_frames(rgb_arr, mf_arr)

        with open(rppg_csv_path, "w", newline="") as rf:
            rw = csv.writer(rf)
            rw.writerow(["algorithm", "window_start_s", "window_end_s",
                         "window_centre_s", "bpm", "snr", "plausible"])
            for algo_name in ["CHROM", "POS", "GREEN"]:
                try:
                    results = compute_bpm_timeseries(
                        rgb_clean, ts_arr,
                        algorithm=algo_name.lower(),
                        window_s=args.window,
                        step_s=args.step,
                        fps=video_fps,
                    )
                    for r in results:
                        plaus = 1 if 42 <= r["bpm"] <= 180 else 0
                        centre = (r["t_start"] + r["t_end"]) / 2
                        rw.writerow([algo_name, f"{r['t_start']:.1f}",
                                     f"{r['t_end']:.1f}", f"{centre:.1f}",
                                     f"{r['bpm']:.1f}", f"{r.get('snr', 0):.2f}", plaus])
                except Exception as e:
                    print(f"  [{algo_name}] failed: {e}")
        print(f"rPPG timeseries saved: {rppg_csv_path}")

    post_session_analysis(csv_path, "")

    # ── Explorer-specific plots (FER formulas + rPPG timeseries) ───────────
    viz_script = os.path.join(SCRIPT_DIR, "visualize_explorer.py")
    if os.path.isfile(viz_script):
        import subprocess
        subprocess.run(
            [sys.executable, viz_script, csv_path],
            check=False,
        )


if __name__ == "__main__":
    main()
