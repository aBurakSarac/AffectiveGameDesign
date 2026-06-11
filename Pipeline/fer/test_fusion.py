"""
La Facade Fissuree - MP+HS+rPPG Triple-Tool Fusion
====================================================
Runs MediaPipe blendshapes, HSEmotion valence-arousal, and rPPG heart rate
simultaneously on the same video frames in a single frame loop.

- MediaPipe: face landmarks + 52 blendshapes → tension, startle, face_valence
- HSEmotion: face crop (Haar/MP fallback) → emotion classification
- rPPG: forehead ROI → RGB accumulation → CHROM/POS/GREEN post-loop

Usage:
    python test_fusion.py --video path/to/video.mp4          # process video file
    python test_fusion.py                                     # test with webcam (default: 0)
    python test_fusion.py --video path/to/video.mp4 --no-session --no-display

Press 'q' to quit. Results saved to logs/ folder.
"""

# ── GPU CONFIGURATION — set before onnxruntime loads ─────────────────────────
USE_GPU = False
GPU_DEVICE_INDEX = 0
# ─────────────────────────────────────────────────────────────────────────────

import os
import onnxruntime as ort

_cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
if USE_GPU and _cuda_available:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_INDEX)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import csv
import time
import argparse
import urllib.request
import numpy as np
from datetime import datetime
import mediapipe as mp_lib
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from session_meta import Session
from post_session_analysis import post_session_analysis
from improved_fear_detection import get_velocity_tag
from rppg_algorithms import (ALGORITHMS, compute_bpm_timeseries,
                             interpolate_motion_frames)

# ── Model path (MediaPipe) ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ── Blendshape groups ─────────────────────────────────────────────────────────
NEGATIVE_BLENDSHAPES = [
    "browDownLeft", "browDownRight",
    "eyeWideLeft", "eyeWideRight",
    "eyeSquintLeft", "eyeSquintRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthPressLeft", "mouthPressRight",
    "noseSneerLeft", "noseSneerRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "cheekPuff",
]

POSITIVE_BLENDSHAPES = [
    "mouthSmileLeft", "mouthSmileRight",
    "cheekSquintLeft", "cheekSquintRight",
    "mouthDimpleLeft", "mouthDimpleRight",
]

AMBIGUOUS_BLENDSHAPES = ["browInnerUp", "jawOpen"]
STRESS_BLENDSHAPES = NEGATIVE_BLENDSHAPES + AMBIGUOUS_BLENDSHAPES + POSITIVE_BLENDSHAPES

KEY_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
    "mouthPressLeft", "mouthPressRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthSmileLeft", "mouthSmileRight", "jawOpen",
    "noseSneerLeft", "noseSneerRight",
    "cheekSquintLeft", "cheekSquintRight",
]

VELOCITY_AUS = [
    "eyeWideLeft", "eyeWideRight", "browInnerUp",
    "jawOpen", "mouthPressLeft", "mouthPressRight",
]
STARTLE_VELOCITY_THRESHOLD = 3.0

EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]

# HUD constants
PANEL_WIDTH = 400
HUD_MIN_HEIGHT = 750  # taller to fit rPPG section
PANEL_BG = (30, 30, 45)
SECTION_DIVIDER_COLOR = (80, 80, 100)

# rPPG forehead ROI ratios (relative to face bbox)
FOREHEAD_Y_START = 0.15
FOREHEAD_Y_END = 0.55
FOREHEAD_X_START = 0.20
FOREHEAD_X_END = 0.80

# rPPG motion detection: blendshapes that move forehead skin
EXPRESSION_ACTIVITY_THRESHOLD = 0.08
EXPRESSION_AUS = [
    "browInnerUp", "browDownLeft", "browDownRight",
    "eyeWideLeft", "eyeWideRight",
    "jawOpen",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "cheekPuff",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FER FUNCTIONS (from test_mp_hs.py)
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_model():
    """Download the FaceLandmarker model if not present."""
    if os.path.isfile(MODEL_PATH):
        return
    print(f"Downloading FaceLandmarker model to {MODEL_PATH} ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def compute_tension(blendshapes_dict):
    """Compute composite facial tension score (v3 formula)."""
    get = blendshapes_dict.get

    smile_level = max(get("mouthSmileLeft", 0.0), get("mouthSmileRight", 0.0))
    cheek_squint = max(get("cheekSquintLeft", 0.0), get("cheekSquintRight", 0.0))
    positive_signal = max(smile_level, cheek_squint * 0.8)

    frown_level = max(get("mouthFrownLeft", 0.0), get("mouthFrownRight", 0.0))
    press_level = max(get("mouthPressLeft", 0.0), get("mouthPressRight", 0.0))
    is_sad = frown_level > press_level and frown_level > 0.1

    brow_down = max(get("browDownLeft", 0.0), get("browDownRight", 0.0))
    is_concentrating = brow_down > 0.2 and press_level < 0.05

    stress_total = 0.0
    stress_wsum = 0.0
    pure_stress = {
        "mouthPressLeft": 1.5, "mouthPressRight": 1.5,
        "noseSneerLeft": 0.8, "noseSneerRight": 0.8,
        "mouthUpperUpLeft": 0.5, "mouthUpperUpRight": 0.5,
        "cheekPuff": 0.3,
    }
    for name, w in pure_stress.items():
        stress_total += get(name, 0.0) * w
        stress_wsum += w

    ctx_total = 0.0
    ctx_wsum = 0.0
    brow_squint = {
        "browDownLeft": 0.7, "browDownRight": 0.7,
        "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
        "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4,
    }
    if is_concentrating:
        ctx_multiplier = 0.2
    elif is_sad:
        ctx_multiplier = 0.5
    else:
        ctx_multiplier = 1.0
    for name, w in brow_squint.items():
        ctx_total += get(name, 0.0) * w * ctx_multiplier
        ctx_wsum += w

    smile_discount = max(0.0, 1.0 - positive_signal * 2.0)
    is_laughing = (get("jawOpen", 0.0) > 0.3 and get("browInnerUp", 0.0) > 0.5)
    brow_inner_up_weight = 0.3 if is_laughing else 1.3

    fear_total = 0.0
    fear_wsum = 0.0
    fear_signals = {
        "browInnerUp": brow_inner_up_weight,
        "eyeWideLeft": 1.5, "eyeWideRight": 1.5,
        "jawOpen": 0.6,
    }
    for name, w in fear_signals.items():
        fear_total += get(name, 0.0) * w * smile_discount
        fear_wsum += w

    total_weighted = stress_total + ctx_total + fear_total
    total_weights = stress_wsum + ctx_wsum + fear_wsum
    raw = total_weighted / total_weights
    raw -= positive_signal * 0.15

    return max(0.0, min(1.0, raw * 2.5))


def compute_face_valence(blendshapes_dict):
    """Compute overall face valence from blendshapes."""
    pos = (
        blendshapes_dict.get("mouthSmileLeft", 0.0)
        + blendshapes_dict.get("mouthSmileRight", 0.0)
        + blendshapes_dict.get("cheekSquintLeft", 0.0) * 0.5
        + blendshapes_dict.get("cheekSquintRight", 0.0) * 0.5
    ) / 3.0
    neg = (
        blendshapes_dict.get("mouthFrownLeft", 0.0)
        + blendshapes_dict.get("mouthFrownRight", 0.0)
        + blendshapes_dict.get("browDownLeft", 0.0)
        + blendshapes_dict.get("browDownRight", 0.0)
        + blendshapes_dict.get("noseSneerLeft", 0.0) * 0.5
        + blendshapes_dict.get("noseSneerRight", 0.0) * 0.5
    ) / 4.0
    return max(-1.0, min(1.0, (pos - neg) * 2.0))


def compute_au_velocities(bs_dict, prev_bs_dict, delta_t):
    """Compute per-frame velocity for fear-relevant AUs."""
    if prev_bs_dict is None or delta_t is None:
        return {au: 0.0 for au in VELOCITY_AUS}, 0.0
    dt = max(delta_t, 1.0 / 120.0)
    velocities = {}
    for au in VELOCITY_AUS:
        v = (bs_dict.get(au, 0.0) - prev_bs_dict.get(au, 0.0)) / dt
        velocities[au] = max(0.0, v)
    startle_score = max(velocities.values()) if velocities else 0.0
    return velocities, startle_score


def compute_ctx_tag(bs_dict, smile_level):
    """Compute MediaPipe context tag from blendshapes."""
    frown_lvl = max(bs_dict.get("mouthFrownLeft", 0), bs_dict.get("mouthFrownRight", 0))
    press_lvl = max(bs_dict.get("mouthPressLeft", 0), bs_dict.get("mouthPressRight", 0))
    brow_dn = max(bs_dict.get("browDownLeft", 0), bs_dict.get("browDownRight", 0))
    eye_wd = max(bs_dict.get("eyeWideLeft", 0), bs_dict.get("eyeWideRight", 0))

    if smile_level > 0.3:
        return "JOY"
    elif eye_wd > 0.3 and bs_dict.get("browInnerUp", 0) > 0.2:
        return "FEAR"
    elif brow_dn > 0.2 and press_lvl < 0.05:
        return "CONC"
    elif frown_lvl > press_lvl and frown_lvl > 0.1:
        return "SAD"
    elif press_lvl > 0.15:
        return "STRESS"
    else:
        return "---"


def compute_expression_activity(bs_dict):
    """Mean activation of expression-related AUs (for rPPG motion gating)."""
    total = sum(bs_dict.get(au, 0.0) for au in EXPRESSION_AUS)
    return total / len(EXPRESSION_AUS)


def compute_composite_fear(hs_arousal, mp_tension):
    """Two-tool composite fear score. 60% HS arousal + 40% MP tension."""
    return min(1.0, max(0.0, 0.60 * hs_arousal + 0.40 * mp_tension))


def compute_agreement(mp_ctx_tag, mp_tension, hs_dominant, hs_arousal, hs_emotions):
    """Compute agreement and veto between MP and HS readings."""
    hs_fear = hs_emotions.get("Fear", 0)
    hs_surprise = hs_emotions.get("Surprise", 0)
    hs_anger = hs_emotions.get("Anger", 0)
    hs_contempt = hs_emotions.get("Contempt", 0)
    hs_happiness = hs_emotions.get("Happiness", 0)
    hs_sadness = hs_emotions.get("Sadness", 0)

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


# ═══════════════════════════════════════════════════════════════════════════════
# rPPG ROI EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_forehead_rgb(frame, face_bbox):
    """Extract mean RGB from forehead region of face bounding box.

    Args:
        frame: BGR image
        face_bbox: (x1, y1, x2, y2) face bounding box

    Returns:
        (mean_r, mean_g, mean_b) or None if ROI is empty
    """
    x1, y1, x2, y2 = face_bbox
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return None

    roi_y1 = y1 + int(FOREHEAD_Y_START * h)
    roi_y2 = y1 + int(FOREHEAD_Y_END * h)
    roi_x1 = x1 + int(FOREHEAD_X_START * w)
    roi_x2 = x1 + int(FOREHEAD_X_END * w)

    # Clamp to frame bounds
    roi_y1 = max(0, roi_y1)
    roi_y2 = min(frame.shape[0], roi_y2)
    roi_x1 = max(0, roi_x1)
    roi_x2 = min(frame.shape[1], roi_x2)

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None

    mean_bgr = roi.mean(axis=(0, 1))
    return (mean_bgr[2], mean_bgr[1], mean_bgr[0])  # BGR → RGB


# ═══════════════════════════════════════════════════════════════════════════════
# HUD DRAWING
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hud(canvas, panel_x, frame_h, mp_data, hs_data, fusion_data, rppg_data):
    """Draw the 4-section side panel HUD (MP + HS + FUSION + rPPG)."""
    pw = PANEL_WIDTH
    font = cv2.FONT_HERSHEY_SIMPLEX

    canvas[0:frame_h, panel_x:panel_x + pw] = PANEL_BG

    y = 10
    px = panel_x + 10
    bar_x = panel_x + 170
    bar_max = 180

    # ── MP TRIGGERS ──────────────────────────────────────────────────────
    cv2.putText(canvas, "MP TRIGGERS", (px, y + 14), font, 0.55, (100, 200, 255), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
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
        ctx_colors = {
            "JOY": (0, 200, 0), "FEAR": (0, 0, 255),
            "CONC": (200, 200, 0), "SAD": (200, 100, 0),
            "STRESS": (0, 50, 255), "---": (150, 150, 150),
        }
        ctx_col = ctx_colors.get(ctx, (150, 150, 150))
        cv2.putText(canvas, f"State: [{ctx}]", (px, y + 12), font, 0.50, ctx_col, 2)
        y += 20

        vel_tag = mp_data["velocity_tag"]
        ss = mp_data["startle_score"]
        vel_col = (0, 100, 255) if vel_tag == "STARTLE" else (100, 100, 100)
        cv2.putText(canvas, f"Startle: [{vel_tag}] {ss:.1f}/s",
                    (px, y + 12), font, 0.40, vel_col, 1)
        y += 18
    else:
        cv2.putText(canvas, "No face detected", (px, y + 12), font, 0.45, (100, 100, 100), 1)
        y += 20

    y += 5

    # ── HS TRIGGERS ──────────────────────────────────────────────────────
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8
    crop_src = hs_data.get("crop_source", "none")
    crop_label = {"haar": "[H]", "mp": "[M]", "none": "[-]"}.get(crop_src, "[-]")
    cv2.putText(canvas, f"HS TRIGGERS {crop_label}", (px, y + 14), font, 0.55, (0, 200, 255), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
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
    else:
        cv2.putText(canvas, "No face detected", (px, y + 12), font, 0.45, (100, 100, 100), 1)
        y += 20

    y += 5

    # ── FUSION ───────────────────────────────────────────────────────────
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8
    cv2.putText(canvas, "FUSION", (px, y + 14), font, 0.55, (255, 100, 255), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    if fusion_data:
        cf = fusion_data["composite_fear"]
        cf_bar = int(min(cf, 1.0) * bar_max)
        cv2.putText(canvas, f"Fear: {cf:.2f}", (px, y + 12), font, 0.45, (0, 0, 255), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + cf_bar, y + 14), (0, 50, 255), -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 20

        agree = fusion_data.get("agreement_tag", "---")
        agree_col = (0, 255, 0) if "AGREE" in agree else (255, 100, 100)
        cv2.putText(canvas, f"[{agree}]", (px, y + 12), font, 0.45, agree_col, 1)
        y += 18
    else:
        cv2.putText(canvas, "---", (px, y + 12), font, 0.45, (100, 100, 100), 1)
        y += 18

    y += 5

    # ── rPPG ─────────────────────────────────────────────────────────────
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8
    cv2.putText(canvas, "rPPG", (px, y + 14), font, 0.55, (0, 255, 200), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    if rppg_data["face_detected"]:
        n_collected = rppg_data["n_collected"]
        cv2.putText(canvas, f"Frames: {n_collected}", (px, y + 12), font, 0.40,
                    (200, 200, 200), 1)
        y += 16

        bpm = rppg_data.get("current_bpm")
        if bpm is not None and bpm > 0:
            bpm_col = (0, 255, 0) if 50 <= bpm <= 150 else (0, 0, 255)
            cv2.putText(canvas, f"BPM: {bpm:.0f}", (px, y + 12), font, 0.50, bpm_col, 2)
            y += 20

            trend = rppg_data.get("trend", "")
            if trend:
                trend_col = (0, 100, 255) if trend == "RISING" else (0, 200, 100)
                cv2.putText(canvas, f"Trend: {trend}", (px, y + 12), font, 0.40, trend_col, 1)
                y += 16
        else:
            cv2.putText(canvas, "Collecting...", (px, y + 12), font, 0.40, (150, 150, 150), 1)
            y += 16

        # Signal quality bar
        quality = rppg_data.get("quality", 0)
        q_bar = int(min(quality, 1.0) * bar_max)
        cv2.putText(canvas, f"Signal: {quality:.0%}", (px, y + 12), font, 0.35,
                    (200, 200, 200), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + q_bar, y + 12), (0, 200, 150), -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 12), (60, 60, 80), 1)
        y += 16
    else:
        cv2.putText(canvas, "No face for rPPG", (px, y + 12), font, 0.40, (100, 100, 100), 1)
        y += 16

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MP+HS+rPPG Triple-Tool Fusion")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. Omit for webcam (default: 0)")
    parser.add_argument("--algorithm", default="all",
                        choices=["chrom", "pos", "green", "all"],
                        help="rPPG algorithm (default: all)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="BPM estimation window in seconds (default: 30)")
    parser.add_argument("--step", type=float, default=5.0,
                        help="BPM window step in seconds (default: 5)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Max seconds to process (0 = full video)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no HUD window)")
    parser.add_argument("--no-session", action="store_true",
                        help="Skip session_meta interactive prompts")
    args = parser.parse_args()

    ensure_model()

    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # ── Session metadata ──────────────────────────────────────────────────
    session = Session("fusion", log_dir)
    if not args.no_session:
        session.pre_session_prompt()
    else:
        session.subject_id = "auto"
        session.session_label = "fusion_benchmark"

    csv_path = os.path.join(log_dir, f"{session.session_id}_fusion_temp.csv")
    csv_compact_path = os.path.join(log_dir, f"{session.session_id}_fusion_compact_temp.csv")
    rppg_csv_path = os.path.join(log_dir, f"{session.session_id}_fusion_rppg_temp.csv")
    summary_path = None

    session.device_used = "cpu"
    session.video_source = args.video if args.video else "webcam"

    # ── Initialize MediaPipe ──────────────────────────────────────────────
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

    # ── Initialize HSEmotion ──────────────────────────────────────────────
    print("Loading HSEmotion model...")
    hs_model_name = "enet_b0_8_va_mtl"
    recognizer = HSEmotionRecognizer(model_name=hs_model_name)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print(f"HSEmotion loaded: {hs_model_name}")

    # ── Open video / webcam ────────────────────────────────────────────────
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        src = f"file '{args.video}'" if args.video else "webcam"
        print(f"ERROR: Could not open {src}.")
        return

    video_fps = None
    if args.video:
        _fps = cap.get(cv2.CAP_PROP_FPS)
        video_fps = _fps if _fps and _fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / video_fps if video_fps > 0 else 0
        max_frames = int(args.duration * video_fps) if args.duration > 0 else total_frames
        print(f"Video: {args.video}")
        print(f"  FPS: {video_fps:.1f} | Total: {total_frames} frames ({total_duration:.1f}s)")
        if args.duration > 0:
            print(f"  Processing first {args.duration:.0f}s ({max_frames} frames)")
    else:
        # Webcam mode
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_fps = 30.0
        total_frames = None
        total_duration = None
        max_frames = None
        print("Webcam mode (capture until 'q' pressed)")

    print("\n" + "=" * 60)
    print("MP+HS+rPPG Triple-Tool Fusion")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print("Press 'q' to quit\n")

    session.start()

    # ── CSV headers ───────────────────────────────────────────────────────
    bs_columns = ["mp_" + n for n in KEY_BLENDSHAPES]
    velocity_cols = ["mp_vel_" + au for au in VELOCITY_AUS] + ["mp_tension_velocity"]
    hs_emotion_cols = ["hs_" + l.lower() for l in EMOTION_LABELS]

    compact_header = (
        ["frame", "timestamp", "hs_crop_source", "composite_fear"]
        + ["hs_dominant"]
        + ["mp_ctx_tag", "mp_velocity_tag", "agreement_tag", "veto_tag"]
        + ["hs_fear", "hs_arousal", "hs_dominant_score"]
        + ["mp_tension", "mp_startle_score"]
        + ["rppg_face_detected", "rppg_mean_r", "rppg_mean_g", "rppg_mean_b"]
    )

    csv_header = (
        ["frame", "timestamp", "hs_crop_source", "composite_fear"]
        + ["hs_dominant"]
        + ["mp_ctx_tag", "mp_velocity_tag", "agreement_tag", "veto_tag"]
        + ["hs_fear", "hs_arousal", "hs_dominant_score"]
        + ["mp_tension", "mp_startle_score"]
        + ["mp_face_valence", "mp_smile_level"]
        + bs_columns + velocity_cols
        + hs_emotion_cols
        + ["rppg_face_detected", "rppg_roi_source",
           "rppg_mean_r", "rppg_mean_g", "rppg_mean_b"]
        + ["mp_face_detected", "hs_face_detected"]
        + ["latency_ms", "cpu_percent", "ram_mb"]
    )

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    csv_compact_file = open(csv_compact_path, "w", newline="")
    csv_compact_writer = csv.writer(csv_compact_file)
    csv_compact_writer.writerow(compact_header)

    # ── rPPG state ────────────────────────────────────────────────────────
    rppg_rgbs = []          # accumulated [R, G, B] per face-detected frame
    rppg_timestamps = []    # corresponding timestamps
    rppg_motion_flags = []  # True when facial expression active (motion artifact)
    rppg_face_count = 0
    current_bpm = None
    bpm_trend = ""
    prev_bpm = None

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()
    prev_bs_dict = None
    prev_elapsed = None
    prev_tension = 0.0

    # ── Frame loop ────────────────────────────────────────────────────────
    while (max_frames is None or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed = frame_count / video_fps
        start_time = time.perf_counter()

        # Default values
        mp_face_detected = False
        hs_face_detected = False
        rppg_face_detected = False

        tension = 0.0
        face_valence = 0.0
        smile_level = 0.0
        ctx_tag = "---"
        bs_dict = {}
        au_velocities = {au: 0.0 for au in VELOCITY_AUS}
        startle_score = 0.0
        tension_velocity = 0.0
        velocity_tag = "---"
        mp_face_bbox = None

        valence_hs = 0.0
        arousal_hs = 0.0
        emotions_hs = {l: 0.0 for l in EMOTION_LABELS}
        dominant_hs = "Neutral"
        dominant_score_hs = 0.0
        hs_face_bbox = None
        hs_crop_source = "none"

        composite_fear = 0.0
        agreement_tag = ""
        veto_tag = "---"

        rppg_roi_source = "none"
        rppg_mean_r = 0.0
        rppg_mean_g = 0.0
        rppg_mean_b = 0.0

        try:
            # ── MediaPipe inference ──────────────────────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_lib.Image(
                image_format=mp_lib.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_count * 1000 / video_fps)

            mp_results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if mp_results.face_blendshapes and len(mp_results.face_blendshapes) > 0:
                mp_face_detected = True
                blendshapes = mp_results.face_blendshapes[0]
                bs_dict = {bs.category_name: bs.score for bs in blendshapes}

                tension = compute_tension(bs_dict)
                face_valence = compute_face_valence(bs_dict)
                smile_level = max(
                    bs_dict.get("mouthSmileLeft", 0),
                    bs_dict.get("mouthSmileRight", 0),
                )

                delta_t = (elapsed - prev_elapsed) if prev_elapsed is not None else None
                au_velocities, startle_score = compute_au_velocities(
                    bs_dict, prev_bs_dict, delta_t,
                )
                if prev_elapsed is not None and delta_t and delta_t > 0:
                    tension_velocity = max(0.0, (tension - prev_tension) / delta_t)

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
                prev_tension = tension

            # ── HSEmotion inference (Haar-first, MP-fallback) ────────
            face_img = None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

            if len(faces) > 0:
                fx, fy, fw_f, fh_f = max(faces, key=lambda f: f[2] * f[3])
                face_img = frame[fy:fy + fh_f, fx:fx + fw_f]
                hs_face_bbox = (fx, fy, fx + fw_f, fy + fh_f)
                hs_crop_source = "haar"
            elif mp_face_detected and mp_face_bbox:
                x1, y1, x2, y2 = mp_face_bbox
                w, h = x2 - x1, y2 - y1
                pad = 0.15
                x1p = max(0, int(x1 - w * pad))
                y1p = max(0, int(y1 - h * pad))
                x2p = min(frame.shape[1], int(x2 + w * pad))
                y2p = min(frame.shape[0], int(y2 + h * pad))
                face_img = frame[y1p:y2p, x1p:x2p]
                hs_face_bbox = (x1p, y1p, x2p, y2p)
                hs_crop_source = "mp"

            if face_img is not None:
                hs_face_detected = True
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)

                valence_hs = float(scores[-2])
                arousal_hs = float(scores[-1])
                for i, label in enumerate(EMOTION_LABELS):
                    if i < len(scores) - 2:
                        emotions_hs[label] = float(scores[i])

                dominant_hs = emotion if emotion else max(emotions_hs, key=emotions_hs.get)
                dominant_score_hs = emotions_hs.get(dominant_hs, 0)

            # ── rPPG: extract forehead RGB (near-zero cost) ──────────
            # Reuse the best available face bbox
            rppg_bbox = hs_face_bbox if hs_face_bbox else mp_face_bbox
            if rppg_bbox:
                rgb = extract_forehead_rgb(frame, rppg_bbox)
                if rgb is not None:
                    rppg_face_detected = True
                    rppg_mean_r, rppg_mean_g, rppg_mean_b = rgb
                    rppg_roi_source = hs_crop_source if hs_face_bbox else "mp"
                    rppg_rgbs.append([rppg_mean_r, rppg_mean_g, rppg_mean_b])
                    rppg_timestamps.append(elapsed)
                    expr_activity = compute_expression_activity(bs_dict)
                    rppg_motion_flags.append(expr_activity > EXPRESSION_ACTIVITY_THRESHOLD)
                    rppg_face_count += 1

            # ── Fusion ───────────────────────────────────────────────
            if mp_face_detected and hs_face_detected:
                composite_fear = compute_composite_fear(arousal_hs, tension)
                agreement_tag, veto_tag = compute_agreement(
                    ctx_tag, tension, dominant_hs, arousal_hs, emotions_hs,
                )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            print(f"\rFrame {frame_count:4d} | Error: {str(e)[:60]}", end="")

        # ── Live rPPG BPM (updated every 5s worth of frames) ──────────
        step_frames = int(args.step * video_fps)
        if (len(rppg_rgbs) >= int(args.window * video_fps)
                and frame_count % step_frames == 0):
            eff_fps = len(rppg_rgbs) / (rppg_timestamps[-1] - rppg_timestamps[0]) \
                if len(rppg_timestamps) > 1 else video_fps
            rgbs_arr = np.array(rppg_rgbs)
            # Apply motion interpolation before rPPG algorithm
            if rppg_motion_flags:
                rgbs_arr = interpolate_motion_frames(rgbs_arr, rppg_motion_flags)
            from rppg_algorithms import pos as pos_algo, estimate_bpm
            winsize = max(int(eff_fps * 1.5), 15)
            sig = pos_algo(rgbs_arr, winsize=winsize)
            # Use last window_s seconds
            win_frames = int(args.window * eff_fps)
            segment = sig[-win_frames:]
            bpm_val, _, _, _, _ = estimate_bpm(segment, eff_fps)
            if 50 <= bpm_val <= 150:
                prev_bpm = current_bpm
                current_bpm = bpm_val
                if prev_bpm and current_bpm:
                    if current_bpm > prev_bpm + 2:
                        bpm_trend = "RISING"
                    elif current_bpm < prev_bpm - 2:
                        bpm_trend = "FALLING"
                    else:
                        bpm_trend = "STABLE"

        # ── Session resources ──────────────────────────────────────────
        meta = session.log_frame(
            face_detected=(mp_face_detected or hs_face_detected),
            face_confidence=1.0 if mp_face_detected else 0.0,
            face_bbox_area=0,
        )

        # ── CSV row ────────────────────────────────────────────────────
        bs_values = [f"{bs_dict.get(n, 0):.4f}" for n in KEY_BLENDSHAPES]
        au_velocity_values = [f"{au_velocities.get(au, 0.0):.4f}" for au in VELOCITY_AUS]
        tension_velocity_value = f"{tension_velocity:.4f}"
        hs_emotion_values = [f"{emotions_hs.get(l, 0):.4f}" for l in EMOTION_LABELS]

        core_row = (
            [frame_count, f"{elapsed:.3f}", hs_crop_source, f"{composite_fear:.4f}"]
            + [dominant_hs]
            + [ctx_tag, velocity_tag, agreement_tag, veto_tag]
            + [f"{emotions_hs.get('Fear', 0):.4f}", f"{arousal_hs:.4f}",
               f"{dominant_score_hs:.4f}"]
            + [f"{tension:.4f}", f"{startle_score:.4f}"]
        )

        # Compact CSV: core + rPPG RGB
        compact_row = core_row + [
            int(rppg_face_detected),
            f"{rppg_mean_r:.1f}", f"{rppg_mean_g:.1f}", f"{rppg_mean_b:.1f}",
        ]
        csv_compact_writer.writerow(compact_row)

        # Full CSV: core + detailed + rPPG
        full_row = core_row + (
            [f"{face_valence:.4f}", f"{smile_level:.4f}"]
            + bs_values
            + au_velocity_values + [tension_velocity_value]
            + hs_emotion_values
            + [int(rppg_face_detected), rppg_roi_source,
               f"{rppg_mean_r:.1f}", f"{rppg_mean_g:.1f}", f"{rppg_mean_b:.1f}"]
            + [int(mp_face_detected), int(hs_face_detected)]
            + [f"{latency_ms:.1f}", meta["cpu_percent"], meta["ram_mb"]]
        )
        csv_writer.writerow(full_row)

        # ── Console ────────────────────────────────────────────────────
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        agree_short = agreement_tag[:10] if agreement_tag else "---"
        crop_ind = hs_crop_source[0].upper() if hs_crop_source != "none" else "-"
        bpm_str = f"{current_bpm:.0f}" if current_bpm else "---"
        print(f"\rFrame {frame_count:4d} | "
              f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
              f"Lat: {latency_ms:5.1f}ms | "
              f"T:{tension:.2f} [{ctx_tag:5s}] | "
              f"A:{arousal_hs:.2f} [{dominant_hs[:4]:4s}]{crop_ind} | "
              f"BPM:{bpm_str:>4s} | "
              f"[{agree_short}]", end="")

        # ── HUD ────────────────────────────────────────────────────────
        if not args.no_display:
            fh, fw = frame.shape[:2]
            canvas_h = HUD_MIN_HEIGHT
            scale = canvas_h / fh
            video_w = int(fw * scale)
            canvas_w = video_w + PANEL_WIDTH

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            resized = cv2.resize(frame, (video_w, canvas_h))
            canvas[0:canvas_h, 0:video_w] = resized

            # Face bboxes
            if mp_face_bbox:
                bx1, by1 = int(mp_face_bbox[0] * scale), int(mp_face_bbox[1] * scale)
                bx2, by2 = int(mp_face_bbox[2] * scale), int(mp_face_bbox[3] * scale)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 100, 0), 2)
            if hs_face_bbox:
                bx1, by1 = int(hs_face_bbox[0] * scale), int(hs_face_bbox[1] * scale)
                bx2, by2 = int(hs_face_bbox[2] * scale), int(hs_face_bbox[3] * scale)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            cv2.putText(canvas, f"TIME: {elapsed_min:02d}:{elapsed_sec:02d}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, f"{latency_ms:.0f}ms",
                        (video_w - 80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            mp_data = {
                "face_detected": mp_face_detected, "tension": tension,
                "face_valence": face_valence, "smile_level": smile_level,
                "ctx_tag": ctx_tag, "startle_score": startle_score,
                "velocity_tag": velocity_tag, "bs_dict": bs_dict,
            }
            hs_data_hud = {
                "face_detected": hs_face_detected, "arousal": arousal_hs,
                "valence": valence_hs, "dominant": dominant_hs,
                "dominant_score": dominant_score_hs, "emotions": emotions_hs,
                "crop_source": hs_crop_source,
            }
            fusion_data_hud = {
                "composite_fear": composite_fear,
                "agreement_tag": agreement_tag if agreement_tag else "---",
                "veto_tag": veto_tag,
            }
            rppg_quality = rppg_face_count / frame_count if frame_count > 0 else 0
            rppg_data_hud = {
                "face_detected": rppg_face_detected,
                "n_collected": rppg_face_count,
                "current_bpm": current_bpm,
                "trend": bpm_trend,
                "quality": rppg_quality,
            }

            canvas = draw_hud(canvas, video_w, canvas_h,
                              mp_data, hs_data_hud, fusion_data_hud, rppg_data_hud)

            cv2.imshow("MP+HS+rPPG Fusion", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    print()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    csv_file.close()
    csv_compact_file.close()

    # ── Post-loop rPPG: full BPM timeseries ──────────────────────────────
    rppg_summary_rows = []
    if len(rppg_rgbs) >= int(video_fps * 10):
        rgbs_arr = np.array(rppg_rgbs)
        ts_arr = np.array(rppg_timestamps)
        eff_fps = len(rgbs_arr) / (ts_arr[-1] - ts_arr[0]) if len(ts_arr) > 1 else video_fps

        # Motion-aware cleaning
        motion_flags_arr = rppg_motion_flags if rppg_motion_flags else None
        if motion_flags_arr:
            motion_pct_total = sum(motion_flags_arr) / len(motion_flags_arr) * 100
            rgbs_cleaned = interpolate_motion_frames(rgbs_arr, motion_flags_arr)
        else:
            motion_pct_total = 0.0
            rgbs_cleaned = rgbs_arr

        print(f"\n--- rPPG Post-Processing ---")
        print(f"  Face frames collected: {len(rgbs_arr)} / {frame_count} "
              f"({len(rgbs_arr)/frame_count*100:.1f}%)")
        print(f"  Effective FPS: {eff_fps:.1f}")
        print(f"  Motion-flagged frames: {motion_pct_total:.1f}%")

        rppg_summary_rows = compute_bpm_timeseries(
            rgbs_cleaned, ts_arr, eff_fps,
            algorithm=args.algorithm,
            window_s=args.window, step_s=args.step,
            motion_flags=motion_flags_arr,
        )

        # Write rPPG summary CSV
        with open(rppg_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "window_idx", "t_start", "t_center", "t_end",
                "algorithm", "bpm", "bpm_smoothed", "bpm_plausible",
                "snr", "prominence", "motion_pct", "n_face_frames",
                "bpm_spread",
            ])
            w.writeheader()
            w.writerows(rppg_summary_rows)

        # Print summary per algorithm
        algos_seen = set()
        for row in rppg_summary_rows:
            algo = row["algorithm"]
            if algo not in algos_seen:
                algos_seen.add(algo)
                algo_rows = [r for r in rppg_summary_rows if r["algorithm"] == algo]
                bpms = [r["bpm"] for r in algo_rows]
                smoothed = [r["bpm_smoothed"] for r in algo_rows]
                snrs = [r["snr"] for r in algo_rows]
                plausible = sum(1 for r in algo_rows if r["bpm_plausible"])
                q25, q75 = np.percentile(smoothed, [25, 75])
                iqr = q75 - q25
                pct_resting = sum(1 for b in smoothed if 50 <= b <= 100) / len(smoothed) * 100
                pct_low_snr = sum(1 for s in snrs if s < 3.0) / len(snrs) * 100
                stability = "GOOD" if iqr < 10 else ("MODERATE" if iqr < 20 else "POOR")
                print(f"\n  {algo}:")
                print(f"    Raw BPM:   {np.mean(bpms):.1f} +/- {np.std(bpms):.1f} "
                      f"(range {np.min(bpms):.1f}-{np.max(bpms):.1f})")
                print(f"    Smoothed:  {np.mean(smoothed):.1f} +/- {np.std(smoothed):.1f} "
                      f"(range {np.min(smoothed):.1f}-{np.max(smoothed):.1f})")
                print(f"    Plausible: {plausible}/{len(algo_rows)} windows")
                print(f"    IQR: {iqr:.1f} BPM | Resting range: {pct_resting:.0f}% "
                      f"| Low-SNR: {pct_low_snr:.0f}% | Stability: {stability}")
    else:
        print(f"\n  rPPG: Only {len(rppg_rgbs)} face frames collected. "
              f"Need {int(video_fps * 10)} (10s) for BPM estimation.")

    # ── Session finalization ──────────────────────────────────────────────
    if not args.no_session:
        if not session.post_session_confirm(csv_path):
            return

    csv_path, summary_path = session.finalize_session_files(csv_path)

    final_compact = csv_path.replace(".csv", "_compact.csv")
    if os.path.isfile(csv_compact_path):
        os.rename(csv_compact_path, final_compact)

    final_rppg = csv_path.replace(".csv", "_rppg.csv")
    if os.path.isfile(rppg_csv_path):
        os.rename(rppg_csv_path, final_rppg)

    if latencies:
        record, extra_lines = session.finish(
            latencies, frame_count, csv_path, summary_path,
        )

        summary_lines = [
            "", "=" * 60,
            "PERFORMANCE SUMMARY — MP+HS+rPPG Triple-Tool Fusion",
            "=" * 60,
            f"Session date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session duration:       {record['duration_s']:.1f} seconds",
            f"Total frames processed: {frame_count}",
            f"Average latency:        {record['avg_latency_ms']:.1f} ms (all tools)",
            f"Median latency:         {record['median_latency_ms']:.1f} ms",
            f"95th percentile:        {record['p95_latency_ms']:.1f} ms",
            f"Effective FPS:          {record['fps_actual']:.1f}",
            "",
            f"rPPG face frames:       {rppg_face_count}/{frame_count} "
            f"({rppg_face_count/frame_count*100:.1f}%)" if frame_count > 0 else "",
            f"rPPG windows computed:  {len(rppg_summary_rows)}",
            "=" * 60,
            f"CSV (full):    {csv_path}",
            f"CSV (compact): {final_compact}",
            f"CSV (rPPG):    {final_rppg}",
            f"Summary:       {summary_path}",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nDone! {frame_count} frames logged.")

        post_session_analysis(csv_path, summary_path,
                              run_viz=True, run_emotion=False)


if __name__ == "__main__":
    main()
