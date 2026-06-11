"""
La Facade Fissuree - MediaPipe Blendshapes Test
================================================
Tests MediaPipe FaceLandmarker (Task API) for 52 real blendshape scores
representing individual facial muscle activations.
Instead of classifying emotions, this gives raw action scores
(brow raise, jaw open, eye squint, lip pucker, etc.).
YOU decide what combination = "stressed" or "aroused".

Requires: models/face_landmarker.task (auto-downloaded on first run)

Usage:
    python test_mediapipe.py

Press 'q' to quit. Results saved to logs/ folder.
"""

import cv2
import time
import os
import csv
import argparse
import urllib.request
import numpy as np
from datetime import datetime
import mediapipe as mp
from session_meta import Session
from post_session_analysis import post_session_analysis
from improved_fear_detection import get_velocity_tag

# --- Model path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ---- Blendshape groups ----
# Negative-valence: these fire during stress, fear, anger, disgust
NEGATIVE_BLENDSHAPES = [
    "browDownLeft",      # Brow furrow — anger/concentration
    "browDownRight",
    "eyeWideLeft",       # Wide eyes — fear/surprise
    "eyeWideRight",
    "eyeSquintLeft",     # Eye squint — tension/discomfort
    "eyeSquintRight",
    "mouthFrownLeft",    # Frown — sadness/displeasure
    "mouthFrownRight",
    "mouthPressLeft",    # Lip press — tension/stress
    "mouthPressRight",
    "noseSneerLeft",     # Nose sneer — disgust
    "noseSneerRight",
    "mouthUpperUpLeft",  # Lip raise — disgust (v3)
    "mouthUpperUpRight",
    "cheekPuff",         # Cheek puff — bracing/exhale
]

# Positive-valence: these fire during smiling/joy
POSITIVE_BLENDSHAPES = [
    "mouthSmileLeft",    # Smile
    "mouthSmileRight",
    "cheekSquintLeft",   # Duchenne smile (genuine)
    "cheekSquintRight",
    "mouthDimpleLeft",   # Dimple with smile
    "mouthDimpleRight",
]

# Ambiguous signals: browInnerUp and jawOpen fire for both stress and positive emotions.
# browInnerUp  = fear/surprise BUT also excited smile / suppressed laughter
# jawOpen      = shock/surprise BUT also laughing
# Handled inside compute_tension() with a laughter proxy gate (jawOpen + browInnerUp together).

AMBIGUOUS_BLENDSHAPES = [
    "browInnerUp",  # fear/surprise BUT also excited smile / suppressed laughter
    "jawOpen",      # shock/surprise BUT also laughing
]

# All 52 MediaPipe Face Landmarker blendshape names (sorted, matches MediaPipe Task API output).
# Used for writing CSV header before the capture loop so the file is always parseable,
# even if no face is detected in the session.
BLENDSHAPE_NAMES = sorted([
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
])

# All blendshapes shown on the overlay
STRESS_BLENDSHAPES = NEGATIVE_BLENDSHAPES + AMBIGUOUS_BLENDSHAPES + POSITIVE_BLENDSHAPES

# ---- Velocity tracking: AUs monitored for startle detection ----
VELOCITY_AUS = [
    "eyeWideLeft", "eyeWideRight", "browInnerUp",
    "jawOpen", "mouthPressLeft", "mouthPressRight",
]
# CALIBRATION PENDING: 0.4 change in <150ms ≈ 2.67/s, using 3.0 as initial estimate.
# Adjust after 3+ jumpscare sessions using data-driven approach (see phase2_sprint.md).
STARTLE_VELOCITY_THRESHOLD = 3.0


def ensure_model():
    """Download the FaceLandmarker model if not present.

    Checks MODEL_PATH for the face_landmarker.task file and
    downloads it from MODEL_URL if missing. Creates the models/
    directory if it does not exist.
    """
    if os.path.isfile(MODEL_PATH):
        return
    print(f"Downloading FaceLandmarker model to {MODEL_PATH} ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def compute_tension(blendshapes_dict):
    """Compute composite facial tension score (v3 formula).

    Research-informed stress/fear score using FACS-based signal
    layers with contextual discounts for sadness, concentration,
    and smiling.

    v3 improvements over v2:
      1. mouthPress is the HIGHEST weighted signal (Navarro: lip
         compression is the most reliable involuntary stress indicator)
      2. Sadness discount: browDown/eyeSquint contribute 50% less
         when mouthFrown > mouthPress (sadness, not anger)
      3. Gamer lean guard: browDown alone with low mouthPress =
         concentration, not stress
      4. Smile discount: ambiguous signals discounted when smile
         is detected (carried over from v2)
      5. Added mouthUpperUp for disgust detection
      6. Laughter proxy gate: browInnerUp weight reduced when
         jawOpen + browInnerUp spike together (suppressed laughter pattern)

    Architecture note:
      The three "layers" (stress, context-dependent, fear) are organizational.
      Step 4 merges them into a single flat weighted average:
          (stress_total + ctx_total + fear_total) / total_weights
      The layers do NOT have independent scales — adding high-weight signals
      to any layer dilutes the others.

    Scaling note:
      The final `* 2.5` factor was calibrated on the developer's face to spread
      output across 0–1. It may need per-subject adjustment: subjects with flat
      affect will saturate near 0.4, expressive subjects will saturate at 1.0.
      Plan to normalize per-subject during experimental analysis.

    Args:
        blendshapes_dict: Dict mapping MediaPipe blendshape names
            (str) to activation values (float, 0.0-1.0).

    Returns:
        Float between 0.0 and 1.0. Higher = more stressed/tense.
    """
    get = blendshapes_dict.get  # shorthand

    # ---- Detect context signals first ----

    # Smile level (positive valence indicator)
    smile_level = max(get("mouthSmileLeft", 0.0), get("mouthSmileRight", 0.0))
    cheek_squint = max(get("cheekSquintLeft", 0.0), get("cheekSquintRight", 0.0))
    positive_signal = max(smile_level, cheek_squint * 0.8)

    # Sadness level (frown without press = sad, not stressed)
    frown_level = max(get("mouthFrownLeft", 0.0), get("mouthFrownRight", 0.0))
    press_level = max(get("mouthPressLeft", 0.0), get("mouthPressRight", 0.0))
    is_sad = frown_level > press_level and frown_level > 0.1

    # Gamer lean: browDown high but mouthPress near zero = concentration
    brow_down = max(get("browDownLeft", 0.0), get("browDownRight", 0.0))
    is_concentrating = brow_down > 0.2 and press_level < 0.05

    # ---- Step 1: Pure stress signals (always count at full weight) ----
    # mouthPress is the #1 stress indicator (Navarro, lip compression research)
    stress_total = 0.0
    stress_wsum = 0.0
    pure_stress = {
        "mouthPressLeft":  1.5,   # #1 stress signal — lip compression
        "mouthPressRight": 1.5,
        "noseSneerLeft":   0.8,   # disgust/anger
        "noseSneerRight":  0.8,
        "mouthUpperUpLeft":  0.5, # disgust (lip raise) — new in v3
        "mouthUpperUpRight": 0.5,
        "cheekPuff":       0.3,   # bracing/exhale
    }
    for name, w in pure_stress.items():
        stress_total += get(name, 0.0) * w
        stress_wsum += w

    # ---- Step 2: Context-dependent signals ----
    # browDown & eyeSquint: sadness discount (0.5x) OR gamer lean (0.2x)
    ctx_total = 0.0
    ctx_wsum = 0.0
    brow_squint = {
        "browDownLeft":   0.7,    # lowered from 1.2 — ambiguous signal
        "browDownRight":  0.7,
        "eyeSquintLeft":  0.6,    # squint — tension OR sadness OR smile
        "eyeSquintRight": 0.6,
        "mouthFrownLeft": 0.4,    # lowered — mainly sadness, not stress
        "mouthFrownRight": 0.4,
    }

    # Determine context multiplier
    if is_concentrating:
        ctx_multiplier = 0.2    # gamer lean: mostly ignore brow furrow
    elif is_sad:
        ctx_multiplier = 0.5    # sadness: half credit
    else:
        ctx_multiplier = 1.0    # anger/stress: full credit

    for name, w in brow_squint.items():
        ctx_total += get(name, 0.0) * w * ctx_multiplier
        ctx_wsum += w

    # ---- Step 3: Fear/startle signals — discount when smiling ----
    # Same smile discount as v2
    smile_discount = max(0.0, 1.0 - positive_signal * 2.0)  # 0 when smile>=0.5

    # Laughter proxy gate: jawOpen + browInnerUp firing together = suppressed laughter, not fear.
    # Observed: browInnerUp hits 0.968 during try-not-to-laugh (vs 0.732 during jumpscare).
    # When this gate fires, reduce browInnerUp contribution to avoid inflating tension.
    is_laughing = (get("jawOpen", 0.0) > 0.3 and get("browInnerUp", 0.0) > 0.5)
    brow_inner_up_weight = 0.3 if is_laughing else 1.3

    fear_total = 0.0
    fear_wsum = 0.0
    fear_signals = {
        "browInnerUp":  brow_inner_up_weight,  # weight reduced when laughter proxy fires
        "eyeWideLeft":  1.5,   # wide eyes — strongest fear signal
        "eyeWideRight": 1.5,
        "jawOpen":      0.6,   # surprise/shock jaw drop
    }
    for name, w in fear_signals.items():
        fear_total += get(name, 0.0) * w * smile_discount
        fear_wsum += w

    # ---- Step 4: Combine all layers ----
    total_weighted = stress_total + ctx_total + fear_total
    total_weights = stress_wsum + ctx_wsum + fear_wsum
    raw = total_weighted / total_weights

    # Subtract smile penalty (happy face ≠ stressed)
    raw -= positive_signal * 0.15

    return max(0.0, min(1.0, raw * 2.5))  # scale up, clamp 0-1


def compute_face_valence(blendshapes_dict):
    """Compute overall face valence from blendshapes.

    Positive valence indicates happiness (smile, cheek squint).
    Negative valence indicates stress or displeasure (frown,
    brow furrow, nose sneer).

    Args:
        blendshapes_dict: Dict mapping MediaPipe blendshape names
            (str) to activation values (float, 0.0-1.0).

    Returns:
        Float between -1.0 and +1.0. Positive = happy,
        negative = stressed/angry.
    """
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
    """Compute per-frame velocity for fear-relevant AUs.

    Velocity = (current - previous) / delta_t (units/second).
    Only positive velocity is returned (AU moving toward fear expression).
    Negative (relaxing) is clamped to 0.0.

    Args:
        bs_dict: Current frame blendshape dict.
        prev_bs_dict: Previous frame blendshape dict. None on first frame.
        delta_t: Seconds since previous frame. None on first frame.

    Returns:
        (velocities_dict, startle_score):
          - velocities_dict: {au: velocity} for each AU in VELOCITY_AUS, clipped >= 0
          - startle_score: max() across all AU velocities this frame
    """
    if prev_bs_dict is None or delta_t is None:
        return {au: 0.0 for au in VELOCITY_AUS}, 0.0

    dt = max(delta_t, 1.0 / 120.0)  # guard: cap at 120 fps minimum dt
    velocities = {}
    for au in VELOCITY_AUS:
        v = (bs_dict.get(au, 0.0) - prev_bs_dict.get(au, 0.0)) / dt
        velocities[au] = max(0.0, v)

    startle_score = max(velocities.values()) if velocities else 0.0
    return velocities, startle_score


def main():
    """Run the MediaPipe blendshape detection loop.

    Opens the webcam, initializes MediaPipe FaceLandmarker, and
    runs a real-time frame-by-frame loop. Computes tension and
    valence scores from 52 facial blendshapes. Displays an
    annotated video overlay with color-coded blendshape bars and
    context state tags (JOY/FEAR/CONC/SAD/STRESS). Logs all data
    to a timestamped CSV in the logs/ directory.

    Press 'q' to quit. A performance summary is printed and saved
    at the end of the session.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file. Omit to use webcam.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip video display (faster processing).")
    args = parser.parse_args()

    ensure_model()

    # Create logs directory
    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Session metadata — prompt before webcam opens
    session = Session("mediapipe", log_dir)
    session.pre_session_prompt()

    csv_path = os.path.join(log_dir, f"{session.session_id}_mediapipe_temp.csv")
    summary_path = None  # set after finalize

    session.device_used = "cpu"  # MediaPipe has no desktop GPU support on Windows
    session.video_source = args.video if args.video else "webcam"

    # --- Initialize FaceLandmarker (Task API) ---
    print("Loading MediaPipe FaceLandmarker...")
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,       # The key flag — gives 52 blendshapes
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    print("FaceLandmarker loaded! (52 blendshapes enabled)")

    # Open webcam or video file
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
        print(f"Video mode: {args.video} @ {video_fps:.1f} fps")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("MediaPipe FaceLandmarker — 52 Blendshape Tracking")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print("Key stress-related blendshapes tracked:")
    for bs in STRESS_BLENDSHAPES:
        print(f"  * {bs}")
    print("\nPress 'q' to quit")
    print()

    # Start session monitoring
    session.start()

    # CSV setup — header written before the capture loop so the file is always parseable
    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    velocity_column_names = (
        ["vel_" + au for au in VELOCITY_AUS]
        + ["tension_velocity", "startle_score", "velocity_tag"]
    )
    csv_writer.writerow(
        ["frame", "timestamp", "latency_ms", "tension", "face_valence", "smile_level"]
        + BLENDSHAPE_NAMES + meta_columns + velocity_column_names
    )

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()
    prev_bs_dict = None
    prev_elapsed = None
    prev_tension = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.perf_counter()

        try:
            # Convert to MediaPipe Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Timestamp in ms for VIDEO mode
            timestamp_ms = int(frame_count * 1000 / video_fps) if video_fps else int((time.perf_counter() - session_start) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            if results.face_blendshapes and len(results.face_blendshapes) > 0:
                blendshapes = results.face_blendshapes[0]

                # Build dict: name -> score (0-1)
                bs_dict = {}
                for bs in blendshapes:
                    bs_dict[bs.category_name] = bs.score

                # Compute composite tension and valence
                tension = compute_tension(bs_dict)
                face_valence = compute_face_valence(bs_dict)
                smile_level = max(
                    bs_dict.get("mouthSmileLeft", 0),
                    bs_dict.get("mouthSmileRight", 0),
                )

                # --- Velocity computation (both approaches run simultaneously) ---
                elapsed = frame_count / video_fps if video_fps else time.perf_counter() - session_start
                delta_t = (elapsed - prev_elapsed) if prev_elapsed is not None else None

                au_velocities, startle_score = compute_au_velocities(
                    bs_dict, prev_bs_dict, delta_t
                )

                if prev_elapsed is not None and delta_t and delta_t > 0:
                    tension_velocity = max(0.0, (tension - prev_tension) / delta_t)
                else:
                    tension_velocity = 0.0

                velocity_tag = get_velocity_tag(startle_score, STARTLE_VELOCITY_THRESHOLD)

                # Compute face bbox area from landmarks
                face_bbox_area = 0
                if results.face_landmarks and len(results.face_landmarks) > 0:
                    h, w = frame.shape[:2]
                    lms = results.face_landmarks[0]
                    xs = [lm.x * w for lm in lms]
                    ys = [lm.y * h for lm in lms]
                    face_bbox_area = int((max(xs) - min(xs)) * (max(ys) - min(ys)))

                # Session resource + face metadata
                # face_confidence: MediaPipe Task API does not expose a detection score,
                # so we use face bbox area as a size proxy (larger = more visible face).
                frame_area = frame.shape[0] * frame.shape[1]
                face_size_ratio = face_bbox_area / frame_area if frame_area > 0 else 0.0
                meta = session.log_frame(
                    face_detected=True, face_confidence=face_size_ratio,
                    face_bbox_area=face_bbox_area,
                )

                # Log to CSV — use BLENDSHAPE_NAMES order to match header
                csv_writer.writerow(
                    [frame_count, f"{elapsed:.3f}", f"{latency_ms:.1f}",
                     f"{tension:.4f}", f"{face_valence:.4f}", f"{smile_level:.4f}"]
                    + [f"{bs_dict.get(n, 0):.4f}" for n in BLENDSHAPE_NAMES]
                    + [meta["cpu_percent"], meta["ram_mb"], meta["face_detected"],
                       meta["face_confidence"], meta["face_bbox_area"]]
                    + [f"{au_velocities.get(au, 0.0):.4f}" for au in VELOCITY_AUS]
                    + [f"{tension_velocity:.4f}", f"{startle_score:.4f}", velocity_tag]
                )

                # Update previous-frame state for next velocity computation
                prev_bs_dict = bs_dict.copy()
                prev_elapsed = elapsed
                prev_tension = tension

                # Determine context state for display
                frown_lvl = max(bs_dict.get("mouthFrownLeft", 0),
                                bs_dict.get("mouthFrownRight", 0))
                press_lvl = max(bs_dict.get("mouthPressLeft", 0),
                                bs_dict.get("mouthPressRight", 0))
                brow_dn = max(bs_dict.get("browDownLeft", 0),
                              bs_dict.get("browDownRight", 0))
                eye_wd = max(bs_dict.get("eyeWideLeft", 0),
                             bs_dict.get("eyeWideRight", 0))
                if smile_level > 0.3:
                    ctx_tag = "JOY"
                elif eye_wd > 0.3 and bs_dict.get("browInnerUp", 0) > 0.2:
                    ctx_tag = "FEAR"
                elif brow_dn > 0.2 and press_lvl < 0.05:
                    ctx_tag = "CONC"   # concentration / gamer lean
                elif frown_lvl > press_lvl and frown_lvl > 0.1:
                    ctx_tag = "SAD"
                elif press_lvl > 0.15:
                    ctx_tag = "STRESS"
                else:
                    ctx_tag = "---"

                # Console print with elapsed time
                val_label = "+" if face_valence >= 0 else ""
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                print(f"\rFrame {frame_count:4d} | "
                      f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
                      f"Lat: {latency_ms:5.1f}ms | "
                      f"T: {tension:.2f} | "
                      f"V: {val_label}{face_valence:.2f} | "
                      f"[{ctx_tag:5s}] [{velocity_tag}] SS:{startle_score:.1f}", end="")

                # --- Draw metrics on frame ---
                y_off = 30

                # Elapsed time (top-left)
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                cv2.putText(frame, f"TIME: {elapsed_min:02d}:{elapsed_sec:02d}", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                y_off += 28

                # Tension bar (red)
                t_bar = int(min(tension, 1.0) * 200)
                cv2.putText(frame, f"TENSION: {tension:.2f}", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (180, y_off - 12), (180 + t_bar, y_off),
                              (0, 0, 255), -1)
                y_off += 28

                # Valence bar (green=positive, red=negative)
                v_color = (0, 200, 0) if face_valence >= 0 else (0, 0, 255)
                v_label = f"VALENCE: {val_label}{face_valence:.2f}"
                cv2.putText(frame, v_label, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, v_color, 1)
                v_bar = int(abs(face_valence) * 150)
                cv2.rectangle(frame, (180, y_off - 10), (180 + v_bar, y_off),
                              v_color, -1)
                y_off += 25

                # Smile bar (cyan)
                s_bar = int(min(smile_level, 1.0) * 150)
                cv2.putText(frame, f"Smile:  {smile_level:.2f}", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (180, y_off - 10), (180 + s_bar, y_off),
                              (255, 255, 0), -1)
                y_off += 25

                # Context tag (what state was detected)
                ctx_colors = {
                    "JOY": (0, 200, 0), "FEAR": (0, 0, 255),
                    "CONC": (200, 200, 0), "SAD": (200, 100, 0),
                    "STRESS": (0, 50, 255), "---": (150, 150, 150),
                }
                ctx_col = ctx_colors.get(ctx_tag, (150, 150, 150))
                cv2.putText(frame, f"State: [{ctx_tag}]", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ctx_col, 2)
                y_off += 28

                # Velocity tag (startle detection — separate from ctx_tag)
                vel_col = (0, 100, 255) if velocity_tag == "STARTLE" else (100, 100, 100)
                cv2.putText(frame, f"Startle: [{velocity_tag}] {startle_score:.1f}/s",
                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vel_col, 1)
                y_off += 22

                # --- Blendshape detail bars (color-coded by group) ---
                cv2.putText(frame, "--- Blendshapes (neg/ambig/pos) ---", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y_off += 16
                neg_set = set(NEGATIVE_BLENDSHAPES)
                pos_set = set(POSITIVE_BLENDSHAPES)
                for bs_name in STRESS_BLENDSHAPES:
                    val = bs_dict.get(bs_name, 0)
                    bar_len = int(min(val, 1.0) * 120)
                    # Color by category
                    if bs_name in neg_set:
                        col = (0, 0, 200) if val > 0.15 else (80, 80, 120)  # red
                    elif bs_name in pos_set:
                        col = (0, 200, 0) if val > 0.15 else (80, 120, 80)  # green
                    else:  # ambiguous
                        col = (0, 200, 200) if val > 0.15 else (80, 120, 120)  # yellow
                    cv2.putText(frame, f"{bs_name[:14]:14s} {val:.2f}", (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
                    cv2.rectangle(frame, (150, y_off - 7), (150 + bar_len, y_off),
                                  col, -1)
                    y_off += 13

                # Latency top-right
                cv2.putText(frame, f"{latency_ms:.0f}ms", (frame.shape[1] - 80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                session.log_frame(face_detected=False)
                print(f"\rFrame {frame_count:4d} | No face | {latency_ms:.1f}ms", end="")

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            print(f"\rFrame {frame_count:4d} | Error: {str(e)[:60]} | {latency_ms:.1f}ms", end="")

        if not args.no_display:
            cv2.imshow("MediaPipe - Blendshapes & Tension", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    csv_file.close()

    if not session.post_session_confirm(csv_path):
        return

    csv_path, summary_path = session.finalize_session_files(csv_path)

    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "", "=" * 60,
            "PERFORMANCE SUMMARY — MediaPipe FaceLandmarker (Blendshapes)",
            "=" * 60,
            f"Session date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session duration:       {record['duration_s']:.1f} seconds",
            f"Total frames processed: {frame_count}",
            f"Average latency:        {record['avg_latency_ms']:.1f} ms",
            f"Median latency:         {record['median_latency_ms']:.1f} ms",
            f"95th percentile:        {record['p95_latency_ms']:.1f} ms",
            f"Min latency:            {np.min(latencies):.1f} ms",
            f"Max latency:            {record['max_latency_ms']:.1f} ms",
            f"Effective FPS:          {record['fps_actual']:.1f}",
            "=" * 60,
            f"CSV log:  {csv_path}",
            f"Summary:  {summary_path}",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"\nDone! {frame_count} frames logged.")

        # Auto-run post-session analysis (visualization + stats)
        post_session_analysis(csv_path, summary_path, run_viz=True, run_emotion=False)


if __name__ == "__main__":
    main()
