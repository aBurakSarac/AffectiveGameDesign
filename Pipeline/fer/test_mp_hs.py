"""MP+HS Dual-Tool FER Pipeline — La Façade Fissuréе.

Pattern: [Facade]          — orchestrates face detection, FER, smoothing,
                             event detection, session logging, and live sender.
Pattern: [Strategy]        — MediaPipe blendshapes and HSEmotion valence-arousal
                             run as two independent FER strategies; outputs fused.
Pattern: [Pipeline/Filter] — frame → FaceDetector → fusion → TwoGateDetector
                             → SessionLogger → LivePipelineSender.

Parameters (edit at top of file):
    USE_GPU            bool   Use CUDA for ONNX inference (default False — CPU is stable)
    GPU_DEVICE_INDEX   int    GPU index when USE_GPU=True (default 0)

Usage:
    python Pipeline/fer/test_mp_hs.py --video path/to/video.mp4
    python Pipeline/fer/test_mp_hs.py --video path/to/video.mp4 --mode fusion
    python Pipeline/fer/test_mp_hs.py --video path/to/video.mp4 --mode independent

    Press 'q' to quit. Results saved to Pipeline/logs/<session>/
"""

# ── GPU CONFIGURATION — set before onnxruntime loads ─────────────────────────
USE_GPU = False
GPU_DEVICE_INDEX = 0
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

# Add Pipeline/ to path so all subpackage imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort

_cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
if USE_GPU and _cuda_available:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_INDEX)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import time
import csv
import argparse
import numpy as np
from collections import deque
from datetime import datetime
import mediapipe as mp
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

from fer.blendshapes import (
    NEGATIVE_BLENDSHAPES, POSITIVE_BLENDSHAPES, STRESS_BLENDSHAPES,
    ALL_BLENDSHAPE_NAMES, KEY_BLENDSHAPES,
    VELOCITY_AUS, STARTLE_VELOCITY_THRESHOLD,
    EMOTION_LABELS, PANEL_WIDTH, HUD_MIN_HEIGHT, NEUTRAL_TENSION_MAX,
)
from fer.face_detector import (
    MODEL_PATH, ensure_model,
    compute_tension, compute_face_valence, compute_au_velocities, compute_ctx_tag,
)
from fer.fusion import compute_all_formulas, compute_composite_fear
from fer.hud import draw_hud, run_mp_hs_comparison, draw_on_video_bars
from fer.video_output import render_annotated_video
from fer.two_gate_detector import TwoGateDetector
from fer.improved_fear_detection import get_velocity_tag
from utils.session_meta import Session
from utils.post_session_analysis import post_session_analysis

# Pipeline root — logs/ and models/ live here
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_arg_parser():
    parser = argparse.ArgumentParser(description="MP+HS dual-tool fusion test")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. Omit to use webcam.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip live video display (required for headless runs).")
    parser.add_argument("--mode", choices=["independent", "fusion", "both"],
                        default="both", help="Analysis mode (default: both)")
    parser.add_argument("--all-blendshapes", action="store_true",
                        help="Log all 52 MP blendshapes (default: key subset)")
    # ── Two-gate detector tuning ──
    parser.add_argument("--formula",
                        choices=["f0", "f1", "f2", "f3", "f4", "f5", "f6",
                                 "f7", "f8", "f9", "f10", "f11", "f12", "f13"],
                        default="f12",
                        help="Formula to feed into rolling average and detector (default: f12)")
    parser.add_argument("--onset-threshold", type=float, default=0.015,
                        help="Gate A derivative threshold per frame (default: 0.015)")
    parser.add_argument("--onset-window", type=int, default=10,
                        help="Gate A derivative window in frames (default: 10)")
    parser.add_argument("--sustain-frames", type=int, default=15,
                        help="Gate B minimum sustained frames (default: 15)")
    parser.add_argument("--floor-threshold", type=float, default=0.30,
                        help="Gate B floor threshold (default: 0.30)")
    parser.add_argument("--cooldown-frames", type=int, default=30,
                        help="Refractory frames after event ends (default: 30)")
    # ── Render-only mode ──
    parser.add_argument("--render", action="store_true",
                        help="Skip analysis; render annotated video from an existing CSV.")
    parser.add_argument("--render-csv", type=str, default=None,
                        help="Path to compact CSV (used with --render).")
    parser.add_argument("--render-out", type=str, default=None,
                        help="Output video path (used with --render). "
                             "Defaults to <csv_stem>_annotated.mp4.")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Disable CLAHE preprocessing on face ROI "
                             "(CLAHE is on by default for low-light robustness).")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames)")
    return parser


def _init_models(args):
    """Load MediaPipe FaceLandmarker and HSEmotion recognizer. Returns (landmarker, recognizer, face_cascade)."""
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
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)
    print("MediaPipe loaded.")

    print("Loading HSEmotion model...")
    hs_model_name = "enet_b0_8_va_mtl"
    recognizer = HSEmotionRecognizer(model_name=hs_model_name)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print(f"HSEmotion loaded: {hs_model_name}")
    return landmarker, recognizer, face_cascade


def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()

    # ── Render-only shortcut ──────────────────────────────────────────────
    if args.render:
        if not args.render_csv or not args.video:
            print("ERROR: --render requires both --video and --render-csv.")
            return
        out_path = args.render_out or os.path.splitext(args.render_csv)[0] + "_annotated.mp4"
        render_annotated_video(args.video, args.render_csv, out_path, args.mode)
        return

    ensure_model()

    log_dir = os.path.join(PIPELINE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Session metadata
    session = Session("mp_hs", log_dir)
    session.pre_session_prompt()

    csv_path = os.path.join(log_dir, f"{session.session_id}_mp_hs_temp.csv")
    csv_compact_path = os.path.join(log_dir, f"{session.session_id}_mp_hs_compact_temp.csv")
    summary_path = None

    session.device_used = "cpu"
    session.video_source = args.video if args.video else "webcam"

    landmarker, recognizer, face_cascade = _init_models(args)

    # ── Open video / webcam ──────────────────────────────────────────────
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
        video_fps = 30.0
        print("Live mode: webcam @ 30.0 fps (target)")

    fps_for_detector = video_fps

    # ── Two-gate detector and rolling average ────────────────────────────
    detector = TwoGateDetector(
        onset_threshold=args.onset_threshold,
        onset_window=args.onset_window,
        sustain_frames=args.sustain_frames,
        floor_threshold=args.floor_threshold,
        cooldown_frames=args.cooldown_frames,
        fps=fps_for_detector,
    )
    _smoothing_buf = deque(maxlen=50)

    print("\n" + "=" * 60)
    print(f"MP+HS Dual-Tool Fusion — Mode: {args.mode.upper()}")
    print(f"Formula: {args.formula.upper()} | Detector: onset≥{args.onset_threshold} "
          f"floor≥{args.floor_threshold} sustain={args.sustain_frames}fr "
          f"cooldown={args.cooldown_frames}fr")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print(f"           {csv_compact_path}")
    print("Press 'q' to quit\n")

    session.start()

    # ── CSV header ───────────────────────────────────────────────────────
    if args.all_blendshapes:
        bs_columns = ["mp_" + n for n in ALL_BLENDSHAPE_NAMES]
    else:
        bs_columns = ["mp_" + n for n in KEY_BLENDSHAPES]

    velocity_cols = (
        ["mp_vel_" + au for au in VELOCITY_AUS]
        + ["mp_tension_velocity"]
    )
    hs_emotion_cols = ["hs_" + l.lower() for l in EMOTION_LABELS]

    # ── COMPACT CSV HEADER ──────────────────────────────────────────────
    compact_header = (
        ["frame", "timestamp", "hs_crop_source", "composite_fear"]
        + ["hs_dominant"]
        + ["mp_ctx_tag", "mp_velocity_tag", "agreement_tag", "veto_tag"]
        + ["hs_fear", "hs_arousal", "hs_dominant_score"]
        + ["mp_tension", "mp_startle_score"]
        + ["smoothed_composite", "onset_slope", "event_status", "face_bbox"]
        + ["mp_valence", "mp_smile"]
    )

    # ── FULL CSV HEADER ─────────────────────────────────────────────────
    csv_header = (
        ["frame", "timestamp", "hs_crop_source", "composite_fear"]
        + ["hs_dominant"]
        + ["mp_ctx_tag", "mp_velocity_tag", "agreement_tag", "veto_tag"]
        + ["hs_fear", "hs_arousal", "hs_dominant_score"]
        + ["mp_tension", "mp_startle_score"]
        + ["f0", "f1", "f2", "f3", "f4", "f5", "f6",
           "f7", "f8", "f9", "f10", "f11", "f12", "f13"]
        + ["smoothed_composite", "onset_slope", "event_status", "is_neutral_frame"]
        + ["mp_face_valence", "mp_smile_level"]
        + bs_columns + velocity_cols
        + hs_emotion_cols
        + ["mp_face_detected", "hs_face_detected"]
        + ["latency_ms", "cpu_percent", "ram_mb"]
    )

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    csv_compact_file = open(csv_compact_path, "w", newline="")
    csv_compact_writer = csv.writer(csv_compact_file)
    csv_compact_writer.writerow(compact_header)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()
    prev_bs_dict = None
    prev_elapsed = None
    prev_tension = 0.0
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if not args.no_clahe else None

    # Pre-allocate display buffers
    _hud_scale = _hud_vid_w = _canvas = _resized = None
    if not args.no_display:
        ret_peek, frame_peek = cap.read()
        if ret_peek:
            _fh, _fw = frame_peek.shape[:2]
            _hud_scale = HUD_MIN_HEIGHT / _fh
            _hud_vid_w = int(_fw * _hud_scale)
            _canvas = np.zeros((HUD_MIN_HEIGHT, _hud_vid_w + PANEL_WIDTH, 3), dtype=np.uint8)
            _resized = np.zeros((HUD_MIN_HEIGHT, _hud_vid_w, 3), dtype=np.uint8)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\nNote: video rendering is done after analysis (two-phase pipeline).\n")

    # ── Frame loop — analysis only, no video writer ───────────────────────
    try:
      while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if args.skip_frames > 1 and frame_count % args.skip_frames != 0:
            continue
        elapsed = (frame_count / video_fps) if video_fps else (time.perf_counter() - session_start)
        start_time = time.perf_counter()

        # Default values (used if try block fails before assignment)
        mp_face_detected = False
        hs_face_detected = False
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
        hs_crop_source = "none"
        composite_fear = 0.0
        agreement_tag = ""
        veto_tag = "---"
        formulas = {"F0": 0.0, "F1": 0.0, "F2": 0.0, "F3": 0.0,
                    "F4": 0.0, "F5": 0.0, "F6": 0.0, "F7": 0.0,
                    "F8": 0.0, "F9": 0.0, "F10": 0.0, "F11": 0.0,
                    "F12": 0.0, "F13": 0.0}
        smoothed_composite = 0.0
        is_neutral_frame = 0
        event_status = "IDLE"
        onset_slope = 0.0
        hs_fear = 0.0   # initialized here so live display can use it even on exception frames

        try:
            # ── MediaPipe inference ──────────────────────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_count * 1000 / video_fps) if video_fps else int(
                (time.perf_counter() - session_start) * 1000
            )

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

                # Face bbox from landmarks
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
                hs_crop_source = "haar"
            elif mp_face_detected and mp_face_bbox:
                x1, y1, x2, y2 = mp_face_bbox
                w, h = x2 - x1, y2 - y1
                pad = int(0.15 * max(w, h))
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(frame.shape[1], x2 + pad)
                y2p = min(frame.shape[0], y2 + pad)
                face_img = frame[y1p:y2p, x1p:x2p]
                hs_crop_source = "mp"

            if face_img is not None:
                hs_face_detected = True
                if _clahe is not None:
                    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
                    face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)

                valence_hs = float(scores[-2])
                arousal_hs = float(scores[-1])
                for i, label in enumerate(EMOTION_LABELS):
                    if i < len(scores) - 2:
                        emotions_hs[label] = float(scores[i])

                dominant_hs = emotion if emotion else max(emotions_hs, key=emotions_hs.get)
                dominant_score_hs = emotions_hs.get(dominant_hs, 0)

            # ── Fusion ───────────────────────────────────────────────
            hs_fear     = emotions_hs.get("Fear", 0.0)
            if mp_face_detected and hs_face_detected and args.mode != "independent":
                composite_fear = compute_composite_fear(hs_fear, arousal_hs, tension)
                # agreement_tag, veto_tag = compute_agreement(
                #     ctx_tag, tension, dominant_hs, arousal_hs, emotions_hs,
                # )

            # ── All formulas ─────────────────────────────────────────
            hs_surprise = emotions_hs.get("Surprise", 0.0)
            hs_anger    = emotions_hs.get("Anger", 0.0)
            formulas = compute_all_formulas(
                hs_fear, hs_surprise, arousal_hs, hs_anger,
                tension, startle_score,
            )
            selected_composite = formulas[args.formula.upper()]

            # ── 50-frame rolling average ──────────────────────────────
            _smoothing_buf.append(selected_composite)
            smoothed_composite = sum(_smoothing_buf) / len(_smoothing_buf)

            # ── is_neutral_frame flag ─────────────────────────────────
            is_neutral_frame = int(
                (dominant_hs == "Neutral") and (tension <= NEUTRAL_TENSION_MAX)
            )

            # ── Two-gate detector ─────────────────────────────────────
            raw_vals = {"hs_fear": hs_fear, "hs_surprise": hs_surprise,
                        "mp_tension": tension}
            event_status = detector.update(frame_count, smoothed_composite, raw_vals)
            onset_slope  = detector.current_onset_slope

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            print(f"\rFrame {frame_count:4d} | Error: {str(e)[:60]} | {latency_ms:.1f}ms",
                  end="")

        # ── Log session resources ────────────────────────────────────
        meta = session.log_frame(
            face_detected=(mp_face_detected or hs_face_detected),
            face_confidence=1.0 if mp_face_detected else 0.0,
            face_bbox_area=0,
        )

        # ── CSV row construction ─────────────────────────────────
        if args.all_blendshapes:
            bs_values = [f"{bs_dict.get(n, 0):.4f}" for n in ALL_BLENDSHAPE_NAMES]
        else:
            bs_values = [f"{bs_dict.get(n, 0):.4f}" for n in KEY_BLENDSHAPES]

        au_velocity_values = [f"{au_velocities.get(au, 0.0):.4f}" for au in VELOCITY_AUS]
        tension_velocity_value = f"{tension_velocity:.4f}"
        hs_emotion_values = [f"{emotions_hs.get(l, 0):.4f}" for l in EMOTION_LABELS]

        core_row = (
            [frame_count, f"{elapsed:.3f}", hs_crop_source, f"{composite_fear:.4f}"]
            + [dominant_hs]
            + [ctx_tag, velocity_tag, agreement_tag, veto_tag]
            + [f"{emotions_hs.get('Fear', 0):.4f}", f"{arousal_hs:.4f}", f"{dominant_score_hs:.4f}"]
            + [f"{tension:.4f}", f"{startle_score:.4f}"]
        )

        bbox_str = ",".join(map(str, mp_face_bbox)) if mp_face_bbox else ""

        # Compact CSV
        csv_compact_writer.writerow(
            core_row + [
                f"{smoothed_composite:.4f}",
                f"{onset_slope:.6f}",
                event_status,
                bbox_str,
                f"{face_valence:.4f}",
                f"{smile_level:.4f}",
            ]
        )

        # Full CSV
        full_row = core_row + (
            [f"{formulas['F0']:.4f}", f"{formulas['F1']:.4f}", f"{formulas['F2']:.4f}",
             f"{formulas['F3']:.4f}", f"{formulas['F4']:.4f}", f"{formulas['F5']:.4f}",
             f"{formulas['F6']:.4f}", f"{formulas['F7']:.4f}", f"{formulas['F8']:.4f}",
             f"{formulas['F9']:.4f}", f"{formulas['F10']:.4f}", f"{formulas['F11']:.4f}",
             f"{formulas['F12']:.4f}", f"{formulas['F13']:.4f}"]
            + [f"{smoothed_composite:.4f}", f"{onset_slope:.6f}", event_status,
               is_neutral_frame]
            + [f"{face_valence:.4f}", f"{smile_level:.4f}"]
            + bs_values
            + au_velocity_values + [tension_velocity_value]
            + hs_emotion_values
            + [int(mp_face_detected), int(hs_face_detected)]
            + [f"{latency_ms:.1f}", meta["cpu_percent"], meta["ram_mb"]]
        )
        csv_writer.writerow(full_row)

        # ── Console print ────────────────────────────────────────────
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        status_short = event_status[:4] if event_status != "IDLE" else "----"
        print(f"\rFrame {frame_count:4d} | "
              f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
              f"Lat: {latency_ms:5.1f}ms | "
              f"f12: {composite_fear:.2f} (fear={hs_fear:.2f} ar={arousal_hs:.2f} t={tension:.2f}) | "
              f"Sm: {smoothed_composite:.2f} [{status_short}] "
              f"ev={len(detector.events)}", end="")

        # ── Live display (optional) ───────────────────────────────────────
        if not args.no_display and _canvas is not None:
            _canvas.fill(0)
            cv2.resize(frame, (_hud_vid_w, HUD_MIN_HEIGHT), dst=_resized)
            _canvas[0:HUD_MIN_HEIGHT, 0:_hud_vid_w] = _resized

            _mp_data = {
                "face_detected": mp_face_detected, "tension": tension,
                "face_valence": face_valence, "smile_level": smile_level,
                "ctx_tag": ctx_tag, "startle_score": startle_score,
                "velocity_tag": velocity_tag, "bs_dict": bs_dict,
            }
            _hs_data = {
                "face_detected": hs_face_detected, "arousal": arousal_hs,
                "valence": valence_hs, "dominant": dominant_hs,
                "dominant_score": dominant_score_hs, "emotions": emotions_hs,
                "crop_source": hs_crop_source,
            }

            if mp_face_bbox:
                x1, y1, x2, y2 = [int(c * _hud_scale) for c in mp_face_bbox]
                _color = (0, 200, 0)
                if event_status == "EVENT_CONFIRMED":
                    _color = (0, 0, 220)
                elif event_status in ("ONSET", "SUSTAINING"):
                    _color = (0, 200, 255)
                cv2.rectangle(_canvas, (x1, y1), (x2, y2), _color, 2)

            draw_on_video_bars(_canvas[0:HUD_MIN_HEIGHT, 0:_hud_vid_w], hs_fear, tension, smoothed_composite)

            # Timer, frame counter, latency overlay on video area
            _font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(_canvas, f"TIME: {elapsed_min:02d}:{elapsed_sec:02d}",
                        (10, 25), _font, 0.6, (200, 200, 200), 2)
            cv2.putText(_canvas, f"Frame {frame_count:04d}",
                        (10, HUD_MIN_HEIGHT - 10), _font, 0.4, (150, 150, 150), 1)
            cv2.putText(_canvas, f"{latency_ms:.0f}ms",
                        (_hud_vid_w - 80, 25), _font, 0.6, (0, 255, 255), 2)

            _fusion_data = None
            if args.mode != "independent":
                _fusion_data = {"composite_fear": composite_fear}
            _canvas = draw_hud(_canvas, _hud_vid_w, HUD_MIN_HEIGHT,
                               _mp_data, _hs_data, _fusion_data, args.mode)
            if event_status not in ("IDLE", ""):
                _badge_colors = {"ONSET": (0, 200, 255), "SUSTAINING": (0, 130, 255),
                                 "EVENT_CONFIRMED": (0, 0, 220), "EVENT_ENDED": (30, 180, 30)}
                _bc = _badge_colors.get(event_status, (150, 150, 150))
                (_tw, _th), _ = cv2.getTextSize(event_status, _font, 0.50, 2)
                _bpad, _bx2 = 5, _hud_vid_w - 8
                _bx1 = _bx2 - _tw - _bpad * 2
                cv2.rectangle(_canvas, (_bx1, 45), (_bx2, 45 + _th + _bpad * 2), _bc, -1)
                cv2.putText(_canvas, event_status, (_bx1 + _bpad, 45 + _th + _bpad - 2),
                            _font, 0.50, (255, 255, 255), 2)
            cv2.imshow("MP+HS Analysis", _canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        csv_file.close()
        csv_compact_file.close()

    # Flush detector: save any mid-video event that satisfied Gate B
    detector.flush(frame_count)

    if not session.post_session_confirm(csv_path):
        return

    csv_path, summary_path = session.finalize_session_files(csv_path)
    # Compact CSV: derive name from final full CSV path
    final_compact = csv_path.replace(".csv", "_compact.csv")
    if os.path.isfile(csv_compact_path):
        os.rename(csv_compact_path, final_compact)
    csv_compact_path = final_compact

    annotated_video_path = None

    # Write events CSV
    events_csv_path = csv_path.replace(".csv", "_events.csv")
    detector.write_events_csv(events_csv_path)

    if latencies:
        record, extra_lines = session.finish(
            latencies, frame_count, csv_path, summary_path,
        )

        summary_lines = [
            "", "=" * 60,
            f"PERFORMANCE SUMMARY — MP+HS Dual-Tool (Mode: {args.mode})",
            "=" * 60,
            f"Session date:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session duration:       {record['duration_s']:.1f} seconds",
            f"Total frames processed: {frame_count}",
            f"Average latency:        {record['avg_latency_ms']:.1f} ms (both tools)",
            f"Median latency:         {record['median_latency_ms']:.1f} ms",
            f"95th percentile:        {record['p95_latency_ms']:.1f} ms",
            f"Min latency:            {np.min(latencies):.1f} ms",
            f"Max latency:            {record['max_latency_ms']:.1f} ms",
            f"Effective FPS:          {record['fps_actual']:.1f}",
            "=" * 60,
            f"CSV (full):    {csv_path}",
            f"CSV (compact): {csv_compact_path}",
            f"Events CSV:    {events_csv_path}",
            f"Summary:       {summary_path}",
            f"Events found:  {len(detector.events)}",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"\nDone! {frame_count} frames logged.")

        gen_comparison = input("\nGenerate comparison analysis? [y/N]: ").strip().lower() == "y"
        gen_plot = input("Generate session plot? [y/N]: ").strip().lower() == "y"
        gen_render = False
        if args.video:
            gen_render = input("Render annotated video? [y/N]: ").strip().lower() == "y"

        if gen_comparison:
            run_mp_hs_comparison(csv_path)
        if gen_plot:
            post_session_analysis(csv_path, summary_path,
                                  run_viz=True, run_emotion=False)
        if gen_render:
            out_video = csv_path.replace(".csv", "_annotated.mp4")
            render_annotated_video(args.video, csv_compact_path, out_video, args.mode)
            print(f"Annotated video: {os.path.basename(out_video)}")


if __name__ == "__main__":
    main()
