"""HS-only FER pipeline — La Façade Fissuréе.

Records hs_fear, hs_arousal, and all 8 emotion scores per frame for offline evaluation.

Face detection:
    Haar cascade      — primary for HSEmotion crop (fast frontal-face detection)
    MP FaceLandmarker — runs simultaneously as fallback; provides bbox when Haar misses

Evaluation after recording:
    python Pipeline/fer/compare_ground_truth_v2.py \\
        --gt-csv Annotations/<session>.csv \\
        --model-csv Pipeline/logs/sessions/<session>/model.csv \\
        --signal-col hs_fear

Usage:
    python Pipeline/fer/test_hsemotion.py --video path/to/video.mp4
    Press 'q' to quit. Results saved to Pipeline/logs/<session>/
"""

# ── GPU CONFIGURATION — set before onnxruntime loads ─────────────────────────
USE_GPU = False
GPU_DEVICE_INDEX = 0
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

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
    EMOTION_LABELS,
    HUD_MIN_HEIGHT,
)
from fer.face_detector import MODEL_PATH, ensure_model
from fer.hud import draw_on_video_bars
from fer.two_gate_detector import TwoGateDetector
from utils.session_meta import Session
from utils.post_session_analysis import post_session_analysis

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="HS-only FER pipeline — Haar-first + MP-fallback face detection"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. Omit for webcam.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip live video display (headless mode).")
    # ── Two-gate detector tuning ──
    parser.add_argument("--onset-threshold", type=float, default=0.015,
                        dest="onset_threshold")
    parser.add_argument("--onset-window", type=int, default=10,
                        dest="onset_window")
    parser.add_argument("--sustain-frames", type=int, default=15,
                        dest="sustain_frames")
    parser.add_argument("--floor-threshold", type=float, default=0.30,
                        dest="floor_threshold")
    parser.add_argument("--cooldown-frames", type=int, default=30,
                        dest="cooldown_frames")
    return parser


def _init_models():
    """Load MP FaceLandmarker (bbox only), Haar cascade, and HSEmotion recognizer."""
    print("Loading MediaPipe FaceLandmarker (bbox mode)...")
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe loaded.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print("Haar cascade loaded.")

    hs_model_name = "enet_b0_8_va_mtl"
    print(f"Loading HSEmotion model ({hs_model_name})...")
    recognizer = HSEmotionRecognizer(model_name=hs_model_name)
    print(f"HSEmotion loaded.")
    return landmarker, face_cascade, recognizer


def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()

    ensure_model()

    log_dir = os.path.join(PIPELINE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    session = Session("hs_only", log_dir)
    session.pre_session_prompt()

    csv_path = os.path.join(log_dir, f"{session.session_id}_hs_only_temp.csv")
    summary_path = None

    session.device_used = "cpu"
    session.video_source = args.video if args.video else "webcam"

    landmarker, face_cascade, recognizer = _init_models()

    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        src = f"file '{args.video}'" if args.video else "webcam"
        print(f"ERROR: Could not open {src}.")
        return

    if args.video:
        _fps = cap.get(cv2.CAP_PROP_FPS)
        video_fps = _fps if _fps and _fps > 0 else 30.0
        print(f"Video mode: {args.video} @ {video_fps:.1f} fps")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_fps = 30.0
        print("Live mode: webcam @ 30.0 fps (target)")

    detector = TwoGateDetector(
        onset_threshold=args.onset_threshold,
        onset_window=args.onset_window,
        sustain_frames=args.sustain_frames,
        floor_threshold=args.floor_threshold,
        cooldown_frames=args.cooldown_frames,
        fps=video_fps,
    )
    _smoothing_buf = deque(maxlen=50)

    print("\n" + "=" * 60)
    print("HS-only Pipeline — face detection: Haar-first, MP-fallback")
    print(f"Detector: onset≥{args.onset_threshold} floor≥{args.floor_threshold} "
          f"sustain={args.sustain_frames}fr cooldown={args.cooldown_frames}fr")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print("Press 'q' to quit\n")

    session.start()

    # ── CSV header ───────────────────────────────────────────────────────────
    hs_emotion_cols = ["hs_" + l.lower() for l in EMOTION_LABELS]

    csv_header = (
        ["frame", "timestamp", "hs_crop_source"]
        + ["hs_fear", "hs_arousal", "hs_valence", "hs_dominant", "hs_dominant_score"]
        + hs_emotion_cols
        + ["mp_face_detected", "hs_face_detected"]
        + ["smoothed_hs_fear", "onset_slope", "event_status"]
        + ["mp_face_bbox_area", "latency_ms", "cpu_percent", "ram_mb"]
    )

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    print("\nNote: video rendering is done after analysis (two-phase pipeline).\n")

    try:
      while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed = (frame_count / video_fps) if video_fps else (
            time.perf_counter() - session_start)
        start_time = time.perf_counter()

        # ── Default values ───────────────────────────────────────────────
        mp_face_detected = False
        hs_face_detected = False
        mp_face_bbox = None
        mp_face_bbox_area = 0
        hs_crop_source = "none"
        hs_fear = 0.0
        hs_arousal = 0.0
        hs_valence = 0.0
        emotions_hs = {l: 0.0 for l in EMOTION_LABELS}
        dominant_hs = "Neutral"
        dominant_score_hs = 0.0
        event_status = "IDLE"
        onset_slope = 0.0
        smoothed_hs_fear = 0.0

        try:
            # ── MediaPipe FaceLandmarker — bbox only ─────────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_count * 1000 / video_fps) if video_fps else int(
                (time.perf_counter() - session_start) * 1000
            )
            mp_results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if mp_results.face_landmarks and len(mp_results.face_landmarks) > 0:
                mp_face_detected = True
                fh_px, fw_px = frame.shape[:2]
                lms = mp_results.face_landmarks[0]
                xs = [lm.x * fw_px for lm in lms]
                ys = [lm.y * fh_px for lm in lms]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                mp_face_bbox = (x1, y1, x2, y2)
                mp_face_bbox_area = (x2 - x1) * (y2 - y1)

            # ── Haar-first, MP-fallback — HSEmotion crop ─────────────────
            face_img = None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

            if len(haar_faces) > 0:
                fx, fy, fw_f, fh_f = max(haar_faces, key=lambda f: f[2] * f[3])
                face_img = frame[fy:fy + fh_f, fx:fx + fw_f]
                hs_crop_source = "haar"
            elif mp_face_detected and mp_face_bbox:
                bx1, by1, bx2, by2 = mp_face_bbox
                bw, bh = bx2 - bx1, by2 - by1
                pad = int(0.15 * max(bw, bh))
                face_img = frame[
                    max(0, by1 - pad):min(frame.shape[0], by2 + pad),
                    max(0, bx1 - pad):min(frame.shape[1], bx2 + pad),
                ]
                hs_crop_source = "mp"

            # ── HSEmotion inference ───────────────────────────────────────
            if face_img is not None and face_img.size > 0:
                hs_face_detected = True
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)

                hs_valence = float(scores[-2])
                hs_arousal = float(scores[-1])
                for i, label in enumerate(EMOTION_LABELS):
                    if i < len(scores) - 2:
                        emotions_hs[label] = float(scores[i])

                dominant_hs = emotion if emotion else max(emotions_hs, key=emotions_hs.get)
                dominant_score_hs = emotions_hs.get(dominant_hs, 0.0)
                hs_fear = emotions_hs.get("Fear", 0.0)

            # ── Smoothing and detector ───────────────────────────────────
            _smoothing_buf.append(hs_fear)
            smoothed_hs_fear = sum(_smoothing_buf) / len(_smoothing_buf)

            hs_surprise = emotions_hs.get("Surprise", 0.0)
            raw_vals = {"hs_fear": hs_fear, "hs_surprise": hs_surprise, "mp_tension": 0.0}
            event_status = detector.update(frame_count, smoothed_hs_fear, raw_vals)
            onset_slope = detector.current_onset_slope

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            print(f"\rFrame {frame_count:4d} | Error: {str(e)[:60]} | {latency_ms:.1f}ms",
                  end="")

        # ── Session resource logging ────────────────────────────────────
        meta = session.log_frame(
            face_detected=(mp_face_detected or hs_face_detected),
            face_confidence=1.0 if hs_face_detected else 0.0,
            face_bbox_area=mp_face_bbox_area,
        )

        # ── CSV row ─────────────────────────────────────────────────────
        hs_emotion_values = [f"{emotions_hs.get(l, 0):.4f}" for l in EMOTION_LABELS]

        csv_writer.writerow(
            [frame_count, f"{elapsed:.3f}", hs_crop_source]
            + [f"{hs_fear:.4f}", f"{hs_arousal:.4f}", f"{hs_valence:.4f}",
               dominant_hs, f"{dominant_score_hs:.4f}"]
            + hs_emotion_values
            + [int(mp_face_detected), int(hs_face_detected)]
            + [f"{smoothed_hs_fear:.4f}", f"{onset_slope:.6f}", event_status]
            + [mp_face_bbox_area, f"{latency_ms:.1f}", meta["cpu_percent"], meta["ram_mb"]]
        )

        # ── Console print ────────────────────────────────────────────────
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        status_short = event_status[:4] if event_status != "IDLE" else "----"
        crop_ind = hs_crop_source[0].upper() if hs_crop_source != "none" else "-"
        print(f"\rFrame {frame_count:4d} | "
              f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
              f"Lat: {latency_ms:5.1f}ms | "
              f"HS_F:{hs_fear:.2f} HS_A:{hs_arousal:.2f} "
              f"[{dominant_hs[:5]:5s}]{crop_ind} | "
              f"Sm:{smoothed_hs_fear:.2f} [{status_short}]", end="")

        # ── Live display ─────────────────────────────────────────────────
        if not args.no_display:
            _fh, _fw = frame.shape[:2]
            _scale = min(HUD_MIN_HEIGHT / _fh, 1.0)
            _dw, _dh = int(_fw * _scale), int(_fh * _scale)
            _disp = cv2.resize(frame, (_dw, _dh))

            if mp_face_bbox:
                x1, y1, x2, y2 = mp_face_bbox
                sx1, sy1 = int(x1 * _scale), int(y1 * _scale)
                sx2, sy2 = int(x2 * _scale), int(y2 * _scale)
                _color = ((0, 0, 220) if event_status == "EVENT_CONFIRMED"
                          else (0, 200, 255) if event_status in ("ONSET", "SUSTAINING")
                          else (0, 200, 0))
                cv2.rectangle(_disp, (sx1, sy1), (sx2, sy2), _color, 2)

            draw_on_video_bars(_disp, hs_fear, 0.0, smoothed_hs_fear)
            cv2.imshow("HS-only Analysis", _disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        csv_file.close()

    detector.flush(frame_count)

    if not session.post_session_confirm(csv_path):
        return

    csv_path, summary_path = session.finalize_session_files(csv_path)

    events_csv_path = csv_path.replace(".csv", "_events.csv")
    detector.write_events_csv(events_csv_path)

    if latencies:
        record, extra_lines = session.finish(
            latencies, frame_count, csv_path, summary_path,
        )

        summary_lines = [
            "", "=" * 60,
            "PERFORMANCE SUMMARY — HS-only Pipeline",
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
            f"CSV (full):  {csv_path}",
            f"Events CSV:  {events_csv_path}",
            f"Summary:     {summary_path}",
            f"Events found: {len(detector.events)}",
            "=" * 60,
            "",
            "To evaluate:",
            f"  python Pipeline/fer/compare_ground_truth_v2.py \\",
            f"      --model-csv {csv_path}",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))
        print(f"\nDone! {frame_count} frames logged.")

        post_session_analysis(csv_path, summary_path, run_viz=True, run_emotion=False)


if __name__ == "__main__":
    main()
