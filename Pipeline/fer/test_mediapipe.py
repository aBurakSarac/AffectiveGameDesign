"""MP-only FER pipeline — La Façade Fissuréе.

Records all mp_tension variants (v3, v4, v5a–v5e) per frame so a single recording
can be evaluated against all formulas offline via compare_ground_truth_mp.py.

Face detection:
    MP FaceLandmarker — primary (blendshapes + bbox)
    Haar cascade      — fallback; marks face present when MP misses (no blendshapes)

Evaluation after recording:
    python Pipeline/fer/compare_ground_truth_mp.py \\
        --gt-csv Annotations/<session>.csv \\
        --model-csv Pipeline/logs/sessions/<session>/model.csv

Usage:
    python Pipeline/fer/test_mediapipe.py --video path/to/video.mp4
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

from fer.blendshapes import (
    ALL_BLENDSHAPE_NAMES, KEY_BLENDSHAPES,
    VELOCITY_AUS, STARTLE_VELOCITY_THRESHOLD,
    HUD_MIN_HEIGHT, PANEL_WIDTH,
)
from fer.face_detector import (
    MODEL_PATH, ensure_model,
    compute_tension, compute_tension_v4,
    compute_tension_v5a, compute_tension_v5b, compute_tension_v5c,
    compute_tension_v5d, compute_tension_v5e,
    compute_face_valence, compute_au_velocities, compute_ctx_tag,
    detect_scene_cut,
)
from fer.hud import draw_mp_variants_hud
from fer.two_gate_detector import TwoGateDetector
from fer.improved_fear_detection import get_velocity_tag
from utils.session_meta import Session
from utils.post_session_analysis import post_session_analysis

PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ordered list of tension variants — defines CSV column order
_TENSION_FNS = {
    "v3":  compute_tension,
    "v4":  compute_tension_v4,
    "v5a": compute_tension_v5a,
    "v5b": compute_tension_v5b,
    "v5c": compute_tension_v5c,
    "v5d": compute_tension_v5d,
    "v5e": compute_tension_v5e,
}
TENSION_VARIANTS = list(_TENSION_FNS.keys())


def _setup_arg_parser():
    parser = argparse.ArgumentParser(
        description="MP-only FER pipeline — records all tension variants per frame"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. Omit for webcam.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip live video display (headless mode).")
    parser.add_argument("--all-blendshapes", action="store_true",
                        help="Log all 52 MP blendshapes (default: key subset).")
    parser.add_argument("--detector-tension",
                        choices=TENSION_VARIANTS, default="v5c",
                        help="Tension variant that drives TwoGateDetector (default: v5c).")
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
    """Load MP FaceLandmarker and Haar cascade."""
    print("Loading MediaPipe FaceLandmarker...")
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
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe loaded.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    print("Haar cascade loaded.")
    return landmarker, face_cascade


def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()

    ensure_model()

    log_dir = os.path.join(PIPELINE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    session = Session("mp_only", log_dir)
    session.pre_session_prompt()

    csv_path = os.path.join(log_dir, f"{session.session_id}_mp_only_temp.csv")
    summary_path = None

    session.device_used = "cpu"
    session.video_source = args.video if args.video else "webcam"

    landmarker, face_cascade = _init_models()

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
    print(f"MP-only Pipeline — detector on {args.detector_tension.upper()}")
    print(f"Tension variants: {', '.join(TENSION_VARIANTS)}")
    print(f"Detector: onset≥{args.onset_threshold} floor≥{args.floor_threshold} "
          f"sustain={args.sustain_frames}fr cooldown={args.cooldown_frames}fr")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print("Press 'q' to quit\n")

    session.start()

    # ── CSV header ───────────────────────────────────────────────────────────
    if args.all_blendshapes:
        bs_columns = ["mp_" + n for n in ALL_BLENDSHAPE_NAMES]
    else:
        bs_columns = ["mp_" + n for n in KEY_BLENDSHAPES]

    velocity_cols = ["mp_vel_" + au for au in VELOCITY_AUS] + ["mp_tension_velocity"]
    tension_variant_cols = [f"mp_tension_{v}" for v in TENSION_VARIANTS]

    csv_header = (
        ["frame", "timestamp", "mp_face_detected", "haar_face_detected"]
        + ["mp_tension", "mp_startle_score", "mp_ctx_tag", "mp_velocity_tag"]
        + tension_variant_cols
        + ["smoothed_tension", "onset_slope", "event_status"]
        + ["mp_face_valence", "mp_smile_level"]
        + bs_columns + velocity_cols
        + ["mp_face_bbox_area", "latency_ms", "cpu_percent", "ram_mb"]
    )

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()
    prev_bs_dict = None
    prev_elapsed = None
    prev_tension = 0.0
    no_face_streak = 0

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
        haar_face_detected = False
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
        mp_face_bbox_area = 0
        tension_variants = {v: 0.0 for v in TENSION_VARIANTS}
        event_status = "IDLE"
        onset_slope = 0.0
        smoothed_tension = 0.0

        try:
            # ── MediaPipe inference ──────────────────────────────────────
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

                tension_variants = {k: fn(bs_dict) for k, fn in _TENSION_FNS.items()}
                tension = tension_variants["v4"]

                face_valence = compute_face_valence(bs_dict)
                smile_level = max(bs_dict.get("mouthSmileLeft", 0),
                                  bs_dict.get("mouthSmileRight", 0))

                delta_t = (elapsed - prev_elapsed) if prev_elapsed is not None else None
                au_velocities, startle_score = compute_au_velocities(
                    bs_dict, prev_bs_dict, delta_t)
                if prev_elapsed is not None and delta_t and delta_t > 0:
                    tension_velocity = max(0.0, (tension - prev_tension) / delta_t)

                velocity_tag = get_velocity_tag(startle_score, STARTLE_VELOCITY_THRESHOLD)
                ctx_tag = compute_ctx_tag(bs_dict, smile_level)

                if mp_results.face_landmarks and len(mp_results.face_landmarks) > 0:
                    fh_px, fw_px = frame.shape[:2]
                    lms = mp_results.face_landmarks[0]
                    xs = [lm.x * fw_px for lm in lms]
                    ys = [lm.y * fh_px for lm in lms]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    mp_face_bbox = (x1, y1, x2, y2)
                    mp_face_bbox_area = (x2 - x1) * (y2 - y1)

                # Scene-cut detection: clear smoother so stale clip values don't carry over
                if no_face_streak >= 3 or detect_scene_cut(prev_bs_dict, bs_dict):
                    _smoothing_buf.clear()
                no_face_streak = 0

                prev_bs_dict = bs_dict.copy()
                prev_elapsed = elapsed
                prev_tension = tension

            else:
                no_face_streak += 1
                # Haar fallback — face presence only, no blendshapes
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
                if len(faces) > 0:
                    haar_face_detected = True

            # ── Smoothing and detector ───────────────────────────────────
            selected_tension = tension_variants.get(args.detector_tension, tension)
            _smoothing_buf.append(selected_tension)
            smoothed_tension = sum(_smoothing_buf) / len(_smoothing_buf)

            raw_vals = {"hs_fear": 0.0, "hs_surprise": 0.0, "mp_tension": selected_tension}
            event_status = detector.update(frame_count, smoothed_tension, raw_vals)
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
            face_detected=(mp_face_detected or haar_face_detected),
            face_confidence=1.0 if mp_face_detected else 0.0,
            face_bbox_area=mp_face_bbox_area,
        )

        # ── CSV row ─────────────────────────────────────────────────────
        if args.all_blendshapes:
            bs_values = [f"{bs_dict.get(n, 0):.4f}" for n in ALL_BLENDSHAPE_NAMES]
        else:
            bs_values = [f"{bs_dict.get(n, 0):.4f}" for n in KEY_BLENDSHAPES]

        au_velocity_values = [f"{au_velocities.get(au, 0.0):.4f}" for au in VELOCITY_AUS]
        tension_variant_values = [f"{tension_variants[v]:.4f}" for v in TENSION_VARIANTS]

        csv_writer.writerow(
            [frame_count, f"{elapsed:.3f}", int(mp_face_detected), int(haar_face_detected)]
            + [f"{tension:.4f}", f"{startle_score:.4f}", ctx_tag, velocity_tag]
            + tension_variant_values
            + [f"{smoothed_tension:.4f}", f"{onset_slope:.6f}", event_status]
            + [f"{face_valence:.4f}", f"{smile_level:.4f}"]
            + bs_values
            + au_velocity_values + [f"{tension_velocity:.4f}"]
            + [mp_face_bbox_area, f"{latency_ms:.1f}", meta["cpu_percent"], meta["ram_mb"]]
        )

        # ── Console print ────────────────────────────────────────────────
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        status_short = event_status[:4] if event_status != "IDLE" else "----"
        src_ind = "M" if mp_face_detected else ("H" if haar_face_detected else "-")
        print(f"\rFrame {frame_count:4d} | "
              f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
              f"Lat: {latency_ms:5.1f}ms | "
              f"Face:{src_ind} | "
              f"T_v4:{tension:.2f} T_v5e:{tension_variants.get('v5e', 0):.2f} "
              f"[{ctx_tag:5s}] | "
              f"Sm:{smoothed_tension:.2f} [{status_short}]", end="")

        # ── Live display ─────────────────────────────────────────────────
        if not args.no_display:
            _hud_scale = HUD_MIN_HEIGHT / frame.shape[0]
            _hud_vid_w = int(frame.shape[1] * _hud_scale)
            _canvas = np.zeros((HUD_MIN_HEIGHT, _hud_vid_w + PANEL_WIDTH, 3), dtype=np.uint8)
            _canvas[0:HUD_MIN_HEIGHT, 0:_hud_vid_w] = cv2.resize(frame, (_hud_vid_w, HUD_MIN_HEIGHT))

            if mp_face_bbox:
                x1, y1, x2, y2 = [int(c * _hud_scale) for c in mp_face_bbox]
                _color = ((0, 0, 220) if event_status == "EVENT_CONFIRMED"
                          else (0, 200, 255) if event_status in ("ONSET", "SUSTAINING")
                          else (0, 200, 0))
                cv2.rectangle(_canvas, (x1, y1), (x2, y2), _color, 2)

            _mp_data = {
                "face_detected": mp_face_detected,
                "tension_variants": tension_variants,
                "face_valence": face_valence,
                "smile_level": smile_level,
                "ctx_tag": ctx_tag,
                "startle_score": startle_score,
                "velocity_tag": velocity_tag,
                "bs_dict": bs_dict,
            }
            _canvas = draw_mp_variants_hud(
                _canvas, _hud_vid_w, HUD_MIN_HEIGHT, _mp_data,
                detector_variant=args.detector_tension,
                event_status=event_status,
                smoothed_tension=smoothed_tension,
                onset_slope=onset_slope,
            )

            if event_status not in ("IDLE", ""):
                _badge_colors = {
                    "ONSET": (0, 200, 255), "SUSTAINING": (0, 130, 255),
                    "EVENT_CONFIRMED": (0, 0, 220), "EVENT_ENDED": (30, 180, 30),
                }
                _bc = _badge_colors.get(event_status, (150, 150, 150))
                _font = cv2.FONT_HERSHEY_SIMPLEX
                (_tw, _th), _ = cv2.getTextSize(event_status, _font, 0.50, 2)
                _bpad, _bx2 = 5, _hud_vid_w - 8
                _bx1 = _bx2 - _tw - _bpad * 2
                cv2.rectangle(_canvas, (_bx1, 45), (_bx2, 45 + _th + _bpad * 2), _bc, -1)
                cv2.putText(_canvas, event_status, (_bx1 + _bpad, 45 + _th + _bpad - 2),
                            _font, 0.50, (255, 255, 255), 2)

            cv2.imshow("MP-only Analysis", _canvas)
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
            "PERFORMANCE SUMMARY — MP-only Pipeline",
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
            f"Detector tension:       {args.detector_tension}",
            f"Tension variants saved: {', '.join(tension_variant_cols)}",
            "=" * 60,
            f"CSV (full):  {csv_path}",
            f"Events CSV:  {events_csv_path}",
            f"Summary:     {summary_path}",
            f"Events found: {len(detector.events)}",
            "=" * 60,
            "",
            "To evaluate:",
            f"  python Pipeline/fer/compare_ground_truth_mp.py \\",
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
