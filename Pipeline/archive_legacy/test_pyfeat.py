"""
La Facade Fissuree - Py-Feat FER Test
=====================================
Tests Py-Feat's facial expression recognition on live webcam feed.
Displays real-time emotion scores, measures latency, and saves CSV log.

Usage:
    python test_pyfeat.py

Press 'q' to quit. Results saved to logs/ folder.

NOTE: Py-Feat's detect_image() only accepts file paths, not numpy arrays or
PIL Images. Each frame is written to a reused temp JPEG file before detection.
This adds ~5-15ms of I/O overhead per frame, which is included in latency numbers.
"""

# ── GPU CONFIGURATION ────────────────────────────────────────────────────────
USE_GPU = False       # Set True to use CUDA (requires CUDA PyTorch)
GPU_DEVICE_INDEX = 0  # 0 = first CUDA device (RTX 4070 on this machine)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import cv2
import time
import os
import csv
import tempfile
import numpy as np
import torch
from datetime import datetime
from feat import Detector
from session_meta import Session
from post_session_analysis import post_session_analysis

EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


def _emotions_valid(result):
    """Return True if result has at least one non-NaN emotion value."""
    if result is None or len(result) == 0:
        return False
    for label in EMOTION_LABELS:
        if label in result.columns:
            val = result[label].values[0]
            if not (isinstance(val, float) and np.isnan(val)):
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file. Omit to use webcam.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip video display (faster processing).")
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    session = Session("pyfeat", log_dir)
    session.pre_session_prompt()

    csv_path = os.path.join(log_dir, f"{session.session_id}_pyfeat_temp.csv")
    summary_path = None

    if USE_GPU and torch.cuda.is_available():
        _device = "cuda"
        torch.cuda.set_device(GPU_DEVICE_INDEX)
        print(f"GPU: {torch.cuda.get_device_name(GPU_DEVICE_INDEX)} (device {GPU_DEVICE_INDEX})")
    else:
        _device = "cpu"
        if USE_GPU:
            print("GPU requested but CUDA unavailable — falling back to CPU.")
            print("  To enable: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("GPU: disabled (USE_GPU=False)")

    session.device_used = _device
    session.video_source = args.video if args.video else "webcam"
    print(f"Loading Py-Feat detector on {_device} (this may take a moment on first run)...")
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        device=_device,
    )
    print("Detector loaded successfully!")

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
    print("Py-Feat Live Emotion Detection")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print("Press 'q' to quit\n")

    session.start()

    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "timestamp", "latency_ms",
        "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral",
        "dominant", "dominant_score", "arousal",
    ] + meta_columns)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    # One reusable temp file — overwritten each frame, avoids per-frame allocation
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_time = time.perf_counter()

            try:
                # Write frame to temp file (BGR is fine — PIL loads it correctly via libjpeg)
                cv2.imwrite(tmp_path, frame)

                # detect_image() requires a file path string, not a numpy array or PIL Image
                result = detector.detect_image(tmp_path, progress_bar=False)

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                if _emotions_valid(result):
                    emotions = {}
                    for label in EMOTION_LABELS:
                        if label in result.columns:
                            val = result[label].values[0]
                            emotions[label] = float(val) if not np.isnan(val) else 0.0

                    dominant = max(emotions, key=emotions.get)
                    dominant_score = emotions[dominant]
                    # Arousal proxy: fear only (surprise fires on pose/lighting, not genuine arousal)
                    arousal = emotions.get("fear", 0)

                    face_bbox_area = 0
                    face_conf = 0.0
                    try:
                        if "FaceRectWidth" in result.columns and "FaceRectHeight" in result.columns:
                            fw = float(result["FaceRectWidth"].values[0])
                            fh = float(result["FaceRectHeight"].values[0])
                            face_bbox_area = int(fw * fh)
                        if "FaceScore" in result.columns:
                            face_conf = float(result["FaceScore"].values[0])
                    except (IndexError, ValueError):
                        pass

                    meta = session.log_frame(
                        face_detected=True, face_confidence=face_conf,
                        face_bbox_area=face_bbox_area,
                    )

                    elapsed = frame_count / video_fps if video_fps else time.perf_counter() - session_start
                    csv_writer.writerow([
                        frame_count, f"{elapsed:.3f}", f"{latency_ms:.1f}",
                        f"{emotions.get('anger', 0):.4f}",
                        f"{emotions.get('disgust', 0):.4f}",
                        f"{emotions.get('fear', 0):.4f}",
                        f"{emotions.get('happiness', 0):.4f}",
                        f"{emotions.get('sadness', 0):.4f}",
                        f"{emotions.get('surprise', 0):.4f}",
                        f"{emotions.get('neutral', 0):.4f}",
                        dominant, f"{dominant_score:.4f}", f"{arousal:.4f}",
                        meta["cpu_percent"], meta["ram_mb"], meta["face_detected"],
                        meta["face_confidence"], meta["face_bbox_area"],
                    ])

                    # Format elapsed time
                    elapsed_min = int(elapsed // 60)
                    elapsed_sec = int(elapsed % 60)
                    print(f"\rFrame {frame_count:4d} | "
                          f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
                          f"{latency_ms:6.1f}ms | "
                          f"{dominant:10s} ({dominant_score:.2f}) | "
                          f"Fear: {emotions.get('fear', 0):.2f} | "
                          f"Arousal: {arousal:.2f}", end="")

                    y_offset = 30

                    # Elapsed time (top-left)
                    elapsed_min = int(elapsed // 60)
                    elapsed_sec = int(elapsed % 60)
                    cv2.putText(frame, f"TIME: {elapsed_min:02d}:{elapsed_sec:02d}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    y_offset += 25

                    for label, score in emotions.items():
                        bar_length = int(score * 200)
                        color = (0, 255, 0) if label == dominant else (200, 200, 200)
                        cv2.putText(frame, f"{label}: {score:.2f}", (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.rectangle(frame, (160, y_offset - 12), (160 + bar_length, y_offset),
                                      color, -1)
                        y_offset += 25
                    cv2.putText(frame, f"AROUSAL: {arousal:.2f}", (10, y_offset + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"{latency_ms:.0f}ms", (frame.shape[1] - 80, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                else:
                    meta = session.log_frame(face_detected=False)
                    print(f"\rFrame {frame_count:4d} | No face | {latency_ms:.1f}ms", end="")

            except Exception as e:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                session.log_frame(face_detected=False)
                print(f"\rFrame {frame_count:4d} | Error: {str(e)[:60]} | {latency_ms:.1f}ms", end="")

            if not args.no_display:
                cv2.imshow("Py-Feat - Emotion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not session.post_session_confirm(csv_path):
        return

    csv_path, summary_path = session.finalize_session_files(csv_path)

    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "",
            "=" * 60,
            "PERFORMANCE SUMMARY — Py-Feat",
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
            f"CSV log saved to:       {csv_path}",
            f"Summary saved to:       {summary_path}",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nDone! {frame_count} frames logged to {csv_path}")

        post_session_analysis(csv_path, summary_path, run_viz=True, run_emotion=False)


if __name__ == "__main__":
    main()
