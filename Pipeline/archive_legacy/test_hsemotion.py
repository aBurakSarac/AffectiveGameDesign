"""
La Facade Fissuree - HSEmotion Valence-Arousal Test
====================================================
Tests HSEmotion for continuous valence-arousal prediction on live webcam feed.
Unlike DeepFace (categorical emotions), this outputs continuous VA scores
that better capture subtle emotional shifts.

Usage:
    python test_hsemotion.py

Press 'q' to quit. Results saved to logs/ folder.
"""

# ── GPU CONFIGURATION — set before onnxruntime loads ─────────────────────────
USE_GPU = False       # Set True to use CUDA (requires onnxruntime-gpu)
GPU_DEVICE_INDEX = 0  # 0 = first CUDA device (RTX 4070 on this machine)
# ─────────────────────────────────────────────────────────────────────────────

import os
import argparse
import onnxruntime as ort

_cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
if USE_GPU and _cuda_available:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_INDEX)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import time
import csv
import numpy as np
from datetime import datetime
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from session_meta import Session
from post_session_analysis import post_session_analysis

# HSEmotion emotion labels
EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def main():
    """Run the HSEmotion valence-arousal detection loop.

    Opens the webcam, initializes the HSEmotion model
    (enet_b0_8_va_mtl) and OpenCV Haar cascade face detector.
    Runs a real-time frame-by-frame loop predicting continuous
    valence and arousal scores along with 8 discrete emotion
    probabilities. Displays annotated video with emotion bars
    and VA scores. Logs all data to a timestamped CSV in the
    logs/ directory.

    Press 'q' to quit. A performance summary is printed and
    saved at the end of the session.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file. Omit to use webcam.")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip video display (faster processing).")
    args = parser.parse_args()

    # Create logs directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Session metadata — prompt before webcam opens
    session = Session("hsemotion", log_dir)
    session.pre_session_prompt()

    csv_path = os.path.join(log_dir, f"{session.session_id}_hsemotion_temp.csv")
    summary_path = None  # set after finalize

    if USE_GPU and _cuda_available:
        _device = "cuda"
        print(f"GPU: CUDAExecutionProvider enabled (device {GPU_DEVICE_INDEX})")
    else:
        _device = "cpu"
        if USE_GPU:
            print("GPU requested but onnxruntime-gpu not installed — using CPU.")
            print("  To enable: pip install onnxruntime-gpu")
        else:
            print("GPU: disabled (USE_GPU=False)")
    session.device_used = _device
    session.video_source = args.video if args.video else "webcam"

    # Initialize HSEmotion — model with valence-arousal multi-task learning
    print("Loading HSEmotion model...")
    model_name = "enet_b0_8_va_mtl"  # EfficientNet-B0 with VA + emotion multi-task
    recognizer = HSEmotionRecognizer(model_name=model_name)
    print(f"HSEmotion loaded: {model_name}")

    # Initialize face detector (OpenCV Haar cascade — fast, built-in)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
    print("HSEmotion Live Valence-Arousal Detection")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print("Press 'q' to quit")
    print()

    # Start session monitoring
    session.start()

    # New CSV columns from session metadata
    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]

    # Open CSV
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "timestamp", "latency_ms",
        "valence", "arousal",
        "anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
        "dominant", "dominant_score"
    ] + meta_columns)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.perf_counter()

        try:
            # Detect face using Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

            if len(faces) > 0:
                # Take largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_img = frame[y:y+h, x:x+w]

                # HSEmotion prediction — MTL model returns 10 scores:
                # first 8 = emotion probabilities, last 2 = valence & arousal
                emotion, scores = recognizer.predict_emotions(face_img, logits=False)

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                # Extract valence and arousal from the last 2 elements
                valence = float(scores[-2])      # range ~ -1 to +1
                arousal_val = float(scores[-1])   # range ~ 0 to 1

                # Build emotion dict from first 8 scores
                emotions = {}
                for i, label in enumerate(EMOTION_LABELS):
                    if i < len(scores) - 2:  # exclude VA from emotion list
                        emotions[label] = float(scores[i])

                dominant = emotion if emotion else max(emotions, key=emotions.get)
                dominant_score = emotions.get(dominant, 0)

                # Session resource + face metadata
                # face_confidence: HSEmotion (Haar cascade) does not expose a detection score,
                # so we use face bbox area as a size proxy (larger = more visible face).
                face_bbox_area = int(w * h)
                frame_area = frame.shape[0] * frame.shape[1]
                face_size_ratio = face_bbox_area / frame_area if frame_area > 0 else 0.0
                meta = session.log_frame(
                    face_detected=True, face_confidence=face_size_ratio,
                    face_bbox_area=face_bbox_area,
                )

                # Log to CSV
                elapsed = frame_count / video_fps if video_fps else time.perf_counter() - session_start
                csv_writer.writerow([
                    frame_count, f"{elapsed:.3f}", f"{latency_ms:.1f}",
                    f"{valence:.4f}", f"{arousal_val:.4f}",
                    *[f"{emotions.get(l, 0):.4f}" for l in EMOTION_LABELS],
                    dominant, f"{dominant_score:.4f}",
                    meta["cpu_percent"], meta["ram_mb"], meta["face_detected"],
                    meta["face_confidence"], meta["face_bbox_area"],
                ])

                # Print with elapsed time
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                print(f"\rFrame {frame_count:4d} | "
                      f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
                      f"Latency: {latency_ms:5.1f}ms | "
                      f"V: {valence:+.2f} A: {arousal_val:.2f} | "
                      f"Dom: {dominant:10s} ({dominant_score:.2f})", end="")

                # Draw face box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Elapsed time (top-left)
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                cv2.putText(frame, f"TIME: {elapsed_min:02d}:{elapsed_sec:02d}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                # Draw VA scores prominently
                cv2.putText(frame, f"Valence: {valence:+.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, f"Arousal: {arousal_val:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw emotion bars
                y_offset = 130
                for label in EMOTION_LABELS:
                    score = emotions.get(label, 0)
                    bar_length = int(score * 200)
                    color = (0, 255, 0) if label == dominant else (180, 180, 180)
                    cv2.putText(frame, f"{label[:4]}: {score:.2f}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.rectangle(frame, (100, y_offset - 10), (100 + bar_length, y_offset),
                                  color, -1)
                    y_offset += 20

                # Latency
                cv2.putText(frame, f"{latency_ms:.0f}ms", (frame.shape[1] - 80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                session.log_frame(face_detected=False)
                print(f"\rFrame {frame_count:4d} | No face | {latency_ms:.1f}ms", end="")

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            session.log_frame(face_detected=False)
            print(f"\rFrame {frame_count:4d} | Error: {str(e)[:60]} | {latency_ms:.1f}ms", end="")

        if not args.no_display:
            cv2.imshow("HSEmotion - Valence/Arousal", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    if not session.post_session_confirm(csv_path):
        return

    csv_path, summary_path = session.finalize_session_files(csv_path)

    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "", "=" * 60,
            "PERFORMANCE SUMMARY — HSEmotion (Valence-Arousal)",
            "=" * 60,
            f"Model:                  {model_name}",
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
