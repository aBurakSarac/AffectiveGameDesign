"""
La Facade Fissuree - DeepFace FER Test
======================================
Tests DeepFace's facial expression recognition on live webcam feed.
Displays real-time emotion scores, measures latency, and saves CSV log.

Usage:
    python test_deepface.py

Press 'q' to quit. Results saved to logs/ folder.
"""

# ── GPU CONFIGURATION — must be set before TensorFlow/DeepFace import ────────
USE_GPU = False       # Set True to use CUDA (not available on Windows with TF)
GPU_DEVICE_INDEX = 0  # 0 = first CUDA device (RTX 4070 on this machine)
# ─────────────────────────────────────────────────────────────────────────────

import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_INDEX) if USE_GPU else "-1"

import cv2
import time
import csv
import numpy as np
from datetime import datetime
from deepface import DeepFace
from session_meta import Session
from post_session_analysis import post_session_analysis

# DeepFace emotion labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main():
    """Run the DeepFace emotion detection loop.

    Opens the webcam and runs DeepFace.analyze() frame by frame
    to classify 7 categorical emotions (angry, disgust, fear,
    happy, sad, surprise, neutral). Displays annotated video
    with emotion probability bars, dominant emotion label, and
    a computed arousal score (max of fear and surprise). Logs
    all data to a timestamped CSV in the logs/ directory.

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
    session = Session("deepface", log_dir)
    session.pre_session_prompt()

    csv_path = os.path.join(log_dir, f"{session.session_id}_deepface_temp.csv")
    summary_path = None  # set after finalize

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

    import tensorflow as tf
    _gpu_devs = tf.config.list_physical_devices("GPU")
    if _gpu_devs:
        _device = "cuda"
        print(f"GPU: {_gpu_devs[0].name} (device {GPU_DEVICE_INDEX})")
    else:
        _device = "cpu"
        if USE_GPU:
            print("GPU requested but TF sees no GPU — using CPU.")
            print("  To enable: pip install tensorflow[and-cuda]")
        else:
            print("GPU: disabled (USE_GPU=False)")
    session.device_used = _device
    session.video_source = args.video if args.video else "webcam"

    print("\n" + "=" * 60)
    print("DeepFace Live Emotion Detection")
    print("=" * 60)
    print(f"Logging to: {csv_path}")
    print("Press 'q' to quit")
    print("(First frame may be slow — model is loading...)")
    print()

    # Start session monitoring
    session.start()

    # New CSV columns from session metadata
    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]

    # Open CSV file for logging
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "timestamp", "latency_ms",
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
        "dominant", "dominant_score", "arousal"
    ] + meta_columns)

    # Performance tracking
    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            if not args.video:
                print("ERROR: Failed to capture frame.")
            break

        frame_count += 1

        # Measure latency for DeepFace detection
        start_time = time.perf_counter()

        try:
            # DeepFace analyze — enforce_detection=False avoids crash when no face found
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # DeepFace returns a list of results (one per face)
            if results and len(results) > 0:
                result = results[0]
                emotions = result.get("emotion", {})
                dominant = result.get("dominant_emotion", "unknown")

                # Normalize scores to 0-1 range (DeepFace returns percentages 0-100)
                emotions_normalized = {k: v / 100.0 for k, v in emotions.items()}

                # Arousal proxy: use fear alone.
                # surprise is excluded — it fires on pose/lighting changes, not genuine arousal,
                # which would inflate arousal scores during mundane stimuli.
                arousal = emotions_normalized.get("fear", 0)

                # Face bbox area from DeepFace region
                region = result.get("region", {})
                face_bbox_area = int(region.get("w", 0) * region.get("h", 0))
                face_conf = result.get("face_confidence", 0.0)

                # Session resource + face metadata
                meta = session.log_frame(
                    face_detected=True, face_confidence=face_conf,
                    face_bbox_area=face_bbox_area,
                )

                # Log to CSV
                elapsed = frame_count / video_fps if video_fps else time.perf_counter() - session_start
                csv_writer.writerow([
                    frame_count,
                    f"{elapsed:.3f}",
                    f"{latency_ms:.1f}",
                    f"{emotions_normalized.get('angry', 0):.4f}",
                    f"{emotions_normalized.get('disgust', 0):.4f}",
                    f"{emotions_normalized.get('fear', 0):.4f}",
                    f"{emotions_normalized.get('happy', 0):.4f}",
                    f"{emotions_normalized.get('sad', 0):.4f}",
                    f"{emotions_normalized.get('surprise', 0):.4f}",
                    f"{emotions_normalized.get('neutral', 0):.4f}",
                    dominant,
                    f"{emotions_normalized.get(dominant, 0):.4f}",
                    f"{arousal:.4f}",
                    meta["cpu_percent"], meta["ram_mb"], meta["face_detected"],
                    meta["face_confidence"], meta["face_bbox_area"],
                ])

                # Print results with elapsed time
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                print(f"\rFrame {frame_count:4d} | "
                      f"[{elapsed_min:02d}:{elapsed_sec:02d}] | "
                      f"Latency: {latency_ms:6.1f}ms | "
                      f"Dominant: {dominant:10s} ({emotions_normalized.get(dominant, 0):.2f}) | "
                      f"Fear: {emotions_normalized.get('fear', 0):.2f} | "
                      f"Surprise: {emotions_normalized.get('surprise', 0):.2f} | "
                      f"Arousal: {arousal:.2f}", end="")

                # Draw on frame
                y_offset = 30

                # Elapsed time (top-left)
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                cv2.putText(frame, f"TIME: {elapsed_min:02d}:{elapsed_sec:02d}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                y_offset += 25

                for label in EMOTION_LABELS:
                    score = emotions_normalized.get(label, 0)
                    bar_length = int(score * 200)
                    color = (0, 255, 0) if label == dominant else (200, 200, 200)
                    cv2.putText(frame, f"{label}: {score:.2f}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.rectangle(frame, (140, y_offset - 12), (140 + bar_length, y_offset),
                                  color, -1)
                    y_offset += 25

                # Draw arousal bar
                cv2.putText(frame, f"AROUSAL: {arousal:.2f}", (10, y_offset + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw latency
                cv2.putText(frame, f"{latency_ms:.0f}ms", (frame.shape[1] - 80, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            session.log_frame(face_detected=False)
            print(f"\rFrame {frame_count:4d} | Error: {str(e)[:50]} | {latency_ms:.1f}ms", end="")

        # Display the frame
        if not args.no_display:
            cv2.imshow("DeepFace - Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    if not session.post_session_confirm(csv_path):
        return

    csv_path, summary_path = session.finalize_session_files(csv_path)

    # Print and save summary statistics
    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "",
            "=" * 60,
            "PERFORMANCE SUMMARY — DeepFace",
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

        # Auto-run post-session analysis (visualization + stats)
        post_session_analysis(csv_path, summary_path, run_viz=True, run_emotion=False)


if __name__ == "__main__":
    main()
