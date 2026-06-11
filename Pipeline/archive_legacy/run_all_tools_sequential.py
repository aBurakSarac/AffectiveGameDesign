"""
La Facade Fissuree - Sequential Multi-Tool Runner
==================================================
Runs all 4 FER tools (MediaPipe, HSEmotion, DeepFace, Pyfeat) sequentially
in a single session without re-prompting for metadata each time.

The metadata (subject, content, lighting, notes) is collected once at the start
and applied to all 4 tools. Each tool opens the webcam fresh, records its own
data, and saves separate CSVs with the same session metadata.

Usage:
    python run_all_tools_sequential.py [--no-display]

Press 'q' during any tool to skip to the next tool.
All CSVs are saved to logs/ with tool-specific columns.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from datetime import datetime

import cv2

# Import core components from individual tools
from session_meta import Session
from post_session_analysis import post_session_analysis


# ============================================================================
# SHARED METADATA SETUP
# ============================================================================

def setup_shared_metadata(log_dir):
    """
    Prompt once for session metadata (subject, content, lighting, notes).
    Returns a Session object with metadata pre-filled.
    """
    session = Session("multi_tool", log_dir)

    print("\n" + "=" * 70)
    print("  SEQUENTIAL MULTI-TOOL RUNNER - Single Session Metadata Setup")
    print("=" * 70)

    session.subject_id = (
        input("  Subject ID [S01]: ").strip() or "S01"
    )
    session.session_label = (
        input("  Session label (optional, e.g. Phase3_test) []: ").strip()
    )

    from session_meta import KNOWN_CONTENT_TYPES, KNOWN_STIMULUS_TYPES, KNOWN_LIGHTING

    session.content_type = (
        input(f"  Content type ({'/'.join(sorted(KNOWN_CONTENT_TYPES))}) [mixed]: ").strip()
        or "mixed"
    )
    if session.content_type not in KNOWN_CONTENT_TYPES:
        print(f"    (note: '{session.content_type}' not in known set, using as-is)")

    session.stimulus_type = (
        input(f"  Stimulus type ({'/'.join(sorted(KNOWN_STIMULUS_TYPES))}) [mixed]: ").strip()
        or "mixed"
    )
    if session.stimulus_type not in KNOWN_STIMULUS_TYPES:
        print(f"    (note: '{session.stimulus_type}' not in known set, using as-is)")

    session.lighting = (
        input(f"  Lighting ({'/'.join(sorted(KNOWN_LIGHTING))}) [bright]: ").strip()
        or "bright"
    )
    if session.lighting not in KNOWN_LIGHTING:
        print(f"    (note: '{session.lighting}' not in known set, using as-is)")

    session.notes = input("  Notes []: ").strip()

    print(f"\n  Metadata locked in:")
    print(f"    Subject:      {session.subject_id}")
    if session.session_label:
        print(f"    Label:        {session.session_label}")
    print(f"    Content:      {session.content_type}")
    print(f"    Stimulus:     {session.stimulus_type}")
    print(f"    Lighting:     {session.lighting}")
    print(f"    Notes:        {session.notes or '(none)'}")
    print(f"    GPU:          {session.hardware['gpu']}")
    print("=" * 70)
    print("\n  Starting tool sequence in 3 seconds...\n")
    time.sleep(3)

    return session


# ============================================================================
# MEDIAPIPE TOOL
# ============================================================================

def run_mediapipe(shared_session, log_dir, no_display):
    """Run MediaPipe FaceLandmarker for blendshape detection."""
    print("\n" + "=" * 70)
    print("  [1/4] MEDIAPIPE BLENDSHAPES")
    print("=" * 70)

    try:
        import mediapipe as mp
        from improved_fear_detection import get_velocity_tag
        import urllib.request
    except ImportError as e:
        print(f"ERROR: Missing dependency for MediaPipe: {e}")
        return False

    # Setup MediaPipe model
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
    MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        print(f"Downloading MediaPipe model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # Create session object for this tool
    session = Session("mediapipe", log_dir)
    session.subject_id = shared_session.subject_id
    session.content_type = shared_session.content_type
    session.lighting = shared_session.lighting
    session.notes = shared_session.notes
    session.session_label = shared_session.session_label
    session.stimulus_type = shared_session.stimulus_type

    csv_path = os.path.join(log_dir, f"{session.session_id}_mediapipe_temp.csv")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Detect device
    try:
        import tensorflow as tf
        _gpu_devs = tf.config.list_physical_devices("GPU")
        session.device_used = "cuda" if _gpu_devs else "cpu"
        print(f"GPU: {_gpu_devs[0].name if _gpu_devs else 'CPU'}")
    except:
        session.device_used = "cpu"
        print("GPU: CPU only")

    session.start()

    # Setup CSV
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    blendshapes = [
        "browDownLeft", "browDownRight", "eyeWideLeft", "eyeWideRight",
        "eyeSquintLeft", "eyeSquintRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthPucker", "mouthRollUpper", "mouthRollLower", "mouthShrugUpper",
        "mouthShrugLower", "jawOpen", "noseSneerLeft", "noseSneerRight",
        "eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight",
        "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
        "eyeBlinkLeft", "eyeBlinkRight", "cheekPuff", "cheekSquintLeft",
        "cheekSquintRight", "winkLeft", "winkRight", "jawForward", "jawLeft",
        "jawRight", "mouthFunnel", "mouthPress", "mouthLeft", "mouthRight",
        "mouthSmileLeft", "mouthSmileRight", "mouthDimpleLeft", "mouthDimpleRight",
        "mouthStretchLeft", "mouthStretchRight", "mouthRollLower", "mouthLowerDownLeft",
        "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "eyeClosedLeft",
        "eyeClosedRight", "face_valence", "face_arousal", "tension",
    ]

    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]

    csv_writer.writerow(
        ["frame", "timestamp", "latency_ms"] + blendshapes + meta_columns
    )

    # Detection model
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    print(f"Logging to: {csv_path}")
    print("Press 'q' to finish and move to next tool\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.perf_counter()

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = detector.detect(mp_image)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            face_detected = False
            face_conf = 0.0
            face_bbox_area = 0
            blendshape_scores = [0.0] * len(blendshapes)

            if results.face_blendshapes and len(results.face_blendshapes) > 0:
                face_detected = True
                face_conf = 0.95  # MediaPipe doesn't give explicit confidence

                blendshapes_list = results.face_blendshapes[0]
                for i, bs in enumerate(blendshapes_list):
                    if i < len(blendshapes):
                        blendshape_scores[i] = bs.score

            meta = session.log_frame(face_detected=face_detected, face_confidence=face_conf, face_bbox_area=face_bbox_area)

            elapsed = frame_count / 30.0  # Assume 30 FPS
            csv_writer.writerow(
                [frame_count, f"{elapsed:.3f}", f"{latency_ms:.1f}"] +
                [f"{s:.4f}" for s in blendshape_scores] +
                [meta["cpu_percent"], meta["ram_mb"], meta["face_detected"], meta["face_confidence"], meta["face_bbox_area"]]
            )

            print(f"\rMediaPipe Frame {frame_count:4d} | Latency: {latency_ms:6.1f}ms | Face: {face_detected}", end="")

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            session.log_frame(face_detected=False)
            print(f"\rMediaPipe Frame {frame_count:4d} | Error: {str(e)[:40]} | {latency_ms:.1f}ms", end="")

        if not no_display:
            cv2.imshow("MediaPipe - Blendshapes", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    # Finalize
    csv_path, summary_path = session.finalize_session_files(csv_path)

    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "",
            "=" * 60,
            "MEDIAPIPE SUMMARY",
            "=" * 60,
            f"Duration:       {record['duration_s']:.1f}s",
            f"Frames:         {frame_count}",
            f"Avg latency:    {record['avg_latency_ms']:.1f}ms",
            f"Median latency: {record['median_latency_ms']:.1f}ms",
            f"P95 latency:    {record['p95_latency_ms']:.1f}ms",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nMediaPipe CSV: {csv_path}")

    return True


# ============================================================================
# HSEMOTION TOOL
# ============================================================================

def run_hsemotion(shared_session, log_dir, no_display):
    """Run HSEmotion for valence-arousal detection."""
    print("\n" + "=" * 70)
    print("  [2/4] HSEMOTION VALENCE-AROUSAL")
    print("=" * 70)

    try:
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    except ImportError as e:
        print(f"ERROR: Missing dependency for HSEmotion: {e}")
        return False

    session = Session("hsemotion", log_dir)
    session.subject_id = shared_session.subject_id
    session.content_type = shared_session.content_type
    session.lighting = shared_session.lighting
    session.notes = shared_session.notes
    session.session_label = shared_session.session_label
    session.stimulus_type = shared_session.stimulus_type

    csv_path = os.path.join(log_dir, f"{session.session_id}_hsemotion_temp.csv")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    session.device_used = "cpu"

    session.start()

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]

    csv_writer.writerow(
        ["frame", "timestamp", "latency_ms", "valence", "arousal", "dominant"] +
        [f"{e.lower()}" for e in EMOTION_LABELS] + meta_columns
    )

    # Initialize model
    model = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl", pretrained=True, device="cpu")

    # Face detector
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    print(f"Logging to: {csv_path}")
    print("Press 'q' to finish and move to next tool\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.perf_counter()

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face_detected = len(faces) > 0
            face_conf = 0.95 if face_detected else 0.0
            face_bbox_area = 0

            valence = 0.0
            arousal = 0.0
            dominant = "unknown"
            emotion_scores = [0.0] * len(EMOTION_LABELS)

            if face_detected:
                x, y, w, h = faces[0]
                face_bbox_area = w * h
                roi = frame[y:y+h, x:x+w]

                # Predict emotions
                pred = model.predict_emotions(roi, logits=False)

                if pred:
                    valence = float(pred[0]) if len(pred) > 0 else 0.0
                    arousal = float(pred[1]) if len(pred) > 1 else 0.0

                    # Get dominant emotion (fallback to neutral)
                    dominant = "neutral"
                    max_prob = 0.0

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            meta = session.log_frame(face_detected=face_detected, face_confidence=face_conf, face_bbox_area=face_bbox_area)

            elapsed = frame_count / 30.0
            csv_writer.writerow(
                [frame_count, f"{elapsed:.3f}", f"{latency_ms:.1f}", f"{valence:.4f}", f"{arousal:.4f}", dominant] +
                [f"{s:.4f}" for s in emotion_scores] +
                [meta["cpu_percent"], meta["ram_mb"], meta["face_detected"], meta["face_confidence"], meta["face_bbox_area"]]
            )

            print(f"\rHSEmotion Frame {frame_count:4d} | Latency: {latency_ms:6.1f}ms | VA: ({valence:.2f}, {arousal:.2f})", end="")

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            session.log_frame(face_detected=False)
            print(f"\rHSEmotion Frame {frame_count:4d} | Error: {str(e)[:40]} | {latency_ms:.1f}ms", end="")

        if not no_display:
            cv2.imshow("HSEmotion - Valence-Arousal", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    csv_path, summary_path = session.finalize_session_files(csv_path)

    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "",
            "=" * 60,
            "HSEMOTION SUMMARY",
            "=" * 60,
            f"Duration:       {record['duration_s']:.1f}s",
            f"Frames:         {frame_count}",
            f"Avg latency:    {record['avg_latency_ms']:.1f}ms",
            f"Median latency: {record['median_latency_ms']:.1f}ms",
            f"P95 latency:    {record['p95_latency_ms']:.1f}ms",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nHSEmotion CSV: {csv_path}")

    return True


# ============================================================================
# DEEPFACE TOOL
# ============================================================================

def run_deepface(shared_session, log_dir, no_display):
    """Run DeepFace for emotion classification."""
    print("\n" + "=" * 70)
    print("  [3/4] DEEPFACE EMOTION CLASSIFICATION")
    print("=" * 70)

    try:
        from deepface import DeepFace
    except ImportError as e:
        print(f"ERROR: Missing dependency for DeepFace: {e}")
        return False

    session = Session("deepface", log_dir)
    session.subject_id = shared_session.subject_id
    session.content_type = shared_session.content_type
    session.lighting = shared_session.lighting
    session.notes = shared_session.notes
    session.session_label = shared_session.session_label
    session.stimulus_type = shared_session.stimulus_type

    csv_path = os.path.join(log_dir, f"{session.session_id}_deepface_temp.csv")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        import tensorflow as tf
        _gpu_devs = tf.config.list_physical_devices("GPU")
        session.device_used = "cuda" if _gpu_devs else "cpu"
    except:
        session.device_used = "cpu"

    session.start()

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]

    csv_writer.writerow(
        ["frame", "timestamp", "latency_ms"] + EMOTION_LABELS +
        ["dominant", "dominant_score", "arousal"] + meta_columns
    )

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    print(f"Logging to: {csv_path}")
    print("Press 'q' to finish and move to next tool\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.perf_counter()

        try:
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            face_detected = False
            face_conf = 0.0
            face_bbox_area = 0
            emotions_normalized = {e: 0.0 for e in EMOTION_LABELS}
            dominant = "unknown"
            dominant_score = 0.0
            arousal = 0.0

            if results and len(results) > 0:
                result = results[0]
                face_detected = True

                emotions = result.get("emotion", {})
                emotions_normalized = {k: v / 100.0 for k, v in emotions.items()}
                dominant = result.get("dominant_emotion", "unknown")
                dominant_score = emotions_normalized.get(dominant, 0)
                arousal = emotions_normalized.get("fear", 0)

                region = result.get("region", {})
                face_bbox_area = int(region.get("w", 0) * region.get("h", 0))
                face_conf = result.get("face_confidence", 0.0)

            meta = session.log_frame(face_detected=face_detected, face_confidence=face_conf, face_bbox_area=face_bbox_area)

            elapsed = frame_count / 30.0
            csv_writer.writerow(
                [frame_count, f"{elapsed:.3f}", f"{latency_ms:.1f}"] +
                [f"{emotions_normalized.get(e, 0):.4f}" for e in EMOTION_LABELS] +
                [dominant, f"{dominant_score:.4f}", f"{arousal:.4f}"] +
                [meta["cpu_percent"], meta["ram_mb"], meta["face_detected"], meta["face_confidence"], meta["face_bbox_area"]]
            )

            print(f"\rDeepFace Frame {frame_count:4d} | Latency: {latency_ms:6.1f}ms | Dominant: {dominant:10s} | Arousal: {arousal:.2f}", end="")

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            session.log_frame(face_detected=False)
            print(f"\rDeepFace Frame {frame_count:4d} | Error: {str(e)[:40]} | {latency_ms:.1f}ms", end="")

        if not no_display:
            cv2.imshow("DeepFace - Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    csv_path, summary_path = session.finalize_session_files(csv_path)

    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "",
            "=" * 60,
            "DEEPFACE SUMMARY",
            "=" * 60,
            f"Duration:       {record['duration_s']:.1f}s",
            f"Frames:         {frame_count}",
            f"Avg latency:    {record['avg_latency_ms']:.1f}ms",
            f"Median latency: {record['median_latency_ms']:.1f}ms",
            f"P95 latency:    {record['p95_latency_ms']:.1f}ms",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nDeepFace CSV: {csv_path}")

    return True


# ============================================================================
# PYFEAT TOOL
# ============================================================================

def run_pyfeat(shared_session, log_dir, no_display):
    """Run Py-Feat for emotion detection."""
    print("\n" + "=" * 70)
    print("  [4/4] PY-FEAT EMOTION DETECTION")
    print("=" * 70)

    try:
        from feat import Detector
    except ImportError as e:
        print(f"ERROR: Missing dependency for Py-Feat: {e}")
        return False

    session = Session("pyfeat", log_dir)
    session.subject_id = shared_session.subject_id
    session.content_type = shared_session.content_type
    session.lighting = shared_session.lighting
    session.notes = shared_session.notes
    session.session_label = shared_session.session_label
    session.stimulus_type = shared_session.stimulus_type

    csv_path = os.path.join(log_dir, f"{session.session_id}_pyfeat_temp.csv")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    session.device_used = "cpu"

    session.start()

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    meta_columns = ["cpu_percent", "ram_mb", "face_detected", "face_confidence", "face_bbox_area"]

    csv_writer.writerow(
        ["frame", "timestamp", "latency_ms", "arousal", "valence", "dominant"] +
        [f"{e.lower()}" for e in EMOTION_LABELS] + meta_columns
    )

    # Initialize detector
    detector = Detector(face_model="retinaface", emotion_model="xgb", facs_model=None)

    latencies = []
    frame_count = 0
    session_start = time.perf_counter()

    print(f"Logging to: {csv_path}")
    print("Press 'q' to finish and move to next tool\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.perf_counter()

        try:
            # Detect faces and emotions
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_emotions(frame_rgb)

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            face_detected = False
            face_conf = 0.0
            face_bbox_area = 0
            arousal = 0.0
            valence = 0.0
            dominant = "unknown"
            emotion_scores = [0.0] * len(EMOTION_LABELS)

            if results is not None and len(results) > 0:
                face_detected = True
                face_conf = 0.95

                # Take first face
                face_data = results.iloc[0]

                # Extract emotion scores
                for i, emotion in enumerate(EMOTION_LABELS):
                    if emotion in face_data:
                        emotion_scores[i] = float(face_data[emotion])

                # Compute dominant emotion
                if emotion_scores:
                    max_idx = np.argmax(emotion_scores)
                    dominant = EMOTION_LABELS[max_idx]

                # Simple arousal proxy
                arousal = emotion_scores[2]  # fear index
                valence = 0.5  # Py-Feat doesn't give explicit valence

            meta = session.log_frame(face_detected=face_detected, face_confidence=face_conf, face_bbox_area=face_bbox_area)

            elapsed = frame_count / 30.0
            csv_writer.writerow(
                [frame_count, f"{elapsed:.3f}", f"{latency_ms:.1f}", f"{arousal:.4f}", f"{valence:.4f}", dominant] +
                [f"{s:.4f}" for s in emotion_scores] +
                [meta["cpu_percent"], meta["ram_mb"], meta["face_detected"], meta["face_confidence"], meta["face_bbox_area"]]
            )

            print(f"\rPy-Feat Frame {frame_count:4d} | Latency: {latency_ms:6.1f}ms | Dominant: {dominant:10s} | Arousal: {arousal:.2f}", end="")

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            session.log_frame(face_detected=False)
            print(f"\rPy-Feat Frame {frame_count:4d} | Error: {str(e)[:40]} | {latency_ms:.1f}ms", end="")

        if not no_display:
            cv2.imshow("Py-Feat - Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    csv_path, summary_path = session.finalize_session_files(csv_path)

    if latencies:
        record, extra_lines = session.finish(latencies, frame_count, csv_path, summary_path)

        summary_lines = [
            "",
            "=" * 60,
            "PY-FEAT SUMMARY",
            "=" * 60,
            f"Duration:       {record['duration_s']:.1f}s",
            f"Frames:         {frame_count}",
            f"Avg latency:    {record['avg_latency_ms']:.1f}ms",
            f"Median latency: {record['median_latency_ms']:.1f}ms",
            f"P95 latency:    {record['p95_latency_ms']:.1f}ms",
            "=" * 60,
        ]
        summary_lines.extend(extra_lines)

        for line in summary_lines:
            print(line)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print(f"\nPy-Feat CSV: {csv_path}")

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all 4 FER tools sequentially with shared metadata."
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Skip video display (faster processing)."
    )
    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Setup metadata once
    shared_session = setup_shared_metadata(log_dir)

    # Run all 4 tools in sequence
    tools = [
        ("MediaPipe", run_mediapipe),
        ("HSEmotion", run_hsemotion),
        ("DeepFace", run_deepface),
        ("Py-Feat", run_pyfeat),
    ]

    results = {}
    for tool_name, tool_func in tools:
        try:
            success = tool_func(shared_session, log_dir, args.no_display)
            results[tool_name] = "✓ completed" if success else "✗ failed"
        except Exception as e:
            print(f"\nERROR in {tool_name}: {e}")
            results[tool_name] = f"✗ error: {str(e)[:30]}"

        print()

    # Final summary
    print("\n" + "=" * 70)
    print("  SESSION COMPLETE")
    print("=" * 70)
    print(f"  Subject:     {shared_session.subject_id}")
    print(f"  Content:     {shared_session.content_type}")
    print(f"  Lighting:    {shared_session.lighting}")
    print(f"  Label:       {shared_session.session_label or '(none)'}")
    print()
    print("  TOOL RESULTS:")
    for tool_name, status in results.items():
        print(f"    {tool_name:12s}  {status}")
    print("=" * 70)
    print(f"\n  All CSVs saved to: {log_dir}")


if __name__ == "__main__":
    main()
