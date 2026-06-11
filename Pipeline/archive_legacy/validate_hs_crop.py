"""
La Facade Fissuree - Haar vs MP Crop Validation for HSEmotion
=============================================================
Compares HSEmotion output when fed Haar cascade crops vs MediaPipe
landmark-derived crops on the same frame. One-time validation before
replacing Haar with MP crops in the fusion pipeline.

Usage:
    python validate_hs_crop.py --video path/to/video.mp4
    python validate_hs_crop.py --video path/to/video.mp4 --sweep
    python validate_hs_crop.py --video path/to/video.mp4 --padding 0.15

Outputs saved to Pipeline/logs/validation/
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
import mediapipe as mp
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ── Constants ────────────────────────────────────────────────────────────────
EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]

DEFAULT_PADDING = 0.15
PADDING_SWEEP = [0, 0.05, 0.10, 0.12, 0.15, 0.18]

# PASS/FAIL thresholds
PASS_AROUSAL_R = 0.85
PASS_AROUSAL_MAD = 0.10
PASS_VALENCE_R = 0.80
PASS_VALENCE_MAD = 0.15
PASS_DOMINANT_AGREE = 0.60

MIN_CROP_SIZE = 20


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_model():
    """Download the FaceLandmarker model if not present."""
    if os.path.isfile(MODEL_PATH):
        return
    print(f"Downloading FaceLandmarker model to {MODEL_PATH} ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def get_mp_face_crop(frame, landmarks, padding=0.20):
    """Extract face crop from MP landmarks with configurable padding.

    Returns (face_crop, bbox_tuple) or (None, None) if too small.
    bbox_tuple = (x1, y1, x2, y2) after padding and clamping.
    """
    fh, fw = frame.shape[:2]
    xs = [lm.x * fw for lm in landmarks]
    ys = [lm.y * fh for lm in landmarks]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Add padding
    w = x2 - x1
    h = y2 - y1
    x1 -= w * padding
    x2 += w * padding
    y1 -= h * padding
    y2 += h * padding

    # Clamp to frame
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(fw, int(x2))
    y2 = min(fh, int(y2))

    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
        return None, None

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def get_haar_face_crop(frame, face_cascade):
    """Extract face crop using Haar cascade (largest face).

    Returns (face_crop, bbox_tuple) or (None, None) if no face detected.
    bbox_tuple = (x1, y1, x2, y2).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return frame[y:y+h, x:x+w], (x, y, x+w, y+h)


def run_hs(recognizer, crop):
    """Run HSEmotion on a crop. Returns (dominant, arousal, valence, emotion_dict)."""
    emotion, scores = recognizer.predict_emotions(crop, logits=False)
    arousal = float(scores[-1])
    valence = float(scores[-2])
    emotions = {}
    for i, label in enumerate(EMOTION_LABELS):
        if i < len(scores) - 2:
            emotions[label] = float(scores[i])
    dominant = emotion if emotion else max(emotions, key=emotions.get)
    return dominant, arousal, valence, emotions


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate HSEmotion: Haar crop vs MP landmark crop"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Path to video file")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING,
                        help=f"MP bbox padding fraction (default: {DEFAULT_PADDING})")
    parser.add_argument("--sweep", action="store_true",
                        help="Test multiple padding values (10%%,15%%,20%%,25%%)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit frames to process (0 = all)")
    args = parser.parse_args()

    ensure_model()

    # Output directory
    out_dir = os.path.join(SCRIPT_DIR, "logs", "validation")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Initialize models ────────────────────────────────────────────────
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
        output_face_blendshapes=False,
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

    # ── Open video ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open '{args.video}'.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video} @ {video_fps:.1f} fps, {total_frames} frames")

    paddings = PADDING_SWEEP if args.sweep else [args.padding]
    print(f"Padding values to test: {[f'{p:.0%}' for p in paddings]}")
    print("=" * 60)

    # ── Per-padding accumulators ─────────────────────────────────────────
    results = {p: [] for p in paddings}

    frame_count = 0
    both_count = 0
    mp_only = 0
    haar_only = 0
    neither = 0
    skipped_small = 0
    t_start = time.perf_counter()

    # ── Frame loop ───────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break

        elapsed = frame_count / video_fps

        # Haar detection
        haar_crop, haar_bbox = get_haar_face_crop(frame, face_cascade)

        # MediaPipe detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int(frame_count * 1000 / video_fps)
        mp_results = landmarker.detect_for_video(mp_image, ts_ms)

        mp_has_face = (mp_results.face_landmarks
                       and len(mp_results.face_landmarks) > 0)

        if haar_crop is None and not mp_has_face:
            neither += 1
            continue
        if haar_crop is None:
            mp_only += 1
            continue
        if not mp_has_face:
            haar_only += 1
            continue

        # Both detected — run HS on Haar crop once
        haar_dom, haar_arousal, haar_valence, haar_emotions = run_hs(
            recognizer, haar_crop
        )

        both_count += 1
        lms = mp_results.face_landmarks[0]

        # Run HS on each padding level
        for pad in paddings:
            mp_crop, mp_bbox = get_mp_face_crop(frame, lms, padding=pad)
            if mp_crop is None:
                skipped_small += 1
                continue

            mp_dom, mp_arousal, mp_valence, mp_emotions = run_hs(
                recognizer, mp_crop
            )

            haar_area = (haar_bbox[2] - haar_bbox[0]) * (haar_bbox[3] - haar_bbox[1])
            mp_area = (mp_bbox[2] - mp_bbox[0]) * (mp_bbox[3] - mp_bbox[1])
            crop_ratio = mp_area / haar_area if haar_area > 0 else 0

            results[pad].append({
                "frame": frame_count,
                "timestamp": f"{elapsed:.3f}",
                "haar_arousal": round(haar_arousal, 4),
                "mp_arousal": round(mp_arousal, 4),
                "arousal_diff": round(mp_arousal - haar_arousal, 4),
                "haar_valence": round(haar_valence, 4),
                "mp_valence": round(mp_valence, 4),
                "valence_diff": round(mp_valence - haar_valence, 4),
                "haar_dominant": haar_dom,
                "mp_dominant": mp_dom,
                "dominant_agree": 1 if haar_dom == mp_dom else 0,
                "crop_size_ratio": round(crop_ratio, 3),
            })

        # Progress
        if frame_count % 200 == 0:
            pct = frame_count / total_frames * 100 if total_frames else 0
            print(f"\r  Frame {frame_count}/{total_frames} ({pct:.0f}%) | "
                  f"Both: {both_count} | Haar-only: {haar_only} | "
                  f"MP-only: {mp_only}", end="", flush=True)

    cap.release()
    landmarker.close()

    runtime = time.perf_counter() - t_start
    print(f"\n\nProcessing done in {runtime:.1f}s ({frame_count} frames)")

    # ── Analysis ─────────────────────────────────────────────────────────
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    print(f"\n{'=' * 60}")
    print("CROP VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total frames processed: {frame_count}")
    print(f"Both detected:  {both_count}")
    print(f"MP-only:        {mp_only}")
    print(f"Haar-only:      {haar_only}")
    print(f"Neither:        {neither}")
    print(f"Skipped (small): {skipped_small}")

    best_padding = None
    best_r = -1
    all_stats = {}

    for pad in paddings:
        rows = results[pad]
        if len(rows) < 10:
            print(f"\nPadding {pad:.0%}: Too few samples ({len(rows)}), skipping")
            continue

        df = pd.DataFrame(rows)

        # Save CSV
        csv_name = f"{timestamp}_crop_validation_pad{int(pad * 100)}.csv"
        csv_path = os.path.join(out_dir, csv_name)
        df.to_csv(csv_path, index=False)

        # Statistics
        r_arousal, p_arousal = stats.pearsonr(df["haar_arousal"], df["mp_arousal"])
        r_valence, p_valence = stats.pearsonr(df["haar_valence"], df["mp_valence"])
        mad_arousal = df["arousal_diff"].abs().mean()
        mad_valence = df["valence_diff"].abs().mean()
        dominant_agree_pct = df["dominant_agree"].mean()
        mean_crop_ratio = df["crop_size_ratio"].mean()

        if r_arousal > best_r:
            best_r = r_arousal
            best_padding = pad

        checks = {
            f"Arousal r > {PASS_AROUSAL_R}": r_arousal > PASS_AROUSAL_R,
            f"Arousal MAD < {PASS_AROUSAL_MAD}": mad_arousal < PASS_AROUSAL_MAD,
            f"Valence r > {PASS_VALENCE_R}": r_valence > PASS_VALENCE_R,
            f"Valence MAD < {PASS_VALENCE_MAD}": mad_valence < PASS_VALENCE_MAD,
            f"Dominant agree > {PASS_DOMINANT_AGREE:.0%}": dominant_agree_pct > PASS_DOMINANT_AGREE,
        }
        verdict = "PASS" if all(checks.values()) else "FAIL"

        all_stats[pad] = {
            "n": len(rows),
            "r_arousal": r_arousal,
            "p_arousal": p_arousal,
            "mad_arousal": mad_arousal,
            "r_valence": r_valence,
            "p_valence": p_valence,
            "mad_valence": mad_valence,
            "dominant_agree": dominant_agree_pct,
            "mean_crop_ratio": mean_crop_ratio,
            "checks": checks,
            "verdict": verdict,
        }

        print(f"\n{'-' * 60}")
        print(f"Padding: {pad:.0%} ({len(rows)} frames)")
        print(f"  Arousal  — r={r_arousal:.4f} (p={p_arousal:.2e}), MAD={mad_arousal:.4f}")
        print(f"  Valence  — r={r_valence:.4f} (p={p_valence:.2e}), MAD={mad_valence:.4f}")
        print(f"  Dominant — agree={dominant_agree_pct:.1%}")
        print(f"  Crop size ratio — mean={mean_crop_ratio:.2f}")
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {check_name}")
        print(f"  >>> VERDICT: {verdict}")

    if best_padding is None:
        print("\nNo padding produced enough samples. Check video/detectors.")
        return

    print(f"\n{'=' * 60}")
    print(f"BEST PADDING: {best_padding:.0%} (arousal r={best_r:.4f})")
    print(f"{'=' * 60}")

    # ── Plots (4-panel, using best padding data) ─────────────────────────
    df = pd.DataFrame(results[best_padding])
    st = all_stats[best_padding]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#1a1a2e")
    fig.suptitle(
        f"Haar vs MP Crop Validation — Padding {best_padding:.0%} "
        f"(n={st['n']}, verdict={st['verdict']})",
        color="white", fontsize=14, fontweight="bold",
    )

    # Panel 1: Arousal scatter
    ax = axes[0, 0]
    ax.set_facecolor("#16213e")
    ax.scatter(df["haar_arousal"], df["mp_arousal"], alpha=0.3, s=8, c="#ff8844")
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="y=x")
    ax.set_xlabel("Haar Arousal", color="white")
    ax.set_ylabel("MP Arousal", color="white")
    ax.set_title(f"Arousal (r={st['r_arousal']:.3f}, MAD={st['mad_arousal']:.3f})",
                 color="white")
    ax.tick_params(colors="white")
    ax.legend()

    # Panel 2: Arousal difference histogram
    ax = axes[0, 1]
    ax.set_facecolor("#16213e")
    ax.hist(df["arousal_diff"], bins=50, color="#44aaff", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    mean_diff = df["arousal_diff"].mean()
    ax.axvline(mean_diff, color="yellow", linestyle=":", label=f"mean={mean_diff:.3f}")
    ax.set_xlabel("Arousal Diff (MP - Haar)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title(f"Arousal Difference Distribution", color="white")
    ax.tick_params(colors="white")
    ax.legend()

    # Panel 3: Valence scatter
    ax = axes[1, 0]
    ax.set_facecolor("#16213e")
    ax.scatter(df["haar_valence"], df["mp_valence"], alpha=0.3, s=8, c="#44ff88")
    ax.plot([-1, 1], [-1, 1], "r--", alpha=0.5)
    ax.set_xlabel("Haar Valence", color="white")
    ax.set_ylabel("MP Valence", color="white")
    ax.set_title(f"Valence (r={st['r_valence']:.3f}, MAD={st['mad_valence']:.3f})",
                 color="white")
    ax.tick_params(colors="white")

    # Panel 4: Dominant emotion agreement over time
    ax = axes[1, 1]
    ax.set_facecolor("#16213e")
    t = df["timestamp"].astype(float)
    agree_rolling = df["dominant_agree"].rolling(50, min_periods=1).mean()
    ax.plot(t, agree_rolling, color="#cc44ff", linewidth=1.2)
    ax.axhline(PASS_DOMINANT_AGREE, color="red", linestyle="--", alpha=0.5,
               label=f"Threshold ({PASS_DOMINANT_AGREE:.0%})")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Agreement (rolling 50)", color="white")
    ax.set_title("Dominant Emotion Agreement Over Time", color="white")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="white")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir,
                             f"{timestamp}_crop_validation_pad{int(best_padding * 100)}.png")
    fig.savefig(plot_path, dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    # ── If sweep, add padding comparison bar chart ───────────────────────
    if args.sweep and len(all_stats) > 1:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1a1a2e")
        fig2.suptitle("Padding Sweep Comparison", color="white",
                      fontsize=14, fontweight="bold")

        pads_sorted = sorted(all_stats.keys())
        x_labels = [f"{p:.0%}" for p in pads_sorted]
        x_pos = range(len(pads_sorted))

        # Arousal r
        ax = axes2[0]
        ax.set_facecolor("#16213e")
        vals = [all_stats[p]["r_arousal"] for p in pads_sorted]
        bars = ax.bar(x_pos, vals, color="#ff8844", alpha=0.8)
        ax.axhline(PASS_AROUSAL_R, color="red", linestyle="--", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, color="white")
        ax.set_ylabel("Pearson r", color="white")
        ax.set_title("Arousal Correlation", color="white")
        ax.tick_params(colors="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", color="white", fontsize=9)

        # Arousal MAD
        ax = axes2[1]
        ax.set_facecolor("#16213e")
        vals = [all_stats[p]["mad_arousal"] for p in pads_sorted]
        bars = ax.bar(x_pos, vals, color="#44aaff", alpha=0.8)
        ax.axhline(PASS_AROUSAL_MAD, color="red", linestyle="--", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, color="white")
        ax.set_ylabel("Mean Abs Diff", color="white")
        ax.set_title("Arousal MAD", color="white")
        ax.tick_params(colors="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", color="white", fontsize=9)

        # Dominant agreement
        ax = axes2[2]
        ax.set_facecolor("#16213e")
        vals = [all_stats[p]["dominant_agree"] for p in pads_sorted]
        bars = ax.bar(x_pos, vals, color="#44ff88", alpha=0.8)
        ax.axhline(PASS_DOMINANT_AGREE, color="red", linestyle="--", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, color="white")
        ax.set_ylabel("Agreement %", color="white")
        ax.set_title("Dominant Emotion Agreement", color="white")
        ax.tick_params(colors="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", color="white", fontsize=9)

        plt.tight_layout()
        sweep_path = os.path.join(out_dir, f"{timestamp}_crop_validation_sweep.png")
        fig2.savefig(sweep_path, dpi=150, facecolor="#1a1a2e")
        plt.close()
        print(f"Sweep plot saved: {sweep_path}")

    # ── Summary text ─────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, f"{timestamp}_crop_validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CROP VALIDATION SUMMARY\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Total frames: {frame_count}\n")
        f.write(f"Both detected: {both_count}\n")
        f.write(f"MP-only: {mp_only} | Haar-only: {haar_only} | Neither: {neither}\n")
        f.write(f"Runtime: {runtime:.1f}s\n\n")

        for pad in sorted(all_stats.keys()):
            s = all_stats[pad]
            f.write(f"{'-' * 60}\n")
            f.write(f"Padding: {pad:.0%} (n={s['n']})\n")
            f.write(f"  Arousal  — r={s['r_arousal']:.4f}, MAD={s['mad_arousal']:.4f}\n")
            f.write(f"  Valence  — r={s['r_valence']:.4f}, MAD={s['mad_valence']:.4f}\n")
            f.write(f"  Dominant — agree={s['dominant_agree']:.1%}\n")
            f.write(f"  Crop ratio — mean={s['mean_crop_ratio']:.2f}\n")
            for check_name, passed in s["checks"].items():
                status = "PASS" if passed else "FAIL"
                f.write(f"    [{status}] {check_name}\n")
            f.write(f"  >>> VERDICT: {s['verdict']}\n\n")

        f.write(f"{'=' * 60}\n")
        f.write(f"BEST PADDING: {best_padding:.0%} (arousal r={best_r:.4f})\n")
        f.write(f"OVERALL: {all_stats[best_padding]['verdict']}\n")

    print(f"Summary saved: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
