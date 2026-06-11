"""Standalone rPPG feasibility test.

Pattern: [Script] — self-contained smoke test for CHROM rPPG.  Signal
processing functions are imported from rppg_algorithms (no duplication).

Usage:
    python smoke_test_rppg.py --video <path> [--duration 60] [--show]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PIPELINE = Path(__file__).resolve().parent.parent  # Pipeline/
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from rppg.rppg_algorithms import chrom, estimate_bpm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="rPPG smoke test: CHROM algorithm on a video file")
    parser.add_argument("--video",    required=True, help="Path to video file")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Max seconds to process (default: 60)")
    parser.add_argument("--window",   type=float, default=30.0,
                        help="BPM estimation window in seconds (default: 30)")
    parser.add_argument("--show",     action="store_true",
                        help="Show live face ROI during processing")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"ERROR: File not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    max_frames = int(args.duration * fps)

    print(f"Video: {video_path.name}")
    print(f"  FPS: {fps:.1f}  |  Total: {total_frames} frames ({total_duration:.1f}s)")
    print(f"  Processing first {args.duration:.0f}s ({max_frames} frames)")

    haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar)

    rgbs: list = []
    timestamps: list = []
    face_detected_count = 0
    t0 = time.time()
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            roi_y1 = y + int(0.15 * h)
            roi_y2 = y + int(0.55 * h)
            roi_x1 = x + int(0.20 * w)
            roi_x2 = x + int(0.80 * w)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            if roi.size > 0:
                mean_bgr = roi.mean(axis=(0, 1))
                rgbs.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
                timestamps.append(t)
                face_detected_count += 1

            if args.show:
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2),
                              (0, 255, 0), 2)
                cv2.imshow("rPPG ROI", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    detection_rate = face_detected_count / frame_idx * 100 if frame_idx > 0 else 0

    print(f"\n  Processed: {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / elapsed:.1f} FPS)")
    print(f"  Face detected: {face_detected_count}/{frame_idx} "
          f"({detection_rate:.1f}%)")

    if len(rgbs) < int(fps * 10):
        print(f"\n  ERROR: Only {len(rgbs)} face frames collected. "
              f"Need at least {int(fps * 10)} (10s) for BPM estimation.")
        return

    rgbs_arr = np.array(rgbs)
    timestamps_arr = np.array(timestamps)
    effective_fps = (len(rgbs_arr) / (timestamps_arr[-1] - timestamps_arr[0])
                     if len(timestamps_arr) > 1 else fps)

    print(f"  Effective FPS (face frames): {effective_fps:.1f}")
    print(f"\n  Mean RGB: R={rgbs_arr[:, 0].mean():.1f}  "
          f"G={rgbs_arr[:, 1].mean():.1f}  B={rgbs_arr[:, 2].mean():.1f}")

    print("\n--- CHROM Algorithm ---")
    winsize = max(int(effective_fps * 1.5), 15)
    rppg_signal = chrom(rgbs_arr, winsize=winsize)

    window_frames = int(args.window * effective_fps)
    step_frames   = max(int(5 * effective_fps), 1)

    print(f"  Window: {args.window:.0f}s ({window_frames} frames)  "
          f"|  Step: 5s ({step_frames} frames)")

    bpm_results = []
    for start in range(0, len(rppg_signal) - window_frames + 1, step_frames):
        end = start + window_frames
        segment  = rppg_signal[start:end]
        t_center = timestamps_arr[start + window_frames // 2]
        bpm, _, _, _, _ = estimate_bpm(segment, effective_fps)
        bpm_results.append((t_center, bpm))

    if not bpm_results:
        print("  ERROR: Not enough data for BPM estimation")
        return

    print(f"\n  BPM estimates ({len(bpm_results)} windows):")
    print(f"  {'Time (s)':>10}  {'BPM':>8}")
    print(f"  {'-' * 10}  {'-' * 8}")
    for t, bpm in bpm_results:
        flag = " *" if bpm < 50 or bpm > 150 else ""
        print(f"  {t:10.1f}  {bpm:8.1f}{flag}")

    bpms     = [b for _, b in bpm_results]
    mean_bpm = np.mean(bpms)
    std_bpm  = np.std(bpms)

    print(f"\n  Summary:")
    print(f"    Mean BPM: {mean_bpm:.1f} +/- {std_bpm:.1f}")
    print(f"    Range: {np.min(bpms):.1f} - {np.max(bpms):.1f}")
    print(f"    Plausible (50-150 BPM): "
          f"{sum(1 for b in bpms if 50 <= b <= 150)}/{len(bpms)} windows")

    if 50 <= mean_bpm <= 120:
        print(f"\n  PASS: Mean BPM {mean_bpm:.1f} is in plausible resting range")
    elif 50 <= mean_bpm <= 150:
        print(f"\n  PASS: Mean BPM {mean_bpm:.1f} is plausible (elevated)")
    else:
        print(f"\n  WARNING: Mean BPM {mean_bpm:.1f} outside plausible range (50-150)")
        print("  This may indicate poor signal quality or lighting conditions")

    status = "FEASIBLE" if 50 <= mean_bpm <= 150 else "NEEDS INVESTIGATION"
    print(f"\n  Conclusion: rPPG signal extraction {status}")


if __name__ == "__main__":
    main()
