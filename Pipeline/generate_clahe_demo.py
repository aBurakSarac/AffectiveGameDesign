"""generate_clahe_demo.py — Create real CLAHE before/after screenshots for the website.

Extracts a face crop from a dim session, saves raw and CLAHE-enhanced PNGs,
and computes luminance histograms for the website's ClaheDemo wiper.

Usage:
    python Pipeline/generate_clahe_demo.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
PRES_DIR = _HERE / "presentation"
OUT_DIR = _HERE.parent / "Website" / "media" / "clahe"

STEM = "S02_Vid04"
TARGET_FRAME_IDX = 9000

CLAHE_CLIP = 2.0
CLAHE_GRID = (4, 4)
HIST_BINS = 16


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pres = PRES_DIR / STEM

    frames_csv = pres / "frames.csv"
    video_path = pres / "raw_video.mp4"
    if not video_path.exists():
        video_path = Path(_HERE.parent / "Recordings" / f"{STEM}.mp4")

    print(f"Loading frames.csv from {frames_csv}")
    df = pd.read_csv(frames_csv, comment="#")

    detected = df[df["face_detected"] == 1]
    if len(detected) == 0:
        print("No face-detected frames found!")
        return

    closest = detected.iloc[(detected["frame_idx"] - TARGET_FRAME_IDX).abs().argsort()[:1]]
    fidx = int(closest["frame_idx"].values[0])
    bbox_str = str(closest["bbox"].values[0])

    print(f"Using frame {fidx}, bbox: {bbox_str}")

    parts = bbox_str.split(",")
    x1, y1, x2, y2 = [int(float(p)) for p in parts[:4]]

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to read frame {fidx}")
        return

    h, w = frame.shape[:2]
    pad = 8
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    face_raw = frame[cy1:cy2, cx1:cx2].copy()

    if face_raw.size == 0:
        print("Face crop is empty!")
        return

    scale = max(1, 400 // max(face_raw.shape[:2]))
    if scale > 1:
        face_raw = cv2.resize(face_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    lab = cv2.cvtColor(face_raw, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    face_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    off_path = OUT_DIR / "clahe_off.png"
    on_path = OUT_DIR / "clahe_on.png"
    cv2.imwrite(str(off_path), face_raw)
    cv2.imwrite(str(on_path), face_clahe)
    print(f"Saved {off_path.name} ({face_raw.shape[1]}x{face_raw.shape[0]})")
    print(f"Saved {on_path.name} ({face_clahe.shape[1]}x{face_clahe.shape[0]})")

    gray_raw = cv2.cvtColor(face_raw, cv2.COLOR_BGR2GRAY)
    gray_clahe = cv2.cvtColor(face_clahe, cv2.COLOR_BGR2GRAY)

    hist_raw = cv2.calcHist([gray_raw], [0], None, [HIST_BINS], [0, 256]).flatten()
    hist_clahe = cv2.calcHist([gray_clahe], [0], None, [HIST_BINS], [0, 256]).flatten()

    total_raw = hist_raw.sum() or 1
    total_clahe = hist_clahe.sum() or 1
    hist_raw_norm = [round(float(v / total_raw * 100), 1) for v in hist_raw]
    hist_clahe_norm = [round(float(v / total_clahe * 100), 1) for v in hist_clahe]

    print(f"\nHIST_BEFORE (raw):   {hist_raw_norm}")
    print(f"HIST_AFTER  (CLAHE): {hist_clahe_norm}")

    meta = {
        "stem": STEM,
        "frame_idx": fidx,
        "bbox": [x1, y1, x2, y2],
        "clahe_clip": CLAHE_CLIP,
        "clahe_grid": list(CLAHE_GRID),
        "hist_before": hist_raw_norm,
        "hist_after": hist_clahe_norm,
    }
    meta_path = OUT_DIR / "clahe_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == "__main__":
    main()
