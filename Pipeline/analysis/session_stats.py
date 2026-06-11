"""
Session Stats — Quick face detection and signal quality summary.

Usage:
    python -m analysis.session_stats <csv_path>
    python -m analysis.session_stats Pipeline/logs/sessions/S07_Vid17_bright/*.csv
"""

import sys
import pandas as pd
from pathlib import Path


def session_stats(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    total = len(df)
    if total == 0:
        print(f"  Empty CSV: {csv_path}")
        return {}

    stats = {"total_frames": total, "csv": Path(csv_path).name}

    if "mp_face_detected" in df.columns:
        mp_det = df["mp_face_detected"].astype(int).sum()
        stats["mp_face_detected"] = mp_det
        stats["mp_face_pct"] = 100.0 * mp_det / total

    if "hs_face_detected" in df.columns:
        hs_det = df["hs_face_detected"].astype(int).sum()
        stats["hs_face_detected"] = hs_det
        stats["hs_face_pct"] = 100.0 * hs_det / total

    if "mp_face_detected" in df.columns and "hs_face_detected" in df.columns:
        both = ((df["mp_face_detected"].astype(int) == 1) &
                (df["hs_face_detected"].astype(int) == 1)).sum()
        stats["both_detected"] = int(both)
        stats["both_pct"] = 100.0 * both / total

    for col in ["hs_fear", "hs_arousal", "hs_surprise", "hs_anger", "mp_tension"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            stats[f"{col}_mean"] = vals.mean()
            stats[f"{col}_std"] = vals.std()
            stats[f"{col}_max"] = vals.max()

    duration_s = total / 30.0
    stats["duration_s"] = duration_s

    return stats


def print_stats(stats: dict):
    if not stats:
        return
    print(f"\n{'='*60}")
    print(f"  SESSION STATS: {stats.get('csv', '?')}")
    print(f"{'='*60}")
    print(f"  Frames: {stats['total_frames']:,}  ({stats.get('duration_s', 0):.0f}s ≈ {stats.get('duration_s', 0)/60:.1f} min)")
    print()
    print(f"  Face Detection:")
    if "mp_face_pct" in stats:
        print(f"    MediaPipe:  {stats['mp_face_detected']:,} / {stats['total_frames']:,}  ({stats['mp_face_pct']:.1f}%)")
    if "hs_face_pct" in stats:
        print(f"    HSEmotion:  {stats['hs_face_detected']:,} / {stats['total_frames']:,}  ({stats['hs_face_pct']:.1f}%)")
    if "both_pct" in stats:
        print(f"    Both:       {stats['both_detected']:,} / {stats['total_frames']:,}  ({stats['both_pct']:.1f}%)")
    print()
    print(f"  Signal Means (on detected frames):")
    for col in ["hs_fear", "hs_arousal", "hs_surprise", "hs_anger", "mp_tension"]:
        m = stats.get(f"{col}_mean")
        if m is not None:
            print(f"    {col:<16} mean={m:.4f}  std={stats[f'{col}_std']:.4f}  max={stats[f'{col}_max']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.session_stats <csv_path> [csv_path2 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        stats = session_stats(path)
        print_stats(stats)
