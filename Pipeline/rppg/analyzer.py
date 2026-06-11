"""File-I/O wrapper around compute_bpm_timeseries.

Pattern: [Utility] — thin layer; all signal-processing logic lives in
rppg_algorithms.py.  RppgAnalyzer only handles loading frames.csv, running
the algorithm, and writing analysis.csv.

Usage:
    from rppg.analyzer import RppgAnalyzer, AnalysisConfig
    cfg = AnalysisConfig()
    analyzer = RppgAnalyzer()
    results = analyzer.analyze(Path("pipeline/sessions/s01/frames.csv"), cfg)
    analyzer.save(results, Path("pipeline/sessions/s01/analysis.csv"))
"""

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from rppg.rppg_algorithms import compute_bpm_timeseries


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    algorithm: str  = "all"
    window_s: float = 30.0
    step_s: float   = 5.0
    motion_gate: bool = True           # use motion detection when loading frames
    snr_consensus_threshold: float = 2.0  # minimum SNR to vote in CONSENSUS
    roi_mode: str = "primary"          # "primary" | "forehead" | "glabella" | "malar"
    bpm_min: int = 60                  # raised from 42 to exclude breathing artifacts
    bpm_max: int = 180
    harmonic_disambiguation: bool = True   # check 2*f when dominant peak may be sub-harmonic
    harmonic_ratio: float = 0.6            # power threshold for preferring the 2x harmonic


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class RppgAnalyzer:

    def analyze(self, frames_csv: Path, config: Optional[AnalysisConfig] = None) -> List[dict]:
        """Load frames.csv and compute BPM timeseries.

        Returns list of window dicts (same schema as compute_bpm_timeseries).
        Returns empty list if fewer than 2 face-detected frames.
        """
        config = config or AnalysisConfig()
        rgbs, timestamps, motion_flags, session_start_utc = self._load_frames(
            frames_csv, roi_mode=config.roi_mode
        )

        if len(rgbs) < 2:
            print(f"[Analyzer] Not enough face frames in {frames_csv} — skipping.")
            return []

        duration = timestamps[-1] - timestamps[0]
        fps_eff = len(rgbs) / duration if duration > 0 else 30.0

        m_flags = motion_flags if config.motion_gate else None

        results = compute_bpm_timeseries(
            rgbs, timestamps, fps_eff,
            algorithm=config.algorithm,
            window_s=config.window_s,
            step_s=config.step_s,
            motion_flags=m_flags,
            snr_consensus_threshold=config.snr_consensus_threshold,
            low_bpm=config.bpm_min,
            high_bpm=config.bpm_max,
            harmonic_disambiguation=config.harmonic_disambiguation,
            harmonic_ratio=config.harmonic_ratio,
        )

        # Attach session metadata for downstream use
        for row in results:
            row["session_start_utc"] = session_start_utc

        return results

    def save(self, results: List[dict], output_path: Path) -> None:
        if not results:
            return
        output_path = Path(output_path)
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[Analyzer] Saved {len(results)} window rows → {output_path.name}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_frames(frames_csv: Path, roi_mode: str = "primary"):
        """Parse frames.csv.

        Args:
            frames_csv: path to frames.csv
            roi_mode: which RGB columns to use — "primary" (r/g/b), "forehead",
                      "glabella", or "malar". Falls back to primary if columns absent.

        Returns:
            rgbs (Nx3 ndarray), timestamps (N ndarray),
            motion_flags (N bool ndarray — True when head motion detected),
            session_start_utc (str)
        """
        frames_csv = Path(frames_csv)
        session_start_utc = ""
        rows = []
        bbox_strs = []

        with open(frames_csv, newline="", encoding="utf-8") as f:
            first_line = f.readline()
            # Check for metadata comment header
            if first_line.startswith("# session_start_utc="):
                session_start_utc = first_line.split("=", 1)[1].strip()
                content = first_line + f.read()
            else:
                content = first_line + f.read()

        # Re-parse through csv.DictReader skipping comment lines
        lines = [ln for ln in content.splitlines() if not ln.strip().startswith("#")]
        reader = csv.DictReader(lines)

        # Determine RGB column names based on roi_mode
        _roi_col_map = {
            "forehead": ("r_forehead", "g_forehead", "b_forehead"),
            "glabella":  ("r_glabella", "g_glabella", "b_glabella"),
            "malar":     ("r_malar",    "g_malar",    "b_malar"),
        }
        roi_cols = _roi_col_map.get(roi_mode)  # None means use primary r/g/b

        for row in reader:
            try:
                face_ok = int(row.get("face_detected", 0)) == 1
                if not face_ok:
                    continue

                # Select RGB columns by roi_mode; fall back to primary if absent
                if roi_cols and all(c in row for c in roi_cols):
                    r = float(row[roi_cols[0]])
                    g = float(row[roi_cols[1]])
                    b = float(row[roi_cols[2]])
                    # Skip frames where this ROI was unavailable (zero-filled)
                    if r == 0.0 and g == 0.0 and b == 0.0:
                        continue
                else:
                    if roi_mode != "primary" and roi_cols is not None:
                        # First miss only — roi_cols=None acts as "already warned" flag
                        warnings.warn(
                            f"[Analyzer] roi_mode='{roi_mode}' columns not found in "
                            f"{frames_csv.name} — falling back to primary r/g/b",
                            stacklevel=2,
                        )
                        roi_cols = None
                    r = float(row["r"])
                    g = float(row["g"])
                    b = float(row["b"])

                ts = float(row["timestamp"])
                rows.append((ts, r, g, b))
                bbox_strs.append(row.get("bbox", ""))
            except (KeyError, ValueError):
                continue

        if not rows:
            return np.zeros((0, 3)), np.array([]), np.array([], dtype=bool), session_start_utc

        arr = np.array(rows, dtype=float)       # Nx4: ts, r, g, b
        timestamps = arr[:, 0]
        rgbs = arr[:, 1:4]

        # Motion detection: bbox centroid displacement (head movement proxy)
        motion_flags = _detect_motion_from_bboxes(bbox_strs, len(rows))

        return rgbs, timestamps, motion_flags, session_start_utc


def _detect_motion_from_bboxes(bbox_strs: list, n: int) -> np.ndarray:
    """Detect head motion using bbox centroid displacement between frames.

    Falls back to RGB-delta-free all-False array if bboxes are missing or
    unparseable for >50% of frames.

    Returns:
        motion_flags: N-length bool array (True = motion-corrupted frame)
    """
    motion_flags = np.zeros(n, dtype=bool)
    if n < 2:
        return motion_flags

    centroids = []
    for s in bbox_strs:
        try:
            parts = [float(x) for x in str(s).split(",")]
            if len(parts) == 4:
                cx = (parts[0] + parts[2]) / 2.0
                cy = (parts[1] + parts[3]) / 2.0
                centroids.append((cx, cy))
            else:
                centroids.append(None)
        except (ValueError, AttributeError):
            centroids.append(None)

    valid = [c for c in centroids if c is not None]
    if len(valid) < n * 0.5:
        # Fewer than half the frames have parseable bboxes — skip motion detection
        return motion_flags

    # Fill gaps (None entries) with nearest valid centroid
    filled = list(centroids)
    last_valid = valid[0]
    for i, c in enumerate(filled):
        if c is None:
            filled[i] = last_valid
        else:
            last_valid = c

    pts = np.array(filled, dtype=float)               # Nx2
    displacements = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))  # (N-1,)
    threshold = np.median(displacements) + 2.5 * np.std(displacements)
    motion_flags[1:] = displacements > threshold
    return motion_flags
