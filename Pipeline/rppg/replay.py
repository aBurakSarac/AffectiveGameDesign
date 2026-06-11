"""rPPG HUD video renderer — reconstructs the analysis session as a video.

Pattern: [Template Method] — BaseReplayRenderer.render() owns the frame loop
and VideoWriter lifecycle; subclasses override draw_hud_panel() to customise
the panel content.  RppgReplayRenderer fills in the rPPG BPM panel.

Usage:
    from rppg.replay import ReplayRenderer
    renderer = ReplayRenderer()
    renderer.render(Path("Pipeline/sessions/20260502_pilot01"), Path("replay.mp4"))
"""

import csv
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from rppg.hud_constants import (
    PANEL_WIDTH, HUD_MIN_HEIGHT,
    PANEL_BG, SECTION_DIVIDER, BAR_MAX_PX, BAR_HEIGHT, BAR_OUTLINE,
    ALGO_ORDER, ALGO_COLORS, TEXT_HEADER, TEXT_LABEL, TEXT_GT,
    TEXT_DELTA_POS, TEXT_DELTA_NEG, TEXT_WARN,
    BPM_LOW, BPM_HIGH, BPM_MIN_DISPLAY, BPM_MAX_DISPLAY,
)


# ---------------------------------------------------------------------------
# Base renderer (Template Method)
# ---------------------------------------------------------------------------

class BaseReplayRenderer(ABC):

    def render(self, video_path: Path, analysis_csv: Path, output_path: Path,
               gt_csv: Optional[Path] = None,
               frames_csv: Optional[Path] = None) -> None:
        """Frame loop — open video, draw HUD, write output.

        Subclasses implement draw_hud_panel().
        frames_csv: optional path to frames.csv for bbox/roi_source overlay.
        """
        video_path  = Path(video_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale     = HUD_MIN_HEIGHT / src_h
        vid_w     = int(src_w * scale)
        canvas_w  = vid_w + PANEL_WIDTH
        canvas_h  = HUD_MIN_HEIGHT

        writer = self._open_writer(output_path, src_fps, canvas_w, canvas_h)

        windows = self._load_analysis(analysis_csv)
        gt_map  = self._load_gt(gt_csv) if gt_csv and Path(gt_csv).exists() else {}
        frame_bbox_map = (
            self._load_frame_bboxes(Path(frames_csv))
            if frames_csv and Path(frames_csv).exists() else {}
        )

        # Pre-build frame→window and frame→gt lookups
        fps_float = src_fps if src_fps > 0 else 30.0
        frame_to_window = self._build_frame_window_map(windows, fps_float, n_frames)

        canvas  = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        resized = np.zeros((canvas_h, vid_w,    3), dtype=np.uint8)

        frame_idx = 0
        print(f"[Replay] Rendering {n_frames} frames → {output_path.name}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                canvas.fill(0)
                cv2.resize(frame, (vid_w, canvas_h), dst=resized)
                self._draw_roi_bbox(resized, frame_bbox_map.get(frame_idx),
                                    src_w, src_h, vid_w, canvas_h)
                canvas[:, :vid_w] = resized

                win_row = frame_to_window.get(frame_idx)
                t_center = float(win_row["t_center"]) if win_row else 0.0
                gt_bpm = gt_map.get(round(t_center, 1))

                self.draw_hud_panel(canvas, vid_w, canvas_h, win_row, gt_bpm, t_center)

                writer.write(canvas)

                if frame_idx % 100 == 0:
                    pct = 100 * frame_idx / max(n_frames, 1)
                    print(f"  {frame_idx}/{n_frames} ({pct:.0f}%)")

                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        print(f"[Replay] Done → {output_path}")

    @abstractmethod
    def draw_hud_panel(self, canvas: np.ndarray, panel_x: int, frame_h: int,
                        win_row: Optional[dict], gt_bpm: Optional[float],
                        t_center: float) -> None:
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_writer(path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
        for codec in ("avc1", "XVID", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            if writer.isOpened():
                return writer
            writer.release()
        raise RuntimeError(f"Could not open VideoWriter for {path}")

    @staticmethod
    def _load_analysis(csv_path: Path) -> List[dict]:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    @staticmethod
    def _load_gt(csv_path: Path) -> Dict[float, float]:
        """Build t_center → gt_bpm mapping from gt_aligned.csv."""
        mapping: Dict[float, float] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = round(float(row["t_center"]), 1)
                    gt = row.get("gt_bpm", "")
                    if gt and gt.lower() not in ("nan", "none", ""):
                        mapping[t] = float(gt)
                except (KeyError, ValueError):
                    continue
        return mapping

    @staticmethod
    def _load_frame_bboxes(frames_csv: Path) -> Dict[int, tuple]:
        """Build {frame_idx: (bbox_str, roi_source)} from frames.csv.

        Prefers roi_bbox (tight polygon bbox) over bbox (full face extent) when
        both columns are present, so the overlay shows the actual sampled region.
        """
        mapping: Dict[int, tuple] = {}
        try:
            with open(frames_csv, newline="", encoding="utf-8") as f:
                lines = [ln for ln in f if not ln.strip().startswith("#")]
            reader = csv.DictReader(lines)
            for row in reader:
                try:
                    idx = int(row["frame_idx"])
                    roi_bbox = row.get("roi_bbox", "").strip()
                    bbox     = row.get("bbox", "").strip()
                    bbox_str = roi_bbox if roi_bbox else bbox
                    roi_src  = row.get("roi_source", "")
                    mapping[idx] = (bbox_str, roi_src)
                except (KeyError, ValueError):
                    continue
        except OSError:
            pass
        return mapping

    @staticmethod
    def _draw_roi_bbox(
        frame: np.ndarray,
        bbox_entry: Optional[tuple],
        src_w: int, src_h: int,
        dst_w: int, dst_h: int,
    ) -> None:
        """Draw green bbox + roi_source label on a resized frame in-place."""
        if not bbox_entry:
            return
        bbox_str, roi_src = bbox_entry
        if not bbox_str:
            return
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox_str.split(",")]
        except (ValueError, AttributeError):
            return
        sx = dst_w / src_w if src_w > 0 else 1.0
        sy = dst_h / src_h if src_h > 0 else 1.0
        rx1, ry1 = int(x1 * sx), int(y1 * sy)
        rx2, ry2 = int(x2 * sx), int(y2 * sy)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 100), 1)
        if roi_src:
            cv2.putText(frame, roi_src, (rx1, max(8, ry1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

    @staticmethod
    def _build_frame_window_map(
        windows: List[dict], fps: float, n_frames: int
    ) -> Dict[int, dict]:
        """Map each frame index to the analysis window that covers it."""
        if not windows:
            return {}
        mapping: Dict[int, dict] = {}
        # Build sorted list of (t_start_frame, t_end_frame, row)
        intervals = []
        for row in windows:
            try:
                t_start = float(row["t_start"])
                t_end   = float(row["t_end"])
                f_start = int(t_start * fps)
                f_end   = int(t_end   * fps)
                intervals.append((f_start, f_end, row))
            except (KeyError, ValueError):
                continue

        for f_idx in range(n_frames):
            best = None
            for (f_start, f_end, row) in intervals:
                if f_start <= f_idx <= f_end:
                    best = row
                    break
            if best is None and intervals:
                # Nearest window (for frames before first window)
                f_start, _, row = min(intervals,
                    key=lambda iv: abs(iv[0] - f_idx))
                best = row
            if best is not None:
                mapping[f_idx] = best

        return mapping


# ---------------------------------------------------------------------------
# rPPG-focused HUD panel drawing
# ---------------------------------------------------------------------------

class RppgHudPanel:
    """Draws the rPPG BPM panel onto a canvas region."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, canvas: np.ndarray, panel_x: int, frame_h: int,
             win_row: Optional[dict], gt_bpm: Optional[float],
             t_center: float) -> None:
        pw = PANEL_WIDTH
        canvas[:frame_h, panel_x:panel_x + pw] = PANEL_BG

        px = panel_x + 10
        y  = 14

        # Header
        time_str = f"{int(t_center // 60):02d}:{int(t_center % 60):02d}"
        cv2.putText(canvas, f"rPPG BPM  t={time_str}",
                    (px, y), self.FONT, 0.55, TEXT_HEADER, 2)
        y += 20
        cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y),
                 SECTION_DIVIDER, 1)
        y += 10

        bar_x = panel_x + 120

        if win_row is None:
            cv2.putText(canvas, "Waiting for first window...",
                        (px, y + 14), self.FONT, 0.45, TEXT_LABEL, 1)
            return

        # Per-algorithm rows — iterate in fixed order, show each algo once
        seen_algos = set()
        for algo in ALGO_ORDER:
            # Find this algo's row (analysis.csv has one row per algorithm per window)
            bpm  = self._get_field(win_row, f"bpm",  algo)
            snr  = self._get_field(win_row, f"snr",  algo)

            # When analysis.csv is grouped (algorithm field present), filter
            row_algo = win_row.get("algorithm", "").upper()
            if row_algo and row_algo != algo:
                continue
            if algo in seen_algos:
                continue
            seen_algos.add(algo)

            if bpm is None:
                continue

            is_consensus = (algo == "CONSENSUS")
            in_range = BPM_LOW <= bpm <= BPM_HIGH
            color = ALGO_COLORS.get(algo, TEXT_LABEL)
            if not in_range:
                color = TEXT_WARN

            # Divider before CONSENSUS
            if is_consensus:
                cv2.line(canvas, (panel_x + 5, y - 2),
                         (panel_x + pw - 5, y - 2), SECTION_DIVIDER, 1)

            bar_len = self._bpm_to_bar(bpm)
            cv2.putText(canvas, f"{algo[:6]:6s} {bpm:5.1f}",
                        (px, y + BAR_HEIGHT - 2), self.FONT, 0.42, color, 1)
            cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_len, y + BAR_HEIGHT),
                          color, -1)
            cv2.rectangle(canvas, (bar_x, y), (bar_x + BAR_MAX_PX, y + BAR_HEIGHT),
                          BAR_OUTLINE, 1)
            if snr is not None:
                cv2.putText(canvas, f"{snr:.1f}",
                            (bar_x + BAR_MAX_PX + 5, y + BAR_HEIGHT - 1),
                            self.FONT, 0.38, (130, 130, 130), 1)
            y += BAR_HEIGHT + 6

        # GT HR line
        y += 6
        cv2.line(canvas, (panel_x + 5, y - 2), (panel_x + pw - 5, y - 2),
                 SECTION_DIVIDER, 1)
        y += 8
        if gt_bpm is not None and not math.isnan(gt_bpm):
            gt_bar = self._bpm_to_bar(gt_bpm)
            cv2.putText(canvas, f"GT HR  {gt_bpm:5.1f}",
                        (px, y + BAR_HEIGHT - 2), self.FONT, 0.45, TEXT_GT, 1)
            cv2.rectangle(canvas, (bar_x, y), (bar_x + gt_bar, y + BAR_HEIGHT),
                          TEXT_GT, 2)
            cv2.rectangle(canvas, (bar_x, y), (bar_x + BAR_MAX_PX, y + BAR_HEIGHT),
                          BAR_OUTLINE, 1)
            y += BAR_HEIGHT + 8

            # Delta: CONSENSUS vs GT
            consensus_bpm_raw = win_row.get("bpm") if win_row else None
            if consensus_bpm_raw is not None:
                try:
                    delta = float(consensus_bpm_raw) - gt_bpm
                    d_color = TEXT_DELTA_POS if delta >= 0 else TEXT_DELTA_NEG
                    cv2.putText(canvas, f"CONS vs GT  {delta:+.1f} BPM",
                                (px, y + 14), self.FONT, 0.44, d_color, 1)
                    y += 20
                except (ValueError, TypeError):
                    pass
        else:
            cv2.putText(canvas, "GT HR  (no data)",
                        (px, y + BAR_HEIGHT - 2), self.FONT, 0.44, (80, 80, 80), 1)

    # ------------------------------------------------------------------

    @staticmethod
    def _bpm_to_bar(bpm: float) -> int:
        ratio = (bpm - BPM_MIN_DISPLAY) / (BPM_MAX_DISPLAY - BPM_MIN_DISPLAY)
        return max(1, int(np.clip(ratio, 0.0, 1.0) * BAR_MAX_PX))

    @staticmethod
    def _get_field(row: dict, field: str, algo: str) -> Optional[float]:
        """Try to get a value from the row dict.  Returns None on missing/invalid."""
        val = row.get(field)
        if val is None or str(val).strip() in ("", "nan", "None"):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Concrete renderer
# ---------------------------------------------------------------------------

class ReplayRenderer(BaseReplayRenderer):
    """Renders a session to a replay video with the rPPG-focused HUD."""

    def __init__(self):
        self._panel = RppgHudPanel()
        # Each analysis.csv has one row per algorithm per window.
        # We accumulate the current window's algo rows to build a snapshot.
        self._current_window_idx: Optional[int] = None
        self._window_snapshot: dict = {}   # algo → {bpm, snr}

    def render(self, session_dir: Path, output_path: Path) -> None:
        """Convenience wrapper: auto-discovers artefacts in session_dir."""
        session_dir = Path(session_dir)
        video_path   = session_dir / "raw_video.mp4"
        analysis_csv = session_dir / "analysis.csv"
        gt_csv       = session_dir / "gt_aligned.csv"

        if not video_path.exists():
            raise FileNotFoundError(f"raw_video.mp4 not found in {session_dir}")
        if not analysis_csv.exists():
            raise FileNotFoundError(f"analysis.csv not found in {session_dir}")

        gt_arg = gt_csv if gt_csv.exists() else None
        frames_csv = session_dir / "frames.csv"

        # Load analysis and build per-window snapshot index
        all_windows = self._load_analysis(analysis_csv)
        self._snapshot_index = self._build_snapshot_index(all_windows)

        super().render(video_path, analysis_csv, output_path, gt_arg,
                       frames_csv=frames_csv if frames_csv.exists() else None)

    def draw_hud_panel(self, canvas: np.ndarray, panel_x: int, frame_h: int,
                        win_row: Optional[dict], gt_bpm: Optional[float],
                        t_center: float) -> None:
        # win_row here is the row for the CONSENSUS algorithm in the current window.
        # We want to show all algorithms, so we pass the full snapshot.
        snapshot_row = self._get_snapshot_row(win_row)
        self._panel.draw(canvas, panel_x, frame_h, snapshot_row, gt_bpm, t_center)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_snapshot_index(windows: List[dict]) -> dict:
        """Build {window_idx: {algo: {bpm, snr}}} from analysis.csv rows."""
        from collections import defaultdict
        index: dict = defaultdict(dict)
        for row in windows:
            try:
                widx = int(row.get("window_idx", 0))
                algo = row.get("algorithm", "?").upper()
                bpm  = float(row.get("bpm_smoothed") or row.get("bpm", 0))
                snr  = float(row.get("snr", 0))
                index[widx][algo] = {"bpm": bpm, "snr": snr,
                                     "t_start": row.get("t_start"),
                                     "t_end":   row.get("t_end"),
                                     "t_center": row.get("t_center")}
            except (KeyError, ValueError):
                continue
        return dict(index)

    def _get_snapshot_row(self, win_row: Optional[dict]) -> Optional[dict]:
        """Build a synthetic row mapping algo → bpm for RppgHudPanel.draw()."""
        if win_row is None:
            return None
        try:
            widx = int(win_row.get("window_idx", 0))
        except (ValueError, TypeError):
            return win_row

        snap = self._snapshot_index.get(widx, {})
        if not snap:
            return win_row

        # Build a flat dict that RppgHudPanel can iterate
        # We return the CONSENSUS row as base and attach per-algo bpm sub-keys
        consensus = snap.get("CONSENSUS", {})
        out = dict(win_row)
        for algo, data in snap.items():
            out[f"bpm_{algo}"] = data["bpm"]
            out[f"snr_{algo}"] = data["snr"]
        return out

    def draw_hud_panel(self, canvas, panel_x, frame_h, win_row, gt_bpm, t_center):
        # Override: build per-algo snapshot from index and pass to panel
        if win_row is None:
            self._panel.draw(canvas, panel_x, frame_h, None, gt_bpm, t_center)
            return

        try:
            widx = int(win_row.get("window_idx", -1))
        except (ValueError, TypeError):
            widx = -1

        snap = self._snapshot_index.get(widx, {})

        # Pass snapshot dict so panel can iterate ALGO_ORDER
        self._panel.draw_snapshot(canvas, panel_x, frame_h, snap, gt_bpm, t_center)


# ---------------------------------------------------------------------------
# Extend RppgHudPanel with snapshot-aware draw
# ---------------------------------------------------------------------------

def _draw_snapshot(
    self: RppgHudPanel,
    canvas: np.ndarray, panel_x: int, frame_h: int,
    snap: dict, gt_bpm: Optional[float], t_center: float,
) -> None:
    """Draw panel from a {algo: {bpm, snr}} snapshot dict."""
    pw = PANEL_WIDTH
    canvas[:frame_h, panel_x:panel_x + pw] = PANEL_BG

    px = panel_x + 10
    y  = 14
    time_str = f"{int(t_center // 60):02d}:{int(t_center % 60):02d}"
    cv2.putText(canvas, f"rPPG BPM  t={time_str}",
                (px, y), self.FONT, 0.55, TEXT_HEADER, 2)
    y += 20
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER, 1)
    y += 10

    bar_x = panel_x + 120

    if not snap:
        cv2.putText(canvas, "Waiting for first window...",
                    (px, y + 14), self.FONT, 0.45, TEXT_LABEL, 1)
        return

    for algo in ALGO_ORDER:
        data = snap.get(algo)
        if data is None:
            continue

        bpm = data.get("bpm", 0.0)
        snr = data.get("snr", 0.0)

        is_consensus = (algo == "CONSENSUS")
        in_range = BPM_LOW <= bpm <= BPM_HIGH
        color = ALGO_COLORS.get(algo, TEXT_LABEL)
        if not in_range:
            color = TEXT_WARN

        if is_consensus:
            cv2.line(canvas, (panel_x + 5, y - 2),
                     (panel_x + pw - 5, y - 2), SECTION_DIVIDER, 1)

        bar_len = self._bpm_to_bar(bpm)
        cv2.putText(canvas, f"{algo[:6]:6s} {bpm:5.1f}",
                    (px, y + BAR_HEIGHT - 2), self.FONT, 0.42, color, 1)
        cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_len, y + BAR_HEIGHT),
                      color, -1)
        cv2.rectangle(canvas, (bar_x, y), (bar_x + BAR_MAX_PX, y + BAR_HEIGHT),
                      BAR_OUTLINE, 1)
        cv2.putText(canvas, f"{snr:.1f}",
                    (bar_x + BAR_MAX_PX + 5, y + BAR_HEIGHT - 1),
                    self.FONT, 0.38, (130, 130, 130), 1)
        y += BAR_HEIGHT + 6

    # GT HR
    y += 6
    cv2.line(canvas, (panel_x + 5, y - 2), (panel_x + pw - 5, y - 2),
             SECTION_DIVIDER, 1)
    y += 8
    if gt_bpm is not None and not math.isnan(gt_bpm):
        gt_bar = self._bpm_to_bar(gt_bpm)
        cv2.putText(canvas, f"GT HR  {gt_bpm:5.1f}",
                    (px, y + BAR_HEIGHT - 2), self.FONT, 0.45, TEXT_GT, 1)
        cv2.rectangle(canvas, (bar_x, y), (bar_x + gt_bar, y + BAR_HEIGHT),
                      TEXT_GT, 2)
        cv2.rectangle(canvas, (bar_x, y), (bar_x + BAR_MAX_PX, y + BAR_HEIGHT),
                      BAR_OUTLINE, 1)
        y += BAR_HEIGHT + 8
        consensus_data = snap.get("CONSENSUS")
        if consensus_data:
            delta = consensus_data.get("bpm", float("nan")) - gt_bpm
            if not math.isnan(delta):
                d_color = TEXT_DELTA_POS if delta >= 0 else TEXT_DELTA_NEG
                cv2.putText(canvas, f"CONS vs GT  {delta:+.1f} BPM",
                            (px, y + 14), self.FONT, 0.44, d_color, 1)
    else:
        cv2.putText(canvas, "GT HR  (no data)",
                    (px, y + BAR_HEIGHT - 2), self.FONT, 0.44, (80, 80, 80), 1)


# Monkey-patch the snapshot method onto RppgHudPanel
RppgHudPanel.draw_snapshot = _draw_snapshot
