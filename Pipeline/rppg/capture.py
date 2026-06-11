"""Live webcam capture with video recording and multi-region face ROI extraction.

Pattern: [Observer] — LiveCaptureSession fires registered frame_callbacks on
every captured frame, decoupling capture from display / live-BPM preview.
The capture loop itself has no dependency on OpenCV imshow or any display code.

Usage (direct):
    from rppg.capture import CaptureConfig, LiveCaptureSession
    cfg = CaptureConfig(preview=True)
    session = LiveCaptureSession(cfg)
    folder = session.run("pilot_01")   # blocks until Q pressed or duration elapsed
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from rppg.rppg_algorithms import compute_bpm_timeseries
from rppg.extractors import (
    FrameRecord,
    MultiROIExtractor,
    VideoReextractor,        # re-exported so callers can still do: from rppg.capture import ...
    write_mp_frames_csv,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    camera_index: int = 0
    output_dir: Path = Path("Recordings/Live")
    fps_target: int = 30
    face_backend: str = "mp"     # MediaPipe FaceLandmarker
    roi_preset: str = "forehead" # kept for API compat; MultiROIExtractor extracts all 3
    preview: bool = True
    max_duration_s: float = 0.0  # 0 = unlimited


# Type alias for callbacks
FrameCallback = Callable[[np.ndarray, FrameRecord], None]


# ---------------------------------------------------------------------------
# Live BPM preview state (internal)
# ---------------------------------------------------------------------------

class _LiveBpmState:
    """Rolling BPM estimate shown in the preview window.

    Recomputes every `step_s` seconds using the last `window_s` seconds of
    accumulated RGB frames.  Only CHROM + POS are computed to keep latency low.
    """

    ALGOS = ("chrom", "pos")

    def __init__(self, window_s: float = 30.0, step_s: float = 5.0):
        self._window_s = window_s
        self._step_s = step_s
        self._last_compute = 0.0
        self.estimates: dict = {a: {"bpm": 0.0, "snr": 0.0} for a in self.ALGOS}

    def update(self, rgbs: List, timestamps: List, now: float) -> bool:
        """Recompute if step interval elapsed. Returns True when updated."""
        if now - self._last_compute < self._step_s:
            return False
        if len(rgbs) < 2:
            return False

        arr_rgb = np.array(rgbs, dtype=float)
        arr_ts  = np.array(timestamps, dtype=float)
        duration = arr_ts[-1] - arr_ts[0]
        if duration < 1.0:
            return False

        fps_eff = len(arr_rgb) / duration
        window_frames = int(self._window_s * fps_eff)
        if len(arr_rgb) > window_frames:
            arr_rgb = arr_rgb[-window_frames:]
            arr_ts  = arr_ts[-window_frames:]

        for algo in self.ALGOS:
            try:
                rows = compute_bpm_timeseries(
                    arr_rgb, arr_ts, fps_eff,
                    algorithm=algo, window_s=self._window_s, step_s=self._window_s
                )
                if rows:
                    last = rows[-1]
                    self.estimates[algo] = {"bpm": last["bpm"], "snr": last["snr"]}
            except Exception:
                pass

        self._last_compute = now
        return True


# ---------------------------------------------------------------------------
# Preview callback
# ---------------------------------------------------------------------------

def _make_preview_callback(bpm_state: _LiveBpmState) -> FrameCallback:
    """Return a frame callback that draws ROI bbox + roi_source + BPM overlay."""

    def _callback(frame: np.ndarray, record: FrameRecord) -> None:
        display = frame.copy()

        if record.face_detected:
            # Draw tight ROI polygon bbox (the actual region sampled for rPPG)
            draw_str = record.roi_bbox_str or record.bbox_str
            if draw_str:
                bx1, by1, bx2, by2 = map(int, draw_str.split(","))
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 100), 2)
                if record.roi_source:
                    cv2.putText(display, record.roi_source,
                                (bx1, max(0, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 24
        cv2.putText(display, f"t={record.timestamp:.1f}s  frame={record.frame_idx}",
                    (10, y_pos), font, 0.5, (200, 200, 200), 1)
        y_pos += 22

        for algo, data in bpm_state.estimates.items():
            bpm   = data["bpm"]
            snr   = data["snr"]
            color = (0, 220, 100) if 42 <= bpm <= 180 else (60, 60, 200)
            cv2.putText(display,
                        f"{algo.upper():6s} {bpm:5.1f} BPM  SNR {snr:.1f}",
                        (10, y_pos), font, 0.5, color, 1)
            y_pos += 20

        cv2.putText(display, "Press Q to stop",
                    (10, display.shape[0] - 10), font, 0.45, (150, 150, 150), 1)
        cv2.imshow("rPPG Capture", display)

    return _callback


# ---------------------------------------------------------------------------
# Main capture session
# ---------------------------------------------------------------------------

class LiveCaptureSession:
    """Records webcam to video file and extracts per-frame multi-region face ROI.

    Usage:
        session = LiveCaptureSession(CaptureConfig())
        folder_info = session.run("my_label")
        # returns dict with keys: video_path, frames_csv, session_start_utc, ...
    """

    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or CaptureConfig()
        self._extractor = MultiROIExtractor()
        self.frame_callbacks: List[FrameCallback] = []

    def run(self, label: str) -> dict:
        """Capture until Q pressed (or max_duration exceeded).

        Returns:
            dict with video_path (Path), frames_csv (Path), session_start_utc (str)
        """
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_")
        video_filename = f"{ts_str}_{safe_label}.mp4"
        video_path = output_dir / video_filename
        frames_csv_path = output_dir / f"{ts_str}_{safe_label}_frames.csv"

        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cfg.camera_index}")

        actual_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_target
        if actual_fps <= 0:
            actual_fps = cfg.fps_target
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = self._open_writer(video_path, actual_fps, frame_w, frame_h)

        bpm_state = _LiveBpmState()
        callbacks = list(self.frame_callbacks)
        if cfg.preview:
            callbacks.append(_make_preview_callback(bpm_state))

        session_start_unix = time.time()
        session_start_utc = datetime.now(timezone.utc).isoformat()

        _acc_rgb: List[np.ndarray] = []
        _acc_ts: List[float]       = []

        records: List[FrameRecord] = []
        frame_idx = 0
        t0 = time.time()

        print(f"\n[rPPG Capture] Starting — {video_path.name}")
        print(f"  Camera: {frame_w}x{frame_h} @ {actual_fps:.0f} fps")
        print(f"  Press Q to stop.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                now     = time.time()
                elapsed = now - t0

                if cfg.max_duration_s > 0 and elapsed >= cfg.max_duration_s:
                    break

                roi          = self._extractor.extract(frame)
                primary      = roi["primary"]
                face_ok      = primary is not None
                bbox         = roi["bbox"]
                roi_bbox     = roi["roi_bbox"]
                bbox_str     = (",".join(map(str, bbox))     if bbox     is not None else "")
                roi_bbox_str = (",".join(map(str, roi_bbox)) if roi_bbox is not None else "")

                def _t(arr):
                    if arr is None:
                        return 0.0, 0.0, 0.0
                    return round(float(arr[0]), 3), round(float(arr[1]), 3), round(float(arr[2]), 3)

                pr, pg, pb    = _t(primary)
                fr, fg, fb    = _t(roi["forehead"])
                gr, gg, gb    = _t(roi["glabella"])
                mr, mg, mb    = _t(roi["malar"])
                bgr, bgg, bgb = _t(roi["bg"])

                record = FrameRecord(
                    frame_idx=frame_idx,
                    timestamp=round(elapsed, 4),
                    r=pr, g=pg, b=pb,
                    r_forehead=fr, g_forehead=fg, b_forehead=fb,
                    r_glabella=gr, g_glabella=gg, b_glabella=gb,
                    r_malar=mr,    g_malar=mg,    b_malar=mb,
                    roi_source=roi["roi_source"] or "",
                    r_bg=bgr, g_bg=bgg, b_bg=bgb,
                    face_detected=face_ok,
                    bbox_str=bbox_str,
                    roi_bbox_str=roi_bbox_str,
                )
                records.append(record)

                if face_ok:
                    _acc_rgb.append(np.array([pr, pg, pb]))
                    _acc_ts.append(elapsed)
                    bpm_state.update(_acc_rgb, _acc_ts, now)

                writer.write(frame)

                for cb in callbacks:
                    cb(frame, record)

                if cfg.preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q") or key == 27:
                        break

                frame_idx += 1

                if frame_idx % 150 == 0:
                    face_count = sum(1 for r in records if r.face_detected)
                    print(f"  {elapsed:6.1f}s  {frame_idx} frames  "
                          f"face {face_count}/{frame_idx} "
                          f"({100*face_count/frame_idx:.0f}%)")

        finally:
            self._extractor.close()
            cap.release()
            writer.release()
            if cfg.preview:
                cv2.destroyAllWindows()

        write_mp_frames_csv(frames_csv_path, records, session_start_utc)

        face_total = sum(1 for r in records if r.face_detected)
        print(f"\n[rPPG Capture] Done — {frame_idx} frames, "
              f"{face_total} with face ({100*face_total/max(frame_idx,1):.1f}%)")
        print(f"  Video  : {video_path}")
        print(f"  Frames : {frames_csv_path}")

        return {
            "video_path": video_path,
            "frames_csv": frames_csv_path,
            "session_start_utc": session_start_utc,
            "session_start_unix": session_start_unix,
            "label": label,
            "ts_str": ts_str,
        }

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
