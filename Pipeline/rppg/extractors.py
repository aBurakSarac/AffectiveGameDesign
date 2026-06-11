"""ROI extractor classes — single source of truth for face-region extraction.

Two extractors:
    HaarROIExtractor    — Haar cascade + fixed forehead crop (v1).  Matches the
                          exact logic used for sessions captured before the
                          MediaPipe upgrade.  Use ``--extractor haar`` to recover
                          lost frames.csv from those recordings.
    MultiROIExtractor   — MediaPipe FaceLandmarker, 3 independent facial ROIs (v2).

VideoReextractor applies either extractor to a saved raw_video.mp4 and writes a
fresh frames.csv, so algorithm changes can be applied to existing recordings.

CSV writers (module-level, used by both live capture and re-extraction):
    write_haar_frames_csv(path, records, session_start_utc)  — v1 minimal format
    write_mp_frames_csv(path, records, session_start_utc)    — v2 multi-ROI format

Usage (data recovery):
    from rppg.extractors import VideoReextractor
    VideoReextractor(extractor_type="haar").reextract(
        video_path=Path("Pipeline/sessions/20260502_162448_pilot01/raw_video.mp4"),
        output_csv=Path("Pipeline/sessions/20260502_162448_pilot01/frames.csv"),
        original_csv=Path("Pipeline/sessions/20260502_162448_pilot01/frames_pre_reextract.csv"),
    )
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared data type
# ---------------------------------------------------------------------------

@dataclass
class FrameRecord:
    frame_idx: int
    timestamp: float       # seconds since session start
    r: float               # primary ROI (forehead > glabella > malar)
    g: float
    b: float
    r_forehead: float
    g_forehead: float
    b_forehead: float
    r_glabella: float
    g_glabella: float
    b_glabella: float
    r_malar: float
    g_malar: float
    b_malar: float
    roi_source: str        # "forehead" | "glabella" | "malar" | ""
    r_bg: float            # background ROI (top-left 50×50)
    g_bg: float
    b_bg: float
    face_detected: bool
    bbox_str: str          # "x1,y1,x2,y2" — full face extent from all landmarks
    roi_bbox_str: str = "" # "x1,y1,x2,y2" — tight bbox of the primary ROI polygon


# ---------------------------------------------------------------------------
# CSV writers — one per format version
# ---------------------------------------------------------------------------

def write_haar_frames_csv(
    path: Path,
    records: List[FrameRecord],
    session_start_utc: str,
) -> None:
    """Write v1 minimal format — matches sessions captured with HaarROIExtractor."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"# session_start_utc={session_start_utc}"])
        writer.writerow(["frame_idx", "timestamp", "r", "g", "b",
                         "face_detected", "bbox"])
        for r in records:
            writer.writerow([r.frame_idx, r.timestamp, r.r, r.g, r.b,
                             int(r.face_detected), r.bbox_str])


def write_mp_frames_csv(
    path: Path,
    records: List[FrameRecord],
    session_start_utc: str,
) -> None:
    """Write v2 multi-ROI format — matches sessions captured with MultiROIExtractor."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"# session_start_utc={session_start_utc}"])
        writer.writerow([
            "frame_idx", "timestamp",
            "r", "g", "b",
            "r_forehead", "g_forehead", "b_forehead",
            "r_glabella", "g_glabella", "b_glabella",
            "r_malar",    "g_malar",    "b_malar",
            "roi_source",
            "r_bg", "g_bg", "b_bg",
            "face_detected", "bbox", "roi_bbox",
        ])
        for r in records:
            writer.writerow([
                r.frame_idx, r.timestamp,
                r.r, r.g, r.b,
                r.r_forehead, r.g_forehead, r.b_forehead,
                r.r_glabella, r.g_glabella, r.b_glabella,
                r.r_malar,    r.g_malar,    r.b_malar,
                r.roi_source,
                r.r_bg, r.g_bg, r.b_bg,
                int(r.face_detected), r.bbox_str, r.roi_bbox_str,
            ])


# ---------------------------------------------------------------------------
# v1 extractor — Haar cascade + fixed forehead crop
# ---------------------------------------------------------------------------

class HaarROIExtractor:
    """Haar-cascade face detector with fixed forehead ROI crop (v1).

    ROI formula (matches original pilot sessions captured before MediaPipe upgrade):
        y: [face_y + 0.15*h,  face_y + 0.55*h]
        x: [face_x + 0.20*w,  face_x + 0.80*w]

    extract() returns (mean_rgb, bbox) where bbox is the forehead crop coords,
    NOT the full face rect.  Matches the bbox column in legacy frames.csv files.
    """

    def __init__(self) -> None:
        haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(haar)

    def extract(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Return (mean_rgb [R,G,B], forehead_bbox) or (None, None)."""
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        y1 = y + int(0.15 * h)
        y2 = y + int(0.55 * h)
        x1 = x + int(0.20 * w)
        x2 = x + int(0.80 * w)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None

        mean_bgr = roi.mean(axis=(0, 1))
        mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=float)
        return mean_rgb, (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# v2 extractor — MediaPipe FaceLandmarker, three facial ROIs
# ---------------------------------------------------------------------------

_MP_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"


class MultiROIExtractor:
    """MediaPipe FaceLandmarker-based extractor for three facial ROIs.

    Regions (all defined by Face Mesh landmark indices):
        forehead  — above the eyebrows, skin-checked
        glabella  — between the eyebrows (highly vascular, hair-resistant)
        malar     — upper cheekbones (left + right averaged)

    Returns a dict with all three ROIs (None when unavailable), a 'primary'
    fallback chain (forehead → glabella → malar), and a background patch for
    interference decoupling.
    """

    _FOREHEAD_LMS = [10, 338, 297, 284, 336, 9, 107, 54, 103, 67, 109]
    _GLABELLA_LMS = [9, 336, 285, 8, 55, 107]
    _MALAR_L_LMS  = [101, 118, 117, 123, 50, 36]
    _MALAR_R_LMS  = [330, 347, 346, 352, 280, 266]

    def __init__(self, model_path: Path = _MP_MODEL_PATH) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def extract(self, frame: np.ndarray) -> dict:
        """Return multi-ROI dict for one BGR frame.

        Keys: forehead, glabella, malar, primary, bbox, roi_bbox, roi_source, bg.
        All RGB arrays are [R,G,B] float or None when unavailable.
        """
        _empty = {
            "forehead": None, "glabella": None, "malar": None,
            "primary": None, "bbox": None, "roi_bbox": None, "roi_source": None,
            "bg": self._extract_background(frame),
        }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=rgb_frame)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return _empty

        h, w = frame.shape[:2]
        lms = result.face_landmarks[0]
        pts = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in lms], dtype=np.int32
        )

        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        bbox = (x1, y1, x2, y2)

        forehead_rgb = self._polygon_mean_rgb(frame, pts, self._FOREHEAD_LMS)
        glabella_rgb = self._polygon_mean_rgb(frame, pts, self._GLABELLA_LMS)
        malar_rgb    = self._bilateral_mean_rgb(frame, pts,
                                                self._MALAR_L_LMS, self._MALAR_R_LMS)

        if forehead_rgb is not None:
            forehead_rgb = self._skin_check(frame, pts, self._FOREHEAD_LMS,
                                            forehead_rgb)

        if forehead_rgb is not None:
            primary, roi_source = forehead_rgb, "forehead"
        elif glabella_rgb is not None:
            primary, roi_source = glabella_rgb, "glabella"
        elif malar_rgb is not None:
            primary, roi_source = malar_rgb, "malar"
        else:
            primary, roi_source = None, None

        # Tight bounding box around the primary ROI polygon (not the full face)
        roi_bbox = None
        if roi_source == "forehead":
            roi_pts = pts[self._FOREHEAD_LMS]
        elif roi_source == "glabella":
            roi_pts = pts[self._GLABELLA_LMS]
        elif roi_source == "malar":
            roi_pts = np.vstack([pts[self._MALAR_L_LMS], pts[self._MALAR_R_LMS]])
        else:
            roi_pts = None
        if roi_pts is not None:
            roi_bbox = (
                int(roi_pts[:, 0].min()), int(roi_pts[:, 1].min()),
                int(roi_pts[:, 0].max()), int(roi_pts[:, 1].max()),
            )

        return {
            "forehead":   forehead_rgb,
            "glabella":   glabella_rgb,
            "malar":      malar_rgb,
            "primary":    primary,
            "bbox":       bbox,
            "roi_bbox":   roi_bbox,
            "roi_source": roi_source,
            "bg":         self._extract_background(frame),
        }

    @staticmethod
    def _polygon_mean_rgb(
        frame: np.ndarray, pts: np.ndarray, lm_indices: list,
    ) -> Optional[np.ndarray]:
        poly = pts[lm_indices]
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        if mask.sum() == 0:
            return None
        mean_bgr = cv2.mean(frame, mask=mask)[:3]
        return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=float)

    @staticmethod
    def _bilateral_mean_rgb(
        frame: np.ndarray, pts: np.ndarray,
        left_indices: list, right_indices: list,
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts[left_indices], pts[right_indices]], 255)
        if mask.sum() == 0:
            return None
        mean_bgr = cv2.mean(frame, mask=mask)[:3]
        return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=float)

    @staticmethod
    def _skin_check(
        frame: np.ndarray, pts: np.ndarray,
        lm_indices: list, current_rgb: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return None if >40% of polygon pixels fail skin-color HSV test."""
        poly = pts[lm_indices]
        h, w = frame.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [poly], 255)
        roi_pixels = int(roi_mask.sum()) // 255
        if roi_pixels == 0:
            return current_rgb

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(
            hsv,
            np.array([0, 20, 40],    dtype=np.uint8),
            np.array([50, 255, 255], dtype=np.uint8),
        )
        skin_in_roi = cv2.bitwise_and(skin_mask, skin_mask, mask=roi_mask)
        skin_ratio  = (int(skin_in_roi.sum()) // 255) / roi_pixels
        return current_rgb if skin_ratio >= 0.60 else None

    @staticmethod
    def _extract_background(frame: np.ndarray) -> np.ndarray:
        # Background ROI for interference decoupling (Shao et al., 2025)
        bg = frame[0:50, 0:50]
        mean_bgr = bg.mean(axis=(0, 1))
        return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=float)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_timestamps_from_csv(csv_path: Optional[Path]) -> tuple:
    """Return ({frame_idx: timestamp}, session_start_utc) from an existing frames.csv."""
    if csv_path is None or not Path(csv_path).exists():
        return {}, ""
    ts_map: dict = {}
    session_start_utc = ""
    try:
        with open(csv_path, encoding="utf-8") as f:
            first = f.readline()
            if first.startswith("# session_start_utc="):
                session_start_utc = first.split("=", 1)[1].strip()
            content = first + f.read()
        lines  = [ln for ln in content.splitlines() if not ln.strip().startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                ts_map[int(row["frame_idx"])] = float(row["timestamp"])
            except (KeyError, ValueError):
                continue
    except OSError:
        pass
    return ts_map, session_start_utc


def _rgb_triplet(arr: Optional[np.ndarray]) -> Tuple[float, float, float]:
    if arr is None:
        return 0.0, 0.0, 0.0
    return round(float(arr[0]), 3), round(float(arr[1]), 3), round(float(arr[2]), 3)


# ---------------------------------------------------------------------------
# Video re-extractor
# ---------------------------------------------------------------------------

class VideoReextractor:
    """Re-runs an ROI extractor on a saved raw_video.mp4 to regenerate frames.csv.

    extractor_type="haar"  — use HaarROIExtractor; writes v1 minimal CSV.
                             Use this to recover frames.csv for sessions that
                             were originally captured with the Haar extractor.
    extractor_type="mp"    — use MultiROIExtractor; writes v2 multi-ROI CSV.
                             Use this to re-extract sessions captured with MP
                             or to apply updated landmark definitions.

    Timestamps are borrowed from original_csv (by frame_idx) when available,
    keeping downstream analysis windows consistent with the original capture.

    Usage:
        from rppg.extractors import VideoReextractor
        VideoReextractor(extractor_type="haar").reextract(
            video_path=Path("sessions/pilot01/raw_video.mp4"),
            output_csv=Path("sessions/pilot01/frames.csv"),
            original_csv=Path("sessions/pilot01/frames_pre_reextract.csv"),
        )
    """

    def __init__(self, extractor_type: str = "mp") -> None:
        if extractor_type not in ("haar", "mp"):
            raise ValueError(f"extractor_type must be 'haar' or 'mp', got '{extractor_type}'")
        self._type = extractor_type
        if extractor_type == "haar":
            self._extractor = HaarROIExtractor()
        else:
            self._extractor = MultiROIExtractor()

    def reextract(
        self,
        video_path: Path,
        output_csv: Path,
        original_csv: Optional[Path] = None,
    ) -> None:
        """Process video_path with current extractor and write new frames.csv."""
        video_path = Path(video_path)
        output_csv = Path(output_csv)

        ts_map, session_start_utc = _load_timestamps_from_csv(original_csv)
        if not session_start_utc:
            session_start_utc = datetime.now(timezone.utc).isoformat()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        records: List[FrameRecord] = []
        frame_idx = 0

        print(f"[Reextract:{self._type}] {n_frames} frames from {video_path.name}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                elapsed = ts_map.get(frame_idx, frame_idx / fps)
                if self._type == "haar":
                    records.append(self._process_haar(frame, frame_idx, elapsed))
                else:
                    records.append(self._process_mp(frame, frame_idx, elapsed))

                if frame_idx % 150 == 0:
                    pct = 100 * frame_idx / max(n_frames, 1)
                    print(f"  {frame_idx}/{n_frames} ({pct:.0f}%)")
                frame_idx += 1
        finally:
            cap.release()
            if hasattr(self._extractor, "close"):
                self._extractor.close()

        if self._type == "haar":
            write_haar_frames_csv(output_csv, records, session_start_utc)
        else:
            write_mp_frames_csv(output_csv, records, session_start_utc)

        face_total = sum(1 for r in records if r.face_detected)
        print(f"[Reextract:{self._type}] Done — {frame_idx} frames, "
              f"{face_total} with face ({100 * face_total / max(frame_idx, 1):.1f}%)")
        print(f"  → {output_csv}")

    def _process_haar(
        self, frame: np.ndarray, frame_idx: int, elapsed: float
    ) -> FrameRecord:
        mean_rgb, bbox = self._extractor.extract(frame)  # type: ignore[union-attr]
        face_ok  = mean_rgb is not None
        bbox_str = (",".join(map(str, bbox)) if face_ok else "")
        r, g, b  = _rgb_triplet(mean_rgb)
        return FrameRecord(
            frame_idx=frame_idx, timestamp=round(elapsed, 4),
            r=r, g=g, b=b,
            r_forehead=0.0, g_forehead=0.0, b_forehead=0.0,
            r_glabella=0.0, g_glabella=0.0, b_glabella=0.0,
            r_malar=0.0,    g_malar=0.0,    b_malar=0.0,
            roi_source="",
            r_bg=0.0, g_bg=0.0, b_bg=0.0,
            face_detected=face_ok,
            bbox_str=bbox_str,
        )

    def _process_mp(
        self, frame: np.ndarray, frame_idx: int, elapsed: float
    ) -> FrameRecord:
        roi          = self._extractor.extract(frame)  # type: ignore[union-attr]
        primary      = roi["primary"]
        face_ok      = primary is not None
        bbox         = roi["bbox"]
        roi_bbox     = roi["roi_bbox"]
        bbox_str     = (",".join(map(str, bbox))     if bbox     is not None else "")
        roi_bbox_str = (",".join(map(str, roi_bbox)) if roi_bbox is not None else "")
        pr, pg, pb   = _rgb_triplet(primary)
        fr, fg, fb   = _rgb_triplet(roi["forehead"])
        gr, gg, gb   = _rgb_triplet(roi["glabella"])
        mr, mg, mb   = _rgb_triplet(roi["malar"])
        bgr, bgg, bgb = _rgb_triplet(roi["bg"])
        return FrameRecord(
            frame_idx=frame_idx, timestamp=round(elapsed, 4),
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
