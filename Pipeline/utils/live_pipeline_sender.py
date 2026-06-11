"""
live_pipeline_sender.py — Real-time FER+rPPG pipeline for Unity
================================================================
Pattern: [Adapter] — adapts the FER pipeline output (MP+HS blendshape/emotion
scores) into newline-delimited JSON over TCP, matching the contract expected
by Unity's SocketManager. Hides all ML inference details behind a single
per-frame JSON payload.

Opens the system webcam, runs the MP+HS fusion loop per frame, and emits
JSON over TCP to Unity's SocketManager on localhost:5005.

Composite fear uses the F12 formula (hybrid_amp): (0.7*hs_fear + 0.3*hs_arousal) * (1 + mp_tension)

TCP server model: this script listens; Unity connects as a client.
JSON objects are newline-delimited (\\n), matching MockPipelineSender exactly.

Usage:
    python live_pipeline_sender.py
    python live_pipeline_sender.py --camera 1 --port 5005 --no-rppg

Press Ctrl+C to stop.
"""

import json
import socket
import time
import threading
import argparse
import sys
import os
import collections

# ── Make sibling Pipeline modules importable regardless of CWD ────────────────
# The helpers are imported by bare name but live in sibling folders:
#   test_fusion, improved_fear_detection -> fer/ · rppg_algorithms -> rppg/
#   session_meta, post_session_analysis -> utils/
_PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/Pipeline
for _sub in ("utils", "rppg", "fer"):
    _p = os.path.join(_PIPELINE_ROOT, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Force unbuffered output so prints appear immediately
sys.stdout.reconfigure(line_buffering=True)

print("[LivePipeline] Loading standard libraries...", flush=True)

import cv2
import numpy as np

# ── Pipeline imports ──────────────────────────────────────────────────────────
# Disable CUDA for consistent behaviour (same as test_fusion.py default)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

print("[LivePipeline] Loading MediaPipe (slow on first run)...", flush=True)
import mediapipe as mp_lib

print("[LivePipeline] Loading HSEmotion ONNX...", flush=True)
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

print("[LivePipeline] Loading fusion helpers...", flush=True)
from test_fusion import (
    ensure_model,
    compute_tension,
    extract_forehead_rgb,
    compute_expression_activity,
    EMOTION_LABELS,
    MODEL_PATH,
    EXPRESSION_ACTIVITY_THRESHOLD,
)
from rppg_algorithms import compute_bpm_timeseries


# ═══════════════════════════════════════════════════════════════════════════════
# F12 composite fear formula (hybrid_amp) — selected phase 4 evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_composite_fear_f12(hs_fear: float, hs_arousal: float, mp_tension: float) -> float:
    """F12 (hybrid_amp): (0.7*hs_fear + 0.3*hs_arousal) * (1 + mp_tension)."""
    return min(1.0, max(0.0, (0.7 * hs_fear + 0.3 * hs_arousal) * (1.0 + mp_tension)))


# ── F15: rPPG-augmented composite fear ───────────────────────────────────────
# Production formula (see render_merged_hud.py / the report):
#     F15 = clamp( F12 * (1 + RPPG_COEFF * bpm_norm) )
RPPG_COEFF = 0.5


class BpmBaseline:
    """Trailing rolling-median resting-HR baseline for live bpm_norm.

    Mirrors the offline pipeline — bpm_norm = clip((bpm - baseline)/baseline, 0, 1),
    baseline floored at 70 — but uses a TRAILING window (no future frames live).
    Calibration begins the moment the script runs; the partial buffer is used
    during the first ~window seconds.
    """
    def __init__(self, window_s: float = 60.0, floor_bpm: float = 70.0):
        self.window_s = window_s
        self.floor_bpm = floor_bpm
        self._bpms = collections.deque()
        self._ts = collections.deque()

    def bpm_norm(self, bpm: float, now: float) -> float:
        if bpm and bpm > 0.0:
            self._bpms.append(bpm)
            self._ts.append(now)
        while self._ts and now - self._ts[0] > self.window_s:
            self._ts.popleft()
            self._bpms.popleft()
        if not self._bpms or not bpm or bpm <= 0.0:
            return 0.0
        baseline = max(float(np.median(self._bpms)), self.floor_bpm)
        return min(1.0, max(0.0, (bpm - baseline) / baseline))


def compute_f15(f12: float, bpm_norm: float) -> float:
    """F15 (production): F12 augmented by the normalised heart-rate rise."""
    return min(1.0, max(0.0, f12 * (1.0 + RPPG_COEFF * bpm_norm)))


# ═══════════════════════════════════════════════════════════════════════════════
# rPPG background thread
# ═══════════════════════════════════════════════════════════════════════════════

class RppgWorker(threading.Thread):
    """
    Periodically estimates BPM from a rolling buffer of forehead RGB values.
    Runs in a daemon thread; the main loop feeds data via push().
    """

    WINDOW_S = 10.0   # seconds of data per BPM estimate (live: first BPM ~10s in)
    STEP_S   = 2.0    # re-estimate every N seconds

    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._rgbs: list = []          # [R, G, B] per face-detected frame
        self._timestamps: list = []    # float epoch timestamps
        self._motion_flags: list = []  # True when strong facial expression
        self._bpm: float = 0.0
        self._running = True
        self._last_compute = 0.0

    def push(self, r: float, g: float, b: float, ts: float, motion: bool):
        with self._lock:
            self._rgbs.append([r, g, b])
            self._timestamps.append(ts)
            self._motion_flags.append(motion)
            # Discard data older than WINDOW_S * 2
            cutoff = ts - self.WINDOW_S * 2
            while self._timestamps and self._timestamps[0] < cutoff:
                self._rgbs.pop(0)
                self._timestamps.pop(0)
                self._motion_flags.pop(0)

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def status(self):
        """(state, eta_seconds): 'ok' once BPM is flowing, else 'warming' with an
        estimate of how many more seconds of facial data are needed."""
        if self._bpm > 0.0:
            return "ok", 0.0
        with self._lock:
            n = len(self._timestamps)
            dur = (self._timestamps[-1] - self._timestamps[0]) if n >= 2 else 0.0
        return "warming", max(0.0, self.WINDOW_S - dur)

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            time.sleep(0.5)
            now = time.time()
            if now - self._last_compute < self.STEP_S:
                continue
            self._last_compute = now

            with self._lock:
                if len(self._timestamps) < 60:
                    continue
                rgbs   = list(self._rgbs)
                ts     = list(self._timestamps)
                motion = list(self._motion_flags)

            try:
                # Infer effective fps from timestamps
                duration = ts[-1] - ts[0]
                if duration < 5.0:
                    continue
                fps = len(ts) / duration

                results = compute_bpm_timeseries(
                    np.array(rgbs, dtype=np.float32),
                    ts,
                    fps=fps,
                    algorithm="chrom",
                    window_s=self.WINDOW_S,
                    step_s=self.STEP_S,
                    motion_flags=motion,
                )
                # compute_bpm_timeseries returns a LIST OF PER-WINDOW DICTS
                # (keys: bpm, bpm_smoothed, bpm_plausible, ...). Take the most
                # recent plausible estimate. (The old code indexed results[0][-1]
                # as if it were a series, which threw and was silently swallowed,
                # leaving BPM pinned at 0.)
                if results:
                    usable = [r for r in results
                              if r.get("bpm_plausible", True)
                              and (r.get("bpm_smoothed") or r.get("bpm"))]
                    rows = usable or results
                    last = rows[-1]
                    val = last.get("bpm_smoothed") or last.get("bpm") or 0.0
                    if val and val > 0:
                        self._bpm = float(val)
            except Exception as exc:
                print(f"[LivePipeline] rPPG estimate failed: {exc}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MediaPipe + HSEmotion initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def init_mediapipe():
    ensure_model()
    BaseOptions         = mp_lib.tasks.BaseOptions
    FaceLandmarker      = mp_lib.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp_lib.tasks.vision.FaceLandmarkerOptions
    RunningMode         = mp_lib.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def init_hsemotion():
    recognizer   = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return recognizer, face_cascade


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame inference
# ═══════════════════════════════════════════════════════════════════════════════

def process_frame(frame, landmarker, recognizer, face_cascade,
                  frame_count: int, fps: float):
    """
    Run MediaPipe + HSEmotion on a single BGR frame.

    Returns a dict with all fields needed for the JSON payload, plus
    rPPG-accumulation values (r, g, b, motion_flag).
    """
    fh_px, fw_px = frame.shape[:2]
    elapsed_ms   = int(frame_count * 1000 / fps)

    # ── Defaults ─────────────────────────────────────────────────────────────
    mp_tension       = 0.0
    hs_fear          = 0.0
    hs_surprise      = 0.0
    hs_sadness       = 0.0
    hs_anger         = 0.0
    hs_happiness     = 0.0
    hs_arousal       = 0.0
    face_bbox_area   = 0.0
    rppg_r = rppg_g  = rppg_b = 0.0
    motion_flag      = False
    mp_face_bbox     = None
    hs_face_bbox     = None
    bs_dict          = {}

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=frame_rgb)
    mp_results = landmarker.detect_for_video(mp_image, elapsed_ms)

    if mp_results.face_blendshapes and mp_results.face_blendshapes:
        blendshapes = mp_results.face_blendshapes[0]
        bs_dict     = {bs.category_name: bs.score for bs in blendshapes}
        mp_tension  = compute_tension(bs_dict)

        if mp_results.face_landmarks and mp_results.face_landmarks:
            lms = mp_results.face_landmarks[0]
            xs  = [lm.x * fw_px for lm in lms]
            ys  = [lm.y * fh_px for lm in lms]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            mp_face_bbox = (x1, y1, x2, y2)
            face_bbox_area = ((x2 - x1) * (y2 - y1)) / (fw_px * fh_px)

    # ── HSEmotion (Haar-first, MP fallback) ───────────────────────────────────
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

    face_img = None
    if len(faces) > 0:
        fx, fy, fw_f, fh_f = max(faces, key=lambda f: f[2] * f[3])
        face_img     = frame[fy:fy + fh_f, fx:fx + fw_f]
        hs_face_bbox = (fx, fy, fx + fw_f, fy + fh_f)
    elif mp_face_bbox is not None:
        x1, y1, x2, y2 = mp_face_bbox
        w, h  = x2 - x1, y2 - y1
        pad   = 0.15
        x1p   = max(0, int(x1 - w * pad))
        y1p   = max(0, int(y1 - h * pad))
        x2p   = min(frame.shape[1], int(x2 + w * pad))
        y2p   = min(frame.shape[0], int(y2 + h * pad))
        face_img     = frame[y1p:y2p, x1p:x2p]
        hs_face_bbox = (x1p, y1p, x2p, y2p)

    if face_img is not None and face_img.size > 0:
        _, scores = recognizer.predict_emotions(face_img, logits=False)
        # scores layout: [Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, valence, arousal]
        label_map = {l: i for i, l in enumerate(EMOTION_LABELS)}
        hs_fear      = float(scores[label_map.get("Fear",       3)])
        hs_surprise  = float(scores[label_map.get("Surprise",   7)])
        hs_sadness   = float(scores[label_map.get("Sadness",    6)])
        hs_anger     = float(scores[label_map.get("Anger",      0)])
        hs_happiness = float(scores[label_map.get("Happiness",  4)])
        # Arousal is the last element in the MTL output
        if len(scores) > len(EMOTION_LABELS):
            hs_arousal = float(scores[-1])

    # ── rPPG ROI ──────────────────────────────────────────────────────────────
    rppg_bbox = hs_face_bbox if hs_face_bbox else mp_face_bbox
    if rppg_bbox:
        rgb = extract_forehead_rgb(frame, rppg_bbox)
        if rgb is not None:
            rppg_r, rppg_g, rppg_b = rgb
            motion_flag = compute_expression_activity(bs_dict) > EXPRESSION_ACTIVITY_THRESHOLD

    # ── F12 composite fear (hybrid_amp) ─────────────────────────────────────
    composite_fear = compute_composite_fear_f12(hs_fear, hs_arousal, mp_tension)

    return {
        "composite_fear": composite_fear,
        "mp_tension":     mp_tension,
        "hs_fear":        hs_fear,
        "hs_surprise":    hs_surprise,
        "hs_sadness":     hs_sadness,
        "hs_anger":       hs_anger,
        "hs_happiness":   hs_happiness,
        "hs_arousal":     hs_arousal,
        "face_bbox_area": face_bbox_area,
        # rPPG accumulation (not sent directly)
        "_rppg_r": rppg_r, "_rppg_g": rppg_g, "_rppg_b": rppg_b,
        "_motion": motion_flag,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TCP server loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_server(args):
    if args.no_calib:
        print("[LivePipeline] --no-calib: Unity will use raw composite_fear (no baseline subtraction).", flush=True)

    print("[LivePipeline] Initialising models...", flush=True)
    landmarker              = init_mediapipe()
    recognizer, face_cascade = init_hsemotion()
    print("[LivePipeline] Models ready.", flush=True)

    # rPPG background thread
    rppg_worker = None
    if not args.no_rppg:
        rppg_worker = RppgWorker()
        rppg_worker.start()
        print("[LivePipeline] rPPG thread started.", flush=True)
    else:
        print("[LivePipeline] rPPG disabled (--no-rppg).", flush=True)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[LivePipeline] ERROR: Cannot open camera index {args.camera}.", flush=True)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam_fps = 30.0
    print(f"[LivePipeline] Webcam opened (index {args.camera}).", flush=True)

    # TCP server — same pattern as MockPipelineSender
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("localhost", args.port))
    server.listen(1)
    server.settimeout(1.0)   # allow Ctrl+C to interrupt accept()

    print(f"[LivePipeline] Listening on localhost:{args.port}", flush=True)
    print("[LivePipeline] Waiting for Unity to connect... (press Ctrl+C to stop)\n", flush=True)

    frame_count   = 0
    fps_report_t  = time.time()
    fps_frames    = 0
    bpm_baseline  = BpmBaseline()   # trailing-median resting-HR baseline for F15

    try:
        while True:
            # ── Wait for Unity connection ─────────────────────────────────
            conn = None
            while conn is None:
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
            print(f"[LivePipeline] Unity connected: {addr}", flush=True)
            conn.setblocking(False)

            # ── Per-frame loop ────────────────────────────────────────────
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("[LivePipeline] ERROR: Webcam read failed.", flush=True)
                        break

                    frame_count  += 1
                    fps_frames   += 1
                    now           = time.time()

                    result = process_frame(
                        frame, landmarker, recognizer, face_cascade,
                        frame_count, webcam_fps,
                    )

                    # Feed rPPG worker
                    if rppg_worker is not None and result["_rppg_r"] > 0:
                        rppg_worker.push(
                            result["_rppg_r"], result["_rppg_g"], result["_rppg_b"],
                            now, result["_motion"],
                        )

                    # Build payload — upgrade F12 → F15 using the live rPPG baseline
                    bpm = rppg_worker.bpm if rppg_worker else 0.0
                    bpm_state, bpm_eta = rppg_worker.status if rppg_worker else ("off", 0.0)
                    f12 = result["composite_fear"]
                    bpm_norm = bpm_baseline.bpm_norm(bpm, now)
                    f15 = compute_f15(f12, bpm_norm)
                    payload = {
                        "composite_fear": round(f15, 4),   # F15 (FER × rPPG) — what Unity reads
                        "f12":            round(f12, 4),    # FER-only composite (debug)
                        "bpm_norm":       round(bpm_norm, 4),
                        "mp_tension":     round(result["mp_tension"],     4),
                        "hs_fear":        round(result["hs_fear"],        4),
                        "hs_surprise":    round(result["hs_surprise"],    4),
                        "hs_sadness":     round(result["hs_sadness"],     4),
                        "hs_anger":       round(result["hs_anger"],       4),
                        "hs_happiness":   round(result["hs_happiness"],   4),
                        "hs_arousal":     round(result["hs_arousal"],     4),
                        "BPM":            round(bpm, 1),
                        "bpm_status":     bpm_state,          # "warming" | "ok" | "off"
                        "bpm_eta":        round(bpm_eta, 1),  # seconds until first BPM expected
                        "face_bbox_area": round(result["face_bbox_area"], 4),
                        "timestamp":      now,
                        "no_calib":       args.no_calib,
                    }

                    line = json.dumps(payload) + "\n"
                    try:
                        conn.sendall(line.encode("utf-8"))
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print("[LivePipeline] Unity disconnected.", flush=True)
                        break

                    # FPS report every 5 seconds
                    if now - fps_report_t >= 5.0:
                        elapsed = now - fps_report_t
                        fps_actual = fps_frames / elapsed if elapsed > 0 else 0
                        bpm_txt = (f"bpm={payload['BPM']:.0f}" if bpm_state == "ok"
                                   else f"bpm=warming(~{bpm_eta:.0f}s)")
                        print(f"[LivePipeline] {fps_actual:.1f} fps  "
                              f"fear={payload['composite_fear']:.2f}  "
                              f"{bpm_txt}  "
                              f"face_area={payload['face_bbox_area']:.3f}",
                              flush=True)
                        fps_report_t = now
                        fps_frames   = 0

            except (BrokenPipeError, ConnectionResetError):
                print("[LivePipeline] Unity disconnected.", flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            print("[LivePipeline] Waiting for next Unity connection...", flush=True)

    except KeyboardInterrupt:
        print("\n[LivePipeline] Shutting down.", flush=True)
    finally:
        if rppg_worker:
            rppg_worker.stop()
        cap.release()
        server.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Live FER+rPPG pipeline sender for Unity (La Facade Fissuree)")
    parser.add_argument("--camera",  type=int,  default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--port",    type=int,  default=5005,
                        help="TCP port to listen on (default: 5005)")
    parser.add_argument("--no-rppg", action="store_true",
                        help="Disable rPPG background thread (BPM will be 0.0)")
    parser.add_argument("--no-calib", action="store_true",
                        help="Skip baseline calibration: Unity uses raw composite_fear directly "
                             "instead of delta-from-baseline. Useful for quick testing.")
    args = parser.parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
