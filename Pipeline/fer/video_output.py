"""Phase 2 annotated video renderer for the La Façade Fissuréе FER pipeline.

Pattern: [Builder] — constructs an annotated video frame-by-frame from a compact
CSV + original video source; VideoWriter is opened after ML inference is done to
keep RAM flat during both phases.
"""

import cv2
import csv
import numpy as np

from fer.blendshapes import PANEL_WIDTH, HUD_MIN_HEIGHT, EMOTION_LABELS
from fer.hud import draw_hud, draw_on_video_bars


def render_annotated_video(video_path, compact_csv_path, output_path, mode="both"):
    """Read compact CSV + original video and write an annotated HUD video."""
    print("\n" + "=" * 60)
    print("PHASE 2 — Rendering annotated video from CSV")
    print(f"  Video : {video_path}")
    print(f"  CSV   : {compact_csv_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # ── Load CSV into a frame-indexed lookup ─────────────────────────────
    rows = {}  # frame_number (int) → row dict
    try:
        with open(compact_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fn = int(row["frame"])
                    rows[fn] = row
                except (KeyError, ValueError):
                    pass
    except FileNotFoundError:
        print(f"ERROR: compact CSV not found: {compact_csv_path}")
        return

    if not rows:
        print("ERROR: No rows found in compact CSV — aborting render.")
        return

    # ── Open source video ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── HUD canvas dimensions (same formula as analysis phase) ───────────
    hud_scale  = HUD_MIN_HEIGHT / src_h
    hud_vid_w  = int(src_w * hud_scale)
    canvas_w   = hud_vid_w + PANEL_WIDTH
    canvas_h   = HUD_MIN_HEIGHT

    # ── Open VideoWriter ──────────────────────────────────────────────────
    video_writer = None
    for codec in ("avc1", "XVID", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video_writer = cv2.VideoWriter(output_path, fourcc, src_fps,
                                       (canvas_w, canvas_h))
        if video_writer.isOpened():
            print(f"Codec: {codec} | {canvas_w}x{canvas_h} @ {src_fps:.1f} fps")
            break
        video_writer.release()
        video_writer = None

    if video_writer is None:
        print("ERROR: Could not open VideoWriter — render aborted.")
        cap.release()
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0

    # Pre-allocate reusable buffers — avoids per-frame numpy allocation
    canvas  = np.zeros((canvas_h, canvas_w,  3), dtype=np.uint8)
    resized = np.zeros((canvas_h, hud_vid_w, 3), dtype=np.uint8)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ── Build canvas ──────────────────────────────────────────────────
            canvas.fill(0)
            cv2.resize(frame, (hud_vid_w, canvas_h), dst=resized)
            canvas[0:canvas_h, 0:hud_vid_w] = resized

            row = rows.get(frame_idx)
            if row:
                try:
                    crop_src = row.get("hs_crop_source", "none")
                    mp_detected = crop_src in ("haar", "mp")
                    hs_detected = crop_src != "none"

                    event_status = row.get("event_status", "IDLE")

                    elapsed    = float(row.get("timestamp", 0))
                    arousal    = float(row.get("hs_arousal", 0))
                    tension    = float(row.get("mp_tension", 0))
                    comp_fear  = float(row.get("composite_fear", 0))
                    smoothed   = float(row.get("smoothed_composite", 0))
                    latency    = float(row.get("latency_ms", 0)) if "latency_ms" in row else 0.0
                    hs_fear_sc = float(row.get("hs_fear", 0))
                    startle    = float(row.get("mp_startle_score", 0))
                    dom_score  = float(row.get("hs_dominant_score", 0))

                    bbox_str = row.get("face_bbox", "")
                    bbox = list(map(int, bbox_str.split(","))) if bbox_str else None

                    if bbox:
                        x1, y1, x2, y2 = [int(c * hud_scale) for c in bbox]
                        color = (0, 200, 0)
                        if event_status == "EVENT_CONFIRMED":
                            color = (0, 0, 220)
                        elif event_status in ("ONSET", "SUSTAINING"):
                            color = (0, 200, 255)
                        cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)

                    draw_on_video_bars(resized, hs_fear_sc, tension, smoothed)

                except (ValueError, TypeError):
                    elapsed = arousal = tension = comp_fear = smoothed = latency = 0.0
                    hs_fear_sc = startle = dom_score = 0.0

                ctx_tag      = row.get("mp_ctx_tag", "---")
                velocity_tag = row.get("mp_velocity_tag", "---")
                agreement    = row.get("agreement_tag", "---") or "---"
                veto_tag     = row.get("veto_tag", "---")
                dominant_hs  = row.get("hs_dominant", "Neutral")
                crop_src     = row.get("hs_crop_source", "none")
                event_status = row.get("event_status", "IDLE")
                # compact CSV doesn't carry face_detected columns — infer from crop_source
                mp_detected  = crop_src in ("haar", "mp")
                hs_detected  = crop_src != "none"

                # Reconstruct minimal emotion dict for HUD (only Fear available in compact)
                emotions_approx = {l: 0.0 for l in EMOTION_LABELS}
                emotions_approx["Fear"] = hs_fear_sc
                emotions_approx[dominant_hs] = dom_score

                # Timestamp overlay on video area
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                cv2.putText(canvas, f"TIME: {elapsed_min:02d}:{elapsed_sec:02d}",
                            (10, 25), font, 0.6, (200, 200, 200), 2)
                cv2.putText(canvas, f"{latency:.0f}ms",
                            (hud_vid_w - 80, 25), font, 0.6, (0, 255, 255), 2)

                # Event status badge in top-right of video area
                if event_status not in ("IDLE", ""):
                    badge_colors = {
                        "ONSET": (0, 200, 255),
                        "SUSTAINING": (0, 130, 255),
                        "EVENT_CONFIRMED": (0, 0, 220),
                        "EVENT_ENDED": (30, 180, 30),
                    }
                    bc = badge_colors.get(event_status, (150, 150, 150))
                    (tw, th), _ = cv2.getTextSize(event_status, font, 0.50, 2)
                    bpad = 5
                    bx2 = hud_vid_w - 8
                    bx1 = bx2 - tw - bpad * 2
                    by2 = 45 + th + bpad * 2
                    cv2.rectangle(canvas, (bx1, 45), (bx2, by2), bc, -1)
                    cv2.putText(canvas, event_status, (bx1 + bpad, 45 + th + bpad - 2),
                                font, 0.50, (255, 255, 255), 2)

                # Frame counter
                cv2.putText(canvas, f"Frame {frame_idx:04d}",
                            (10, canvas_h - 10), font, 0.4, (150, 150, 150), 1)

                # Side HUD panel
                mp_data_hud = {
                    "face_detected": mp_detected,
                    "tension": tension,
                    "face_valence": 0.0,
                    "smile_level": 0.0,
                    "ctx_tag": ctx_tag,
                    "startle_score": startle,
                    "velocity_tag": velocity_tag,
                    "bs_dict": {},
                }
                hs_data_hud = {
                    "face_detected": bool(hs_detected),
                    "arousal": arousal,
                    "valence": 0.0,
                    "dominant": dominant_hs,
                    "dominant_score": dom_score,
                    "emotions": emotions_approx,
                    "crop_source": crop_src,
                }
                fusion_data_hud = None
                if mode != "independent":
                    fusion_data_hud = {
                        "composite_fear": comp_fear,
                    }
                canvas = draw_hud(canvas, hud_vid_w, canvas_h,
                                  mp_data_hud, hs_data_hud, fusion_data_hud, mode)
            else:
                # Frame not in CSV (e.g. before analysis started) — plain timestamp
                cv2.putText(canvas, f"Frame {frame_idx:04d} | no data",
                            (10, 25), font, 0.5, (100, 100, 100), 1)

            video_writer.write(canvas)

            if frame_idx % 100 == 0:
                pct = 100 * frame_idx / max(total_frames, 1)
                print(f"\r  Rendering... {frame_idx}/{total_frames} ({pct:.1f}%)", end="")

        print(f"\r  Rendering... {frame_idx}/{total_frames} (100.0%) — done.          ")

    finally:
        cap.release()
        video_writer.release()

    print(f"Annotated video saved: {output_path}\n")
