"""prepare_merged_session.py — turn ONE recording into a render-ready merged session.

Bridges the two pipelines so the merged HUD (`render_merged_hud.py`) can run:

    recording.mp4  ──►  FER full CSV     (fer/test_mp_hs.py — or reuse existing)
                   ──►  rPPG frames.csv  (rppg offline ROI extraction — or cache)
                   ──►  rPPG analysis.csv (rppg/analyzer.py, POS+all @ 30s/5s)
                   ──►  merged_hud.mp4    (render_merged_hud.render_session)

Everything is collected into a self-contained session folder so it is reproducible
and so the batch driver (`build_presentation.py`) can iterate over many recordings.

Modular: import `prepare_session()` / `render_merged()` and call them, or run this
file directly for a guided single-recording run (no CLI flags)::

    python Pipeline/prepare_merged_session.py

IMPORTANT — offline, not live:
- FER runs offline via `fer/test_mp_hs.py --video PATH` (needs the `hsemotion`
  package). If `hsemotion` is unavailable or an FER CSV already exists for the
  recording, the existing FER output under `logs/sessions/` is reused.
- rPPG is offline too. `live_rppg.py` is camera-only and has NO --video mode, so
  we use its actual offline code path: `rppg/video_extractor.extract_or_load_cached`
  (MediaPipe ROI extraction, disk-cached) + `rppg/analyzer.RppgAnalyzer`.
"""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── path setup ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent            # Pipeline/
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rppg.analyzer import RppgAnalyzer, AnalysisConfig          # noqa: E402
from rppg.video_extractor import extract_or_load_cached          # noqa: E402
import render_merged_hud as rmh                                  # noqa: E402

# ── locations ────────────────────────────────────────────────────────────────
RECORDINGS_DIR    = REPO / "Recordings"
RPPG_CACHE_DIR    = HERE / "logs" / "rppg_cache"          # shared ROI-extraction cache
FER_SESSIONS_DIR  = HERE / "logs" / "sessions"            # where test_mp_hs.py keeps FER runs
PRESENTATION_DIR  = HERE / "presentation"
ANNOTATIONS_DIR   = REPO / "Annotations"

# rPPG analysis config — POS is production; "all" so the other-algo chips populate.
RPPG_ANALYSIS = AnalysisConfig(
    algorithm="all",
    window_s=float(rmh.POS_WINDOW_S),     # 30
    step_s=float(rmh.POS_STEP_S),         # 5
    bpm_min=60, bpm_max=180,
)

# Output encoding. OpenCV falls back to the bulky mp4v codec (no H.264 HW encoder
# here), so we post-transcode to H.264 (libx264) — ~5x smaller, universally playable.
TRANSCODE_H264 = True
# TRANSCODE_H264 = False     # keep the raw mp4v VideoWriter output
H264_CRF = 20                # quality (lower = better/larger); 18-23 is the sane range


# ============================================================================
# metadata helpers
# ============================================================================

def derive_ids(stem: str) -> Tuple[str, str]:
    """'S02_Vid04' -> (subject 'S02', video_id 'Vid04'). Falls back gracefully."""
    parts = stem.split("_", 1)
    subject = parts[0] if parts else stem
    video_id = parts[1] if len(parts) > 1 else ""
    return subject, video_id


def _hsemotion_available() -> bool:
    return importlib.util.find_spec("hsemotion") is not None


# ============================================================================
# FER side
# ============================================================================

def find_existing_fer_csv(subject: str, video_id: str, lighting: str) -> Optional[Path]:
    """Newest FER *full* CSV under logs/sessions/<subject>_<video>_<lighting>/.

    test_mp_hs.py renames its temp CSV to
    ``<sid>_mp_hs_<subject>_<video>_<lighting>.csv`` inside that folder.
    """
    folder = FER_SESSIONS_DIR / f"{subject}_{video_id}_{lighting}"
    if not folder.is_dir():
        return None
    cands = [p for p in folder.glob(f"*_mp_hs_{subject}_{video_id}_{lighting}.csv")
             if "compact" not in p.name]
    if not cands:
        # looser fallback: any non-compact mp_hs csv in the folder
        cands = [p for p in folder.glob("*_mp_hs_*.csv") if "compact" not in p.name]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def run_fer_offline(video: Path, subject: str, video_id: str, lighting: str) -> Optional[Path]:
    """Run fer/test_mp_hs.py on the recording (offline). Requires `hsemotion`.

    Drives the script's interactive prompts via stdin
    (Subject ID, Video ID, Lighting, then 'Keep? [Y/n]') and returns the FER full
    CSV it produces under logs/sessions/. Returns None on failure.
    """
    if not _hsemotion_available():
        print("  [FER] 'hsemotion' not installed — cannot run test_mp_hs.py here.")
        return None
    cmd = [sys.executable, str(HERE / "fer" / "test_mp_hs.py"),
           "--video", str(video), "--no-display"]
    stdin_data = f"{subject}\n{video_id}\n{lighting}\nY\n"
    print(f"  [FER] running test_mp_hs.py on {video.name} (this loads models + is slow) ...")
    try:
        subprocess.run(cmd, input=stdin_data, text=True, cwd=str(HERE), check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  [FER] test_mp_hs.py failed: {e}")
        return None
    return find_existing_fer_csv(subject, video_id, lighting)


def prepare_fer(video: Path, subject: str, video_id: str, lighting: str,
                force: bool = False) -> Optional[Path]:
    """Return a FER full CSV for the recording.

    Default: reuse an existing FER run if present (fast, works without hsemotion).
    force=True: re-run test_mp_hs.py (needs hsemotion), falling back to existing.
    """
    existing = find_existing_fer_csv(subject, video_id, lighting)
    if existing and not force:
        print(f"  [FER] reusing existing {existing.relative_to(HERE)}")
        return existing
    produced = run_fer_offline(video, subject, video_id, lighting)
    if produced:
        return produced
    if existing:
        print("  [FER] falling back to existing FER CSV.")
        return existing
    return None


# ============================================================================
# rPPG side
# ============================================================================

def prepare_rppg(video: Path, out_dir: Path, force: bool = False) -> Tuple[Path, Path]:
    """Offline rPPG: extract (cached) frames.csv + write analysis.csv into out_dir.

    Returns (frames_csv, analysis_csv).
    """
    RPPG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames_csv = extract_or_load_cached(video, RPPG_CACHE_DIR, extractor_type="mp",
                                        force=force)
    analysis_csv = out_dir / "analysis.csv"
    if analysis_csv.exists() and not force:
        print(f"  [rPPG] reusing existing {analysis_csv.relative_to(HERE)}")
        return frames_csv, analysis_csv
    print(f"  [rPPG] analysing {Path(frames_csv).name} "
          f"({RPPG_ANALYSIS.algorithm} @ {RPPG_ANALYSIS.window_s:.0f}s/"
          f"{RPPG_ANALYSIS.step_s:.0f}s) ...")
    results = RppgAnalyzer().analyze(Path(frames_csv), RPPG_ANALYSIS)
    if not results:
        raise RuntimeError(f"rPPG analysis produced no windows for {video.name}")
    RppgAnalyzer().save(results, analysis_csv)
    return Path(frames_csv), analysis_csv


# ============================================================================
# orchestration
# ============================================================================

def _link(src: Path, dst: Path) -> None:
    """Symlink dst -> src (absolute); replace if it already exists. Copy on failure."""
    src = Path(src).resolve()
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def prepare_session(video: Path, lighting: str, out_dir: Optional[Path] = None,
                    force_fer: bool = False, force_rppg: bool = False) -> Dict:
    """Assemble a render-ready session folder for one recording.

    Returns a dict of resolved paths + metadata (also written as meta.json).
    Raises if a required artefact (FER CSV or rPPG analysis) can't be produced.
    """
    video = Path(video)
    stem = video.stem
    subject, video_id = derive_ids(stem)
    out_dir = Path(out_dir) if out_dir else (PRESENTATION_DIR / stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] {stem}  (subject={subject} video={video_id} lighting={lighting})")

    fer_csv = prepare_fer(video, subject, video_id, lighting, force=force_fer)
    if not fer_csv:
        raise RuntimeError(
            f"No FER CSV for {stem}: none found under {FER_SESSIONS_DIR} and "
            f"test_mp_hs.py could not run (install 'hsemotion' to generate one).")
    frames_csv, analysis_csv = prepare_rppg(video, out_dir, force=force_rppg)

    # collect artefacts into the session folder
    _link(video, out_dir / "raw_video.mp4")
    _link(frames_csv, out_dir / "frames.csv")
    _link(fer_csv, out_dir / "fer.csv")

    # TODO(annotations): compare HUD verdicts (F12/F15 over time) against the
    # hand-annotated fear events in Annotations/<stem>.csv — e.g. precision/recall
    # of F15>=0.80 windows vs annotated onsets. Not implemented yet.
    annotation_csv = ANNOTATIONS_DIR / f"{stem}.csv"

    meta = {
        "stem": stem,
        "subject": subject,
        "video_id": video_id,
        "lighting": lighting,            # metadata only (not in folder name)
        "source_video": str(video.resolve()),
        "fer_csv": str(Path(fer_csv).resolve()),
        "frames_csv": str(Path(frames_csv).resolve()),
        "analysis_csv": str(analysis_csv.resolve()),
        "annotation_csv": str(annotation_csv) if annotation_csv.exists() else None,
        "rppg_config": {"algorithm": RPPG_ANALYSIS.algorithm,
                        "window_s": RPPG_ANALYSIS.window_s,
                        "step_s": RPPG_ANALYSIS.step_s},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    meta["out_dir"] = str(out_dir)
    return meta


def _ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None


def transcode_to_h264(mp4_path: Path, crf: int = H264_CRF) -> Path:
    """In-place transcode of an mp4v file to H.264 (libx264). No-op if ffmpeg/libx264
    is unavailable — the original mp4v is left untouched in that case."""
    mp4_path = Path(mp4_path)
    if not _ffmpeg_available():
        print("  [transcode] ffmpeg not found — keeping mp4v output.")
        return mp4_path
    tmp = mp4_path.with_name(mp4_path.stem + "__h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path),
           "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
           "-pix_fmt", "yuv420p", str(tmp)]
    print(f"  [transcode] mp4v → H.264 (crf {crf}) ...")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"  [transcode] failed ({e}) — keeping mp4v.")
        if tmp.exists():
            tmp.unlink()
        return mp4_path
    tmp.replace(mp4_path)        # replace original with the H.264 version (same name)
    return mp4_path


def render_merged(meta: Dict, out_mp4: Optional[Path] = None,
                  frame_stride: int = 1, max_frames: Optional[int] = None,
                  transcode: bool = TRANSCODE_H264) -> Path:
    """Render the merged HUD mp4 for a prepared session (dict from prepare_session).

    With transcode=True the (bulky mp4v) render is re-encoded in place to H.264.
    """
    out_dir = Path(meta["out_dir"])
    out_mp4 = Path(out_mp4) if out_mp4 else (out_dir / "merged_hud.mp4")
    rmh.render_session(
        video_path=out_dir / "raw_video.mp4",
        fer_csv=Path(meta["fer_csv"]),
        analysis_csv=Path(meta["analysis_csv"]),
        frames_csv=out_dir / "frames.csv",
        output_path=out_mp4,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    if transcode:
        transcode_to_h264(out_mp4)
    return out_mp4


def prepare_and_render(video: Path, lighting: str, out_dir: Optional[Path] = None,
                       frame_stride: int = 1, max_frames: Optional[int] = None,
                       force_fer: bool = False, force_rppg: bool = False) -> Path:
    meta = prepare_session(video, lighting, out_dir, force_fer, force_rppg)
    return render_merged(meta, frame_stride=frame_stride, max_frames=max_frames)


# ============================================================================
# interactive entry point
# ============================================================================

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        return input(f"{prompt}{suffix}: ").strip() or default
    except EOFError:
        return default


def main() -> None:
    print("=" * 72)
    print("  Prepare a merged FER + rPPG session from a recording, then render HUD")
    print("=" * 72)
    vids = sorted(RECORDINGS_DIR.glob("*.mp4"))
    if vids:
        print("\nRecordings in Recordings/:")
        for i, v in enumerate(vids):
            print(f"  {i:>2}  {v.name}")
    choice = _ask("\nPick a recording # (or type a video path)")
    if choice.isdigit() and int(choice) < len(vids):
        video = vids[int(choice)]
    else:
        video = Path(choice)
    if not video.exists():
        print(f"[error] video not found: {video}")
        sys.exit(1)

    subject, video_id = derive_ids(video.stem)
    lighting = _ask("Lighting (bright/dim/dark/mixed)", "bright")
    stride = _ask("Frame stride (1=full, 2-3=faster preview)", "1")
    stride = int(stride) if stride.isdigit() else 1
    out = prepare_and_render(video, lighting, frame_stride=stride)
    print(f"\nDone → {out}")


if __name__ == "__main__":
    main()
