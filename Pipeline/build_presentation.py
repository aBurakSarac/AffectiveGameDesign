"""build_presentation.py — batch-build merged FER + rPPG HUDs for the demo set.

For each listed recording in Recordings/ this:
  1. prepares a render-ready session (FER full CSV + rPPG frames/analysis), and
  2. renders the merged HUD mp4,
into Pipeline/presentation/<recording-stem>/  (folder name excludes lighting;
lighting is kept as session metadata in meta.json).

Run (no flags)::

    python Pipeline/build_presentation.py

Tunables are module constants below. Long recordings (10-15 min) render slowly at
full fidelity (~1 frame/~90ms → ~25-30 min each); set PREVIEW_STRIDE>1 or
PREVIEW_MAX_FRAMES for a fast first pass.

TODO(annotations): once HUDs are built, compare F12/F15 verdict windows against the
hand-annotated fear events in Annotations/<stem>.csv (precision/recall of
F15>=0.80 vs annotated onsets) and emit a per-recording agreement report. The
per-session annotation path is already recorded in each meta.json.
"""

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prepare_merged_session as pms      # noqa: E402

# ── the demo set: (recording stem, lighting). Lighting is metadata only — it is
#    NOT part of the output folder name (folder = stem, e.g. "S02_Vid04"). ──
RECORDINGS = [
    ("S02_Vid04", "dim"),
    ("S02_Vid05", "bright"),
    ("S04_Vid09", "bright"),
    ("S05_Vid10", "bright"),
    ("S06_Vid16", "bright"),
    ("S08_Vid18", "mixed"),
    ("S10_Vid13", "bright"),
]

# ── tunables ─────────────────────────────────────────────────────────────────
PREVIEW_STRIDE    = 1       # 1 = full 30fps fidelity; 2-3 = faster preview, real-time playback
PREVIEW_MAX_FRAMES = None   # e.g. 900 for a ~quick smoke render per video; None = whole video
FORCE_FER         = False   # re-run test_mp_hs.py even if an FER CSV exists (needs hsemotion)
FORCE_RPPG        = False   # re-extract + re-analyse rPPG even if cached/analysed
SKIP_IF_RENDERED  = True    # skip a recording whose merged_hud.mp4 already exists


def main() -> None:
    print("=" * 78)
    print("  Building merged FER + rPPG presentation HUDs")
    print(f"  → {pms.PRESENTATION_DIR}")
    print("=" * 78)

    results = []
    for stem, lighting in RECORDINGS:
        video = pms.RECORDINGS_DIR / f"{stem}.mp4"
        out_dir = pms.PRESENTATION_DIR / stem
        out_mp4 = out_dir / "merged_hud.mp4"
        print("\n" + "-" * 78)
        if not video.exists():
            print(f"[skip] {stem}: recording not found at {video}")
            results.append((stem, "missing-video", None))
            continue
        if SKIP_IF_RENDERED and out_mp4.exists():
            print(f"[skip] {stem}: {out_mp4.name} already exists "
                  f"(set SKIP_IF_RENDERED=False to rebuild)")
            results.append((stem, "skipped", out_mp4))
            continue
        t0 = time.time()
        try:
            meta = pms.prepare_session(video, lighting, out_dir,
                                       force_fer=FORCE_FER, force_rppg=FORCE_RPPG)
            pms.render_merged(meta, out_mp4=out_mp4,
                              frame_stride=PREVIEW_STRIDE, max_frames=PREVIEW_MAX_FRAMES)
            dt = time.time() - t0
            print(f"[ok] {stem}: {out_mp4}  ({dt:.0f}s)")
            results.append((stem, "ok", out_mp4))
        except Exception as e:        # keep the batch going if one recording fails
            print(f"[FAIL] {stem}: {e}")
            traceback.print_exc()
            results.append((stem, f"fail: {e}", None))

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for stem, status, out in results:
        loc = f"  → {out}" if out else ""
        print(f"  {stem:14s}  {status}{loc}")


if __name__ == "__main__":
    main()
