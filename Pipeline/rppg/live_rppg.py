"""rPPG CLI entry point — capture, analyze, replay, compare, list.

Usage:
    python Pipeline/rppg/live_rppg.py capture [--label NAME] [--gt PATH] [--no-preview] [--duration S]
    python Pipeline/rppg/live_rppg.py analyze  --session ID   [--gt PATH]
    python Pipeline/rppg/live_rppg.py replay   --session ID
    python Pipeline/rppg/live_rppg.py compare  --session ID   --gt PATH
    python Pipeline/rppg/live_rppg.py list
"""

import argparse
import sys
from pathlib import Path

# Allow running from anywhere: python Pipeline/rppg/live_rppg.py ...
_HERE = Path(__file__).resolve().parent           # Pipeline/rppg/
_PIPELINE = _HERE.parent                          # Pipeline/
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))            # Pipeline/ — rppg is a sub-package here


from rppg.capture import CaptureConfig
from rppg.analyzer import AnalysisConfig
from rppg.session import RppgSession, SessionRepository


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_capture(args: argparse.Namespace) -> None:
    cfg = CaptureConfig(
        camera_index=args.camera,
        preview=not args.no_preview,
        max_duration_s=args.duration,
    )
    analysis_cfg = AnalysisConfig(
        window_s=args.window,
        step_s=args.step,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        harmonic_disambiguation=not args.no_harmonic_disambig,
        harmonic_ratio=args.harmonic_ratio,
    )
    gt_path = Path(args.gt) if args.gt else None
    session = RppgSession()
    session.capture(args.label, capture_config=cfg,
                    analysis_config=analysis_cfg, gt_path=gt_path)


def cmd_reextract(args: argparse.Namespace) -> None:
    analysis_cfg = AnalysisConfig(
        window_s=args.window,
        step_s=args.step,
        roi_mode=args.roi,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        harmonic_disambiguation=not args.no_harmonic_disambig,
        harmonic_ratio=args.harmonic_ratio,
    )
    gt_path = Path(args.gt) if args.gt else None
    session = RppgSession()
    session.reextract(args.session, extractor_type=args.extractor,
                      analysis_config=analysis_cfg, gt_path=gt_path)


def cmd_analyze(args: argparse.Namespace) -> None:
    analysis_cfg = AnalysisConfig(
        window_s=args.window,
        step_s=args.step,
        roi_mode=args.roi,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        harmonic_disambiguation=not args.no_harmonic_disambig,
        harmonic_ratio=args.harmonic_ratio,
    )
    gt_path = Path(args.gt) if args.gt else None
    session = RppgSession()
    session.analyze(args.session, analysis_config=analysis_cfg, gt_path=gt_path)


def cmd_replay(args: argparse.Namespace) -> None:
    session = RppgSession()
    replay_path = session.replay(args.session)
    print(f"Replay saved: {replay_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    gt_path = Path(args.gt)
    session = RppgSession()
    session.compare(args.session, gt_path=gt_path)


def cmd_list(args: argparse.Namespace) -> None:
    repo = SessionRepository()
    folders = repo.list()
    if not folders:
        print("No sessions found.")
        return
    print(f"\n{'ID':40s}  {'Video':5s}  {'Analysis':8s}  {'GT':3s}  {'Replay':6s}")
    print("-" * 70)
    for f in folders:
        has_video    = "yes" if f.raw_video    else "no"
        has_analysis = "yes" if f.analysis_csv else "no"
        has_gt       = "yes" if f.gt_csv       else "no"
        has_replay   = "yes" if f.replay_video else "no"
        print(f"{f.id:40s}  {has_video:5s}  {has_analysis:8s}  "
              f"{has_gt:3s}  {has_replay:6s}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_rppg",
        description="rPPG live capture and analysis pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── capture ──────────────────────────────────────────────────────
    p_cap = sub.add_parser("capture", help="Record a live session")
    p_cap.add_argument("--label",      default="session",
                       help="Short label for the session folder (default: session)")
    p_cap.add_argument("--gt",         default=None,
                       help="Path to Zepp ZIP or GT CSV for ground-truth comparison")
    p_cap.add_argument("--no-preview", action="store_true",
                       help="Disable live preview window")
    p_cap.add_argument("--camera",     type=int, default=0,
                       help="Camera index (default: 0)")
    p_cap.add_argument("--duration",   type=float, default=0.0,
                       help="Max capture duration in seconds (0 = unlimited)")
    p_cap.add_argument("--window",     type=float, default=30.0,
                       help="rPPG analysis window in seconds (default: 30)")
    p_cap.add_argument("--step",       type=float, default=5.0,
                       help="rPPG step size in seconds (default: 5)")
    p_cap.add_argument("--bpm-min",    type=int, default=60,
                       help="Minimum BPM (default: 60)")
    p_cap.add_argument("--bpm-max",    type=int, default=180,
                       help="Maximum BPM (default: 180)")
    p_cap.add_argument("--no-harmonic-disambig", action="store_true",
                       help="Disable 2x harmonic disambiguation (use raw FFT peak)")
    p_cap.add_argument("--harmonic-ratio", type=float, default=0.6,
                       help="Power ratio threshold for preferring 2x harmonic (default: 0.6)")

    # ── reextract ────────────────────────────────────────────────────
    p_rex = sub.add_parser(
        "reextract",
        help="Re-run ROI extraction on raw_video.mp4 then re-analyze "
             "(use after changing landmark/ROI code)",
    )
    p_rex.add_argument("--session", required=True,
                       help="Session ID (folder name under Pipeline/sessions/)")
    p_rex.add_argument("--gt",      default=None,
                       help="Path to Zepp ZIP or GT CSV")
    p_rex.add_argument("--window",  type=float, default=30.0)
    p_rex.add_argument("--step",    type=float, default=5.0)
    p_rex.add_argument("--bpm-min", type=int, default=60)
    p_rex.add_argument("--bpm-max", type=int, default=180)
    p_rex.add_argument("--no-harmonic-disambig", action="store_true")
    p_rex.add_argument("--harmonic-ratio", type=float, default=0.6)
    p_rex.add_argument("--roi",       default="primary",
                       choices=["primary", "forehead", "glabella", "malar"],
                       help="ROI source for analysis (default: primary)")
    p_rex.add_argument("--extractor", default="mp",
                       choices=["haar", "mp"],
                       help="Extractor to use: 'haar' recovers v1 sessions "
                            "(Haar cascade, fixed forehead crop), "
                            "'mp' uses current MediaPipe extractor (default)")

    # ── analyze ──────────────────────────────────────────────────────
    p_ana = sub.add_parser("analyze",
                            help="Re-analyze an existing captured session")
    p_ana.add_argument("--session", required=True,
                       help="Session ID (folder name under Pipeline/sessions/)")
    p_ana.add_argument("--gt",      default=None,
                       help="Path to Zepp ZIP or GT CSV")
    p_ana.add_argument("--window",  type=float, default=30.0)
    p_ana.add_argument("--step",    type=float, default=5.0)
    p_ana.add_argument("--bpm-min", type=int, default=60)
    p_ana.add_argument("--bpm-max", type=int, default=180)
    p_ana.add_argument("--no-harmonic-disambig", action="store_true")
    p_ana.add_argument("--harmonic-ratio", type=float, default=0.6)
    p_ana.add_argument("--roi",     default="primary",
                       choices=["primary", "forehead", "glabella", "malar"],
                       help="ROI source for analysis (default: primary)")

    # ── replay ───────────────────────────────────────────────────────
    p_rep = sub.add_parser("replay",
                            help="Re-render replay video for an analyzed session")
    p_rep.add_argument("--session", required=True)

    # ── compare ──────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare",
                            help="Add GT comparison to an already-analyzed session")
    p_cmp.add_argument("--session", required=True)
    p_cmp.add_argument("--gt",      required=True,
                       help="Path to Zepp ZIP or GT CSV")

    # ── list ─────────────────────────────────────────────────────────
    sub.add_parser("list", help="List all rPPG sessions")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "capture":    cmd_capture,
        "reextract":  cmd_reextract,
        "analyze":    cmd_analyze,
        "replay":     cmd_replay,
        "compare":    cmd_compare,
        "list":       cmd_list,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
