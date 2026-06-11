"""Annotation helper for La Façade Fissuréе FER session CSVs.

Pattern: [Command] — keyboard dispatch table maps keystrokes to labeled annotation events.
Pattern: [Facade]  — entry point orchestrates annotation_io, annotation_algorithms, annotation_events.

Reads a test_mp_hs.py full CSV and produces an annotation template CSV with
ground-truth event timing (start_s, end_s) and category labels for later
precision/recall evaluation.

Two clustering methods:
    v1  peak detection + fixed 6 s window (original; predictable rhythm)
    v2  flood-fill clustering with confidence scoring (adaptive width)

Output CSV columns: frame, timestamp, start_s, end_s, category, confidence, notes

Usage:
    python fer/annotate_helper.py --csv logs/sessions/<session>/<file>.csv
    python fer/annotate_helper.py --csv <path> --version v1
    python fer/annotate_helper.py --csv <path> --version v2
    python fer/annotate_helper.py --csv <path> --version both

Keyboard shortcuts (interactive review mode):
    SPACE / n   next frame
    p           previous frame
    1–9         assign category label to current event
    s           save and advance
    q           quit and write output CSV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

from fer.annotation_events import FORMULAS, FORMULA_VOTE_THR
from fer.annotation_io import load_csv, recompute_composite_fear
from fer.annotation_algorithms import run_v1, run_v2, run_v3
from fer.compare_ground_truth_v2 import run_advanced_analysis, run_formula_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotation template from test_mp_hs.py CSV. "
                    "v3 (default): multi-channel voting. "
                    "v1: peak detection. v2: flood-fill. both: v1+v2.")
    parser.add_argument("--csv", required=True,
                        help="Path to mp_hs CSV (full, not compact)")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "both"],
                        default="v3",
                        help="Clustering method: v3 (multi-channel voting, default), "
                             "v1 (peak+6s), v2 (flood-fill), both (v1+v2)")
    # v1-specific
    parser.add_argument("--startle-threshold", type=float, default=2.5,
                        help="[v1] MP startle_score threshold (default: 2.5)")
    parser.add_argument("--fear-threshold", type=float, default=0.3,
                        help="[v1] HS fear score threshold (default: 0.3)")
    parser.add_argument("--min-gap", type=float, default=1.5,
                        help="[v1] Min gap between peaks in seconds (default: 1.5)")
    parser.add_argument("--window", type=float, default=3.0,
                        help="[v1] Window radius around each peak (default: 3.0)")
    # v2/v3-shared
    parser.add_argument("--gap-tolerance", type=float, default=None,
                        help="[v2/v3] Max gap between consecutive seed frames "
                             "in seconds (default: 1.0 for v2, 0.5 for v3)")
    parser.add_argument("--min-frames", type=int, default=2,
                        help="[v2/v3] Minimum seed frames per cluster (default: 2)")
    parser.add_argument("--seed", choices=["dominant", "threshold"], default="threshold",
                        help="[v2] Seed strategy: dominant (hs_dominant in Fear/Surprise) "
                             "or threshold (hs_fear>=0.15 OR hs_surprise>=0.25). "
                             "Default: threshold (more clusters)")
    parser.add_argument("--fear-seed-thr", type=float, default=0.15,
                        help="[v2] hs_fear threshold for threshold seed mode (default: 0.15)")
    parser.add_argument("--surprise-seed-thr", type=float, default=0.25,
                        help="[v2] hs_surprise threshold for threshold seed mode (default: 0.25)")
    parser.add_argument("--sustained-threshold", type=float, default=0.15,
                        help="[v2] composite_fear threshold for sustained%% "
                             "(default: 0.15)")
    parser.add_argument("--arousal-threshold", type=float, default=0.10,
                        help="[v2] mean_arousal threshold for HIGH confidence "
                             "(default: 0.10)")
    parser.add_argument("--sustained-pct-threshold", type=float, default=15.0,
                        help="[v2] sustained%% threshold for HIGH confidence "
                             "(default: 15.0)")
    # v3-specific
    parser.add_argument("--mode", choices=["auto", "continuous", "compilation"],
                        default="auto",
                        help="[v3] Video type: auto (default), continuous, "
                             "compilation (enables scene-cut detection)")
    parser.add_argument("--video", type=str, default=None,
                        help="[v3] Path to video file (for scene-cut detection "
                             "in compilation mode)")
    parser.add_argument("--vote-confirm", type=int, default=5,
                        help="[v3] Min channel votes for CONFIRMED tier (default: 5)")
    parser.add_argument("--vote-candidate", type=int, default=3,
                        help="[v3] Min channel votes for CANDIDATE tier (default: 3)")
    parser.add_argument("--min-duration", type=float, default=0.3,
                        help="[v3] Min cluster duration in seconds (default: 0.3)")
    parser.add_argument("--hide-candidate", action="store_true",
                        help="[v3] Only output CONFIRMED tier clusters")
    parser.add_argument("--cuts-file", type=str, default=None,
                        help="[v3] Path to manual scene-cut timestamps file")
    parser.add_argument("--rppg-csv", type=str, default=None,
                        help="[v3] Path to _rppg.csv sidecar (auto-detected if omitted). "
                             "Fills rppg_bpm_*/rppg_delta_* columns per cluster. "
                             "Leave blank to fill rppg_impression/rppg_notes manually.")
    parser.add_argument("--ch-fear-thr", type=float, default=0.15,
                        help="[v3] FEAR_THR channel: hs_fear threshold (default: 0.15)")
    parser.add_argument("--ch-tension-thr", type=float, default=0.20,
                        help="[v3] TENSION channel: mp_tension threshold (default: 0.20)")
    parser.add_argument("--ch-startle-thr", type=float, default=2.0,
                        help="[v3] STARTLE channel: mp_startle threshold (default: 2.0)")
    parser.add_argument("--ch-arousal-thr", type=float, default=0.30,
                        help="[v3] AROUSAL channel: hs_arousal threshold (default: 0.30)")
    parser.add_argument("--ch-cross-fear-thr", type=float, default=0.10,
                        help="[v3] CROSS_MODAL channel: hs_fear threshold (default: 0.10)")
    parser.add_argument("--ch-cross-tension-thr", type=float, default=0.10,
                        help="[v3] CROSS_MODAL channel: mp_tension threshold (default: 0.10)")
    parser.add_argument("--formula-vote-thr", type=float, default=FORMULA_VOTE_THR,
                        help=f"[v3] Peak score threshold for FX_vote / formulas_voted "
                             f"columns (default: {FORMULA_VOTE_THR})")
    # Shared (v1/v2)
    parser.add_argument("--formula", choices=list(FORMULAS.keys()), default="F2",
                        help="[v1/v2] Composite fear formula for scoring: "
                             "original (pre-computed), F2 (fear+surprise+tension), "
                             "F1 (fear+tension). Default: F2")
    parser.add_argument("--fn-rows", type=int, default=10,
                        help="Number of blank FN rows to append (default: 10)")
    parser.add_argument("--gt-file", type=str, default=None,
                        help="Path to GT annotation CSV. When provided, runs "
                             "compare_ground_truth_v2 P/R/F1 evaluation and "
                             "formula ranking after annotation.")
    args = parser.parse_args()

    # Resolve version-dependent defaults
    if args.gap_tolerance is None:
        args.gap_tolerance = 0.5 if args.version == "v3" else 1.0

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    # Load once, share between versions
    print(f"Loading: {csv_path.name}")
    rows = load_csv(csv_path)
    total_frames = len(rows)
    duration = rows[-1]["timestamp"] if rows else 0
    print(f"  Frames: {total_frames:,}  |  Duration: {duration:.1f}s")

    # Recompute composite_fear with selected formula (v1/v2 only; v3 uses F2 internally)
    if args.version in ("v1", "v2", "both"):
        recompute_composite_fear(rows, args.formula)

    if args.version == "v3":
        print("\n" + "=" * 60)
        run_v3(args, rows, csv_path)

    if args.version in ("v1", "both"):
        print("\n" + "=" * 60)
        run_v1(args, rows, csv_path)

    if args.version in ("v2", "both"):
        print("\n" + "=" * 60)
        run_v2(args, rows, csv_path)

    if args.version == "both":
        print("\n" + "=" * 60)
        print("Both templates generated. Compare side-by-side on the same video.")

    # ── GT evaluation (optional) ──────────────────────────────────────────────
    if args.gt_file:
        gt_path = Path(args.gt_file).resolve()
        if not gt_path.exists():
            print(f"\nERROR: GT file not found: {gt_path}")
        else:
            model_path = str(csv_path)
            print(f"\n{'='*62}")
            print(f"  GT Evaluation against: {gt_path.name}")
            print(f"{'='*62}")
            run_advanced_analysis(
                gt_file    = str(gt_path),
                model_file = model_path,
            )
            run_formula_comparison(
                gt_file    = str(gt_path),
                model_file = model_path,
            )


if __name__ == "__main__":
    main()
