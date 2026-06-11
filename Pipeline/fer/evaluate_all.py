"""Multi-video GT evaluation wrapper — La Façade Fissuréе.

Add annotation/model pairs to SESSIONS below, then run:
    python Pipeline/fer/evaluate_all.py

Flow:
  1. Load all sessions.
  2. Run combined v2 parameter sweep → find best (threshold, min_frames, window) per formula.
  3. Print per-session metrics table using the best combined params.
  4. Print combined aggregate using the best combined params.
  5. Generate per-session v2 + mp report log files (quiet).
  6. Run combined MP tension sweep (memory-efficient) using best v2 detection params.

Output saved to Pipeline/logs/comparisons/YYYYMMDD_evaluate_all.txt
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── Add Pipeline/ to path ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fer.compare_ground_truth_v2 import (
    SCOREABLE_LABELS,
    time_to_sec,
    _ALL_FORMULA_COLS,
    _FORMULA_LABELS,
    generate_v2_report,
    compute_metrics,
    print_parameter_sweep,
    print_best_configs_summary,
)

from fer.compare_ground_truth_mp import (
    generate_mp_report,
    run_combined_mp_sweep,
    MP_FORMULA_COLS,
)

# ---------------------------------------------------------------------------
# SESSIONS — add pairs here manually.
# Each entry: (gt_csv_path, model_csv_path)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESSIONS = [
    (
        os.path.join(_REPO_ROOT, "Annotations", "S02_Vid04.csv"),
        os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions", "S02_Vid04_dim", "20260503_153301_mp_hs_S02_Vid04_dim.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations", "S02_Vid05.csv"),
        os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions", "S02_Vid05_bright", "20260518_092016_mp_hs_S02_Vid05_bright.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations", "S08_Vid18.csv"),
        os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions", "S08_Vid18_mixed", "20260503_155009_mp_hs_S08_Vid18_mixed.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations", "S06_Vid16.csv"),
        os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions", "S06_Vid16_bright", "20260503_155724_mp_hs_S06_Vid16_bright.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations", "S04_Vid09.csv"),
        os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions", "S04_Vid09_bright", "20260517_154229_mp_hs_S04_Vid09_bright.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations", "S05_Vid10.csv"),
        os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions", "S05_Vid10_bright", "20260517_175428_mp_hs_S05_Vid10_bright.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations", "S10_Vid13.csv"),
        os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions", "S10_Vid13_bright", "20260517_200406_mp_hs_S10_Vid13_bright.csv"),
    ),
    # S08_Vid15 — annotation exists but no model CSV yet
    # S03_Vid08 — excluded (side-profile face detection failure)
    # S01_Vid03, S06_Vid14 — not annotated yet
]

# ---------------------------------------------------------------------------
# Fixed params (used for per-session report log generation only)
# The actual detection params for metrics come from the combined v2 sweep.
# ---------------------------------------------------------------------------

PAD_START  = 0.5
PAD_END    = 1.0
FILL_RATIO = 0.65

# Fallback detection params — used only if the v2 sweep produces no results.
_DEFAULT_THRESHOLD   = 0.70
_DEFAULT_MIN_FRAMES  = 10
_DEFAULT_WINDOW_SIZE = 15

# ---------------------------------------------------------------------------
# Log output
# ---------------------------------------------------------------------------

_LOG_DIR  = os.path.join(_REPO_ROOT, "Pipeline", "logs", "comparisons")
_RUN_DIR  = os.path.join(_LOG_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
_LOG_PATH = os.path.join(_RUN_DIR, "evaluate_all.txt")


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _load_gt(gt_file):
    raw_gt = pd.read_csv(gt_file, low_memory=False)
    raw_gt['start_val']  = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val']    = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    return raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)


def _enrich_formulas(df, anger_coeff=0.6):
    """Compute f7–f13 and recompute composite_fear in-place."""
    hs_fear    = pd.to_numeric(df.get('hs_fear',    0), errors='coerce').fillna(0.0)
    mp_t       = pd.to_numeric(df.get('mp_tension', 0), errors='coerce').fillna(0.0)
    hs_anger   = pd.to_numeric(df.get('hs_anger',   0), errors='coerce').fillna(0.0)
    hs_arousal = pd.to_numeric(df.get('hs_arousal',  0), errors='coerce').fillna(0.0)
    if 'f7'  not in df.columns: df['f7']  = hs_fear.clip(0, 1)
    if 'f8'  not in df.columns: df['f8']  = mp_t.clip(0, 1)
    if 'f9'  not in df.columns: df['f9']  = np.maximum(hs_fear, mp_t).clip(0, 1)
    if 'f10' not in df.columns: df['f10'] = np.sqrt((hs_fear * mp_t).clip(0, 1)).clip(0, 1)
    if 'f11' not in df.columns: df['f11'] = (hs_fear - anger_coeff * hs_anger).clip(0, 1)
    if 'f12' not in df.columns: df['f12'] = ((0.7 * hs_fear + 0.3 * hs_arousal) * (1.0 + mp_t)).clip(0, 1)
    if 'f13' not in df.columns: df['f13'] = (hs_fear * (1.0 + mp_t)).clip(0, 1)
    df['composite_fear'] = (hs_fear * (1.0 + mp_t)).clip(0, 1)


def run_f12_weight_sweep(loaded_sessions, threshold, min_frames, window_size,
                         fill_ratio, pad_start, pad_end):
    """Sweep fear/arousal weights and MP tension scale in the f12 formula."""
    fear_weights = np.arange(0.50, 0.95, 0.05)
    tension_scales = [0.5, 1.0, 1.5, 2.0]

    print(f"\n\n{'='*70}")
    print(f"  f12 Weight Sweep — {len(loaded_sessions)} sessions")
    print(f"  f12 = (w_fear * hs_fear + w_aro * hs_arousal) * (1 + k * mp_tension)")
    print(f"  Detection: thresh={threshold} mf={min_frames} "
          f"win={'strict' if window_size is None else window_size} fill={fill_ratio:.0%}")
    print(f"{'='*70}")
    print(f"  {'w_fear':>7} {'w_aro':>6} {'k_tens':>7}  "
          f"{'Prec':>8}  {'Rec':>8}  {'F1':>8}")

    best_f1, best_config = 0.0, None
    results = []

    for w_fear in fear_weights:
        w_aro = round(1.0 - w_fear, 2)
        for k in tension_scales:
            tot_tp = tot_fp = tot_gt_caught = tot_n_gt = 0
            for sd in loaded_sessions:
                df = sd['model_df']
                hs_fear = pd.to_numeric(df.get('hs_fear', 0), errors='coerce').fillna(0.0)
                hs_aro = pd.to_numeric(df.get('hs_arousal', 0), errors='coerce').fillna(0.0)
                mp_t = pd.to_numeric(df.get('mp_tension', 0), errors='coerce').fillna(0.0)
                df['_f12_sweep'] = ((w_fear * hs_fear + w_aro * hs_aro)
                                    * (1.0 + k * mp_t)).clip(0, 1)
                tp, fp, gt_caught, n_gt = compute_metrics(
                    df, sd['gt_df'], threshold, min_frames,
                    pad_start, pad_end, window_size, fill_ratio, '_f12_sweep',
                )
                tot_tp += tp; tot_fp += fp
                tot_gt_caught += gt_caught; tot_n_gt += n_gt

            p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
            r = tot_gt_caught / tot_n_gt if tot_n_gt > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            is_current = abs(w_fear - 0.70) < 0.001 and abs(k - 1.0) < 0.001
            marker = ""
            if f1 > best_f1:
                best_f1, best_config = f1, (w_fear, w_aro, k)
            if is_current:
                marker += " <-- CURRENT"
            results.append((w_fear, w_aro, k, p, r, f1, is_current))

    for w_fear, w_aro, k, p, r, f1, is_current in results:
        marker = ""
        if is_current:
            marker += " <-- CURRENT"
        if best_config and abs(w_fear - best_config[0]) < 0.001 \
                and abs(k - best_config[2]) < 0.001:
            marker += " <-- BEST" if not marker else " / BEST"
        print(f"  {w_fear:>7.2f} {w_aro:>6.2f} {k:>7.1f}  "
              f"{p:>8.2%}  {r:>8.2%}  {f1:>8.4f}{marker}")

    if best_config:
        print(f"\n  Best: w_fear={best_config[0]:.2f}  w_aro={best_config[1]:.2f}  "
              f"k_tension={best_config[2]:.1f}  F1={best_f1:.4f}")

    # cleanup temp column
    for sd in loaded_sessions:
        sd['model_df'].drop(columns=['_f12_sweep'], errors='ignore', inplace=True)
    sys.stdout.flush()


def run_evaluate_all():
    if not SESSIONS:
        print("SESSIONS list is empty — add (gt_csv, model_csv) pairs at the top of the file.")
        return

    # ── 1. Load sessions ────────────────────────────────────────────────────
    loaded_sessions = []
    all_cols = set()
    for gt_file, model_file in SESSIONS:
        try:
            model_df = pd.read_csv(model_file, low_memory=False)
            _enrich_formulas(model_df)
            gt_df    = _load_gt(gt_file)
            loaded_sessions.append({
                'gt_file':    gt_file,
                'model_file': model_file,
                'gt_df':      gt_df,
                'model_df':   model_df,
            })
            all_cols.update(c for c in _ALL_FORMULA_COLS if c in model_df.columns)
        except FileNotFoundError as e:
            print(f"  WARNING: file not found: {e.filename}", flush=True)

    present_cols = [c for c in _ALL_FORMULA_COLS if c in all_cols]
    if not present_cols:
        print("No recognised formula columns found in any model CSV.")
        return

    print(f"\n  Loaded {len(loaded_sessions)} session(s): "
          + ", ".join(os.path.basename(sd['model_file']) for sd in loaded_sessions),
          flush=True)

    # ── 2. Combined v2 parameter sweep (runs first — determines best params) ──
    print(f"\n\n{'='*70}")
    print(f"  --- Combined v2 Parameter Sweep ({len(loaded_sessions)} sessions) ---")
    print(f"  pad_start={PAD_START}s  pad_end={PAD_END}s  fill_ratio={FILL_RATIO:.0%}")
    print(f"{'='*70}", flush=True)

    sweep_thresholds   = [0.4, 0.5, 0.6, 0.70, 0.80]
    sweep_min_frames   = [5, 8, 10, 12, 15]
    sweep_window_sizes = [None, 10, 15, 20]

    best_per_formula = []

    for col in present_cols:
        label = _FORMULA_LABELS.get(col, col)

        def eval_combined(thresh, mf, ws, current_col=col):
            tot_tp = tot_fp = tot_gt_caught = tot_n_gt = 0
            for sd in loaded_sessions:
                if current_col in sd['model_df'].columns:
                    tp, fp, gt_caught, n_gt = compute_metrics(
                        sd['model_df'], sd['gt_df'], thresh, mf, PAD_START, PAD_END,
                        ws, FILL_RATIO, current_col,
                    )
                    tot_tp += tp; tot_fp += fp
                    tot_gt_caught += gt_caught; tot_n_gt += n_gt
            return tot_tp, tot_fp, tot_gt_caught, tot_n_gt

        best = print_parameter_sweep(
            formula_col=col,
            label=label,
            eval_func=eval_combined,
            thresholds=sweep_thresholds,
            min_frames=sweep_min_frames,
            window_sizes=sweep_window_sizes,
            pad_start=PAD_START,
            pad_end=PAD_END,
            fill_ratio=FILL_RATIO,
            print_table=False,
        )
        if best:
            best_per_formula.append(best)

    print_best_configs_summary(
        best_per_formula,
        f"Best v2 Detection Config Per Formula (Combined — {len(loaded_sessions)} sessions)"
    )
    sys.stdout.flush()

    # Extract best detection params (highest F1 formula wins)
    if best_per_formula:
        best_v2     = max(best_per_formula, key=lambda r: r['f1'])
        best_thresh = best_v2['thresh']
        best_min_fr = best_v2['min_fr']
        best_win    = None if best_v2['win'] == 'strict' else int(best_v2['win'])
        print(f"\n  → Best overall: {best_v2['formula']} ({best_v2['label']})"
              f"  thresh={best_thresh}  min_frames={best_min_fr}  window={best_v2['win']}"
              f"  F1={best_v2['f1']:.4f}", flush=True)
    else:
        best_thresh = _DEFAULT_THRESHOLD
        best_min_fr = _DEFAULT_MIN_FRAMES
        best_win    = _DEFAULT_WINDOW_SIZE
        print(f"  → No sweep results; using defaults: "
              f"thresh={best_thresh}  min_frames={best_min_fr}  window={best_win}", flush=True)

    # ── 3. Per-session metrics table (using best combined params) ────────────
    print(f"\n\n{'='*70}")
    print(f"  evaluate_all.py — {len(loaded_sessions)} session(s)")
    print(f"  threshold={best_thresh}  min_frames={best_min_fr}  "
          f"window={best_win}  fill={FILL_RATIO:.0%}  [best combined params]")
    print(f"  pad_start={PAD_START}s  pad_end={PAD_END}s")
    print(f"{'='*70}", flush=True)

    aggregate = {col: {'tp': 0, 'fp': 0, 'gt_caught': 0, 'n_gt': 0} for col in present_cols}

    for session_data in loaded_sessions:
        session_name = os.path.basename(session_data['model_file'])
        print(f"\n--- Session: {session_name} ---")
        print(f"    GT: {os.path.basename(session_data['gt_file'])}")
        print(f"  {'Formula':<10}  {'Label':<12}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
              f"{'Prec':>8}  {'Rec':>8}  {'F1':>8}")

        for col in present_cols:
            if col not in session_data['model_df'].columns:
                print(f"  {col:<10}  {'(missing)':<12}  {'—':>4}")
                continue

            tp, fp, gt_caught, n_gt = compute_metrics(
                session_data['model_df'], session_data['gt_df'],
                best_thresh, best_min_fr, PAD_START, PAD_END,
                best_win, FILL_RATIO, col,
            )
            prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec   = gt_caught / n_gt if n_gt > 0 else 0.0
            f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            label = _FORMULA_LABELS.get(col, col)
            fn    = n_gt - gt_caught

            print(f"  {col:<10}  {label:<12}  {tp:>4}  {fp:>4}  {fn:>4}  "
                  f"{prec:>8.2%}  {rec:>8.2%}  {f1:>8.4f}")

            aggregate[col]['tp']        += tp
            aggregate[col]['fp']        += fp
            aggregate[col]['gt_caught'] += gt_caught
            aggregate[col]['n_gt']      += n_gt

        sys.stdout.flush()

    # ── 4. Combined aggregate ───────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  --- Combined Aggregate ({len(loaded_sessions)} sessions) ---")
    print(f"{'='*70}")
    print(f"  {'Formula':<10}  {'Label':<12}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
          f"{'Prec':>8}  {'Rec':>8}  {'F1':>8}")

    best_f1  = 0.0
    agg_rows = []
    for col in present_cols:
        a = aggregate[col]
        tp, fp, gt_caught, n_gt = a['tp'], a['fp'], a['gt_caught'], a['n_gt']
        fn   = n_gt - gt_caught
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = gt_caught / n_gt if n_gt > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        agg_rows.append((col, _FORMULA_LABELS.get(col, col), tp, fp, fn, prec, rec, f1))
        best_f1 = max(best_f1, f1)

    for col, label, tp, fp, fn, prec, rec, f1 in agg_rows:
        marker = "  <-- BEST" if abs(f1 - best_f1) < 1e-9 else ""
        print(f"  {col:<10}  {label:<12}  {tp:>4}  {fp:>4}  {fn:>4}  "
              f"{prec:>8.2%}  {rec:>8.2%}  {f1:>8.4f}{marker}")
    print()
    sys.stdout.flush()

    # ── 5. Generate per-session report log files (v2 + mp) ──────────────────
    print(f"\n  Generating per-session report logs...", flush=True)
    for session_data in loaded_sessions:
        generate_v2_report(
            gt_file     = session_data['gt_file'],
            model_file  = session_data['model_file'],
            threshold   = best_thresh,
            min_frames  = best_min_fr,
            pad_start   = PAD_START,
            pad_end     = PAD_END,
            window_size = best_win,
            fill_ratio  = FILL_RATIO,
            quiet       = True,
            output_dir  = _RUN_DIR,
        )
        generate_mp_report(
            gt_file     = session_data['gt_file'],
            model_file  = session_data['model_file'],
            threshold   = best_thresh,
            min_frames  = best_min_fr,
            pad_start   = PAD_START,
            pad_end     = PAD_END,
            window_size = best_win,
            fill_ratio  = FILL_RATIO,
            quiet       = True,
            output_dir  = _RUN_DIR,
        )
    sys.stdout.flush()

    # ── 6. Combined MP tension sweep (memory-efficient, using best v2 params) ─
    # All 12 formula columns are always produced by _recompute_formulas — no need
    # to filter by what's pre-stored in the raw model CSV.
    has_mp = any('mp_tension' in sd['model_df'].columns for sd in loaded_sessions)
    if not has_mp:
        print("\n  [mp sweep] No MP tension column found — skipping.")
        return

    run_combined_mp_sweep(
        sessions    = loaded_sessions,
        threshold   = best_thresh,
        min_frames  = best_min_fr,
        window_size = best_win,
        fill_ratio  = FILL_RATIO,
        pad_start   = PAD_START,
        pad_end     = PAD_END,
        formula_cols = MP_FORMULA_COLS,
        title=(f"MP Tension Config Sweep — Combined {len(loaded_sessions)} sessions  "
               f"[thresh={best_thresh} mf={best_min_fr} win={best_win}]"),
    )

    # ── 7. f12 weight sweep ─────────────────────────────────────────────────
    run_f12_weight_sweep(
        loaded_sessions, best_thresh, best_min_fr, best_win,
        FILL_RATIO, PAD_START, PAD_END,
    )


# ---------------------------------------------------------------------------
# FER Statistical Validation Suite (A9)
# ---------------------------------------------------------------------------

def run_fer_validation(loaded_sessions):
    """A9: FER statistical validation — permutation, leave-k-out CV, per-session F1, weighting."""
    from itertools import combinations

    n_perms = 10000
    formula_col = "f12"

    sweep_thresholds = [0.4, 0.5, 0.6, 0.70, 0.80]
    sweep_min_frames = [5, 8, 10, 12, 15]
    sweep_window_sizes = [None, 10, 15, 20]

    def _eval_sessions(sessions, col=formula_col):
        """Evaluate f12 across given sessions, return (tp, fp, gt_caught, n_gt)."""
        tp = fp = gc = ng = 0
        for sd in sessions:
            if col in sd["model_df"].columns:
                t, f, g, n = compute_metrics(
                    sd["model_df"], sd["gt_df"], _DEFAULT_THRESHOLD, _DEFAULT_MIN_FRAMES,
                    PAD_START, PAD_END, None, FILL_RATIO, col,
                )
                tp += t; fp += f; gc += g; ng += n
        return tp, fp, gc, ng

    def _f1_from_counts(tp, fp, gc, ng):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = gc / ng if ng > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def _sweep_best(sessions, col=formula_col):
        """Run parameter sweep on sessions, return best config dict."""
        def ef(thresh, mf, ws, _col=col, _ss=sessions):
            tp = fp = gc = ng = 0
            for sd in _ss:
                if _col in sd["model_df"].columns:
                    t, f, g, n = compute_metrics(
                        sd["model_df"], sd["gt_df"], thresh, mf,
                        PAD_START, PAD_END, ws, FILL_RATIO, _col,
                    )
                    tp += t; fp += f; gc += g; ng += n
            return tp, fp, gc, ng

        best = print_parameter_sweep(
            formula_col=col, label="f12_cv", eval_func=ef,
            thresholds=sweep_thresholds, min_frames=sweep_min_frames,
            window_sizes=sweep_window_sizes, pad_start=PAD_START,
            pad_end=PAD_END, fill_ratio=FILL_RATIO, print_table=False,
        )
        return best

    def _eval_with_params(sessions, thresh, mf, ws, col=formula_col):
        """Evaluate sessions with specific params."""
        tp = fp = gc = ng = 0
        for sd in sessions:
            if col in sd["model_df"].columns:
                t, f, g, n = compute_metrics(
                    sd["model_df"], sd["gt_df"], thresh, mf,
                    PAD_START, PAD_END, ws, FILL_RATIO, col,
                )
                tp += t; fp += f; gc += g; ng += n
        return _f1_from_counts(tp, fp, gc, ng)

    # ── Baseline F1 at locked params ──
    tp0, fp0, gc0, ng0 = _eval_sessions(loaded_sessions)
    base_f1 = _f1_from_counts(tp0, fp0, gc0, ng0)
    base_prec = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
    base_rec = gc0 / ng0 if ng0 > 0 else 0.0

    print(f"\n  Baseline f12 (locked params): F1={base_f1:.4f} "
          f"(P={base_prec:.1%}, R={base_rec:.1%}, TP={tp0}, FP={fp0}, "
          f"GT={gc0}/{ng0})")

    # ── A9a: Permutation test ──
    print(f"\n  {'='*60}")
    print(f"  A9a. Permutation Test — f12 (n={n_perms})")
    print(f"  {'='*60}")

    all_gt_events = []
    session_durations = []
    for sd in loaded_sessions:
        gt_df = sd["gt_df"]
        frame_ts = pd.to_numeric(
            sd["model_df"]["timestamp"], errors="coerce"
        ).fillna(0.0).values
        dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0
        session_durations.append(dur)
        for _, row in gt_df.iterrows():
            all_gt_events.append({
                "start_val": row["start_val"], "end_val": row["end_val"],
                "duration": row["end_val"] - row["start_val"],
            })

    rng = np.random.default_rng(42)
    n_exceed = 0

    for perm_i in range(n_perms):
        tot_tp = tot_fp = tot_gc = tot_ng = 0
        for si, sd in enumerate(loaded_sessions):
            gt_df = sd["gt_df"]
            dur = session_durations[si]
            n_events = len(gt_df)
            if n_events == 0 or dur <= 0:
                continue
            fake_starts = rng.uniform(0, max(dur - 5, 1), size=n_events)
            durations = (gt_df["end_val"] - gt_df["start_val"]).values
            fake_gt = pd.DataFrame({
                "start_val": fake_starts,
                "end_val": fake_starts + durations,
                "label_norm": "fear",
                "start_s": fake_starts,
                "label": "fear",
            })
            tp, fp, gc, ng = compute_metrics(
                sd["model_df"], fake_gt, _DEFAULT_THRESHOLD, _DEFAULT_MIN_FRAMES,
                PAD_START, PAD_END, None, FILL_RATIO, formula_col,
            )
            tot_tp += tp; tot_fp += fp; tot_gc += gc; tot_ng += ng

        perm_f1 = _f1_from_counts(tot_tp, tot_fp, tot_gc, tot_ng)
        if perm_f1 >= base_f1:
            n_exceed += 1

    perm_p = (n_exceed + 1) / (n_perms + 1)
    print(f"  Observed F1 = {base_f1:.4f}")
    print(f"  Permutations with F1 ≥ observed: {n_exceed}/{n_perms}")
    print(f"  Permutation p-value: {perm_p:.4f}")

    # ── A9b: Leave-k-out cross-validation ──
    print(f"\n  {'='*60}")
    print(f"  A9b. Leave-k-out Cross-Validation — f12")
    print(f"  {'='*60}")

    n_sess = len(loaded_sessions)
    for k in [1, 2, 3]:
        if k >= n_sess:
            continue
        folds = list(combinations(range(n_sess), k))
        test_f1s_locked = []
        test_f1s_optimized = []

        for held_out_idx in folds:
            train = [loaded_sessions[i] for i in range(n_sess) if i not in held_out_idx]
            test = [loaded_sessions[i] for i in held_out_idx]

            tp_t, fp_t, gc_t, ng_t = _eval_sessions(test)
            test_f1_locked = _f1_from_counts(tp_t, fp_t, gc_t, ng_t)
            test_f1s_locked.append(test_f1_locked)

            train_best = _sweep_best(train)
            if train_best:
                opt_ws = None if train_best["win"] == "strict" else int(train_best["win"])
                test_f1_opt = _eval_with_params(
                    test, train_best["thresh"], train_best["min_fr"], opt_ws)
            else:
                test_f1_opt = test_f1_locked
            test_f1s_optimized.append(test_f1_opt)

        mean_locked = float(np.mean(test_f1s_locked))
        std_locked = float(np.std(test_f1s_locked, ddof=1)) if len(test_f1s_locked) > 1 else 0
        mean_opt = float(np.mean(test_f1s_optimized))
        std_opt = float(np.std(test_f1s_optimized, ddof=1)) if len(test_f1s_optimized) > 1 else 0

        print(f"\n  Leave-{k}-out: {len(folds)} folds")
        print(f"    Locked params:     F1 = {mean_locked:.4f} ± {std_locked:.4f}")
        print(f"    Train-optimized:   F1 = {mean_opt:.4f} ± {std_opt:.4f}")
        print(f"    Overfit gap:       {mean_opt - mean_locked:+.4f}")

    # ── A9c: Per-session F1 breakdown ──
    print(f"\n  {'='*60}")
    print(f"  A9c. Per-Session F1 Breakdown — f12")
    print(f"  {'='*60}")
    print(f"  {'Session':<40}  {'n_GT':>5}  {'F1_locked':>10}  {'F1_optimal':>10}  {'Gap':>7}")

    for sd in loaded_sessions:
        tag = os.path.basename(sd["model_file"])[:25]
        n_gt = len(sd["gt_df"])
        tp_l, fp_l, gc_l, ng_l = _eval_sessions([sd])
        f1_locked = _f1_from_counts(tp_l, fp_l, gc_l, ng_l)
        best_sd = _sweep_best([sd])
        f1_opt = best_sd["f1"] if best_sd else f1_locked
        gap = f1_opt - f1_locked
        print(f"  {tag:<40}  {n_gt:>5}  {f1_locked:>10.4f}  {f1_opt:>10.4f}  {gap:>+7.4f}")

    # ── A9d: Per-session vs per-event weighting ──
    print(f"\n  {'='*60}")
    print(f"  A9d. Per-Session vs Per-Event Weighting — f12")
    print(f"  {'='*60}")

    per_session_f1s = []
    per_session_weights = []
    for sd in loaded_sessions:
        tp_s, fp_s, gc_s, ng_s = _eval_sessions([sd])
        f1_s = _f1_from_counts(tp_s, fp_s, gc_s, ng_s)
        per_session_f1s.append(f1_s)
        per_session_weights.append(ng_s)

    per_event_f1 = base_f1

    total_events = sum(per_session_weights)
    weighted_f1 = sum(f * w for f, w in zip(per_session_f1s, per_session_weights)) / total_events if total_events > 0 else 0
    macro_f1 = float(np.mean(per_session_f1s)) if per_session_f1s else 0

    print(f"\n  Per-event (micro) F1:    {per_event_f1:.4f}  (each event counts equally)")
    print(f"  Event-weighted F1:       {weighted_f1:.4f}  (Σ session_F1 × n_events / total)")
    print(f"  Per-session (macro) F1:  {macro_f1:.4f}  (each session counts equally)")
    print(f"\n  Per-session F1 values: {[f'{f:.4f}' for f in per_session_f1s]}")
    print(f"  Per-session GT counts: {per_session_weights}")

    if len(per_session_f1s) > 1:
        std_f1 = float(np.std(per_session_f1s, ddof=1))
        print(f"  Session F1 std:          {std_f1:.4f}")

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="Multi-video GT evaluation")
    _parser.add_argument("--rppg", action="store_true",
                         help="Also run rPPG × FER correlation analysis")
    _parser.add_argument("--rppg-ext-all", action="store_true",
                         help="Run all Phase 2 rPPG extensions")
    _parser.add_argument("--validate", action="store_true",
                         help="Run FER statistical validation suite (A9)")
    _args = _parser.parse_args()

    os.makedirs(_RUN_DIR, exist_ok=True)
    with open(_LOG_PATH, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(sys.stdout, _fh)
        try:
            run_evaluate_all()
        finally:
            sys.stdout = _orig
    print(f"\nLog saved → {_LOG_PATH}")

    if _args.validate:
        loaded_sessions = []
        for gt_file, model_file in SESSIONS:
            try:
                model_df = pd.read_csv(model_file, low_memory=False)
                _enrich_formulas(model_df)
                gt_df = _load_gt(gt_file)
                loaded_sessions.append({
                    'gt_file': gt_file, 'model_file': model_file,
                    'gt_df': gt_df, 'model_df': model_df,
                })
            except FileNotFoundError:
                pass
        if loaded_sessions:
            val_log = os.path.join(_RUN_DIR, "fer_validation.txt")
            with open(val_log, 'w') as _fh:
                _orig, sys.stdout = sys.stdout, _Tee(sys.stdout, _fh)
                try:
                    print(f"\n{'='*70}")
                    print(f"  FER Statistical Validation Suite — {len(loaded_sessions)} sessions")
                    print(f"{'='*70}")
                    run_fer_validation(loaded_sessions)
                finally:
                    sys.stdout = _orig
            print(f"Validation log saved → {val_log}")

    if _args.rppg:
        from rppg.evaluate_rppg import run_evaluate_rppg
        _ext = {"facedet": True, "peak": True, "multiwin": True,
                "zscore": True, "variability": True} if _args.rppg_ext_all else {}
        for _mode in ("ALL", "GAMEPLAY"):
            _rppg_log = os.path.join(_RUN_DIR, f"evaluate_rppg_{_mode}.txt")
            with open(_rppg_log, 'w') as _fh:
                _orig, sys.stdout = sys.stdout, _Tee(sys.stdout, _fh)
                try:
                    run_evaluate_rppg(_mode, output_dir=_RUN_DIR,
                                      ext_flags=_ext)
                finally:
                    sys.stdout = _orig
        print(f"rPPG logs saved → {_RUN_DIR}")
