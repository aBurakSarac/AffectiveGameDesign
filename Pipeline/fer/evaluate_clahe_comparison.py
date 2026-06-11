"""CLAHE A/B comparison — evaluate same sessions with and without CLAHE preprocessing.

Runs the same parameter sweep + per-session/aggregate evaluation for two session sets
(CLAHE vs non-CLAHE) and prints a side-by-side comparison table.

Usage:
    python Pipeline/fer/evaluate_clahe_comparison.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fer.compare_ground_truth_v2 import (
    SCOREABLE_LABELS,
    time_to_sec,
    _ALL_FORMULA_COLS,
    _FORMULA_LABELS,
    compute_metrics,
    print_parameter_sweep,
    print_best_configs_summary,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Session sets: (gt_csv, model_csv)
# ---------------------------------------------------------------------------

CLAHE_SESSIONS = [
    (
        os.path.join(_REPO_ROOT, "Annotations/S02_Vid04.csv"),
        os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S02_Vid04_dim/20260503_153301_mp_hs_S02_Vid04_dim.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations/S08_Vid18.csv"),
        os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S08_Vid18_mixed/20260503_155009_mp_hs_S08_Vid18_mixed.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations/S06_Vid16.csv"),
        os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S06_Vid16_bright/20260503_155724_mp_hs_S06_Vid16_bright.csv"),
    ),
]

NOCLAHE_SESSIONS = [
    (
        os.path.join(_REPO_ROOT, "Annotations/S02_Vid04.csv"),
        os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S02_Vid04_dim_noclahe/20260507_151655_mp_hs_S02_Vid04_dim_noclahe.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations/S08_Vid18.csv"),
        os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S08_Vid18_mixed_noclahe/20260507_152746_mp_hs_S08_Vid18_mixed_noclahe.csv"),
    ),
    (
        os.path.join(_REPO_ROOT, "Annotations/S06_Vid16.csv"),
        os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S06_Vid16_bright_noclahe/20260507_152753_mp_hs_S06_Vid16_bright_noclahe.csv"),
    ),
]

# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

PAD_START  = 0.5
PAD_END    = 1.0
FILL_RATIO = 0.65

SWEEP_THRESHOLDS   = [0.4, 0.5, 0.6, 0.70, 0.80]
SWEEP_MIN_FRAMES   = [5, 8, 10, 12, 15]
SWEEP_WINDOW_SIZES = [None, 10, 15, 20]


def _enrich_formulas(df, anger_coeff=0.6):
    hs_fear  = pd.to_numeric(df.get('hs_fear',  0), errors='coerce').fillna(0.0)
    mp_t     = pd.to_numeric(df.get('mp_tension', 0), errors='coerce').fillna(0.0)
    hs_anger = pd.to_numeric(df.get('hs_anger',  0), errors='coerce').fillna(0.0)
    if 'f7'  not in df.columns: df['f7']  = hs_fear.clip(0, 1)
    if 'f8'  not in df.columns: df['f8']  = mp_t.clip(0, 1)
    if 'f9'  not in df.columns: df['f9']  = np.maximum(hs_fear, mp_t).clip(0, 1)
    if 'f10' not in df.columns: df['f10'] = np.sqrt((hs_fear * mp_t).clip(0, 1)).clip(0, 1)
    if 'f11' not in df.columns: df['f11'] = (hs_fear - anger_coeff * hs_anger).clip(0, 1)


def _load_gt(gt_file):
    raw_gt = pd.read_csv(gt_file, low_memory=False)
    raw_gt['start_val']  = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val']    = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    return raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)


def _load_sessions(session_list, label):
    loaded = []
    for gt_file, model_file in session_list:
        if model_file is None:
            continue
        try:
            model_df = pd.read_csv(model_file, low_memory=False)
            _enrich_formulas(model_df)
            gt_df = _load_gt(gt_file)
            loaded.append({
                'gt_file': gt_file,
                'model_file': model_file,
                'gt_df': gt_df,
                'model_df': model_df,
            })
        except FileNotFoundError as e:
            print(f"  WARNING [{label}]: file not found: {e}")
    return loaded


def _run_sweep(loaded_sessions, present_cols, label):
    """Run parameter sweep and return best_per_formula list and best overall params."""
    best_per_formula = []

    for col in present_cols:
        formula_label = _FORMULA_LABELS.get(col, col)

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
            formula_col=col, label=formula_label, eval_func=eval_combined,
            thresholds=SWEEP_THRESHOLDS, min_frames=SWEEP_MIN_FRAMES,
            window_sizes=SWEEP_WINDOW_SIZES,
            pad_start=PAD_START, pad_end=PAD_END, fill_ratio=FILL_RATIO,
            print_table=False,
        )
        if best:
            best_per_formula.append(best)

    if best_per_formula:
        best_overall = max(best_per_formula, key=lambda r: r['f1'])
        best_thresh = best_overall['thresh']
        best_min_fr = best_overall['min_fr']
        best_win = None if best_overall['win'] == 'strict' else int(best_overall['win'])
    else:
        best_thresh, best_min_fr, best_win = 0.70, 10, 15

    return best_per_formula, best_thresh, best_min_fr, best_win


def _evaluate_set(loaded_sessions, present_cols, best_thresh, best_min_fr, best_win, label):
    """Evaluate one session set and return per-session + aggregate results."""
    aggregate = {col: {'tp': 0, 'fp': 0, 'gt_caught': 0, 'n_gt': 0} for col in present_cols}
    per_session_results = []

    for sd in loaded_sessions:
        session_name = os.path.basename(sd['model_file'])
        session_results = {'name': session_name}

        for col in present_cols:
            if col not in sd['model_df'].columns:
                continue
            tp, fp, gt_caught, n_gt = compute_metrics(
                sd['model_df'], sd['gt_df'],
                best_thresh, best_min_fr, PAD_START, PAD_END,
                best_win, FILL_RATIO, col,
            )
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = gt_caught / n_gt if n_gt > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            session_results[col] = {'tp': tp, 'fp': fp, 'fn': n_gt - gt_caught,
                                    'prec': prec, 'rec': rec, 'f1': f1}
            aggregate[col]['tp'] += tp
            aggregate[col]['fp'] += fp
            aggregate[col]['gt_caught'] += gt_caught
            aggregate[col]['n_gt'] += n_gt

        per_session_results.append(session_results)

    agg_results = {}
    for col in present_cols:
        a = aggregate[col]
        tp, fp, gt_caught, n_gt = a['tp'], a['fp'], a['gt_caught'], a['n_gt']
        fn = n_gt - gt_caught
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = gt_caught / n_gt if n_gt > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        agg_results[col] = {'tp': tp, 'fp': fp, 'fn': fn, 'prec': prec, 'rec': rec, 'f1': f1}

    return per_session_results, agg_results


def main():
    # Auto-discover noclahe session files if placeholders are None
    noclahe_dir = os.path.join(_REPO_ROOT, "Pipeline/logs/sessions")
    for i, (gt_file, model_file) in enumerate(NOCLAHE_SESSIONS):
        if model_file is None:
            gt_base = os.path.splitext(os.path.basename(gt_file))[0]
            for folder in os.listdir(noclahe_dir):
                if 'noclahe' in folder and gt_base.split('_')[0] in folder:
                    folder_path = os.path.join(noclahe_dir, folder)
                    csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv') and '_compact' not in f and '_events' not in f]
                    if csvs:
                        NOCLAHE_SESSIONS[i] = (gt_file, os.path.join(folder_path, sorted(csvs)[0]))

    print("=" * 70)
    print("  CLAHE A/B COMPARISON — Group D")
    print("=" * 70)

    clahe_loaded   = _load_sessions(CLAHE_SESSIONS, "CLAHE")
    noclahe_loaded = _load_sessions(NOCLAHE_SESSIONS, "NO-CLAHE")

    print(f"\n  CLAHE sessions loaded:    {len(clahe_loaded)}")
    print(f"  NO-CLAHE sessions loaded: {len(noclahe_loaded)}")

    if not clahe_loaded or not noclahe_loaded:
        print("\n  ERROR: Cannot compare — one or both session sets empty.")
        return

    all_cols = set()
    for sd in clahe_loaded + noclahe_loaded:
        all_cols.update(c for c in _ALL_FORMULA_COLS if c in sd['model_df'].columns)
    present_cols = [c for c in _ALL_FORMULA_COLS if c in all_cols]

    # Run sweep on CLAHE set (the established baseline)
    print(f"\n\n{'='*70}")
    print("  Parameter Sweep — CLAHE sessions")
    print(f"{'='*70}")
    clahe_bpf, clahe_thresh, clahe_mf, clahe_win = _run_sweep(clahe_loaded, present_cols, "CLAHE")
    print_best_configs_summary(clahe_bpf, "Best Config (CLAHE)")
    print(f"\n  → CLAHE best: thresh={clahe_thresh} mf={clahe_mf} win={clahe_win}")

    # Run sweep on non-CLAHE set
    print(f"\n\n{'='*70}")
    print("  Parameter Sweep — NO-CLAHE sessions")
    print(f"{'='*70}")
    noclahe_bpf, noclahe_thresh, noclahe_mf, noclahe_win = _run_sweep(noclahe_loaded, present_cols, "NO-CLAHE")
    print_best_configs_summary(noclahe_bpf, "Best Config (NO-CLAHE)")
    print(f"\n  → NO-CLAHE best: thresh={noclahe_thresh} mf={noclahe_mf} win={noclahe_win}")

    # Evaluate both sets with THEIR OWN best params (fair comparison)
    print(f"\n\n{'='*70}")
    print("  Per-Session Comparison (each set uses its own best params)")
    print(f"{'='*70}")

    _, clahe_agg = _evaluate_set(clahe_loaded, present_cols, clahe_thresh, clahe_mf, clahe_win, "CLAHE")
    clahe_per_session, _ = _evaluate_set(clahe_loaded, present_cols, clahe_thresh, clahe_mf, clahe_win, "CLAHE")
    noclahe_per_session, noclahe_agg = _evaluate_set(noclahe_loaded, present_cols, noclahe_thresh, noclahe_mf, noclahe_win, "NO-CLAHE")

    # Also evaluate both with SAME params (controlled comparison)
    # Use CLAHE params as the control
    _, clahe_agg_ctrl = _evaluate_set(clahe_loaded, present_cols, clahe_thresh, clahe_mf, clahe_win, "CLAHE-ctrl")
    _, noclahe_agg_ctrl = _evaluate_set(noclahe_loaded, present_cols, clahe_thresh, clahe_mf, clahe_win, "NOCLAHE-ctrl")

    # Print comparison table — aggregate with own best params
    key_formulas = ['f1', 'f7', 'f11', 'f0', 'f2']
    key_formulas = [f for f in key_formulas if f in present_cols]

    print(f"\n\n{'='*70}")
    print("  AGGREGATE COMPARISON — Own Best Params")
    print(f"  CLAHE:    thresh={clahe_thresh} mf={clahe_mf} win={clahe_win}")
    print(f"  NO-CLAHE: thresh={noclahe_thresh} mf={noclahe_mf} win={noclahe_win}")
    print(f"{'='*70}")
    print(f"  {'Formula':<10}  {'Label':<16}  {'CLAHE F1':>10}  {'noCLAHE F1':>10}  {'Delta':>8}")
    print(f"  {'-'*10}  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*8}")

    for col in present_cols:
        label = _FORMULA_LABELS.get(col, col)
        c_f1 = clahe_agg.get(col, {}).get('f1', 0.0)
        n_f1 = noclahe_agg.get(col, {}).get('f1', 0.0)
        delta = n_f1 - c_f1
        sign = "+" if delta >= 0 else ""
        print(f"  {col:<10}  {label:<16}  {c_f1:>10.4f}  {n_f1:>10.4f}  {sign}{delta:>7.4f}")

    # Print controlled comparison (same params)
    print(f"\n\n{'='*70}")
    print(f"  CONTROLLED COMPARISON — Same Params (CLAHE's best: thresh={clahe_thresh} mf={clahe_mf} win={clahe_win})")
    print(f"{'='*70}")
    print(f"  {'Formula':<10}  {'Label':<16}  {'CLAHE F1':>10}  {'noCLAHE F1':>10}  {'Delta':>8}")
    print(f"  {'-'*10}  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*8}")

    for col in present_cols:
        label = _FORMULA_LABELS.get(col, col)
        c_f1 = clahe_agg_ctrl.get(col, {}).get('f1', 0.0)
        n_f1 = noclahe_agg_ctrl.get(col, {}).get('f1', 0.0)
        delta = n_f1 - c_f1
        sign = "+" if delta >= 0 else ""
        print(f"  {col:<10}  {label:<16}  {c_f1:>10.4f}  {n_f1:>10.4f}  {sign}{delta:>7.4f}")

    # Per-session breakdown for key formulas
    print(f"\n\n{'='*70}")
    print("  PER-SESSION BREAKDOWN (key formulas, own best params)")
    print(f"{'='*70}")

    session_labels = ["S02 (dim)", "S08 (mixed)", "S06 (bright)"]
    for formula in key_formulas:
        label = _FORMULA_LABELS.get(formula, formula)
        print(f"\n  {formula} ({label}):")
        print(f"  {'Session':<16}  {'CLAHE F1':>10}  {'noCLAHE F1':>10}  {'Delta':>8}")
        for i in range(min(len(clahe_per_session), len(noclahe_per_session))):
            slabel = session_labels[i] if i < len(session_labels) else f"Session {i}"
            c_f1 = clahe_per_session[i].get(formula, {}).get('f1', 0.0)
            n_f1 = noclahe_per_session[i].get(formula, {}).get('f1', 0.0)
            delta = n_f1 - c_f1
            sign = "+" if delta >= 0 else ""
            print(f"  {slabel:<16}  {c_f1:>10.4f}  {n_f1:>10.4f}  {sign}{delta:>7.4f}")

    # Detailed per-session P/R/F1 for best formula (f1)
    print(f"\n\n{'='*70}")
    print("  DETAILED PER-SESSION — f1 (hs_fear × (1+mp_tension))")
    print(f"{'='*70}")
    print(f"  {'Session':<16}  {'Set':<10}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
    for i in range(min(len(clahe_per_session), len(noclahe_per_session))):
        slabel = session_labels[i] if i < len(session_labels) else f"Session {i}"
        for tag, results in [("CLAHE", clahe_per_session[i]), ("noCLAHE", noclahe_per_session[i])]:
            m = results.get('f1', {})
            if m:
                print(f"  {slabel:<16}  {tag:<10}  {m['tp']:>4}  {m['fp']:>4}  {m['fn']:>4}  "
                      f"{m['prec']:>8.2%}  {m['rec']:>8.2%}  {m['f1']:>8.4f}")

    print(f"\n{'='*70}")
    print("  CLAHE A/B comparison complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    _LOG_DIR  = os.path.join(_REPO_ROOT, "Pipeline", "logs", "comparisons")
    os.makedirs(_LOG_DIR, exist_ok=True)
    _LOG_PATH = os.path.join(_LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_clahe_comparison.txt")

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
        def flush(self):
            for f in self.files:
                f.flush()

    with open(_LOG_PATH, 'w') as _fh:
        _orig, sys.stdout = sys.stdout, _Tee(sys.stdout, _fh)
        try:
            main()
        finally:
            sys.stdout = _orig
    print(f"\nLog saved → {_LOG_PATH}")
