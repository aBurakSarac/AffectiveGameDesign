"""Group E: FER Detection Parameter Optimization (Solutions 8, 9, 12, 14).

Sweeps four dimensions beyond the existing threshold/min_frames/window grid:
  - Fill ratio: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
  - Anger coefficient for f11: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  - Temporal smoothing window: [None, 30, 40, 50, 60, 70] frames rolling avg
  - Per-lighting breakdown of best config

Uses leave-one-subject-out cross-validation to avoid overfitting with only 3 subjects.

Usage:
    python Pipeline/fer/evaluate_group_e.py
"""

import os
import sys
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fer.compare_ground_truth_v2 import (
    SCOREABLE_LABELS,
    time_to_sec,
    _ALL_FORMULA_COLS,
    _FORMULA_LABELS,
    compute_metrics,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESSIONS = [
    {
        'label': 'S02 (dim)',
        'lighting': 'dim',
        'gt': os.path.join(_REPO_ROOT, "Annotations/S02_Vid04.csv"),
        'model': os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S02_Vid04_dim/20260503_153301_mp_hs_S02_Vid04_dim.csv"),
    },
    {
        'label': 'S08 (mixed)',
        'lighting': 'mixed',
        'gt': os.path.join(_REPO_ROOT, "Annotations/S08_Vid18.csv"),
        'model': os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S08_Vid18_mixed/20260503_155009_mp_hs_S08_Vid18_mixed.csv"),
    },
    {
        'label': 'S06 (bright)',
        'lighting': 'bright',
        'gt': os.path.join(_REPO_ROOT, "Annotations/S06_Vid16.csv"),
        'model': os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S06_Vid16_bright/20260503_155724_mp_hs_S06_Vid16_bright.csv"),
    },
]

PAD_START = 0.5
PAD_END = 1.0


def _load_gt(gt_file):
    raw_gt = pd.read_csv(gt_file, low_memory=False)
    raw_gt['start_val'] = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val'] = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    return raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)


def _enrich_formulas(df, anger_coeff=0.6):
    hs_fear = pd.to_numeric(df.get('hs_fear', 0), errors='coerce').fillna(0.0)
    mp_t = pd.to_numeric(df.get('mp_tension', 0), errors='coerce').fillna(0.0)
    hs_anger = pd.to_numeric(df.get('hs_anger', 0), errors='coerce').fillna(0.0)
    if 'f7' not in df.columns: df['f7'] = hs_fear.clip(0, 1)
    if 'f8' not in df.columns: df['f8'] = mp_t.clip(0, 1)
    if 'f9' not in df.columns: df['f9'] = np.maximum(hs_fear, mp_t).clip(0, 1)
    if 'f10' not in df.columns: df['f10'] = np.sqrt((hs_fear * mp_t).clip(0, 1)).clip(0, 1)
    df['f11'] = (hs_fear - anger_coeff * hs_anger).clip(0, 1)


def _apply_smoothing(df, cols, smooth_window):
    """Apply rolling mean smoothing to formula columns in-place."""
    if smooth_window is None:
        return df
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).rolling(
                window=smooth_window, min_periods=1, center=False
            ).mean()
    return df


def _eval_combined(loaded, col, thresh, mf, ws, fill_ratio):
    tot_tp = tot_fp = tot_gt_caught = tot_n_gt = 0
    for sd in loaded:
        if col in sd['model_df'].columns:
            tp, fp, gt_caught, n_gt = compute_metrics(
                sd['model_df'], sd['gt_df'], thresh, mf, PAD_START, PAD_END,
                ws, fill_ratio, col,
            )
            tot_tp += tp; tot_fp += fp
            tot_gt_caught += gt_caught; tot_n_gt += n_gt
    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    rec = tot_gt_caught / tot_n_gt if tot_n_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'tp': tot_tp, 'fp': tot_fp, 'fn': tot_n_gt - tot_gt_caught,
            'prec': prec, 'rec': rec, 'f1': f1}


def _load_sessions(anger_coeff=0.6, smooth_window=None):
    loaded = []
    for s in SESSIONS:
        model_df = pd.read_csv(s['model'], low_memory=False)
        _enrich_formulas(model_df, anger_coeff=anger_coeff)
        if smooth_window is not None:
            formula_cols = [c for c in _ALL_FORMULA_COLS if c in model_df.columns]
            model_df = _apply_smoothing(model_df, formula_cols, smooth_window)
        gt_df = _load_gt(s['gt'])
        loaded.append({
            'label': s['label'],
            'lighting': s['lighting'],
            'gt_df': gt_df,
            'model_df': model_df,
        })
    return loaded


def main():
    # Use key formulas only to keep sweep tractable
    key_formulas = ['f1', 'f7', 'f11', 'f3', 'f5', 'f6']

    # Detection grid (same as evaluate_all.py baseline)
    thresholds = [0.4, 0.5, 0.6, 0.70, 0.80]
    min_frames_list = [5, 8, 10, 12, 15]
    window_sizes = [None, 10, 15, 20]

    # ===================================================================
    # SWEEP 1: Fill Ratio
    # ===================================================================
    print("=" * 70)
    print("  SWEEP 1: Fill Ratio")
    print("  Fixed: anger_coeff=0.6, smoothing=None")
    print("=" * 70)

    fill_ratios = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    loaded = _load_sessions(anger_coeff=0.6, smooth_window=None)

    fill_results = {}
    for fr in fill_ratios:
        best_f1 = 0.0
        best_cfg = None
        for col in key_formulas:
            for thresh in thresholds:
                for mf in min_frames_list:
                    for ws in window_sizes:
                        r = _eval_combined(loaded, col, thresh, mf, ws, fr)
                        if r['f1'] > best_f1:
                            best_f1 = r['f1']
                            best_cfg = {'formula': col, 'thresh': thresh, 'mf': mf,
                                        'ws': ws, 'fill': fr, **r}
        fill_results[fr] = best_cfg
        label = _FORMULA_LABELS.get(best_cfg['formula'], best_cfg['formula']) if best_cfg else '—'
        print(f"  fill={fr:.2f}  best F1={best_f1:.4f}  "
              f"formula={best_cfg['formula'] if best_cfg else '—'} ({label})  "
              f"thresh={best_cfg['thresh'] if best_cfg else '—'}  "
              f"mf={best_cfg['mf'] if best_cfg else '—'}  "
              f"ws={best_cfg['ws'] if best_cfg else '—'}  "
              f"P={best_cfg['prec']:.2%}  R={best_cfg['rec']:.2%}" if best_cfg else "")

    print(f"\n  Current default: fill=0.65")
    best_fill = max(fill_results.items(), key=lambda x: x[1]['f1'] if x[1] else 0)
    print(f"  Best fill ratio: {best_fill[0]:.2f} → F1={best_fill[1]['f1']:.4f}")

    # ===================================================================
    # SWEEP 2: Anger Coefficient (f11 only)
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("  SWEEP 2: Anger Coefficient for f11 = hs_fear - k*hs_anger")
    print("  Fixed: fill=0.65, smoothing=None")
    print("=" * 70)

    anger_coeffs = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    anger_results = {}
    for ac in anger_coeffs:
        loaded_ac = _load_sessions(anger_coeff=ac, smooth_window=None)
        best_f1 = 0.0
        best_cfg = None
        for thresh in thresholds:
            for mf in min_frames_list:
                for ws in window_sizes:
                    r = _eval_combined(loaded_ac, 'f11', thresh, mf, ws, 0.65)
                    if r['f1'] > best_f1:
                        best_f1 = r['f1']
                        best_cfg = {'anger_coeff': ac, 'thresh': thresh, 'mf': mf,
                                    'ws': ws, **r}
        anger_results[ac] = best_cfg
        print(f"  anger_coeff={ac:.1f}  best F1={best_f1:.4f}  "
              f"thresh={best_cfg['thresh']}  mf={best_cfg['mf']}  "
              f"ws={best_cfg['ws']}  "
              f"P={best_cfg['prec']:.2%}  R={best_cfg['rec']:.2%}" if best_cfg else "")

    print(f"\n  Current default: anger_coeff=0.6")
    best_anger = max(anger_results.items(), key=lambda x: x[1]['f1'] if x[1] else 0)
    print(f"  Best anger coeff: {best_anger[0]:.1f} → F1={best_anger[1]['f1']:.4f}")

    # Also compare against f7 (hs_fear only, equivalent to anger_coeff=0)
    loaded_base = _load_sessions(anger_coeff=0.6, smooth_window=None)
    best_f7 = 0.0
    for thresh in thresholds:
        for mf in min_frames_list:
            for ws in window_sizes:
                r = _eval_combined(loaded_base, 'f7', thresh, mf, ws, 0.65)
                if r['f1'] > best_f7:
                    best_f7 = r['f1']
    print(f"  f7 (hs_fear only, no anger suppression) best F1={best_f7:.4f}")

    # ===================================================================
    # SWEEP 3: Temporal Smoothing
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("  SWEEP 3: Temporal Smoothing (rolling mean window)")
    print("  Fixed: fill=0.65, anger_coeff=0.6")
    print("=" * 70)

    smooth_windows = [None, 20, 30, 40, 50, 60, 70]

    smooth_results = {}
    for sw in smooth_windows:
        loaded_sw = _load_sessions(anger_coeff=0.6, smooth_window=sw)
        best_f1 = 0.0
        best_cfg = None
        for col in key_formulas:
            for thresh in thresholds:
                for mf in min_frames_list:
                    for ws in window_sizes:
                        r = _eval_combined(loaded_sw, col, thresh, mf, ws, 0.65)
                        if r['f1'] > best_f1:
                            best_f1 = r['f1']
                            best_cfg = {'smooth': sw, 'formula': col, 'thresh': thresh,
                                        'mf': mf, 'ws': ws, **r}
        smooth_results[sw] = best_cfg
        sw_str = f"{sw}fr" if sw else "none"
        label = _FORMULA_LABELS.get(best_cfg['formula'], best_cfg['formula']) if best_cfg else '—'
        print(f"  smooth={sw_str:<6s}  best F1={best_f1:.4f}  "
              f"formula={best_cfg['formula'] if best_cfg else '—'} ({label})  "
              f"thresh={best_cfg['thresh'] if best_cfg else '—'}  "
              f"mf={best_cfg['mf'] if best_cfg else '—'}  "
              f"ws={best_cfg['ws'] if best_cfg else '—'}  "
              f"P={best_cfg['prec']:.2%}  R={best_cfg['rec']:.2%}" if best_cfg else "")

    # ===================================================================
    # SWEEP 4: Per-Lighting Threshold Analysis
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("  SWEEP 4: Per-Lighting Optimal Threshold (f1)")
    print("  Fixed: fill=0.65, anger_coeff=0.6, smoothing=None")
    print("=" * 70)

    loaded_base = _load_sessions(anger_coeff=0.6, smooth_window=None)

    for sd in loaded_base:
        best_f1 = 0.0
        best_cfg = None
        n_gt = len(sd['gt_df'])
        for thresh in thresholds:
            for mf in min_frames_list:
                for ws in window_sizes:
                    tp, fp, gt_caught, n = compute_metrics(
                        sd['model_df'], sd['gt_df'], thresh, mf,
                        PAD_START, PAD_END, ws, 0.65, 'f1',
                    )
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = gt_caught / n if n > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cfg = {'thresh': thresh, 'mf': mf, 'ws': ws,
                                    'prec': prec, 'rec': rec, 'f1': f1,
                                    'tp': tp, 'fp': fp, 'fn': n - gt_caught}
        print(f"\n  {sd['label']} ({sd['lighting']}) — {n_gt} GT events:")
        print(f"    Best f1 F1={best_f1:.4f}  thresh={best_cfg['thresh']}  "
              f"mf={best_cfg['mf']}  ws={best_cfg['ws']}  "
              f"P={best_cfg['prec']:.2%}  R={best_cfg['rec']:.2%}  "
              f"TP={best_cfg['tp']} FP={best_cfg['fp']} FN={best_cfg['fn']}")

    # Combined best (current approach) for reference
    best_combined_f1 = 0.0
    best_combined_cfg = None
    for thresh in thresholds:
        for mf in min_frames_list:
            for ws in window_sizes:
                r = _eval_combined(loaded_base, 'f1', thresh, mf, ws, 0.65)
                if r['f1'] > best_combined_f1:
                    best_combined_f1 = r['f1']
                    best_combined_cfg = {'thresh': thresh, 'mf': mf, 'ws': ws, **r}
    print(f"\n  Combined best (single threshold for all): "
          f"F1={best_combined_f1:.4f}  thresh={best_combined_cfg['thresh']}  "
          f"mf={best_combined_cfg['mf']}  ws={best_combined_cfg['ws']}")

    # ===================================================================
    # CROSS-VALIDATION: Leave-One-Subject-Out
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("  CROSS-VALIDATION: Leave-One-Subject-Out (f1, fill=0.65)")
    print("  Train on 2 subjects, test on 1, rotate")
    print("=" * 70)

    loaded_cv = _load_sessions(anger_coeff=0.6, smooth_window=None)
    cv_f1_scores = []

    for hold_idx in range(len(loaded_cv)):
        train = [loaded_cv[i] for i in range(len(loaded_cv)) if i != hold_idx]
        test = [loaded_cv[hold_idx]]

        # Find best params on train set
        best_train_f1 = 0.0
        best_params = None
        for thresh in thresholds:
            for mf in min_frames_list:
                for ws in window_sizes:
                    r = _eval_combined(train, 'f1', thresh, mf, ws, 0.65)
                    if r['f1'] > best_train_f1:
                        best_train_f1 = r['f1']
                        best_params = (thresh, mf, ws)

        # Evaluate on held-out subject
        t, m, w = best_params
        test_r = _eval_combined(test, 'f1', t, m, w, 0.65)

        print(f"\n  Hold out: {loaded_cv[hold_idx]['label']}")
        print(f"    Train best: thresh={t} mf={m} ws={w} → train F1={best_train_f1:.4f}")
        print(f"    Test F1={test_r['f1']:.4f}  P={test_r['prec']:.2%}  R={test_r['rec']:.2%}  "
              f"TP={test_r['tp']} FP={test_r['fp']} FN={test_r['fn']}")
        cv_f1_scores.append(test_r['f1'])

    mean_cv = np.mean(cv_f1_scores)
    std_cv = np.std(cv_f1_scores)
    print(f"\n  LOO-CV mean F1: {mean_cv:.4f} ± {std_cv:.4f}")
    print(f"  Per-fold: {', '.join(f'{s:.4f}' for s in cv_f1_scores)}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("  GROUP E SUMMARY")
    print("=" * 70)
    print(f"  Current baseline (f1, fill=0.65, anger=0.6, smooth=none): F1={best_combined_f1:.4f}")
    print(f"  Best fill ratio:     {best_fill[0]:.2f} → F1={best_fill[1]['f1']:.4f}  "
          f"(delta {best_fill[1]['f1'] - best_combined_f1:+.4f})")
    best_anger_entry = max(anger_results.items(), key=lambda x: x[1]['f1'] if x[1] else 0)
    print(f"  Best anger coeff:    {best_anger_entry[0]:.1f} → f11 F1={best_anger_entry[1]['f1']:.4f}")
    best_smooth_entry = max(smooth_results.items(), key=lambda x: x[1]['f1'] if x[1] else 0)
    sw_str = f"{best_smooth_entry[0]}fr" if best_smooth_entry[0] else "none"
    print(f"  Best smoothing:      {sw_str} → F1={best_smooth_entry[1]['f1']:.4f}  "
          f"(delta {best_smooth_entry[1]['f1'] - best_combined_f1:+.4f})")
    print(f"  LOO-CV mean F1:      {mean_cv:.4f} ± {std_cv:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    _LOG_DIR = os.path.join(_REPO_ROOT, "Pipeline", "logs", "comparisons")
    os.makedirs(_LOG_DIR, exist_ok=True)
    _LOG_PATH = os.path.join(_LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_group_e.txt")

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
