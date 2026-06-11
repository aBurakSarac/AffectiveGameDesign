"""Group F: FER Formula Innovation (Solutions 10, 11, 13, 15).

Tests novel formula variants computed post-hoc from existing CSV columns:
  - Sol 10: hs_fear × (1 + k × hs_surprise) — pure HS multiplicative
  - Sol 13: hs_fear × hs_arousal — arousal-gated fear
  - Sol 11: Ensemble voting — weighted combinations of top formulas
  - Sol 15: Two-Gate Detector offline evaluation (event_status column)

All new formulas are computed post-hoc from raw columns already in the CSV,
so this is a controlled experiment: same data, same detection pipeline,
only the signal column changes.

Usage:
    python Pipeline/fer/evaluate_group_f.py
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
    build_trigger_blocks,
    match_triggers,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SESSIONS = [
    {
        'label': 'S02 (dim)',
        'gt': os.path.join(_REPO_ROOT, "Annotations/S02_Vid04.csv"),
        'model': os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S02_Vid04_dim/20260503_153301_mp_hs_S02_Vid04_dim.csv"),
    },
    {
        'label': 'S08 (mixed)',
        'gt': os.path.join(_REPO_ROOT, "Annotations/S08_Vid18.csv"),
        'model': os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S08_Vid18_mixed/20260503_155009_mp_hs_S08_Vid18_mixed.csv"),
    },
    {
        'label': 'S06 (bright)',
        'gt': os.path.join(_REPO_ROOT, "Annotations/S06_Vid16.csv"),
        'model': os.path.join(_REPO_ROOT, "Pipeline/logs/sessions/S06_Vid16_bright/20260503_155724_mp_hs_S06_Vid16_bright.csv"),
    },
]

PAD_START = 0.5
PAD_END = 1.0
FILL_RATIO = 0.65

THRESHOLDS = [0.4, 0.5, 0.6, 0.70, 0.80]
MIN_FRAMES = [5, 8, 10, 12, 15]
WINDOW_SIZES = [None, 10, 15, 20]


def _load_gt(gt_file):
    raw_gt = pd.read_csv(gt_file, low_memory=False)
    raw_gt['start_val'] = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val'] = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    return raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)


def _add_novel_formulas(df):
    """Compute new formula columns post-hoc from raw CSV columns."""
    hs_fear = pd.to_numeric(df.get('hs_fear', 0), errors='coerce').fillna(0.0)
    hs_surprise = pd.to_numeric(df.get('hs_surprise', 0), errors='coerce').fillna(0.0)
    hs_arousal = pd.to_numeric(df.get('hs_arousal', 0), errors='coerce').fillna(0.0)
    hs_anger = pd.to_numeric(df.get('hs_anger', 0), errors='coerce').fillna(0.0)
    mp_tension = pd.to_numeric(df.get('mp_tension', 0), errors='coerce').fillna(0.0)

    # Existing formulas for reference (ensure present)
    if 'f1' not in df.columns:
        df['f1'] = (hs_fear * (1 + mp_tension)).clip(0, 1)
    if 'f7' not in df.columns:
        df['f7'] = hs_fear.clip(0, 1)
    if 'f11' not in df.columns:
        df['f11'] = (hs_fear - 0.6 * hs_anger).clip(0, 1)

    # --- Solution 10: HS multiplicative with surprise ---
    for k in [0.5, 1.0, 1.5, 2.0]:
        col = f'f_surp_{k:.1f}'
        df[col] = (hs_fear * (1 + k * hs_surprise)).clip(0, 1)

    # --- Solution 13: Arousal × Fear interaction ---
    df['f_arousal_mult'] = (hs_fear * hs_arousal).clip(0, 1)
    df['f_arousal_gate'] = (hs_fear * (1 + hs_arousal)).clip(0, 1)
    df['f_arousal_sqrt'] = np.sqrt((hs_fear * hs_arousal).clip(0, 1))

    # --- Solution 11: Ensemble combinations ---
    f1_vals = pd.to_numeric(df.get('f1', 0), errors='coerce').fillna(0.0)
    f7_vals = pd.to_numeric(df.get('f7', 0), errors='coerce').fillna(0.0)
    f11_vals = pd.to_numeric(df.get('f11', 0), errors='coerce').fillna(0.0)
    f3_vals = pd.to_numeric(df.get('f3', 0), errors='coerce').fillna(0.0)

    df['f_ens_avg'] = ((f1_vals + f7_vals + f11_vals) / 3.0).clip(0, 1)
    df['f_ens_max'] = np.maximum(np.maximum(f1_vals, f7_vals), f11_vals).clip(0, 1)
    df['f_ens_w'] = (0.5 * f1_vals + 0.3 * f7_vals + 0.2 * f11_vals).clip(0, 1)

    # Surprise-boosted f1 variant: hs_fear × (1 + mp_tension + hs_surprise)
    df['f_surp_t'] = (hs_fear * (1 + mp_tension + hs_surprise)).clip(0, 1)

    # Fear × (1 + arousal) — analogous to f1 structure but HS-only
    df['f_fear_arousal_boost'] = (hs_fear * (1 + 0.5 * hs_arousal)).clip(0, 1)

    return df


NOVEL_FORMULAS = {
    'f_surp_0.5': 'fear×(1+0.5×surp)',
    'f_surp_1.0': 'fear×(1+surp)',
    'f_surp_1.5': 'fear×(1+1.5×surp)',
    'f_surp_2.0': 'fear×(1+2×surp)',
    'f_arousal_mult': 'fear×arousal',
    'f_arousal_gate': 'fear×(1+arousal)',
    'f_arousal_sqrt': '√(fear×arousal)',
    'f_ens_avg': 'avg(f1,f7,f11)',
    'f_ens_max': 'max(f1,f7,f11)',
    'f_ens_w': '0.5f1+0.3f7+0.2f11',
    'f_surp_t': 'fear×(1+t+surp)',
    'f_fear_arousal_boost': 'fear×(1+0.5×ar)',
}

BASELINE_FORMULAS = {
    'f1': 'fear×(1+mp_t)',
    'f7': 'hs_fear only',
    'f11': 'fear-0.6×anger',
    'f3': 'composite',
}


def _eval_combined(loaded, col, thresh, mf, ws):
    tot_tp = tot_fp = tot_gt_caught = tot_n_gt = 0
    for sd in loaded:
        if col in sd['model_df'].columns:
            tp, fp, gt_caught, n_gt = compute_metrics(
                sd['model_df'], sd['gt_df'], thresh, mf, PAD_START, PAD_END,
                ws, FILL_RATIO, col,
            )
            tot_tp += tp; tot_fp += fp
            tot_gt_caught += gt_caught; tot_n_gt += n_gt
    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    rec = tot_gt_caught / tot_n_gt if tot_n_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'tp': tot_tp, 'fp': tot_fp, 'fn': tot_n_gt - tot_gt_caught,
            'prec': prec, 'rec': rec, 'f1': f1}


def _best_config(loaded, col):
    best_f1 = 0.0
    best_cfg = None
    for thresh in THRESHOLDS:
        for mf in MIN_FRAMES:
            for ws in WINDOW_SIZES:
                r = _eval_combined(loaded, col, thresh, mf, ws)
                if r['f1'] > best_f1:
                    best_f1 = r['f1']
                    best_cfg = {'thresh': thresh, 'mf': mf, 'ws': ws, **r}
    return best_cfg


def _eval_two_gate(loaded):
    """Solution 15: Evaluate Two-Gate Detector events from event_status column."""
    print(f"\n{'='*70}")
    print("  Solution 15: Two-Gate Detector Offline Evaluation")
    print("=" * 70)

    tot_tp = tot_fp = tot_gt_caught = tot_n_gt = 0

    for sd in loaded:
        model_df = sd['model_df']
        gt_df = sd['gt_df']

        if 'event_status' not in model_df.columns:
            print(f"  {sd['label']}: event_status column missing — skipping")
            continue

        if 'timestamp' not in model_df.columns:
            print(f"  {sd['label']}: timestamp column missing — skipping")
            continue

        # Extract event blocks from event_status column
        # Values: IDLE, ONSET, SUSTAIN, EVENT_END, COOLDOWN
        in_event = model_df['event_status'].isin(['ONSET', 'SUSTAIN', 'EVENT_END'])
        model_df = model_df.copy()
        model_df['_tg_event'] = in_event.astype(int)
        model_df['_tg_block'] = (model_df['_tg_event'] != model_df['_tg_event'].shift()).cumsum()

        blocks = []
        for _, group in model_df[model_df['_tg_event'] == 1].groupby('_tg_block'):
            if len(group) >= 1:
                ts = pd.to_numeric(group['timestamp'], errors='coerce')
                blocks.append({
                    'start_s': float(ts.iloc[0]),
                    'end_s': float(ts.iloc[-1]),
                    'duration_frames': len(group),
                    'hit': False,
                })

        n_gt = len(gt_df)
        gt_caught = 0
        for _, gt_row in gt_df.iterrows():
            gs = gt_row['start_val'] - PAD_START
            ge = gt_row['end_val'] + PAD_END
            for b in blocks:
                if b['start_s'] <= ge and b['end_s'] >= gs:
                    gt_caught += 1
                    b['hit'] = True
                    break

        fps = sum(1 for b in blocks if not b['hit'])
        tp = gt_caught

        prec = tp / (tp + fps) if (tp + fps) > 0 else 0.0
        rec = gt_caught / n_gt if n_gt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"\n  {sd['label']}: {len(blocks)} detector events, {n_gt} GT events")
        print(f"    TP={tp}  FP={fps}  FN={n_gt - gt_caught}  "
              f"P={prec:.2%}  R={rec:.2%}  F1={f1:.4f}")

        tot_tp += tp; tot_fp += fps
        tot_gt_caught += gt_caught; tot_n_gt += n_gt

    if tot_n_gt > 0:
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        rec = tot_gt_caught / tot_n_gt if tot_n_gt > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"\n  Combined: TP={tot_tp}  FP={tot_fp}  FN={tot_n_gt - tot_gt_caught}  "
              f"P={prec:.2%}  R={rec:.2%}  F1={f1:.4f}")
        return {'tp': tot_tp, 'fp': tot_fp, 'fn': tot_n_gt - tot_gt_caught,
                'prec': prec, 'rec': rec, 'f1': f1}
    return None


def main():
    # Load sessions and add novel formulas
    loaded = []
    for s in SESSIONS:
        model_df = pd.read_csv(s['model'], low_memory=False)
        model_df = _add_novel_formulas(model_df)
        gt_df = _load_gt(s['gt'])
        loaded.append({
            'label': s['label'],
            'gt_df': gt_df,
            'model_df': model_df,
        })

    # ===================================================================
    # Baseline formulas (for reference)
    # ===================================================================
    print("=" * 70)
    print("  GROUP F: Formula Innovation — Baseline Reference")
    print("=" * 70)
    print(f"\n  {'Formula':<22s}  {'Label':<22s}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'Thresh':>6}  {'mf':>3}  {'ws':>6}")
    print(f"  {'-'*22}  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*3}  {'-'*6}")

    baseline_results = {}
    for col, label in BASELINE_FORMULAS.items():
        cfg = _best_config(loaded, col)
        if cfg:
            baseline_results[col] = cfg
            ws_str = str(cfg['ws']) if cfg['ws'] else 'strict'
            print(f"  {col:<22s}  {label:<22s}  {cfg['f1']:>7.4f}  {cfg['prec']:>7.2%}  "
                  f"{cfg['rec']:>7.2%}  {cfg['tp']:>4}  {cfg['fp']:>4}  {cfg['fn']:>4}  "
                  f"{cfg['thresh']:>6.2f}  {cfg['mf']:>3}  {ws_str:>6}")

    best_baseline_f1 = max(r['f1'] for r in baseline_results.values()) if baseline_results else 0

    # ===================================================================
    # Novel formulas
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("  Novel Formulas — Full Sweep")
    print("=" * 70)
    print(f"\n  {'Formula':<22s}  {'Label':<22s}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'Thresh':>6}  {'mf':>3}  {'ws':>6}  {'vs f1':>7}")
    print(f"  {'-'*22}  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*3}  {'-'*6}  {'-'*7}")

    novel_results = {}
    f1_baseline = baseline_results.get('f1', {}).get('f1', 0)

    for col, label in NOVEL_FORMULAS.items():
        cfg = _best_config(loaded, col)
        if cfg:
            novel_results[col] = cfg
            ws_str = str(cfg['ws']) if cfg['ws'] else 'strict'
            delta = cfg['f1'] - f1_baseline
            sign = "+" if delta >= 0 else ""
            marker = " ***" if cfg['f1'] > best_baseline_f1 else ""
            print(f"  {col:<22s}  {label:<22s}  {cfg['f1']:>7.4f}  {cfg['prec']:>7.2%}  "
                  f"{cfg['rec']:>7.2%}  {cfg['tp']:>4}  {cfg['fp']:>4}  {cfg['fn']:>4}  "
                  f"{cfg['thresh']:>6.2f}  {cfg['mf']:>3}  {ws_str:>6}  {sign}{delta:>6.4f}{marker}")

    # ===================================================================
    # Per-session breakdown for top novel formulas
    # ===================================================================
    all_results = {**baseline_results, **novel_results}
    top_novel = sorted(novel_results.items(), key=lambda x: x[1]['f1'], reverse=True)[:5]

    print(f"\n\n{'='*70}")
    print("  Per-Session Breakdown — Top 5 Novel vs f1 Baseline")
    print("=" * 70)

    compare_formulas = ['f1'] + [col for col, _ in top_novel]
    compare_labels = {'f1': BASELINE_FORMULAS['f1']}
    compare_labels.update({col: NOVEL_FORMULAS[col] for col, _ in top_novel})

    for col in compare_formulas:
        label = compare_labels.get(col, col)
        print(f"\n  {col} ({label}):")
        print(f"  {'Session':<16}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
        for sd in loaded:
            cfg = _best_config([sd], col)
            if cfg:
                print(f"  {sd['label']:<16}  {cfg['f1']:>7.4f}  {cfg['prec']:>7.2%}  "
                      f"{cfg['rec']:>7.2%}  {cfg['tp']:>4}  {cfg['fp']:>4}  {cfg['fn']:>4}")

    # ===================================================================
    # Solution 15: Two-Gate Detector
    # ===================================================================
    tg_result = _eval_two_gate(loaded)

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n\n{'='*70}")
    print("  GROUP F SUMMARY")
    print("=" * 70)

    print(f"\n  Best baseline: f1 (fear×(1+mp_t)) F1={f1_baseline:.4f}")
    if top_novel:
        best_novel = top_novel[0]
        print(f"  Best novel:    {best_novel[0]} ({NOVEL_FORMULAS[best_novel[0]]}) "
              f"F1={best_novel[1]['f1']:.4f}  delta={best_novel[1]['f1'] - f1_baseline:+.4f}")
    if tg_result:
        print(f"  Two-Gate:      F1={tg_result['f1']:.4f}  "
              f"P={tg_result['prec']:.2%}  R={tg_result['rec']:.2%}")

    beats_baseline = [(col, r) for col, r in novel_results.items() if r['f1'] > best_baseline_f1]
    if beats_baseline:
        print(f"\n  Formulas beating all baselines ({best_baseline_f1:.4f}):")
        for col, r in sorted(beats_baseline, key=lambda x: x[1]['f1'], reverse=True):
            print(f"    {col:<22s} ({NOVEL_FORMULAS[col]:<22s})  F1={r['f1']:.4f}  "
                  f"delta={r['f1'] - best_baseline_f1:+.4f}")
    else:
        print(f"\n  No novel formula beats the best baseline ({best_baseline_f1:.4f}).")

    print("=" * 70)


if __name__ == "__main__":
    _LOG_DIR = os.path.join(_REPO_ROOT, "Pipeline", "logs", "comparisons")
    os.makedirs(_LOG_DIR, exist_ok=True)
    _LOG_PATH = os.path.join(_LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_group_f.txt")

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
