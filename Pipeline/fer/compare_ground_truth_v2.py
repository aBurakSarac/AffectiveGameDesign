import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurable label sets
# ---------------------------------------------------------------------------

# Only these labels are scored against the model.
# Everything else is reported as "excluded" and used for contextual FP checking.
SCOREABLE_LABELS = {'fear'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_to_sec(t_str):
    s = str(t_str).strip()
    if ':' in s:
        parts = s.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    return float(s)


def overlaps(trigger_start, trigger_end, gt_start, gt_end):
    """Any overlap between a trigger block and a padded GT window."""
    return max(trigger_start, gt_start) <= min(trigger_end, gt_end)


def _make_block(group):
    return {
        'start':        group['timestamp'].iloc[0],
        'end':          group['timestamp'].iloc[-1],
        'peak_hs_fear': round(group['hs_fear'].max(), 3),
        'mean_hs_fear': round(group['hs_fear'].mean(), 3),
        'duration_s':   round(
            group['timestamp'].iloc[-1] - group['timestamp'].iloc[0], 2
        ),
        'hit': False,
    }


def build_trigger_blocks(model_df, threshold, min_frames,
                         window_size=None, fill_ratio=0.65,
                         signal_col='hs_fear'):
    """
    Detect sustained fear blocks in the model signal.

    signal_col: which column to threshold (default 'hs_fear').

    Strict mode (window_size=None):
        Requires min_frames *consecutive* frames >= threshold.
        Original behaviour — clean but brittle against classifier noise.

    Sliding-window mode (window_size set):
        Scans every window_size-frame window; if fill_ratio of frames in that
        window are >= threshold the whole window is marked as qualified.
        Adjacent qualified frames are merged into blocks.
        Tolerates the short dips caused by HSEmotion classifier noise without
        lowering the threshold or the min_frames requirement.

    fill_ratio=0.65 with window_size=15 means 10/15 frames must qualify —
    equivalent effective density to the original min_frames=10 consecutive
    requirement, but robust to single-frame dropouts.
    """
    model_df = model_df.copy()
    model_df['is_fear'] = (pd.to_numeric(model_df[signal_col], errors='coerce').fillna(0.0)
                           >= threshold).astype(int)

    if window_size is None:
        # --- strict consecutive mode ---
        model_df['block_id'] = (
            model_df['is_fear'] != model_df['is_fear'].shift()
        ).cumsum()
        blocks = []
        for _, group in model_df[model_df['is_fear'] == 1].groupby('block_id'):
            if len(group) >= min_frames:
                blocks.append(_make_block(group))
        return blocks

    # --- sliding-window mode ---
    fear_vals = model_df['is_fear'].values
    required  = int(window_size * fill_ratio)   # e.g. 10 out of 15
    qualified = np.zeros(len(fear_vals), dtype=bool)

    for i in range(len(fear_vals) - window_size + 1):
        if fear_vals[i:i + window_size].sum() >= required:
            qualified[i:i + window_size] = True

    model_df['qualified'] = qualified
    model_df['block_id']  = (
        model_df['qualified'] != model_df['qualified'].shift()
    ).cumsum()

    blocks = []
    for _, group in model_df[model_df['qualified']].groupby('block_id'):
        if len(group) >= min_frames:
            blocks.append(_make_block(group))
    return blocks


def match_triggers(gt_df, trigger_blocks, pad_start, pad_end):
    """
    Match GT events to trigger blocks using simple any-overlap with
    asymmetric padding:
      pad_start — seconds before GT start (left-edge rounding slack)
      pad_end   — seconds after GT end (reaction delay + right-edge slack)

    Mutates trigger_blocks in-place (sets hit=True).
    Returns gt_caught count and missed_events list.
    """
    gt_caught     = 0
    missed_events = []

    for _, row in gt_df.iterrows():
        gt_s   = row['start_val'] - pad_start
        gt_e   = row['end_val']   + pad_end
        caught = False

        for trigger in trigger_blocks:
            if overlaps(trigger['start'], trigger['end'], gt_s, gt_e):
                caught        = True
                trigger['hit'] = True

        if caught:
            gt_caught += 1
        else:
            missed_events.append({'time': row['start_s'], 'label': row['label']})

    return gt_caught, missed_events


def compute_metrics(model_df, gt_df, threshold, min_frames, pad_start, pad_end,
                    window_size, fill_ratio, signal_col='hs_fear'):
    """Lightweight pass used by the threshold/window sweep."""
    blocks       = build_trigger_blocks(model_df, threshold, min_frames,
                                        window_size, fill_ratio, signal_col)
    gt_caught, _ = match_triggers(gt_df, blocks, pad_start, pad_end)

    tps  = gt_caught                              # GT-centric: one GT event = 1 TP
    fps  = sum(1 for t in blocks if not t['hit'])

    return tps, fps, gt_caught, len(gt_df)


def print_parameter_sweep(
    formula_col,
    label,
    eval_func,
    thresholds,
    min_frames,
    window_sizes,
    pad_start,
    pad_end,
    fill_ratio,
    current_params=None,
    print_table=True
):
    """
    eval_func(thresh, mf, ws) -> (tps, fps, fn, n_triggers)
    """
    if print_table:
        print(f"\n{'='*70}")
        print(f"  === Formula: {formula_col} ({label}) ===")
        print(f"  pad_start={pad_start}s  pad_end={pad_end}s  fill_ratio={fill_ratio:.0%}")
        print(f"{'='*70}")
        print(f"{'Thresh':>7}  {'MinFr':>6}  {'Win':>5}  "
              f"{'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Triggers':>9}")

    rows = []
    for thresh in thresholds:
        for mf in min_frames:
            for ws in window_sizes:
                tps, fps, gt_caught, n_gt = eval_func(thresh, mf, ws)
                p = tps / (tps + fps) if (tps + fps) > 0 else 0.0
                r = gt_caught / n_gt if n_gt > 0 else 0.0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                rows.append((thresh, mf, ws, p, r, f, tps + fps))

    if not rows:
        return None

    best_f1 = max(row[5] for row in rows)
    best_row = next(row for row in rows if abs(row[5] - best_f1) < 1e-9)

    prev_thresh = None
    for thresh, mf, ws, p, r, f, n in rows:
        if print_table and prev_thresh is not None and thresh != prev_thresh:
            print()
        prev_thresh = thresh
        win_str = str(ws) if ws else 'strict'
        
        is_current = current_params and thresh == current_params[0] and mf == current_params[1] and ws == current_params[2]
        is_best    = abs(f - best_f1) < 1e-9
        if is_current and is_best:
            marker = " <-- current / BEST F1"
        elif is_current:
            marker = " <-- current"
        elif is_best:
            marker = " <-- BEST F1"
        else:
            marker = ""
            
        if print_table:
            print(f"{thresh:>7.2f}  {mf:>6}  {win_str:>5}  "
                  f"{p:>8.2%}  {r:>8.2%}  {f:>8.4f}  {n:>9}{marker}")

    return {
        'formula': formula_col,
        'label':   label,
        'thresh':  best_row[0],
        'min_fr':  best_row[1],
        'win':     str(best_row[2]) if best_row[2] else 'strict',
        'prec':    best_row[3],
        'rec':     best_row[4],
        'f1':      best_row[5],
    }


def print_best_configs_summary(best_per_formula, title="Best Config Per Formula"):
    if not best_per_formula:
        return
    print(f"\n\n{'='*70}")
    print(f"  --- {title} ---")
    print(f"{'='*70}")
    print(f"{'Formula':<10}  {'Label':<12}  {'Thresh':>7}  {'MinFr':>6}  "
          f"{'Win':>6}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}")
    best_f1_overall = max(r['f1'] for r in best_per_formula)
    for r in best_per_formula:
        marker = "  <-- BEST" if abs(r['f1'] - best_f1_overall) < 1e-9 else ""
        print(f"{r['formula']:<10}  {r['label']:<12}  {r['thresh']:>7.2f}  "
              f"{r['min_fr']:>6}  {r['win']:>6}  {r['prec']:>8.2%}  "
              f"{r['rec']:>8.2%}  {r['f1']:>8.4f}{marker}")
    print()# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_advanced_analysis(
    gt_file,
    model_file,
    threshold   = 0.70,
    min_frames  = 10,
    pad_start   = 0.5,    # seconds before GT start
    pad_end     = 1.0,    # seconds after GT end
    window_size = 15,     # sliding-window width in frames (None = strict mode)
    fill_ratio  = 0.65,   # fraction of window frames that must be >= threshold
    signal_col  = 'hs_fear',
):
    # ------------------------------------------------------------------
    # 1. Load & split GT into scoreable vs excluded
    # ------------------------------------------------------------------
    raw_gt   = pd.read_csv(gt_file, low_memory=False)
    model_df = pd.read_csv(model_file, low_memory=False)

    raw_gt['start_val'] = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val']   = raw_gt['end_s'].apply(time_to_sec)

    # Normalise label case for matching
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm       = {l.lower() for l in SCOREABLE_LABELS}

    gt_df      = raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)
    excluded   = raw_gt[~raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Skipped-labels summary
    # ------------------------------------------------------------------
    if len(excluded) > 0:
        print(f"\n--- Excluded from Scoring ({len(excluded)} events) ---")
        print(excluded[['start_s', 'end_s', 'label']].to_string(index=False))
        print("  (not scored — model is not expected to detect these labels)")

    # ------------------------------------------------------------------
    # 3. Build triggers and match against scoreable GT
    # ------------------------------------------------------------------
    trigger_blocks           = build_trigger_blocks(model_df, threshold, min_frames,
                                                    window_size, fill_ratio, signal_col)
    gt_caught, missed_events = match_triggers(gt_df, trigger_blocks, pad_start, pad_end)

    tps       = gt_caught                                         # GT-centric: one cluster = 1 TP
    fps_count = sum(1 for t in trigger_blocks if not t['hit'])
    precision = tps / (tps + fps_count) if (tps + fps_count) > 0 else 0.0
    recall    = gt_caught / len(gt_df)   if len(gt_df) > 0         else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    mode_str = (f"window={window_size}fr  fill={fill_ratio:.0%}"
                if window_size else "strict consecutive")

    # ------------------------------------------------------------------
    # 4. Headline stats
    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  Advanced Analysis: {model_file.split('/')[-1]}")
    print(f"  signal_col={signal_col}  threshold={threshold}  min_frames={min_frames}  {mode_str}")
    print(f"  pad_start=-{pad_start}s  pad_end=+{pad_end}s")
    print(f"{'='*62}")
    print(f"  Precision : {precision:.2%}  (how clean are your triggers?)")
    print(f"  Recall    : {recall:.2%}  (how many scares did you catch?)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Triggers  : {len(trigger_blocks)} total  ({fps_count} FP / {len(trigger_blocks)-fps_count} hit-any-GT)")
    print(f"  GT Events : {gt_caught}/{len(gt_df)} caught  "
          f"({len(excluded)} excluded from scoring)")

    # ------------------------------------------------------------------
    # 5. Per-label recall breakdown (scoreable labels only)
    # ------------------------------------------------------------------
    print(f"\n--- Recall by Label (scoreable) ---")
    label_stats = {}
    for _, row in gt_df.iterrows():
        lbl = row['label']
        label_stats.setdefault(lbl, {'total': 0, 'caught': 0})
        label_stats[lbl]['total'] += 1
        gt_s = row['start_val'] - pad_start
        gt_e = row['end_val']   + pad_end
        if any(overlaps(t['start'], t['end'], gt_s, gt_e) for t in trigger_blocks):
            label_stats[lbl]['caught'] += 1

    rows = []
    for lbl, s in sorted(label_stats.items()):
        pct = s['caught'] / s['total'] * 100
        rows.append({'label': lbl, 'caught': s['caught'],
                     'total': s['total'], 'recall_%': f"{pct:.0f}%"})
    print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Missed events — signal strength and face coverage in window
    # ------------------------------------------------------------------
    if missed_events:
        print(f"\n--- Missed Events (False Negatives) ---")
        for ev in missed_events:
            t0     = time_to_sec(ev['time'])
            window = model_df[
                (model_df['timestamp'] >= t0 - pad_start) &
                (model_df['timestamp'] <= t0 + pad_end + 2.0)
            ]
            ev['peak_hs_fear']    = round(window['hs_fear'].max(), 3)  if len(window) else float('nan')
            ev['mean_hs_fear']    = round(window['hs_fear'].mean(), 3) if len(window) else float('nan')
            ev['face_detected_%'] = (
                f"{window['mp_face_detected'].mean()*100:.0f}%"
                if len(window) else 'n/a'
            )
        print(pd.DataFrame(missed_events)[
            ['time', 'label', 'peak_hs_fear', 'mean_hs_fear', 'face_detected_%']
        ].to_string(index=False))

    # ------------------------------------------------------------------
    # 7. False positive characterisation
    #    — distance to nearest scoreable GT
    #    — distance to nearest excluded GT (contextual check)
    # ------------------------------------------------------------------
    fps_list = [t for t in trigger_blocks if not t['hit']]
    if fps_list:
        for fp in fps_list:
            fp_mid = (fp['start'] + fp['end']) / 2

            # Nearest scoreable GT event
            fp['nearest_fear_gt_s'] = round(
                min(abs(fp_mid - r['start_val']) for _, r in gt_df.iterrows())
                if len(gt_df) else 999, 2
            )

            # Nearest excluded GT event + its label
            if len(excluded) > 0:
                dists = (excluded['start_val'] - fp_mid).abs()
                idx   = dists.idxmin()
                fp['nearest_excl_gt_s']  = round(dists[idx], 2)
                fp['nearest_excl_label'] = excluded.loc[idx, 'label']
            else:
                fp['nearest_excl_gt_s']  = 999
                fp['nearest_excl_label'] = '—'

        print(f"\n--- False Positive Regions ({fps_count} total) ---")
        print(pd.DataFrame(fps_list)[[
            'start', 'end', 'duration_s',
            'peak_hs_fear', 'mean_hs_fear',
            'nearest_fear_gt_s', 'nearest_excl_gt_s', 'nearest_excl_label'
        ]].to_string(index=False))

        near_fear = sum(1 for fp in fps_list if fp['nearest_fear_gt_s'] <= 3.0)
        near_excl = sum(1 for fp in fps_list
                        if fp['nearest_excl_gt_s'] <= 3.0
                        and fp['nearest_fear_gt_s'] > 3.0)

        if near_fear:
            print(f"\n  -> {near_fear}/{fps_count} FPs within 3s of a scoreable GT "
                  f"(likely real reactions outside annotation window)")
        if near_excl:
            print(f"  -> {near_excl}/{fps_count} FPs near an excluded event "
                  f"(model fired on {', '.join(set(fp['nearest_excl_label'] for fp in fps_list if fp['nearest_excl_gt_s'] <= 3.0 and fp['nearest_fear_gt_s'] > 3.0))})"
                  f" — not a model error, label scope decision")

    # ------------------------------------------------------------------
    # 8. Threshold + window sweep
    # ------------------------------------------------------------------
    sweep_thresholds   = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep_min_frames   = [5, 8, 10, 12, 15]
    sweep_window_sizes = [None, 10, 15, 20]   # None = strict mode

    print_parameter_sweep(
        formula_col=signal_col,
        label=signal_col,
        eval_func=lambda thresh, mf, ws: compute_metrics(
            model_df, gt_df, thresh, mf, pad_start, pad_end, ws, fill_ratio, signal_col
        ),
        thresholds=sweep_thresholds,
        min_frames=sweep_min_frames,
        window_sizes=sweep_window_sizes,
        pad_start=pad_start,
        pad_end=pad_end,
        fill_ratio=fill_ratio,
        current_params=(threshold, min_frames, window_size)
    )


# Human-readable labels for known formula columns
_FORMULA_LABELS = {
    'hs_fear': 'raw_hs',
    'composite_fear': 'F*(1+T)',
    'smoothed_composite': 'smoothed',
    'f0': 'f0', 'f1': 'f1', 'f2': 'f2', 'f3': 'f3',
    'f4': 'f4', 'f5': 'f5', 'f6': 'f6',
    'f7': 'hs_only', 'f8': 'mp_only',
    'f9': 'max_fusion', 'f10': 'geo_mean', 'f11': 'hs-anger',
    'f12': 'hybrid_amp',
    'f13': 'fear*(1+T)',
}

_ALL_FORMULA_COLS = ['hs_fear', 'composite_fear', 'smoothed_composite',
                     'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
                     'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']


def run_formula_comparison(
    gt_file,
    model_file,
    pad_start   = 0.5,
    pad_end     = 1.0,
    fill_ratio  = 0.65,
):
    """Run the parameter sweep for every formula column found in model_file.

    Prints one sweep table per formula, then a best-per-formula summary.
    Missing columns are skipped with a note.
    """
    raw_gt   = pd.read_csv(gt_file, low_memory=False)
    model_df = pd.read_csv(model_file, low_memory=False)

    raw_gt['start_val'] = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val']   = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    gt_df = raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)

    sweep_thresholds   = [0.4, 0.5, 0.6, 0.70, 0.80]
    sweep_min_frames   = [5, 8, 10, 12, 15]
    sweep_window_sizes = [None, 10, 15, 20]

    present_cols  = [c for c in _ALL_FORMULA_COLS if c in model_df.columns]
    missing_cols  = [c for c in _ALL_FORMULA_COLS if c not in model_df.columns]

    if missing_cols:
        print(f"\n  [formula sweep] Skipping (not in CSV): {', '.join(missing_cols)}")

    best_per_formula = []

    for col in present_cols:
        label = _FORMULA_LABELS.get(col, col)
        best = print_parameter_sweep(
            formula_col=col,
            label=label,
            eval_func=lambda thresh, mf, ws: compute_metrics(
                model_df, gt_df, thresh, mf, pad_start, pad_end, ws, fill_ratio, col
            ),
            thresholds=sweep_thresholds,
            min_frames=sweep_min_frames,
            window_sizes=sweep_window_sizes,
            pad_start=pad_start,
            pad_end=pad_end,
            fill_ratio=fill_ratio
        )
        if best:
            best_per_formula.append(best)

    print_best_configs_summary(best_per_formula, "Best Config Per Formula")


def print_fp_fn_analysis(model_df, gt_df, signal_col='composite_fear',
                         n_fp=15, gap_frames=30,
                         pad_start=0.5, pad_end=1.0):
    """FP + FN focused analysis for a composite signal column.

    Section 1 — FALSE POSITIVES (UNLABELED peaks):
        Clusters consecutive above-threshold frames into regions, filters
        out any that overlap a GT event, and reports the top n_fp by peak
        value.  One entry per distinct region.

    Section 2 — FALSE NEGATIVES (missed GT events):
        For every GT event not covered by any high-signal region, reports
        the peak and mean composite score inside that GT window so you can
        see *why* the detector missed it.
    """
    if signal_col not in model_df.columns:
        print(f"\n  [fp/fn] Column '{signal_col}' not in CSV — skipping.")
        return

    sig = pd.to_numeric(model_df[signal_col], errors='coerce').fillna(0.0).values
    ts  = pd.to_numeric(model_df['timestamp'], errors='coerce').fillna(0.0).values

    # ── cluster high-signal regions ──────────────────────────────────────
    min_thresh = 0.15
    above = sig >= min_thresh

    regions = []
    in_region = False
    start_i = 0
    gap_count = 0

    for i in range(len(above)):
        if above[i]:
            if not in_region:
                in_region = True
                start_i = i
                gap_count = 0
            else:
                gap_count = 0
        else:
            if in_region:
                gap_count += 1
                if gap_count >= gap_frames:
                    end_i = i - gap_count
                    regions.append((start_i, end_i))
                    in_region = False
    if in_region:
        regions.append((start_i, len(sig) - 1))

    # ── classify each region as GT_HIT or UNLABELED ──────────────────────
    all_entries = []
    for s_i, e_i in regions:
        seg = sig[s_i:e_i + 1]
        peak_val = float(seg.max())
        mean_val = float(seg.mean())
        t_start = float(ts[s_i])
        t_end   = float(ts[min(e_i, len(ts) - 1)])
        dur     = t_end - t_start

        is_hit = False
        for _, row in gt_df.iterrows():
            gt_s = row['start_val'] - pad_start
            gt_e = row['end_val']   + pad_end
            if overlaps(t_start, t_end, gt_s, gt_e):
                is_hit = True
                break

        hs_fear_peak = float(model_df['hs_fear'].iloc[s_i:e_i + 1].max()) if 'hs_fear' in model_df.columns else 0.0
        mp_tension_peak = float(model_df['mp_tension'].iloc[s_i:e_i + 1].max()) if 'mp_tension' in model_df.columns else 0.0

        all_entries.append({
            'start_s': round(t_start, 1),
            'end_s':   round(t_end, 1),
            'dur':     round(dur, 1),
            'peak':    round(peak_val, 3),
            'mean':    round(mean_val, 3),
            'pk_fear': round(hs_fear_peak, 3),
            'pk_tens': round(mp_tension_peak, 3),
            'is_hit':  is_hit,
        })

    # ── Section 1: FALSE POSITIVES (unlabeled peaks) ─────────────────────
    fps = [e for e in all_entries if not e['is_hit']]
    fps.sort(key=lambda e: e['peak'], reverse=True)
    fps = fps[:n_fp]

    print(f"\n{'='*70}")
    print(f"  FALSE POSITIVES — Top {len(fps)} Unlabeled Peaks — {signal_col}")
    print(f"  (high composite regions NOT matching any GT event)")
    print(f"{'='*70}")

    if fps:
        print(f"{'#':>3}  {'Start':>7}  {'End':>7}  {'Dur':>5}  {'Peak':>6}  "
              f"{'Mean':>6}  {'Fear':>6}  {'Tens':>6}")
        for i, e in enumerate(fps, 1):
            print(f"{i:>3}  {e['start_s']:>7.1f}  {e['end_s']:>7.1f}  {e['dur']:>5.1f}  "
                  f"{e['peak']:>6.3f}  {e['mean']:>6.3f}  "
                  f"{e['pk_fear']:>6.3f}  {e['pk_tens']:>6.3f}")
        print(f"\n  -> {len(fps)} unlabeled region(s) with signal >= {min_thresh}")
        print(f"     Review these timestamps — could be GT omissions or true FPs")
    else:
        print("  No unlabeled high-signal regions found.")

    # ── Section 2: FALSE NEGATIVES (missed GT events) ────────────────────
    missed = []
    for _, row in gt_df.iterrows():
        gt_s = row['start_val']
        gt_e = row['end_val']
        padded_s = gt_s - pad_start
        padded_e = gt_e + pad_end

        caught = any(
            overlaps(e['start_s'], e['end_s'], padded_s, padded_e)
            for e in all_entries
        )
        if caught:
            continue

        window = model_df[
            (pd.to_numeric(model_df['timestamp'], errors='coerce') >= gt_s - 1.0) &
            (pd.to_numeric(model_df['timestamp'], errors='coerce') <= gt_e + 2.0)
        ]

        peak_sig  = round(float(pd.to_numeric(window[signal_col], errors='coerce').max()), 3) if len(window) else 0.0
        mean_sig  = round(float(pd.to_numeric(window[signal_col], errors='coerce').mean()), 3) if len(window) else 0.0
        peak_fear = round(float(window['hs_fear'].astype(float).max()), 3) if len(window) and 'hs_fear' in window.columns else 0.0
        peak_tens = round(float(window['mp_tension'].astype(float).max()), 3) if len(window) and 'mp_tension' in window.columns else 0.0
        face_pct  = f"{window['mp_face_detected'].astype(float).mean()*100:.0f}%" if len(window) and 'mp_face_detected' in window.columns else 'n/a'

        missed.append({
            'gt_start': round(gt_s, 1),
            'gt_end':   round(gt_e, 1),
            'label':    row.get('label', ''),
            'pk_comp':  peak_sig,
            'mn_comp':  mean_sig,
            'pk_fear':  peak_fear,
            'pk_tens':  peak_tens,
            'face_%':   face_pct,
        })

    print(f"\n{'='*70}")
    print(f"  FALSE NEGATIVES — Missed GT Events — {signal_col}")
    print(f"  (GT events with no high-signal region overlapping)")
    print(f"{'='*70}")

    if missed:
        print(f"{'#':>3}  {'GT_Start':>8}  {'GT_End':>7}  {'Label':<8}  "
              f"{'PkComp':>7}  {'MnComp':>7}  {'PkFear':>7}  {'PkTens':>7}  {'Face%':>6}")
        for i, m in enumerate(missed, 1):
            print(f"{i:>3}  {m['gt_start']:>8.1f}  {m['gt_end']:>7.1f}  {m['label']:<8}  "
                  f"{m['pk_comp']:>7.3f}  {m['mn_comp']:>7.3f}  "
                  f"{m['pk_fear']:>7.3f}  {m['pk_tens']:>7.3f}  {m['face_%']:>6}")
        print(f"\n  -> {len(missed)}/{len(gt_df)} GT events missed")
        print(f"     Low PkComp = detector legitimately low; low Face% = face not detected")
    else:
        print(f"  All {len(gt_df)} GT events have overlapping high-signal regions.")


def generate_timeline_plot(model_df, gt_df, trigger_blocks, signal_col,
                           threshold, out_path, excluded_df=None, title_suffix=""):
    """Visual timeline: signal trace + GT spans + trigger blocks (FP/TP)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("  [timeline] matplotlib not available — skipping plot.")
        return

    DARK_BG = "#1a1a2e"
    PANEL_BG = "#16213e"
    GRID_COL = "#2a2a4a"

    ts = pd.to_numeric(model_df['timestamp'], errors='coerce').fillna(0.0).values
    sig = pd.to_numeric(model_df[signal_col], errors='coerce').fillna(0.0).values
    smooth_w = 15
    sig_smooth = pd.Series(sig).rolling(smooth_w, center=True, min_periods=1).mean().values

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), facecolor=DARK_BG,
                             gridspec_kw={"hspace": 0.30, "height_ratios": [3, 1]})

    # --- Panel 1: Signal trace + overlays ---
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.set_title(f"Timeline — {signal_col} {title_suffix}", color="white", fontsize=12, pad=8)

    ax.fill_between(ts, 0, sig, alpha=0.15, color="#cc44ff")
    ax.plot(ts, sig_smooth, color="#cc44ff", linewidth=1.2, label=f"{signal_col} (smooth)")
    ax.axhline(y=threshold, color="#ff6666", linestyle="--", alpha=0.5, linewidth=0.8,
               label=f"threshold={threshold}")

    # GT event spans
    for _, row in gt_df.iterrows():
        gs = row['start_val']
        ge = row['end_val']
        ax.axvspan(gs, ge, ymin=0, ymax=1, alpha=0.20, color="#00cc00")
        ax.text(gs, 0.98, row.get('label', 'GT'), fontsize=7, color="#00cc00",
                va='top', transform=ax.get_xaxis_transform())

    # Excluded GT spans (dimmer)
    if excluded_df is not None and len(excluded_df) > 0:
        for _, row in excluded_df.iterrows():
            ax.axvspan(row['start_val'], row['end_val'], ymin=0, ymax=1,
                       alpha=0.08, color="#888888")

    # Trigger blocks
    for blk in trigger_blocks:
        color = "#4488ff" if blk['hit'] else "#ff4444"
        alpha = 0.30 if blk['hit'] else 0.35
        ax.axvspan(blk['start'], blk['end'], ymin=0, ymax=0.08, alpha=alpha, color=color)
        label_text = "TP" if blk['hit'] else "FP"
        ax.text((blk['start'] + blk['end']) / 2, 0.04, label_text, fontsize=7,
                color=color, ha='center', va='center', fontweight='bold',
                transform=ax.get_xaxis_transform())

    # Missed GT events (FN markers)
    for _, row in gt_df.iterrows():
        gs = row['start_val']
        ge = row['end_val']
        caught = any(overlaps(t['start'], t['end'], gs - 0.5, ge + 1.0) for t in trigger_blocks)
        if not caught:
            mid = (gs + ge) / 2
            ax.annotate("FN", xy=(mid, 0.02), fontsize=9, color="#ffaa00",
                        ha='center', fontweight='bold',
                        xycoords=('data', 'axes fraction'))
            ax.axvspan(gs, ge, ymin=0, ymax=1, alpha=0.10, color="#ffaa00")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(signal_col, color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=GRID_COL, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel 2: hs_fear + mp_tension breakdown ---
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.set_title("Component Signals", color="white", fontsize=10, pad=6)

    if 'hs_fear' in model_df.columns:
        fear = pd.to_numeric(model_df['hs_fear'], errors='coerce').fillna(0.0).values
        fear_s = pd.Series(fear).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax2.plot(ts, fear_s, color="#ff4488", linewidth=1.0, label="hs_fear")

    if 'mp_tension' in model_df.columns:
        tens = pd.to_numeric(model_df['mp_tension'], errors='coerce').fillna(0.0).values
        tens_s = pd.Series(tens).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax2.plot(ts, tens_s, color="#44ff88", linewidth=1.0, label="mp_tension")

    if 'hs_arousal' in model_df.columns:
        arou = pd.to_numeric(model_df['hs_arousal'], errors='coerce').fillna(0.0).values
        arou_s = pd.Series(arou).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax2.plot(ts, arou_s, color="#ff8844", linewidth=0.8, alpha=0.6, label="hs_arousal")

    for _, row in gt_df.iterrows():
        ax2.axvspan(row['start_val'], row['end_val'], ymin=0, ymax=1,
                    alpha=0.12, color="#00cc00")

    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Time (s)", color="white", fontsize=10)
    ax2.set_ylabel("Score", color="white", fontsize=10)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.grid(True, color=GRID_COL, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"\n  [timeline] Plot saved → {out_path}")


class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files: f.write(text)
    def flush(self):
        for f in self.files: f.flush()

def generate_v2_report(gt_file, model_file, threshold=0.70, min_frames=10,
                       pad_start=0.5, pad_end=1.0, window_size=15, fill_ratio=0.65,
                       signal_col='hs_fear', quiet=False, output_dir=None):
    log_stem = os.path.splitext(os.path.basename(model_file))[0]
    if output_dir:
        log_dir = output_dir
        session_tag = log_stem.split("_mp_hs_")[-1] if "_mp_hs_" in log_stem else log_stem
        log_path = os.path.join(log_dir, f"{session_tag}_v2.txt")
    else:
        base_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'comparisons'))
        ts_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_tag = log_stem.split("_mp_hs_")[-1] if "_mp_hs_" in log_stem else log_stem
        log_dir = os.path.join(base_dir, "standalone", f"{ts_prefix}_{session_tag}")
        log_path = os.path.join(log_dir, f"{session_tag}_v2.txt")
    os.makedirs(log_dir, exist_ok=True)

    with open(log_path, 'w') as _fh:
        _orig = sys.stdout
        sys.stdout = _fh if quiet else _Tee(sys.stdout, _fh)
        try:
            run_advanced_analysis(
                gt_file     = gt_file,
                model_file  = model_file,
                threshold   = threshold,
                min_frames  = min_frames,
                pad_start   = pad_start,
                pad_end     = pad_end,
                window_size = window_size,
                fill_ratio  = fill_ratio,
                signal_col  = signal_col,
            )
            run_formula_comparison(
                gt_file    = gt_file,
                model_file = model_file,
                pad_start  = pad_start,
                pad_end    = pad_end,
                fill_ratio = fill_ratio,
            )
        finally:
            sys.stdout = _orig
    print(f"\n[v2 report] Log saved → {log_path}")

    # --- Top composite peaks + timeline plot (outside Tee to avoid binary in log) ---
    model_df = pd.read_csv(model_file, low_memory=False)
    raw_gt   = pd.read_csv(gt_file, low_memory=False)
    raw_gt['start_val'] = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val']   = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    gt_df    = raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)
    excluded = raw_gt[~raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)

    peaks_path = os.path.join(log_dir, f"{session_tag}_peaks.txt")
    with open(peaks_path, 'w') as _pk:
        _orig2 = sys.stdout
        sys.stdout = _Tee(sys.stdout, _pk)
        try:
            for col in ['composite_fear', 'smoothed_composite']:
                if col in model_df.columns:
                    print_fp_fn_analysis(model_df, gt_df, signal_col=col,
                                         pad_start=pad_start, pad_end=pad_end)
        finally:
            sys.stdout = _orig2
    print(f"[v2 report] FP/FN analysis saved → {peaks_path}")

    plot_col = 'composite_fear' if 'composite_fear' in model_df.columns else signal_col
    trigger_blocks = build_trigger_blocks(model_df, threshold, min_frames,
                                          window_size, fill_ratio, plot_col)
    match_triggers(gt_df, trigger_blocks, pad_start, pad_end)

    plot_path = os.path.join(log_dir, f"{session_tag}_timeline.png")
    generate_timeline_plot(model_df, gt_df, trigger_blocks, plot_col,
                           threshold, plot_path, excluded_df=excluded,
                           title_suffix=f"(thresh={threshold}, mf={min_frames})")

if __name__ == "__main__":
    _GT_FILE    = '/home/burak/Desktop/Bitirme/Annotations/S10_Vid13.csv'
    _MODEL_FILE = '/home/burak/Desktop/Bitirme/Pipeline/logs/sessions/S10_Vid13_bright/20260517_200406_mp_hs_S10_Vid13_bright.csv'
    generate_v2_report(_GT_FILE, _MODEL_FILE)