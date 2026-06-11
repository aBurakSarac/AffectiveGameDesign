import os
import sys
from datetime import datetime

import pandas as pd


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


def build_trigger_blocks(model_df, threshold, min_frames):
    """Return list of trigger dicts from a model DataFrame."""
    model_df = model_df.copy()
    model_df['is_fear']  = (model_df['hs_fear'] >= threshold).astype(int)
    model_df['block_id'] = (model_df['is_fear'] != model_df['is_fear'].shift()).cumsum()

    blocks = []
    for _, group in model_df[model_df['is_fear'] == 1].groupby('block_id'):
        if len(group) >= min_frames:
            blocks.append({
                'start':        group['timestamp'].iloc[0],
                'end':          group['timestamp'].iloc[-1],
                'peak_hs_fear': round(group['hs_fear'].max(), 3),
                'mean_hs_fear': round(group['hs_fear'].mean(), 3),
                'duration_s':   round(group['timestamp'].iloc[-1] - group['timestamp'].iloc[0], 2),
                'hit':          False,
            })
    return blocks


def match_triggers(gt_df, trigger_blocks, pad_start, pad_end):
    """
    Match GT events to trigger blocks using simple any-overlap with
    asymmetric padding:
      pad_start — seconds subtracted from GT start (accounts for early model
                  response and minute:second rounding on the left edge)
      pad_end   — seconds added to GT end (accounts for reaction delay and
                  rounding slack on the right edge)

    Mutates trigger_blocks in-place (sets hit=True on matched blocks).
    Returns gt_caught count and missed_events list.
    """
    gt_caught     = 0
    missed_events = []

    for _, row in gt_df.iterrows():
        gt_s = row['start_val'] - pad_start
        gt_e = row['end_val']   + pad_end
        caught = False

        for trigger in trigger_blocks:
            if overlaps(trigger['start'], trigger['end'], gt_s, gt_e):
                caught = True
                trigger['hit'] = True

        if caught:
            gt_caught += 1
        else:
            missed_events.append({
                'time':  row['start_s'],
                'label': row['label'],
            })

    return gt_caught, missed_events


def compute_metrics(model_df, gt_df, threshold, min_frames, pad_start, pad_end):
    """Lightweight metric pass used by the threshold sweep."""
    blocks       = build_trigger_blocks(model_df, threshold, min_frames)
    gt_caught, _ = match_triggers(gt_df, blocks, pad_start, pad_end)

    tps  = sum(1 for t in blocks if t['hit'])
    fps  = sum(1 for t in blocks if not t['hit'])
    prec = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    rec  = gt_caught / len(gt_df) if len(gt_df) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, len(blocks)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_advanced_analysis(
    gt_file,
    model_file,
    threshold  = 0.7,
    min_frames =10,
    pad_start  = 0.5,   # seconds before GT start
    pad_end    = 1.0,   # seconds after GT end
):
    # 1. Load data
    gt_df    = pd.read_csv(gt_file)
    model_df = pd.read_csv(model_file)

    # Only evaluate fear-family labels; other labels (anger, disgust, obscured,
    # side_profile) exist in the annotation file for documentation but are not
    # targets for the fear detector.
    
    #FEAR_LABELS = {'fear', 'slight_fear', 'Slight_fear'}
    FEAR_LABELS = {'fear'}
    gt_df = gt_df[gt_df['label'].isin(FEAR_LABELS)].reset_index(drop=True)

    gt_df['start_val'] = gt_df['start_s'].apply(time_to_sec)
    gt_df['end_val']   = gt_df['end_s'].apply(time_to_sec)

    # 2. Build triggers and match
    trigger_blocks           = build_trigger_blocks(model_df, threshold, min_frames)
    gt_caught, missed_events = match_triggers(gt_df, trigger_blocks, pad_start, pad_end)

    tps       = sum(1 for t in trigger_blocks if t['hit'])
    fps       = sum(1 for t in trigger_blocks if not t['hit'])
    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall    = gt_caught / len(gt_df) if len(gt_df) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ------------------------------------------------------------------
    # 3. Headline stats
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Advanced Analysis: {model_file.split('/')[-1]}")
    print(f"  threshold={threshold}  min_frames={min_frames}  "
          f"pad_start=-{pad_start}s  pad_end=+{pad_end}s")
    print(f"{'='*60}")
    print(f"  Precision : {precision:.2%}  (how clean are your triggers?)")
    print(f"  Recall    : {recall:.2%}  (how many scares did you catch?)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Triggers  : {len(trigger_blocks)} total  ({tps} TP / {fps} FP)")
    print(f"  GT Events : {gt_caught}/{len(gt_df)} caught")

    # ------------------------------------------------------------------
    # 4. Per-label recall breakdown
    # ------------------------------------------------------------------
    print(f"\n--- Recall by Label ---")
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
        rows.append({
            'label':    lbl,
            'caught':   s['caught'],
            'total':    s['total'],
            'recall_%': f"{pct:.0f}%",
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Missed events — peak hs_fear and face coverage in window
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
    # 6. False positive characterisation — distance to nearest GT event
    # ------------------------------------------------------------------
    fps_list = [t for t in trigger_blocks if not t['hit']]
    if fps_list:
        for fp in fps_list:
            fp_mid = (fp['start'] + fp['end']) / 2
            fp['nearest_gt_s'] = round(
                min(abs(fp_mid - row['start_val']) for _, row in gt_df.iterrows()), 2
            )

        print(f"\n--- False Positive Regions ({fps} total) ---")
        print(pd.DataFrame(fps_list)[
            ['start', 'end', 'duration_s', 'peak_hs_fear', 'mean_hs_fear', 'nearest_gt_s']
        ].to_string(index=False))

        near = sum(1 for fp in fps_list if fp['nearest_gt_s'] <= 3.0)
        if near:
            print(f"\n  -> {near}/{fps} FPs are within 3s of a GT event "
                  f"(likely real reactions outside annotation window)")

    # ------------------------------------------------------------------
    # 7. Threshold sweep
    # ------------------------------------------------------------------
    print(f"\n--- Threshold Sweep "
          f"(min_frames={min_frames}, pad_start={pad_start}s, pad_end={pad_end}s) ---")
    print(f"{'Thresh':>7}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'Triggers':>9}")
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8]:
        for min_frames in [1, 2, 5, 10, 15, 20, 30]:
            p, r, f, n = compute_metrics(model_df, gt_df, thresh, min_frames, pad_start, pad_end)
            marker = " <--" if thresh == threshold else ""
            print(f"{thresh:>7.2f} {min_frames:>2} {p:>8.2%}  {r:>8.2%}  {f:>8.4f}  {n:>9}{marker}")
 
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_GT_FILE    = '/home/burak/Desktop/Bitirme/Annotations/S02_Vid4.csv'
_MODEL_FILE = '/home/burak/Desktop/Bitirme/Pipeline/logs/sessions/S02_AlienCompilation_dim/20260429_093554_mp_hs_S02_AlienCompilation_dim.csv'
_LOG_DIR    = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'comparisons'))
_LOG_STEM   = os.path.splitext(os.path.basename(_MODEL_FILE))[0]
_LOG_PATH   = os.path.join(_LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_LOG_STEM}_v1.txt")


class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, text):
        for f in self.files: f.write(text)
    def flush(self):
        for f in self.files: f.flush()


os.makedirs(_LOG_DIR, exist_ok=True)
with open(_LOG_PATH, 'w') as _fh:
    _orig, sys.stdout = sys.stdout, _Tee(sys.stdout, _fh)
    try:
        run_advanced_analysis(
            gt_file    = _GT_FILE,
            model_file = _MODEL_FILE,
            threshold  = 0.7,
            min_frames = 10,
            pad_start  = 0.5,
            pad_end    = 1.0,
        )
    finally:
        sys.stdout = _orig

print(f"\nLog saved → {_LOG_PATH}")