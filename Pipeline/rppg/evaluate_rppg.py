"""rPPG × FER fear correlation analysis — La Façade Fissurée.

Extracts rPPG BPM from annotated FER videos and checks whether heart-rate
signals correlate with hand-annotated fear events. Runs two evaluation
groups: ALL (gameplay + compilation) and GAMEPLAY-only.

Usage:
    python Pipeline/rppg/evaluate_rppg.py
    python Pipeline/rppg/evaluate_rppg.py --force-reextract
    python Pipeline/rppg/evaluate_rppg.py --mode GAMEPLAY
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Add Pipeline/ to path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fer.compare_ground_truth_v2 import (
    SCOREABLE_LABELS,
    time_to_sec,
    _FORMULA_LABELS,
    compute_metrics,
    print_parameter_sweep,
    print_best_configs_summary,
)
from rppg.analyzer import RppgAnalyzer, AnalysisConfig
from rppg.video_extractor import extract_or_load_cached

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE_DIR = os.path.join(_REPO_ROOT, "Pipeline", "logs", "rppg_cache")

# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

RPPG_SESSIONS = [
    {
        "tag": "S02_Vid04",
        "video": os.path.join(_REPO_ROOT, "Recordings", "S02_Vid04.mp4"),
        "gt_csv": os.path.join(_REPO_ROOT, "Annotations", "S02_Vid04.csv"),
        "model_csv": os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions",
                                  "S02_Vid04_dim",
                                  "20260503_153301_mp_hs_S02_Vid04_dim.csv"),
        "type": "compilation",
    },
    {
        "tag": "S02_Vid05",
        "video": os.path.join(_REPO_ROOT, "Recordings", "S02_Vid05.mp4"),
        "gt_csv": os.path.join(_REPO_ROOT, "Annotations", "S02_Vid05.csv"),
        "model_csv": os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions",
                                  "S02_Vid05_bright",
                                  "20260518_092016_mp_hs_S02_Vid05_bright.csv"),
        "type": "livestream",
    },
    {
        "tag": "S04_Vid09",
        "video": os.path.join(_REPO_ROOT, "Recordings", "S04_Vid09.mp4"),
        "gt_csv": os.path.join(_REPO_ROOT, "Annotations", "S04_Vid09.csv"),
        "model_csv": os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions",
                                  "S04_Vid09_bright",
                                  "20260517_154229_mp_hs_S04_Vid09_bright.csv"),
        "type": "gameplay",
    },
    {
        "tag": "S05_Vid10",
        "video": os.path.join(_REPO_ROOT, "Recordings", "S05_Vid10.mp4"),
        "gt_csv": os.path.join(_REPO_ROOT, "Annotations", "S05_Vid10.csv"),
        "model_csv": os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions",
                                  "S05_Vid10_bright",
                                  "20260517_175428_mp_hs_S05_Vid10_bright.csv"),
        "type": "gameplay",
    },
    {
        "tag": "S06_Vid16",
        "video": os.path.join(_REPO_ROOT, "Recordings", "S06_Vid16.mp4"),
        "gt_csv": os.path.join(_REPO_ROOT, "Annotations", "S06_Vid16.csv"),
        "model_csv": os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions",
                                  "S06_Vid16_bright",
                                  "20260503_155724_mp_hs_S06_Vid16_bright.csv"),
        "type": "gameplay",
    },
    {
        "tag": "S08_Vid18",
        "video": os.path.join(_REPO_ROOT, "Recordings", "S08_Vid18.mp4"),
        "gt_csv": os.path.join(_REPO_ROOT, "Annotations", "S08_Vid18.csv"),
        "model_csv": os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions",
                                  "S08_Vid18_mixed",
                                  "20260503_155009_mp_hs_S08_Vid18_mixed.csv"),
        "type": "compilation",
    },
    {
        "tag": "S10_Vid13",
        "video": os.path.join(_REPO_ROOT, "Recordings", "S10_Vid13.mp4"),
        "gt_csv": os.path.join(_REPO_ROOT, "Annotations", "S10_Vid13.csv"),
        "model_csv": os.path.join(_REPO_ROOT, "Pipeline", "logs", "sessions",
                                  "S10_Vid13_bright",
                                  "20260517_200406_mp_hs_S10_Vid13_bright.csv"),
        "type": "gameplay",
    },
]

# Detection params from evaluate_all best-combined (f12)
PAD_START = 0.5
PAD_END = 1.0
FILL_RATIO = 0.65

# rPPG analysis config — 15s window, 3s step for temporal resolution
RPPG_CONFIG = AnalysisConfig(
    algorithm="all",
    window_s=15.0,
    step_s=3.0,
    bpm_min=60,
    bpm_max=180,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


def _load_gt(gt_file):
    raw_gt = pd.read_csv(gt_file, low_memory=False)
    if raw_gt.empty:
        return raw_gt
    raw_gt['start_val'] = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val'] = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    return raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)


def _enrich_formulas(df, anger_coeff=0.6):
    """Compute all f0-f13 + composite_fear in-place (mirrors evaluate_all)."""
    hs_fear = pd.to_numeric(df.get('hs_fear', 0), errors='coerce').fillna(0.0)
    mp_t = pd.to_numeric(df.get('mp_tension', 0), errors='coerce').fillna(0.0)
    hs_anger = pd.to_numeric(df.get('hs_anger', 0), errors='coerce').fillna(0.0)
    hs_arousal = pd.to_numeric(df.get('hs_arousal', 0), errors='coerce').fillna(0.0)
    hs_surprise = pd.to_numeric(df.get('hs_surprise', 0), errors='coerce').fillna(0.0)
    startle = pd.to_numeric(df.get('mp_startle_score', 0), errors='coerce').fillna(0.0)
    startle = (startle / 10.0).clip(0, 1)
    if 'f0' not in df.columns:
        df['f0'] = (0.60 * hs_arousal + 0.40 * mp_t).clip(0, 1)
    if 'f1' not in df.columns:
        df['f1'] = (0.60 * hs_fear + 0.40 * mp_t).clip(0, 1)
    if 'f2' not in df.columns:
        df['f2'] = (0.40 * hs_fear + 0.20 * hs_surprise + 0.40 * mp_t).clip(0, 1)
    if 'f3' not in df.columns:
        df['f3'] = (0.50 * hs_fear + 0.20 * hs_arousal + 0.30 * mp_t).clip(0, 1)
    if 'f4' not in df.columns:
        f0_vals = (0.60 * hs_arousal + 0.40 * mp_t).clip(0, 1)
        df['f4'] = (f0_vals * (hs_anger <= hs_fear).astype(float)).clip(0, 1)
    if 'f5' not in df.columns:
        df['f5'] = (0.50 * hs_fear + 0.30 * mp_t + 0.20 * startle).clip(0, 1)
    if 'f6' not in df.columns:
        df['f6'] = ((0.60 * hs_fear + 0.40 * mp_t) * (1.0 - hs_anger)).clip(0, 1)
    if 'f7' not in df.columns:
        df['f7'] = hs_fear.clip(0, 1)
    if 'f8' not in df.columns:
        df['f8'] = mp_t.clip(0, 1)
    if 'f9' not in df.columns:
        df['f9'] = np.maximum(hs_fear, mp_t).clip(0, 1)
    if 'f10' not in df.columns:
        df['f10'] = np.sqrt((hs_fear * mp_t).clip(0, 1)).clip(0, 1)
    if 'f11' not in df.columns:
        df['f11'] = (hs_fear - anger_coeff * hs_anger).clip(0, 1)
    if 'f12' not in df.columns:
        df['f12'] = ((0.7 * hs_fear + 0.3 * hs_arousal) * (1.0 + mp_t)).clip(0, 1)
    if 'f13' not in df.columns:
        df['f13'] = (hs_fear * (1.0 + mp_t)).clip(0, 1)
    df['composite_fear'] = (hs_fear * (1.0 + mp_t)).clip(0, 1)


# ---------------------------------------------------------------------------
# BPM processing
# ---------------------------------------------------------------------------

def filter_consensus(bpm_results):
    """Extract CONSENSUS rows as a DataFrame."""
    rows = [r for r in bpm_results if r["algorithm"] == "CONSENSUS"]
    if not rows:
        return pd.DataFrame(columns=["t_center", "bpm_smoothed", "snr",
                                     "bpm_plausible", "t_start", "t_end"])
    df = pd.DataFrame(rows)
    return df.sort_values("t_center").reset_index(drop=True)


def filter_algorithm(bpm_results, algo_name):
    """Extract rows for a single algorithm as a DataFrame."""
    rows = [r for r in bpm_results if r["algorithm"] == algo_name.upper()]
    if not rows:
        return pd.DataFrame(columns=["t_center", "bpm_smoothed", "snr",
                                     "bpm_plausible", "t_start", "t_end"])
    df = pd.DataFrame(rows)
    return df.sort_values("t_center").reset_index(drop=True)


def filter_multi_algorithm(bpm_results, algo_names, snr_threshold=2.0):
    """Build a mini-CONSENSUS from a subset of algorithms (SNR-weighted mean)."""
    algo_set = {a.upper() for a in algo_names}
    by_window = {}
    for r in bpm_results:
        if r["algorithm"] not in algo_set:
            continue
        key = round(r["t_center"], 3)
        by_window.setdefault(key, []).append(r)

    rows = []
    for t_center, algo_rows in sorted(by_window.items()):
        eligible = [r for r in algo_rows if r.get("snr", 0) >= snr_threshold]
        if not eligible:
            eligible = algo_rows
        snrs = np.array([r["snr"] for r in eligible])
        bpms = np.array([r["bpm_smoothed"] for r in eligible])
        weights = snrs / snrs.sum() if snrs.sum() > 0 else np.ones(len(snrs)) / len(snrs)
        rows.append({
            "t_center": eligible[0]["t_center"],
            "t_start": eligible[0].get("t_start", 0),
            "t_end": eligible[0].get("t_end", 0),
            "bpm_smoothed": float(np.dot(weights, bpms)),
            "snr": float(np.mean(snrs)),
            "bpm_plausible": 1,
        })
    if not rows:
        return pd.DataFrame(columns=["t_center", "bpm_smoothed", "snr",
                                     "bpm_plausible", "t_start", "t_end"])
    return pd.DataFrame(rows).sort_values("t_center").reset_index(drop=True)


def interpolate_bpm_to_frames(bpm_consensus, frame_timestamps):
    """Linearly interpolate CONSENSUS BPM to per-frame timestamps."""
    if bpm_consensus.empty or len(frame_timestamps) == 0:
        return np.zeros(len(frame_timestamps))
    t_centers = bpm_consensus["t_center"].values
    bpms = bpm_consensus["bpm_smoothed"].values
    return np.interp(frame_timestamps, t_centers, bpms)


def compute_bpm_norm(bpm_at_frame, frame_timestamps, baseline_window_s=60.0):
    """Normalized BPM: fractional increase over rolling median baseline.

    Returns values clipped to [0, 1] — only elevations count.
    """
    n = len(bpm_at_frame)
    if n == 0:
        return np.zeros(0)

    dt = np.median(np.diff(frame_timestamps)) if n > 1 else 0.033
    win_frames = max(int(baseline_window_s / dt), 1)
    baseline = pd.Series(bpm_at_frame).rolling(
        win_frames, center=True, min_periods=max(win_frames // 4, 1)
    ).median().values

    safe_baseline = np.where(baseline > 30, baseline, 70.0)
    bpm_norm = (bpm_at_frame - safe_baseline) / safe_baseline
    return np.clip(bpm_norm, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Phase 2 Extension 1: Face Detection Comparison (FER vs rPPG)
# ---------------------------------------------------------------------------

def compute_face_detection_comparison(loaded_sessions, cache_dir):
    """Compare per-frame face detection rates between FER and rPPG.

    Returns list of dicts with FER MP%, Haar%, rPPG% per session.
    """
    results = []
    for sd in loaded_sessions:
        tag = sd["tag"]
        model_df = sd["model_df"]

        n_fer = len(model_df)
        mp_det = int((pd.to_numeric(model_df.get("mp_face_detected", 0),
                                    errors="coerce").fillna(0) == 1).sum())
        hs_det = int((pd.to_numeric(model_df.get("hs_face_detected", 0),
                                    errors="coerce").fillna(0) == 1).sum())
        any_det = int(((pd.to_numeric(model_df.get("mp_face_detected", 0),
                                      errors="coerce").fillna(0) == 1) |
                       (pd.to_numeric(model_df.get("hs_face_detected", 0),
                                      errors="coerce").fillna(0) == 1)).sum())

        rppg_csv = os.path.join(cache_dir, f"{tag}_rppg_frames.csv")
        rppg_det, n_rppg = 0, 0
        if os.path.exists(rppg_csv):
            rppg_df = pd.read_csv(rppg_csv, comment="#")
            n_rppg = len(rppg_df)
            rppg_det = int((rppg_df["face_detected"] == 1).sum())

        results.append({
            "tag": tag,
            "type": sd["type"],
            "n_fer": n_fer,
            "fer_mp_pct": round(100 * mp_det / max(n_fer, 1), 1),
            "fer_haar_pct": round(100 * hs_det / max(n_fer, 1), 1),
            "fer_any_pct": round(100 * any_det / max(n_fer, 1), 1),
            "n_rppg": n_rppg,
            "rppg_pct": round(100 * rppg_det / max(n_rppg, 1), 1),
            "gap": round(100 * rppg_det / max(n_rppg, 1) - 100 * mp_det / max(n_fer, 1), 1),
        })
    return results


def print_face_detection_comparison(comparison):
    """Print face detection comparison table."""
    print(f"\n  {'Tag':<14} {'Type':<12} {'FER MP%':>8} {'FER Haar%':>10} "
          f"{'FER Any%':>9} {'rPPG det%':>10} {'Gap':>7}")
    for r in comparison:
        print(f"  {r['tag']:<14} {r['type']:<12} {r['fer_mp_pct']:>8.1f} "
              f"{r['fer_haar_pct']:>10.1f} {r['fer_any_pct']:>9.1f} "
              f"{r['rppg_pct']:>10.1f} {r['gap']:>+7.1f}")

    worst = max(comparison, key=lambda x: abs(x["gap"]))
    if abs(worst["gap"]) < 2.0:
        print(f"\n  → FER and rPPG face detection rates are virtually identical.")
        print(f"    Detection failures are video-level (face not visible), not detector-level.")
    else:
        print(f"\n  → {worst['tag']} shows the largest gap ({worst['gap']:+.1f}%).")
        print(f"    FER Haar fallback recovers {worst['fer_any_pct'] - worst['fer_mp_pct']:.1f}% "
              f"additional frames over MP alone.")
        if worst["type"] == "compilation":
            print(f"    As a compilation video, scene cuts cause face tracking loss — "
                  f"this is video-level, not detector-level.")


# ---------------------------------------------------------------------------
# GT alignment — per-event BPM statistics
# ---------------------------------------------------------------------------

def compute_event_bpm_stats(bpm_consensus, gt_df,
                            pre_window_s=30.0, pre_gap_s=5.0):
    """For each GT fear event, compute BPM during vs before.

    Returns list of dicts with: event_idx, start_s, end_s, label,
    bpm_during, bpm_pre, bpm_delta, bpm_post, n_windows_during, n_windows_pre.
    """
    if bpm_consensus.empty or gt_df.empty:
        return []

    t_centers = bpm_consensus["t_center"].values
    bpms = bpm_consensus["bpm_smoothed"].values
    results = []

    for idx, row in gt_df.iterrows():
        gs, ge = row["start_val"], row["end_val"]

        mask_during = (t_centers >= gs) & (t_centers <= ge)
        mask_pre = ((t_centers >= gs - pre_window_s - pre_gap_s) &
                    (t_centers < gs - pre_gap_s))
        mask_post = (t_centers > ge) & (t_centers <= ge + 15.0)

        bpm_during = float(np.mean(bpms[mask_during])) if mask_during.any() else np.nan
        bpm_pre = float(np.mean(bpms[mask_pre])) if mask_pre.any() else np.nan
        bpm_post = float(np.mean(bpms[mask_post])) if mask_post.any() else np.nan

        bpm_delta = bpm_during - bpm_pre if not (np.isnan(bpm_during) or np.isnan(bpm_pre)) else np.nan

        results.append({
            "event_idx": idx,
            "start_s": gs,
            "end_s": ge,
            "label": row.get("label", "fear"),
            "bpm_during": round(bpm_during, 1) if not np.isnan(bpm_during) else None,
            "bpm_pre": round(bpm_pre, 1) if not np.isnan(bpm_pre) else None,
            "bpm_delta": round(bpm_delta, 1) if not np.isnan(bpm_delta) else None,
            "bpm_post": round(bpm_post, 1) if not np.isnan(bpm_post) else None,
            "n_windows_during": int(mask_during.sum()),
            "n_windows_pre": int(mask_pre.sum()),
        })

    return results


def sample_nonfear_intervals(bpm_consensus, gt_df, session_duration_s,
                             n_samples=50, seed=42):
    """Sample random non-fear intervals and compute their mean BPM."""
    if bpm_consensus.empty:
        return []

    rng = np.random.RandomState(seed)
    t_centers = bpm_consensus["t_center"].values
    bpms = bpm_consensus["bpm_smoothed"].values

    fear_spans = list(zip(gt_df["start_val"].values, gt_df["end_val"].values)) if not gt_df.empty else []
    durations = [e - s for s, e in fear_spans] if fear_spans else [5.0]
    mean_dur = float(np.mean(durations))

    results = []
    attempts = 0
    while len(results) < n_samples and attempts < n_samples * 10:
        attempts += 1
        start = rng.uniform(15.0, max(session_duration_s - mean_dur - 5.0, 20.0))
        end = start + mean_dur

        overlaps_fear = any(not (end < fs or start > fe) for fs, fe in fear_spans)
        if overlaps_fear:
            continue

        mask = (t_centers >= start) & (t_centers <= end)
        if not mask.any():
            continue

        results.append({
            "start_s": round(start, 1),
            "end_s": round(end, 1),
            "bpm_during": round(float(np.mean(bpms[mask])), 1),
            "n_windows": int(mask.sum()),
        })

    return results


# ---------------------------------------------------------------------------
# Phase 2 Extension 2: Post-Event Peak Analysis (autonomic delay)
# ---------------------------------------------------------------------------

def compute_event_peak_stats(bpm_consensus, gt_df,
                             pre_window_s=30.0, pre_gap_s=5.0,
                             post_peak_window_s=20.0):
    """Extended event stats with post-event mean BPM analysis.

    Uses MEAN (not MAX) in the post-event window to avoid cherry-picking.
    Adds isolation metadata so callers can filter to uncontaminated events.
    """
    if bpm_consensus.empty or gt_df.empty:
        return []

    t_centers = bpm_consensus["t_center"].values
    bpms = bpm_consensus["bpm_smoothed"].values

    gt_starts = gt_df["start_val"].values
    gt_ends = gt_df["end_val"].values
    results = []

    for i, (gs, ge) in enumerate(zip(gt_starts, gt_ends)):
        next_gap = float(gt_starts[i + 1] - ge) if i + 1 < len(gt_starts) else np.inf

        mask_pre = ((t_centers >= gs - pre_window_s - pre_gap_s) &
                    (t_centers < gs - pre_gap_s))
        mask_post = ((t_centers > ge) &
                     (t_centers <= ge + post_peak_window_s))

        bpm_pre = float(np.mean(bpms[mask_pre])) if mask_pre.any() else np.nan
        bpm_mean_post = float(np.mean(bpms[mask_post])) if mask_post.any() else np.nan

        bpm_delta = np.nan
        if not (np.isnan(bpm_mean_post) or np.isnan(bpm_pre)):
            bpm_delta = bpm_mean_post - bpm_pre

        results.append({
            "event_idx": i,
            "start_s": gs,
            "end_s": ge,
            "bpm_pre": round(bpm_pre, 1) if not np.isnan(bpm_pre) else None,
            "bpm_mean_post": round(bpm_mean_post, 1) if not np.isnan(bpm_mean_post) else None,
            "bpm_post_delta": round(bpm_delta, 1) if not np.isnan(bpm_delta) else None,
            "next_event_gap": round(next_gap, 1),
            "is_isolated": next_gap > 30.0,
        })

    return results


def _ttest_1samp_summary(values):
    """One-sample t-test on *values* vs 0. Returns dict with stats or Nones."""
    from scipy import stats as sp_stats
    if len(values) < 3:
        return {"t": None, "p": None, "d": None, "ci_95": None,
                "mean": None, "std": None, "n": len(values)}
    arr = np.asarray(values, dtype=float)
    t_stat, p_val = sp_stats.ttest_1samp(arr, 0.0)
    se = np.std(arr, ddof=1) / np.sqrt(len(arr))
    sd = float(np.std(arr, ddof=1))
    d = float(np.mean(arr)) / sd if sd > 0 else 0.0
    return {
        "t": round(float(t_stat), 3),
        "p": round(float(p_val), 4),
        "d": round(d, 3),
        "ci_95": (round(float(np.mean(arr) - 1.96 * se), 2),
                  round(float(np.mean(arr) + 1.96 * se), 2)),
        "mean": round(float(np.mean(arr)), 2),
        "std": round(sd, 2),
        "n": len(arr),
    }


def compute_peak_aggregate_stats(peak_events, original_fear_stats):
    """Aggregate post-event MEAN BPM delta — ALL events + ISOLATED subset."""
    all_deltas = [e["bpm_post_delta"] for e in peak_events
                  if e["bpm_post_delta"] is not None]
    iso_deltas = [e["bpm_post_delta"] for e in peak_events
                  if e["bpm_post_delta"] is not None and e["is_isolated"]]
    orig_deltas = [s["bpm_delta"] for s in original_fear_stats
                   if s["bpm_delta"] is not None]

    n_total = len(peak_events)
    n_contaminated = sum(1 for e in peak_events if e["next_event_gap"] < 20.0)

    return {
        "n_total": n_total,
        "n_contaminated": n_contaminated,
        "contamination_pct": round(100 * n_contaminated / max(n_total, 1), 1),
        "all": _ttest_1samp_summary(all_deltas),
        "isolated": _ttest_1samp_summary(iso_deltas),
        "orig_mean_delta": round(float(np.mean(orig_deltas)), 2) if orig_deltas else None,
    }


def print_peak_analysis(peak_stats, mode):
    """Print post-event MEAN BPM analysis with contamination warning."""
    ps = peak_stats
    print(f"\n  WARNING: {ps['contamination_pct']}% of events ({ps['n_contaminated']}/{ps['n_total']})"
          f" have next fear event within 20s.")
    print(f"  Post-event windows overlap with adjacent fear events.")
    print(f"  Using MEAN post-event BPM (not MAX) to mitigate cherry-picking.\n")

    def _print_subgroup(label, stats):
        if stats["n"] == 0:
            print(f"    {label}: no events")
            return
        print(f"    {label} (n={stats['n']}):")
        print(f"      mean Δ = {stats['mean']} BPM,  std = {stats['std']}")
        if stats["t"] is not None:
            sig = "NULL RESULT" if stats["p"] > 0.05 else "SIGNIFICANT"
            print(f"      t = {stats['t']},  p = {stats['p']},  Cohen's d = {stats['d']}"
                  f"  [{sig}]")
            print(f"      95% CI = ({stats['ci_95'][0]}, {stats['ci_95'][1]})")
        else:
            print(f"      Not enough events for t-test")

    print(f"  Post-event mean BPM delta (H0: delta=0):")
    _print_subgroup("All events", ps["all"])
    _print_subgroup("Isolated only (gap>30s)", ps["isolated"])

    print(f"\n  Original in-event delta: mean Δ = {ps['orig_mean_delta']} BPM")
    print(f"  → No evidence of autonomic-delay HR response in post-event windows.")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def compute_stats(fear_stats, nonfear_stats):
    """Welch's t-test + Cohen's d for BPM fear vs non-fear."""
    from scipy import stats as sp_stats

    fear_bpms = [s["bpm_during"] for s in fear_stats if s["bpm_during"] is not None]
    nonfear_bpms = [s["bpm_during"] for s in nonfear_stats]

    result = {
        "n_fear": len(fear_bpms),
        "n_nonfear": len(nonfear_bpms),
        "mean_fear_bpm": round(float(np.mean(fear_bpms)), 1) if fear_bpms else None,
        "mean_nonfear_bpm": round(float(np.mean(nonfear_bpms)), 1) if nonfear_bpms else None,
        "std_fear_bpm": round(float(np.std(fear_bpms, ddof=1)), 1) if len(fear_bpms) > 1 else None,
        "std_nonfear_bpm": round(float(np.std(nonfear_bpms, ddof=1)), 1) if len(nonfear_bpms) > 1 else None,
    }

    if len(fear_bpms) >= 3 and len(nonfear_bpms) >= 3:
        t_stat, p_val = sp_stats.ttest_ind(fear_bpms, nonfear_bpms, equal_var=False)
        pooled_std = np.sqrt((np.var(fear_bpms, ddof=1) + np.var(nonfear_bpms, ddof=1)) / 2)
        cohens_d = (np.mean(fear_bpms) - np.mean(nonfear_bpms)) / pooled_std if pooled_std > 0 else 0
        result["t_stat"] = round(float(t_stat), 3)
        result["p_value"] = round(float(p_val), 4)
        result["cohens_d"] = round(float(cohens_d), 3)
    else:
        result["t_stat"] = None
        result["p_value"] = None
        result["cohens_d"] = None

    return result


def compute_delta_stats(fear_stats):
    """One-sample t-test: is mean bpm_delta > 0?"""
    from scipy import stats as sp_stats

    deltas = [s["bpm_delta"] for s in fear_stats if s["bpm_delta"] is not None]
    result = {
        "n_events": len(deltas),
        "mean_delta": round(float(np.mean(deltas)), 2) if deltas else None,
        "std_delta": round(float(np.std(deltas, ddof=1)), 2) if len(deltas) > 1 else None,
    }

    if len(deltas) >= 3:
        t_stat, p_val = sp_stats.ttest_1samp(deltas, 0.0)
        se = np.std(deltas, ddof=1) / np.sqrt(len(deltas))
        ci_lo = np.mean(deltas) - 1.96 * se
        ci_hi = np.mean(deltas) + 1.96 * se
        result["t_stat"] = round(float(t_stat), 3)
        result["p_value"] = round(float(p_val), 4)
        result["ci_95"] = (round(float(ci_lo), 2), round(float(ci_hi), 2))
    else:
        result["t_stat"] = None
        result["p_value"] = None
        result["ci_95"] = None

    return result


def compute_fer_bpm_correlation(model_df, bpm_consensus, fer_col="f12",
                                resample_s=3.0):
    """Pearson + Spearman correlation between FER signal and BPM."""
    from scipy import stats as sp_stats

    if bpm_consensus.empty or fer_col not in model_df.columns:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None, "n_points": 0}

    ts_model = pd.to_numeric(model_df["timestamp"], errors="coerce").values
    fer_vals = pd.to_numeric(model_df[fer_col], errors="coerce").fillna(0.0).values

    t_min = max(ts_model.min(), bpm_consensus["t_center"].min())
    t_max = min(ts_model.max(), bpm_consensus["t_center"].max())
    grid = np.arange(t_min, t_max, resample_s)

    if len(grid) < 5:
        return {"pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None, "n_points": 0}

    fer_resampled = np.interp(grid, ts_model, fer_vals)
    bpm_resampled = np.interp(grid, bpm_consensus["t_center"].values,
                              bpm_consensus["bpm_smoothed"].values)

    pr, pp = sp_stats.pearsonr(fer_resampled, bpm_resampled)
    sr, sp_val = sp_stats.spearmanr(fer_resampled, bpm_resampled)

    return {
        "pearson_r": round(float(pr), 4),
        "pearson_p": round(float(pp), 4),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp_val), 4),
        "n_points": len(grid),
    }


# ---------------------------------------------------------------------------
# Phase 2 Extension 3: Multi-Window BPM Comparison
# ---------------------------------------------------------------------------

MULTI_WINDOW_CONFIGS = [
    AnalysisConfig(algorithm="all", window_s=30.0, step_s=5.0, bpm_min=60, bpm_max=180),
    AnalysisConfig(algorithm="all", window_s=15.0, step_s=3.0, bpm_min=60, bpm_max=180),
    AnalysisConfig(algorithm="all", window_s=10.0, step_s=2.0, bpm_min=60, bpm_max=180),
    AnalysisConfig(algorithm="all", window_s=5.0,  step_s=1.0, bpm_min=60, bpm_max=180),
]


def run_multi_window_comparison(loaded_sessions, analyzer):
    """Run BPM analysis at multiple window sizes and compare results.

    The 15s config reuses already-computed data from loaded_sessions.
    10s and 5s re-run analyzer.analyze() per session (fast — reads cached frames).
    """
    summary = []

    for cfg in MULTI_WINDOW_CONFIGS:
        all_fear_ev, all_nonfear_ev = [], []

        for sd in loaded_sessions:
            if cfg.window_s == 15.0:
                bpm_cons = sd["bpm_consensus"]
            else:
                bpm_results = analyzer.analyze(Path(sd["frames_csv"]), cfg)
                bpm_cons = filter_consensus(bpm_results)

            gt_df = sd["gt_df"]
            frame_ts = pd.to_numeric(sd["model_df"]["timestamp"],
                                     errors="coerce").fillna(0.0).values
            session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0

            fear_ev = compute_event_bpm_stats(bpm_cons, gt_df)
            nonfear_ev = sample_nonfear_intervals(bpm_cons, gt_df, session_dur)

            for ev in fear_ev:
                ev["session"] = sd["tag"]
            all_fear_ev.extend(fear_ev)
            all_nonfear_ev.extend(nonfear_ev)

        bpm_stats = compute_stats(all_fear_ev, all_nonfear_ev)
        delta_stats = compute_delta_stats(all_fear_ev)

        n_plaus = sum(1 for e in all_fear_ev
                      if e["bpm_during"] is not None)

        summary.append({
            "window_s": cfg.window_s,
            "step_s": cfg.step_s,
            "p_fear_vs_nf": bpm_stats.get("p_value"),
            "d_fear_vs_nf": bpm_stats.get("cohens_d"),
            "p_delta": delta_stats.get("p_value"),
            "mean_delta": delta_stats.get("mean_delta"),
            "n_fear": bpm_stats.get("n_fear"),
            "n_events_with_bpm": n_plaus,
        })

    return summary


def print_multi_window_comparison(summary):
    """Print multi-window comparison table."""
    print(f"\n  {'Window':>7} {'Step':>5}  {'p_fear':>8} {'d_fear':>8} "
          f"{'p_delta':>8} {'mean_Δ':>8} {'n_fear':>7} {'w/BPM':>6}")
    for r in summary:
        p_f = f"{r['p_fear_vs_nf']:.4f}" if r["p_fear_vs_nf"] is not None else "N/A"
        d_f = f"{r['d_fear_vs_nf']:.3f}" if r["d_fear_vs_nf"] is not None else "N/A"
        p_d = f"{r['p_delta']:.4f}" if r["p_delta"] is not None else "N/A"
        m_d = f"{r['mean_delta']:+.2f}" if r["mean_delta"] is not None else "N/A"
        print(f"  {r['window_s']:>5.1f}s {r['step_s']:>4.1f}s  {p_f:>8} "
              f"{d_f:>8} {p_d:>8} {m_d:>8} {r['n_fear']:>7} "
              f"{r['n_events_with_bpm']:>6}")

    best = min(summary, key=lambda x: x["p_delta"] if x["p_delta"] is not None else 999)
    print(f"\n  → Lowest delta p-value: {best['window_s']}s window "
          f"(p={best['p_delta']})")
    if best["window_s"] != 15.0:
        print(f"    Shorter window may improve temporal alignment, but check "
              f"BPM accuracy at {best['window_s']}s.")
    print(f"\n  Note: 5s window = ~150 frames at 30fps = ~12 BPM frequency resolution.")
    print(f"  CONSENSUS filtering handles noise, but precision degrades at short windows.")


# ---------------------------------------------------------------------------
# Phase 2 Extension 4: Per-Subject Z-Score Normalization
# ---------------------------------------------------------------------------

def compute_zscore_analysis(loaded_sessions):
    """Per-subject BPM z-score normalization.

    Groups sessions by subject, computes per-subject BPM mean/std from all
    windows (fear + non-fear), then z-scores each event's BPM and tests
    fear z-scores vs non-fear z-scores.
    """
    from scipy import stats as sp_stats

    subj_bpms = {}
    for sd in loaded_sessions:
        subj = sd["tag"].split("_")[0]
        if subj not in subj_bpms:
            subj_bpms[subj] = {"tags": [], "all_bpms": []}
        subj_bpms[subj]["tags"].append(sd["tag"])
        if not sd["bpm_consensus"].empty:
            subj_bpms[subj]["all_bpms"].extend(
                sd["bpm_consensus"]["bpm_smoothed"].values.tolist()
            )

    subj_stats = {}
    for subj, data in subj_bpms.items():
        bpms = data["all_bpms"]
        if len(bpms) > 1:
            subj_stats[subj] = {
                "mean": float(np.mean(bpms)),
                "std": max(float(np.std(bpms, ddof=1)), 1.0),
                "n_windows": len(bpms),
                "tags": data["tags"],
            }

    fear_z, nonfear_z = [], []
    for sd in loaded_sessions:
        subj = sd["tag"].split("_")[0]
        if subj not in subj_stats:
            continue
        mu, sigma = subj_stats[subj]["mean"], subj_stats[subj]["std"]

        for ev in sd["fear_stats"]:
            if ev["bpm_during"] is not None:
                fear_z.append((ev["bpm_during"] - mu) / sigma)

        for nf in sd["nonfear_stats"]:
            nonfear_z.append((nf["bpm_during"] - mu) / sigma)

    result = {
        "subject_stats": subj_stats,
        "n_fear_z": len(fear_z),
        "n_nonfear_z": len(nonfear_z),
        "mean_fear_z": round(float(np.mean(fear_z)), 3) if fear_z else None,
        "mean_nonfear_z": round(float(np.mean(nonfear_z)), 3) if nonfear_z else None,
        "std_fear_z": round(float(np.std(fear_z, ddof=1)), 3) if len(fear_z) > 1 else None,
        "std_nonfear_z": round(float(np.std(nonfear_z, ddof=1)), 3) if len(nonfear_z) > 1 else None,
    }

    if len(fear_z) >= 3 and len(nonfear_z) >= 3:
        t_stat, p_val = sp_stats.ttest_ind(fear_z, nonfear_z, equal_var=False)
        pooled_std = np.sqrt((np.var(fear_z, ddof=1) + np.var(nonfear_z, ddof=1)) / 2)
        cohens_d = (np.mean(fear_z) - np.mean(nonfear_z)) / pooled_std if pooled_std > 0 else 0
        result["t_stat"] = round(float(t_stat), 3)
        result["p_value"] = round(float(p_val), 4)
        result["cohens_d"] = round(float(cohens_d), 3)
    else:
        result["t_stat"] = None
        result["p_value"] = None
        result["cohens_d"] = None

    return result


def print_zscore_analysis(zs):
    """Print per-subject z-score analysis."""
    print(f"\n  Per-Subject BPM baseline:")
    print(f"  {'Subject':<10} {'Sessions':<20} {'n_windows':>10} "
          f"{'mean BPM':>10} {'std BPM':>10}")
    for subj, st in sorted(zs["subject_stats"].items()):
        tags = ", ".join(st["tags"])
        print(f"  {subj:<10} {tags:<20} {st['n_windows']:>10} "
              f"{st['mean']:>10.1f} {st['std']:>10.1f}")

    print(f"\n  Z-Score Fear vs Non-Fear (Welch's t-test):")
    print(f"    Fear:     n={zs['n_fear_z']}, mean_z={zs['mean_fear_z']}, "
          f"std_z={zs['std_fear_z']}")
    print(f"    Non-fear: n={zs['n_nonfear_z']}, mean_z={zs['mean_nonfear_z']}, "
          f"std_z={zs['std_nonfear_z']}")
    if zs["t_stat"] is not None:
        print(f"    t = {zs['t_stat']},  p = {zs['p_value']},  Cohen's d = {zs['cohens_d']}")
    sig = "significant" if zs.get("p_value") and zs["p_value"] < 0.05 else "not significant"
    print(f"\n  → Within-subject normalization: effect is {sig}.")


# ---------------------------------------------------------------------------
# Phase 2 Extension 5: Approximate BPM Variability
# ---------------------------------------------------------------------------

def compute_contiguous_rmssd(bpm_array, mask):
    """Compute RMSSD/SDNN only from within-contiguous-block diffs.

    Splits *bpm_array* into contiguous runs where *mask* is True,
    computes np.diff inside each run, and pools the diffs.  Diffs that
    would span a gap between non-adjacent timepoints are excluded.
    """
    indices = np.where(mask)[0]
    if len(indices) < 2:
        return {"rmssd": np.nan, "sdnn": np.nan, "n": int(mask.sum()), "n_diffs": 0}

    splits = np.where(np.diff(indices) > 1)[0] + 1
    segments = np.split(indices, splits)

    all_diffs = []
    for seg in segments:
        if len(seg) >= 2:
            all_diffs.append(np.diff(bpm_array[seg]))

    values = bpm_array[indices]
    sdnn = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan

    if not all_diffs:
        return {"rmssd": np.nan, "sdnn": round(sdnn, 2), "n": len(indices), "n_diffs": 0}

    pooled = np.concatenate(all_diffs)
    rmssd = float(np.sqrt(np.mean(pooled ** 2)))
    return {"rmssd": round(rmssd, 2), "sdnn": round(sdnn, 2),
            "n": len(indices), "n_diffs": len(pooled)}


def compute_variability_comparison(loaded_sessions, n_bootstrap=500):
    """Compare BPM variability during fear vs non-fear per session.

    Uses contiguous-block RMSSD (no cross-boundary diffs) and adds a
    duration-matched bootstrap to control for fear/non-fear length asymmetry.
    """
    from scipy import stats as sp_stats
    rng = np.random.default_rng(42)

    session_results = []
    for sd in loaded_sessions:
        bpm_cons = sd["bpm_consensus"]
        if bpm_cons.empty:
            continue
        t_centers = bpm_cons["t_center"].values
        bpms = bpm_cons["bpm_smoothed"].values
        gt_df = sd["gt_df"]

        fear_mask = np.zeros(len(t_centers), dtype=bool)
        if not gt_df.empty:
            for _, row in gt_df.iterrows():
                fear_mask |= ((t_centers >= row["start_val"] - 5) &
                              (t_centers <= row["end_val"] + 5))

        fear_var = compute_contiguous_rmssd(bpms, fear_mask)
        nonfear_var = compute_contiguous_rmssd(bpms, ~fear_mask)

        target_n = min(fear_var["n"], nonfear_var["n"])
        boot_rmssd = []
        if target_n >= 4 and nonfear_var["n"] >= target_n:
            nf_indices = np.where(~fear_mask)[0]
            for _ in range(n_bootstrap):
                sample_idx = rng.choice(nf_indices, size=target_n, replace=False)
                sample_idx.sort()
                sample_mask = np.zeros(len(bpms), dtype=bool)
                sample_mask[sample_idx] = True
                sv = compute_contiguous_rmssd(bpms, sample_mask)
                if not np.isnan(sv["rmssd"]):
                    boot_rmssd.append(sv["rmssd"])

        boot_p = np.nan
        if boot_rmssd and not np.isnan(fear_var["rmssd"]):
            boot_p = float(np.mean(np.array(boot_rmssd) <= fear_var["rmssd"]))

        boot_degenerate = (target_n >= nonfear_var["n"])

        session_results.append({
            "tag": sd["tag"],
            "fear_rmssd": fear_var["rmssd"],
            "fear_sdnn": fear_var["sdnn"],
            "fear_n": fear_var["n"],
            "fear_n_diffs": fear_var["n_diffs"],
            "nonfear_rmssd": nonfear_var["rmssd"],
            "nonfear_sdnn": nonfear_var["sdnn"],
            "nonfear_n": nonfear_var["n"],
            "nonfear_n_diffs": nonfear_var["n_diffs"],
            "boot_mean_rmssd": round(float(np.mean(boot_rmssd)), 2) if boot_rmssd else np.nan,
            "boot_std_rmssd": round(float(np.std(boot_rmssd)), 2) if boot_rmssd else np.nan,
            "boot_p": round(boot_p, 4) if not np.isnan(boot_p) else np.nan,
            "boot_degenerate": boot_degenerate,
            "rmssd_diff": round(fear_var["rmssd"] - nonfear_var["rmssd"], 2)
                          if not (np.isnan(fear_var["rmssd"]) or np.isnan(nonfear_var["rmssd"]))
                          else np.nan,
        })

    result = {"sessions": session_results, "n_bootstrap": n_bootstrap}

    fear_rmssd = [s["fear_rmssd"] for s in session_results if not np.isnan(s["fear_rmssd"])]
    nonfear_rmssd = [s["nonfear_rmssd"] for s in session_results if not np.isnan(s["nonfear_rmssd"])]
    rmssd_diffs = [s["rmssd_diff"] for s in session_results if not np.isnan(s.get("rmssd_diff", np.nan))]

    if len(fear_rmssd) >= 3 and len(fear_rmssd) == len(nonfear_rmssd):
        t_r, p_r = sp_stats.ttest_rel(fear_rmssd, nonfear_rmssd)
        result["paired_t"] = round(float(t_r), 3)
        result["paired_p"] = round(float(p_r), 4)
        mean_diff = float(np.mean(rmssd_diffs))
        sd_diff = float(np.std(rmssd_diffs, ddof=1))
        result["cohens_dz"] = round(mean_diff / sd_diff, 3) if sd_diff > 0 else 0.0
    else:
        result["paired_t"] = None
        result["paired_p"] = None
        result["cohens_dz"] = None

    result["mean_fear_rmssd"] = round(float(np.mean(fear_rmssd)), 2) if fear_rmssd else None
    result["mean_nonfear_rmssd"] = round(float(np.mean(nonfear_rmssd)), 2) if nonfear_rmssd else None

    return result


def print_variability_comparison(vc):
    """Print contiguous-block RMSSD comparison with bootstrap."""
    print(f"\n  Caveats:")
    print(f"    - Approximate BPM variability from 3s-resolution timeseries,")
    print(f"      NOT true beat-to-beat HRV (requires RR intervals)")
    print(f"    - RMSSD computed within contiguous blocks only (no cross-gap diffs)")
    print(f"    - 80% BPM window overlap reduces all RMSSD values equally")
    print(f"    - n={len(vc['sessions'])} sessions — low statistical power\n")

    print(f"  {'Tag':<14} {'Fear RMSSD':>11} {'NF RMSSD':>9} {'Diff':>7}"
          f"  {'Fear n':>7} {'NF n':>5}  {'Boot p':>7}")
    for s in vc["sessions"]:
        bp = f"{s['boot_p']:.4f}" if not np.isnan(s.get("boot_p", np.nan)) else "  N/A"
        if s.get("boot_degenerate"):
            bp += "*"
        diff = f"{s['rmssd_diff']:+.2f}" if not np.isnan(s.get("rmssd_diff", np.nan)) else " N/A"
        print(f"  {s['tag']:<14} {s['fear_rmssd']:>11.2f} {s['nonfear_rmssd']:>9.2f} {diff:>7}"
              f"  {s['fear_n']:>7} {s['nonfear_n']:>5}  {bp:>7}")

    if vc.get("paired_t") is not None:
        sig = "SIGNIFICANT" if vc["paired_p"] < 0.05 else "NOT SIGNIFICANT"
        print(f"\n  Paired t-test ({len(vc['sessions'])} sessions):")
        print(f"    RMSSD: fear={vc['mean_fear_rmssd']}, nonfear={vc['mean_nonfear_rmssd']}")
        print(f"    t = {vc['paired_t']},  p = {vc['paired_p']},  Cohen's dz = {vc['cohens_dz']}"
              f"  [{sig}]")
    else:
        print(f"\n  Not enough sessions for paired t-test.")

    boot_ps = [s["boot_p"] for s in vc["sessions"] if not np.isnan(s.get("boot_p", np.nan))]
    n_degen = sum(1 for s in vc["sessions"] if s.get("boot_degenerate"))
    if boot_ps:
        print(f"\n  Duration-matched bootstrap ({vc['n_bootstrap']} samples per session):")
        print(f"    Per-session p-values: {', '.join(f'{p:.4f}' for p in boot_ps)}")
        valid_ps = [s["boot_p"] for s in vc["sessions"]
                    if not np.isnan(s.get("boot_p", np.nan)) and not s.get("boot_degenerate")]
        if valid_ps:
            all_below = all(p < 0.05 for p in valid_ps)
            print(f"    Non-degenerate sessions (fear_n <= nonfear_n) fear < bootstrap: "
                  f"{'YES' if all_below else 'NO'}")
        if n_degen:
            print(f"    * = degenerate bootstrap (fear_n > nonfear_n, all non-fear sampled every time)")


# ---------------------------------------------------------------------------
# Phase 2 Extension 6: Per-Algorithm Fear Correlation
# ---------------------------------------------------------------------------

ALGO_CONFIGS = [
    ("POS", ["POS"]),
    ("POS+CHROM", ["POS", "CHROM"]),
    ("CONSENSUS", None),
]


def run_per_algorithm_evaluation(loaded_sessions, analyzer):
    """Run fear correlation separately for POS, POS+CHROM, and CONSENSUS."""
    from scipy import stats as sp_stats

    print(f"\n  {'Config':<14}  {'n_fear':>7}  {'n_nf':>6}  "
          f"{'Fear BPM':>9}  {'NF BPM':>7}  {'t':>7}  {'p':>8}  {'d':>6}  "
          f"{'mean Δ':>7}  {'p_delta':>8}")

    for config_name, algo_names in ALGO_CONFIGS:
        all_fear_ev, all_nonfear_ev = [], []

        for sd in loaded_sessions:
            if algo_names is None:
                bpm_df = sd["bpm_consensus"]
            elif len(algo_names) == 1:
                bpm_df = filter_algorithm(sd["bpm_results"], algo_names[0])
            else:
                bpm_df = filter_multi_algorithm(sd["bpm_results"], algo_names)

            gt_df = sd["gt_df"]
            frame_ts = pd.to_numeric(sd["model_df"]["timestamp"],
                                     errors="coerce").fillna(0.0).values
            session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0

            fear_ev = compute_event_bpm_stats(bpm_df, gt_df)
            nonfear_ev = sample_nonfear_intervals(bpm_df, gt_df, session_dur)

            for ev in fear_ev:
                ev["session"] = sd["tag"]
            all_fear_ev.extend(fear_ev)
            all_nonfear_ev.extend(nonfear_ev)

        bpm_stats = compute_stats(all_fear_ev, all_nonfear_ev)
        delta_stats = compute_delta_stats(all_fear_ev)

        t_str = f"{bpm_stats['t_stat']:.3f}" if bpm_stats.get("t_stat") is not None else "N/A"
        p_str = f"{bpm_stats['p_value']:.4f}" if bpm_stats.get("p_value") is not None else "N/A"
        d_str = f"{bpm_stats['cohens_d']:.3f}" if bpm_stats.get("cohens_d") is not None else "N/A"
        md_str = f"{delta_stats['mean_delta']:+.2f}" if delta_stats.get("mean_delta") is not None else "N/A"
        pd_str = f"{delta_stats['p_value']:.4f}" if delta_stats.get("p_value") is not None else "N/A"

        print(f"  {config_name:<14}  {bpm_stats.get('n_fear', 0):>7}  "
              f"{bpm_stats.get('n_nonfear', 0):>6}  "
              f"{bpm_stats.get('mean_fear_bpm', 'N/A'):>9}  "
              f"{bpm_stats.get('mean_nonfear_bpm', 'N/A'):>7}  "
              f"{t_str:>7}  {p_str:>8}  {d_str:>6}  "
              f"{md_str:>7}  {pd_str:>8}")

    print()


# ---------------------------------------------------------------------------
# Phase 2 Extension 7: Temporal Shift of Fear Windows
# ---------------------------------------------------------------------------

SHIFT_VALUES = [0, 3, 5, 8, 10, 15]


def compute_shifted_event_bpm_stats(bpm_df, gt_df, shift_s,
                                     pre_window_s=30.0, pre_gap_s=5.0,
                                     session_dur=None):
    """Shift GT fear windows forward by shift_s and compute BPM during vs pre."""
    if bpm_df.empty or gt_df.empty:
        return [], 0, 0

    t_centers = bpm_df["t_center"].values
    bpms = bpm_df["bpm_smoothed"].values
    gt_starts = gt_df["start_val"].values
    gt_ends = gt_df["end_val"].values

    results = []
    n_overlap = 0
    n_excluded = 0

    for i, (gs, ge) in enumerate(zip(gt_starts, gt_ends)):
        shifted_start = gs + shift_s
        shifted_end = ge + shift_s

        if session_dur is not None and shifted_end > session_dur:
            n_excluded += 1
            continue

        overlaps_next = False
        for j, (gs2, ge2) in enumerate(zip(gt_starts, gt_ends)):
            if j == i:
                continue
            if not (shifted_end < gs2 or shifted_start > ge2):
                overlaps_next = True
                break
        if overlaps_next:
            n_overlap += 1

        mask_during = (t_centers >= shifted_start) & (t_centers <= shifted_end)
        mask_pre = ((t_centers >= gs - pre_window_s - pre_gap_s) &
                    (t_centers < gs - pre_gap_s))

        bpm_during = float(np.mean(bpms[mask_during])) if mask_during.any() else np.nan
        bpm_pre = float(np.mean(bpms[mask_pre])) if mask_pre.any() else np.nan
        bpm_delta = bpm_during - bpm_pre if not (np.isnan(bpm_during) or np.isnan(bpm_pre)) else np.nan

        results.append({
            "bpm_during": bpm_during,
            "bpm_pre": bpm_pre,
            "bpm_delta": bpm_delta,
            "overlaps_other": overlaps_next,
        })

    return results, n_overlap, n_excluded


def run_temporal_shift_sweep(loaded_sessions):
    """Sweep temporal shifts and report fear vs baseline BPM."""
    from scipy import stats as sp_stats

    print(f"\n  {'Shift':>6}  {'n':>4}  {'n_overlap':>10}  {'overlap%':>9}  "
          f"{'mean Δ':>7}  {'t':>7}  {'p':>8}  {'d':>6}  "
          f"{'iso_n':>6}  {'iso_p':>8}  {'iso_d':>6}")

    for shift_s in SHIFT_VALUES:
        all_events = []
        total_overlap = 0
        total_excluded = 0

        for sd in loaded_sessions:
            frame_ts = pd.to_numeric(sd["model_df"]["timestamp"],
                                     errors="coerce").fillna(0.0).values
            session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else None
            events, n_ov, n_ex = compute_shifted_event_bpm_stats(
                sd["bpm_consensus"], sd["gt_df"], shift_s,
                session_dur=session_dur)
            all_events.extend(events)
            total_overlap += n_ov
            total_excluded += n_ex

        deltas = [e["bpm_delta"] for e in all_events
                  if not np.isnan(e.get("bpm_delta", np.nan))]
        n = len(deltas)
        overlap_pct = 100 * total_overlap / max(len(all_events), 1)

        if n >= 3:
            t_val, p_val = sp_stats.ttest_1samp(deltas, 0)
            mean_d = float(np.mean(deltas))
            std_d = float(np.std(deltas, ddof=1))
            cohens_d = mean_d / std_d if std_d > 0 else 0.0
        else:
            t_val, p_val, mean_d, cohens_d = np.nan, np.nan, np.nan, np.nan

        iso_deltas = [e["bpm_delta"] for e in all_events
                      if not np.isnan(e.get("bpm_delta", np.nan))
                      and not e.get("overlaps_other", False)]
        iso_n = len(iso_deltas)
        if iso_n >= 3:
            iso_t, iso_p = sp_stats.ttest_1samp(iso_deltas, 0)
            iso_std = float(np.std(iso_deltas, ddof=1))
            iso_d = float(np.mean(iso_deltas)) / iso_std if iso_std > 0 else 0.0
        else:
            iso_p, iso_d = np.nan, np.nan

        t_str = f"{t_val:.3f}" if not np.isnan(t_val) else "N/A"
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        d_str = f"{cohens_d:.3f}" if not np.isnan(cohens_d) else "N/A"
        md_str = f"{mean_d:+.2f}" if not np.isnan(mean_d) else "N/A"
        ip_str = f"{iso_p:.4f}" if not np.isnan(iso_p) else "N/A"
        id_str = f"{iso_d:.3f}" if not np.isnan(iso_d) else "N/A"

        print(f"  {shift_s:>5}s  {n:>4}  {total_overlap:>10}  "
              f"{overlap_pct:>8.1f}%  {md_str:>7}  {t_str:>7}  {p_str:>8}  "
              f"{d_str:>6}  {iso_n:>6}  {ip_str:>8}  {id_str:>6}")

    print(f"\n  iso_n/iso_p/iso_d = isolated events only (no overlap with adjacent fear events)")


# ---------------------------------------------------------------------------
# Phase 2 Extension 8: POS-only at 30s window
# ---------------------------------------------------------------------------

RPPG_CONFIG_POS_30S = AnalysisConfig(
    algorithm="pos",
    window_s=30.0,
    step_s=5.0,
    bpm_min=60,
    bpm_max=180,
)


def run_pos_30s_evaluation(loaded_sessions, analyzer):
    """Run fear correlation with POS algorithm at 30s window."""
    from scipy import stats as sp_stats

    all_fear, all_nonfear = [], []
    for sd in loaded_sessions:
        bpm_results = analyzer.analyze(Path(sd["frames_csv"]), RPPG_CONFIG_POS_30S)
        bpm_df = filter_algorithm(bpm_results, "POS")

        frame_ts = pd.to_numeric(sd["model_df"]["timestamp"],
                                 errors="coerce").fillna(0.0).values
        session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0

        fear_ev = compute_event_bpm_stats(bpm_df, sd["gt_df"])
        nonfear_ev = sample_nonfear_intervals(bpm_df, sd["gt_df"], session_dur)
        for ev in fear_ev:
            ev["session"] = sd["tag"]
        all_fear.extend(fear_ev)
        all_nonfear.extend(nonfear_ev)

    bpm_stats = compute_stats(all_fear, all_nonfear)
    delta_stats = compute_delta_stats(all_fear)

    print(f"\n  POS-only @ 30s window / 5s step:")
    print(f"    Fear: n={bpm_stats.get('n_fear', 0)}, "
          f"mean={bpm_stats.get('mean_fear_bpm', 'N/A')}")
    print(f"    Non-fear: n={bpm_stats.get('n_nonfear', 0)}, "
          f"mean={bpm_stats.get('mean_nonfear_bpm', 'N/A')}")
    if bpm_stats.get("p_value") is not None:
        print(f"    t = {bpm_stats['t_stat']:.3f},  p = {bpm_stats['p_value']:.4f},  "
              f"Cohen's d = {bpm_stats['cohens_d']:.3f}")
    print(f"    BPM delta: mean Δ = {delta_stats.get('mean_delta', 'N/A')}")
    if delta_stats.get("p_value") is not None:
        print(f"    Delta t = {delta_stats['t_stat']:.3f},  "
              f"p = {delta_stats['p_value']:.4f}")


# ---------------------------------------------------------------------------
# Formula augmentation
# ---------------------------------------------------------------------------

_BASE_FORMULA_COLS = [
    "f0", "f1", "f2", "f3", "f4", "f5", "f6",
    "f7", "f8", "f9", "f10", "f11", "f12", "f13",
]

_BASE_FORMULA_LABELS = {
    "f0": "aro+T",       "f1": "fear+T",      "f2": "fear+surp+T",
    "f3": "balanced",     "f4": "f0_gated",    "f5": "startle",
    "f6": "anger_supp",   "f7": "hs_only",     "f8": "mp_only",
    "f9": "max_fusion",   "f10": "geo_mean",   "f11": "hs-anger",
    "f12": "hybrid_amp",  "f13": "fear*(1+T)",
}


def enrich_rppg_formulas(model_df, bpm_norm, coeff=0.3, suffix="_rppg"):
    """Create augmented columns for every base formula present.

    Augmentation strategy: fi{suffix} = fi × (1 + coeff × bpm_norm).
    """
    bn = np.asarray(bpm_norm, dtype=float)
    n = min(len(model_df), len(bn))
    bn = bn[:n]
    model_df = model_df.iloc[:n].copy()

    for col in _BASE_FORMULA_COLS:
        if col not in model_df.columns:
            continue
        base = pd.to_numeric(model_df[col], errors="coerce").fillna(0.0).values[:n]
        model_df[f"{col}{suffix}"] = np.clip(base * (1.0 + coeff * bn), 0, 1)

    return model_df


def run_pos_augmentation_sweep(loaded_sessions, analyzer):
    """Compare formula augmentation: CONSENSUS BPM vs POS@30s BPM at multiple coefficients."""
    aug_coeffs = [0.3, 0.5, 0.8, 1.0]
    target_formulas = ["f12"]

    print(f"\n  BPM source comparison for formula augmentation:")
    print(f"  Coefficients: {aug_coeffs}")
    print(f"  Formulas: {target_formulas}")

    pos_bpm_per_session = {}
    for sd in loaded_sessions:
        bpm_results = analyzer.analyze(Path(sd["frames_csv"]), RPPG_CONFIG_POS_30S)
        bpm_df = filter_algorithm(bpm_results, "POS")
        pos_bpm_per_session[sd["tag"]] = bpm_df

    sweep_thresholds = [0.4, 0.5, 0.6, 0.70, 0.80]
    sweep_min_frames = [5, 8, 10, 12, 15]
    sweep_window_sizes = [None, 10, 15, 20]

    results = []

    for bpm_label, get_bpm_df in [
        ("CONSENSUS (15s)", lambda sd: sd["bpm_consensus"]),
        ("POS-only (30s)", lambda sd: pos_bpm_per_session[sd["tag"]]),
    ]:
        for coeff in aug_coeffs:
            sfx = f"_aug_{coeff}"
            aug_sessions = []
            for sd in loaded_sessions:
                bpm_df = get_bpm_df(sd)
                frame_ts = pd.to_numeric(
                    sd["model_df"]["timestamp"], errors="coerce"
                ).fillna(0.0).values
                bpm_at_frame = interpolate_bpm_to_frames(bpm_df, frame_ts)
                bn = compute_bpm_norm(bpm_at_frame, frame_ts)
                aug_df = enrich_rppg_formulas(sd["model_df"], bn, coeff=coeff, suffix=sfx)
                aug_sessions.append({"gt_df": sd["gt_df"], "model_df": aug_df})

            for fcol in target_formulas:
                aug_col = f"{fcol}{sfx}"

                def eval_fn(thresh, mf, ws, _col=aug_col, _asess=aug_sessions):
                    tp = fp = gc = ng = 0
                    for asd in _asess:
                        if _col in asd["model_df"].columns:
                            t, f, g, n = compute_metrics(
                                asd["model_df"], asd["gt_df"], thresh, mf,
                                PAD_START, PAD_END, ws, FILL_RATIO, _col,
                            )
                            tp += t; fp += f; gc += g; ng += n
                    return tp, fp, gc, ng

                best = print_parameter_sweep(
                    formula_col=aug_col,
                    label=f"{fcol}+{bpm_label}(c={coeff})",
                    eval_func=eval_fn,
                    thresholds=sweep_thresholds,
                    min_frames=sweep_min_frames,
                    window_sizes=sweep_window_sizes,
                    pad_start=PAD_START, pad_end=PAD_END,
                    fill_ratio=FILL_RATIO, print_table=False,
                )
                if best:
                    results.append({
                        "bpm_source": bpm_label,
                        "coeff": coeff,
                        "formula": fcol,
                        "f1": best["f1"],
                        "prec": best["prec"],
                        "rec": best["rec"],
                        "thresh": best["thresh"],
                    })

    # Also get base f12 (no augmentation) for reference
    base_f1 = None
    def eval_base(thresh, mf, ws):
        tp = fp = gc = ng = 0
        for sd in loaded_sessions:
            t, f, g, n = compute_metrics(
                sd["model_df"], sd["gt_df"], thresh, mf,
                PAD_START, PAD_END, ws, FILL_RATIO, "f12",
            )
            tp += t; fp += f; gc += g; ng += n
        return tp, fp, gc, ng

    base_best = print_parameter_sweep(
        formula_col="f12", label="f12 (no rPPG)",
        eval_func=eval_base,
        thresholds=sweep_thresholds, min_frames=sweep_min_frames,
        window_sizes=sweep_window_sizes,
        pad_start=PAD_START, pad_end=PAD_END,
        fill_ratio=FILL_RATIO, print_table=False,
    )
    if base_best:
        base_f1 = base_best["f1"]

    print(f"\n{'='*70}")
    print(f"  POS-only Augmentation Sweep Summary")
    print(f"{'='*70}")
    print(f"  {'BPM Source':<22}  {'Coeff':>5}  {'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'ΔF1':>8}")
    if base_f1 is not None:
        print(f"  {'(no rPPG)':<22}  {'—':>5}  "
              f"{base_best['prec']:>7.1%}  {base_best['rec']:>7.1%}  "
              f"{base_f1:>8.4f}  {'—':>8}  <-- BASE")
    best_result = max(results, key=lambda x: x["f1"]) if results else None
    for r in results:
        delta = r["f1"] - base_f1 if base_f1 is not None else 0
        marker = " <-- BEST" if r is best_result else ""
        print(f"  {r['bpm_source']:<22}  {r['coeff']:>5.1f}  "
              f"{r['prec']:>7.1%}  {r['rec']:>7.1%}  "
              f"{r['f1']:>8.4f}  {delta:>+8.4f}{marker}")

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Phase 2 Extension 10: Comprehensive Controlled Sweep
# ---------------------------------------------------------------------------

_SWEEP_WINDOWS = [
    (30.0, 5.0),
    (15.0, 3.0),
    (10.0, 2.0),
    (5.0,  1.0),
]

_SWEEP_WINDOWS_16 = [
    (30.0, 5.0), (30.0, 3.0), (30.0, 2.0), (30.0, 1.0),
    (15.0, 5.0), (15.0, 3.0), (15.0, 2.0), (15.0, 1.0),
    (10.0, 5.0), (10.0, 3.0), (10.0, 2.0), (10.0, 1.0),
    (5.0,  5.0), (5.0,  3.0), (5.0,  2.0), (5.0,  1.0),
]

_SWEEP_ALGO_FILTERS = [
    ("CONSENSUS",  lambda bpm_all: filter_consensus(bpm_all)),
    ("POS+CHROM",  lambda bpm_all: filter_multi_algorithm(bpm_all, ["POS", "CHROM"])),
    ("POS",        lambda bpm_all: filter_algorithm(bpm_all, "POS")),
]

_AUG_COEFFS = [0.3, 0.5, 0.8, 1.0]


def run_comprehensive_sweep(loaded_sessions, analyzer, windows=None):
    """Full combinatorial sweep: 3 algos × N windows × (3 methods + 4 aug coeffs)."""
    from scipy import stats as sp_stats

    if windows is None:
        windows = _SWEEP_WINDOWS

    n_configs = len(windows) * len(_SWEEP_ALGO_FILTERS)
    print(f"\n  Extracting BPM for {len(windows)} windows × "
          f"{len(_SWEEP_ALGO_FILTERS)} algorithms = {n_configs} configs")

    # --- Step 1: Extract BPM for each window/step (algorithm="all" gives us everything) ---
    raw_per_window = {}
    for win, step in windows:
        key = f"{win:.0f}s/{step:.0f}s"
        if key in raw_per_window:
            continue
        print(f"  Extracting window={win:.0f}s step={step:.0f}s ...", end="", flush=True)
        cfg = AnalysisConfig(algorithm="all", window_s=win, step_s=step,
                             bpm_min=60, bpm_max=180)
        per_session = {}
        for sd in loaded_sessions:
            bpm_all = analyzer.analyze(Path(sd["frames_csv"]), cfg)
            per_session[sd["tag"]] = bpm_all
        raw_per_window[key] = per_session
        print(" done.")

    # --- Step 2: Apply algorithm filters to get derived BPM DataFrames ---
    derived = {}
    for win, step in windows:
        wk = f"{win:.0f}s/{step:.0f}s"
        for algo_name, algo_fn in _SWEEP_ALGO_FILTERS:
            config_key = f"{algo_name}@{win:.0f}s/{step:.0f}s"
            bpm_map = {}
            for tag, bpm_all in raw_per_window[wk].items():
                bpm_map[tag] = algo_fn(bpm_all)
            derived[config_key] = {"bpm_map": bpm_map, "win": win, "step": step,
                                   "algo": algo_name}

    # --- Step 3: Run 3 methods for each config ---
    fear_cache = {}
    correlation_rows = []

    for config_key, cfg_data in derived.items():
        bpm_map = cfg_data["bpm_map"]

        # Compute fear/non-fear event stats
        all_fear, all_nonfear = [], []
        for sd in loaded_sessions:
            bpm_df = bpm_map.get(sd["tag"], pd.DataFrame())
            if bpm_df.empty:
                continue
            frame_ts = pd.to_numeric(
                sd["model_df"]["timestamp"], errors="coerce"
            ).fillna(0.0).values
            session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0
            fear_ev = compute_event_bpm_stats(bpm_df, sd["gt_df"])
            nonfear_ev = sample_nonfear_intervals(bpm_df, sd["gt_df"], session_dur)
            for ev in fear_ev:
                ev["session"] = sd["tag"]
            for nf in nonfear_ev:
                nf["session"] = sd["tag"]
            all_fear.extend(fear_ev)
            all_nonfear.extend(nonfear_ev)

        fear_cache[config_key] = (all_fear, all_nonfear)

        # Method 1: Mean BPM
        bpm_stats = compute_stats(all_fear, all_nonfear)
        correlation_rows.append({
            "config": config_key, "method": "Mean BPM",
            "n": bpm_stats.get("n_fear", 0),
            "p": bpm_stats.get("p_value"), "d": bpm_stats.get("cohens_d"),
            "val_fear": bpm_stats.get("mean_fear_bpm"),
            "val_nf": bpm_stats.get("mean_nonfear_bpm"),
        })

        # Method 2: Z-score normalization
        subj_pools = {}
        for sd in loaded_sessions:
            subj = sd["tag"].split("_")[0]
            bpm_df = bpm_map.get(sd["tag"], pd.DataFrame())
            if bpm_df.empty:
                continue
            subj_pools.setdefault(subj, []).extend(
                bpm_df["bpm_smoothed"].values.tolist()
            )

        subj_mu_sig = {}
        for subj, bpms in subj_pools.items():
            if len(bpms) > 1:
                subj_mu_sig[subj] = (float(np.mean(bpms)),
                                     max(float(np.std(bpms, ddof=1)), 1.0))

        fear_z, nonfear_z = [], []
        for ev in all_fear:
            subj = ev.get("session", "").split("_")[0]
            if subj in subj_mu_sig and ev.get("bpm_during") is not None:
                mu, sig = subj_mu_sig[subj]
                fear_z.append((ev["bpm_during"] - mu) / sig)
        for nf in all_nonfear:
            subj = nf.get("session", "").split("_")[0]
            if subj in subj_mu_sig and nf.get("bpm_during") is not None:
                mu, sig = subj_mu_sig[subj]
                nonfear_z.append((nf["bpm_during"] - mu) / sig)

        zp, zd = None, None
        if len(fear_z) >= 3 and len(nonfear_z) >= 3:
            _, zp_raw = sp_stats.ttest_ind(fear_z, nonfear_z, equal_var=False)
            zp = round(float(zp_raw), 4)
            ps = np.sqrt((np.var(fear_z, ddof=1) + np.var(nonfear_z, ddof=1)) / 2)
            zd = round((np.mean(fear_z) - np.mean(nonfear_z)) / ps, 3) if ps > 0 else 0

        correlation_rows.append({
            "config": config_key, "method": "Z-score",
            "n": len(fear_z), "p": zp, "d": zd,
            "val_fear": round(float(np.mean(fear_z)), 3) if fear_z else None,
            "val_nf": round(float(np.mean(nonfear_z)), 3) if nonfear_z else None,
        })

        # Method 3: BPM Variability (contiguous RMSSD)
        fear_rmssd_vals, nonfear_rmssd_vals = [], []
        for sd in loaded_sessions:
            bpm_df = bpm_map.get(sd["tag"], pd.DataFrame())
            if bpm_df.empty:
                continue
            t_c = bpm_df["t_center"].values
            bpms = bpm_df["bpm_smoothed"].values
            gt_df = sd["gt_df"]
            fm = np.zeros(len(t_c), dtype=bool)
            if not gt_df.empty:
                for _, row in gt_df.iterrows():
                    fm |= ((t_c >= row["start_val"] - 5) &
                            (t_c <= row["end_val"] + 5))
            fv = compute_contiguous_rmssd(bpms, fm)
            nfv = compute_contiguous_rmssd(bpms, ~fm)
            if not np.isnan(fv["rmssd"]):
                fear_rmssd_vals.append(fv["rmssd"])
            if not np.isnan(nfv["rmssd"]):
                nonfear_rmssd_vals.append(nfv["rmssd"])

        vp, vd = None, None
        if (len(fear_rmssd_vals) >= 3 and
                len(fear_rmssd_vals) == len(nonfear_rmssd_vals)):
            _, vp_raw = sp_stats.ttest_rel(fear_rmssd_vals, nonfear_rmssd_vals)
            vp = round(float(vp_raw), 4)
            diffs = [f - n for f, n in zip(fear_rmssd_vals, nonfear_rmssd_vals)]
            md, sd_d = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
            vd = round(md / sd_d, 3) if sd_d > 0 else 0

        correlation_rows.append({
            "config": config_key, "method": "Variability",
            "n": len(fear_rmssd_vals), "p": vp, "d": vd,
            "val_fear": round(float(np.mean(fear_rmssd_vals)), 2) if fear_rmssd_vals else None,
            "val_nf": round(float(np.mean(nonfear_rmssd_vals)), 2) if nonfear_rmssd_vals else None,
        })

    # --- Step 4: Augmentation sweep (each config × 4 coefficients) ---
    sweep_thresholds = [0.4, 0.5, 0.6, 0.70, 0.80]
    sweep_min_frames = [5, 8, 10, 12, 15]
    sweep_window_sizes = [None, 10, 15, 20]

    # Base f12 score (no augmentation)
    def eval_base(thresh, mf, ws):
        tp = fp = gc = ng = 0
        for sd in loaded_sessions:
            t, f, g, n = compute_metrics(
                sd["model_df"], sd["gt_df"], thresh, mf,
                PAD_START, PAD_END, ws, FILL_RATIO, "f12",
            )
            tp += t; fp += f; gc += g; ng += n
        return tp, fp, gc, ng

    base_best = print_parameter_sweep(
        formula_col="f12", label="f12_base", eval_func=eval_base,
        thresholds=sweep_thresholds, min_frames=sweep_min_frames,
        window_sizes=sweep_window_sizes, pad_start=PAD_START,
        pad_end=PAD_END, fill_ratio=FILL_RATIO, print_table=False,
    )
    base_f1 = base_best["f1"] if base_best else 0

    aug_rows = []
    for config_key, cfg_data in derived.items():
        bpm_map = cfg_data["bpm_map"]
        for coeff in _AUG_COEFFS:
            sfx = f"_csweep_{coeff}"
            aug_sessions = []
            for sd in loaded_sessions:
                bpm_df = bpm_map.get(sd["tag"], pd.DataFrame())
                frame_ts = pd.to_numeric(
                    sd["model_df"]["timestamp"], errors="coerce"
                ).fillna(0.0).values
                bpm_at_frame = interpolate_bpm_to_frames(bpm_df, frame_ts)
                bn = compute_bpm_norm(bpm_at_frame, frame_ts)
                aug_df = enrich_rppg_formulas(sd["model_df"], bn, coeff=coeff, suffix=sfx)
                aug_sessions.append({"gt_df": sd["gt_df"], "model_df": aug_df})

            aug_col = f"f12{sfx}"

            def eval_fn(thresh, mf, ws, _col=aug_col, _as=aug_sessions):
                tp = fp = gc = ng = 0
                for asd in _as:
                    if _col in asd["model_df"].columns:
                        t, f, g, n = compute_metrics(
                            asd["model_df"], asd["gt_df"], thresh, mf,
                            PAD_START, PAD_END, ws, FILL_RATIO, _col,
                        )
                        tp += t; fp += f; gc += g; ng += n
                return tp, fp, gc, ng

            best = print_parameter_sweep(
                formula_col=aug_col,
                label=f"f12+{config_key}(c={coeff})",
                eval_func=eval_fn,
                thresholds=sweep_thresholds,
                min_frames=sweep_min_frames,
                window_sizes=sweep_window_sizes,
                pad_start=PAD_START, pad_end=PAD_END,
                fill_ratio=FILL_RATIO, print_table=False,
            )
            if best:
                aug_rows.append({
                    "config": config_key, "coeff": coeff,
                    "f1": best["f1"], "prec": best["prec"], "rec": best["rec"],
                })

    # --- Print Table 1: Fear Correlation Matrix ---
    print(f"\n\n{'='*70}")
    print(f"  Table 1: Fear Correlation Matrix — {len(loaded_sessions)} sessions")
    print(f"  3 methods × {len(_SWEEP_ALGO_FILTERS)} algos × {len(windows)} windows")
    print(f"{'='*70}")
    print(f"  {'Config':<26}  {'Method':<12}  {'n':>4}  "
          f"{'Fear':>8}  {'NF':>8}  {'p':>8}  {'d':>7}")

    best_overall_p = 1.0
    best_overall = None
    for r in correlation_rows:
        p_str = f"{r['p']:.4f}" if r["p"] is not None else "   N/A"
        d_str = f"{r['d']:.3f}" if r["d"] is not None else "  N/A"
        f_str = f"{r['val_fear']}" if r["val_fear"] is not None else "N/A"
        n_str = f"{r['val_nf']}" if r["val_nf"] is not None else "N/A"
        if r["p"] is not None and r["p"] < best_overall_p:
            best_overall_p = r["p"]
            best_overall = r
        print(f"  {r['config']:<26}  {r['method']:<12}  {r['n']:>4}  "
              f"{f_str:>8}  {n_str:>8}  {p_str:>8}  {d_str:>7}")

    if best_overall:
        sig = "**SIGNIFICANT**" if best_overall_p < 0.05 else "not significant"
        print(f"\n  → Best cell: {best_overall['config']} / {best_overall['method']} "
              f"(p={best_overall['p']:.4f}, d={best_overall['d']:.3f}) [{sig}]")

    # --- Print Table 2: Augmentation Matrix ---
    print(f"\n\n{'='*70}")
    print(f"  Table 2: f12 Augmentation Sweep — {len(loaded_sessions)} sessions")
    print(f"  {n_configs} configs × {len(_AUG_COEFFS)} coefficients")
    print(f"{'='*70}")

    coeff_headers = "  ".join(f"c={c:<4}" for c in _AUG_COEFFS)
    print(f"  {'Config':<26}  {coeff_headers}")
    if base_best:
        base_str = f"  {'(no rPPG)':<26}"
        for _ in _AUG_COEFFS:
            base_str += f"  {base_f1:.4f}"
        print(f"{base_str}  <-- BASE")

    best_aug_f1 = base_f1
    best_aug_row = None
    for config_key in derived:
        line = f"  {config_key:<26}"
        for coeff in _AUG_COEFFS:
            match = next((r for r in aug_rows
                          if r["config"] == config_key and r["coeff"] == coeff), None)
            if match:
                delta = match["f1"] - base_f1
                line += f"  {delta:+.4f}"
                if match["f1"] > best_aug_f1:
                    best_aug_f1 = match["f1"]
                    best_aug_row = match
            else:
                line += f"     N/A"
        print(line)

    if best_aug_row:
        print(f"\n  → Best augmentation: {best_aug_row['config']} c={best_aug_row['coeff']} "
              f"(F1={best_aug_row['f1']:.4f}, Δ={best_aug_row['f1'] - base_f1:+.4f}, "
              f"P={best_aug_row['prec']:.1%}, R={best_aug_row['rec']:.1%})")
    else:
        print(f"\n  → No augmentation improved over base F1={base_f1:.4f}")

    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Phase 2 Extension 11: rPPG Statistical Validation Suite (A8)
# ---------------------------------------------------------------------------

_VALIDATION_CONFIG = AnalysisConfig(
    algorithm="all", window_s=10.0, step_s=2.0, bpm_min=60, bpm_max=180
)


def _extract_fear_nonfear_bpms(loaded_sessions, analyzer, cfg=None,
                               algo_filter=None):
    """Extract fear/non-fear BPM lists. Returns (fear_bpms, nonfear_bpms, per_session_data)."""
    if cfg is None:
        cfg = _VALIDATION_CONFIG
    if algo_filter is None:
        algo_filter = lambda bpm_all: filter_algorithm(bpm_all, "POS")

    per_session = []
    for sd in loaded_sessions:
        bpm_all = analyzer.analyze(Path(sd["frames_csv"]), cfg)
        bpm_df = algo_filter(bpm_all)
        if bpm_df.empty:
            continue
        frame_ts = pd.to_numeric(
            sd["model_df"]["timestamp"], errors="coerce"
        ).fillna(0.0).values
        session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0
        fear_ev = compute_event_bpm_stats(bpm_df, sd["gt_df"])
        nonfear_ev = sample_nonfear_intervals(bpm_df, sd["gt_df"], session_dur)
        f_bpms = [e["bpm_during"] for e in fear_ev if e["bpm_during"] is not None]
        nf_bpms = [e["bpm_during"] for e in nonfear_ev if e["bpm_during"] is not None]
        per_session.append({
            "tag": sd["tag"], "fear_bpms": f_bpms, "nonfear_bpms": nf_bpms,
        })

    all_fear = []
    all_nonfear = []
    for ps in per_session:
        all_fear.extend(ps["fear_bpms"])
        all_nonfear.extend(ps["nonfear_bpms"])
    return all_fear, all_nonfear, per_session


_VALIDATION_CONFIGS = [
    ("POS@10s/2s", AnalysisConfig(algorithm="all", window_s=10.0, step_s=2.0, bpm_min=60, bpm_max=180)),
    ("POS@10s/5s", AnalysisConfig(algorithm="all", window_s=10.0, step_s=5.0, bpm_min=60, bpm_max=180)),
]


def run_rppg_validation(loaded_sessions, analyzer):
    """A8: rPPG statistical validation — run for each candidate config."""
    for cfg_label, cfg in _VALIDATION_CONFIGS:
        print(f"\n\n  {'#'*60}")
        print(f"  ## Validation: {cfg_label}")
        print(f"  {'#'*60}")
        _run_rppg_validation_single(loaded_sessions, analyzer, cfg, cfg_label)

    _run_family_wise_permutation(loaded_sessions, analyzer)


def _run_rppg_validation_single(loaded_sessions, analyzer, cfg, cfg_label):
    """A8a-e for a single config."""
    from scipy import stats as sp_stats
    from itertools import combinations

    n_perms = 10000

    all_fear, all_nonfear, per_session = _extract_fear_nonfear_bpms(
        loaded_sessions, analyzer, cfg=cfg
    )
    n_f, n_nf = len(all_fear), len(all_nonfear)
    obs_delta = float(np.mean(all_fear) - np.mean(all_nonfear)) if n_f and n_nf else 0

    # ── A8a: Per-session consistency ──
    print(f"\n  {'='*60}")
    print(f"  A8a. Per-Session Consistency — {cfg_label}")
    print(f"  {'='*60}")
    print(f"  {'Session':<20}  {'n_fear':>6}  {'Fear BPM':>9}  {'NF BPM':>9}  {'Delta':>7}  {'d':>7}  {'Dir':>4}")

    n_positive = 0
    for ps in per_session:
        fb, nfb = ps["fear_bpms"], ps["nonfear_bpms"]
        if not fb or not nfb:
            continue
        mf, mnf = float(np.mean(fb)), float(np.mean(nfb))
        delta = mf - mnf
        ps_var = (np.var(fb, ddof=1) + np.var(nfb, ddof=1)) / 2
        d = delta / np.sqrt(ps_var) if ps_var > 0 else 0
        direction = "+" if delta > 0 else "-"
        if delta > 0:
            n_positive += 1
        print(f"  {ps['tag']:<20}  {len(fb):>6}  {mf:>9.1f}  {mnf:>9.1f}  "
              f"{delta:>+7.1f}  {d:>7.3f}  {direction:>4}")

    n_sess = len([ps for ps in per_session if ps["fear_bpms"] and ps["nonfear_bpms"]])
    print(f"\n  Direction consistency: {n_positive}/{n_sess} sessions show fear > non-fear")

    # ── A8b: Permutation test — single config ──
    print(f"\n  {'='*60}")
    print(f"  A8b. Permutation Test — {cfg_label} (n={n_perms})")
    print(f"  {'='*60}")

    pooled = np.array(all_fear + all_nonfear, dtype=float)
    rng = np.random.default_rng(42)
    n_exceed = 0
    for _ in range(n_perms):
        rng.shuffle(pooled)
        perm_delta = float(np.mean(pooled[:n_f]) - np.mean(pooled[n_f:]))
        if perm_delta >= obs_delta:
            n_exceed += 1

    perm_p = (n_exceed + 1) / (n_perms + 1)
    print(f"  Observed Δ = {obs_delta:+.2f} BPM")
    print(f"  Permutations with Δ ≥ observed: {n_exceed}/{n_perms}")
    print(f"  Permutation p-value (one-sided): {perm_p:.4f}")

    # ── A8d: Leave-k-out consistency ──
    print(f"\n  {'='*60}")
    print(f"  A8d. Leave-k-out Consistency — {cfg_label}")
    print(f"  {'='*60}")

    session_tags = [ps["tag"] for ps in per_session if ps["fear_bpms"]]
    n_total = len(session_tags)

    for k in [1, 2, 3]:
        if k >= n_total:
            continue
        folds = list(combinations(range(n_total), k))
        n_pos = 0
        fold_deltas = []
        for held_out_idx in folds:
            train_f, train_nf = [], []
            for i in range(n_total):
                if i not in held_out_idx:
                    train_f.extend(per_session[i]["fear_bpms"])
                    train_nf.extend(per_session[i]["nonfear_bpms"])
            if train_f and train_nf:
                d = float(np.mean(train_f) - np.mean(train_nf))
                fold_deltas.append(d)
                if d > 0:
                    n_pos += 1

        print(f"\n  Leave-{k}-out: {len(folds)} folds")
        print(f"    Direction consistency: {n_pos}/{len(folds)} folds show fear > non-fear")
        if fold_deltas:
            print(f"    Mean Δ across folds: {np.mean(fold_deltas):+.2f} BPM "
                  f"(std={np.std(fold_deltas, ddof=1):.2f})")

    # ── A8e: Per-session vs per-event weighting ──
    print(f"\n  {'='*60}")
    print(f"  A8e. Per-Session vs Per-Event Weighting — {cfg_label}")
    print(f"  {'='*60}")

    if n_f >= 3 and n_nf >= 3:
        t_event, p_event = sp_stats.ttest_ind(all_fear, all_nonfear, equal_var=False)
        print(f"\n  Per-event (n_fear={n_f}, n_nf={n_nf}):")
        print(f"    t = {t_event:.3f},  p = {p_event:.4f}")

    session_deltas = []
    for ps in per_session:
        fb, nfb = ps["fear_bpms"], ps["nonfear_bpms"]
        if fb and nfb:
            session_deltas.append(float(np.mean(fb) - np.mean(nfb)))

    if len(session_deltas) >= 3:
        t_sess, p_sess = sp_stats.ttest_1samp(session_deltas, 0.0)
        print(f"\n  Per-session (n_sessions={len(session_deltas)}):")
        print(f"    Session deltas: {[f'{d:+.1f}' for d in session_deltas]}")
        print(f"    t = {t_sess:.3f},  p = {p_sess:.4f}")
    elif session_deltas:
        print(f"\n  Per-session (n={len(session_deltas)}): too few for t-test")
        print(f"    Session deltas: {[f'{d:+.1f}' for d in session_deltas]}")

    sys.stdout.flush()


def _run_family_wise_permutation(loaded_sessions, analyzer):
    """A8c: Family-wise permutation test across all configs (run once)."""
    from scipy import stats as sp_stats

    n_perms = 10000
    rng = np.random.default_rng(42)

    print(f"\n\n  {'#'*60}")
    print(f"  ## A8c. Family-Wise Permutation Test — all configs (n={n_perms})")
    print(f"  {'#'*60}")

    algo_filters = [
        ("CONSENSUS", lambda bpm_all: filter_consensus(bpm_all)),
        ("POS+CHROM", lambda bpm_all: filter_multi_algorithm(bpm_all, ["POS", "CHROM"])),
        ("POS",       lambda bpm_all: filter_algorithm(bpm_all, "POS")),
    ]
    test_windows = _SWEEP_WINDOWS_16

    configs_data = {}
    extracted = {}
    for win, step in test_windows:
        wk = f"{win:.0f}s/{step:.0f}s"
        if wk not in extracted:
            cfg = AnalysisConfig(algorithm="all", window_s=win, step_s=step,
                                 bpm_min=60, bpm_max=180)
            raw_bpm = {}
            for sd in loaded_sessions:
                raw_bpm[sd["tag"]] = analyzer.analyze(Path(sd["frames_csv"]), cfg)
            extracted[wk] = raw_bpm

        for algo_name, algo_fn in algo_filters:
            ckey = f"{algo_name}@{win:.0f}s/{step:.0f}s"
            all_f, all_nf = [], []
            for sd in loaded_sessions:
                bpm_df = algo_fn(extracted[wk][sd["tag"]])
                if bpm_df.empty:
                    continue
                frame_ts = pd.to_numeric(
                    sd["model_df"]["timestamp"], errors="coerce"
                ).fillna(0.0).values
                session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0
                fear_ev = compute_event_bpm_stats(bpm_df, sd["gt_df"])
                nonfear_ev = sample_nonfear_intervals(bpm_df, sd["gt_df"], session_dur)
                all_f.extend([e["bpm_during"] for e in fear_ev if e["bpm_during"] is not None])
                all_nf.extend([e["bpm_during"] for e in nonfear_ev if e["bpm_during"] is not None])
            if len(all_f) >= 3 and len(all_nf) >= 3:
                configs_data[ckey] = (np.array(all_f), np.array(all_nf))

    obs_best_p = 1.0
    obs_best_config = None
    for ckey, (f_arr, nf_arr) in configs_data.items():
        _, p = sp_stats.ttest_ind(f_arr, nf_arr, equal_var=False)
        if p < obs_best_p:
            obs_best_p = p
            obs_best_config = ckey

    print(f"\n  Observed best p-value: {obs_best_p:.6f} ({obs_best_config})")
    print(f"  Testing {len(configs_data)} configs across {n_perms} permutations...")

    n_family_exceed = 0
    for i in range(n_perms):
        worst_best_p = 1.0
        for ckey, (f_arr, nf_arr) in configs_data.items():
            pooled_c = np.concatenate([f_arr, nf_arr])
            rng.shuffle(pooled_c)
            nf_c = len(f_arr)
            _, pp = sp_stats.ttest_ind(pooled_c[:nf_c], pooled_c[nf_c:],
                                       equal_var=False)
            if pp < worst_best_p:
                worst_best_p = pp
        if worst_best_p <= obs_best_p:
            n_family_exceed += 1

    fwer_p = (n_family_exceed + 1) / (n_perms + 1)
    print(f"  Permutations with best-p ≤ {obs_best_p:.6f}: {n_family_exceed}/{n_perms}")
    print(f"  Family-wise corrected p-value: {fwer_p:.4f}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COL = "#2a2a4a"


def generate_rppg_timeline(session_data, output_path, threshold=0.70):
    """4-panel per-session timeline: FER + BPM + components + delta bars."""
    plt = _setup_matplotlib()

    model_df = session_data["model_df"]
    gt_df = session_data["gt_df"]
    bpm_cons = session_data["bpm_consensus"]
    fear_stats = session_data.get("fear_stats", [])
    tag = session_data["tag"]

    ts = pd.to_numeric(model_df["timestamp"], errors="coerce").fillna(0.0).values
    smooth_w = 15

    has_delta = any(s["bpm_delta"] is not None for s in fear_stats)
    n_panels = 4 if has_delta else 3
    ratios = [3, 2, 2, 1] if has_delta else [3, 2, 2]

    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels),
                              facecolor=DARK_BG,
                              gridspec_kw={"hspace": 0.35, "height_ratios": ratios})

    # --- Panel 1: FER f12 signal ---
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.set_title(f"rPPG × FER Timeline — {tag}", color="white", fontsize=12, pad=8)

    if "f12" in model_df.columns:
        f12 = pd.to_numeric(model_df["f12"], errors="coerce").fillna(0.0).values
        f12_smooth = pd.Series(f12).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax.fill_between(ts, 0, f12, alpha=0.15, color="#cc44ff")
        ax.plot(ts, f12_smooth, color="#cc44ff", linewidth=1.2, label="f12 (smooth)")
    ax.axhline(y=threshold, color="#ff6666", linestyle="--", alpha=0.5, linewidth=0.8,
               label=f"threshold={threshold}")

    for _, row in gt_df.iterrows():
        ax.axvspan(row["start_val"], row["end_val"], alpha=0.20, color="#00cc00")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("f12", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=GRID_COL, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel 2: BPM timeseries ---
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.set_title("CONSENSUS BPM (15s window, 3s step)", color="white", fontsize=10, pad=6)

    if not bpm_cons.empty:
        tc = bpm_cons["t_center"].values
        bpm_vals = bpm_cons["bpm_smoothed"].values
        snr_vals = bpm_cons["snr"].values

        for i in range(len(tc) - 1):
            snr = snr_vals[i]
            color = "#00cc66" if snr >= 3 else ("#cccc00" if snr >= 2 else "#cc3333")
            ax2.axvspan(tc[i], tc[i + 1], alpha=0.08, color=color)

        ax2.plot(tc, bpm_vals, color="#00ccff", linewidth=1.5, marker=".", markersize=3,
                 label="CONSENSUS BPM")
        ax2.set_ylabel("BPM", color="white", fontsize=10)

        bpm_median = float(np.median(bpm_vals))
        ax2.axhline(y=bpm_median, color="#00ccff", linestyle=":", alpha=0.4,
                     label=f"median={bpm_median:.0f}")

    for _, row in gt_df.iterrows():
        ax2.axvspan(row["start_val"], row["end_val"], alpha=0.20, color="#00cc00")

    ax2.tick_params(colors="white", labelsize=8)
    ax2.grid(True, color=GRID_COL, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    # --- Panel 3: Component signals + bpm_norm ---
    ax3 = axes[2]
    ax3.set_facecolor(PANEL_BG)
    ax3.set_title("Component Signals + bpm_norm", color="white", fontsize=10, pad=6)

    if "hs_fear" in model_df.columns:
        fear = pd.to_numeric(model_df["hs_fear"], errors="coerce").fillna(0.0).values
        fear_s = pd.Series(fear).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax3.plot(ts, fear_s, color="#ff4488", linewidth=1.0, label="hs_fear")

    if "mp_tension" in model_df.columns:
        tens = pd.to_numeric(model_df["mp_tension"], errors="coerce").fillna(0.0).values
        tens_s = pd.Series(tens).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax3.plot(ts, tens_s, color="#44ff88", linewidth=1.0, label="mp_tension")

    if "hs_arousal" in model_df.columns:
        arou = pd.to_numeric(model_df["hs_arousal"], errors="coerce").fillna(0.0).values
        arou_s = pd.Series(arou).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax3.plot(ts, arou_s, color="#ff8844", linewidth=0.8, alpha=0.6, label="hs_arousal")

    if "bpm_norm" in model_df.columns:
        bn = model_df["bpm_norm"].values
        bn_s = pd.Series(bn).rolling(smooth_w, center=True, min_periods=1).mean().values
        ax3_r = ax3.twinx()
        ax3_r.plot(ts[:len(bn_s)], bn_s, color="#00ccff", linewidth=1.2, alpha=0.7,
                   label="bpm_norm")
        ax3_r.set_ylim(0, 1.05)
        ax3_r.set_ylabel("bpm_norm", color="#00ccff", fontsize=9)
        ax3_r.tick_params(colors="#00ccff", labelsize=7)
        ax3_r.legend(loc="upper left", fontsize=7)

    for _, row in gt_df.iterrows():
        ax3.axvspan(row["start_val"], row["end_val"], alpha=0.12, color="#00cc00")

    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("Time (s)", color="white", fontsize=10)
    ax3.set_ylabel("Score", color="white", fontsize=10)
    ax3.tick_params(colors="white", labelsize=8)
    ax3.grid(True, color=GRID_COL, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    # --- Panel 4: Delta bars per GT event ---
    if has_delta:
        ax4 = axes[3]
        ax4.set_facecolor(PANEL_BG)
        ax4.set_title("BPM Delta per GT Event", color="white", fontsize=10, pad=6)

        events_with_delta = [s for s in fear_stats if s["bpm_delta"] is not None]
        if events_with_delta:
            x_pos = [s["start_s"] for s in events_with_delta]
            deltas = [s["bpm_delta"] for s in events_with_delta]
            colors = ["#00cc66" if d >= 0 else "#cc3333" for d in deltas]
            widths = [max(s["end_s"] - s["start_s"], 2.0) for s in events_with_delta]
            ax4.bar(x_pos, deltas, width=widths, color=colors, alpha=0.7, align="edge")
            ax4.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)
            ax4.set_ylabel("ΔBPM", color="white", fontsize=9)

        ax4.set_xlabel("Time (s)", color="white", fontsize=10)
        ax4.tick_params(colors="white", labelsize=8)
        ax4.grid(True, color=GRID_COL, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [rppg-timeline] → {output_path}")


def generate_bpm_distribution(all_fear, all_nonfear, bpm_stats, delta_stats,
                              output_path, title=""):
    """Box plots + delta histogram."""
    plt = _setup_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)

    # Left: box plots
    ax1.set_facecolor(PANEL_BG)
    ax1.set_title(f"BPM: Fear vs Non-Fear {title}", color="white", fontsize=11, pad=8)

    fear_bpms = [s["bpm_during"] for s in all_fear if s["bpm_during"] is not None]
    nonfear_bpms = [s["bpm_during"] for s in all_nonfear]

    bp = ax1.boxplot([fear_bpms, nonfear_bpms], labels=["Fear", "Non-Fear"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#ff4488")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#4488ff")
    bp["boxes"][1].set_alpha(0.6)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("white")

    annotation = ""
    if bpm_stats.get("p_value") is not None:
        annotation = (f"p = {bpm_stats['p_value']:.4f}\n"
                      f"Cohen's d = {bpm_stats['cohens_d']:.3f}")
    ax1.text(0.05, 0.95, annotation, transform=ax1.transAxes, color="white",
             fontsize=9, va="top", fontfamily="monospace")
    ax1.set_ylabel("BPM", color="white", fontsize=10)
    ax1.tick_params(colors="white", labelsize=9)
    ax1.grid(True, color=GRID_COL, alpha=0.3, axis="y")

    # Right: delta histogram
    ax2.set_facecolor(PANEL_BG)
    ax2.set_title(f"BPM Delta (Fear event − Baseline) {title}", color="white",
                  fontsize=11, pad=8)

    deltas = [s["bpm_delta"] for s in all_fear if s["bpm_delta"] is not None]
    if deltas:
        ax2.hist(deltas, bins=max(len(deltas) // 3, 5), color="#00ccff", alpha=0.7,
                 edgecolor="white", linewidth=0.5)
        ax2.axvline(x=0, color="white", linewidth=1, alpha=0.5)
        if delta_stats.get("mean_delta") is not None:
            ax2.axvline(x=delta_stats["mean_delta"], color="#ff8844", linewidth=2,
                        label=f"mean={delta_stats['mean_delta']:.1f}")
            if delta_stats.get("ci_95"):
                lo, hi = delta_stats["ci_95"]
                ax2.axvspan(lo, hi, alpha=0.15, color="#ff8844", label=f"95% CI [{lo:.1f}, {hi:.1f}]")

        annotation2 = ""
        if delta_stats.get("p_value") is not None:
            annotation2 = f"p = {delta_stats['p_value']:.4f}  (H₀: δ=0)"
        ax2.text(0.05, 0.95, annotation2, transform=ax2.transAxes, color="white",
                 fontsize=9, va="top", fontfamily="monospace")
        ax2.legend(loc="upper right", fontsize=8)

    ax2.set_xlabel("ΔBPM", color="white", fontsize=10)
    ax2.set_ylabel("Count", color="white", fontsize=10)
    ax2.tick_params(colors="white", labelsize=9)
    ax2.grid(True, color=GRID_COL, alpha=0.3, axis="y")

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [rppg-distribution] → {output_path}")


def generate_correlation_plot(corr_results, formula_scores, output_path, title=""):
    """Correlation summary + rPPG ΔF1 bar chart per formula."""
    plt = _setup_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)

    # Left: correlation info as text
    ax1.set_facecolor(PANEL_BG)
    ax1.set_title(f"FER × BPM Correlation {title}", color="white", fontsize=11, pad=8)

    text_lines = []
    for c in corr_results:
        tag = c.get("tag", "?")
        pr = c.get("pearson_r")
        sr = c.get("spearman_r")
        n = c.get("n_points", 0)
        pr_str = f"{pr:.3f}" if pr is not None else "N/A"
        sr_str = f"{sr:.3f}" if sr is not None else "N/A"
        text_lines.append(f"{tag:15s}  r={pr_str}  ρ={sr_str}  (n={n})")

    if corr_results:
        valid_pr = [c["pearson_r"] for c in corr_results if c.get("pearson_r") is not None]
        valid_sr = [c["spearman_r"] for c in corr_results if c.get("spearman_r") is not None]
        if valid_pr:
            text_lines.append(f"\n{'MEAN':15s}  r={np.mean(valid_pr):.3f}  "
                              f"ρ={np.mean(valid_sr):.3f}")

    ax1.text(0.05, 0.95, "\n".join(text_lines), transform=ax1.transAxes,
             color="white", fontsize=9, va="top", fontfamily="monospace")
    ax1.axis("off")

    # Right: ΔF1 bar chart (rPPG improvement per formula)
    ax2.set_facecolor(PANEL_BG)
    ax2.set_title(f"rPPG ΔF1 per Formula {title}", color="white", fontsize=11, pad=8)

    base_scores = {k: v for k, v in formula_scores.items() if not k.endswith("_rppg")}
    rppg_scores = {k.replace("_rppg", ""): v for k, v in formula_scores.items()
                   if k.endswith("_rppg")}

    common = [c for c in _BASE_FORMULA_COLS if c in base_scores and c in rppg_scores]
    if common:
        deltas = [rppg_scores[c] - base_scores[c] for c in common]
        colors = ["#00cc66" if d >= 0 else "#cc3333" for d in deltas]

        bars = ax2.bar(common, deltas, color=colors, alpha=0.8,
                       edgecolor="white", linewidth=0.5)
        for bar, val, c in zip(bars, deltas, common):
            y_off = 0.003 if val >= 0 else -0.008
            ax2.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                     f"{val:+.3f}", ha="center", color="white", fontsize=7,
                     fontweight="bold" if c == "f12" else "normal")

        ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.5)
        ax2.set_ylabel("ΔF1 (base → +rPPG)", color="white", fontsize=10)
        ax2.tick_params(colors="white", labelsize=8)
        ax2.tick_params(axis="x", rotation=45)
    else:
        ax2.text(0.5, 0.5, "No formula scores available", transform=ax2.transAxes,
                 color="white", ha="center", fontsize=10)
        ax2.axis("off")

    ax2.grid(True, color=GRID_COL, alpha=0.3, axis="y")

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [rppg-correlation] → {output_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_evaluate_rppg(mode="ALL", output_dir=None, force_reextract=False,
                      ext_flags=None):
    """Run the full rPPG × FER evaluation.

    Args:
        mode: "ALL" or "GAMEPLAY"
        output_dir: where to write results (default: new timestamped dir)
        force_reextract: re-extract ROI even if cache exists
        ext_flags: dict of Phase 2 extension flags (facedet, peak, etc.)
    """
    if ext_flags is None:
        ext_flags = {}
    sessions = RPPG_SESSIONS if mode == "ALL" else [
        s for s in RPPG_SESSIONS if s["type"] == "gameplay"
    ]

    if output_dir is None:
        log_dir = os.path.join(_REPO_ROOT, "Pipeline", "logs", "comparisons")
        output_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"evaluate_rppg_{mode}.txt")

    print(f"\n{'='*70}")
    print(f"  rPPG × FER Evaluation — {mode} ({len(sessions)} sessions)")
    print(f"  Window: {RPPG_CONFIG.window_s}s  Step: {RPPG_CONFIG.step_s}s")
    print(f"{'='*70}\n")

    # ── 1. Load sessions: GT, model CSV, extract rPPG, compute BPM ─────
    loaded = []
    analyzer = RppgAnalyzer()

    for sess in sessions:
        tag = sess["tag"]
        print(f"\n--- {tag} ({sess['type']}) ---")

        try:
            gt_df = _load_gt(sess["gt_csv"])
        except FileNotFoundError:
            print(f"  WARNING: GT not found: {sess['gt_csv']}")
            continue

        try:
            model_df = pd.read_csv(sess["model_csv"], low_memory=False)
            _enrich_formulas(model_df)
        except FileNotFoundError:
            print(f"  WARNING: model CSV not found: {sess['model_csv']}")
            continue

        frames_csv = extract_or_load_cached(
            Path(sess["video"]), Path(_CACHE_DIR), force=force_reextract,
        )

        bpm_results = analyzer.analyze(frames_csv, RPPG_CONFIG)
        bpm_cons = filter_consensus(bpm_results)

        frame_ts = pd.to_numeric(model_df["timestamp"], errors="coerce").fillna(0.0).values
        bpm_at_frame = interpolate_bpm_to_frames(bpm_cons, frame_ts)
        bpm_norm = compute_bpm_norm(bpm_at_frame, frame_ts)
        model_df["bpm_norm"] = 0.0
        model_df.loc[:len(bpm_norm) - 1, "bpm_norm"] = bpm_norm

        n_plausible = int((bpm_cons["bpm_plausible"] == 1).sum()) if not bpm_cons.empty else 0
        n_total = len(bpm_cons)
        mean_snr = float(bpm_cons["snr"].mean()) if not bpm_cons.empty else 0.0

        quality = {
            "n_bpm_windows": n_total,
            "n_plausible": n_plausible,
            "pct_plausible": round(100 * n_plausible / max(n_total, 1), 1),
            "mean_snr": round(mean_snr, 2),
            "median_bpm": round(float(bpm_cons["bpm_smoothed"].median()), 1) if not bpm_cons.empty else 0,
        }

        print(f"  BPM windows: {n_total}  plausible: {n_plausible} ({quality['pct_plausible']}%)"
              f"  SNR: {mean_snr:.2f}  median BPM: {quality['median_bpm']}")
        print(f"  GT events: {len(gt_df)}")

        session_dur = float(frame_ts[-1]) if len(frame_ts) > 0 else 0
        fear_stats = compute_event_bpm_stats(bpm_cons, gt_df)
        nonfear_stats = sample_nonfear_intervals(bpm_cons, gt_df, session_dur)

        loaded.append({
            "tag": tag,
            "type": sess["type"],
            "gt_df": gt_df,
            "gt_file": sess["gt_csv"],
            "model_df": model_df,
            "model_file": sess["model_csv"],
            "bpm_consensus": bpm_cons,
            "bpm_results": bpm_results,
            "fear_stats": fear_stats,
            "nonfear_stats": nonfear_stats,
            "quality": quality,
            "frames_csv": str(frames_csv),
        })

    if not loaded:
        print("No sessions loaded — exiting.")
        return

    # ── 2. Per-session quality & fear BPM summary ──────────────────────
    print(f"\n\n{'='*70}")
    print(f"  Per-Session rPPG Quality & BPM Summary — {mode}")
    print(f"{'='*70}")
    print(f"  {'Tag':<18}  {'Type':<12}  {'BPM Win':>8}  {'Plaus%':>7}  "
          f"{'SNR':>6}  {'Med BPM':>8}  {'GT Evts':>8}  {'Fear BPM':>9}  {'Pre BPM':>8}  {'ΔBPM':>7}")

    for sd in loaded:
        q = sd["quality"]
        fear_bpms = [s["bpm_during"] for s in sd["fear_stats"] if s["bpm_during"] is not None]
        pre_bpms = [s["bpm_pre"] for s in sd["fear_stats"] if s["bpm_pre"] is not None]
        deltas = [s["bpm_delta"] for s in sd["fear_stats"] if s["bpm_delta"] is not None]

        mean_fear = f"{np.mean(fear_bpms):.1f}" if fear_bpms else "N/A"
        mean_pre = f"{np.mean(pre_bpms):.1f}" if pre_bpms else "N/A"
        mean_delta = f"{np.mean(deltas):+.1f}" if deltas else "N/A"

        print(f"  {sd['tag']:<18}  {sd['type']:<12}  {q['n_bpm_windows']:>8}  "
              f"{q['pct_plausible']:>6.1f}%  {q['mean_snr']:>6.2f}  "
              f"{q['median_bpm']:>8.1f}  {len(sd['gt_df']):>8}  "
              f"{mean_fear:>9}  {mean_pre:>8}  {mean_delta:>7}")

    sys.stdout.flush()

    # ── 3. Aggregate statistical tests ─────────────────────────────────
    all_fear = []
    all_nonfear = []
    for sd in loaded:
        for fs in sd["fear_stats"]:
            fs_copy = dict(fs)
            fs_copy["session"] = sd["tag"]
            all_fear.append(fs_copy)
        for nf in sd["nonfear_stats"]:
            nf_copy = dict(nf)
            nf_copy["session"] = sd["tag"]
            all_nonfear.append(nf_copy)

    bpm_stats = compute_stats(all_fear, all_nonfear)
    delta_stats = compute_delta_stats(all_fear)

    print(f"\n\n{'='*70}")
    print(f"  Aggregate Statistics — {mode} ({len(loaded)} sessions)")
    print(f"{'='*70}")

    print(f"\n  BPM Fear vs Non-Fear (Welch's t-test):")
    print(f"    Fear events:    n={bpm_stats['n_fear']}, mean={bpm_stats['mean_fear_bpm']}, "
          f"std={bpm_stats['std_fear_bpm']}")
    print(f"    Non-fear:       n={bpm_stats['n_nonfear']}, mean={bpm_stats['mean_nonfear_bpm']}, "
          f"std={bpm_stats['std_nonfear_bpm']}")
    if bpm_stats["p_value"] is not None:
        print(f"    t = {bpm_stats['t_stat']:.3f},  p = {bpm_stats['p_value']:.4f},  "
              f"Cohen's d = {bpm_stats['cohens_d']:.3f}")
    else:
        print(f"    Not enough samples for t-test")

    print(f"\n  BPM Delta (one-sample t-test, H₀: δ=0):")
    print(f"    n={delta_stats['n_events']},  mean Δ={delta_stats['mean_delta']},  "
          f"std={delta_stats['std_delta']}")
    if delta_stats["p_value"] is not None:
        print(f"    t = {delta_stats['t_stat']:.3f},  p = {delta_stats['p_value']:.4f},  "
              f"95% CI = {delta_stats['ci_95']}")
    else:
        print(f"    Not enough samples for t-test")

    # ── 4. FER × BPM correlation per session ───────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  FER (f12) × BPM Correlation — {mode}")
    print(f"{'='*70}")
    print(f"  {'Tag':<18}  {'Pearson r':>10}  {'p':>8}  {'Spearman ρ':>11}  {'p':>8}  {'n':>6}")

    corr_results = []
    for sd in loaded:
        corr = compute_fer_bpm_correlation(sd["model_df"], sd["bpm_consensus"])
        corr["tag"] = sd["tag"]
        corr_results.append(corr)

        pr_str = f"{corr['pearson_r']:.4f}" if corr["pearson_r"] is not None else "N/A"
        pp_str = f"{corr['pearson_p']:.4f}" if corr["pearson_p"] is not None else "N/A"
        sr_str = f"{corr['spearman_r']:.4f}" if corr["spearman_r"] is not None else "N/A"
        sp_str = f"{corr['spearman_p']:.4f}" if corr["spearman_p"] is not None else "N/A"
        print(f"  {sd['tag']:<18}  {pr_str:>10}  {pp_str:>8}  {sr_str:>11}  {sp_str:>8}  {corr['n_points']:>6}")

    valid_pr = [c["pearson_r"] for c in corr_results if c.get("pearson_r") is not None]
    valid_sr = [c["spearman_r"] for c in corr_results if c.get("spearman_r") is not None]
    if valid_pr:
        print(f"\n  {'MEAN':<18}  {np.mean(valid_pr):>10.4f}  {'':>8}  {np.mean(valid_sr):>11.4f}")

    sys.stdout.flush()

    # ── 5. Formula augmentation (f14, f15, f16) + v2 detection sweep ───
    print(f"\n\n{'='*70}")
    print(f"  rPPG-Augmented Formula Evaluation — {mode}")
    print(f"{'='*70}")

    sweep_thresholds = [0.4, 0.5, 0.6, 0.70, 0.80]
    sweep_min_frames = [5, 8, 10, 12, 15]
    sweep_window_sizes = [None, 10, 15, 20]

    augmented_sessions = []
    for sd in loaded:
        frame_ts = pd.to_numeric(sd["model_df"]["timestamp"], errors="coerce").fillna(0.0).values
        bpm_at_frame = interpolate_bpm_to_frames(sd["bpm_consensus"], frame_ts)
        bn = compute_bpm_norm(bpm_at_frame, frame_ts)
        aug_df = enrich_rppg_formulas(sd["model_df"], bn)
        augmented_sessions.append({
            "gt_df": sd["gt_df"],
            "model_df": aug_df,
        })

    # Build sweep column list: all base formulas + their _rppg augmented versions
    present_base = [c for c in _BASE_FORMULA_COLS
                    if any(c in asd["model_df"].columns for asd in augmented_sessions)]
    all_sweep_cols = []
    for c in present_base:
        all_sweep_cols.append(c)
        rppg_col = f"{c}_rppg"
        if any(rppg_col in asd["model_df"].columns for asd in augmented_sessions):
            all_sweep_cols.append(rppg_col)

    best_per_formula = []

    for col in all_sweep_cols:
        base_name = col.replace("_rppg", "")
        base_label = _BASE_FORMULA_LABELS.get(base_name, _FORMULA_LABELS.get(base_name, base_name))
        label = f"{base_label}+rppg" if col.endswith("_rppg") else base_label

        def eval_combined(thresh, mf, ws, current_col=col):
            tot_tp = tot_fp = tot_gt_caught = tot_n_gt = 0
            for asd in augmented_sessions:
                if current_col in asd["model_df"].columns:
                    tp, fp, gt_caught, n_gt = compute_metrics(
                        asd["model_df"], asd["gt_df"], thresh, mf,
                        PAD_START, PAD_END, ws, FILL_RATIO, current_col,
                    )
                    tot_tp += tp
                    tot_fp += fp
                    tot_gt_caught += gt_caught
                    tot_n_gt += n_gt
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
        f"rPPG Formula Comparison — {mode} ({len(loaded)} sessions)"
    )

    # ── 5b. Delta table: F1 improvement per formula from rPPG ──────────
    formula_scores = {}
    for r in best_per_formula:
        formula_scores[r["formula"]] = r["f1"]

    base_scores = {r["formula"]: r["f1"] for r in best_per_formula
                   if not r["formula"].endswith("_rppg")}
    rppg_scores = {r["formula"].replace("_rppg", ""): r["f1"] for r in best_per_formula
                   if r["formula"].endswith("_rppg")}

    print(f"\n{'='*70}")
    print(f"  rPPG Benefit per Formula — {mode}")
    print(f"{'='*70}")
    print(f"  {'Formula':<10}  {'Base F1':>8}  {'+ rPPG F1':>10}  {'ΔF1':>8}  {'Change':>8}")

    best_delta = -999
    best_delta_formula = ""
    for col in present_base:
        base_f1 = base_scores.get(col, 0.0)
        rppg_f1 = rppg_scores.get(col, 0.0)
        delta = rppg_f1 - base_f1
        pct = 100 * delta / base_f1 if base_f1 > 0 else 0
        marker = ""
        if delta > best_delta:
            best_delta = delta
            best_delta_formula = col
        print(f"  {col:<10}  {base_f1:>8.4f}  {rppg_f1:>10.4f}  {delta:>+8.4f}  {pct:>+7.1f}%{marker}")

    if best_delta_formula:
        print(f"\n  → Largest rPPG benefit: {best_delta_formula} "
              f"(ΔF1 = {best_delta:+.4f})")
    print()

    sys.stdout.flush()

    # ── 6. Generate plots ──────────────────────────────────────────────
    print(f"\n  Generating plots...", flush=True)

    for sd in loaded:
        plot_path = os.path.join(output_dir, f"{sd['tag']}_rppg_timeline.png")
        generate_rppg_timeline(sd, plot_path)

    dist_path = os.path.join(output_dir, f"rppg_bpm_distribution_{mode}.png")
    generate_bpm_distribution(all_fear, all_nonfear, bpm_stats, delta_stats,
                              dist_path, title=f"({mode})")

    corr_path = os.path.join(output_dir, f"rppg_correlation_{mode}.png")
    generate_correlation_plot(corr_results, formula_scores, corr_path, title=f"({mode})")

    # ── 7. Save event stats CSV ────────────────────────────────────────
    if all_fear:
        events_df = pd.DataFrame(all_fear)
        events_path = os.path.join(output_dir, f"rppg_event_stats_{mode}.csv")
        events_df.to_csv(events_path, index=False)
        print(f"  [rppg-events] → {events_path}")

    if formula_scores:
        scores_df = pd.DataFrame([
            {"formula": k, "f1": v} for k, v in formula_scores.items()
        ])
        scores_path = os.path.join(output_dir, f"rppg_formula_comparison_{mode}.csv")
        scores_df.to_csv(scores_path, index=False)
        print(f"  [rppg-formulas] → {scores_path}")

    # ── 8. Phase 2 Extensions ────────────────────────────────────────
    any_ext = any(ext_flags.get(k) for k in ("facedet", "peak", "multiwin",
                                              "zscore", "variability",
                                              "peralgo", "shift", "pos30",
                                              "posaug", "fullsweep",
                                              "fullsweep16", "validate_rppg"))
    if any_ext:
        print(f"\n\n{'='*70}")
        print(f"  Phase 2 Extended Analysis — {mode}")
        print(f"{'='*70}")

        if ext_flags.get("facedet"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2a. Face Detection: FER vs rPPG ---")
            print(f"{'='*70}")
            facedet_cmp = compute_face_detection_comparison(loaded, _CACHE_DIR)
            print_face_detection_comparison(facedet_cmp)

        if ext_flags.get("peak"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2b. Post-Event Peak Analysis (Autonomic Delay) ---")
            print(f"{'='*70}")
            all_peak_events = []
            for sd in loaded:
                peaks = compute_event_peak_stats(sd["bpm_consensus"], sd["gt_df"])
                for p in peaks:
                    p["session"] = sd["tag"]
                all_peak_events.extend(peaks)
            peak_agg = compute_peak_aggregate_stats(all_peak_events, all_fear)
            print_peak_analysis(peak_agg, mode)

        if ext_flags.get("multiwin"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2c. Multi-Window BPM Comparison ---")
            print(f"{'='*70}")
            mw_summary = run_multi_window_comparison(loaded, analyzer)
            print_multi_window_comparison(mw_summary)

        if ext_flags.get("zscore"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2d. Per-Subject Z-Score Normalization ---")
            print(f"{'='*70}")
            zs_result = compute_zscore_analysis(loaded)
            print_zscore_analysis(zs_result)

        if ext_flags.get("variability"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2e. Approximate BPM Variability ---")
            print(f"{'='*70}")
            var_result = compute_variability_comparison(loaded)
            print_variability_comparison(var_result)

        if ext_flags.get("peralgo"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2f. Per-Algorithm Fear Correlation ---")
            print(f"{'='*70}")
            run_per_algorithm_evaluation(loaded, analyzer)

        if ext_flags.get("shift"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2g. Temporal Shift of Fear Windows ---")
            print(f"{'='*70}")
            run_temporal_shift_sweep(loaded)

        if ext_flags.get("pos30"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2h. POS-only @ 30s Window ---")
            print(f"{'='*70}")
            run_pos_30s_evaluation(loaded, analyzer)

        if ext_flags.get("posaug"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2i. POS-only BPM Augmentation Sweep ---")
            print(f"{'='*70}")
            run_pos_augmentation_sweep(loaded, analyzer)

        if ext_flags.get("fullsweep"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2j. Comprehensive Controlled Sweep ---")
            print(f"{'='*70}")
            run_comprehensive_sweep(loaded, analyzer)

        if ext_flags.get("fullsweep16"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2k. Full 16 Window/Step Sweep ---")
            print(f"{'='*70}")
            run_comprehensive_sweep(loaded, analyzer, windows=_SWEEP_WINDOWS_16)

        if ext_flags.get("validate_rppg"):
            print(f"\n\n{'='*70}")
            print(f"  --- 2l. rPPG Statistical Validation Suite ---")
            print(f"{'='*70}")
            run_rppg_validation(loaded, analyzer)

    print(f"\n  Log → {log_path}")
    print(f"  Done — {mode} evaluation complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="rPPG × FER fear correlation analysis")
    parser.add_argument("--mode", choices=["ALL", "GAMEPLAY", "BOTH"], default="BOTH",
                        help="Evaluation group (default: BOTH)")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Re-extract ROI even if cache exists")
    parser.add_argument("--ext-facedet", action="store_true",
                        help="Phase 2: face detection comparison (FER vs rPPG)")
    parser.add_argument("--ext-peak", action="store_true",
                        help="Phase 2: post-event peak analysis (autonomic delay)")
    parser.add_argument("--ext-multiwin", action="store_true",
                        help="Phase 2: multi-window BPM comparison (15s/10s/5s)")
    parser.add_argument("--ext-zscore", action="store_true",
                        help="Phase 2: per-subject z-score normalization")
    parser.add_argument("--ext-variability", action="store_true",
                        help="Phase 2: approximate BPM variability analysis")
    parser.add_argument("--ext-peralgo", action="store_true",
                        help="Phase 2: per-algorithm fear correlation (POS, POS+CHROM, CONSENSUS)")
    parser.add_argument("--ext-shift", action="store_true",
                        help="Phase 2: temporal shift of fear windows [0,3,5,8,10,15]s")
    parser.add_argument("--ext-pos30", action="store_true",
                        help="Phase 2: POS-only at 30s window")
    parser.add_argument("--ext-posaug", action="store_true",
                        help="Phase 2: POS-only BPM augmentation sweep (coeff × bpm_norm)")
    parser.add_argument("--ext-fullsweep", action="store_true",
                        help="Phase 2: comprehensive sweep (3 algos × 4 windows × 3 methods + augmentation)")
    parser.add_argument("--ext-fullsweep16", action="store_true",
                        help="Phase 2: full 16 window/step sweep (4 win × 4 step × 3 algos)")
    parser.add_argument("--ext-validate-rppg", action="store_true",
                        help="Phase 2: rPPG statistical validation (permutation, leave-k-out, weighting)")
    parser.add_argument("--ext-all", action="store_true",
                        help="Phase 2: run all extensions")
    args = parser.parse_args()

    ext_flags = {
        "facedet": args.ext_facedet or args.ext_all,
        "peak": args.ext_peak or args.ext_all,
        "multiwin": args.ext_multiwin or args.ext_all,
        "zscore": args.ext_zscore or args.ext_all,
        "variability": args.ext_variability or args.ext_all,
        "peralgo": args.ext_peralgo or args.ext_all,
        "shift": args.ext_shift or args.ext_all,
        "pos30": args.ext_pos30 or args.ext_all,
        "posaug": args.ext_posaug or args.ext_all,
        "fullsweep": args.ext_fullsweep or args.ext_all,
        "fullsweep16": args.ext_fullsweep16 or args.ext_all,
        "validate_rppg": args.ext_validate_rppg or args.ext_all,
    }

    log_dir = os.path.join(_REPO_ROOT, "Pipeline", "logs", "comparisons")
    run_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    modes = ["ALL", "GAMEPLAY"] if args.mode == "BOTH" else [args.mode]

    for m in modes:
        log_path = os.path.join(run_dir, f"evaluate_rppg_{m}.txt")
        with open(log_path, "w") as fh:
            orig, sys.stdout = sys.stdout, _Tee(sys.stdout, fh)
            try:
                run_evaluate_rppg(m, output_dir=run_dir,
                                  force_reextract=args.force_reextract,
                                  ext_flags=ext_flags)
            finally:
                sys.stdout = orig

    print(f"\nAll results → {run_dir}")


if __name__ == "__main__":
    main()
