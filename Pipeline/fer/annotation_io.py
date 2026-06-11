"""File I/O utilities for the La Façade Fissuréе annotation pipeline.

Pattern: [Repository] — encapsulates all data access: CSV loading,
    field parsing, output path resolution, rPPG sidecar loading.
"""

import csv
import sys
from pathlib import Path

from fer.annotation_events import REQUIRED, FORMULAS, RPPG_ALGOS


def parse_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def load_csv(path):
    """Load CSV and return list of dicts with parsed numeric fields."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            print(f"ERROR: CSV is missing columns: {missing}")
            sys.exit(1)
        for r in reader:
            rows.append({
                "frame":            parse_int(r["frame"]),
                "timestamp":        parse_float(r["timestamp"]),
                "hs_dominant":      r.get("hs_dominant", "Neutral").strip(),
                "hs_fear":          parse_float(r.get("hs_fear")),
                "hs_surprise":      parse_float(r.get("hs_surprise")),
                "hs_anger":         parse_float(r.get("hs_anger")),
                "hs_arousal":       parse_float(r.get("hs_arousal")),
                "mp_startle_score": parse_float(r.get("mp_startle_score")),
                "mp_tension":       parse_float(r.get("mp_tension")),
                "mp_ctx_tag":       r.get("mp_ctx_tag", "---").strip(),
                "composite_fear":   parse_float(r.get("composite_fear")),
                "agreement_tag":    r.get("agreement_tag", "").strip(),
                "veto_tag":         r.get("veto_tag", "---").strip(),
                "mp_face_detected":   parse_int(r.get("mp_face_detected")),
                "hs_face_detected":   parse_int(r.get("hs_face_detected")),
                "event_status":       r.get("event_status", "IDLE").strip(),
                "smoothed_composite": parse_float(r.get("smoothed_composite")),
            })
    return rows


def recompute_composite_fear(rows, formula_name):
    """Recompute composite_fear on each row using the named formula.
    Updates rows in place. Skips 'original' (uses pre-computed column)."""
    if formula_name == "original":
        return
    fn = FORMULAS[formula_name]
    for r in rows:
        r["composite_fear"] = fn(r)
    print(f"  Recomputed composite_fear with formula: {formula_name}")


def _output_dir(csv_path):
    """Resolve annotation output directory: logs/annotations/
    CSV lives in logs/sessions/<session>/, so 3 levels up."""
    out_dir = csv_path.parent.parent.parent / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_scene_cuts(cuts_file, auto_cuts=None, dedup_tolerance=1.0):
    """Load manual cuts from file and merge with auto-detected cuts.

    cuts_file: path to .cuts.txt (one timestamp per line, or comma-separated)
    """
    manual_cuts = []
    if cuts_file and Path(cuts_file).exists():
        text = Path(cuts_file).read_text().strip()
        for token in text.replace("\n", ",").split(","):
            token = token.strip()
            if token:
                try:
                    manual_cuts.append(float(token))
                except ValueError:
                    pass
        print(f"  Manual cuts loaded: {len(manual_cuts)} from {cuts_file}")

    all_cuts = sorted(set((auto_cuts or []) + manual_cuts))

    # Deduplicate within tolerance
    if len(all_cuts) <= 1:
        return all_cuts
    deduped = [all_cuts[0]]
    for c in all_cuts[1:]:
        if c - deduped[-1] > dedup_tolerance:
            deduped.append(c)
    return deduped


def load_rppg_csv(path):
    """Load _rppg.csv sidecar into per-algorithm BPM timeseries.

    Returns dict: {algo_name: [(t_center, bpm_smoothed), ...]}
    """
    data = {a: [] for a in RPPG_ALGOS}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                algo = row.get("algorithm", "").strip()
                if algo not in data:
                    continue
                try:
                    t = float(row["t_center"])
                    bpm = float(row["bpm_smoothed"])
                    data[algo].append((t, bpm))
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        print(f"  WARNING: Could not load rPPG CSV: {e}")
    for algo in RPPG_ALGOS:
        data[algo].sort(key=lambda x: x[0])
    return data


def rppg_cluster_stats(rppg_data, start_s, end_s, pre_window_s=30, pre_gap_s=5):
    """Compute mean BPM during cluster and in the pre-window for each algorithm.

    pre-window = [start_s - pre_window_s,  start_s - pre_gap_s]
    Returns dict with keys like rppg_bpm_chrom, rppg_delta_chrom, etc.
    (all empty strings if no data available)
    """
    result = {}
    pre_start = start_s - pre_window_s
    pre_end   = start_s - pre_gap_s

    for algo in RPPG_ALGOS:
        series = rppg_data.get(algo, [])
        during = [bpm for t, bpm in series if start_s <= t <= end_s and bpm > 0]
        pre    = [bpm for t, bpm in series if pre_start <= t <= pre_end and bpm > 0]

        mean_during = sum(during) / len(during) if during else None
        mean_pre    = sum(pre)    / len(pre)    if pre    else None

        result[f"rppg_bpm_{algo}"]   = f"{mean_during:.1f}" if mean_during else ""
        result[f"rppg_pre_bpm_{algo}"] = f"{mean_pre:.1f}" if mean_pre else ""
        if mean_during and mean_pre:
            result[f"rppg_delta_{algo}"] = f"{mean_during - mean_pre:+.1f}"
        else:
            result[f"rppg_delta_{algo}"] = ""

    return result
