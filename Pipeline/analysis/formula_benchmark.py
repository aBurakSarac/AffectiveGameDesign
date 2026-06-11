"""
Formula Benchmark — Composite Fear Formula × Seed × VETO Sweep
===============================================================
Pattern: [Utility / Reporter] — post-hoc batch evaluator; reads existing CSV logs,
applies all formula × seed × VETO combinations, and ranks results by F1 against GT.

Post-hoc evaluation of composite fear formulas on existing fusion CSVs.
Runs all combinations and ranks by F1 against ground truth annotations.

Usage:
    python formula_benchmark.py --csv <fusion_csv> --gt <gt_csv>       # Full benchmark
    python formula_benchmark.py --csv <fusion_csv>                      # Stats only
    python formula_benchmark.py --csv <fusion_csv> --gt <gt> --plot     # + timeline plot
    python formula_benchmark.py --list                                  # Print definitions
"""

import argparse
import csv
import math
import os
from collections import OrderedDict

# ═══════════════════════════════════════════════════════════════════════════════
# FORMULA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _clamp(v):
    return min(1.0, max(0.0, v))


FORMULAS = OrderedDict([
    ("F0_CURRENT",       lambda r: 0.60 * r["hs_arousal"] + 0.40 * r["mp_tension"]),
    ("F1_FEAR_TENSION",  lambda r: 0.60 * r["hs_fear"] + 0.40 * r["mp_tension"]),
    ("F2_FEAR_SURPRISE", lambda r: 0.40 * r["hs_fear"] + 0.20 * r["hs_surprise"] + 0.40 * r["mp_tension"]),
    ("F3_FEAR_HEDGED",   lambda r: 0.50 * r["hs_fear"] + 0.20 * r["hs_arousal"] + 0.30 * r["mp_tension"]),
    ("F4_HARD_VETO",     lambda r: (0.60 * r["hs_arousal"] + 0.40 * r["mp_tension"])
                                    * (1.0 if r["hs_anger"] <= r["hs_fear"] else 0.0)),
    ("F5_FEAR_STARTLE",  lambda r: 0.50 * r["hs_fear"] + 0.30 * r["mp_tension"]
                                    + 0.20 * _clamp(r["mp_startle_score"] / 10.0)),
    ("F6_SOFT_VETO",     lambda r: (0.60 * r["hs_fear"] + 0.40 * r["mp_tension"])
                                    * max(0.0, 1.0 - r["hs_anger"])),
])

# ═══════════════════════════════════════════════════════════════════════════════
# SEED STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

SEED_EMOTIONS = {"Fear", "Surprise"}

SEEDS = OrderedDict([
    ("S0_DOMINANT",     lambda r: r["hs_dominant"] in SEED_EMOTIONS),
    ("S1_fear>0.10",    lambda r: r["hs_fear"] > 0.10),
    ("S1_fear>0.15",    lambda r: r["hs_fear"] > 0.15),
    ("S1_fear>0.20",    lambda r: r["hs_fear"] > 0.20),
    ("S1_fear>0.30",    lambda r: r["hs_fear"] > 0.30),
    ("S1_fear>0.40",    lambda r: r["hs_fear"] > 0.40),
    ("S1_fear>0.50",    lambda r: r["hs_fear"] > 0.50),
    ("S2_f>0.10|s>0.20", lambda r: r["hs_fear"] > 0.10 or r["hs_surprise"] > 0.20),
    ("S2_f>0.15|s>0.25", lambda r: r["hs_fear"] > 0.15 or r["hs_surprise"] > 0.25),
    ("S2_f>0.20|s>0.30", lambda r: r["hs_fear"] > 0.20 or r["hs_surprise"] > 0.30),
])

# ═══════════════════════════════════════════════════════════════════════════════
# VETO STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

VETOS = OrderedDict([
    ("V0_NO_VETO",      lambda score, r: score),
    ("V1_CURRENT_VETO", lambda score, r: 0.0 if r["veto_tag"] not in ("---", "") else score),
    ("V2_ANGER_HARD",   lambda score, r: score * (1.0 if r["hs_anger"] <= r["hs_fear"] else 0.0)),
    ("V3_ANGER_SOFT",   lambda score, r: score * max(0.0, 1.0 - r["hs_anger"])),
    ("V4_CONC_DISCOUNT", lambda score, r: score * 0.5
                          if r["mp_ctx_tag"] in ("CONC", "---") else score),
])

# ═══════════════════════════════════════════════════════════════════════════════
# CSV LOADING
# ═══════════════════════════════════════════════════════════════════════════════

FLOAT_COLS = [
    "timestamp", "composite_fear", "hs_arousal", "hs_fear", "hs_surprise",
    "hs_anger", "hs_contempt", "hs_disgust", "hs_happiness", "hs_neutral",
    "hs_sadness", "mp_tension", "mp_startle_score", "hs_dominant_score",
]

INT_COLS = ["frame", "mp_face_detected", "hs_face_detected"]


def load_fusion_csv(path):
    """Load fusion CSV into list of dicts with parsed numeric fields."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = dict(raw)
            for col in FLOAT_COLS:
                try:
                    row[col] = float(row.get(col, 0) or 0)
                except (ValueError, TypeError):
                    row[col] = 0.0
            for col in INT_COLS:
                try:
                    row[col] = int(float(row.get(col, 0) or 0))
                except (ValueError, TypeError):
                    row[col] = 0
            # Ensure string fields have defaults
            row.setdefault("hs_dominant", "Neutral")
            row.setdefault("mp_ctx_tag", "---")
            row.setdefault("veto_tag", "---")
            row.setdefault("agreement_tag", "")
            rows.append(row)
    return rows


def load_gt_csv(path):
    """Load ground truth annotation CSV (v1 format with verdict column)."""
    events = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            eid = raw.get("event_id", "")
            verdict = raw.get("verdict", "").strip()
            if not eid or not verdict:
                continue
            try:
                start = float(raw.get("start_s") or raw.get("cluster_start") or 0)
                end = float(raw.get("end_s") or raw.get("cluster_end") or 0)
            except (ValueError, TypeError):
                continue
            if start == 0 and end == 0:
                continue
            events.append({
                "event_id": eid,
                "verdict": verdict,
                "start_s": start,
                "end_s": end,
                "context": raw.get("context", ""),
            })
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# FLOOD-FILL CLUSTERING (generalized)
# ═══════════════════════════════════════════════════════════════════════════════

def _padded_window(rows, s_idx, e_idx, pad_s=3.0):
    """Return (pad_s_idx, pad_e_idx) padded by +-pad_s seconds."""
    t_lo = rows[s_idx]["timestamp"] - pad_s
    t_hi = rows[e_idx]["timestamp"] + pad_s
    pad_s_idx = s_idx
    pad_e_idx = e_idx
    for i in range(s_idx - 1, -1, -1):
        if rows[i]["timestamp"] >= t_lo:
            pad_s_idx = i
        else:
            break
    for i in range(e_idx + 1, len(rows)):
        if rows[i]["timestamp"] <= t_hi:
            pad_e_idx = i
        else:
            break
    return pad_s_idx, pad_e_idx


def flood_fill_generic(rows, seed_fn, score_key, gap_tolerance=1.0,
                       min_frames=3, sustained_thr=0.15):
    """Generalized flood-fill clustering.

    Args:
        rows: list of dicts with at least 'timestamp' and score_key
        seed_fn: callable(row) -> bool, which frames can seed clusters
        score_key: string key for the composite score in each row
        gap_tolerance: max seconds between consecutive seeds
        min_frames: minimum seed frames per cluster
        sustained_thr: threshold for sustained_pct computation

    Returns:
        list of cluster dicts with stats
    """
    # 1. Collect seed indices
    seed_indices = sorted(i for i, r in enumerate(rows) if seed_fn(r))
    if not seed_indices:
        return []

    # 2. Chain into runs by gap tolerance
    runs = []
    current_run = [seed_indices[0]]
    for idx in seed_indices[1:]:
        gap = rows[idx]["timestamp"] - rows[current_run[-1]]["timestamp"]
        if gap <= gap_tolerance:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]
    runs.append(current_run)

    # 3. Build clusters
    clusters = []
    for indices in runs:
        if len(indices) < min_frames:
            continue

        s_idx = indices[0]
        e_idx = indices[-1]
        peak_idx = max(indices, key=lambda i: rows[i][score_key])

        cluster_rows = rows[s_idx:e_idx + 1]
        n_total = len(cluster_rows)

        peak_score = rows[peak_idx][score_key]
        duration = rows[e_idx]["timestamp"] - rows[s_idx]["timestamp"]

        # Sustained scoring on padded window
        pad_s, pad_e = _padded_window(rows, s_idx, e_idx)
        pad_cluster = rows[pad_s:pad_e + 1]
        n_pad = len(pad_cluster)
        n_sustained = sum(1 for r in pad_cluster if r[score_key] >= sustained_thr)
        sustained_pct = (n_sustained / n_pad * 100) if n_pad > 0 else 0.0

        # Dominant emotion at peak
        dom_at_peak = rows[peak_idx].get("hs_dominant", "?")

        clusters.append({
            "s_idx": s_idx,
            "e_idx": e_idx,
            "start_t": rows[s_idx]["timestamp"],
            "end_t": rows[e_idx]["timestamp"],
            "peak_t": rows[peak_idx]["timestamp"],
            "peak_score": peak_score,
            "sustained_pct": sustained_pct,
            "duration": duration,
            "n_seed_frames": len(indices),
            "n_total_frames": n_total,
            "dom_at_peak": dom_at_peak,
            "auto_confidence": "HIGH" if sustained_pct >= 25.0 else "LOW",
        })

    return clusters


# ═══════════════════════════════════════════════════════════════════════════════
# GT EVENT MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

POSITIVE_VERDICTS = {"Fear", "Stress"}


def temporal_iou(c_start, c_end, gt_start, gt_end):
    """Intersection-over-union of two time intervals."""
    inter_start = max(c_start, gt_start)
    inter_end = min(c_end, gt_end)
    inter = max(0.0, inter_end - inter_start)
    union = max(0.0, (c_end - c_start) + (gt_end - gt_start) - inter)
    return inter / union if union > 0 else 0.0


def match_clusters_to_gt(clusters, gt_events, iou_threshold=0.3,
                         confidence_filter=True):
    """Match detected clusters to GT events, compute P/R/F1.

    When confidence_filter=True, only HIGH-confidence clusters count as
    detections. This makes the formula matter: different formulas produce
    different sustained_pct → different confidence → different F1.

    Returns dict with tp, fp, fn, precision, recall, f1, and match details.
    """
    if confidence_filter:
        active_clusters = [c for c in clusters if c["auto_confidence"] == "HIGH"]
    else:
        active_clusters = clusters

    gt_matched = set()
    tp = 0
    fp = 0
    matches = []

    for cl in active_clusters:
        best_iou = 0.0
        best_gt = None
        best_gt_idx = -1

        for gi, gt in enumerate(gt_events):
            iou = temporal_iou(cl["start_t"], cl["end_t"], gt["start_s"], gt["end_s"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt is not None:
            verdict = best_gt["verdict"]
            if verdict in POSITIVE_VERDICTS:
                tp += 1
                gt_matched.add(best_gt_idx)
                matches.append(("TP", cl, best_gt, best_iou))
            else:
                fp += 1
                matches.append(("FP_matched", cl, best_gt, best_iou))
        else:
            fp += 1
            matches.append(("FP_unmatched", cl, best_gt, best_iou))

    # Count FN: positive GT events not matched
    fn = 0
    for gi, gt in enumerate(gt_events):
        if gt["verdict"] in POSITIVE_VERDICTS and gi not in gt_matched:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_clusters": len(clusters),
        "n_active": len(active_clusters),
        "matches": matches,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_formulas(rows):
    """Compute all formula outputs and store in each row."""
    for row in rows:
        for fid, fn in FORMULAS.items():
            row[f"cf_{fid}"] = _clamp(fn(row))


def apply_veto(rows, formula_id, veto_id, veto_fn):
    """Apply a VETO to a formula's output, store as new key."""
    key_in = f"cf_{formula_id}"
    key_out = f"cfv_{formula_id}_{veto_id}"
    for row in rows:
        row[key_out] = _clamp(veto_fn(row[key_in], row))
    return key_out


def run_phase_a(rows, gt_events=None):
    """Phase A: Formula × Seed (no VETO). Returns ranked results."""
    results = []

    for fid in FORMULAS:
        score_key = f"cf_{fid}"
        for sid, seed_fn in SEEDS.items():
            clusters = flood_fill_generic(rows, seed_fn, score_key)

            entry = {
                "formula": fid,
                "seed": sid,
                "n_clusters": len(clusters),
                "mean_peak": (sum(c["peak_score"] for c in clusters) / len(clusters)
                              if clusters else 0),
                "n_high": sum(1 for c in clusters if c["auto_confidence"] == "HIGH"),
            }

            if gt_events:
                metrics = match_clusters_to_gt(clusters, gt_events)
                entry.update(metrics)

            results.append(entry)

    # Sort by F1 if GT available, else by n_high descending
    if gt_events:
        results.sort(key=lambda r: (-r["f1"], -r["precision"], r["fp"]))
    else:
        results.sort(key=lambda r: (-r["n_high"], -r["mean_peak"]))

    return results


def run_phase_b(rows, top_formulas, top_seeds, gt_events=None):
    """Phase B: Top formulas × Top seeds × all VETOs."""
    results = []

    for fid in top_formulas:
        for sid, seed_fn in [(s, SEEDS[s]) for s in top_seeds]:
            for vid, veto_fn in VETOS.items():
                score_key = apply_veto(rows, fid, vid, veto_fn)
                clusters = flood_fill_generic(rows, seed_fn, score_key)

                entry = {
                    "formula": fid,
                    "seed": sid,
                    "veto": vid,
                    "n_clusters": len(clusters),
                    "mean_peak": (sum(c["peak_score"] for c in clusters) / len(clusters)
                                  if clusters else 0),
                    "n_high": sum(1 for c in clusters if c["auto_confidence"] == "HIGH"),
                }

                if gt_events:
                    metrics = match_clusters_to_gt(clusters, gt_events)
                    entry.update(metrics)

                results.append(entry)

    if gt_events:
        results.sort(key=lambda r: (-r["f1"], -r["precision"], r["fp"]))
    else:
        results.sort(key=lambda r: (-r["n_high"], -r["mean_peak"]))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME-LEVEL STATS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_frame_stats(rows):
    """Per-formula frame-level statistics, split by emotion."""
    stats = {}
    for fid in FORMULAS:
        key = f"cf_{fid}"
        all_vals = [r[key] for r in rows]
        anger_vals = [r[key] for r in rows if r["hs_dominant"] == "Anger"]
        fear_vals = [r[key] for r in rows if r["hs_dominant"] == "Fear"]
        surprise_vals = [r[key] for r in rows if r["hs_dominant"] == "Surprise"]
        neutral_vals = [r[key] for r in rows if r["hs_dominant"] == "Neutral"]

        def _stats(vals):
            if not vals:
                return {"mean": 0, "std": 0, "max": 0, "n": 0}
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            return {"mean": mean, "std": math.sqrt(var), "max": max(vals), "n": n}

        stats[fid] = {
            "all": _stats(all_vals),
            "anger": _stats(anger_vals),
            "fear": _stats(fear_vals),
            "surprise": _stats(surprise_vals),
            "neutral": _stats(neutral_vals),
        }
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# PRINTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_frame_stats(stats):
    print("\n" + "=" * 90)
    print("FRAME-LEVEL STATISTICS — Per Formula, By Dominant Emotion")
    print("=" * 90)
    print(f"{'Formula':<20} {'All':>12} {'Anger':>12} {'Fear':>12} "
          f"{'Surprise':>12} {'Neutral':>12}")
    print(f"{'':20} {'mean±std':>12} {'mean(n)':>12} {'mean(n)':>12} "
          f"{'mean(n)':>12} {'mean(n)':>12}")
    print("-" * 90)
    for fid, s in stats.items():
        a = s["all"]
        ang = s["anger"]
        fear = s["fear"]
        sur = s["surprise"]
        neu = s["neutral"]
        print(f"{fid:<20} "
              f"{a['mean']:.3f}±{a['std']:.3f} "
              f"{ang['mean']:.3f}({ang['n']:>4}) "
              f"{fear['mean']:.3f}({fear['n']:>4}) "
              f"{sur['mean']:.3f}({sur['n']:>4}) "
              f"{neu['mean']:.3f}({neu['n']:>4})")


def print_phase_a(results, top_n=15):
    has_gt = "f1" in results[0] if results else False
    print("\n" + "=" * 100)
    print("PHASE A — Formula × Seed (no VETO)")
    print("=" * 100)

    if has_gt:
        print(f"{'#':<4} {'Formula':<20} {'Seed':<20} "
              f"{'All':>4} {'HIGH':>4} {'P':>6} {'R':>6} {'F1':>6} "
              f"{'TP':>4} {'FP':>4} {'FN':>4} {'MnPk':>6}")
        print("-" * 105)
        for i, r in enumerate(results[:top_n]):
            print(f"{i+1:<4} {r['formula']:<20} {r['seed']:<20} "
                  f"{r['n_clusters']:>4} {r.get('n_active', r['n_high']):>4} "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
                  f"{r['f1']:>6.3f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
                  f"{r['mean_peak']:>6.3f}")
    else:
        print(f"{'#':<4} {'Formula':<20} {'Seed':<20} "
              f"{'Clust':>5} {'HIGH':>5} {'MnPk':>6}")
        print("-" * 100)
        for i, r in enumerate(results[:top_n]):
            print(f"{i+1:<4} {r['formula']:<20} {r['seed']:<20} "
                  f"{r['n_clusters']:>5} {r['n_high']:>5} {r['mean_peak']:>6.3f}")


def print_phase_b(results, top_n=15):
    has_gt = "f1" in results[0] if results else False
    print("\n" + "=" * 115)
    print("PHASE B — Top Formulas × Top Seeds × VETO")
    print("=" * 115)

    if has_gt:
        print(f"{'#':<4} {'Formula':<20} {'Seed':<20} {'VETO':<18} "
              f"{'All':>4} {'HIGH':>4} {'P':>6} {'R':>6} {'F1':>6} "
              f"{'TP':>4} {'FP':>4} {'FN':>4}")
        print("-" * 120)
        for i, r in enumerate(results[:top_n]):
            print(f"{i+1:<4} {r['formula']:<20} {r['seed']:<20} {r['veto']:<18} "
                  f"{r['n_clusters']:>4} {r.get('n_active', r['n_high']):>4} "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} "
                  f"{r['f1']:>6.3f} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}")
    else:
        print(f"{'#':<4} {'Formula':<20} {'Seed':<20} {'VETO':<18} "
              f"{'Clust':>5} {'HIGH':>5} {'MnPk':>6}")
        print("-" * 115)
        for i, r in enumerate(results[:top_n]):
            print(f"{i+1:<4} {r['formula']:<20} {r['seed']:<20} {r['veto']:<18} "
                  f"{r['n_clusters']:>5} {r['n_high']:>5} {r['mean_peak']:>6.3f}")


def print_cluster_detail(rows, best_result, gt_events=None):
    """Print per-cluster detail for the best combo vs F0 baseline."""
    best_fid = best_result["formula"]
    best_sid = best_result["seed"]
    best_vid = best_result.get("veto")

    # Determine score key
    if best_vid:
        score_key = f"cfv_{best_fid}_{best_vid}"
    else:
        score_key = f"cf_{best_fid}"
    baseline_key = "cf_F0_CURRENT"

    seed_fn = SEEDS[best_sid]
    clusters = flood_fill_generic(rows, seed_fn, score_key)

    print(f"\n{'=' * 95}")
    print(f"CLUSTER DETAIL — Best: {best_fid} + {best_sid}"
          + (f" + {best_vid}" if best_vid else ""))
    print(f"{'=' * 95}")
    print(f"{'#':<4} {'Start':>7} {'End':>7} {'Dur':>5} {'Peak':>6} "
          f"{'F0_Pk':>6} {'Sust%':>6} {'Conf':>5} {'DomPk':<10} {'GT':>10}")
    print("-" * 95)

    for i, cl in enumerate(clusters):
        # Get F0 baseline peak in same window
        f0_peak = max((r[baseline_key] for r in rows[cl["s_idx"]:cl["e_idx"] + 1]),
                      default=0)

        # Match to GT
        gt_label = "---"
        if gt_events:
            for gt in gt_events:
                iou = temporal_iou(cl["start_t"], cl["end_t"], gt["start_s"], gt["end_s"])
                if iou >= 0.3:
                    gt_label = gt["verdict"]
                    break

        print(f"{i+1:<4} {cl['start_t']:>7.1f} {cl['end_t']:>7.1f} "
              f"{cl['duration']:>5.1f} {cl['peak_score']:>6.3f} "
              f"{f0_peak:>6.3f} {cl['sustained_pct']:>6.1f} "
              f"{cl['auto_confidence']:>5} {cl['dom_at_peak']:<10} {gt_label:>10}")


def print_list():
    """Print all formula, seed, and VETO definitions."""
    print("\n=== FORMULAS ===")
    descs = {
        "F0_CURRENT":       "0.60*hs_arousal + 0.40*mp_tension",
        "F1_FEAR_TENSION":  "0.60*hs_fear + 0.40*mp_tension",
        "F2_FEAR_SURPRISE": "0.40*hs_fear + 0.20*hs_surprise + 0.40*mp_tension",
        "F3_FEAR_HEDGED":   "0.50*hs_fear + 0.20*hs_arousal + 0.30*mp_tension",
        "F4_HARD_VETO":     "F0 * (1 if hs_anger <= hs_fear else 0)",
        "F5_FEAR_STARTLE":  "0.50*hs_fear + 0.30*mp_tension + 0.20*clamp(startle/10)",
        "F6_SOFT_VETO":     "F1 * max(0, 1 - hs_anger)",
    }
    for fid in FORMULAS:
        print(f"  {fid:<20} {descs.get(fid, '?')}")

    print("\n=== SEEDS ===")
    seed_descs = {
        "S0_DOMINANT":       "hs_dominant in {Fear, Surprise}",
        "S1_fear>0.10":      "hs_fear > 0.10",
        "S1_fear>0.15":      "hs_fear > 0.15",
        "S1_fear>0.20":      "hs_fear > 0.20",
        "S1_fear>0.30":      "hs_fear > 0.30",
        "S1_fear>0.40":      "hs_fear > 0.40",
        "S1_fear>0.50":      "hs_fear > 0.50",
        "S2_f>0.10|s>0.20":  "hs_fear > 0.10 or hs_surprise > 0.20",
        "S2_f>0.15|s>0.25":  "hs_fear > 0.15 or hs_surprise > 0.25",
        "S2_f>0.20|s>0.30":  "hs_fear > 0.20 or hs_surprise > 0.30",
    }
    for sid in SEEDS:
        print(f"  {sid:<22} {seed_descs.get(sid, '?')}")

    print("\n=== VETOS ===")
    veto_descs = {
        "V0_NO_VETO":        "score * 1.0 (passthrough)",
        "V1_CURRENT_VETO":   "0 if veto_tag active, else score",
        "V2_ANGER_HARD":     "0 if hs_anger > hs_fear, else score",
        "V3_ANGER_SOFT":     "score * max(0, 1 - hs_anger)",
        "V4_CONC_DISCOUNT":  "score * 0.5 if mp_ctx_tag in {CONC, ---}",
    }
    for vid in VETOS:
        print(f"  {vid:<20} {veto_descs.get(vid, '?')}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(rows, gt_events, best_result, out_path):
    """Multi-panel timeline: formulas overlaid + emotion bands + GT markers."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("[WARN] matplotlib/numpy not available, skipping plot")
        return

    timestamps = np.array([r["timestamp"] for r in rows])

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1.5]})
    fig.patch.set_facecolor("#0a0a14")

    def style_ax(ax):
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#cccccc")
        for sp in ax.spines.values():
            sp.set_color("#444")

    # ── Panel 1: Formula outputs ──
    ax1 = axes[0]
    style_ax(ax1)
    colors = ["#ff4444", "#44ff44", "#4488ff", "#ffaa00", "#ff44ff", "#44ffff", "#ffff44"]
    for i, fid in enumerate(FORMULAS):
        key = f"cf_{fid}"
        vals = np.array([r[key] for r in rows])
        ax1.plot(timestamps, vals, color=colors[i % len(colors)],
                 alpha=0.7, linewidth=0.8, label=fid)
    ax1.set_ylabel("Composite Fear", color="#ccc")
    ax1.legend(fontsize=7, loc="upper right", facecolor="#1a1a2e", edgecolor="#444",
               labelcolor="#ccc")
    ax1.set_title("Formula Comparison — Composite Fear Over Time",
                  color="#eee", fontsize=12)

    # GT event markers
    if gt_events:
        for gt in gt_events:
            color = "#00ff88" if gt["verdict"] in POSITIVE_VERDICTS else "#ff6666"
            ax1.axvspan(gt["start_s"], gt["end_s"], alpha=0.15, color=color)
            ax1.text(gt["start_s"], ax1.get_ylim()[1] * 0.95,
                     f"{gt['event_id']}:{gt['verdict'][:4]}",
                     fontsize=6, color=color, rotation=90, va="top")

    # ── Panel 2: Dominant emotion bands ──
    ax2 = axes[1]
    style_ax(ax2)
    emo_colors = {
        "Anger": "#ff2222", "Fear": "#aa44ff", "Surprise": "#ffdd00",
        "Happiness": "#44ff44", "Neutral": "#666666", "Sadness": "#4466ff",
        "Disgust": "#88aa00", "Contempt": "#aa6600",
    }
    for i, r in enumerate(rows):
        if i == 0:
            continue
        t0 = rows[i - 1]["timestamp"]
        t1 = r["timestamp"]
        c = emo_colors.get(r["hs_dominant"], "#333333")
        ax2.add_patch(Rectangle((t0, 0), t1 - t0, 1, color=c, alpha=0.8))
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("HS Dominant", color="#ccc")
    ax2.set_yticks([])
    # Legend for emotions
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=4, label=e)
               for e, c in emo_colors.items()]
    ax2.legend(handles=handles, fontsize=6, loc="upper right",
               facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ccc", ncol=4)

    # ── Panel 3: Cluster bars for best combo ──
    ax3 = axes[2]
    style_ax(ax3)
    best_fid = best_result["formula"]
    best_sid = best_result["seed"]
    best_vid = best_result.get("veto")
    if best_vid:
        score_key = f"cfv_{best_fid}_{best_vid}"
    else:
        score_key = f"cf_{best_fid}"
    seed_fn = SEEDS[best_sid]

    # Best combo clusters
    clusters_best = flood_fill_generic(rows, seed_fn, score_key)
    for cl in clusters_best:
        ax3.add_patch(Rectangle((cl["start_t"], 0.6), cl["duration"] or 0.1, 0.35,
                                color="#44ff44", alpha=0.7))

    # F0 baseline clusters for comparison
    clusters_f0 = flood_fill_generic(rows, SEEDS["S0_DOMINANT"], "cf_F0_CURRENT")
    for cl in clusters_f0:
        ax3.add_patch(Rectangle((cl["start_t"], 0.1), cl["duration"] or 0.1, 0.35,
                                color="#ff4444", alpha=0.7))

    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Clusters", color="#ccc")
    ax3.set_yticks([0.3, 0.8])
    ax3.set_yticklabels(["F0 baseline", f"Best"], fontsize=8)
    ax3.set_xlabel("Time (s)", color="#ccc")

    label_str = f"Best: {best_fid} + {best_sid}" + (f" + {best_vid}" if best_vid else "")
    ax3.set_title(label_str, color="#aaa", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[PLOT] Saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Formula Benchmark — Composite Fear Formula × Seed × VETO Sweep")
    parser.add_argument("--csv", type=str, help="Path to fusion/mp_hs CSV")
    parser.add_argument("--gt", type=str, help="Path to GT annotation CSV (v1 format)")
    parser.add_argument("--plot", action="store_true", help="Generate timeline plot")
    parser.add_argument("--list", action="store_true", help="Print all definitions")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of top results to print (default: 15)")
    args = parser.parse_args()

    if args.list:
        print_list()
        return

    if not args.csv:
        parser.error("--csv is required (or use --list)")

    # Load data
    print(f"Loading CSV: {args.csv}")
    rows = load_fusion_csv(args.csv)
    print(f"  {len(rows)} frames loaded")

    gt_events = None
    if args.gt:
        print(f"Loading GT: {args.gt}")
        gt_events = load_gt_csv(args.gt)
        n_pos = sum(1 for e in gt_events if e["verdict"] in POSITIVE_VERDICTS)
        n_neg = sum(1 for e in gt_events if e["verdict"] not in POSITIVE_VERDICTS)
        print(f"  {len(gt_events)} events ({n_pos} positive, {n_neg} negative)")

    # Apply all formulas
    print("\nApplying 7 formulas to all frames...")
    apply_formulas(rows)

    # Frame-level stats
    stats = compute_frame_stats(rows)
    print_frame_stats(stats)

    # Phase A: Formula × Seed
    print("\nRunning Phase A: 7 formulas × 10 seeds = 70 combinations...")
    phase_a = run_phase_a(rows, gt_events)
    print_phase_a(phase_a, args.top_n)

    # Extract top 3 unique formulas and top 3 unique seeds
    seen_formulas = []
    seen_seeds = []
    for r in phase_a:
        if r["formula"] not in seen_formulas and len(seen_formulas) < 3:
            seen_formulas.append(r["formula"])
        if r["seed"] not in seen_seeds and len(seen_seeds) < 3:
            seen_seeds.append(r["seed"])
        if len(seen_formulas) >= 3 and len(seen_seeds) >= 3:
            break

    print(f"\nTop 3 formulas: {seen_formulas}")
    print(f"Top 3 seeds:    {seen_seeds}")

    # Phase B: Top × Top × VETO
    print(f"\nRunning Phase B: {len(seen_formulas)} formulas × {len(seen_seeds)} seeds "
          f"× {len(VETOS)} VETOs = {len(seen_formulas) * len(seen_seeds) * len(VETOS)} "
          f"combinations...")
    phase_b = run_phase_b(rows, seen_formulas, seen_seeds, gt_events)
    print_phase_b(phase_b, args.top_n)

    # Phase C: Sustained threshold sweep (confidence cutoff sensitivity)
    if gt_events:
        print("\n" + "=" * 110)
        print("PHASE C — Sustained Threshold Sweep (confidence cutoff for HIGH)")
        print("  Cluster boundaries fixed by seed; different sust_pct thresholds change which clusters count")
        print("=" * 110)
        sust_thresholds = [5, 10, 15, 20, 25, 30, 40, 50]
        # Test all formulas with the best seed
        best_seed_id = seen_seeds[0] if seen_seeds else "S0_DOMINANT"
        best_seed_fn = SEEDS[best_seed_id]
        print(f"Seed: {best_seed_id}\n")
        print(f"{'Formula':<20} " + " ".join(f"{'st=' + str(t) + '%':>9}" for t in sust_thresholds))
        print("-" * (20 + 10 * len(sust_thresholds)))
        for fid in FORMULAS:
            score_key = f"cf_{fid}"
            clusters = flood_fill_generic(rows, best_seed_fn, score_key)
            f1_values = []
            for st in sust_thresholds:
                # Re-classify confidence with different threshold
                for cl in clusters:
                    cl["auto_confidence"] = "HIGH" if cl["sustained_pct"] >= st else "LOW"
                metrics = match_clusters_to_gt(clusters, gt_events, confidence_filter=True)
                f1_values.append(metrics["f1"])
            print(f"{fid:<20} " + " ".join(f"{f1:>9.3f}" for f1 in f1_values))

    # Best result detail
    best = phase_b[0] if phase_b else (phase_a[0] if phase_a else None)
    if best:
        print_cluster_detail(rows, best, gt_events)

    # Plot
    if args.plot and best:
        csv_base = os.path.splitext(args.csv)[0]
        plot_path = csv_base + "_formula_benchmark.png"
        plot_comparison(rows, gt_events, best, plot_path)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
