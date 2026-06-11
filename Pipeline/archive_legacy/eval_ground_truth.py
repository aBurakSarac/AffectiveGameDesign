# eval_ground_truth.py
# Precision / recall evaluation of all 4 FER tools against the 36-event S02 ground truth.
# Usage: conda activate facade && python Pipeline/eval_ground_truth.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
LOGS = Path(__file__).resolve().parent / "logs"
SESS = LOGS / "sessions"

CSV = {
    "mediapipe": SESS / "S02_mixed_monitor-only" / "20260318_233756_mediapipe_S02_mixed_monitor-only.csv",
    "hsemotion": SESS / "S02_mixed_monitor-only" / "20260318_234030_hsemotion_S02_mixed_monitor-only.csv",
    "deepface":  SESS / "S02_mixed_monitor-only" / "20260318_234214_deepface_S02_mixed_monitor-only.csv",
    "pyfeat":    SESS / "S02_YoutubeReaction_monitor-only" / "20260318_234701_pyfeat_S02_YoutubeReaction_monitor-only.csv",
}
OUT_PNG = LOGS / "comparisons" / "eval_ground_truth_S02.png"
MATCH_WINDOW = 2.0   # ±2 s to count a detection as hitting an event

# ── Ground truth (36 events) ────────────────────────────────────────────────
# Categories: startle, fear, anger, surprise, joy
GT = {
    "startle":  [4, 8, 13, 26, 43, 44, 68, 78, 90, 107, 149, 154, 197, 212, 218],
    "fear":     [20, 23, 37, 41, 55, 96, 99, 101, 123, 130, 134, 151, 161, 164,
                 180, 184, 190, 214],   # 188-192 midpoint = 190
    "anger":    [136],                  # 135-138 midpoint
    "surprise": [216],
    "joy":      [201],
}
GT_ALL = sorted(set(t for ts in GT.values() for t in ts))  # 36 unique times

# True-negative probe windows: confirmed nothing there
TN_WINDOWS = [(33, 35), (48, 50), (126, 128), (230, 234)]

# Session duration
SESSION_END = 234.0

# ── Style ────────────────────────────────────────────────────────────────────
DARK  = "#1a1a2e"
PANEL = "#16213e"
GRID  = "#2a2a4a"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
YELL  = "\033[93m"
GREEN = "\033[92m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load(tool):
    df = pd.read_csv(CSV[tool])
    df = df[df["face_detected"] == 1].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def detect_events(timestamps, scores, threshold, min_gap=1.5):
    """Return list of peak-timestamps where score exceeds threshold,
    merging detections within min_gap seconds."""
    above = timestamps[scores >= threshold]
    if len(above) == 0:
        return []
    events, cur_start, cur_peak_t = [], above[0], above[0]
    cur_peak_v = scores[scores.index[timestamps == above[0]]].values[0] if hasattr(scores, 'index') else 0
    prev = above[0]
    for t in above[1:]:
        if t - prev <= min_gap:
            if scores.iloc[np.searchsorted(timestamps.values, t)] > cur_peak_v if hasattr(scores, 'iloc') else True:
                cur_peak_t = t
        else:
            events.append(cur_peak_t)
            cur_start, cur_peak_t = t, t
        prev = t
    events.append(cur_peak_t)
    return events


def pr_at_threshold(timestamps, scores, threshold, gt_events=GT_ALL, window=MATCH_WINDOW):
    """Compute precision, recall, F1 for a given threshold."""
    det_mask = scores >= threshold
    det_ts = timestamps[det_mask].values
    if len(det_ts) == 0:
        return 0.0, 0.0, 0.0, 0

    # Merge detections within min_gap
    merged = []
    prev = -999
    for t in det_ts:
        if t - prev > 1.5:
            merged.append(t)
        prev = t
    det_events = np.array(merged)

    # Match to ground truth
    tp = 0
    matched_gt = set()
    for d in det_events:
        for i, g in enumerate(gt_events):
            if i not in matched_gt and abs(d - g) <= window:
                tp += 1
                matched_gt.add(i)
                break

    fp = len(det_events) - tp
    fn = len(gt_events) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, len(det_events)


def sweep(timestamps, scores, lo, hi, steps=50, gt_events=GT_ALL):
    thresholds = np.linspace(lo, hi, steps)
    rows = []
    for thr in thresholds:
        p, r, f1, n_det = pr_at_threshold(timestamps, scores, thr, gt_events)
        rows.append({"threshold": thr, "precision": p, "recall": r, "f1": f1, "n_det": n_det})
    return pd.DataFrame(rows)


def best_row(df_sweep):
    return df_sweep.loc[df_sweep["f1"].idxmax()]


def tn_fp_rate(timestamps, scores, threshold):
    """How often does each tool fire inside confirmed-nothing windows?"""
    fp_count = 0
    for w0, w1 in TN_WINDOWS:
        mask = (timestamps >= w0) & (timestamps <= w1) & (scores >= threshold)
        fp_count += mask.sum()
    return int(fp_count)


# ═════════════════════════════════════════════════════════════════════════════
# Load data
# ═════════════════════════════════════════════════════════════════════════════
print("Loading CSVs...")
mp = load("mediapipe")
hs = load("hsemotion")
df = load("deepface")
pf = load("pyfeat")

print(f"  MP  frames: {len(mp):,}  | duration: {mp['timestamp'].max():.1f}s")
print(f"  HS  frames: {len(hs):,}  | duration: {hs['timestamp'].max():.1f}s")
print(f"  DF  frames: {len(df):,}  | duration: {df['timestamp'].max():.1f}s")
print(f"  PF  frames: {len(pf):,}  | duration: {pf['timestamp'].max():.1f}s")


# ═════════════════════════════════════════════════════════════════════════════
# Define signals to evaluate
# ═════════════════════════════════════════════════════════════════════════════
# MediaPipe: startle_score for startle detection only (15 events)
# HSEmotion: arousal for all 36 events (arousal-based proxy)
# DeepFace:  fear score for fear events (18 events)
# Py-Feat:   fear score + arousal for fear events (18 events)

TOOLS = [
    # (name, df, col, lo, hi, gt_events, label)
    ("MediaPipe\nstartle_score",  mp, "startle_score", 0.5,  8.0, GT["startle"],  "Startles only (n=15)"),
    ("MediaPipe\nstartle (all)",  mp, "startle_score", 0.5,  8.0, GT_ALL,         "All events (n=36)"),
    ("HSEmotion\narousal",        hs, "arousal",       0.05, 0.9, GT_ALL,         "All events (n=36)"),
    ("DeepFace\nfear",            df, "fear",          0.05, 0.95, GT["fear"],    "Fear events (n=18)"),
    ("DeepFace\nfear (all)",      df, "fear",          0.05, 0.95, GT_ALL,        "All events (n=36)"),
    ("Py-Feat\nfear",             pf, "fear",          0.05, 0.95, GT["fear"],    "Fear events (n=18)"),
    ("Py-Feat\narousal",          pf, "arousal",       0.05, 0.9,  GT_ALL,        "All events (n=36)"),
]


# ═════════════════════════════════════════════════════════════════════════════
# Console results
# ═════════════════════════════════════════════════════════════════════════════
print()
print(f"{BOLD}{'='*88}{RESET}")
print(f"{BOLD}  GROUND TRUTH EVALUATION — S02 Panda mixed session  |  ±{MATCH_WINDOW}s match window{RESET}")
print(f"{BOLD}{'='*88}{RESET}")
print(f"  Ground truth: {len(GT_ALL)} events — {len(GT['startle'])} startles, {len(GT['fear'])} fear, "
      f"1 anger, 1 surprise, 1 joy")
print()

sweeps = {}
for (name, data, col, lo, hi, gt_ev, lbl) in TOOLS:
    key = f"{name}|{col}|{lbl}"
    sw = sweep(data["timestamp"], data[col], lo, hi, steps=80, gt_events=gt_ev)
    sweeps[key] = sw
    best = best_row(sw)
    opt_thr = best["threshold"]
    tn_fps = tn_fp_rate(data["timestamp"], data[col], opt_thr)

    tag = name.replace("\n", " ")
    print(f"{CYAN}{tag:<30}{RESET}  vs {lbl}")
    print(f"  {'thr':>7} {'prec':>7} {'rec':>7} {'F1':>7} {'n_det':>6}  TN-window FPs")
    print(f"  {'-'*55}")

    # Print a few reference points
    for thr_ref in [lo + (hi - lo)*0.2, lo + (hi - lo)*0.4, lo + (hi - lo)*0.6,
                    lo + (hi - lo)*0.8]:
        p, r, f1, nd = pr_at_threshold(data["timestamp"], data[col], thr_ref, gt_ev)
        tfps = tn_fp_rate(data["timestamp"], data[col], thr_ref)
        row_str = f"  {thr_ref:>7.3f} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {nd:>6}  {tfps}"
        if f1 > 0.4:
            print(f"{GREEN}{row_str}{RESET}")
        elif f1 > 0.25:
            print(f"{YELL}{row_str}{RESET}")
        else:
            print(f"{DIM}{row_str}{RESET}")

    print(f"  {BOLD}BEST{RESET} {opt_thr:>7.3f} {best['precision']:>7.3f} {best['recall']:>7.3f} "
          f"{best['f1']:>7.3f} {best['n_det']:>6.0f}  {tn_fps}  <- optimal F1={best['f1']:.3f}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Per-event audit: which events does each tool catch/miss at best threshold?
# ═════════════════════════════════════════════════════════════════════════════
print(f"{BOLD}{'='*88}{RESET}")
print(f"{BOLD}  PER-EVENT AUDIT (at best F1 threshold, ±{MATCH_WINDOW}s window){RESET}")
print(f"{BOLD}{'='*88}{RESET}")
print()

AUDIT_TOOLS = [
    ("MP startle_score", mp, "startle_score", 0.5, 8.0, "startle"),
    ("HS arousal",       hs, "arousal",       0.05, 0.9, "all"),
    ("DF fear",          df, "fear",          0.05, 0.95,"fear"),
    ("PF fear",          pf, "fear",          0.05, 0.95,"fear"),
    ("PF arousal",       pf, "arousal",       0.05, 0.9, "all"),
]

# Determine best threshold per tool
best_thrs = {}
for (name, data, col, lo, hi, gt_key) in AUDIT_TOOLS:
    gt_ev = GT[gt_key] if gt_key != "all" else GT_ALL
    sw = sweep(data["timestamp"], data[col], lo, hi, steps=80, gt_events=gt_ev)
    best_thrs[name] = best_row(sw)["threshold"]

# Print header
header = f"  {'t(s)':>5}  {'category':<10}"
for (name, _, _, _, _, _) in AUDIT_TOOLS:
    header += f"  {name:<15}"
print(header)
print(f"  {'-'*95}")

for t_ev in sorted(GT_ALL):
    cat = next((c for c, ts in GT.items() if t_ev in ts), "?")
    row = f"  {t_ev:>5}  {cat:<10}"
    for (name, data, col, lo, hi, gt_key) in AUDIT_TOOLS:
        thr = best_thrs[name]
        # Check if any frame within ±MATCH_WINDOW fires
        window = data[(data["timestamp"] >= t_ev - MATCH_WINDOW) &
                      (data["timestamp"] <= t_ev + MATCH_WINDOW)]
        fired = (window[col] >= thr).any()
        peak  = window[col].max() if len(window) > 0 else float("nan")
        mark = "HIT " if fired else "miss"
        row += f"  {mark} ({peak:5.2f}){' ':2}"
    print(row)

# Check TN windows
print()
print(f"  {'--- TN probes ---'}")
for w0, w1 in TN_WINDOWS:
    t_mid = (w0 + w1) / 2
    row = f"  {t_mid:>5.0f}  {'NOTHING':<10}"
    for (name, data, col, lo, hi, gt_key) in AUDIT_TOOLS:
        thr = best_thrs[name]
        window = data[(data["timestamp"] >= w0) & (data["timestamp"] <= w1)]
        fired = (window[col] >= thr).any()
        peak  = window[col].max() if len(window) > 0 else float("nan")
        mark = f"{RED}FP  {RESET}" if fired else f"{GREEN}OK  {RESET}"
        row += f"  {mark}({peak:5.2f}){' ':3}"
    print(row)

print()
print(f"{BOLD}{'='*88}{RESET}")


# ═════════════════════════════════════════════════════════════════════════════
# Summary table
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}SUMMARY TABLE{RESET}")
print(f"  {'Tool + Signal':<28} {'vs':<22} {'BestThr':>8} {'P':>6} {'R':>6} {'F1':>6} {'TN-FPs':>7}")
print(f"  {'-'*84}")

for (name, data, col, lo, hi, gt_ev, lbl) in TOOLS:
    key = f"{name}|{col}|{lbl}"
    sw = sweeps[key]
    best = best_row(sw)
    tn_fps = tn_fp_rate(data["timestamp"], data[col], best["threshold"])
    tag = name.replace('\n', ' ')
    print(f"  {tag:<28} {lbl:<22} {best['threshold']:>8.3f} "
          f"{best['precision']:>6.3f} {best['recall']:>6.3f} {best['f1']:>6.3f} {tn_fps:>7}")

print()


# ═════════════════════════════════════════════════════════════════════════════
# Plot: PR curves
# ═════════════════════════════════════════════════════════════════════════════
print(f"Generating PR plot -> {OUT_PNG}")

TOOL_PLOT = [
    ("MP startle (startles)",   "mediapipe", "startle_score", 0.5, 8.0,  GT["startle"], "#4db8ff", "-"),
    ("HS arousal (all)",        "hsemotion", "arousal",       0.05, 0.9,  GT_ALL,        "#ff4444", "-"),
    ("DF fear (fear events)",   "deepface",  "fear",          0.05, 0.95, GT["fear"],    "#ffdd00", "--"),
    ("DF fear (all events)",    "deepface",  "fear",          0.05, 0.95, GT_ALL,        "#ff8800", ":"),
    ("PF fear (fear events)",   "pyfeat",    "fear",          0.05, 0.95, GT["fear"],    "#cc44ff", "--"),
    ("PF arousal (all events)", "pyfeat",    "arousal",       0.05, 0.9,  GT_ALL,        "#00ff88", ":"),
]
DATA = {"mediapipe": mp, "hsemotion": hs, "deepface": df, "pyfeat": pf}

fig, (ax_pr, ax_f1) = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK)
fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.12, wspace=0.28)

for ax in (ax_pr, ax_f1):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#aaaacc", labelsize=9)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)

ax_pr.set_xlabel("Recall", fontsize=10)
ax_pr.set_ylabel("Precision", fontsize=10)
ax_pr.set_title("Precision–Recall curves  |  S02 Panda ground truth",
                color="#ddddff", fontsize=10)
ax_pr.set_xlim(0, 1.02)
ax_pr.set_ylim(0, 1.05)

ax_f1.set_xlabel("Threshold", fontsize=10)
ax_f1.set_ylabel("F1", fontsize=10)
ax_f1.set_title("F1 vs Threshold  |  each tool at best GT set",
                color="#ddddff", fontsize=10)
ax_f1.set_ylim(0, 1.05)

for (lbl, tool_key, col, lo, hi, gt_ev, color, ls) in TOOL_PLOT:
    data = DATA[tool_key]
    sw = sweep(data["timestamp"], data[col], lo, hi, steps=80, gt_events=gt_ev)
    best = best_row(sw)

    # PR curve
    ax_pr.plot(sw["recall"], sw["precision"], color=color, linewidth=1.6,
               linestyle=ls, label=lbl, alpha=0.9)
    ax_pr.scatter([best["recall"]], [best["precision"]], color=color,
                  s=60, zorder=5, edgecolors="white", linewidth=0.8)
    ax_pr.annotate(f"F1={best['f1']:.2f}", (best["recall"], best["precision"]),
                   textcoords="offset points", xytext=(5, 4),
                   fontsize=7, color=color, alpha=0.9)

    # F1 vs threshold
    ax_f1.plot(sw["threshold"], sw["f1"], color=color, linewidth=1.6,
               linestyle=ls, label=lbl, alpha=0.9)
    ax_f1.axvline(best["threshold"], color=color, linewidth=0.7,
                  linestyle=":", alpha=0.5)

ax_pr.legend(fontsize=8, loc="lower left", facecolor=PANEL, edgecolor=GRID,
             labelcolor="#ddddff", framealpha=0.85)
ax_f1.legend(fontsize=8, loc="upper right", facecolor=PANEL, edgecolor=GRID,
             labelcolor="#ddddff", framealpha=0.85)

# Annotate GT info
ax_pr.text(0.02, 0.02, f"{len(GT_ALL)} ground truth events  |  ±{MATCH_WINDOW}s match window",
           transform=ax_pr.transAxes, fontsize=8, color="#888899", va="bottom")

plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Saved: {OUT_PNG}")
print("Done.")
