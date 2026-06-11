"""
La Facade Fissuree — Explorer Visualizer
==========================================
Pattern: [Utility / Visualization] — standalone plot generator; reads a
benchmark_explorer CSV and writes two PNG files (FER formulas + rPPG timeseries).

Generates two dedicated plots from a benchmark_explorer CSV:

  Plot 1 — FER Formulas Timeline
    Seven formula scores (F0-F6) over time as coloured lines.
    If an annotation CSV is found alongside the data CSV, event bands
    are overlaid (colour-coded by verdict).

  Plot 2 — rPPG Methods Timeline
    All rPPG BPM estimates (CHROM/POS/GREEN/ICA/WAVELET, 30s + 10s windows,
    gated, multi-ROI) as step plots over time.
    Same event-band overlay when annotation is available.

Usage:
    python visualize_explorer.py logs/20260327_XXXXXX_explorer.csv
    python visualize_explorer.py   # uses most recent _explorer.csv in logs/
"""

import os
import sys
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = os.path.join(SCRIPT_DIR, "logs", "annotations")

# ── Colour palettes ──────────────────────────────────────────────────────────
FORMULA_COLORS = {
    "f0": ("#e05050", "F0 AROUSAL [broken]"),
    "f1": ("#50c878", "F1 FEAR+TEN"),
    "f2": ("#00ff88", "F2 FEAR+SURP [recom]"),
    "f3": ("#ffd700", "F3 HEDGED"),
    "f4": ("#888888", "F4 HARDVETO"),
    "f5": ("#00bfff", "F5 STARTLE"),
    "f6": ("#ff69b4", "F6 SOFTVETO"),
}

RPPG_COLORS = {
    "rppg_chrom_30s":    ("#00ffc8", "CHROM 30s"),
    "rppg_pos_30s":      ("#ffc800", "POS 30s"),
    "rppg_green_30s":    ("#00c832", "GREEN 30s"),
    "rppg_ica_30s":      ("#c896ff", "ICA 30s"),
    "rppg_wavelet_30s":  ("#ff64c8", "WAVELET 30s"),
    "rppg_multi_roi_30s":("#64dcff", "MULTI-ROI 30s"),
    "rppg_chrom_10s":    ("#00ffc8", "CHROM 10s"),
    "rppg_pos_10s":      ("#ffc800", "POS 10s"),
    "rppg_green_10s":    ("#00c832", "GREEN 10s"),
    "rppg_ica_10s":      ("#c896ff", "ICA 10s"),
    "rppg_wavelet_10s":  ("#ff64c8", "WAVELET 10s"),
    "rppg_chrom_gated":  ("#00a090", "CHROM gated"),
    "rppg_pos_gated":    ("#a08000", "POS gated"),
    "rppg_green_gated":  ("#008020", "GREEN gated"),
}

VERDICT_BAND_COLORS = {
    "Fear":    ("#ff2020", 0.18),
    "Stress":  ("#ff8c00", 0.14),
    "Angry":   ("#ff00ff", 0.12),
    "Neutral": ("#808080", 0.08),
    "Neutral/Slightly angry": ("#808080", 0.08),
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_explorer_csv():
    # Search both old root location and new sessions subfolders
    patterns = [
        os.path.join(SCRIPT_DIR, "logs", "*_explorer*.csv"),
        os.path.join(SCRIPT_DIR, "logs", "sessions", "**", "*_explorer*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = [f for f in files if "_rppg" not in f]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_explorer_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_annotation_csv(_explorer_csv_path):
    """Look for any annotation CSV in logs/annotations/ that overlaps timeframe."""
    if not os.path.isdir(ANNOTATIONS_DIR):
        return None
    candidates = glob.glob(os.path.join(ANNOTATIONS_DIR, "*_annotation*.csv"))
    return candidates[0] if candidates else None


def load_annotations(ann_path):
    """Returns list of event dicts with id, verdict, start, end."""
    events = []
    if not ann_path or not os.path.isfile(ann_path):
        return events
    with open(ann_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("event_id", "").startswith("E") and row.get("start_s"):
                try:
                    events.append({
                        "id":      row["event_id"],
                        "verdict": row.get("verdict", ""),
                        "start":   float(row["start_s"]),
                        "end":     float(row["end_s"]),
                        "peak":    float(row["peak_t"]) if row.get("peak_t") else None,
                    })
                except ValueError:
                    pass
    return events


def draw_event_bands(ax, events, _ymin, ymax):
    """Shade event windows and mark peak timestamps."""
    for ev in events:
        col, alpha = VERDICT_BAND_COLORS.get(ev["verdict"], ("#aaaaaa", 0.10))
        ax.axvspan(ev["start"], ev["end"], color=col, alpha=alpha, zorder=0)
        if ev["peak"]:
            ax.axvline(ev["peak"], color=col, alpha=0.55, linewidth=0.8,
                       linestyle="--", zorder=1)
        mid = (ev["start"] + ev["end"]) / 2
        ax.text(mid, ymax * 0.97, ev["id"], fontsize=5.5, ha="center",
                va="top", color=col, alpha=0.9, zorder=5)


def _col_timeseries(rows, col):
    """Extract (t, value) pairs for a column, skipping empty cells."""
    ts, vals = [], []
    for r in rows:
        v = r.get(col, "")
        if v:
            try:
                ts.append(float(r["timestamp"]))
                vals.append(float(v))
            except ValueError:
                pass
    return np.array(ts), np.array(vals)


# ── Plot 1: FER Formulas ─────────────────────────────────────────────────────

def plot_fer_formulas(rows, events, out_path):
    t_all = np.array([float(r["timestamp"]) for r in rows])

    fig = plt.figure(figsize=(18, 9), facecolor="#1a1a2e")
    gs = GridSpec(2, 1, figure=fig, hspace=0.38,
                  top=0.92, bottom=0.08, left=0.05, right=0.97)

    # ── Top panel: all 7 formulas ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0f0f1e")

    for col, (color, label) in FORMULA_COLORS.items():
        if col not in rows[0]:
            continue
        vals = np.array([float(r[col]) for r in rows])
        lw = 2.0 if col == "f2" else 1.2
        ax1.plot(t_all, vals, color=color, linewidth=lw, label=label, alpha=0.9)

    if events:
        draw_event_bands(ax1, events, 0, 1)

    ax1.set_ylabel("Score", color="white", fontsize=9)
    ax1.set_ylim(0, None)
    ax1.set_xlim(t_all[0], t_all[-1])
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_title("FER Formulas — All 7 methods over time", color="white",
                  fontsize=11, pad=6)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.3,
               labelcolor="white", facecolor="#1a1a2e")
    ax1.spines[:].set_color("#444466")

    # ── Bottom panel: F2 vs F0 vs F3 closeup ──────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0f0f1e")

    for col in ["f0", "f2", "f3"]:
        color, label = FORMULA_COLORS[col]
        vals = np.array([float(r[col]) for r in rows])
        lw = 2.2 if col == "f2" else 1.4
        ax2.plot(t_all, vals, color=color, linewidth=lw, label=label, alpha=0.95)

    if events:
        draw_event_bands(ax2, events, 0, 1)

    ax2.set_xlabel("Time (s)", color="white", fontsize=9)
    ax2.set_ylabel("Score", color="white", fontsize=9)
    ax2.set_ylim(0, None)
    ax2.set_xlim(t_all[0], t_all[-1])
    ax2.tick_params(colors="white", labelsize=7)
    ax2.set_title("F0 vs F2 vs F3 — discrimination candidates", color="white",
                  fontsize=10, pad=6)
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.3,
               labelcolor="white", facecolor="#1a1a2e")
    ax2.spines[:].set_color("#444466")

    # ── Verdict legend ────────────────────────────────────────────────────
    verdict_patches = [
        mpatches.Patch(color=c, alpha=a + 0.3, label=v)
        for v, (c, a) in VERDICT_BAND_COLORS.items()
    ]
    fig.legend(handles=verdict_patches, loc="lower center", ncol=5,
               fontsize=7, framealpha=0.3, labelcolor="white",
               facecolor="#1a1a2e", bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"FER Formula Comparison — {os.path.basename(out_path).replace('_fer_formulas.png','')}",
                 color="white", fontsize=13)

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"FER formula plot saved: {out_path}")


# ── Plot 2: rPPG Timeseries ───────────────────────────────────────────────────

def plot_rppg_timeseries(rows, events, out_path):
    fig = plt.figure(figsize=(18, 11), facecolor="#1a1a2e")
    gs = GridSpec(3, 1, figure=fig, hspace=0.42,
                  top=0.92, bottom=0.08, left=0.06, right=0.97)

    # ── Panel 1: 30s window — all algorithms ─────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#0f0f1e")

    primary_30s = ["rppg_chrom_30s", "rppg_pos_30s", "rppg_green_30s",
                   "rppg_ica_30s", "rppg_wavelet_30s", "rppg_multi_roi_30s"]
    for col in primary_30s:
        if col not in rows[0]:
            continue
        t, v = _col_timeseries(rows, col)
        if len(t) == 0:
            continue
        color, label = RPPG_COLORS[col]
        lw = 2.2 if "chrom" in col else 1.4
        ax1.step(t, v, color=color, linewidth=lw, label=label,
                 where="post", alpha=0.9)

    if events:
        draw_event_bands(ax1, events, 0, 200)
    ax1.axhline(60, color="white", linewidth=0.5, alpha=0.3, linestyle=":")
    ax1.set_ylabel("BPM", color="white", fontsize=9)
    ax1.set_ylim(40, 110)
    if rows:
        ax1.set_xlim(0, float(rows[-1]["timestamp"]))
    ax1.tick_params(colors="white", labelsize=7)
    ax1.set_title("rPPG 30s window — all algorithms", color="white",
                  fontsize=11, pad=6)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.3,
               labelcolor="white", facecolor="#1a1a2e")
    ax1.spines[:].set_color("#444466")

    # ── Panel 2: 10s window — CHROM, POS, GREEN, ICA, WAVELET ────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#0f0f1e")

    primary_10s = ["rppg_chrom_10s", "rppg_pos_10s", "rppg_green_10s",
                   "rppg_ica_10s", "rppg_wavelet_10s"]
    for col in primary_10s:
        if col not in rows[0]:
            continue
        t, v = _col_timeseries(rows, col)
        if len(t) == 0:
            continue
        color, label = RPPG_COLORS[col]
        ax2.step(t, v, color=color, linewidth=1.4, label=label,
                 where="post", alpha=0.85)

    if events:
        draw_event_bands(ax2, events, 0, 200)
    ax2.axhline(60, color="white", linewidth=0.5, alpha=0.3, linestyle=":")
    ax2.set_ylabel("BPM", color="white", fontsize=9)
    ax2.set_ylim(40, 130)
    if rows:
        ax2.set_xlim(0, float(rows[-1]["timestamp"]))
    ax2.tick_params(colors="white", labelsize=7)
    ax2.set_title("rPPG 10s window (~6 BPM resolution)", color="white",
                  fontsize=10, pad=6)
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.3,
               labelcolor="white", facecolor="#1a1a2e")
    ax2.spines[:].set_color("#444466")

    # ── Panel 3: CHROM comparison — 30s / gated / 10s ────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor("#0f0f1e")

    chrom_variants = {
        "rppg_chrom_30s":   ("#00ffc8", "CHROM 30s", 2.0),
        "rppg_chrom_gated": ("#00a090", "CHROM gated (Concern I)", 1.4),
        "rppg_chrom_10s":   ("#88ffee", "CHROM 10s", 1.2),
    }
    for col, (color, label, lw) in chrom_variants.items():
        if col not in rows[0]:
            continue
        t, v = _col_timeseries(rows, col)
        if len(t) == 0:
            continue
        ax3.step(t, v, color=color, linewidth=lw, label=label,
                 where="post", alpha=0.9)

    if events:
        draw_event_bands(ax3, events, 0, 200)
    ax3.axhline(60, color="white", linewidth=0.5, alpha=0.3, linestyle=":")
    ax3.set_xlabel("Time (s)", color="white", fontsize=9)
    ax3.set_ylabel("BPM", color="white", fontsize=9)
    ax3.set_ylim(40, 100)
    if rows:
        ax3.set_xlim(0, float(rows[-1]["timestamp"]))
    ax3.tick_params(colors="white", labelsize=7)
    ax3.set_title("CHROM variants: 30s vs 10s vs neutral-gated",
                  color="white", fontsize=10, pad=6)
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.3,
               labelcolor="white", facecolor="#1a1a2e")
    ax3.spines[:].set_color("#444466")

    # ── Verdict legend ────────────────────────────────────────────────────
    verdict_patches = [
        mpatches.Patch(color=c, alpha=a + 0.3, label=v)
        for v, (c, a) in VERDICT_BAND_COLORS.items()
    ]
    fig.legend(handles=verdict_patches, loc="lower center", ncol=5,
               fontsize=7, framealpha=0.3, labelcolor="white",
               facecolor="#1a1a2e", bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"rPPG Method Comparison — {os.path.basename(out_path).replace('_rppg_timeseries.png','')}",
                 color="white", fontsize=13)

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"rPPG timeseries plot saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_explorer_csv()
        if not csv_path:
            print("No explorer CSV found.")
            sys.exit(1)
        print(f"Using: {csv_path}")

    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    rows = load_explorer_csv(csv_path)
    if not rows:
        print("CSV is empty.")
        sys.exit(1)

    # Check this is an explorer CSV
    if "f0" not in rows[0]:
        print("Not an explorer CSV (no f0 column). Skipping.")
        sys.exit(0)

    ann_path = find_annotation_csv(csv_path)
    events   = load_annotations(ann_path)
    if events:
        print(f"Overlaying {len(events)} annotated events from {os.path.basename(ann_path)}")
    else:
        print("No annotation file found — plots will show raw signals only.")

    base = csv_path.replace(".csv", "")
    plot_fer_formulas(rows, events, base + "_fer_formulas.png")
    plot_rppg_timeseries(rows, events, base + "_rppg_timeseries.png")


if __name__ == "__main__":
    main()
