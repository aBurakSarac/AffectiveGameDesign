# compare_all_sessions.py
# Generic multi-subject FER comparison: confirmed + sub-threshold event tables + 4-panel plot
# Subjects S02–S05 (S01 excluded by default). Uses sessions.json as registry.
# Usage: conda activate facade && python Pipeline/compare_all_sessions.py [options]
#   --subject S03        filter to one subject
#   --content jumpscare  filter by content type
#   --lighting dim       filter by lighting
#   --mp-threshold 3.0   MediaPipe startle_score confirmed threshold
#   --hs-threshold 0.6   HSEmotion arousal confirmed threshold
#   --sub-ratio 0.6      sub-threshold fraction (show events at >= 60% of threshold)
#   --no-plot            skip plot generation

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ───────────────────────────────────────────────────────────────────────
LOGS          = Path(__file__).resolve().parent / "logs"
SESSIONS_JSON = LOGS / "sessions.json"

# ── Style ───────────────────────────────────────────────────────────────────────
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COL = "#2a2a4a"

BOLD  = "\033[1m"
CYAN  = "\033[96m"
YELL  = "\033[93m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"

smooth = lambda s, w=15: pd.Series(s).rolling(w, center=True, min_periods=1).mean().values

HS_EMOTION_COLORS = {
    "hs_anger":     "#ff4444",
    "hs_contempt":  "#ff8800",
    "hs_disgust":   "#aa44ff",
    "hs_fear":      "#ffdd00",
    "hs_happiness": "#00ff88",
    "hs_neutral":   "#888888",
    "hs_sadness":   "#4488ff",
    "hs_surprise":  "#ff44ff",
}

# ── Table columns: (header, col_key, fmt_spec, min_width) ──────────────────────
# fmt_spec: "6.2f" for numeric, "str" for string, "__trigger__" for trigger col
TABLE_COLS = [
    ("t(s)",      "timestamp",      "6.2f",  6),
    ("startle",   "startle_score",  "8.3f",  8),
    ("tension",   "tension",        "7.4f",  7),
    ("hs_aro",    "hs_arousal",     "7.4f",  7),
    ("hs_val",    "hs_valence",     "7.4f",  7),
    ("hs_dom",    "hs_dominant",    "str",  10),
    ("hs_ang",    "hs_anger",       "6.3f",  6),
    ("hs_con",    "hs_contempt",    "6.3f",  6),
    ("hs_dis",    "hs_disgust",     "6.3f",  6),
    ("hs_fear",   "hs_fear",        "6.3f",  7),
    ("hs_hap",    "hs_happiness",   "6.3f",  6),
    ("hs_neu",    "hs_neutral",     "6.3f",  6),
    ("hs_sad",    "hs_sadness",     "6.3f",  6),
    ("hs_sur",    "hs_surprise",    "6.3f",  6),
    ("df_fear",   "df_fear",        "7.3f",  7),
    ("pf_fear",   "pf_fear",        "7.3f",  7),
    ("fear_2d",   "fear_2d",        "7.3f",  7),
    ("fear_cons", "fear_consensus", "8.3f",  9),
    ("TRIGGER",   "__trigger__",    "str",  14),
]


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def _norm_content(c: str) -> str:
    if "youtube" in c.lower():
        return "mixed"
    return c.lower()


def load_registry(path: Path) -> dict:
    """Build {(subject, content, lighting): {tool: session_entry}} from sessions.json.
    For duplicate tool entries (same subject/content/lighting) keeps the earliest."""
    with open(path) as f:
        data = json.load(f)
    reg = {}
    for s in data["sessions"]:
        key = (s["subject_id"], _norm_content(s["content_type"]), s["lighting"])
        tool = s["tool"]
        if key not in reg:
            reg[key] = {}
        if tool in reg[key]:
            if s["session_id"] < reg[key][tool]["session_id"]:
                reg[key][tool] = s
        else:
            reg[key][tool] = s
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading & alignment
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(session: dict, logs_dir: Path) -> "pd.DataFrame | None":
    p = logs_dir / session["csv_path"]
    if not p.exists():
        print(f"  [WARN] Missing CSV: {p.name}")
        return None
    df = pd.read_csv(p)
    if "face_detected" in df.columns:
        df = df[df["face_detected"] == 1].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def align_sessions(session_set: dict, logs_dir: Path) -> "pd.DataFrame | None":
    """Merge all available tool CSVs onto MediaPipe timestamp base via merge_asof."""
    if "mediapipe" not in session_set:
        return None
    mp = _load_csv(session_set["mediapipe"], logs_dir)
    if mp is None or len(mp) == 0:
        return None
    merged = mp.copy()

    # HSEmotion — tolerance 0.15 s
    if "hsemotion" in session_set:
        hs = _load_csv(session_set["hsemotion"], logs_dir)
        if hs is not None:
            keep = [c for c in ["timestamp", "valence", "arousal", "anger", "contempt",
                                 "disgust", "fear", "happiness", "neutral", "sadness",
                                 "surprise", "dominant"] if c in hs.columns]
            hs_sub = hs[keep].rename(
                columns={c: f"hs_{c}" for c in keep if c != "timestamp"})
            merged = pd.merge_asof(merged, hs_sub, on="timestamp",
                                   direction="nearest", tolerance=0.15)

    # DeepFace — tolerance 0.3 s
    if "deepface" in session_set:
        df = _load_csv(session_set["deepface"], logs_dir)
        if df is not None:
            keep = [c for c in ["timestamp", "angry", "disgust", "fear", "happy", "sad",
                                 "surprise", "neutral", "dominant", "arousal"] if c in df.columns]
            df_sub = df[keep].rename(
                columns={c: f"df_{c}" for c in keep if c != "timestamp"})
            merged = pd.merge_asof(merged, df_sub, on="timestamp",
                                   direction="nearest", tolerance=0.3)

    # Py-Feat — tolerance 2.0 s
    if "pyfeat" in session_set:
        pf = _load_csv(session_set["pyfeat"], logs_dir)
        if pf is not None:
            keep = [c for c in ["timestamp", "fear", "anger", "disgust", "happiness",
                                 "sadness", "surprise", "neutral", "dominant",
                                 "arousal"] if c in pf.columns]
            pf_sub = pf[keep].rename(
                columns={c: f"pf_{c}" for c in keep if c != "timestamp"})
            merged = pd.merge_asof(merged, pf_sub, on="timestamp",
                                   direction="nearest", tolerance=2.0)

    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fear proxies
# ─────────────────────────────────────────────────────────────────────────────

def add_fear_proxies(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    aro  = m.get("hs_arousal",   pd.Series(0.0, index=m.index)).fillna(0)
    val  = m.get("hs_valence",   pd.Series(0.0, index=m.index)).fillna(0)
    star = m.get("startle_score", pd.Series(0.0, index=m.index)).fillna(0)

    m["fear_arousal"]   = aro
    m["fear_2d"]        = aro * (-val).clip(0, 1)

    max_aro  = aro.max()  or 1
    max_star = star.max() or 1
    m["fear_consensus"] = 0.6 * (aro / max_aro) + 0.4 * (star / max_star)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Event detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_events(m: pd.DataFrame, mp_thresh: float, hs_thresh: float,
                  sub_ratio: float, min_gap_s: float = 1.5):
    """Return confirmed and sub-threshold event peak rows + boolean masks."""
    mp_v = m["startle_score"].fillna(0) if "startle_score" in m.columns \
           else pd.Series(0.0, index=m.index)
    hs_v = m["hs_arousal"].fillna(0)   if "hs_arousal"    in m.columns \
           else pd.Series(0.0, index=m.index)

    mp_conf = mp_v >= mp_thresh
    hs_conf = hs_v >= hs_thresh
    mp_sub  = (mp_v >= mp_thresh * sub_ratio) & ~mp_conf
    hs_sub  = (hs_v >= hs_thresh * sub_ratio) & ~hs_conf

    conf_mask = (mp_conf | hs_conf).values
    sub_mask  = ((mp_sub | hs_sub) & ~(mp_conf | hs_conf)).values

    def cluster_peaks(bool_mask):
        idxs = np.where(bool_mask)[0]
        if len(idxs) == 0:
            return []
        clusters, cur = [], [idxs[0]]
        for idx in idxs[1:]:
            if m.loc[idx, "timestamp"] - m.loc[cur[-1], "timestamp"] > min_gap_s:
                clusters.append(cur)
                cur = []
            cur.append(idx)
        clusters.append(cur)
        peaks = []
        for c in clusters:
            score = mp_v.iloc[c] / (mp_thresh or 1) + hs_v.iloc[c] / (hs_thresh or 1)
            peaks.append(c[int(score.values.argmax())])
        return peaks

    def trig_str(pos_idx):
        parts = []
        if mp_conf.iloc[pos_idx]: parts.append("MP")
        if hs_conf.iloc[pos_idx]: parts.append("HS")
        if parts:
            return "+".join(parts)
        near = []
        if mp_sub.iloc[pos_idx]: near.append("MP")
        if hs_sub.iloc[pos_idx]: near.append("HS")
        return f"NEAR[{','.join(near)}]"

    confirmed_events = [(m.loc[i], trig_str(i)) for i in cluster_peaks(conf_mask)]
    sub_events       = [(m.loc[i], trig_str(i)) for i in cluster_peaks(sub_mask)]
    return confirmed_events, sub_events, conf_mask, sub_mask


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

def _cell(row: pd.Series, col: str, fmt: str, width: int) -> str:
    v = row.get(col)
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "  --  ".ljust(width)
    if fmt == "str":
        return str(v)[:width].ljust(width)
    try:
        return format(float(v), fmt).ljust(width)
    except (TypeError, ValueError):
        return "  —  ".ljust(width)


def print_table(events: list, title: str, color: str = RED) -> None:
    print()
    print(color + BOLD + f"  {title}  ({len(events)} events)" + RESET)
    if not events:
        print(DIM + "  (none)" + RESET)
        return
    header = " | ".join(h.ljust(w) for h, _, _, w in TABLE_COLS)
    sep    = "-+-".join("-" * w for _, _, _, w in TABLE_COLS)
    print(CYAN + "  " + header + RESET)
    print(DIM  + "  " + sep    + RESET)
    for row, trig in events:
        cells = []
        for _, col, fmt, w in TABLE_COLS:
            cells.append(trig.ljust(w) if col == "__trigger__" else _cell(row, col, fmt, w))
        print("  " + " | ".join(cells))


def print_summary(session_set: dict, merged: pd.DataFrame,
                  confirmed: list, sub: list) -> None:
    print()
    print(DIM + "-" * 110 + RESET)
    print(BOLD + "  SUMMARY" + RESET)
    for tool_key, lbl in [("mediapipe", "MediaPipe"), ("hsemotion", "HSEmotion"),
                           ("deepface",  "DeepFace"),  ("pyfeat",   "Py-Feat")]:
        if tool_key not in session_set:
            continue
        s   = session_set[tool_key]
        fdr = s.get("face_detection_rate", float("nan"))
        fps = s.get("fps_actual",          float("nan"))
        dur = s.get("duration_s",          float("nan"))
        lat = s.get("avg_latency_ms",      float("nan"))
        try:
            print(f"  {lbl:12} | FDR={fdr*100:.1f}%  FPS={fps:.1f}  dur={dur:.0f}s  avg_lat={lat:.0f}ms")
        except TypeError:
            print(f"  {lbl:12} | (data unavailable)")
    if "hs_arousal" in merged.columns:
        aro = merged["hs_arousal"].dropna()
        if len(aro):
            pk = aro.idxmax()
            print(f"  HS arousal    | mean={aro.mean():.3f}  peak={aro.max():.3f}"
                  f" @ t={merged.loc[pk,'timestamp']:.2f}s  nz={100*(aro>0).mean():.0f}%")
    if "startle_score" in merged.columns:
        ss = merged["startle_score"].fillna(0)
        print(f"  MP startle    | peak={ss.max():.2f}  tension_mean={merged['tension'].mean():.4f}")
    agree = sum(1 for _, trig in confirmed if "+" in trig)
    print(f"  Events        | confirmed={len(confirmed)}  sub-threshold={len(sub)}"
          f"  multi-tool_agree={agree}")
    print(DIM + "-" * 110 + RESET)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.yaxis.label.set_color("#aaaacc")
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.grid(axis="both", color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)


def _shade(ax, mask: np.ndarray, t: np.ndarray, color: str, alpha: float) -> None:
    in_run, t0 = False, 0.0
    for ti, flag in zip(t, mask):
        if flag and not in_run:
            t0, in_run = ti, True
        elif not flag and in_run:
            ax.axvspan(t0 - 0.1, ti + 0.1, alpha=alpha, color=color, zorder=0)
            in_run = False
    if in_run:
        ax.axvspan(t0 - 0.1, t[-1] + 0.1, alpha=alpha, color=color, zorder=0)


def generate_plot(merged: pd.DataFrame, conf_mask: np.ndarray, sub_mask: np.ndarray,
                  confirmed: list, _sub: list,
                  mp_thresh: float, hs_thresh: float,
                  title: str, out_path: Path) -> None:
    t = merged["timestamp"].values

    fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True,
                             facecolor=DARK_BG,
                             gridspec_kw={"height_ratios": [2, 2, 1.5, 1]})
    fig.subplots_adjust(hspace=0.06, left=0.07, right=0.97, top=0.93, bottom=0.05)

    def shade_all(ax):
        _shade(ax, conf_mask, t, "#ff3333", 0.18)
        _shade(ax, sub_mask,  t, "#ff8800", 0.12)

    # ── Panel 1: MediaPipe startle + tension ──────────────────────────────────
    ax1 = axes[0]; _style_ax(ax1); shade_all(ax1)
    if "tension" in merged.columns:
        ax1.plot(t, smooth(merged["tension"].fillna(0).values),
                 color="#4db8ff", linewidth=1.2, label="tension", alpha=0.9)
    if "startle_score" in merged.columns:
        ss   = merged["startle_score"].fillna(0).values
        cm   = ss >= mp_thresh
        sm   = (ss >= mp_thresh * 0.6) & ~cm
        ax1.scatter(t[cm], ss[cm].clip(max=15),
                    color="#ff4444", s=20, zorder=5, label=f"startle≥{mp_thresh}")
        ax1.scatter(t[sm], ss[sm].clip(max=15) * 0.3,
                    color="#ff9900", s=12, zorder=5, marker="^", label="sub-thresh startle")
        ax1.axhline(mp_thresh, color="#ff4444", linewidth=0.5, linestyle="--", alpha=0.4)
    ax1.set_ylabel("MediaPipe", fontsize=9, color="#aaaacc")
    ax1.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
               edgecolor=GRID_COL, labelcolor="#aaaacc", framealpha=0.8)
    ax1.set_title(title + "  |  red=confirmed  orange=sub-threshold",
                  fontsize=9, color="#ddddff", pad=5)

    # ── Panel 2: HS arousal + valence + fear_2d ───────────────────────────────
    ax2 = axes[1]; _style_ax(ax2); shade_all(ax2)
    if "hs_arousal" in merged.columns:
        aro = merged["hs_arousal"].fillna(0).values
        ax2.plot(t, smooth(aro), color="#ff4444", linewidth=1.3,
                 label="hs_arousal", alpha=0.9)
        ax2.axhline(hs_thresh, color="#ffaa00", linewidth=0.6, linestyle="--",
                    alpha=0.5, label=f"threshold ({hs_thresh})")
        ax2.axhline(hs_thresh * 0.6, color="#ffaa00", linewidth=0.4, linestyle=":",
                    alpha=0.3, label=f"sub ({hs_thresh*0.6:.2f})")
        for row, trig in confirmed:
            v = row.get("hs_arousal", np.nan)
            if pd.notna(v):
                ax2.annotate(f"t={row['timestamp']:.1f}s\n{trig}",
                             xy=(row["timestamp"], float(v)),
                             fontsize=6, color="#ffdd00", ha="center", va="bottom",
                             xytext=(0, 5), textcoords="offset points")
    if "hs_valence" in merged.columns:
        ax2.plot(t, smooth(merged["hs_valence"].fillna(0).values, 30),
                 color="#888888", linewidth=0.7, linestyle=":", label="hs_valence", alpha=0.5)
        ax2.axhline(0, color="#555588", linewidth=0.4, alpha=0.4)
    if "fear_2d" in merged.columns:
        ax2.plot(t, smooth(merged["fear_2d"].fillna(0).values, 20),
                 color="#ff8800", linewidth=0.9, linestyle="-.", label="fear_2d proxy", alpha=0.7)
    ax2.set_ylabel("HSEmotion", fontsize=9, color="#aaaacc")
    ax2.set_ylim(-0.8, 1.6)
    ax2.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
               edgecolor=GRID_COL, labelcolor="#aaaacc", framealpha=0.8)

    # ── Panel 3: All 8 HS emotion scores ─────────────────────────────────────
    ax3 = axes[2]; _style_ax(ax3); shade_all(ax3)
    for col, color in HS_EMOTION_COLORS.items():
        if col in merged.columns:
            ax3.plot(t, smooth(merged[col].fillna(0).values, 10),
                     color=color, linewidth=0.8, label=col.replace("hs_", ""), alpha=0.75)
    ax3.set_ylabel("HS Emotions", fontsize=9, color="#aaaacc")
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
               edgecolor=GRID_COL, labelcolor="#aaaacc", framealpha=0.8, ncol=4)

    # ── Panel 4: DF fear + PF fear + consensus proxy ──────────────────────────
    ax4 = axes[3]; _style_ax(ax4); shade_all(ax4)
    if "df_fear" in merged.columns:
        ax4.plot(t, smooth(merged["df_fear"].fillna(0).values, 20),
                 color="#00ccff", linewidth=1.0, label="DeepFace fear", alpha=0.85)
    if "pf_fear" in merged.columns:
        ax4.plot(t, smooth(merged["pf_fear"].fillna(0).values, 10),
                 color="#ffdd00", linewidth=0.9, linestyle="--", label="Py-Feat fear", alpha=0.8)
    if "fear_consensus" in merged.columns:
        ax4.plot(t, smooth(merged["fear_consensus"].fillna(0).values, 20),
                 color="#00ff88", linewidth=0.9, linestyle="-.", label="consensus proxy", alpha=0.7)
    ax4.set_ylabel("Reference", fontsize=9, color="#aaaacc")
    ax4.set_xlabel("time (s)", fontsize=9, color="#aaaacc")
    ax4.set_ylim(-0.05, 1.1)
    ax4.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
               edgecolor=GRID_COL, labelcolor="#aaaacc", framealpha=0.8)

    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  [PLOT] Saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Multi-subject FER comparison (S02–S05)")
    p.add_argument("--subject",      help="Filter subject (e.g. S03)")
    p.add_argument("--content",      help="Filter content type (e.g. jumpscare)")
    p.add_argument("--lighting",     help="Filter lighting (e.g. dim)")
    p.add_argument("--mp-threshold", type=float, default=3.0, metavar="T",
                   help="MP startle_score confirmed threshold (default 3.0)")
    p.add_argument("--hs-threshold", type=float, default=0.6, metavar="T",
                   help="HS arousal confirmed threshold (default 0.6)")
    p.add_argument("--sub-ratio",    type=float, default=0.6, metavar="R",
                   help="Sub-threshold fraction of threshold (default 0.6)")
    p.add_argument("--no-plot",      action="store_true", help="Skip plot generation")
    args = p.parse_args()

    registry = load_registry(SESSIONS_JSON)
    EXCLUDE  = {"S01"}

    keys = sorted(registry.keys())
    if args.subject:
        keys = [k for k in keys if k[0] == args.subject.upper()]
    else:
        keys = [k for k in keys if k[0] not in EXCLUDE]
    if args.content:
        keys = [k for k in keys if k[1] == args.content.lower()]
    if args.lighting:
        keys = [k for k in keys if k[2] == args.lighting.lower()]

    if not keys:
        print("No matching sessions found.")
        sys.exit(1)

    print(f"\n{BOLD}{CYAN}  FER Multi-Subject Comparison"
          f"  |  MP_thresh={args.mp_threshold}  HS_thresh={args.hs_threshold}"
          f"  sub_ratio={args.sub_ratio}{RESET}")
    print(f"  Sessions to compare: {len(keys)}\n")

    for subj, content, lighting in keys:
        session_set = registry[(subj, content, lighting)]
        tools       = sorted(session_set.keys())

        print(f"\n{'='*110}")
        print(f"{BOLD}  {subj}  |  {content}  |  {lighting}"
              f"  |  tools: {', '.join(tools)}{RESET}")
        print(f"{'='*110}")

        merged = align_sessions(session_set, LOGS)
        if merged is None or len(merged) == 0:
            print("  [WARN] No data after alignment — skipping.")
            continue

        merged = add_fear_proxies(merged)
        confirmed, sub, conf_mask, sub_mask = detect_events(
            merged, args.mp_threshold, args.hs_threshold, args.sub_ratio)

        print_table(confirmed, "CONFIRMED EVENTS",     color=RED)
        print_table(sub,       "SUB-THRESHOLD EVENTS", color=YELL)
        print_summary(session_set, merged, confirmed, sub)

        if not args.no_plot:
            title    = f"{subj}  {content}  {lighting}  [{', '.join(tools)}]"
            out_path = LOGS / "comparisons" / f"compare_all_sessions_{subj}_{content}_{lighting}.png"
            generate_plot(merged, conf_mask, sub_mask, confirmed, sub,
                          args.mp_threshold, args.hs_threshold, title, out_path)


if __name__ == "__main__":
    main()
