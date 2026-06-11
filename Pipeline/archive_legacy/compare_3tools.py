# compare_3tools.py
# 3-tool comparison: MediaPipe + HSEmotion + DeepFace
# Includes baseline correction, sub-threshold events, timestamp inspection.
# Usage: conda activate facade && python Pipeline/compare_3tools.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
LOGS = Path(__file__).resolve().parent / "logs"
SESS = LOGS / "sessions"

GAME = {
    "mp": SESS / "S01_game_bright" / "20260318_213728_mediapipe_S01_game_bright.csv",
    "hs": SESS / "S01_chase_bright" / "20260318_214136_hsemotion_S01_chase_bright.csv",
    "df": SESS / "S01_chase_bright" / "20260318_214548_deepface_S01_chase_bright.csv",
}
NEUTRAL = {
    "mp": SESS / "S01_neutral_bright" / "20260317_124653_mediapipe_S01_neutral_bright.csv",
    "hs": SESS / "S01_neutral_bright" / "20260317_125310_hsemotion_S01_neutral_bright.csv",
    "df": SESS / "S01_neutral_bright" / "20260317_125503_deepface_S01_neutral_bright.csv",
}
OUT_PNG = LOGS / "comparisons" / "compare_3tools_20260318_game.png"

# ── Thresholds ─────────────────────────────────────────────────────────────────
STARTLE_CONFIRMED  = 3.0
STARTLE_SUSPICIOUS = 1.5
MIN_GAP_S = 1.5

# ── User-specified moments ─────────────────────────────────────────────────────
USER_MOMENTS = {
    180: "startle buildup",
    217: "214-220s window",
    223: "startle approach",
    237: "confirmed startle",
    246: "huge stress (unlisted)",
    264: "user-observed stress",
    328: "pre-big-startle",
}

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COL = "#2a2a4a"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
YELL  = "\033[93m"
GREEN = "\033[92m"
RED   = "\033[91m"
DIM   = "\033[2m"
RESET = "\033[0m"

smooth = lambda s, w=15: pd.Series(s).rolling(w, center=True, min_periods=1).mean().values

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("Loading game session CSVs...")
mp_g = pd.read_csv(GAME["mp"])
hs_g = pd.read_csv(GAME["hs"])
df_g = pd.read_csv(GAME["df"])

print("Loading neutral baseline CSVs...")
mp_n = pd.read_csv(NEUTRAL["mp"])
hs_n = pd.read_csv(NEUTRAL["hs"])
df_n = pd.read_csv(NEUTRAL["df"])

# Keep only face-detected rows
for df in [mp_g, hs_g, df_g, mp_n, hs_n, df_n]:
    df.query("face_detected == 1", inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute neutral baselines
# ─────────────────────────────────────────────────────────────────────────────
BL = {
    "mp_tension":     mp_n["tension"].mean(),
    "mp_browDown":    mp_n[["browDownLeft","browDownRight"]].mean(axis=1).mean(),
    "mp_browInnerUp": mp_n["browInnerUp"].mean(),
    "hs_arousal":     hs_n["arousal"].mean(),
    "hs_valence":     hs_n["valence"].mean(),
    "hs_anger":       hs_n["anger"].mean(),
    "df_sad":         df_n["sad"].mean(),
    "df_fear":        df_n["fear"].mean(),
    "df_neutral_pct": (df_n["dominant"].str.lower() == "neutral").mean() * 100,
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Align 3 game CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("Aligning game CSVs...")
mp_s = mp_g.sort_values("timestamp").reset_index(drop=True)
hs_s = hs_g[["timestamp","arousal","valence","anger","fear","happiness",
              "neutral","sadness","surprise","contempt","dominant"]].sort_values("timestamp").reset_index(drop=True)
df_s = df_g[["timestamp","angry","disgust","fear","happy","sad","surprise",
              "neutral","dominant","dominant_score"]].sort_values("timestamp").reset_index(drop=True)

m = pd.merge_asof(mp_s, hs_s.rename(columns={c: f"hs_{c}" for c in hs_s.columns if c != "timestamp"}),
                  on="timestamp", direction="nearest", tolerance=0.05)
m = pd.merge_asof(m,   df_s.rename(columns={c: f"df_{c}" for c in df_s.columns if c != "timestamp"}),
                  on="timestamp", direction="nearest", tolerance=0.05)
m = m.dropna(subset=["hs_arousal","df_sad"]).reset_index(drop=True)
print(f"  Aligned frames: {len(m):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline-corrected signals
# ─────────────────────────────────────────────────────────────────────────────
m["bc_tension"]  = m["tension"]    - BL["mp_tension"]
m["bc_arousal"]  = m["hs_arousal"] - BL["hs_arousal"]
m["bc_df_sad"]   = m["df_sad"]     - BL["df_sad"]
m["bc_valence_delta"] = abs(m["hs_valence"] - BL["hs_valence"])  # should be near 0

# Normalise each BC channel to [0,1] for composite
def norm01(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

m["composite"] = (
    0.50 * norm01(m["bc_arousal"].clip(lower=0))
    + 0.30 * norm01(m["bc_tension"].clip(lower=0))
    + 0.20 * norm01(m["bc_df_sad"].clip(lower=0))
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Startle windows (confirmed + suspicious)
# ─────────────────────────────────────────────────────────────────────────────
def find_startle_windows(series, lo, hi, min_gap=MIN_GAP_S):
    mask = (series > lo) & (series <= hi)
    windows, in_win, win = [], False, {}
    for i in range(len(series)):
        t, sc = m["timestamp"].iloc[i], series.iloc[i]
        if mask.iloc[i]:
            if not in_win:
                win = {"t_start": t, "t_end": t, "peak": sc}
                in_win = True
            else:
                win["t_end"] = t
                win["peak"] = max(win["peak"], sc)
        else:
            if in_win:
                windows.append(win)
                in_win = False
    if in_win:
        windows.append(win)
    # merge close windows
    merged = []
    for w in windows:
        if merged and (w["t_start"] - merged[-1]["t_end"]) < min_gap:
            merged[-1]["t_end"] = w["t_end"]
            merged[-1]["peak"] = max(merged[-1]["peak"], w["peak"])
        else:
            merged.append(dict(w))
    return merged

confirmed_wins  = find_startle_windows(m["startle_score"], STARTLE_CONFIRMED, 999)
suspicious_wins = find_startle_windows(m["startle_score"], STARTLE_SUSPICIOUS, STARTLE_CONFIRMED)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Console output
# ─────────────────────────────────────────────────────────────────────────────
print()
print(f"{BOLD}{'='*72}{RESET}")
print(f"{BOLD}  3-TOOL COMPARISON + BASELINE ANALYSIS — Game Session 2026-03-18{RESET}")
print(f"{BOLD}{'='*72}{RESET}")

# ── Phase comparison block ─────────────────────────────────────────────────────
print(f"\n{CYAN}[PHASE COMPARISON: NEUTRAL BASELINE vs GAME SESSION]{RESET}")
print(f"  {'Signal':<30} {'Neutral':>9} {'Game':>9} {'Delta':>8}  {'Verdict'}")
print(f"  {'-'*68}")

rows = [
    ("HS arousal",         BL["hs_arousal"],     m["hs_arousal"].mean(),       "RELIABLE  -- use as primary fear proxy"),
    ("HS valence",         BL["hs_valence"],      m["hs_valence"].mean(),       "USELESS   -- flat across conditions"),
    ("HS anger%",          BL["hs_anger"]*100,    m["hs_anger"].mean()*100,     "BIASED    -- 18% is resting face"),
    ("MP tension",         BL["mp_tension"],      m["tension"].mean(),          "RELIABLE  -- real but subtle"),
    ("MP browDown avg",    BL["mp_browDown"],     m[["browDownLeft","browDownRight"]].mean(axis=1).mean(), "BIASED    -- gaming concentration"),
    ("DF sad",             BL["df_sad"],          m["df_sad"].mean(),            "ARTIFACT  -- tension reads as sad"),
    ("DF fear/arousal",    BL["df_fear"],         m["df_fear"].mean(),           "USELESS   -- lower during game than rest"),
    ("DF neutral%",        BL["df_neutral_pct"],  (m["df_dominant"].str.lower()=="neutral").mean()*100, "REFERENCE -- resting face = neutral"),
]
for label, n_val, g_val, verdict in rows:
    delta = g_val - n_val
    sign = "+" if delta >= 0 else ""
    print(f"  {label:<30} {n_val:>9.3f} {g_val:>9.3f} {sign}{delta:>7.3f}  {verdict}")

# ── Confirmed startles ─────────────────────────────────────────────────────────
print(f"\n{RED}[CONFIRMED STARTLES]{RESET}  (startle_score > {STARTLE_CONFIRMED})")
print(f"  {'#':<4} {'t_start':>8} {'t_end':>8}  {'peak_score':>10}  {'hs_arousal_mean':>16}  {'df_sad_mean':>12}")
print(f"  {'-'*64}")
for i, w in enumerate(confirmed_wins):
    win_rows = m[(m["timestamp"] >= w["t_start"]) & (m["timestamp"] <= w["t_end"] + 0.3)]
    print(f"  {i+1:<4} {w['t_start']:>7.1f}s {w['t_end']:>7.1f}s  {w['peak']:>10.2f}"
          f"  {win_rows['hs_arousal'].mean():>16.3f}  {win_rows['df_sad'].mean():>12.3f}")

# ── Suspicious sub-threshold startles ─────────────────────────────────────────
print(f"\n{YELL}[SUSPICIOUS SUB-THRESHOLD]{RESET}  ({STARTLE_SUSPICIOUS} < startle_score <= {STARTLE_CONFIRMED})")
print(f"  {'#':<4} {'t_start':>8} {'t_end':>8}  {'peak_score':>10}  {'hs_arousal_mean':>16}  {'bc_arousal_mean':>16}")
print(f"  {'-'*68}")
for i, w in enumerate(suspicious_wins):
    win_rows = m[(m["timestamp"] >= w["t_start"]) & (m["timestamp"] <= w["t_end"] + 0.3)]
    print(f"  {i+1:<4} {w['t_start']:>7.1f}s {w['t_end']:>7.1f}s  {w['peak']:>10.2f}"
          f"  {win_rows['hs_arousal'].mean():>16.3f}  {win_rows['bc_arousal'].mean():>16.3f}")

# ── Per-timestamp inspection ───────────────────────────────────────────────────
print(f"\n{GREEN}[TIMESTAMP INSPECTION]{RESET}  (+-4s windows, flagged rows: bc_arousal>0.4 OR startle>1.0 OR bc_tension>0.03)")
for t_center, label in USER_MOMENTS.items():
    win = m[(m["timestamp"] >= t_center - 4) & (m["timestamp"] <= t_center + 4)].copy()
    if len(win) == 0:
        print(f"\n  t={t_center}s ({label}): NO DATA")
        continue
    print(f"\n  {BOLD}t={t_center}s  |  {label}{RESET}")
    print(f"  {'t':>7}  {'tension':>8}  {'startle':>8}  {'hs_arous':>9}  {'hs_dom':<10}  {'df_fear':>8}  {'df_sad':>8}  {'df_dom':<10}  flag")
    print(f"  {'-'*90}")
    for _, r in win.iterrows():
        flag = ""
        if (r["bc_arousal"] > 0.4 or r["startle_score"] > 1.0 or r["bc_tension"] > 0.03):
            parts = []
            if r["bc_arousal"] > 0.4:   parts.append("AROUSAL")
            if r["startle_score"] > 1.0: parts.append("STARTLE")
            if r["bc_tension"] > 0.03:   parts.append("TENSION")
            flag = f"[{'+'.join(parts)}]"
        hs_dom = str(r.get("hs_dominant",""))[:10]
        df_dom = str(r.get("df_dominant",""))[:10]
        row_str = (f"  {r['timestamp']:>6.1f}s  {r['tension']:>8.4f}  {r['startle_score']:>8.3f}"
                   f"  {r['hs_arousal']:>9.3f}  {hs_dom:<10}  {r['df_fear']:>8.4f}"
                   f"  {r['df_sad']:>8.4f}  {df_dom:<10}  {flag}")
        if flag:
            print(f"{YELL}{row_str}{RESET}")
        else:
            print(f"{DIM}{row_str}{RESET}")

print(f"\n{'='*72}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 4-panel plot
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating plot -> {OUT_PNG}")

fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True,
                         facecolor=DARK_BG,
                         gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5]})
fig.subplots_adjust(hspace=0.06, left=0.07, right=0.97, top=0.93, bottom=0.06)

t = m["timestamp"].values

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.yaxis.label.set_color("#aaaacc")
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.grid(axis="both", color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)

def shade_all(ax):
    for w in confirmed_wins:
        ax.axvspan(w["t_start"] - 0.2, w["t_end"] + 0.3, alpha=0.18, color="#ff3333", zorder=0)
    for w in suspicious_wins:
        ax.axvspan(w["t_start"] - 0.1, w["t_end"] + 0.2, alpha=0.14, color="#ff8800", zorder=0)
    for t_m, lbl in USER_MOMENTS.items():
        ax.axvline(t_m, color="#ffffff", linewidth=0.6, linestyle="--", alpha=0.4, zorder=1)

# ── Panel 1: MediaPipe ─────────────────────────────────────────────────────────
ax1 = axes[0]
style_ax(ax1)
shade_all(ax1)
ax1.plot(t, smooth(m["tension"].values), color="#4db8ff", linewidth=1.2,
         label="tension (smoothed)", alpha=0.9)
ax1.axhline(BL["mp_tension"], color="#4db8ff", linewidth=0.7, linestyle=":",
            alpha=0.5, label=f"neutral baseline ({BL['mp_tension']:.3f})")
# startle score as dots
conf_mask = m["startle_score"] > STARTLE_CONFIRMED
susp_mask = (m["startle_score"] > STARTLE_SUSPICIOUS) & (~conf_mask)
ax1.scatter(m.loc[conf_mask,"timestamp"], m.loc[conf_mask,"startle_score"].clip(upper=0.6),
            color="#ff4444", s=18, zorder=5, label="confirmed startle")
ax1.scatter(m.loc[susp_mask,"timestamp"], m.loc[susp_mask,"startle_score"].clip(upper=0.6) * 0.3,
            color="#ff9900", s=10, zorder=5, label="suspicious startle", marker="^")
ax1.set_ylabel("MediaPipe", fontsize=9)
ax1.set_ylim(0, max(m["tension"].max() * 1.2, 0.5))
ax1.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
           labelcolor="#aaaacc", framealpha=0.8)
ax1.set_title("3-Tool Comparison — Game Session 2026-03-18  |  "
              "red=confirmed startle  orange=suspicious  white dashes=user moments",
              fontsize=9, color="#ddddff", pad=5)

# User moment labels on top panel
for t_m, lbl in USER_MOMENTS.items():
    ax1.text(t_m, ax1.get_ylim()[1] * 0.92, f"t={t_m}", fontsize=5.5,
             color="#ddddaa", ha="center", rotation=90, va="top", alpha=0.8)

# ── Panel 2: HSEmotion arousal only ───────────────────────────────────────────
ax2 = axes[1]
style_ax(ax2)
shade_all(ax2)
ax2.plot(t, smooth(m["hs_arousal"].values), color="#ff4444", linewidth=1.3,
         label="arousal (smoothed) — RELIABLE", alpha=0.9)
ax2.axhline(BL["hs_arousal"], color="#ff4444", linewidth=0.7, linestyle=":",
            alpha=0.5, label=f"neutral baseline ({BL['hs_arousal']:.3f})")
ax2.axhline(0.6, color="#ffaa00", linewidth=0.5, linestyle="--", alpha=0.35,
            label="strong arousal (0.6)")
# Valence dimmed for reference (proven useless)
ax2.plot(t, smooth(m["hs_valence"].values, 30), color="#888888", linewidth=0.6,
         linestyle=":", label=f"valence (useless, delta={m['hs_valence'].mean()-BL['hs_valence']:+.3f})",
         alpha=0.4)
ax2.axhline(BL["hs_valence"], color="#888888", linewidth=0.5, linestyle=":",
            alpha=0.3)
ax2.set_ylabel("HSEmotion", fontsize=9)
ax2.set_ylim(-0.8, 1.1)
ax2.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
           labelcolor="#aaaacc", framealpha=0.8)

# ── Panel 3: DeepFace ──────────────────────────────────────────────────────────
ax3 = axes[2]
style_ax(ax3)
shade_all(ax3)
ax3.plot(t, smooth(m["df_sad"].values, 20), color="#00ccff", linewidth=1.1,
         label="sad (tension artifact)", alpha=0.85)
ax3.plot(t, smooth(m["df_fear"].values, 20), color="#ffdd00", linewidth=0.9,
         linestyle="--", label="fear (unreliable)", alpha=0.75)
ax3.axhline(BL["df_sad"],  color="#00ccff", linewidth=0.7, linestyle=":",
            alpha=0.5, label=f"sad baseline ({BL['df_sad']:.3f})")
ax3.axhline(BL["df_fear"], color="#ffdd00", linewidth=0.5, linestyle=":",
            alpha=0.4, label=f"fear baseline ({BL['df_fear']:.3f})")
ax3.set_ylabel("DeepFace", fontsize=9)
ax3.set_ylim(0, 0.7)
ax3.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
           labelcolor="#aaaacc", framealpha=0.8)

# ── Panel 4: Baseline-corrected composite ─────────────────────────────────────
ax4 = axes[3]
style_ax(ax4)
shade_all(ax4)
ax4.fill_between(t, smooth(m["composite"].values, 20), alpha=0.3, color="#00ff88")
ax4.plot(t, smooth(m["composite"].values, 20), color="#00ff88", linewidth=1.3,
         label="BC composite (arousal 50% + tension 30% + df_sad 20%)")
ax4.axhline(0.3, color="#ffaa00", linewidth=0.7, linestyle="--", alpha=0.5, label="threshold 0.3")
ax4.set_ylabel("Composite", fontsize=9)
ax4.set_ylim(0, 1.05)
ax4.set_xlabel("Time (s)", fontsize=9, color="#aaaacc")
ax4.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
           labelcolor="#aaaacc", framealpha=0.8)

# Legend
p_conf = mpatches.Patch(color="#ff3333", alpha=0.5, label="Confirmed startle")
p_susp = mpatches.Patch(color="#ff8800", alpha=0.5, label="Suspicious sub-threshold")
p_user = mpatches.Patch(color="#ffffff", alpha=0.4, label="User-identified moment")
fig.legend(handles=[p_conf, p_susp, p_user], loc="lower center", ncol=3,
           fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL,
           labelcolor="#ddddff", framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved: {OUT_PNG}")
print("Done.")
