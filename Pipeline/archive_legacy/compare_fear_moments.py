# compare_fear_moments.py
# Cross-tool fear analysis: HSEmotion vs MediaPipe
# Identifies non-startle fear moments and explains anger dominance.
# Usage: conda activate facade && python Pipeline/compare_fear_moments.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
LOGS = Path(__file__).resolve().parent / "logs"
SESS = LOGS / "sessions"

MP_CSV  = SESS / "S01_game_bright" / "20260318_213728_mediapipe_S01_game_bright.csv"
HS_CSV  = SESS / "S01_chase_bright" / "20260318_214136_hsemotion_S01_chase_bright.csv"
OUT_PNG = LOGS / "comparisons" / "compare_fear_20260318_game.png"

# ── Thresholds ─────────────────────────────────────────────────────────────────
STARTLE_THRESH   = 3.0    # startle_score from mediapipe
FEAR_SCORE_THRESH = 0.30  # composite fear score threshold (lowered: browInnerUp stays low in gaming)
MIN_WINDOW_GAP_S  = 2.0   # merge windows closer than this
MIN_WINDOW_DUR_S  = 0.3   # discard windows shorter than this

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading CSVs...")
mp = pd.read_csv(MP_CSV)
hs = pd.read_csv(HS_CSV)

# Keep only face-detected rows for stats, but keep all rows for timeline
mp_det = mp[mp["face_detected"] == 1].copy()
hs_det = hs[hs["face_detected"] == 1].copy()

print(f"  MediaPipe : {len(mp):,} frames, {mp['face_detected'].mean()*100:.1f}% face detected")
print(f"  HSEmotion : {len(hs):,} frames, {hs['face_detected'].mean()*100:.1f}% face detected")

# ── 2. Align by timestamp (merge_asof, 50ms tolerance) ────────────────────────
print("Aligning timestamps...")
mp_s = mp_det.sort_values("timestamp").reset_index(drop=True)
hs_s = hs_det.sort_values("timestamp").reset_index(drop=True)

merged = pd.merge_asof(
    mp_s,
    hs_s[["timestamp","valence","arousal","anger","fear","happiness",
          "neutral","sadness","surprise","contempt","dominant","dominant_score",
          "face_detected"]].rename(columns={
              "valence":        "hs_valence",
              "arousal":        "hs_arousal",
              "anger":          "hs_anger",
              "fear":           "hs_fear",
              "happiness":      "hs_happiness",
              "neutral":        "hs_neutral",
              "sadness":        "hs_sadness",
              "surprise":       "hs_surprise",
              "contempt":       "hs_contempt",
              "dominant":       "hs_dominant",
              "dominant_score": "hs_dominant_score",
              "face_detected":  "hs_face_detected",
          }),
    on="timestamp",
    direction="nearest",
    tolerance=0.05,
)
# Drop rows where HSEmotion had no match
merged = merged.dropna(subset=["hs_arousal"]).copy()
print(f"  Aligned frames: {len(merged):,}")

# ── 3. Composite fear score ────────────────────────────────────────────────────
# Channel 1: MediaPipe tension (normalized to session max)
t_max = merged["tension"].max()
chan_mp_tension = merged["tension"] / t_max if t_max > 0 else merged["tension"]

# Channel 2: MediaPipe fear-AU (browInnerUp + eyeWide - browDown, no smile)
# browDown columns: browDownLeft, browDownRight
# eyeWide: eyeWideLeft, eyeWideRight
fear_au_raw = (
    merged["browInnerUp"]
    + 0.5 * merged["eyeWideLeft"]
    + 0.5 * merged["eyeWideRight"]
    - 0.5 * merged["browDownLeft"]
    - 0.5 * merged["browDownRight"]
).clip(lower=0)
au_max = fear_au_raw.max()
chan_mp_fear_au = fear_au_raw / au_max if au_max > 0 else fear_au_raw

# Channel 3: HSEmotion arousal, only when valence is negative (filter excited-happy)
hs_arousal_filtered = merged["hs_arousal"].copy()
hs_arousal_filtered[merged["hs_valence"] >= -0.2] = 0.0
hs_ar_max = hs_arousal_filtered.max()
chan_hs_arousal = hs_arousal_filtered / hs_ar_max if hs_ar_max > 0 else hs_arousal_filtered

merged["chan_mp_tension"]  = chan_mp_tension.values
merged["chan_mp_fear_au"]  = chan_mp_fear_au.values
merged["chan_hs_arousal"]  = chan_hs_arousal.values

merged["fear_score"] = (
    0.35 * merged["chan_mp_tension"]
    + 0.35 * merged["chan_mp_fear_au"]
    + 0.30 * merged["chan_hs_arousal"]
)

is_startle = merged["startle_score"] > STARTLE_THRESH
merged["is_startle"] = is_startle

# ── 4. Identify fear candidate windows ────────────────────────────────────────
is_candidate = (merged["fear_score"] > FEAR_SCORE_THRESH) & (~is_startle)

# Collapse consecutive candidate frames into windows
merged = merged.reset_index(drop=True)
is_candidate = is_candidate.reset_index(drop=True)

windows = []
in_win = False
win = {}
for i in range(len(merged)):
    row = merged.iloc[i]
    t = row["timestamp"]
    if is_candidate.iloc[i]:
        if not in_win:
            win = {"t_start": t, "t_end": t,
                   "peak_fear": row["fear_score"],
                   "peak_arousal": row["hs_arousal"],
                   "peak_tension": row["tension"],
                   "peak_browInnerUp": row["browInnerUp"]}
            in_win = True
        else:
            win["t_end"] = t
            win["peak_fear"]        = max(win["peak_fear"],        row["fear_score"])
            win["peak_arousal"]     = max(win["peak_arousal"],     row["hs_arousal"])
            win["peak_tension"]     = max(win["peak_tension"],     row["tension"])
            win["peak_browInnerUp"] = max(win["peak_browInnerUp"], row["browInnerUp"])
    else:
        if in_win:
            windows.append(win)
            in_win = False
if in_win:
    windows.append(win)

# Merge windows closer than MIN_WINDOW_GAP_S
merged_windows = []
for w in windows:
    if merged_windows and (w["t_start"] - merged_windows[-1]["t_end"]) < MIN_WINDOW_GAP_S:
        prev = merged_windows[-1]
        prev["t_end"]          = w["t_end"]
        prev["peak_fear"]      = max(prev["peak_fear"],      w["peak_fear"])
        prev["peak_arousal"]   = max(prev["peak_arousal"],   w["peak_arousal"])
        prev["peak_tension"]   = max(prev["peak_tension"],   w["peak_tension"])
        prev["peak_browInnerUp"] = max(prev["peak_browInnerUp"], w["peak_browInnerUp"])
    else:
        merged_windows.append(dict(w))

# Filter by minimum duration
merged_windows = [w for w in merged_windows
                  if (w["t_end"] - w["t_start"]) >= MIN_WINDOW_DUR_S]

# Sort by peak fear descending
merged_windows.sort(key=lambda w: w["peak_fear"], reverse=True)

# Also collect startle windows for plotting
startle_frames = merged[merged["is_startle"]]
startle_windows = []
in_win = False
for idx, row in startle_frames.iterrows():
    t = row["timestamp"]
    if not in_win:
        sw = {"t_start": t, "t_end": t}
        in_win = True
    else:
        if t - sw["t_end"] < 1.0:
            sw["t_end"] = t
        else:
            startle_windows.append(sw)
            sw = {"t_start": t, "t_end": t}
    sw["t_end"] = t
if in_win:
    startle_windows.append(sw)

# ── 5. Console output ──────────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
YELL  = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

print()
print(f"{BOLD}{'='*70}{RESET}")
print(f"{BOLD}  CROSS-TOOL FEAR ANALYSIS — Game Session 2026-03-18{RESET}")
print(f"{BOLD}{'='*70}{RESET}")

# Startle summary
n_startles = len(startle_windows)
startle_times = [f"t={sw['t_start']:.1f}s" for sw in startle_windows]
print(f"\n{CYAN}[STARTLES]{RESET}  {n_startles} events detected by MediaPipe velocity threshold")
for sw in startle_windows:
    mp_row = merged[(merged["timestamp"] >= sw["t_start"]) &
                    (merged["timestamp"] <= sw["t_end"] + 0.5)]
    peak_ss = mp_row["startle_score"].max() if len(mp_row) else 0
    print(f"   t={sw['t_start']:.1f}–{sw['t_end']:.1f}s  |  peak startle_score={peak_ss:.2f}")

# Fear candidates
print(f"\n{YELL}[NON-STARTLE FEAR CANDIDATES]{RESET}  "
      f"(fear_score>{FEAR_SCORE_THRESH}, not a startle event)")
print(f"  {'#':<4} {'t_start':>8} {'t_end':>8} {'dur':>6} "
      f"{'peak_fear':>10} {'peak_arousal':>13} {'peak_tension':>13} {'peak_browUp':>10}")
print(f"  {'-'*75}")
for i, w in enumerate(merged_windows[:15]):
    dur = w["t_end"] - w["t_start"]
    print(f"  {i+1:<4} {w['t_start']:>7.1f}s {w['t_end']:>7.1f}s {dur:>5.1f}s "
          f"  {w['peak_fear']:>9.3f}  {w['peak_arousal']:>12.3f}  "
          f"{w['peak_tension']:>12.3f}  {w['peak_browInnerUp']:>9.3f}")

# Top HSEmotion arousal peaks (independent)
print(f"\n{CYAN}[TOP HSEMOTION AROUSAL PEAKS]{RESET}  (all, including startles)")
hs_top = merged.nlargest(8, "hs_arousal")[
    ["timestamp","hs_arousal","hs_valence","hs_dominant","tension","browInnerUp","is_startle"]
].reset_index(drop=True)
print(f"  {'t':>8}  {'arousal':>8}  {'valence':>8}  {'dominant':<12}  "
      f"{'mp_tension':>10}  {'browInnerUp':>10}  {'startle':>8}")
print(f"  {'-'*70}")
for _, r in hs_top.iterrows():
    flag = "STARTLE" if r["is_startle"] else "       "
    print(f"  {r['timestamp']:>7.1f}s  {r['hs_arousal']:>8.3f}  {r['hs_valence']:>8.3f}  "
          f"{str(r['hs_dominant']):<12}  {r['tension']:>10.3f}  "
          f"{r['browInnerUp']:>10.3f}  {flag}")

# ── Why anger dominates ────────────────────────────────────────────────────────
print(f"\n{GREEN}[WHY HSEmotion SHOWS ANGER NOT FEAR]{RESET}")
brow_down_mean  = merged[["browDownLeft","browDownRight"]].mean(axis=1).mean()
brow_inner_mean = merged["browInnerUp"].mean()
pct_brow_down_dominant = (
    merged[["browDownLeft","browDownRight"]].mean(axis=1) > merged["browInnerUp"]
).mean() * 100

mean_valence_all   = merged["hs_valence"].mean()
mean_arousal_all   = merged["hs_arousal"].mean()
dom_lower = merged["hs_dominant"].str.lower()
mean_arousal_anger = merged.loc[dom_lower == "anger", "hs_arousal"].mean()
mean_arousal_fear  = merged.loc[dom_lower == "fear",  "hs_arousal"].mean()
anger_pct = (dom_lower == "anger").mean() * 100

eye_wide_mean = merged[["eyeWideLeft","eyeWideRight"]].mean(axis=1).mean()

print(f"  Anger dominant in {anger_pct:.1f}% of aligned frames")
print(f"  Mean browDown (L+R avg):  {brow_down_mean:.3f}")
print(f"  Mean browInnerUp:         {brow_inner_mean:.3f}")
print(f"  Mean eyeWide (L+R avg):   {eye_wide_mean:.3f}")
print(f"  browDown > browInnerUp in {pct_brow_down_dominant:.1f}% of frames")
print(f"  -> Gaming concentration face = furrowed brow -> anger label in HSEmotion")
print()
print(f"  VA space (all frames):   valence={mean_valence_all:.3f}, arousal={mean_arousal_all:.3f}")
print(f"  Anger-dominant frames:   mean arousal={mean_arousal_anger:.3f}")
if not np.isnan(mean_arousal_fear):
    print(f"  Fear-dominant frames:    mean arousal={mean_arousal_fear:.3f}")
else:
    print(f"  Fear-dominant frames:    too few to average")
print(f"  -> Both fear+anger share high arousal + negative valence quadrant")
print(f"  -> The brow direction (down=anger, inner-up=fear) is the key discriminator")
print(f"  -> MediaPipe's browInnerUp+eyeWide channels are more sensitive to true fear")

print(f"\n{'='*70}\n")

# ── 6. Plot ────────────────────────────────────────────────────────────────────
print(f"Generating plot -> {OUT_PNG}")

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COL = "#2a2a4a"

fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True,
                         facecolor=DARK_BG,
                         gridspec_kw={"height_ratios": [2, 2, 1.5]})
fig.subplots_adjust(hspace=0.08, left=0.07, right=0.97, top=0.93, bottom=0.07)

t_mp = merged["timestamp"].values
t_hs = merged["timestamp"].values  # same after merge

def apply_style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.yaxis.label.set_color("#aaaacc")
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(axis="both", color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)

def shade_events(ax):
    for sw in startle_windows:
        ax.axvspan(sw["t_start"] - 0.3, sw["t_end"] + 0.3,
                   alpha=0.20, color="#ff3333", zorder=0)
    for w in merged_windows[:10]:
        ax.axvspan(w["t_start"], w["t_end"],
                   alpha=0.18, color="#ff8800", zorder=0)

# ── Panel 1: MediaPipe ────────────────────────────────────────────────────────
ax1 = axes[0]
apply_style(ax1)
shade_events(ax1)
# Smooth with rolling window for readability
smooth = lambda s, w=15: pd.Series(s).rolling(w, center=True, min_periods=1).mean().values
ax1.plot(t_mp, smooth(merged["tension"].values),
         color="#4db8ff", linewidth=1.2, label="tension (smoothed)", alpha=0.9)
ax1.plot(t_mp, merged["browInnerUp"].values,
         color="#00ffcc", linewidth=0.7, linestyle="--", label="browInnerUp", alpha=0.7)
eye_wide_avg = merged[["eyeWideLeft","eyeWideRight"]].mean(axis=1).values
ax1.plot(t_mp, smooth(eye_wide_avg, 10),
         color="#ffffff", linewidth=0.6, linestyle=":", label="eyeWide avg", alpha=0.5)
ax1.set_ylabel("MediaPipe", fontsize=9)
ax1.set_ylim(0, max(merged["tension"].max() * 1.15, 0.6))
ax1.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
           labelcolor="#aaaacc", framealpha=0.8)
ax1.set_title("Cross-Tool Fear Analysis — Game Session 2026-03-18  |  "
              "red=startle  orange=fear candidate",
              fontsize=10, color="#ddddff", pad=6)

# ── Panel 2: HSEmotion ────────────────────────────────────────────────────────
ax2 = axes[1]
apply_style(ax2)
shade_events(ax2)
ax2.plot(t_hs, smooth(merged["hs_arousal"].values),
         color="#ff4444", linewidth=1.2, label="arousal (smoothed)", alpha=0.9)
ax2.plot(t_hs, merged["hs_fear"].values,
         color="#ffdd00", linewidth=0.8, linestyle="--", label="fear prob", alpha=0.8)
ax2.plot(t_hs, smooth(merged["hs_anger"].values, 20),
         color="#888899", linewidth=0.7, linestyle=":", label="anger (smoothed)", alpha=0.55)
ax2.set_ylabel("HSEmotion", fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.axhline(0.6, color="#ff4444", linewidth=0.5, linestyle="--", alpha=0.4)
ax2.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
           labelcolor="#aaaacc", framealpha=0.8)

# ── Panel 3: Composite fear score ─────────────────────────────────────────────
ax3 = axes[2]
apply_style(ax3)
shade_events(ax3)
ax3.fill_between(t_mp, merged["fear_score"].values, alpha=0.35, color="#00cc66")
ax3.plot(t_mp, smooth(merged["fear_score"].values, 20),
         color="#00ff88", linewidth=1.4, label="fear_score (smoothed)")
ax3.axhline(FEAR_SCORE_THRESH, color="#ffaa00", linewidth=0.9, linestyle="--",
            alpha=0.7, label=f"threshold ({FEAR_SCORE_THRESH})")

# Annotate top fear windows
annotated = 0
for w in merged_windows[:6]:
    t_mid = (w["t_start"] + w["t_end"]) / 2
    score = w["peak_fear"]
    ax3.annotate(f"t={t_mid:.0f}s",
                 xy=(t_mid, score), xytext=(t_mid, min(score + 0.08, 0.95)),
                 fontsize=6.5, color="#ffaa00", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#ffaa00", lw=0.6))
    annotated += 1

ax3.set_ylabel("Fear score", fontsize=9)
ax3.set_ylim(0, 1.05)
ax3.set_xlabel("Time (s)", fontsize=9, color="#aaaacc")
ax3.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COL,
           labelcolor="#aaaacc", framealpha=0.8)

# Legend patches
patch_s = mpatches.Patch(color="#ff3333", alpha=0.5, label="Startle (MediaPipe)")
patch_f = mpatches.Patch(color="#ff8800", alpha=0.5, label="Fear candidate")
fig.legend(handles=[patch_s, patch_f], loc="lower center", ncol=2,
           fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL,
           labelcolor="#ddddff", framealpha=0.9,
           bbox_to_anchor=(0.5, 0.0))

plt.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved: {OUT_PNG}")
print("Done.")
