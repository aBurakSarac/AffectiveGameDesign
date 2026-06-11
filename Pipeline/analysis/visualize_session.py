"""
La Facade Fissuree — Session Visualizer
========================================
Pattern: [Utility / Visualization] — auto-detects the FER tool from CSV columns
and routes to the appropriate plot function (MediaPipe, HSEmotion, or categorical).

Reads any FER tool CSV log and produces a timeline plot. Auto-detects
which tool produced the CSV from column names.

Supported tools:
  - MediaPipe: Tension+Valence timeline, blendshape detail, state bar chart
  - DeepFace/Py-Feat: Arousal timeline, 7 emotion lines, emotion distribution bar
  - HSEmotion: Valence+Arousal timeline, 8 emotion lines, emotion distribution bar

Usage:
    python visualize_session.py                          # uses latest log (any tool)
    python visualize_session.py logs/mediapipe_XYZ.csv   # specific file

Output: a PNG saved next to the CSV (same name, _plot.png extension).
"""

import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter


# ---- Tool detection ----
def detect_tool(columns):
    """Detect which FER tool produced a CSV from its column names."""
    cols = set(columns)

    # Fusion scripts (MP+HS or MP+HS+rPPG) have prefixed columns
    if ("mp_tension" in cols or "mp_face_valence" in cols) and \
       ("hs_arousal" in cols or "hs_dominant" in cols):
        return "fusion"

    # Single tools
    if "tension" in cols and "face_valence" in cols:
        return "mediapipe"
    if ("valence" in cols or "hs_valence" in cols) and \
       ("arousal" in cols or "hs_arousal" in cols):
        return "hsemotion"
    if "arousal" in cols and "angry" in cols:
        return "deepface"
    if "arousal" in cols and "anger" in cols:
        return "pyfeat"
    return "unknown"


# ---- Session metadata lookup ----
def load_session_meta(csv_path):
    """Try to find matching session record in sessions.json."""
    # sessions.json lives in the logs/ root (parent of sessions/)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    registry = os.path.join(log_dir, "sessions.json")
    csv_basename = os.path.basename(csv_path)
    if not os.path.isfile(registry):
        return None
    try:
        with open(registry, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("sessions", []):
            # Match by filename (csv_path may include sessions/ prefix)
            if os.path.basename(s.get("csv_path", "")) == csv_basename:
                return s
    except (json.JSONDecodeError, IOError):
        pass
    return None


# ---- Common helpers ----
def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def pick_latest_csv():
    """Pick the latest FER CSV (any tool type, any naming convention).

    Searches logs/sessions/ subfolders recursively.
    Excludes _temp.csv files (in-progress sessions).
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    sessions_dir = os.path.join(log_dir, "sessions")
    candidates = []
    if os.path.isdir(sessions_dir):
        for root, _dirs, files in os.walk(sessions_dir):
            for f in files:
                if (f.endswith(".csv")
                        and not f.endswith("_temp.csv")
                        and not f.endswith("_summary.csv")):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"No FER CSV found in {sessions_dir}")
    # Sort by filename (timestamp prefix ensures chronological order)
    candidates.sort(key=lambda p: os.path.basename(p))
    return candidates[-1]


def smooth(arr, w):
    """Rolling mean smoothing."""
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def style_ax(ax):
    """Apply dark theme to an axis."""
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="#cccccc")
    for sp in ax.spines.values():
        sp.set_color("#444")


def build_subtitle(csv_path, meta, n_frames, duration_s, fps_actual):
    """Build plot title with optional session metadata."""
    fname = os.path.basename(csv_path)
    title = f"{fname}  |  {n_frames:,} frames  |  {duration_s:.0f}s  |  {fps_actual:.0f} FPS"
    if meta:
        parts = []
        if meta.get("subject_id"):
            parts.append(f"Subject: {meta['subject_id']}")
        if meta.get("content_type"):
            parts.append(f"Content: {meta['content_type']}")
        if meta.get("lighting"):
            parts.append(f"Light: {meta['lighting']}")
        if parts:
            title += "\n" + "  |  ".join(parts)
    return title


# ---- Context state detection (mirrors test_mediapipe.py logic) ----
STATE_COLORS = {
    "JOY":    "#00c853",
    "FEAR":   "#d50000",
    "CONC":   "#ffd600",
    "SAD":    "#e65100",
    "STRESS": "#aa00ff",
    "---":    "#e0e0e0",
}
STATE_ALPHA = 0.18
ALL_STATES = ["JOY", "FEAR", "CONC", "SAD", "STRESS", "---"]

# Mirror of STARTLE_VELOCITY_THRESHOLD in test_mediapipe.py — keep in sync manually.
# CALIBRATION PENDING — update both if the threshold is adjusted after sessions.
STARTLE_VELOCITY_THRESHOLD_DISPLAY = 3.0


def get_ctx_tag(row):
    smile = float(row.get("mouthSmileLeft", 0))
    eye_wd = max(float(row.get("eyeWideLeft", 0)), float(row.get("eyeWideRight", 0)))
    brow_inner = float(row.get("browInnerUp", 0))
    brow_dn = max(float(row.get("browDownLeft", 0)), float(row.get("browDownRight", 0)))
    frown = max(float(row.get("mouthFrownLeft", 0)), float(row.get("mouthFrownRight", 0)))
    press = max(float(row.get("mouthPressLeft", 0)), float(row.get("mouthPressRight", 0)))

    if smile > 0.3:
        return "JOY"
    elif eye_wd > 0.3 and brow_inner > 0.2:
        return "FEAR"
    elif brow_dn > 0.2 and press < 0.05:
        return "CONC"
    elif frown > press and frown > 0.1:
        return "SAD"
    elif press > 0.15:
        return "STRESS"
    else:
        return "---"


def draw_state_bands(ax, t, tags):
    if not tags:
        return
    prev_state = tags[0]
    seg_start = t[0]
    for i in range(1, len(tags)):
        if tags[i] != prev_state or i == len(tags) - 1:
            seg_end = t[i]
            ax.axvspan(seg_start, seg_end,
                       color=STATE_COLORS[prev_state],
                       alpha=STATE_ALPHA, linewidth=0)
            prev_state = tags[i]
            seg_start = t[i]


def draw_state_bar(ax, state_pct):
    """Draw horizontal bar chart of state distribution."""
    style_ax(ax)
    states_show = [s for s in ALL_STATES if state_pct.get(s, 0) > 0]
    pcts = [state_pct[s] for s in states_show]
    cols = [STATE_COLORS[s] for s in states_show]

    if not pcts:
        return

    bars = ax.barh(states_show, pcts, color=cols, height=0.6)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="left",
                color="#cccccc", fontsize=9)

    ax.set_xlim(0, max(pcts) * 1.2)
    ax.set_xlabel("% of session time", color="#cccccc", fontsize=9)
    ax.set_title("Time per context state", color="#cccccc", fontsize=9)
    ax.tick_params(axis="y", labelcolor="#cccccc", labelsize=9)
    ax.tick_params(axis="x", labelcolor="#999999", labelsize=8)


# ---- Emotion colors for categorical tools ----
EMOTION_COLORS = {
    "angry": "#ff4444", "anger": "#ff4444",
    "disgust": "#88cc00",
    "fear": "#ff8800",
    "happy": "#00cc44", "happiness": "#00cc44",
    "sad": "#4488ff", "sadness": "#4488ff",
    "surprise": "#cc44ff",
    "neutral": "#aaaaaa",
    "contempt": "#ff88cc",
}


# ========== MEDIAPIPE PLOT ==========
def plot_mediapipe(rows, t, csv_path, meta):
    tension = np.array([float(r["tension"]) for r in rows])
    valence = np.array([float(r["face_valence"]) for r in rows])
    smile_arr = np.array([float(r["smile_level"]) for r in rows])

    eye_wide = np.array([max(float(r.get("eyeWideLeft", 0)),
                             float(r.get("eyeWideRight", 0))) for r in rows])
    brow_up = np.array([float(r.get("browInnerUp", 0)) for r in rows])
    jaw_open = np.array([float(r.get("jawOpen", 0)) for r in rows])
    mouth_press = np.array([max(float(r.get("mouthPressLeft", 0)),
                                float(r.get("mouthPressRight", 0))) for r in rows])
    brow_down = np.array([max(float(r.get("browDownLeft", 0)),
                              float(r.get("browDownRight", 0))) for r in rows])

    tags = [get_ctx_tag(r) for r in rows]

    fps_est = len(t) / (t[-1] - t[0]) if t[-1] > t[0] else 30
    win = max(1, int(fps_est))

    tension_s = smooth(tension, win)
    valence_s = smooth(valence, win)
    smile_s = smooth(smile_arr, win)
    eye_wide_s = smooth(eye_wide, win)
    press_s = smooth(mouth_press, win)

    state_counts = Counter(tags)
    total = len(tags)
    state_pct = {s: state_counts.get(s, 0) / total * 100 for s in ALL_STATES}

    # Build figure — 4 panels when velocity data present, 3 panels for old CSVs
    has_velocity = "startle_score" in rows[0]

    if has_velocity:
        fig = plt.figure(figsize=(18, 14), facecolor="#1a1a2e")
        gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 2, 1.5, 1], hspace=0.38)
        ax_main = fig.add_subplot(gs[0])
        ax_feat = fig.add_subplot(gs[1], sharex=ax_main)
        ax_vel  = fig.add_subplot(gs[2], sharex=ax_main)
        ax_bar  = fig.add_subplot(gs[3])
    else:
        fig = plt.figure(figsize=(18, 11), facecolor="#1a1a2e")
        gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 2, 1], hspace=0.35)
        ax_main = fig.add_subplot(gs[0])
        ax_feat = fig.add_subplot(gs[1], sharex=ax_main)
        ax_bar  = fig.add_subplot(gs[2])
        ax_vel  = None

    for ax in [ax_main, ax_feat] + ([ax_vel] if ax_vel else []):
        style_ax(ax)

    draw_state_bands(ax_main, t, tags)
    draw_state_bands(ax_feat, t, tags)
    if ax_vel is not None:
        draw_state_bands(ax_vel, t, tags)

    # Main panel: Tension & Valence
    ax_main.plot(t, tension, color="#ff4444", alpha=0.2, linewidth=0.5)
    ax_main.plot(t, tension_s, color="#ff4444", linewidth=2.0, label="Tension (smoothed)")
    ax_main.plot(t, valence, color="#44aaff", alpha=0.2, linewidth=0.5)
    ax_main.plot(t, valence_s, color="#44aaff", linewidth=1.5, label="Valence (smoothed)")
    ax_main.plot(t, smile_s, color="#44ff88", linewidth=1.5, linestyle="--", label="Smile")
    ax_main.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    ax_main.set_ylim(-0.6, 1.05)
    ax_main.set_ylabel("Score", color="#cccccc", fontsize=11)

    n_frames = len(rows)
    duration_s = t[-1] - t[0]
    fps_actual = n_frames / duration_s if duration_s > 0 else 0
    title = build_subtitle(csv_path, meta, n_frames, duration_s, fps_actual)
    title += f"  |  Tension avg={tension.mean():.3f}  max={tension.max():.3f}"
    ax_main.set_title(title, color="#eeeeee", fontsize=10, pad=8)

    # Feature panel: blendshapes
    ax_feat.plot(t, eye_wide, color="#ff8800", alpha=0.25, linewidth=0.5)
    ax_feat.plot(t, eye_wide_s, color="#ff8800", linewidth=1.8, label="eyeWide (max L/R)")
    ax_feat.plot(t, brow_up, color="#ffdd00", alpha=0.25, linewidth=0.5)
    ax_feat.plot(t, smooth(brow_up, win), color="#ffdd00", linewidth=1.5, label="browInnerUp")
    ax_feat.plot(t, jaw_open, color="#cc44ff", alpha=0.25, linewidth=0.5)
    ax_feat.plot(t, smooth(jaw_open, win), color="#cc44ff", linewidth=1.2, label="jawOpen")
    ax_feat.plot(t, mouth_press, color="#ff2266", alpha=0.25, linewidth=0.5)
    ax_feat.plot(t, press_s, color="#ff2266", linewidth=1.5, label="mouthPress (max L/R)")
    ax_feat.plot(t, brow_down, color="#aaaaff", alpha=0.2, linewidth=0.5)
    ax_feat.plot(t, smooth(brow_down, win), color="#aaaaff", linewidth=1.0,
                 linestyle="--", label="browDown (max L/R)")
    ax_feat.axhline(0.3, color="#ff8800", linewidth=0.8, linestyle=":",
                    label="eyeWide threshold (0.30)")
    ax_feat.axhline(0.2, color="#ffdd00", linewidth=0.8, linestyle=":",
                    label="browInnerUp threshold (0.20)")
    ax_feat.set_ylim(-0.02, 1.0)
    ax_feat.set_ylabel("Blendshape score", color="#cccccc", fontsize=10)
    ax_feat.set_xlabel("Time (seconds)", color="#cccccc", fontsize=10)

    # Legends
    patch_legend = [mpatches.Patch(color=STATE_COLORS[s], alpha=0.7, label=s) for s in ALL_STATES]
    ax_main.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="#cccccc",
                   fontsize=9, framealpha=0.8)
    ax_feat.legend(handles=ax_feat.get_lines()[:5] + patch_legend,
                   loc="upper right", facecolor="#1a1a2e", labelcolor="#cccccc",
                   fontsize=7.5, framealpha=0.85, ncol=2)

    # Velocity panel (only when startle_score column present in CSV)
    if ax_vel is not None:
        startle_arr = np.array([float(r.get("startle_score", 0)) for r in rows])
        startle_s = smooth(startle_arr, win)

        ax_vel.fill_between(t, 0, startle_arr, color="#ff6600", alpha=0.15, linewidth=0)
        ax_vel.plot(t, startle_arr, color="#ff6600", alpha=0.3, linewidth=0.5)
        ax_vel.plot(t, startle_s, color="#ff6600", linewidth=2.0,
                    label=f"startle_score (max={startle_arr.max():.2f}/s)")
        ax_vel.axhline(STARTLE_VELOCITY_THRESHOLD_DISPLAY, color="#ffdd00",
                       linewidth=1.2, linestyle="--",
                       label=f"Threshold ({STARTLE_VELOCITY_THRESHOLD_DISPLAY:.1f}/s"
                             f" \u2014 CALIBRATION PENDING)")

        startle_times = [t[i] for i, r in enumerate(rows) if r.get("velocity_tag") == "STARTLE"]
        for st in startle_times:
            ax_vel.axvline(st, color="#ff0000", alpha=0.6, linewidth=1.0)
        if startle_times:
            ax_vel.axvline(startle_times[0], color="#ff0000", alpha=0.6, linewidth=1.0,
                           label=f"STARTLE events ({len(startle_times)} detected)", zorder=5)

        ax_vel.set_ylabel("AU velocity (units/s)", color="#cccccc", fontsize=10)
        ax_vel.set_xlabel("Time (seconds)", color="#cccccc", fontsize=10)
        ax_vel.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="#cccccc",
                      fontsize=8, framealpha=0.85)
        ax_feat.set_xlabel("")  # ax_vel now owns the time axis label

    # State bar chart
    draw_state_bar(ax_bar, state_pct)

    plt.tight_layout(pad=1.5)

    vel_info = {}
    if has_velocity:
        startle_arr_ret = np.array([float(r.get("startle_score", 0)) for r in rows])
        vel_info = {
            "n_startle_events": len([r for r in rows if r.get("velocity_tag") == "STARTLE"]),
            "startle_score_max": float(startle_arr_ret.max()),
        }
    return fig, {
        "n_frames": n_frames, "duration_s": duration_s, "fps_actual": fps_actual,
        "state_pct": state_pct,
        "stats": {
            "tension": tension, "eye_wide": eye_wide, "brow_up": brow_up,
            "jaw_open": jaw_open, "mouth_press": mouth_press,
        },
        **vel_info,
    }


# ========== DEEPFACE / PYFEAT PLOT (categorical 7-emotion) ==========
def plot_categorical(rows, t, csv_path, meta, tool_name):
    """Plot for DeepFace or Py-Feat (7 categorical emotions + arousal)."""
    # Column names differ slightly
    if tool_name == "deepface":
        labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    else:  # pyfeat
        labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

    emotions = {}
    for label in labels:
        emotions[label] = np.array([float(r.get(label, 0)) for r in rows])

    arousal = np.array([float(r.get("arousal", 0)) for r in rows])
    dominant = [r.get("dominant", "") for r in rows]

    fps_est = len(t) / (t[-1] - t[0]) if t[-1] > t[0] else 30
    win = max(1, int(fps_est))

    # Dominant emotion distribution
    dom_counts = Counter(dominant)
    total = len(dominant)

    # Build figure
    fig = plt.figure(figsize=(18, 11), facecolor="#1a1a2e")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 2, 1], hspace=0.35)
    ax_main = fig.add_subplot(gs[0])
    ax_emo = fig.add_subplot(gs[1], sharex=ax_main)
    ax_bar = fig.add_subplot(gs[2])

    for ax in [ax_main, ax_emo]:
        style_ax(ax)

    # Draw dominant emotion background bands
    emo_band_colors = {
        "angry": "#ff4444", "anger": "#ff4444",
        "happy": "#00cc44", "happiness": "#00cc44",
        "fear": "#ff8800", "surprise": "#cc44ff",
        "sad": "#4488ff", "sadness": "#4488ff",
        "disgust": "#88cc00", "neutral": "#666666",
    }
    if dominant:
        prev_dom = dominant[0]
        seg_start = t[0]
        for i in range(1, len(dominant)):
            if dominant[i] != prev_dom or i == len(dominant) - 1:
                seg_end = t[i]
                col = emo_band_colors.get(prev_dom, "#444444")
                ax_main.axvspan(seg_start, seg_end, color=col, alpha=0.12, linewidth=0)
                ax_emo.axvspan(seg_start, seg_end, color=col, alpha=0.12, linewidth=0)
                prev_dom = dominant[i]
                seg_start = t[i]

    # Main panel: Arousal timeline
    arousal_s = smooth(arousal, win)
    ax_main.plot(t, arousal, color="#ff4444", alpha=0.2, linewidth=0.5)
    ax_main.plot(t, arousal_s, color="#ff4444", linewidth=2.0, label="Arousal (smoothed)")

    # Also plot fear and surprise specifically
    fear_key = "fear"
    surprise_key = "surprise"
    ax_main.plot(t, smooth(emotions[fear_key], win), color="#ff8800",
                 linewidth=1.5, label=f"{fear_key} (smoothed)")
    ax_main.plot(t, smooth(emotions[surprise_key], win), color="#cc44ff",
                 linewidth=1.5, label=f"{surprise_key} (smoothed)")

    ax_main.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.set_ylabel("Score", color="#cccccc", fontsize=11)

    n_frames = len(rows)
    duration_s = t[-1] - t[0]
    fps_actual = n_frames / duration_s if duration_s > 0 else 0
    title = build_subtitle(csv_path, meta, n_frames, duration_s, fps_actual)
    title += f"  |  Arousal avg={arousal.mean():.3f}  max={arousal.max():.3f}"
    ax_main.set_title(title, color="#eeeeee", fontsize=10, pad=8)
    ax_main.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="#cccccc",
                   fontsize=9, framealpha=0.8)

    # Emotion panel: all 7 emotions
    for label in labels:
        col = EMOTION_COLORS.get(label, "#ffffff")
        ax_emo.plot(t, emotions[label], color=col, alpha=0.25, linewidth=0.5)
        ax_emo.plot(t, smooth(emotions[label], win), color=col, linewidth=1.5, label=label)

    ax_emo.set_ylim(-0.02, 1.0)
    ax_emo.set_ylabel("Emotion probability", color="#cccccc", fontsize=10)
    ax_emo.set_xlabel("Time (seconds)", color="#cccccc", fontsize=10)
    ax_emo.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="#cccccc",
                  fontsize=8, framealpha=0.8, ncol=2)

    # Dominant emotion distribution bar chart
    style_ax(ax_bar)
    dom_labels = sorted(dom_counts.keys(), key=lambda x: dom_counts[x], reverse=True)
    dom_pcts = [dom_counts[l] / total * 100 for l in dom_labels]
    dom_cols = [emo_band_colors.get(l, "#888888") for l in dom_labels]

    if dom_pcts:
        bars = ax_bar.barh(dom_labels, dom_pcts, color=dom_cols, height=0.6)
        for bar, pct in zip(bars, dom_pcts):
            ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", va="center", ha="left",
                        color="#cccccc", fontsize=9)
        ax_bar.set_xlim(0, max(dom_pcts) * 1.2)

    tool_label = "DeepFace" if tool_name == "deepface" else "Py-Feat"
    ax_bar.set_xlabel("% of frames as dominant", color="#cccccc", fontsize=9)
    ax_bar.set_title(f"Dominant emotion distribution ({tool_label})", color="#cccccc", fontsize=9)
    ax_bar.tick_params(axis="y", labelcolor="#cccccc", labelsize=9)
    ax_bar.tick_params(axis="x", labelcolor="#999999", labelsize=8)

    plt.tight_layout(pad=1.5)
    return fig, {
        "n_frames": n_frames, "duration_s": duration_s, "fps_actual": fps_actual,
        "emotions": {l: emotions[l] for l in labels}, "arousal": arousal,
        "dom_pcts": dict(zip(dom_labels, dom_pcts)),
    }


# ========== HSEMOTION PLOT (VA + 8 emotions) ==========
def plot_hsemotion(rows, t, csv_path, meta):
    """Plot for HSEmotion (continuous VA + 8 discrete emotions). Handles both HSEmotion and fusion CSVs."""
    labels = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    # Handle both HSEmotion ("valence") and fusion ("hs_valence") column names
    valence = np.array([float(r.get("valence", 0)) or float(r.get("hs_valence", 0)) for r in rows])
    arousal = np.array([float(r.get("arousal", 0)) or float(r.get("hs_arousal", 0)) for r in rows])

    emotions = {}
    for label in labels:
        # Try with and without "hs_" prefix
        emotions[label] = np.array([float(r.get(label, 0)) or float(r.get(f"hs_{label}", 0)) for r in rows])

    dominant = [r.get("dominant", "") or r.get("hs_dominant", "") for r in rows]

    fps_est = len(t) / (t[-1] - t[0]) if t[-1] > t[0] else 30
    win = max(1, int(fps_est))

    dom_counts = Counter(dominant)
    total = len(dominant)

    # Build figure
    fig = plt.figure(figsize=(18, 11), facecolor="#1a1a2e")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 2, 1], hspace=0.35)
    ax_main = fig.add_subplot(gs[0])
    ax_emo = fig.add_subplot(gs[1], sharex=ax_main)
    ax_bar = fig.add_subplot(gs[2])

    for ax in [ax_main, ax_emo]:
        style_ax(ax)

    # Main panel: Valence + Arousal timeline
    valence_s = smooth(valence, win)
    arousal_s = smooth(arousal, win)

    ax_main.plot(t, valence, color="#44aaff", alpha=0.2, linewidth=0.5)
    ax_main.plot(t, valence_s, color="#44aaff", linewidth=2.0, label="Valence (smoothed)")
    ax_main.plot(t, arousal, color="#ff4444", alpha=0.2, linewidth=0.5)
    ax_main.plot(t, arousal_s, color="#ff4444", linewidth=2.0, label="Arousal (smoothed)")

    ax_main.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    ax_main.set_ylim(-1.1, 1.1)
    ax_main.set_ylabel("Score", color="#cccccc", fontsize=11)

    n_frames = len(rows)
    duration_s = t[-1] - t[0]
    fps_actual = n_frames / duration_s if duration_s > 0 else 0
    title = build_subtitle(csv_path, meta, n_frames, duration_s, fps_actual)
    title += f"  |  V avg={valence.mean():+.3f}  A avg={arousal.mean():.3f}"
    ax_main.set_title(title, color="#eeeeee", fontsize=10, pad=8)
    ax_main.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="#cccccc",
                   fontsize=9, framealpha=0.8)

    # Emotion panel: 8 emotions
    for label in labels:
        col = EMOTION_COLORS.get(label, "#ffffff")
        ax_emo.plot(t, emotions[label], color=col, alpha=0.25, linewidth=0.5)
        ax_emo.plot(t, smooth(emotions[label], win), color=col, linewidth=1.5, label=label)

    ax_emo.set_ylim(-0.02, 1.0)
    ax_emo.set_ylabel("Emotion probability", color="#cccccc", fontsize=10)
    ax_emo.set_xlabel("Time (seconds)", color="#cccccc", fontsize=10)
    ax_emo.legend(loc="upper right", facecolor="#1a1a2e", labelcolor="#cccccc",
                  fontsize=8, framealpha=0.8, ncol=2)

    # Dominant emotion distribution bar chart
    style_ax(ax_bar)
    dom_labels = sorted(dom_counts.keys(), key=lambda x: dom_counts[x], reverse=True)
    dom_pcts = [dom_counts[l] / total * 100 for l in dom_labels]
    dom_cols = [EMOTION_COLORS.get(l, "#888888") for l in dom_labels]

    if dom_pcts:
        bars = ax_bar.barh(dom_labels, dom_pcts, color=dom_cols, height=0.6)
        for bar, pct in zip(bars, dom_pcts):
            ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", va="center", ha="left",
                        color="#cccccc", fontsize=9)
        ax_bar.set_xlim(0, max(dom_pcts) * 1.2)

    ax_bar.set_xlabel("% of frames as dominant", color="#cccccc", fontsize=9)
    ax_bar.set_title("Dominant emotion distribution (HSEmotion)", color="#cccccc", fontsize=9)
    ax_bar.tick_params(axis="y", labelcolor="#cccccc", labelsize=9)
    ax_bar.tick_params(axis="x", labelcolor="#999999", labelsize=8)

    plt.tight_layout(pad=1.5)
    return fig, {
        "n_frames": n_frames, "duration_s": duration_s, "fps_actual": fps_actual,
        "valence": valence, "arousal": arousal,
        "emotions": {l: emotions[l] for l in labels},
        "dom_pcts": dict(zip(dom_labels, dom_pcts)),
    }


# ========== MAIN ==========
def main():
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = pick_latest_csv()

    print(f"Loading: {csv_path}")
    rows = load_csv(csv_path)

    if not rows:
        print("ERROR: CSV is empty (no data rows).")
        return

    # Detect tool
    columns = list(rows[0].keys())
    tool = detect_tool(columns)
    print(f"Detected tool: {tool}")

    if tool == "unknown":
        print("ERROR: Could not detect which FER tool produced this CSV.")
        print(f"  Columns: {columns[:15]}...")
        return

    # Load session metadata
    meta = load_session_meta(csv_path)
    if meta:
        print(f"  Session: {meta.get('subject_id', '?')} / {meta.get('content_type', '?')} / {meta.get('lighting', '?')}")

    # Extract timestamps
    t = np.array([float(r["timestamp"]) for r in rows])

    # Route to the right plotter
    if tool == "mediapipe":
        fig, info = plot_mediapipe(rows, t, csv_path, meta)
    elif tool == "hsemotion" or tool == "fusion":
        # Fusion uses HS arousal/valence + MP tension (route to HS plotter for now)
        fig, info = plot_hsemotion(rows, t, csv_path, meta)
    else:  # deepface or pyfeat
        fig, info = plot_categorical(rows, t, csv_path, meta, tool)

    # Save
    out_path = csv_path.replace(".csv", "_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved: {out_path}")

    # Print quick stats
    n = info["n_frames"]
    dur = info["duration_s"]
    fps = info["fps_actual"]
    print(f"\n--- Quick Stats ({tool}) ---")
    print(f"Duration:  {dur:.1f}s  |  {n} frames  |  {fps:.0f} FPS")

    if tool == "mediapipe":
        st = info["stats"]
        ten = st["tension"]
        print(f"Tension:         mean={ten.mean():.4f}  max={ten.max():.4f}  "
              f"min={ten.min():.4f}  std={ten.std():.4f}")
        print(f"eyeWide max:     {st['eye_wide'].max():.4f}  (FEAR threshold: 0.30 — "
              f"{'REACHED' if st['eye_wide'].max() >= 0.3 else 'NEVER REACHED'})")
        print(f"browInnerUp max: {st['brow_up'].max():.4f}  (FEAR threshold: 0.20 — "
              f"{'REACHED' if st['brow_up'].max() >= 0.2 else 'NEVER REACHED'})")
        print(f"jawOpen max:     {st['jaw_open'].max():.4f}")
        print(f"mouthPress max:  {st['mouth_press'].max():.4f}")
        print("\nState distribution:")
        for s in ALL_STATES:
            pct = info["state_pct"].get(s, 0)
            bar = "#" * int(pct / 2)
            print(f"  {s:6s} {pct:5.1f}%  {bar}")

    elif tool == "hsemotion":
        v = info["valence"]
        a = info["arousal"]
        print(f"Valence:   mean={v.mean():+.4f}  min={v.min():+.4f}  max={v.max():+.4f}  std={v.std():.4f}")
        print(f"Arousal:   mean={a.mean():.4f}  min={a.min():.4f}  max={a.max():.4f}  std={a.std():.4f}")
        print("\nDominant emotion distribution:")
        for emo, pct in sorted(info["dom_pcts"].items(), key=lambda x: -x[1]):
            bar = "#" * int(pct / 2)
            print(f"  {emo:12s} {pct:5.1f}%  {bar}")

    else:  # deepface / pyfeat
        a = info["arousal"]
        print(f"Arousal:   mean={a.mean():.4f}  min={a.min():.4f}  max={a.max():.4f}  std={a.std():.4f}")
        print("\nDominant emotion distribution:")
        for emo, pct in sorted(info["dom_pcts"].items(), key=lambda x: -x[1]):
            bar = "#" * int(pct / 2)
            print(f"  {emo:12s} {pct:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
