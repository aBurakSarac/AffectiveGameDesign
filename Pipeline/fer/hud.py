"""Live HUD display and post-session comparison analysis.

Pattern: [Facade] — HUD composes all signal layers (MP, HS, Fusion) into one surface.
"""

import cv2
import numpy as np

from fer.blendshapes import (
    EMOTION_LABELS, PANEL_WIDTH, HUD_MIN_HEIGHT,
    PANEL_BG, SECTION_DIVIDER_COLOR,
    NEGATIVE_BLENDSHAPES, POSITIVE_BLENDSHAPES, STRESS_BLENDSHAPES,
)
# gated_composite_fear removed — veto system commented out


def draw_hud(canvas, panel_x, frame_h, mp_data, hs_data, fusion_data, mode):
    """Draw the 3-section side panel HUD onto canvas."""
    pw = PANEL_WIDTH
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Fill panel background
    canvas[0:frame_h, panel_x:panel_x + pw] = PANEL_BG

    y = 10
    px = panel_x + 10
    bar_x = panel_x + 170
    bar_max = 180

    # ── MP TRIGGERS ──────────────────────────────────────────────────────
    mp_crop_src = mp_data.get("crop_source", "none")
    # [M] = full-frame MediaPipe detection; [H] = Haar provided the crop for retry
    mp_crop_label = {"full_frame": "[M]", "haar_retry": "[H]", "none": "[-]"}.get(mp_crop_src, "[-]")
    cv2.putText(canvas, f"MP TRIGGERS {mp_crop_label}", (px, y + 14), font, 0.55, (100, 200, 255), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    if mp_data["face_detected"]:
        # Tension bar (AGC-normalised)
        tension = mp_data["tension"]
        t_bar = int(min(tension, 1.0) * bar_max)
        cv2.putText(canvas, f"Tension: {tension:.2f}", (px, y + 12), font, 0.45, (0, 0, 255), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + t_bar, y + 14), (0, 0, 255), -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 20
        # Raw (pre-AGC) tension for comparison
        tension_v3 = mp_data.get("tension_v3", 0.0)
        v3_bar = int(min(tension_v3, 1.0) * bar_max)
        cv2.putText(canvas, f"Raw:     {tension_v3:.2f}", (px, y + 12), font, 0.45, (80, 80, 180), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + v3_bar, y + 14), (80, 80, 180), -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 20

        # Valence
        val = mp_data["face_valence"]
        v_color = (0, 200, 0) if val >= 0 else (0, 0, 255)
        v_bar = int(abs(val) * bar_max)
        cv2.putText(canvas, f"Valence: {val:+.2f}", (px, y + 12), font, 0.45, v_color, 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + v_bar, y + 14), v_color, -1)
        y += 20

        # Smile
        smile = mp_data["smile_level"]
        s_bar = int(min(smile, 1.0) * bar_max)
        cv2.putText(canvas, f"Smile:   {smile:.2f}", (px, y + 12), font, 0.45, (255, 255, 0), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + s_bar, y + 14), (255, 255, 0), -1)
        y += 20

        # Context tag
        ctx = mp_data["ctx_tag"]
        ctx_colors = {
            "JOY": (0, 200, 0), "FEAR": (0, 0, 255),
            "CONC": (200, 200, 0), "SAD": (200, 100, 0),
            "STRESS": (0, 50, 255), "---": (150, 150, 150),
        }
        ctx_col = ctx_colors.get(ctx, (150, 150, 150))
        cv2.putText(canvas, f"State: [{ctx}]", (px, y + 12), font, 0.50, ctx_col, 2)
        y += 20

        # Startle
        vel_tag = mp_data["velocity_tag"]
        ss = mp_data["startle_score"]
        vel_col = (0, 100, 255) if vel_tag == "STARTLE" else (100, 100, 100)
        cv2.putText(canvas, f"Startle: [{vel_tag}] {ss:.1f}/s",
                    (px, y + 12), font, 0.40, vel_col, 1)
        y += 18

        # Top activated blendshapes
        cv2.putText(canvas, "Key AUs:", (px, y + 11), font, 0.35, (180, 180, 180), 1)
        y += 14
        bs_dict = mp_data["bs_dict"]
        sorted_bs = sorted(
            [(name, bs_dict.get(name, 0)) for name in STRESS_BLENDSHAPES],
            key=lambda x: x[1], reverse=True,
        )
        neg_set = set(NEGATIVE_BLENDSHAPES)
        pos_set = set(POSITIVE_BLENDSHAPES)
        shown = 0
        for name, bval in sorted_bs:
            if bval < 0.05 or shown >= 6:
                break
            b_len = int(min(bval, 1.0) * 100)
            if name in neg_set:
                col = (0, 0, 200) if bval > 0.15 else (80, 80, 120)
            elif name in pos_set:
                col = (0, 200, 0) if bval > 0.15 else (80, 120, 80)
            else:
                col = (0, 200, 200) if bval > 0.15 else (80, 120, 120)
            cv2.putText(canvas, f"{name[:12]:12s} {bval:.2f}", (px, y + 10),
                        font, 0.30, col, 1)
            cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + b_len, y + 10), col, -1)
            y += 13
            shown += 1
    else:
        cv2.putText(canvas, "No face detected", (px, y + 12), font, 0.45, (100, 100, 100), 1)
        y += 20

    y += 5

    # ── HS TRIGGERS ──────────────────────────────────────────────────────
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8
    crop_src = hs_data.get("crop_source", "none")
    crop_label = {"haar": "[H]", "mp": "[M]", "none": "[-]"}.get(crop_src, "[-]")
    cv2.putText(canvas, f"HS TRIGGERS {crop_label}", (px, y + 14), font, 0.55, (0, 200, 255), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    if hs_data["face_detected"]:
        arousal = hs_data["arousal"]
        a_bar = int(min(max(arousal, 0), 1.0) * bar_max)
        cv2.putText(canvas, f"Arousal: {arousal:.2f}", (px, y + 12), font, 0.45, (0, 0, 255), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + a_bar, y + 14), (0, 0, 255), -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 20

        hs_val = hs_data["valence"]
        hv_col = (0, 200, 0) if hs_val >= 0 else (0, 100, 255)
        cv2.putText(canvas, f"Valence: {hs_val:+.2f}", (px, y + 12), font, 0.45, hv_col, 1)
        y += 20

        dom = hs_data["dominant"]
        dom_score = hs_data["dominant_score"]
        cv2.putText(canvas, f"Dom: {dom} ({dom_score:.2f})", (px, y + 12),
                    font, 0.45, (255, 255, 255), 1)
        y += 20

        emotions = hs_data["emotions"]
        for label in EMOTION_LABELS:
            score = emotions.get(label, 0)
            e_bar = int(max(score, 0) * 140)
            color = (0, 255, 0) if label == dom else (140, 140, 140)
            cv2.putText(canvas, f"{label[:5]:5s} {score:.2f}", (px, y + 10),
                        font, 0.32, color, 1)
            cv2.rectangle(canvas, (px + 80, y + 2), (px + 80 + e_bar, y + 10), color, -1)
            y += 14
    else:
        cv2.putText(canvas, "No face detected", (px, y + 12), font, 0.45, (100, 100, 100), 1)
        y += 20

    y += 5

    # ── FUSION ───────────────────────────────────────────────────────────
    if mode != "independent" and fusion_data is not None:
        cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
        y += 8
        cv2.putText(canvas, "FUSION", (px, y + 14), font, 0.55, (255, 200, 0), 2)
        y += 22
        cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
        y += 8

        comp = fusion_data["composite_fear"]
        c_bar = int(min(max(comp, 0), 1.0) * bar_max)
        if comp > 0.5:
            c_col = (0, 0, 255)
        elif comp > 0.3:
            c_col = (0, 100, 255)
        else:
            c_col = (100, 100, 100)
        cv2.putText(canvas, f"F*(1+T): {comp:.2f}", (px, y + 12), font, 0.45, c_col, 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + c_bar, y + 14), c_col, -1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 22

        # Veto/agreement system commented out — using hs_fear*(1+mp_tension)
        # agree = fusion_data.get("agreement_tag", "---")
        # ...

        y += 5
        mp_face = "Y" if mp_data["face_detected"] else "N"
        hs_face = "Y" if hs_data["face_detected"] else "N"
        cv2.putText(canvas, f"Face: MP={mp_face} HS={hs_face}",
                    (px, y + 12), font, 0.38, (150, 150, 150), 1)
        y += 18

        # ── FORMULAS PANEL (F0–F11, 2-column grid) ───────────────────────
        formulas = fusion_data.get("formulas")
        selected = fusion_data.get("selected_formula", "").upper()
        if formulas:
            y += 4
            cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y),
                     SECTION_DIVIDER_COLOR, 1)
            y += 6
            cv2.putText(canvas, "FORMULAS", (px, y + 11), font, 0.45, (200, 180, 80), 2)
            y += 16

            col_w = (pw - 20) // 2   # two equal columns
            bar_max_f = col_w - 60   # bar width budget per column
            formula_keys = ["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11"]
            row_h = 13

            for i, key in enumerate(formula_keys):
                val = formulas.get(key, 0.0)
                col_idx = i % 2          # 0 = left, 1 = right
                row_idx = i // 2
                cx = px + col_idx * col_w
                cy = y + row_idx * row_h

                is_sel = (key == selected)
                txt_col = (0, 220, 255) if is_sel else (160, 160, 160)
                b_col   = (0, 180, 220) if is_sel else (80, 100, 120)

                b_len = int(min(val, 1.0) * bar_max_f)
                cv2.putText(canvas, f"{key}:{val:.2f}", (cx, cy + 9),
                            font, 0.30, txt_col, 1)
                cv2.rectangle(canvas, (cx + 52, cy + 2),
                              (cx + 52 + b_len, cy + 9), b_col, -1)

            y += (len(formula_keys) // 2) * row_h + 4

    return canvas


_VARIANT_COLORS = {
    "v3":  (100, 100, 200),
    "v4":  (50,  220,  50),
    "v5a": (50,  200, 200),
    "v5b": (200, 160,  50),
    "v5c": (200,  50, 200),
    "v5d": (160, 220, 100),
    "v5e": (50,  120, 255),
}


def draw_mp_variants_hud(canvas, panel_x, frame_h, mp_data, detector_variant="v4",
                          event_status="IDLE", smoothed_tension=0.0, onset_slope=0.0):
    """Draw the MP-only side panel showing all tension variants and MP state."""
    pw = PANEL_WIDTH
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas[0:frame_h, panel_x:panel_x + pw] = PANEL_BG

    y = 10
    px = panel_x + 10
    bar_x = panel_x + 140
    bar_max = pw - 140 - 20  # 240px

    # ── TENSION VARIANTS ─────────────────────────────────────────────────
    cv2.putText(canvas, "MP TENSION VARIANTS", (px, y + 14), font, 0.55, (100, 200, 255), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    variants = mp_data.get("tension_variants", {})
    for vname, color in _VARIANT_COLORS.items():
        val = variants.get(vname, 0.0)
        b_len = int(min(val, 1.0) * bar_max)
        is_sel = (vname == detector_variant)

        label = f"[{vname}]" if is_sel else f" {vname} "
        txt_col = (255, 255, 255) if is_sel else color
        weight = 2 if is_sel else 1

        cv2.putText(canvas, f"{label}: {val:.3f}", (px, y + 12), font, 0.45, txt_col, weight)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + b_len, y + 14), color, -1 if is_sel else 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
        y += 20

    y += 4
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    # ── DETECTOR STATUS ──────────────────────────────────────────────────
    cv2.putText(canvas, "DETECTOR", (px, y + 14), font, 0.55, (255, 200, 0), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    sm_bar = int(min(smoothed_tension, 1.0) * bar_max)
    cv2.putText(canvas, f"Smoothed: {smoothed_tension:.3f}", (px, y + 12), font, 0.45, (0, 180, 255), 1)
    cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + sm_bar, y + 14), (0, 180, 255), -1)
    cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + bar_max, y + 14), (60, 60, 80), 1)
    y += 20

    cv2.putText(canvas, f"Slope:    {onset_slope:.5f}", (px, y + 12), font, 0.40, (150, 150, 150), 1)
    y += 18

    badge_colors = {
        "ONSET": (0, 200, 255), "SUSTAINING": (0, 130, 255),
        "EVENT_CONFIRMED": (0, 0, 220), "EVENT_ENDED": (30, 180, 30),
    }
    bc = badge_colors.get(event_status, (80, 80, 80))
    cv2.rectangle(canvas, (px, y), (panel_x + pw - 10, y + 22), bc, -1)
    cv2.putText(canvas, event_status, (px + 5, y + 15), font, 0.50, (255, 255, 255), 2)
    y += 30

    y += 4
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    # ── MP STATE ─────────────────────────────────────────────────────────
    cv2.putText(canvas, "MP STATE", (px, y + 14), font, 0.55, (100, 200, 255), 2)
    y += 22
    cv2.line(canvas, (panel_x + 5, y), (panel_x + pw - 5, y), SECTION_DIVIDER_COLOR, 1)
    y += 8

    if mp_data.get("face_detected", False):
        val = mp_data.get("face_valence", 0.0)
        v_color = (0, 200, 0) if val >= 0 else (0, 0, 255)
        cv2.putText(canvas, f"Valence: {val:+.2f}", (px, y + 12), font, 0.45, v_color, 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + int(abs(val) * bar_max), y + 14), v_color, -1)
        y += 20

        smile = mp_data.get("smile_level", 0.0)
        cv2.putText(canvas, f"Smile:   {smile:.2f}", (px, y + 12), font, 0.45, (255, 255, 0), 1)
        cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + int(min(smile, 1.0) * bar_max), y + 14),
                      (255, 255, 0), -1)
        y += 20

        ctx = mp_data.get("ctx_tag", "---")
        ctx_colors = {
            "JOY": (0, 200, 0), "FEAR": (0, 0, 255), "CONC": (200, 200, 0),
            "SAD": (200, 100, 0), "STRESS": (0, 50, 255), "---": (150, 150, 150),
        }
        cv2.putText(canvas, f"State: [{ctx}]", (px, y + 12), font, 0.50,
                    ctx_colors.get(ctx, (150, 150, 150)), 2)
        y += 20

        vel_tag = mp_data.get("velocity_tag", "---")
        ss = mp_data.get("startle_score", 0.0)
        vel_col = (0, 100, 255) if vel_tag == "STARTLE" else (100, 100, 100)
        cv2.putText(canvas, f"Startle: [{vel_tag}] {ss:.1f}/s", (px, y + 12), font, 0.40, vel_col, 1)
        y += 18

        y += 4
        cv2.putText(canvas, "Key AUs:", (px, y + 11), font, 0.35, (180, 180, 180), 1)
        y += 14
        bs_dict = mp_data.get("bs_dict", {})
        sorted_bs = sorted(
            [(name, bs_dict.get(name, 0)) for name in STRESS_BLENDSHAPES],
            key=lambda x: x[1], reverse=True,
        )
        neg_set = set(NEGATIVE_BLENDSHAPES)
        pos_set = set(POSITIVE_BLENDSHAPES)
        shown = 0
        for name, bval in sorted_bs:
            if bval < 0.05 or shown >= 8:
                break
            b_len = int(min(bval, 1.0) * 100)
            if name in neg_set:
                col = (0, 0, 200) if bval > 0.15 else (80, 80, 120)
            elif name in pos_set:
                col = (0, 200, 0) if bval > 0.15 else (80, 120, 80)
            else:
                col = (0, 200, 200) if bval > 0.15 else (80, 120, 120)
            cv2.putText(canvas, f"{name[:14]:14s} {bval:.2f}", (px, y + 10), font, 0.30, col, 1)
            cv2.rectangle(canvas, (bar_x, y + 2), (bar_x + b_len, y + 10), col, -1)
            y += 13
            shown += 1
    else:
        cv2.putText(canvas, "No face detected", (px, y + 12), font, 0.45, (100, 100, 100), 1)

    return canvas


def run_mp_hs_comparison(csv_path):
    """Post-session comparison analysis: stats + 4-panel plot."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  Skipping comparison analysis (missing dependency: {e})")
        return

    print("\n" + "=" * 60)
    print("MP+HS COMPARISON ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    total_frames = len(df)

    both = df[(df["mp_face_detected"] == 1) & (df["hs_face_detected"] == 1)]
    both_count = len(both)

    print(f"Total frames: {total_frames}")
    print(f"Both tools detected face: {both_count} ({100 * both_count / max(total_frames, 1):.1f}%)")
    print(f"MP-only face: {((df['mp_face_detected'] == 1) & (df['hs_face_detected'] == 0)).sum()}")
    print(f"HS-only face: {((df['mp_face_detected'] == 0) & (df['hs_face_detected'] == 1)).sum()}")
    print(f"Neither: {((df['mp_face_detected'] == 0) & (df['hs_face_detected'] == 0)).sum()}")

    if both_count == 0:
        print("No frames with both faces detected — skipping analysis.")
        return

    r_val = both[["mp_tension", "hs_arousal"]].corr().iloc[0, 1]
    print(f"\nCorrelation (tension vs arousal): r = {r_val:.3f}")

    if "hs_fear" in both.columns:
        r_fear = both[["hs_fear", "mp_tension"]].corr().iloc[0, 1]
        print(f"Correlation (hs_fear vs tension): r = {r_fear:.3f}")

    if "composite_fear" in both.columns:
        top_peaks = both.nlargest(5, "composite_fear")
        print(f"\nTop 5 composite fear moments — hs_fear × (1 + mp_tension):")
        for _, row in top_peaks.iterrows():
            print(f"  t={float(row['timestamp']):.1f}s | "
                  f"Comp={float(row['composite_fear']):.2f} | "
                  f"Fear={float(row.get('hs_fear', 0)):.2f} | "
                  f"MP_T={float(row['mp_tension']):.2f} | "
                  f"HS_A={float(row['hs_arousal']):.2f} [{row['hs_dominant']}]")

    # ── 4-Panel Comparison Plot ──────────────────────────────────────────
    DARK_BG = "#1a1a2e"
    PANEL_BG_PLOT = "#16213e"
    GRID_COL = "#2a2a4a"

    smooth = lambda s, w=15: pd.Series(s).rolling(w, center=True, min_periods=1).mean().values

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor=DARK_BG,
                             gridspec_kw={"hspace": 0.35})

    t = df["timestamp"].astype(float).values

    # Panel 1: MP tension
    ax = axes[0]
    ax.set_facecolor(PANEL_BG_PLOT)
    ax.set_title("MediaPipe Tension", color="white", fontsize=11, pad=8)
    tension_vals = df["mp_tension"].fillna(0).astype(float).values
    ax.plot(t, smooth(tension_vals), color="#ff4444", linewidth=1.2, label="Tension")
    ax.fill_between(t, 0, smooth(tension_vals), alpha=0.2, color="#ff4444")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Tension", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=GRID_COL, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ctx_colors_plot = {"JOY": "#00cc00", "FEAR": "#ff0000", "CONC": "#cccc00",
                       "SAD": "#cc6600", "STRESS": "#ff3300", "---": "#333333"}
    if "mp_ctx_tag" in df.columns:
        for i in range(len(t) - 1):
            tag = str(df["mp_ctx_tag"].iloc[i]) if pd.notna(df["mp_ctx_tag"].iloc[i]) else "---"
            color = ctx_colors_plot.get(tag, "#333333")
            ax.axvspan(t[i], t[i + 1], ymin=0, ymax=0.05, color=color, alpha=0.8)

    # Panel 2: HS arousal
    ax = axes[1]
    ax.set_facecolor(PANEL_BG_PLOT)
    ax.set_title("HSEmotion Arousal", color="white", fontsize=11, pad=8)
    arousal_vals = df["hs_arousal"].fillna(0).astype(float).values
    ax.plot(t, smooth(arousal_vals), color="#ff8844", linewidth=1.2, label="Arousal")
    ax.fill_between(t, 0, smooth(arousal_vals), alpha=0.2, color="#ff8844")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Arousal", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=GRID_COL, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 3: Composite fear
    ax = axes[2]
    ax.set_facecolor(PANEL_BG_PLOT)
    ax.set_title("Composite Fear — hs_fear × (1 + mp_tension)", color="white",
                 fontsize=11, pad=8)
    comp_vals = df["composite_fear"].fillna(0).astype(float).values
    ax.plot(t, smooth(comp_vals), color="#cc44ff", linewidth=1.2, label="Composite")
    ax.fill_between(t, 0, smooth(comp_vals), alpha=0.2, color="#cc44ff")
    ax.axhline(y=0.5, color="#ff6666", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Composite", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=GRID_COL, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Panel 4: HS Fear trace
    ax = axes[3]
    ax.set_facecolor(PANEL_BG_PLOT)
    ax.set_title("HS Fear Score", color="white", fontsize=11, pad=8)
    if "hs_fear" in df.columns:
        fear_vals = df["hs_fear"].fillna(0).astype(float).values
        ax.plot(t, smooth(fear_vals), color="#ff4488", linewidth=1.2, label="HS Fear")
        ax.fill_between(t, 0, smooth(fear_vals), alpha=0.2, color="#ff4488")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fear", color="white", fontsize=9)
    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color=GRID_COL, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    out_png = csv_path.replace(".csv", "_comparison.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"\nComparison plot saved: {out_png}")
    print("=" * 60)


def _draw_signal_bar(img, x, y, value, width, height, color, label, font, scale=0.35):
    """Draw a labeled horizontal bar on img at (x, y)."""
    cv2.putText(img, label, (x, y + height - 1), font, scale, (200, 200, 200), 1)
    bar_end = x + 90 + int(min(max(value, 0.0), 1.0) * width)
    cv2.rectangle(img, (x + 90, y), (bar_end, y + height), color, -1)
    cv2.putText(img, f"{value:.2f}", (x + 90 + width + 4, y + height - 1),
                font, scale, (200, 200, 200), 1)


def draw_on_video_bars(frame, hs_fear, mp_tension, smoothed_comp):
    """Draw semi-transparent signal bars on the bottom-left of the video frame."""
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    panel_w, panel_h = 260, 75
    panel_x, panel_y = 10, fh - panel_h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    bx, by = panel_x + 5, panel_y + 10
    _draw_signal_bar(frame, bx, by,      hs_fear,       100, 12, (50, 50, 200),  "Fear",    font)
    _draw_signal_bar(frame, bx, by + 18, mp_tension,    100, 12, (50, 200, 50),  "Tension", font)
    _draw_signal_bar(frame, bx, by + 36, smoothed_comp, 100, 18, (0, 180, 255), "F*(1+T)", font, 0.4)
