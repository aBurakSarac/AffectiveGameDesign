"""MP feature sweep — solid comparison tool.

Reads a FULL model CSV (needs mp_<blendshape> columns + hs_* columns).
Re-derives mp_tension under different formula/AGC configurations, then evaluates
each configuration against ground truth using the same detection logic as
compare_ground_truth_v2.py (GT-centric TP, cluster-aware).

Fixed (CLI-configurable, not swept):  threshold, min_frames, window_size, fill_ratio
Swept:  tension formula (v3/v4), global_scale, joy suppression threshold, AGC gain

Usage (direct):
  python Pipeline/fer/compare_ground_truth_mp.py              # uses constants below
  python Pipeline/fer/compare_ground_truth_mp.py \\
      --gt-csv Annotations/S06_Vid16.csv \\
      --model-csv Pipeline/logs/sessions/.../model.csv \\
      --threshold 0.5 --min-frames 15 --window-size 30 --fill-ratio 0.25 \\
      --sweep-scale 1.4 1.6 1.8 2.0 2.2 \\
      --sweep-gain 0.0 0.40 0.50 0.60 0.70 \\
      --sweep-joy 0.0 0.20

Usage (as module from evaluate_all.py):
  from fer.compare_ground_truth_mp import (
      generate_mp_report, compute_metrics_mp,
      print_mp_parameter_sweep, make_mp_sweep_configs,
      MP_FORMULA_COLS, MP_FORMULA_LABELS,
  )

Notes:
  - mouthUpperUpLeft/Right and cheekPuff are in the v3 stress group but NOT in
    KEY_BLENDSHAPES, so recomputed v3 tension is approximate (~11% stress weight missing).
    Use stored_v3 mode for exact v3 values.
  - sweep-gain 0.0 means no-AGC (raw tension used directly as score).
  - sweep-joy 0.0 means joy suppression disabled.
  - f7 (hs_only) and f11 (hs-anger) do not use mp_tension — their F1 is constant
    across all MP configs and serves as the HS-only baseline.
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# ── Import shared helpers from compare_ground_truth_v2 ───────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fer.compare_ground_truth_v2 import (
    SCOREABLE_LABELS,
    time_to_sec,
    compute_metrics,
)


# ── Blendshape columns stored in full CSV (KEY_BLENDSHAPES subset) ───────────

_STORED_BS = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
    "mouthPressLeft", "mouthPressRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthSmileLeft", "mouthSmileRight", "jawOpen",
    "noseSneerLeft", "noseSneerRight",
    "cheekSquintLeft", "cheekSquintRight",
]


def _get_bs(row):
    """Extract blendshape dict from CSV row. Missing columns → 0."""
    return {n: float(row.get(f"mp_{n}", 0.0) or 0.0) for n in _STORED_BS}


# ── HS column patching for mp_only CSVs ─────────────────────────────────────

_HS_COLS = ('hs_fear', 'hs_arousal', 'hs_surprise', 'hs_anger')


def _patch_hs_columns(df):
    """Add zero-filled hs_* columns to df in-place if absent (mp_only CSV).

    Needed because _recompute_formulas calls df.get('hs_fear', 0) which returns
    a scalar when the column is absent, breaking subsequent .fillna() calls.
    """
    for col in _HS_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


# ── Tension recomputation from blendshapes ────────────────────────────────────

def _recompute_v3(bs):
    """Approximate compute_tension (v3) from stored blendshapes.

    Missing: mouthUpperUpLeft/Right (w=0.5 each) and cheekPuff (w=0.3) —
    their weights are kept in stress_wsum so the denominator stays correct.
    """
    get = bs.get
    smile   = max(get("mouthSmileLeft", 0), get("mouthSmileRight", 0))
    cheek_s = max(get("cheekSquintLeft", 0), get("cheekSquintRight", 0))
    pos_sig = max(smile, cheek_s * 0.8)

    frown  = max(get("mouthFrownLeft", 0), get("mouthFrownRight", 0))
    press  = max(get("mouthPressLeft", 0), get("mouthPressRight", 0))
    brow_d = max(get("browDownLeft", 0), get("browDownRight", 0))
    is_sad  = frown > press and frown > 0.1
    is_conc = brow_d > 0.2 and press < 0.05

    # stress group — missing mouthUpperUp×0.5 and cheekPuff×0.3 (kept in wsum)
    stress = {"mouthPressLeft": 1.5, "mouthPressRight": 1.5,
              "noseSneerLeft": 0.8, "noseSneerRight": 0.8}
    st = sum(get(n, 0) * w for n, w in stress.items())
    sw = sum(stress.values()) + 0.5 + 0.5 + 0.3   # include missing weights in denominator

    # context group
    ctx_m = 0.2 if is_conc else (0.5 if is_sad else 1.0)
    ctx   = {"browDownLeft": 0.7, "browDownRight": 0.7,
             "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
             "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4}
    ct = sum(get(n, 0) * w * ctx_m for n, w in ctx.items())
    cw = sum(ctx.values())

    # fear group
    sd = max(0.0, 1.0 - pos_sig * 2.0)
    is_laughing = get("jawOpen", 0) > 0.3 and get("browInnerUp", 0) > 0.5
    biu_w = 0.3 if is_laughing else 1.3
    fear  = {"browInnerUp": biu_w, "eyeWideLeft": 1.5, "eyeWideRight": 1.5, "jawOpen": 0.6}
    ft = sum(get(n, 0) * w * sd for n, w in fear.items())
    fw = 1.3 + 1.5 + 1.5 + 0.6   # use max biu_w for denominator consistency

    raw = (st + ct + ft) / (sw + cw + fw) - pos_sig * 0.15
    return max(0.0, min(1.0, raw * 2.5))


def _recompute_v4(bs, global_scale=1.8, joy_thresh=0.20):
    """Approximate compute_tension_v4 from stored blendshapes."""
    get = bs.get
    smile   = max(get("mouthSmileLeft", 0), get("mouthSmileRight", 0))
    cheek_s = max(get("cheekSquintLeft", 0), get("cheekSquintRight", 0))
    pos_sig = max(smile, cheek_s * 0.8)

    frown  = max(get("mouthFrownLeft", 0), get("mouthFrownRight", 0))
    press  = max(get("mouthPressLeft", 0), get("mouthPressRight", 0))
    brow_d = max(get("browDownLeft", 0), get("browDownRight", 0))
    is_sad    = frown > press and frown > 0.1
    is_conc   = brow_d > 0.2 and press < 0.05
    is_joyful = smile > joy_thresh if joy_thresh is not None else False

    # stress group (approximate — missing mouthUpperUp and cheekPuff)
    stress = {"mouthPressLeft": 1.5, "mouthPressRight": 1.5,
              "noseSneerLeft": 0.8, "noseSneerRight": 0.8}
    st = sum(get(n, 0) * w for n, w in stress.items())
    sw = sum(stress.values()) + 0.5 + 0.5 + 0.3
    stress_score = st / sw

    # context group
    ctx_m = (0.2 if is_conc else
             0.25 if is_joyful else
             0.5 if is_sad else 1.0)
    ctx = {"browDownLeft": 0.7, "browDownRight": 0.7,
           "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
           "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4}
    ct = sum(get(n, 0) * w * ctx_m for n, w in ctx.items())
    cw = sum(ctx.values())
    ctx_score = ct / cw

    # fear group
    sd = max(0.0, 1.0 - pos_sig * 2.0)
    is_laughing = get("jawOpen", 0) > 0.3 and get("browInnerUp", 0) > 0.5
    biu_w = 0.3 if is_laughing else 1.3
    fear = {"browInnerUp": biu_w, "eyeWideLeft": 1.5, "eyeWideRight": 1.5, "jawOpen": 0.6}
    ft = sum(get(n, 0) * w * sd for n, w in fear.items())
    fw = biu_w + 1.5 + 1.5 + 0.6
    fear_score = ft / fw

    scores = sorted([stress_score, ctx_score, fear_score])
    blend  = 0.50 * scores[2] + 0.30 * scores[1] + 0.20 * scores[0]
    raw    = blend * global_scale - pos_sig * 0.15
    return max(0.0, min(1.0, raw))


def _recompute_v4_batch(model_df, global_scale=1.8, joy_thresh=0.20):
    """Vectorized v4 computation. Same result as per-row _recompute_v4 but ~100x faster."""
    def gc(name):
        col = f"mp_{name}"
        return (pd.to_numeric(model_df[col], errors='coerce').fillna(0.0).values
                if col in model_df.columns else np.zeros(len(model_df)))

    face_det = (model_df.get("mp_face_detected", pd.Series(0, index=model_df.index))
                .fillna(0).astype(int).values.astype(bool))

    smile_l, smile_r = gc("mouthSmileLeft"), gc("mouthSmileRight")
    cheek_sl, cheek_sr = gc("cheekSquintLeft"), gc("cheekSquintRight")
    frown_l, frown_r = gc("mouthFrownLeft"), gc("mouthFrownRight")
    press_l, press_r = gc("mouthPressLeft"), gc("mouthPressRight")
    brow_dl, brow_dr = gc("browDownLeft"), gc("browDownRight")
    biu = gc("browInnerUp")
    eye_wl, eye_wr = gc("eyeWideLeft"), gc("eyeWideRight")
    eye_sql, eye_sqr = gc("eyeSquintLeft"), gc("eyeSquintRight")
    jaw_open = gc("jawOpen")
    nose_snl, nose_snr = gc("noseSneerLeft"), gc("noseSneerRight")

    smile   = np.maximum(smile_l, smile_r)
    cheek_s = np.maximum(cheek_sl, cheek_sr)
    pos_sig = np.maximum(smile, cheek_s * 0.8)
    frown   = np.maximum(frown_l, frown_r)
    press   = np.maximum(press_l, press_r)
    brow_d  = np.maximum(brow_dl, brow_dr)

    is_conc   = (brow_d > 0.2) & (press < 0.05)
    is_sad    = (frown > press) & (frown > 0.1)
    is_joyful = (smile > joy_thresh) if joy_thresh is not None else np.zeros(len(model_df), dtype=bool)

    # stress score — unaffected by smile (mouthPress + noseSneer)
    sw = (1.5 + 1.5 + 0.8 + 0.8) + 0.5 + 0.5 + 0.3   # includes missing blendshape weights
    stress_score = (press_l * 1.5 + press_r * 1.5 + nose_snl * 0.8 + nose_snr * 0.8) / sw

    # context score — suppressed when joyful
    cw = 0.7 + 0.7 + 0.6 + 0.6 + 0.4 + 0.4
    ct_raw = (brow_dl * 0.7 + brow_dr * 0.7 +
              eye_sql * 0.6 + eye_sqr * 0.6 +
              frown_l * 0.4 + frown_r * 0.4)
    ctx_m = np.where(is_conc, 0.2, np.where(is_joyful, 0.25, np.where(is_sad, 0.5, 1.0)))
    ctx_score = ct_raw * ctx_m / cw

    # fear score — suppressed when smiling (sd → 0 when pos_sig > 0.5)
    sd = np.maximum(0.0, 1.0 - pos_sig * 2.0)
    is_laughing = (jaw_open > 0.3) & (biu > 0.5)
    biu_w = np.where(is_laughing, 0.3, 1.3)
    fw = biu_w + 1.5 + 1.5 + 0.6
    fear_score = (biu * biu_w + eye_wl * 1.5 + eye_wr * 1.5 + jaw_open * 0.6) * sd / fw

    stacked = np.stack([stress_score, ctx_score, fear_score], axis=1)
    s = np.sort(stacked, axis=1)
    blend  = 0.50 * s[:, 2] + 0.30 * s[:, 1] + 0.20 * s[:, 0]
    result = np.clip(blend * global_scale - pos_sig * 0.15, 0.0, 1.0)
    result[~face_det] = 0.0
    return result


def _recompute_v5_batch(model_df, global_scale=1.8, joy_thresh=0.20,
                        fix_laughing=True, harder_smile_gate=False,
                        remove_eye_squint=False, joy_jaw_suppress=False,
                        smile_penalty=0.15):
    """Parametric v5 tension formula addressing two complementary v4 failures.

    v4 failures:
      - Spikes on positive: smile gate requires pos_sig > 0.5 to zero fear group;
        typical happy expression (smile≈0.3) still lets fear fire at 40%.
      - Low during screaming fear: is_laughing fires when jaw+biu raise together
        (which is exactly what screaming looks like), reducing biu weight 1.3→0.3.

    Flags (all independent, can be combined):
      fix_laughing      — is_laughing requires smile > 0.15 (fixes scream misclassification)
      harder_smile_gate — fear sd gate uses smile*4.0; zeroes at smile>0.25 vs pos_sig>0.50
      remove_eye_squint — drop eyeSquint from ctx group (Duchenne marker, fires during joy)
      joy_jaw_suppress  — jawOpen weight → 0.0 when is_joyful (laugh jaw ≠ fear jaw)
      smile_penalty     — coefficient for final `- smile * c` subtraction (default 0.15)
    """
    def gc(name):
        col = f"mp_{name}"
        return (pd.to_numeric(model_df[col], errors='coerce').fillna(0.0).values
                if col in model_df.columns else np.zeros(len(model_df)))

    face_det = (model_df.get("mp_face_detected", pd.Series(0, index=model_df.index))
                .fillna(0).astype(int).values.astype(bool))

    smile_l, smile_r   = gc("mouthSmileLeft"),   gc("mouthSmileRight")
    cheek_sl, cheek_sr = gc("cheekSquintLeft"),  gc("cheekSquintRight")
    frown_l, frown_r   = gc("mouthFrownLeft"),   gc("mouthFrownRight")
    press_l, press_r   = gc("mouthPressLeft"),   gc("mouthPressRight")
    brow_dl, brow_dr   = gc("browDownLeft"),     gc("browDownRight")
    biu                = gc("browInnerUp")
    eye_wl, eye_wr     = gc("eyeWideLeft"),      gc("eyeWideRight")
    eye_sql, eye_sqr   = gc("eyeSquintLeft"),    gc("eyeSquintRight")
    jaw_open           = gc("jawOpen")
    nose_snl, nose_snr = gc("noseSneerLeft"),    gc("noseSneerRight")

    smile   = np.maximum(smile_l, smile_r)
    cheek_s = np.maximum(cheek_sl, cheek_sr)
    pos_sig = np.maximum(smile, cheek_s * 0.8)
    frown   = np.maximum(frown_l, frown_r)
    press   = np.maximum(press_l, press_r)
    brow_d  = np.maximum(brow_dl, brow_dr)

    is_conc   = (brow_d > 0.2) & (press < 0.05)
    is_sad    = (frown > press) & (frown > 0.1)
    is_joyful = (smile > joy_thresh) if joy_thresh is not None else np.zeros(len(model_df), dtype=bool)

    # stress score — unchanged from v4
    sw = (1.5 + 1.5 + 0.8 + 0.8) + 0.5 + 0.5 + 0.3
    stress_score = (press_l * 1.5 + press_r * 1.5 + nose_snl * 0.8 + nose_snr * 0.8) / sw

    # ctx score — optionally drop eyeSquint (Duchenne smile marker)
    if remove_eye_squint:
        cw     = 0.7 + 0.7 + 0.4 + 0.4
        ct_raw = brow_dl * 0.7 + brow_dr * 0.7 + frown_l * 0.4 + frown_r * 0.4
    else:
        cw     = 0.7 + 0.7 + 0.6 + 0.6 + 0.4 + 0.4
        ct_raw = (brow_dl * 0.7 + brow_dr * 0.7 +
                  eye_sql * 0.6 + eye_sqr * 0.6 +
                  frown_l * 0.4 + frown_r * 0.4)
    ctx_m     = np.where(is_conc, 0.2, np.where(is_joyful, 0.25, np.where(is_sad, 0.5, 1.0)))
    ctx_score = ct_raw * ctx_m / cw

    # fear score — smile gate and laughing check
    if harder_smile_gate:
        sd = np.maximum(0.0, 1.0 - smile * 4.0)   # zeroes at smile>0.25 vs v4's pos_sig>0.50
    else:
        sd = np.maximum(0.0, 1.0 - pos_sig * 2.0)

    if fix_laughing:
        is_laughing = (jaw_open > 0.3) & (biu > 0.5) & (smile > 0.15)
    else:
        is_laughing = (jaw_open > 0.3) & (biu > 0.5)
    biu_w = np.where(is_laughing, 0.3, 1.3)

    jaw_w = np.where(is_joyful, 0.0, 0.6) if joy_jaw_suppress else np.full(len(model_df), 0.6)

    fw         = biu_w + 1.5 + 1.5 + jaw_w
    fear_score = (biu * biu_w + eye_wl * 1.5 + eye_wr * 1.5 + jaw_open * jaw_w) * sd / fw

    stacked = np.stack([stress_score, ctx_score, fear_score], axis=1)
    s       = np.sort(stacked, axis=1)
    blend   = 0.50 * s[:, 2] + 0.30 * s[:, 1] + 0.20 * s[:, 0]
    result  = np.clip(blend * global_scale - smile * smile_penalty, 0.0, 1.0)
    result[~face_det] = 0.0
    return result


def _recompute_v5a_batch(model_df, global_scale=1.8, joy_thresh=0.20):
    """v5a: fix is_laughing only. Targets screaming fear underdetection."""
    return _recompute_v5_batch(model_df, global_scale, joy_thresh,
                               fix_laughing=True)


def _recompute_v5b_batch(model_df, global_scale=1.8, joy_thresh=0.20):
    """v5b: fix laughing + harder smile gate. Targets joy-spike FPs."""
    return _recompute_v5_batch(model_df, global_scale, joy_thresh,
                               fix_laughing=True, harder_smile_gate=True)


def _recompute_v5c_batch(model_df, global_scale=1.8, joy_thresh=0.20):
    """v5c: fix laughing + remove eyeSquint from ctx (Duchenne marker)."""
    return _recompute_v5_batch(model_df, global_scale, joy_thresh,
                               fix_laughing=True, remove_eye_squint=True)


def _recompute_v5d_batch(model_df, global_scale=1.8, joy_thresh=0.20):
    """v5d: fix laughing + jaw suppression when joyful."""
    return _recompute_v5_batch(model_df, global_scale, joy_thresh,
                               fix_laughing=True, joy_jaw_suppress=True)


def _recompute_v5e_batch(model_df, global_scale=1.8, joy_thresh=0.20):
    """v5e: all v5 fixes combined + stronger smile penalty (0.35)."""
    return _recompute_v5_batch(model_df, global_scale, joy_thresh,
                               fix_laughing=True, harder_smile_gate=True,
                               remove_eye_squint=True, joy_jaw_suppress=True,
                               smile_penalty=0.35)


_V5_DISPATCH = {
    "v5a": _recompute_v5a_batch,
    "v5b": _recompute_v5b_batch,
    "v5c": _recompute_v5c_batch,
    "v5d": _recompute_v5d_batch,
    "v5e": _recompute_v5e_batch,
}


def _recompute_fear_only_batch(model_df):
    """Vectorized fear-group-only score: browInnerUp, eyeWide{L,R}, jawOpen.

    Drops stress and context groups. Designed for per-segment normalization:
    a subject with suppressed AUs (e.g. glasses dampening eyeWide) will have
    their segment max mapped to 1.0 rather than being diluted by session average.
    """
    eye_wide = (model_df.get("mp_eyeWideLeft",    pd.Series(0.0, index=model_df.index)) +
                model_df.get("mp_eyeWideRight",   pd.Series(0.0, index=model_df.index))) / 2.0
    brow_up  =  model_df.get("mp_browInnerUp",    pd.Series(0.0, index=model_df.index))
    jaw_open =  model_df.get("mp_jawOpen",        pd.Series(0.0, index=model_df.index))
    smile    = (model_df.get("mp_mouthSmileLeft", pd.Series(0.0, index=model_df.index)) +
                model_df.get("mp_mouthSmileRight",pd.Series(0.0, index=model_df.index))) / 2.0
    discount = (1.0 - smile / 0.25).clip(lower=0.0)
    raw      = (eye_wide * 1.5 + brow_up * 1.3 + jaw_open * 0.6) / (1.5 + 1.3 + 0.6)
    return (raw * discount).fillna(0.0).values


def _compute_raw_series(model_df, tension_fn, global_scale, joy_thresh):
    """Derive raw (pre-AGC) tension array for every row.

    tension_fn: "stored_v3"         — use stored mp_tension_v3 column (exact)
                "stored_v4"         — use stored mp_tension_v4 column
                "stored_v5a"-"v5e"  — use stored mp_tension_v5a…v5e columns
                "v4"                — vectorized recomputation from blendshape columns
                "v5a"-"v5e"         — parametric v5 variants (see _recompute_v5_batch)
                "fear_only"         — browInnerUp+eyeWide+jawOpen only; use with AGC
                "fear_only_seg"     — same score, normalized per segment; bypass AGC
    """
    if tension_fn.startswith("stored_"):
        col_name = "mp_tension_" + tension_fn[len("stored_"):]
        if col_name in model_df.columns:
            return model_df[col_name].fillna(0.0).values.copy()
        raise KeyError(f"Column '{col_name}' not found in model CSV — "
                       f"re-record with test_mediapipe.py to generate it.")
    if tension_fn == "fear_only":
        return _recompute_fear_only_batch(model_df)
    if tension_fn == "fear_only_seg":
        raw      = _recompute_fear_only_batch(model_df)
        face_det = (model_df.get("mp_face_detected",
                                 pd.Series(1, index=model_df.index))
                    .fillna(0).astype(int).values)
        bs_arr   = _build_bs_arr(model_df)
        cuts     = _detect_cut_frames(face_det, bs_arr)
        return _normalize_per_segment(raw, cuts)
    if tension_fn in _V5_DISPATCH:
        return _V5_DISPATCH[tension_fn](model_df, global_scale or 1.8, joy_thresh)
    return _recompute_v4_batch(model_df, global_scale, joy_thresh)


# ── AGC replay ────────────────────────────────────────────────────────────────

def _replay_agc(raw_arr, face_det_arr, gain, decay=0.9990, init_ambient=0.12, burnin=60,
                bs_arr=None, no_face_gap=3, l2_threshold=0.3):
    """Replay TensionAGC on a raw tension array. Returns normalized array.

    gain=None → no-AGC mode: returns raw_arr unchanged.
    bs_arr: optional (N, F) float array of per-frame blendshape vectors. When
    provided, an L2 jump > l2_threshold between consecutive frames triggers a
    burn-in reset (hard cut with continuous face detection). A gap of ≥ no_face_gap
    consecutive no-face frames also resets burn-in when the face returns.
    """
    if gain is None:
        return raw_arr.copy()

    out          = np.zeros(len(raw_arr))
    ambient      = init_ambient
    face_count   = 0
    no_face_streak = 0

    for i in range(len(raw_arr)):
        raw  = float(raw_arr[i])
        face = bool(face_det_arr[i])

        if face:
            # Reset burn-in on no-face gap (clip cut with black frames)
            if no_face_streak >= no_face_gap:
                face_count = 0
            no_face_streak = 0

            # Reset burn-in on L2 blendshape jump (hard cut, face never dropped)
            if bs_arr is not None and i > 0:
                diff = float(np.linalg.norm(bs_arr[i] - bs_arr[i - 1]))
                if diff > l2_threshold:
                    face_count = 0

            face_count += 1
            if face_count <= burnin:
                ambient = 0.90 * ambient + 0.10 * raw
                out[i]  = 0.0
                continue
            ambient = decay * ambient + (1.0 - decay) * raw
        else:
            no_face_streak += 1

        relative = raw / max(ambient, 0.005)
        out[i]   = min(1.0, max(0.0, (relative - 1.0) * gain))

    return out


# ── Scene-cut helpers ─────────────────────────────────────────────────────────

def _build_bs_arr(model_df):
    """Extract blendshape matrix (N, F) from stored mp_<au> columns.

    Returns None if no blendshape columns are present (compact CSV).
    """
    bs_cols = [
        c for c in model_df.columns
        if c.startswith("mp_")
        and c not in ("mp_face_detected", "mp_startle_score")
        and not c.startswith("mp_tension_")
        and not c.startswith("mp_velocity_")
        and c not in ("mp_ctx_tag", "mp_velocity_tag")
    ]
    if not bs_cols:
        return None
    return model_df[bs_cols].fillna(0.0).values.astype(np.float32)


def _detect_cut_frames(face_det_arr, bs_arr=None, no_face_gap=3, l2_threshold=0.3):
    """Return boolean array: True at the frame where a scene cut is detected.

    Two signals: ① no-face gap ≥ no_face_gap → cut on first face-return frame.
                 ② L2 blendshape jump > l2_threshold → cut on the new frame.
    """
    cuts   = np.zeros(len(face_det_arr), dtype=bool)
    streak = 0
    for i, face in enumerate(face_det_arr):
        if not face:
            streak += 1
        else:
            if streak >= no_face_gap:
                cuts[i] = True
            streak = 0
    if bs_arr is not None:
        norms = np.linalg.norm(np.diff(bs_arr.astype(np.float64), axis=0), axis=1)
        jump_frames = np.where(norms > l2_threshold)[0] + 1  # index of the new frame
        for j in jump_frames:
            if j < len(face_det_arr) and face_det_arr[j]:
                cuts[j] = True
    return cuts


def _normalize_per_segment(raw_arr, cut_frames):
    """Normalize each segment between scene cuts to [0, 1] by its own maximum.

    S06 eyeWide peaks at 0.12 (glasses) — per-segment normalization maps
    that 0.12 to 1.0, the correct relative ceiling for that subject.
    """
    out       = raw_arr.copy().astype(np.float64)
    seg_start = 0
    indices   = list(np.where(cut_frames)[0]) + [len(raw_arr)]
    for cut_idx in indices:
        seg     = out[seg_start:cut_idx]
        seg_max = seg.max()
        if seg_max > 1e-6:
            out[seg_start:cut_idx] = seg / seg_max
        seg_start = cut_idx
    return np.clip(out, 0.0, 1.0)


# ── Formula recomputation ────────────────────────────────────────────────────

def _recompute_formulas(model_df, new_mp_tension, anger_coeff=0.6):
    """Return copy of model_df with F0-F11 recomputed using new_mp_tension."""
    df   = model_df.copy()
    mp_t = pd.Series(new_mp_tension, index=df.index)

    hs_fear    = pd.to_numeric(df.get("hs_fear",    0), errors='coerce').fillna(0.0)
    hs_arousal = pd.to_numeric(df.get("hs_arousal", 0), errors='coerce').fillna(0.0)
    hs_surp    = pd.to_numeric(df.get("hs_surprise",0), errors='coerce').fillna(0.0)
    hs_anger   = pd.to_numeric(df.get("hs_anger",   0), errors='coerce').fillna(0.0)
    startle    = pd.to_numeric(df.get("mp_startle_score", 0), errors='coerce').fillna(0.0).clip(0, 10) / 10.0

    df['f0']  = (0.60 * hs_arousal + 0.40 * mp_t).clip(0, 1)
    df['f1']  = (0.60 * hs_fear    + 0.40 * mp_t).clip(0, 1)
    df['f2']  = (0.40 * hs_fear + 0.20 * hs_surp + 0.40 * mp_t).clip(0, 1)
    df['f3']  = (0.50 * hs_fear + 0.20 * hs_arousal + 0.30 * mp_t).clip(0, 1)
    df['f4']  = (df['f0'] * (hs_anger <= hs_fear).astype(float)).clip(0, 1)
    df['f5']  = (0.50 * hs_fear + 0.30 * mp_t + 0.20 * startle).clip(0, 1)
    df['f6']  = ((0.60 * hs_fear + 0.40 * mp_t) * (1.0 - hs_anger).clip(0, 1)).clip(0, 1)
    df['f7']  = hs_fear.clip(0, 1)
    df['f8']  = mp_t.clip(0, 1)
    df['f9']  = np.maximum(hs_fear, mp_t).clip(0, 1)
    df['f10'] = np.sqrt((hs_fear * mp_t).clip(0, 1)).clip(0, 1)
    df['f11'] = (hs_fear - anger_coeff * hs_anger).clip(0, 1)
    return df


# ── Public formula constants ─────────────────────────────────────────────────

MP_FORMULA_COLS = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
                   'f7', 'f8', 'f9', 'f10', 'f11']

MP_FORMULA_LABELS = {
    'f0': 'F0', 'f1': 'F1', 'f2': 'F2', 'f3': 'F3',
    'f4': 'F4', 'f5': 'F5', 'f6': 'F6',
    'f7': 'hs_only', 'f8': 'mp_only',
    'f9': 'max', 'f10': 'geomean', 'f11': 'hs-anger',
}

# Formulas that don't use mp_tension — excluded from BestFml ranking in mp sweeps
# (f7 = raw hs_fear, f11 = hs_fear - anger; their F1 is constant across all mp configs)
_HS_ONLY_COLS   = {'f7', 'f11'}
_MP_SENSITIVE_COLS = [c for c in MP_FORMULA_COLS if c not in _HS_ONLY_COLS]


# ── Sweep config builder ─────────────────────────────────────────────────────

def make_mp_sweep_configs(sweep_scales=None, sweep_gains=None, sweep_joy=None,
                          sweep_v5=True, model_df=None):
    """Return list of (tension_fn, global_scale, joy_thresh, agc_gain) tuples.

    sweep_v5=True appends one entry per v5 variant (v5a–v5e) × all agc_gains at
    scale=1.8, joy_thresh=0.20 (fixed — no scale sweep for v5 yet).
    If the model CSV has stored mp_tension_v* columns (from test_mediapipe.py),
    also add "stored_v*" entries which read columns directly (no blendshape approx).
    """
    if sweep_scales is None:
        sweep_scales = [1.4, 1.6, 1.8, 2.0, 2.2]
    if sweep_gains is None:
        sweep_gains = [None, 0.40, 0.50, 0.60, 0.70]
    if sweep_joy is None:
        sweep_joy = [None, 0.20]

    stored_variants = ["v3", "v4", "v5a", "v5b", "v5c", "v5d", "v5e"]
    configs = []
    for variant in stored_variants:
        if model_df is None or f"mp_tension_{variant}" in model_df.columns:
            for gain in sweep_gains:
                configs.append((f"stored_{variant}", None, None, gain))
    for scale in sweep_scales:
        for joy in sweep_joy:
            for gain in sweep_gains:
                configs.append(("v4", scale, joy, gain))
    if sweep_v5:
        for v5_fn in ["v5a", "v5b", "v5c", "v5d", "v5e"]:
            for gain in sweep_gains:
                configs.append((v5_fn, 1.8, 0.20, gain))
    # Fear-group-only score: browInnerUp + eyeWide + jawOpen, no stress/context groups.
    # fear_only uses AGC normalization; fear_only_seg uses per-segment max normalization
    # (bypasses AGC — pass gain=None to _replay_agc).
    for gain in sweep_gains:
        configs.append(("fear_only", None, None, gain))
    configs.append(("fear_only_seg", None, None, None))  # gain=None → bypass AGC
    return configs


def make_mp_sweep_configs_stored(model_df, sweep_gains=None):
    """Configs that read pre-computed tension columns directly (no blendshape approx).

    Use when model CSV was produced by test_mediapipe.py, which saves mp_tension_v3
    through mp_tension_v5e. Skips variants whose column is absent.
    """
    if sweep_gains is None:
        sweep_gains = [None, 0.40, 0.50, 0.60, 0.70]

    stored_variants = ["v3", "v4", "v5a", "v5b", "v5c", "v5d", "v5e"]
    configs = []
    for variant in stored_variants:
        col = f"mp_tension_{variant}"
        if col in model_df.columns:
            for gain in sweep_gains:
                configs.append((f"stored_{variant}", None, None, gain))
    return configs


# ── Per-formula metrics for one MP config ────────────────────────────────────

def compute_metrics_mp(model_df, gt_df, threshold, min_frames, pad_start, pad_end,
                       window_size, fill_ratio, col,
                       tension_fn, global_scale, joy_thresh, agc_gain,
                       agc_decay=0.9990, agc_init=0.12, agc_burnin=60, anger_coeff=0.6):
    """Evaluate (tp, fp, gt_caught, n_gt) for one MP tension config + formula column.

    Delegates to v2's compute_metrics after recomputing tension and formulas.
    TP is GT-centric (one annotated cluster = 1 TP regardless of how many
    trigger blocks fall inside it).
    Accepts mp_only CSVs (no hs_* columns) — patches zeros in-place.
    """
    _patch_hs_columns(model_df)
    face_det = (model_df.get("mp_face_detected", pd.Series(1, index=model_df.index))
                .fillna(0).astype(int).values)
    bs_arr   = _build_bs_arr(model_df)
    raw      = _compute_raw_series(model_df, tension_fn, global_scale or 1.8, joy_thresh)
    normed   = _replay_agc(raw, face_det, agc_gain, agc_decay, agc_init, agc_burnin,
                           bs_arr=bs_arr)
    eval_df  = _recompute_formulas(model_df, normed, anger_coeff)
    return compute_metrics(eval_df, gt_df, threshold, min_frames, pad_start, pad_end,
                           window_size, fill_ratio, col)


# ── Sweep printer (used by run_mp_analysis and evaluate_all.py) ──────────────

def print_mp_parameter_sweep(
    eval_func,
    sweep_configs=None,
    formula_cols=None,
    spot_cols=None,
    title="MP Tension Config Sweep",
):
    """Print a ranked tension-config sweep table and return results list.

    eval_func(col, tension_fn, global_scale, joy_thresh, agc_gain)
        -> (tp, fp, gt_caught, n_gt)

    BestFml ranks only mp-sensitive formulas (excludes hs-only f7/f11).
    Results are sorted by best F1 across mp-sensitive formula_cols per config.
    """
    if sweep_configs is None:
        sweep_configs = make_mp_sweep_configs()
    if formula_cols is None:
        formula_cols = MP_FORMULA_COLS
    if spot_cols is None:
        spot_cols = ['f7', 'f8', 'f1', 'f9', 'f10']

    n_configs = len(sweep_configs)
    results = []
    for cfg_idx, (tension_fn, scale, joy, gain) in enumerate(sweep_configs, 1):
        print(f"\r  [{cfg_idx:>3}/{n_configs}] Evaluating tension config... ", end='', flush=True)
        formula_f1 = {}
        for col in formula_cols:
            tp, fp, gt_caught, n_gt = eval_func(col, tension_fn, scale, joy, gain)
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = gt_caught / n_gt if n_gt > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            formula_f1[col] = f1

        # BestFml: only rank mp-sensitive formulas (f7/f11 are hs-only baselines)
        sensitive = {c: formula_f1[c] for c in formula_cols if c in _MP_SENSITIVE_COLS}
        best_col = max(sensitive, key=sensitive.get) if sensitive else (
                   max(formula_f1, key=formula_f1.get) if formula_f1 else None)
        best_f1  = formula_f1.get(best_col, 0.0)

        if tension_fn.startswith("stored_"):
            cfg_str = tension_fn
        elif tension_fn in _V5_DISPATCH:
            cfg_str = f"{tension_fn}(s={scale or 1.8:.1f})"
        elif tension_fn in ("fear_only", "fear_only_seg"):
            cfg_str = tension_fn
        else:
            joy_tag = f"j={joy:.2f}" if joy is not None else "j=off"
            cfg_str = f"v4(s={scale or 1.8:.1f},{joy_tag})"
        gain_str = f"gain={gain:.2f}" if gain is not None else "noAGC"

        results.append({
            "tension_fn": tension_fn, "scale": scale, "joy": joy, "gain": gain,
            "cfg_str": cfg_str, "gain_str": gain_str,
            "formula_f1": formula_f1, "best_col": best_col, "best_f1": best_f1,
        })

    print(f"\r  Done — {n_configs} configs evaluated.{' ' * 20}", flush=True)
    results.sort(key=lambda r: r["best_f1"], reverse=True)
    overall_best = results[0]["best_f1"] if results else 0.0

    hdr_spots = "  ".join(f"{c:>7}" for c in spot_cols)
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"  (BestFml ranks mp-sensitive formulas only; f7/f11 shown as spots)")
    print(f"{'='*72}")
    print(f"\n{'TensionConfig':<30}  {'Gain':<8}  {'BestFml':<9}  {'BestF1':>7}  {hdr_spots}")
    print("-" * (30 + 8 + 9 + 7 + 5 + len(spot_cols) * 9 + 12))

    prev_fn = None
    for r in results:
        if r["tension_fn"] != prev_fn:
            print()
            prev_fn = r["tension_fn"]
        marker = "  <-- BEST" if abs(r["best_f1"] - overall_best) < 1e-9 else ""
        spots  = "  ".join(f"{r['formula_f1'].get(c, 0.0):>7.4f}" for c in spot_cols)
        print(f"{r['cfg_str']:<30}  {r['gain_str']:<8}  "
              f"{r['best_col'] or '---':<9}  {r['best_f1']:>7.4f}  {spots}{marker}")

    print(f"\n\n{'='*72}")
    print("  === Top 5 Configurations — All Formulas ===")
    print(f"{'='*72}")
    for rank, r in enumerate(results[:5], 1):
        print(f"\n  #{rank}: {r['cfg_str']}  {r['gain_str']}"
              f"  →  best={r['best_col']} (F1={r['best_f1']:.4f})")
        for col in formula_cols:
            f1    = r["formula_f1"].get(col, 0.0)
            label = MP_FORMULA_LABELS.get(col, col)
            mark  = "  <--" if col == r["best_col"] else ""
            print(f"    {col:<5} ({label:<10}): {f1:.4f}{mark}")
    print()

    return results


# ── Memory-efficient combined sweep across multiple sessions ─────────────────

def run_combined_mp_sweep(
    sessions,
    threshold, min_frames, window_size, fill_ratio,
    pad_start, pad_end,
    formula_cols=None,
    sweep_configs=None,
    spot_cols=None,
    agc_decay=0.9990, agc_init=0.12, agc_burnin=60, anger_coeff=0.6,
    title="MP Tension Config Sweep",
):
    """Memory-efficient combined mp sweep across multiple loaded sessions.

    sessions: list of {'model_df': pd.DataFrame, 'gt_df': pd.DataFrame}
    One config is processed at a time; eval_dfs are freed after each config.
    BestFml ranks only mp-sensitive formulas (f7/f11 excluded — constant across configs).
    """
    if formula_cols is None:
        formula_cols = MP_FORMULA_COLS
    if sweep_configs is None:
        # Build a representative model_df intersection so stored_* configs are only
        # included when the column is present in every session.
        shared_cols = set.intersection(*(set(sd['model_df'].columns) for sd in sessions))
        _rep_df = sessions[0]['model_df'][list(shared_cols)]
        sweep_configs = make_mp_sweep_configs(model_df=_rep_df)
    if spot_cols is None:
        spot_cols = ['f7', 'f8', 'f1', 'f9', 'f10']

    # Patch hs_* columns in-place for any mp_only sessions
    for sd in sessions:
        _patch_hs_columns(sd['model_df'])

    # Pre-extract face_det and blendshape arrays once (reused across all configs)
    face_det_arrays = [
        (sd['model_df'].get("mp_face_detected", pd.Series(1, index=sd['model_df'].index))
         .fillna(0).astype(int).values)
        for sd in sessions
    ]
    bs_arrays = [_build_bs_arr(sd['model_df']) for sd in sessions]

    n_configs = len(sweep_configs)
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"  {n_configs} tension configs × {len(formula_cols)} formulas × {len(sessions)} session(s)")
    print(f"{'='*72}", flush=True)

    results = []
    for cfg_idx, (tension_fn, scale, joy, gain) in enumerate(sweep_configs, 1):
        print(f"\r  [{cfg_idx:>3}/{n_configs}] Computing tension config... ", end='', flush=True)

        # Compute eval_df per session for this config only — freed at end of iteration
        session_eval_dfs = [
            _recompute_formulas(
                sd['model_df'],
                _replay_agc(
                    _compute_raw_series(sd['model_df'], tension_fn, scale or 1.8, joy),
                    face_det, gain, agc_decay, agc_init, agc_burnin,
                    bs_arr=bs_arr,
                ),
                anger_coeff,
            )
            for sd, face_det, bs_arr in zip(sessions, face_det_arrays, bs_arrays)
        ]

        formula_f1 = {}
        for col in formula_cols:
            tot_tp = tot_fp = tot_gt_caught = tot_n_gt = 0
            for sd, eval_df in zip(sessions, session_eval_dfs):
                if col not in eval_df.columns:
                    continue
                tp, fp, gt_caught, n_gt = compute_metrics(
                    eval_df, sd['gt_df'],
                    threshold, min_frames, pad_start, pad_end,
                    window_size, fill_ratio, col,
                )
                tot_tp += tp; tot_fp += fp
                tot_gt_caught += gt_caught; tot_n_gt += n_gt
            p  = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
            r  = tot_gt_caught / tot_n_gt if tot_n_gt > 0 else 0.0
            formula_f1[col] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        session_eval_dfs = None   # allow GC to free the dataframe copies

        sensitive = {c: formula_f1[c] for c in formula_cols if c in _MP_SENSITIVE_COLS}
        best_col = max(sensitive, key=sensitive.get) if sensitive else (
                   max(formula_f1, key=formula_f1.get) if formula_f1 else None)
        best_f1  = formula_f1.get(best_col, 0.0)

        if tension_fn.startswith("stored_"):
            cfg_str = tension_fn
        elif tension_fn in _V5_DISPATCH:
            cfg_str = f"{tension_fn}(s={scale or 1.8:.1f})"
        elif tension_fn in ("fear_only", "fear_only_seg"):
            cfg_str = tension_fn
        else:
            joy_tag = f"j={joy:.2f}" if joy is not None else "j=off"
            cfg_str = f"v4(s={scale or 1.8:.1f},{joy_tag})"
        gain_str = f"gain={gain:.2f}" if gain is not None else "noAGC"

        results.append({
            "tension_fn": tension_fn, "scale": scale, "joy": joy, "gain": gain,
            "cfg_str": cfg_str, "gain_str": gain_str,
            "formula_f1": formula_f1, "best_col": best_col, "best_f1": best_f1,
        })

    print(f"\r  Done — {n_configs} configs evaluated.{' ' * 20}", flush=True)
    results.sort(key=lambda r: r["best_f1"], reverse=True)
    overall_best = results[0]["best_f1"] if results else 0.0

    hdr_spots = "  ".join(f"{c:>7}" for c in spot_cols)
    print(f"\n{'TensionConfig':<30}  {'Gain':<8}  {'BestFml':<9}  {'BestF1':>7}  {hdr_spots}")
    print(f"  (BestFml ranks mp-sensitive formulas only; f7/f11 shown as spots)")
    print("-" * (30 + 8 + 9 + 7 + 5 + len(spot_cols) * 9 + 12))

    prev_fn = None
    for r in results:
        if r["tension_fn"] != prev_fn:
            print()
            prev_fn = r["tension_fn"]
        marker = "  <-- BEST" if abs(r["best_f1"] - overall_best) < 1e-9 else ""
        spots  = "  ".join(f"{r['formula_f1'].get(c, 0.0):>7.4f}" for c in spot_cols)
        print(f"{r['cfg_str']:<30}  {r['gain_str']:<8}  "
              f"{r['best_col'] or '---':<9}  {r['best_f1']:>7.4f}  {spots}{marker}")
    sys.stdout.flush()

    print(f"\n\n{'='*72}")
    print("  === Top 5 Configurations — All Formulas ===")
    print(f"{'='*72}")
    for rank, r in enumerate(results[:5], 1):
        print(f"\n  #{rank}: {r['cfg_str']}  {r['gain_str']}"
              f"  →  best={r['best_col']} (F1={r['best_f1']:.4f})")
        for col in formula_cols:
            f1    = r["formula_f1"].get(col, 0.0)
            label = MP_FORMULA_LABELS.get(col, col)
            hs_tag = "  [hs-only]" if col in _HS_ONLY_COLS else ""
            mark   = "  <--" if col == r["best_col"] else ""
            print(f"    {col:<5} ({label:<10}): {f1:.4f}{mark}{hs_tag}")
    print()
    sys.stdout.flush()

    return results


# ── Per-session MP analysis ───────────────────────────────────────────────────

def run_mp_analysis(
    gt_file,
    model_file,
    threshold    = 0.50,
    min_frames   = 15,
    window_size  = 30,
    fill_ratio   = 0.25,
    pad_start    = 0.5,
    pad_end      = 1.0,
    sweep_scales = None,
    sweep_gains  = None,
    sweep_joy    = None,
    agc_decay    = 0.9990,
    agc_init     = 0.12,
    agc_burnin   = 60,
    anger_coeff  = 0.6,
):
    """Run MP feature sweep for a single session and print ranked results."""
    model_df = pd.read_csv(model_file, low_memory=False)
    raw_gt   = pd.read_csv(gt_file, low_memory=False)
    raw_gt['start_val']  = raw_gt['start_s'].apply(time_to_sec)
    raw_gt['end_val']    = raw_gt['end_s'].apply(time_to_sec)
    raw_gt['label_norm'] = raw_gt['label'].str.strip().str.lower()
    scoreable_norm = {l.lower() for l in SCOREABLE_LABELS}
    gt_df    = raw_gt[raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)
    excluded = raw_gt[~raw_gt['label_norm'].isin(scoreable_norm)].reset_index(drop=True)

    mp_only_mode = 'hs_fear' not in model_df.columns
    _patch_hs_columns(model_df)   # adds zero hs_* cols if absent; no-op otherwise

    face_det = (model_df.get("mp_face_detected",
                              pd.Series(1, index=model_df.index))
                .fillna(0).astype(int).values)
    bs_arr   = _build_bs_arr(model_df)

    _sweep_scales = sweep_scales
    missing = [f"mp_{n}" for n in _STORED_BS if f"mp_{n}" not in model_df.columns]
    if missing and not mp_only_mode:
        print(f"\n  [WARNING] {len(missing)} blendshape columns absent (compact CSV?) — "
              f"v4 recomputation disabled.")
        _sweep_scales = []

    print(f"\n{'='*72}")
    print(f"  MP Feature Sweep — {os.path.basename(model_file)}")
    if mp_only_mode:
        stored = [c for c in model_df.columns if c.startswith("mp_tension_")]
        print(f"  [MP-ONLY] No hs_* columns — sweeping stored tension variants only.")
        print(f"  Stored variants: {', '.join(stored)}")
        print(f"  f7/f10/f11 will be zero; f8/f9 carry full signal; f0-f6 are attenuated.")
    print(f"  GT: {os.path.basename(gt_file)}")
    print(f"  GT fear events: {len(gt_df)}  |  excluded (not scored): {len(excluded)}")
    print(f"  Fixed: threshold={threshold}  min_frames={min_frames}  "
          f"window_size={window_size}  fill_ratio={fill_ratio:.0%}")
    print(f"  pad=(-{pad_start}s / +{pad_end}s)  anger_coeff={anger_coeff}")
    print(f"  AGC: decay={agc_decay}  init={agc_init}  burnin={agc_burnin}fr")
    print(f"{'='*72}")

    if mp_only_mode:
        configs = make_mp_sweep_configs_stored(model_df, sweep_gains)
    else:
        configs = make_mp_sweep_configs(_sweep_scales, sweep_gains, sweep_joy, model_df=model_df)

    # Cache eval_df per config to avoid recomputing tension for each formula
    config_cache = {}

    def eval_func(col, tension_fn, scale, joy, gain):
        key = (tension_fn, scale, joy, gain)
        if key not in config_cache:
            raw    = _compute_raw_series(model_df, tension_fn, scale or 1.8, joy)
            normed = _replay_agc(raw, face_det, gain, agc_decay, agc_init, agc_burnin,
                                 bs_arr=bs_arr)
            config_cache[key] = _recompute_formulas(model_df, normed, anger_coeff)
        return compute_metrics(config_cache[key], gt_df, threshold, min_frames,
                               pad_start, pad_end, window_size, fill_ratio, col)

    spot_cols = ['f8', 'f9', 'f5', 'f1', 'f0'] if mp_only_mode else None
    return print_mp_parameter_sweep(
        eval_func, configs,
        spot_cols=spot_cols,
        title=f"MP Tension Sweep — {os.path.basename(model_file)}"
    )


# ── Report generator with log file ───────────────────────────────────────────

class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)

    def flush(self):
        for f in self.files:
            f.flush()


_LOG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'comparisons')
)


def generate_mp_report(
    gt_file,
    model_file,
    threshold    = 0.50,
    min_frames   = 15,
    window_size  = 30,
    fill_ratio   = 0.25,
    pad_start    = 0.5,
    pad_end      = 1.0,
    sweep_scales = None,
    sweep_gains  = None,
    sweep_joy    = None,
    agc_decay    = 0.9990,
    agc_init     = 0.12,
    agc_burnin   = 60,
    anger_coeff  = 0.6,
    quiet        = False,
    output_dir   = None,
):
    """Run the full MP tension sweep for one session and save a log file.

    Mirrors generate_v2_report() — call with quiet=True from evaluate_all.py
    to produce a per-session log without cluttering the terminal.
    """
    log_stem = os.path.splitext(os.path.basename(model_file))[0]
    if output_dir:
        log_dir = output_dir
        session_tag = log_stem.split("_mp_hs_")[-1] if "_mp_hs_" in log_stem else log_stem
        log_path = os.path.join(log_dir, f"{session_tag}_mp_sweep.txt")
    else:
        base_dir = _LOG_DIR
        ts_prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_tag = log_stem.split("_mp_hs_")[-1] if "_mp_hs_" in log_stem else log_stem
        log_dir = os.path.join(base_dir, "standalone", f"{ts_prefix}_{session_tag}")
        log_path = os.path.join(log_dir, f"{session_tag}_mp_sweep.txt")
    os.makedirs(log_dir, exist_ok=True)

    with open(log_path, 'w') as _fh:
        _orig      = sys.stdout
        sys.stdout = _fh if quiet else _Tee(sys.stdout, _fh)
        try:
            run_mp_analysis(
                gt_file      = gt_file,
                model_file   = model_file,
                threshold    = threshold,
                min_frames   = min_frames,
                window_size  = window_size,
                fill_ratio   = fill_ratio,
                pad_start    = pad_start,
                pad_end      = pad_end,
                sweep_scales = sweep_scales,
                sweep_gains  = sweep_gains,
                sweep_joy    = sweep_joy,
                agc_decay    = agc_decay,
                agc_init     = agc_init,
                agc_burnin   = agc_burnin,
                anger_coeff  = anger_coeff,
            )
        finally:
            sys.stdout = _orig
    print(f"\n[mp report] Log saved → {log_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

_GT_FILE    = '/home/burak/Desktop/Bitirme/Annotations/S02_Vid04.csv'
_MODEL_FILE = '/home/burak/Desktop/Bitirme/Pipeline/logs/sessions/S02_Vid04_dim/20260502_095248_mp_only_S02_Vid04_dim.csv'


def _parse_args():
    p = argparse.ArgumentParser(description="MP feature sweep comparison tool")
    p.add_argument("--gt-csv",      dest="gt_csv",    default=_GT_FILE)
    p.add_argument("--model-csv",   dest="model_csv", default=_MODEL_FILE)
    p.add_argument("--threshold",   type=float, default=0.50)
    p.add_argument("--min-frames",  type=int,   default=15,   dest="min_frames")
    p.add_argument("--window-size", type=int,   default=30,   dest="window_size")
    p.add_argument("--fill-ratio",  type=float, default=0.25, dest="fill_ratio")
    p.add_argument("--pad-start",   type=float, default=0.5,  dest="pad_start")
    p.add_argument("--pad-end",     type=float, default=1.0,  dest="pad_end")
    p.add_argument("--sweep-scale", nargs="+",  type=float,
                   default=[1.4, 1.6, 1.8, 2.0, 2.2], dest="sweep_scale")
    p.add_argument("--sweep-gain",  nargs="+",  type=float,
                   default=[0.0, 0.40, 0.50, 0.60, 0.70], dest="sweep_gain",
                   help="AGC gain values (0.0 = no-AGC mode)")
    p.add_argument("--sweep-joy",   nargs="+",  type=float,
                   default=[0.0, 0.20], dest="sweep_joy",
                   help="Joy-suppress thresholds (0.0 = disabled)")
    p.add_argument("--agc-decay",   type=float, default=0.9990, dest="agc_decay")
    p.add_argument("--agc-init",    type=float, default=0.12,   dest="agc_init")
    p.add_argument("--agc-burnin",  type=int,   default=60,     dest="agc_burnin")
    p.add_argument("--anger-coeff", type=float, default=0.6,    dest="anger_coeff")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    sweep_gains = [None if g == 0.0 else g for g in args.sweep_gain]
    sweep_joy   = [None if j == 0.0 else j for j in args.sweep_joy]

    generate_mp_report(
        gt_file      = args.gt_csv,
        model_file   = args.model_csv,
        threshold    = args.threshold,
        min_frames   = args.min_frames,
        window_size  = args.window_size,
        fill_ratio   = args.fill_ratio,
        pad_start    = args.pad_start,
        pad_end      = args.pad_end,
        sweep_scales = args.sweep_scale,
        sweep_gains  = sweep_gains,
        sweep_joy    = sweep_joy,
        agc_decay    = args.agc_decay,
        agc_init     = args.agc_init,
        agc_burnin   = args.agc_burnin,
        anger_coeff  = args.anger_coeff,
    )
