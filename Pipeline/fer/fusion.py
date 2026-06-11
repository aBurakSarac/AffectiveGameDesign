"""FER math and composite formula computation for the La Façade Fissuréе pipeline.

Pattern: [Strategy B] — HSEmotion valence-arousal strategy output fed here.
Pattern: [Pipeline/Filter] — formula fusion layer combining MP and HS signals.

All functions are pure (no state, no I/O). Safe to call in any order.

Formula registry
----------------
F0–F6  : LEGACY — require mp_tension; retained for cross-evaluation and reporting.
          mp_tension removed from primary FER scope 2026-05-02 (D-113).
F7     : hs_fear ablation baseline — active.
F8     : LEGACY — mp_tension only.
F9–F10 : LEGACY — max / geometric mean of hs_fear and mp_tension.
F11    : hs_fear − anger penalty — active.
F12    : hybrid_amp — WINNER (phase 4 evaluation, 2026-05-17). Primary detection formula.
F13    : former composite_fear (hs_fear × (1+mp_tension)). Preserved for reference.
"""

import math


def _clamp(v):
    return min(1.0, max(0.0, v))


def compute_all_formulas(hs_fear, hs_surprise, hs_arousal, hs_anger,
                         mp_tension, mp_startle_score):
    """Compute all FER formulas F0–F13. Returns dict: formula_id → score [0,1].

    F0–F6, F8–F10 are LEGACY (mp_tension removed from scope, D-113).
    F12 is the primary detection formula (hybrid_amp, selected phase 4).
    F13 is the former composite_fear preserved for reference.
    Pass mp_tension=0.0 when running HS-only sessions; legacy scores will be zero.
    """
    startle_norm = _clamp(mp_startle_score / 10.0)

    # --- LEGACY: mp_tension required (D-113) ---
    f0 = _clamp(0.60 * hs_arousal + 0.40 * mp_tension)
    f1 = _clamp(0.60 * hs_fear   + 0.40 * mp_tension)
    f2 = _clamp(0.40 * hs_fear   + 0.20 * hs_surprise + 0.40 * mp_tension)
    f3 = _clamp(0.50 * hs_fear   + 0.20 * hs_arousal  + 0.30 * mp_tension)
    f4 = _clamp(f0 * (1.0 if hs_anger <= hs_fear else 0.0))
    f5 = _clamp(0.50 * hs_fear   + 0.30 * mp_tension  + 0.20 * startle_norm)
    f6 = _clamp((0.60 * hs_fear  + 0.40 * mp_tension) * max(0.0, 1.0 - hs_anger))

    # --- Active: HS-only ablations and combinations ---
    f7  = _clamp(hs_fear)
    f8  = _clamp(mp_tension)                              # LEGACY: mp_tension only
    f9  = _clamp(max(hs_fear, mp_tension))                # LEGACY
    f10 = _clamp(math.sqrt(hs_fear * mp_tension))         # LEGACY
    f11 = _clamp(hs_fear - 0.6 * hs_anger)               # partial subtraction — 1:1 is too aggressive

    # --- Active: fear+arousal base amplified by tension (x=0.7 from f3 ratio) ---
    f12 = _clamp((0.7 * hs_fear + 0.3 * hs_arousal) * (1.0 + mp_tension))

    # --- Former composite_fear, preserved as f13 ---
    f13 = _clamp(hs_fear * (1.0 + mp_tension))

    return {"F0": f0, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "F5": f5, "F6": f6,
            "F7": f7, "F8": f8, "F9": f9, "F10": f10, "F11": f11, "F12": f12,
            "F13": f13}


def compute_composite_fear(hs_fear, hs_arousal, mp_tension):
    """Primary detection formula — f12 (hybrid_amp).

    Selected as best performer in phase 4 evaluation (F1=0.6971, 6 sessions).
    """
    return _clamp((0.7 * hs_fear + 0.3 * hs_arousal) * (1.0 + mp_tension))


def compute_composite_fear_legacy(hs_fear, mp_tension):
    """Former primary detection: hs_fear × (1+mp_tension). Now registered as f13."""
    return _clamp(hs_fear * (1.0 + mp_tension))


# ── Veto / agreement system — commented out, replaced by hs_fear*(1+mp_tension) ──
#
# AGREEMENT_CONFIDENCE = {
#     "AGREE_FEAR":   1.00,
#     "AGREE_STRESS": 0.90,
#     "AGREE_JOY":    0.85,
#     "AGREE_SAD":    0.85,
#     "AMBIGUOUS":    0.50,
#     "VETO":         0.05,
# }
#
#
# def gated_composite_fear(composite, agreement_tag):
#     """Weight composite_fear by cross-tool agreement confidence."""
#     return composite * AGREEMENT_CONFIDENCE.get(agreement_tag, 0.50)
#
#
# def compute_agreement(mp_ctx_tag, _mp_tension, hs_dominant, hs_arousal, hs_emotions):
#     """Compute agreement and veto between MP and HS readings."""
#     hs_fear = hs_emotions.get("Fear", 0)
#     hs_surprise = hs_emotions.get("Surprise", 0)
#     hs_anger = hs_emotions.get("Anger", 0)
#     hs_contempt = hs_emotions.get("Contempt", 0)
#     hs_happiness = hs_emotions.get("Happiness", 0)
#     hs_sadness = hs_emotions.get("Sadness", 0)
#
#     if mp_ctx_tag == "JOY":
#         if hs_dominant == "Happiness" or hs_happiness > 0.4:
#             return "AGREE_JOY", "---"
#         elif hs_dominant in ("Fear", "Anger") and hs_arousal > 0.5:
#             return "VETO", f"MP:JOY/HS:{hs_dominant}"
#
#     elif mp_ctx_tag == "FEAR":
#         if hs_fear > 0.2 or hs_surprise > 0.3 or hs_arousal > 0.5:
#             return "AGREE_FEAR", "---"
#         elif hs_dominant == "Happiness" and hs_arousal < 0.3:
#             return "VETO", "MP:FEAR/HS:Happy"
#
#     elif mp_ctx_tag == "STRESS":
#         if hs_anger > 0.2 or hs_contempt > 0.2 or hs_arousal > 0.4:
#             return "AGREE_STRESS", "---"
#         elif hs_dominant == "Happiness" and hs_arousal < 0.25:
#             return "VETO", "MP:STRESS/HS:Happy"
#
#     elif mp_ctx_tag == "SAD":
#         if hs_sadness > 0.3:
#             return "AGREE_SAD", "---"
#         elif hs_dominant == "Happiness":
#             return "VETO", "MP:SAD/HS:Happy"
#     return "AMBIGUOUS", "---"
