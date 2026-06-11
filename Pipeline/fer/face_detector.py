"""MediaPipe FaceLandmarker wrapper for the La Façade Fissuréе FER pipeline.

Pattern: [Strategy A] — MediaPipe blendshape-based FER strategy.
    Provides face detection, blendshape extraction, tension score, valence,
    AU velocity, and context tag. Runs independently from HSEmotion (Strategy B).
"""

import os
import urllib.request
import numpy as np

from fer.blendshapes import VELOCITY_AUS, STRESS_BLENDSHAPES

# ── Model path constants ──────────────────────────────────────────────────────
# go one level up from fer/ to reach Pipeline/ so models/ resolves correctly
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PIPELINE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model():
    """Download the FaceLandmarker model if not present."""
    if os.path.isfile(MODEL_PATH):
        return
    print(f"Downloading FaceLandmarker model to {MODEL_PATH} ...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def compute_tension(blendshapes_dict):
    """Compute composite facial tension score (v3 formula)."""
    get = blendshapes_dict.get

    smile_level = max(get("mouthSmileLeft", 0.0), get("mouthSmileRight", 0.0))
    cheek_squint = max(get("cheekSquintLeft", 0.0), get("cheekSquintRight", 0.0))
    positive_signal = max(smile_level, cheek_squint * 0.8)

    frown_level = max(get("mouthFrownLeft", 0.0), get("mouthFrownRight", 0.0))
    press_level = max(get("mouthPressLeft", 0.0), get("mouthPressRight", 0.0))
    is_sad = frown_level > press_level and frown_level > 0.1

    brow_down = max(get("browDownLeft", 0.0), get("browDownRight", 0.0))
    is_concentrating = brow_down > 0.2 and press_level < 0.05

    stress_total = 0.0
    stress_wsum = 0.0
    pure_stress = {
        "mouthPressLeft": 1.5, "mouthPressRight": 1.5,
        "noseSneerLeft": 0.8, "noseSneerRight": 0.8,
        "mouthUpperUpLeft": 0.5, "mouthUpperUpRight": 0.5,
        "cheekPuff": 0.3,
    }
    for name, w in pure_stress.items():
        stress_total += get(name, 0.0) * w
        stress_wsum += w

    ctx_total = 0.0
    ctx_wsum = 0.0
    brow_squint = {
        "browDownLeft": 0.7, "browDownRight": 0.7,
        "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
        "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4,
    }
    if is_concentrating:
        ctx_multiplier = 0.2
    elif is_sad:
        ctx_multiplier = 0.5
    else:
        ctx_multiplier = 1.0
    for name, w in brow_squint.items():
        ctx_total += get(name, 0.0) * w * ctx_multiplier
        ctx_wsum += w

    smile_discount = max(0.0, 1.0 - positive_signal * 2.0)
    is_laughing = (get("jawOpen", 0.0) > 0.3 and get("browInnerUp", 0.0) > 0.5
                   and smile_level > 0.15)
    brow_inner_up_weight = 0.3 if is_laughing else 1.3

    fear_total = 0.0
    fear_wsum = 0.0
    fear_signals = {
        "browInnerUp": brow_inner_up_weight,
        "eyeWideLeft": 1.5, "eyeWideRight": 1.5,
        "jawOpen": 0.6,
    }
    for name, w in fear_signals.items():
        fear_total += get(name, 0.0) * w * smile_discount
        fear_wsum += w

    total_weighted = stress_total + ctx_total + fear_total
    total_weights = stress_wsum + ctx_wsum + fear_wsum
    raw = total_weighted / total_weights
    raw -= positive_signal * 0.15

    return max(0.0, min(1.0, raw * 2.5))


class TensionAGC:
    """Adaptive gain control for lighting-agnostic tension normalization.

    The v3 formula was calibrated under normal indoor lighting. In dim/monitor-only
    conditions (experiment setup), blendshape amplitudes are globally suppressed,
    compressing tension into 0.09–0.22 regardless of emotional content.

    AGC fixes this by normalizing relative to the session's own ambient baseline
    rather than an absolute scale. Output semantics:
        raw ≈ ambient  →  output ≈ 0        (neutral)
        raw = 2× ambient  →  output ≈ 0.40  (mild reaction)
        raw = 3× ambient  →  output ≈ 0.80  (strong reaction)

    Ambient is updated only on detected-face frames so compilation black frames
    and scene cuts do not corrupt the baseline.
    """

    def __init__(self, decay: float = 0.9990, init_ambient: float = 0.12,
                 gain: float = 0.40, burnin_frames: int = 60):
        # decay=0.9990 → half-life ≈ 23 s at 30 fps once burn-in completes
        self.decay = decay
        self.ambient = init_ambient
        self._gain = gain
        self._burnin = burnin_frames   # face-detected frames before output is produced
        self._face_frames = 0          # counts only detected-face frames

    def update(self, raw_tension: float, face_detected: bool = True) -> float:
        """Update ambient baseline and return normalized tension in [0, 1].

        First `burnin_frames` face-detected frames: fast-converge ambient (decay=0.90)
        and return 0.0. This eliminates init_ambient calibration sensitivity — ambient
        converges to the true session baseline before any output is produced.
        Scene cuts (face_detected=False) do not increment the counter or move ambient.
        """
        if face_detected:
            self._face_frames += 1
            if self._face_frames <= self._burnin:
                self.ambient = 0.90 * self.ambient + 0.10 * raw_tension
                return 0.0
            self.ambient = self.decay * self.ambient + (1.0 - self.decay) * raw_tension
        relative = raw_tension / max(self.ambient, 0.005)
        return min(1.0, max(0.0, (relative - 1.0) * self._gain))

    def reset_burnin(self) -> None:
        """Re-enter fast-adapt burn-in — call on detected scene cut."""
        self._face_frames = 0


def detect_scene_cut(prev_bs_dict: dict, curr_bs_dict: dict,
                     threshold: float = 0.3) -> bool:
    """Return True when consecutive blendshape vectors diverge by > threshold (L2).

    A hard cut between compilation clips replaces one person's neutral face with
    another's, causing a sudden jump across all AUs even when face detection
    never drops. threshold=0.3 is roughly a 10-AU shift of 0.09 each.
    """
    if prev_bs_dict is None or curr_bs_dict is None:
        return False
    diff_sq = sum(
        (curr_bs_dict.get(k, 0.0) - prev_bs_dict.get(k, 0.0)) ** 2
        for k in curr_bs_dict
    )
    return diff_sq ** 0.5 > threshold


def compute_fear_score(bs_dict: dict, smile_discount_thresh: float = 0.25) -> float:
    """Fear-group-only raw score: browInnerUp, eyeWide{L,R}, jawOpen.

    Drops stress and context groups. Intended for per-segment normalization in
    offline replay — segment max maps to 1.0 regardless of absolute amplitude.
    """
    eye_wide = (bs_dict.get("eyeWideLeft", 0.0) + bs_dict.get("eyeWideRight", 0.0)) / 2.0
    brow_up  = bs_dict.get("browInnerUp", 0.0)
    jaw_open = bs_dict.get("jawOpen", 0.0)
    smile    = (bs_dict.get("mouthSmileLeft", 0.0) + bs_dict.get("mouthSmileRight", 0.0)) / 2.0
    discount = max(0.0, 1.0 - smile / max(smile_discount_thresh, 1e-6))
    raw = (eye_wide * 1.5 + brow_up * 1.3 + jaw_open * 0.6) / (1.5 + 1.3 + 0.6)
    return raw * discount


def compute_tension_v4(blendshapes_dict, global_scale: float = 1.8):
    """Compute facial tension score (v4 formula).

    Fixes v3's denominator dilution: each group is normalized by its own weight
    sum independently, then combined with a max-weighted blend. A global scale
    ensures a single strongly-active group can reach the 0.50–0.80 target range.

    Example: pure fear at 100% → fear_score=0.50 → blend=0.50 × 1.8 = 0.90.
    Compare to v3 where the same expression gives raw≈0.15 due to shared denominator.

    Disable/compare: set --tension-normalize v3 to fall back to compute_tension().
    """
    get = blendshapes_dict.get

    smile_level = max(get("mouthSmileLeft", 0.0), get("mouthSmileRight", 0.0))
    cheek_squint = max(get("cheekSquintLeft", 0.0), get("cheekSquintRight", 0.0))
    positive_signal = max(smile_level, cheek_squint * 0.8)

    frown_level = max(get("mouthFrownLeft", 0.0), get("mouthFrownRight", 0.0))
    press_level = max(get("mouthPressLeft", 0.0), get("mouthPressRight", 0.0))
    is_sad = frown_level > press_level and frown_level > 0.1

    brow_down = max(get("browDownLeft", 0.0), get("browDownRight", 0.0))
    is_concentrating = brow_down > 0.2 and press_level < 0.05

    # ── Stress group — mouth clenching, sneer ──────────────────────────────
    stress_total = 0.0
    stress_wsum = 0.0
    pure_stress = {
        "mouthPressLeft": 1.5, "mouthPressRight": 1.5,
        "noseSneerLeft": 0.8, "noseSneerRight": 0.8,
        "mouthUpperUpLeft": 0.5, "mouthUpperUpRight": 0.5,
        "cheekPuff": 0.3,
    }
    for name, w in pure_stress.items():
        stress_total += get(name, 0.0) * w
        stress_wsum += w
    stress_score = stress_total / stress_wsum

    # ── Context group — brow furrow, squint, frown ────────────────────────
    ctx_total = 0.0
    ctx_wsum = 0.0
    brow_squint = {
        "browDownLeft": 0.7, "browDownRight": 0.7,
        "eyeSquintLeft": 0.6, "eyeSquintRight": 0.6,
        "mouthFrownLeft": 0.4, "mouthFrownRight": 0.4,
    }
    # Joy suppression: eyeSquint during smiling is a JOY marker, not a stress marker.
    # Without this, cheek-raised smiles produce high ctx_score as a false positive.
    is_joyful = smile_level > 0.20
    ctx_multiplier = (
        0.2 if is_concentrating else
        0.25 if is_joyful else    # squinting from smiling, not stress
        0.5 if is_sad else
        1.0
    )
    for name, w in brow_squint.items():
        ctx_total += get(name, 0.0) * w * ctx_multiplier
        ctx_wsum += w
    ctx_score = ctx_total / ctx_wsum

    # ── Fear group — wide eyes, jaw drop, brow raise ──────────────────────
    smile_discount = max(0.0, 1.0 - positive_signal * 2.0)
    is_laughing = (get("jawOpen", 0.0) > 0.3 and get("browInnerUp", 0.0) > 0.5
                   and smile_level > 0.15)
    brow_inner_up_weight = 0.3 if is_laughing else 1.3

    fear_total = 0.0
    fear_wsum = 0.0
    fear_signals = {
        "browInnerUp": brow_inner_up_weight,
        "eyeWideLeft": 1.5, "eyeWideRight": 1.5,
        "jawOpen": 0.6,
    }
    for name, w in fear_signals.items():
        fear_total += get(name, 0.0) * w * smile_discount
        fear_wsum += w
    fear_score = fear_total / fear_wsum

    # ── Max-weighted blend — highest-active group dominates ───────────────
    scores = sorted([stress_score, ctx_score, fear_score])   # ascending
    blend = 0.50 * scores[2] + 0.30 * scores[1] + 0.20 * scores[0]

    raw = blend * global_scale - positive_signal * 0.15
    return max(0.0, min(1.0, raw))


def compute_tension_v5(blendshapes_dict, global_scale=1.8, joy_thresh=0.20,
                       fix_laughing=True, harder_smile_gate=False,
                       remove_eye_squint=False, joy_jaw_suppress=False,
                       smile_penalty=0.15):
    """Parametric v5 tension formula — per-frame version of _recompute_v5_batch.

    Flags mirror compare_ground_truth_mp._recompute_v5_batch exactly:
      fix_laughing      — is_laughing requires smile > 0.15 (fixes scream misclassification)
      harder_smile_gate — fear sd gate = max(0, 1-smile*4); zeroes at smile>0.25 vs v4's >0.50
      remove_eye_squint — drop eyeSquint from ctx group (Duchenne marker, fires during joy)
      joy_jaw_suppress  — jawOpen weight → 0 when smiling (laugh jaw ≠ fear jaw)
      smile_penalty     — coefficient for final `- smile * c` (default 0.15)
    """
    get = blendshapes_dict.get

    smile_level = max(get("mouthSmileLeft", 0.0), get("mouthSmileRight", 0.0))
    cheek_squint = max(get("cheekSquintLeft", 0.0), get("cheekSquintRight", 0.0))
    positive_signal = max(smile_level, cheek_squint * 0.8)

    frown_level = max(get("mouthFrownLeft", 0.0), get("mouthFrownRight", 0.0))
    press_level = max(get("mouthPressLeft", 0.0), get("mouthPressRight", 0.0))
    brow_down = max(get("browDownLeft", 0.0), get("browDownRight", 0.0))

    is_concentrating = brow_down > 0.2 and press_level < 0.05
    is_sad = frown_level > press_level and frown_level > 0.1
    is_joyful = smile_level > joy_thresh if joy_thresh is not None else False

    # stress group — unchanged from v4
    stress_total, stress_wsum = 0.0, 0.0
    for name, w in (("mouthPressLeft", 1.5), ("mouthPressRight", 1.5),
                    ("noseSneerLeft", 0.8), ("noseSneerRight", 0.8),
                    ("mouthUpperUpLeft", 0.5), ("mouthUpperUpRight", 0.5),
                    ("cheekPuff", 0.3)):
        stress_total += get(name, 0.0) * w
        stress_wsum += w
    stress_score = stress_total / stress_wsum

    # ctx group — optionally drop eyeSquint
    ctx_multiplier = (0.2 if is_concentrating else
                      0.25 if is_joyful else
                      0.5 if is_sad else 1.0)
    ctx_signals = (
        [("browDownLeft", 0.7), ("browDownRight", 0.7),
         ("mouthFrownLeft", 0.4), ("mouthFrownRight", 0.4)]
        if remove_eye_squint else
        [("browDownLeft", 0.7), ("browDownRight", 0.7),
         ("eyeSquintLeft", 0.6), ("eyeSquintRight", 0.6),
         ("mouthFrownLeft", 0.4), ("mouthFrownRight", 0.4)]
    )
    ctx_total = sum(get(n, 0.0) * w * ctx_multiplier for n, w in ctx_signals)
    ctx_wsum  = sum(w for _, w in ctx_signals)
    ctx_score = ctx_total / ctx_wsum

    # fear group
    if harder_smile_gate:
        smile_discount = max(0.0, 1.0 - smile_level * 4.0)
    else:
        smile_discount = max(0.0, 1.0 - positive_signal * 2.0)

    if fix_laughing:
        is_laughing = (get("jawOpen", 0.0) > 0.3 and get("browInnerUp", 0.0) > 0.5
                       and smile_level > 0.15)
    else:
        is_laughing = (get("jawOpen", 0.0) > 0.3 and get("browInnerUp", 0.0) > 0.5)
    biu_w = 0.3 if is_laughing else 1.3

    jaw_w = 0.0 if (joy_jaw_suppress and is_joyful) else 0.6
    fear_wsum = biu_w + 1.5 + 1.5 + jaw_w
    fear_total = (get("browInnerUp", 0.0) * biu_w
                  + get("eyeWideLeft", 0.0) * 1.5
                  + get("eyeWideRight", 0.0) * 1.5
                  + get("jawOpen", 0.0) * jaw_w) * smile_discount
    fear_score = fear_total / fear_wsum if fear_wsum > 0 else 0.0

    scores = sorted([stress_score, ctx_score, fear_score])
    blend  = 0.50 * scores[2] + 0.30 * scores[1] + 0.20 * scores[0]
    raw    = blend * global_scale - smile_level * smile_penalty
    return max(0.0, min(1.0, raw))


def compute_tension_v5a(blendshapes_dict):
    """v5a: fix is_laughing only. Targets screaming fear underdetection."""
    return compute_tension_v5(blendshapes_dict, fix_laughing=True)


def compute_tension_v5b(blendshapes_dict):
    """v5b: fix laughing + harder smile gate. Targets joy-spike FPs."""
    return compute_tension_v5(blendshapes_dict, fix_laughing=True, harder_smile_gate=True)


def compute_tension_v5c(blendshapes_dict):
    """v5c: fix laughing + remove eyeSquint from ctx (Duchenne marker)."""
    return compute_tension_v5(blendshapes_dict, fix_laughing=True, remove_eye_squint=True)


def compute_tension_v5d(blendshapes_dict):
    """v5d: fix laughing + jaw suppression when joyful."""
    return compute_tension_v5(blendshapes_dict, fix_laughing=True, joy_jaw_suppress=True)


def compute_tension_v5e(blendshapes_dict):
    """v5e: all v5 fixes combined + stronger smile penalty (0.35)."""
    return compute_tension_v5(blendshapes_dict,
                               fix_laughing=True, harder_smile_gate=True,
                               remove_eye_squint=True, joy_jaw_suppress=True,
                               smile_penalty=0.35)


def compute_face_valence(blendshapes_dict):
    """Compute overall face valence from blendshapes."""
    pos = (
        blendshapes_dict.get("mouthSmileLeft", 0.0)
        + blendshapes_dict.get("mouthSmileRight", 0.0)
        + blendshapes_dict.get("cheekSquintLeft", 0.0) * 0.5
        + blendshapes_dict.get("cheekSquintRight", 0.0) * 0.5
    ) / 3.0
    neg = (
        blendshapes_dict.get("mouthFrownLeft", 0.0)
        + blendshapes_dict.get("mouthFrownRight", 0.0)
        + blendshapes_dict.get("browDownLeft", 0.0)
        + blendshapes_dict.get("browDownRight", 0.0)
        + blendshapes_dict.get("noseSneerLeft", 0.0) * 0.5
        + blendshapes_dict.get("noseSneerRight", 0.0) * 0.5
    ) / 4.0
    return max(-1.0, min(1.0, (pos - neg) * 2.0))


def compute_au_velocities(bs_dict, prev_bs_dict, delta_t):
    """Compute per-frame velocity for fear-relevant AUs."""
    if prev_bs_dict is None or delta_t is None:
        return {au: 0.0 for au in VELOCITY_AUS}, 0.0
    dt = max(delta_t, 1.0 / 120.0)
    velocities = {}
    for au in VELOCITY_AUS:
        v = (bs_dict.get(au, 0.0) - prev_bs_dict.get(au, 0.0)) / dt
        velocities[au] = max(0.0, v)
    startle_score = max(velocities.values()) if velocities else 0.0
    return velocities, startle_score


def compute_ctx_tag(bs_dict, smile_level):
    """Compute MediaPipe context tag from blendshapes."""
    frown_lvl = max(bs_dict.get("mouthFrownLeft", 0), bs_dict.get("mouthFrownRight", 0))
    press_lvl = max(bs_dict.get("mouthPressLeft", 0), bs_dict.get("mouthPressRight", 0))
    brow_dn = max(bs_dict.get("browDownLeft", 0), bs_dict.get("browDownRight", 0))
    eye_wd = max(bs_dict.get("eyeWideLeft", 0), bs_dict.get("eyeWideRight", 0))

    if smile_level > 0.3:
        return "JOY"
    elif eye_wd > 0.3 and bs_dict.get("browInnerUp", 0) > 0.2:
        return "FEAR"
    elif brow_dn > 0.2 and press_lvl < 0.05:
        return "CONC"
    elif frown_lvl > press_lvl and frown_lvl > 0.1:
        return "SAD"
    elif press_lvl > 0.15:
        return "STRESS"
    else:
        return "---"
