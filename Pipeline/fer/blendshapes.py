"""Blendshape groups, velocity AUs, emotion labels, and HUD display constants.

Pattern: [Constants] — pure data, no logic, no imports.
"""

# ── Blendshape groups ─────────────────────────────────────────────────────────
NEGATIVE_BLENDSHAPES = [
    "browDownLeft", "browDownRight",
    "eyeWideLeft", "eyeWideRight",
    "eyeSquintLeft", "eyeSquintRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthPressLeft", "mouthPressRight",
    "noseSneerLeft", "noseSneerRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "cheekPuff",
]

POSITIVE_BLENDSHAPES = [
    "mouthSmileLeft", "mouthSmileRight",
    "cheekSquintLeft", "cheekSquintRight",
    "mouthDimpleLeft", "mouthDimpleRight",
]

AMBIGUOUS_BLENDSHAPES = ["browInnerUp", "jawOpen"]

STRESS_BLENDSHAPES = NEGATIVE_BLENDSHAPES + AMBIGUOUS_BLENDSHAPES + POSITIVE_BLENDSHAPES

ALL_BLENDSHAPE_NAMES = sorted([
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
])

# Key blendshapes for CSV (subset used in tension/valence/fear formulas)
KEY_BLENDSHAPES = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "eyeWideLeft", "eyeWideRight", "eyeSquintLeft", "eyeSquintRight",
    "mouthPressLeft", "mouthPressRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthSmileLeft", "mouthSmileRight", "jawOpen",
    "noseSneerLeft", "noseSneerRight",
    "cheekSquintLeft", "cheekSquintRight",
]

# Velocity tracking AUs (used for startle detection)
VELOCITY_AUS = [
    "eyeWideLeft", "eyeWideRight", "browInnerUp",
    "jawOpen", "mouthPressLeft", "mouthPressRight",
]
STARTLE_VELOCITY_THRESHOLD = 3.0

# HSEmotion emotion class labels (output order matches model)
EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]

# ── HUD display constants ─────────────────────────────────────────────────────
PANEL_WIDTH = 400
HUD_MIN_HEIGHT = 800   # extended to fit formulas panel (F0–F11) + raw tension row
PANEL_BG = (30, 30, 45)
SECTION_DIVIDER_COLOR = (80, 80, 100)

# Neutral frame gate (matches benchmark_explorer.py)
NEUTRAL_TENSION_MAX = 0.15
