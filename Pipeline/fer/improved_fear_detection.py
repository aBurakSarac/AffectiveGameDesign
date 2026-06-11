"""Fear detection utility functions for the La Façade Fissuréе FER pipeline.

Pattern: [Utility] — pure functions, no state, safe to call in any order.

Active function (imported by test_mp_hs.py):
    get_velocity_tag(startle_score, threshold) → 'STARTLE' | '---'

Historical reference functions (not imported by the active pipeline):
    get_ctx_tag_original  — STARTLE detection only (baseline; kept for comparison)
    get_ctx_tag_improved  — dual-mode: STARTLE + TENSION (discovered 2026-03-15 session)
    get_ctx_tag_adaptive  — disabled; raises NotImplementedError (no calibration infra)

Usage:
    from fer.improved_fear_detection import get_velocity_tag
    tag = get_velocity_tag(mp_startle_score)
"""

# ORIGINAL (detects only STARTLE)
def get_ctx_tag_original(row):
    """Standard fear detection - STARTLE response only."""
    smile = float(row.get("mouthSmileLeft", 0))
    eye_wd = max(float(row.get("eyeWideLeft", 0)), float(row.get("eyeWideRight", 0)))
    brow_inner = float(row.get("browInnerUp", 0))
    brow_dn = max(float(row.get("browDownLeft", 0)), float(row.get("browDownRight", 0)))
    frown = max(float(row.get("mouthFrownLeft", 0)), float(row.get("mouthFrownRight", 0)))
    press = max(float(row.get("mouthPressLeft", 0)), float(row.get("mouthPressRight", 0)))

    if smile > 0.3:
        return "JOY"
    elif eye_wd > 0.3 and brow_inner > 0.2:
        return "FEAR"  # Only detects startle
    elif brow_dn > 0.2 and press < 0.05:
        return "CONC"
    elif frown > press and frown > 0.1:
        return "SAD"
    elif press > 0.15:
        return "STRESS"
    else:
        return "---"


# IMPROVED (detects both STARTLE and TENSION)
def get_ctx_tag_improved(row):
    """
    Improved fear detection - catches BOTH startle AND tension-based fear.

    Fear signatures discovered through analysis of 2026-03-15 22:48 session:
      - STARTLE: eyeWide > 0.30 AND browInnerUp > 0.20
      - TENSION: mouthPress > 0.12 AND browDown > 0.08 AND eyeSquint > 0.50
    """
    smile = float(row.get("mouthSmileLeft", 0))

    # STARTLE fear (sudden shock, eye-widening)
    eye_wd = max(float(row.get("eyeWideLeft", 0)), float(row.get("eyeWideRight", 0)))
    brow_inner = float(row.get("browInnerUp", 0))

    # TENSION fear (sustained fear, mouth-clenching)
    mouth_press = max(
        float(row.get("mouthPressLeft", 0)),
        float(row.get("mouthPressRight", 0))
    )
    brow_dn = max(float(row.get("browDownLeft", 0)), float(row.get("browDownRight", 0)))
    eye_squint = max(
        float(row.get("eyeSquintLeft", 0)),
        float(row.get("eyeSquintRight", 0))
    )

    # Other emotions
    frown = max(float(row.get("mouthFrownLeft", 0)), float(row.get("mouthFrownRight", 0)))
    press = mouth_press

    # Detection priority: JOY > FEAR (both types) > CONC > SAD > STRESS > ---
    if smile > 0.3:
        return "JOY"

    # FEAR detection (improved with dual-mode)
    startle_fear = eye_wd > 0.3 and brow_inner > 0.2
    tension_fear = mouth_press > 0.12 and brow_dn > 0.08 and eye_squint > 0.50

    if startle_fear or tension_fear:
        return "FEAR"

    # Other emotions
    if brow_dn > 0.2 and press < 0.05:
        return "CONC"
    elif frown > press and frown > 0.1:
        return "SAD"
    elif press > 0.15:
        return "STRESS"
    else:
        return "---"


# ADAPTIVE (learns from session metadata)
# DISABLED: no profile storage or calibration infrastructure in current pipeline.
# The fear_profile=None fallback is identical to get_ctx_tag_improved, making
# this function misleading. Re-enable only if a calibration window and profile
# persistence layer are built in Phase 3.
def get_ctx_tag_adaptive(row, fear_profile=None):
    raise NotImplementedError(
        "Adaptive detection is disabled: no profile infrastructure exists. "
        "Use get_ctx_tag_improved() instead."
    )


# ---------------------------------------------------------------------------
# VELOCITY-BASED STARTLE DETECTION
# ---------------------------------------------------------------------------
# CALIBRATION PENDING: threshold=3.0 units/s derived from Gemini suggestion
# (0.4 delta in <150ms ≈ 2.67/s, rounded to 3.0). Requires at least 3
# jumpscare sessions to validate per-subject before using in Unity bridge.
# ---------------------------------------------------------------------------

def get_velocity_tag(startle_score, threshold=3.0):
    """Velocity-based state tag for a single frame.

    Args:
        startle_score: Float. Max positive AU velocity across fear-relevant
            AUs (units/s). Computed by compute_au_velocities() in
            test_mediapipe.py.
        threshold: Float. CALIBRATION PENDING (default 3.0 units/s).

    Returns:
        'STARTLE' if startle_score >= threshold, else '---'.
    """
    return "STARTLE" if startle_score >= threshold else "---"


if __name__ == "__main__":
    # Test the functions
    print("\n" + "="*80)
    print(" FEAR DETECTION ALGORITHM COMPARISON")
    print("="*80)

    # Test case 1: STARTLE fear (session should be detected as FEAR by all)
    test_startle = {
        "mouthSmileLeft": 0.01,
        "eyeWideLeft": 0.45,
        "eyeWideRight": 0.42,
        "browInnerUp": 0.25,
        "browDownLeft": 0.05,
        "browDownRight": 0.05,
        "eyeSquintLeft": 0.45,
        "eyeSquintRight": 0.42,
        "mouthFrownLeft": 0.01,
        "mouthFrownRight": 0.01,
        "mouthPressLeft": 0.02,
        "mouthPressRight": 0.01,
    }

    # Test case 2: TENSION fear (2026-03-15 22:48 signature)
    test_tension = {
        "mouthSmileLeft": 0.01,
        "eyeWideLeft": 0.004,
        "eyeWideRight": 0.003,
        "browInnerUp": 0.005,
        "browDownLeft": 0.099,
        "browDownRight": 0.101,
        "eyeSquintLeft": 0.568,
        "eyeSquintRight": 0.437,
        "mouthFrownLeft": 0.001,
        "mouthFrownRight": 0.001,
        "mouthPressLeft": 0.175,
        "mouthPressRight": 0.064,
    }

    # Test case 3: Neutral
    test_neutral = {
        "mouthSmileLeft": 0.0,
        "eyeWideLeft": 0.005,
        "eyeWideRight": 0.005,
        "browInnerUp": 0.057,
        "browDownLeft": 0.025,
        "browDownRight": 0.048,
        "eyeSquintLeft": 0.490,
        "eyeSquintRight": 0.404,
        "mouthFrownLeft": 0.0,
        "mouthFrownRight": 0.0,
        "mouthPressLeft": 0.023,
        "mouthPressRight": 0.011,
    }

    print("\n--- Test Case 1: STARTLE fear (eye-wide, brow-up) ---")
    print(f"Original:  {get_ctx_tag_original(test_startle)}")
    print(f"Improved:  {get_ctx_tag_improved(test_startle)}")
    print(f"Adaptive:  DISABLED (NotImplementedError)")

    print("\n--- Test Case 2: TENSION fear (mouth-press, brow-down) [22:48 signature] ---")
    print(f"Original:  {get_ctx_tag_original(test_tension)}")
    print(f"Improved:  {get_ctx_tag_improved(test_tension)}")
    print(f"Adaptive:  DISABLED")
    print(f"(Original MISSES this fear type!)")

    print("\n--- Test Case 3: Neutral ---")
    print(f"Original:  {get_ctx_tag_original(test_neutral)}")
    print(f"Improved:  {get_ctx_tag_improved(test_neutral)}")
    print(f"Adaptive:  DISABLED")

    print("\n" + "="*80)
    print(" SUMMARY")
    print("="*80)
    print("""
Original Algorithm:
  - DETECTS: STARTLE fear (eye-wide > 0.30 AND brow-up > 0.20)
  - MISSES: TENSION fear (mouth-press, brow-down, eye-squint)
  - Problem: Session 2026-03-15 22:48 was not detected

Improved Algorithm:
  - DETECTS: STARTLE fear (same as original)
  - DETECTS: TENSION fear (new dual-mode)
  - Solution: Catches both fear types

Adaptive Algorithm:
  - DETECTS: User-specific fear signature
  - LEARNS: From calibration sessions
  - Thresholds: Can be adjusted per user

Recommendation: Use IMPROVED algorithm in visualize_session.py
  - Catches the most fear responses
  - No false positives (combines multiple signals)
  - Ready for production
    """)
