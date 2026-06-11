"""
Generate focused emotion comparison report from CSV data.
Focus on fear-relevant features across sessions.
"""

import os
import pandas as pd
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Key features for emotion analysis
FEAR_FEATURES = ['eyeWideLeft', 'eyeWideRight', 'browInnerUp', 'jawOpen', 'mouthFunnel']
JOY_FEATURES = ['mouthSmileLeft', 'mouthSmileRight', 'cheekSquintLeft', 'cheekSquintRight']
STRESS_FEATURES = ['browDownLeft', 'browDownRight', 'eyeSquintLeft', 'eyeSquintRight']

def analyze_emotion_session(csv_file, content_type="Unknown"):
    """Analyze emotional features from a CSV file."""
    csv_path = os.path.join(LOG_DIR, csv_file)

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)

        results = {
            'file': csv_file,
            'content': content_type,
            'frames': len(df),
            'duration': df['timestamp'].max() - df['timestamp'].min() if 'timestamp' in df.columns else 0,
        }

        # Valence (positive/negative axis)
        if 'face_valence' in df.columns:
            results['valence_mean'] = df['face_valence'].mean()

        # Fear features
        fear_activation = {}
        for feat in FEAR_FEATURES:
            if feat in df.columns:
                fear_activation[feat] = df[feat].mean()
        results['fear_activation'] = fear_activation

        # Joy features
        joy_activation = {}
        for feat in JOY_FEATURES:
            if feat in df.columns:
                joy_activation[feat] = df[feat].mean()
        results['joy_activation'] = joy_activation

        # Stress features
        stress_activation = {}
        for feat in STRESS_FEATURES:
            if feat in df.columns:
                stress_activation[feat] = df[feat].mean()
        results['stress_activation'] = stress_activation

        return results
    except Exception as e:
        print(f"Error analyzing {csv_file}: {e}")
        return None

def print_emotion_profile(results):
    """Print emotion profile for a session."""
    if not results:
        return

    print(f"\n{'-'*70}")
    print(f"{results['file']}")
    print(f"Content: {results['content']}")
    print(f"Duration: {results['duration']:.1f}s | Frames: {results['frames']}")
    print(f"{'-'*70}")

    if 'valence_mean' in results:
        val = results['valence_mean']
        bar = '▌' * int(abs(val) * 20) if val >= 0 else '▌' * int(abs(val) * 20)
        side = 'Positive' if val > 0 else 'Negative' if val < 0 else 'Neutral'
        print(f"Valence: {val:.4f} ({side})")

    print(f"\nFear activation (fear response indicators):")
    for feat, val in results['fear_activation'].items():
        print(f"  {feat:<15}: {val:.4f}")

    print(f"\nJoy activation (smile, laugh):")
    for feat, val in results['joy_activation'].items():
        print(f"  {feat:<15}: {val:.4f}")

    print(f"\nStress activation (tension, concentration):")
    for feat, val in results['stress_activation'].items():
        print(f"  {feat:<15}: {val:.4f}")

def main():
    print("\n" + "="*70)
    print(" EMOTION PROFILE COMPARISON: MARCH 15 vs MARCH 16")
    print("="*70)

    # March 15 sessions
    sessions_15 = [
        ("mediapipe_20260315_224814.csv", "Unknown"),
        ("mediapipe_20260315_225845.csv", "Unknown"),
        ("mediapipe_20260315_231250.csv", "Unknown"),
    ]

    # March 16 sessions
    sessions_16 = [
        ("mediapipe_20260316_084912_lying_variant.csv", "Neutral (lying)"),
        ("mediapipe_20260316_090205.csv", "Neutral"),
        ("mediapipe_20260316_091539.csv", "Funny"),
    ]

    print("\nMARCH 15 SESSIONS (yesterday)")
    results_15 = []
    for csv_file, content in sessions_15:
        result = analyze_emotion_session(csv_file, content)
        if result:
            results_15.append(result)
            print_emotion_profile(result)

    print("\n\nMARCH 16 SESSIONS (today)")
    results_16 = []
    for csv_file, content in sessions_16:
        result = analyze_emotion_session(csv_file, content)
        if result:
            results_16.append(result)
            print_emotion_profile(result)

    # Comparison summary
    print("\n\n" + "="*70)
    print(" KEY FINDINGS")
    print("="*70)

    if results_16:
        neutral = [r for r in results_16 if 'Neutral' in r['content']]
        funny = [r for r in results_16 if 'funny' in r['content'].lower()]

        if neutral:
            print(f"\nNEUTRAL baseline (n={len(neutral)} sessions):")
            avg_fear = np.mean([np.mean(list(r['fear_activation'].values())) for r in neutral])
            avg_joy = np.mean([np.mean(list(r['joy_activation'].values())) for r in neutral])
            avg_stress = np.mean([np.mean(list(r['stress_activation'].values())) for r in neutral])
            print(f"  Avg Fear activation:   {avg_fear:.4f}")
            print(f"  Avg Joy activation:    {avg_joy:.4f}")
            print(f"  Avg Stress activation: {avg_stress:.4f}")

        if funny:
            print(f"\nFUNNY baseline (n={len(funny)} sessions):")
            avg_fear = np.mean([np.mean(list(r['fear_activation'].values())) for r in funny])
            avg_joy = np.mean([np.mean(list(r['joy_activation'].values())) for r in funny])
            avg_stress = np.mean([np.mean(list(r['stress_activation'].values())) for r in funny])
            print(f"  Avg Fear activation:   {avg_fear:.4f}")
            print(f"  Avg Joy activation:    {avg_joy:.4f}")
            print(f"  Avg Stress activation: {avg_stress:.4f}")

    print(f"\n\nMARCH 15 sessions (content types need verification):")
    print(f"  Check sessions.json or video history for:")
    for i, (csv_file, _) in enumerate(sessions_15, 1):
        print(f"    {i}. {csv_file}")

    print(f"\n\nTO DO THIS MORNING:")
    print(f"  1. Identify what was recorded in March 15 sessions")
    print(f"     - Check if any were jumpscare (Fear emotion)")
    print(f"     - Update sessions.json with content_type")
    print(f"  2. Use this comparison to understand fear patterns")
    print(f"     - Compare fear features to neutral/funny baselines")
    print(f"  3. Plan evening Fear session protocol")
    print(f"     - Use jumpscare videos you found effective before")
    print(f"     - Expected fear features: eyeWide, browInnerUp, jawOpen high")
    print(f"     - Should differ significantly from neutral baseline")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()
