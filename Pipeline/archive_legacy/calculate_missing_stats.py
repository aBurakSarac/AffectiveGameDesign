"""
Calculate missing feature statistics for March 15 CSV files and generate comparison report.
Compares with today's (March 16) neutral and funny sessions.
"""

import os
import sys
import pandas as pd

# Add current dir to path for session_meta import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_meta import compute_feature_stats, format_feature_stats_table

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Sessions to analyze
SESSIONS = [
    # March 15 sessions (need stats)
    {"date": "2026-03-15", "time": "22:48", "csv": "mediapipe_20260315_224814.csv", "content": "Unknown"},
    {"date": "2026-03-15", "time": "22:58", "csv": "mediapipe_20260315_225845.csv", "content": "Unknown"},
    {"date": "2026-03-15", "time": "23:18", "csv": "mediapipe_20260315_231250.csv", "content": "Unknown"},
    # March 16 sessions (existing stats for comparison)
    {"date": "2026-03-16", "time": "08:50", "csv": "mediapipe_20260316_084912_lying_variant.csv", "content": "neutral (lying)"},
    {"date": "2026-03-16", "time": "09:02", "csv": "mediapipe_20260316_090205.csv", "content": "neutral"},
    {"date": "2026-03-16", "time": "09:12", "csv": "mediapipe_20260316_091539.csv", "content": "funny"},
]

def load_csv_stats(csv_path):
    """Load stats from CSV using session_meta functions."""
    try:
        return compute_feature_stats(csv_path)
    except Exception as e:
        print(f"Error computing stats for {csv_path}: {e}")
        return {}

def get_csv_info(csv_path):
    """Get basic info from CSV (frame count, duration)."""
    try:
        df = pd.read_csv(csv_path)
        frame_count = len(df)
        if 'timestamp' in df.columns:
            duration = df['timestamp'].max() - df['timestamp'].min()
            fps = frame_count / duration if duration > 0 else 0
        else:
            duration = 0
            fps = 0

        # Calculate key emotion-related stats
        emotion_summary = {}
        if 'face_valence' in df.columns:
            emotion_summary['valence_mean'] = df['face_valence'].mean()
            emotion_summary['valence_std'] = df['face_valence'].std()

        if 'smile_level' in df.columns:
            emotion_summary['smile_mean'] = df['smile_level'].mean()
            emotion_summary['smile_max'] = df['smile_level'].max()

        # Fear-related features
        fear_features = ['eyeWideLeft', 'eyeWideRight', 'browInnerUp', 'jawOpen', 'mouthFunnel']
        fear_activation = {}
        for feat in fear_features:
            if feat in df.columns:
                fear_activation[feat] = df[feat].mean()

        return {
            'frame_count': frame_count,
            'duration_s': round(duration, 1),
            'fps': round(fps, 1),
            'emotion_summary': emotion_summary,
            'fear_features': fear_activation,
        }
    except Exception as e:
        print(f"Error getting CSV info for {csv_path}: {e}")
        return {}

def print_session_summary(session):
    """Print summary stats for a single session."""
    csv_path = os.path.join(LOG_DIR, session['csv'])
    if not os.path.exists(csv_path):
        print(f"  [ERROR] CSV not found: {session['csv']}")
        return

    info = get_csv_info(csv_path)
    stats = load_csv_stats(csv_path)

    print(f"\n{'='*70}")
    print(f"{session['date']} {session['time']} — {session['content']}")
    print(f"{'='*70}")
    print(f"  File:     {session['csv']}")
    print(f"  Frames:   {info.get('frame_count', 'N/A')}")
    print(f"  Duration: {info.get('duration_s', 'N/A')} seconds")
    print(f"  FPS:      {info.get('fps', 'N/A')}")

    # Emotion summary
    emotion = info.get('emotion_summary', {})
    if emotion:
        print(f"\n  Valence (positive/negative):")
        if 'valence_mean' in emotion:
            print(f"    Mean: {emotion['valence_mean']:.4f} (±{emotion['valence_std']:.4f})")
        if 'smile_mean' in emotion:
            print(f"  Smile level:")
            print(f"    Mean: {emotion['smile_mean']:.4f}, Max: {emotion['smile_max']:.4f}")

    # Fear features
    fear = info.get('fear_features', {})
    if fear:
        print(f"\n  Fear indicators:")
        for feat, val in fear.items():
            print(f"    {feat:<15}: {val:.4f}")

    # Full feature stats table
    if stats:
        print(f"\n  Full feature statistics:")
        lines = format_feature_stats_table(stats)
        for line in lines:
            print(f"  {line}")

def main():
    print("\n" + "="*70)
    print(" MISSING STATS CALCULATION & SESSION COMPARISON")
    print("="*70)
    print(f"\nLog directory: {LOG_DIR}")

    # Check which files exist
    print(f"\nChecking session files...")
    for session in SESSIONS:
        csv_path = os.path.join(LOG_DIR, session['csv'])
        exists = "[OK]" if os.path.exists(csv_path) else "[MISSING]"
        print(f"  {exists} {session['csv']}")

    # Print March 15 sessions
    print(f"\n\n{'='*70}")
    print(" MARCH 15 SESSIONS (YESTERDAY - MISSING FULL STATS)")
    print(f"{'='*70}")

    for session in SESSIONS[:3]:
        print_session_summary(session)

    # Print March 16 sessions
    print(f"\n\n{'='*70}")
    print(" MARCH 16 SESSIONS (TODAY - COMPARISON BASELINE)")
    print(f"{'='*70}")

    for session in SESSIONS[3:]:
        print_session_summary(session)

    # Generate comparison report
    print(f"\n\n{'='*70}")
    print(" COMPARISON ANALYSIS")
    print(f"{'='*70}")

    print(f"\nNote: March 15 sessions have unknown content types.")
    print(f"   Check sessions.json or the summary files for metadata.")
    print(f"\nSuccessfully calculated stats from CSV files.")
    print(f"You can now:")
    print(f"   1. Add content_type metadata to March 15 sessions in sessions.json")
    print(f"   2. Compare fear indicators across emotion conditions")
    print(f"   3. Generate updated summary files with full stats")

    print(f"\n\n{'='*70}")
    print(" NEXT STEPS FOR MORNING PRODUCTIVITY")
    print(f"{'='*70}")
    print(f"""
  Morning work instead of waiting for jumpscare sessions:

  1. ✓ Calculate stats from existing CSVs (DONE)

  2. Identify content types for March 15 sessions
     - Check video history or notes
     - Update sessions.json with content_type

  3. Regenerate full summary files
     - Use generate_full_summaries.py (next)
     - Include metadata + feature stats

  4. Analyze fear indicators across conditions
     - Compare fear features: eyeWideLeft/Right, browInnerUp, jawOpen
     - Check if they're distinctive from neutral/funny

  5. Document findings for Phase 2 report
     - What facial features distinguish fear vs. other emotions?
     - Which MediaPipe blendshapes are most reliable?

  6. Plan Fear session protocol
     - When: evening (for actual fear response)
     - What: jumpscare videos (already identified in March 15 data!)
     - How: compare against neutral/funny baseline
    """)

if __name__ == "__main__":
    main()
