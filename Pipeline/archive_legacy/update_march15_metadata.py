"""
Helper tool to identify and update March 15 session content types.
This is an interactive tool to document what was actually recorded.
"""

import os
import json
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSIONS_JSON = os.path.join(LOG_DIR, "sessions.json")

SESSIONS_TO_IDENTIFY = [
    {
        "csv": "mediapipe_20260315_224814.csv",
        "time": "22:48",
        "frames": 4524,
        "duration": "157.3s",
        "hint": "Negative valence, low fear/joy, high eye squint (stress)"
    },
    {
        "csv": "mediapipe_20260315_225845.csv",
        "time": "22:58",
        "frames": 18039,
        "duration": "607.1s",
        "hint": "Longest session, negative valence, moderate stress"
    },
    {
        "csv": "mediapipe_20260315_231250.csv",
        "time": "23:18",
        "frames": 9333,
        "duration": "320.7s",
        "hint": "Positive valence, elevated smile (0.10) and browInnerUp (0.12) - likely FUNNY"
    },
]

CONTENT_TYPES = ["jumpscare", "funny", "neutral", "mixed", "game", "posed", "other"]

def show_hint(session_idx):
    """Show emotion profile hint for a session."""
    session = SESSIONS_TO_IDENTIFY[session_idx]
    print(f"\n  Hint from emotion profile analysis:")
    print(f"    {session['hint']}")

def prompt_content_type(session_idx):
    """Prompt user for content type of a session."""
    session = SESSIONS_TO_IDENTIFY[session_idx]

    print(f"\n{'='*70}")
    print(f"Session {session_idx + 1}/3: {session['csv']}")
    print(f"{'='*70}")
    print(f"Time:      {session['time']}")
    print(f"Duration:  {session['duration']}")
    print(f"Frames:    {session['frames']}")

    show_hint(session_idx)

    print(f"\nWhat was the content type?")
    for i, ct in enumerate(CONTENT_TYPES, 1):
        print(f"  {i}. {ct}")

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(CONTENT_TYPES)}) [or 'skip']: ").strip().lower()
            if choice == 'skip':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(CONTENT_TYPES):
                return CONTENT_TYPES[idx]
            else:
                print(f"Invalid choice. Please enter 1-{len(CONTENT_TYPES)}")
        except ValueError:
            print("Please enter a number or 'skip'")

    return None

def load_sessions_json():
    """Load the sessions.json file."""
    if not os.path.exists(SESSIONS_JSON):
        print(f"Error: {SESSIONS_JSON} not found")
        return None

    try:
        with open(SESSIONS_JSON, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def find_march15_sessions(data):
    """Find March 15 sessions in the registry."""
    sessions = data.get("sessions", [])
    march15 = []

    for session in sessions:
        session_id = session.get("session_id", "")
        if "20260315" in session_id:
            march15.append(session)

    return march15

def update_sessions_json(data, updates):
    """Update sessions.json with new content types."""
    sessions = data.get("sessions", [])

    for session in sessions:
        session_id = session.get("session_id", "")
        if "20260315" in session_id:
            csv_path = session.get("csv_path", "")
            # Match by time or CSV path
            for csv_file, content_type in updates.items():
                if csv_file in csv_path:
                    session["content_type"] = content_type
                    print(f"Updated {session_id}: content_type = {content_type}")

    return data

def main():
    print("\n" + "="*70)
    print(" IDENTIFY MARCH 15 SESSION CONTENT TYPES")
    print("="*70)
    print(f"\nThis tool helps you document what was recorded in each session.")
    print(f"Use the emotion profile hints to identify content types.\n")

    # Load current sessions.json
    data = load_sessions_json()
    if not data:
        print("Cannot proceed without sessions.json")
        return

    # Check what March 15 sessions exist in registry
    march15_sessions = find_march15_sessions(data)
    print(f"Found {len(march15_sessions)} March 15 sessions in registry:")
    for s in march15_sessions:
        print(f"  - {s.get('session_id')}: {s.get('content_type', 'UNKNOWN')}")

    # Prompt for each session
    updates = {}
    for i, session_info in enumerate(SESSIONS_TO_IDENTIFY):
        content_type = prompt_content_type(i)
        if content_type:
            updates[session_info['csv']] = content_type

    # Show summary
    if updates:
        print(f"\n\n{'='*70}")
        print(" SUMMARY OF UPDATES")
        print(f"{'='*70}")

        for csv_file, content_type in updates.items():
            print(f"  {csv_file:<40} -> {content_type}")

        # Ask for confirmation
        confirm = input(f"\nSave these updates to sessions.json? (yes/no): ").strip().lower()

        if confirm in ['yes', 'y']:
            updated_data = update_sessions_json(data, updates)
            try:
                with open(SESSIONS_JSON, "w") as f:
                    json.dump(updated_data, f, indent=2, ensure_ascii=False)
                print(f"\nSuccessfully updated {SESSIONS_JSON}")
            except Exception as e:
                print(f"Error saving file: {e}")
        else:
            print("Updates not saved.")
    else:
        print("\nNo updates made.")

    print(f"\nNext: Run emotion_comparison_report.py again to get updated analysis.")

if __name__ == "__main__":
    main()
