"""Post-session analysis runner for the La Façade Fissuréе FER pipeline.

Pattern: [Utility] — stateless; called by test_mp_hs.py via session.finish().

Automatically triggers visualization and stats after a recording session ends:
  - Generates visualize_session.py plot for the completed CSV
  - Prints quick stats summary to terminal
  - Optionally runs emotion profile comparison

Usage:
    from utils.post_session_analysis import post_session_analysis
    post_session_analysis(csv_path, session_record)
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_visualization(csv_path, silent=False):
    """
    Run visualize_session.py on the given CSV.
    Returns True if successful, False otherwise.
    """
    try:
        if not silent:
            print(f"\n{'='*70}")
            print(" Generating visualization...")
            print(f"{'='*70}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        viz_script = os.path.join(script_dir, "..", "analysis", "visualize_session.py")
        python_exe = sys.executable

        result = subprocess.run(
            [python_exe, viz_script, csv_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            if not silent:
                print(f"Visualization failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Visualization timed out")
        return False
    except Exception as e:
        if not silent:
            print(f"Could not run visualization: {e}")
        return False

def run_emotion_comparison(_csv_path, silent=False):
    """Placeholder for emotion comparison report. Print reminder only."""
    if not silent:
        print("Run this later: emotion_comparison_report.py")
    return True

def load_session_summary(summary_path):
    """
    Load quick summary info from the _summary.txt file.
    Returns dict with key metrics.
    """
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            content = f.read()

        summary = {}
        for line in content.split("\n"):
            line = line.strip()
            if "Session date:" in line:
                summary["date"] = line.split(":")[-1].strip()
            elif "Session duration:" in line:
                summary["duration"] = line.split(":")[-1].strip()
            elif "Total frames processed:" in line:
                summary["frames"] = line.split(":")[-1].strip()
            elif "Average latency:" in line:
                summary["avg_latency"] = line.split(":")[-1].strip()
            elif "Median latency:" in line:
                summary["median_latency"] = line.split(":")[-1].strip()
            elif "Effective FPS:" in line:
                summary["fps"] = line.split(":")[-1].strip()


        return summary

    except Exception as e:
        return {}

def print_session_summary(summary_path, csv_path):
    """Print a nice summary of the session."""
    try:
        summary = load_session_summary(summary_path)

        print(f"\n{'='*70}")
        print(" SESSION SUMMARY")
        print(f"{'='*70}")

        for key, value in summary.items():
            print(f"{key.capitalize():<20}: {value}")

        plot_path = csv_path.replace(".csv", "_plot.png")
        if os.path.exists(plot_path):
            print(f"\nVisualization saved: {os.path.basename(plot_path)}")

        print(f"\nFull session log: {os.path.basename(summary_path)}")
        print(f"Data file:        {os.path.basename(csv_path)}")

    except Exception as e:
        print(f"Could not load summary: {e}")

def post_session_analysis(csv_path, summary_path, run_viz=True, run_emotion=False, silent=False):
    """
    Run all post-session analysis.

    Args:
        csv_path: Path to the CSV file
        summary_path: Path to the _summary.txt file
        run_viz: Whether to run visualization (default: True)
        run_emotion: Whether to run emotion comparison (default: False)
        silent: Whether to suppress detailed output (default: False)

    Returns:
        True if all requested analysis completed successfully
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return False

    success = True

    if os.path.exists(summary_path):
        print_session_summary(summary_path, csv_path)
    else:
        print(f"\nSession data saved: {os.path.basename(csv_path)}")

    if run_viz:
        if not run_visualization(csv_path, silent=silent):
            success = False

    if run_emotion:
        if not run_emotion_comparison(csv_path, silent=silent):
            success = False

    if not silent:
        print(f"\n{'='*70}")
        if success:
            print(" Analysis complete!")
        else:
            print(" Analysis completed with warnings (see above)")
        print(f"{'='*70}\n")

    return success


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python post_session_analysis.py <csv_path> [--no-viz] [--emotion]")
        sys.exit(1)

    csv_path = sys.argv[1]
    summary_path = csv_path.replace(".csv", "_summary.txt")

    run_viz = "--no-viz" not in sys.argv
    run_emotion = "--emotion" in sys.argv

    post_session_analysis(csv_path, summary_path, run_viz=run_viz, run_emotion=run_emotion)
