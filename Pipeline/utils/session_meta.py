"""Session metadata, hardware monitoring, and session registry for the FER pipeline.

Pattern: [Observer target] — Session instances are consumed by test_mp_hs.py and
    post_session_analysis.py as a shared state container for a single recording run.

Provides:
  - Pre-session terminal questionnaire (subject, content, lighting, notes)
  - Hardware auto-detection (CPU, RAM, GPU)
  - Background resource monitoring (CPU%, RAM MB via psutil) on a daemon thread
  - Session registry (appends to logs/sessions.json)
  - Per-column descriptive statistics for the session CSV

Usage:
    from utils.session_meta import Session
    session = Session("mp_hs", log_dir)
    session.prompt()
    session.start()
    # ... frame loop ...
    record = session.finish(latencies, frame_count, csv_path, summary_path)
"""

import os
import json
import time
import platform
import subprocess
import threading
from datetime import datetime

import numpy as np
import psutil
import pandas as pd


# ---- Default prompt values ----
DEFAULT_SUBJECT = "S01"
DEFAULT_CONTENT = "mixed"
DEFAULT_LIGHTING = "bright"

KNOWN_CONTENT_TYPES = {"jumpscare", "funny", "neutral", "mixed", "game", "posed", "stealth", "calibration"}
KNOWN_STIMULUS_TYPES = {"jumpscare", "stealth", "neutral", "mixed", "calibration", "custom"}
KNOWN_LIGHTING = {"bright", "dim", "dark", "monitor-only"}


class _ResourceMonitor(threading.Thread):
    """Background daemon thread sampling CPU% and process RAM every 0.5s."""

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._process = psutil.Process()
        # Latest values (read by main thread via log_frame)
        self.cpu_percent = 0.0
        self.ram_mb = 0.0
        self.vram_mb = 0.0
        # History for summary stats
        self.cpu_samples = []
        self.ram_samples = []
        self.vram_samples = []

    @staticmethod
    def _sample_vram_mb():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return 0.0

    def run(self):
        # Prime the first cpu_percent call (returns 0.0 on first invocation)
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            self.cpu_percent = psutil.cpu_percent(interval=None)
            self.ram_mb = self._process.memory_info().rss / (1024 * 1024)
            self.vram_mb = self._sample_vram_mb()
            self.cpu_samples.append(self.cpu_percent)
            self.ram_samples.append(self.ram_mb)
            self.vram_samples.append(self.vram_mb)
            self._stop_event.wait(0.5)

    def stop(self):
        self._stop_event.set()

    def get_summary(self):
        if not self.cpu_samples:
            return {
                "cpu_avg": 0, "cpu_peak": 0,
                "ram_avg_mb": 0, "ram_peak_mb": 0,
                "vram_avg_mb": 0, "vram_peak_mb": 0,
            }
        return {
            "cpu_avg": round(float(np.mean(self.cpu_samples)), 1),
            "cpu_peak": round(float(np.max(self.cpu_samples)), 1),
            "ram_avg_mb": round(float(np.mean(self.ram_samples))),
            "ram_peak_mb": round(float(np.max(self.ram_samples))),
            "vram_avg_mb": round(float(np.mean(self.vram_samples))),
            "vram_peak_mb": round(float(np.max(self.vram_samples))),
        }


def _detect_hardware():
    """Detect CPU, RAM, and GPU. Called once per session."""
    hw = {
        "cpu": platform.processor() or "unknown",
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "gpu": "unknown",
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            hw["gpu"] = result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return hw


def compute_feature_stats(csv_path):
    """Compute full descriptive statistics for every numeric column in a CSV.

    Returns a dict of {column_name: {min, max, mean, median, std, p5, p25, p75, p95, range, non_zero_pct}}.
    """
    df = pd.read_csv(csv_path)
    # Keep only numeric columns, drop 'frame' (just an index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = {"frame"}
    numeric_cols = [c for c in numeric_cols if c not in skip]

    stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        stats[col] = {
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "p5": round(float(np.percentile(s, 5)), 4),
            "p25": round(float(np.percentile(s, 25)), 4),
            "p75": round(float(np.percentile(s, 75)), 4),
            "p95": round(float(np.percentile(s, 95)), 4),
            "range": round(float(s.max() - s.min()), 4),
            "non_zero_pct": round(float((s.abs() > 0.001).sum() / len(s) * 100), 1),
        }
    return stats


def format_feature_stats_table(feature_stats):
    """Format feature_stats dict as a human-readable fixed-width table (list of strings)."""
    lines = [
        "",
        "-" * 120,
        "FEATURE STATISTICS",
        "-" * 120,
        f"{'Feature':<24s} {'min':>7s} {'max':>7s} {'mean':>7s} {'median':>7s} "
        f"{'std':>7s} {'p5':>7s} {'p25':>7s} {'p75':>7s} {'p95':>7s} "
        f"{'range':>7s} {'nz%':>6s}",
    ]
    for col, st in feature_stats.items():
        lines.append(
            f"{col:<24s} {st['min']:7.4f} {st['max']:7.4f} {st['mean']:7.4f} "
            f"{st['median']:7.4f} {st['std']:7.4f} {st['p5']:7.4f} {st['p25']:7.4f} "
            f"{st['p75']:7.4f} {st['p95']:7.4f} {st['range']:7.4f} {st['non_zero_pct']:5.1f}%"
        )
    lines.append("-" * 120)
    return lines


def _detect_tool_from_cols(columns):
    """Detect FER tool from CSV column names (mirrors visualize_session.detect_tool)."""
    cols = set(columns)
    if "tension" in cols and "face_valence" in cols:
        return "mediapipe"
    if "valence" in cols and "arousal" in cols and "contempt" in cols:
        return "hsemotion"
    if "arousal" in cols and "angry" in cols:
        return "deepface"
    if "arousal" in cols and "anger" in cols:
        return "pyfeat"
    return "unknown"


def _find_peaks(timestamps, values, top_n=5, min_gap_s=1.0):
    """Return up to top_n (timestamp, value, array_idx) tuples for maxima at least min_gap_s apart."""
    if len(values) == 0:
        return []
    sorted_idx = np.argsort(values)[::-1]
    peaks = []
    for idx in sorted_idx:
        t = float(timestamps[idx])
        v = float(values[idx])
        if all(abs(t - p[0]) >= min_gap_s for p in peaks):
            peaks.append((t, v, int(idx)))
        if len(peaks) >= top_n:
            break
    peaks.sort(key=lambda x: x[0])
    return peaks


def _compute_fingerprint(df, tool):
    """Return SESSION FINGERPRINT lines (list of strings) computed from a session CSV DataFrame."""
    lines = ["", "SESSION FINGERPRINT"]
    if tool == "mediapipe":
        if "ctx_tag" in df.columns:
            ctx_counts = df["ctx_tag"].value_counts()
            if len(ctx_counts) > 0:
                dom = ctx_counts.index[0]
                pct = ctx_counts.iloc[0] / len(df) * 100
                lines.append(f"  Dominant state:       {dom} ({pct:.1f}%)")
        if "tension" in df.columns:
            mask = df["tension"].notna()
            vals = df.loc[mask, "tension"].values
            tss = df.loc[mask, "timestamp"].values if "timestamp" in df.columns else np.arange(mask.sum(), dtype=float)
            if len(vals) > 0:
                pk = int(np.argmax(vals))
                lines.append(f"  Peak tension:         {vals[pk]:.2f} at t={float(tss[pk]):.2f}s")
                lines.append(f"  Tension range:        {vals.min():.2f} \u2013 {vals.max():.2f}")
                lines.append(f"  Tension std:          {vals.std():.3f}")
        if "velocity_tag" in df.columns:
            lines.append(f"  Startle events:       {(df['velocity_tag'] == 'STARTLE').sum()}")
        if "startle_score" in df.columns:
            mask = df["startle_score"].notna()
            vals = df.loc[mask, "startle_score"].values
            tss = df.loc[mask, "timestamp"].values if "timestamp" in df.columns else np.arange(mask.sum(), dtype=float)
            if len(vals) > 0:
                pk = int(np.argmax(vals))
                lines.append(f"  Peak startle:         {vals[pk]:.2f} at t={float(tss[pk]):.2f}s")
    elif tool == "hsemotion":
        if "dominant" in df.columns:
            dom_counts = df["dominant"].value_counts()
            if len(dom_counts) > 0:
                lines.append(f"  Dominant emotion:     {dom_counts.index[0]} ({dom_counts.iloc[0] / len(df) * 100:.1f}%)")
        if "arousal" in df.columns:
            mask = df["arousal"].notna()
            vals = df.loc[mask, "arousal"].values
            tss = df.loc[mask, "timestamp"].values if "timestamp" in df.columns else np.arange(mask.sum(), dtype=float)
            if len(vals) > 0:
                pk = int(np.argmax(vals))
                lines.append(f"  Peak arousal:         {vals[pk]:.2f} at t={float(tss[pk]):.2f}s")
                lines.append(f"  Arousal std:          {vals.std():.3f}")
        if "valence" in df.columns:
            mask = df["valence"].notna()
            vals = df.loc[mask, "valence"].values
            if len(vals) > 0:
                mean_v = vals.mean()
                direction = "positive" if mean_v >= 0 else "negative"
                lines.append(f"  Mean valence:         {mean_v:.3f} ({direction})")
    elif tool in ("deepface", "pyfeat"):
        if "dominant" in df.columns:
            dom_counts = df["dominant"].value_counts()
            if len(dom_counts) > 0:
                lines.append(f"  Dominant emotion:     {dom_counts.index[0]} ({dom_counts.iloc[0] / len(df) * 100:.1f}%)")
        if "fear" in df.columns:
            mask = df["fear"].notna()
            vals = df.loc[mask, "fear"].values
            tss = df.loc[mask, "timestamp"].values if "timestamp" in df.columns else np.arange(mask.sum(), dtype=float)
            if len(vals) > 0:
                pk = int(np.argmax(vals))
                lines.append(f"  Peak fear:            {vals[pk]:.2f} at t={float(tss[pk]):.2f}s")
                lines.append(f"  Fear range:           {vals.min():.2f} \u2013 {vals.max():.2f}")
        if "arousal" in df.columns:
            mask = df["arousal"].notna()
            vals = df.loc[mask, "arousal"].values
            tss = df.loc[mask, "timestamp"].values if "timestamp" in df.columns else np.arange(mask.sum(), dtype=float)
            if len(vals) > 0:
                pk = int(np.argmax(vals))
                lines.append(f"  Peak arousal:         {vals[pk]:.2f} at t={float(tss[pk]):.2f}s")
    else:
        lines.append("  (unknown tool \u2014 no fingerprint available)")
    return lines


def _compute_key_moments(df, tool, top_n=5, min_gap_s=1.0):
    """Return KEY MOMENTS lines listing the top N emotional peaks with timestamps."""
    lines = ["", "KEY MOMENTS"]
    primary = {"mediapipe": "tension", "hsemotion": "arousal"}.get(tool, "fear")
    if primary not in df.columns:
        lines.append("  (primary signal column not found)")
        return lines

    mask = df[primary].notna()
    sig_vals = df.loc[mask, primary].values
    sig_ts = df.loc[mask, "timestamp"].values if "timestamp" in df.columns else np.arange(mask.sum(), dtype=float)

    peaks = _find_peaks(sig_ts, sig_vals, top_n=top_n, min_gap_s=min_gap_s)
    if not peaks:
        lines.append("  (no peaks found)")
        return lines

    ts_series = df.loc[mask, "timestamp"] if "timestamp" in df.columns else None
    for rank, (t, v, _) in enumerate(peaks, 1):
        extra = ""
        if ts_series is not None:
            row_idx = (ts_series - t).abs().idxmin()
            if tool == "mediapipe":
                if "velocity_tag" in df.columns:
                    vtag = df.at[row_idx, "velocity_tag"]
                    if pd.notna(vtag) and str(vtag).strip():
                        extra = f" ({vtag})"
                if not extra and "ctx_tag" in df.columns:
                    ctag = df.at[row_idx, "ctx_tag"]
                    if pd.notna(ctag) and str(ctag).strip():
                        extra = f" ({ctag})"
            else:
                if "dominant" in df.columns:
                    dom = df.at[row_idx, "dominant"]
                    if pd.notna(dom) and str(dom).strip():
                        extra = f" ({dom})"
        lines.append(f"  {rank}. t={t:.2f}s  {primary}={v:.3f}{extra}")
    return lines


class Session:
    """Manages metadata, resource monitoring, and registry for one FER session."""

    def __init__(self, tool_name, log_dir):
        self.tool_name = tool_name
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hardware = _detect_hardware()

        # Filled by prompt()
        self.subject_id = DEFAULT_SUBJECT
        self.video_id = ""          # specific video identifier, e.g. "Vid04"
        self.content_type = DEFAULT_CONTENT
        self.lighting = DEFAULT_LIGHTING
        self.notes = ""
        self.session_label = ""
        self.stimulus_type = "mixed"
        self.video_source = "webcam"

        # Set by the test script after device detection: session.device_used = _device
        self.device_used = "cpu"

        # Filled by start()
        self.start_time = None
        self._monitor = None

        # Per-frame tracking
        self.faces_detected = 0
        self.faces_missed = 0

    def pre_session_prompt(self):
        """Ask for subject ID, video ID, and lighting before the session starts."""
        print("\n" + "=" * 50)
        print("  SESSION SETUP")
        print("=" * 50)
        self.subject_id = (
            input(f"  Subject ID [{DEFAULT_SUBJECT}]: ").strip() or DEFAULT_SUBJECT
        )
        self.video_id = input("  Video ID (e.g. Vid04) []: ").strip()
        lighting = (
            input(f"  Lighting ({'/'.join(sorted(KNOWN_LIGHTING))}) [{DEFAULT_LIGHTING}]: ").strip()
            or DEFAULT_LIGHTING
        )
        self.lighting = lighting
        if self.lighting not in KNOWN_LIGHTING:
            print(f"    (note: '{self.lighting}' not in known set, using as-is)")
        print(f"\n  Subject:  {self.subject_id}")
        if self.video_id:
            print(f"  Video ID: {self.video_id}")
        print(f"  Lighting: {self.lighting}")
        print(f"  Tool:     {self.tool_name}")
        print(f"  GPU:      {self.hardware['gpu']}")
        print("=" * 50)
        print("  Starting in 2 seconds...\n")
        time.sleep(2)

    def post_session_confirm(self, csv_path):
        """Show recorded metadata and offer keep/discard. No additional questions.

        Returns:
            True  — keep the session
            False — discard: deletes csv_path from disk
        """
        print("\n" + "=" * 50)
        print("  SESSION COMPLETE")
        print("=" * 50)
        print(f"  Subject:  {self.subject_id}")
        if self.video_id:
            print(f"  Video ID: {self.video_id}")
        print(f"  Lighting: {self.lighting}")
        print(f"  Tool:     {self.tool_name}")

        action = input("\n  Keep session? [Y/n]: ").strip().lower()
        if action in ("n", "no"):
            if os.path.isfile(csv_path):
                os.remove(csv_path)
            print("  Session discarded.")
            return False

        print("=" * 50)
        return True

    def finalize_session_files(self, temp_csv_path):
        """Rename the temp CSV to the final convention name after post_session_confirm().

        Final name: {session_id}_{tool}_{subject_id}_{content_type}_{lighting}.csv
        Saved to: logs/sessions/{subject_id}_{content_type}_{lighting}/

        Returns:
            (final_csv_path, final_summary_path) — both with the canonical basename.
        """
        id_part = self.video_id if self.video_id else self.content_type
        basename = f"{self.session_id}_{self.tool_name}_{self.subject_id}_{id_part}_{self.lighting}"
        session_folder = f"{self.subject_id}_{id_part}_{self.lighting}"
        session_dir = os.path.join(self.log_dir, "sessions", session_folder)
        os.makedirs(session_dir, exist_ok=True)
        final_csv = os.path.join(session_dir, basename + ".csv")
        final_summary = os.path.join(session_dir, basename + "_summary.txt")
        if os.path.isfile(temp_csv_path):
            os.rename(temp_csv_path, final_csv)
        return final_csv, final_summary

    def prompt(self):
        """Full pre-session questionnaire (subject + content + lighting + notes).

        Legacy method used in standalone tests. For real sessions prefer
        pre_session_prompt() + post_session_confirm().
        """
        print("\n" + "=" * 50)
        print("  SESSION SETUP")
        print("=" * 50)

        self.subject_id = (
            input(f"  Subject ID [{DEFAULT_SUBJECT}]: ").strip() or DEFAULT_SUBJECT
        )
        self.content_type = (
            input(f"  Content type ({'/'.join(sorted(KNOWN_CONTENT_TYPES))}) [{DEFAULT_CONTENT}]: ").strip()
            or DEFAULT_CONTENT
        )
        if self.content_type not in KNOWN_CONTENT_TYPES:
            print(f"    (note: '{self.content_type}' is not in the known set, using as-is)")

        self.lighting = (
            input(f"  Lighting ({'/'.join(sorted(KNOWN_LIGHTING))}) [{DEFAULT_LIGHTING}]: ").strip()
            or DEFAULT_LIGHTING
        )
        if self.lighting not in KNOWN_LIGHTING:
            print(f"    (note: '{self.lighting}' is not in the known set, using as-is)")

        self.notes = input("  Notes []: ").strip()

        print(f"\n  Tool:     {self.tool_name}")
        print(f"  Subject:  {self.subject_id}")
        print(f"  Content:  {self.content_type}")
        print(f"  Lighting: {self.lighting}")
        print(f"  GPU:      {self.hardware['gpu']}")
        print(f"  CPU:      {self.hardware['cpu']}")
        print(f"  RAM:      {self.hardware['ram_gb']} GB")
        if self.notes:
            print(f"  Notes:    {self.notes}")
        print("=" * 50)
        print("  Starting in 2 seconds...\n")
        time.sleep(2)

    def start(self):
        """Record start time and begin resource monitoring."""
        self.start_time = time.perf_counter()
        self._monitor = _ResourceMonitor()
        self._monitor.start()

    def log_frame(self, face_detected, face_confidence=0.0, face_bbox_area=0):
        """Called per frame. Returns dict with resource + face metadata for CSV columns.

        Non-blocking — reads cached values from the monitor thread.
        """
        if face_detected:
            self.faces_detected += 1
        else:
            self.faces_missed += 1

        cpu = self._monitor.cpu_percent if self._monitor else 0.0
        ram = self._monitor.ram_mb if self._monitor else 0.0

        return {
            "cpu_percent": round(cpu, 1),
            "ram_mb": round(ram, 1),
            "face_detected": 1 if face_detected else 0,
            "face_confidence": round(face_confidence, 4),
            "face_bbox_area": face_bbox_area,
        }

    def finish(self, latencies, frame_count, csv_path, summary_path):
        """Stop monitoring, compute stats, write to sessions.json, return the record.

        Also computes comprehensive feature statistics from the CSV and appends
        enhanced summary sections to the summary file.
        """
        end_time = time.perf_counter()

        resource_summary = {}
        if self._monitor:
            self._monitor.stop()
            self._monitor.join(timeout=2)
            resource_summary = self._monitor.get_summary()

        session_duration = end_time - self.start_time if self.start_time else 0
        total_faces = self.faces_detected + self.faces_missed
        face_detection_rate = (
            round(self.faces_detected / total_faces, 4) if total_faces > 0 else 0
        )

        lat = np.array(latencies) if latencies else np.array([0])
        first_frame_latency = float(lat[0]) if len(lat) > 0 else 0
        steady_lat = lat[1:] if len(lat) > 1 else lat

        feature_stats = {}
        try:
            feature_stats = compute_feature_stats(csv_path)
        except Exception as e:
            print(f"\n  Warning: Could not compute feature stats: {e}")

        record = {
            "session_id": self.session_id,
            "tool": self.tool_name,
            "subject_id": self.subject_id,
            "video_id": self.video_id,
            "content_type": self.content_type,
            "lighting": self.lighting,
            "notes": self.notes,
            "session_label": self.session_label,
            "stimulus_type": self.stimulus_type,
            "video_source": self.video_source,
            "device_used": self.device_used,
            "start_time": datetime.fromtimestamp(
                time.time() - session_duration
            ).isoformat(timespec="seconds"),
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(session_duration, 1),
            "total_frames": frame_count,
            "fps_actual": round(frame_count / session_duration, 1) if session_duration > 0 else 0,
            "avg_latency_ms": round(float(np.mean(lat)), 1),
            "median_latency_ms": round(float(np.median(lat)), 1),
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 1),
            "max_latency_ms": round(float(np.max(lat)), 1),
            "first_frame_latency_ms": round(first_frame_latency, 1),
            "steady_avg_latency_ms": round(float(np.mean(steady_lat)), 1),
            "steady_p95_latency_ms": round(float(np.percentile(steady_lat, 95)), 1),
            "face_detection_rate": face_detection_rate,
            "faces_detected": self.faces_detected,
            "faces_missed": self.faces_missed,
            **resource_summary,
            "csv_path": os.path.relpath(csv_path, self.log_dir).replace("\\", "/"),
            "summary_path": os.path.relpath(summary_path, self.log_dir).replace("\\", "/"),
            "hardware": self.hardware,
            "feature_stats": feature_stats,
        }

        self._save_to_registry(record)
        extra_lines = self._build_enhanced_summary(record, feature_stats, csv_path=csv_path)

        return record, extra_lines

    def _save_to_registry(self, record):
        """Append session record to logs/sessions.json."""
        registry_path = os.path.join(self.log_dir, "sessions.json")
        data = {"sessions": []}
        if os.path.isfile(registry_path):
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {"sessions": []}

        data["sessions"].append(record)
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _build_enhanced_summary(self, record, feature_stats, csv_path=None):
        """Build additional summary lines for the enhanced summary file."""
        lines = [
            "",
            "-" * 60,
            "SESSION METADATA",
            "-" * 60,
            f"Session ID:             {record['session_id']}",
            f"Subject:                {record['subject_id']}",
            f"Video ID:               {record.get('video_id') or '(none)'}",
            f"Content:                {record['content_type']}",
            f"Stimulus type:          {record.get('stimulus_type', 'mixed')}",
            f"Lighting:               {record['lighting']}",
            f"Video source:           {record.get('video_source', 'webcam')}",
            f"Notes:                  {record['notes'] or '(none)'}",
            "",
            "-" * 60,
            "RESOURCE USAGE",
            "-" * 60,
            f"Device used:            {record['device_used']}",
            f"GPU (hardware):         {self.hardware['gpu']}",
            f"CPU avg:                {record.get('cpu_avg', 'N/A')}%",
            f"CPU peak:               {record.get('cpu_peak', 'N/A')}%",
            f"RAM avg:                {record.get('ram_avg_mb', 'N/A')} MB",
            f"RAM peak:               {record.get('ram_peak_mb', 'N/A')} MB",
            f"VRAM avg:               {record.get('vram_avg_mb', 'N/A')} MB",
            f"VRAM peak:              {record.get('vram_peak_mb', 'N/A')} MB",
            "",
            "-" * 60,
            "FACE DETECTION",
            "-" * 60,
            f"Frames with face:       {record['faces_detected']} / {record['total_frames']} "
            f"({record['face_detection_rate'] * 100:.1f}%)",
            f"Frames without face:    {record['faces_missed']} / {record['total_frames']} "
            f"({(1 - record['face_detection_rate']) * 100:.1f}%)",
            "",
            "-" * 60,
            "TIMING BREAKDOWN",
            "-" * 60,
            f"First-frame latency:    {record['first_frame_latency_ms']:.1f} ms (model warmup)",
            f"Steady-state avg:       {record['steady_avg_latency_ms']:.1f} ms (excluding first frame)",
            f"Steady-state p95:       {record['steady_p95_latency_ms']:.1f} ms (excluding first frame)",
        ]

        # Session fingerprint and key moments from CSV
        if csv_path:
            try:
                _fp_df = pd.read_csv(csv_path)
                _tool = _detect_tool_from_cols(_fp_df.columns.tolist())
                lines.extend(_compute_fingerprint(_fp_df, _tool))
                lines.extend(_compute_key_moments(_fp_df, _tool))
            except Exception as _fp_e:
                lines.append(f"\n  Warning: Could not compute fingerprint: {_fp_e}")

        # Feature statistics table
        if feature_stats:
            lines.extend(format_feature_stats_table(feature_stats))

        lines.append("=" * 60)
        return lines


# ---- Standalone test ----
if __name__ == "__main__":
    print("session_meta.py — standalone test")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    session = Session("test", log_dir)
    session.prompt()
    session.start()

    print("Monitoring for 5 seconds...")
    for i in range(10):
        res = session.log_frame(face_detected=(i % 3 != 0))
        print(f"  Frame {i}: CPU={res['cpu_percent']}%, RAM={res['ram_mb']}MB, "
              f"face={res['face_detected']}")
        time.sleep(0.5)

    # Create a tiny dummy CSV for feature stats test
    dummy_csv = os.path.join(log_dir, "test_dummy.csv")
    pd.DataFrame({
        "frame": [1, 2, 3],
        "timestamp": [0.1, 0.2, 0.3],
        "latency_ms": [10, 12, 11],
        "tension": [0.1, 0.3, 0.2],
    }).to_csv(dummy_csv, index=False)

    record, extra_lines = session.finish(
        latencies=[10, 12, 11],
        frame_count=10,
        csv_path=dummy_csv,
        summary_path=dummy_csv.replace(".csv", "_summary.txt"),
    )

    print("\n--- Session Record ---")
    for k, v in record.items():
        if k == "feature_stats":
            print(f"  feature_stats: {len(v)} features")
        else:
            print(f"  {k}: {v}")

    print("\n--- Enhanced Summary ---")
    for line in extra_lines:
        print(line)

    # Cleanup dummy file
    os.remove(dummy_csv)
    print(f"\nRegistry saved to: {os.path.join(log_dir, 'sessions.json')}")
    print("Done!")
