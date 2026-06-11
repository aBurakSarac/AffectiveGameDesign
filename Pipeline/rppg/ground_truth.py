"""Ground-truth HR loader and accuracy evaluation for Amazfit / Zepp exports.

Pattern: [Factory] — GTLoaderFactory.create(path) dispatches to the correct
loader without callers branching on format.  Both loaders normalise to the
same list[GTSample] so downstream code is format-agnostic.

Zepp Health (Amazfit Active 2) exports a ZIP containing HEARTRATE_AUTO.json:
    [{"time": "2026-05-02 10:30:00", "heartRate": 65}, ...]
Fallback variant with unix "timestamp" field is also handled.

Usage:
    from rppg.ground_truth import GTLoaderFactory, GTAligner, AccuracyReport
    samples = GTLoaderFactory.create(Path("zepp_export.zip")).load(path)
    aligned = GTAligner().align(samples, analysis_rows, session_start_unix)
    report  = AccuracyReport().compute(aligned)
"""

import csv
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class GTSample:
    timestamp_unix: float  # seconds since Unix epoch
    bpm: int


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class ZeppGTLoader:
    """Loads heart-rate data from a Zepp Health app export ZIP."""

    _CANDIDATE_FILES = (
        "HEARTRATE_AUTO.json",
        "HEARTRATE.json",
        "heartrate_auto.json",
        "heartrate.json",
    )

    def load(self, path: Path) -> List[GTSample]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        raw = self._read_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON array in {path.name}, got {type(raw)}")

        samples: List[GTSample] = []
        for entry in raw:
            ts = self._parse_timestamp(entry, path.name)
            bpm_val = entry.get("heartRate") or entry.get("heart_rate") or entry.get("bpm")
            if ts is None or bpm_val is None:
                continue
            bpm_int = int(bpm_val)
            if bpm_int <= 0:
                continue
            samples.append(GTSample(timestamp_unix=ts, bpm=bpm_int))

        if not samples:
            raise ValueError(
                f"No valid HR samples found in {path.name}. "
                f"Keys seen: {list(raw[0].keys()) if raw else '[]'}"
            )
        return sorted(samples, key=lambda s: s.timestamp_unix)

    def _read_json(self, path: Path) -> list:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                target = next(
                    (n for c in self._CANDIDATE_FILES for n in names
                     if n.endswith(c)),
                    None,
                )
                if target is None:
                    raise ValueError(
                        f"Could not find heart-rate JSON in ZIP. "
                        f"Files in archive: {names}"
                    )
                with zf.open(target) as f:
                    return json.load(f)
        else:
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    @staticmethod
    def _parse_timestamp(entry: dict, source_name: str) -> Optional[float]:
        # Variant 1: unix integer/float "timestamp"
        if "timestamp" in entry:
            return float(entry["timestamp"])

        # Variant 2: human-readable "time" string
        for key in ("time", "date_time", "dateTime"):
            if key in entry:
                raw = entry[key]
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                             "%Y-%m-%dT%H:%M:%SZ"):
                    try:
                        dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                        return dt.timestamp()
                    except ValueError:
                        continue
                # Last resort: fromisoformat (Python 3.7+)
                try:
                    dt = datetime.fromisoformat(raw)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.timestamp()
                except ValueError:
                    pass

        return None


class CsvGTLoader:
    """Loads heart-rate data from a simple CSV with timestamp + bpm columns.

    Also handles the headerless format produced by amazfit_connection.py:
        <unix_timestamp>,<bpm>,<rr_intervals_list>
    Detection: if the first field of the first row looks like a unix timestamp
    (large float > 1e9), treat columns as (0=timestamp, 1=bpm).
    """

    def load(self, path: Path) -> List[GTSample]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        samples: List[GTSample] = []
        with open(path, newline="", encoding="utf-8") as f:
            lines = [ln for ln in f if not ln.lstrip().startswith("#")]

        if not lines:
            return samples

        # Detect headerless format: first cell is a large float (unix ts ~1.7e9)
        first_cell = lines[0].split(",")[0].strip().strip('"')
        try:
            first_val = float(first_cell)
            is_headerless = first_val > 1_000_000_000
        except ValueError:
            is_headerless = False

        if is_headerless:
            return self._load_positional(lines)

        reader = csv.DictReader(lines)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path.name} has no header row")

        ts_col  = self._find_col(reader.fieldnames, ("timestamp", "time", "date_time"))
        bpm_col = self._find_col(reader.fieldnames, ("bpm", "heart_rate", "heartRate", "hr"))

        if ts_col is None or bpm_col is None:
            raise ValueError(
                f"CSV {path.name} missing timestamp/bpm columns. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            ts  = self._parse_ts(row[ts_col])
            bpm = int(float(row[bpm_col]))
            if ts is not None and bpm > 0:
                samples.append(GTSample(timestamp_unix=ts, bpm=bpm))

        if not samples:
            raise ValueError(f"No valid samples in {path.name}")
        return sorted(samples, key=lambda s: s.timestamp_unix)

    @staticmethod
    def _load_positional(lines: list) -> "List[GTSample]":
        """Parse amazfit_connection.py format: timestamp,bpm[,rr_intervals]."""
        samples = []
        for line in lines:
            parts = line.strip().split(",", 2)   # split at most 2 times; keep rr as blob
            if len(parts) < 2:
                continue
            try:
                ts  = float(parts[0].strip())
                bpm = int(float(parts[1].strip()))
                if bpm > 0:
                    samples.append(GTSample(timestamp_unix=ts, bpm=bpm))
            except (ValueError, IndexError):
                continue
        return sorted(samples, key=lambda s: s.timestamp_unix)

    @staticmethod
    def _find_col(fieldnames, candidates) -> Optional[str]:
        for name in fieldnames:
            if name.lower().replace("_", "") in [c.lower().replace("_", "")
                                                  for c in candidates]:
                return name
        return None

    @staticmethod
    def _parse_ts(value: str) -> Optional[float]:
        value = value.strip()
        # Try plain float (unix)
        try:
            return float(value)
        except ValueError:
            pass
        # Try ISO formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# TCX loader (Zepp workout export — Training Center XML)
# ---------------------------------------------------------------------------

class TcxGTLoader:
    """Parses a TCX file exported from the Zepp app.

    Each <Trackpoint> element that contains a <HeartRateBpm><Value> is
    extracted.  Timestamps come from the <Time> element (ISO 8601, UTC).

    TCX trackpoints typically have 1-second resolution during workouts.
    """

    _NS = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"

    def load(self, path: Path) -> List[GTSample]:
        import xml.etree.ElementTree as ET

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        tree = ET.parse(path)
        root = tree.getroot()

        # Strip namespace prefix from tag comparisons
        ns = self._NS
        samples: List[GTSample] = []

        for tp in root.iter(f"{{{ns}}}Trackpoint"):
            time_el = tp.find(f"{{{ns}}}Time")
            hr_el   = tp.find(f".//{{{ns}}}HeartRateBpm/{{{ns}}}Value")
            if time_el is None or hr_el is None:
                continue
            try:
                bpm = int(hr_el.text.strip())
                if bpm <= 0:
                    continue
                ts = self._parse_time(time_el.text.strip())
                if ts is not None:
                    samples.append(GTSample(timestamp_unix=ts, bpm=bpm))
            except (ValueError, AttributeError):
                continue

        if not samples:
            raise ValueError(
                f"No HR trackpoints found in {path.name}. "
                "Check that the TCX file contains HeartRateBpm elements."
            )
        return sorted(samples, key=lambda s: s.timestamp_unix)

    @staticmethod
    def _parse_time(value: str) -> Optional[float]:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class GTLoaderFactory:
    @staticmethod
    def create(path: Path):
        """Return the appropriate loader for the given file path."""
        suffix = Path(path).suffix.lower()
        if suffix == ".zip":
            return ZeppGTLoader()
        if suffix == ".tcx":
            return TcxGTLoader()
        if suffix in (".csv", ".tsv", ".txt"):
            return CsvGTLoader()
        # Try ZeppGTLoader for bare .json too
        if suffix == ".json":
            return ZeppGTLoader()
        raise ValueError(
            f"Unrecognised GT file extension '{suffix}'. "
            "Expected .tcx (Zepp workout), .zip (Zepp full export), .json (Zepp), or .csv."
        )


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class GTAligner:
    """Aligns GT HR samples to rPPG analysis windows by timestamp interpolation."""

    MAX_GAP_S = 90.0  # mark as NaN if nearest GT sample is farther than this

    def align(
        self,
        gt_samples: List[GTSample],
        rppg_windows: List[dict],
        session_start_unix: float,
    ) -> List[dict]:
        """Add 'gt_bpm' column to each rPPG window row.

        Args:
            gt_samples: list of GTSample (absolute unix timestamps)
            rppg_windows: list of dicts from analysis.csv
                          (must contain 't_center' = seconds since session start)
            session_start_unix: unix timestamp of the first captured frame

        Returns:
            Copy of rppg_windows with 'gt_bpm' key added (float or nan).
        """
        if not gt_samples:
            return [{**r, "gt_bpm": float("nan")} for r in rppg_windows]

        gt_ts  = np.array([s.timestamp_unix for s in gt_samples], dtype=float)
        gt_bpm = np.array([s.bpm for s in gt_samples], dtype=float)

        result = []
        for row in rppg_windows:
            window_unix = session_start_unix + float(row["t_center"])
            # Distance to nearest GT sample
            dists = np.abs(gt_ts - window_unix)
            nearest_idx = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_idx])

            if nearest_dist > self.MAX_GAP_S:
                interpolated = float("nan")
            elif len(gt_ts) == 1:
                interpolated = float(gt_bpm[0])
            else:
                # Linear interpolation between the two surrounding samples
                interpolated = float(np.interp(window_unix, gt_ts, gt_bpm))

            result.append({**row, "gt_bpm": round(interpolated, 1)
                            if not math.isnan(interpolated) else float("nan")})
        return result

    def save(self, aligned_rows: List[dict], output_path: Path) -> None:
        if not aligned_rows:
            return
        output_path = Path(output_path)
        fieldnames = list(aligned_rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in aligned_rows:
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------

class AccuracyReport:
    """Computes per-algorithm accuracy metrics against aligned GT HR."""

    def compute(self, aligned_rows: List[dict], snr_min: float = 0.0) -> dict:
        """Return dict of per-algorithm metrics.

        Args:
            aligned_rows: output of GTAligner.align(); must contain 'algorithm'
                          and 'gt_bpm' columns.
            snr_min: if > 0, only include windows with SNR >= this value.

        Returns:
            {algorithm: {mae, rmse, pearson_r, bias, n}} plus an 'ALL' summary.
        """
        from collections import defaultdict

        by_algo: dict = defaultdict(lambda: {"pred": [], "gt": []})
        for row in aligned_rows:
            gt = row.get("gt_bpm")
            if gt is None or (isinstance(gt, float) and math.isnan(gt)):
                continue
            if snr_min > 0:
                snr = float(row.get("snr", 0))
                if snr < snr_min:
                    continue
            pred = row.get("bpm_smoothed") or row.get("bpm")
            if pred is None:
                continue
            algo = row.get("algorithm", "UNKNOWN")
            by_algo[algo]["pred"].append(float(pred))
            by_algo[algo]["gt"].append(float(gt))

        report = {}
        all_pred, all_gt = [], []
        for algo, data in by_algo.items():
            pred = np.array(data["pred"])
            gt   = np.array(data["gt"])
            metrics = self._metrics(pred, gt)
            report[algo] = metrics
            all_pred.extend(data["pred"])
            all_gt.extend(data["gt"])

        if all_pred:
            report["ALL"] = self._metrics(np.array(all_pred), np.array(all_gt))

        return report

    @staticmethod
    def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
        if len(pred) == 0:
            return {"mae": float("nan"), "rmse": float("nan"),
                    "pearson_r": float("nan"), "bias": float("nan"), "n": 0}
        diff = pred - gt
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
        n    = len(pred)
        if n >= 2 and np.std(pred) > 0 and np.std(gt) > 0:
            r = float(np.corrcoef(pred, gt)[0, 1])
        else:
            r = float("nan")
        return {"mae": round(mae, 2), "rmse": round(rmse, 2),
                "pearson_r": round(r, 3), "bias": round(bias, 2), "n": n}

    def print_table(self, report: dict) -> None:
        print(f"\n{'Algorithm':12s}  {'MAE':>6}  {'RMSE':>6}  {'r':>6}  {'Bias':>6}  {'N':>4}")
        print("-" * 52)
        for algo in sorted(report.keys(), key=lambda a: (a == "ALL", a)):
            m = report[algo]
            r_str = f"{m['pearson_r']:6.3f}" if not math.isnan(m["pearson_r"]) else "   nan"
            print(f"{algo:12s}  {m['mae']:6.2f}  {m['rmse']:6.2f}  "
                  f"{r_str}  {m['bias']:+6.2f}  {m['n']:4d}")

    def save(self, report: dict, output_path: Path) -> None:
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
