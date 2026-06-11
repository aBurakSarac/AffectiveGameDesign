"""Session folder management and RppgSession facade.

Pattern: [Facade + Repository]

SessionRepository handles all path logic for Pipeline/sessions/<id>/ folders.
RppgSession is the single public entry point that orchestrates capture →
analyze → GT alignment → replay without callers touching individual modules.

Usage:
    from rppg.session import RppgSession, SessionRepository
    # Full capture + analysis
    session = RppgSession()
    folder = session.capture("pilot_01", gt_path=Path("zepp.zip"))

    # Re-analyze an existing session
    folder = session.analyze("20260502_123456_pilot_01", gt_path=Path("zepp.zip"))

    # Render replay only
    replay_path = session.replay("20260502_123456_pilot_01")
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rppg.analyzer import AnalysisConfig, RppgAnalyzer
from rppg.capture import CaptureConfig, LiveCaptureSession
from rppg.extractors import VideoReextractor
from rppg.ground_truth import AccuracyReport, GTAligner, GTLoaderFactory
from rppg.replay import ReplayRenderer


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SessionFolder:
    """Paths for all artefacts inside a session directory.

    Missing files have their field set to None — callers must check existence.
    """
    id: str
    path: Path
    raw_video:    Optional[Path] = None
    frames_csv:   Optional[Path] = None
    analysis_csv: Optional[Path] = None
    gt_csv:       Optional[Path] = None
    replay_video: Optional[Path] = None
    accuracy_json: Optional[Path] = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "path": str(self.path),
            "raw_video":     str(self.raw_video)    if self.raw_video    else None,
            "frames_csv":    str(self.frames_csv)   if self.frames_csv   else None,
            "analysis_csv":  str(self.analysis_csv) if self.analysis_csv else None,
            "gt_csv":        str(self.gt_csv)       if self.gt_csv       else None,
            "replay_video":  str(self.replay_video) if self.replay_video else None,
            "accuracy_json": str(self.accuracy_json)if self.accuracy_json else None,
        }


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

class SessionRepository:
    """CRUD for Pipeline/sessions/<session-id>/ directories."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Resolve relative to this file: Pipeline/rppg/session.py → Pipeline/sessions/
            base_dir = Path(__file__).parent.parent / "sessions"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self, label: str, ts_str: Optional[str] = None) -> SessionFolder:
        if ts_str is None:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_")
        session_id = f"{ts_str}_{safe_label}"
        session_path = self.base_dir / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        return SessionFolder(id=session_id, path=session_path)

    def load(self, session_id: str) -> SessionFolder:
        session_path = self.base_dir / session_id
        if not session_path.exists():
            raise FileNotFoundError(f"Session not found: {session_path}")

        def maybe(name: str) -> Optional[Path]:
            p = session_path / name
            return p if p.exists() else None

        return SessionFolder(
            id=session_id,
            path=session_path,
            raw_video=    maybe("raw_video.mp4"),
            frames_csv=   maybe("frames.csv"),
            analysis_csv= maybe("analysis.csv"),
            gt_csv=       maybe("gt_aligned.csv"),
            replay_video= maybe("replay.mp4"),
            accuracy_json=maybe("accuracy_report.json"),
        )

    def list(self) -> List[SessionFolder]:
        folders = []
        for p in sorted(self.base_dir.iterdir()):
            if p.is_dir():
                try:
                    folders.append(self.load(p.name))
                except Exception:
                    pass
        return folders


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class RppgSession:
    """Orchestrates the full rPPG pipeline for a single session."""

    def __init__(
        self,
        repo: Optional[SessionRepository] = None,
        recordings_dir: Optional[Path] = None,
    ):
        self._repo = repo or SessionRepository()
        self._recordings_dir = recordings_dir or (
            Path(__file__).parent.parent.parent / "Recordings" / "Live"
        )
        self._recordings_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(
        self,
        label: str,
        capture_config: Optional[CaptureConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
        gt_path: Optional[Path] = None,
    ) -> SessionFolder:
        """Run live capture, then analyze + replay.

        Returns the fully populated SessionFolder.
        """
        cfg = capture_config or CaptureConfig(output_dir=self._recordings_dir)
        cfg.output_dir = self._recordings_dir

        # 1. Capture
        capture_session = LiveCaptureSession(cfg)
        raw = capture_session.run(label)

        # 2. Create session folder
        folder = self._repo.create(label, ts_str=raw["ts_str"])

        # 3. Symlink / copy video and frames CSV into session folder
        folder = self._link_artefacts(folder, raw["video_path"], raw["frames_csv"])

        # 4. Analyze
        folder = self._run_analysis(folder, analysis_config, raw["session_start_unix"])

        # 5. GT alignment + accuracy (optional)
        if gt_path is not None:
            folder = self._run_gt(folder, Path(gt_path), raw["session_start_unix"])

        # 6. Replay
        folder = self._run_replay(folder)

        self._print_summary(folder)
        return folder

    def reextract(
        self,
        session_id: str,
        extractor_type: str = "mp",
        analysis_config: Optional[AnalysisConfig] = None,
        gt_path: Optional[Path] = None,
    ) -> SessionFolder:
        """Re-run MultiROIExtractor on raw_video.mp4, regenerate frames.csv, then analyze.

        Use this after changing landmark definitions or ROI logic so that the new
        extraction code is applied to an existing recording.
        Original timestamps are preserved from the existing frames.csv if present.
        """
        folder = self._repo.load(session_id)
        if folder.raw_video is None:
            raise FileNotFoundError(f"No raw_video.mp4 in session {session_id}")

        frames_csv_path = folder.path / "frames.csv"
        original_csv    = frames_csv_path if frames_csv_path.exists() else None

        # Back up existing frames.csv before overwriting so old extraction is preserved
        if original_csv is not None:
            import shutil as _shutil
            backup = folder.path / "frames_pre_reextract.csv"
            _shutil.copy2(original_csv, backup)
            print(f"[Session] Backed up original frames.csv → {backup.name}")

        print(f"[Session] Re-extracting ROI from {folder.raw_video.name} ...")
        VideoReextractor(extractor_type=extractor_type).reextract(
            video_path=folder.raw_video,
            output_csv=frames_csv_path,
            original_csv=original_csv,
        )
        folder.frames_csv = frames_csv_path

        folder = self._run_analysis(folder, analysis_config, None)

        if gt_path is not None:
            folder = self._run_gt(folder, Path(gt_path), None)

        folder = self._run_replay(folder)
        self._print_summary(folder)
        return folder

    def analyze(
        self,
        session_id: str,
        analysis_config: Optional[AnalysisConfig] = None,
        gt_path: Optional[Path] = None,
        session_start_unix: Optional[float] = None,
    ) -> SessionFolder:
        """Re-run analysis (and optionally GT + replay) on an existing session."""
        folder = self._repo.load(session_id)

        if folder.frames_csv is None:
            raise FileNotFoundError(f"No frames.csv in session {session_id}")

        folder = self._run_analysis(folder, analysis_config, session_start_unix)

        if gt_path is not None:
            folder = self._run_gt(folder, Path(gt_path), session_start_unix)

        folder = self._run_replay(folder)
        self._print_summary(folder)
        return folder

    def replay(self, session_id: str) -> Path:
        """Re-render the replay video for an existing analyzed session."""
        folder = self._repo.load(session_id)
        folder = self._run_replay(folder)
        return folder.replay_video

    def compare(
        self,
        session_id: str,
        gt_path: Path,
        session_start_unix: Optional[float] = None,
    ) -> SessionFolder:
        """Add GT alignment + accuracy report to an already-analyzed session."""
        folder = self._repo.load(session_id)
        if folder.analysis_csv is None:
            raise FileNotFoundError(
                f"analysis.csv missing in {session_id} — run analyze first."
            )
        folder = self._run_gt(folder, Path(gt_path), session_start_unix)
        folder = self._run_replay(folder)
        self._print_summary(folder)
        return folder

    def list_sessions(self) -> List[SessionFolder]:
        return self._repo.list()

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _link_artefacts(
        self, folder: SessionFolder, video_path: Path, frames_csv: Path
    ) -> SessionFolder:
        dst_video  = folder.path / "raw_video.mp4"
        dst_frames = folder.path / "frames.csv"

        if not dst_video.exists():
            try:
                os.symlink(video_path.resolve(), dst_video)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(video_path, dst_video)

        if not dst_frames.exists():
            try:
                os.symlink(frames_csv.resolve(), dst_frames)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(frames_csv, dst_frames)

        folder.raw_video  = dst_video
        folder.frames_csv = dst_frames
        return folder

    def _run_analysis(
        self, folder: SessionFolder,
        config: Optional[AnalysisConfig],
        session_start_unix: Optional[float],
    ) -> SessionFolder:
        analyzer = RppgAnalyzer()
        results  = analyzer.analyze(folder.frames_csv, config)
        if not results:
            print("[Session] No analysis results — check frames.csv.")
            return folder

        out_path = folder.path / "analysis.csv"
        analyzer.save(results, out_path)
        folder.analysis_csv = out_path
        return folder

    def _run_gt(
        self, folder: SessionFolder,
        gt_path: Path,
        session_start_unix: Optional[float],
    ) -> SessionFolder:
        if folder.analysis_csv is None:
            return folder

        # Load GT
        loader   = GTLoaderFactory.create(gt_path)
        samples  = loader.load(gt_path)
        print(f"[Session] Loaded {len(samples)} GT samples from {gt_path.name}")

        # Determine session start unix from analysis CSV if not supplied
        start_unix = session_start_unix
        if start_unix is None:
            start_unix = self._read_session_start_unix(folder.frames_csv)
        if start_unix is None:
            print("[Session] WARNING: session_start_unix unknown — GT alignment may be off.")
            start_unix = 0.0

        # Load analysis rows
        import csv as _csv
        with open(folder.analysis_csv, newline="", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))

        # Align
        aligner = GTAligner()
        aligned = aligner.align(samples, rows, start_unix)

        gt_csv_path = folder.path / "gt_aligned.csv"
        aligner.save(aligned, gt_csv_path)
        folder.gt_csv = gt_csv_path

        # Accuracy report
        report      = AccuracyReport()
        metrics     = report.compute(aligned)
        report.print_table(metrics)

        acc_path = folder.path / "accuracy_report.json"
        report.save(metrics, acc_path)
        folder.accuracy_json = acc_path

        # SNR-filtered reports for quality analysis
        for snr_thresh in [3.0, 5.0]:
            filtered = report.compute(aligned, snr_min=snr_thresh)
            if filtered:
                tag = f"snr{snr_thresh:.0f}"
                filt_path = folder.path / f"accuracy_{tag}.json"
                report.save(filtered, filt_path)
                print(f"\n[SNR >= {snr_thresh:.0f}]")
                report.print_table(filtered)

        return folder

    def _run_replay(self, folder: SessionFolder) -> SessionFolder:
        if folder.raw_video is None or folder.analysis_csv is None:
            print("[Session] Skipping replay — missing video or analysis.csv.")
            return folder

        replay_path = folder.path / "replay.mp4"
        renderer    = ReplayRenderer()
        renderer.render(folder.path, replay_path)
        folder.replay_video = replay_path
        return folder

    @staticmethod
    def _read_session_start_unix(frames_csv: Optional[Path]) -> Optional[float]:
        if frames_csv is None or not frames_csv.exists():
            return None
        from datetime import timezone as _tz
        with open(frames_csv, encoding="utf-8") as f:
            first = f.readline().strip()
        if first.startswith("# session_start_utc="):
            utc_str = first.split("=", 1)[1]
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(utc_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_tz.utc)
                return dt.timestamp()
            except ValueError:
                pass
        return None

    @staticmethod
    def _print_summary(folder: SessionFolder) -> None:
        print(f"\n{'='*60}")
        print(f"Session: {folder.id}")
        print(f"  Path       : {folder.path}")
        print(f"  Video      : {folder.raw_video}")
        print(f"  Frames     : {folder.frames_csv}")
        print(f"  Analysis   : {folder.analysis_csv}")
        print(f"  GT aligned : {folder.gt_csv}")
        print(f"  Accuracy   : {folder.accuracy_json}")
        print(f"  Replay     : {folder.replay_video}")
        print(f"{'='*60}\n")
