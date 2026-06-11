"""rPPG pipeline package.

Public API:
    RppgSession          — full pipeline facade (capture / analyze / replay / compare)
    SessionFolder        — dataclass for session artefact paths
    SessionRepository    — CRUD for Pipeline/sessions/ directories
    compute_bpm_timeseries — core BPM timeseries from RGB frames
    ALGORITHMS           — registry of available rPPG algorithms
    GTLoaderFactory      — creates the right GT loader for Zepp ZIP or CSV
    AccuracyReport       — computes and prints per-algorithm accuracy metrics
"""

from rppg.session import RppgSession, SessionFolder, SessionRepository
from rppg.rppg_algorithms import compute_bpm_timeseries, ALGORITHMS
from rppg.ground_truth import GTLoaderFactory, AccuracyReport

__all__ = [
    "RppgSession",
    "SessionFolder",
    "SessionRepository",
    "compute_bpm_timeseries",
    "ALGORITHMS",
    "GTLoaderFactory",
    "AccuracyReport",
]
