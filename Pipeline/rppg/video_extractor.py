"""Cached batch ROI extraction from pre-recorded videos.

Thin wrapper around VideoReextractor that adds disk caching. Extracts
face ROI RGB values from MP4 files and stores them as frames.csv for
downstream rPPG analysis.

Usage:
    from rppg.video_extractor import extract_or_load_cached
    frames_csv = extract_or_load_cached(
        Path("Recordings/S06_Vid16.mp4"),
        Path("Pipeline/logs/rppg_cache"),
    )
"""

from pathlib import Path

from rppg.extractors import VideoReextractor


def extract_or_load_cached(
    video_path: Path,
    cache_dir: Path,
    extractor_type: str = "mp",
    force: bool = False,
) -> Path:
    """Extract face ROI RGB from video, or return cached frames.csv.

    Args:
        video_path: path to the MP4 file
        cache_dir: directory for cached frames.csv files
        extractor_type: "mp" (MediaPipe) or "haar"
        force: if True, re-extract even if cache exists

    Returns:
        Path to the frames.csv file (cached or newly extracted).
    """
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{video_path.stem}_rppg_frames.csv"

    if not force and cache_path.exists():
        if cache_path.stat().st_mtime >= video_path.stat().st_mtime:
            print(f"[video_extractor] Using cached {cache_path.name}")
            return cache_path

    print(f"[video_extractor] Extracting ROI from {video_path.name} ...")
    reextractor = VideoReextractor(extractor_type=extractor_type)
    reextractor.reextract(video_path=video_path, output_csv=cache_path)
    return cache_path
