"""Constants and registries for the La Façade Fissuréе annotation pipeline.

Pattern: [Constants / Registry] — all annotation-domain constants, headers,
    channel definitions, and small pure helpers that other modules import.
"""

import csv

# ── Required columns (full CSV from test_mp_hs.py / benchmark_explorer) ─
REQUIRED = [
    "frame", "timestamp",
    "hs_dominant", "hs_fear", "hs_surprise", "hs_arousal",
    "mp_startle_score", "mp_tension", "mp_ctx_tag",
    "composite_fear",
    "agreement_tag", "veto_tag",
    "mp_face_detected", "hs_face_detected",
]

# ── Composite fear formulas (post-hoc recomputation from raw signals) ─────
FORMULAS = {
    "original": lambda r: r["composite_fear"],  # use pre-computed column as-is
    "F2":       lambda r: min(1.0, max(0.0,
                    0.40 * r["hs_fear"] + 0.20 * r["hs_surprise"]
                    + 0.40 * r["mp_tension"])),
    "F1":       lambda r: min(1.0, max(0.0,
                    0.60 * r["hs_fear"] + 0.40 * r["mp_tension"])),
}

def _c(v):
    return min(1.0, max(0.0, v))

# ── All 7 benchmark formulas (v3 audit columns — mirrors formula_benchmark.py) ──
BENCHMARK_FORMULAS = [
    ("F0", lambda r: _c(0.60 * r["hs_arousal"] + 0.40 * r["mp_tension"])),
    ("F1", lambda r: _c(0.60 * r["hs_fear"]    + 0.40 * r["mp_tension"])),
    ("F2", lambda r: _c(0.40 * r["hs_fear"] + 0.20 * r["hs_surprise"] + 0.40 * r["mp_tension"])),
    ("F3", lambda r: _c(0.50 * r["hs_fear"] + 0.20 * r["hs_arousal"]  + 0.30 * r["mp_tension"])),
    ("F4", lambda r: _c((0.60 * r["hs_arousal"] + 0.40 * r["mp_tension"])
                        * (1.0 if r["hs_anger"] <= r["hs_fear"] else 0.0))),
    ("F5", lambda r: _c(0.50 * r["hs_fear"] + 0.30 * r["mp_tension"]
                        + 0.20 * _c(r["mp_startle_score"] / 10.0))),
    ("F6", lambda r: _c((0.60 * r["hs_fear"] + 0.40 * r["mp_tension"])
                        * max(0.0, 1.0 - r["hs_anger"]))),
]

# Emotions that seed the flood-fill (Method B: Fear + Surprise combined)
SEED_EMOTIONS = {"Fear", "Surprise"}

# All HS dominant labels for per-cluster frame counting
HS_EMOTIONS = {
    "Fear": "n_fear_frames",
    "Surprise": "n_surprise_frames",
    "Anger": "n_angry_frames",
    "Happiness": "n_happy_frames",
    "Sadness": "n_sad_frames",
    "Contempt": "n_contempt_frames",
    "Disgust": "n_disgust_frames",
    "Neutral": "n_neutral_frames",
}

# ── v1: Template columns (peak detection + fixed window) ────────────────
HEADER_V1 = [
    # Identity & timing
    "event_id", "cluster_start", "cluster_end", "peak_t",
    # Manual
    "verdict", "category", "start_s", "end_s", "context",
    "face_visible_in_gaps", "no_face_pct",
    # Fusion
    "peak_composite", "n_veto_frames", "veto_reasons",
    # HS signals
    "cluster_emotion", "source",
    "hs_dominant_at_peak", "peak_hs_fear", "peak_hs_surprise", "peak_arousal",
    "n_fear_frames", "n_surprise_frames",
    "n_angry_frames", "n_happy_frames", "n_sad_frames",
    "n_contempt_frames", "n_disgust_frames", "n_neutral_frames",
    # MP signals
    "peak_startle", "peak_tension", "mp_ctx_tags",
    # Data quality (end)
    "n_no_face_frames",
    # Calculated
    "duration_s",
]

# ── v2: Template columns (flood-fill + confidence) ──────────────────────
HEADER_V2 = [
    # Identity & timing
    "event_id", "cluster_start", "cluster_end", "peak_t",
    # Manual columns (annotator fills these)
    "label", "start_s", "end_s", "context",
    # Auto
    "auto_confidence",
    "face_visible_in_gaps", "no_face_pct",
    # Fusion
    "peak_composite", "sustained_pct", "mean_arousal",
    "n_veto_frames", "veto_reasons",
    # HS signals
    "cluster_emotion", "source",
    "hs_dominant_at_peak", "peak_hs_fear", "peak_hs_surprise", "peak_arousal",
    "n_fear_frames", "n_surprise_frames",
    "n_angry_frames", "n_happy_frames", "n_sad_frames",
    "n_contempt_frames", "n_disgust_frames", "n_neutral_frames",
    # MP signals
    "peak_startle", "peak_tension", "mp_ctx_tags",
    # Data quality (end)
    "n_no_face_frames",
    # Calculated
    "duration_s",
]

# ── v3: Template columns (multi-channel voting) ───────────────────────
HEADER_V3 = [
    # Identity & timing
    "event_id", "tier", "cluster_start_s", "cluster_end_s", "duration_s", "peak_t",
    # System emotion assessment
    "cluster_emotion",
    # Manual annotation block (annotator fills first)
    "verdict", "category", "context",
    # Manual rPPG observation (smartwatch)
    "rppg_watch",
    # Manual timestamps (precise response window)
    "start_s", "end_s",
    # Per-formula emotion vote: cluster_emotion if formula fired, "---" otherwise
    "F0_vote", "F1_vote", "F2_vote", "F3_vote", "F4_vote", "F5_vote", "F6_vote",
    # Formula and channel summaries
    "formulas_voted",    # e.g. "F0,F1,F3" — formulas with peak score >= threshold
    "channels_filtered", # active channels whose emotion vote matches cluster_emotion
    # Voting summary
    "vote_count_peak", "vote_count_mean", "channels_active",
    # Per-channel binary (0/1)
    "ch_DOM_FEAR", "ch_DOM_SURP", "ch_FEAR_THR",
    "ch_TENSION", "ch_STARTLE", "ch_AROUSAL", "ch_CROSS_MODAL", "ch_TWO_GATE",
    "gate_overlap_pct",
    # Raw formula scores (peak and mean for each formula)
    "peak_f0", "mean_f0",
    "peak_f1", "mean_f1",
    "peak_f2", "mean_f2",
    "peak_f3", "mean_f3",
    "peak_f4", "mean_f4",
    "peak_f5", "mean_f5",
    "peak_f6", "mean_f6",
    # HS signals at peak
    "peak_hs_fear", "peak_hs_surprise", "peak_hs_arousal",
    # MP signals at peak
    "peak_mp_tension", "peak_mp_startle",
    # Per-emotion frame counts
    "n_fear_frames", "n_surprise_frames",
    "n_angry_frames", "n_happy_frames", "n_sad_frames",
    "n_contempt_frames", "n_disgust_frames", "n_neutral_frames",
    # Data quality
    "no_face_pct", "n_veto_frames", "scene_cut_bounded",
    # rPPG (auto-filled if --rppg-csv provided, blank otherwise — fill manually)
    "rppg_bpm_chrom", "rppg_pre_bpm_chrom", "rppg_delta_chrom",
    "rppg_bpm_pos",   "rppg_pre_bpm_pos",   "rppg_delta_pos",
    "rppg_bpm_green", "rppg_pre_bpm_green",  "rppg_delta_green",
    # Manual rPPG opinion
    "rppg_impression",   # fill: RISING / FALLING / STABLE / UNCLEAR
    "rppg_notes",
]

# ── v3: Channel source (HS = HSEmotion, MP = MediaPipe, BOTH = cross-modal) ──
CHANNEL_SOURCE = {
    "DOM_FEAR":    "HS",
    "DOM_SURP":    "HS",
    "FEAR_THR":    "HS",
    "AROUSAL":     "HS",
    "TENSION":     "MP",
    "STARTLE":     "MP",
    "CROSS_MODAL": "BOTH",
}

# ── v3: Which emotion each channel semantically votes for ─────────────────
CHANNEL_EMOTION_VOTE = {
    "DOM_FEAR":    "Fear",
    "DOM_SURP":    "Surprise",
    "FEAR_THR":    "Fear",
    "TENSION":     "Fear",     # tension -> fear in horror context
    "STARTLE":     "Surprise",
    "AROUSAL":     None,       # non-specific arousal
    "CROSS_MODAL": "Fear",     # cross-modal fear+tension agreement
}

# ── v3: Default threshold for formula vote columns (FX_vote / formulas_voted) ─
FORMULA_VOTE_THR = 0.25

# ── v3: Category dropdown options ────────────────────────────────────────
CATEGORY_OPTIONS = (
    "FEAR,STARTLED,HORRIFIED,SURPRISED,TENSE,"
    "THRILLED,RELIEVED,CALM,ANGRY,DISGUSTED,SAD,AMUSED,CONFUSED"
)

# ── rPPG algorithm names ──────────────────────────────────────────────────
RPPG_ALGOS = ["chrom", "pos", "green"]


def _count_emotions(cluster):
    """Count per-emotion frames in a cluster slice."""
    counts = {}
    for emo, col in HS_EMOTIONS.items():
        counts[col] = sum(1 for r in cluster if r["hs_dominant"] == emo)
    return counts


def _all_formula_scores(span):
    """Compute peak and mean for all 7 benchmark formulas across a cluster span.
    Returns dict of {'peak_f0': ..., 'mean_f0': ..., ..., 'peak_f6': ..., 'mean_f6': ...}."""
    result = {}
    for fname, fn in BENCHMARK_FORMULAS:
        key = fname.lower()
        vals = [fn(r) for r in span]
        result[f"peak_{key}"] = max(vals) if vals else 0.0
        result[f"mean_{key}"] = sum(vals) / len(vals) if vals else 0.0
    return result


def _channels_active_at(row, channels):
    """Return list of channel names that fire for a given row."""
    return [name for name, fn in channels if fn(row)]
