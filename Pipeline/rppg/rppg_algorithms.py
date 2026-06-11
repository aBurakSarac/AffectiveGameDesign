"""Standalone rPPG signal processing algorithms.

Pattern: [Strategy] — CHROM, POS, GREEN, ICA, and WAVELET are interchangeable
algorithm implementations registered in the ALGORITHMS dict. Callers select
by name; compute_bpm_timeseries() applies any registered strategy.

All algorithms accept an Nx3 array of mean [R, G, B] values from the face
forehead ROI and return a 1D rPPG signal. BPM is estimated via FFT.

References:
    CHROM — de Haan & Jeanne (2013), IEEE Trans. Biomed. Eng.
    POS   — Wang et al. (2017), IEEE Trans. Biomed. Eng.
"""

import warnings

import numpy as np
from scipy.signal import butter, detrend, filtfilt

# ---------------------------------------------------------------------------
# Startup dependency check — warn loudly instead of silently returning zeros
# ---------------------------------------------------------------------------

_MISSING_DEPS: list = []
try:
    from sklearn.decomposition import FastICA as _FastICA  # noqa: F401
except ImportError:
    _MISSING_DEPS.append("scikit-learn (required for ICA algorithm)")
try:
    import pywt as _pywt  # noqa: F401
except ImportError:
    _MISSING_DEPS.append("PyWavelets (required for WAVELET algorithm)")

if _MISSING_DEPS:
    warnings.warn(
        f"[rppg_algorithms] Missing dependencies: {', '.join(_MISSING_DEPS)}. "
        f"Install via: pip install {' '.join(d.split()[0] for d in _MISSING_DEPS)}",
        ImportWarning,
        stacklevel=2,
    )


__all__ = [
    "ALGORITHMS",
    "compute_bpm_timeseries",
    "estimate_bpm",
    "interpolate_motion_frames",
    "median_smooth",
    "chrom", "pos", "green", "ica", "wavelet",
]


# ---------------------------------------------------------------------------
# Bandpass filter + BPM estimation (shared by all algorithms)
# ---------------------------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def parabolic_interpolation(spectrum, peak_idx):
    """Refine FFT peak location via 3-point parabolic interpolation.

    Returns fractional bin offset from peak_idx.
    """
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return float(peak_idx)
    alpha = spectrum[peak_idx - 1]
    beta = spectrum[peak_idx]
    gamma = spectrum[peak_idx + 1]
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-10:
        return float(peak_idx)
    p = 0.5 * (alpha - gamma) / denom
    return peak_idx + p


def compute_fft_snr(fft_power, peak_idx):
    """Compute SNR of FFT peak vs noise floor in the band.

    Returns:
        snr: peak power / median noise (float)
        prominence: peak power - mean power (float)
    """
    n = len(fft_power)
    if n < 3:
        return 0.0, 0.0
    # Exclude peak +/- 2 bins from noise estimate
    noise_mask = np.ones(n, dtype=bool)
    noise_mask[max(0, peak_idx - 2):min(n, peak_idx + 3)] = False
    noise_vals = fft_power[noise_mask]
    if len(noise_vals) == 0 or np.median(noise_vals) < 1e-10:
        return 0.0, 0.0
    snr = float(fft_power[peak_idx] / np.median(noise_vals))
    prominence = float(fft_power[peak_idx] - np.mean(noise_vals))
    return snr, prominence


def estimate_bpm(signal, fps, low_bpm=60, high_bpm=180,
                 harmonic_disambiguation=True, harmonic_ratio=0.6):
    """Estimate BPM from an rPPG signal using FFT peak detection.

    Args:
        signal: 1D rPPG signal array
        fps: sampling rate
        low_bpm: minimum detectable BPM (default 60, raised from 42 to
                 exclude sub-cardiac breathing artifacts)
        high_bpm: maximum detectable BPM
        harmonic_disambiguation: if True, check whether the dominant peak
                 is a sub-harmonic by looking for power at 2x the frequency
        harmonic_ratio: threshold for preferring the 2x harmonic — if the
                 power at 2*f is >= this fraction of the power at f, prefer 2*f

    Returns:
        bpm: estimated heart rate (float)
        fft_freqs: frequency axis (ndarray)
        fft_power: power spectrum (ndarray)
        snr: peak signal-to-noise ratio (float)
        prominence: peak prominence above noise (float)
    """
    n = len(signal)
    if n < 2:
        return 0.0, np.array([]), np.array([]), 0.0, 0.0

    # Linear detrend before bandpass to reduce spectral smearing
    signal = detrend(signal, type="linear")

    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0
    b, a = butter_bandpass(low_hz, high_hz, fps)
    filtered = filtfilt(b, a, signal)

    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    fft_vals = np.abs(np.fft.rfft(filtered))

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return 0.0, freqs, fft_vals, 0.0, 0.0

    valid_freqs = freqs[mask]
    valid_power = fft_vals[mask]
    peak_idx = np.argmax(valid_power)

    # Parabolic interpolation for sub-bin BPM accuracy
    refined_idx = parabolic_interpolation(valid_power, peak_idx)
    refined_idx = max(0.0, min(float(len(valid_freqs) - 1), refined_idx))
    freq_step = valid_freqs[1] - valid_freqs[0] if len(valid_freqs) > 1 else 1.0
    peak_freq = valid_freqs[0] + refined_idx * freq_step
    peak_freq = max(low_hz, min(high_hz, peak_freq))

    # 2x harmonic disambiguation: if the dominant peak looks like a
    # sub-harmonic (breathing artifact), check whether 2*f has significant
    # power and prefer it if so.
    if harmonic_disambiguation:
        peak_freq_2x = 2.0 * peak_freq
        if low_hz <= peak_freq_2x <= high_hz:
            idx_2x = np.argmin(np.abs(valid_freqs - peak_freq_2x))
            power_at_2x = valid_power[idx_2x]
            power_at_peak = valid_power[peak_idx]
            if power_at_peak > 1e-10 and power_at_2x >= harmonic_ratio * power_at_peak:
                peak_idx = idx_2x
                peak_freq = peak_freq_2x

    bpm = peak_freq * 60.0

    snr, prominence = compute_fft_snr(valid_power, peak_idx)

    return bpm, freqs, fft_vals, snr, prominence


# ---------------------------------------------------------------------------
# CHROM — Chrominance-based rPPG (de Haan & Jeanne, 2013)
# ---------------------------------------------------------------------------

def chrom(rgbs, winsize=45):
    """CHROM algorithm (xovery variant).

    X = R - G, Y = 0.5R + 0.5G - B.
    Causal rolling mean of X/Y ratio — fully vectorized via cumsum.

    Args:
        rgbs: Nx3 array of mean [R, G, B] from face ROI
        winsize: moving average window (frames)

    Returns:
        signal: 1D rPPG signal (length N)
    """
    n = len(rgbs)
    xs = rgbs[:, 0] - rgbs[:, 1]
    ys = 0.5 * rgbs[:, 0] + 0.5 * rgbs[:, 1] - rgbs[:, 2]

    idx     = np.arange(n)
    starts  = np.maximum(0, idx - winsize + 1)
    counts  = idx - starts + 1

    xs_cs   = np.concatenate([[0.0], np.cumsum(xs)])
    ys_cs   = np.concatenate([[0.0], np.cumsum(ys)])
    xs_mean = (xs_cs[idx + 1] - xs_cs[starts]) / counts
    ys_mean = (ys_cs[idx + 1] - ys_cs[starts]) / counts

    ys_safe = np.where(np.abs(ys_mean) > 1e-10, ys_mean, 1.0)
    return xs_mean / ys_safe - 1.0


# ---------------------------------------------------------------------------
# POS — Plane-Orthogonal-to-Skin (Wang et al., 2017)
# ---------------------------------------------------------------------------

def pos(rgbs, winsize=45):
    """POS algorithm.

    Projects temporal RGB variations onto a plane orthogonal to the skin
    tone direction. Adaptive alpha weighting based on std ratio.
    Fully vectorized via cumsum (no Python loops).

    Reference: Wang et al. (2017), "Algorithmic Principles of Remote PPG",
    IEEE Trans. Biomed. Eng.

    Args:
        rgbs: Nx3 array of mean [R, G, B] from face ROI
        winsize: temporal normalization window (frames)

    Returns:
        signal: 1D rPPG signal (length N)
    """
    n = len(rgbs)
    if n < 3:
        return np.zeros(n)

    idx    = np.arange(n)
    starts = np.maximum(0, idx - winsize + 1)
    counts = (idx - starts + 1).reshape(-1, 1)   # Nx1

    # Causal rolling mean per channel via cumsum
    cs     = np.vstack([np.zeros((1, 3)), np.cumsum(rgbs, axis=0)])  # (N+1)x3
    means  = (cs[idx + 1] - cs[starts]) / counts                      # Nx3
    means  = np.where(means < 1e-6, 1.0, means)

    normalized = rgbs / means   # Nx3

    S1 = normalized[:, 1] - normalized[:, 2]                           # G - B
    S2 = normalized[:, 1] + normalized[:, 2] - 2.0 * normalized[:, 0] # G + B - 2R

    # Causal rolling std for adaptive alpha via E[x²] - E[x]²
    counts1d = counts.ravel()

    def _rolling_std(v):
        vc  = np.concatenate([[0.0], np.cumsum(v)])
        vc2 = np.concatenate([[0.0], np.cumsum(v ** 2)])
        mu  = (vc[idx + 1]  - vc[starts])  / counts1d
        mu2 = (vc2[idx + 1] - vc2[starts]) / counts1d
        return np.sqrt(np.maximum(0.0, mu2 - mu ** 2))

    std_S1 = _rolling_std(S1)
    std_S2 = _rolling_std(S2)
    std_S2_safe = np.where(std_S2 > 1e-10, std_S2, 1.0)
    alpha       = np.where(std_S2 > 1e-10, std_S1 / std_S2_safe, 0.0)

    return S1 + alpha * S2


# ---------------------------------------------------------------------------
# GREEN — Trivial baseline (green channel only)
# ---------------------------------------------------------------------------

def green(rgbs, _winsize=None):
    """GREEN channel baseline.

    Simply uses mean green channel intensity as the rPPG signal.
    The bandpass filter in estimate_bpm handles frequency isolation.

    Args:
        rgbs: Nx3 array of mean [R, G, B] from face ROI
        winsize: unused (kept for API consistency)

    Returns:
        signal: 1D rPPG signal (length N)
    """
    g = rgbs[:, 1].copy()
    g -= np.mean(g)  # detrend (remove DC)
    return g


# ---------------------------------------------------------------------------
# ICA — Independent Component Analysis (Poh et al. 2010)
# ---------------------------------------------------------------------------

def ica(rgbs, _winsize=None):
    """ICA-based rPPG.

    Decomposes Nx3 RGB into 3 statistically independent components via
    FastICA. Selects the component with the most power in the HR band
    (42-180 BPM, fps≈30 assumed for component selection only).

    Args:
        rgbs: Nx3 array of mean [R, G, B] from face ROI
        winsize: unused (kept for API consistency)

    Returns:
        signal: 1D rPPG signal (length N)

    Raises:
        ImportError: if scikit-learn is not installed
    """
    n = len(rgbs)
    if n < 15:
        return np.zeros(n)
    from sklearn.decomposition import FastICA  # Let ImportError propagate

    centered = rgbs - rgbs.mean(axis=0)
    try:
        model = FastICA(n_components=3, random_state=42, max_iter=500, tol=0.01)
        comps = model.fit_transform(centered)           # shape: Nx3
    except Exception as exc:
        warnings.warn(f"[rppg] ICA convergence failure: {exc}")
        return np.zeros(n)

    # Pick component with highest energy in 42–180 BPM band (fps≈30 for selection)
    fps_est = 30.0
    low_hz, high_hz = 42 / 60.0, 180 / 60.0
    best_comp, best_power = comps[:, 0], -1.0
    for i in range(3):
        c = detrend(comps[:, i], type="linear")
        freqs = np.fft.rfftfreq(n, d=1.0 / fps_est)
        pwr   = np.abs(np.fft.rfft(c)) ** 2
        mask  = (freqs >= low_hz) & (freqs <= high_hz)
        band_power = float(np.sum(pwr[mask])) if mask.any() else 0.0
        if band_power > best_power:
            best_power, best_comp = band_power, c
    return best_comp


# ---------------------------------------------------------------------------
# WAVELET — CWT Morlet ridge extraction
# ---------------------------------------------------------------------------

def wavelet(rgbs, _winsize=None):
    """Wavelet-based rPPG using Morlet CWT on GREEN channel.

    Computes the CWT and extracts the instantaneous dominant frequency
    via ridge tracking (scale with max power at each time step).

    Args:
        rgbs: Nx3 array of mean [R, G, B] from face ROI
        winsize: unused (kept for API consistency)

    Returns:
        signal: 1D rPPG signal (length N)

    Raises:
        ImportError: if PyWavelets is not installed
    """
    n = len(rgbs)
    if n < 15:
        return np.zeros(n)
    import pywt  # Let ImportError propagate

    g = rgbs[:, 1].astype(float)
    g -= np.mean(g)

    # Scales covering approx 42–180 BPM at 30 fps for Morlet wavelet
    scales = np.arange(4, 40)
    try:
        coef, _ = pywt.cwt(g, scales, "morl")          # shape: (n_scales, n)
        power   = np.abs(coef) ** 2
        ridge_idx = np.argmax(power, axis=0)            # dominant scale per time step
        signal = np.array([coef[ridge_idx[t], t].real for t in range(n)])
        return signal
    except Exception as exc:
        warnings.warn(f"[rppg] Wavelet CWT failed: {exc}")
        return np.zeros(n)


# ---------------------------------------------------------------------------
# Registry — algorithm name → function mapping
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "chrom":   chrom,
    "pos":     pos,
    "green":   green,
    "ica":     ica,
    "wavelet": wavelet,
}


def median_smooth(values, kernel=3):
    """Rolling median filter to reject isolated BPM spikes."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        return arr.copy()
    half = kernel // 2
    # Pad with edge values so every window has exactly `kernel` elements
    padded = np.pad(arr, (half, half), mode="edge")
    # Build (n, kernel) view via stride tricks — no Python loop
    shape = (n, kernel)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.median(windows, axis=1)


def interpolate_motion_frames(rgbs, motion_flags):
    """Replace motion-flagged frames with linearly interpolated RGB values.

    Args:
        rgbs: Nx3 array of mean [R, G, B]
        motion_flags: N-length boolean list/array (True = motion detected)

    Returns:
        cleaned: Nx3 array with flagged frames interpolated
    """
    cleaned = rgbs.copy()
    flags = np.asarray(motion_flags, dtype=bool)
    n = len(rgbs)
    if n < 2:
        return cleaned

    flagged = np.where(flags)[0]
    clean = np.where(~flags)[0]

    # If too few clean frames (<40%), interpolation is unreliable
    if len(clean) < 2 or len(flagged) / n > 0.60:
        return cleaned

    for ch in range(3):
        cleaned[flagged, ch] = np.interp(flagged, clean, rgbs[clean, ch])

    return cleaned


def compute_bpm_timeseries(rgbs, timestamps, fps, algorithm="all",
                           window_s=30.0, step_s=5.0, motion_flags=None,
                           snr_consensus_threshold=2.0,
                           low_bpm=60, high_bpm=180,
                           harmonic_disambiguation=True, harmonic_ratio=0.6):
    """Run rPPG algorithm(s) and produce per-window BPM estimates.

    Args:
        rgbs: Nx3 array of mean [R, G, B] from forehead ROI
        timestamps: N-length array of frame timestamps (seconds)
        fps: effective frames per second (face-detected frames only)
        algorithm: "chrom", "pos", "green", "ica", "wavelet", or "all"
        window_s: BPM estimation window in seconds
        step_s: window step in seconds
        motion_flags: optional N-length bool array (True = motion detected)
        snr_consensus_threshold: minimum SNR for an algorithm to vote in CONSENSUS
        low_bpm: minimum detectable BPM (default 60)
        high_bpm: maximum detectable BPM (default 180)
        harmonic_disambiguation: check 2*f when dominant peak may be sub-harmonic
        harmonic_ratio: power threshold for preferring the 2x harmonic

    Returns:
        results: list of dicts with keys:
            window_idx, t_start, t_center, t_end, algorithm,
            bpm, bpm_smoothed, bpm_plausible, snr, prominence,
            motion_pct, n_face_frames, bpm_spread
            (CONSENSUS rows also have n_eligible)
    """
    algos = ALGORITHMS if algorithm == "all" else {algorithm: ALGORITHMS[algorithm]}
    winsize_chrom = max(int(fps * 1.5), 15)

    window_frames = int(window_s * fps)
    step_frames = max(int(step_s * fps), 1)

    has_motion = motion_flags is not None and len(motion_flags) == len(rgbs)
    m_flags = np.asarray(motion_flags, dtype=bool) if has_motion else None

    # Interpolate motion-corrupted frames before algorithm processing
    if has_motion and m_flags.any():
        rgbs = interpolate_motion_frames(rgbs, m_flags)

    results = []
    for algo_name, algo_fn in algos.items():
        try:
            ws = winsize_chrom if algo_name in ("chrom", "pos") else None
            rppg_signal = algo_fn(rgbs, winsize=ws) if ws is not None else algo_fn(rgbs)
        except ImportError as exc:
            warnings.warn(f"[rppg] Skipping {algo_name}: {exc}")
            continue
        except Exception as exc:
            warnings.warn(f"[rppg] {algo_name} failed: {exc}")
            continue

        algo_rows = []
        window_idx = 0
        for start in range(0, len(rppg_signal) - window_frames + 1, step_frames):
            end = start + window_frames
            segment = rppg_signal[start:end]

            t_start = timestamps[start]
            t_end = timestamps[min(end - 1, len(timestamps) - 1)]
            t_center = timestamps[start + window_frames // 2]

            bpm, _, _, snr, prominence = estimate_bpm(
                segment, fps, low_bpm=low_bpm, high_bpm=high_bpm,
                harmonic_disambiguation=harmonic_disambiguation,
                harmonic_ratio=harmonic_ratio,
            )

            motion_pct = 0.0
            if has_motion:
                motion_pct = float(m_flags[start:end].sum()) / window_frames

            algo_rows.append({
                "window_idx": window_idx,
                "t_start": round(t_start, 2),
                "t_center": round(t_center, 2),
                "t_end": round(t_end, 2),
                "algorithm": algo_name.upper(),
                "bpm": round(bpm, 1),
                "bpm_smoothed": 0.0,
                "bpm_plausible": 1 if low_bpm <= bpm <= high_bpm else 0,
                "snr": round(snr, 2),
                "prominence": round(prominence, 2),
                "motion_pct": round(motion_pct, 3),
                "n_face_frames": window_frames,
                "bpm_spread": 0.0,
                "n_eligible": 0,
            })
            window_idx += 1

        if algo_rows:
            raw_bpms = np.array([r["bpm"] for r in algo_rows])
            smoothed = median_smooth(raw_bpms, kernel=5)
            for i, row in enumerate(algo_rows):
                row["bpm_smoothed"] = round(float(smoothed[i]), 1)

        results.extend(algo_rows)

    # CONSENSUS: SNR-weighted mean, excluding algorithms below SNR threshold
    if algorithm == "all" and results:
        from collections import defaultdict
        by_window = defaultdict(list)
        for r in results:
            by_window[r["window_idx"]].append(r)

        consensus_rows = []
        for widx in sorted(by_window.keys()):
            rows = by_window[widx]

            eligible = [r for r in rows if r["snr"] >= snr_consensus_threshold]
            if eligible:
                snrs = np.array([r["snr"] for r in eligible])
                bpms = np.array([r["bpm_smoothed"] for r in eligible])
                weights = snrs / snrs.sum()
                consensus_bpm = float(np.dot(weights, bpms))
                spread = float(max(bpms) - min(bpms))
                mean_snr = float(np.mean(snrs))
            else:
                # Fallback to median of all if none pass SNR threshold
                bpms_all = [r["bpm_smoothed"] for r in rows]
                consensus_bpm = float(np.median(bpms_all))
                spread = float(max(bpms_all) - min(bpms_all))
                mean_snr = float(np.mean([r["snr"] for r in rows]))

            mpcts = [r["motion_pct"] for r in rows]
            n_eligible = len(eligible)

            consensus_rows.append({
                "window_idx": widx,
                "t_start": rows[0]["t_start"],
                "t_center": rows[0]["t_center"],
                "t_end": rows[0]["t_end"],
                "algorithm": "CONSENSUS",
                "bpm": round(consensus_bpm, 1),
                "bpm_smoothed": round(consensus_bpm, 1),
                "bpm_plausible": 1 if low_bpm <= consensus_bpm <= high_bpm else 0,
                "snr": round(mean_snr, 2),
                "prominence": 0.0,
                "motion_pct": round(float(np.mean(mpcts)), 3),
                "n_face_frames": rows[0]["n_face_frames"],
                "bpm_spread": round(spread, 1),
                "n_eligible": n_eligible,
            })
        results.extend(consensus_rows)

    return results
