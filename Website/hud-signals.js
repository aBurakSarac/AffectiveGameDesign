/* hud-signals.js — mock session timeline + faithful F12/F15/bpm_norm math.
 *
 * Pure data + math, no DOM. Exposes window.HUD with:
 *   DURATION, FPS, THRESH, EMOTION_LABELS, computeFrame(t), sampleTrace(key, step)
 *
 * Math mirrors Pipeline/fer/fusion.py and Pipeline/rppg/evaluate_rppg.py:
 *   base = 0.7*hs_fear + 0.3*hs_arousal
 *   F12  = clamp( base * (1 + mp_tension) )
 *   bpm_norm = clip( (bpm - baseline)/baseline, 0, 1 )   // 60s rolling median ~ resting
 *   F15  = clamp( F12 * (1 + 0.5*bpm_norm) )
 *   verdict thresholds: F12 -> 0.70, F15 -> 0.80 (production)
 */
(function () {
  "use strict";

  const DURATION = 48;            // seconds
  const FPS = 30;
  const BPM_BASELINE = 74;        // resting (60s rolling median) BPM
  const RPPG_COEFF = 0.5;         // c in F15 = F12*(1 + c*bpm_norm)
  const THRESH = { F12: 0.70, F15: 0.80 };

  const EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                          "Happiness", "Neutral", "Sadness", "Surprise"];
  const RPPG_ALGOS = ["CHROM", "POS", "GREEN", "ICA", "WAVELET", "CONSENSUS"];

  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const smooth = (a) => a * a * (3 - 2 * a);             // smoothstep

  // Keyframe interpolation over [{t, v}] (v scalar or array). Smoothstep eased.
  function track(keys) {
    return function (t) {
      if (t <= keys[0].t) return keys[0].v;
      const last = keys[keys.length - 1];
      if (t >= last.t) return last.v;
      for (let i = 0; i < keys.length - 1; i++) {
        const a = keys[i], b = keys[i + 1];
        if (t >= a.t && t <= b.t) {
          const f = smooth((t - a.t) / (b.t - a.t));
          if (Array.isArray(a.v)) return a.v.map((av, k) => av + (b.v[k] - av) * f);
          return a.v + (b.v - a.v) * f;
        }
      }
      return last.v;
    };
  }

  // ── Raw input tracks ──────────────────────────────────────────────────
  // Emotion distribution (8-vector). Fear here IS hs_fear (single source).
  // [Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]
  const distTrack = track([
    { t: 0,  v: [0.03, 0.05, 0.02, 0.05, 0.10, 0.62, 0.08, 0.05] }, // calm
    { t: 6,  v: [0.03, 0.05, 0.02, 0.06, 0.10, 0.61, 0.08, 0.05] },
    { t: 11, v: [0.04, 0.03, 0.03, 0.64, 0.02, 0.10, 0.04, 0.10] }, // FALSE ALARM (fear face)
    { t: 14, v: [0.04, 0.03, 0.03, 0.62, 0.02, 0.12, 0.04, 0.10] },
    { t: 18, v: [0.03, 0.05, 0.02, 0.10, 0.12, 0.58, 0.07, 0.03] }, // recover
    { t: 24, v: [0.04, 0.04, 0.03, 0.22, 0.06, 0.49, 0.06, 0.06] },
    { t: 29, v: [0.05, 0.02, 0.04, 0.58, 0.02, 0.18, 0.04, 0.07] }, // genuine onset
    { t: 32, v: [0.06, 0.02, 0.04, 0.72, 0.01, 0.05, 0.03, 0.07] }, // GENUINE FEAR peak
    { t: 35, v: [0.06, 0.02, 0.04, 0.70, 0.01, 0.07, 0.03, 0.07] },
    { t: 40, v: [0.05, 0.03, 0.03, 0.30, 0.05, 0.42, 0.07, 0.05] }, // settle
    { t: 44, v: [0.04, 0.04, 0.03, 0.12, 0.08, 0.55, 0.09, 0.05] },
    { t: 48, v: [0.03, 0.05, 0.02, 0.07, 0.10, 0.60, 0.08, 0.05] },
  ]);

  const arousalTrack = track([
    { t: 0, v: 0.18 }, { t: 8, v: 0.20 }, { t: 11, v: 0.27 }, // stays LOW during false alarm
    { t: 15, v: 0.25 }, { t: 20, v: 0.16 }, { t: 26, v: 0.40 },
    { t: 30, v: 0.62 }, { t: 33, v: 0.66 }, { t: 36, v: 0.60 },
    { t: 40, v: 0.42 }, { t: 44, v: 0.22 }, { t: 48, v: 0.16 },
  ]);

  const tensionTrack = track([
    { t: 0, v: 0.07 }, { t: 8, v: 0.09 }, { t: 11, v: 0.16 },
    { t: 15, v: 0.14 }, { t: 20, v: 0.09 }, { t: 26, v: 0.18 },
    { t: 30, v: 0.23 }, { t: 33, v: 0.25 }, { t: 36, v: 0.22 },
    { t: 40, v: 0.15 }, { t: 44, v: 0.11 }, { t: 48, v: 0.08 },
  ]);

  const bpmTrack = track([
    { t: 0, v: 72 }, { t: 8, v: 73 }, { t: 12, v: 74 },   // flat during false alarm
    { t: 18, v: 73 }, { t: 24, v: 76 }, { t: 28, v: 85 },
    { t: 31, v: 92 }, { t: 34, v: 93 }, { t: 37, v: 88 },
    { t: 41, v: 81 }, { t: 44, v: 77 }, { t: 48, v: 74 },
  ]);

  const valenceTrack = track([
    { t: 0, v: 0.10 }, { t: 9, v: -0.18 }, { t: 14, v: -0.20 },
    { t: 18, v: 0.05 }, { t: 26, v: -0.30 }, { t: 33, v: -0.52 },
    { t: 40, v: -0.20 }, { t: 48, v: 0.05 },
  ]);

  const smileTrack = track([
    { t: 0, v: 0.06 }, { t: 18, v: 0.10 }, { t: 24, v: 0.02 },
    { t: 33, v: 0.0 }, { t: 48, v: 0.05 },
  ]);

  // brief startle spikes at the two onsets
  const startleTrack = track([
    { t: 0, v: 0.0 }, { t: 8.5, v: 0.0 }, { t: 9.2, v: 4.2 }, { t: 10.5, v: 0.6 },
    { t: 25, v: 0.4 }, { t: 27, v: 5.1 }, { t: 28.5, v: 1.2 }, { t: 33, v: 0.5 },
    { t: 48, v: 0.0 },
  ]);

  // small deterministic noise (no Math.random — reproducible per t)
  const noise = (t, seed) =>
    (Math.sin(t * 12.9898 + seed * 78.233) * 43758.5453) % 1;

  function computeFrame(t) {
    t = Math.max(0, Math.min(DURATION, t));
    const dist = distTrack(t).map((v) => Math.max(0, v));
    const sum = dist.reduce((a, b) => a + b, 0) || 1;
    const emotions = {};
    EMOTION_LABELS.forEach((lab, i) => (emotions[lab] = dist[i] / sum));

    const hs_fear = emotions.Fear;
    const hs_arousal = clamp01(arousalTrack(t));
    const mp_tension = clamp01(tensionTrack(t));
    const bpm = bpmTrack(t);

    // dominant emotion
    let dom = "Neutral", domScore = 0;
    EMOTION_LABELS.forEach((lab) => {
      if (emotions[lab] > domScore) { domScore = emotions[lab]; dom = lab; }
    });

    // ── formula chain ──
    const base = 0.7 * hs_fear + 0.3 * hs_arousal;
    const mp_mult = 1 + mp_tension;
    const F12 = clamp01(base * mp_mult);
    const bpm_norm = clamp01((bpm - BPM_BASELINE) / BPM_BASELINE);
    const rppg_mult = 1 + RPPG_COEFF * bpm_norm;
    const F15 = clamp01(F12 * rppg_mult);

    const isFear = F15 >= THRESH.F15;

    // MediaPipe tentative hint (was "State: [STRESS]") — NOT a verdict
    const valence = valenceTrack(t);
    const startle = Math.max(0, startleTrack(t));
    let hint = "—", hintKind = "idle";
    if (startle > 3) { hint = "startle?"; hintKind = "alert"; }
    else if (mp_tension > 0.18 && valence < -0.15) { hint = "stress?"; hintKind = "warn"; }
    else if (mp_tension > 0.12 && valence < 0) { hint = "tension?"; hintKind = "warn"; }

    // rPPG per-algorithm spread around POS (POS == headline bpm)
    const algos = {};
    const spread = { CHROM: 1.045, POS: 1.0, GREEN: 0.86, ICA: 0.955, WAVELET: 0.925, CONSENSUS: 0.998 };
    const snrBase = { CHROM: 4.1, POS: 4.0, GREEN: 2.7, ICA: 3.4, WAVELET: 3.8, CONSENSUS: 3.6 };
    RPPG_ALGOS.forEach((a, i) => {
      const jit = noise(t, i + 1) * 1.4;
      algos[a] = { bpm: bpm * spread[a] + jit, snr: snrBase[a] + noise(t, i + 9) * 0.4 };
    });

    // telemetry
    const frame = Math.floor(t * FPS);
    const latency = 23 + Math.abs(noise(t, 3)) * 14 + (startle > 3 ? 9 : 0);
    const fps = 30 - Math.abs(noise(t, 7)) * 1.6;

    return {
      t, frame, latency, fps,
      hs_fear, hs_arousal, emotions, dom, domScore,
      mp_tension, valence, smile: clamp01(smileTrack(t)), startle, hint, hintKind,
      bpm, bpm_norm, baseline: BPM_BASELINE, algos,
      base, mp_mult, rppg_mult, F12, F15, isFear,
    };
  }

  // Sample a derived scalar across the whole session (for the trace chart).
  function sampleTrace(key, step) {
    step = step || 0.2;
    const pts = [];
    for (let t = 0; t <= DURATION + 1e-6; t += step) {
      const f = computeFrame(t);
      pts.push({ t, v: f[key] });
    }
    return pts;
  }

  window.HUD = {
    DURATION, FPS, THRESH, BPM_BASELINE, RPPG_COEFF,
    EMOTION_LABELS, RPPG_ALGOS, computeFrame, sampleTrace, clamp01,
  };
})();
