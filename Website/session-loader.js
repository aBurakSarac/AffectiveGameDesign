/* session-loader.js — real per-session data, replacing hud-signals.js for the site.
 *
 * Provides the same window.HUD API (DURATION, THRESH, computeFrame, sampleTrace)
 * but backed by precomputed per-frame JSON from export_session_json.py.
 *
 * Usage from site-app.jsx:
 *   await window.HUD.loadSession("S06_Vid16");  // fetches frames.json
 *   const f = window.HUD.computeFrame(t);         // binary search in real data
 */
(function () {
  "use strict";

  const THRESH = { F12: 0.70, F15: 0.80 };
  const EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                          "Happiness", "Neutral", "Sadness", "Surprise"];
  const RPPG_ALGOS = ["CHROM", "POS", "GREEN", "ICA", "WAVELET", "CONSENSUS"];
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  let _frames = [];
  let _duration = 1;
  let _fps = 30;
  let _loaded = null;
  let _videoWidth = 290;
  let _videoHeight = 240;
  const _cache = {};

  const _fallback = {
    t: 0, frame: 0, latency: 0, fps: 30,
    hs_fear: 0, hs_arousal: 0,
    emotions: Object.fromEntries(EMOTION_LABELS.map((e) => [e, e === "Neutral" ? 1 : 0])),
    dom: "Neutral", domScore: 1,
    mp_tension: 0, valence: 0, smile: 0, startle: 0,
    hint: "—", hintKind: "idle",
    bpm: 0, bpm_norm: 0, baseline: 74,
    algos: {},
    base: 0, mp_mult: 1, rppg_mult: 1,
    F12: 0, F15: 0, isFear: false,
  };

  function computeFrame(t) {
    if (!_frames.length) return { ..._fallback, t: t || 0 };
    t = Math.max(0, Math.min(_duration, t));
    let lo = 0, hi = _frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (_frames[mid].t < t) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(_frames[lo - 1].t - t) < Math.abs(_frames[lo].t - t)) lo--;
    return _frames[lo];
  }

  function sampleTrace(key, step) {
    step = step || 0.2;
    if (!_frames.length) return [{ t: 0, v: 0 }];
    const pts = [];
    for (let t = 0; t <= _duration + 1e-6; t += step) {
      const f = computeFrame(t);
      pts.push({ t, v: f[key] != null ? f[key] : 0 });
    }
    return pts;
  }

  async function loadSession(stem) {
    if (_cache[stem]) {
      _frames = _cache[stem].frames;
      _duration = _cache[stem].duration;
      _fps = _cache[stem].fps;
      _videoWidth = _cache[stem].videoWidth;
      _videoHeight = _cache[stem].videoHeight;
      _loaded = stem;
      return;
    }
    const resp = await fetch("media/sessions/" + stem + "/frames.json?v=2");
    if (!resp.ok) throw new Error("Failed to load session " + stem + ": " + resp.status);
    const data = await resp.json();
    _frames = data.frames;
    _duration = data.duration;
    _fps = data.fps || 30;
    _videoWidth = data.videoWidth || 290;
    _videoHeight = data.videoHeight || 240;
    _cache[stem] = {
      frames: _frames,
      duration: _duration,
      fps: _fps,
      videoWidth: _videoWidth,
      videoHeight: _videoHeight,
    };
    _loaded = stem;
  }

  window.HUD = {
    get DURATION() { return _duration; },
    get FPS() { return _fps; },
    THRESH: THRESH,
    BPM_BASELINE: 74,
    RPPG_COEFF: 0.5,
    EMOTION_LABELS: EMOTION_LABELS,
    RPPG_ALGOS: RPPG_ALGOS,
    computeFrame: computeFrame,
    sampleTrace: sampleTrace,
    clamp01: clamp01,
    loadSession: loadSession,
    getVideoUrl: function (stem) { return "media/sessions/" + stem + "/video.mp4"; },
    get loaded() { return _loaded; },
    get videoWidth() { return _videoWidth; },
    get videoHeight() { return _videoHeight; },
  };
})();
