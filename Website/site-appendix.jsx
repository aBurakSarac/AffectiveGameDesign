/* site-appendix.jsx — tail report sections built from the final report:
 *   §2 literature · §3 gaps + SWOT · §5.9 latency/cost ·
 *   §8.2 contributions · §8.3 limitations · §8.4 future · §8.5 conclusion ·
 *   §A/§B glossary (plain ↔ technical toggle, mirroring the HUD label mode).
 * Uses SiteIcon (from site-components.jsx) and the polling in-view hook.
 */
const { useState: useStateA, useRef: useRefA, useEffect: useEffectA } = React;

function useSeenA(margin = 0.85) {
  const ref = useRefA(null);
  const [seen, setSeen] = useStateA(false);
  useEffectA(() => {
    if (seen) return; let done = false;
    const check = () => { if (done) return; const el = ref.current; if (!el) return;
      if (el.getBoundingClientRect().top < window.innerHeight * margin) { done = true; setSeen(true); } };
    window.addEventListener("scroll", check, { passive: true });
    window.addEventListener("resize", check);
    const id = setInterval(check, 200); check();
    return () => { window.removeEventListener("scroll", check); window.removeEventListener("resize", check); clearInterval(id); };
  }, [seen, margin]);
  return [ref, seen];
}

/* ── BACKGROUND: literature + gaps + SWOT ────────────────────────────── */
const LIT = [
  { ic: "eye", t: "Facial emotion recognition", refs: [
    { c: "Li et al.", y: "2025", d: "Fair comparison of 12 FER algorithms — best (Poster) 75.98%, HOG-SVM 50.12%. <b>Chose an EfficientNet-B0 backbone.</b>" },
    { c: "Savchenko", y: "2025", d: "HSEmotion at ABAW-8; temporal smoothing adds +3.93% F1. <b>Adopted a 50-frame rolling average.</b>" },
    { c: "Abdeldayem et al.", y: "2026", d: "Fear is the hardest class — it overlaps surprise (AU 4 vs AU 5). <b>Motivated the arousal channel.</b>" },
  ]},
  { ic: "pulse", t: "rPPG & low-light sensing", refs: [
    { c: "Acharya et al.", y: "2025", d: "POS reaches 1.1 BPM MAE, beating deep nets — but degrades at high heart rate. <b>POS as primary; z-score normalization.</b>" },
    { c: "Wang et al.", y: "2017", d: "Algorithm hierarchy ends at POS (highest SNR, no skin-tone prior). <b>POS suits coloured monitor light.</b>" },
    { c: "Moghimi & Grailu", y: "2024", d: "Mid-forehead is the optimal ROI; exclude the lower face under expression. <b>Upper-face ROI only.</b>" },
  ]},
  { ic: "fuse", t: "Multimodal fusion", refs: [
    { c: "Karani & Desai", y: "2022", d: "Decision-level fusion is most robust when modalities run at different rates (FER 30 fps, rPPG ~0.33 Hz). <b>F15 fuses at the decision level.</b>" },
    { c: "Yan et al.", y: "2024", d: "Each added modality gives consistent accuracy gains. <b>Supports a multimodal approach.</b>" },
  ]},
  { ic: "shield", t: "Biofeedback & privacy", refs: [
    { c: "Moschovitis & Denisova", y: "2022", d: "Caroline: a Relax-to-Win horror game on a single contact heart-rate sensor. <b>The gap we fill: multimodal, webcam-only.</b>" },
    { c: "Yang et al.", y: "2026", d: "Blendshape Distribution Alignment — <1 ms per-subject calibration. <b>Lead candidate for future work.</b>" },
    { c: "Gutiérrez et al.", y: "2025", d: "Federated learning + AES-256 hits 87% — privacy and accuracy aren't exclusive. <b>Validates fully local processing.</b>" },
  ]},
];
const GAPS = [
  { n: "G1", t: "FER robustness in low light", d: "Benchmarks run under controlled light; reliability under a horror game's monitor-only lighting is undocumented." },
  { n: "G2", t: "The rPPG stress paradox", d: "rPPG accuracy drops exactly when heart rate spikes — the most informative moments for an adaptive game." },
  { n: "G3", t: "Mismatched temporal resolutions", d: "Fusing an instant facial signal with a slow physiological one has not been validated in a game context." },
  { n: "G4", t: "Single-signal biofeedback", d: "Existing game biofeedback relies on one contact sensor; none combine FER + rPPG from a single webcam." },
];
const SWOT = {
  s: ["Pipeline is independent — testable in isolation", "Modular architecture with graceful degradation", "Fully local processing (privacy)", "Mature open-source tools (MediaPipe, OpenCV)", "Very low MediaPipe latency (~6 ms)"],
  w: ["Single-camera dependence — no backup sensor", "Small expected sample limits statistical power", "Limited control over the player's face lighting"],
  o: ["Growing webcam-biometrics adoption (Affectiva, iMotions)", "Architecture reusable for therapeutic apps", "Continuous valence–arousal complements discrete classes", "MediaPipe blendshapes enable personalized metrics"],
  t: ["rPPG may lack sensitivity for fear-induced cardiac change", "FER low-light degradation may exceed thresholds", "Participant recruitment difficulty", "Risk of exceeding the 100 ms latency budget"],
};

function BackgroundSection() {
  return (
    <section className="section-block divline" id="background">
      <div className="wrap">
        <span className="kicker reveal">Background</span>
        <h2 className="sec-title reveal">What the literature already knew — and didn't</h2>
        <p className="sec-lead reveal">
          The review follows the pipeline's own logic: facial emotion recognition, remote photoplethysmography,
          multimodal fusion, biofeedback, and privacy. Each result below shaped a concrete design decision.
        </p>
        <div className="lit-grid">
          {LIT.map((g) => (
            <div className="lit-card reveal" key={g.t}>
              <div className="lc-h"><span className="lc-ic"><SiteIcon name={g.ic} s={18} /></span><span className="lc-t">{g.t}</span></div>
              <div className="lit-refs">
                {g.refs.map((r) => (
                  <div className="lref" key={r.c}>
                    <span className="lr-cite">{r.c}<small>{r.y}</small></span>
                    <span className="lr-d" dangerouslySetInnerHTML={{ __html: r.d }} />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <span className="kicker k-plain reveal" style={{ color: "var(--ink-4)", marginTop: "clamp(40px,6vw,64px)", display: "inline-flex" }}>Four cumulative gaps</span>
        <div className="gaps">
          {GAPS.map((g) => (
            <div className="gap reveal" key={g.n}>
              <div className="gp-n">{g.n}</div><div className="gp-t">{g.t}</div><div className="gp-d">{g.d}</div>
            </div>
          ))}
        </div>

        <div className="swot">
          {[["s", "Strengths", SWOT.s], ["w", "Weaknesses", SWOT.w], ["o", "Opportunities", SWOT.o], ["t", "Threats", SWOT.t]].map(([k, lab, items]) => (
            <div className={`swot-q ${k} reveal`} key={k}>
              <div className="sq-h">{lab}</div>
              <ul>{items.map((it, i) => <li key={i}>{it}</li>)}</ul>
            </div>
          ))}
        </div>
        <div className="swot-note reveal">
          <b>Retrospective:</b> the hardware failure turned the “small sample” weakness and the “recruitment”
          threat into a video-based evaluation that ultimately produced a more diverse subject pool.
        </div>
      </div>
    </section>
  );
}

/* ── PERFORMANCE: latency budget ─────────────────────────────────────── */
const LAT = [
  { nm: "Camera capture", ms: 33, note: "frame period @ 30 FPS" },
  { nm: "Haar + MP detect", ms: 9, note: "hybrid face detection" },
  { nm: "HSEmotion", ms: 10, note: "EfficientNet-B0, CPU" },
  { nm: "Formula", ms: 0.5, note: "arithmetic only" },
  { nm: "Socket emit", ms: 1, note: "TCP localhost" },
];
function PerformanceSection() {
  const [ref, seen] = useSeenA();
  const max = 33;
  const col = (ms) => ms >= 20 ? "var(--instrument)" : ms >= 8 ? "var(--arousal)" : "var(--clear)";
  return (
    <section className="section-block divline" id="performance">
      <div className="wrap">
        <span className="kicker reveal">Performance &amp; cost</span>
        <h2 className="sec-title reveal">~53 ms, end to end, on a consumer laptop</h2>
        <p className="sec-lead reveal">
          The latency budget (measured on a 197 s session at 30 FPS) sits well inside the 100 ms real-time
          target. The camera's own frame period is the largest single cost; with pipelined capture the
          effective throughput reaches 27–30 FPS.
        </p>
        <div className="lat-wrap" ref={ref}>
          <div>
            <div className="lat-bar">
              {LAT.map((r) => (
                <div className="lat-row" key={r.nm}>
                  <span className="lr-nm">{r.nm}</span>
                  <span className="lr-track"><span className="lr-fill" style={{ width: seen ? `${(r.ms / max) * 100}%` : 0, background: col(r.ms) }} /></span>
                  <span className="lr-ms">{r.ms < 1 ? "<1" : r.ms}</span>
                </div>
              ))}
            </div>
            <div className="lat-total">
              <span className="lt-v">53<span style={{ fontSize: "0.5em", color: "var(--ink-3)" }}>ms</span></span>
              <span className="lt-k">total per-frame latency · ~19 FPS (27–30 pipelined)</span>
              <span className="lt-budget">budget: &lt;100 ms ✓</span>
            </div>
          </div>
          <div className="lat-side">
            <h4>CPU utilization</h4>
            <div className="ls-row"><span className="ls-k">MediaPipe (C++ runtime)</span><span className="ls-v">4%</span></div>
            <div className="ls-row"><span className="ls-k">HSEmotion (PyTorch, CPU)</span><span className="ls-v">67.5%</span></div>
            <div className="ls-note">No CUDA under Windows during offline processing. The rPPG pipeline runs
              on a non-blocking background thread — ROI extraction ~1 ms/frame, FFT every 3 s.</div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── OUTLOOK: contributions / limitations / future / conclusion ──────── */
const CONTRIB = [
  { t: "Composite fear formula F12", d: "A multiplicative multimodal score where MediaPipe tension amplifies but never triggers fear. F1 = 0.70 over 6 subjects / 88 events." },
  { t: "rPPG cardiac fear response", d: "First empirical evidence that horror-game fear produces a detectable webcam-rPPG cardiac response (d = 0.70, FWER p = 0.029)." },
  { t: "Hybrid low-light detection", d: "A Haar + MediaPipe fallback that raises face coverage from 30.4% to 99.9% with no classification artefacts." },
  { t: "Open-source real-time pipeline", d: "A complete Python FER + rPPG + TCP pipeline at ~53 ms, ready to drop into any game engine." },
  { t: "Event-level evaluation method", d: "A sliding-window detection framework with temporal padding and false-positive proximity analysis, tuned to affective-game granularity." },
];
const LIMITS = [
  { t: "Sample size", d: "Six subjects, single annotator; no inter-annotator agreement. Insufficient for population-level claims." },
  { t: "Ecological validity", d: "Videos processed offline, not live — validates detection accuracy, not closed-loop dynamics." },
  { t: "No gameplay experiment", d: "The hardware failure blocked the planned two-condition study; RQ3 is unanswered." },
  { t: "MediaPipe variance", d: "Blendshape activations stay sparse across subjects; no per-subject calibration was implemented." },
  { t: "Single-camera dependence", d: "Momentary facial occlusion causes signal loss with no sensor-level fallback." },
  { t: "Post-hoc rPPG config", d: "The winning window was found by sweep, not pre-registered. Survives FWER but needs replication." },
];
const FUTURE = [
  { t: "Live gameplay study", d: "Finish v2.0 and run the closed-loop experiment to answer RQ3, validating the rPPG finding on a larger sample." },
  { t: "Cross-subject calibration", d: "Implement Blendshape Distribution Alignment (BDA) to normalize MediaPipe activation ranges per player." },
  { t: "Cardiac ground truth", d: "Pair webcam rPPG with a contact PPG sensor to capture high-resolution heart-rate-variability metrics." },
];
function OutlookSection() {
  return (
    <section className="section-block divline" id="outlook">
      <div className="wrap">
        <span className="kicker reveal">Contributions &amp; outlook</span>
        <h2 className="sec-title reveal">What it proved, and what's next</h2>
        <div className="outlook-grid">
          <div className="ol-col contrib reveal">
            <h3><span className="ol-ic"><SiteIcon name="shield" s={17} /></span>Contributions</h3>
            <div className="ol-list">
              {CONTRIB.map((c, i) => (
                <div className="ol-item" key={i}>
                  <span className="oi-n">{String(i + 1).padStart(2, "0")}</span>
                  <span><span className="oi-t">{c.t}</span><span className="oi-d">{c.d}</span></span>
                </div>
              ))}
            </div>
          </div>
          <div className="ol-col limit reveal">
            <h3><span className="ol-ic"><SiteIcon name="info" s={17} /></span>Limitations</h3>
            <div className="ol-list">
              {LIMITS.map((c, i) => (
                <div className="ol-item" key={i}>
                  <span className="oi-n">{String(i + 1).padStart(2, "0")}</span>
                  <span><span className="oi-t">{c.t}</span><span className="oi-d">{c.d}</span></span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="future reveal">
          <span className="kicker k-plain" style={{ color: "var(--ink-4)" }}>Future work — three axes</span>
          <div className="future-grid">
            {FUTURE.map((f) => (
              <div className="fut" key={f.t}><div className="ft-t">{f.t}</div><div className="ft-d">{f.d}</div></div>
            ))}
          </div>
        </div>

        <div className="conclusion reveal">
          <div className="cc-k">Conclusion</div>
          <div className="cc-t">The technical barrier to affective gaming is <em>no longer the hardware.</em></div>
          <div className="cc-d">
            A standard webcam, two open-source models and a composite formula running in 53 ms on a consumer
            laptop are enough to sense fear. What remains is evaluation methodology, cross-subject calibration,
            and the closed-loop validation that a live gameplay experiment will provide.
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── GLOSSARY: plain ↔ technical ─────────────────────────────────────── */
const GLOSS = [
  { t: "rPPG", ab: "remote photoplethysmography",
    plain: "Reading your pulse from video. A webcam sees tiny colour changes in your skin as blood moves with each heartbeat — no sensor touches you.",
    tech: "Contactless PPG from an RGB camera: sub-pixel chromatic variation in facial skin is extracted from temporal fluctuations of mean ROI pixel values; the cardiac-band periodic component yields BPM." },
  { t: "POS", ab: "plane-orthogonal-to-skin",
    plain: "The most accurate pulse method here — ~3 BPM error vs a smartwatch. It mathematically separates blood-flow colour change from movement and lighting.",
    tech: "Projects the RGB signal onto a plane orthogonal to the mean skin-tone direction, removing intensity variation with no skin-tone prior. Highest SNR in Wang et al. (5.16 dB); MAE = 2.96 BPM here." },
  { t: "CONSENSUS", ab: "multi-algorithm blend",
    plain: "An attempt to combine all five pulse methods. It backfired: the weak methods dragged down the accurate one (POS), making the blend worse than POS alone.",
    tech: "SNR-weighted average of five extractors. Weak algorithms (GREEN, WAVELET >13 BPM MAE) diluted POS, collapsing the cardiac fear effect (d = 0.042) until POS was isolated (d = 0.696)." },
  { t: "CLAHE", ab: "adaptive histogram equalization",
    plain: "A contrast booster for dark video. It brightens the face region so detail hidden in shadow becomes readable before emotion is measured.",
    tech: "Contrast-Limited Adaptive Histogram Equalization on the L channel (clipLimit 2.0, 4×4 tiles) — local equalization with contrast clipping to avoid noise amplification." },
  { t: "FER", ab: "facial emotion recognition",
    plain: "Software that reads emotion from a face. This project uses two: a neural model (HSEmotion) and a geometric face tracker (MediaPipe).",
    tech: "HSEmotion (EfficientNet-B0 / AffectNet) gives 8-class softmax probabilities; MediaPipe yields 52 facial blendshape activations for geometric tension." },
  { t: "Blendshapes", ab: "MediaPipe AU proxies",
    plain: "52 sliders describing facial muscle movements — brow raise, jaw clench, eye widen. Together they sketch an expression.",
    tech: "Per-frame activation coefficients for facial action proxies; fear-critical channels show median activations of 0.001–0.004, structurally too sparse to threshold alone." },
  { t: "F1 score", ab: "precision × recall",
    plain: "One number for detector quality, balancing false alarms against missed events. 1.0 is perfect; this system reaches 0.77.",
    tech: "Harmonic mean of precision and recall, evaluated at the event level via sliding-window matching with temporal padding." },
  { t: "Cohen's d", ab: "effect size",
    plain: "How big a difference is, not just whether it exists. ~0.7 means the fear heart-rate response is a solid, medium-to-large effect.",
    tech: "Standardized mean difference. The isolated-POS cardiac fear response reached d = 0.696 (FWER-corrected p = 0.029)." },
  { t: "FFT", ab: "fast Fourier transform",
    plain: "A tool that finds repeating rhythms in a signal — like picking out which notes make up a chord. It locates the heartbeat frequency in the skin-colour signal.",
    tech: "O(n log n) discrete Fourier transform on windowed rPPG segments; the dominant peak in the 1.0–3.0 Hz cardiac band gives the BPM estimate." },
  { t: "Relax-to-Win", ab: "the core mechanic",
    plain: "The game gets scarier when you're scared. Your only defence is to calm down — self-regulation is the controller.",
    tech: "A positive-feedback biofeedback loop: detected fear (F12/F15 > θ) escalates enemy state; the player must suppress visible fear to de-escalate." },
];
function GlossarySection() {
  const [mode, setMode] = useStateA("plain");
  return (
    <section className="section-block divline" id="glossary">
      <div className="wrap">
        <span className="kicker reveal">Glossary</span>
        <h2 className="sec-title reveal">Every term, two ways</h2>
        <div className="gloss-head reveal">
          <p className="sec-lead" style={{ marginTop: 0, flex: "1 1 320px" }}>
            The same plain ↔ technical switch the HUD uses, applied to the vocabulary of the report.
          </p>
          <div className="gloss-toggle">
            <button className={mode === "plain" ? "on" : ""} onClick={() => setMode("plain")}>Plain language</button>
            <button className={mode === "tech" ? "on" : ""} onClick={() => setMode("tech")}>Technical</button>
          </div>
        </div>
        <div className="gloss-grid">
          {GLOSS.map((g) => (
            <div className="gterm" key={g.t}>
              <div className="gt-h"><span className="gt-t">{g.t}</span><span className="gt-ab">{g.ab}</span></div>
              <div className={`gt-d ${mode === "tech" ? "tech" : ""}`}>{mode === "tech" ? g.tech : g.plain}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { BackgroundSection, PerformanceSection, OutlookSection, GlossarySection });
