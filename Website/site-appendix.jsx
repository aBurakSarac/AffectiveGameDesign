/* site-appendix.jsx — tail report sections built from the final report:
 *   §2 literature · §3 gaps + SWOT · §5.9 latency/cost ·
 *   §8.2 contributions · §8.3 limitations · §8.4 future · §8.5 conclusion ·
 *   §A/§B glossary (plain ↔ technical toggle, mirroring the HUD label mode).
 * Uses SiteIcon (from site-components.jsx) and the polling in-view hook.
 *
 * i18n: the data sets are builder functions (not module constants) so they
 * re-resolve from the current language each render. Author citations, years,
 * acronyms and numbers stay literal; prose is keyed in lang/*.js.
 */
const { useState: useStateA, useRef: useRefA, useEffect: useEffectA } = React;
const tA = (k, fb) => window.I18N.t(k, fb);

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
function getLIT() {
  return [
    { ic: "eye", t: tA("bg.lit.0.t"), refs: [
      { c: "Li et al.", y: "2025", d: tA("bg.lit.0.r0.d") },
      { c: "Savchenko", y: "2025", d: tA("bg.lit.0.r1.d") },
      { c: "Abdeldayem et al.", y: "2026", d: tA("bg.lit.0.r2.d") },
    ]},
    { ic: "pulse", t: tA("bg.lit.1.t"), refs: [
      { c: "Acharya et al.", y: "2025", d: tA("bg.lit.1.r0.d") },
      { c: "Wang et al.", y: "2017", d: tA("bg.lit.1.r1.d") },
      { c: "Moghimi & Grailu", y: "2024", d: tA("bg.lit.1.r2.d") },
    ]},
    { ic: "fuse", t: tA("bg.lit.2.t"), refs: [
      { c: "Karani & Desai", y: "2022", d: tA("bg.lit.2.r0.d") },
      { c: "Yan et al.", y: "2024", d: tA("bg.lit.2.r1.d") },
    ]},
    { ic: "shield", t: tA("bg.lit.3.t"), refs: [
      { c: "Moschovitis & Denisova", y: "2022", d: tA("bg.lit.3.r0.d") },
      { c: "Yang et al.", y: "2026", d: tA("bg.lit.3.r1.d") },
      { c: "Gutiérrez et al.", y: "2025", d: tA("bg.lit.3.r2.d") },
    ]},
  ];
}
function getGAPS() {
  return [
    { n: "G1", t: tA("bg.gap.0.t"), d: tA("bg.gap.0.d") },
    { n: "G2", t: tA("bg.gap.1.t"), d: tA("bg.gap.1.d") },
    { n: "G3", t: tA("bg.gap.2.t"), d: tA("bg.gap.2.d") },
    { n: "G4", t: tA("bg.gap.3.t"), d: tA("bg.gap.3.d") },
  ];
}
function getSWOT() {
  return {
    s: [tA("bg.swot.s.0"), tA("bg.swot.s.1"), tA("bg.swot.s.2"), tA("bg.swot.s.3"), tA("bg.swot.s.4")],
    w: [tA("bg.swot.w.0"), tA("bg.swot.w.1"), tA("bg.swot.w.2")],
    o: [tA("bg.swot.o.0"), tA("bg.swot.o.1"), tA("bg.swot.o.2"), tA("bg.swot.o.3")],
    t: [tA("bg.swot.t.0"), tA("bg.swot.t.1"), tA("bg.swot.t.2"), tA("bg.swot.t.3")],
  };
}

function BackgroundSection() {
  const LIT = getLIT(), GAPS = getGAPS(), SWOT = getSWOT();
  return (
    <section className="section-block divline" id="background">
      <div className="wrap">
        <span className="kicker reveal">{tA("bg.kicker", "Background")}</span>
        <h2 className="sec-title reveal">{tA("bg.title", "What the literature already knew — and didn't")}</h2>
        <p className="sec-lead reveal">
          {tA("bg.lead", "The review follows the pipeline's own logic: facial emotion recognition, remote photoplethysmography, multimodal fusion, biofeedback, and privacy. Each result below shaped a concrete design decision.")}
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

        <span className="kicker k-plain reveal" style={{ color: "var(--ink-4)", marginTop: "clamp(40px,6vw,64px)", display: "inline-flex" }}>{tA("bg.gapsKicker", "Four cumulative gaps")}</span>
        <div className="gaps">
          {GAPS.map((g) => (
            <div className="gap reveal" key={g.n}>
              <div className="gp-n">{g.n}</div><div className="gp-t">{g.t}</div><div className="gp-d">{g.d}</div>
            </div>
          ))}
        </div>

        <div className="swot">
          {[["s", tA("bg.swot.strengths", "Strengths"), SWOT.s], ["w", tA("bg.swot.weaknesses", "Weaknesses"), SWOT.w], ["o", tA("bg.swot.opportunities", "Opportunities"), SWOT.o], ["t", tA("bg.swot.threats", "Threats"), SWOT.t]].map(([k, lab, items]) => (
            <div className={`swot-q ${k} reveal`} key={k}>
              <div className="sq-h">{lab}</div>
              <ul>{items.map((it, i) => <li key={i}>{it}</li>)}</ul>
            </div>
          ))}
        </div>
        <div className="swot-note reveal" dangerouslySetInnerHTML={{ __html: tA("bg.swotNote", "<b>Retrospective:</b> the hardware failure turned the “small sample” weakness and the “recruitment” threat into a video-based evaluation that ultimately produced a more diverse subject pool.") }} />
      </div>
    </section>
  );
}

/* ── PERFORMANCE: latency budget ─────────────────────────────────────── */
function getLAT() {
  return [
    { nm: tA("perf.lat.0.nm"), ms: 33, note: tA("perf.lat.0.note") },
    { nm: tA("perf.lat.1.nm"), ms: 9, note: tA("perf.lat.1.note") },
    { nm: tA("perf.lat.2.nm"), ms: 10, note: tA("perf.lat.2.note") },
    { nm: tA("perf.lat.3.nm"), ms: 0.5, note: tA("perf.lat.3.note") },
    { nm: tA("perf.lat.4.nm"), ms: 1, note: tA("perf.lat.4.note") },
  ];
}
function PerformanceSection() {
  const [ref, seen] = useSeenA();
  const LAT = getLAT();
  const max = 33;
  const col = (ms) => ms >= 20 ? "var(--instrument)" : ms >= 8 ? "var(--arousal)" : "var(--clear)";
  return (
    <section className="section-block divline" id="performance">
      <div className="wrap">
        <span className="kicker reveal">{tA("perf.kicker", "Performance & cost")}</span>
        <h2 className="sec-title reveal">{tA("perf.title", "~53 ms, end to end, on a consumer laptop")}</h2>
        <p className="sec-lead reveal">
          {tA("perf.lead", "The latency budget (measured on a 197 s session at 30 FPS) sits well inside the 100 ms real-time target. The camera's own frame period is the largest single cost; with pipelined capture the effective throughput reaches 27–30 FPS.")}
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
              <span className="lt-k">{tA("perf.latTotal", "total per-frame latency · ~19 FPS (27–30 pipelined)")}</span>
              <span className="lt-budget">{tA("perf.latBudget", "budget: <100 ms ✓")}</span>
            </div>
          </div>
          <div className="lat-side">
            <h4>{tA("perf.cpuTitle", "CPU utilization")}</h4>
            <div className="ls-row"><span className="ls-k">{tA("perf.cpu1", "MediaPipe (C++ runtime)")}</span><span className="ls-v">4%</span></div>
            <div className="ls-row"><span className="ls-k">{tA("perf.cpu2", "HSEmotion (PyTorch, CPU)")}</span><span className="ls-v">67.5%</span></div>
            <div className="ls-note">{tA("perf.cpuNote", "No CUDA under Windows during offline processing. The rPPG pipeline runs on a non-blocking background thread — ROI extraction ~1 ms/frame, FFT every 3 s.")}</div>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── OUTLOOK: contributions / limitations / future / conclusion ──────── */
function getCONTRIB() {
  return [
    { t: tA("outlook.contrib.0.t"), d: tA("outlook.contrib.0.d") },
    { t: tA("outlook.contrib.1.t"), d: tA("outlook.contrib.1.d") },
    { t: tA("outlook.contrib.2.t"), d: tA("outlook.contrib.2.d") },
    { t: tA("outlook.contrib.3.t"), d: tA("outlook.contrib.3.d") },
    { t: tA("outlook.contrib.4.t"), d: tA("outlook.contrib.4.d") },
  ];
}
function getLIMITS() {
  return [
    { t: tA("outlook.limit.0.t"), d: tA("outlook.limit.0.d") },
    { t: tA("outlook.limit.1.t"), d: tA("outlook.limit.1.d") },
    { t: tA("outlook.limit.2.t"), d: tA("outlook.limit.2.d") },
    { t: tA("outlook.limit.3.t"), d: tA("outlook.limit.3.d") },
    { t: tA("outlook.limit.4.t"), d: tA("outlook.limit.4.d") },
    { t: tA("outlook.limit.5.t"), d: tA("outlook.limit.5.d") },
  ];
}
function getFUTURE() {
  return [
    { t: tA("outlook.future.0.t"), d: tA("outlook.future.0.d") },
    { t: tA("outlook.future.1.t"), d: tA("outlook.future.1.d") },
    { t: tA("outlook.future.2.t"), d: tA("outlook.future.2.d") },
  ];
}
function OutlookSection() {
  const CONTRIB = getCONTRIB(), LIMITS = getLIMITS(), FUTURE = getFUTURE();
  return (
    <section className="section-block divline" id="outlook">
      <div className="wrap">
        <span className="kicker reveal">{tA("outlook.kicker", "Contributions & outlook")}</span>
        <h2 className="sec-title reveal">{tA("outlook.title", "What it proved, and what's next")}</h2>
        <div className="outlook-grid">
          <div className="ol-col contrib reveal">
            <h3><span className="ol-ic"><SiteIcon name="shield" s={17} /></span>{tA("outlook.contribTitle", "Contributions")}</h3>
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
            <h3><span className="ol-ic"><SiteIcon name="info" s={17} /></span>{tA("outlook.limitTitle", "Limitations")}</h3>
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
          <span className="kicker k-plain" style={{ color: "var(--ink-4)" }}>{tA("outlook.futureKicker", "Future work — three axes")}</span>
          <div className="future-grid">
            {FUTURE.map((f) => (
              <div className="fut" key={f.t}><div className="ft-t">{f.t}</div><div className="ft-d">{f.d}</div></div>
            ))}
          </div>
        </div>

        <div className="conclusion reveal">
          <div className="cc-k">{tA("outlook.conclTag", "Conclusion")}</div>
          <div className="cc-t" dangerouslySetInnerHTML={{ __html: tA("outlook.conclTitle", "The technical barrier to affective gaming is <em>no longer the hardware.</em>") }} />
          <div className="cc-d">
            {tA("outlook.conclDesc", "A standard webcam, two open-source models and a composite formula running in 53 ms on a consumer laptop are enough to sense fear. What remains is evaluation methodology, cross-subject calibration, and the closed-loop validation that a live gameplay experiment will provide.")}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── GLOSSARY: plain ↔ technical ─────────────────────────────────────── */
function getGLOSS() {
  // term acronyms stay literal; ab/plain/tech are translated
  return [
    { t: "rPPG", ab: tA("gloss.term.0.ab"), plain: tA("gloss.term.0.plain"), tech: tA("gloss.term.0.tech") },
    { t: "POS", ab: tA("gloss.term.1.ab"), plain: tA("gloss.term.1.plain"), tech: tA("gloss.term.1.tech") },
    { t: "CONSENSUS", ab: tA("gloss.term.2.ab"), plain: tA("gloss.term.2.plain"), tech: tA("gloss.term.2.tech") },
    { t: "CLAHE", ab: tA("gloss.term.3.ab"), plain: tA("gloss.term.3.plain"), tech: tA("gloss.term.3.tech") },
    { t: "FER", ab: tA("gloss.term.4.ab"), plain: tA("gloss.term.4.plain"), tech: tA("gloss.term.4.tech") },
    { t: "Blendshapes", ab: tA("gloss.term.5.ab"), plain: tA("gloss.term.5.plain"), tech: tA("gloss.term.5.tech") },
    { t: "F1 score", ab: tA("gloss.term.6.ab"), plain: tA("gloss.term.6.plain"), tech: tA("gloss.term.6.tech") },
    { t: "Cohen's d", ab: tA("gloss.term.7.ab"), plain: tA("gloss.term.7.plain"), tech: tA("gloss.term.7.tech") },
    { t: "FFT", ab: tA("gloss.term.8.ab"), plain: tA("gloss.term.8.plain"), tech: tA("gloss.term.8.tech") },
    { t: "Relax-to-Win", ab: tA("gloss.term.9.ab"), plain: tA("gloss.term.9.plain"), tech: tA("gloss.term.9.tech") },
  ];
}
function GlossarySection() {
  const [mode, setMode] = useStateA("plain");
  const GLOSS = getGLOSS();
  return (
    <section className="section-block divline" id="glossary">
      <div className="wrap">
        <span className="kicker reveal">{tA("gloss.kicker", "Glossary")}</span>
        <h2 className="sec-title reveal">{tA("gloss.title", "Every term, two ways")}</h2>
        <div className="gloss-head reveal">
          <p className="sec-lead" style={{ marginTop: 0, flex: "1 1 320px" }}>
            {tA("gloss.lead", "The same plain ↔ technical switch the HUD uses, applied to the vocabulary of the report.")}
          </p>
          <div className="gloss-toggle">
            <button className={mode === "plain" ? "on" : ""} onClick={() => setMode("plain")}>{tA("gloss.plainBtn", "Plain language")}</button>
            <button className={mode === "tech" ? "on" : ""} onClick={() => setMode("tech")}>{tA("gloss.techBtn", "Technical")}</button>
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
