const { useState: useStateA, useRef: useRefA, useEffect: useEffectA } = React;
const tA = (k, fb) => window.I18N.t(k, fb);
function useSeenA(margin = 0.85) {
  const ref = useRefA(null);
  const [seen, setSeen] = useStateA(false);
  useEffectA(() => {
    if (seen) return;
    let done = false;
    const check = () => {
      if (done) return;
      const el = ref.current;
      if (!el) return;
      if (el.getBoundingClientRect().top < window.innerHeight * margin) {
        done = true;
        setSeen(true);
      }
    };
    window.addEventListener("scroll", check, { passive: true });
    window.addEventListener("resize", check);
    const id = setInterval(check, 200);
    check();
    return () => {
      window.removeEventListener("scroll", check);
      window.removeEventListener("resize", check);
      clearInterval(id);
    };
  }, [seen, margin]);
  return [ref, seen];
}
function getLIT() {
  return [
    { ic: "eye", t: tA("bg.lit.0.t"), refs: [
      { c: "Li et al.", y: "2025", d: tA("bg.lit.0.r0.d") },
      { c: "Savchenko", y: "2025", d: tA("bg.lit.0.r1.d") },
      { c: "Abdeldayem et al.", y: "2026", d: tA("bg.lit.0.r2.d") }
    ] },
    { ic: "pulse", t: tA("bg.lit.1.t"), refs: [
      { c: "Acharya et al.", y: "2025", d: tA("bg.lit.1.r0.d") },
      { c: "Wang et al.", y: "2017", d: tA("bg.lit.1.r1.d") },
      { c: "Moghimi & Grailu", y: "2024", d: tA("bg.lit.1.r2.d") }
    ] },
    { ic: "fuse", t: tA("bg.lit.2.t"), refs: [
      { c: "Karani & Desai", y: "2022", d: tA("bg.lit.2.r0.d") },
      { c: "Yan et al.", y: "2024", d: tA("bg.lit.2.r1.d") }
    ] },
    { ic: "shield", t: tA("bg.lit.3.t"), refs: [
      { c: "Moschovitis & Denisova", y: "2022", d: tA("bg.lit.3.r0.d") },
      { c: "Yang et al.", y: "2026", d: tA("bg.lit.3.r1.d") },
      { c: "Guti\xE9rrez et al.", y: "2025", d: tA("bg.lit.3.r2.d") }
    ] }
  ];
}
function getGAPS() {
  return [
    { n: "G1", t: tA("bg.gap.0.t"), d: tA("bg.gap.0.d") },
    { n: "G2", t: tA("bg.gap.1.t"), d: tA("bg.gap.1.d") },
    { n: "G3", t: tA("bg.gap.2.t"), d: tA("bg.gap.2.d") },
    { n: "G4", t: tA("bg.gap.3.t"), d: tA("bg.gap.3.d") }
  ];
}
function getSWOT() {
  return {
    s: [tA("bg.swot.s.0"), tA("bg.swot.s.1"), tA("bg.swot.s.2"), tA("bg.swot.s.3"), tA("bg.swot.s.4")],
    w: [tA("bg.swot.w.0"), tA("bg.swot.w.1"), tA("bg.swot.w.2")],
    o: [tA("bg.swot.o.0"), tA("bg.swot.o.1"), tA("bg.swot.o.2"), tA("bg.swot.o.3")],
    t: [tA("bg.swot.t.0"), tA("bg.swot.t.1"), tA("bg.swot.t.2"), tA("bg.swot.t.3")]
  };
}
function BackgroundSection() {
  const LIT = getLIT(), GAPS = getGAPS(), SWOT = getSWOT();
  return /* @__PURE__ */ React.createElement("section", { className: "section-block divline", id: "background" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, tA("bg.kicker", "Background")), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, tA("bg.title", "What the literature already knew \u2014 and didn't")), /* @__PURE__ */ React.createElement("p", { className: "sec-lead reveal" }, tA("bg.lead", "The review follows the pipeline's own logic: facial emotion recognition, remote photoplethysmography, multimodal fusion, biofeedback, and privacy. Each result below shaped a concrete design decision.")), /* @__PURE__ */ React.createElement("div", { className: "lit-grid" }, LIT.map((g) => /* @__PURE__ */ React.createElement("div", { className: "lit-card reveal", key: g.t }, /* @__PURE__ */ React.createElement("div", { className: "lc-h" }, /* @__PURE__ */ React.createElement("span", { className: "lc-ic" }, /* @__PURE__ */ React.createElement(SiteIcon, { name: g.ic, s: 18 })), /* @__PURE__ */ React.createElement("span", { className: "lc-t" }, g.t)), /* @__PURE__ */ React.createElement("div", { className: "lit-refs" }, g.refs.map((r) => /* @__PURE__ */ React.createElement("div", { className: "lref", key: r.c }, /* @__PURE__ */ React.createElement("span", { className: "lr-cite" }, r.c, /* @__PURE__ */ React.createElement("small", null, r.y)), /* @__PURE__ */ React.createElement("span", { className: "lr-d", dangerouslySetInnerHTML: { __html: r.d } }))))))), /* @__PURE__ */ React.createElement("span", { className: "kicker k-plain reveal", style: { color: "var(--ink-4)", marginTop: "clamp(40px,6vw,64px)", display: "inline-flex" } }, tA("bg.gapsKicker", "Four cumulative gaps")), /* @__PURE__ */ React.createElement("div", { className: "gaps" }, GAPS.map((g) => /* @__PURE__ */ React.createElement("div", { className: "gap reveal", key: g.n }, /* @__PURE__ */ React.createElement("div", { className: "gp-n" }, g.n), /* @__PURE__ */ React.createElement("div", { className: "gp-t" }, g.t), /* @__PURE__ */ React.createElement("div", { className: "gp-d" }, g.d)))), /* @__PURE__ */ React.createElement("div", { className: "swot" }, [["s", tA("bg.swot.strengths", "Strengths"), SWOT.s], ["w", tA("bg.swot.weaknesses", "Weaknesses"), SWOT.w], ["o", tA("bg.swot.opportunities", "Opportunities"), SWOT.o], ["t", tA("bg.swot.threats", "Threats"), SWOT.t]].map(([k, lab, items]) => /* @__PURE__ */ React.createElement("div", { className: `swot-q ${k} reveal`, key: k }, /* @__PURE__ */ React.createElement("div", { className: "sq-h" }, lab), /* @__PURE__ */ React.createElement("ul", null, items.map((it, i) => /* @__PURE__ */ React.createElement("li", { key: i }, it)))))), /* @__PURE__ */ React.createElement("div", { className: "swot-note reveal", dangerouslySetInnerHTML: { __html: tA("bg.swotNote", "<b>Retrospective:</b> the hardware failure turned the \u201Csmall sample\u201D weakness and the \u201Crecruitment\u201D threat into a video-based evaluation that ultimately produced a more diverse subject pool.") } })));
}
function getLAT() {
  return [
    { nm: tA("perf.lat.0.nm"), ms: 33, note: tA("perf.lat.0.note") },
    { nm: tA("perf.lat.1.nm"), ms: 9, note: tA("perf.lat.1.note") },
    { nm: tA("perf.lat.2.nm"), ms: 10, note: tA("perf.lat.2.note") },
    { nm: tA("perf.lat.3.nm"), ms: 0.5, note: tA("perf.lat.3.note") },
    { nm: tA("perf.lat.4.nm"), ms: 1, note: tA("perf.lat.4.note") }
  ];
}
function PerformanceSection() {
  const [ref, seen] = useSeenA();
  const LAT = getLAT();
  const max = 33;
  const col = (ms) => ms >= 20 ? "var(--instrument)" : ms >= 8 ? "var(--arousal)" : "var(--clear)";
  return /* @__PURE__ */ React.createElement("section", { className: "section-block divline", id: "performance" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, tA("perf.kicker", "Performance & cost")), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, tA("perf.title", "~53 ms, end to end, on a consumer laptop")), /* @__PURE__ */ React.createElement("p", { className: "sec-lead reveal" }, tA("perf.lead", "The latency budget (measured on a 197 s session at 30 FPS) sits well inside the 100 ms real-time target. The camera's own frame period is the largest single cost; with pipelined capture the effective throughput reaches 27\u201330 FPS.")), /* @__PURE__ */ React.createElement("div", { className: "lat-wrap", ref }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("div", { className: "lat-bar" }, LAT.map((r) => /* @__PURE__ */ React.createElement("div", { className: "lat-row", key: r.nm }, /* @__PURE__ */ React.createElement("span", { className: "lr-nm" }, r.nm), /* @__PURE__ */ React.createElement("span", { className: "lr-track" }, /* @__PURE__ */ React.createElement("span", { className: "lr-fill", style: { width: seen ? `${r.ms / max * 100}%` : 0, background: col(r.ms) } })), /* @__PURE__ */ React.createElement("span", { className: "lr-ms" }, r.ms < 1 ? "<1" : r.ms)))), /* @__PURE__ */ React.createElement("div", { className: "lat-total" }, /* @__PURE__ */ React.createElement("span", { className: "lt-v" }, "53", /* @__PURE__ */ React.createElement("span", { style: { fontSize: "0.5em", color: "var(--ink-3)" } }, "ms")), /* @__PURE__ */ React.createElement("span", { className: "lt-k" }, tA("perf.latTotal", "total per-frame latency \xB7 ~19 FPS (27\u201330 pipelined)")), /* @__PURE__ */ React.createElement("span", { className: "lt-budget" }, tA("perf.latBudget", "budget: <100 ms \u2713")))), /* @__PURE__ */ React.createElement("div", { className: "lat-side" }, /* @__PURE__ */ React.createElement("h4", null, tA("perf.cpuTitle", "CPU utilization")), /* @__PURE__ */ React.createElement("div", { className: "ls-row" }, /* @__PURE__ */ React.createElement("span", { className: "ls-k" }, tA("perf.cpu1", "MediaPipe (C++ runtime)")), /* @__PURE__ */ React.createElement("span", { className: "ls-v" }, "4%")), /* @__PURE__ */ React.createElement("div", { className: "ls-row" }, /* @__PURE__ */ React.createElement("span", { className: "ls-k" }, tA("perf.cpu2", "HSEmotion (PyTorch, CPU)")), /* @__PURE__ */ React.createElement("span", { className: "ls-v" }, "67.5%")), /* @__PURE__ */ React.createElement("div", { className: "ls-note" }, tA("perf.cpuNote", "No CUDA under Windows during offline processing. The rPPG pipeline runs on a non-blocking background thread \u2014 ROI extraction ~1 ms/frame, FFT every 3 s."))))));
}
function getCONTRIB() {
  return [
    { t: tA("outlook.contrib.0.t"), d: tA("outlook.contrib.0.d") },
    { t: tA("outlook.contrib.1.t"), d: tA("outlook.contrib.1.d") },
    { t: tA("outlook.contrib.2.t"), d: tA("outlook.contrib.2.d") },
    { t: tA("outlook.contrib.3.t"), d: tA("outlook.contrib.3.d") },
    { t: tA("outlook.contrib.4.t"), d: tA("outlook.contrib.4.d") }
  ];
}
function getLIMITS() {
  return [
    { t: tA("outlook.limit.0.t"), d: tA("outlook.limit.0.d") },
    { t: tA("outlook.limit.1.t"), d: tA("outlook.limit.1.d") },
    { t: tA("outlook.limit.2.t"), d: tA("outlook.limit.2.d") },
    { t: tA("outlook.limit.3.t"), d: tA("outlook.limit.3.d") },
    { t: tA("outlook.limit.4.t"), d: tA("outlook.limit.4.d") },
    { t: tA("outlook.limit.5.t"), d: tA("outlook.limit.5.d") }
  ];
}
function getFUTURE() {
  return [
    { t: tA("outlook.future.0.t"), d: tA("outlook.future.0.d") },
    { t: tA("outlook.future.1.t"), d: tA("outlook.future.1.d") },
    { t: tA("outlook.future.2.t"), d: tA("outlook.future.2.d") }
  ];
}
function OutlookSection() {
  const CONTRIB = getCONTRIB(), LIMITS = getLIMITS(), FUTURE = getFUTURE();
  return /* @__PURE__ */ React.createElement("section", { className: "section-block divline", id: "outlook" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, tA("outlook.kicker", "Contributions & outlook")), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, tA("outlook.title", "What it proved, and what's next")), /* @__PURE__ */ React.createElement("div", { className: "outlook-grid" }, /* @__PURE__ */ React.createElement("div", { className: "ol-col contrib reveal" }, /* @__PURE__ */ React.createElement("h3", null, /* @__PURE__ */ React.createElement("span", { className: "ol-ic" }, /* @__PURE__ */ React.createElement(SiteIcon, { name: "shield", s: 17 })), tA("outlook.contribTitle", "Contributions")), /* @__PURE__ */ React.createElement("div", { className: "ol-list" }, CONTRIB.map((c, i) => /* @__PURE__ */ React.createElement("div", { className: "ol-item", key: i }, /* @__PURE__ */ React.createElement("span", { className: "oi-n" }, String(i + 1).padStart(2, "0")), /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("span", { className: "oi-t" }, c.t), /* @__PURE__ */ React.createElement("span", { className: "oi-d" }, c.d)))))), /* @__PURE__ */ React.createElement("div", { className: "ol-col limit reveal" }, /* @__PURE__ */ React.createElement("h3", null, /* @__PURE__ */ React.createElement("span", { className: "ol-ic" }, /* @__PURE__ */ React.createElement(SiteIcon, { name: "info", s: 17 })), tA("outlook.limitTitle", "Limitations")), /* @__PURE__ */ React.createElement("div", { className: "ol-list" }, LIMITS.map((c, i) => /* @__PURE__ */ React.createElement("div", { className: "ol-item", key: i }, /* @__PURE__ */ React.createElement("span", { className: "oi-n" }, String(i + 1).padStart(2, "0")), /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("span", { className: "oi-t" }, c.t), /* @__PURE__ */ React.createElement("span", { className: "oi-d" }, c.d))))))), /* @__PURE__ */ React.createElement("div", { className: "future reveal" }, /* @__PURE__ */ React.createElement("span", { className: "kicker k-plain", style: { color: "var(--ink-4)" } }, tA("outlook.futureKicker", "Future work \u2014 three axes")), /* @__PURE__ */ React.createElement("div", { className: "future-grid" }, FUTURE.map((f) => /* @__PURE__ */ React.createElement("div", { className: "fut", key: f.t }, /* @__PURE__ */ React.createElement("div", { className: "ft-t" }, f.t), /* @__PURE__ */ React.createElement("div", { className: "ft-d" }, f.d))))), /* @__PURE__ */ React.createElement("div", { className: "conclusion reveal" }, /* @__PURE__ */ React.createElement("div", { className: "cc-k" }, tA("outlook.conclTag", "Conclusion")), /* @__PURE__ */ React.createElement("div", { className: "cc-t", dangerouslySetInnerHTML: { __html: tA("outlook.conclTitle", "The technical barrier to affective gaming is <em>no longer the hardware.</em>") } }), /* @__PURE__ */ React.createElement("div", { className: "cc-d" }, tA("outlook.conclDesc", "A standard webcam, two open-source models and a composite formula running in 53 ms on a consumer laptop are enough to sense fear. What remains is evaluation methodology, cross-subject calibration, and the closed-loop validation that a live gameplay experiment will provide.")))));
}
function getGLOSS() {
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
    { t: "Relax-to-Win", ab: tA("gloss.term.9.ab"), plain: tA("gloss.term.9.plain"), tech: tA("gloss.term.9.tech") }
  ];
}
function GlossarySection() {
  const [mode, setMode] = useStateA("plain");
  const GLOSS = getGLOSS();
  return /* @__PURE__ */ React.createElement("section", { className: "section-block divline", id: "glossary" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, tA("gloss.kicker", "Glossary")), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, tA("gloss.title", "Every term, two ways")), /* @__PURE__ */ React.createElement("div", { className: "gloss-head reveal" }, /* @__PURE__ */ React.createElement("p", { className: "sec-lead", style: { marginTop: 0, flex: "1 1 320px" } }, tA("gloss.lead", "The same plain \u2194 technical switch the HUD uses, applied to the vocabulary of the report.")), /* @__PURE__ */ React.createElement("div", { className: "gloss-toggle" }, /* @__PURE__ */ React.createElement("button", { className: mode === "plain" ? "on" : "", onClick: () => setMode("plain") }, tA("gloss.plainBtn", "Plain language")), /* @__PURE__ */ React.createElement("button", { className: mode === "tech" ? "on" : "", onClick: () => setMode("tech") }, tA("gloss.techBtn", "Technical")))), /* @__PURE__ */ React.createElement("div", { className: "gloss-grid" }, GLOSS.map((g) => /* @__PURE__ */ React.createElement("div", { className: "gterm", key: g.t }, /* @__PURE__ */ React.createElement("div", { className: "gt-h" }, /* @__PURE__ */ React.createElement("span", { className: "gt-t" }, g.t), /* @__PURE__ */ React.createElement("span", { className: "gt-ab" }, g.ab)), /* @__PURE__ */ React.createElement("div", { className: `gt-d ${mode === "tech" ? "tech" : ""}` }, mode === "tech" ? g.tech : g.plain))))));
}
Object.assign(window, { BackgroundSection, PerformanceSection, OutlookSection, GlossarySection });
