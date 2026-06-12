const { useState: useStateM, useRef: useRefM, useEffect: useEffectM, useCallback: useCbM } = React;
const tMe = (k, fb) => window.I18N.t(k, fb);
function useInView(margin = 0.84) {
  const ref = useRefM(null);
  const [seen, setSeen] = useStateM(false);
  useEffectM(() => {
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
    const onScroll = () => check();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    const id = setInterval(check, 200);
    check();
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      clearInterval(id);
    };
  }, [seen, margin]);
  return [ref, seen];
}
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const pctM = (v) => `${clamp01(v) * 100}%`;
function DetectionFallback() {
  const [ref, seen] = useInView();
  const rows = [
    { nm: tMe("m1.rowBright", "Bright"), mp: 0.06, miss: 0.02, pc: tMe("m1.pcBright", "Haar handles most") },
    { nm: tMe("m1.rowDim", "Dim"), mp: 0.46, miss: 0.05, pc: tMe("m1.pcDim", "fallback on ~46%") },
    { nm: tMe("m1.rowMixed", "Mixed"), mp: 0.55, miss: 0.06, pc: tMe("m1.pcMixed", "fallback on ~55%") }
  ];
  return /* @__PURE__ */ React.createElement("div", { className: "mblock", ref }, /* @__PURE__ */ React.createElement("div", { className: "mcap" }, /* @__PURE__ */ React.createElement("div", { className: "m-no" }, tMe("methods.label", "METHOD"), " 01"), /* @__PURE__ */ React.createElement("h3", null, tMe("m1.title", "Hybrid detection that survives the dark")), /* @__PURE__ */ React.createElement("p", null, tMe("m1.p", "A fast Haar cascade crops the face every frame. When it loses the face in low light, a MediaPipe FaceLandmarker fallback takes over \u2014 so detection holds even as the room dims. The darker and more uneven the scene, the more the fallback earns its keep.")), /* @__PURE__ */ React.createElement("div", { className: "m-stat" }, /* @__PURE__ */ React.createElement("span", { className: "v" }, "99.9%"), /* @__PURE__ */ React.createElement("span", { className: "k" }, tMe("m1.stat", "face coverage across analyzed frames"))), /* @__PURE__ */ React.createElement("div", { className: "m-foot" }, tMe("m1.foot", "Per-condition shares bind to each session's detector log."))), /* @__PURE__ */ React.createElement("div", { className: "mvis" }, /* @__PURE__ */ React.createElement("div", { className: "mvis-head" }, /* @__PURE__ */ React.createElement("span", { className: "dot" }), /* @__PURE__ */ React.createElement("span", { className: "mh-t" }, tMe("m1.visTitle", "Detector by lighting")), " ", /* @__PURE__ */ React.createElement("span", { className: "mh-r" }, tMe("m1.visR", "per frame"))), /* @__PURE__ */ React.createElement("div", { className: "detect-stage" }, rows.map((r) => /* @__PURE__ */ React.createElement("div", { className: "detect-row", key: r.nm }, /* @__PURE__ */ React.createElement("div", { className: "dl" }, /* @__PURE__ */ React.createElement("span", { className: "nm" }, r.nm), /* @__PURE__ */ React.createElement("span", { className: "pc" }, r.pc)), /* @__PURE__ */ React.createElement("div", { className: "dbar" }, /* @__PURE__ */ React.createElement("div", { className: "seg-haar", style: { width: seen ? pctM(1 - r.mp - r.miss) : 0 } }, /* @__PURE__ */ React.createElement("span", { className: "seg-lab" }, Math.round((1 - r.mp - r.miss) * 100), "%")), /* @__PURE__ */ React.createElement("div", { className: "seg-mp", style: { width: seen ? pctM(r.mp) : 0 } }, r.mp > 0.12 ? /* @__PURE__ */ React.createElement("span", { className: "seg-lab" }, Math.round(r.mp * 100), "%") : null), /* @__PURE__ */ React.createElement("div", { className: "seg-miss", style: { width: seen ? pctM(r.miss) : 0 }, title: `${Math.round(r.miss * 100)}% missed by both` })))), /* @__PURE__ */ React.createElement("div", { className: "detect-legend" }, /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("i", { style: { background: "var(--clear)" } }), " ", tMe("m1.legHaar", "Haar cascade")), /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("i", { style: { background: "var(--instrument)" } }), " ", tMe("m1.legMp", "MediaPipe fallback")), /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("i", { className: "hatch" }), " ", tMe("m1.legMiss", "missed by both"))), /* @__PURE__ */ React.createElement("div", { className: "detect-note" }, tMe("m1.note", "A few frames are caught by neither detector \u2014 and that's expected. Recorded video has no guarantee a face is present 100% of the time: occlusion (hands, hair), extreme head turns, or edits that cut away from the camera all leave short gaps.")))));
}
const HIST_BEFORE = [33.8, 14.9, 12.4, 13.3, 14, 10.3, 1.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0];
const HIST_AFTER = [19.5, 20.5, 7.4, 6.8, 7, 6.7, 7.5, 7.4, 6.9, 4.8, 4.2, 1.2, 0, 0, 0, 0];
function Histo({ data, label, after }) {
  const max = Math.max(...data);
  return /* @__PURE__ */ React.createElement("div", { className: `histo ${after ? "after" : ""}` }, /* @__PURE__ */ React.createElement("div", { className: "ht" }, label), /* @__PURE__ */ React.createElement("div", { className: "bars" }, data.map((v, i) => /* @__PURE__ */ React.createElement("i", { key: i, style: { height: `${v / max * 100}%` } }))));
}
function ClaheDemo() {
  const [ref, seen] = useInView();
  const [wipe, setWipe] = useStateM(50);
  const viewRef = useRefM(null);
  const drag = useRefM(false);
  const set = useCbM((clientX) => {
    const el = viewRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setWipe(Math.max(2, Math.min(98, (clientX - r.left) / r.width * 100)));
  }, []);
  useEffectM(() => {
    if (!seen) return;
    let raf, t0 = performance.now(), active = true;
    const anim = (now) => {
      if (!active || drag.current) return;
      const el = (now - t0) / 1e3;
      if (el > 2.6) return;
      setWipe(50 + 32 * Math.sin(el * 2.2));
      raf = requestAnimationFrame(anim);
    };
    raf = requestAnimationFrame(anim);
    return () => {
      active = false;
      cancelAnimationFrame(raf);
    };
  }, [seen]);
  return /* @__PURE__ */ React.createElement("div", { className: "mblock flip", ref }, /* @__PURE__ */ React.createElement("div", { className: "mcap" }, /* @__PURE__ */ React.createElement("div", { className: "m-no" }, tMe("methods.label", "METHOD"), " 02"), /* @__PURE__ */ React.createElement("h3", null, tMe("m2.title", "CLAHE restores the face before reading it")), /* @__PURE__ */ React.createElement("p", null, tMe("m2.p", "Contrast-Limited Adaptive Histogram Equalization stretches the cramped, dark tones of the face ROI across the full range \u2014 pulling expression detail out of the shadows so the emotion model and the skin-colour pulse signal both get a clean read. Drag to compare.")), /* @__PURE__ */ React.createElement("div", { className: "m-stat" }, /* @__PURE__ */ React.createElement("span", { className: "v" }, "+"), /* @__PURE__ */ React.createElement("span", { className: "k" }, tMe("m2.stat", "recovered detail in dim & mixed light"))), /* @__PURE__ */ React.createElement("div", { className: "m-foot" }, tMe("m2.foot", "Real face crop from session S02_Vid04 (dim lighting). CLAHE: clipLimit 2.0, 4\xD74 tiles, LAB L-channel."))), /* @__PURE__ */ React.createElement("div", { className: "mvis" }, /* @__PURE__ */ React.createElement("div", { className: "mvis-head" }, /* @__PURE__ */ React.createElement("span", { className: "dot" }), /* @__PURE__ */ React.createElement("span", { className: "mh-t" }, tMe("m2.visTitle", "Face ROI \xB7 raw \u2192 CLAHE")), " ", /* @__PURE__ */ React.createElement("span", { className: "mh-r" }, tMe("m2.visR", "drag \u21C6"))), /* @__PURE__ */ React.createElement("div", { className: "clahe-wrap" }, /* @__PURE__ */ React.createElement(
    "div",
    {
      className: "clahe-view",
      ref: viewRef,
      style: { "--wipe": `${wipe}%` },
      onPointerDown: (e) => {
        drag.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        set(e.clientX);
      },
      onPointerMove: (e) => {
        if (drag.current) set(e.clientX);
      },
      onPointerUp: () => {
        drag.current = false;
      }
    },
    /* @__PURE__ */ React.createElement("div", { className: "clahe-face" }, /* @__PURE__ */ React.createElement("img", { src: "media/clahe/clahe_off.png", alt: tMe("m2.altRaw", "Raw face ROI") })),
    /* @__PURE__ */ React.createElement("div", { className: "clahe-face after" }, /* @__PURE__ */ React.createElement("img", { src: "media/clahe/clahe_on.png", alt: tMe("m2.altOn", "CLAHE enhanced") })),
    /* @__PURE__ */ React.createElement("div", { className: "clahe-roi" }, /* @__PURE__ */ React.createElement("span", { className: "rl" }, tMe("m2.roiLabel", "analysis ROI"))),
    /* @__PURE__ */ React.createElement("span", { className: "clahe-tag l" }, "RAW"),
    /* @__PURE__ */ React.createElement("span", { className: "clahe-tag r" }, "CLAHE"),
    /* @__PURE__ */ React.createElement("div", { className: "clahe-divider" }, /* @__PURE__ */ React.createElement("span", { className: "gp" }, "\u21C6"))
  ), /* @__PURE__ */ React.createElement("div", { className: "clahe-hist" }, /* @__PURE__ */ React.createElement(Histo, { data: HIST_BEFORE, label: tMe("m2.histRaw", "Raw histogram") }), /* @__PURE__ */ React.createElement(Histo, { data: seen ? HIST_AFTER : HIST_BEFORE, label: tMe("m2.histEq", "Equalized"), after: true })))));
}
function FusionLadder() {
  const [ref] = useInView();
  const [tension, setTension] = useStateM(0.24);
  const [hr, setHr] = useStateM(0.3);
  const fear = 0.66, arousal = 0.52;
  const base = 0.7 * fear + 0.3 * arousal;
  const f12 = Math.min(1, base * (1 + tension));
  const f15 = Math.min(1, f12 * (1 + 0.5 * hr));
  const isFear = f15 >= 0.8;
  const Step = ({ lab, val, color, thr }) => /* @__PURE__ */ React.createElement("div", { className: "fl-step" }, /* @__PURE__ */ React.createElement("span", { className: "sl", dangerouslySetInnerHTML: { __html: lab } }), /* @__PURE__ */ React.createElement("div", { className: "fl-track" }, /* @__PURE__ */ React.createElement("div", { className: "ff", style: { width: pctM(val), background: color } }), thr != null ? /* @__PURE__ */ React.createElement("div", { className: "thr-m", style: { left: pctM(thr) } }) : null), /* @__PURE__ */ React.createElement("span", { className: "sv", style: { color } }, val.toFixed(2)));
  return /* @__PURE__ */ React.createElement("div", { className: "mblock", ref }, /* @__PURE__ */ React.createElement("div", { className: "mcap" }, /* @__PURE__ */ React.createElement("div", { className: "m-no" }, tMe("methods.label", "METHOD"), " 03"), /* @__PURE__ */ React.createElement("h3", null, tMe("m3.title", "Fusion that multiplies, not averages")), /* @__PURE__ */ React.createElement("p", { dangerouslySetInnerHTML: { __html: tMe("m3.p", "Facial tension and a rising heart rate don't get averaged into the score \u2014 they act as <em> amplifiers</em> on the base fear reading. Because they multiply, agreement compounds: a calm body barely moves the needle, while genuine arousal pushes a true fear event clear over the line. Move the sliders to feel it.") } }), /* @__PURE__ */ React.createElement("div", { className: "m-formula" }, /* @__PURE__ */ React.createElement("div", { className: "mf-row" }, /* @__PURE__ */ React.createElement("span", { className: "mf-lhs" }, "base"), /* @__PURE__ */ React.createElement("span", { className: "mf-eq" }, "="), /* @__PURE__ */ React.createElement("span", { className: "mf-rhs" }, "0.7\xB7fear + 0.3\xB7arousal")), /* @__PURE__ */ React.createElement("div", { className: "mf-row" }, /* @__PURE__ */ React.createElement("span", { className: "mf-lhs" }, "F12"), /* @__PURE__ */ React.createElement("span", { className: "mf-eq" }, "="), /* @__PURE__ */ React.createElement("span", { className: "mf-rhs" }, "base \xB7 (1 + tension)")), /* @__PURE__ */ React.createElement("div", { className: "mf-row" }, /* @__PURE__ */ React.createElement("span", { className: "mf-lhs" }, "F15"), /* @__PURE__ */ React.createElement("span", { className: "mf-eq" }, "="), /* @__PURE__ */ React.createElement("span", { className: "mf-rhs" }, "F12 \xB7 (1 + \xBD\xB7HR)")))), /* @__PURE__ */ React.createElement("div", { className: "mvis" }, /* @__PURE__ */ React.createElement("div", { className: "mvis-head" }, /* @__PURE__ */ React.createElement("span", { className: "dot" }), /* @__PURE__ */ React.createElement("span", { className: "mh-t" }, tMe("m3.visTitle", "Live fusion \xB7 F12 \u2265 0.70 \xB7 F15 \u2265 0.80"))), /* @__PURE__ */ React.createElement("div", { className: "fusion-lab" }, /* @__PURE__ */ React.createElement("div", { className: "fl-controls" }, /* @__PURE__ */ React.createElement("div", { className: "fl-ctl" }, /* @__PURE__ */ React.createElement("div", { className: "fl-top" }, /* @__PURE__ */ React.createElement("span", { className: "nm", style: { color: "var(--tension)" } }, tMe("hud.amp.tensionTitle", "Facial tension")), /* @__PURE__ */ React.createElement("span", { className: "vl" }, "+", (tension * 100).toFixed(0), "%")), /* @__PURE__ */ React.createElement("input", { className: "range r-tension", type: "range", min: "0", max: "0.6", step: "0.01", value: tension, onChange: (e) => setTension(+e.target.value) })), /* @__PURE__ */ React.createElement("div", { className: "fl-ctl" }, /* @__PURE__ */ React.createElement("div", { className: "fl-top" }, /* @__PURE__ */ React.createElement("span", { className: "nm", style: { color: "var(--heart)" } }, tMe("teach.rowBpm", "Heart-rate rise")), /* @__PURE__ */ React.createElement("span", { className: "vl" }, "+", (hr * 100).toFixed(0), "%")), /* @__PURE__ */ React.createElement("input", { className: "range r-hr", type: "range", min: "0", max: "0.6", step: "0.01", value: hr, onChange: (e) => setHr(+e.target.value) }))), /* @__PURE__ */ React.createElement("div", { className: "fl-ladder" }, /* @__PURE__ */ React.createElement(Step, { lab: tMe("m3.stepBase", "Base <b>(face + arousal)</b>"), val: base, color: "var(--ink-3)" }), /* @__PURE__ */ React.createElement(Step, { lab: tMe("m3.stepF12", "\xD7 tension \u2192 <b>F12</b>"), val: f12, color: f12 >= 0.7 ? "var(--ink)" : "var(--ink-2)", thr: 0.7 }), /* @__PURE__ */ React.createElement(Step, { lab: tMe("m3.stepF15", "\xD7 heart rate \u2192 <b>F15</b>"), val: f15, color: isFear ? "var(--danger)" : "var(--ink-2)", thr: 0.8 })), /* @__PURE__ */ React.createElement("div", { className: `fl-verdict ${isFear ? "fear" : "clear"}` }, /* @__PURE__ */ React.createElement("span", { className: "st" }, isFear ? tMe("hud.verdict.fear", "FEAR DETECTED") : tMe("hud.verdict.noFear", "NO FEAR")), /* @__PURE__ */ React.createElement("span", { className: "ex" }, isFear ? tMe("m3.exFear", "body confirms the face") : tMe("m3.exClear", "face alone isn't enough"))))));
}
function RppgSweep() {
  const [ref, seen] = useInView(0.9);
  const algos = ["GREEN", "ICA", "WAVELET", "CHROM", "POS"];
  const windows = ["15 s", "20 s", "30 s", "45 s"];
  const grid = [
    [0.3, 0.42, 0.5, 0.46],
    [0.4, 0.55, 0.63, 0.58],
    [0.48, 0.64, 0.74, 0.69],
    [0.55, 0.74, 0.86, 0.8],
    [0.58, 0.8, 0.95, 0.88]
  ];
  const bestRow = 4, bestCol = 2;
  const HEAT_STOPS = [
    { p: 0, L: 0.5, C: 0.17, H: 27 },
    // red
    { p: 0.5, L: 0.74, C: 0.15, H: 85 },
    // amber
    { p: 1, L: 0.72, C: 0.16, H: 150 }
    // green
  ];
  const heat = (q) => {
    q = Math.max(0, Math.min(1, q));
    let a = HEAT_STOPS[0], b = HEAT_STOPS[HEAT_STOPS.length - 1];
    for (let i = 0; i < HEAT_STOPS.length - 1; i++) {
      if (q >= HEAT_STOPS[i].p && q <= HEAT_STOPS[i + 1].p) {
        a = HEAT_STOPS[i];
        b = HEAT_STOPS[i + 1];
        break;
      }
    }
    const t = (q - a.p) / (b.p - a.p || 1);
    const L = a.L + (b.L - a.L) * t, C = a.C + (b.C - a.C) * t, H = (a.H + (b.H - a.H) * t) % 360;
    return `oklch(${L.toFixed(3)} ${C.toFixed(3)} ${H.toFixed(1)})`;
  };
  return /* @__PURE__ */ React.createElement("div", { className: "mblock flip", ref }, /* @__PURE__ */ React.createElement("div", { className: "mcap" }, /* @__PURE__ */ React.createElement("div", { className: "m-no" }, tMe("methods.label", "METHOD"), " 04"), /* @__PURE__ */ React.createElement("h3", null, tMe("m4.title", "Six estimators, one 30-second window")), /* @__PURE__ */ React.createElement("p", { dangerouslySetInnerHTML: { __html: tMe("m4.p", 'Reading a heartbeat from skin colour needs the right algorithm and the right time window. All six rPPG estimators are computed over a <b style="color: var(--clear)">30-second window at a 3-second step</b>; <b style="color: var(--clear)">POS</b> is used as the headline pulse for its robustness under motion \u2014 the cell that lit up brightest.') } }), /* @__PURE__ */ React.createElement("div", { className: "m-stat" }, /* @__PURE__ */ React.createElement("span", { className: "v" }, "POS \xB7 30 s"), /* @__PURE__ */ React.createElement("span", { className: "k" }, tMe("m4.stat", "headline estimator \xB7 3 s step"))), /* @__PURE__ */ React.createElement("div", { className: "m-foot" }, tMe("m4.foot", "The grid scores how cleanly each estimator \xD7 window separates fear from calm \u2014 red is poor, green is strong."))), /* @__PURE__ */ React.createElement("div", { className: "mvis" }, /* @__PURE__ */ React.createElement("div", { className: "mvis-head" }, /* @__PURE__ */ React.createElement("span", { className: "dot" }), /* @__PURE__ */ React.createElement("span", { className: "mh-t" }, tMe("m4.visTitle", "rPPG sweep \xB7 estimators")), " ", /* @__PURE__ */ React.createElement("span", { className: "mh-r" }, tMe("m4.visR", "relative"))), /* @__PURE__ */ React.createElement("div", { className: `sweep ${seen ? "in" : ""}` }, /* @__PURE__ */ React.createElement("div", { className: "sweep-grid" }, /* @__PURE__ */ React.createElement("div", { className: "corner" }), windows.map((w) => /* @__PURE__ */ React.createElement("div", { className: "colh", key: w }, w)), algos.map((a, ri) => /* @__PURE__ */ React.createElement(React.Fragment, { key: a }, /* @__PURE__ */ React.createElement("div", { className: "rowh" }, a), grid[ri].map((q, ci) => {
    const best = ri === bestRow && ci === bestCol;
    return /* @__PURE__ */ React.createElement(
      "div",
      {
        key: ci,
        className: `cell ${best ? "best" : ""}`,
        style: { background: heat(q), color: "oklch(0.17 0.02 60)", transitionDelay: seen ? `${(ri * 4 + ci) * 26}ms` : "0ms" },
        title: `${a} \xB7 ${windows[ci]} \u2014 detectability ${q.toFixed(2)} of 1.00`
      },
      q.toFixed(2)
    );
  })))), /* @__PURE__ */ React.createElement("div", { className: "sweep-foot" }, /* @__PURE__ */ React.createElement("span", { className: "sweep-scale" }, tMe("m4.scaleLow", "low"), " ", /* @__PURE__ */ React.createElement("span", { className: "bar" }), " ", tMe("m4.scaleHigh", "high")), /* @__PURE__ */ React.createElement("span", { className: "sweep-pick" }, tMe("m4.bestLabel", "best:"), " ", /* @__PURE__ */ React.createElement("b", null, "POS @ 30 s \u2192 0.95"))), /* @__PURE__ */ React.createElement("div", { className: "sweep-key", dangerouslySetInnerHTML: { __html: tMe("m4.key", "Each cell is a <b>detectability score from 0 to 1</b> \u2014 how cleanly that estimator and window separate genuine fear events from calm baseline. <b>1.00</b> would be perfect separation; POS at a 30 s window tops out at <b>0.95</b>, while short windows and weaker estimators (red) blur the cardiac signal. <i>Illustrative \u2014 exact values bind to the benchmark output.</i>") } }))));
}
function MethodsSectionFull({ onNav }) {
  const M = window.SITE.METHODS;
  return /* @__PURE__ */ React.createElement("section", { className: "section-block divline", id: "methods" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, tMe("methods.kicker", "Methods")), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, tMe("methods.title", "What made it work in the dark")), /* @__PURE__ */ React.createElement("p", { className: "sec-lead reveal" }, tMe("methods.leadFull", "Four engineering choices carry the pipeline through horror-game lighting. Each one is broken down below \u2014 detection fallback, contrast recovery, multiplicative fusion, and the rPPG sweep.")), /* @__PURE__ */ React.createElement("div", { className: "feature-grid" }, M.map((m) => /* @__PURE__ */ React.createElement("div", { key: m.t, className: "card reveal" }, /* @__PURE__ */ React.createElement("span", { className: "c-ic" }, /* @__PURE__ */ React.createElement(SiteIcon, { name: m.ic, s: 20 })), /* @__PURE__ */ React.createElement("h3", null, m.t), /* @__PURE__ */ React.createElement("p", null, m.p)))), /* @__PURE__ */ React.createElement("div", { className: "methods-deep" }, /* @__PURE__ */ React.createElement(DetectionFallback, null), /* @__PURE__ */ React.createElement(ClaheDemo, null), /* @__PURE__ */ React.createElement(FusionLadder, null), /* @__PURE__ */ React.createElement(RppgSweep, null))));
}
Object.assign(window, { MethodsSectionFull });
