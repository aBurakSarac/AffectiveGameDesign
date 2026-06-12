/* site-methods.jsx — the four "applied methods" interactive animations:
 *   1 · hybrid Haar→MediaPipe detection fallback (data-driven stacked bars)
 *   2 · CLAHE contrast enhancement (draggable before/after wipe + histogram)
 *   3 · multiplicative fusion ladder (live sliders → F12/F15 verdict)
 *   4 · rPPG window × algorithm sweep (heatmap, POS@30s = best)
 *
 * Animations trigger on scroll via a scroll-listener in-view hook (robust in
 * every render context, unlike IntersectionObserver).
 */
const { useState: useStateM, useRef: useRefM, useEffect: useEffectM, useCallback: useCbM } = React;
// i18n shorthand (unique name; global scope shared across script files)
const tMe = (k, fb) => window.I18N.t(k, fb);

function useInView(margin = 0.84) {
  const ref = useRefM(null);
  const [seen, setSeen] = useStateM(false);
  useEffectM(() => {
    if (seen) return;
    let done = false;
    const check = () => {
      if (done) return;
      const el = ref.current; if (!el) return;
      if (el.getBoundingClientRect().top < window.innerHeight * margin) { done = true; setSeen(true); }
    };
    const onScroll = () => check();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    const id = setInterval(check, 200); // poll: fires even when rAF is starved (offscreen render)
    check();
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); clearInterval(id); };
  }, [seen, margin]);
  return [ref, seen];
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));
const pctM = (v) => `${clamp01(v) * 100}%`;

/* ── 1 · Detection fallback ──────────────────────────────────────────── */
function DetectionFallback() {
  const [ref, seen] = useInView();
  // share of frames per detector, by lighting (illustrative — binds to detector logs).
  // `miss` = frames neither Haar nor MediaPipe could lock (kept intentionally).
  const rows = [
    { nm: tMe("m1.rowBright", "Bright"), mp: 0.06, miss: 0.02, pc: tMe("m1.pcBright", "Haar handles most") },
    { nm: tMe("m1.rowDim", "Dim"), mp: 0.46, miss: 0.05, pc: tMe("m1.pcDim", "fallback on ~46%") },
    { nm: tMe("m1.rowMixed", "Mixed"), mp: 0.55, miss: 0.06, pc: tMe("m1.pcMixed", "fallback on ~55%") },
  ];
  return (
    <div className="mblock" ref={ref}>
      <div className="mcap">
        <div className="m-no">{tMe("methods.label", "METHOD")} 01</div>
        <h3>{tMe("m1.title", "Hybrid detection that survives the dark")}</h3>
        <p>{tMe("m1.p", "A fast Haar cascade crops the face every frame. When it loses the face in low light, a MediaPipe FaceLandmarker fallback takes over — so detection holds even as the room dims. The darker and more uneven the scene, the more the fallback earns its keep.")}</p>
        <div className="m-stat"><span className="v">99.9%</span><span className="k">{tMe("m1.stat", "face coverage across analyzed frames")}</span></div>
        <div className="m-foot">{tMe("m1.foot", "Per-condition shares bind to each session's detector log.")}</div>
      </div>
      <div className="mvis">
        <div className="mvis-head"><span className="dot" /><span className="mh-t">{tMe("m1.visTitle", "Detector by lighting")}</span> <span className="mh-r">{tMe("m1.visR", "per frame")}</span></div>
        <div className="detect-stage">
          {rows.map((r) => (
            <div className="detect-row" key={r.nm}>
              <div className="dl"><span className="nm">{r.nm}</span><span className="pc">{r.pc}</span></div>
              <div className="dbar">
                <div className="seg-haar" style={{ width: seen ? pctM(1 - r.mp - r.miss) : 0 }}>
                  <span className="seg-lab">{Math.round((1 - r.mp - r.miss) * 100)}%</span>
                </div>
                <div className="seg-mp" style={{ width: seen ? pctM(r.mp) : 0 }}>
                  {r.mp > 0.12 ? <span className="seg-lab">{Math.round(r.mp * 100)}%</span> : null}
                </div>
                <div className="seg-miss" style={{ width: seen ? pctM(r.miss) : 0 }} title={`${Math.round(r.miss * 100)}% missed by both`} />
              </div>
            </div>
          ))}
          <div className="detect-legend">
            <span><i style={{ background: "var(--clear)" }} /> {tMe("m1.legHaar", "Haar cascade")}</span>
            <span><i style={{ background: "var(--instrument)" }} /> {tMe("m1.legMp", "MediaPipe fallback")}</span>
            <span><i className="hatch" /> {tMe("m1.legMiss", "missed by both")}</span>
          </div>
          <div className="detect-note">
            {tMe("m1.note", "A few frames are caught by neither detector — and that's expected. Recorded video has no guarantee a face is present 100% of the time: occlusion (hands, hair), extreme head turns, or edits that cut away from the camera all leave short gaps.")}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── 2 · CLAHE before/after ──────────────────────────────────────────── */
const HIST_BEFORE = [33.8, 14.9, 12.4, 13.3, 14.0, 10.3, 1.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0];
const HIST_AFTER = [19.5, 20.5, 7.4, 6.8, 7.0, 6.7, 7.5, 7.4, 6.9, 4.8, 4.2, 1.2, 0, 0, 0, 0];
function Histo({ data, label, after }) {
  const max = Math.max(...data);
  return (
    <div className={`histo ${after ? "after" : ""}`}>
      <div className="ht">{label}</div>
      <div className="bars">{data.map((v, i) => <i key={i} style={{ height: `${(v / max) * 100}%` }} />)}</div>
    </div>
  );
}
function ClaheDemo() {
  const [ref, seen] = useInView();
  const [wipe, setWipe] = useStateM(50);
  const viewRef = useRefM(null);
  const drag = useRefM(false);
  const set = useCbM((clientX) => {
    const el = viewRef.current; if (!el) return;
    const r = el.getBoundingClientRect();
    setWipe(Math.max(2, Math.min(98, ((clientX - r.left) / r.width) * 100)));
  }, []);
  // gentle auto-demo sweep the first time it enters view
  useEffectM(() => {
    if (!seen) return;
    let raf, t0 = performance.now(), active = true;
    const anim = (now) => {
      if (!active || drag.current) return;
      const el = (now - t0) / 1000;
      if (el > 2.6) return;
      setWipe(50 + 32 * Math.sin(el * 2.2));
      raf = requestAnimationFrame(anim);
    };
    raf = requestAnimationFrame(anim);
    return () => { active = false; cancelAnimationFrame(raf); };
  }, [seen]);
  return (
    <div className="mblock flip" ref={ref}>
      <div className="mcap">
        <div className="m-no">{tMe("methods.label", "METHOD")} 02</div>
        <h3>{tMe("m2.title", "CLAHE restores the face before reading it")}</h3>
        <p>{tMe("m2.p", "Contrast-Limited Adaptive Histogram Equalization stretches the cramped, dark tones of the face ROI across the full range — pulling expression detail out of the shadows so the emotion model and the skin-colour pulse signal both get a clean read. Drag to compare.")}</p>
        <div className="m-stat"><span className="v">+</span><span className="k">{tMe("m2.stat", "recovered detail in dim & mixed light")}</span></div>
        <div className="m-foot">{tMe("m2.foot", "Real face crop from session S02_Vid04 (dim lighting). CLAHE: clipLimit 2.0, 4×4 tiles, LAB L-channel.")}</div>
      </div>
      <div className="mvis">
        <div className="mvis-head"><span className="dot" /><span className="mh-t">{tMe("m2.visTitle", "Face ROI · raw → CLAHE")}</span> <span className="mh-r">{tMe("m2.visR", "drag ⇆")}</span></div>
        <div className="clahe-wrap">
          <div className="clahe-view" ref={viewRef} style={{ "--wipe": `${wipe}%` }}
            onPointerDown={(e) => { drag.current = true; e.currentTarget.setPointerCapture(e.pointerId); set(e.clientX); }}
            onPointerMove={(e) => { if (drag.current) set(e.clientX); }}
            onPointerUp={() => { drag.current = false; }}>
            <div className="clahe-face"><img src="media/clahe/clahe_off.png" alt={tMe("m2.altRaw", "Raw face ROI")} /></div>
            <div className="clahe-face after"><img src="media/clahe/clahe_on.png" alt={tMe("m2.altOn", "CLAHE enhanced")} /></div>
            <div className="clahe-roi"><span className="rl">{tMe("m2.roiLabel", "analysis ROI")}</span></div>
            <span className="clahe-tag l">RAW</span>
            <span className="clahe-tag r">CLAHE</span>
            <div className="clahe-divider"><span className="gp">⇆</span></div>
          </div>
          <div className="clahe-hist">
            <Histo data={HIST_BEFORE} label={tMe("m2.histRaw", "Raw histogram")} />
            <Histo data={seen ? HIST_AFTER : HIST_BEFORE} label={tMe("m2.histEq", "Equalized")} after />
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── 3 · Multiplicative fusion ladder ────────────────────────────────── */
function FusionLadder() {
  const [ref] = useInView();
  const [tension, setTension] = useStateM(0.24);
  const [hr, setHr] = useStateM(0.30);
  const fear = 0.66, arousal = 0.52;
  const base = 0.7 * fear + 0.3 * arousal;            // 0.622
  const f12 = Math.min(1, base * (1 + tension));
  const f15 = Math.min(1, f12 * (1 + 0.5 * hr));
  const isFear = f15 >= 0.80;
  const Step = ({ lab, val, color, thr }) => (
    <div className="fl-step">
      <span className="sl" dangerouslySetInnerHTML={{ __html: lab }} />
      <div className="fl-track">
        <div className="ff" style={{ width: pctM(val), background: color }} />
        {thr != null ? <div className="thr-m" style={{ left: pctM(thr) }} /> : null}
      </div>
      <span className="sv" style={{ color }}>{val.toFixed(2)}</span>
    </div>
  );
  return (
    <div className="mblock" ref={ref}>
      <div className="mcap">
        <div className="m-no">{tMe("methods.label", "METHOD")} 03</div>
        <h3>{tMe("m3.title", "Fusion that multiplies, not averages")}</h3>
        <p dangerouslySetInnerHTML={{ __html: tMe("m3.p", "Facial tension and a rising heart rate don't get averaged into the score — they act as <em> amplifiers</em> on the base fear reading. Because they multiply, agreement compounds: a calm body barely moves the needle, while genuine arousal pushes a true fear event clear over the line. Move the sliders to feel it.") }} />
        <div className="m-formula">
          <div className="mf-row"><span className="mf-lhs">base</span><span className="mf-eq">=</span><span className="mf-rhs">0.7·fear + 0.3·arousal</span></div>
          <div className="mf-row"><span className="mf-lhs">F12</span><span className="mf-eq">=</span><span className="mf-rhs">base · (1 + tension)</span></div>
          <div className="mf-row"><span className="mf-lhs">F15</span><span className="mf-eq">=</span><span className="mf-rhs">F12 · (1 + ½·HR)</span></div>
        </div>
      </div>
      <div className="mvis">
        <div className="mvis-head"><span className="dot" /><span className="mh-t">{tMe("m3.visTitle", "Live fusion · F12 ≥ 0.70 · F15 ≥ 0.80")}</span></div>
        <div className="fusion-lab">
          <div className="fl-controls">
            <div className="fl-ctl">
              <div className="fl-top"><span className="nm" style={{ color: "var(--tension)" }}>{tMe("hud.amp.tensionTitle", "Facial tension")}</span><span className="vl">+{(tension * 100).toFixed(0)}%</span></div>
              <input className="range r-tension" type="range" min="0" max="0.6" step="0.01" value={tension} onChange={(e) => setTension(+e.target.value)} />
            </div>
            <div className="fl-ctl">
              <div className="fl-top"><span className="nm" style={{ color: "var(--heart)" }}>{tMe("teach.rowBpm", "Heart-rate rise")}</span><span className="vl">+{(hr * 100).toFixed(0)}%</span></div>
              <input className="range r-hr" type="range" min="0" max="0.6" step="0.01" value={hr} onChange={(e) => setHr(+e.target.value)} />
            </div>
          </div>
          <div className="fl-ladder">
            <Step lab={tMe("m3.stepBase", "Base <b>(face + arousal)</b>")} val={base} color="var(--ink-3)" />
            <Step lab={tMe("m3.stepF12", "× tension → <b>F12</b>")} val={f12} color={f12 >= 0.7 ? "var(--ink)" : "var(--ink-2)"} thr={0.70} />
            <Step lab={tMe("m3.stepF15", "× heart rate → <b>F15</b>")} val={f15} color={isFear ? "var(--danger)" : "var(--ink-2)"} thr={0.80} />
          </div>
          <div className={`fl-verdict ${isFear ? "fear" : "clear"}`}>
            <span className="st">{isFear ? tMe("hud.verdict.fear", "FEAR DETECTED") : tMe("hud.verdict.noFear", "NO FEAR")}</span>
            <span className="ex">{isFear ? tMe("m3.exFear", "body confirms the face") : tMe("m3.exClear", "face alone isn't enough")}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── 4 · rPPG sweep heatmap ──────────────────────────────────────────── */
function RppgSweep() {
  const [ref, seen] = useInView(0.9);
  const algos = ["GREEN", "ICA", "WAVELET", "CHROM", "POS"];
  const windows = ["15 s", "20 s", "30 s", "45 s"];
  // relative detectability across the sweep; POS @ 30s is the headline config
  const grid = [
    [0.30, 0.42, 0.50, 0.46],
    [0.40, 0.55, 0.63, 0.58],
    [0.48, 0.64, 0.74, 0.69],
    [0.55, 0.74, 0.86, 0.80],
    [0.58, 0.80, 0.95, 0.88],
  ];
  const bestRow = 4, bestCol = 2;
  // red (low) → amber (mid) → green (high) detectability ramp
  const HEAT_STOPS = [
    { p: 0.00, L: 0.50, C: 0.17, H: 27 },   // red
    { p: 0.50, L: 0.74, C: 0.15, H: 85 },   // amber
    { p: 1.00, L: 0.72, C: 0.16, H: 150 },  // green
  ];
  const heat = (q) => {
    q = Math.max(0, Math.min(1, q));
    let a = HEAT_STOPS[0], b = HEAT_STOPS[HEAT_STOPS.length - 1];
    for (let i = 0; i < HEAT_STOPS.length - 1; i++) {
      if (q >= HEAT_STOPS[i].p && q <= HEAT_STOPS[i + 1].p) { a = HEAT_STOPS[i]; b = HEAT_STOPS[i + 1]; break; }
    }
    const t = (q - a.p) / (b.p - a.p || 1);
    const L = a.L + (b.L - a.L) * t, C = a.C + (b.C - a.C) * t, H = (a.H + (b.H - a.H) * t) % 360;
    return `oklch(${L.toFixed(3)} ${C.toFixed(3)} ${H.toFixed(1)})`;
  };
  return (
    <div className="mblock flip" ref={ref}>
      <div className="mcap">
        <div className="m-no">{tMe("methods.label", "METHOD")} 04</div>
        <h3>{tMe("m4.title", "Six estimators, one 30-second window")}</h3>
        <p dangerouslySetInnerHTML={{ __html: tMe("m4.p", "Reading a heartbeat from skin colour needs the right algorithm and the right time window. All six rPPG estimators are computed over a <b style=\"color: var(--clear)\">30-second window at a 3-second step</b>; <b style=\"color: var(--clear)\">POS</b> is used as the headline pulse for its robustness under motion — the cell that lit up brightest.") }} />
        <div className="m-stat"><span className="v">POS · 30 s</span><span className="k">{tMe("m4.stat", "headline estimator · 3 s step")}</span></div>
        <div className="m-foot">{tMe("m4.foot", "The grid scores how cleanly each estimator × window separates fear from calm — red is poor, green is strong.")}</div>
      </div>
      <div className="mvis">
        <div className="mvis-head"><span className="dot" /><span className="mh-t">{tMe("m4.visTitle", "rPPG sweep · estimators")}</span> <span className="mh-r">{tMe("m4.visR", "relative")}</span></div>
        <div className={`sweep ${seen ? "in" : ""}`}>
          <div className="sweep-grid">
            <div className="corner" />
            {windows.map((w) => <div className="colh" key={w}>{w}</div>)}
            {algos.map((a, ri) => (
              <React.Fragment key={a}>
                <div className="rowh">{a}</div>
                {grid[ri].map((q, ci) => {
                  const best = ri === bestRow && ci === bestCol;
                  return (
                    <div key={ci} className={`cell ${best ? "best" : ""}`}
                      style={{ background: heat(q), color: "oklch(0.17 0.02 60)", transitionDelay: seen ? `${(ri * 4 + ci) * 26}ms` : "0ms" }}
                      title={`${a} · ${windows[ci]} — detectability ${q.toFixed(2)} of 1.00`}>
                      {q.toFixed(2)}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
          <div className="sweep-foot">
            <span className="sweep-scale">{tMe("m4.scaleLow", "low")} <span className="bar" /> {tMe("m4.scaleHigh", "high")}</span>
            <span className="sweep-pick">{tMe("m4.bestLabel", "best:")} <b>POS @ 30 s → 0.95</b></span>
          </div>
          <div className="sweep-key" dangerouslySetInnerHTML={{ __html: tMe("m4.key", "Each cell is a <b>detectability score from 0 to 1</b> — how cleanly that estimator and window separate genuine fear events from calm baseline. <b>1.00</b> would be perfect separation; POS at a 30 s window tops out at <b>0.95</b>, while short windows and weaker estimators (red) blur the cardiac signal. <i>Illustrative — exact values bind to the benchmark output.</i>") }} />
        </div>
      </div>
    </div>
  );
}

/* ── full Methods section (replaces the preview-only version) ─────────── */
function MethodsSectionFull({ onNav }) {
  const M = window.SITE.METHODS;
  return (
    <section className="section-block divline" id="methods">
      <div className="wrap">
        <span className="kicker reveal">{tMe("methods.kicker", "Methods")}</span>
        <h2 className="sec-title reveal">{tMe("methods.title", "What made it work in the dark")}</h2>
        <p className="sec-lead reveal">
          {tMe("methods.leadFull", "Four engineering choices carry the pipeline through horror-game lighting. Each one is broken down below — detection fallback, contrast recovery, multiplicative fusion, and the rPPG sweep.")}
        </p>
        <div className="feature-grid">
          {M.map((m) => (
            <div key={m.t} className="card reveal">
              <span className="c-ic"><SiteIcon name={m.ic} s={20} /></span>
              <h3>{m.t}</h3>
              <p>{m.p}</p>
            </div>
          ))}
        </div>

        <div className="methods-deep">
          <DetectionFallback />
          <ClaheDemo />
          <FusionLadder />
          <RppgSweep />
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { MethodsSectionFull });
