/* hud-components.jsx — presentational pieces for the merged HUD.
 * All components read a computed frame `f` from window.HUD.computeFrame(t).
 * Exported to window at the bottom.
 */
const { useMemo: useMemoH, useRef: useRefH, useCallback: useCbH } = React;

const fmt = (v, d = 2) => v.toFixed(d);
const pct = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
// i18n shorthand (unique name; global scope is shared across script files)
const tH = (k, fb) => window.I18N.t(k, fb);

// label helper: mode = 'plain' | 'plainOnly' | 'tech'
function Lab({ plain, tech, mode }) {
  if (mode === "tech") return <span className="sig-tech" style={{ fontSize: 15, color: "var(--ink)" }}>{tech}</span>;
  return (
    <>
      <span className="sig-name">{plain}</span>
      {mode !== "plainOnly" && tech ? <span className="sig-tech">{tech}</span> : null}
    </>
  );
}

/* ── Telemetry top bar ─────────────────────────────────────────────── */
function TelemetryBar({ f, accent }) {
  const mm = String(Math.floor(f.t / 60)).padStart(2, "0");
  const ss = String(Math.floor(f.t % 60)).padStart(2, "0");
  const cs = String(Math.floor((f.t % 1) * 100)).padStart(2, "0");
  return (
    <div className="telemetry">
      <div className="brand">
        <span className="rec-dot" />
        <div>
          <div className="title">{tH("hud.telemetry.title", "Fear Analysis HUD")}</div>
          <div className="sub">{tH("hud.telemetry.sub", "FER + rPPG · replay")}</div>
        </div>
      </div>
      <div className="tele-group">
        <div className="tele"><span className="k">{tH("hud.telemetry.time", "Time")}</span><span className="v">{mm}:{ss}<span style={{ color: "var(--ink-4)", fontSize: 14 }}>.{cs}</span></span></div>
        <div className="tele-sep" />
        <div className="tele"><span className="k">{tH("hud.telemetry.frame", "Frame")}</span><span className="v">{String(f.frame).padStart(5, "0")}</span></div>
        <div className="tele-sep" />
        <div className="tele"><span className="k">{tH("hud.telemetry.latency", "Latency")}</span><span className="v" style={{ color: f.latency > 40 ? "var(--arousal)" : "var(--ink)" }}>{Math.round(f.latency)}<span style={{ fontSize: 13, color: "var(--ink-4)" }}>ms</span></span></div>
        <div className="tele-sep" />
        <div className="tele"><span className="k">{tH("hud.telemetry.throughput", "Throughput")}</span><span className="v dim">{f.fps.toFixed(1)}<span style={{ fontSize: 13, color: "var(--ink-4)" }}>fps</span></span></div>
      </div>
    </div>
  );
}

/* ── Video area (image slot + forehead ROI) ────────────────────────── */
function VideoArea({ f }) {
  const fear = f.isFear;
  return (
    <div className="video-wrap">
      <image-slot id="subject-feed" shape="rect" fit="cover"
        placeholder="Drop your session frame / webcam capture"></image-slot>
      <div className="video-overlay">
        <div className="roi" style={{ left: "39%", top: "20%", width: "23%", height: "12%" }}>
          <span className="roi-label">forehead · rPPG ROI</span>
          {/* [COMMENTED OUT 2026-06-01 — animated scan-line cut per review; static box only
              (simpler to render in the Python mp4 export). Kept for rollback]
          <span className="scan" style={{ top: `${20 + 60 * (0.5 + 0.5 * Math.sin(f.t * 3))}%` }} />
          */}
        </div>
        <div className="vid-badge">analysis replay</div>
        {/* [COMMENTED OUT 2026-06-01 — bottom-left status pill cut per review; redundant
            with the verdict card on the right. Kept for rollback]
        <div className="vid-caption">
          <span className="dot" style={{ background: fear ? "var(--danger)" : "var(--clear)" }} />
          {fear ? "Fear moment — score above threshold" : "Monitoring — no fear detected"}
        </div>
        */}
      </div>
    </div>
  );
}

/* ── Trace timeline + scrubber ─────────────────────────────────────── */
function TraceTimeline({ f, traces, headline, playing, onSeek, onToggle }) {
  const W = 1000, H = 120, padT = 8, padB = 8, usable = H - padT - padB;
  const D = window.HUD.DURATION;
  const x = (t) => (t / D) * W;
  const y = (v) => padT + (1 - Math.max(0, Math.min(1, v))) * usable;
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  const area = (pts) => `M${x(pts[0].t)},${y(0)} ` + pts.map((p) => `L${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ") + ` L${x(pts[pts.length - 1].t)},${y(0)} Z`;

  const thr = window.HUD.THRESH.F15;       // [CHANGED 2026-06-01] always show both; F15 drives shading
  const thr12 = window.HUD.THRESH.F12;
  // fear windows where F15 (production) trace >= its threshold
  const windows = useMemoH(() => {
    const tr = traces.F15; const out = []; let start = null;
    tr.forEach((p) => {
      if (p.v >= thr && start === null) start = p.t;
      else if (p.v < thr && start !== null) { out.push([start, p.t]); start = null; }
    });
    if (start !== null) out.push([start, D]);
    return out;
  }, [traces, thr, D]);

  const scrubRef = useRefH(null);
  const seekAt = useCbH((clientX) => {
    const el = scrubRef.current; if (!el) return;
    const r = el.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    onSeek(frac * D);
  }, [onSeek, D]);
  const drag = useRefH(false);

  return (
    <div className="timeline">
      <div className="timeline-head">
        <span className="ttl">Fear score over time</span>
        <div className="legend">
          <span><i style={{ background: "var(--ink-2)" }} />F15 (+ heart rate)</span>
          <span><i style={{ background: "var(--ink-4)" }} />F12 (face only)</span>
          <span><i style={{ background: "var(--ink-2)", height: 2 }} />thr {fmt(thr)}</span>
          <span><i style={{ background: "var(--ink-4)", height: 2 }} />thr {fmt(thr12)}</span>
        </div>
      </div>
      <div className="trace" onClick={(e) => seekAt(e.clientX)}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {windows.map((w, i) => (
            <rect key={i} x={x(w[0])} y={0} width={x(w[1]) - x(w[0])} height={H}
              fill="var(--danger)" opacity="0.12" />
          ))}
          <line x1="0" y1={y(thr12)} x2={W} y2={y(thr12)} stroke="var(--ink-4)" strokeWidth="1" strokeDasharray="4 6" opacity="0.7" />
          <line x1="0" y1={y(thr)} x2={W} y2={y(thr)} stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="5 5" opacity="0.8" />
          {/* F12 secondary trace */}
          <path d={path(traces.F12)} fill="none" stroke="var(--ink-4)" strokeWidth="1.6" opacity="0.9" />
          {/* F15 primary trace */}
          <path d={area(traces.F15)} fill="var(--ink)" opacity="0.05" />
          <path d={path(traces.F15)} fill="none" stroke="var(--ink-2)" strokeWidth="2.4" />
          {/* playhead */}
          <line x1={x(f.t)} y1="0" x2={x(f.t)} y2={H} stroke="var(--ink)" strokeWidth="1.5" opacity="0.5" />
          <circle cx={x(f.t)} cy={y(f.F15)} r="4.5" fill="var(--ink)" stroke="var(--bg)" strokeWidth="2" />
        </svg>
      </div>
      <div className="scrub-row">
        <button className="play-btn" onClick={onToggle} aria-label={playing ? "Pause" : "Play"}>
          {playing
            ? <svg width="14" height="16" viewBox="0 0 14 16"><rect x="0" y="0" width="4.5" height="16" rx="1" fill="currentColor" /><rect x="9.5" y="0" width="4.5" height="16" rx="1" fill="currentColor" /></svg>
            : <svg width="14" height="16" viewBox="0 0 14 16"><path d="M0 0 L14 8 L0 16 Z" fill="currentColor" /></svg>}
        </button>
        <div className="scrub" ref={scrubRef}
          onPointerDown={(e) => { drag.current = true; e.currentTarget.setPointerCapture(e.pointerId); seekAt(e.clientX); }}
          onPointerMove={(e) => { if (drag.current) seekAt(e.clientX); }}
          onPointerUp={(e) => { drag.current = false; }}>
          <div className="scrub-track" />
          <div className="scrub-fill" style={{ width: pct(f.t / D) }} />
          <div className="scrub-knob" style={{ left: pct(f.t / D) }} />
        </div>
        <span className="scrub-time">{String(Math.floor(f.t / 60)).padStart(2, "0")}:{String(Math.floor(f.t % 60)).padStart(2, "0")} / {String(Math.floor(D / 60)).padStart(2, "0")}:{String(Math.floor(D % 60)).padStart(2, "0")}</span>
      </div>
    </div>
  );
}

/* ── Verdict card ──────────────────────────────────────────────────── */
function VerdictCard({ f, headline }) {
  // [CHANGED 2026-06-01] Show BOTH F12 and F15 with their own threshold lines,
  // instead of a single toggled headline. F15 (production) drives the verdict
  // state; the two gauges let viewers see how rPPG nudges F12 → F15.
  const noFace = !f.roi;
  const fear = !noFace && f.F15 >= window.HUD.THRESH.F15;

  /* [COMMENTED OUT 2026-06-01 — single-headline score/gap, superseded by dual gauges below.
     Kept for rollback.
  const score = f[headline];
  const thr = window.HUD.THRESH[headline];
  const gap = score - thr;
  let reason;
  if (fear) reason = "Fearful expression, high arousal and an elevated heart rate — the combined score has crossed the line.";
  else if (f.hs_fear > 0.45 && f.hs_arousal < 0.35) reason = "Strong fear expression — but low arousal and a calm heart rate hold the score below the line. Not counted as fear.";
  else if (score > 0.45) reason = "Some negative signals present, but the score stays under the fear threshold.";
  else reason = "Calm. No meaningful fear indicators.";
  const fillColor = fear ? "var(--danger)" : score >= thr * 0.7 ? "var(--arousal)" : "var(--instrument)";
  */

  // one gauge row: label · fill · own threshold tick · pass/fail value
  function GaugeRow({ tag, sub, score, thr, color }) {
    const over = score >= thr;
    // [CHANGED 2026-06-01] F12/F15 bars are neutral gray; the verdict state (banner,
    // OVER/UNDER text + number going red) carries the meaning, not the bar color.
    // Mid-gray fill keeps the white threshold tick visible on top.
    const fillColor = over ? "var(--ink-2)" : "var(--ink-4)";
    return (
      <div className="grow">
        <div className="grow-head">
          <span className="grow-tag" style={{ color }}>{tag}</span>
          <span className="grow-sub">{sub}</span>
          <span className="grow-thr">{tH("hud.verdict.needs", "needs")} {fmt(thr)}</span>
          <span className="grow-state" style={{ color: over ? "var(--danger)" : "var(--ink-3)" }}>
            {over ? tH("hud.verdict.over", "OVER") : tH("hud.verdict.under", "UNDER")}
          </span>
          <span className="grow-num" style={{ color: over ? "var(--danger)" : "var(--ink)" }}>{fmt(score)}</span>
        </div>
        <div className="gauge-track">
          <div className="gauge-fill" style={{ width: pct(score), background: fillColor }} />
          <div className="gauge-thresh" style={{ left: pct(thr) }} />
        </div>
      </div>
    );
  }

  return (
    <div className={`section verdict ${noFace ? "warn" : fear ? "fear" : "clear"}`}>
      <div className="verdict-top">
        <div className="verdict-icon">
          {noFace
            ? <svg width="30" height="30" viewBox="0 0 24 24"><path d="M1 1 L23 23" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" /><path d="M6.5 6.5 A8 8 0 0 0 4 12 c0 4.4 3.6 8 8 8 a8 8 0 0 0 5.5-2.2" fill="none" stroke="currentColor" strokeWidth="2.2" /><path d="M9 9 a8 8 0 0 1 11 3" fill="none" stroke="currentColor" strokeWidth="2.2" /></svg>
            : fear
            ? <svg width="30" height="30" viewBox="0 0 24 24"><path d="M12 2 L22 20 H2 Z" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" /><line x1="12" y1="9" x2="12" y2="14" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" /><circle cx="12" cy="17" r="1.3" fill="currentColor" /></svg>
            : <svg width="30" height="30" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2.2" /><path d="M7 12.5 L10.5 16 L17 8.5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg>}
        </div>
        <div className="verdict-words">
          <div className="verdict-state">{noFace ? tH("hud.verdict.faceLost", "FACE LOST") : fear ? tH("hud.verdict.fear", "FEAR DETECTED") : tH("hud.verdict.noFear", "NO FEAR")}</div>
          <div className="verdict-byline">{noFace ? tH("hud.verdict.bylineFaceLost", "subject out of frame · detection paused") : tH("hud.verdict.byline", "final decision · F15 (production)")}</div>
        </div>
        <div className="verdict-score">
          <div className="num" style={{ color: fear ? "var(--danger)" : "var(--ink)" }}>{fmt(f.F15)}</div>
          <div className="cap">{tH("hud.verdict.scoreCap", "F15 score")}</div>
        </div>
      </div>
      <div className="gauge gauge-dual">
        <GaugeRow tag="F12" sub={tH("hud.verdict.f12Sub", "face only")} score={f.F12} thr={window.HUD.THRESH.F12} color="var(--ink-2)" />
        <GaugeRow tag="F15" sub={tH("hud.verdict.f15Sub", "+ heart rate")} score={f.F15} thr={window.HUD.THRESH.F15} color="var(--ink-2)" />
        <div className="gauge-scale"><span>0.00</span><span>0.50</span><span>1.00</span></div>
      </div>
    </div>
  );
}

/* ── Primary signals: Fear (hero) + Arousal + secondary emotions ───── */
function PrimarySignals({ f, mode }) {
  const labels = window.HUD.EMOTION_LABELS;
  const others = labels.filter((l) => l !== "Fear");
  return (
    <div className="section">
      <div className="sec-head">
        <span className="idx">1</span>
        <span className="label">{tH("hud.primary.label", "Primary signal · facial emotion")}</span>
        <span className="note">HSEmotion</span>
      </div>
      <div className="sec-body">
        <div className="sig primary hero">
          <div className="sig-top">
            <Lab plain={tH("emo.Fear", "Fear")} tech="hs_fear" mode={mode} />
            <span className="sig-val" style={{ color: "var(--danger)" }}>{fmt(f.hs_fear)}</span>
          </div>
          <div className="bar"><div className="bar-fill" style={{ width: pct(f.hs_fear), background: "var(--danger)" }} /></div>
        </div>
        <div className="sig primary">
          <div className="sig-top">
            <Lab plain={tH("hud.primary.arousal", "Arousal")} tech="hs_arousal" mode={mode} />
            <span className="sig-val" style={{ color: "var(--arousal)" }}>{fmt(f.hs_arousal)}</span>
          </div>
          <div className="bar"><div className="bar-fill" style={{ width: pct(f.hs_arousal), background: "var(--arousal)" }} /></div>
        </div>
        <div className="base-note">
          <span>{tH("hud.primary.baseScore", "Base score")}</span>
          <span style={{ fontFamily: "var(--mono)", color: "var(--ink-3)" }}>{tH("hud.primary.baseFormula", "= 0.7×Fear + 0.3×Arousal =")}</span>
          <b>{fmt(f.base)}</b>
        </div>
        <div className="emo-strip">
          <div className="cap">{tH("hud.primary.distPrefix", "Full emotion distribution · dominant:")} {tH("emo." + f.dom, f.dom)} ({fmt(f.domScore)})</div>
          <div className="emo-grid">
            {others.map((l) => (
              <div key={l} className={`emo ${l === f.dom ? "dom" : ""}`}>
                <span className="en">{tH("emo." + l, l)}</span>
                <span className="eb"><span className="ef" style={{ width: pct(f.emotions[l]) }} /></span>
                <span className="ev">{fmt(f.emotions[l])}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Amplifiers + Formula chain (merged) ──────────────────────────── */
function Amplifiers({ f, mode, showAlgos, setShowAlgos }) {
  const algoOrder = window.HUD.RPPG_ALGOS;
  const bnMult = f.rppg_mult;
  const fear = f.isFear;

  return (
    <div className="section">
      <div className="sec-head">
        <span className="idx">2</span>
        <span className="label">{tH("hud.amp.label", "Amplifiers & score build-up")}</span>
        <span className="note">{tH("hud.amp.note", "F12 = base × tension · F15 = F12 × heart")}</span>
      </div>
      <div className="sec-body">
        <div className="amp-chain-grid">
          {/* MediaPipe tension */}
          <div className="amp" style={{ borderColor: "oklch(0.70 0.15 305 / 0.4)" }}>
            <div className="amp-head">
              <span className="amp-dot" style={{ background: "var(--tension)" }} />
              <span className="amp-title">{tH("hud.amp.tensionTitle", "Facial tension")}</span>
              <span className="amp-mult" style={{ background: "oklch(0.70 0.15 305 / 0.16)", color: "var(--tension)" }}>×{fmt(f.mp_mult)}</span>
            </div>
            <div className="amp-sub">{mode === "tech" ? "mp_tension · ×(1+mp_tension)" : tH("hud.amp.tensionSub", "MediaPipe · brow / jaw / eye muscle strain")}</div>
            <div className="amp-main">
              <span className="amp-num" style={{ color: "var(--tension)" }}>{fmt(f.mp_tension)}</span>
              <span className="amp-unit">/ 1.00</span>
            </div>
            <div className="amp-row">
              <span className="mini-bar"><span className="mini-fill" style={{ width: pct(f.mp_tension), background: "var(--tension)" }} /></span>
            </div>
            <div className="mp-stats">
              <div className="mp-stat"><span className="k">{tH("hud.amp.valence", "Valence")}</span><span className="v" style={{ color: f.valence < 0 ? "var(--danger)" : "var(--clear)" }}>{f.valence >= 0 ? "+" : ""}{fmt(f.valence)}</span></div>
              <div className="mp-stat"><span className="k">{tH("hud.amp.smile", "Smile")}</span><span className="v">{fmt(f.smile)}</span></div>
              <div className="mp-stat"><span className="k">{tH("hud.amp.startle", "Startle")}</span><span className="v">{fmt(f.startle, 1)}/s</span></div>
            </div>
          </div>

          {/* rPPG heart rate */}
          {(() => {
            const calibrating = !f.bpm && (!f.algos || Object.keys(f.algos).length === 0);
            return (
              <div className="amp" style={{ borderColor: calibrating ? "oklch(0.46 0.012 250 / 0.4)" : "oklch(0.70 0.175 12 / 0.4)" }}>
                <div className="amp-head">
                  <span className="amp-dot" style={{ background: calibrating ? "var(--ink-4)" : "var(--heart)" }} />
                  <span className="amp-title">{tH("hud.amp.heartTitle", "Heart rate")}</span>
                  {calibrating
                    ? <span className="amp-mult" style={{ background: "oklch(0.46 0.012 250 / 0.16)", color: "var(--ink-3)" }}>{tH("hud.amp.calibrating", "calibrating")}</span>
                    : <span className="amp-mult" style={{ background: "oklch(0.70 0.175 12 / 0.16)", color: "var(--heart)" }}>×{fmt(bnMult)}</span>}
                </div>
                {calibrating ? (
                  <>
                    <div className="amp-sub">{tH("hud.amp.calSub", "rPPG needs ~30 s of skin-colour data before the first pulse estimate")}</div>
                    <div className="amp-main">
                      <span className="amp-num" style={{ color: "var(--ink-4)" }}>—</span>
                      <span className="amp-unit" style={{ color: "var(--ink-4)" }}>BPM</span>
                    </div>
                    <div className="rppg-cal-note">{tH("hud.amp.calNote", "Collecting forehead ROI frames… multiplier locked at ×1.00 until ready")}</div>
                  </>
                ) : (
                  <>
                    <div className="amp-sub">{mode === "tech" ? "POS bpm · ×(1+0.5·bpm_norm)" : tH("hud.amp.heartSub", "rPPG (POS) · pulse read from skin colour")}</div>
                    <div className="amp-main">
                      <span className="amp-num" style={{ color: "var(--heart)" }}>{Math.round(f.bpm)}</span>
                      <span className="amp-unit">BPM</span>
                      <span style={{ marginLeft: "auto", fontFamily: "var(--mono)", fontSize: 12, color: "var(--ink-3)" }}>
                        {tH("hud.amp.rise", "rise")} {mode === "tech" ? "bpm_norm " : ""}{fmt(f.bpm_norm)}
                      </span>
                    </div>
                    <div className="hr-strip">
                      <div className="hr-line" />
                      <div className="hr-base" style={{ left: pct(Math.max(0.04, Math.min(0.96, (f.baseline - 50) / 80))) }}><span className="tag">{tH("hud.amp.rest", "rest")} {f.baseline}</span></div>
                      <div className="hr-now" style={{ left: pct(Math.max(0.04, Math.min(0.96, (f.bpm - 50) / 80))) }}><span className="tag">{tH("hud.amp.now", "now")} {Math.round(f.bpm)}</span></div>
                    </div>
                    <div className="algos">
                      {setShowAlgos && (
                        <label className="ctl-toggle algo-toggle">
                          <input type="checkbox" checked={showAlgos} onChange={(e) => setShowAlgos(e.target.checked)} />
                          <span className="track"><span className="knob" /></span>
                          {tH("hud.amp.otherAlgos", "Other rPPG algorithms")}
                        </label>
                      )}
                      {showAlgos && (
                        <div className="algo-chips">
                          {algoOrder.filter((a) => a !== "POS" && f.algos[a]).map((a) => (
                            <div key={a} className="algo-chip">
                              <div className="an">{a.slice(0, 5)}</div>
                              <div className="av">{Math.round(f.algos[a].bpm)}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            );
          })()}

          {/* Formula chain — inline below the two amp cards */}
          <div className="chain-inline">
            <div className="chain">
              <div className="chain-node">
                <div className="cn-cap">{tH("hud.chain.base", "Base")}</div>
                <div className="cn-val">{fmt(f.base)}</div>
                <div className="cn-form">{tH("hud.chain.baseForm", "fear + arousal")}</div>
              </div>
              <div className="chain-op">
                <span className="op" style={{ color: "var(--tension)" }}>× {fmt(f.mp_mult)}</span>
                <span className="arr">→</span>
                <span className="lift" style={{ background: "oklch(0.70 0.15 305 / 0.16)", color: "var(--tension)" }}>{tH("hud.chain.tension", "tension")}</span>
              </div>
              <div className="chain-node">
                <div className="cn-cap">{tH("hud.chain.f12", "F12 · face only")}</div>
                <div className="cn-val" style={{ color: "var(--ink)" }}>{fmt(f.F12)}</div>
                <div className="cn-form">≥ {fmt(window.HUD.THRESH.F12)}?</div>
              </div>
              <div className="chain-op">
                <span className="op" style={{ color: "var(--heart)" }}>× {fmt(f.rppg_mult)}</span>
                <span className="arr">→</span>
                <span className="lift" style={{ background: "oklch(0.70 0.175 12 / 0.16)", color: "var(--heart)" }}>{tH("hud.chain.heart", "heart")}</span>
              </div>
              <div className={`chain-node final ${fear ? "" : "clear-state"}`}>
                <div className="cn-cap">{tH("hud.chain.f15", "F15 · production")}</div>
                <div className="cn-val">{fmt(f.F15)}</div>
                <div className="cn-form">≥ {fmt(window.HUD.THRESH.F15)}?</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function FormulaChain() { return null; }

Object.assign(window, {
  TelemetryBar, VideoArea, TraceTimeline, VerdictCard,
  PrimarySignals, Amplifiers, FormulaChain,
});
