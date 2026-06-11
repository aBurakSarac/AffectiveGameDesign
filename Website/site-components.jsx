/* site-components.jsx — presentational sections for the study site.
 * Reuses the approved HUD leaf components from hud-components.jsx
 * (TelemetryBar, VerdictCard, PrimarySignals, Amplifiers, FormulaChain)
 * and adds the responsive site chrome around them.
 */
const { useMemo: useMemoC, useRef: useRefC, useCallback: useCbC, useState: useStateC } = React;

const f2 = (v, d = 2) => Number(v).toFixed(d);
const pctC = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/* small inline icon set (stroke, currentColor) */
function Icon({ name, s = 18 }) {
  const p = {
    play: <path d="M5 3 L19 12 L5 21 Z" fill="currentColor" stroke="none" />,
    pause: <g fill="currentColor" stroke="none"><rect x="5" y="3" width="5" height="18" rx="1" /><rect x="14" y="3" width="5" height="18" rx="1" /></g>,
    arrow: <path d="M4 12 H20 M14 6 L20 12 L14 18" />,
    menu: <path d="M3 6 H21 M3 12 H21 M3 18 H21" />,
    close: <path d="M5 5 L19 19 M19 5 L5 19" />,
    sliders: <g><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h14M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="8" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></g>,
    up: <path d="M12 19 V5 M6 11 L12 5 L18 11" />,
    down: <path d="M12 5 V19 M6 13 L12 19 L18 13" />,
    reset: <path d="M3 12 a9 9 0 1 0 3-6.7 M3 4 v4 h4" />,
    dot6: <g fill="currentColor" stroke="none"><circle cx="9" cy="6" r="1.4" /><circle cx="15" cy="6" r="1.4" /><circle cx="9" cy="12" r="1.4" /><circle cx="15" cy="12" r="1.4" /><circle cx="9" cy="18" r="1.4" /><circle cx="15" cy="18" r="1.4" /></g>,
    info: <g><circle cx="12" cy="12" r="9" /><path d="M12 11 V16 M12 8 h0.01" /></g>,
    shield: <path d="M12 3 L20 6 V11 C20 16 16.5 19.5 12 21 C7.5 19.5 4 16 4 11 V6 Z M9 12 l2 2 l4 -4" />,
    bolt: <path d="M13 3 L5 13 H11 L10 21 L19 10 H13 Z" />,
    eye: <g><path d="M2 12 S6 5 12 5 S22 12 22 12 S18 19 12 19 S2 12 2 12 Z" /><circle cx="12" cy="12" r="3" /></g>,
    lighting: <g><circle cx="12" cy="12" r="4" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" /></g>,
    contrast: <g><circle cx="12" cy="12" r="9" /><path d="M12 3 a9 9 0 0 0 0 18 Z" fill="currentColor" stroke="none" /></g>,
    fuse: <g><circle cx="7" cy="7" r="3.4" /><circle cx="17" cy="7" r="3.4" /><path d="M7 10.4 V13 a5 5 0 0 0 10 0 V10.4 M12 18 v3" /></g>,
    pulse: <path d="M2 12 H7 L9 6 L13 18 L15 12 H22" />,
  }[name];
  return <svg width={s} height={s} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{p}</svg>;
}

/* ── Top nav ──────────────────────────────────────────────────────────── */
function Nav({ active, open, setOpen, onNav, links, onCustomize }) {
  const go = (e, id) => { e.preventDefault(); setOpen(false); onNav(id); };
  return (
    <nav className="nav">
      <div className="wrap">
        <a href="#overview" className="nav-brand" onClick={(e) => go(e, "overview")}>
          <span className="rec-dot" />
          <span>
            <span className="bt">La Façade <em>Fissurée</em></span>
          </span>
          <span className="bs">FER + rPPG</span>
        </a>
        <div className={`nav-links ${open ? "open" : ""}`}>
          {links.map(([id, lab]) => (
            <a key={id} href={`#${id}`} className={active === id ? "active" : ""} onClick={(e) => go(e, id)}>{lab}</a>
          ))}
          <a href="#player" className="nav-cta" onClick={(e) => go(e, "player")}>Open the HUD ▸</a>
        </div>
        <button className="nav-custom" onClick={onCustomize} aria-label="Customize sections" title="Reorder & hide sections">
          <Icon name="sliders" s={17} />
        </button>
        <button className="nav-toggle" onClick={() => setOpen((o) => !o)} aria-label="Menu">
          <Icon name={open ? "close" : "menu"} />
        </button>
      </div>
    </nav>
  );
}

/* ── Hero / overview ──────────────────────────────────────────────────── */
function Hero({ onNav }) {
  const R = window.SITE.RESULTS;
  return (
    <header className="hero section-block" id="overview">
      <div className="hero-aura" />
      <div className="wrap">
        <span className="hero-tag reveal">
          <span className="pill">Final project</span>
          Galatasaray University · Computer Engineering
        </span>
        <h1 className="reveal">
          Can a plain webcam<br />feel a player's <em>fear</em>?
        </h1>
        <p className="hero-sub reveal">
          A real-time, privacy-preserving pipeline that fuses facial emotion recognition with
          a contactless heart rate read from skin colour — scoring fear under the dim,
          flickering light of horror games. Everything runs locally; no video ever leaves the device.
        </p>
        <div className="hero-actions reveal">
          <button className="btn btn-primary" onClick={() => onNav("player")}>
            <Icon name="play" s={17} /> Watch real sessions
          </button>
          <button className="btn btn-ghost" onClick={() => onNav("fusion")}>
            Does the heartbeat help? <Icon name="arrow" s={16} />
          </button>
        </div>

        <div className="result-strip reveal">
          {R.map((c) => (
            <div key={c.k} className={`result-cell ${c.tone}`}>
              <div className="rc-k">{c.k}</div>
              <div className="rc-v">{c.v}{c.unit ? <small>{c.unit}</small> : null}</div>
              <div className="rc-s">{c.s}</div>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
}

/* ── Player: session video + forehead ROI ─────────────────────────────── */
function PlayerVideo({ f, session, videoRef, sessionReady, onToggle }) {
  const vw = session.videoWidth || window.HUD.videoWidth || 290;
  const vh = session.videoHeight || window.HUD.videoHeight || 240;
  const bboxStyle = (box) => box ? ({
    left: `${(box[0] / vw) * 100}%`,
    top: `${(box[1] / vh) * 100}%`,
    width: `${((box[2] - box[0]) / vw) * 100}%`,
    height: `${((box[3] - box[1]) / vh) * 100}%`,
  }) : null;
  const faceStyle = bboxStyle(f.roi);
  const foreheadStyle = bboxStyle(f.foreheadRoi);
  return (
    <div className="video-wrap">
      <video ref={videoRef} className="session-video" muted playsInline preload="auto"
        poster={`media/sessions/${session.id}/thumb.jpg`}
        src={sessionReady ? window.HUD.getVideoUrl(session.id) : undefined}
        onClick={onToggle} style={{ cursor: "pointer" }} />
      <div className="video-overlay">
        {faceStyle && <div className="roi roi-face" style={faceStyle}>
          <span className="roi-label">FER · face ROI</span>
        </div>}
        {foreheadStyle && <div className="roi roi-forehead" style={foreheadStyle}>
          <span className="roi-label">rPPG · forehead</span>
        </div>}
        <div className="vid-badge">{session.id} · analysis replay</div>
        <div className="vid-caption">
          {f.roi
            ? <>
                <span className="dot" style={{ background: f.isFear ? "var(--danger)" : "var(--clear)" }} />
                {f.isFear ? "Fear moment — score over threshold" : "Monitoring — no fear detected"}
              </>
            : <>
                <span className="dot" style={{ background: "var(--arousal)" }} />
                Face lost — subject out of frame
              </>}
        </div>
      </div>
    </div>
  );
}

/* ── Player trace == the scrubber ─────────────────────────────────────── */
function PlayerTrace({ f, traces, D, playing, onSeek, onToggle, speed, setSpeed }) {
  const W = 1000, H = 132, padT = 12, padB = 10, usable = H - padT - padB;
  const THR = window.HUD.THRESH;
  const x = (t) => (t / D) * W;
  const y = (v) => padT + (1 - Math.max(0, Math.min(1, v))) * usable;
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  const area = (pts) => `M${x(pts[0].t)},${y(0)} ` + pts.map((p) => `L${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ") + ` L${x(pts[pts.length - 1].t)},${y(0)} Z`;

  const windows = useMemoC(() => {
    const tr = traces.F15, out = []; let start = null;
    tr.forEach((p) => {
      if (p.v >= THR.F15 && start === null) start = p.t;
      else if (p.v < THR.F15 && start !== null) { out.push([start, p.t]); start = null; }
    });
    if (start !== null) out.push([start, D]);
    return out;
  }, [traces, D]);

  const plotRef = useRefC(null);
  const drag = useRefC(false);
  const seekAt = useCbC((clientX) => {
    const el = plotRef.current; if (!el) return;
    const r = el.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * D);
  }, [onSeek, D]);

  const frac = f.t / D;
  return (
    <div className="ptrace">
      <div className="ptrace-head">
        <span className="ttl">Fear score · drag anywhere to scrub</span>
        <div className="legend">
          <span><i style={{ background: "var(--ink-2)" }} />F15 +heart</span>
          <span><i style={{ background: "var(--ink-4)" }} />F12 face</span>
          <span><i style={{ background: "var(--danger)", opacity: 0.5 }} />fear window</span>
        </div>
      </div>
      <div className="ptrace-plot" ref={plotRef}
        onPointerDown={(e) => { drag.current = true; e.currentTarget.setPointerCapture(e.pointerId); seekAt(e.clientX); }}
        onPointerMove={(e) => { if (drag.current) seekAt(e.clientX); }}
        onPointerUp={() => { drag.current = false; }}>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {windows.map((w, i) => (
            <rect key={i} x={x(w[0])} y="0" width={x(w[1]) - x(w[0])} height={H} fill="var(--danger)" opacity="0.12" />
          ))}
          <line x1="0" y1={y(THR.F12)} x2={W} y2={y(THR.F12)} stroke="var(--ink-4)" strokeWidth="1" strokeDasharray="4 6" opacity="0.7" />
          <line x1="0" y1={y(THR.F15)} x2={W} y2={y(THR.F15)} stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="5 5" opacity="0.8" />
          <path d={path(traces.F12)} fill="none" stroke="var(--ink-4)" strokeWidth="1.6" opacity="0.9" />
          <path d={area(traces.F15)} fill="var(--ink)" opacity="0.05" />
          <path d={path(traces.F15)} fill="none" stroke="var(--ink-2)" strokeWidth="2.4" />
          <line x1={x(f.t)} y1="0" x2={x(f.t)} y2={H} stroke={f.isFear ? "var(--danger)" : "var(--ink)"} strokeWidth="1.5" opacity="0.55" />
          <circle cx={x(f.t)} cy={y(f.F15)} r="5" fill={f.isFear ? "var(--danger)" : "var(--ink)"} stroke="var(--bg)" strokeWidth="2" />
        </svg>
        <div className="pt-time" style={{ left: pctC(Math.max(0.05, Math.min(0.95, frac))) }}>{f.F15.toFixed(2)}</div>
      </div>
      <div className="ptrace-ctl">
        <button className="play-btn" onClick={onToggle} aria-label={playing ? "Pause" : "Play"}>
          <Icon name={playing ? "pause" : "play"} s={15} />
        </button>
        <div className="seg seg-speed">
          {["0.5×", "1×", "2×"].map((o) => (
            <button key={o} className={speed === o ? "on" : ""} onClick={() => setSpeed(o)}>{o}</button>
          ))}
        </div>
        <span className="scrub-time">{mmss(f.t)} / {mmss(D)}</span>
      </div>
    </div>
  );
}

/* segmented control */
function Seg({ label, value, options, onChange }) {
  return (
    <div className="seg-field">
      <span className="seg-lab">{label}</span>
      <div className="seg">
        {options.map((o) => (
          <button key={o} className={value === o ? "on" : ""} onClick={() => onChange(o)}>{o}</button>
        ))}
      </div>
    </div>
  );
}

function PlayerControls({ labels, setLabels, showAlgos, setShowAlgos, speed, setSpeed }) {
  return (
    <div className="player-controls">
      <Seg label="Labels" value={labels} options={["Plain", "Plain + tech", "Technical"]} onChange={setLabels} />
      <Seg label="Speed" value={speed} options={["0.5×", "1×", "2×"]} onChange={setSpeed} />
      <label className="ctl-toggle">
        <input type="checkbox" checked={showAlgos} onChange={(e) => setShowAlgos(e.target.checked)} />
        <span className="track"><span className="knob" /></span>
        Show other rPPG algorithms
      </label>
    </div>
  );
}

/* ── Session rail ─────────────────────────────────────────────────────── */
function SessionRail({ sessions, sel, onSelect }) {
  return (
    <div className="session-rail">
      <div className="rail-head">
        <span className="rh-t">Recorded sessions</span>
        <span className="rh-n">{sessions.length} clips · 6 subjects · 3 lighting conditions</span>
      </div>
      <div className="session-list">
        {sessions.map((s) => (
          <button key={s.id} className={`s-card ${s.id === sel ? "sel" : ""}`} onClick={() => onSelect(s.id)}>
            <span className="s-thumb">
              <img className="s-thumb-img" src={`media/sessions/${s.id}/thumb.jpg`} alt="" loading="lazy" />
              <span className="play-ic"><Icon name={s.id === sel ? "pause" : "play"} s={15} /></span>
              <span className="s-dur">{mmss(s.dur)}</span>
            </span>
            <span className="s-meta">
              <span className="sm-top">
                <span className="sm-id">{s.subject}</span>
                <span className={`lightchip ${s.lighting}`}>{s.lighting}</span>
                <span className="sm-vid">{s.vid}</span>
              </span>
              <span className="sm-sub">{s.note}</span>
            </span>
          </button>
        ))}
      </div>
      <div className="note-strip">
        <Icon name="info" s={17} />
        <span>Each card streams its own real per-frame data + <span className="mono">raw_video.mp4</span>. The numbers
          match the offline HUD renderer exactly.</span>
      </div>
    </div>
  );
}

/* ── Player section assembly ──────────────────────────────────────────── */
function HudFit({ children }) {
  const wrapRef = useRefC(null);
  const [box, setBox] = useStateC({ s: 1, h: 900 });
  React.useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const CW = 1600, CH = 900;
    const measure = () => {
      const w = el.clientWidth;
      const s = Math.min(1, w / CW);
      setBox({ s, h: CH * s });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    window.addEventListener("resize", measure);
    return () => { ro.disconnect(); window.removeEventListener("resize", measure); };
  }, []);
  return (
    <div className="hud-fit" ref={wrapRef} style={{ height: box.h }}>
      <div className="hud-canvas" style={{ transform: `scale(${box.s})` }}>{children}</div>
    </div>
  );
}

function PlayerSection({ f, traces, D, playing, onSeek, onToggle, sessions, sel, onSelect, mode, showAlgos, setShowAlgos, speed, setSpeed, compact, videoRef, sessionReady }) {
  const session = sessions.find((s) => s.id === sel) || sessions[0];
  const Body = (
    <>
      <TelemetryBar f={f} />
      <div className="hud-body">
        <div className="hud-left">
          <PlayerVideo f={f} session={session} videoRef={videoRef} sessionReady={sessionReady} onToggle={onToggle} />
          <PlayerTrace f={f} traces={traces} D={D} playing={playing} onSeek={onSeek} onToggle={onToggle} speed={speed} setSpeed={setSpeed} />
        </div>
        <div className="hud-right">
          <VerdictCard f={f} headline="F15" />
          <PrimarySignals f={f} mode={mode} />
          <Amplifiers f={f} mode={mode} showAlgos={showAlgos} setShowAlgos={setShowAlgos} />
        </div>
      </div>
    </>
  );
  return (
    <section className="section-block tight" id="player">
      <div className="wrap">
        <span className="kicker reveal">The centerpiece</span>
        <h2 className="sec-title reveal">The fear-analysis HUD, on real sessions</h2>
        <p className="sec-lead reveal">
          The same instrument the pipeline renders to video — now live. Pick a recorded session below;
          the fear-score trace doubles as the scrubber, and every readout updates from the current frame.
        </p>
      </div>

      <div className="player-stage player-wide reveal">
        {compact
          ? <div className="hud-shell">{Body}</div>
          : <HudFit>{Body}</HudFit>}

        <div className="wrap">
          <SessionRail sessions={sessions} sel={sel} onSelect={onSelect} />
        </div>
      </div>
    </section>
  );
}

/* ── F12 vs F15 — does the heartbeat help? ────────────────────────────── */
function teachCompute(c) {
  const base = 0.7 * c.fear + 0.3 * c.arousal;
  const F12 = Math.min(1, base * (1 + c.tension));
  const F15 = Math.min(1, F12 * (1 + 0.5 * c.bpmRise));
  return { base, F12, F15, fear12: F12 >= 0.70, fear15: F15 >= 0.80 };
}

function TeachReadout({ c }) {
  const r = teachCompute(c);
  const Row = ({ nm, val, color, thr }) => (
    <div className="tro">
      <div className="tro-top"><span className="nm">{nm}</span><span className="vl" style={{ color }}>{f2(val)}</span></div>
      <div className="tbar">
        <div className="tf" style={{ width: pctC(val), background: color }} />
        {thr != null ? <div className="thr" style={{ left: pctC(thr) }} /> : null}
      </div>
    </div>
  );
  return (
    <>
      <div className="teach-side">
        <div className="ts-cap">Signals this moment</div>
        <div className="teach-readout">
          <Row nm="Fear (face)" val={c.fear} color="var(--danger)" />
          <Row nm="Arousal" val={c.arousal} color="var(--arousal)" />
          <Row nm="Facial tension" val={c.tension} color="var(--tension)" />
          <Row nm="Heart-rate rise" val={c.bpmRise} color="var(--heart)" />
        </div>
      </div>
      <div className="teach-side">
        <div className="ts-cap">Two verdicts, two thresholds</div>
        <div className="teach-readout">
          <Row nm="F12 · face only" val={r.F12} color={r.fear12 ? "var(--ink)" : "var(--ink-2)"} thr={0.70} />
          <Row nm="F15 · + heart rate" val={r.F15} color={r.fear15 ? "var(--danger)" : "var(--ink-2)"} thr={0.80} />
        </div>
        <div className={`teach-verdict ${r.fear15 ? "fear" : "clear"}`}>
          <div className="tv-state">{r.fear15 ? "FEAR DETECTED" : "NO FEAR"}</div>
          <div className="tv-why">{c.f15Why}</div>
        </div>
      </div>
    </>
  );
}

function FusionSection() {
  const { FUSE, TEACH } = window.SITE;
  const [ci, setCi] = useStateC(0);
  const Eq = ({ parts, hr }) => (
    <span>{parts[0]}<span className={hr ? "hr" : "op"}>{parts[1]}</span></span>
  );
  return (
    <section className="section-block" id="fusion">
      <div className="wrap">
        <div className="fuse-head-grid">
          <div>
            <span className="kicker reveal">Fusion</span>
            <h2 className="sec-title reveal">Does the heartbeat help?</h2>
            <p className="sec-lead reveal">
              A fearful face can lie. Adding a contactless heart-rate read turns a face-only guess
              into a body-confirmed decision — a calm pulse holds back the
              <b className="mono" style={{ color: "var(--ink)" }}> F12 ≥ 0.70</b> face score, while a
              genuine cardiac response pushes the fused
              <b className="mono" style={{ color: "var(--accent)" }}> F15 over 0.80</b>.
            </p>
          </div>
          <div className="fuse-formulas reveal">
            <div className="fcard f12">
              <div className="fc-top"><span className="fc-tag">F12</span><span className="fc-name">face only</span><span className="fc-thr">≥ 0.70</span></div>
              <div className="fc-eq mono"><Eq parts={FUSE.f12.eq} /></div>
            </div>
            <div className="fcard f15">
              <div className="fc-top"><span className="fc-tag">F15</span><span className="fc-name">+ heart rate</span><span className="fc-thr">≥ 0.80</span></div>
              <div className="fc-eq mono"><Eq parts={FUSE.f15.eq} hr /></div>
            </div>
          </div>
        </div>

        <div className="teach reveal">
          <div className="teach-tabs">
            {TEACH.map((c, i) => (
              <button key={c.key} className={`teach-tab ${i === ci ? "on" : ""}`} onClick={() => setCi(i)}>
                <div className="tt-k">Case {i + 1}</div>
                <div className="tt-t">{c.tab}</div>
              </button>
            ))}
          </div>
          <div className="teach-body">
            <TeachReadout c={TEACH[ci]} />
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── Methods preview ──────────────────────────────────────────────────── */
function MethodsSection({ onNav }) {
  const M = window.SITE.METHODS;
  return (
    <section className="section-block divline" id="methods">
      <div className="wrap">
        <span className="kicker reveal">Methods</span>
        <h2 className="sec-title reveal">What made it work in the dark</h2>
        <p className="sec-lead reveal">
          Four engineering choices carry the pipeline through horror-game lighting. Each gets a dedicated
          interactive breakdown — detection fallback, contrast recovery, multiplicative fusion, and the rPPG sweep.
        </p>
        <div className="feature-grid">
          {M.map((m) => (
            <div key={m.t} className="card reveal">
              <span className="c-ic"><Icon name={m.ic} s={20} /></span>
              <h3>{m.t}</h3>
              <p>{m.p}</p>
            </div>
          ))}
          <div className="card reveal" style={{ justifyContent: "center", alignItems: "flex-start", background: "var(--surface-2)" }}>
            <span className="kicker k-plain" style={{ color: "var(--ink-4)" }}>Coming next</span>
            <h3 style={{ marginTop: 6 }}>Interactive method animations</h3>
            <p>Before/after CLAHE, the lighting-by-fallback bar chart, and the 48-config sweep — each as a live, scrubbable visual.</p>
            <button className="btn btn-ghost" style={{ marginTop: 8 }} onClick={() => onNav("player")}>Back to the HUD <Icon name="arrow" s={15} /></button>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── Footer ───────────────────────────────────────────────────────────── */
function Footer() {
  return (
    <footer className="footer" id="about">
      <div className="wrap footer-grid">
        <div className="f-brand">
          <div className="bt">La Façade <em>Fissurée</em></div>
          <p>Real-time emotion analysis for adaptive enemy AI in affective game design.
            A webcam-only, privacy-preserving approach to sensing fear — fusing facial emotion
            recognition with contactless heart rate.</p>
        </div>
        <div className="f-col">
          <h4>Project</h4>
          <ul>
            <li><span className="lab">Author</span>Ali Burak Saraç</li>
            <li><span className="lab">Advisor</span>Asst. Prof. Reis Burak Arslan</li>
            <li><span className="lab">Date</span>May 2026</li>
          </ul>
        </div>
        <div className="f-col">
          <h4>Institution</h4>
          <ul>
            <li><span className="lab">University</span>Galatasaray University</li>
            <li><span className="lab">Faculty</span>Engineering &amp; Technology</li>
            <li><span className="lab">Department</span>Computer Engineering</li>
          </ul>
        </div>
      </div>
      <div className="wrap">
        <div className="footer-base">
          <span>HSEmotion · EfficientNet-B0</span><span className="sep" />
          <span>MediaPipe FaceLandmarker</span><span className="sep" />
          <span>Haar-first detection · CLAHE</span><span className="sep" />
          <span>rPPG · POS @ 30 s</span>
        </div>
      </div>
    </footer>
  );
}

/* ── Section customizer — ships ON the site, for visitors ─────────────── */
function SectionCustomizer({ open, onClose, sections, order, hidden, move, toggle, reset, onJump }) {
  if (!open) return null;
  const ordered = order.map((id) => sections.find((s) => s.id === id)).filter(Boolean);
  const visibleCount = ordered.filter((s) => !hidden[s.id]).length;
  return (
    <>
      <div className="cust-scrim" onClick={onClose} />
      <aside className="cust" role="dialog" aria-label="Customize sections">
        <div className="cust-head">
          <div>
            <div className="cust-t">Customize this page</div>
            <div className="cust-s">{visibleCount} of {sections.length} sections shown · saved on this device</div>
          </div>
          <button className="cust-x" onClick={onClose} aria-label="Close"><Icon name="close" s={16} /></button>
        </div>
        <div className="cust-list">
          {ordered.map((s, i) => {
            const off = !!hidden[s.id];
            const locked = s.locked;
            return (
              <div className={`cust-row ${off ? "off" : ""}`} key={s.id}>
                <span className="cust-grip"><Icon name="dot6" s={16} /></span>
                <button className="cust-name" onClick={() => !off && onJump(s.id)} disabled={off} title={off ? "Hidden" : "Jump to section"}>
                  <span className="cn-i">{String(i + 1).padStart(2, "0")}</span>{s.label}
                  {locked ? <span className="cn-lock">always on</span> : null}
                </button>
                <span className="cust-acts">
                  <button onClick={() => move(s.id, -1)} disabled={i === 0} aria-label="Move up"><Icon name="up" s={14} /></button>
                  <button onClick={() => move(s.id, 1)} disabled={i === ordered.length - 1} aria-label="Move down"><Icon name="down" s={14} /></button>
                  <label className={`cust-eye ${locked ? "locked" : ""}`} title={locked ? "This section can't be hidden" : (off ? "Show" : "Hide")}>
                    <input type="checkbox" checked={!off} disabled={locked} onChange={() => toggle(s.id)} />
                    <span className="track"><span className="knob" /></span>
                  </label>
                </span>
              </div>
            );
          })}
        </div>
        <div className="cust-foot">
          <button className="cust-reset" onClick={reset}><Icon name="reset" s={14} /> Reset to default</button>
          <span className="cust-hint">Reorder with ↑↓ · toggle to hide</span>
        </div>
      </aside>
    </>
  );
}

Object.assign(window, {
  SiteIcon: Icon, Nav, Hero, PlayerSection, FusionSection, MethodsSection, Footer, SectionCustomizer,
});
