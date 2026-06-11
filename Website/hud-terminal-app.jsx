/* hud-terminal-app.jsx — original neon-on-black aesthetic, same signals/decisions. */
const { useState: tUseState, useEffect: tUseEffect, useRef: tUseRef, useMemo: tUseMemo, useCallback: tUseCallback } = React;

const tFmt = (v, d = 2) => Number(v).toFixed(d);
const tPct = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;

const ALGO_COLOR = {
  CHROM: "var(--t-cyan)", POS: "var(--t-yellow)", GREEN: "var(--t-green)",
  ICA: "var(--t-pink)", WAVELET: "var(--t-aqua2)", CONSENSUS: "var(--t-white)",
};
const ALGO_SHORT = { CHROM: "CHROM", POS: "POS", GREEN: "GREEN", ICA: "ICA", WAVELET: "WAVELE", CONSENSUS: "CONSEN" };

/* ── verdict + dual F12/F15 gauges ─────────────────────────────────── */
function TVerdict({ f }) {
  const T = window.HUD.THRESH;
  const fear = f.F15 >= T.F15;
  const Row = ({ tag, score, thr, color }) => {
    const over = score >= thr;
    return (
      <div className="t-gauge">
        <span className="gtag" style={{ color }}>{tag}</span>
        <div className="gbar">
          <div className="gfill" style={{ width: tPct(score), background: over ? "var(--t-red)" : color, opacity: over ? 1 : 0.5 }} />
          <div className="t-tick" style={{ left: tPct(thr) }}><span className="tl">{tFmt(thr)}</span></div>
        </div>
        <span className="gright">
          <span className="gstate" style={{ color: over ? "var(--t-red)" : "var(--t-dim)" }}>{over ? "OVER " : "UNDER "}</span>
          <span style={{ color: over ? "var(--t-red)" : "var(--t-ink)" }}>{tFmt(score)}</span>
        </span>
      </div>
    );
  };
  return (
    <div className="t-sec">
      <div className={`t-verdict ${fear ? "fear" : "clear"}`}>
        <span className="vbox">{fear ? "!" : "\u2713"}</span>
        <span className="vstate">{fear ? "FEAR DETECTED" : "NO FEAR"}</span>
        <span className="vmeta"><div className="vn" style={{ color: fear ? "var(--t-red)" : "var(--t-ink)" }}>{tFmt(f.F15)}</div><div className="vc">F15 · FINAL</div></span>
      </div>
      <div className="t-rule" />
      <Row tag="F12" score={f.F12} thr={T.F12} color="var(--t-cyan)" />
      <Row tag="F15" score={f.F15} thr={T.F15} color="var(--t-yellow)" />
      <div className="t-note">F12 face-only crosses at 0.70 · F15 (+ heart rate) crosses at 0.80 — separate lines, separate thresholds.</div>
    </div>
  );
}

/* ── HS TRIGGERS ───────────────────────────────────────────────────── */
function THsTriggers({ f, mode }) {
  const labels = window.HUD.EMOTION_LABELS;
  const others = labels.filter((l) => l !== "Fear");
  const tech = mode === "tech";
  return (
    <div className="t-sec">
      <div className="t-head" style={{ color: "var(--t-cyan)" }}>HS TRIGGERS<span className="tag" style={{ color: "var(--t-dim)" }}>[H]</span></div>
      <div className="t-rule" />
      <div className="t-row t-hero">
        <span className="rl" style={{ color: "var(--t-red)" }}>{tech ? "hs_fear" : "Fear"}</span>
        <span className="rv" style={{ color: "var(--t-red)" }}>{tFmt(f.hs_fear)}</span>
        <div className="t-bar"><div className="fill" style={{ width: tPct(f.hs_fear), background: "var(--t-red)" }} /></div>
      </div>
      <div className="t-row">
        <span className="rl" style={{ color: "var(--t-orange)" }}>{tech ? "hs_arousal" : "Arousal"}</span>
        <span className="rv" style={{ color: "var(--t-orange)" }}>{tFmt(f.hs_arousal)}</span>
        <div className="t-bar"><div className="fill" style={{ width: tPct(f.hs_arousal), background: "var(--t-orange)" }} /></div>
      </div>
      <div className="t-note" style={{ color: "var(--t-ink)", fontSize: 17, marginTop: 6, marginBottom: 8 }}>
        base = 0.7&times;fear + 0.3&times;arousal = <b>{tFmt(f.base)}</b> &nbsp;·&nbsp; Dom: {f.dom} ({tFmt(f.domScore)})
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", columnGap: 26, rowGap: 0 }}>
        {others.map((l) => (
          <div key={l} className={`t-emo ${l === f.dom ? "dom" : ""}`}>
            <span className="el">{l}</span>
            <span className="ev">{tFmt(f.emotions[l])}</span>
            <span className="eb"><span className="ef" style={{ width: tPct(f.emotions[l]) }} /></span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── MP TRIGGERS ───────────────────────────────────────────────────── */
function TMpTriggers({ f, mode }) {
  const tech = mode === "tech";
  return (
    <div className="t-sec">
      <div className="t-head" style={{ color: "var(--t-yellow)" }}>MP TRIGGERS</div>
      <div className="t-rule" />
      <div className="t-row t-hero" style={{ borderLeftColor: "var(--t-red)" }}>
        <span className="rl" style={{ color: "var(--t-red)" }}>{tech ? "mp_tension" : "Tension"}</span>
        <span className="rv" style={{ color: "var(--t-red)" }}>{tFmt(f.mp_tension)}</span>
        <div className="t-bar"><div className="fill" style={{ width: tPct(f.mp_tension), background: "var(--t-red)" }} /></div>
      </div>
      <div className="t-row">
        <span className="rl" style={{ color: "var(--t-green)" }}>Valence</span>
        <span className="rv" style={{ color: f.valence < 0 ? "var(--t-red)" : "var(--t-green)" }}>{f.valence >= 0 ? "+" : ""}{tFmt(f.valence)}</span>
        <div className="t-bar"><div className="fill" style={{ left: "50%", width: `${Math.abs(f.valence) * 50}%`, background: f.valence < 0 ? "var(--t-red)" : "var(--t-green)", position: "absolute", top: 0, bottom: 0, transform: f.valence < 0 ? "translateX(-100%)" : "none" }} /></div>
      </div>
      <div className="t-row">
        <span className="rl" style={{ color: "var(--t-cyan)" }}>Smile</span>
        <span className="rv" style={{ color: "var(--t-cyan)" }}>{tFmt(f.smile)}</span>
        <div className="t-bar"><div className="fill" style={{ width: tPct(f.smile), background: "var(--t-cyan)" }} /></div>
      </div>
      <div className="t-row">
        <span className="rl" style={{ color: "var(--t-dim)" }}>Startle</span>
        <span className="rv" style={{ color: "var(--t-dim)" }}>{tFmt(f.startle, 1)}/s</span>
        <div className="t-bar"><div className="fill" style={{ width: tPct(f.startle / 6), background: "var(--t-dim)" }} /></div>
      </div>
      {/* [NOTE 2026-06-01] The old "State: [STRESS]" tag is intentionally removed:
          only mp_tension feeds the formula; the tag read like a verdict. */}
    </div>
  );
}

/* ── rPPG ──────────────────────────────────────────────────────────── */
function TRppg({ f, mode }) {
  const order = window.HUD.RPPG_ALGOS;
  const tech = mode === "tech";
  const mm = String(Math.floor(f.t / 60)).padStart(2, "0");
  const ss = String(Math.floor(f.t % 60)).padStart(2, "0");
  return (
    <div className="t-sec">
      <div className="t-head" style={{ color: "var(--t-yellow)" }}>rPPG BPM<span className="tag" style={{ color: "var(--t-dim)" }}>t={mm}:{ss}</span></div>
      <div className="t-rule" />
      {order.map((a) => {
        const v = f.algos[a].bpm;
        return (
          <div key={a} className={`t-algo ${a === "POS" ? "headline" : ""}`}>
            <span className="al" style={{ color: ALGO_COLOR[a] }}>{ALGO_SHORT[a]}{a === "POS" ? " \u25c0" : ""}</span>
            <span className="av" style={{ color: ALGO_COLOR[a] }}>{tFmt(v, 1)}</span>
            <span className="ab"><span className="af" style={{ width: tPct((v - 50) / 80), background: ALGO_COLOR[a] }} /></span>
            <span className="snr">{tFmt(f.algos[a].snr, 1)}</span>
          </div>
        );
      })}
      <div className="t-note">POS is the production algorithm · rise {tech ? "bpm_norm " : ""}{tFmt(f.bpm_norm)} (rest {f.baseline} &rarr; now {Math.round(f.bpm)}) · SNR on right</div>
    </div>
  );
}

/* ── FUSION chain ──────────────────────────────────────────────────── */
function TFusion({ f }) {
  const T = window.HUD.THRESH;
  return (
    <div className="t-sec">
      <div className="t-head" style={{ color: "var(--t-cyan)" }}>FUSION</div>
      <div className="t-rule" />
      <div className="t-chain">
        <div className="line">
          <span className="node">base {tFmt(f.base)}</span>
          <span className="op">&times; (1 + tension {tFmt(f.mp_tension)}) =</span>
          <span className="res" style={{ color: f.F12 >= T.F12 ? "var(--t-red)" : "var(--t-ink)" }}>F12 {tFmt(f.F12)}</span>
        </div>
        <div className="line">
          <span className="node">F12 {tFmt(f.F12)}</span>
          <span className="op">&times; (1 + 0.5&middot;bpm_norm {tFmt(f.bpm_norm)}) =</span>
          <span className="res" style={{ color: f.F15 >= T.F15 ? "var(--t-red)" : "var(--t-yellow)" }}>F15 {tFmt(f.F15)}</span>
        </div>
      </div>
    </div>
  );
}

/* ── shell: video + telemetry ──────────────────────────────────────── */
function TVideo({ f }) {
  const mm = String(Math.floor(f.t / 60)).padStart(2, "0");
  const ss = String(Math.floor(f.t % 60)).padStart(2, "0");
  return (
    <div className="t-left">
      <div className="t-video">
        <image-slot id="t-subject-feed" shape="rect" fit="cover" placeholder="Drop your session frame"></image-slot>
        <div className="t-vid-overlay">
          <div className="t-roi" style={{ left: "40%", top: "22%", width: "22%", height: "13%" }}>
            <span className="lab">forehead</span>
          </div>
        </div>
        <div className="t-tele">
          <span className="time">TIME: {mm}:{ss}</span>
          <span className="lat">{Math.round(f.latency)}ms</span>
        </div>
        <div className="t-frame">Frame {String(f.frame).padStart(4, "0")} &middot; {f.fps.toFixed(1)} fps</div>
      </div>
    </div>
  );
}

const T_LABEL_MAP = { "Plain": "plain", "Technical": "tech" };
const T_SPEED_MAP = { "0.5\u00d7": 0.5, "1\u00d7": 1, "2\u00d7": 2 };
const T_DEFAULTS = /*EDITMODE-BEGIN*/{
  "labels": "Plain",
  "speed": "1\u00d7"
}/*EDITMODE-END*/;
const T_STORE = "terminalHudTime";

function TApp() {
  const [t, setTweak] = useTweaks(T_DEFAULTS);
  const mode = T_LABEL_MAP[t.labels] || "plain";
  const speed = T_SPEED_MAP[t.speed] || 1;
  const D = window.HUD.DURATION;

  const init = (() => { const v = parseFloat(localStorage.getItem(T_STORE)); return isNaN(v) ? 0 : Math.min(D, Math.max(0, v)); })();
  const [time, setTime] = tUseState(init);
  const [playing, setPlaying] = tUseState(true);
  const timeRef = tUseRef(init), playingRef = tUseRef(true), speedRef = tUseRef(speed);
  tUseEffect(() => { speedRef.current = speed; }, [speed]);
  tUseEffect(() => { playingRef.current = playing; }, [playing]);

  tUseEffect(() => {
    let raf, last = performance.now(), acc = 0; const STEP = 1 / 30;
    const loop = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      if (playingRef.current) {
        let nt = timeRef.current + dt * speedRef.current;
        if (nt >= D) nt = 0;
        timeRef.current = nt; acc += dt;
        if (acc >= STEP) { acc = 0; setTime(nt); }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [D]);
  tUseEffect(() => {
    const id = setInterval(() => localStorage.setItem(T_STORE, timeRef.current.toFixed(2)), 500);
    return () => { clearInterval(id); localStorage.setItem(T_STORE, timeRef.current.toFixed(2)); };
  }, []);

  const seek = tUseCallback((nt) => { nt = Math.max(0, Math.min(D, nt)); timeRef.current = nt; setTime(nt); }, [D]);
  tUseEffect(() => { window.__seek = (nt) => { setPlaying(false); seek(nt); }; }, [seek]);

  const [scale, setScale] = tUseState(1);
  tUseEffect(() => {
    const fit = () => setScale(Math.min(window.innerWidth / 1920, window.innerHeight / 1080));
    fit(); window.addEventListener("resize", fit); return () => window.removeEventListener("resize", fit);
  }, []);

  const f = tUseMemo(() => window.HUD.computeFrame(time), [time]);
  const mm = String(Math.floor(time / 60)).padStart(2, "0"), ss = String(Math.floor(time % 60)).padStart(2, "0");

  return (
    <div id="stage">
      <div id="canvas" style={{ transform: `scale(${scale})` }}>
        <TVideo f={f} />
        <div className="t-right">
          <TVerdict f={f} />
          <THsTriggers f={f} mode={mode} />
          <TMpTriggers f={f} mode={mode} />
          <TRppg f={f} mode={mode} />
          <TFusion f={f} />
          {/* scrubber */}
          <div style={{ marginTop: "auto", display: "flex", alignItems: "center", gap: 14, paddingTop: 8 }}>
            <button onClick={() => setPlaying((p) => !p)} style={{ width: 38, height: 38, flex: "none", background: "#111", border: "1px solid var(--t-rule)", color: "var(--t-ink)", cursor: "pointer", fontFamily: "var(--t-mono)" }}>
              {playing ? "❙❙" : "▶"}
            </button>
            <input type="range" min="0" max={D} step="0.05" value={time}
              onChange={(e) => { setPlaying(false); seek(parseFloat(e.target.value)); }}
              style={{ flex: 1, accentColor: "var(--t-yellow)" }} />
            <span style={{ fontSize: 17, color: "var(--t-dim)", minWidth: 110, textAlign: "right" }}>{mm}:{ss} / {String(Math.floor(D / 60)).padStart(2, "0")}:{String(Math.floor(D % 60)).padStart(2, "0")}</span>
          </div>
        </div>
      </div>

      <TweaksPanel>
        <TweakSection label="Readout" />
        <TweakRadio label="Labels" value={t.labels} options={["Plain", "Technical"]} onChange={(v) => setTweak("labels", v)} />
        <TweakSection label="Playback" />
        <TweakRadio label="Speed" value={t.speed} options={["0.5\u00d7", "1\u00d7", "2\u00d7"]} onChange={(v) => setTweak("speed", v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<TApp />);
