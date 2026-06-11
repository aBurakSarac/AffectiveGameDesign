/* hud-app.jsx — clock, scaling, scrubbing, tweaks. Mounts the merged HUD. */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "headline": "F15",
  "labels": "Plain + tech",
  "showAlgos": true,
  "speed": "1\u00d7",
  "accent": "#e85742"
}/*EDITMODE-END*/;

const LABEL_MAP = { "Plain + tech": "plain", "Plain only": "plainOnly", "Technical": "tech" };
const SPEED_MAP = { "0.5\u00d7": 0.5, "1\u00d7": 1, "2\u00d7": 2 };
const STORE_KEY = "mergedHudTime";

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const headline = t.headline === "F12" ? "F12" : "F15";
  const mode = LABEL_MAP[t.labels] || "plain";
  const speed = SPEED_MAP[t.speed] || 1;

  const D = window.HUD.DURATION;
  const init = (() => { const v = parseFloat(localStorage.getItem(STORE_KEY)); return isNaN(v) ? 0 : Math.min(D, Math.max(0, v)); })();
  const [time, setTime] = useState(init);
  const [playing, setPlaying] = useState(true);

  const timeRef = useRef(init);
  const playingRef = useRef(true);
  const speedRef = useRef(speed);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { playingRef.current = playing; }, [playing]);

  // rAF clock, throttled render to ~30fps
  useEffect(() => {
    let raf, last = performance.now(), acc = 0;
    const STEP = 1 / 30;
    const loop = (now) => {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      if (playingRef.current) {
        let nt = timeRef.current + dt * speedRef.current;
        if (nt >= D) nt = 0; // loop
        timeRef.current = nt; acc += dt;
        if (acc >= STEP) { acc = 0; setTime(nt); }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [D]);

  // persist position (throttled)
  useEffect(() => {
    const id = setInterval(() => localStorage.setItem(STORE_KEY, timeRef.current.toFixed(2)), 500);
    return () => { clearInterval(id); localStorage.setItem(STORE_KEY, timeRef.current.toFixed(2)); };
  }, []);

  const seek = useCallback((nt) => { nt = Math.max(0, Math.min(D, nt)); timeRef.current = nt; setTime(nt); }, [D]);
  const toggle = useCallback(() => setPlaying((p) => !p), []);

  // debug/screenshot hook
  useEffect(() => { window.__seek = (nt) => { setPlaying(false); seek(nt); }; }, [seek]);

  // accent var
  useEffect(() => { document.documentElement.style.setProperty("--accent", t.accent); }, [t.accent]);

  // scaling
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const fit = () => setScale(Math.min(window.innerWidth / 1920, window.innerHeight / 1080));
    fit(); window.addEventListener("resize", fit); return () => window.removeEventListener("resize", fit);
  }, []);

  const traces = useMemo(() => ({ F15: window.HUD.sampleTrace("F15", 0.15), F12: window.HUD.sampleTrace("F12", 0.15) }), []);
  const f = useMemo(() => window.HUD.computeFrame(time), [time]);

  return (
    <div id="stage">
      <div id="canvas" style={{ transform: `scale(${scale})` }}>
        <TelemetryBar f={f} />
        <div className="body">
          <div className="left">
            <VideoArea f={f} />
            <TraceTimeline f={f} traces={traces} headline={headline}
              playing={playing} onSeek={seek} onToggle={toggle} />
          </div>
          <div className="right">
            <VerdictCard f={f} headline={headline} />
            <PrimarySignals f={f} mode={mode} />
            <Amplifiers f={f} mode={mode} showAlgos={t.showAlgos} />
            <FormulaChain f={f} headline={headline} />
          </div>
        </div>
      </div>

      <TweaksPanel>
        {/* [COMMENTED OUT 2026-06-01 — headline F12/F15 toggle removed per review:
            both scores are now shown side-by-side on the verdict + chain + trace,
            so there is nothing to toggle. Kept for rollback]
        <TweakSection label="Score & verdict" />
        <TweakRadio label="Headline score" value={t.headline} options={["F15", "F12"]}
          onChange={(v) => setTweak("headline", v)} />
        */}
        <TweakSection label="Readout" />
        <TweakRadio label="Labels" value={t.labels} options={["Plain + tech", "Plain only", "Technical"]}
          onChange={(v) => setTweak("labels", v)} />
        <TweakToggle label="Show other rPPG algorithms" value={t.showAlgos}
          onChange={(v) => setTweak("showAlgos", v)} />
        <TweakSection label="Playback" />
        <TweakRadio label="Speed" value={t.speed} options={["0.5\u00d7", "1\u00d7", "2\u00d7"]}
          onChange={(v) => setTweak("speed", v)} />
        <TweakSection label="Theme" />
        <TweakColor label="Score accent" value={t.accent}
          options={["#e85742", "#d98a3a", "#c2557a", "#5a8fe0"]}
          onChange={(v) => setTweak("accent", v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
