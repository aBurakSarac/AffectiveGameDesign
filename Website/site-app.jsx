/* site-app.jsx — top-level app: playback clock, session + label state,
 * smooth-scroll nav with scroll-spy, and reveal-on-scroll. Mounts the site.
 */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

const LABEL_MAP = { "Plain": "plainOnly", "Plain + tech": "plain", "Technical": "tech" };
const SPEED_MAP = { "0.5×": 0.5, "1×": 1, "2×": 2 };
const STORE_T = "facadeHudTime";
const STORE_VIEW = "facadeView_v1";

// Default running order (decided): context first, then the instrument, then the
// findings, the game, and the reference appendix. `locked` sections can't be hidden.
const SECTION_DEFS = [
  { id: "overview",    label: "Overview",    locked: true },
  { id: "background",  label: "Background" },
  { id: "pipeline",    label: "Pipeline" },
  { id: "player",      label: "HUD Player",  locked: true },
  { id: "fusion",      label: "F12 vs F15" },
  { id: "methods",     label: "Methods" },
  { id: "results",     label: "Results" },
  { id: "performance", label: "Performance" },
  { id: "game",        label: "The Game" },
  { id: "path",        label: "Design Path" },
  { id: "outlook",     label: "Outlook" },
  { id: "glossary",    label: "Glossary" },
];
const DEFAULT_ORDER = SECTION_DEFS.map((s) => s.id);
// short labels for the top nav (kept terse to fit the row)
const NAV_LABELS = { overview: "Overview", background: "Background", pipeline: "Pipeline", player: "Player", fusion: "Fusion", methods: "Methods", results: "Results", performance: "Perf", game: "Game", path: "Path", outlook: "Outlook", glossary: "Glossary" };

function loadView() {
  try {
    const v = JSON.parse(localStorage.getItem(STORE_VIEW));
    if (!v || !Array.isArray(v.order)) throw 0;
    // reconcile with current defs (add new sections, drop removed ones)
    const known = new Set(DEFAULT_ORDER);
    const order = v.order.filter((id) => known.has(id));
    DEFAULT_ORDER.forEach((id) => { if (!order.includes(id)) order.push(id); });
    const hidden = {};
    Object.keys(v.hidden || {}).forEach((id) => { if (known.has(id)) hidden[id] = !!v.hidden[id]; });
    SECTION_DEFS.forEach((s) => { if (s.locked) delete hidden[s.id]; });
    return { order, hidden };
  } catch (e) { return { order: [...DEFAULT_ORDER], hidden: {} }; }
}

function SiteApp() {
  // ── playback state ──
  const [D, setD] = useState(1);
  const init = (() => { const v = parseFloat(localStorage.getItem(STORE_T)); return isNaN(v) ? 0 : Math.max(0, v); })();
  const [time, setTime] = useState(init);
  const [playing, setPlaying] = useState(false);
  const [labels, setLabels] = useState("Plain + tech");
  const [showAlgos, setShowAlgos] = useState(true);
  const [speed, setSpeed] = useState("1×");
  const [sel, setSel] = useState(window.SITE.SESSIONS[0].id);
  const [sessionReady, setSessionReady] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const [active, setActive] = useState("overview");
  const [custOpen, setCustOpen] = useState(false);
  const videoRef = useRef(null);
  const sessionVer = useRef(0);

  // ── section view (order + hidden), persisted on this device ──
  const [view, setView] = useState(loadView);
  useEffect(() => { localStorage.setItem(STORE_VIEW, JSON.stringify(view)); }, [view]);
  const moveSection = useCallback((id, dir) => setView((v) => {
    const order = [...v.order]; const i = order.indexOf(id); const j = i + dir;
    if (i < 0 || j < 0 || j >= order.length) return v;
    [order[i], order[j]] = [order[j], order[i]];
    return { ...v, order };
  }), []);
  const toggleSection = useCallback((id) => setView((v) => {
    if (SECTION_DEFS.find((s) => s.id === id)?.locked) return v;
    return { ...v, hidden: { ...v.hidden, [id]: !v.hidden[id] } };
  }), []);
  const resetView = useCallback(() => setView({ order: [...DEFAULT_ORDER], hidden: {} }), []);

  const visibleOrder = useMemo(() => view.order.filter((id) => !view.hidden[id]), [view]);
  const navLinks = useMemo(() => visibleOrder.filter((id) => id !== "overview").map((id) => [id, NAV_LABELS[id] || id]), [visibleOrder]);

  // compact = stacked fluid HUD; wide = scaled fixed canvas. 920px crossover.
  const [compact, setCompact] = useState(() => window.innerWidth < 920);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 919px)");
    const on = () => setCompact(mq.matches);
    on(); mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const mode = LABEL_MAP[labels] || "plain";
  const timeRef = useRef(init), speedRef = useRef(1);
  useEffect(() => {
    speedRef.current = SPEED_MAP[speed] || 1;
    if (videoRef.current) videoRef.current.playbackRate = speedRef.current;
  }, [speed]);

  // ── Load first session on mount ──
  useEffect(() => {
    const firstStem = window.SITE.SESSIONS[0].id;
    window.HUD.loadSession(firstStem).then(() => {
      setD(window.HUD.DURATION);
      setSessionReady(true);
      sessionVer.current++;
    });
  }, []);

  // ── Video-driven playback clock ──
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid || !sessionReady) return;
    let raf;
    const loop = () => {
      if (vid.readyState >= 2) {
        const ct = vid.currentTime;
        if (Math.abs(ct - timeRef.current) > 0.016) {
          timeRef.current = ct;
          setTime(ct);
        }
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [sessionReady, sel]);

  // ── Persist position ──
  useEffect(() => {
    const id = setInterval(() => localStorage.setItem(STORE_T, timeRef.current.toFixed(2)), 500);
    return () => { clearInterval(id); localStorage.setItem(STORE_T, timeRef.current.toFixed(2)); };
  }, []);

  const seek = useCallback((nt) => {
    const vid = videoRef.current;
    nt = Math.max(0, Math.min(D, nt));
    timeRef.current = nt;
    setTime(nt);
    if (vid && vid.readyState >= 1) vid.currentTime = nt;
  }, [D]);

  const toggle = useCallback(() => {
    const vid = videoRef.current;
    if (!vid) return;
    if (vid.paused) { vid.play().catch(() => {}); setPlaying(true); }
    else { vid.pause(); setPlaying(false); }
  }, []);

  const selectSession = useCallback(async (id) => {
    const vid = videoRef.current;
    if (vid) vid.pause();
    setPlaying(false);
    setSel(id);
    await window.HUD.loadSession(id);
    const newD = window.HUD.DURATION;
    setD(newD);
    sessionVer.current++;
    if (vid) {
      vid.src = window.HUD.getVideoUrl(id);
      vid.load();
      vid.currentTime = 0;
      vid.playbackRate = speedRef.current;
      vid.oncanplay = () => {
        vid.oncanplay = null;
        vid.play().catch(() => {});
        setPlaying(true);
      };
    }
    timeRef.current = 0;
    setTime(0);
  }, []);

  const traceVer = sessionVer.current;
  const traces = useMemo(() => ({
    F15: window.HUD.sampleTrace("F15", D > 200 ? 0.5 : 0.15),
    F12: window.HUD.sampleTrace("F12", D > 200 ? 0.5 : 0.15),
  }), [traceVer, D]);
  const f = useMemo(() => window.HUD.computeFrame(time), [time, traceVer]);

  // ── smooth scroll nav ──
  const navTo = useCallback((id) => {
    const el = document.getElementById(id); if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - 56;
    window.scrollTo({ top, behavior: "smooth" });
  }, []);

  // ── scroll-spy + reveal (scroll-listener based — robust in every context,
  //    incl. preview iframes where IntersectionObserver never fires) ──
  useEffect(() => {
    let ticking = false;
    const update = () => {
      ticking = false;
      const vh = window.innerHeight;
      // reveal anything whose top has entered the lower 92% of the viewport
      document.querySelectorAll(".reveal:not(.in)").forEach((el) => {
        if (el.getBoundingClientRect().top < vh * 0.92) el.classList.add("in");
      });
      // active section = last one whose top has crossed the nav line
      let cur = visibleOrder[0] || "overview";
      for (const id of visibleOrder) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top - 72 <= 0) cur = id;
      }
      setActive(cur);
    };
    const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
    const warm = [80, 240, 600].map((d) => setTimeout(update, d)); // catch late layout/font shifts
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); warm.forEach(clearTimeout); };
  }, [visibleOrder]);

  // section id → rendered node (built once per relevant-state change)
  const nodes = useMemo(() => ({
    overview: <Hero onNav={navTo} />,
    background: <BackgroundSection />,
    pipeline: <PipelineSection />,
    player: <PlayerSection
      f={f} traces={traces} D={D} playing={playing} onSeek={seek} onToggle={toggle}
      sessions={window.SITE.SESSIONS} sel={sel} onSelect={selectSession}
      mode={mode} showAlgos={showAlgos} setShowAlgos={setShowAlgos} speed={speed} setSpeed={setSpeed}
      compact={compact} videoRef={videoRef} sessionReady={sessionReady} />,
    fusion: <FusionSection />,
    methods: <MethodsSectionFull onNav={navTo} />,
    results: <ResultsSection />,
    performance: <PerformanceSection />,
    game: <GameSection />,
    path: <DesignPathSection />,
    outlook: <OutlookSection />,
    glossary: <GlossarySection />,
  }), [f, traces, playing, seek, toggle, sel, selectSession, mode, showAlgos, speed, compact, navTo]);

  return (
    <>
      <Nav active={active} open={navOpen} setOpen={setNavOpen} onNav={navTo}
        links={navLinks} onCustomize={() => setCustOpen(true)} />
      <main>
        {visibleOrder.map((id) => <React.Fragment key={id}>{nodes[id]}</React.Fragment>)}
      </main>
      <Footer />
      <SectionCustomizer
        open={custOpen} onClose={() => setCustOpen(false)}
        sections={SECTION_DEFS} order={view.order} hidden={view.hidden}
        move={moveSection} toggle={toggleSection} reset={resetView}
        onJump={(id) => { setCustOpen(false); setTimeout(() => navTo(id), 60); }} />
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<SiteApp />);
