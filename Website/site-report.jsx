/* site-report.jsx — report sections: Pipeline, Results, The Game, Design Path.
 * Reuses SiteIcon (from site-components.jsx) and the in-view hook pattern.
 * Data comes from window.SITE.
 */
const { useState: useStateR, useRef: useRefR, useEffect: useEffectR } = React;

function useSeen(margin = 0.85) {
  const ref = useRefR(null);
  const [seen, setSeen] = useStateR(false);
  useEffectR(() => {
    if (seen) return;
    let done = false;
    const check = () => {
      if (done) return; const el = ref.current; if (!el) return;
      if (el.getBoundingClientRect().top < window.innerHeight * margin) { done = true; setSeen(true); }
    };
    window.addEventListener("scroll", check, { passive: true });
    window.addEventListener("resize", check);
    const id = setInterval(check, 200); check();
    return () => { window.removeEventListener("scroll", check); window.removeEventListener("resize", check); clearInterval(id); };
  }, [seen, margin]);
  return [ref, seen];
}
const pctR = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;

/* ── PIPELINE ────────────────────────────────────────────────────────── */
function PipelineSection() {
  const P = window.SITE.PIPELINE;
  const RQ = window.SITE.RQS;
  const legend = [
    ["var(--instrument)", "pre-processing"], ["var(--arousal)", "facial channel"],
    ["var(--heart)", "physiological channel"], ["var(--accent)", "fusion"], ["var(--clear)", "to the game"],
  ];
  return (
    <section className="section-block divline" id="pipeline">
      <div className="wrap">
        <span className="kicker reveal">The pipeline</span>
        <h2 className="sec-title reveal">One webcam, eight stages, fully on-device</h2>
        <p className="sec-lead reveal">
          From a single consumer webcam to enemy AI — no dedicated sensors, no cloud. Each frame is
          enhanced, the face is found, emotion and pulse are read in parallel, then fused into one fear
          score and a discrete event the game can act on. No video ever leaves the machine.
        </p>
        <div className="pipe-flow reveal">
          {P.map((s) => (
            <div className="pstage" data-tag={s.tag} key={s.k}>
              <span className="pdot" />
              <span className="pk">{s.k}</span>
              <span className="pt">{s.t}</span>
              <span className="ps">{s.s}</span>
              <span className="pd">{s.d}</span>
            </div>
          ))}
        </div>
        <div className="pipe-legend reveal">
          {legend.map(([c, l]) => <span key={l}><i style={{ background: c }} />{l}</span>)}
        </div>

        <div className="rq-band">
          {RQ.map((r) => (
            <div className={`rq ${r.status} reveal`} key={r.id}>
              <div className="rq-top">
                <span className="rq-id">{r.id}</span>
                <span className="rq-badge">{r.status === "answered" ? "answered" : r.status === "partial" ? "promising" : "open"}</span>
              </div>
              <div className="rq-q">{r.q}</div>
              <div className="rq-a">{r.a}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ── RESULTS ─────────────────────────────────────────────────────────── */
function ResultsSection() {
  const { FINDINGS, ASYMMETRY, RPPG_CONFIGS } = window.SITE;
  const [aRef, aSeen] = useSeen();
  const A = ASYMMETRY;
  return (
    <section className="section-block divline" id="results">
      <div className="wrap">
        <span className="kicker reveal">Results</span>
        <h2 className="sec-title reveal">What the evaluation found</h2>
        <p className="sec-lead reveal">
          Six subjects, 88 annotated fear events, three lighting conditions. Five findings shaped the
          production formula — and one of them began as an apparent failure.
        </p>

        <div className="findings">
          {FINDINGS.map((f) => (
            <div className="finding reveal" key={f.n}>
              <span className="fn-n">{f.n}</span>
              <div className="fn-b"><h4>{f.t}</h4><p>{f.p}</p></div>
            </div>
          ))}
        </div>

        {/* FER asymmetry */}
        <div className="asym reveal" ref={aRef}>
          <span className="kicker k-plain" style={{ color: "var(--ink-4)" }}>The unexpected asymmetry</span>
          <div className="asym-grid" style={{ marginTop: 18 }}>
            <div className="asym-card det">
              <div className="ac-role">{A.detector.role}</div>
              <div className="ac-name">{A.detector.name}</div>
              <div className="ac-sub">{A.detector.sub}</div>
              <div className="ac-f1"><span className="v">{A.detector.f1.toFixed(2)}</span><span className="k">standalone F1</span></div>
              <div className="ac-bar"><i style={{ width: aSeen ? pctR(A.detector.f1) : 0 }} /></div>
              <div className="ac-note">{A.detector.note}</div>
            </div>
            <div className="asym-op">×</div>
            <div className="asym-card amp">
              <div className="ac-role">{A.amplifier.role}</div>
              <div className="ac-name">{A.amplifier.name}</div>
              <div className="ac-sub">{A.amplifier.sub}</div>
              <div className="ac-f1"><span className="v">{A.amplifier.f1.toFixed(3)}</span><span className="k">standalone F1</span></div>
              <div className="ac-bar"><i style={{ width: aSeen ? pctR(A.amplifier.f1) : 0 }} /></div>
              <div className="ac-note">{A.amplifier.note}</div>
            </div>
          </div>
          <p className="asym-cap">
            The two FER tools were assumed interchangeable. The data said otherwise: <b>HSEmotion is the
            detector, MediaPipe is the amplifier.</b> The multiplicative formula encodes exactly this —
            tension can lift a fear reading, but can never trigger one on its own.
          </p>
        </div>

        {/* rPPG config table */}
        <div className="cfg-table reveal">
          <div className="cfg-head">
            <span>rPPG config</span><span>boost c</span><span className="h-p">precision</span>
            <span className="h-r">recall</span><span>F1</span><span className="h-why">why</span>
          </div>
          {RPPG_CONFIGS.map((c) => (
            <div className={`cfg-row ${c.best ? "best" : ""}`} key={c.cfg}>
              <span className="c-cfg">{c.best ? <span className="star">★</span> : null}{c.cfg}</span>
              <span className="c-num">{c.c.toFixed(1)}</span>
              <span className="c-num c-p">{c.p.toFixed(1)}%</span>
              <span className="c-num c-r">{c.r.toFixed(1)}%</span>
              <span className="c-f1">{c.f1.toFixed(3)}</span>
              <span className="c-why">{c.why}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ── THE GAME ────────────────────────────────────────────────────────── */
function GameSection() {
  const { ENEMIES } = window.SITE;
  return (
    <section className="section-block divline" id="game">
      <div className="wrap">
        <span className="kicker reveal">The game</span>
        <h2 className="sec-title reveal">La Façade Fissurée — where the fear signal goes</h2>
        <p className="sec-lead reveal">
          A first-person horror game in an abandoned, dimly-lit building. The pipeline is the controller:
          enemy AI reacts to your detected fear in real time. It inverts the genre — your physiological
          state is an active input, not a passive reaction.
        </p>

        <div className="game-intro">
          <div className="reveal">
            <h3 style={{ fontSize: "clamp(20px,2.4vw,28px)", letterSpacing: "-0.01em" }}>Relax-to-Win</h3>
            <p className="sec-lead" style={{ marginTop: 14 }}>
              Show fear and the enemy grows aggressive — a positive feedback loop you have to break by
              actively calming down. The only defence is emotional self-regulation: suppress visible fear
              to de-escalate the threat. The mechanic has direct relevance to exposure therapy, where
              controlled exposure paired with self-regulation training is a core technique.
            </p>
          </div>
          <div className="r2w reveal">
            <div className="r2w-t">The feedback loop</div>
            <h3>Your calm is the controller</h3>
            <div className="r2w-loop">
              <div className="r2w-step fear">
                <span className="ic"><SiteIcon name="bolt" s={17} /></span>
                <span className="tx"><b>You show fear</b> — F12/F15 crosses the threshold.</span>
              </div>
              <div className="r2w-arrow">↓</div>
              <div className="r2w-step">
                <span className="ic"><SiteIcon name="shield" s={17} /></span>
                <span className="tx">The enemy escalates: <b>WANDER → ALERT → CHASE</b>.</span>
              </div>
              <div className="r2w-arrow">↓</div>
              <div className="r2w-step calm">
                <span className="ic"><SiteIcon name="pulse" s={17} /></span>
                <span className="tx"><b>You calm down</b> — fear drops, the enemy disengages.</span>
              </div>
            </div>
          </div>
        </div>

        {/* Enemies */}
        <div className="enemies">
          {ENEMIES.map((e) => (
            <div className={`enemy ${e.accent} reveal`} key={e.key}>
              <div className="e-head">
                <div className="e-aka">{e.aka}</div>
                <div className="e-name">{e.name}</div>
                <div className="e-cls">{e.cls}</div>
              </div>
              <div className="fsm">
                {e.fsm.map((st, i) => (
                  <React.Fragment key={st}>
                    {i ? <span className="ar">→</span> : null}
                    <span className="st">{st}</span>
                  </React.Fragment>
                ))}
              </div>
              {e.matrix ? (
                <div className="e-matrix">
                  <div className="mh"></div><div className="mh">calm</div><div className="mh">afraid</div>
                  {e.matrix.map((row) => (
                    <React.Fragment key={row.see}>
                      <div className="ml">{row.see}</div>
                      <div className={row.calm.includes("game") || row.calm.includes("STRIKE") ? "strike" : (row.calm === "Despawns" ? "safe" : "")}>{row.calm}</div>
                      <div className={row.afraid.includes("game") || row.afraid.includes("STRIKE") ? "strike" : (row.afraid === "Despawns" ? "safe" : "")}>{row.afraid}</div>
                    </React.Fragment>
                  ))}
                </div>
              ) : null}
              <div className="e-channels">
                {e.channels.map((ch) => (
                  <div className="e-ch" key={ch.t}>
                    <span className="ch-t">{ch.t}</span>
                    <span className="ch-d">{ch.d}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* v1 → v2 */}
        <div className="versions">
          <div className="ver v1 reveal">
            <span className="v-tag"><span className="badge">validated</span> Prototype v1</span>
            <h3>4-room graybox</h3>
            <p>A working build with NavMesh enemy AI, a four-state FSM, safe-room triggers, TCP socket
              link, a 26-column per-frame logger, and a calibration step disguised as a “Security Scan.”
              This is the prototype the six-subject evaluation was built on.</p>
          </div>
          <div className="ver v2 reveal">
            <span className="v-tag"><span className="badge">in development</span> Redesign v2.0</span>
            <h3>17 rooms · garden ring</h3>
            <ul>
              <li>17 rooms + 6 corridors + 4 garden zones in a 50×70 m footprint (~3.5× the usable area).</li>
              <li>Wanderer gains a fifth POUNCE state and three independent detection channels.</li>
              <li>The Watcher — a new proximity entity with a 2×2 see/fear decision matrix.</li>
              <li>Key-gated escalation across 7 tiers; 20-minute target session.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── DESIGN PATH ─────────────────────────────────────────────────────── */
function DesignPathSection() {
  const { PATH } = window.SITE;
  return (
    <section className="section-block divline" id="path">
      <div className="wrap">
        <span className="kicker reveal">The design path</span>
        <h2 className="sec-title reveal">A plan that bifurcated, not derailed</h2>
        <p className="sec-lead reveal">
          The project was meant to be one straight line: build the pipeline, wire it into the game, run a
          player study. A hardware failure split that line in two — and reshaped what the evaluation could be.
        </p>
        <div className="path-track">
          {PATH.map((p) => (
            <div className={`pnode ${p.state} reveal`} key={p.n}>
              <div className="pn-rail">
                <span className="pn-num">{p.n}</span>
                <span className="pn-state">{p.state === "shipped" ? "shipped" : "disrupted"}</span>
              </div>
              <div className="pn-body">
                <div className="pn-t">{p.t}</div>
                <div className="pn-meta">{p.meta}</div>
                <p className="pn-d" dangerouslySetInnerHTML={{ __html: p.n === "03"
                  ? p.d.replace("The main development machine failed after 13 April 2026", "<b>The main development machine failed after 13 April 2026</b>")
                  : p.d }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { PipelineSection, ResultsSection, GameSection, DesignPathSection });
