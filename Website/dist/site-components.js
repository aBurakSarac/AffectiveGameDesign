const { useMemo: useMemoC, useRef: useRefC, useCallback: useCbC, useState: useStateC } = React;
const f2 = (v, d = 2) => Number(v).toFixed(d);
const pctC = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const t = (k, fb) => window.I18N.t(k, fb);
const tSlot = (str, slot, node) => window.I18N.tSlot(str, slot, node);
function Icon({ name, s = 18 }) {
  const p = {
    play: /* @__PURE__ */ React.createElement("path", { d: "M5 3 L19 12 L5 21 Z", fill: "currentColor", stroke: "none" }),
    pause: /* @__PURE__ */ React.createElement("g", { fill: "currentColor", stroke: "none" }, /* @__PURE__ */ React.createElement("rect", { x: "5", y: "3", width: "5", height: "18", rx: "1" }), /* @__PURE__ */ React.createElement("rect", { x: "14", y: "3", width: "5", height: "18", rx: "1" })),
    arrow: /* @__PURE__ */ React.createElement("path", { d: "M4 12 H20 M14 6 L20 12 L14 18" }),
    menu: /* @__PURE__ */ React.createElement("path", { d: "M3 6 H21 M3 12 H21 M3 18 H21" }),
    close: /* @__PURE__ */ React.createElement("path", { d: "M5 5 L19 19 M19 5 L5 19" }),
    sliders: /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("path", { d: "M4 6h10M18 6h2M4 12h2M10 12h10M4 18h14M20 18h0" }), /* @__PURE__ */ React.createElement("circle", { cx: "16", cy: "6", r: "2" }), /* @__PURE__ */ React.createElement("circle", { cx: "8", cy: "12", r: "2" }), /* @__PURE__ */ React.createElement("circle", { cx: "18", cy: "18", r: "2" })),
    up: /* @__PURE__ */ React.createElement("path", { d: "M12 19 V5 M6 11 L12 5 L18 11" }),
    down: /* @__PURE__ */ React.createElement("path", { d: "M12 5 V19 M6 13 L12 19 L18 13" }),
    reset: /* @__PURE__ */ React.createElement("path", { d: "M3 12 a9 9 0 1 0 3-6.7 M3 4 v4 h4" }),
    dot6: /* @__PURE__ */ React.createElement("g", { fill: "currentColor", stroke: "none" }, /* @__PURE__ */ React.createElement("circle", { cx: "9", cy: "6", r: "1.4" }), /* @__PURE__ */ React.createElement("circle", { cx: "15", cy: "6", r: "1.4" }), /* @__PURE__ */ React.createElement("circle", { cx: "9", cy: "12", r: "1.4" }), /* @__PURE__ */ React.createElement("circle", { cx: "15", cy: "12", r: "1.4" }), /* @__PURE__ */ React.createElement("circle", { cx: "9", cy: "18", r: "1.4" }), /* @__PURE__ */ React.createElement("circle", { cx: "15", cy: "18", r: "1.4" })),
    info: /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("circle", { cx: "12", cy: "12", r: "9" }), /* @__PURE__ */ React.createElement("path", { d: "M12 11 V16 M12 8 h0.01" })),
    shield: /* @__PURE__ */ React.createElement("path", { d: "M12 3 L20 6 V11 C20 16 16.5 19.5 12 21 C7.5 19.5 4 16 4 11 V6 Z M9 12 l2 2 l4 -4" }),
    bolt: /* @__PURE__ */ React.createElement("path", { d: "M13 3 L5 13 H11 L10 21 L19 10 H13 Z" }),
    eye: /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("path", { d: "M2 12 S6 5 12 5 S22 12 22 12 S18 19 12 19 S2 12 2 12 Z" }), /* @__PURE__ */ React.createElement("circle", { cx: "12", cy: "12", r: "3" })),
    lighting: /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("circle", { cx: "12", cy: "12", r: "4" }), /* @__PURE__ */ React.createElement("path", { d: "M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" })),
    contrast: /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("circle", { cx: "12", cy: "12", r: "9" }), /* @__PURE__ */ React.createElement("path", { d: "M12 3 a9 9 0 0 0 0 18 Z", fill: "currentColor", stroke: "none" })),
    fuse: /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("circle", { cx: "7", cy: "7", r: "3.4" }), /* @__PURE__ */ React.createElement("circle", { cx: "17", cy: "7", r: "3.4" }), /* @__PURE__ */ React.createElement("path", { d: "M7 10.4 V13 a5 5 0 0 0 10 0 V10.4 M12 18 v3" })),
    pulse: /* @__PURE__ */ React.createElement("path", { d: "M2 12 H7 L9 6 L13 18 L15 12 H22" }),
    globe: /* @__PURE__ */ React.createElement("g", null, /* @__PURE__ */ React.createElement("circle", { cx: "12", cy: "12", r: "9" }), /* @__PURE__ */ React.createElement("path", { d: "M3 12 h18 M12 3 c3 3 3 15 0 18 M12 3 c-3 3 -3 15 0 18" }))
  }[name];
  return /* @__PURE__ */ React.createElement("svg", { width: s, height: s, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.8", strokeLinecap: "round", strokeLinejoin: "round" }, p);
}
function LangSwitcher({ lang, setLang }) {
  return /* @__PURE__ */ React.createElement("div", { className: "lang-switch", role: "group", "aria-label": t("nav.language", "Language") }, /* @__PURE__ */ React.createElement(Icon, { name: "globe", s: 15 }), window.I18N.SUPPORTED.map((code) => /* @__PURE__ */ React.createElement(
    "button",
    {
      key: code,
      className: code === lang ? "on" : "",
      onClick: () => setLang(code),
      "aria-pressed": code === lang,
      title: window.I18N.NAMES[code]
    },
    code.toUpperCase()
  )));
}
function Nav({ active, open, setOpen, onNav, links, onCustomize, hidden, lang, setLang }) {
  const go = (e, id) => {
    e.preventDefault();
    setOpen(false);
    onNav(id);
  };
  return /* @__PURE__ */ React.createElement("nav", { className: `nav ${hidden ? "nav-hidden" : ""}` }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("a", { href: "#overview", className: "nav-brand", onClick: (e) => go(e, "overview") }, /* @__PURE__ */ React.createElement("span", { className: "rec-dot" }), /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("span", { className: "bt" }, "La Fa\xE7ade ", /* @__PURE__ */ React.createElement("em", null, "Fissur\xE9e"))), /* @__PURE__ */ React.createElement("span", { className: "bs" }, t("nav.brandTag", "FER + rPPG"))), /* @__PURE__ */ React.createElement("div", { className: "nav-right" }, /* @__PURE__ */ React.createElement(LangSwitcher, { lang, setLang }), /* @__PURE__ */ React.createElement("a", { href: "#player", className: "nav-cta", onClick: (e) => go(e, "player") }, t("nav.openHud", "Open the HUD \u25B8")), /* @__PURE__ */ React.createElement("button", { className: "nav-custom", onClick: onCustomize, "aria-label": t("nav.customize", "Reorder & hide sections"), title: t("nav.customize", "Reorder & hide sections") }, /* @__PURE__ */ React.createElement(Icon, { name: "sliders", s: 17 })), /* @__PURE__ */ React.createElement("button", { className: "nav-toggle", onClick: () => setOpen((o) => !o), "aria-label": t("nav.menu", "Menu") }, /* @__PURE__ */ React.createElement(Icon, { name: open ? "close" : "menu" }))), /* @__PURE__ */ React.createElement("div", { className: `nav-links ${open ? "open" : ""}` }, links.map(([id, lab]) => /* @__PURE__ */ React.createElement("a", { key: id, href: `#${id}`, className: active === id ? "active" : "", onClick: (e) => go(e, id) }, lab)))));
}
function SideNav({ active, links, onNav }) {
  return /* @__PURE__ */ React.createElement("aside", { className: "sidenav", "aria-label": "Sections" }, /* @__PURE__ */ React.createElement("ul", { className: "sidenav-list" }, links.map(([id, lab]) => /* @__PURE__ */ React.createElement("li", { key: id }, /* @__PURE__ */ React.createElement(
    "a",
    {
      href: `#${id}`,
      className: active === id ? "active" : "",
      onClick: (e) => {
        e.preventDefault();
        onNav(id);
      }
    },
    /* @__PURE__ */ React.createElement("span", { className: "sn-dot" }),
    /* @__PURE__ */ React.createElement("span", { className: "sn-label" }, lab)
  )))));
}
function Hero({ onNav }) {
  const R = window.SITE.RESULTS;
  return /* @__PURE__ */ React.createElement("header", { className: "hero section-block", id: "overview" }, /* @__PURE__ */ React.createElement("div", { className: "hero-aura" }), /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "hero-tag reveal" }, /* @__PURE__ */ React.createElement("span", { className: "pill" }, t("hero.badge", "Final project")), t("hero.affil", "Galatasaray University \xB7 Computer Engineering")), /* @__PURE__ */ React.createElement("h1", { className: "reveal" }, (() => {
    const p = tSlot(
      t("hero.title", "Can a plain webcam feel a player's {fear}?"),
      "fear",
      /* @__PURE__ */ React.createElement("em", null, t("hero.fearWord", "fear"))
    );
    return /* @__PURE__ */ React.createElement(React.Fragment, null, p[0], p[1], p[2]);
  })()), /* @__PURE__ */ React.createElement("p", { className: "hero-sub reveal" }, t("hero.sub")), /* @__PURE__ */ React.createElement("div", { className: "hero-actions reveal" }, /* @__PURE__ */ React.createElement("button", { className: "btn btn-primary", onClick: () => onNav("player") }, /* @__PURE__ */ React.createElement(Icon, { name: "play", s: 17 }), " ", t("hero.ctaSessions", "Watch real sessions")), /* @__PURE__ */ React.createElement("button", { className: "btn btn-ghost", onClick: () => onNav("fusion") }, t("hero.ctaFusion", "Does the heartbeat help?"), " ", /* @__PURE__ */ React.createElement(Icon, { name: "arrow", s: 16 }))), /* @__PURE__ */ React.createElement("div", { className: "result-strip reveal" }, R.map((c) => /* @__PURE__ */ React.createElement("div", { key: c.k, className: `result-cell ${c.tone}` }, /* @__PURE__ */ React.createElement("div", { className: "rc-k" }, c.k), /* @__PURE__ */ React.createElement("div", { className: "rc-v" }, c.v, c.unit ? /* @__PURE__ */ React.createElement("small", null, c.unit) : null), /* @__PURE__ */ React.createElement("div", { className: "rc-s" }, c.s))))));
}
function PlayerVideo({ f, session, videoRef, sessionReady, onToggle }) {
  const vw = session.videoWidth || window.HUD.videoWidth || 290;
  const vh = session.videoHeight || window.HUD.videoHeight || 240;
  const bboxStyle = (box) => box ? {
    left: `${box[0] / vw * 100}%`,
    top: `${box[1] / vh * 100}%`,
    width: `${(box[2] - box[0]) / vw * 100}%`,
    height: `${(box[3] - box[1]) / vh * 100}%`
  } : null;
  const faceStyle = bboxStyle(f.roi);
  const foreheadStyle = bboxStyle(f.foreheadRoi);
  return /* @__PURE__ */ React.createElement("div", { className: "video-wrap" }, /* @__PURE__ */ React.createElement(
    "video",
    {
      ref: videoRef,
      className: "session-video",
      muted: true,
      playsInline: true,
      preload: "auto",
      poster: `media/sessions/${session.id}/thumb.jpg`,
      src: sessionReady ? window.HUD.getVideoUrl(session.id) : void 0,
      onClick: onToggle,
      style: { cursor: "pointer" }
    }
  ), /* @__PURE__ */ React.createElement("div", { className: "video-overlay" }, faceStyle && /* @__PURE__ */ React.createElement("div", { className: "roi roi-face", style: faceStyle }, /* @__PURE__ */ React.createElement("span", { className: "roi-label" }, t("player.faceRoi", "FER \xB7 face ROI"))), foreheadStyle && /* @__PURE__ */ React.createElement("div", { className: "roi roi-forehead", style: foreheadStyle }, /* @__PURE__ */ React.createElement("span", { className: "roi-label" }, t("player.foreheadRoi", "rPPG \xB7 forehead"))), /* @__PURE__ */ React.createElement("div", { className: "vid-badge" }, session.id, " \xB7 ", t("player.replay", "analysis replay")), /* @__PURE__ */ React.createElement("div", { className: "vid-caption" }, f.roi ? /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("span", { className: "dot", style: { background: f.isFear ? "var(--danger)" : "var(--clear)" } }), f.isFear ? t("player.capFear", "Fear moment \u2014 score over threshold") : t("player.capMonitor", "Monitoring \u2014 no fear detected")) : /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("span", { className: "dot", style: { background: "var(--arousal)" } }), t("player.capFaceLost", "Face lost \u2014 subject out of frame")))));
}
function PlayerTrace({ f, traces, D, playing, onSeek, onToggle, speed, setSpeed }) {
  const W = 1e3, H = 132, padT = 12, padB = 10, usable = H - padT - padB;
  const THR = window.HUD.THRESH;
  const x = (t2) => t2 / D * W;
  const y = (v) => padT + (1 - Math.max(0, Math.min(1, v))) * usable;
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
  const area = (pts) => `M${x(pts[0].t)},${y(0)} ` + pts.map((p) => `L${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ") + ` L${x(pts[pts.length - 1].t)},${y(0)} Z`;
  const windows = useMemoC(() => {
    const tr = traces.F15, out = [];
    let start = null;
    tr.forEach((p) => {
      if (p.v >= THR.F15 && start === null) start = p.t;
      else if (p.v < THR.F15 && start !== null) {
        out.push([start, p.t]);
        start = null;
      }
    });
    if (start !== null) out.push([start, D]);
    return out;
  }, [traces, D]);
  const plotRef = useRefC(null);
  const drag = useRefC(false);
  const seekAt = useCbC((clientX) => {
    const el = plotRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    onSeek(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * D);
  }, [onSeek, D]);
  const frac = f.t / D;
  return /* @__PURE__ */ React.createElement("div", { className: "ptrace" }, /* @__PURE__ */ React.createElement("div", { className: "ptrace-head" }, /* @__PURE__ */ React.createElement("span", { className: "ttl" }, t("player.traceTitle", "Fear score \xB7 drag anywhere to scrub")), /* @__PURE__ */ React.createElement("div", { className: "legend" }, /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("i", { style: { background: "var(--ink-2)" } }), t("player.legF15", "F15 +heart")), /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("i", { style: { background: "var(--ink-4)" } }), t("player.legF12", "F12 face")), /* @__PURE__ */ React.createElement("span", null, /* @__PURE__ */ React.createElement("i", { style: { background: "var(--danger)", opacity: 0.5 } }), t("player.legWindow", "fear window")))), /* @__PURE__ */ React.createElement(
    "div",
    {
      className: "ptrace-plot",
      ref: plotRef,
      onPointerDown: (e) => {
        drag.current = true;
        e.currentTarget.setPointerCapture(e.pointerId);
        seekAt(e.clientX);
      },
      onPointerMove: (e) => {
        if (drag.current) seekAt(e.clientX);
      },
      onPointerUp: () => {
        drag.current = false;
      }
    },
    /* @__PURE__ */ React.createElement("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" }, windows.map((w, i) => /* @__PURE__ */ React.createElement("rect", { key: i, x: x(w[0]), y: "0", width: x(w[1]) - x(w[0]), height: H, fill: "var(--danger)", opacity: "0.12" })), /* @__PURE__ */ React.createElement("line", { x1: "0", y1: y(THR.F12), x2: W, y2: y(THR.F12), stroke: "var(--ink-4)", strokeWidth: "1", strokeDasharray: "4 6", opacity: "0.7" }), /* @__PURE__ */ React.createElement("line", { x1: "0", y1: y(THR.F15), x2: W, y2: y(THR.F15), stroke: "var(--ink-2)", strokeWidth: "1", strokeDasharray: "5 5", opacity: "0.8" }), /* @__PURE__ */ React.createElement("path", { d: path(traces.F12), fill: "none", stroke: "var(--ink-4)", strokeWidth: "1.6", opacity: "0.9" }), /* @__PURE__ */ React.createElement("path", { d: area(traces.F15), fill: "var(--ink)", opacity: "0.05" }), /* @__PURE__ */ React.createElement("path", { d: path(traces.F15), fill: "none", stroke: "var(--ink-2)", strokeWidth: "2.4" }), /* @__PURE__ */ React.createElement("line", { x1: x(f.t), y1: "0", x2: x(f.t), y2: H, stroke: f.isFear ? "var(--danger)" : "var(--ink)", strokeWidth: "1.5", opacity: "0.55" }), /* @__PURE__ */ React.createElement("circle", { cx: x(f.t), cy: y(f.F15), r: "5", fill: f.isFear ? "var(--danger)" : "var(--ink)", stroke: "var(--bg)", strokeWidth: "2" })),
    /* @__PURE__ */ React.createElement("div", { className: "pt-time", style: { left: pctC(Math.max(0.05, Math.min(0.95, frac))) } }, f.F15.toFixed(2))
  ), /* @__PURE__ */ React.createElement("div", { className: "ptrace-ctl" }, /* @__PURE__ */ React.createElement("button", { className: "play-btn", onClick: onToggle, "aria-label": playing ? "Pause" : "Play" }, /* @__PURE__ */ React.createElement(Icon, { name: playing ? "pause" : "play", s: 15 })), /* @__PURE__ */ React.createElement("div", { className: "seg seg-speed" }, ["0.5\xD7", "1\xD7", "2\xD7"].map((o) => /* @__PURE__ */ React.createElement("button", { key: o, className: speed === o ? "on" : "", onClick: () => setSpeed(o) }, o))), /* @__PURE__ */ React.createElement("span", { className: "scrub-time" }, mmss(f.t), " / ", mmss(D))));
}
function Seg({ label, value, options, onChange }) {
  return /* @__PURE__ */ React.createElement("div", { className: "seg-field" }, /* @__PURE__ */ React.createElement("span", { className: "seg-lab" }, label), /* @__PURE__ */ React.createElement("div", { className: "seg" }, options.map((o) => /* @__PURE__ */ React.createElement("button", { key: o, className: value === o ? "on" : "", onClick: () => onChange(o) }, o))));
}
function PlayerControls({ labels, setLabels, showAlgos, setShowAlgos, speed, setSpeed }) {
  return /* @__PURE__ */ React.createElement("div", { className: "player-controls" }, /* @__PURE__ */ React.createElement(Seg, { label: "Labels", value: labels, options: ["Plain", "Plain + tech", "Technical"], onChange: setLabels }), /* @__PURE__ */ React.createElement(Seg, { label: "Speed", value: speed, options: ["0.5\xD7", "1\xD7", "2\xD7"], onChange: setSpeed }), /* @__PURE__ */ React.createElement("label", { className: "ctl-toggle" }, /* @__PURE__ */ React.createElement("input", { type: "checkbox", checked: showAlgos, onChange: (e) => setShowAlgos(e.target.checked) }), /* @__PURE__ */ React.createElement("span", { className: "track" }, /* @__PURE__ */ React.createElement("span", { className: "knob" })), "Show other rPPG algorithms"));
}
function SessionRail({ sessions, sel, onSelect }) {
  return /* @__PURE__ */ React.createElement("div", { className: "session-rail" }, /* @__PURE__ */ React.createElement("div", { className: "rail-head" }, /* @__PURE__ */ React.createElement("span", { className: "rh-t" }, t("rail.title", "Recorded sessions")), /* @__PURE__ */ React.createElement("span", { className: "rh-n" }, sessions.length, " ", t("rail.meta", "clips \xB7 6 subjects \xB7 3 lighting conditions"))), /* @__PURE__ */ React.createElement("div", { className: "session-list" }, sessions.map((s) => /* @__PURE__ */ React.createElement("button", { key: s.id, className: `s-card ${s.id === sel ? "sel" : ""}`, onClick: () => onSelect(s.id) }, /* @__PURE__ */ React.createElement("span", { className: "s-thumb" }, /* @__PURE__ */ React.createElement("img", { className: "s-thumb-img", src: `media/sessions/${s.id}/thumb.jpg`, alt: "", loading: "lazy" }), /* @__PURE__ */ React.createElement("span", { className: "play-ic" }, /* @__PURE__ */ React.createElement(Icon, { name: s.id === sel ? "pause" : "play", s: 15 })), /* @__PURE__ */ React.createElement("span", { className: "s-dur" }, mmss(s.dur))), /* @__PURE__ */ React.createElement("span", { className: "s-meta" }, /* @__PURE__ */ React.createElement("span", { className: "sm-top" }, /* @__PURE__ */ React.createElement("span", { className: "sm-id" }, s.subject), /* @__PURE__ */ React.createElement("span", { className: `lightchip ${s.lighting}` }, t("light." + s.lighting, s.lighting)), /* @__PURE__ */ React.createElement("span", { className: "sm-vid" }, s.vid)), /* @__PURE__ */ React.createElement("span", { className: "sm-sub" }, s.note))))), /* @__PURE__ */ React.createElement("div", { className: "note-strip" }, /* @__PURE__ */ React.createElement(Icon, { name: "info", s: 17 }), /* @__PURE__ */ React.createElement("span", null, (() => {
    const p = tSlot(t("rail.note", "Each card streams its own real per-frame data + {file}. The numbers match the offline HUD renderer exactly."), "file", /* @__PURE__ */ React.createElement("span", { className: "mono" }, "raw_video.mp4"));
    return /* @__PURE__ */ React.createElement(React.Fragment, null, p[0], p[1], p[2]);
  })())));
}
function HudFit({ children }) {
  const wrapRef = useRefC(null);
  const [box, setBox] = useStateC({ s: 1, h: 900 });
  React.useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
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
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);
  return /* @__PURE__ */ React.createElement("div", { className: "hud-fit", ref: wrapRef, style: { height: box.h } }, /* @__PURE__ */ React.createElement("div", { className: "hud-canvas", style: { transform: `scale(${box.s})` } }, children));
}
function PlayerSection({ f, traces, D, playing, onSeek, onToggle, sessions, sel, onSelect, mode, showAlgos, setShowAlgos, speed, setSpeed, compact, videoRef, sessionReady }) {
  const session = sessions.find((s) => s.id === sel) || sessions[0];
  const Body = /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement(TelemetryBar, { f }), /* @__PURE__ */ React.createElement("div", { className: "hud-body" }, /* @__PURE__ */ React.createElement("div", { className: "hud-left" }, /* @__PURE__ */ React.createElement(PlayerVideo, { f, session, videoRef, sessionReady, onToggle }), /* @__PURE__ */ React.createElement(PlayerTrace, { f, traces, D, playing, onSeek, onToggle, speed, setSpeed })), /* @__PURE__ */ React.createElement("div", { className: "hud-right" }, /* @__PURE__ */ React.createElement(VerdictCard, { f, headline: "F15" }), /* @__PURE__ */ React.createElement(PrimarySignals, { f, mode }), /* @__PURE__ */ React.createElement(Amplifiers, { f, mode, showAlgos, setShowAlgos }))));
  return /* @__PURE__ */ React.createElement("section", { className: "section-block tight", id: "player" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, t("player.kicker", "The centerpiece")), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, t("player.title", "The fear-analysis HUD, on real sessions")), /* @__PURE__ */ React.createElement("p", { className: "sec-lead reveal" }, t("player.lead", "The same instrument the pipeline renders to video \u2014 now live. Pick a recorded session below; the fear-score trace doubles as the scrubber, and every readout updates from the current frame."))), /* @__PURE__ */ React.createElement("div", { className: "player-stage player-wide reveal" }, compact ? /* @__PURE__ */ React.createElement("div", { className: "hud-shell" }, Body) : /* @__PURE__ */ React.createElement(HudFit, null, Body), /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement(SessionRail, { sessions, sel, onSelect }))));
}
function teachCompute(c) {
  const base = 0.7 * c.fear + 0.3 * c.arousal;
  const F12 = Math.min(1, base * (1 + c.tension));
  const F15 = Math.min(1, F12 * (1 + 0.5 * c.bpmRise));
  return { base, F12, F15, fear12: F12 >= 0.7, fear15: F15 >= 0.8 };
}
function TeachReadout({ c }) {
  const r = teachCompute(c);
  const Row = ({ nm, val, color, thr }) => /* @__PURE__ */ React.createElement("div", { className: "tro" }, /* @__PURE__ */ React.createElement("div", { className: "tro-top" }, /* @__PURE__ */ React.createElement("span", { className: "nm" }, nm), /* @__PURE__ */ React.createElement("span", { className: "vl", style: { color } }, f2(val))), /* @__PURE__ */ React.createElement("div", { className: "tbar" }, /* @__PURE__ */ React.createElement("div", { className: "tf", style: { width: pctC(val), background: color } }), thr != null ? /* @__PURE__ */ React.createElement("div", { className: "thr", style: { left: pctC(thr) } }) : null));
  return /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("div", { className: "teach-side" }, /* @__PURE__ */ React.createElement("div", { className: "ts-cap" }, t("teach.signalsCap", "Signals this moment")), /* @__PURE__ */ React.createElement("div", { className: "teach-readout" }, /* @__PURE__ */ React.createElement(Row, { nm: t("teach.rowFear", "Fear (face)"), val: c.fear, color: "var(--danger)" }), /* @__PURE__ */ React.createElement(Row, { nm: t("teach.rowArousal", "Arousal"), val: c.arousal, color: "var(--arousal)" }), /* @__PURE__ */ React.createElement(Row, { nm: t("teach.rowTension", "Facial tension"), val: c.tension, color: "var(--tension)" }), /* @__PURE__ */ React.createElement(Row, { nm: t("teach.rowBpm", "Heart-rate rise"), val: c.bpmRise, color: "var(--heart)" }))), /* @__PURE__ */ React.createElement("div", { className: "teach-side" }, /* @__PURE__ */ React.createElement("div", { className: "ts-cap" }, t("teach.verdictsCap", "Two verdicts, two thresholds")), /* @__PURE__ */ React.createElement("div", { className: "teach-readout" }, /* @__PURE__ */ React.createElement(Row, { nm: t("teach.rowF12", "F12 \xB7 face only"), val: r.F12, color: r.fear12 ? "var(--ink)" : "var(--ink-2)", thr: 0.7 }), /* @__PURE__ */ React.createElement(Row, { nm: t("teach.rowF15", "F15 \xB7 + heart rate"), val: r.F15, color: r.fear15 ? "var(--danger)" : "var(--ink-2)", thr: 0.8 })), /* @__PURE__ */ React.createElement("div", { className: `teach-verdict ${r.fear15 ? "fear" : "clear"}` }, /* @__PURE__ */ React.createElement("div", { className: "tv-state" }, r.fear15 ? t("hud.verdict.fear", "FEAR DETECTED") : t("hud.verdict.noFear", "NO FEAR")), /* @__PURE__ */ React.createElement("div", { className: "tv-why" }, c.f15Why))));
}
function FusionSection() {
  const { FUSE, TEACH } = window.SITE;
  const [ci, setCi] = useStateC(0);
  const Eq = ({ parts, hr }) => /* @__PURE__ */ React.createElement("span", null, parts[0], /* @__PURE__ */ React.createElement("span", { className: hr ? "hr" : "op" }, parts[1]));
  return /* @__PURE__ */ React.createElement("section", { className: "section-block", id: "fusion" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("div", { className: "fuse-head-grid" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, t("fusion.kicker", "Fusion")), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, t("fusion.title", "Does the heartbeat help?")), /* @__PURE__ */ React.createElement("p", { className: "sec-lead reveal" }, t("fusion.lead", "A fearful face can lie. Adding a contactless heart-rate read turns a face-only guess into a body-confirmed decision \u2014 a calm pulse holds back the {f12} face score, while a genuine cardiac response pushes the fused {f15}.").split(/(\{f12\}|\{f15\})/).map((p, i) => p === "{f12}" ? /* @__PURE__ */ React.createElement("b", { key: i, className: "mono", style: { color: "var(--ink)" } }, t("fusion.leadF12", "F12 \u2265 0.70")) : p === "{f15}" ? /* @__PURE__ */ React.createElement("b", { key: i, className: "mono", style: { color: "var(--accent)" } }, t("fusion.leadF15", "F15 over 0.80")) : p))), /* @__PURE__ */ React.createElement("div", { className: "fuse-formulas reveal" }, /* @__PURE__ */ React.createElement("div", { className: "fcard f12" }, /* @__PURE__ */ React.createElement("div", { className: "fc-top" }, /* @__PURE__ */ React.createElement("span", { className: "fc-tag" }, "F12"), /* @__PURE__ */ React.createElement("span", { className: "fc-name" }, t("hud.verdict.f12Sub", "face only")), /* @__PURE__ */ React.createElement("span", { className: "fc-thr" }, "\u2265 0.70")), /* @__PURE__ */ React.createElement("div", { className: "fc-eq mono" }, /* @__PURE__ */ React.createElement(Eq, { parts: FUSE.f12.eq }))), /* @__PURE__ */ React.createElement("div", { className: "fcard f15" }, /* @__PURE__ */ React.createElement("div", { className: "fc-top" }, /* @__PURE__ */ React.createElement("span", { className: "fc-tag" }, "F15"), /* @__PURE__ */ React.createElement("span", { className: "fc-name" }, t("hud.verdict.f15Sub", "+ heart rate")), /* @__PURE__ */ React.createElement("span", { className: "fc-thr" }, "\u2265 0.80")), /* @__PURE__ */ React.createElement("div", { className: "fc-eq mono" }, /* @__PURE__ */ React.createElement(Eq, { parts: FUSE.f15.eq, hr: true }))))), /* @__PURE__ */ React.createElement("div", { className: "teach reveal" }, /* @__PURE__ */ React.createElement("div", { className: "teach-tabs" }, TEACH.map((c, i) => /* @__PURE__ */ React.createElement("button", { key: c.key, className: `teach-tab ${i === ci ? "on" : ""}`, onClick: () => setCi(i) }, /* @__PURE__ */ React.createElement("div", { className: "tt-k" }, t("fusion.case", "Case"), " ", i + 1), /* @__PURE__ */ React.createElement("div", { className: "tt-t" }, c.tab)))), /* @__PURE__ */ React.createElement("div", { className: "teach-body" }, /* @__PURE__ */ React.createElement(TeachReadout, { c: TEACH[ci] })))));
}
function MethodsSection({ onNav }) {
  const M = window.SITE.METHODS;
  return /* @__PURE__ */ React.createElement("section", { className: "section-block divline", id: "methods" }, /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("span", { className: "kicker reveal" }, "Methods"), /* @__PURE__ */ React.createElement("h2", { className: "sec-title reveal" }, "What made it work in the dark"), /* @__PURE__ */ React.createElement("p", { className: "sec-lead reveal" }, "Four engineering choices carry the pipeline through horror-game lighting. Each gets a dedicated interactive breakdown \u2014 detection fallback, contrast recovery, multiplicative fusion, and the rPPG sweep."), /* @__PURE__ */ React.createElement("div", { className: "feature-grid" }, M.map((m) => /* @__PURE__ */ React.createElement("div", { key: m.t, className: "card reveal" }, /* @__PURE__ */ React.createElement("span", { className: "c-ic" }, /* @__PURE__ */ React.createElement(Icon, { name: m.ic, s: 20 })), /* @__PURE__ */ React.createElement("h3", null, m.t), /* @__PURE__ */ React.createElement("p", null, m.p))), /* @__PURE__ */ React.createElement("div", { className: "card reveal", style: { justifyContent: "center", alignItems: "flex-start", background: "var(--surface-2)" } }, /* @__PURE__ */ React.createElement("span", { className: "kicker k-plain", style: { color: "var(--ink-4)" } }, "Coming next"), /* @__PURE__ */ React.createElement("h3", { style: { marginTop: 6 } }, "Interactive method animations"), /* @__PURE__ */ React.createElement("p", null, "Before/after CLAHE, the lighting-by-fallback bar chart, and the 48-config sweep \u2014 each as a live, scrubbable visual."), /* @__PURE__ */ React.createElement("button", { className: "btn btn-ghost", style: { marginTop: 8 }, onClick: () => onNav("player") }, "Back to the HUD ", /* @__PURE__ */ React.createElement(Icon, { name: "arrow", s: 15 }))))));
}
function Footer() {
  return /* @__PURE__ */ React.createElement("footer", { className: "footer", id: "about" }, /* @__PURE__ */ React.createElement("div", { className: "wrap footer-grid" }, /* @__PURE__ */ React.createElement("div", { className: "f-brand" }, /* @__PURE__ */ React.createElement("div", { className: "bt" }, "La Fa\xE7ade ", /* @__PURE__ */ React.createElement("em", null, "Fissur\xE9e")), /* @__PURE__ */ React.createElement("p", null, t("footer.blurb", "Real-time emotion analysis for adaptive enemy AI in affective game design. A webcam-only, privacy-preserving approach to sensing fear \u2014 fusing facial emotion recognition with contactless heart rate."))), /* @__PURE__ */ React.createElement("div", { className: "f-col" }, /* @__PURE__ */ React.createElement("h4", null, t("footer.project", "Project")), /* @__PURE__ */ React.createElement("ul", null, /* @__PURE__ */ React.createElement("li", null, /* @__PURE__ */ React.createElement("span", { className: "lab" }, t("footer.author", "Author")), "Ali Burak Sara\xE7"), /* @__PURE__ */ React.createElement("li", null, /* @__PURE__ */ React.createElement("span", { className: "lab" }, t("footer.advisor", "Advisor")), "Asst. Prof. Reis Burak Arslan"), /* @__PURE__ */ React.createElement("li", null, /* @__PURE__ */ React.createElement("span", { className: "lab" }, t("footer.date", "Date")), t("footer.dateVal", "May 2026")))), /* @__PURE__ */ React.createElement("div", { className: "f-col" }, /* @__PURE__ */ React.createElement("h4", null, t("footer.institution", "Institution")), /* @__PURE__ */ React.createElement("ul", null, /* @__PURE__ */ React.createElement("li", null, /* @__PURE__ */ React.createElement("span", { className: "lab" }, t("footer.university", "University")), t("footer.uniVal", "Galatasaray University")), /* @__PURE__ */ React.createElement("li", null, /* @__PURE__ */ React.createElement("span", { className: "lab" }, t("footer.faculty", "Faculty")), t("footer.facultyVal", "Engineering & Technology")), /* @__PURE__ */ React.createElement("li", null, /* @__PURE__ */ React.createElement("span", { className: "lab" }, t("footer.department", "Department")), t("footer.deptVal", "Computer Engineering"))))), /* @__PURE__ */ React.createElement("div", { className: "wrap" }, /* @__PURE__ */ React.createElement("div", { className: "footer-base" }, /* @__PURE__ */ React.createElement("span", null, "HSEmotion \xB7 EfficientNet-B0"), /* @__PURE__ */ React.createElement("span", { className: "sep" }), /* @__PURE__ */ React.createElement("span", null, "MediaPipe FaceLandmarker"), /* @__PURE__ */ React.createElement("span", { className: "sep" }), /* @__PURE__ */ React.createElement("span", null, "Haar-first detection \xB7 CLAHE"), /* @__PURE__ */ React.createElement("span", { className: "sep" }), /* @__PURE__ */ React.createElement("span", null, "rPPG \xB7 POS @ 30 s"))));
}
function SectionCustomizer({ open, onClose, sections, order, hidden, move, toggle, reset, onJump }) {
  if (!open) return null;
  const ordered = order.map((id) => sections.find((s) => s.id === id)).filter(Boolean);
  const visibleCount = ordered.filter((s) => !hidden[s.id]).length;
  return /* @__PURE__ */ React.createElement(React.Fragment, null, /* @__PURE__ */ React.createElement("div", { className: "cust-scrim", onClick: onClose }), /* @__PURE__ */ React.createElement("aside", { className: "cust", role: "dialog", "aria-label": t("cust.aria", "Customize sections") }, /* @__PURE__ */ React.createElement("div", { className: "cust-head" }, /* @__PURE__ */ React.createElement("div", null, /* @__PURE__ */ React.createElement("div", { className: "cust-t" }, t("cust.title", "Customize this page")), /* @__PURE__ */ React.createElement("div", { className: "cust-s" }, t("cust.subtitle", "{shown} of {total} sections shown \xB7 saved on this device").replace("{shown}", visibleCount).replace("{total}", sections.length))), /* @__PURE__ */ React.createElement("button", { className: "cust-x", onClick: onClose, "aria-label": t("cust.close", "Close") }, /* @__PURE__ */ React.createElement(Icon, { name: "close", s: 16 }))), /* @__PURE__ */ React.createElement("div", { className: "cust-list" }, ordered.map((s, i) => {
    const off = !!hidden[s.id];
    const locked = s.locked;
    return /* @__PURE__ */ React.createElement("div", { className: `cust-row ${off ? "off" : ""}`, key: s.id }, /* @__PURE__ */ React.createElement("span", { className: "cust-grip" }, /* @__PURE__ */ React.createElement(Icon, { name: "dot6", s: 16 })), /* @__PURE__ */ React.createElement("button", { className: "cust-name", onClick: () => !off && onJump(s.id), disabled: off, title: off ? t("cust.hidden", "Hidden") : t("cust.jump", "Jump to section") }, /* @__PURE__ */ React.createElement("span", { className: "cn-i" }, String(i + 1).padStart(2, "0")), t("section." + s.id, s.label), locked ? /* @__PURE__ */ React.createElement("span", { className: "cn-lock" }, t("cust.alwaysOn", "always on")) : null), /* @__PURE__ */ React.createElement("span", { className: "cust-acts" }, /* @__PURE__ */ React.createElement("button", { onClick: () => move(s.id, -1), disabled: i === 0, "aria-label": t("cust.moveUp", "Move up") }, /* @__PURE__ */ React.createElement(Icon, { name: "up", s: 14 })), /* @__PURE__ */ React.createElement("button", { onClick: () => move(s.id, 1), disabled: i === ordered.length - 1, "aria-label": t("cust.moveDown", "Move down") }, /* @__PURE__ */ React.createElement(Icon, { name: "down", s: 14 })), /* @__PURE__ */ React.createElement("label", { className: `cust-eye ${locked ? "locked" : ""}`, title: locked ? t("cust.cantHide", "This section can't be hidden") : off ? t("cust.show", "Show") : t("cust.hide", "Hide") }, /* @__PURE__ */ React.createElement("input", { type: "checkbox", checked: !off, disabled: locked, onChange: () => toggle(s.id) }), /* @__PURE__ */ React.createElement("span", { className: "track" }, /* @__PURE__ */ React.createElement("span", { className: "knob" })))));
  })), /* @__PURE__ */ React.createElement("div", { className: "cust-foot" }, /* @__PURE__ */ React.createElement("button", { className: "cust-reset", onClick: reset }, /* @__PURE__ */ React.createElement(Icon, { name: "reset", s: 14 }), " ", t("cust.reset", "Reset to default")), /* @__PURE__ */ React.createElement("span", { className: "cust-hint" }, t("cust.hint", "Reorder with \u2191\u2193 \xB7 toggle to hide")))));
}
Object.assign(window, {
  SiteIcon: Icon,
  Nav,
  SideNav,
  LangSwitcher,
  Hero,
  PlayerSection,
  FusionSection,
  MethodsSection,
  Footer,
  SectionCustomizer
});
