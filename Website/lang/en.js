/* lang/en.js — ENGLISH, the canonical source of truth.
 *
 * This is the reference dictionary. fr.js and tr.js mirror these keys.
 * When you add a new user-facing string anywhere in the site, add its key
 * HERE first, then to fr.js and tr.js.
 *
 * Conventions (see TRANSLATION_GUIDE.md):
 *  - Keys are flat + dotted: "<section>.<thing>".
 *  - {slots} are placeholders filled with styled JSX by the component.
 *    Keep the {slot} token verbatim; you MAY move it within the sentence to
 *    fit the target grammar, but never translate or delete the token.
 *  - Some values carry inline HTML (<b>, <em>, <i>) — keep the tags verbatim,
 *    translate only the words between them.
 *  - Proper nouns / technical tokens stay untranslated: "La Façade Fissurée",
 *    FER, rPPG, HSEmotion, MediaPipe, POS, CLAHE, HUD, F12, F15, BPM, ROI.
 *  - ⚠ comments mark SPLIT sentences (one thought spread across several keys
 *    for styling) and ASSEMBLED strings (a key joined to a runtime number/
 *    value). The comment shows the full assembled sentence — translate the
 *    whole group coherently and mind word order around the inserted value.
 */
window.I18N._register("en", {
  // ── top nav + side nav ──
  "nav.brandTag": "FER + rPPG",
  "nav.openHud": "Open the HUD ▸",
  "nav.customize": "Reorder & hide sections",
  "nav.menu": "Menu",
  "nav.language": "Language",
  "nav.overview": "Overview",
  "nav.background": "Background",
  "nav.pipeline": "Pipeline",
  "nav.player": "Player",
  "nav.fusion": "Fusion",
  "nav.methods": "Methods",
  "nav.results": "Results",
  "nav.performance": "Perf",
  "nav.game": "Game",
  "nav.path": "Path",
  "nav.outlook": "Outlook",
  "nav.glossary": "Glossary",

  // ── hero / overview ──
  "hero.badge": "Final project",
  "hero.affil": "Galatasaray University · Computer Engineering",
  // ⚠ SPLIT SENTENCE — the big italic title. hero.fearWord is the emphasized
  //   word that renders inside the {fear} slot. Full sentence reads:
  //       "Can a plain webcam feel a player's fear?"
  //   Translate hero.title and hero.fearWord TOGETHER as one thought. You may
  //   move {fear} anywhere in the sentence to fit grammar, but keep the token.
  "hero.title": "Can a plain webcam feel a player's {fear}?",
  "hero.fearWord": "fear",
  "hero.sub": "A real-time, privacy-preserving pipeline that fuses facial emotion recognition with a contactless heart rate read from skin colour — scoring fear under the dim, flickering light of horror games. Everything runs locally; no video ever leaves the device.",
  "hero.ctaSessions": "Watch real sessions",
  "hero.ctaFusion": "Does the heartbeat help?",

  // ── data: session notes (site-data.js) ──
  "data.sess.S06_Vid16.note": "Well-lit baseline — clean FER, strong rPPG signal.",
  "data.sess.S02_Vid04.note": "Dim room — MediaPipe fallback carries detection.",
  "data.sess.S08_Vid18.note": "Flickering screen-lit scene — hardest case.",
  "data.sess.S04_Vid09.note": "Steady lighting, several genuine startles.",
  "data.sess.S05_Vid10.note": "High arousal range — fusion separates true fear.",
  "data.sess.S10_Vid13.note": "Expressive subject — fear-face false alarms tested.",

  // ── data: headline results strip ──
  "data.results.0.k": "F15 event F1",
  "data.results.0.s": "rPPG-augmented · 90% precision",
  "data.results.1.k": "End-to-end latency",
  "data.results.1.s": "fully on-device · <100 ms budget",
  "data.results.2.k": "Face detection",
  "data.results.2.s": "hybrid Haar + MediaPipe, low light",
  "data.results.3.k": "Subjects · events",
  "data.results.3.s": "annotated fear events, 3 lighting sets",

  // ── data: fusion formulas ──
  // ⚠ SPLIT FORMULA — eq0 + eq1 render side by side as ONE equation (eq1 is the
  //   styled/coloured half). Assembled:  f12 = "(0.7·Fear + 0.3·Arousal) × (1 + facial tension)"
  //                                       f15 = "F12 × (1 + 0.5·heart-rate rise)"
  //   Mostly math — translate only the words ("Fear", "Arousal", "facial
  //   tension", "heart-rate rise"); keep the numbers, ·, × and the leading space.
  "data.fuse.f12.name": "face only",
  "data.fuse.f12.eq0": "(0.7·Fear + 0.3·Arousal)",
  "data.fuse.f12.eq1": " × (1 + facial tension)",
  "data.fuse.f15.name": "+ heart rate",
  "data.fuse.f15.eq0": "F12",
  "data.fuse.f15.eq1": " × (1 + 0.5·heart-rate rise)",

  // ── data: F12-vs-F15 teaching cases ──
  "data.teach.0.tab": "Fearful face, calm body",
  "data.teach.0.title": "A scare face — but the body is calm",
  "data.teach.0.f12Why": "A strong fear expression alone pushes F12 close to the line…",
  "data.teach.0.f15Why": "…but flat arousal and a resting heart rate hold the fused score below 0.80. Correctly read as not fear — the kind of false alarm a face-only model over-calls.",
  "data.teach.1.tab": "Genuine fear",
  "data.teach.1.title": "Fearful face, high arousal, elevated pulse",
  "data.teach.1.f12Why": "Fear and arousal together already lift F12 over its 0.70 line.",
  "data.teach.1.f15Why": "An elevated heart rate amplifies it further — F15 crosses 0.80 with high confidence. A genuine fear event the body confirms.",

  // ── data: methods cards ──
  "data.methods.0.t": "Haar-first, MediaPipe fallback",
  "data.methods.0.p": "A fast Haar cascade crops the face every frame; when it loses the face in low light, MediaPipe FaceLandmarker steps in. The hybrid keeps a face locked on 99.9% of analyzed frames.",
  "data.methods.1.t": "CLAHE contrast enhancement",
  "data.methods.1.p": "Contrast-Limited Adaptive Histogram Equalization (clipLimit 2.0, 4×4 tiles, on the L channel) restores detail in the face ROI before emotion is read.",
  "data.methods.2.t": "Multiplicative fusion",
  "data.methods.2.p": "HSEmotion detects; MediaPipe tension and heart-rate amplify. They multiply rather than average, so a calm body can't trigger fear but a confirming one compounds it.",
  "data.methods.3.t": "rPPG estimator sweep",
  "data.methods.3.p": "A 48-configuration sweep over estimators, windows and boost coefficients settled on POS at a 30 s window / 3 s step — the only choice that preserved 90% precision.",

  // ── data: pipeline stages (s of stages 2/4/6 are technical labels) ──
  "data.pipeline.0.t": "Capture",
  "data.pipeline.0.s": "Consumer webcam",
  "data.pipeline.0.d": "A single ordinary webcam — no dedicated biometric hardware. Everything downstream runs on the player's own machine.",
  "data.pipeline.1.t": "CLAHE",
  "data.pipeline.1.s": "Contrast recovery",
  "data.pipeline.1.d": "Adaptive histogram equalization lifts the dark, low-contrast face ROI typical of horror lighting.",
  "data.pipeline.2.t": "Hybrid detect",
  "data.pipeline.2.s": "Haar → MediaPipe",
  "data.pipeline.2.d": "Fast Haar cascade first; MediaPipe FaceLandmarker rescues frames it loses. 99.9% coverage.",
  "data.pipeline.3.t": "FER",
  "data.pipeline.3.s": "HSEmotion + MediaPipe",
  "data.pipeline.3.d": "HSEmotion (EfficientNet-B0) reads fear & arousal — the detector. MediaPipe's 52 blendshapes give facial tension — the amplifier.",
  "data.pipeline.4.t": "rPPG",
  "data.pipeline.4.s": "POS @ 30 s / 3 s",
  "data.pipeline.4.d": "Skin-colour pulse from the forehead ROI. POS estimator over a 30-second window, stepped every 3 s.",
  "data.pipeline.5.t": "Fusion",
  "data.pipeline.5.s": "F12 → F15",
  "data.pipeline.5.d": "Multiplicative composite: face score, amplified by tension, then by heart-rate rise. Thresholds 0.70 / 0.80.",
  "data.pipeline.6.t": "Dual-gate event",
  "data.pipeline.6.s": "IDLE · ONSET · CONFIRMED",
  "data.pipeline.6.d": "Two gates reject slow drifts and brief spikes, so only sustained fear becomes a discrete event.",
  "data.pipeline.7.t": "Game over TCP",
  "data.pipeline.7.s": "Enemy AI",
  "data.pipeline.7.d": "Discrete event state + smoothed score stream to the Unity game — driving enemy escalation and adaptive behaviour.",

  // ── data: research questions ──
  "data.rqs.0.q": "Can facial emotion recognition stay accurate in horror-game low light?",
  "data.rqs.0.a": "Yes — F1 = 0.77 under monitor-only lighting, with 99.9% face coverage from the hybrid detector.",
  "data.rqs.1.q": "Does decision-level FER + rPPG fusion improve detection reliability?",
  "data.rqs.1.a": "Promising — rPPG augmentation lifts F1 from 0.70 to 0.77 at 90% precision on the four sessions with usable pulse. Preliminary; needs more participants.",
  "data.rqs.2.q": "Does a “Relax-to-Win” adaptation improve the player experience?",
  "data.rqs.2.a": "Formally unanswered. The hardware failure forced a video-based evaluation on passive viewers, which omits the closed-loop biofeedback. This is the primary future-work target.",

  // ── data: key findings ──
  "data.findings.0.t": "Multiply, don't average",
  "data.findings.0.p": "Equal-weight fusion of HSEmotion and MediaPipe collapses across subjects — MediaPipe's tension activations are structurally too sparse. A multiplicative modifier (tension amplifies but can't trigger fear alone) fixes it.",
  "data.findings.1.t": "The arousal channel",
  "data.findings.1.p": "Adding 30% hs_arousal to the fear signal recovers events lost to fear–surprise confusion, producing F12 at F1 = 0.70.",
  "data.findings.2.t": "The rPPG null was an artefact",
  "data.findings.2.p": "The CONSENSUS estimator diluted POS's precise signal with weaker algorithms (d = 0.042). Isolating POS and matching the window to event duration revealed a medium-large cardiac fear response (d = 0.696, FWER p = 0.029).",
  "data.findings.3.t": "rPPG augmentation is promising",
  "data.findings.3.p": "F15 lifts F1 to 0.77 while preserving 90% precision on four sessions. The pulse lags FER by ~8–18 s, creating a complementary two-phase “sustained terror” dynamic.",
  "data.findings.4.t": "Low-light robustness confirmed",
  "data.findings.4.p": "F1 = 0.77 under monitor-only lighting, with 99.9% face-detection coverage from the hybrid Haar + MediaPipe approach.",

  // ── data: FER asymmetry ──
  "data.asym.detector.role": "THE DETECTOR",
  "data.asym.detector.sub": "EfficientNet-B0 · AffectNet · 8-class softmax",
  "data.asym.detector.note": "Holistic neural classification. Robust across faces with no per-subject calibration.",
  "data.asym.amplifier.role": "THE AMPLIFIER",
  "data.asym.amplifier.sub": "52 facial blendshapes · geometric tension",
  "data.asym.amplifier.note": "Sparse activations — weak alone, but confirms HSEmotion when both agree on a fear moment.",

  // ── data: rPPG sweep configs (cfg strings stay literal) ──
  "data.rppg.0.why": "Highest F1, preserves 90% precision — the production choice.",
  "data.rppg.1.why": "Comparable F1 but trades 10pp precision for recall — bad for a game.",
  "data.rppg.2.why": "Solid, but lower F1 than the 30 s window.",

  // ── data: enemies (names Le Rodeur / Le Veilleur + FSM states stay literal) ──
  "data.enemies.wanderer.aka": "The Wanderer",
  "data.enemies.wanderer.cls": "Giant 8-legged spider · hunts by fear, not by sight",
  "data.enemies.wanderer.ch0.t": "Motion cone",
  "data.enemies.wanderer.ch0.d": "90° forward, 12 m, raycast-occluded — only fires on movement. Stay still and you're invisible, even inside the cone.",
  "data.enemies.wanderer.ch1.t": "Sprint-hearing bubble",
  "data.enemies.wanderer.ch1.d": "360°, 7 m radius, wall-occluded — hears sprinting but not walking.",
  "data.enemies.wanderer.ch2.t": "Fear-ping",
  "data.enemies.wanderer.ch2.d": "Global, >15 m only — reads your smoothed composite fear and steers the enemy toward you. The direct biofeedback channel.",
  "data.enemies.watcher.aka": "The Watcher",
  "data.enemies.watcher.cls": "Directionless proximity entity · attacks only the unseen and the afraid",
  "data.enemies.watcher.m0.see": "Look at it",
  "data.enemies.watcher.m0.calm": "Freezes",
  "data.enemies.watcher.m0.afraid": "Freezes",
  "data.enemies.watcher.m1.see": "Look away",
  "data.enemies.watcher.m1.calm": "Despawns",
  "data.enemies.watcher.m1.afraid": "STRIKE → game over",
  "data.enemies.watcher.ch0.t": "Proximity spawn",
  "data.enemies.watcher.ch0.d": "Appears 5–9 m away, always just outside the camera frustum.",
  "data.enemies.watcher.ch1.t": "2×2 decision",
  "data.enemies.watcher.ch1.d": "Behaviour is set by (do you see it) × (are you afraid). Eye contact freezes it; calm makes it leave; fear while looking away is fatal.",
  "data.enemies.watcher.ch2.t": "Key escalation",
  "data.enemies.watcher.ch2.d": "Spawn interval tightens per collected key across 7 tiers — from ~17 min at 0 keys down to ~2 min at 6 keys.",

  // ── data: design path ──
  "data.path.0.t": "The pipeline",
  "data.path.0.meta": "Phases 1–2 · built & validated",
  "data.path.0.d": "A real-time, on-device affective-computing pipeline: hybrid detection, CLAHE, HSEmotion + MediaPipe FER, POS rPPG, multiplicative fusion and a dual-gate event detector. Preserved in git throughout.",
  "data.path.1.t": "The evaluation pivot",
  "data.path.1.meta": "Phase 4 · video-based study",
  "data.path.1.d": "Validated on 6 diverse subjects and 88 annotated fear events across three lighting conditions, using horror-reaction videos with independent manual annotation — a broader, more diverse corpus than the lab study originally planned.",
  "data.path.2.t": "The game integration",
  "data.path.2.meta": "Phase 3 · planned, hardware-hit",
  "data.path.2.d": "La Façade Fissurée — a first-person horror game (177 files, 20k+ lines of Unity) where enemy AI reacts to your detected fear. <b>The main development machine failed after 13 April 2026</b>; the prototype couldn't run on the backup, so the closed-loop game integration was frozen while the pipeline carried on. The Relax-to-Win player study is the primary future-work target.",

  // ── HUD: telemetry bar (ms/fps units stay literal) ──
  "hud.telemetry.title": "Fear Analysis HUD",
  "hud.telemetry.sub": "FER + rPPG · replay",
  "hud.telemetry.time": "Time",
  "hud.telemetry.frame": "Frame",
  "hud.telemetry.latency": "Latency",
  "hud.telemetry.throughput": "Throughput",

  // ── HUD: verdict card (F12/F15 tags stay literal) ──
  "hud.verdict.faceLost": "FACE LOST",
  "hud.verdict.fear": "FEAR DETECTED",
  "hud.verdict.noFear": "NO FEAR",
  "hud.verdict.bylineFaceLost": "subject out of frame · detection paused",
  "hud.verdict.byline": "final decision · F15 (production)",
  "hud.verdict.scoreCap": "F15 score",
  // ⚠ ASSEMBLED — followed by the threshold number: needs + " 0.80".
  "hud.verdict.needs": "needs",
  "hud.verdict.over": "OVER",
  "hud.verdict.under": "UNDER",
  "hud.verdict.f12Sub": "face only",
  "hud.verdict.f15Sub": "+ heart rate",

  // ── HUD: primary signals (HSEmotion, hs_fear/hs_arousal stay literal) ──
  "hud.primary.label": "Primary signal · facial emotion",
  "hud.primary.arousal": "Arousal",
  "hud.primary.baseScore": "Base score",
  "hud.primary.baseFormula": "= 0.7×Fear + 0.3×Arousal =",
  // ⚠ ASSEMBLED — followed at runtime by the dominant emotion + score:
  //   distPrefix + " Neutral (0.53)". Keep the trailing colon.
  "hud.primary.distPrefix": "Full emotion distribution · dominant:",

  // ── HUD: emotion labels (8-class AffectNet; keys double as data lookups) ──
  "emo.Anger": "Anger",
  "emo.Contempt": "Contempt",
  "emo.Disgust": "Disgust",
  "emo.Fear": "Fear",
  "emo.Happiness": "Happiness",
  "emo.Neutral": "Neutral",
  "emo.Sadness": "Sadness",
  "emo.Surprise": "Surprise",

  // ── HUD: amplifiers + formula chain (mp_tension/bpm_norm/BPM stay literal) ──
  "hud.amp.label": "Amplifiers & score build-up",
  "hud.amp.note": "F12 = base × tension · F15 = F12 × heart",
  "hud.amp.tensionTitle": "Facial tension",
  "hud.amp.tensionSub": "MediaPipe · brow / jaw / eye muscle strain",
  "hud.amp.valence": "Valence",
  "hud.amp.smile": "Smile",
  "hud.amp.startle": "Startle",
  "hud.amp.heartTitle": "Heart rate",
  "hud.amp.calibrating": "calibrating",
  "hud.amp.calSub": "rPPG needs ~30 s of skin-colour data before the first pulse estimate",
  "hud.amp.calNote": "Collecting forehead ROI frames… multiplier locked at ×1.00 until ready",
  "hud.amp.heartSub": "rPPG (POS) · pulse read from skin colour",
  // ⚠ ASSEMBLED — each precedes a number: rise + " 0.11", rest + " 72", now + " 88".
  "hud.amp.rise": "rise",
  "hud.amp.rest": "rest",
  "hud.amp.now": "now",
  "hud.amp.otherAlgos": "Other rPPG algorithms",
  "hud.chain.base": "Base",
  "hud.chain.baseForm": "fear + arousal",
  "hud.chain.tension": "tension",
  "hud.chain.heart": "heart",
  "hud.chain.f12": "F12 · face only",
  "hud.chain.f15": "F15 · production",

  // ── player section ──
  "player.kicker": "The centerpiece",
  "player.title": "The fear-analysis HUD, on real sessions",
  "player.lead": "The same instrument the pipeline renders to video — now live. Pick a recorded session below; the fear-score trace doubles as the scrubber, and every readout updates from the current frame.",
  "player.faceRoi": "FER · face ROI",
  "player.foreheadRoi": "rPPG · forehead",
  // ⚠ ASSEMBLED — preceded by the session id: "S06_Vid16 · " + player.replay.
  "player.replay": "analysis replay",
  "player.capFear": "Fear moment — score over threshold",
  "player.capMonitor": "Monitoring — no fear detected",
  "player.capFaceLost": "Face lost — subject out of frame",
  "player.traceTitle": "Fear score · drag anywhere to scrub",
  "player.legF15": "F15 +heart",
  "player.legF12": "F12 face",
  "player.legWindow": "fear window",

  // ── session rail (light.* double as CSS keys — keep them short) ──
  "rail.title": "Recorded sessions",
  // ⚠ ASSEMBLED — preceded by the clip count: "6 " + rail.meta → "6 clips · …".
  "rail.meta": "clips · 6 subjects · 3 lighting conditions",
  // ⚠ SLOT — {file} renders as the monospaced filename "raw_video.mp4" (a literal,
  //   not a key). Keep {file} verbatim; it may move within the sentence.
  "rail.note": "Each card streams its own real per-frame data + {file}. The numbers match the offline HUD renderer exactly.",
  "light.bright": "bright",
  "light.dim": "dim",
  "light.mixed": "mixed",

  // ── fusion section ──
  "fusion.kicker": "Fusion",
  "fusion.title": "Does the heartbeat help?",
  // ⚠ SPLIT SENTENCE — fusion.lead has two slots filled by bold mono fragments
  //   fusion.leadF12 ({f12}) and fusion.leadF15 ({f15}). Assembled:
  //   "…a calm pulse holds back the F12 ≥ 0.70 face score, while a genuine
  //    cardiac response pushes the fused F15 over 0.80."
  //   Translate the three together; keep {f12}/{f15} tokens and the F12/F15/≥/0.70
  //   tokens. "over" in leadF15 is a real word — translate it.
  "fusion.lead": "A fearful face can lie. Adding a contactless heart-rate read turns a face-only guess into a body-confirmed decision — a calm pulse holds back the {f12} face score, while a genuine cardiac response pushes the fused {f15}.",
  "fusion.leadF12": "F12 ≥ 0.70",
  "fusion.leadF15": "F15 over 0.80",
  // ⚠ ASSEMBLED — followed by the case number: fusion.case + " 1" → "Case 1".
  "fusion.case": "Case",

  // ── F12-vs-F15 teaching readout ──
  "teach.signalsCap": "Signals this moment",
  "teach.rowFear": "Fear (face)",
  "teach.rowArousal": "Arousal",
  "teach.rowTension": "Facial tension",
  "teach.rowBpm": "Heart-rate rise",
  "teach.verdictsCap": "Two verdicts, two thresholds",
  "teach.rowF12": "F12 · face only",
  "teach.rowF15": "F15 · + heart rate",

  // ── footer (names HSEmotion/MediaPipe/author/advisor stay literal) ──
  "footer.blurb": "Real-time emotion analysis for adaptive enemy AI in affective game design. A webcam-only, privacy-preserving approach to sensing fear — fusing facial emotion recognition with contactless heart rate.",
  "footer.project": "Project",
  "footer.author": "Author",
  "footer.advisor": "Advisor",
  "footer.date": "Date",
  "footer.dateVal": "May 2026",
  "footer.institution": "Institution",
  "footer.university": "University",
  "footer.uniVal": "Galatasaray University",
  "footer.faculty": "Faculty",
  "footer.facultyVal": "Engineering & Technology",
  "footer.department": "Department",
  "footer.deptVal": "Computer Engineering",

  // ── section customizer ({shown}/{total} replaced with counts) ──
  "cust.aria": "Customize sections",
  "cust.title": "Customize this page",
  // ⚠ {shown}/{total} are replaced with counts at runtime (e.g. "10 of 12 …").
  //   Keep both tokens; you may reorder them to fit grammar.
  "cust.subtitle": "{shown} of {total} sections shown · saved on this device",
  "cust.close": "Close",
  "cust.hidden": "Hidden",
  "cust.jump": "Jump to section",
  "cust.alwaysOn": "always on",
  "cust.moveUp": "Move up",
  "cust.moveDown": "Move down",
  "cust.cantHide": "This section can't be hidden",
  "cust.show": "Show",
  "cust.hide": "Hide",
  "cust.reset": "Reset to default",
  "cust.hint": "Reorder with ↑↓ · toggle to hide",

  // ── section customizer full labels (nav.* are the terse rail/menu labels) ──
  "section.overview": "Overview",
  "section.background": "Background",
  "section.pipeline": "Pipeline",
  "section.player": "HUD Player",
  "section.fusion": "F12 vs F15",
  "section.methods": "Methods",
  "section.results": "Results",
  "section.performance": "Performance",
  "section.game": "The Game",
  "section.path": "Design Path",
  "section.outlook": "Outlook",
  "section.glossary": "Glossary",

  // ── methods section (full) ──
  "methods.kicker": "Methods",
  "methods.title": "What made it work in the dark",
  "methods.leadFull": "Four engineering choices carry the pipeline through horror-game lighting. Each one is broken down below — detection fallback, contrast recovery, multiplicative fusion, and the rPPG sweep.",
  // ⚠ ASSEMBLED — followed by the method number: methods.label + " 01" → "METHOD 01".
  "methods.label": "METHOD",

  // method 01 — detection fallback
  "m1.title": "Hybrid detection that survives the dark",
  "m1.p": "A fast Haar cascade crops the face every frame. When it loses the face in low light, a MediaPipe FaceLandmarker fallback takes over — so detection holds even as the room dims. The darker and more uneven the scene, the more the fallback earns its keep.",
  "m1.stat": "face coverage across analyzed frames",
  "m1.foot": "Per-condition shares bind to each session's detector log.",
  "m1.visTitle": "Detector by lighting",
  "m1.visR": "per frame",
  "m1.rowBright": "Bright",
  "m1.rowDim": "Dim",
  "m1.rowMixed": "Mixed",
  "m1.pcBright": "Haar handles most",
  "m1.pcDim": "fallback on ~46%",
  "m1.pcMixed": "fallback on ~55%",
  "m1.legHaar": "Haar cascade",
  "m1.legMp": "MediaPipe fallback",
  "m1.legMiss": "missed by both",
  "m1.note": "A few frames are caught by neither detector — and that's expected. Recorded video has no guarantee a face is present 100% of the time: occlusion (hands, hair), extreme head turns, or edits that cut away from the camera all leave short gaps.",

  // method 02 — CLAHE (RAW/CLAHE tags stay literal)
  "m2.title": "CLAHE restores the face before reading it",
  "m2.p": "Contrast-Limited Adaptive Histogram Equalization stretches the cramped, dark tones of the face ROI across the full range — pulling expression detail out of the shadows so the emotion model and the skin-colour pulse signal both get a clean read. Drag to compare.",
  "m2.stat": "recovered detail in dim & mixed light",
  "m2.foot": "Real face crop from session S02_Vid04 (dim lighting). CLAHE: clipLimit 2.0, 4×4 tiles, LAB L-channel.",
  "m2.visTitle": "Face ROI · raw → CLAHE",
  "m2.visR": "drag ⇆",
  "m2.altRaw": "Raw face ROI",
  "m2.altOn": "CLAHE enhanced",
  "m2.roiLabel": "analysis ROI",
  "m2.histRaw": "Raw histogram",
  "m2.histEq": "Equalized",

  // method 03 — fusion ladder (HTML <b>/<em> + F12/F15 stay literal in markup)
  "m3.title": "Fusion that multiplies, not averages",
  "m3.p": "Facial tension and a rising heart rate don't get averaged into the score — they act as <em> amplifiers</em> on the base fear reading. Because they multiply, agreement compounds: a calm body barely moves the needle, while genuine arousal pushes a true fear event clear over the line. Move the sliders to feel it.",
  "m3.visTitle": "Live fusion · F12 ≥ 0.70 · F15 ≥ 0.80",
  "m3.stepBase": "Base <b>(face + arousal)</b>",
  "m3.stepF12": "× tension → <b>F12</b>",
  "m3.stepF15": "× heart rate → <b>F15</b>",
  "m3.exFear": "body confirms the face",
  "m3.exClear": "face alone isn't enough",

  // method 04 — rPPG sweep (estimator names GREEN/ICA/.../POS + windows stay literal)
  "m4.title": "Six estimators, one 30-second window",
  "m4.p": "Reading a heartbeat from skin colour needs the right algorithm and the right time window. All six rPPG estimators are computed over a <b style=\"color: var(--clear)\">30-second window at a 3-second step</b>; <b style=\"color: var(--clear)\">POS</b> is used as the headline pulse for its robustness under motion — the cell that lit up brightest.",
  "m4.stat": "headline estimator · 3 s step",
  "m4.foot": "The grid scores how cleanly each estimator × window separates fear from calm — red is poor, green is strong.",
  "m4.visTitle": "rPPG sweep · estimators",
  "m4.visR": "relative",
  "m4.scaleLow": "low",
  "m4.scaleHigh": "high",
  "m4.bestLabel": "best:",
  "m4.key": "Each cell is a <b>detectability score from 0 to 1</b> — how cleanly that estimator and window separate genuine fear events from calm baseline. <b>1.00</b> would be perfect separation; POS at a 30 s window tops out at <b>0.95</b>, while short windows and weaker estimators (red) blur the cardiac signal. <i>Illustrative — exact values bind to the benchmark output.</i>",

  // ── report: pipeline section ──
  "report.pipe.kicker": "The pipeline",
  "report.pipe.title": "One webcam, eight stages, fully on-device",
  "report.pipe.lead": "From a single consumer webcam to enemy AI — no dedicated sensors, no cloud. Each frame is enhanced, the face is found, emotion and pulse are read in parallel, then fused into one fear score and a discrete event the game can act on. No video ever leaves the machine.",
  "report.pipe.leg1": "pre-processing",
  "report.pipe.leg2": "facial channel",
  "report.pipe.leg3": "physiological channel",
  "report.pipe.leg4": "fusion",
  "report.pipe.leg5": "to the game",
  "report.rq.answered": "answered",
  "report.rq.promising": "promising",
  "report.rq.open": "open",

  // ── report: results section ──
  "report.res.kicker": "Results",
  "report.res.title": "What the evaluation found",
  "report.res.lead": "Six subjects, 88 annotated fear events, three lighting conditions. Five findings shaped the production formula — and one of them began as an apparent failure.",
  "report.res.asymKicker": "The unexpected asymmetry",
  "report.res.standaloneF1": "standalone F1",
  "report.res.asymCap": "The two FER tools were assumed interchangeable. The data said otherwise: <b>HSEmotion is the detector, MediaPipe is the amplifier.</b> The multiplicative formula encodes exactly this — tension can lift a fear reading, but can never trigger one on its own.",
  "report.cfg.config": "rPPG config",
  "report.cfg.boost": "boost c",
  "report.cfg.precision": "precision",
  "report.cfg.recall": "recall",
  "report.cfg.f1": "F1",
  "report.cfg.why": "why",

  // ── report: game section (Relax-to-Win / enemy names / FSM stay literal) ──
  "report.game.kicker": "The game",
  "report.game.title": "La Façade Fissurée — where the fear signal goes",
  "report.game.lead": "A first-person horror game in an abandoned, dimly-lit building. The pipeline is the controller: enemy AI reacts to your detected fear in real time. It inverts the genre — your physiological state is an active input, not a passive reaction.",
  "report.game.r2wIntro": "Show fear and the enemy grows aggressive — a positive feedback loop you have to break by actively calming down. The only defence is emotional self-regulation: suppress visible fear to de-escalate the threat. The mechanic has direct relevance to exposure therapy, where controlled exposure paired with self-regulation training is a core technique.",
  "report.game.loopTag": "The feedback loop",
  "report.game.loopTitle": "Your calm is the controller",
  "report.game.step1": "<b>You show fear</b> — F12/F15 crosses the threshold.",
  "report.game.step2": "The enemy escalates: <b>WANDER → ALERT → CHASE</b>.",
  "report.game.step3": "<b>You calm down</b> — fear drops, the enemy disengages.",
  "report.game.calm": "calm",
  "report.game.afraid": "afraid",
  "report.game.validated": "validated",
  "report.game.protoV1": "Prototype v1",
  "report.game.v1Title": "4-room graybox",
  "report.game.v1Desc": "A working build with NavMesh enemy AI, a four-state FSM, safe-room triggers, TCP socket link, a 26-column per-frame logger, and a calibration step disguised as a “Security Scan.” This is the prototype the six-subject evaluation was built on.",
  "report.game.inDev": "in development",
  "report.game.redesignV2": "Redesign v2.0",
  "report.game.v2Title": "17 rooms · garden ring",
  "report.game.v2li1": "17 rooms + 6 corridors + 4 garden zones in a 50×70 m footprint (~3.5× the usable area).",
  "report.game.v2li2": "The Wanderer gains a fifth POUNCE state and three independent detection channels.",
  "report.game.v2li3": "The Watcher — a new proximity entity with a 2×2 see/fear decision matrix.",
  "report.game.v2li4": "Key-gated escalation across 7 tiers; 20-minute target session.",

  // ── report: design path section ──
  "report.path.kicker": "The design path",
  "report.path.title": "A plan that bifurcated, not derailed",
  "report.path.lead": "The project was meant to be one straight line: build the pipeline, wire it into the game, run a player study. A hardware failure split that line in two — and reshaped what the evaluation could be.",
  "report.path.shipped": "shipped",
  "report.path.disrupted": "disrupted",

  // ── background: literature (citations/years stay literal; <b> kept in markup) ──
  "bg.kicker": "Background",
  "bg.title": "What the literature already knew — and didn't",
  "bg.lead": "The review follows the pipeline's own logic: facial emotion recognition, remote photoplethysmography, multimodal fusion, biofeedback, and privacy. Each result below shaped a concrete design decision.",
  "bg.lit.0.t": "Facial emotion recognition",
  "bg.lit.0.r0.d": "Fair comparison of 12 FER algorithms — best (Poster) 75.98%, HOG-SVM 50.12%. <b>Chose an EfficientNet-B0 backbone.</b>",
  "bg.lit.0.r1.d": "HSEmotion at ABAW-8; temporal smoothing adds +3.93% F1. <b>Adopted a 50-frame rolling average.</b>",
  "bg.lit.0.r2.d": "Fear is the hardest class — it overlaps surprise (AU 4 vs AU 5). <b>Motivated the arousal channel.</b>",
  "bg.lit.1.t": "rPPG & low-light sensing",
  "bg.lit.1.r0.d": "POS reaches 1.1 BPM MAE, beating deep nets — but degrades at high heart rate. <b>POS as primary; z-score normalization.</b>",
  "bg.lit.1.r1.d": "Algorithm hierarchy ends at POS (highest SNR, no skin-tone prior). <b>POS suits coloured monitor light.</b>",
  "bg.lit.1.r2.d": "Mid-forehead is the optimal ROI; exclude the lower face under expression. <b>Upper-face ROI only.</b>",
  "bg.lit.2.t": "Multimodal fusion",
  "bg.lit.2.r0.d": "Decision-level fusion is most robust when modalities run at different rates (FER 30 fps, rPPG ~0.33 Hz). <b>F15 fuses at the decision level.</b>",
  "bg.lit.2.r1.d": "Each added modality gives consistent accuracy gains. <b>Supports a multimodal approach.</b>",
  "bg.lit.3.t": "Biofeedback & privacy",
  "bg.lit.3.r0.d": "Caroline: a Relax-to-Win horror game on a single contact heart-rate sensor. <b>The gap we fill: multimodal, webcam-only.</b>",
  "bg.lit.3.r1.d": "Blendshape Distribution Alignment — <1 ms per-subject calibration. <b>Lead candidate for future work.</b>",
  "bg.lit.3.r2.d": "Federated learning + AES-256 hits 87% — privacy and accuracy aren't exclusive. <b>Validates fully local processing.</b>",

  // ── background: gaps + SWOT ──
  "bg.gapsKicker": "Four cumulative gaps",
  "bg.gap.0.t": "FER robustness in low light",
  "bg.gap.0.d": "Benchmarks run under controlled light; reliability under a horror game's monitor-only lighting is undocumented.",
  "bg.gap.1.t": "The rPPG stress paradox",
  "bg.gap.1.d": "rPPG accuracy drops exactly when heart rate spikes — the most informative moments for an adaptive game.",
  "bg.gap.2.t": "Mismatched temporal resolutions",
  "bg.gap.2.d": "Fusing an instant facial signal with a slow physiological one has not been validated in a game context.",
  "bg.gap.3.t": "Single-signal biofeedback",
  "bg.gap.3.d": "Existing game biofeedback relies on one contact sensor; none combine FER + rPPG from a single webcam.",
  "bg.swot.strengths": "Strengths",
  "bg.swot.weaknesses": "Weaknesses",
  "bg.swot.opportunities": "Opportunities",
  "bg.swot.threats": "Threats",
  "bg.swot.s.0": "Pipeline is independent — testable in isolation",
  "bg.swot.s.1": "Modular architecture with graceful degradation",
  "bg.swot.s.2": "Fully local processing (privacy)",
  "bg.swot.s.3": "Mature open-source tools (MediaPipe, OpenCV)",
  "bg.swot.s.4": "Very low MediaPipe latency (~6 ms)",
  "bg.swot.w.0": "Single-camera dependence — no backup sensor",
  "bg.swot.w.1": "Small expected sample limits statistical power",
  "bg.swot.w.2": "Limited control over the player's face lighting",
  "bg.swot.o.0": "Growing webcam-biometrics adoption (Affectiva, iMotions)",
  "bg.swot.o.1": "Architecture reusable for therapeutic apps",
  "bg.swot.o.2": "Continuous valence–arousal complements discrete classes",
  "bg.swot.o.3": "MediaPipe blendshapes enable personalized metrics",
  "bg.swot.t.0": "rPPG may lack sensitivity for fear-induced cardiac change",
  "bg.swot.t.1": "FER low-light degradation may exceed thresholds",
  "bg.swot.t.2": "Participant recruitment difficulty",
  "bg.swot.t.3": "Risk of exceeding the 100 ms latency budget",
  "bg.swotNote": "<b>Retrospective:</b> the hardware failure turned the “small sample” weakness and the “recruitment” threat into a video-based evaluation that ultimately produced a more diverse subject pool.",

  // ── performance section ──
  "perf.kicker": "Performance & cost",
  "perf.title": "~53 ms, end to end, on a consumer laptop",
  "perf.lead": "The latency budget (measured on a 197 s session at 30 FPS) sits well inside the 100 ms real-time target. The camera's own frame period is the largest single cost; with pipelined capture the effective throughput reaches 27–30 FPS.",
  "perf.lat.0.nm": "Camera capture",
  "perf.lat.0.note": "frame period @ 30 FPS",
  "perf.lat.1.nm": "Haar + MP detect",
  "perf.lat.1.note": "hybrid face detection",
  "perf.lat.2.nm": "HSEmotion",
  "perf.lat.2.note": "EfficientNet-B0, CPU",
  "perf.lat.3.nm": "Formula",
  "perf.lat.3.note": "arithmetic only",
  "perf.lat.4.nm": "Socket emit",
  "perf.lat.4.note": "TCP localhost",
  "perf.latTotal": "total per-frame latency · ~19 FPS (27–30 pipelined)",
  "perf.latBudget": "budget: <100 ms ✓",
  "perf.cpuTitle": "CPU utilization",
  "perf.cpu1": "MediaPipe (C++ runtime)",
  "perf.cpu2": "HSEmotion (PyTorch, CPU)",
  "perf.cpuNote": "No CUDA under Windows during offline processing. The rPPG pipeline runs on a non-blocking background thread — ROI extraction ~1 ms/frame, FFT every 3 s.",

  // ── outlook: contributions / limitations / future / conclusion ──
  "outlook.kicker": "Contributions & outlook",
  "outlook.title": "What it proved, and what's next",
  "outlook.contribTitle": "Contributions",
  "outlook.limitTitle": "Limitations",
  "outlook.contrib.0.t": "Composite fear formula F12",
  "outlook.contrib.0.d": "A multiplicative multimodal score where MediaPipe tension amplifies but never triggers fear. F1 = 0.70 over 6 subjects / 88 events.",
  "outlook.contrib.1.t": "rPPG cardiac fear response",
  "outlook.contrib.1.d": "First empirical evidence that horror-game fear produces a detectable webcam-rPPG cardiac response (d = 0.70, FWER p = 0.029).",
  "outlook.contrib.2.t": "Hybrid low-light detection",
  "outlook.contrib.2.d": "A Haar + MediaPipe fallback that raises face coverage from 30.4% to 99.9% with no classification artefacts.",
  "outlook.contrib.3.t": "Open-source real-time pipeline",
  "outlook.contrib.3.d": "A complete Python FER + rPPG + TCP pipeline at ~53 ms, ready to drop into any game engine.",
  "outlook.contrib.4.t": "Event-level evaluation method",
  "outlook.contrib.4.d": "A sliding-window detection framework with temporal padding and false-positive proximity analysis, tuned to affective-game granularity.",
  "outlook.limit.0.t": "Sample size",
  "outlook.limit.0.d": "Six subjects, single annotator; no inter-annotator agreement. Insufficient for population-level claims.",
  "outlook.limit.1.t": "Ecological validity",
  "outlook.limit.1.d": "Videos processed offline, not live — validates detection accuracy, not closed-loop dynamics.",
  "outlook.limit.2.t": "No gameplay experiment",
  "outlook.limit.2.d": "The hardware failure blocked the planned two-condition study; RQ3 is unanswered.",
  "outlook.limit.3.t": "MediaPipe variance",
  "outlook.limit.3.d": "Blendshape activations stay sparse across subjects; no per-subject calibration was implemented.",
  "outlook.limit.4.t": "Single-camera dependence",
  "outlook.limit.4.d": "Momentary facial occlusion causes signal loss with no sensor-level fallback.",
  "outlook.limit.5.t": "Post-hoc rPPG config",
  "outlook.limit.5.d": "The winning window was found by sweep, not pre-registered. Survives FWER but needs replication.",
  "outlook.futureKicker": "Future work — three axes",
  "outlook.future.0.t": "Live gameplay study",
  "outlook.future.0.d": "Finish v2.0 and run the closed-loop experiment to answer RQ3, validating the rPPG finding on a larger sample.",
  "outlook.future.1.t": "Cross-subject calibration",
  "outlook.future.1.d": "Implement Blendshape Distribution Alignment (BDA) to normalize MediaPipe activation ranges per player.",
  "outlook.future.2.t": "Cardiac ground truth",
  "outlook.future.2.d": "Pair webcam rPPG with a contact PPG sensor to capture high-resolution heart-rate-variability metrics.",
  "outlook.conclTag": "Conclusion",
  "outlook.conclTitle": "The technical barrier to affective gaming is <em>no longer the hardware.</em>",
  "outlook.conclDesc": "A standard webcam, two open-source models and a composite formula running in 53 ms on a consumer laptop are enough to sense fear. What remains is evaluation methodology, cross-subject calibration, and the closed-loop validation that a live gameplay experiment will provide.",

  // ── glossary (term acronyms stay literal; ab/plain/tech translated) ──
  "gloss.kicker": "Glossary",
  "gloss.title": "Every term, two ways",
  "gloss.lead": "The same plain ↔ technical switch the HUD uses, applied to the vocabulary of the report.",
  "gloss.plainBtn": "Plain language",
  "gloss.techBtn": "Technical",
  "gloss.term.0.ab": "remote photoplethysmography",
  "gloss.term.0.plain": "Reading your pulse from video. A webcam sees tiny colour changes in your skin as blood moves with each heartbeat — no sensor touches you.",
  "gloss.term.0.tech": "Contactless PPG from an RGB camera: sub-pixel chromatic variation in facial skin is extracted from temporal fluctuations of mean ROI pixel values; the cardiac-band periodic component yields BPM.",
  "gloss.term.1.ab": "plane-orthogonal-to-skin",
  "gloss.term.1.plain": "The most accurate pulse method here — ~3 BPM error vs a smartwatch. It mathematically separates blood-flow colour change from movement and lighting.",
  "gloss.term.1.tech": "Projects the RGB signal onto a plane orthogonal to the mean skin-tone direction, removing intensity variation with no skin-tone prior. Highest SNR in Wang et al. (5.16 dB); MAE = 2.96 BPM here.",
  "gloss.term.2.ab": "multi-algorithm blend",
  "gloss.term.2.plain": "An attempt to combine all five pulse methods. It backfired: the weak methods dragged down the accurate one (POS), making the blend worse than POS alone.",
  "gloss.term.2.tech": "SNR-weighted average of five extractors. Weak algorithms (GREEN, WAVELET >13 BPM MAE) diluted POS, collapsing the cardiac fear effect (d = 0.042) until POS was isolated (d = 0.696).",
  "gloss.term.3.ab": "adaptive histogram equalization",
  "gloss.term.3.plain": "A contrast booster for dark video. It brightens the face region so detail hidden in shadow becomes readable before emotion is measured.",
  "gloss.term.3.tech": "Contrast-Limited Adaptive Histogram Equalization on the L channel (clipLimit 2.0, 4×4 tiles) — local equalization with contrast clipping to avoid noise amplification.",
  "gloss.term.4.ab": "facial emotion recognition",
  "gloss.term.4.plain": "Software that reads emotion from a face. This project uses two: a neural model (HSEmotion) and a geometric face tracker (MediaPipe).",
  "gloss.term.4.tech": "HSEmotion (EfficientNet-B0 / AffectNet) gives 8-class softmax probabilities; MediaPipe yields 52 facial blendshape activations for geometric tension.",
  "gloss.term.5.ab": "MediaPipe AU proxies",
  "gloss.term.5.plain": "52 sliders describing facial muscle movements — brow raise, jaw clench, eye widen. Together they sketch an expression.",
  "gloss.term.5.tech": "Per-frame activation coefficients for facial action proxies; fear-critical channels show median activations of 0.001–0.004, structurally too sparse to threshold alone.",
  "gloss.term.6.ab": "precision × recall",
  "gloss.term.6.plain": "One number for detector quality, balancing false alarms against missed events. 1.0 is perfect; this system reaches 0.77.",
  "gloss.term.6.tech": "Harmonic mean of precision and recall, evaluated at the event level via sliding-window matching with temporal padding.",
  "gloss.term.7.ab": "effect size",
  "gloss.term.7.plain": "How big a difference is, not just whether it exists. ~0.7 means the fear heart-rate response is a solid, medium-to-large effect.",
  "gloss.term.7.tech": "Standardized mean difference. The isolated-POS cardiac fear response reached d = 0.696 (FWER-corrected p = 0.029).",
  "gloss.term.8.ab": "fast Fourier transform",
  "gloss.term.8.plain": "A tool that finds repeating rhythms in a signal — like picking out which notes make up a chord. It locates the heartbeat frequency in the skin-colour signal.",
  "gloss.term.8.tech": "O(n log n) discrete Fourier transform on windowed rPPG segments; the dominant peak in the 1.0–3.0 Hz cardiac band gives the BPM estimate.",
  "gloss.term.9.ab": "the core mechanic",
  "gloss.term.9.plain": "The game gets scarier when you're scared. Your only defence is to calm down — self-regulation is the controller.",
  "gloss.term.9.tech": "A positive-feedback biofeedback loop: detected fear (F12/F15 > θ) escalates enemy state; the player must suppress visible fear to de-escalate.",
});
