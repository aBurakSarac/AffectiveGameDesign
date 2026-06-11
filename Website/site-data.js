/* site-data.js — site copy + the real presentation session list + report content.
 * Sessions mirror Pipeline/presentation/<stem>/ (subject, video_id, lighting).
 * Figures are taken from the final report (La Façade Fissurée, May 2026) and the
 * pipeline source; numbers labelled "illustrative" are bindable to per-session logs.
 */
(function () {
  "use strict";

  // 7 real presentation sessions (from Pipeline/presentation/*/meta.json)
  const SESSIONS = [
    { id: "S06_Vid16", subject: "S06", vid: "Vid16", lighting: "bright", dur: 876, videoWidth: 290, videoHeight: 240, events: 11, note: "Well-lit baseline — clean FER, strong rPPG signal." },
    { id: "S02_Vid04", subject: "S02", vid: "Vid04", lighting: "dim",    dur: 598, videoWidth: 200, videoHeight: 240, events: 14, note: "Dim room — MediaPipe fallback carries detection." },
    { id: "S08_Vid18", subject: "S08", vid: "Vid18", lighting: "mixed",  dur: 275, videoWidth: 480, videoHeight: 320, events: 17, note: "Flickering screen-lit scene — hardest case." },
    { id: "S04_Vid09", subject: "S04", vid: "Vid09", lighting: "bright", dur: 504, videoWidth: 240, videoHeight: 260, events: 9,  note: "Steady lighting, several genuine startles." },
    { id: "S05_Vid10", subject: "S05", vid: "Vid10", lighting: "bright", dur: 343, videoWidth: 270, videoHeight: 260, events: 12, note: "High arousal range — fusion separates true fear." },
    { id: "S10_Vid13", subject: "S10", vid: "Vid13", lighting: "bright", dur: 425, videoWidth: 220, videoHeight: 210, events: 13, note: "Expressive subject — fear-face false alarms tested." },
  ];

  // headline outcomes — REAL numbers from the final report (§6.6, abstract)
  const RESULTS = [
    { k: "F15 event F1", v: "0.77", s: "rPPG-augmented · 90% precision", tone: "accent" },
    { k: "End-to-end latency", v: "53", unit: "ms", s: "fully on-device · <100 ms budget", tone: "clear" },
    { k: "Face detection", v: "99.9", unit: "%", s: "hybrid Haar + MediaPipe, low light", tone: "" },
    { k: "Subjects · events", v: "6 · 88", s: "annotated fear events, 3 lighting sets", tone: "" },
  ];

  const FUSE = {
    f12: { tag: "F12", name: "face only", thr: 0.70,
           eq: ['(0.7·Fear + 0.3·Arousal)', ' × (1 + facial tension)'] },
    f15: { tag: "F15", name: "+ heart rate", thr: 0.80,
           eq: ['F12', ' × (1 + 0.5·heart-rate rise)'] },
  };

  // teaching cases for the F12 vs F15 page (faithful to the fusion logic)
  const TEACH = [
    {
      key: "false", tab: "Fearful face, calm body",
      title: "A scare face — but the body is calm",
      fear: 0.64, arousal: 0.22, tension: 0.16, bpmRise: 0.02,
      f12Why: "A strong fear expression alone pushes F12 close to the line…",
      f15Why: "…but flat arousal and a resting heart rate hold the fused score below 0.80. Correctly read as not fear — the kind of false alarm a face-only model over-calls.",
    },
    {
      key: "true", tab: "Genuine fear",
      title: "Fearful face, high arousal, elevated pulse",
      fear: 0.72, arousal: 0.66, tension: 0.25, bpmRise: 0.26,
      f12Why: "Fear and arousal together already lift F12 over its 0.70 line.",
      f15Why: "An elevated heart rate amplifies it further — F15 crosses 0.80 with high confidence. A genuine fear event the body confirms.",
    },
  ];

  const METHODS = [
    { ic: "lighting", t: "Haar-first, MediaPipe fallback",
      p: "A fast Haar cascade crops the face every frame; when it loses the face in low light, MediaPipe FaceLandmarker steps in. The hybrid keeps a face locked on 99.9% of analyzed frames." },
    { ic: "contrast", t: "CLAHE contrast enhancement",
      p: "Contrast-Limited Adaptive Histogram Equalization (clipLimit 2.0, 4×4 tiles, on the L channel) restores detail in the face ROI before emotion is read." },
    { ic: "fuse", t: "Multiplicative fusion",
      p: "HSEmotion detects; MediaPipe tension and heart-rate amplify. They multiply rather than average, so a calm body can't trigger fear but a confirming one compounds it." },
    { ic: "pulse", t: "rPPG estimator sweep",
      p: "A 48-configuration sweep over estimators, windows and boost coefficients settled on POS at a 30 s window / 3 s step — the only choice that preserved 90% precision." },
  ];

  // ── PIPELINE stages (architecture flow, §5.1) ──
  const PIPELINE = [
    { k: "01", t: "Capture", s: "Consumer webcam", d: "A single ordinary webcam — no dedicated biometric hardware. Everything downstream runs on the player's own machine.", tag: "input" },
    { k: "02", t: "CLAHE", s: "Contrast recovery", d: "Adaptive histogram equalization lifts the dark, low-contrast face ROI typical of horror lighting.", tag: "pre" },
    { k: "03", t: "Hybrid detect", s: "Haar → MediaPipe", d: "Fast Haar cascade first; MediaPipe FaceLandmarker rescues frames it loses. 99.9% coverage.", tag: "pre" },
    { k: "04", t: "FER", s: "HSEmotion + MediaPipe", d: "HSEmotion (EfficientNet-B0) reads fear & arousal — the detector. MediaPipe's 52 blendshapes give facial tension — the amplifier.", tag: "face" },
    { k: "05", t: "rPPG", s: "POS @ 30 s / 3 s", d: "Skin-colour pulse from the forehead ROI. POS estimator over a 30-second window, stepped every 3 s.", tag: "body" },
    { k: "06", t: "Fusion", s: "F12 → F15", d: "Multiplicative composite: face score, amplified by tension, then by heart-rate rise. Thresholds 0.70 / 0.80.", tag: "fuse" },
    { k: "07", t: "Dual-gate event", s: "IDLE · ONSET · CONFIRMED", d: "Two gates reject slow drifts and brief spikes, so only sustained fear becomes a discrete event.", tag: "fuse" },
    { k: "08", t: "Game over TCP", s: "Enemy AI", d: "Discrete event state + smoothed score stream to the Unity game — driving enemy escalation and adaptive behaviour.", tag: "out" },
  ];

  // ── research questions (§1) ──
  const RQS = [
    { id: "RQ1", q: "Can facial emotion recognition stay accurate in horror-game low light?", a: "Yes — F1 = 0.77 under monitor-only lighting, with 99.9% face coverage from the hybrid detector.", status: "answered" },
    { id: "RQ2", q: "Does decision-level FER + rPPG fusion improve detection reliability?", a: "Promising — rPPG augmentation lifts F1 from 0.70 to 0.77 at 90% precision on the four sessions with usable pulse. Preliminary; needs more participants.", status: "partial" },
    { id: "RQ3", q: "Does a \u201cRelax-to-Win\u201d adaptation improve the player experience?", a: "Formally unanswered. The hardware failure forced a video-based evaluation on passive viewers, which omits the closed-loop biofeedback. This is the primary future-work target.", status: "open" },
  ];

  // ── key findings (§6.6) ──
  const FINDINGS = [
    { n: "01", t: "Multiply, don't average", p: "Equal-weight fusion of HSEmotion and MediaPipe collapses across subjects — MediaPipe's tension activations are structurally too sparse. A multiplicative modifier (tension amplifies but can't trigger fear alone) fixes it." },
    { n: "02", t: "The arousal channel", p: "Adding 30% hs_arousal to the fear signal recovers events lost to fear–surprise confusion, producing F12 at F1 = 0.70." },
    { n: "03", t: "The rPPG null was an artefact", p: "The CONSENSUS estimator diluted POS's precise signal with weaker algorithms (d = 0.042). Isolating POS and matching the window to event duration revealed a medium-large cardiac fear response (d = 0.696, FWER p = 0.029)." },
    { n: "04", t: "rPPG augmentation is promising", p: "F15 lifts F1 to 0.77 while preserving 90% precision on four sessions. The pulse lags FER by ~8–18 s, creating a complementary two-phase \u201csustained terror\u201d dynamic." },
    { n: "05", t: "Low-light robustness confirmed", p: "F1 = 0.77 under monitor-only lighting, with 99.9% face-detection coverage from the hybrid Haar + MediaPipe approach." },
  ];

  // FER asymmetry (§7.1) — standalone F1 of each tool
  const ASYMMETRY = {
    detector: { name: "HSEmotion", sub: "EfficientNet-B0 · AffectNet · 8-class softmax", f1: 0.67, role: "THE DETECTOR", note: "Holistic neural classification. Robust across faces with no per-subject calibration." },
    amplifier: { name: "MediaPipe", sub: "52 facial blendshapes · geometric tension", f1: 0.065, role: "THE AMPLIFIER", note: "Sparse activations — weak alone, but confirms HSEmotion when both agree on a fear moment." },
  };

  // rPPG augmentation sweep best configs (Table 6.8)
  const RPPG_CONFIGS = [
    { cfg: "POS @ 30s / 3s", c: 0.5, p: 90.0, r: 66.7, f1: 0.766, best: true, why: "Highest F1, preserves 90% precision — the production choice." },
    { cfg: "POS @ 10s / 2s", c: 1.0, p: 81.2, r: 72.2, f1: 0.765, best: false, why: "Comparable F1 but trades 10pp precision for recall — bad for a game." },
    { cfg: "POS @ 10s / 5s", c: 0.8, p: 85.0, r: 63.0, f1: 0.740, best: false, why: "Solid, but lower F1 than the 30 s window." },
  ];

  // ── THE GAME (§5.8) ──
  const ENEMIES = [
    {
      key: "wanderer", name: "La Bête", aka: "The Wanderer", cls: "Insectoid quadruped · hunts by fear, not by sight",
      fsm: ["WANDER", "ALERT", "CHASE", "POUNCE", "COOLDOWN"],
      channels: [
        { t: "Motion cone", d: "90° forward, 12 m, raycast-occluded — only fires on movement. Stay still and you're invisible, even inside the cone." },
        { t: "Sprint-hearing bubble", d: "360°, 7 m radius, wall-occluded — hears sprinting but not walking." },
        { t: "Fear-ping", d: "Global, >15 m only — reads your smoothed composite fear and steers the enemy toward you. The direct biofeedback channel." },
      ],
      accent: "danger",
    },
    {
      key: "watcher", name: "Le Veilleur", aka: "The Watcher", cls: "Directionless proximity entity · attacks only the unseen and the afraid",
      fsm: ["IDLE", "APPEAR", "FROZEN / DESPAWN / STRIKE"],
      matrix: [
        { see: "Look at it", calm: "Freezes", afraid: "Freezes" },
        { see: "Look away", calm: "Despawns", afraid: "STRIKE → game over" },
      ],
      channels: [
        { t: "Proximity spawn", d: "Appears 5–9 m away, always just outside the camera frustum." },
        { t: "2×2 decision", d: "Behaviour is set by (do you see it) × (are you afraid). Eye contact freezes it; calm makes it leave; fear while looking away is fatal." },
        { t: "Key escalation", d: "Spawn interval tightens per collected key across 7 tiers — from ~17 min at 0 keys down to ~2 min at 6 keys." },
      ],
      accent: "moon",
    },
  ];

  // ── DESIGN PATH (the three-bullet arc; game = 3rd, hit by hardware) ──
  const PATH = [
    {
      n: "01", t: "The pipeline", state: "shipped", meta: "Phases 1–2 · built & validated",
      d: "A real-time, on-device affective-computing pipeline: hybrid detection, CLAHE, HSEmotion + MediaPipe FER, POS rPPG, multiplicative fusion and a dual-gate event detector. Preserved in git throughout.",
    },
    {
      n: "02", t: "The evaluation pivot", state: "shipped", meta: "Phase 4 · video-based study",
      d: "Validated on 6 diverse subjects and 88 annotated fear events across three lighting conditions, using horror-reaction videos with independent manual annotation — a broader, more diverse corpus than the lab study originally planned.",
    },
    {
      n: "03", t: "The game integration", state: "disrupted", meta: "Phase 3 · planned, hardware-hit",
      d: "La Façade Fissurée — a first-person horror game (177 files, 20k+ lines of Unity) where enemy AI reacts to your detected fear. The main development machine failed after 13 April 2026; the prototype couldn't run on the backup, so the closed-loop game integration was frozen while the pipeline carried on. The Relax-to-Win player study is the primary future-work target.",
    },
  ];

  window.SITE = { SESSIONS, RESULTS, FUSE, TEACH, METHODS, PIPELINE, RQS, FINDINGS, ASYMMETRY, RPPG_CONFIGS, ENEMIES, PATH };
})();
