# La Facade Fissuree

Real-time fear and emotion detection for adaptive horror game AI.

## Project Summary

**Capstone project (INF494)** -- Galatasaray University, Computer Engineering Department

- **Author:** Ali Burak Sarac (21401932)
- **Advisor:** Asst. Prof. Reis Burak Arslan
- **Year:** 2025--2026

This project builds a webcam-based multimodal affective computing pipeline that detects player fear in real time while playing a horror game. It combines **Facial Expression Recognition** (MediaPipe blendshapes + HSEmotion CNN fusion) with **remote photoplethysmography** (rPPG) heart-rate estimation, and streams the composite affect signal to a Unity game engine via TCP socket. The game adapts its enemy AI behavior based on detected fear levels.

## Repository Structure

```
Pipeline/           Python emotion analysis pipeline (FER + rPPG)
  fer/              Facial expression recognition package
  rppg/             Remote photoplethysmography package
  utils/            Shared utilities (session tracking, TCP bridge)
  analysis/         Post-session visualization and benchmarking
  sessions/         rPPG session artifacts (frames, analysis)
  logs/             FER session data, validation, and comparisons
  archive_legacy/   Superseded scripts (kept for reference)
  requirements.txt  Pinned Python dependencies

Website/            Interactive visualization and HUD replay
  media/sessions/   Demo session data (video + per-frame analytics)
  media/clahe/      CLAHE preprocessing illustration
  lang/             Localization (EN / FR / TR)
  vendor/           React runtime (production builds)
  dist/             Compiled JavaScript
  serve.py          Local development server

Annotations/        Ground truth annotation CSVs per session

Report/             Final deliverables
  21401932_AliBurakSarac.pdf          Final report (French)
  main_EN.pdf                         Final report (English)
  AliBurakSarac_Poster.pdf            Project poster
  1/2/3_Ali_Burak_Sarac.pdf           Interim reports
  21401932_RevueDeLitterature.pdf     Literature review
```

## Architecture

```
Webcam --> Face Detection (MediaPipe) --> FER (MP + HSEmotion fusion)
                                      --> rPPG (POS algorithm)
                                           |
                                     Composite Fear Score
                                           |
                                     TCP Socket --> Unity Game
                                           |
                                     Session CSV + HUD Visualization
```

The FER pipeline uses a two-gate state machine for event detection (IDLE -> ONSET -> SUSTAINING -> EVENT_CONFIRMED) and applies exponential smoothing to reduce noise. Three annotation algorithms (peak-window, flood-fill, multi-channel voting) enable offline validation against ground truth.

See [Pipeline/ARCHITECTURE.md](Pipeline/ARCHITECTURE.md) for detailed data flow diagrams and design patterns.

## Setup

### Pipeline (Python 3.12)

```bash
pip install -r Pipeline/requirements.txt
```

**FER analysis (live or from video):**

```bash
python Pipeline/fer/test_mp_hs.py --video <path_to_video>
```

**rPPG capture:**

```bash
python Pipeline/rppg/live_rppg.py capture --label <session_name>
```

**Post-session visualization:**

```bash
python Pipeline/analysis/visualize_session.py <session_csv>
```

### Website

```bash
python Website/serve.py
```

Then open `http://localhost:8000` in a browser. The website provides an interactive HUD replay for recorded sessions and a methodology walkthrough.

## AI Usage Declaration

The majority of the source code in this project was written with the assistance of AI tools. I was responsible for:

- Project planning and milestone management
- System architecture design and technology selection
- Experimental methodology and research direction
- Report content, analysis, and interpretation
- Code review, testing, and corrections

French translations of the reports were AI-assisted. The interactive website, Python pipeline code, and Unity game integration code were developed through iterative AI-assisted sessions directed by me.

## License

MIT License -- see [LICENSE](LICENSE) for details.
