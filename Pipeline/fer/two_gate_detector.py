"""Two-Gate Affective Event Detector.

Pattern: [State Machine] — four states: IDLE → ONSET → SUSTAINING/EVENT_CONFIRMED → COOLDOWN.
Pattern: [Ring Buffer]   — fixed-length deque for O(1) per-frame derivative computation.

Sits downstream of the FER pipeline (rolling-averaged composite fear signal).
Fires discrete affective events using a two-gate mechanism:

  Gate A — derivative threshold: slope > onset_threshold triggers a candidate.
  Gate B — sustain threshold: candidate must stay above floor_threshold for
           sustain_frames consecutive frames to be confirmed as an event.

State transitions:
  IDLE          ──[slope > onset_threshold]──────────────> ONSET
  ONSET         ──[sustain_count >= sustain_frames]──────> EVENT_CONFIRMED
  ONSET         ──[signal drops before Gate B met]───────> IDLE
  EVENT_CONFIRMED──[signal drops below floor_threshold]──> EVENT_ENDED → COOLDOWN
  COOLDOWN      ──[cooldown_remaining reaches 0]─────────> IDLE

Usage:
    detector = TwoGateDetector(fps=video_fps)
    status = detector.update(frame_number, smoothed_composite, raw_values)
    detector.write_events_csv(path)
"""

from collections import deque
import csv
import os


class TwoGateDetector:
    def __init__(
        self,
        # ── Original Gate A + Gate B params ───────────────────────────────────
        onset_threshold=0.015,  # Gate A sensitivity (units/frame)
        onset_window=10,        # frames over which derivative is measured (~0.33s @ 30fps)
        sustain_frames=15,      # Gate B: min consecutive frames above floor to confirm
        floor_threshold=0.30,   # composite fear floor for sustain counting [0,1]
        exit_threshold=0.18,    # hysteresis lower bound — cancel when signal drops below this
        max_gap_frames=3,       # gap tolerance — frames in hysteresis zone before cancelling
        cooldown_frames=30,     # refractory period after event ends (~1s @ 30fps)
        fps=30.0,               # video framerate — converts frame counts to duration_s
        # ── Evaluation-compatible params (offline → live parameter transfer) ──
        signal_col='composite_fear',  # column to threshold; 'composite_fear' = use smoothed_composite
        threshold=None,               # detection threshold; defaults to floor_threshold
        min_frames=None,              # minimum qualifying frames; defaults to sustain_frames
        window_size=None,             # None = strict mode; int = sliding-window Gate B mode
        fill_ratio=0.65,              # fraction of window frames that must be >= threshold
    ):
        self.onset_threshold = onset_threshold
        self.onset_window = onset_window
        self.sustain_frames = sustain_frames
        self.floor_threshold = floor_threshold
        self.exit_threshold = exit_threshold
        self.max_gap_frames = max_gap_frames
        self.cooldown_frames = cooldown_frames
        self.fps = fps

        # Evaluation-compatible Gate B parameters
        self.signal_col   = signal_col
        self._threshold   = threshold   if threshold   is not None else floor_threshold
        self._min_frames  = min_frames  if min_frames  is not None else sustain_frames
        self.window_size  = window_size
        self.fill_ratio   = fill_ratio

        # Ring buffer for sliding-window mode
        self._gate_b_buf   = deque(maxlen=window_size) if window_size else None
        self._qualify_count = 0

        # Bounded ring buffer for derivative computation.
        # maxlen = onset_window + 1 so index [-onset_window] is always valid
        # once the buffer is full.
        self.smoothed_history = deque(maxlen=onset_window + 1)

        # Per-frame public outputs
        self.current_onset_slope = 0.0

        # Internal event state
        self.in_event = False
        self.event_start_frame = None
        self.gate_a_frame = None
        self.sustain_count = 0
        self.confirmed = False
        self._gap_count = 0
        self.current_event_frames = 0
        self.current_event_peak = 0.0
        self.current_event_sum = 0.0
        self._saved_onset_slope = 0.0   # saved at Gate A fire time
        self._peak_hs_fear = 0.0        # per-channel peaks accumulated internally
        self._peak_hs_surprise = 0.0
        self._peak_mp_tension = 0.0

        # Refractory period
        self._cooldown_remaining = 0

        self.events = []
        self._event_id_counter = 0

    # ──────────────────────────────────────────────────────────────────────────

    def _pick_signal(self, smoothed_composite, raw_values):
        """Select the detection signal based on signal_col."""
        if self.signal_col == 'composite_fear' or self.signal_col not in raw_values:
            return smoothed_composite
        return raw_values[self.signal_col]

    def update(self, frame_number, smoothed_composite, raw_values):
        """Process one frame.

        Args:
            frame_number: integer frame index (1-based matches CSV row)
            smoothed_composite: post-rolling-average composite fear value [0,1]
            raw_values: dict with keys hs_fear, hs_surprise, mp_tension, and
                        optionally formula columns (f0–f11) for signal_col support

        Returns:
            str: "IDLE" | "ONSET" | "SUSTAINING" | "EVENT_CONFIRMED" | "EVENT_ENDED"
        """
        self.smoothed_history.append(smoothed_composite)

        hs_fear     = raw_values.get("hs_fear", 0.0)
        hs_surprise = raw_values.get("hs_surprise", 0.0)
        mp_tension  = raw_values.get("mp_tension", 0.0)

        # Update derivative for public output (used every frame in CSV)
        if len(self.smoothed_history) >= self.onset_window:
            self.current_onset_slope = (
                (smoothed_composite - self.smoothed_history[-self.onset_window])
                / self.onset_window
            )
        else:
            self.current_onset_slope = 0.0

        # Tick cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return "IDLE"

        # ── Sliding-window Gate B mode ────────────────────────────────────────
        if self.window_size is not None:
            return self._update_window_mode(
                frame_number, smoothed_composite, raw_values,
                hs_fear, hs_surprise, mp_tension
            )

        # ── Original strict-consecutive mode ─────────────────────────────────
        if not self.in_event:
            # Gate A check
            if (len(self.smoothed_history) >= self.onset_window
                    and self.current_onset_slope > self.onset_threshold):
                self.in_event = True
                self.gate_a_frame = frame_number
                self.event_start_frame = frame_number
                self._saved_onset_slope = self.current_onset_slope
                self.sustain_count = 1 if smoothed_composite >= self.floor_threshold else 0
                self.current_event_peak = smoothed_composite
                self.current_event_sum = smoothed_composite
                self.current_event_frames = 1
                self._peak_hs_fear = hs_fear
                self._peak_hs_surprise = hs_surprise
                self._peak_mp_tension = mp_tension
                return "ONSET"
            return "IDLE"

        else:
            # Accumulate event statistics
            self.current_event_frames += 1
            self.current_event_sum += smoothed_composite
            self.current_event_peak = max(self.current_event_peak, smoothed_composite)
            self._peak_hs_fear    = max(self._peak_hs_fear, hs_fear)
            self._peak_hs_surprise = max(self._peak_hs_surprise, hs_surprise)
            self._peak_mp_tension  = max(self._peak_mp_tension, mp_tension)

            if smoothed_composite >= self.floor_threshold:
                # Above floor — good frame, reset gap counter
                self.sustain_count += 1
                self._gap_count = 0
                if self.sustain_count >= self.sustain_frames:
                    self.confirmed = True
                    return "EVENT_CONFIRMED"
                return "SUSTAINING"

            elif smoothed_composite >= self.exit_threshold:
                # Hysteresis zone: tolerate up to max_gap_frames before cancelling
                self._gap_count += 1
                if self._gap_count <= self.max_gap_frames:
                    return "EVENT_CONFIRMED" if self.confirmed else "SUSTAINING"
                # Gap exceeded — close or discard
                if self.confirmed:
                    event = self._close_event(frame_number)
                    self.events.append(event)
                    self._reset()
                    return "EVENT_ENDED"
                self._reset()
                return "IDLE"

            else:
                # Hard exit: signal below exit_threshold — cancel immediately
                if self.confirmed:
                    event = self._close_event(frame_number)
                    self.events.append(event)
                    self._reset()
                    return "EVENT_ENDED"
                self._reset()
                return "IDLE"

    def _update_window_mode(self, frame_number, smoothed_composite, raw_values,
                            hs_fear, hs_surprise, mp_tension):
        """Sliding-window Gate B: no mandatory Gate A — fires when fill_ratio of
        last window_size frames are >= threshold for min_frames total frames."""
        signal_val = self._pick_signal(smoothed_composite, raw_values)
        self._gate_b_buf.append(signal_val)

        required = int(self.window_size * self.fill_ratio)
        window_qualified = (
            len(self._gate_b_buf) == self.window_size
            and sum(1 for v in self._gate_b_buf if v >= self._threshold) >= required
        )

        if window_qualified:
            if not self.in_event:
                self.in_event = True
                # Back-date event start to beginning of the qualifying window
                self.event_start_frame = max(1, frame_number - self.window_size + 1)
                self.gate_a_frame = (
                    frame_number if self.current_onset_slope > self.onset_threshold else None
                )
                self._saved_onset_slope = self.current_onset_slope
                self.current_event_peak = signal_val
                self.current_event_sum  = signal_val
                self.current_event_frames = 1
                self._peak_hs_fear    = hs_fear
                self._peak_hs_surprise = hs_surprise
                self._peak_mp_tension  = mp_tension
                self._qualify_count   = 1
                self._gap_count       = 0
                return "ONSET"
            else:
                self.current_event_frames += 1
                self.current_event_sum    += signal_val
                self.current_event_peak    = max(self.current_event_peak, signal_val)
                self._peak_hs_fear         = max(self._peak_hs_fear, hs_fear)
                self._peak_hs_surprise     = max(self._peak_hs_surprise, hs_surprise)
                self._peak_mp_tension      = max(self._peak_mp_tension, mp_tension)
                self._qualify_count       += 1
                self._gap_count            = 0
                if self._qualify_count >= self._min_frames:
                    self.confirmed = True
                    return "EVENT_CONFIRMED"
                return "SUSTAINING"

        else:
            # Window not qualified
            if self.in_event:
                # Tolerate short gaps (re-uses max_gap_frames as gap tolerance)
                self._gap_count += 1
                if self._gap_count <= self.max_gap_frames:
                    return "EVENT_CONFIRMED" if self.confirmed else "SUSTAINING"
                if self.confirmed:
                    event = self._close_event(frame_number)
                    self.events.append(event)
                    self._reset()
                    return "EVENT_ENDED"
                self._reset()
                return "IDLE"
            return "IDLE"

    def flush(self, last_frame_number):
        """Call after frame loop ends to close any open mid-video event.

        If the event was in progress and Gate B was already satisfied,
        the event is closed and logged. Otherwise it is silently discarded.
        """
        if self.in_event and self.sustain_count >= self.sustain_frames:
            event = self._close_event(last_frame_number)
            self.events.append(event)
        self._reset()

    # ──────────────────────────────────────────────────────────────────────────

    def _close_event(self, end_frame):
        self._event_id_counter += 1
        return {
            "event_id": f"E{self._event_id_counter:02d}",
            "event_start_frame": self.event_start_frame,
            "event_end_frame": end_frame,
            "gate_a_frame": self.gate_a_frame,
            "duration_frames": self.current_event_frames,
            "duration_seconds": round(self.current_event_frames / self.fps, 3),
            "peak_composite": round(self.current_event_peak, 4),
            "mean_composite": round(
                self.current_event_sum / max(1, self.current_event_frames), 4
            ),
            "onset_slope": round(self._saved_onset_slope, 6),
            "peak_hs_fear": round(self._peak_hs_fear, 4),
            "peak_hs_surprise": round(self._peak_hs_surprise, 4),
            "peak_mp_tension": round(self._peak_mp_tension, 4),
        }

    def _reset(self):
        self.in_event = False
        self.confirmed = False
        self._gap_count = 0
        self.event_start_frame = None
        self.gate_a_frame = None
        self.sustain_count = 0
        self._qualify_count = 0
        self.current_event_frames = 0
        self.current_event_peak = 0.0
        self.current_event_sum = 0.0
        self._saved_onset_slope = 0.0
        self._peak_hs_fear = 0.0
        self._peak_hs_surprise = 0.0
        self._peak_mp_tension = 0.0
        self._cooldown_remaining = self.cooldown_frames

    # ──────────────────────────────────────────────────────────────────────────

    def write_events_csv(self, path):
        """Write the detected event summary to a CSV file."""
        fieldnames = [
            "event_id", "event_start_frame", "event_end_frame", "gate_a_frame",
            "duration_frames", "duration_seconds",
            "peak_composite", "mean_composite", "onset_slope",
            "peak_hs_fear", "peak_hs_surprise", "peak_mp_tension",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.events)
        print(f"[TwoGateDetector] {len(self.events)} events → {os.path.basename(path)}")
