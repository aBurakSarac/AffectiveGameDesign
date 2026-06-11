"""HUD layout constants for the rPPG replay renderer.

Keeping magic numbers here (not in replay.py) lets future renderers reuse
the same layout without importing replay logic.
"""

# Panel dimensions
PANEL_WIDTH   = 400   # pixels — right-side BPM panel
HUD_MIN_HEIGHT = 480  # pixels — minimum canvas height

# Background / UI colours (BGR)
PANEL_BG           = (25,  25,  30)
SECTION_DIVIDER    = (70,  70,  80)
TEXT_HEADER        = (130, 200, 255)  # light blue
TEXT_LABEL         = (200, 200, 200)  # light grey
TEXT_GT            = (100, 255, 100)  # green for GT HR
TEXT_DELTA_POS     = (100, 200, 255)  # cyan — rPPG > GT
TEXT_DELTA_NEG     = (100, 100, 255)  # red  — rPPG < GT
TEXT_WARN          = (60,  60,  220)  # out-of-range BPM

BAR_MAX_PX   = 200   # max bar width in pixels
BAR_HEIGHT   = 12    # bar height
BAR_OUTLINE  = (60,  60,  80)

# Algorithm display order and colours (BGR)
ALGO_ORDER = ("CHROM", "POS", "GREEN", "ICA", "WAVELET", "CONSENSUS")

ALGO_COLORS = {
    "CHROM":     (255, 180,  60),   # orange
    "POS":       (100, 220, 255),   # cyan
    "GREEN":     ( 80, 200,  80),   # green
    "ICA":       (200, 130, 255),   # violet
    "WAVELET":   (255, 255, 100),   # yellow
    "CONSENSUS": (255, 255, 255),   # white
}

BPM_LOW  = 50   # below this → warn colour
BPM_HIGH = 150  # above this → warn colour
BPM_MIN_DISPLAY = 40   # bar scaling floor
BPM_MAX_DISPLAY = 180  # bar scaling ceiling
