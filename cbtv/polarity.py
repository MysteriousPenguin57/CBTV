"""Input-polarity normalisation: make bright-background video look like CBTV's
native dark-background domain.

Why this exists
---------------
Every stage of CBTV's detection path assumes BRIGHT particles on a DARK
background:

* ``cbtv.detect.channels.subtract_background`` estimates the per-channel
  histogram mode and subtracts it. On a cream tray the modal BGR is about
  (178, 186, 186); subtracting it from a red bead at (40, 40, 190) leaves
  (0, 0, 12) - the bead's colour is destroyed, not its background.
* ``build_channels`` equalises Gray / V / L and the watershed strategies run
  Otsu on them. With a bright background those channels vote for the tray.

Fed a bright-background video raw, the detector returns ~0 particles. This
module maps such a frame into the expected polarity while preserving the one
thing CBTV's constellations depend on - each particle's colour.

How it works
------------
Particles are separated from the background in **chromaticity**
(channel / (B+G+R)), not in luminance. Chromaticity is invariant to
multiplicative shading, which is the property that makes this robust: the
tray reads the same whether it is directly lit or in the shadow cast between
two touching beads, so the gaps inside a dense pile score as background. A
luminance-based matte cannot do that - it reports every shadowed gap, and
every shadowed corner of the tray, as particles.

    evidence = |chromaticity(pixel) - chromaticity(background)|
    alpha    = smoothstep(evidence, t0, t1)     (t0/t1 from Otsu)
    output   = original_BGR * alpha

The output keeps each particle's own colour and shading and sends the
background to black, so ``subtract_background`` then finds a modal value
below its skip threshold and passes the frame through untouched.

A morphological closing is applied to alpha before compositing. Glossy
particles carry a bright, near-neutral specular highlight that scores as
background and would otherwise punch a hole through the particle centre -
exactly where the detector needs signal.

Scope
-----
Default is OFF (``--input-polarity dark``), and when off this module is never
imported, so the legacy path is byte-identical. ``auto`` inspects the first
frame and enables normalisation only for bright-background video.
"""
from __future__ import annotations

import cv2
import numpy as np

# The matte ramps from the background shoulder (t0) to RAMP_MULT * t0. The
# ramp is deliberately wide: a hard threshold quantises particle edges and
# costs the detector sub-pixel accuracy.
RAMP_MULT = 3.0

# A frame whose modal HSV value exceeds this is treated as bright-background.
BRIGHT_MODE_THRESHOLD = 127


def _disc(k):
    k = max(3, int(k) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def detect_polarity(frame_bgr):
    """Return 'bright' or 'dark' for a single BGR frame.

    Uses the modal HSV value: the background is by definition the most
    common brightness in a particle-tracking frame. CBTV's own corpus sits
    at mode 0-38 (rendered on black); a lit tray or lightbox sits above 150.
    """
    v = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    mode = int(np.argmax(np.bincount(v.ravel(), minlength=256)))
    return "bright" if mode > BRIGHT_MODE_THRESHOLD else "dark"


def chromaticity(bgr):
    """Shading-invariant (r, g) chromaticity: channel / (B + G + R)."""
    f = bgr.astype(np.float32)
    s = f.sum(2) + 1e-3
    return f[:, :, 2] / s, f[:, :, 1] / s


class PolarityNormaliser:
    """Fits the background chromaticity once, then normalises every frame.

    The fit is deliberately NOT redone per frame. The tray and the lighting
    do not change during a run, and a per-frame refit would make the matte
    breathe - injecting appearance jitter into a tracker whose cost matrix
    has a colour term.
    """

    def __init__(self, r_max=10, verbose=True):
        self.bg_rg = None
        self.t0 = None
        self.t1 = None
        # The highlight scales with the particle, so the hole-filling
        # structuring element does too.
        self.fill_k = max(3, int(round(0.8 * max(1, r_max))))
        self.verbose = verbose

    # -- fitting -------------------------------------------------------
    def fit(self, frames):
        """Estimate background chromaticity and the matte thresholds."""
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
        rs, gs = [], []
        for f in frames:
            r, g = chromaticity(f)
            # Seed on near-achromatic pixels: a tray/lightbox background is
            # neutral, particles are not. Fall back to the whole frame if
            # the seed is too small to be meaningful.
            seed = (np.abs(r - 1 / 3) < 0.03) & (np.abs(g - 1 / 3) < 0.03)
            if seed.sum() < 1000:
                seed = np.ones_like(r, dtype=bool)
            rs.append(np.median(r[seed]))
            gs.append(np.median(g[seed]))
        self.bg_rg = (float(np.mean(rs)), float(np.mean(gs)))

        # Threshold from the data, via the TRIANGLE method rather than Otsu.
        # Otsu assumes two classes; the evidence image has one background
        # peak and SEVERAL particle modes (one per dye - measured here at
        # 0.052 / 0.089 / 0.235). Otsu splits between particle populations,
        # keeping only the most-saturated dye and discarding the rest.
        # Triangle finds the shoulder of the dominant background peak, which
        # is the boundary actually wanted, whatever the particle colours are.
        # Taken as the median over the sampled frames so one atypical frame
        # (a hand entering shot, an exposure step) cannot set the threshold
        # for the whole run.
        splits = []
        for f in frames:
            ev = self.evidence(f)
            hi = float(np.percentile(ev, 99.5))
            q = np.clip(ev / max(1e-6, hi) * 255.0, 0, 255).astype(np.uint8)
            thr, _ = cv2.threshold(q, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            splits.append(float(thr) / 255.0 * hi)
        self.t0 = float(np.median(splits))
        self.t1 = self.t0 * RAMP_MULT
        if self.verbose:
            print(f"[polarity] background chromaticity "
                  f"r={self.bg_rg[0]:.4f} g={self.bg_rg[1]:.4f}; "
                  f"matte ramp {self.t0:.4f}..{self.t1:.4f}; "
                  f"fill kernel {self.fill_k}px")
        return self

    # -- application ---------------------------------------------------
    def evidence(self, bgr):
        """Per-pixel distance from the background chromaticity."""
        r, g = chromaticity(bgr)
        return np.hypot(r - self.bg_rg[0], g - self.bg_rg[1])

    def alpha(self, bgr):
        ev = self.evidence(bgr)
        a = np.clip((ev - self.t0) / max(1e-6, self.t1 - self.t0), 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)                 # smoothstep
        return cv2.morphologyEx(a, cv2.MORPH_CLOSE, _disc(self.fill_k))

    def apply(self, bgr):
        """Return the frame with the background matted to black."""
        a = self.alpha(bgr)
        out = bgr.astype(np.float32) * a[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)


def make_normaliser(cap, mode, r_max=10, n_fit=5, verbose=True):
    """Build a normaliser for a video, or None when no work is needed.

    `cap` is an already-open cv2.VideoCapture; its position is restored
    before returning, so the caller's read loop is unaffected.

    mode: 'dark'   -> always None (legacy path, byte-identical)
          'bright' -> always build one
          'auto'   -> build one only if the first frame looks bright-backed
    """
    if mode == "dark":
        return None

    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in np.linspace(0, max(0, total - 1), n_fit).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    if not frames:
        return None
    if mode == "auto":
        pol = detect_polarity(frames[0])
        if verbose:
            print(f"[polarity] auto-detected {pol}-background input")
        if pol == "dark":
            return None
    return PolarityNormaliser(r_max=r_max, verbose=verbose).fit(frames)
