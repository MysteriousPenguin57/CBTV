"""PIV — anchor-free deformation field (Phase 5 Stage 3.1, Arm P).

Gate 0 showed the COMMIT-anchored local affine is geometrically dead on every
available video at the displacement band (fall-throughs are spatially isolated
from the committed set: PRIMARY ~20%, K05 ~6%, K01 ~4.5% field-constrainable).
PIV routes around that: it estimates a dense motion field directly from the raw
pixels of frame n-1 -> n by image cross-correlation, so it predicts a
displacement mu (+ uncertainty Sigma) for EVERY particle with no particle
anchors at all — including the ~80% of fall-throughs the commit-affine cannot
reach.

The module is a pluggable mu-source: ``sample_piv_affine_at_particle`` returns
the SAME ``(a, B, alpha)`` triple that ``field.fit_local_affine`` produces, so
``field.predict_position`` and the kernel deform hook consume it unchanged. As
the local correlation peak flattens (low texture / shear bands) Sigma grows,
alpha -> 0, and the prediction gracefully degrades to identity (no recentring) —
i.e. PIV never does worse than the appearance-only path where motion is
uninformative.

Pure numpy/scipy/cv2; no module-level globals; CPU (FFT/matchTemplate). The
orchestration (which frames, which particles) stays in run_tracker.
"""
from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Dense PIV grid + O(1) bilinear sampling
# ---------------------------------------------------------------------------

class PIVGridResult:
    """A regular grid of per-window displacement (u, v) and uncertainty sigma.

    Grid point (gj, gi) sits at image (x0 + gj*dx, y0 + gi*dy). ``sample_at``
    bilinearly interpolates and clamps queries to the grid bounds.
    """

    def __init__(self, u_grid, v_grid, sigma_grid, x0, y0, dx, dy):
        self.u = np.asarray(u_grid, dtype=np.float32)
        self.v = np.asarray(v_grid, dtype=np.float32)
        self.sigma = np.asarray(sigma_grid, dtype=np.float32)
        self.x0 = float(x0); self.y0 = float(y0)
        self.dx = float(dx); self.dy = float(dy)
        self.gy, self.gx = self.u.shape

    def sample_at(self, x: float, y: float) -> tuple[float, float, float]:
        if self.gx == 0 or self.gy == 0:
            return 0.0, 0.0, float("inf")
        fx = (x - self.x0) / self.dx if self.dx else 0.0
        fy = (y - self.y0) / self.dy if self.dy else 0.0
        fx = min(max(fx, 0.0), self.gx - 1.0)
        fy = min(max(fy, 0.0), self.gy - 1.0)
        j0 = int(math.floor(fx)); i0 = int(math.floor(fy))
        j1 = min(j0 + 1, self.gx - 1); i1 = min(i0 + 1, self.gy - 1)
        tx = fx - j0; ty = fy - i0

        def bilerp(g):
            a = g[i0, j0] * (1 - tx) + g[i0, j1] * tx
            b = g[i1, j0] * (1 - tx) + g[i1, j1] * tx
            return float(a * (1 - ty) + b * ty)

        return bilerp(self.u), bilerp(self.v), bilerp(self.sigma)


# ---------------------------------------------------------------------------
# Sub-pixel peak + sharpness -> Sigma
# ---------------------------------------------------------------------------

def _subpixel_1d(cm1: float, c0: float, cp1: float) -> float:
    """3-point Gaussian peak offset in [-0.5, 0.5] (standard PIV).

    Falls back to a parabola when values are non-positive (Gaussian needs
    positive samples). Returns 0.0 on a degenerate / flat triple.
    """
    if cm1 > 0 and c0 > 0 and cp1 > 0:
        denom = (math.log(cm1) - 2.0 * math.log(c0) + math.log(cp1))
        if abs(denom) > 1e-12:
            off = 0.5 * (math.log(cm1) - math.log(cp1)) / denom
            return float(min(max(off, -1.0), 1.0))
    denom = (cm1 - 2.0 * c0 + cp1)
    if abs(denom) > 1e-12:
        off = 0.5 * (cm1 - cp1) / denom
        return float(min(max(off, -1.0), 1.0))
    return 0.0


def peak_sharpness_to_sigma(corr_2d: np.ndarray, peak_y: int | np.integer, peak_x: int | np.integer,
                            *, base_sigma: float = 2.0, max_sigma: float = 8.0,
                            disc: int = 3) -> float:
    """Map correlation-peak confidence to a displacement uncertainty sigma.

    Confident match (high, UNAMBIGUOUS normalised peak) -> sigma near
    base_sigma; weak/flat (low texture) or ambiguous (periodic lattice, shear
    band — a near-equal second peak) -> sigma near max_sigma. CCOEFF_NORMED is
    already in [-1, 1], so the peak HEIGHT is the primary confidence signal
    (more robust than curvature on smooth texture); a second-peak ratio adds
    the ambiguity penalty that matters on quasi-lattice regions.

    disc (--piv-ambiguity-disc, px): half-width of the square around the main
    peak excluded from the second-peak search. It is in pixels, so it must
    scale with the image: the peak's width grows with the particle size, and a
    disc narrower than the peak lets its own flank count as a second peak.
    Default 3 = the original fixed value.
    """
    peak = float(corr_2d[peak_y, peak_x])
    if not np.isfinite(peak) or peak < 0.2:
        return float(max_sigma)
    conf = min(max((peak - 0.3) / (0.85 - 0.3), 0.0), 1.0)   # height -> [0,1]
    # Ambiguity: best correlation OUTSIDE a small disc around the main peak.
    h, w = corr_2d.shape
    cc = corr_2d.copy()
    y0, y1 = max(0, peak_y - disc), min(h, peak_y + disc + 1)
    x0, x1 = max(0, peak_x - disc), min(w, peak_x + disc + 1)
    cc[y0:y1, x0:x1] = -1.0
    second = float(cc.max())
    if second > 0.0:
        amb = second / max(peak, 1e-6)                        # 1 -> fully ambiguous
        conf *= min(max((0.9 - amb) / (0.9 - 0.5), 0.0), 1.0)
    return float(max_sigma - (max_sigma - base_sigma) * conf)


# ---------------------------------------------------------------------------
# Bulk (whole-frame) shift estimate — Phase 6g Stage 3.2
# ---------------------------------------------------------------------------

def estimate_bulk_shift(gray_prev: np.ndarray, gray_curr: np.ndarray,
                        min_response: float = 0.03) -> tuple[float, float]:
    """Coarse whole-frame translation n-1 -> n by Hann-windowed phase
    correlation. Returns (u, v) in px; (0, 0) when the correlation response is
    too weak to trust (graceful no-op). Used by --piv-bulk-pass to re-centre
    the PIV interrogation windows so a fast-but-coherent bed stays within the
    per-window search margin."""
    import cv2

    p = gray_prev.astype(np.float32)
    c = gray_curr.astype(np.float32)
    hann = cv2.createHanningWindow((p.shape[1], p.shape[0]), cv2.CV_32F)
    try:
        (du, dv), response = cv2.phaseCorrelate(p, c, hann)
    except cv2.error:
        return 0.0, 0.0
    if not np.isfinite(du) or not np.isfinite(dv) or response < min_response:
        return 0.0, 0.0
    return float(du), float(dv)


# ---------------------------------------------------------------------------
# Dense PIV field by windowed normalised cross-correlation
# ---------------------------------------------------------------------------

def compute_dense_piv_field(gray_prev: np.ndarray, gray_curr: np.ndarray,
                            d_px: float, args, use_gpu: bool = False,
                            bulk: tuple[float, float] | None = None) -> PIVGridResult:
    """Per-interrogation-window forward displacement (n-1 -> n) + sigma.

    For each grid window: template = frame n-1 window; search = frame n region
    (template + 2*search_margin); normalised cross-correlation (cv2 CCOEFF);
    sub-pixel peak -> (u, v); peak sharpness -> sigma. All spatial scales are
    in units of the particle diameter d_px (Fix 0), so the field transfers
    across videos with different px/diameter scales.

    bulk=(u0, v0) [Phase 6g]: re-centre every search window in frame n by the
    (rounded) bulk shift, clamped to the frame; the applied offset is added
    back into the returned (u, v) so the field stays an ABSOLUTE n-1 -> n
    displacement. bulk=None is byte-identical to the pre-6g path.
    """
    import cv2

    win_d = float(getattr(args, "piv_window_d_units", 5.0))
    overlap = float(getattr(args, "piv_step_overlap", 0.5))
    base_sigma = float(getattr(args, "piv_base_sigma", 2.0))
    max_sigma = float(getattr(args, "piv_max_sigma", 8.0))
    disc = int(getattr(args, "piv_ambiguity_disc", 3))

    H, W = gray_prev.shape[:2]
    win = max(8, int(round(win_d * d_px)))
    win += win % 2                                   # even
    h = win // 2
    s = max(2, h // 1)                               # search margin ~ window/2 each side -> 2x window
    stride = max(1, int(round(win * (1.0 - overlap))))

    if gray_prev.dtype != np.uint8:
        gray_prev = np.clip(gray_prev, 0, 255).astype(np.uint8)
    if gray_curr.dtype != np.uint8:
        gray_curr = np.clip(gray_curr, 0, 255).astype(np.uint8)

    bu = bv = 0
    if bulk is not None:
        bu, bv = int(round(bulk[0])), int(round(bulk[1]))

    # Grid centres where the full search region fits inside the frame.
    lo = h + s
    xs = list(range(lo, W - lo, stride)) or [W // 2]
    ys = list(range(lo, H - lo, stride)) or [H // 2]
    gx, gy = len(xs), len(ys)
    u_g = np.zeros((gy, gx), np.float32)
    v_g = np.zeros((gy, gx), np.float32)
    sig_g = np.full((gy, gx), max_sigma, np.float32)

    side = win + 2 * s                               # search-region side length
    for gi, cy in enumerate(ys):
        for gj, cx in enumerate(xs):
            templ = gray_prev[cy - h:cy + h, cx - h:cx + h]
            if bulk is None:
                ry0, rx0 = cy - h - s, cx - h - s
            else:
                # shift the search region by the bulk, clamped inside the frame;
                # the offset ACTUALLY applied is what we add back below.
                ry0 = min(max(cy - h - s + bv, 0), H - side)
                rx0 = min(max(cx - h - s + bu, 0), W - side)
            off_x = rx0 - (cx - h - s)
            off_y = ry0 - (cy - h - s)
            search = gray_curr[ry0:ry0 + side, rx0:rx0 + side]
            try:
                corr = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
            # Sub-pixel refine on the integer peak (3-point Gaussian).
            sub_x = sub_y = 0.0
            if 0 < px < corr.shape[1] - 1:
                sub_x = _subpixel_1d(float(corr[py, px - 1]), float(corr[py, px]),
                                     float(corr[py, px + 1]))
            if 0 < py < corr.shape[0] - 1:
                sub_y = _subpixel_1d(float(corr[py - 1, px]), float(corr[py, px]),
                                     float(corr[py + 1, px]))
            # Zero displacement sits at (s, s) in the corr map (+ applied offset).
            u_g[gi, gj] = (px + sub_x) - s + off_x
            v_g[gi, gj] = (py + sub_y) - s + off_y
            sig_g[gi, gj] = peak_sharpness_to_sigma(corr, py, px,
                                                    base_sigma=base_sigma, max_sigma=max_sigma,
                                                    disc=disc)

    x0 = xs[0]; y0 = ys[0]
    dx = (xs[1] - xs[0]) if gx > 1 else 1.0
    dy = (ys[1] - ys[0]) if gy > 1 else 1.0
    return PIVGridResult(u_g, v_g, sig_g, x0, y0, dx, dy)


# ---------------------------------------------------------------------------
# Per-particle (a, B, alpha) — the field.py contract
# ---------------------------------------------------------------------------

def sample_piv_affine_at_particle(piv: PIVGridResult, x: float, y: float, d_px: float,
                                  *, base_sigma: float = 2.0, max_sigma: float = 8.0):
    """Local affine of the PIV field at (x, y), matching field.fit_local_affine.

    Returns ``(a, B, alpha)``:
      a : (2,) float64   translation = PIV displacement (u, v) at (x, y).
      B : (2, 2) float64 finite-difference Jacobian of (u, v) over +/- d/2.
      alpha : float in [0, 1] — confidence from the local sigma; sharp peak
              -> high alpha (strong recentring), flat/shear -> alpha 0
              (predict_position collapses to identity).
    Consumed verbatim by field.predict_position and the kernel deform hook.
    """
    u, v, sigma_c = piv.sample_at(x, y)
    a = np.array([u, v], dtype=np.float64)

    hstep = max(0.5 * d_px, 1.0)
    uxp, vxp, _ = piv.sample_at(x + hstep, y)
    uxm, vxm, _ = piv.sample_at(x - hstep, y)
    uyp, vyp, _ = piv.sample_at(x, y + hstep)
    uym, vym, _ = piv.sample_at(x, y - hstep)
    inv = 1.0 / (2.0 * hstep)
    B = np.array([[(uxp - uxm) * inv, (uyp - uym) * inv],
                  [(vxp - vxm) * inv, (vyp - vym) * inv]], dtype=np.float64)

    denom = max(max_sigma - base_sigma, 1e-9)
    alpha = float(min(max((max_sigma - sigma_c) / denom, 0.0), 1.0))
    return a, B, alpha
