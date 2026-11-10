"""Edge-fit sub-pixel circle refinement (refine_edge_fit + fit_circle_taubin).

Copied verbatim from the research code that the published runs imported via a
sys.path insert in detect_utils.py. An earlier standalone copy did not ship that
file, so the guarded import silently disabled edge-fit refinement and changed
every result.
"""
from typing import Tuple

import cv2
import numpy as np


def refine_edge_fit(gray: np.ndarray, x0: float, y0: float,
                    radius: float, annulus_width: float = 3) -> Tuple[float, float, float]:
    """
    Edge detection + circle fitting for large error correction.
    Returns refined (x, y, radius).
    """
    # Extract ROI
    margin = int(radius + 10)
    x_int, y_int = int(round(x0)), int(round(y0))

    x_min = max(0, x_int - margin)
    x_max = min(gray.shape[1], x_int + margin)
    y_min = max(0, y_int - margin)
    y_max = min(gray.shape[0], y_int + margin)

    if x_max <= x_min or y_max <= y_min:
        return x0, y0, radius

    roi = gray[y_min:y_max, x_min:x_max]

    # Adaptive Canny thresholds
    median_val = np.median(roi)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))

    edges = cv2.Canny(roi.astype(np.uint8), lower, upper)

    # Get edge points
    edge_y, edge_x = np.where(edges > 0)

    if len(edge_x) < 10:
        return x0, y0, radius

    # Convert to global coordinates
    edge_points = np.column_stack([edge_x + x_min, edge_y + y_min])

    # Filter points near expected radius
    center = np.array([x0, y0])
    distances = np.linalg.norm(edge_points - center, axis=1)
    mask = np.abs(distances - radius) < annulus_width

    ring_points = edge_points[mask]

    if len(ring_points) < 10:
        return x0, y0, radius

    # Taubin circle fit
    try:
        cx, cy, r_fit = fit_circle_taubin(ring_points)

        # Sanity check
        if abs(cx - x0) < 10 and abs(cy - y0) < 10 and abs(r_fit - radius) < 5:
            return cx, cy, r_fit
    except:  # noqa: E722 -- bare on purpose: also swallows KeyboardInterrupt
        pass

    return x0, y0, radius


def fit_circle_taubin(points: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic circle fitting using Taubin's method."""
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)

    # Center data
    mx, my = x.mean(), y.mean()
    x = x - mx
    y = y - my

    # Build design matrix
    Z = np.column_stack([x**2 + y**2, x, y, np.ones(len(x))])

    # SVD
    _, _, V = np.linalg.svd(Z)
    A = V[-1]

    # Extract circle parameters
    if abs(A[0]) < 1e-10:
        return mx, my, np.sqrt(x**2 + y**2).mean()

    a = -A[1] / (2 * A[0])
    b = -A[2] / (2 * A[0])
    r = np.sqrt((A[1]**2 + A[2]**2 - 4*A[0]*A[3]) / (4*A[0]**2))

    return mx + a, my + b, r


