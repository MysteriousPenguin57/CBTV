"""Detection pre-filter & dedup helpers (after peaks, before Bayes)."""

import numpy as np

from .detection_filter import filter_detections_multistage
from .edge_fit import refine_edge_fit

EDGE_FIT_AVAILABLE = True


def radius_based_adaptive_dedup_2c(dets_list, gray, chans, hard, soft, radius_mult=1.0, stats=None):
    """
    Radius-based adaptive deduplication with filter-tested weighted merging.

    For each cluster:
    - Threshold = average radius × radius_mult
    - Try weighted centroid
    - If it passes filter: use weighted
    - If it fails: use keep-best
    """
    if not dets_list:
        return []

    from scipy.spatial import cKDTree

    coords = np.array([[x, y] for x, y, r, s in dets_list])
    scores = np.array([s for x, y, r, s in dets_list])
    radii = np.array([r for x, y, r, s in dets_list])

    if len(coords) == 1:
        return dets_list

    # Search for all pairs within the maximum possible threshold
    max_threshold = np.max(radii) * radius_mult
    tree = cKDTree(coords)

    taken = np.zeros(len(coords), dtype=bool)
    result = []

    # Sort indices by score descending, to process highest confident ones first
    sorted_indices = np.argsort(-scores, kind="stable")

    for i in sorted_indices:
        if taken[i]:
            continue

        cluster_indices = [i]
        taken[i] = True

        # Find nearby detections using max threshold first
        neighbors = tree.query_ball_point(coords[i], max_threshold)

        for j in neighbors:
            if taken[j]:
                continue

            dist = np.linalg.norm(coords[i] - coords[j])
            avg_radius = (radii[i] + radii[j]) / 2.0
            threshold = avg_radius * radius_mult

            if dist <= threshold:
                cluster_indices.append(j)
                taken[j] = True

        if len(cluster_indices) == 1:
            # Single detection - keep as is
            idx = cluster_indices[0]
            result.append((coords[idx, 0], coords[idx, 1], radii[idx], scores[idx]))
        else:
            # Multiple detections - try weighted centroid
            cluster_coords = coords[cluster_indices]
            cluster_scores = scores[cluster_indices]
            cluster_radii = radii[cluster_indices]

            total_score = np.sum(cluster_scores)
            weights = cluster_scores / total_score
            x_w = np.sum(cluster_coords[:, 0] * weights)
            y_w = np.sum(cluster_coords[:, 1] * weights)
            r_w = np.mean(cluster_radii)
            s_w = np.max(cluster_scores)

            # Test weighted centroid through actual filter
            test_det = [(x_w, y_w, r_w, s_w)]
            kept_test, _, _ = filter_detections_multistage(gray, test_det,
                                                            hard=hard, soft=soft,
                                                            channels_dict=chans,
                                                            stats=stats)

            if len(kept_test) > 0:
                # Weighted passes - use it
                result.append((float(x_w), float(y_w), float(r_w), float(s_w)))
            else:
                # Weighted fails - use keep-best
                best_local_idx = np.argmax(cluster_scores)
                best_idx = cluster_indices[best_local_idx]
                result.append((float(coords[best_idx, 0]), float(coords[best_idx, 1]),
                              float(radii[best_idx]), float(scores[best_idx])))

    return result

def apply_edge_fit_refinement_2c(detections, gray):
    """Apply Edge-Fit refinement to all detections"""
    if not EDGE_FIT_AVAILABLE:
        return detections

    refined = []
    for x, y, r, s in detections:
        try:
            x_ref, y_ref, r_ref = refine_edge_fit(gray, x, y, r)
            refined.append((x_ref, y_ref, r_ref, s))
        except:  # noqa: E722 -- bare on purpose: also swallows KeyboardInterrupt
            refined.append((x, y, r, s))
    return refined

def get_relaxed_filter_params(relaxation_factor=0.6):
    """Get relaxed filter parameters for Tracker 2C"""
    hard = dict(
        edge_distance=3,
        max_radius_factor=1.65,
        min_local_variance=47.0 * relaxation_factor,
        min_gradient_strength=4.75 * relaxation_factor,
        min_circularity=0.50 * relaxation_factor,
        intensity_uniform_std=14.0,
        intensity_extreme_low=8,
        intensity_extreme_high=247
    )

    soft = dict(
        max_radius_factor=1.55,
        min_local_variance=80.0 * relaxation_factor,
        min_gradient_strength=5.6 * relaxation_factor,
        min_circularity=0.70 * relaxation_factor
    )

    return hard, soft

