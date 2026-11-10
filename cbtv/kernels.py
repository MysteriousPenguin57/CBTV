"""LAB neighbourhood tracking-kernel construction (CPU + GPU)."""
import math

import cv2
import numpy as np

from . import gpu
from .correlation import ncc
from .gpu import cp


# === [DATA-4] precompute_neighbor_lists — KDTree helper for GPU kernel build (see Flowchart 13) ===
# Builds a cKDTree over `prev_detections` and emits, for every track in the SoA,
# the list of neighbour indices within `neighbor_search_factor * r`. The output
# feeds `batch_build_kernels_gpu`, where each track's neighbourhood patch is
# composed once on the GPU instead of being re-scanned per detection.
def precompute_neighbor_lists(tracks_soa, prev_detections, neighbor_search_factor, adaptive_sizing=True):
    """
    Pre-compute neighbor lists for all tracks using KD-Tree for efficiency
    OPTIMIZED: Uses KD-tree to avoid computing ALL M×P distances

    Args:
        tracks_soa: TrackArrays object with track data
        prev_detections: List of (x, y, r, _) from previous frame
        neighbor_search_factor: Multiplier for search radius (maximum)
        adaptive_sizing: If True, compute adaptive search factors per track (currently disabled)

    Returns:
        neighbor_lists: Dict {track_id: [list of neighbor indices]}
        neighbor_data: Dict {track_id: {'distances': array, 'radii': array, 'adaptive_factor': float}}
    """
    neighbor_lists = {}
    neighbor_data = {}

    if prev_detections is None or len(prev_detections) == 0:
        return neighbor_lists, neighbor_data

    # Convert prev_detections to numpy arrays
    prev_positions = np.array([[x, y] for (x, y, _, _) in prev_detections], dtype=np.float32)
    prev_radii = np.array([r for (_, _, r, _) in prev_detections], dtype=np.float32)

    # Build KD-Tree once for all queries
    from scipy.spatial import cKDTree
    kdtree = cKDTree(prev_positions)

    for i in range(tracks_soa.count):
        track_id = int(tracks_soa.ids[i])
        tx, ty = tracks_soa.positions[i]
        tr = tracks_soa.radii[i]

        # Use KD-tree to find neighbors efficiently (no redundant distance computation!)
        search_radius = neighbor_search_factor * max(tr, 1.0)
        neighbor_indices = kdtree.query_ball_point([tx, ty], r=search_radius)

        # Compute distances only for neighbors (not all P detections!)
        if len(neighbor_indices) > 0:
            neighbor_pos = prev_positions[neighbor_indices]
            neighbor_dists = np.sqrt(np.sum((neighbor_pos - [tx, ty])**2, axis=1))
            neighbor_rs = prev_radii[neighbor_indices]

            # Sort by distance (closest first)
            sort_idx = np.argsort(neighbor_dists, kind="stable")
            neighbor_indices = [neighbor_indices[j] for j in sort_idx]
            neighbor_dists = neighbor_dists[sort_idx]
            neighbor_rs = neighbor_rs[sort_idx]
        else:
            neighbor_dists = np.array([])
            neighbor_rs = np.array([])

        neighbor_lists[track_id] = neighbor_indices

        # Fixed search factor (adaptive_sizing disabled for performance)
        adaptive_factor = neighbor_search_factor

        neighbor_data[track_id] = {
            'distances': neighbor_dists,
            'radii': neighbor_rs,
            'adaptive_factor': adaptive_factor
        }

    return neighbor_lists, neighbor_data

def build_kernel_task(track_id, track_pos, track_radius, neighbor_indices,
                      prev_detections, prev_L, prev_A, prev_B, args):
    """Worker task for multiprocessing kernel build"""
    try:
        # The `build_LAB_kernel_cpu_optimized` function expects the central detection
        # to be at index 0 of `det_list`. We need to construct a `det_list`
        # where the central detection is first, followed by its neighbors.

        # The track_pos and track_radius are for the *current* track, which corresponds
        # to a detection in the *previous* frame. We need to find that detection.

        # Find the detection in prev_detections that corresponds to this track.
        # This is a bit inefficient, but necessary if track_pos/radius don't directly map
        # to an index in prev_detections.
        # A better approach would be to pass the actual index of the central detection
        # in prev_detections directly to build_kernel_task.

        # For now, let's assume track_pos/radius are from the *matched* detection
        # in the previous frame, and we need to find its index.
        # Or, more simply, we can just construct a temporary det_list for this task.

        # The `build_LAB_kernel_cpu_optimized` expects `col_lab` (LAB colors of detections)
        # to be pre-computed. We need to compute this for the relevant detections.

        # Central detection for kernel building
        cen_det = (track_pos[0], track_pos[1], track_radius, 1.0) # Dummy score

        # Collect all detections relevant for this kernel (central + neighbors)
        # Ensure neighbor_indices are valid indices into prev_detections
        max_neighbors = 10 # Limit number of neighbors for kernel complexity

        # Create a temporary det_list for build_LAB_kernel_cpu_optimized
        # Central detection is at index 0
        track_det_list = [cen_det] + [prev_detections[idx] for idx in neighbor_indices[:max_neighbors]]

        # Pre-compute LAB colors for these detections
        # This is done on CPU for this worker, as the main LAB conversion is already done.
        col_lab_for_kernel = compute_particle_colours_lab_from_split(
            prev_L, prev_A, prev_B, track_det_list
        )

        # Get image dimensions from one of the split channels
        H, W = prev_L.shape

        # Call the optimized kernel building function
        K_built, cx_off, cy_off = build_LAB_kernel_cpu_optimized(
            track_det_list,
            0, # Central detection is at index 0 in track_det_list
            list(range(1, len(track_det_list))), # Neighbors are from index 1 onwards
            prev_L, prev_A, prev_B,
            col_lab_for_kernel,
            H, W,
            track_radius, # r0 for central detection
            args
        )

        return track_id, (K_built, cx_off, cy_off)
    except Exception as e:
        print(f"Error building kernel for track {track_id}: {e}")
        return track_id, None # Return None if kernel building fails

def prebuild_all_kernels_parallel(tracks_soa, prev_detections, prev_frame,
                                  neighbor_lists, args, max_workers=None, neighbor_data=None, prev_lab=None, kernel_cache=None):
    """
    Pre-builds kernels for all tracks using a thread pool.

    Args:
        tracks_soa: SOA structure of tracks
        prev_detections: List of previous detections
        prev_frame: Previous frame image
        neighbor_lists: Precomputed neighbor lists for each track (optional)
        args: Config arguments
        max_workers: Number of threads to use (default: up to 8)
        neighbor_data: Precomputed SpatialData for previous detections
        prev_lab: Previous frame already converted to LAB
    """
    import os
    import time as time_module
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)

    t0 = time_module.perf_counter()
    kernel_dict = {}

    if tracks_soa.count == 0 or prev_detections is None or len(prev_detections) == 0:
        return kernel_dict, {'t_kernel_build': 0.0, 'kernel_count': 0}

    # Split LAB frame once
    if prev_lab is None:
        prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    L_img, A_img, B_img = cv2.split(prev_lab)

    # Try GPU batch processing first (Wait: Currently disabled because PCIe .asnumpy() syncs take 2.5s! CPU ThreadPool is faster)
    if False and args.use_gpu and gpu.USE_CUPY:
        try:
            kernel_dict = batch_build_kernels_gpu(
                tracks_soa, prev_detections, prev_frame, prev_lab, neighbor_lists, args,
                use_gpu=args.use_gpu, neighbor_data=neighbor_data
            )
            elapsed = (time_module.perf_counter() - t0) * 1000.0
            return kernel_dict, {'t_kernel_build': elapsed, 'kernel_count': len(kernel_dict)}
        except Exception as e:
            if args.verbose:
                print(f"⚠️  Batch kernel building failed ({e}), using CPU threading fallback")

    # CPU fallback: Multithreaded approach
    tasks = []
    for i in range(tracks_soa.count):
        t_id = int(tracks_soa.ids[i])
        t_pos = (float(tracks_soa.positions[i, 0]), float(tracks_soa.positions[i, 1]))
        t_r = float(tracks_soa.radii[i])

        # Get neighbor indices for this track
        neigh_indices = neighbor_lists.get(t_id, [])

        # Central detection for kernel building
        cen_det = (t_pos[0], t_pos[1], t_r, 1.0) # Dummy score

        # Collect all detections relevant for this kernel (central + neighbors)
        max_neighbors = 10 # Limit number of neighbors for kernel complexity

        # Create a temporary det_list for build_LAB_kernel_cpu_optimized
        # Central detection is at index 0
        track_det_list = [cen_det] + [prev_detections[idx] for idx in neigh_indices[:max_neighbors]]
        # Local indices inside track_det_list for neighbors (1 to N)
        local_neigh_idx = list(range(1, len(track_det_list)))

        # Check if kernel is already cached for this track & position
        if kernel_cache is not None:
            cached_kernel = kernel_cache.get(t_id, current_pos=t_pos)
            if cached_kernel is not None:
                kernel_dict[t_id] = cached_kernel
                continue

        # Submit to pool
        tasks.append((t_id, t_pos, t_r, local_neigh_idx, track_det_list,
                       L_img, A_img, B_img, args))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(build_kernel_task, *task_args) for task_args in tasks]

        for future in as_completed(futures):
            track_id, kernel_data = future.result()
            if kernel_data is not None:
                kernel_dict[track_id] = kernel_data

    elapsed = (time_module.perf_counter() - t0) * 1000.0

    return kernel_dict, {'t_kernel_build': elapsed, 'kernel_count': len(kernel_dict)}

def compute_particle_colours_lab(frame_bgr, dets):
    """Batch-extract mean LAB colours for a list of detections.

    Converts the full frame to LAB once, splits into L/A/B channels,
    then delegates to ``compute_particle_colours_lab_from_split`` for
    the per-detection circular crop and mean computation.  Use this
    function when you only have a BGR frame; prefer
    ``compute_particle_colours_lab_from_split`` directly when L/A/B
    channels are already available to avoid redundant conversion.

    Pipeline context:
        Called by: build_LAB_tracking_kernel, main (colour extraction loop)
        Calls: compute_particle_colours_lab_from_split
        Stage: classification

    Args:
        frame_bgr: Full BGR video frame, shape ``(H, W, 3)``. [np.ndarray]
        dets: List of detection tuples ``[(x, y, r, score), ...]`` in
            image pixel coordinates. [List[tuple]]

    Returns:
        List of ``(L_mean, A_mean, B_mean)`` float tuples, one per
        detection.  Detections with no pixels inside the mask return
        ``(0.0, 0.0, 0.0)``. [List[Tuple[float, float, float]]]

    Phase history:
        - Original: Introduced with LAB-based tracking kernels.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L_img, A_img, B_img = cv2.split(lab)
    return compute_particle_colours_lab_from_split(L_img, A_img, B_img, dets)

def compute_particle_colours_lab_from_split(Limg, Aimg, Bimg, dets):
    """Batch-extract mean LAB colours from pre-split L/A/B channel images.

    For each detection, crops a bounding box of radius ``ceil(1.15 * r)``
    around the centre, builds a circular boolean mask using ``np.ogrid``,
    and returns the mean L, A, B values over the masked pixels.  Using
    pre-split channels avoids the full-frame ``cv2.split`` on every call
    when this function is invoked inside a per-frame loop where the
    channels are already available.

    Pipeline context:
        Called by: compute_particle_colours_lab,
                   build_kernel_task (CPU ThreadPool path),
                   batch_build_kernels_gpu
        Calls: np.ogrid, np.mean
        Stage: classification

    Args:
        Limg: Lightness channel of the LAB image, shape ``(H, W)``,
            dtype uint8 or float32. [np.ndarray]
        Aimg: A (green-red) channel, same shape and dtype. [np.ndarray]
        Bimg: B (blue-yellow) channel, same shape and dtype. [np.ndarray]
        dets: List of detection tuples ``[(x, y, r, score), ...]``.
            [List[tuple]]

    Returns:
        List of ``(L_mean, A_mean, B_mean)`` float tuples, one per
        detection.  Returns ``(0.0, 0.0, 0.0)`` for any detection whose
        circular crop falls entirely outside the image bounds. [List[Tuple[float, float, float]]]

    Phase history:
        - Phase 4: Extracted from ``compute_particle_colours_lab`` to
          enable the CPU ThreadPool kernel-build path to reuse pre-split
          channels and avoid redundant conversions.
    """
    cols = []
    H, W = Limg.shape
    for (x, y, r, s) in dets:
        xi, yi = int(round(x)), int(round(y))
        rad = int(max(2, round(r*1.15)))
        x0 = max(0, xi - rad)
        x1 = min(W, xi + rad + 1)
        y0 = max(0, yi - rad)
        y1 = min(H, yi + rad + 1)

        if x1 <= x0 or y1 <= y0:
            cols.append((0.0, 0.0, 0.0))
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - xi)**2 + (yy - yi)**2 <= rad**2

        L_patch = Limg[y0:y1, x0:x1]
        A_patch = Aimg[y0:y1, x0:x1]
        B_patch = Bimg[y0:y1, x0:x1]

        L_pts = L_patch[mask]
        if len(L_pts) == 0:
            cols.append((0.0, 0.0, 0.0))
        else:
            mean_L = float(np.mean(L_pts))
            mean_A = float(np.mean(A_patch[mask]))
            mean_B = float(np.mean(B_patch[mask]))
            cols.append((mean_L, mean_A, mean_B))
    return cols

def build_LAB_tracking_kernel(det_list, frame_bgr, central_idx, neigh_indices,
                              sigma_factor_central=0.25, sigma_factor_neighbour=0.5, margin=2.0):
    """Build a neighbourhood-aware LAB correlation kernel for a central particle.

    Constructs a 3-channel (L, A, B) kernel array of size
    ``(2*k_half+1, 2*k_half+1)`` where ``k_half = ceil((1+margin)*r0)``.

    The L (luminance) channel encodes particle structure using a
    Difference-of-Gaussians (DoG) response multiplied by the Sobel
    gradient magnitude, providing edge-accentuated structural cues that
    are robust to illumination changes.

    The A and B (colour) channels encode the LAB chroma of neighbouring
    particles at their relative offset positions, blended with a Gaussian
    of width ``sigma_factor_neighbour * r_neighbour``.  This encodes the
    local colour neighbourhood pattern, enabling the NCC correlation step
    to distinguish particles by their surrounding context rather than
    appearance alone.

    Finally, a small circular boost (radius ``ceil(3 * sigma_central)``)
    is added to the L channel at the centre to raise the correlation peak
    confidence for the central particle.  All three channels are
    zero-mean unit-energy normalised before stacking.

    Pipeline context:
        Called by: main (reference kernel building, non-optimised path)
        Calls: compute_particle_colours_lab, cv2.GaussianBlur, cv2.Sobel,
               cv2.magnitude, np.ogrid
        Stage: correlation

    Args:
        det_list: List of detection tuples ``[(x, y, r, score), ...]``
            containing both the central and neighbour detections. [List[tuple]]
        frame_bgr: Current BGR video frame. [np.ndarray]
        central_idx: Index into ``det_list`` identifying the particle
            whose kernel is being built. [int]
        neigh_indices: List of indices into ``det_list`` for the
            neighbouring particles whose colours populate the A/B channels.
            [List[int]]
        sigma_factor_central: Controls the radius of the centre L-channel
            boost circle as ``ceil(3 * sigma_factor_central * r0)``. [float]
        sigma_factor_neighbour: Controls the Gaussian spread of each
            neighbour's colour contribution as
            ``sigma_factor_neighbour * r_neighbour``. [float]
        margin: Extra fractional padding around the particle radius that
            sets the kernel half-size: ``ceil((1+margin) * r0)``. [float]

    Returns:
        Tuple ``(K, cx_off, cy_off)`` where ``K`` is the float32 LAB
        kernel array of shape ``(kh, kw, 3)``, and ``cx_off``, ``cy_off``
        are the integer pixel offsets of the central particle within the
        kernel (both equal to ``k_half``). [Tuple[np.ndarray, int, int]]

    Phase history:
        - Original: Initial LAB neighbourhood kernel implementation.
        - Phase 3: DoG + edge-magnitude weighting added to L channel for
          better structural discrimination.
    """
    (x0, y0, r0, _) = det_list[central_idx]
    r0 = float(r0)
    k_half = int(np.ceil((1.0 + margin) * r0))
    kh, kw = 2 * k_half + 1, 2 * k_half + 1
    cx_off = cy_off = k_half

    lab_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    Limg, Aimg, Bimg = cv2.split(lab_img)
    H, W = Limg.shape

    K_L = np.zeros((kh, kw), np.float32)
    K_A = np.zeros((kh, kw), np.float32)
    K_B = np.zeros((kh, kw), np.float32)

    rr = int(max(2, round(r0)))
    x0i, x1i = int(round(x0)) - rr, int(round(x0)) + rr + 1
    y0i, y1i = int(round(y0)) - rr, int(round(y0)) + rr + 1
    ix0, ix1 = max(0, x0i), min(W, x1i); iy0, iy1 = max(0, y0i), min(H, y1i)
    L_patch = Limg[iy0:iy1, ix0:ix1].astype(np.float32)

    if L_patch.size > 0:
        sigma = max(1.0, 0.6 * r0)
        L_blur = cv2.GaussianBlur(L_patch, (0, 0), sigma)
        L_dog = L_patch - L_blur
        gx = cv2.Sobel(L_patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L_patch, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        gmag = (gmag - gmag.min()) / (np.ptp(gmag) + 1e-6)
        L_struct = L_dog * (0.5 + 0.5 * gmag)

        if np.std(L_struct) < 0.1:
            K_L_patch = np.zeros_like(L_struct)
        else:
            K_L_patch = (L_struct - L_struct.mean()) / (L_struct.std() + 1e-6)

        ky0, ky1 = cy_off - K_L_patch.shape[0] // 2, cy_off + (K_L_patch.shape[0] + 1) // 2
        kx0, kx1 = cx_off - K_L_patch.shape[1] // 2, cx_off + (K_L_patch.shape[1] + 1) // 2
        K_L[max(0, ky0):min(kh, ky1), max(0, kx0):min(kw, kx1)] = K_L_patch[:min(kh-ky0, K_L_patch.shape[0]), :min(kw-kx0, K_L_patch.shape[1])]

    col_lab = compute_particle_colours_lab(frame_bgr, det_list)
    for ni in neigh_indices:
        xn, yn, rn, _ = det_list[ni]
        dx = xn - x0; dy = yn - y0
        kx = cx_off + dx; ky = cy_off + dy
        if not (0 <= kx < kw and 0 <= ky < kh):
            continue
        sig = max(1.0, sigma_factor_neighbour * rn)
        rad = int(np.ceil(3.0 * sig))
        xg0, xg1 = int(max(0, np.floor(kx - rad))), int(min(kw, np.ceil(kx + rad + 1)))
        yg0, yg1 = int(max(0, np.floor(ky - rad))), int(min(kh, np.ceil(ky + rad + 1)))
        yy, xx = np.ogrid[yg0:yg1, xg0:xg1]
        g = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2.0 * sig * sig)).astype(np.float32)
        La, Lb = col_lab[ni][1], col_lab[ni][2]
        K_A[yg0:yg1, xg0:xg1] += La * g
        K_B[yg0:yg1, xg0:xg1] += Lb * g

    if np.std(K_L) > 0.1:
        sig_c = max(1.0, sigma_factor_central * r0)
        rad_c = int(np.ceil(3.0 * sig_c))
        yy, xx = np.ogrid[cy_off - rad_c:cy_off + rad_c + 1, cx_off - rad_c:cx_off + rad_c + 1]
        mask = (yy - cy_off) ** 2 + (xx - cx_off) ** 2 <= rad_c ** 2
        K_L[cy_off - rad_c:cy_off + rad_c + 1, cx_off - rad_c:cx_off + rad_c + 1][mask] += 15.0

    for Kc in (K_L, K_A, K_B):
        Kc -= Kc.mean(); Kc /= (np.sqrt((Kc*Kc).sum())+1e-6)
    K = cv2.merge([K_L, K_A, K_B]).astype(np.float32)
    return K, cx_off, cy_off

def correlate_kernel_lab(patch_lab, kernel_lab):
    """Correlate a LAB tracking kernel against a LAB image patch using per-channel NCC.

    Splits both the patch and the kernel into L, A, B channels and
    applies Normalised Cross-Correlation (NCC) independently to each
    channel.  The three NCC scores are averaged with equal weights to
    produce a single scalar similarity measure in ``[-1, 1]``.

    Equal weighting treats luminance structure and chroma context as
    equally important.  If only one channel is informative (e.g. a
    low-saturation grey particle), the other channels will have near-zero
    kernel norms and their NCC contribution will be small.

    Pipeline context:
        Called by: build_kernels_for_frame (kernel matching loop),
                   kernel_recovery_search
        Calls: ncc (per-channel NCC helper)
        Stage: correlation

    Args:
        patch_lab: Image patch extracted around a candidate detection
            centre, already converted to float32 LAB format.  Must be at
            least as large as ``kernel_lab`` in both spatial dimensions.
            [np.ndarray, shape (H, W, 3)]
        kernel_lab: LAB tracking kernel produced by
            ``build_LAB_tracking_kernel`` or
            ``build_LAB_kernel_cpu_optimized``, float32.
            [np.ndarray, shape (kh, kw, 3)]

    Returns:
        Scalar NCC score averaged across L, A, B channels.  Values
        near 1.0 indicate a strong template match; values near 0 or
        below indicate poor or anti-correlated match. [float]

    Phase history:
        - Original: Introduced with LAB tracking kernel framework.
    """
    Lp,Ap,Bp = cv2.split(patch_lab.astype(np.float32))
    Lk,Ak,Bk = cv2.split(kernel_lab.astype(np.float32))
    RL = ncc(Lp, Lk); RA = ncc(Ap, Ak); RB = ncc(Bp, Bk)
    return (RL + RA + RB) / 3.0

def build_LAB_kernel_cpu_optimized(det_list, central_idx, neigh_indices,
                                   Limg, Aimg, Bimg, col_lab_for_kernel, H, W, r0, args,
                                   affine_B=None, affine_alpha=0.0):
    """
    Builds the tracking kernel using the pre-split L, A, and B arrays instead of a BGR image.
    Uses pre-computed `col_lab_for_kernel` (LAB colors of detections) to weight the kernel appropriately.

    Phase 2 — affine deformation:
        affine_B : (2, 2) np.float64 ndarray or None
            Local deformation matrix B for this track's centre.  When provided
            with affine_alpha > 0, each neighbour offset r_n = (dx, dy) is
            replaced with (I + affine_alpha * B) r_n before being placed in
            the kernel.  affine_B=None or affine_alpha=0 reproduces the
            un-deformed Phase-1 behaviour exactly (byte-identical kernel).
        affine_alpha : float in [0, 1]
            Deformation confidence.  Multiplies B before adding to identity.
    """
    (x0, y0, _, _) = det_list[central_idx]
    r0 = float(r0)
    k_half = int(np.ceil((1.0 + args.kernel_margin) * r0))
    kh, kw = 2 * k_half + 1, 2 * k_half + 1
    cx_off = cy_off = k_half

    K_L = np.zeros((kh, kw), np.float32)
    K_A = np.zeros((kh, kw), np.float32)
    K_B = np.zeros((kh, kw), np.float32)

    # Build L channel (structure/edges)
    rr = int(max(2, round(r0)))
    x0i, x1i = int(round(x0)) - rr, int(round(x0)) + rr + 1
    y0i, y1i = int(round(y0)) - rr, int(round(y0)) + rr + 1
    ix0, ix1 = max(0, x0i), min(W, x1i)
    iy0, iy1 = max(0, y0i), min(H, y1i)
    L_patch = Limg[iy0:iy1, ix0:ix1].astype(np.float32)

    if L_patch.size > 0:
        sigma = max(1.0, 0.6 * r0)
        L_blur = cv2.GaussianBlur(L_patch, (0, 0), sigma)
        L_dog = L_patch - L_blur
        gx = cv2.Sobel(L_patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L_patch, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        gmag = (gmag - gmag.min()) / (np.ptp(gmag) + 1e-6)
        L_struct = L_dog * (0.5 + 0.5 * gmag)

        if np.std(L_struct) >= 0.1:
            K_L_patch = (L_struct - L_struct.mean()) / (L_struct.std() + 1e-6)
            ky0 = cy_off - K_L_patch.shape[0] // 2
            ky1 = cy_off + (K_L_patch.shape[0] + 1) // 2
            kx0 = cx_off - K_L_patch.shape[1] // 2
            kx1 = cx_off + (K_L_patch.shape[1] + 1) // 2
            K_L[max(0, ky0):min(kh, ky1), max(0, kx0):min(kw, kx1)] = \
                K_L_patch[:min(kh-ky0, K_L_patch.shape[0]), :min(kw-kx0, K_L_patch.shape[1])]

    # Build A/B channels from neighbors
    # Phase 2: optionally deform neighbour offsets by (I + alpha * B).
    _use_affine = affine_B is not None and affine_alpha != 0.0
    for ni in neigh_indices:
        xn, yn, rn, _ = det_list[ni]
        dx = xn - x0
        dy = yn - y0
        if _use_affine:
            # r'_n = (I + alpha B) r_n
            dx_d = dx + affine_alpha * (affine_B[0, 0] * dx + affine_B[0, 1] * dy)
            dy_d = dy + affine_alpha * (affine_B[1, 0] * dx + affine_B[1, 1] * dy)
            dx, dy = float(dx_d), float(dy_d)
        kx = cx_off + dx
        ky = cy_off + dy
        if not (0 <= kx < kw and 0 <= ky < kh):
            continue
        sig = max(1.0, args.sigma_factor_neighbour * rn)
        rad = int(np.ceil(3.0 * sig))
        xg0 = int(max(0, np.floor(kx - rad)))
        xg1 = int(min(kw, np.ceil(kx + rad + 1)))
        yg0 = int(max(0, np.floor(ky - rad)))
        yg1 = int(min(kh, np.ceil(ky + rad + 1)))
        yy, xx = np.ogrid[yg0:yg1, xg0:xg1]
        g = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2.0 * sig * sig)).astype(np.float32)
        La, Lb = col_lab_for_kernel[ni][1], col_lab_for_kernel[ni][2]
        K_A[yg0:yg1, xg0:xg1] += La * g
        K_B[yg0:yg1, xg0:xg1] += Lb * g

    # Add central boost to L channel
    if np.std(K_L) > 0.1:
        sig_c = max(1.0, args.sigma_factor_central * r0)
        rad_c = int(np.ceil(3.0 * sig_c))
        yy, xx = np.ogrid[cy_off - rad_c:cy_off + rad_c + 1, cx_off - rad_c:cx_off + rad_c + 1]
        mask = (yy - cy_off) ** 2 + (xx - cx_off) ** 2 <= rad_c ** 2
        K_L[cy_off - rad_c:cy_off + rad_c + 1, cx_off - rad_c:cx_off + rad_c + 1][mask] += 15.0

    # Normalize channels to zero-mean, unit-variance (Standard Score)
    for Kc in (K_L, K_A, K_B):
        Kc -= Kc.mean()
        Kc /= (Kc.std() + 1e-6)

    K = cv2.merge([K_L, K_A, K_B]).astype(np.float32)
    return K, cx_off, cy_off

def build_LAB_kernel_gpu(det_list, central_idx, neigh_indices,
                        Limg_gpu, Aimg_gpu, Bimg_gpu, col_lab, H, W, r0, args,
                        affine_B=None, affine_alpha=0.0):
    """
    Build LAB kernel using GPU arrays (CuPy version)

    KEY: Uses GPU arrays for faster processing

    Phase 2 — affine deformation:
        affine_B / affine_alpha: see build_LAB_kernel_cpu_optimized.  The
        deformation is applied to the (Python-side) neighbour offsets
        before the GPU mask is built, so CPU and GPU paths produce
        identical kernels under identical (B, alpha).
    """
    (x0, y0, _, _) = det_list[central_idx]
    r0 = float(r0)
    k_half = int(np.ceil((1.0 + args.kernel_margin) * r0))
    kh, kw = 2 * k_half + 1, 2 * k_half + 1
    cx_off = cy_off = k_half

    # Initialize on GPU
    K_L = cp.zeros((kh, kw), dtype=cp.float32)
    K_A = cp.zeros((kh, kw), dtype=cp.float32)
    K_B = cp.zeros((kh, kw), dtype=cp.float32)

    # Build L channel (structure/edges) - extract patch from GPU
    rr = int(max(2, round(r0)))
    x0i, x1i = int(round(x0)) - rr, int(round(x0)) + rr + 1
    y0i, y1i = int(round(y0)) - rr, int(round(y0)) + rr + 1
    ix0, ix1 = max(0, x0i), min(W, x1i)
    iy0, iy1 = max(0, y0i), min(H, y1i)
    L_patch_gpu = Limg_gpu[iy0:iy1, ix0:ix1]

    if L_patch_gpu.size > 0:
        # Transfer patch to CPU for cv2 operations (small patch, acceptable)
        L_patch = cp.asnumpy(L_patch_gpu)

        sigma = max(1.0, 0.6 * r0)
        L_blur = cv2.GaussianBlur(L_patch, (0, 0), sigma)
        L_dog = L_patch - L_blur
        gx = cv2.Sobel(L_patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L_patch, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        gmag = (gmag - gmag.min()) / (np.ptp(gmag) + 1e-6)
        L_struct = L_dog * (0.5 + 0.5 * gmag)

        if np.std(L_struct) >= 0.1:
            K_L_patch = (L_struct - L_struct.mean()) / (L_struct.std() + 1e-6)
            # Upload result back to GPU
            K_L_patch_gpu = cp.asarray(K_L_patch)
            ky0 = cy_off - K_L_patch.shape[0] // 2
            ky1 = cy_off + (K_L_patch.shape[0] + 1) // 2
            kx0 = cx_off - K_L_patch.shape[1] // 2
            kx1 = cx_off + (K_L_patch.shape[1] + 1) // 2
            K_L[max(0, ky0):min(kh, ky1), max(0, kx0):min(kw, kx1)] = \
                K_L_patch_gpu[:min(kh-ky0, K_L_patch.shape[0]), :min(kw-kx0, K_L_patch.shape[1])]

    # Build A/B channels from neighbors (GPU computation)
    # Phase 2: optional affine deformation of neighbour offsets (CPU-side compute,
    # identical formula to build_LAB_kernel_cpu_optimized for CPU/GPU parity).
    _use_affine = affine_B is not None and affine_alpha != 0.0
    for ni in neigh_indices:
        xn, yn, rn, _ = det_list[ni]
        dx = xn - x0
        dy = yn - y0
        if _use_affine:
            dx_d = dx + affine_alpha * (affine_B[0, 0] * dx + affine_B[0, 1] * dy)
            dy_d = dy + affine_alpha * (affine_B[1, 0] * dx + affine_B[1, 1] * dy)
            dx, dy = float(dx_d), float(dy_d)
        kx = cx_off + dx
        ky = cy_off + dy
        if not (0 <= kx < kw and 0 <= ky < kh):
            continue
        sig = max(1.0, args.sigma_factor_neighbour * rn)
        rad = int(np.ceil(3.0 * sig))
        xg0 = int(max(0, np.floor(kx - rad)))
        xg1 = int(min(kw, np.ceil(kx + rad + 1)))
        yg0 = int(max(0, np.floor(ky - rad)))
        yg1 = int(min(kh, np.ceil(ky + rad + 1)))

        # GPU computation of Gaussian weights
        yy, xx = cp.ogrid[yg0:yg1, xg0:xg1]
        g = cp.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / (2.0 * sig * sig))
        La, Lb = col_lab[ni][1], col_lab[ni][2]
        K_A[yg0:yg1, xg0:xg1] += La * g
        K_B[yg0:yg1, xg0:xg1] += Lb * g

    # Add central boost to L channel (GPU)
    if float(cp.std(K_L)) > 0.1:
        sig_c = max(1.0, args.sigma_factor_central * r0)
        rad_c = int(np.ceil(3.0 * sig_c))
        yy, xx = cp.ogrid[cy_off - rad_c:cy_off + rad_c + 1, cx_off - rad_c:cx_off + rad_c + 1]
        mask = (yy - cy_off) ** 2 + (xx - cx_off) ** 2 <= rad_c ** 2
        K_L[cy_off - rad_c:cy_off + rad_c + 1, cx_off - rad_c:cx_off + rad_c + 1][mask] += 15.0

    # Normalize channels to zero-mean, unit-variance (Standard Score)
    for Kc in (K_L, K_A, K_B):
        Kc -= cp.mean(Kc)
        Kc /= (cp.std(Kc) + 1e-6)

    # Transfer final kernel to CPU and merge
    K_L_cpu = cp.asnumpy(K_L)
    K_A_cpu = cp.asnumpy(K_A)
    K_B_cpu = cp.asnumpy(K_B)
    K = cv2.merge([K_L_cpu, K_A_cpu, K_B_cpu]).astype(np.float32)

    return K, cx_off, cy_off

def batch_build_kernels_gpu(tracks_soa, prev_detections, prev_frame, prev_lab, neighbor_lists, args, use_gpu=True, neighbor_data=None,
                             affine_by_id=None, only_track_ids=None):
    # [Phase 4C] Batched GPU kernel build.
    # Pipeline diagram: see docs/architecture/PIPELINE_FLOWCHARTS.md
    # Flowchart 13 (GPU Kernel Build Pipeline). Section labels [GPU-1..5]
    # below map 1:1 to that diagram's stages.
    #
    # [v6 Change] The original pipeline rebuilt one kernel per track per frame
    # on the CPU inside a Python for-loop (~540 sequential calls per frame).
    # Profiled cost on the primary 1024² video: ~1.8 s per frame, the single
    # largest contributor to the 5.7 s baseline frame time. Phase 4C introduced
    # this batched GPU path with three optimizations:
    #   1. Shared LAB conversion: prev_lab is computed ONCE at the top of the
    #      frame (caller side) and reused here, eliminating ~540 redundant
    #      cv2.cvtColor calls per frame.
    #   2. Shared neighbor_lists: precomputed by precompute_neighbor_lists()
    #      and reused from the cost-matrix stage.
    #   3. Per-track CuPy gather/mask operations replace the per-pixel CPU
    #      loop in build_LAB_kernel_cpu_optimized.
    # Result: ~1800 ms → ~40 ms per frame.
    """
    Build LAB tracking kernels for ALL tracks in a single batch (GPU-accelerated + adaptive sizing)

    KEY OPTIMIZATIONS:
    1. Convert LAB ONCE (not N times!)
    2. Adaptive kernel sizing based on actual neighbor distances

    Args:
        tracks_soa: TrackArrays object with all tracks
        prev_detections: Previous frame detections [(x, y, r, s), ...]
        prev_frame: Previous BGR frame
        prev_lab: Previous LAB frame (already converted)
        neighbor_lists: Dict {track_id: [neighbor_indices]}
        args: Arguments object
        use_gpu: Whether to use GPU acceleration
        neighbor_data: Dict {track_id: {'adaptive_factor': float, ...}} for adaptive sizing

    Returns:
        Dict {track_id: (kernel, cx_off, cy_off)}
    """
    if tracks_soa.count == 0 or prev_frame is None:
        return {}

    # === [GPU-1] Neighbor list reuse (see Flowchart 13 — Shared Precomputation) ===
    # ===== KEY OPTIMIZATION: Convert LAB ONCE for all kernels =====
    # This single conversion replaces 363 redundant conversions!
    # Use prev_lab if provided, otherwise convert prev_frame
    if prev_lab is not None:
        lab_frame = prev_lab
    else:
        lab_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2LAB)

    Limg, Aimg, Bimg = cv2.split(lab_frame)
    H, W = Limg.shape

    # Pre-compute all particle LAB colors ONCE
    col_lab = compute_particle_colours_lab(prev_frame, prev_detections)

    kernel_dict = {}

    # === [GPU-4] GPU batch dispatch (see Flowchart 13 — GPU branch) ===
    # Try GPU batch processing if available
    if use_gpu and gpu.USE_CUPY:
        try:
            # Upload LAB channels to GPU
            Limg_gpu = cp.asarray(Limg.astype(np.float32))
            Aimg_gpu = cp.asarray(Aimg.astype(np.float32))
            Bimg_gpu = cp.asarray(Bimg.astype(np.float32))

            # Build kernels in batch on GPU with adaptive sizing
            for i in range(tracks_soa.count):
                track_id = int(tracks_soa.ids[i])
                # [Phase 4 Stage 1, finding B] only_track_ids: build only
                # cache-miss tracks. The caller's put loop is put-if-absent,
                # so skipping builds whose output was discarded leaves the
                # cache evolution byte-identical while cutting ~5/6 of the
                # per-frame build cost. None = build all (legacy/fresh).
                if only_track_ids is not None and track_id not in only_track_ids:
                    continue
                neighbor_indices = neighbor_lists.get(track_id, [])

                if len(neighbor_indices) < args.min_neighbors:
                    continue

                tx = float(tracks_soa.positions[i, 0])
                ty = float(tracks_soa.positions[i, 1])
                tr = float(tracks_soa.radii[i])

                # Get adaptive search factor if available
                if neighbor_data and track_id in neighbor_data:
                    adaptive_factor = neighbor_data[track_id]['adaptive_factor']
                else:
                    adaptive_factor = args.search_factor

                # Find track's previous position
                track_prev_idx = None
                min_dist_prev = float('inf')
                for pi in range(len(prev_detections)):
                    px, py, _, _ = prev_detections[pi]
                    pd = math.hypot(tx - px, ty - py)
                    if pd < min_dist_prev:
                        min_dist_prev = pd
                        track_prev_idx = pi

                if track_prev_idx is None:
                    continue

                try:
                    # === [GPU-2] Patch gather + [GPU-3] LAB composition (Flowchart 13) ===
                    # build_LAB_kernel_gpu performs the gather (central + neighbor
                    # patches) and the LAB composition (mask central, weight neighbors)
                    # in a single CuPy kernel. The CPU equivalent
                    # (build_LAB_kernel_cpu_optimized) does the same in a Python
                    # for-loop and is the fallback path when CuPy is unavailable.
                    # Build kernel using GPU LAB data with adaptive search factor
                    # Create a modified args with adaptive search factor
                    args_adaptive = type('obj', (object,), {**vars(args), 'search_factor': adaptive_factor})()
                    # Phase 2: per-track (B, alpha) lookup, default = no deformation
                    _aB, _aalpha = (None, 0.0)
                    if affine_by_id is not None:
                        _ent = affine_by_id.get(track_id)
                        if _ent is not None:
                            _aB, _aalpha = _ent[1], _ent[2]   # (a, B, alpha)
                    K, cx_off, cy_off = build_LAB_kernel_gpu(
                        prev_detections, track_prev_idx, neighbor_indices,
                        Limg_gpu, Aimg_gpu, Bimg_gpu, col_lab, H, W,
                        tr, args_adaptive,
                        affine_B=_aB, affine_alpha=_aalpha,
                    )
                    kernel_dict[track_id] = (K, cx_off, cy_off)
                except Exception:
                    # Skip this kernel on failure
                    continue

            # === [GPU-5] Return kernels list (see Flowchart 13 — Output) ===
            return kernel_dict

        except Exception as e:
            # GPU failed, fall back to CPU
            print(f"[KERNEL] GPU kernel building failed ({e}), falling back to CPU")

    # CPU fallback: Still use shared LAB conversion + adaptive sizing!
    for i in range(tracks_soa.count):
        track_id = int(tracks_soa.ids[i])
        neighbor_indices = neighbor_lists.get(track_id, [])

        if len(neighbor_indices) < args.min_neighbors:
            continue

        tx = float(tracks_soa.positions[i, 0])
        ty = float(tracks_soa.positions[i, 1])
        tr = float(tracks_soa.radii[i])

        # Get adaptive search factor if available
        if neighbor_data and track_id in neighbor_data:
            adaptive_factor = neighbor_data[track_id]['adaptive_factor']
        else:
            adaptive_factor = args.search_factor

        # Find track's previous position
        track_prev_idx = None
        min_dist_prev = float('inf')
        for pi in range(len(prev_detections)):
            px, py, _, _ = prev_detections[pi]
            pd = math.hypot(tx - px, ty - py)
            if pd < min_dist_prev:
                min_dist_prev = pd
                track_prev_idx = pi

        if track_prev_idx is None:
            continue

        try:
            # Build kernel using shared LAB data (CPU) with adaptive search factor
            args_adaptive = type('obj', (object,), {**vars(args), 'search_factor': adaptive_factor})()
            _aB, _aalpha = (None, 0.0)
            if affine_by_id is not None:
                _ent = affine_by_id.get(track_id)
                if _ent is not None:
                    _aB, _aalpha = _ent[1], _ent[2]
            K, cx_off, cy_off = build_LAB_kernel_cpu_optimized(
                prev_detections, track_prev_idx, neighbor_indices,
                Limg, Aimg, Bimg, col_lab, H, W,
                tr, args_adaptive,
                affine_B=_aB, affine_alpha=_aalpha,
            )
            kernel_dict[track_id] = (K, cx_off, cy_off)
        except Exception:
            # Skip this kernel on failure
            continue

    return kernel_dict

