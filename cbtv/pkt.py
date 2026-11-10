"""Predictive Kernel Tracking early-lock (frame n -> n+1 NCC)."""
import cv2
import numpy as np


def predictive_kernel_early_lock(tracks, frame_gray, prev_frame_gray, W, H, args, threshold=0.80, search_r=10):
    # [Phase 4C] PKT bypass — NCC template matching (see Flowchart 7)
    # [v6 Change] Two-tier lock architecture (Phase 4C):
    #   The original pipeline had no PKT — all ~540 tracks went through the full cost matrix
    #   and Hungarian assignment every frame. PKT uses NCC template matching against the
    #   previous frame to lock high-confidence tracks (NCC >= 0.85) before assignment,
    #   reducing the Hungarian matrix from 540x540 to ~4x5 on average.
    #   Template rebuild strategy: templates are rebuilt from current track position each frame
    #   (no caching). The original PKT prototype cached templates, but cached templates
    #   drifted after the March 11 offset fix shifted all coordinates.
    #   Subpixel refinement: parabolic interpolation on the NCC response map for sub-pixel
    #   position accuracy (standard 1D fit in x and y independently).
    #
    # [Phase 0] --pkt-rebuild-template gating:
    #   args.pkt_rebuild_template = 1 (default) -> rebuild every call (current/fixed behavior).
    #   args.pkt_rebuild_template = 0 -> if t.pkt_template is already cached, reuse it
    #       (reproduces the historic "locks tracks backwards" bug for A/B measurement).
    #   backwards_lock_count: incremented when a successful lock would move the track
    #       in a direction OPPOSITE to its previous-frame velocity vector.
    #       This is the magnitude signature of the documented bug.
    """
    Early matching using direct BGR patch correlation from the previous frame.
    Bypasses synthetic kernel generation constraints.

    Returns
    -------
    early_locked_track_ids : set[int]
    track_updates : dict {track_id: (abs_x, abs_y, ncc_score)}  plus a debug
        sentinel key 'max_debug' carrying the largest NCC seen this frame.
    backwards_lock_count : int  — diagnostic counter for the PKT bug.
    """
    early_locked_track_ids = set()
    track_updates = {}
    backwards_lock_count = 0

    if prev_frame_gray is None:
        return early_locked_track_ids, track_updates, backwards_lock_count

    # Phase 0 flag (default = 1 / rebuild every frame, current behavior).
    pkt_rebuild = int(getattr(args, 'pkt_rebuild_template', 1))

    # Optional: convert to float32 for normalized correlation to prevent overflow
    # but matchTemplate handles uint8 BGR vs uint8 BGR perfectly fine.

    for t_idx, t in enumerate(tracks):
        if t.missed == 0 and t.px is not None and t.py is not None:

            # [PKT-1] Template extraction from previous frame
            # Determine template size based on previous radius. Tight crop to exclude neighbors!
            r0 = max(2.0, float(getattr(t, 'pr', t.r)))
            k_half = int(np.ceil(1.1 * r0))  # 1.1x radius to tightly bound the circle
            kh, kw = 2 * k_half + 1, 2 * k_half + 1

            # Phase 0: cache reuse path (--pkt-rebuild-template=0).
            # If a cached template exists, use it. Otherwise fall back to fresh extraction
            # (and store it for next call), so the "buggy" path still produces something.
            template_gray = None
            tcx = tcy = None
            if pkt_rebuild == 0:
                cached = getattr(t, 'pkt_template', None)
                if cached is not None:
                    template_gray = cached
                    tcx = float(getattr(t, 'pkt_tcx', k_half))
                    tcy = float(getattr(t, 'pkt_tcy', k_half))

            if template_gray is None:
                # [PKT-1 cont.] Rebuild template from current track position in prev frame
                tx0 = int(t.x - k_half)
                tx1 = int(t.x + k_half + 1)
                ty0 = int(t.y - k_half)
                ty1 = int(t.y + k_half + 1)

                if tx0 < 0 or ty0 < 0 or tx1 > W or ty1 > H:
                    continue

                template_gray = prev_frame_gray[ty0:ty1, tx0:tx1].copy()
                if template_gray.shape[0] < 3 or template_gray.shape[1] < 3:
                    continue
                if template_gray.std() < 5.0:
                    continue

                template_gray = cv2.GaussianBlur(template_gray, (3, 3), 0)

                tcx = t.x - tx0
                tcy = t.y - ty0

                # If running in cache mode, store for next call (the buggy reuse path).
                if pkt_rebuild == 0:
                    t.pkt_template = template_gray
                    t.pkt_tcx = tcx
                    t.pkt_tcy = tcy

            # [PKT-2] Velocity prediction and search region definition
            vx = t.x - t.px
            vy = t.y - t.py
            pred_x = t.x + vx
            pred_y = t.y + vy

            sx0 = int(pred_x - search_r - k_half)
            sx1 = int(pred_x + search_r + k_half + 1)
            sy0 = int(pred_y - search_r - k_half)
            sy1 = int(pred_y + search_r + k_half + 1)

            if sx0 >= 0 and sy0 >= 0 and sx1 <= W and sy1 <= H:
                search_gray = frame_gray[sy0:sy1, sx0:sx1]

                if search_gray.shape[0] < kh or search_gray.shape[1] < kw:
                    continue

                # [PKT-3] NCC template matching (cv2.TM_CCOEFF_NORMED)
                search_gray = cv2.GaussianBlur(search_gray, (3, 3), 0)
                res = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > track_updates.get('max_debug', -1.0):
                    track_updates['max_debug'] = max_val

                if max_val > threshold:
                    px, py = max_loc  # Top-left of match in search_gray

                    # [PKT-4] Subpixel parabolic refinement
                    dx, dy = 0.0, 0.0
                    res_h, res_w = res.shape
                    if 0 < px < res_w - 1 and 0 < py < res_h - 1:
                        val_c = res[py, px]
                        val_l = res[py, px - 1]
                        val_r = res[py, px + 1]
                        val_u = res[py - 1, px]
                        val_d = res[py + 1, px]
                        # Standard 1D parabolic fit
                        denom_x = 2.0 * (val_l - 2.0 * val_c + val_r)
                        denom_y = 2.0 * (val_u - 2.0 * val_c + val_d)
                        if denom_x != 0:
                            dx = (val_l - val_r) / denom_x
                        if denom_y != 0:
                            dy = (val_u - val_d) / denom_y

                    # [PKT-5] Absolute position computation and lock
                    curr_tx0 = sx0 + px + dx
                    curr_ty0 = sy0 + py + dy

                    # The particle's subpixel center inside the cached template is (tcx, tcy).
                    # Thus, the new absolute subpixel coordinate is the sum of these.
                    abs_x = curr_tx0 + tcx
                    abs_y = curr_ty0 + tcy

                    # Phase 0 diagnostic: "backwards lock" — does the proposed lock move
                    # the track in a direction OPPOSITE to its velocity? This is the
                    # signature of the cached-template bug (template centred on the OLD
                    # position so the NCC peak biases backwards). We measure it regardless
                    # of rebuild mode so the magnitude is comparable across runs.
                    move_x = abs_x - t.x
                    move_y = abs_y - t.y
                    vel_mag = (vx * vx + vy * vy) ** 0.5
                    move_mag = (move_x * move_x + move_y * move_y) ** 0.5
                    # Only count when velocity and move are meaningfully non-zero.
                    if vel_mag > 0.5 and move_mag > 0.5:
                        if (vx * move_x + vy * move_y) < 0.0:
                            backwards_lock_count += 1

                    early_locked_track_ids.add(t.id)
                    track_updates[t.id] = (abs_x, abs_y, max_val)

    return early_locked_track_ids, track_updates, backwards_lock_count
