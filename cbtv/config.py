"""CLI argument definitions for the CBTV tracker (extracted from main())."""
import argparse


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-video", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=-1)

    ap.add_argument("--use-gpu", type=int, default=1)
    ap.add_argument("--gpu-device", type=int, default=None, help="Specific GPU device ID (0, 1, etc.). If not set, uses default GPU.")

    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--threshold-min", type=float, default=0.38)
    ap.add_argument("--threshold-max", type=float, default=0.62)
    ap.add_argument("--adaptive-threshold", type=int, default=1)
    ap.add_argument("--thr-step", type=float, default=0.01)
    ap.add_argument("--r-min", type=int, default=6)
    ap.add_argument("--r-max", type=int, default=10)
    ap.add_argument("--topk-channels", type=int, default=3)
    ap.add_argument("--nms-factor", type=float, default=0.6)
    ap.add_argument("--intelligent-nms", type=int, default=1)
    ap.add_argument("--hough-polarity", type=str, default="auto", choices=["auto","inv","norm","both"])

    ap.add_argument("--refine-bayes", type=int, default=1)
    ap.add_argument("--refine-top-pos", type=int, default=800)
    ap.add_argument("--refine-top-neg", type=int, default=800)
    ap.add_argument("--refine-beta", type=float, default=0.6)

    ap.add_argument("--thr-mode", type=str, default="hybrid", choices=["abs","local","hybrid"])
    ap.add_argument("--thr-local-k", type=float, default=2.5)
    ap.add_argument("--thr-local-ksize", type=int, default=0)
    ap.add_argument("--thr-topk-tile", type=int, default=0)
    ap.add_argument("--thr-tile-factor", type=float, default=3.0)

    ap.add_argument("--dedup-threshold", type=float, default=3.0)
    ap.add_argument("--use-improved-dedup", action="store_true",
                    help="Use improved clustering deduplication (100%% FP reduction)")
    ap.add_argument("--improved-dedup-threshold", type=float, default=10.0,
                    help="Threshold for improved clustering dedup (default: 10.0px)")

    # [Decision Rationale] Cost matrix weights (see Flowchart 8 — Cost Matrix): spatial distance dominates (0.7) because particles
    # are densely packed and position is the most reliable discriminator. Color gets 0.3 because
    # LAB distance is informative but subject to per-frame jitter (especially L01 videos with ~2% RGB noise).
    # The remaining weight (1 - 0.7 - 0.3 = 0.0 base) goes to kernel NCC via adaptive gamma.
    ap.add_argument("--w-dist", type=float, default=0.7)
    ap.add_argument("--w-color", type=float, default=0.3)
    # ------------------------------------------------------------------
    # Phase 3 Stage A — explicit kernel weight.
    # The kernel-NCC term [COST-5] is gated on `w_dist + w_color < 1.0`
    # (cost_matrix.py). Historically the only way to enable it was to
    # lower w_dist/w_color and leave the residual implicit. --w-kernel
    # makes the residual explicit: validate_args() enforces
    # w_dist + w_color + w_kernel == 1, so w_kernel > 0 <=> the kernel
    # branch is live. w_kernel = 0 (default) is byte-identical to the
    # locked Phase-1/2 baseline. The kernel term's magnitude inside the
    # branch is still carried by adaptive gamma (the shipped mechanism);
    # w_kernel sets how much weight dist+colour cede to it.
    # Sweep convention (Phase 3 Stage B): keep w_dist:w_color at 0.7:0.3,
    # i.e. --w-kernel K --w-dist 0.7*(1-K) --w-color 0.3*(1-K).
    ap.add_argument("--w-kernel", type=float, default=0.0,
                    help="Explicit kernel-NCC weight share (default 0.0 = kernel term off; "
                         "must satisfy w_dist + w_color + w_kernel = 1).")
    ap.add_argument("--max-dist", type=float, default=20.0)
    ap.add_argument("--max-color", type=float, default=60.0)
    ap.add_argument("--miss-tolerance", type=int, default=8)
    # [In/outflow F] Per-video scale: the Phase-4D boundary constants, formerly
    # hardcoded in the lifecycle block. Defaults are the old values.
    ap.add_argument("--boundary-margin", type=float, default=20.0,
                    help="Frame-edge band, px (or diameters with --units d): coasting tracks "
                         "inside it get the boundary miss bonus and edge extrapolation [20].")
    ap.add_argument("--boundary-miss-bonus", type=int, default=6,
                    help="Extra miss tolerance (frames) for tracks inside --boundary-margin [6].")
    # --units d: the px-valued gates --max-dist, --boundary-margin,
    # --recovery-search-radius and --recovery-max-move are read in particle
    # DIAMETERS (x --d-p px). Only flags given explicitly on the command line
    # are converted; unspecified ones keep their px defaults.
    ap.add_argument("--units", type=str, default="px", choices=["px", "d"],
                    help="Units of the explicit gate flags listed above: px (default) or d "
                         "(bead diameters, needs --d-p).")
    ap.add_argument("--d-p", type=float, default=0.0,
                    help="Bead diameter in px, for --units d (and the default for cbtv_clean.py).")
    # [In/outflow G] Row tags. 0 (default) = legacy 6-column tracks.csv, no
    # sidecar. 1 = append state, coast_n, region columns and write
    # <csv stem>_meta.csv (birth/death reason, n_matched, n_coast per track).
    ap.add_argument("--row-tags", type=int, default=0, choices=[0, 1],
                    help="Tag every tracks.csv row (state, coast_n, region) and write "
                         "<stem>_meta.csv with per-track birth/death reasons [0].")
    # [In/outflow A-E] Entry/exit/occluder handling. Every default reproduces
    # the legacy tracker exactly (see cbtv/inoutflow.py for the conventions).
    ap.add_argument("--region-map", type=str, default="",
                    help="Grey PNG of region labels in the reference frame: 0 view, 1 exit, "
                         "2 occluder, 3 entry band. Without it the frame edges are the only exits.")
    ap.add_argument("--region-transform", type=str, default="",
                    help="CSV frame,dx,dy[,theta][,cx,cy]: per-source-frame rig->camera "
                         "transform (theta in radians about cx,cy; default image centre).")
    ap.add_argument("--region-transform-mode", type=str, default="camera",
                    choices=["camera", "rig"],
                    help="camera: raw clip, the map moves with the rig. rig: rig-stabilised "
                         "clip, the map is fixed and the visible window moves [camera].")
    ap.add_argument("--coast-predict", type=str, default="freeze",
                    choices=["freeze", "velocity", "flow"],
                    help="Coasting position: freeze (legacy: stay put, edge extrapolation "
                         "only), velocity (advance by the last observed velocity), flow "
                         "(advance by the frame's motion prior, e.g. PIV; else velocity) [freeze].")
    ap.add_argument("--coast-velocity-ema", type=float, default=-1.0,
                    help="Track velocity for coasting: EMA weight of the newest observed step "
                         "(1.0 = last observed step only; lower smooths detection noise for "
                         "long coasts). -1 = auto: 0.3 with --occluder-coast, else 1.0 [auto].")
    ap.add_argument("--exit-kill", type=int, default=0, choices=[0, 1],
                    help="Terminate a track whose predicted position lies in an exit region "
                         "or more than --exit-margin radii outside the visible area; it is "
                         "removed from every matching stage that frame [0].")
    ap.add_argument("--exit-margin", type=float, default=1.0,
                    help="--exit-kill distance outside the visible area, in radii [1.0].")
    ap.add_argument("--occluder-coast", type=int, default=0,
                    help="N>0: tracks coasting inside an occluder advance by velocity and "
                         "their misses do not count toward --miss-tolerance for up to N "
                         "frames; detections inside occluders are ignored [0 = off].")
    ap.add_argument("--occluder-gate", type=float, default=2.0,
                    help="Match gate (radii) around the predicted position of a track that "
                         "coasted through an occluder [2.0].")
    ap.add_argument("--confirm-hits", type=int, default=1,
                    help="K>1: new tracks are tentative until K consecutive matches; a "
                         "tentative track that misses is dropped; confirmed tracks are "
                         "written from their birth frame [1 = off].")
    ap.add_argument("--confirm-hits-entry", type=int, default=0,
                    help="Confirmation K for births in entry bands (0 = same as "
                         "--confirm-hits) [0].")

    ap.add_argument("--use-neighborhood-kernels", type=int, default=1)
    ap.add_argument("--neighbor-search-factor", type=float, default=2.0)
    ap.add_argument("--min-neighbors", type=int, default=2)
    # [Decision Rationale] gamma=8.0 (see Flowchart 8 — Cost Matrix): maximum cost before a match is rejected as impossible.
    # Costs exceeding this are set to infinity in the assignment matrix. Tuned empirically:
    # too low (< 6) causes valid matches to be rejected in dense regions;
    # too high (> 10.5) allows cross-track swaps that degrade IDF1. See RULES.md: gamma <= 10.5.
    ap.add_argument("--gamma", type=float, default=8.0)
    ap.add_argument("--sigma-factor-central", type=float, default=0.25)
    ap.add_argument("--sigma-factor-neighbour", type=float, default=0.5)
    ap.add_argument("--kernel-margin", type=float, default=2.0)
    ap.add_argument("--search-factor", type=float, default=5.0)
    ap.add_argument("--use-adaptive-kernel-search", type=int, default=0,
                    help="Use KD-Tree based adaptive kernel search (0=off, 1=on)")

    ap.add_argument("--color-saturation-weight", type=int, default=1)
    ap.add_argument("--color-nonlinear-k", type=float, default=2.0)
    ap.add_argument("--glare-threshold", type=int, default=220)

    ap.add_argument("--adaptive-gamma", type=int, default=1, help="Adapt gamma based on kernel and color confidence")

    # [Phase 5-D] Persistence bonus: multiply cost by this factor when a detection was assigned
    # to the same track last frame. Lower = stickier (prefers continuing same assignment).
    # [Decision Rationale] persistence_bonus=0.85 (see Flowchart 8 — Cost Matrix): chosen via multi-run AssA sweep (Mar 29): median AssA=91.33%
    # at 0.85 vs 88.86% at default 1.0. Values below 0.80 caused over-stickiness (tracks
    # refused to release particles that had genuinely swapped positions). See EXPERIMENT_LOG.
    ap.add_argument("--persistence-bonus", type=float, default=0.85,
                    help="Cost multiplier for maintaining previous assignment (0.0-1.0, lower=stickier)")
    ap.add_argument("--use-3stage-matching", type=int, default=1)
    ap.add_argument("--conf-margin", type=float, default=4.0)
    ap.add_argument("--conf-maxcost", type=float, default=30.0)
    ap.add_argument("--hungarian-max-cost", type=float, default=120.0)

    ap.add_argument("--cap-mode", type=str, default="off", choices=["warmup","off","fixed","ratio"])
    ap.add_argument("--cap-fixed", type=int, default=2000)
    ap.add_argument("--cap-ratio", type=float, default=1.15)
    ap.add_argument("--cap-extra-frac", type=float, default=0.15)
    ap.add_argument("--cap-min", type=int, default=40)
    ap.add_argument("--cap-warmup", type=int, default=8)

    ap.add_argument("--second-pass", type=int, default=0)
    ap.add_argument("--second-pass-factor", type=float, default=2.0)

    ap.add_argument("--hard-factor", type=float, default=0.6, help="Hard filter relaxation factor")
    ap.add_argument("--soft-factor", type=float, default=0.8, help="Soft filter relaxation factor")
    ap.add_argument("--use-density-adaptive-thresholds", type=int, default=0, help="Enable density-aware threshold scaling (slower but may improve quality)")

    # Advanced numerical methods for improved accuracy
    ap.add_argument("--subpixel-refinement", type=int, default=0, help="Enable sub-pixel peak refinement using 2D parabolic fitting")
    ap.add_argument("--antialias-kernels", type=int, default=0, help="Enable anti-aliased kernel downsampling (Gaussian pre-filtering)")
    ap.add_argument("--better-integration", type=int, default=0, help="Use Simpson's rule for better numerical integration (detection only)")
    ap.add_argument("--use-phase-correlation", type=int, default=0, help="Use phase correlation (frequency domain) for kernel matching")
    ap.add_argument("--strict-multiframe", type=int, default=0, help="Require higher temporal consistency for multiframe detections")
    ap.add_argument("--second-color-factor", type=float, default=1.0)

    ap.add_argument("--kernel-recovery", type=int, default=1)
    ap.add_argument("--kernel-recovery-threshold", type=float, default=0.75, help="(deprecated) Old absolute threshold")
    ap.add_argument("--kernel-recovery-confidence", type=float, default=0.55, help="Confidence threshold (uniqueness+SNR based)")
    ap.add_argument("--kernel-recovery-search-radius", "--recovery-search-radius",
                    dest="kernel_recovery_search_radius", type=float, default=30.0,
                    help="Kernel-recovery NCC search radius, px (or diameters with --units d) [30].")
    ap.add_argument("--kernel-recovery-max-move", "--recovery-max-move",
                    dest="kernel_recovery_max_move", type=float, default=25.0,
                    help="Kernel-recovery max accepted move, px (or diameters with --units d) [25].")
    ap.add_argument("--kernel-recovery-min-neighbors", type=int, default=2)

    ap.add_argument("--dump-first-frame-corr", type=int, default=1)
    ap.add_argument("--dump-dir", type=str, default="debug_hybrid")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--profile-csv", type=str, default=None)

    # Multi-frame detection kernel improvement
    ap.add_argument("--multiframe-detection", type=int, default=1,
                    help="Use multi-frame candidate pooling for better detection kernel (1=on, 0=off)")
    ap.add_argument("--multiframe-offset", type=int, default=5,
                    help="Frame offset for multi-frame detection (default: 5 for t-5, t, t+5)")

    # Catastrophic drop recovery
    # Phase 3f: PKT lock threshold tuning. Default 0.85 (99.7% lock on
    # PRIMARY clean per existing tuning notes). Higher = more conservative
    # (fewer false locks on dense / fast-motion data).
    ap.add_argument("--pkt-lock-thresh", type=float, default=0.85,
                    help="PKT NCC threshold for full track lock "
                         "(0.85=default, 0.90=conservative, 0.80=aggressive)")
    ap.add_argument("--catastrophic-recovery", type=int, default=0,
                    help="Enable catastrophic detection drop recovery (1=on, 0=off)")
    ap.add_argument("--recovery-drop-threshold", type=float, default=0.90,
                    help="Detection count drop threshold (default: 0.90 = trigger if <90%% of moving average)")
    ap.add_argument("--recovery-history-size", type=int, default=5,
                    help="Number of frames for moving average (default: 5)")
    ap.add_argument("--recovery-min-threshold", type=float, default=0.29,
                    help="Minimum threshold for aggressive recovery (default: 0.29)")

    # Diagnostic mode: log detection counts at each pipeline stage
    ap.add_argument("--diagnostic", action="store_true",
                    help="Log per-frame detection counts at each pipeline stage to diagnostic_log.csv")

    # Optimization feature flags
    ap.add_argument("--opt-gpu-blob", type=int, default=1, help="Use GPU for LoG blob detection (1=on, 0=off)")
    # [Phase 3 Stage A, 2026-06-10] PKT default flipped 1 -> 0 ("PKT kill").
    # Phase 0/1 found pkt_off scores higher on the primary and every
    # Phase-1/2 benchmark ran with --opt-predictive-kernel 0; the default
    # path now matches that configuration by construction. The code path
    # is retained (named Phase-4 repair candidate) but DEPRECATED.
    ap.add_argument("--opt-predictive-kernel", type=int, default=0,
                    help="DEPRECATED (default 0 since Phase 3). Predictive Kernel Tracking "
                         "early lock (1=on, 0=off). Known to lock tracks backwards under "
                         "displacement; kept only as a Phase-4 repair candidate.")
    ap.add_argument("--opt-gpu-color", type=int, default=1, help="Use GPU batched color extraction (1=on, 0=off)")
    ap.add_argument("--opt-gpu-kernel", type=int, default=1, help="Use GPU batched kernel generation (1=on, 0=off)")
    # Workstream G2: obstacle mask pre-detection filter
    ap.add_argument("--obstacle-mask", type=str, default="",
                    help="Path to obstacle mask image (255=valid, 0=suppress). Applied pre-detection to zero NCC correlation in masked regions.")
    # [Phase 5-G2] Mask erosion: shrink valid region by erode pixels to create a buffer zone.
    # [Decision Rationale] mask-erode default=8 (see Flowchart 12 — Auto-Radius/Init): K06 diagnostic showed 87.5% of FPs were at the obstacle
    # boundary (within ~10px). Erosion of 8px eliminates most boundary FPs while retaining 6.2%
    # valid area. Sweep results: erode=4 (F1=28%), erode=8 (F1=34.1%), erode=12 (too aggressive).
    ap.add_argument("--mask-erode", type=int, default=8,
                    help="Pixels to erode obstacle mask inward (buffer zone to eliminate boundary FPs). 0=no erosion.")

    # [Phase 5-H] Small-image detection mode: auto-triggers for videos where min(H,W) < 256
    ap.add_argument("--small-image-mode", type=str, default="auto",
                     choices=["auto", "off", "on"],
                     help="Small-image detection mode (auto=trigger when min(H,W)<256)")

    ap.add_argument("--detector", type=str, default="bayes_adapt",
                    help="Detection strategy preset (see cbtv.detectors.available()).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for every RNG on the tracking/detection path (bayes_adapt "
                         "exemplar hold-out shuffle); same seed + inputs = same tracks.csv.")
    # Input polarity. The whole detection path (subtract_background, the
    # equalised Gray/V/L channels, Otsu inside the watershed strategies)
    # assumes bright particles on a dark background; a bright-background
    # video returns ~0 detections. 'bright'/'auto' matte the background to
    # black first (see cbtv.polarity). Default 'dark' never imports the
    # module, so the legacy path is byte-identical.
    ap.add_argument("--input-polarity", type=str, default="dark",
                    choices=["dark", "bright", "auto"],
                    help="Background polarity of the input video: dark "
                         "(legacy, no-op), bright (matte background to "
                         "black), auto (decide from the first frame).")
    # [Phase 6b] shared external detections for --detector csv: the same
    # frame,x,y,r,score CSV (proc-frame index) every baseline linker consumes.
    # [csv-trust] With --detector csv, 1 = the CSV detections are already
    # validated: skip every image-quality stage (multistage filter incl.
    # SPARSE tightening, obstacle mask, dedup merge, edge-fit refinement,
    # final filter, catastrophic recovery); all detections are hard and keep
    # their x, y, r; matched rows report the detection radius. Default 0 =
    # legacy (the image filter runs on csv detections too).
    # [colour source] The colour-distance matrix compares each detection with
    # the CURRENT frame re-sampled at the track's PREVIOUS position ('frame',
    # the legacy default). Once a particle moves more than ~r/2 per frame that
    # patch is mostly background and true matches fail the colour gate.
    # 'track' uses the colour the track stored when last matched (same
    # saturation-weighted LAB statistic, EMA'd with --colour-ema).
    ap.add_argument("--color-source", type=str, default="frame", choices=["frame", "track", "min"],
                    help="Track side of the colour distance: frame (legacy: re-sample the current "
                         "frame at the track's old position) or track (the colour stored at the "
                         "last match) or min (the smaller of the two per pair) [frame].")
    ap.add_argument("--csv-trust", type=int, default=0, choices=[0, 1],
                    help="--detector csv only: 1 = trust the CSV detections as given "
                         "(bypass the image-quality filters; all hard) [0].")
    ap.add_argument("--detections-csv", type=str, default=None,
                    help="[--detector csv] path to the shared detections CSV.")
    ap.add_argument("--force-log", dest="force_log", action="store_true",
                    help="Force LoG exemplar supplement in detection.")

    # ------------------------------------------------------------------
    # Phase 0 — diagnosed-bug flags + instrumentation
    # ------------------------------------------------------------------
    # PKT template-rebuild policy.
    #   1 (default) = rebuild template from prev_frame_gray at each call
    #                 (current Phase-4C-fixed behavior; matches locked baseline).
    #   0           = cache template once per track, reuse across frames
    #                 (reproduces the historic "locks tracks backwards" bug
    #                  described in CLAUDE.md "PKT bug critical for Phase 4C").
    # Used by cbtv.pkt.predictive_kernel_early_lock.
    ap.add_argument("--pkt-rebuild-template", type=int, default=1,
                    help="PKT template rebuild policy: 1=rebuild each frame (default/fixed), "
                         "0=cache and reuse (historic bug — for A/B measurement only).")

    # Colour EMA factor (beta in: c_new = beta*c_det + (1-beta)*c_old).
    #   0.3 (default) = current Phase-4C smoothing; matches locked baseline.
    #   1.0           = wholesale overwrite (the historic colour-jitter bug
    #                   described in CLAUDE.md — L01 oscillation).
    # Used in run_tracker.py first-pass + second-pass match update.
    ap.add_argument("--colour-ema", type=float, default=0.3,
                    help="EMA factor for track-colour update; 0.3=default smoothing, "
                         "1.0=wholesale overwrite (historic bug — A/B only). Bounded [0,1].")

    # V0 profile-CSV instrumentation toggle. When the user passes --profile-csv
    # the extended columns are always written (no separate flag needed).

    # ------------------------------------------------------------------
    # Phase 2 — local affine deformation field (`cbtv.field`)
    # ------------------------------------------------------------------
    # When --use-affine-field 0 (default), the field is NOT computed; the
    # cost-matrix gate and kernel builders run their original (Phase-1)
    # code paths. Byte-identical to the locked baseline.
    #
    # When --use-affine-field 1, run_tracker.py fits a per-track local
    # affine (a, B, alpha) between [MAIN-7] and [MAIN-8], the cost matrix
    # re-centres its distance gate on the predicted position, and the
    # LAB kernel builders deform neighbour offsets by `(I + alpha·B) r_n`.
    ap.add_argument("--use-affine-field", type=int, default=0,
                    help="Local-affine deformation field gate (0=off/baseline, 1=on/deform).")

    # Anchor source for the per-track WLS solve. `prev` uses previous-frame
    # velocities only (Track.px/py); `easy` uses only easy-match committed
    # current-frame anchors; `prev_plus_easy` combines both (default).
    ap.add_argument("--field-source", type=str, default="prev_plus_easy",
                    choices=["prev_plus_easy", "prev", "easy"],
                    help="Anchor source for the per-track affine fit.")

    # [Phase 5 Stage 3] Decouple the Tier-0 easy-COMMIT pass (MAIN-7.4) from
    # the field FIT/gate-recentring (MAIN-7.5/MAIN-8). Phase 4 dropped the
    # field because its bundled easy-commit pass grabbed particles BEFORE
    # constellations at 89-93% accuracy (vs cluster 99%+), poisoning the
    # field. With constellations running first, the cost matrix already
    # operates ONLY on the fall-through residual — so the field's
    # gate-recentring (+ kernel deform) can be applied there WITHOUT the
    # easy-commit liability. Default 1 = byte-identical to the Phase-2/4
    # field-on behaviour; 0 = fit+recentre the field (from prev anchors) but
    # commit no easy matches. Inert when --use-affine-field 0.
    ap.add_argument("--field-easy-commit", type=int, default=1,
                    help="Run the Tier-0 easy-match commit pass when the field is on "
                         "(1=on/parity, 0=field gate-recentring only, no easy commits).")

    # [Phase 5 Stage 3.1, Arm P] mu-source: where each particle's predicted
    # displacement (and uncertainty) comes from. `field` = the commit/velocity
    # affine (Phase 2/4, the parity default). `piv` = an ANCHOR-FREE dense
    # motion field from raw-pixel cross-correlation (cbtv/piv.py) — predicts
    # mu for EVERY particle (incl. the ~80% of fall-throughs the commit-affine
    # cannot reach, per Gate 0). `piv_plus_field` = PIV, with the commit-affine
    # filling tracks where PIV's peak is flat (alpha 0). Default `field` =
    # byte-identical baseline (the PIV branch is never entered).
    # [Phase 6c] `nvp` = the ORDERED-FLOW switch: trackpy-NearestVelocityPredict-
    # style prior. Each track's mu = its OWN last velocity when known, else the
    # median velocity of its --nvp-k nearest tracks with known velocity (pure
    # translation, B=0, alpha=1 -> kernel build byte-identical, only the gate
    # re-centring moves). Motivated by P6: on coherent plug flow (Group-M
    # chutes/hopper) constant-velocity extrapolation is ~exact and trackpy-NVP
    # was best-in-field; PIV's window correlation is noisier there. Everything
    # downstream (constellations, kernel, colour, assignment) is unchanged.
    # [Phase 6g] `nfield` = the PURE NEIGHBOUR-FIELD prior: every track's mu =
    # the component-wise median velocity of its --nfield-k nearest OTHER tracks
    # with known velocity. Own velocity is NEVER used directly (nvp's own-
    # velocity-first was the stale-feedback negative) and new tracks get
    # the field immediately (no birth gap). Pure translation, B=0, alpha=1.
    # [Phase 6g] `none` = EXPLICIT static prior: no prediction block runs at
    # all, gates stay centred on the previous position (behaviourally equal to
    # the default `field` with --use-affine-field 0, but self-documenting and
    # linkdump-labelled; the Stage-3.0 heavy-appearance arm pairs it with a
    # raised --w-kernel and a bulk-sized --max-dist).
    ap.add_argument("--mu-source", type=str, default="field",
                    choices=["field", "piv", "piv_plus_field", "nvp", "nfield", "none"],
                    help="Predicted-displacement source for gate re-centring + kernel deform.")
    ap.add_argument("--nvp-k", type=int, default=1,
                    help="[mu-source nvp] neighbours whose median velocity seeds a "
                         "track with unknown own velocity (1 = nearest, trackpy-like).")
    ap.add_argument("--nfield-k", type=int, default=5,
                    help="[mu-source nfield] neighbour count for the neighbour-field "
                         "median-velocity prior (own track always excluded).")
    ap.add_argument("--piv-window-d-units", type=float, default=5.0,
                    help="PIV interrogation window size in particle diameters.")
    ap.add_argument("--piv-step-overlap", type=float, default=0.5,
                    help="PIV window overlap fraction (0.5 = 50%%).")
    ap.add_argument("--piv-base-sigma", type=float, default=2.0,
                    help="PIV min displacement uncertainty (px) at a sharp peak -> alpha~1.")
    ap.add_argument("--piv-max-sigma", type=float, default=8.0,
                    help="PIV max displacement uncertainty (px) at a flat/ambiguous peak -> alpha~0.")
    ap.add_argument("--piv-ambiguity-disc", type=int, default=3,
                    help="PIV peak test: half-width (px) of the square around the correlation peak that is "
                         "excluded when the second peak is sought. In pixels, so scale it with the image "
                         "(e.g. 6 at twice the image scale); too small and the peak's own flank counts as "
                         "a second peak and the confidence collapses [3].")
    # [Phase 5 Stage 3.1] PIV has a ~1-diameter displacement ceiling: below it
    # PIV is a major win (s4 HOTA +0.16), above it the texture decorrelates and
    # PIV mispredicts -> fragmentation (s8 regression). Gate PIV OFF when the
    # OBSERVED mean displacement/radius exceeds this (2.0 = 1 diameter); above
    # the gate the frame falls back to the kernel/frozen path. 0 = no gate (PIV
    # always on, parity for the ungated screens). Driven by observed motion,
    # never the benchmark step.
    ap.add_argument("--piv-max-disp-ratio", type=float, default=0.0,
                    help="Disable PIV above this observed displacement/radius "
                         "(2.0 ~ 1 diameter; 0 = no gate).")
    # [Phase 6g Stage 3.2] Bulk-compensated PIV: one coarse full-frame phase-
    # correlation pass estimates the bulk (mean) shift; PIV interrogation
    # windows are re-centred by it and the --piv-max-disp-ratio gate applies
    # to the RESIDUAL velocity (own velocity minus bulk), so a fast-but-
    # coherent bed no longer disengages the prior. Default 0 = byte-identical
    # (no bulk estimate is computed).
    ap.add_argument("--piv-bulk-pass", type=int, default=0,
                    help="[mu-source piv] 1 = bulk-compensated PIV (residual gate).")
    # [Phase 6g rev 3, arm 3] Coherence-routed PIV gate: replaces the fixed
    # velocity gate (a speed PROXY for texture decorrelation that lies on
    # fast-but-coherent flow) with a per-particle trust hierarchy:
    #   PIV peak-sharpness confidence (alpha, computed in cbtv/piv.py) >=
    #   --piv-conf-min                       -> PIV prediction;
    #   else local neighbour-velocity dispersion (wrap-cleaned k-NN, the P6f
    #   T_gran definition) <= --nvel-disp-max -> own-velocity extrapolation
    #   (k-NN median velocity for tracks without history);
    #   both bad                              -> static (no prior).
    # Default 'velocity' = the shipped --piv-max-disp-ratio hard gate,
    # byte-identical. Only active under --mu-source piv.
    # [Phase 6h, Part B] `outlier` = the VALIDATED-EXTRAPOLATION gate (gate
    # attempt #3): certify the extrapolator, not PIV. Per track: own velocity
    # is VALIDATED against the k-NN neighbour median (relative-velocity
    # outlier test, absolute tolerance in diameters -- the B0 calibration
    # showed the MAD form is non-discriminative in coherent beds); validated
    # -> gates centred on the extrapolation. Birth tracks: neighbour median
    # if the neighbourhood is internally coherent. Optional PIV override
    # (--outlier-override): a SHARP PIV peak that DISAGREES with the prior
    # replaces it (the impact detector). Fallback: static.
    ap.add_argument("--piv-gate", type=str, default="velocity",
                    choices=["velocity", "coherence", "outlier"],
                    help="[mu-source piv] PIV trust gate: fixed velocity gate "
                         "(ship), per-particle coherence routing, or the "
                         "validated-extrapolation outlier gate.")
    # [Phase 6h, Part A] Per-N cluster cap: P6g showed cap 128 strands ~1/3 of
    # a 594-particle bed and cap 256 under-covers N=2373 (the same mechanism
    # at scale; RES_A_4K knee measured at 768 = alpha 0.324). Rule per frame:
    # cap = min(cap-max, max(--cluster-max-per-frame, ceil(alpha * N_det))).
    # The max() keeps every N<=594 cell at exactly the base cap
    # (byte-identical); default 0 = rule off, byte-identical everywhere.
    ap.add_argument("--cluster-cap-auto", type=int, default=0,
                    help="1 = per-N cluster cap "
                         "min(cap-max, max(base, ceil(alpha*N_det))).")
    ap.add_argument("--cluster-cap-alpha", type=float, default=0.324,
                    help="[cluster-cap-auto] clusters per detection "
                         "(RES_A_4K saturation knee 768/2373).")
    ap.add_argument("--cluster-cap-max", type=int, default=2048,
                    help="[cluster-cap-auto] hard ceiling on the auto cap.")
    # [Phase 6i Part A] Non-dimensionalised px limits: derive r_min/r_max/
    # max_dist from the calibrated px/diam (--nd-dpx, driver-supplied via the
    # publish-protocol convention scale x GT diameter). Ratios match legacy at
    # 12.8 px/diam (byte-identical no-op there, pre-registered). Motion floor:
    # max_dist >= --nd-km x m_est (median coarse-PIV magnitude, first frame
    # pair) so diameter-scaled gates cannot clip large per-step bulk motion.
    ap.add_argument("--nd-gates", type=int, default=0,
                    help="1 = derive r/gate px limits from --nd-dpx.")
    ap.add_argument("--nd-dpx", type=float, default=0.0,
                    help="[nd-gates] calibrated particle diameter in px.")
    ap.add_argument("--nd-km", type=float, default=1.25,
                    help="[nd-gates] motion-floor multiple of m_est.")
    ap.add_argument("--outlier-k", type=int, default=5,
                    help="[piv-gate outlier] neighbours for the validation median.")
    ap.add_argument("--outlier-tol-d", type=float, default=0.2,
                    help="[piv-gate outlier] max |v_own - median_k(v_nbr)| in "
                         "DIAMETERS to validate the extrapolation (B0-calibrated: "
                         "validated precision 0.93-0.997 at 0.2; diag_p6h/"
                         "B0_CALIBRATION.md).")
    ap.add_argument("--outlier-override", type=int, default=0,
                    help="[piv-gate outlier] 1 = enable the sharp-PIV override "
                         "(the impact detector).")
    ap.add_argument("--piv-conf-hi", type=float, default=0.9,
                    help="[outlier override] min PIV alpha for the override "
                         "(B0: alpha>=0.9 -> PIV within tol 93-100%%).")
    ap.add_argument("--override-margin", type=float, default=0.5,
                    help="[outlier override] min |v_piv - prior| in diameters "
                         "for the override to fire (B0 s4: inert at >=0.2 -- "
                         "sharp peaks agree with history; kept for the s8 "
                         "collective-surprise case).")
    ap.add_argument("--piv-conf-min", type=float, default=0.6,
                    help="[piv-gate coherence] min PIV peak-sharpness alpha to "
                         "trust the PIV vector. Calibrated on the P6g rev-3 "
                         "ungated linkdumps (diag/CALIBRATION.md): 0.6 is the "
                         "largest threshold that keeps PIV routing >= 80%% on "
                         "the smooth RES beds (95%%/82%% at s4/s8, GT-joined), "
                         "per the dispatch's pre-registered routing requirement.")
    ap.add_argument("--nvel-disp-max", type=float, default=0.02,
                    help="[piv-gate coherence] max wrap-cleaned neighbour-velocity "
                         "dispersion (diam^2/frame^2) to trust velocity "
                         "extrapolation when PIV confidence is low.")
    ap.add_argument("--nvel-k", type=int, default=8,
                    help="[piv-gate coherence] neighbours for the dispersion "
                         "estimate (P6f T_gran convention).")
    # [Phase 6g Stage 1] Per-frame per-track link diagnostics: prior source,
    # predicted position, match stage, gate distance. Write-only observer --
    # the tracking path is untouched (flag off = no code executes at all).
    ap.add_argument("--debug-linkdump", type=str, default="",
                    help="Path for a per-frame per-track link-diagnostics CSV "
                         "(empty = off).")
    # [Phase 5 gs_disp] The velocity gate above cannot catch the s8 failure
    # mode: PRIMARY is mostly a STATIC settled bed (s8 median velocity ratio
    # 0.35) past which a plug moves 2-3 r — PIV windows on near-static bed
    # particles lock onto the moving texture and predict motion for particles
    # that are not moving (velocity ~0 -> velocity gate blind). This gates on
    # PIV's OWN predicted displacement instead: a vector larger than the
    # decorrelation ceiling is untrustworthy whoever it lands on. Skipped
    # tracks keep their velocity/none prior. Default 0 = off = byte-identical.
    ap.add_argument("--piv-max-pred-ratio", type=float, default=0.0,
                    help="Drop a PIV prediction whose displacement exceeds this "
                         "x mean radius (2.0 ~ 1 diameter = the decorrelation "
                         "ceiling; 0 = no gate).")

    # ------------------------------------------------------------------
    # Phase 3 Stage B — displacement-adaptive cost weights (`cbtv.field`)
    # ------------------------------------------------------------------
    # When 1, the per-frame observed mean easy-match displacement over the
    # mean track radius drives (a) the effective kernel weight share, from 0
    # up to the configured --w-kernel, and (b) a distance-gate widening
    # factor. Driven by *observed* displacement, never the benchmark's
    # --frame-step (the tracker must not know the benchmark exists).
    # Requires --use-affine-field 1 (the easy-match committer supplies the
    # displacement observations). Default 0 = static weights, byte-identical.
    ap.add_argument("--adaptive-cost-weights", type=int, default=0,
                    help="Scale kernel weight + distance gate from observed easy-match "
                         "displacement/radius (0=off/static, 1=on). Needs --use-affine-field 1.")

    # ------------------------------------------------------------------
    # Phase 3 Stage C — colour constellations (`cbtv.clusters`)
    # ------------------------------------------------------------------
    # When 1, after the easy pass + field fit ([MAIN-7.5]) coherent colour
    # clusters of hard (not-yet-matched) tracks are matched as single
    # affine units against the detection set; members above the residual
    # bar are committed jointly and leave the cost matrix. Default 0 =
    # byte-identical to the Stage-B path.
    ap.add_argument("--use-constellations", type=int, default=0,
                    help="Colour-constellation joint assignment (0=off/baseline, 1=on).")
    ap.add_argument("--cluster-min-size", type=int, default=5,
                    help="Minimum constellation size (members).")
    ap.add_argument("--cluster-max-size", type=int, default=12,
                    help="Maximum constellation size (members).")
    ap.add_argument("--cluster-commit-conf", type=float, default=0.6,
                    help="Minimum cluster confidence required to commit members.")
    ap.add_argument("--cluster-residual-bar", type=float, default=0.6,
                    help="Per-member residual bar as a fraction of the member's radius; "
                         "members above the bar fall through to the normal cost matrix.")
    # ------------------------------------------------------------------
    # Phase 4 Stage 2a — constellation coverage levers (parity defaults).
    # Phase-3 finding: commits are 98-99% correct but COVERAGE-capped
    # (~85 members/frame vs ~114 block transitions/frame) — the mechanism
    # is starving, not failing.
    # ------------------------------------------------------------------
    ap.add_argument("--cluster-max-per-frame", type=int, default=64,
                    help="Constellation proposal cap per pass (Phase-3 hard-coded 64).")
    ap.add_argument("--cluster-rounds", type=int, default=1,
                    help="Constellation passes per frame; rounds >1 re-propose on the "
                         "residual hard set after the previous round's commits refresh "
                         "the field.")
    ap.add_argument("--cluster-fallback-min-size", type=int, default=0,
                    help="0=off. 3 or 4: one extra pass of small clusters (stricter "
                         "--cluster-fallback-conf) for block particles no 5-cluster reaches.")
    ap.add_argument("--cluster-fallback-conf", type=float, default=0.75,
                    help="Commit-confidence bar for the small-cluster fallback pass.")
    # ------------------------------------------------------------------
    # Phase 4 Stage 2b — temporal constellation->field propagation.
    # Committed cluster transforms are the highest-grade velocity evidence
    # available (98-99% audit-correct): when 1, tracks that were committed
    # by a constellation in the PREVIOUS frame contribute their velocity
    # anchors to THIS frame's field fit at weight --field-temporal-weight
    # (others stay at 1.0). Lifts field alpha exactly in the dense regions
    # where easy matches are scarce. Default 0 = unweighted fit (parity).
    # ------------------------------------------------------------------
    ap.add_argument("--field-temporal-clusters", type=int, default=0,
                    help="Carry previous-frame constellation commits into the field fit "
                         "as high-weight anchors (0=off/parity, 1=on).")
    ap.add_argument("--field-temporal-weight", type=float, default=2.5,
                    help="WLS weight for previous-frame constellation-committed anchors.")
    # ------------------------------------------------------------------
    # Phase 4 Stage 2c — easy-pass hygiene (Tier-1 exclusion where it pays).
    # Phase-3 mechanism finding: in the >=1.1-diam plug-flow block the
    # radius-gated easy committer grabs the SHIFTED NEIGHBOUR (the only
    # det within 1r of the stale position) and commits it confidently.
    # When 1: (i) a candidate whose velocity-predicted position is also
    # claimed by another unmatched track is not easy (competition at
    # comparable prior); (ii) an implied displacement deviating > ~1
    # nn-spacing from the track's own velocity prediction demotes to hard.
    # Runs BEFORE the field fit, so the available prior is the previous
    # velocity (cluster-grade under --field-temporal-clusters).
    # ------------------------------------------------------------------
    ap.add_argument("--easy-competition-check", type=int, default=0,
                    help="Easy-pass hygiene: competition + displacement-consistency "
                         "checks (0=off/parity, 1=on).")
    # ------------------------------------------------------------------
    # Phase 4 Stage 2d — displacement-aware constellation commit bar.
    # Phase-3 audit: commit accuracy 98-99% at steps 3-4 but 87-88% at
    # step 8. When 1, the commit-confidence bar and the min-prior gate
    # scale with the OBSERVED displacement ratio (never the benchmark
    # step): ratio<0.8 -> base bars; 0.8-1.4 -> +0.05/+0.10; >1.4 ->
    # +0.10/+0.20.
    # ------------------------------------------------------------------
    ap.add_argument("--cluster-disp-aware-bar", type=int, default=0,
                    help="Scale cluster commit conf + min-prior with observed "
                         "displacement (0=off/parity, 1=on).")
    # ------------------------------------------------------------------
    # Phase 4 Stage 3 — science-cell ablation flags (never defaults).
    # ------------------------------------------------------------------
    ap.add_argument("--cluster-colour-blind", type=int, default=0,
                    help="SCIENCE CELL ONLY: zero the colour term in constellation "
                         "member-matching AND disable colour-novelty growth (the moat "
                         "measurement). 0=off/parity.")
    ap.add_argument("--kernel-deform-off", type=int, default=0,
                    help="SCIENCE CELL ONLY: build LAB kernels with the affine "
                         "deformation forced off (alpha=0) while the field still "
                         "drives gates/constellations. 0=off/parity.")
    # ------------------------------------------------------------------
    # Phase 4 Stage 4 — cascade Tier 1 (exclusion/competition pruning).
    # Iterative near-certain commit + target deletion on the fall-through
    # cost matrix (assign-by-elimination), same margin/max-cost bars as
    # assignment stage 1 — never prunes below a measured alternative.
    # ------------------------------------------------------------------
    ap.add_argument("--cascade-tier1", type=int, default=0,
                    help="Cascade Tier-1 exclusion pass before the assignment stages "
                         "(0=off/parity, 1=on).")

    # ------------------------------------------------------------------
    # Phase 3 Stage B — kernel-cache freshness probe
    # ------------------------------------------------------------------
    # The shipped KernelCache is put-if-absent with a 5-frame age limit
    # and a dead position-invalidation (get() is called without
    # current_pos), so NCC kernels are 0-5 processed frames stale while
    # ~5/6 of the per-frame kernel-build output is discarded. At step 4
    # the plug flow moves ~13 px/frame: a 5-frame-stale kernel's
    # neighbour geometry is off by up to ~65 px. 1 = always store the
    # freshly built kernel (same build cost — the build already runs
    # every frame — but NCC consumes age-0 kernels). Default 0 keeps the
    # shipped behaviour byte-identical.
    ap.add_argument("--kernel-cache-fresh", type=int, default=0,
                    help="Always refresh the kernel cache with this frame's build "
                         "(0=shipped put-if-absent/5-frame-stale, 1=fresh).")

    # ------------------------------------------------------------------
    # Phase 5 Speed Stage 2 — flag-gated tracking-hotspot optimisations.
    # All default to the shipped behaviour (byte-identical parity); each
    # targets a profiled hotspot (see data/results/p5/speed_gs/
    # INTERIM_FINDINGS.md for the per-bucket ms/frame profile).
    # ------------------------------------------------------------------
    # Lever 1: sparse-gate -> STRICTLY byte-identical fast assignment. The
    # profiled t_asn hotspot is NOT the Hungarian (the scipy LSA is <0.5% of
    # t_asn at PRIMARY scale) but the Python-loop Stage-1 greedy + Stage-2 2x2
    # (~96% combined): full-width np.argsort/argmin per track + O(len) `in list`
    # membership scans over ~560 remaining tracks/dets. --sparse-gate routes
    # both to fast variants that are byte-identical to the shipped loops --
    # greedy vectorises argmin/partition; 2x2 swaps `in list` for O(1) sets
    # while keeping the IDENTICAL per-row/col np.argsort (so tie order matches
    # bit-for-bit; a whole-matrix argsort precompute was rejected because 2x2's
    # commit cascade turns tie-order differences into cost-CHANGING divergence).
    # ~5x on the 2x2 stage; verified by _smoke_sparse_gate + the
    # CBTV_ASSIGN_PARITY=1 real-data probe. Stage 3 stays baseline.
    ap.add_argument("--sparse-gate", type=int, default=0,
                    help="[Phase5-speed] Byte-identical fast Stage-1 greedy + "
                         "Stage-2 2x2 assignment (0=off/baseline shipped loops, "
                         "1=on/fast). Targets the ~96%% of t_asn outside the LSA.")
    # Lever 2: kernel-recovery gate. batch_kernel_recovery runs every frame
    # whenever --kernel-recovery 1 (the default), INDEPENDENT of w_kernel --
    # so a pure-distance/PIV run (w_kernel=0) still pays the 150-247 ms NCC
    # fallback even though the cost matrix's kernel term is off. When 1, gate
    # recovery on w_kernel>0 (the kernel term being active). Default 0 =
    # always run = byte-identical. ON changes output when w_kernel==0 ->
    # HOTA-validate.
    ap.add_argument("--kernel-recovery-gate", type=int, default=0,
                    help="[Phase5-speed] Gate kernel recovery on w_kernel>0 "
                         "(0=off/always-run/baseline, 1=skip recovery when w_kernel==0).")
    # [Phase 5 Stage 3] Appearance-term operator. 'kernel' (default) = the shipped
    # LAB-neighbourhood kernel + NCC (byte-identical). 'graph-struct' = the
    # kernel-free structural matcher (cbtv.match_ops): anchors a track's neighbour
    # star-graph at the candidate, deforms by the local PIV transform, and scores
    # neighbour alignment (5-50x cheaper than kernel-BUILD; feasibility 12/12).
    # Only active when w_kernel>0 (the appearance term has weight); skips the
    # kernel build entirely.
    ap.add_argument("--match-op", type=str, default="kernel",
                    choices=["kernel", "graph-struct"],
                    help="[Phase5-Stage3] Appearance operator: kernel (default, "
                         "byte-identical) or graph-struct (kernel-free structural).")
    # [Phase 5 Stage 3] Shared appearance gate in PARTICLE DIAMETERS (diameter =
    # 2*r_max), used IDENTICALLY by both --match-op operators for a fair compare:
    #   graph-struct: the neighbour-match tolerance (predicted neighbour must land
    #     within match_gate_d diameters of a detection);
    #   kernel: the NCC relaxed-search gate (which pairs get a kernel evaluation),
    #     replacing the hard-coded 2*max_dist.
    # Default 0 = LEGACY/byte-identical (kernel uses 2*max_dist; graph-struct uses
    # --graph-struct-gate px). >0 overrides both with match_gate_d*2*r_max px.
    ap.add_argument("--match-gate-d", type=float, default=0.0,
                    help="[Phase5-Stage3] Shared appearance gate in particle diameters for BOTH "
                         "--match-op operators (0=legacy/byte-identical; e.g. 1.1 ~ 1 spacing).")
    ap.add_argument("--graph-struct-gate", type=float, default=22.0,
                    help="[Phase5-Stage3] Star-graph neighbour gate (px) for --match-op graph-struct: a "
                         "predicted neighbour must land within this of a detection to count. Must be ~1 "
                         "particle spacing (~2*r), NOT sub-radius: the real PIV field has ~10-15 px residual "
                         "at high displacement, so a tight 8 px gate leaves n_matched~0 -> the score is inert "
                         "(flat across candidates) and the term does nothing. At 22-25 px graph-struct matches/"
                         "beats the kernel (PRIMARY PIV s4 w_kernel=0.5: gate8 HOTA 0.231 vs gate25 0.504 vs "
                         "kernel 0.491). Ideally scales with observed displacement (future).")
    return ap


def validate_args(args):
    """Post-parse invariants (Phase 3 Stage A).

    Enforces the explicit cost-weight budget `w_dist + w_color + w_kernel
    == 1` so the kernel term's weight share is never an implicit residual.
    Raises SystemExit (argparse-style error) on violation.
    """
    total = float(args.w_dist) + float(args.w_color) + float(getattr(args, "w_kernel", 0.0))
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(
            f"--w-dist + --w-color + --w-kernel must sum to 1.0 "
            f"(got {args.w_dist} + {args.w_color} + {getattr(args, 'w_kernel', 0.0)} = {total}). "
            f"Sweep convention for --w-kernel K: --w-dist 0.7*(1-K) --w-color 0.3*(1-K).")
    if getattr(args, "w_kernel", 0.0) < 0.0:
        raise SystemExit(f"--w-kernel must be >= 0 (got {args.w_kernel}).")
    if getattr(args, "adaptive_cost_weights", 0) == 1 and getattr(args, "use_affine_field", 0) != 1:
        raise SystemExit("--adaptive-cost-weights 1 requires --use-affine-field 1 "
                         "(easy-match displacement observations come from the field's easy pass).")
    if getattr(args, "use_constellations", 0) == 1:
        if int(args.cluster_min_size) < 3:
            raise SystemExit("--cluster-min-size must be >= 3 (affine fit needs 3+ members).")
        if int(args.cluster_max_size) < int(args.cluster_min_size):
            raise SystemExit("--cluster-max-size must be >= --cluster-min-size.")
    if int(getattr(args, "piv_ambiguity_disc", 3)) < 1:
        raise SystemExit(f"--piv-ambiguity-disc must be >= 1 (got {args.piv_ambiguity_disc}).")
    if getattr(args, "boundary_margin", 20.0) < 0 or getattr(args, "boundary_miss_bonus", 6) < 0:
        raise SystemExit("--boundary-margin and --boundary-miss-bonus must be >= 0.")
    if getattr(args, "units", "px") == "d":
        _convert_diameter_units(args)
    import os
    for flag in ("region_map", "region_transform"):
        path = getattr(args, flag, "")
        if path and not os.path.isfile(path):
            raise SystemExit(f"--{flag.replace('_', '-')} not found: {path}")
    if getattr(args, "region_transform", "") and not getattr(args, "region_map", "") \
            and getattr(args, "region_transform_mode", "camera") == "camera":
        print("[REGION] NOTE --region-transform without --region-map in camera mode has "
              "no effect (the frame edges do not move); use --region-transform-mode rig "
              "for a stabilised clip.")
    if getattr(args, "csv_trust", 0) == 1 and getattr(args, "detector", "") != "csv":
        raise SystemExit("--csv-trust 1 applies only to --detector csv.")
    if getattr(args, "coast_velocity_ema", -1.0) < 0:
        # auto: occluder coasts span tens of frames, where last-step velocity
        # noise dominates (synthetic: 31% -> 92% re-capture at 0.3)
        args.coast_velocity_ema = 0.3 if getattr(args, "occluder_coast", 0) > 0 else 1.0
    if not 0.0 < args.coast_velocity_ema <= 1.0:
        raise SystemExit("--coast-velocity-ema must be in (0, 1] (or -1 = auto).")
    if getattr(args, "exit_margin", 1.0) < 0:
        raise SystemExit("--exit-margin must be >= 0.")
    if getattr(args, "occluder_coast", 0) < 0 or getattr(args, "occluder_gate", 2.0) <= 0:
        raise SystemExit("--occluder-coast must be >= 0 and --occluder-gate > 0.")
    if getattr(args, "confirm_hits", 1) < 1 or getattr(args, "confirm_hits_entry", 0) < 0:
        raise SystemExit("--confirm-hits must be >= 1 and --confirm-hits-entry >= 0.")
    if getattr(args, "occluder_coast", 0) > 0 and not getattr(args, "region_map", ""):
        raise SystemExit("--occluder-coast needs --region-map with occluder (label 2) pixels.")
    if getattr(args, "exit_kill", 0) == 1 and getattr(args, "coast_predict", "freeze") == "freeze":
        print("[EXIT] NOTE --exit-kill with --coast-predict freeze: coasting tracks never "
              "move, so only motion-prior predictions and exit regions can trigger a kill; "
              "--coast-predict velocity is the intended pairing.")
    return args


def inoutflow_active(args):
    """True when any in/outflow LIFECYCLE flag departs from its legacy default
    (row tags alone only change the writer, not the tracker)."""
    return (getattr(args, "coast_predict", "freeze") != "freeze"
            or getattr(args, "exit_kill", 0) == 1
            or getattr(args, "occluder_coast", 0) > 0
            or getattr(args, "confirm_hits", 1) > 1
            or getattr(args, "confirm_hits_entry", 0) > 1
            or bool(getattr(args, "region_map", ""))
            or bool(getattr(args, "region_transform", "")))


# Gate flags that --units d reinterprets as bead diameters.
_DIAMETER_UNIT_FLAGS = ("max_dist", "boundary_margin",
                        "kernel_recovery_search_radius", "kernel_recovery_max_move")


def _explicit_dests(argv=None):
    """Dests of the options actually present on the command line (defaults
    suppressed), so --units d converts only what the user wrote."""
    import argparse
    import sys
    ap = build_parser()
    for act in ap._actions:
        act.default = argparse.SUPPRESS
        act.required = False
    ns, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    return set(vars(ns))


def _convert_diameter_units(args, argv=None):
    d_p = float(getattr(args, "d_p", 0.0))
    if d_p <= 0.0:
        raise SystemExit("--units d requires --d-p > 0 (bead diameter in px).")
    given = _explicit_dests(argv)
    for dest in _DIAMETER_UNIT_FLAGS:
        if dest in given:
            val = float(getattr(args, dest))
            setattr(args, dest, val * d_p)
            print(f"[UNITS] --{dest.replace('_', '-')} {val:g} d -> {val * d_p:.2f} px "
                  f"(d_p={d_p:g})")
