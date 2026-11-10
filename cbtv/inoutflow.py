"""In/outflow support: region map, row tags, track metadata.

Particles that leave and enter the view (frame edges, crop lines, occluders)
need three things the core tracker does not know about:

* WHERE the view ends and what lies beyond it (``RegionMap``);
* a per-row account of what each tracks.csv row is (``RowWriter`` with
  ``--row-tags 1``: matched / coast_frozen / coast_pred / occluded /
  tentative_backfill, plus a tracks_meta.csv sidecar with birth and death
  reasons), so results can be cleaned automatically (``cbtv_clean.py``);
* a short output delay so that tentative births (``--confirm-hits K``) can be
  written on confirmation without losing their first frames and without
  writing rows out of frame order.

Nothing here is imported unless one of the in/outflow flags is set, and with
every flag at its default the RowWriter writes exactly the legacy rows.

Region labels (``--region-map``, a grey PNG in the reference frame):
    0 view, 1 exit, 2 occluder, 3 entry band.

Region transform (``--region-transform``, CSV with header
``frame,dx,dy[,theta][,cx,cy]``): T_f maps RIG (reference-map) coordinates to
CAMERA coordinates of source frame f,
    camera = R(theta) (rig - c) + c + (dx, dy),
with theta in radians and c = (cx, cy) (default: the image centre). Frames
not listed are linearly interpolated (held at the ends). ``frame`` is the
0-based source-frame index of the input video (``--frame-step`` is applied).

``--region-transform-mode``:
    camera  the video is raw camera frames: the map moves with the rig,
            label(p) = map(T^-1 p); points off the map read 0 (view);
            the visible area is the frame rectangle.
    rig     the video is rig-stabilised: the map is fixed, label(p) = map(p),
            but the VISIBLE window moves: p is visible iff T(p) lies inside
            the camera frame. Stabilised borders must be constant black,
            never BORDER_REPLICATE (replicated pixels look like particles).
"""
from __future__ import annotations

import csv
import math
import os

import cv2
import numpy as np

VIEW, EXIT, OCCLUDER, ENTRY = 0, 1, 2, 3
ROW_STATES = ("matched", "coast_frozen", "coast_pred", "occluded", "tentative_backfill")
BIRTH_REASONS = ("first_frame", "entry_band", "interior", "after_occluder", "restart")
DEATH_REASONS = ("exit_region", "exit_frame", "miss_timeout", "end_of_video")


class RegionMap:
    """Per-frame region labels and visible-area geometry.

    Works without a map (all labels VIEW, visible area = frame) so callers
    need no special cases.
    """

    def __init__(self, W, H, map_path="", transform_path="", mode="camera"):
        self.W, self.H = int(W), int(H)
        self.mode = mode
        self.lab = None
        self.occ_dist = None
        if map_path:
            lab = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            if lab is None:
                raise SystemExit(f"[REGION] cannot read --region-map {map_path}")
            vals = set(np.unique(lab).tolist())
            if not vals <= {VIEW, EXIT, OCCLUDER, ENTRY}:
                raise SystemExit(
                    f"[REGION] --region-map must hold labels 0 view, 1 exit, 2 occluder, "
                    f"3 entry band; found values {sorted(vals)[:12]}")
            if lab.shape[:2] != (self.H, self.W):
                print(f"[REGION] WARNING map {lab.shape[1]}x{lab.shape[0]} resized to "
                      f"video {self.W}x{self.H} (nearest); transform offsets must be in video px")
                lab = cv2.resize(lab, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            self.lab = lab
            if (lab == OCCLUDER).any():
                self.occ_dist = cv2.distanceTransform(
                    (lab != OCCLUDER).astype(np.uint8), cv2.DIST_L2, 5)
            counts = {k: int((lab == k).sum()) for k in (VIEW, EXIT, OCCLUDER, ENTRY)}
            print(f"[REGION] map {map_path}: px view={counts[0]} exit={counts[1]} "
                  f"occluder={counts[2]} entry={counts[3]} mode={mode}")
        self._tf = None
        if transform_path:
            self._tf = _load_transform(transform_path, self.W, self.H)
            print(f"[REGION] transform {transform_path}: {len(self._tf['frame'])} rows, "
                  f"frames {int(self._tf['frame'][0])}..{int(self._tf['frame'][-1])}")
        self.set_frame(0)

    # -- per-frame state -------------------------------------------------
    def set_frame(self, src_idx):
        if self._tf is None:
            self.dx = self.dy = self.th = 0.0
            self.cx, self.cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        else:
            tf = self._tf
            f = float(src_idx)
            self.dx = float(np.interp(f, tf["frame"], tf["dx"]))
            self.dy = float(np.interp(f, tf["frame"], tf["dy"]))
            self.th = float(np.interp(f, tf["frame"], tf["theta"]))
            self.cx = float(np.interp(f, tf["frame"], tf["cx"]))
            self.cy = float(np.interp(f, tf["frame"], tf["cy"]))
        self.c, self.s = math.cos(self.th), math.sin(self.th)

    # -- geometry ---------------------------------------------------------
    def _rig_to_cam(self, x, y):
        ux, uy = x - self.cx, y - self.cy
        return (self.c * ux - self.s * uy + self.cx + self.dx,
                self.s * ux + self.c * uy + self.cy + self.dy)

    def _cam_to_rig(self, x, y):
        ux, uy = x - self.cx - self.dx, y - self.cy - self.dy
        return (self.c * ux + self.s * uy + self.cx,
                -self.s * ux + self.c * uy + self.cy)

    def _map_coords(self, x, y):
        """Image point -> reference-map coordinates."""
        if self.mode == "camera" and self._tf is not None:
            return self._cam_to_rig(x, y)
        return x, y

    def labels(self, xs, ys):
        """Map label at each point (VIEW everywhere when there is no map)."""
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        if self.lab is None:
            return np.zeros(xs.shape, dtype=np.int32)
        qx, qy = self._map_coords(xs, ys)
        ix = np.rint(qx).astype(np.int64)
        iy = np.rint(qy).astype(np.int64)
        inside = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        # Off-map points read VIEW: leaving the visible area is geometry
        # (outside_dist, with --exit-margin), not a label.
        out = np.full(xs.shape, VIEW, dtype=np.int32)
        out[inside] = self.lab[iy[inside], ix[inside]]
        return out

    def label(self, x, y):
        return int(self.labels([x], [y])[0])

    def _rect_out(self, x, y):
        ox = np.maximum(0.0, np.maximum(-x, x - (self.W - 1)))
        oy = np.maximum(0.0, np.maximum(-y, y - (self.H - 1)))
        return np.hypot(ox, oy)

    def _rect_in(self, x, y):
        return np.minimum(np.minimum(x, (self.W - 1) - x),
                          np.minimum(y, (self.H - 1) - y))

    def outside_dist(self, xs, ys):
        """Distance (px) by which each point lies outside the visible area
        (0 inside). The visible area is the video frame and, in rig mode,
        also the moving camera window (rotation preserves distance, so the
        window test is made in camera coordinates)."""
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        d = self._rect_out(xs, ys)
        if self.mode == "rig" and self._tf is not None:
            d = np.maximum(d, self._rect_out(*self._rig_to_cam(xs, ys)))
        return d

    def edge_dist(self, xs, ys):
        """Distance (px) from each point to the nearest visible-area edge
        (negative outside)."""
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        inner = self._rect_in(xs, ys)
        if self.mode == "rig" and self._tf is not None:
            inner = np.minimum(inner, self._rect_in(*self._rig_to_cam(xs, ys)))
        return np.where(inner >= 0, inner, -self.outside_dist(xs, ys))

    def near_edge(self, x, y, margin):
        """The legacy Phase-4D boundary-band test (x < m or x > W - m ...),
        applied to the video frame and, in rig mode, the camera window."""
        def band(px, py):
            return (px < margin or px > self.W - margin or
                    py < margin or py > self.H - margin)
        if band(x, y):
            return True
        if self.mode == "rig" and self._tf is not None:
            cx, cy = self._rig_to_cam(x, y)
            return band(cx, cy)
        return False

    def row_region(self, x, y):
        """Label for the tracks.csv `region` column: EXIT outside the visible
        area, else the map label."""
        if float(self.outside_dist([x], [y])[0]) > 0.0:
            return EXIT
        return self.label(x, y)

    def occluder_dist(self, x, y):
        if self.occ_dist is None:
            return math.inf
        qx, qy = self._map_coords(np.array([x], float), np.array([y], float))
        ix, iy = int(round(float(qx[0]))), int(round(float(qy[0])))
        if 0 <= ix < self.W and 0 <= iy < self.H:
            return float(self.occ_dist[iy, ix])
        return math.inf


def _load_transform(path, W, H):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        cols = {k.strip().lower(): k for k in (rd.fieldnames or [])}
        for need in ("frame", "dx", "dy"):
            if need not in cols:
                raise SystemExit(f"[REGION] --region-transform needs columns frame,dx,dy"
                                 f"[,theta][,cx,cy]; got {rd.fieldnames}")
        rows = []
        for r in rd:
            def g(k, d, r=r):
                return float(r[cols[k]]) if k in cols and r[cols[k]] != "" else d
            rows.append((g("frame", 0), g("dx", 0.0), g("dy", 0.0), g("theta", 0.0),
                         g("cx", (W - 1) / 2.0), g("cy", (H - 1) / 2.0)))
    if not rows:
        raise SystemExit(f"[REGION] --region-transform {path} is empty")
    a = np.array(sorted(rows), dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise SystemExit(f"[REGION] --region-transform {path} contains NaN/inf")
    return {"frame": a[:, 0], "dx": a[:, 1], "dy": a[:, 2], "theta": a[:, 3],
            "cx": a[:, 4], "cy": a[:, 5]}


def birth_reason(processed, x, y, r, region, boundary_margin, occluder_gate,
                 no_live_tracks):
    """Why a track was born here (tracks_meta.csv `birth_reason`)."""
    if processed == 0:
        return "first_frame"
    if no_live_tracks:
        return "restart"            # every track had died: tracker restart
    if region.occluder_dist(x, y) <= max(occluder_gate, 1.0) * r:
        return "after_occluder"
    if region.label(x, y) == ENTRY:
        return "entry_band"
    if region.lab is None or not (region.lab == ENTRY).any():
        # no explicit entry bands: the band inside the visible-area edge
        if float(region.edge_dist([x], [y])[0]) < boundary_margin:
            return "entry_band"
    return "interior"


def io_lifecycle(tracks, matched_tracks, excluded, processed, args, region,
                 prior_positions=None, prior_ids=None):
    """Age / advance / prune step for the in/outflow path (replaces the
    legacy [PRUNE-1..4] block when any lifecycle flag is set).

    Matched tracks: velocity from the last two matches, occluder counter
    reset, tentative hit counting. Unmatched tracks: excluded (exited before
    matching) -> removed; tentative -> dropped; otherwise the miss counter
    advances and the coast position is predicted (--coast-predict; always
    velocity inside an occluder), then exit and miss-tolerance tests.
    Returns the surviving tracks; death reasons are set on the removed ones.
    """
    coast = getattr(args, "coast_predict", "freeze")
    occ_n = int(getattr(args, "occluder_coast", 0))
    exit_kill = getattr(args, "exit_kill", 0) == 1
    margin = float(getattr(args, "exit_margin", 1.0))
    bm = float(getattr(args, "boundary_margin", 20.0))
    bonus = int(getattr(args, "boundary_miss_bonus", 6))
    vel_ema = float(getattr(args, "coast_velocity_ema", 1.0))
    W, H = region.W, region.H
    n_prior = 0 if prior_positions is None else len(prior_positions)
    surviving = []
    for i, t in enumerate(tracks):
        if i in matched_tracks:
            if t.lm_frame is not None and processed > t.lm_frame:
                g = processed - t.lm_frame
                sx, sy = (t.x - t.lm_x) / g, (t.y - t.lm_y) / g
                if vel_ema >= 1.0 or t.lm_frame == t.birth_frame:
                    t.vx, t.vy = sx, sy          # last observed step
                else:
                    t.vx = vel_ema * sx + (1.0 - vel_ema) * t.vx
                    t.vy = vel_ema * sy + (1.0 - vel_ema) * t.vy
            t.lm_x, t.lm_y, t.lm_frame = t.x, t.y, processed
            t.occl_n = 0
            t.coast_state = "matched"
            if t.tentative and t.birth_frame != processed:
                t.hits += 1
                if t.hits >= t.k_need:
                    t.tentative = False
            surviving.append(t)
            continue
        if i in excluded:               # left the view; death_reason already set
            if t.tentative:
                t.dropped = True
            continue
        if t.tentative:                 # unconfirmed birth that missed
            t.dropped = True
            continue
        t.missed += 1
        t.last_det_pos = None
        t.last_det_r = None
        t.coast_state = "coast_frozen"
        vel = (t.x + t.vx, t.y + t.vy)
        new, state = None, "coast_pred"
        if occ_n > 0 and region.label(*vel) == OCCLUDER:
            new, state = vel, "occluded"
            t.occl_n += 1
        elif coast == "velocity":
            new = vel
        elif coast == "flow":
            if i < n_prior and prior_ids and t.id in prior_ids:
                new = (float(prior_positions[i, 0]), float(prior_positions[i, 1]))
            else:
                new = vel
        elif t.px is not None and t.py is not None:
            # freeze: the legacy Phase-4D edge extrapolation, unchanged
            pred_x, pred_y = t.x + (t.x - t.px), t.y + (t.y - t.py)
            near = (pred_x < bm or pred_x > W - bm or pred_y < bm or pred_y > H - bm)
            if near and t.missed <= 3:
                t.px, t.py = t.x, t.y
                t.x = float(np.clip(pred_x, 1, W - 2))
                t.y = float(np.clip(pred_y, 1, H - 2))
                t.coast_state = "coast_pred"
        if new is not None:
            moved = (new[0] != t.x) or (new[1] != t.y)
            t.px, t.py = t.x, t.y
            t.x, t.y = float(new[0]), float(new[1])
            if moved or state == "occluded":
                t.coast_state = state
        if exit_kill:
            if region.label(t.x, t.y) == EXIT:
                t.death_reason = "exit_region"
                continue
            # T038: the coast position (motion prior) or the track's own velocity step past the margin
            if max(float(region.outside_dist([t.x], [t.y])[0]),
                   float(region.outside_dist([vel[0]], [vel[1]])[0])) > margin * t.r:
                t.death_reason = "exit_frame"
                continue
        tol = int(args.miss_tolerance)
        if region.near_edge(t.x, t.y, bm):
            tol += bonus
        tol += min(t.occl_n, occ_n)
        if t.missed > tol:
            t.death_reason = "miss_timeout"
            continue
        surviving.append(t)
    return surviving


class RowWriter:
    """Writes tracks.csv rows (optionally tagged) with an optional delay.

    delay = 0 and row_tags = False reproduces the legacy writer exactly.
    With delay d > 0 each frame's rows are held for d frames; rows of tracks
    that were dropped meanwhile (unconfirmed tentative births) are discarded
    and rows produced while a track was tentative are written with state
    tentative_backfill. d = K-1 guarantees every buffered track is decided
    (confirmed or dropped) before its first row is emitted.
    """

    def __init__(self, csv_wr, row_tags=False, delay=0, region=None, meta_path="",
                 r_from_det=False):
        self.wr = csv_wr
        self.r_from_det = bool(r_from_det)   # --csv-trust: matched rows report the det radius
        self.row_tags = bool(row_tags)
        self.delay = max(0, int(delay))
        self.region = region
        self.meta_path = meta_path
        self.buf = []            # [(frame, [(track, row, tentative), ...]), ...]
        self.registry = {}       # track id -> Track, every track with >= 1 row
        header = ["frame", "track_id", "x", "y", "r", "missed"]
        if self.row_tags:
            header += ["state", "coast_n", "region"]
        self.wr.writerow(header)

    def frame(self, processed, tracks):
        rows = []
        for t in tracks:
            r_out = t.r
            if self.r_from_det and t.missed == 0 and t.last_det_r is not None:
                r_out = t.last_det_r
            row = [processed, t.id, f"{t.x:.3f}", f"{t.y:.3f}", f"{r_out:.3f}", t.missed]
            if self.row_tags:
                state = "matched" if t.missed == 0 else getattr(t, "coast_state", "coast_frozen")
                reg = self.region.row_region(t.x, t.y) if self.region is not None else VIEW
                row += [state, t.missed, reg]
            rows.append((t, row, bool(getattr(t, "tentative", False))))
        if self.delay == 0:
            self._emit(processed, rows)
            return
        self.buf.append((processed, rows))
        while len(self.buf) > self.delay:
            f, rs = self.buf.pop(0)
            self._emit(f, rs)

    def _emit(self, frame, rows):
        for t, row, tent in rows:
            if getattr(t, "dropped", False):
                continue
            if tent and getattr(t, "tentative", False):
                # still undecided: only possible at end of video (see finish)
                continue
            if tent and self.row_tags:
                row[6] = "tentative_backfill"
            self.wr.writerow(row)
            if t.missed == 0 or tent:
                t.n_matched += 1
            else:
                t.n_coast += 1
            t.last_row_frame = frame
            self.registry.setdefault(t.id, t)

    def finish(self, live_tracks):
        """Flush the delay buffer and write tracks_meta.csv. Tracks still
        tentative at the end of the video are unconfirmed and are dropped."""
        n_unconf = 0
        for t in live_tracks:
            if getattr(t, "tentative", False):
                t.dropped = True
                n_unconf += 1
        while self.buf:
            f, rs = self.buf.pop(0)
            self._emit(f, rs)
        if n_unconf:
            print(f"[ROWS] {n_unconf} tentative tracks unconfirmed at end of video (dropped)")
        if not self.meta_path:
            return
        live_ids = {t.id for t in live_tracks}
        with open(self.meta_path, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["track_id", "birth_frame", "birth_reason", "death_frame",
                        "death_reason", "n_matched", "n_coast"])
            for tid in sorted(self.registry):
                t = self.registry[tid]
                dr = getattr(t, "death_reason", "") or ("end_of_video" if tid in live_ids
                                                        else "miss_timeout")
                w.writerow([tid, t.birth_frame, t.birth_reason or "unknown",
                            t.last_row_frame, dr, t.n_matched, t.n_coast])
        print(f"[ROWS] wrote {self.meta_path} ({len(self.registry)} tracks)")


def meta_path_for(output_csv):
    stem, _ = os.path.splitext(output_csv)
    return stem + "_meta.csv"
