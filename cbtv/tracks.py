"""Track state: Track, TrackArrays (SoA mirror), KernelCache."""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
# ============================================================================
# SECTION: TRACK DATA STRUCTURES
# Purpose: Core data structures for tracking. Track holds per-particle state,
#          TrackArrays provides SoA (Structure-of-Arrays) view for vectorized
#          operations, KernelCache stores correlation kernels across frames.
# Key functions: Track, TrackArrays, KernelCache
# Phase history: Original. Velocity fields removed v7 (dead code).
#                last_det_pos/last_det_r added Phase 5 D.
# ============================================================================

# === [DATA-1] Track — per-particle tracking state (see Flowcharts 10, 20) ===
# Mutable per-track state container used by the main() frame loop. Holds
# position, appearance (BGR colour), PKT template cache, and the last
# last-matched detection metadata. Phase history lives in the docstring below.
class Track:
    """Per-particle tracking state for a single active trajectory.

    Each Track instance holds the complete mutable state for one particle
    across frames: current and previous positions, appearance model (BGR
    color), missed-detection counter, PKT (Predictive Kernel Tracking)
    template cache, and the last
    confirmed detection coordinates used for cost-matrix seeding and
    recovery searches.

    Fields are updated in-place each frame by the assignment, spawn, and
    prune logic inside ``main``.  The SoA mirror ``TrackArrays`` is
    rebuilt from this list before every vectorized cost-matrix call.

    Pipeline context:
        Called by: main (spawn, update, prune loops)
        Calls: (data class — no methods)
        Stage: data_structure

    Args:
        id: Unique integer track identifier, assigned at spawn. [int]
        x: Current centroid x position in image pixels. [float]
        y: Current centroid y position in image pixels. [float]
        r: Current particle radius in pixels. [float]
        color_bgr: BGR appearance tuple used for the color cost component.
            Overwritten each matched frame (no temporal smoothing yet —
            see Phase 4C colour vulnerability). [Tuple[float,float,float]]
        missed: Consecutive frames without a matched detection.  Track is
            pruned when this exceeds ``--miss-tolerance``. [int]
        px: Previous-frame centroid x (used for velocity estimation and PKT
            template alignment). None until the second matched frame. [float]
        py: Previous-frame centroid y. None until the second matched frame.
            [float]
        pr: Previous-frame radius. None until the second matched frame.
            [float]
        pkt_template: Grayscale NCC template cached for PKT early-lock.
            Set by ``predictive_kernel_early_lock``; may lock to previous
            positions when coordinates shift (PKT bug — see CLAUDE.md).
            [np.ndarray]
        pkt_tcx: Sub-pixel x offset of the PKT template centre relative
            to (px, py). [float]
        pkt_tcy: Sub-pixel y offset of the PKT template centre relative
            to (px, py). [float]
        last_det_pos: ``(x, y)`` pixel position of the most recent
            *matched* detection (not predicted).  Used by kernel recovery
            and persistence bonus logic. [tuple]
        last_det_r: Radius of the most recent matched detection, mirroring
            ``last_det_pos``. [float]

    Phase history:
        - Original: id, x, y, r, color_bgr, missed, px/py/pr, pkt fields.
        - Phase 5 D: last_det_pos, last_det_r added for persistence bonus
          and recovery seeding.
    """
    id: int
    x: float
    y: float
    r: float
    color_bgr: Tuple[float,float,float]
    missed: int = 0
    px: Optional[float] = None
    py: Optional[float] = None
    pr: Optional[float] = None
    pkt_template: Optional[np.ndarray] = None
    pkt_tcx: Optional[float] = None
    pkt_tcy: Optional[float] = None
    last_det_pos: Optional[tuple] = None   # (x, y) of last matched detection
    last_det_r: Optional[float] = None     # radius of last matched detection
    # [In/outflow] lifecycle + row-tag state (cbtv.inoutflow). Written on
    # every run but only READ when an in/outflow flag is set.
    vx: float = 0.0              # velocity from the last two matches (px/frame)
    vy: float = 0.0
    lm_x: Optional[float] = None           # position + frame of the last match
    lm_y: Optional[float] = None
    lm_frame: Optional[int] = None
    birth_frame: int = -1
    birth_reason: str = ""
    tentative: bool = False      # --confirm-hits: unconfirmed birth
    hits: int = 0                # consecutive matches since birth (tentative only)
    k_need: int = 1              # matches needed to confirm
    dropped: bool = False        # unconfirmed birth discarded (rows never written)
    coast_state: str = "matched" # row state while coasting
    occl_n: int = 0              # frames coasted inside an occluder since last match
    death_reason: str = ""
    n_matched: int = 0           # rows written, by kind
    n_coast: int = 0
    last_row_frame: int = -1
    # [--color-source track] saturation-weighted LAB colour + saturation,
    # sampled at the matched position (EMA'd with --colour-ema).
    lab_w: Optional[tuple] = None
    S_w: float = 0.0

# === [DATA-2] TrackArrays — SoA mirror of the Track list (see Flowcharts 8, 10, 20) ===
# Structure-of-Arrays projection rebuilt from `tracks: List[Track]` before every
# vectorised cost-matrix / kernel-build call. Holds ids, positions, radii,
# colours, missed counts, and last_det_pos in aligned ndarrays so the
# GPU path can operate on contiguous memory without per-track Python overhead.
class TrackArrays:
    """Structure-of-Arrays representation for vectorized track operations"""

    def __init__(self, tracks: List[Track]):
        """Convert list of Track objects to SoA format"""
        if not tracks:
            self.ids = np.array([], dtype=np.int32)
            self.positions = np.zeros((0, 2), dtype=np.float32)
            self.radii = np.array([], dtype=np.float32)
            self.colors = np.zeros((0, 3), dtype=np.float32)
            self.missed = np.array([], dtype=np.int32)
            self.last_det_pos = np.zeros((0, 2), dtype=np.float32)
            self.count = 0
        else:
            self.ids = np.array([t.id for t in tracks], dtype=np.int32)
            self.positions = np.array([[t.x, t.y] for t in tracks], dtype=np.float32)
            self.radii = np.array([t.r for t in tracks], dtype=np.float32)
            self.colors = np.array([list(t.color_bgr) for t in tracks], dtype=np.float32)
            self.missed = np.array([t.missed for t in tracks], dtype=np.int32)
            self.last_det_pos = np.array([[t.last_det_pos[0], t.last_det_pos[1]]
                                          if t.last_det_pos is not None else [-1.0, -1.0]
                                          for t in tracks], dtype=np.float32)
            self.count = len(tracks)
        # [--color-source track] stored colours; None unless every track has one
        self.lab_w = self.S_w = None
        if tracks and all(t.lab_w is not None for t in tracks):
            self.lab_w = np.array([t.lab_w for t in tracks], dtype=np.float64)
            self.S_w = np.array([t.S_w for t in tracks], dtype=np.float64)

    def to_track_list(self) -> List[Track]:
        """Convert SoA back to list of Track objects"""
        tracks = []
        for i in range(self.count):
            t = Track(
                id=int(self.ids[i]),
                x=float(self.positions[i, 0]),
                y=float(self.positions[i, 1]),
                r=float(self.radii[i]),
                color_bgr=tuple(float(c) for c in self.colors[i]),  # type: ignore[arg-type]
                missed=int(self.missed[i])
            )
            tracks.append(t)
        return tracks

    def update_from_matches(self, match_indices, detections, det_colors, smooth_r=0.85):
        """
        Update tracks from matched detections

        Args:
            match_indices: List of (track_idx, det_idx) pairs
            detections: List of (x, y, r, score) tuples
            det_colors: List of (b, g, r) tuples
            smooth_r: Smoothing factor for radius (0.85 means 85% old, 15% new)
        """
        for (track_idx, det_idx) in match_indices:
            x, y, r, _ = detections[det_idx]
            self.positions[track_idx, 0] = x
            self.positions[track_idx, 1] = y
            self.radii[track_idx] = smooth_r * self.radii[track_idx] + (1 - smooth_r) * r
            self.colors[track_idx] = np.array(det_colors[det_idx], dtype=np.float32)
            self.missed[track_idx] = 0

    def increment_missed(self, unmatched_indices):
        """Increment missed counter for unmatched tracks"""
        for idx in unmatched_indices:
            self.missed[idx] += 1

    def get_detections_format(self):
        """Convert to detection format [(x, y, r, _), ...] for compatibility"""
        return [(float(self.positions[i, 0]), float(self.positions[i, 1]),
                 float(self.radii[i]), 1.0) for i in range(self.count)]

class KernelCache:
    """
    Persistent kernel storage across frames to prevent O(M) kernel rebuilds.
    Kernels are retained unless the track moves significantly or too many frames pass.
    """

    def __init__(self, max_frames=5, max_dist=10.0):
        """Initialize kernel cache"""
        self.cache = {}  # {track_id: {'data': (kernel, cx, cy), 'pos': (x,y), 'frame': int}}
        self.max_frames = max_frames
        self.max_dist = max_dist
        self.current_frame = 0

    def set_frame(self, frame_idx):
        """Set current frame index for cache age tracking."""
        self.current_frame = frame_idx

    def get(self, track_id, current_pos=None):
        """
        Retrieve kernel for track if still valid.
        """
        if track_id not in self.cache:
            return None

        entry = self.cache[track_id]

        # Invalidate if too old
        if self.current_frame - entry['frame'] > self.max_frames:
            del self.cache[track_id]
            return None

        # Invalidate if track moved too much since kernel was built
        if current_pos is not None:
            dist = math.hypot(current_pos[0] - entry['pos'][0], current_pos[1] - entry['pos'][1])
            if dist > self.max_dist:
                del self.cache[track_id]
                return None

        return entry['data']

    def put(self, track_id, kernel_data, pos):
        """
        Store kernel in cache with its build position and frame.
        """
        self.cache[track_id] = {
            'data': kernel_data,
            'pos': pos,
            'frame': self.current_frame
        }

    def clear(self):
        """Clear all cached kernels"""
        self.cache.clear()

