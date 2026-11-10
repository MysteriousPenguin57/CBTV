"""CBTV: Colour Bean Tracking Velocimetry for dense granular flows.

Pipeline: detection (exemplars -> NCC -> Bayesian refinement) -> tracking
(LAB kernel built on frame n, NCC-matched on frame n+1 via a 3-stage assignment).

Python API: ``cbtv.track(video, ...) -> pandas.DataFrame`` and ``cbtv.presets()`` (cbtv/api.py).
Entry point: ``python -m cbtv.run_tracker`` (see run_tracker.main).
Detector presets: ``cbtv.detectors.available()``.
"""
from . import gpu  # noqa: F401
from .api import presets, track  # noqa: F401  (public Python API)
