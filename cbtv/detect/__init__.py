"""Detection code used by the shipped detectors (moved here from the research modules).

- :mod:`cbtv.detect.channels`: background subtraction, the channel set and the module-level detection flags
  (ex ``detection_benchmark``).
- :mod:`cbtv.detect.strategies`: the ``ws_multi_vote_gpu`` / ``ws_vgl_log_gpu`` watershed voting strategies
  (ex ``strategy_combination_benchmark``), with CPU fallbacks in
- :mod:`cbtv.detect.gpu_strategies` / :mod:`cbtv.detect.gpu_backend` (CuPy when available, else NumPy/OpenCV).
"""
