"""Validation tooling for tracks on real footage without ground truth (a consensus reference from other linkers).

Modules (each usable from Python or through ``python -m tools.realdata <command>``):
  tracksio   csv I/O (detections, tracks, gt), sequence manifests, sha256
  decimate   temporal decimation (offsets 0..k-1), reversal, lossless video writer
  linkers    CBTV-free reference linkers (relaxation baseline, mutual-NN, Hungarian) and track snapping
  fb         forward-backward identity check
  consensus  consensus pseudo-GT across members and decimation offsets + provenance / circularity guard
  score      link-level scoring of a tracker against a reference (pseudo-GT or true GT)
  audit      blinded audit sampler (crops + annotator csvs) and Cohen's kappa / reference error rate
  synth      synthetic high-frame-rate clip with known GT (validation of the validator)
  validate   the validation table: pseudo-GT error vs true-GT error
"""
