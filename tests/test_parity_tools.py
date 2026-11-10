"""Unit tests for tools/parity/diff_dumps.py and diff_tracks.py on hand-built inputs."""
import diff_dumps
import diff_tracks
import numpy as np
import pandas as pd


def _npz(path, C, valid, tids=None, dets=None):
    M, N = C.shape
    np.savez(path, C=np.asarray(C, np.float64), track_ids=np.asarray(tids if tids is not None else np.arange(M) + 100),
             det_idx=np.asarray(dets if dets is not None else np.arange(N)), valid_pairs=np.asarray(valid, bool))


def _dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    return a, b


def test_identical_dumps_exit_zero(tmp_path, capsys):
    a, b = _dirs(tmp_path)
    C = np.array([[0.1, 1e6], [0.5, 0.2]]); v = C < 1
    for d in (a, b):
        _npz(d / "C_00001.npz", C, v)
    sys_argv = ["x", str(a), str(b)]
    import sys
    old = sys.argv; sys.argv = sys_argv
    try:
        assert diff_dumps.main() == 0
    finally:
        sys.argv = old
    assert "IDENTICAL" in capsys.readouterr().out


def test_first_difference_reported_with_pair_and_valid_bits(tmp_path, capsys):
    a, b = _dirs(tmp_path)
    C = np.array([[0.1, 1e6], [0.5, 0.2]]); v = C < 1
    _npz(a / "C_00001.npz", C, v); _npz(b / "C_00001.npz", C, v)
    C2 = C.copy(); v2 = v.copy(); C2[1, 0] = 0.9; v2[0, 1] = True; C2[0, 1] = 0.7
    _npz(a / "C_00002.npz", C, v); _npz(b / "C_00002.npz", C2, v2)
    import sys
    old = sys.argv; sys.argv = ["x", str(a), str(b), "--summary", str(tmp_path / "s.csv")]
    try:
        rc = diff_dumps.main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert rc == 1 and "first differing frame: 2" in out
    assert "track_id=100 det_idx=1" in out and "valid_A=False valid_B=True" in out   # row-major first diff
    s = pd.read_csv(tmp_path / "s.csv")
    assert list(s["frame"]) == [1, 2] and s.loc[0, "n_c_diff"] == 0 and s.loc[1, "n_valid_diff"] == 1
    assert s.loc[1, "max_abs_dC"] > 0.4


def test_compare_frame_counts_finite_inf_mismatch():
    C = np.array([[0.1, np.inf]]); D = np.array([[0.1, 0.3]]); v = np.array([[True, False]])
    a = dict(C=C, valid_pairs=v, track_ids=np.array([1]), det_idx=np.array([0, 1]))
    b = dict(C=D, valid_pairs=v, track_ids=np.array([1]), det_idx=np.array([0, 1]))
    row, first = diff_dumps.compare_frame(a, b)
    assert row["n_finite_diff"] == 1 and row["n_c_diff"] == 0 and first["det_idx"] == 1


def _tracks(path, rows):
    pd.DataFrame(rows, columns=["frame", "track_id", "x", "y", "r", "missed"]).to_csv(path, index=False)


def test_diff_tracks_identical_and_different(tmp_path):
    p, q, r = tmp_path / "p.csv", tmp_path / "q.csv", tmp_path / "r.csv"
    rows = [(0, 1, 10.0, 10.0, 7.0, 0), (1, 1, 11.0, 10.0, 7.0, 0), (1, 2, 50.0, 50.0, 7.0, 0)]
    _tracks(p, rows); _tracks(q, rows)
    s = diff_tracks.compare(p, q)
    assert s["byte_identical"] and s["shared"] == 3 and s["only_A"] == 0 and s["first_diff_frame"] is None
    rows2 = rows[:2] + [(1, 3, 50.5, 50.0, 7.0, 0)]                      # id 2 -> 3 at frame 1
    _tracks(r, rows2)
    s = diff_tracks.compare(p, r)
    assert not s["byte_identical"] and (s["shared"], s["only_A"], s["only_B"]) == (2, 1, 1)
    assert s["first_diff_frame"] == 1
