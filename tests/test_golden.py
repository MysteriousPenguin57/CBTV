"""GPU run of each fixture must equal the committed golden byte-for-byte (needs GPU 0)."""
import numpy as np
import pytest
from conftest import FIX, FIXTURES, first_diff_message, run_fixture


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_gpu_equals_golden(fixture, tmp_path):
    out = run_fixture(fixture, tmp_path / "gpu", use_gpu=True)
    gold = FIX / fixture / "golden_gpu_tracks.csv"
    assert out.read_bytes() == gold.read_bytes(), "%s GPU != golden\n%s" % (fixture, first_diff_message(out, gold))


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_dump_written_and_does_not_change_tracks(tmp_path):
    dump = tmp_path / "dump"
    out = run_fixture("synth_io", tmp_path / "gpu", use_gpu=True, dump_dir=dump)
    assert out.read_bytes() == (FIX / "synth_io" / "golden_gpu_tracks.csv").read_bytes()
    files = sorted(dump.glob("C_*.npz"))
    assert len(files) == 59, "expected one npz per frame 1..59, got %d" % len(files)
    with np.load(files[9]) as z:                       # C_00010
        assert set(z.files) == {"C", "track_ids", "det_idx", "valid_pairs", "det_xyr"}
        assert z["det_xyr"].shape == (z["C"].shape[1], 3)  # one (x, y, r) per column
        assert z["C"].dtype == np.float64 and z["C"].shape == z["valid_pairs"].shape
