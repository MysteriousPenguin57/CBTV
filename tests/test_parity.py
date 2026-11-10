"""CPU run of each fixture must equal the committed GPU golden (CPU == GPU parity, G2).

After T006 (pair-set fix) all three fixtures are byte-identical on CPU; the ticket's xfail(strict)
for m05_30 / synth_io is therefore NOT applied (it would turn into a strict XPASS failure)."""
from pathlib import Path

import diff_tracks  # tools/parity
import pytest
from conftest import FIX, FIXTURES, first_diff_message, run_fixture


def test_ship_config_matches_run_fixtures_script():
    """tests and tools/run_fixtures.py must use the same ship-v4 and BT flags (they produce the goldens)."""
    import run_cbtv
    src = (Path(__file__).resolve().parent.parent / "tools" / "run_fixtures.py").read_text()
    for tok in run_cbtv.SHIP_CONFIG_V4 + run_cbtv.BT_CONFIG:
        assert '"%s"' % tok in src, "SHIP flag %r missing from tools/run_fixtures.py" % tok


@pytest.mark.slow
@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_cpu_equals_golden_gpu(fixture, tmp_path):
    out = run_fixture(fixture, tmp_path / "cpu", use_gpu=False)
    gold = FIX / fixture / "golden_gpu_tracks.csv"
    if out.read_bytes() != gold.read_bytes():
        s = diff_tracks.compare(out, gold)
        pytest.fail("%s CPU != golden GPU: %s\n%s" % (fixture, s, first_diff_message(out, gold)))
