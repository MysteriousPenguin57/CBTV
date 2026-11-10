"""bt (the default preset) vs ship-v4: preset wiring, and BT goldens on GPU and CPU.

golden_gpu_tracks_bt.csv per fixture is produced on GPU (`tools/run_fixtures.py gpu <dir> --preset bt`);
until it is committed the golden tests SKIP (not pass)."""
import pytest
from conftest import FIX, FIXTURES, first_diff_message, run_fixture

BT_FLAGS = ["--coast-predict", "flow", "--exit-kill", "1", "--color-source", "track"]


def test_bt_is_ship_v4_plus_three_flags():
    import run_cbtv
    assert run_cbtv.BT_CONFIG == run_cbtv.SHIP_CONFIG_V4 + BT_FLAGS
    assert run_cbtv.PRESETS == {"bt": run_cbtv.BT_CONFIG, "ship-v4": run_cbtv.SHIP_CONFIG_V4}


def test_default_preset_is_bt_and_ship_config_aliases_it():
    import run_cbtv
    assert run_cbtv.DEFAULT_PRESET == "bt"
    assert run_cbtv.SHIP_CONFIG is run_cbtv.PRESETS["bt"]


def _bt_golden(fixture):
    gold = FIX / fixture / "golden_gpu_tracks_bt.csv"
    if not gold.exists():
        pytest.skip("%s: golden_gpu_tracks_bt.csv not committed" % fixture)
    return gold


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_gpu_bt_equals_bt_golden(fixture, tmp_path):
    gold = _bt_golden(fixture)
    out = run_fixture(fixture, tmp_path / "gpu_bt", use_gpu=True, preset="bt")
    assert out.read_bytes() == gold.read_bytes(), "%s GPU bt != BT golden\n%s" % (fixture, first_diff_message(out, gold))


@pytest.mark.slow
@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_cpu_bt_equals_bt_golden(fixture, tmp_path):
    gold = _bt_golden(fixture)
    out = run_fixture(fixture, tmp_path / "cpu_bt", use_gpu=False, preset="bt")
    assert out.read_bytes() == gold.read_bytes(), "%s CPU bt != BT golden\n%s" % (fixture, first_diff_message(out, gold))
