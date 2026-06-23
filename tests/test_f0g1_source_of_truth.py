"""
Guards the f0g1 / fn-gn+1 single-source-of-truth migration.

Background: f0g1 pulse params were historically scattered across three stores
(`device.manipulate`, the `ds_storage` CSV `M1` row, and `device.multiphoton`).
Calibration steps read one store while their postprocessors wrote another, so the
coarse->chevron->fine->error-amp "search near the previous result" chain silently
broke. The migration makes `hardware_cfg.device.multiphoton` the SOLE store for
f0g1 params; the autocalibrate Manipulate section and the production readers no
longer touch the `ds_storage` `M1` row.

These tests lock that in. Because the calibration pre/post-processors live inside
the (non-importable) notebook and mock QICK returns all-zero data (so real fits
can't drive the postprocessors), the strongest *runnable* guarantee is structural:
after the migration only one store is referenced, so divergence is impossible by
construction. We back that with a mock-station check that the canonical store is
well-formed for index-based access.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "measurement_notebooks" / "guan" / "single_qubit_autocalibrate_v2.py"
KERR = REPO_ROOT / "experiments" / "qsim" / "kerr.py"
QSIM_NB = REPO_ROOT / "measurement_notebooks" / "guan" / "qsim_experiments.py"

# Markers bounding the *refactored* f0g1 calibration cells in the notebook.
MANIP_START = "# # Manipulate"
MANIP_END = "# ## --- Manipulate sections below not refactored ---"


def _refactored_manipulate_section() -> str:
    text = NOTEBOOK.read_text(encoding="utf-8")
    start = text.index(MANIP_START)
    end = text.index(MANIP_END)  # everything below this marker is explicitly not migrated
    assert start < end, "Manipulate section markers out of order"
    return text[start:end]


# --------------------------------------------------------------------------- #
# Structural guards (fast, deterministic, no hardware/config DB needed)
# --------------------------------------------------------------------------- #
def test_refactored_manipulate_section_does_not_touch_ds_storage():
    """The migrated f0g1 cells must not read or write the ds_storage M1 row,
    nor persist it — that's the dead second source of truth."""
    section = _refactored_manipulate_section()
    for forbidden in ("ds_storage", "snapshot_man1_storage_swap", "snapshot_multiphoton_config"):
        assert forbidden not in section, (
            f"Refactored Manipulate section still references {forbidden!r} — "
            "f0g1 params must live solely in hardware_cfg.device.multiphoton."
        )


def test_refactored_manipulate_section_uses_multiphoton():
    """Sanity: the migrated cells do drive off multiphoton fn-gn+1."""
    section = _refactored_manipulate_section()
    assert "multiphoton" in section
    assert "fn-gn+1" in section


def test_production_f0g1_readers_use_multiphoton():
    """Live (non-notebook) f0g1 readers were repointed off the ds_storage M1 row."""
    kerr = KERR.read_text(encoding="utf-8")
    assert "_ds_storage.get_freq('M1')" not in kerr
    assert "multiphoton['pi']['fn-gn+1']['frequency']" in kerr

    qsim = QSIM_NB.read_text(encoding="utf-8")
    assert "ds_storage.get_gain('M1')" not in qsim
    assert "multiphoton['pi']['fn-gn+1']['gain']" in qsim


# --------------------------------------------------------------------------- #
# Runtime guard: the canonical store is well-formed against the real configs
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mock_station():
    """Real configs (main versions), stubbed instruments. Skips if the config
    DB / main versions aren't reachable on this machine."""
    try:
        from experiments import MultimodeStation
        return MultimodeStation(
            user="guan", experiment_name="260618_f0g1_test",
            mock=True, log_measurements=False,
        )
    except Exception as e:  # noqa: BLE001 - environment-dependent bootstrap
        pytest.skip(f"mock station unavailable ({type(e).__name__}: {e})")


def test_multiphoton_fn_gn1_is_wellformed_for_index_access(mock_station):
    """f0g1 access is `multiphoton['pi']['fn-gn+1'][field][man_mode_no-1]`.
    The field arrays must be equal length so a postproc writing freq/length/gain
    for a mode can't IndexError or silently desync across fields."""
    mp = mock_station.hardware_cfg.device.multiphoton
    block = mp["pi"]["fn-gn+1"]
    lengths = {field: len(block[field]) for field in ("frequency", "gain", "length", "sigma", "type")}
    assert len(set(lengths.values())) == 1, (
        f"multiphoton['pi']['fn-gn+1'] field arrays disagree in length: {lengths}"
    )
    # There is intentionally no hpi['fn-gn+1'] slot: the f0g1 half-pi swap is
    # derived (pi_len/2), not stored. If that changes, the chevron/length-rabi
    # postprocs must start writing it — this tripwire flags the schema change.
    assert "fn-gn+1" not in mp.get("hpi", {}), (
        "Unexpected hpi['fn-gn+1'] slot — update the f0g1 postprocs to write h_pi and fix this test."
    )
