# -*- coding: utf-8 -*-
"""Gate for MBRStarkCalExperiment and MBRCalibrationSetExperiment.

docs/qsim/mbr_redesign.md, section 7, step 2. Code organization, not physics:
these check that the code loads and runs and that outputs have the expected
shape and dtype. The numbers are pinned separately, and non-blocking, by
``test_stark_cal_baseline_matches`` in ``test_mbr_analysis_golden.py``.

- Job class: the Program builds, compiles and acquires in mock mode, on both
  pinned config sets (``gauss`` and ``preload_flattop`` swaps), through
  ``CalibrationSet.acquire(runner)``.
- Assembled class: on converted fixture data (two occupations of the
  September N=3 calibration, through ``tools/migrate_mbr_jobs.py``),
  construct, ``from_manifest``, ``analyze``, ``display`` and ``save`` finish.

Run:  pixi run python -m pytest tests/test_mbr_stark_cal.py -v
"""
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.mbr_calibration_set import MBRCalibrationSetExperiment
from experiments.qsim.mbr_campaign import (
    mbr_defaults,
    mock_station,
    pinned_config_set,
    pinned_sets,
)
from experiments.qsim.mbr_stark_cal import (
    RAMSEY_PHASES,
    MBRStarkCalExperiment,
    MBRStarkCalProgram,
)
from experiments.qsim.mbr_phase_correction import (
    EntireFloquetCyclePhaseCalibrationProgram,
)
from tests.mbr_reference import STARK_CAL_IDS, migration_tool

matplotlib.use("Agg")

OCCUPATIONS = [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2]]
SWAP_STORS = [1, 2, 3, 4]
CYCLE_PAIRS = [0, 1, 2]


# --------------------------------------------------------------------------
# Job class, mock mode
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", params=sorted(pinned_sets()))
def station(request):
    return request.param, mock_station(**pinned_config_set(request.param))


@pytest.fixture(scope="module")
def acquired(station):
    """One CalibrationSet acquired on a mock station (local, no queue)."""
    _, st = station
    assert st.is_mock, "refusing to acquire against real instruments"
    runner = CharacterizationRunner(
        station=st, ExptClass=MBRStarkCalExperiment,
        default_expt_cfg=mbr_defaults(SWAP_STORS, reps=10), show=False)
    calibration = MBRCalibrationSetExperiment(
        OCCUPATIONS, CYCLE_PAIRS, SWAP_STORS, sync_cycles=10, reps=10)
    calibration.acquire(runner, batch_size=2)
    return calibration


def test_acquire_runs_one_job_per_occupation(acquired):
    assert [child.occupation for child in acquired.children] == [
        tuple(o) for o in OCCUPATIONS]
    for child in acquired.children:
        assert isinstance(child, MBRStarkCalExperiment)
        assert child.ProgramClass is MBRStarkCalProgram
        assert np.asarray(child.data["avgi"]).shape == (len(CYCLE_PAIRS), len(RAMSEY_PHASES))
        assert np.asarray(child.data["xpts"]).tolist() == RAMSEY_PHASES
        assert list(child.data["ypts"]) == CYCLE_PAIRS


def test_job_analyze_never_raises_on_unfittable_data(acquired):
    """Mock data is all zeros. The job still analyzes, so it would still save."""
    child = acquired.children[0]
    data = child.analyze()
    assert data["complex_return"].shape == (len(CYCLE_PAIRS),)
    assert data["complex_return"].dtype == complex
    assert np.isnan(data["phase_per_cycle"])
    with pytest.raises(ValueError, match="no phase fit"):
        acquired.analyze()


def test_acquire_rejects_the_wrong_job_class(station):
    _, st = station
    runner = CharacterizationRunner(station=st, ExptClass=object,
                                    default_expt_cfg={}, show=False)
    calibration = MBRCalibrationSetExperiment(OCCUPATIONS, CYCLE_PAIRS, SWAP_STORS)
    with pytest.raises(TypeError, match="MBRStarkCalExperiment"):
        calibration.acquire(runner)


def test_old_program_is_a_subclass():
    """The old one-analyzer-phase program now takes its pulses from the new one."""
    assert issubclass(EntireFloquetCyclePhaseCalibrationProgram, MBRStarkCalProgram)


# --------------------------------------------------------------------------
# Assembled class, on converted fixture data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Two occupations of September N=3, converted with the migration script."""
    root = tmp_path_factory.mktemp("converted")
    calibration = migration_tool().migrate_stark_cal(
        STARK_CAL_IDS[:4], out_root=root, load_shots=True, notes="gate")
    return root, calibration


def test_migration_writes_the_new_layout(converted):
    root, calibration = converted
    files = sorted((root / "converted_data").glob("*.h5"))
    assert len(files) == 2
    with h5py.File(files[0], "r") as handle:
        assert handle.attrs["job_class"] == "MBRStarkCalExperiment"
        assert len(json.loads(handle.attrs["converted_from"])["job_ids"]) == 2
        assert "derived_params" in handle.attrs
        assert handle["avgi"].shape == (65, 4)
        assert handle["idata"].shape[0] == 4 * 65
        assert handle["complex_return"].dtype == complex
    assert sorted(p.suffix for p in (root / "assembled_data").iterdir()) == [".h5", ".yaml"]


def test_migration_keeps_the_shot_order(converted):
    """Row k of the new shots is point (cycle k // 4, ramsey phase k % 4)."""
    from experiments.job_paths import resolve_job_paths
    from experiments.saved_jobs import load_h5

    _, calibration = converted
    old = resolve_job_paths(STARK_CAL_IDS[:2])
    _, phi0 = load_h5(old[STARK_CAL_IDS[0]], load_shots=True)
    _, phi90 = load_h5(old[STARK_CAL_IDS[1]], load_shots=True)
    with h5py.File(calibration.children[0].fname, "r") as handle:
        new = handle["idata"][()]
    cycle = 7
    np.testing.assert_array_equal(new[4 * cycle + 0], phi0["idata"][2 * cycle + 0])
    np.testing.assert_array_equal(new[4 * cycle + 1], phi0["idata"][2 * cycle + 1])
    np.testing.assert_array_equal(new[4 * cycle + 2], phi90["idata"][2 * cycle + 0])
    np.testing.assert_array_equal(new[4 * cycle + 3], phi90["idata"][2 * cycle + 1])


def test_from_manifest_analyze_display_save(converted, tmp_path):
    root, saved = converted
    calibration = MBRCalibrationSetExperiment.from_manifest(saved.manifest_path)
    assert calibration.manifest_path == Path(saved.manifest_path)
    assert all(isinstance(c, MBRStarkCalExperiment) for c in calibration.children)

    data = calibration.analyze()
    assert data.complex_returns.shape == (2, 65)
    assert data.complex_returns.dtype == complex
    assert data.phase_mod180.shape == (2,)
    assert data.phase_mod180.dtype == float
    assert data.physical_cycles.tolist() == list(range(0, 130, 2))

    summary = calibration.display()
    assert summary is not None

    correction = calibration.phase_correction(cycle_branches=0)
    assert set(correction.phase_by_occupation) == set(calibration.occupations)
    assert correction.calibration_manifest == str(saved.manifest_path)
    occupation = calibration.occupations[0]
    assert calibration.phase_for(occupation) == correction.phase_by_occupation[occupation]

    manifest_path = calibration.save(directory=tmp_path)
    reloaded = MBRCalibrationSetExperiment.from_manifest(manifest_path)
    assert reloaded.occupations == calibration.occupations
    assert reloaded.job_ids == calibration.job_ids
    np.testing.assert_array_equal(reloaded.analyze().phase_mod180, data.phase_mod180)
