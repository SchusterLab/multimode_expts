# -*- coding: utf-8 -*-
"""Gate for MBRTimeTraceExperiment and MBRSpectrumExperiment.

docs/qsim/mbr_redesign.md, section 7, step 3. Code organization, not physics:
these check that the code loads and runs and that outputs have the expected
shape and dtype. The numbers are pinned separately, and non-blocking, by the
quick-plot and complete-basis baselines in ``test_mbr_analysis_golden.py``.

- Job class: the Program builds, compiles and acquires in mock mode on both
  pinned config sets, diagonal through ``Spectrum.acquire(runner)`` and one
  off-diagonal job directly. Its pulses are the old diagonal program's.
- Assembled class: on converted fixture data (the August quick-plot set,
  through ``tools/migrate_mbr_jobs.py``), ``from_manifest``, ``analyze``,
  ``display`` and ``save`` finish, and the calibration link survives a save.

Run:  pixi run python -m pytest tests/test_mbr_time_trace.py -v
"""
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.mbr_campaign import (
    mbr_defaults,
    mock_station,
    pinned_config_set,
    pinned_sets,
)
from experiments.qsim.mbr_ramsey import NPhotonHamiltonianSpectroscopyProgram
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_stark_cal import RAMSEY_PHASES
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment, MBRTimeTraceProgram
from tests.asm_golden import render
from tests.mbr_reference import (
    CHARACTERIZATION_ANALYSIS,
    STARK_CAL_IDS,
    converted_quickplot,
    migration_tool,
)

matplotlib.use("Agg")

OCCUPATIONS = [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2]]
SWAP_STORS = [1, 2, 3, 4]
CYCLES = [0, 2, 4]


# --------------------------------------------------------------------------
# Job class, mock mode
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", params=sorted(pinned_sets()))
def station(request):
    return request.param, mock_station(**pinned_config_set(request.param))


def _runner(st):
    assert st.is_mock, "refusing to acquire against real instruments"
    return CharacterizationRunner(
        station=st, ExptClass=MBRTimeTraceExperiment,
        default_expt_cfg=mbr_defaults(SWAP_STORS, reps=10), show=False)


def test_spectrum_acquires_one_diagonal_job_per_occupation(station):
    _, st = station
    spectrum = MBRSpectrumExperiment(OCCUPATIONS, CYCLES, SWAP_STORS, reps=10)
    spectrum.acquire(_runner(st), batch_size=2)
    assert [child.initial_occupation for child in spectrum.children] == [
        tuple(o) for o in OCCUPATIONS]
    for child in spectrum.children:
        assert child.ProgramClass is MBRTimeTraceProgram
        assert child.final_occupation == child.initial_occupation
        assert np.asarray(child.data["avgi"]).shape == (len(CYCLES), len(RAMSEY_PHASES))
        assert child.cfg.expt.calibration_manifest is None
        data = child.analyze()
        assert data["complex_return"].shape == (len(CYCLES),)
        assert data["complex_return"].dtype == complex


def test_an_off_diagonal_job_acquires(station):
    _, st = station
    initial, final = OCCUPATIONS
    override = MBRTimeTraceExperiment.job_config(initial, final, CYCLES, SWAP_STORS,
                                                 phase_per_cycle_deg=12.5, reps=10)
    (child,) = _runner(st).execute(overrides=[override])
    assert child.initial_occupation == tuple(initial)
    assert child.final_occupation == tuple(final)
    assert child.analyze()["complex_return"].shape == (len(CYCLES),)


def test_time_trace_pulses_are_the_old_diagonal_program(station):
    """At one point of the sweep, the new program compiles to the old one's ASM."""
    _, st = station
    base = AttrDict(deepcopy(st.hardware_cfg))
    base.device.storage._ds_storage = st.ds_storage
    base.device.storage._ds_floquet = st.ds_floquet
    expt = AttrDict(mbr_defaults(SWAP_STORS, reps=10))
    expt.update(MBRTimeTraceExperiment.job_config(
        OCCUPATIONS[0], OCCUPATIONS[0], CYCLES, SWAP_STORS, phase_per_cycle_deg=7.))
    expt.floquet_cycle = 4

    new_cfg = deepcopy(base)
    new_cfg.expt = AttrDict(dict(deepcopy(expt), ramsey_phase=[180., 90.]))
    old_cfg = deepcopy(base)
    old_cfg.expt = AttrDict(dict(deepcopy(expt), spectroscopy_prep_phase=180.,
                                 spectroscopy_analyzer_phase=90.))
    new = MBRTimeTraceProgram(soccfg=st.soccfg, cfg=new_cfg)
    old = NPhotonHamiltonianSpectroscopyProgram(soccfg=st.soccfg, cfg=old_cfg)
    assert render(new) == render(old)
    assert len(render(new).splitlines()) > 100, "compiled to almost nothing"


def test_job_overrides_record_the_calibration(tmp_path):
    """Each job carries its correction and the calibration manifest path."""
    calibration = migration_tool().migrate_stark_cal(
        STARK_CAL_IDS[:4], out_root=tmp_path, load_shots=False)
    occupations = calibration.occupations
    spectrum = MBRSpectrumExperiment(occupations, CYCLES, SWAP_STORS,
                                     calibration=calibration, cycle_branches=1)
    correction = calibration.phase_correction(1).phase_by_occupation
    for occupation, override in zip(occupations, spectrum.job_overrides()):
        assert override["final_analyzer_phase_per_cycle_deg"] == correction[occupation]
        assert override["calibration_manifest"] == str(calibration.manifest_path)

    calibration.manifest_path = None
    with pytest.raises(ValueError, match="save"):
        spectrum.job_overrides()


# --------------------------------------------------------------------------
# Assembled class, on converted fixture data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def quickplot(tmp_path_factory):
    return converted_quickplot(tmp_path_factory.mktemp("quickplot"))


def test_converted_quickplot_is_four_time_traces(quickplot):
    assert len(quickplot.children) == 4
    for child in quickplot.children:
        assert isinstance(child, MBRTimeTraceExperiment)
        assert np.asarray(child.data["avgi"]).shape[1] == 4
        assert child.data["complex_return"].dtype == complex


@pytest.mark.parametrize("method", ["fft", "matrix_pencil"])
def test_analyze_display_save(quickplot, method, tmp_path):
    data = quickplot.analyze(**{**CHARACTERIZATION_ANALYSIS, "spectrum_method": method})
    n_cycles = len(quickplot.cycles)
    assert data.acquired_reconstruction.A.shape == (4, n_cycles)
    assert data.acquired_reconstruction.A.dtype == complex
    assert data.reconstruction.A.shape == (4, n_cycles)
    assert np.asarray(data.spectrum.measured_local).shape[0] == 4
    assert quickplot.display() is not None
    assert quickplot.display(occupation=0) is not None

    manifest = quickplot.save(directory=tmp_path)
    reloaded = MBRSpectrumExperiment.from_manifest(manifest)
    assert reloaded.occupations == quickplot.occupations
    assert reloaded.calibration is None
    np.testing.assert_array_equal(
        reloaded.analyze(**CHARACTERIZATION_ANALYSIS).acquired_reconstruction.A,
        quickplot.analyze(**CHARACTERIZATION_ANALYSIS).acquired_reconstruction.A)


def test_the_calibration_link_survives_a_save(quickplot, tmp_path):
    calibration = migration_tool().migrate_stark_cal(
        STARK_CAL_IDS[:4], out_root=tmp_path, load_shots=False)
    spectrum = MBRSpectrumExperiment.from_children(
        quickplot.children, calibration=calibration)
    spectrum.analyze(**CHARACTERIZATION_ANALYSIS)
    reloaded = MBRSpectrumExperiment.from_manifest(spectrum.save(directory=tmp_path))
    assert Path(reloaded.calibration.manifest_path) == Path(calibration.manifest_path).resolve()


def test_a_spectrum_refuses_off_diagonal_traces(quickplot):
    child = deepcopy(quickplot.children[0])
    child.cfg.expt.spectroscopy_final_occupations = list(quickplot.children[1].initial_occupation)
    with pytest.raises(ValueError, match="off-diagonal"):
        MBRSpectrumExperiment.from_children([child])
