"""Gate for MBROrthoColumnExperiment and MBROrthogonalityExperiment.

docs/qsim/mbr_redesign.md, section 7, step 4. Code organization, not physics:
these check that the code loads and runs and that outputs have the expected
shape and dtype. The numbers are compared, non-blocking, with the old
``reconstruct_orthogonality`` on the same raw files.

- Job class: the Program builds, compiles and acquires in mock mode on both
  pinned config sets, through ``Orthogonality.acquire(runner)``. Its pulses
  are the old orthogonality program's at q = 0, and the old propagator
  program's (with the correction on the pulse) at q > 0.
- Assembled class: on converted fixture data (the September N=3 set, through
  ``tools/migrate_mbr_jobs.py``), ``from_manifest``, ``analyze``,
  ``display`` and ``save`` finish.

Run:  pixi run python -m pytest tests/test_mbr_orthogonality.py -v
"""
from copy import deepcopy

import matplotlib
import numpy as np
import pytest
from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.deprecated.legacy_mbr import MBROrthogonalityExperiment as LegacyOrthogonality
from experiments.qsim.mbr_campaign import (
    mbr_defaults,
    mock_station,
    pinned_config_set,
    pinned_sets,
)
from experiments.qsim.mbr_ortho_column import MBROrthoColumnExperiment, MBROrthoColumnProgram
from experiments.qsim.mbr_orthogonality import (
    EncodingOrthogonalityProgram,
    MBROrthogonalityExperiment,
)
from experiments.qsim.mbr_propagator import EncodingPropagatorProgram
from experiments.qsim.mbr_stark_cal import RAMSEY_PHASES
from experiments.saved_jobs import load_job
from tests.asm_golden import render
from tests.mbr_reference import ORTHOGONALITY_IDS, converted_orthogonality

matplotlib.use("Agg")

OCCUPATIONS = [[0, 0, 0, 0, 3], [1, 0, 0, 0, 2], [0, 1, 0, 0, 2]]
SWAP_STORS = [1, 2, 3, 4]


# --------------------------------------------------------------------------
# Job class, mock mode
# --------------------------------------------------------------------------


@pytest.fixture(scope="module", params=sorted(pinned_sets()))
def station(request):
    return request.param, mock_station(**pinned_config_set(request.param))


def _runner(st):
    assert st.is_mock, "refusing to acquire against real instruments"
    return CharacterizationRunner(
        station=st, ExptClass=MBROrthoColumnExperiment,
        default_expt_cfg=mbr_defaults(SWAP_STORS, reps=10), show=False)


@pytest.mark.parametrize("cycle", [0, 4])
def test_orthogonality_acquires_one_column_per_occupation(station, cycle):
    _, st = station
    ortho = MBROrthogonalityExperiment(OCCUPATIONS, SWAP_STORS, cycle=cycle, reps=10)
    ortho.acquire(_runner(st), batch_size=2)
    assert [child.initial_occupation for child in ortho.children] == [
        tuple(o) for o in OCCUPATIONS]
    for child in ortho.children:
        assert child.ProgramClass is MBROrthoColumnProgram
        assert child.cycle == cycle
        assert np.asarray(child.data["avgi"]).shape == (len(OCCUPATIONS), len(RAMSEY_PHASES))
        assert child.cfg.expt.calibration_manifest is None
        data = child.analyze()
        assert data["complex_return"].shape == (len(OCCUPATIONS),)
        assert data["complex_return"].dtype == complex
    data = ortho.analyze()
    assert data.matrix.shape == (len(OCCUPATIONS), len(OCCUPATIONS))
    assert data.matrix.dtype == complex


def _program_cfg(st, **expt):
    cfg = AttrDict(deepcopy(st.hardware_cfg))
    cfg.device.storage._ds_storage = st.ds_storage
    cfg.device.storage._ds_floquet = st.ds_floquet
    cfg.expt = AttrDict(dict(mbr_defaults(SWAP_STORS, reps=10), **expt))
    return cfg


def test_ortho_column_pulses_at_q0_are_the_old_orthogonality_program(station):
    """At one point of the sweep, the new program compiles to the old one's ASM."""
    _, st = station
    initial, decoder = OCCUPATIONS[0], OCCUPATIONS[1]
    new_expt = MBROrthoColumnExperiment.job_config(initial, OCCUPATIONS, SWAP_STORS)
    new = MBROrthoColumnProgram(soccfg=st.soccfg, cfg=_program_cfg(
        st, **new_expt, decoder_occupation=decoder, ramsey_phase=[180., 90.]))
    old = EncodingOrthogonalityProgram(soccfg=st.soccfg, cfg=_program_cfg(
        st, **new_expt, orthogonality_decoder_occupations=OCCUPATIONS,
        orthogonality_analyzer_phases=[0., 90.],
        decoder_analyzer_row=2 * OCCUPATIONS.index(decoder) + 1,
        spectroscopy_prep_phase=180.))
    assert render(new) == render(old)
    assert len(render(new).splitlines()) > 100, "compiled to almost nothing"


def test_ortho_column_pulses_at_q_are_the_old_propagator_program(station):
    """At q > 0, with the decoder correction on the pulse."""
    _, st = station
    initial, decoder, cycle = OCCUPATIONS[0], OCCUPATIONS[2], 4
    corrections = [7., -3., 11.]
    new_expt = MBROrthoColumnExperiment.job_config(
        initial, OCCUPATIONS, SWAP_STORS, cycle=cycle, decoder_phases_deg=corrections)
    new = MBROrthoColumnProgram(soccfg=st.soccfg, cfg=_program_cfg(
        st, **new_expt, decoder_occupation=decoder, ramsey_phase=[0., 90.]))
    old = EncodingPropagatorProgram(soccfg=st.soccfg, cfg=_program_cfg(
        st, **new_expt, propagator_occupations=OCCUPATIONS,
        propagator_decoder_phase_correction_deg=corrections,
        phase_correction_location="pulse",
        cycle_decoder_analyzer=[cycle, *decoder, 90.],
        spectroscopy_prep_phase=0.))
    assert render(new) == render(old)


def test_job_overrides_record_the_calibration():
    """Each column carries the decoder corrections and the calibration manifest."""
    class Calibration:
        manifest_path = "calibration.yaml"

        def phase_correction(self, branches):
            return AttrDict(dict(phase_by_occupation={tuple(o): 10. * i
                                                      for i, o in enumerate(OCCUPATIONS)}))

    ortho = MBROrthogonalityExperiment(OCCUPATIONS, SWAP_STORS, cycle=20,
                                       calibration=Calibration())
    for initial, override in zip(OCCUPATIONS, ortho.job_overrides()):
        assert override["spectroscopy_occupations"] == initial
        assert override["decoder_phase_per_cycle_deg"] == [0., 10., 20.]
        assert override["floquet_cycle"] == 20
        assert override["calibration_manifest"] == "calibration.yaml"

    ortho.calibration.manifest_path = None
    with pytest.raises(ValueError, match="save"):
        ortho.job_overrides()


# --------------------------------------------------------------------------
# Assembled class, on converted fixture data
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def september(tmp_path_factory):
    return converted_orthogonality(tmp_path_factory.mktemp("orthogonality"))


def test_converted_set_is_35_ortho_columns(september):
    assert len(september.children) == 35
    assert september.cycle == 0
    for child in september.children:
        assert isinstance(child, MBROrthoColumnExperiment)
        assert np.asarray(child.data["avgi"]).shape == (35, 4)
        assert child.data["complex_return"].dtype == complex
        assert child.cfg.expt.decoder_phase_per_cycle_deg == [0.] * 35
    assert [child.initial_occupation for child in september.children] == september.occupations


def test_analyze_display_save(september, tmp_path):
    data = september.analyze()
    assert data.matrix.shape == (35, 35)
    assert data.matrix.dtype == complex
    np.testing.assert_array_equal(data.matrix, data.raw_matrix)
    assert data.mode_labels == ["M1", "S2", "S3", "S4", "S5"]
    assert data.column_leakage.shape == (35,)
    assert september.display() is not None
    assert september.children[0].display() is not None

    reloaded = MBROrthogonalityExperiment.from_manifest(september.save(directory=tmp_path))
    assert reloaded.occupations == september.occupations
    assert reloaded.calibration is None
    np.testing.assert_array_equal(reloaded.analyze().matrix, data.matrix)


def test_from_children_puts_columns_in_decoder_order(september):
    shuffled = september.children[::-1]
    ortho = MBROrthogonalityExperiment.from_children(
        shuffled, job_ids=[f"id{i}" for i in range(35)])
    assert ortho.children == september.children
    assert ortho.job_ids == [f"id{34 - i}" for i in range(35)]


def test_an_incomplete_set_is_refused(september):
    with pytest.raises(ValueError, match="decoded occupations"):
        MBROrthogonalityExperiment.from_children(september.children[:-1])


@pytest.mark.xfail(strict=False, reason="physics audit pending")
def test_matrix_matches_the_old_reconstruction(september):
    old = LegacyOrthogonality.reconstruct_orthogonality(
        [load_job(job_id) for job_id in ORTHOGONALITY_IDS])
    assert [tuple(o) for o in old.occupations] == september.occupations
    data = september.analyze()
    for key in ("matrix", "normalized_power", "column_leakage"):
        np.testing.assert_allclose(data[key], old[key], rtol=0, atol=1e-12)
