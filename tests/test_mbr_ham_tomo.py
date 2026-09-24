"""Gate for MBRHamTomoExperiment.

docs/qsim/mbr_redesign.md, section 7, step 5. Code organization, not physics:
on converted fixture data (the August N=1 propagator set, through
``tools/migrate_mbr_jobs.py``), ``from_manifest``, ``analyze``, ``display``
and ``save`` finish, and the outputs have the expected shape and dtype. The
tomography itself is pinned on synthetic data in
``test_propagator_dynamics.py``; the matrices are compared, non-blocking,
with the old ``reconstruct_propagator`` on the same raw files.

Run:  pixi run python -m pytest tests/test_mbr_ham_tomo.py -v
"""
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from experiments import assembled_data
from experiments.qsim.deprecated.legacy_mbr import MBRPropagatorExperiment as LegacyPropagator
from experiments.qsim.mbr_ham_tomo import MBRHamTomoExperiment
from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
from experiments.saved_jobs import load_job
from tests.mbr_reference import PROPAGATOR_IDS, converted_propagator

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def august(tmp_path_factory):
    return converted_propagator(tmp_path_factory.mktemp("propagator"))


def test_converted_set_is_two_orthogonality_parts(august):
    assert august.cycles == [0, 20]
    assert len(august.occupations) == 5
    for part, cycle in zip(august.children, august.cycles):
        assert isinstance(part, MBROrthogonalityExperiment)
        assert part.cycle == cycle
        assert len(part.children) == 5
        assert part.manifest_path is not None
    # The old jobs left the Stark correction to analysis; only q > 0 feels it.
    columns = august.children[1].children
    assert all(np.any(child.analysis_phase_per_cycle_deg) for child in columns)


def test_analyze_display_save(august, tmp_path):
    data = august.analyze()
    assert data.matrices.shape == (2, 5, 5)
    assert data.matrices.dtype == complex
    assert data.raw_matrices.shape == (2, 5, 5)
    np.testing.assert_array_equal(data.matrices[0], data.raw_matrices[0])
    assert "eigenphase" not in data, "no calibration set, so no tomography"
    assert data.floquet_cycle_us > 0
    assert august.display() is not None

    manifest = august.save(directory=tmp_path)
    reloaded = MBRHamTomoExperiment.from_manifest(manifest)
    assert reloaded.cycles == august.cycles
    assert reloaded.occupations == august.occupations
    assert reloaded.job_ids == august.job_ids
    np.testing.assert_array_equal(reloaded.analyze().matrices, data.matrices)
    arrays, attrs = assembled_data.read_assembled_h5(
        Path(manifest).with_suffix(".h5"))
    assert arrays["matrices"].shape == (2, 5, 5)
    assert attrs["class"] == "MBRHamTomoExperiment"


def test_unsaved_parts_are_refused_by_save(august, tmp_path):
    part = MBROrthogonalityExperiment.from_children(august.children[0].children)
    tomo = MBRHamTomoExperiment.from_parts([part])
    tomo.analyze()
    with pytest.raises(ValueError, match="save"):
        tomo.save(directory=tmp_path)


def test_acquire_is_not_offered(august):
    with pytest.raises(NotImplementedError, match="from_parts"):
        august.acquire(runner=None)


def test_parts_must_share_occupations(august):
    first, second = august.children
    shuffled = MBROrthogonalityExperiment.from_children(second.children)
    shuffled.occupations = shuffled.occupations[::-1]
    with pytest.raises(ValueError, match="occupations"):
        MBRHamTomoExperiment.from_parts([first, shuffled])


def test_two_saves_in_one_second_do_not_collide(august, tmp_path):
    part = august.children[0]
    first = part.save(directory=tmp_path)
    second = part.save(directory=tmp_path)
    assert first != second
    assert first.exists() and second.exists()


@pytest.mark.xfail(strict=False, reason="physics audit pending")
def test_matrices_match_the_old_reconstruction(august):
    old = LegacyPropagator.reconstruct_propagator(
        [load_job(job_id) for job_id in PROPAGATOR_IDS])
    assert [tuple(o) for o in old.occupations] == august.occupations
    assert list(old.cycles) == august.cycles
    data = august.analyze()
    np.testing.assert_allclose(data.raw_matrices, old.raw_matrices, rtol=0, atol=1e-12)
    np.testing.assert_allclose(data.matrices, old.matrices, rtol=0, atol=1e-12)
