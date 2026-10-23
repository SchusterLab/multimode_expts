"""CentralBosonAGPExperiment: the direction sector, C_1 and C_2, and its save lifecycle."""
import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")

from experiments.qsim.agp import (  # noqa: E402
    CentralBosonAGPExperiment,
    direction_sector,
    rms_khz,
    sector_dimension,
)
from tests.population_jobs import (  # noqa: E402
    CONFUSION,
    FakeRunner,
    population_job,
    random_labels,
    save_job,
)

CYCLES = [0, 3]


def test_direction_sector_is_complete_ordered_and_uses_physical_ids():
    states = direction_sector(3, [7, 4, 5, 6], 4)
    assert len(states) == 15 and sector_dimension(3, 5) == 35
    assert states == sorted(states)
    assert all(sum(state) == 3 and state[2] > 0 for state in states)
    assert direction_sector(2, [7], 7) == [[0, 2], [1, 1]]
    assert direction_sector(1, [], 0) == [[1]]
    with pytest.raises(ValueError, match="direction 3"):
        direction_sector(2, [4, 5], 3)


def sector_jobs(tmp_path, labels_by_state, swap_stors, direction, confusion=None,
                detunings=None):
    detunings = [0.] * len(swap_stors) if detunings is None else detunings
    jobs = []
    for index, (state, labels) in enumerate(labels_by_state.items()):
        job = population_job(list(state), direction, labels, CYCLES, swap_stors, detunings,
                             confusion=confusion)
        jobs.append(save_job(job, tmp_path / "data" / f"JOB-{index}.h5"))
    return jobs


def test_correlations_match_operator_trace_for_unitary_dynamics(tmp_path):
    # Two photons in [M1, S7]; the states with S7 occupied, at t=0 and after a
    # balanced beamsplitter. Labels are the read S7 number of each shot.
    labels = {(0, 2): [[2, 2, 2, 2], [0, 1, 1, 2]],
              (1, 1): [[1, 1, 1, 1], [0, 0, 2, 2]]}
    agp = CentralBosonAGPExperiment.from_children(
        sector_jobs(tmp_path, labels, [7], 7), direction=7)
    data = agp.analyze()
    assert int(data["sector_dimension"]) == 3  # full sector; [2, 0] is not measured

    # Independent operator calculation in the basis n_S7 = 0, 1, 2.
    unitary = np.array([[.5, -np.sqrt(.5), .5],
                        [np.sqrt(.5), 0., -np.sqrt(.5)],
                        [.5, np.sqrt(.5), .5]])
    number = np.diag([0., 1., 2.])
    for power in (1, 2):
        operator = np.linalg.matrix_power(number, power)
        expected = [np.trace(operator @ operator) / 3,
                    np.trace(unitary.T @ operator @ unitary @ operator) / 3]
        np.testing.assert_allclose(data[f"C{power}"], expected)


@pytest.mark.parametrize("confusion", [None, CONFUSION])
def test_correlations_and_errors_from_the_moments(tmp_path, confusion):
    rng = np.random.default_rng(4)
    states = direction_sector(2, [4, 7], 7)
    labels = {tuple(state): random_labels(rng, CYCLES) for state in states}
    agp = CentralBosonAGPExperiment.from_children(
        sector_jobs(tmp_path, labels, [4, 7], 7, confusion=confusion), direction=7)
    data = agp.analyze()
    initial = np.array([state[2] for state in states], dtype=float)
    dimension = sector_dimension(2, 3)
    suffix = "" if confusion is None else "_corrected"
    for k, moment, error in [(1, "n_mean", "n_err"), (2, "n2_mean", "n2_err")]:
        weights = initial**k / dimension
        np.testing.assert_allclose(data[f"C{k}{suffix}"], weights @ data[f"{moment}{suffix}"])
        # Independent jobs: the shot variances add with the squared weights.
        variance = sum(w**2 * row**2 for w, row in zip(weights, data[error]))
        np.testing.assert_allclose(data[f"C{k}_err"], np.sqrt(variance))
        assert (f"C{k}_corrected" in data) == (confusion is not None)
    agp.display()


def test_children_are_sorted_into_sector_order_and_must_cover_it(tmp_path):
    rng = np.random.default_rng(5)
    states = direction_sector(2, [4, 7], 7)
    labels = {tuple(state): random_labels(rng, CYCLES) for state in reversed(states)}
    jobs = sector_jobs(tmp_path, labels, [4, 7], 7)
    agp = CentralBosonAGPExperiment.from_children(jobs, job_ids=["c", "b", "a"], direction=7)
    assert [list(child.cfg.expt.occupations) for child in agp.children] == states
    assert agp.job_ids == ["a", "b", "c"]
    with pytest.raises(ValueError, match="exactly once"):
        CentralBosonAGPExperiment.from_children(jobs[:2], direction=7)
    with pytest.raises(ValueError, match="direction"):
        CentralBosonAGPExperiment.from_children(jobs)


def test_acquire_save_and_reload_keep_direction_and_disorder(tmp_path):
    detunings = [.03, -.01]
    agp = CentralBosonAGPExperiment(2, 7, [4, 7], CYCLES, detunings, calibration=CONFUSION)
    assert agp.disorder_khz == pytest.approx(rms_khz(detunings))
    overrides = agp.job_overrides()
    assert [o["occupations"] for o in overrides] == direction_sector(2, [4, 7], 7)
    assert {o["ro_stor"] for o in overrides} == {7}

    agp.acquire(FakeRunner(tmp_path / "data"))
    agp.analyze()
    manifest_path = agp.save()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["parameters"]["direction"] == 7
    assert len(manifest["raw_files"]) == 3

    reloaded = CentralBosonAGPExperiment.from_manifest(manifest_path)
    assert reloaded.direction == 7 and reloaded.disorder_khz == pytest.approx(agp.disorder_khz)
    reloaded.analyze()
    for key in ["C1", "C2", "C1_corrected", "C2_corrected", "C1_err"]:
        np.testing.assert_allclose(reloaded.data[key], agp.data[key])
    view = CentralBosonAGPExperiment.from_h5file(manifest_path.with_suffix(".h5"))
    view.display()


def test_read_mode_may_differ_from_direction(tmp_path):
    agp = CentralBosonAGPExperiment(2, 7, [4, 7], CYCLES, [0., 0.], ro_stor=4)
    assert {mode for _, mode in agp.entries} == {4}
    assert all(occupation[2] > 0 for occupation, _ in agp.entries)
