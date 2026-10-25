"""CentralBosonAGPExperiment: the direction sector, C_1 and C_2, the AGP, theory and save lifecycle."""
import json

import h5py
import matplotlib
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.linalg import expm
import yaml

matplotlib.use("Agg")

from experiments.qsim.agp import (  # noqa: E402
    CentralBosonAGPExperiment,
    direction_sector,
    rms_khz,
    sector_dimension,
)
from fitting.qsim.agp import agp_kernel_weights, calibration_error_factors, finite_time_agp  # noqa: E402
from theory.qsim.agp import two_time_correlation  # noqa: E402
from tests.population_jobs import (  # noqa: E402
    CONFUSION,
    COUPLINGS_MHZ,
    KERR_MHZ,
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
    # The Kerr direction needs a pair in M1: n_M1 >= 2.
    assert direction_sector(1, [], 0) == [] and direction_sector(2, [], 0) == [[2]]
    assert direction_sector(2, [7], 0) == [[2, 0]]
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
@pytest.mark.parametrize("mu", [None, 5.915])
def test_correlations_and_errors_from_the_moments(tmp_path, confusion, mu):
    rng = np.random.default_rng(4)
    states = direction_sector(2, [4, 7], 7)
    labels = {tuple(state): random_labels(rng, CYCLES) for state in states}
    agp = CentralBosonAGPExperiment.from_children(
        sector_jobs(tmp_path, labels, [4, 7], 7, confusion=confusion), direction=7)
    # Only the active links (S4, S7) define g; unequal/signed/zero links are allowed.
    for child in agp.children:
        params = child.recorded_derived_params()
        params["couplings_MHz"] = [1., 1., 1., -.02, 1., 1., 0.]
        child.data["attrs"]["derived_params"] = json.dumps(params)
    data = agp.analyze(mu=mu)
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
    # Independent kernel calculation checks kHz/us units, normalization and C1 selection.
    g_khz = 10.  # mean(|-20 kHz|, |0 kHz|); inactive links do not contribute.
    cutoff = g_khz * 3 / dimension if mu is None else mu
    tau = 1000 / (2 * np.pi * cutoff)
    times = data["time_us"]
    dense = np.linspace(0, times[-1], 200001)
    kernel = tau / 2 * (1 - dense / tau) * np.exp(-dense / tau)
    scale = (2 * np.pi * g_khz / 1000)**2
    expected = scale * trapezoid(kernel * np.interp(dense, times, data[f"C1{suffix}"]), x=dense)
    np.testing.assert_allclose(data["agp_mu"], [cutoff])
    np.testing.assert_allclose(data["agp"], [expected], rtol=1e-8)
    # No readable calibration file: the AGP error is the shot error of C1, independent per time.
    weights = scale * np.array([trapezoid(kernel * np.interp(dense, times, row), x=dense)
                                for row in np.eye(len(times))])
    np.testing.assert_allclose(data["agp_err"], [np.sqrt(weights**2 @ data["C1_err"]**2)],
                               rtol=1e-8)
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
    with h5py.File(manifest_path.with_suffix(".h5"), "r") as handle:
        for key in ["agp", "agp_mu", "agp_err", "agp_theory", "agp_theory_window", "C1_theory"]:
            np.testing.assert_allclose(handle[key][()], agp.data[key])

    reloaded = CentralBosonAGPExperiment.from_manifest(manifest_path)
    assert reloaded.direction == 7 and reloaded.disorder_khz == pytest.approx(agp.disorder_khz)
    reloaded.analyze()
    for key in ["C1", "C2", "C1_corrected", "C2_corrected", "C1_err", "agp", "agp_mu",
                "C1_theory", "agp_theory"]:
        np.testing.assert_allclose(reloaded.data[key], agp.data[key])
    view = CentralBosonAGPExperiment.from_h5file(manifest_path.with_suffix(".h5"))
    view.display()


def test_read_mode_may_differ_from_direction(tmp_path):
    agp = CentralBosonAGPExperiment(2, 7, [4, 7], CYCLES, [0., 0.], ro_stor=4)
    assert {mode for _, mode in agp.entries} == {4}
    assert all(occupation[2] > 0 for occupation, _ in agp.entries)
    agp.acquire(FakeRunner(tmp_path / "data"))
    with pytest.warns(UserWarning):
        data = agp.analyze()
    assert "C1" in data and "C2" in data and "agp" not in data


def test_kerr_direction_probes_pairs_with_errors(tmp_path):
    # N = 2 in [M1, S7]: only [2, 0] has a pair in M1. Labels are its read M1 number.
    labels = np.array([[2] * 30 + [1] * 10, [0] * 20 + [1] * 10 + [2] * 10])
    job = population_job([2, 0], 0, labels, CYCLES, [7], [0.])
    agp = CentralBosonAGPExperiment.from_children(
        [save_job(job, tmp_path / "data" / "JOB-0.h5")], direction=0)
    data = agp.analyze(mu=5.)
    pairs = labels * (labels - 1) / 2
    dimension = sector_dimension(2, 2)
    np.testing.assert_allclose(data["C1"], pairs.mean(axis=1) / dimension)
    np.testing.assert_allclose(data["C1_err"], pairs.std(axis=1, ddof=1)
                               / np.sqrt(labels.shape[1]) / dimension)
    assert "C2" not in data and data["agp"].shape == (1,)


def brute_force_correlation(photons, swap_stors, detunings, probe_mode, times):
    """Tr_N[n(t) n] / D with ladder operators on the full truncated space, no Fock-sector code."""
    modes, levels = 1 + len(swap_stors), photons + 1
    lowering = np.diag(np.sqrt(np.arange(1., levels)), 1)

    def on(mode, operator):
        result = np.eye(1)
        for index in range(modes):
            result = np.kron(result, operator if index == mode else np.eye(levels))
        return result

    a = [on(mode, lowering) for mode in range(modes)]
    n = [x.T @ x for x in a]
    hamiltonian = KERR_MHZ / 2 * n[0] @ (n[0] - np.eye(len(n[0])))
    for index, storage in enumerate(swap_stors, start=1):
        hamiltonian = hamiltonian - detunings[index - 1] * n[index]
        hamiltonian = hamiltonian + COUPLINGS_MHZ[storage - 1] * (a[0].T @ a[index]
                                                                  + a[index].T @ a[0])
    sector = np.flatnonzero(np.rint(np.diag(sum(n))).astype(int) == photons)
    probe = n[probe_mode]
    values = []
    for t in times:
        unitary = expm(-2j * np.pi * hamiltonian * t)
        evolved = (unitary.conj().T @ probe @ unitary @ probe)[np.ix_(sector, sector)]
        values.append(np.trace(evolved).real / len(sector))
    return np.asarray(values)


def test_theory_matches_brute_force_dynamics_and_long_window(tmp_path):
    swap_stors, detunings = [4, 7], [.03, -.01]
    rng = np.random.default_rng(6)
    states = direction_sector(2, swap_stors, 7)
    labels = {tuple(state): random_labels(rng, CYCLES) for state in states}
    agp = CentralBosonAGPExperiment.from_children(
        sector_jobs(tmp_path, labels, swap_stors, 7, detunings=detunings), direction=7)
    data = agp.analyze(mu=[2., 8.])
    np.testing.assert_allclose(data["theory_kerr_mhz"], KERR_MHZ)
    expected = brute_force_correlation(2, swap_stors, detunings, 2, data["time_us"])
    np.testing.assert_allclose(data["C1_theory"], expected, atol=1e-12)
    assert data["agp_theory"].shape == data["agp_theory_window"].shape == (2,)

    # The infinite-time AGP is the kernel integral of the closed-system C1 over a long window.
    differences, weights = agp.theory_spectrum()
    for cutoff_khz, value in zip(data["agp_mu"], data["agp_theory"]):
        times = np.linspace(0, 40 * 1000 / (2 * np.pi * cutoff_khz), 400001)
        window = finite_time_agp(times, two_time_correlation(differences, weights, times),
                                 cutoff_khz / 1000)["values"]
        np.testing.assert_allclose((2 * np.pi * .02)**2 * window, value, rtol=1e-5)

    # A Kerr override changes the theory only.
    other = agp.analyze(mu=[2., 8.], kerr_mhz=0.)
    np.testing.assert_allclose(other["C1"], data["C1"])
    assert not np.allclose(other["agp_theory"], data["agp_theory"])


def test_local_t1_compensation_and_its_error(tmp_path):
    rng = np.random.default_rng(7)
    states = direction_sector(2, [4, 7], 7)
    labels = {tuple(state): random_labels(rng, CYCLES) for state in states}
    agp = CentralBosonAGPExperiment.from_children(
        sector_jobs(tmp_path, labels, [4, 7], 7), direction=7)
    plain = agp.analyze(mu=4.)["agp"].copy()
    data = agp.analyze(mu=4., t1_us=50., t1_err_us=5.)
    np.testing.assert_allclose(data["agp"], plain)
    times = data["time_us"]
    dense = np.linspace(0, times[-1], 200001)
    tau = 1000 / (2 * np.pi * 4.)
    kernel = tau / 2 * (1 - dense / tau) * np.exp(-dense / tau) * np.exp(dense / 50.)
    expected = (2 * np.pi * .02)**2 * trapezoid(
        kernel * np.interp(dense, times, data["C1"]), x=dense)
    np.testing.assert_allclose(data["agp_t1"], [expected], rtol=1e-8)
    without = agp.analyze(mu=4., t1_us=50.)
    assert data["agp_t1_err"][0] > without["agp_t1_err"][0]


def test_calibration_file_adds_the_shared_calibration_error(tmp_path):
    rng = np.random.default_rng(8)
    states = direction_sector(2, [4, 7], 7)
    labels = {tuple(state): random_labels(rng, CYCLES) for state in states}
    agp = CentralBosonAGPExperiment.from_children(
        sector_jobs(tmp_path, labels, [4, 7], 7, confusion=CONFUSION), direction=7)
    with pytest.warns(UserWarning, match="shot-only"):
        shot_only = agp.analyze(mu=4.)["agp_err"].copy()
    calibration = tmp_path / "calibration.h5"
    with h5py.File(calibration, "w") as handle:  # 4 prepared states x 500 shots x 4 lanes
        handle.attrs["config"] = json.dumps(dict(read_num=4))
        handle.create_dataset("idata", data=np.zeros((4, 2000)))
    agp.calibration["fname"] = str(calibration)
    data = agp.analyze(mu=4.)
    assert data["C1_calibration_factors"].shape == (len(CYCLES), 4)
    weights = (2 * np.pi * .02)**2 * agp_kernel_weights(data["time_us"], .004)
    rows = calibration_error_factors(data["p_mod_corrected"], CONFUSION, 500, np.arange(4.))
    factors = np.einsum("r,rta->ta", data["occupations"][:, 2] / 6, rows)
    expected = np.sqrt(shot_only**2 + ((weights @ factors)**2).sum())
    np.testing.assert_allclose(data["agp_err"], expected)
    assert data["agp_err"][0] > shot_only[0]
