"""FStateHoldExperiment (experiments/qsim/f_state_hold.py) and its estimators
(fitting/qsim/f_hold.py).

The estimators are checked on synthetic shots and curves with known answers. The
Experiment runs end to end on the mock station: every program compiles, the
|f,2> prefix plus a wait fits the board's envelope memory, the points get the
right pulses, and the analysis runs on the (zero) mock data without an error.
"""
import h5py
import matplotlib
import numpy as np
import pytest
from slab import AttrDict

matplotlib.use("Agg")

from experiments.characterization_runner import CharacterizationRunner  # noqa: E402
from experiments.qsim.f_state_hold import FStateHoldExperiment, hold_points  # noqa: E402
from experiments.qsim.mbr_campaign import mock_station  # noqa: E402
from fitting.qsim import f_hold  # noqa: E402

CENTERS = np.array([-100 + 0j, 50 + 0j, 50 + 120j])  # g, e, f


def cloud(rng, probs, n, sigma=30.0):
    """n complex shots drawn from the three Gaussian clouds with weights probs."""
    idx = rng.choice(3, size=n, p=probs)
    return CENTERS[idx] + sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))


def test_hold_points_order():
    assert hold_points([0, 2], [0, 5]) == [
        ["g", 0.0], ["e", 0.0], ["f", 0.0], [0, 0.0], [0, 5.0], [2, 0.0], [2, 5.0]]


def test_corrected_populations_recover_a_known_mixture():
    rng = np.random.default_rng(1)
    refs = [cloud(rng, np.eye(3)[k], 20000, sigma=40) for k in range(3)]
    centers = f_hold.blob_centers(refs)
    assert np.allclose(centers, CENTERS, atol=2)
    A = f_hold.assignment_matrix(refs, centers)
    assert np.allclose(A.sum(axis=1), 1)
    assert A[1, 2] > 0.03  # the e and f clouds overlap

    # Mostly f: the f -> e misassignment is not cancelled by e -> f, so the raw
    # fractions are wrong and the correction must fix them.
    truth = np.array([0.05, 0.05, 0.90])
    raw = f_hold.assigned_fractions(cloud(rng, truth, 20000, sigma=40), centers)
    assert raw[2] < truth[2] - 0.03
    assert np.allclose(f_hold.corrected_populations(raw, A), truth, atol=0.015)


def test_corrected_populations_with_a_singular_matrix_do_not_raise():
    A = np.array([[1.0, 0, 0]] * 3)  # all clouds at one point, as with zero mock data
    pop = f_hold.corrected_populations(np.ones((2, 4, 3)) / 3, A)
    assert pop.shape == (2, 4, 3)


@pytest.mark.parametrize("tau, share", [(20.0, 0.6), (2.6, 0.95), (120.0, 0.0)])
def test_decay_fit_recovers_rate_and_share_to_g(tau, share):
    t = np.array([0, 2, 5, 10, 20, 40, 60], float)
    pf = 0.85 * np.exp(-t / tau) + 0.05
    pg = 0.02 + share * (0.90 - pf)
    pop = np.column_stack([pg, 1 - pg - pf, pf])
    fit = f_hold.decay_fit(t, pop)
    assert fit["rate"] == pytest.approx(1 / tau, rel=1e-3)
    assert fit["share_to_g"] == pytest.approx(share, abs=1e-6)
    assert fit["rate_to_g"] == pytest.approx(share / tau, rel=1e-3, abs=1e-9)
    assert fit["offset"] == pytest.approx(0.05, abs=1e-3)


@pytest.mark.parametrize("t, pf", [
    ([0, 2, 5, 10], [0.5, 0.5, 0.5, 0.5]),  # no change in P(f)
    ([0, 5], [0.9, 0.5]),                   # too few points
])
def test_decay_fit_gives_nan_where_it_cannot_fit(t, pf):
    pop = np.column_stack([np.zeros(len(t)), 1 - np.array(pf), pf])
    fit = f_hold.decay_fit(t, pop)
    assert all(np.isnan(v) for v in fit.values())


def test_transition_frequencies_of_the_1111_045mA_config():
    # CFG-HW-20261111-00097: device.qubit f_ge, f_ef and multiphoton fn-gn+1 frequencies.
    s = [1997.9653974698153, 1997.3440405852505, 1996.8705599700575, 1997.9653974698153]
    omega, nu = f_hold.transition_frequencies(3563.120382419127, 3419.6981002648113, s, [0, 1, 2, 5])
    assert omega == pytest.approx(4984.853, abs=1e-3)
    assert np.isnan(nu[0]) and np.isnan(nu[3])
    assert nu[1:3] == pytest.approx([11967.050, 11966.577], abs=1e-3)


@pytest.fixture(scope="module")
def mock_hold():
    station = mock_station()
    assert station.is_mock, "refusing to acquire against real instruments"
    defaults = AttrDict(dict(
        reps=2, rounds=1, qubit=0, qubits=[0], relax_delay=10,
        active_reset=False, man_reset=False, storage_reset=False,
        hold_ns=[0, 1, 2], hold_waits=[0, 5]))
    runner = CharacterizationRunner(station=station, ExptClass=FStateHoldExperiment,
                                    default_expt_cfg=defaults, show=False)
    # A mock station turns the analysis off by default; the worker always runs it.
    return runner.execute(log=False, show=False, go_kwargs=dict(analyze=True))


def test_mock_job_runs_every_point(mock_hold):
    expt = mock_hold
    assert len(expt.data["idata"]) == 3 + 3 * 2
    assert expt.data["populations"].shape == (3, 2, 3)
    assert expt.data["assignment_matrix"].shape == (3, 3)
    assert np.isnan(expt.data["nu"][0]) and np.all(np.isfinite(expt.data["nu"][1:]))


def test_last_point_plays_f2_and_the_wait(mock_hold):
    prog_expt = mock_hold.prog.cfg.expt
    assert prog_expt.prepulse and not prog_expt.pulse_e and not prog_expt.pulse_f
    assert prog_expt.pre_sweep_pulse == [
        ["multiphoton", "g0-e0", "pi", 0], ["multiphoton", "e0-f0", "pi", 0],
        ["multiphoton", "f0-g1", "pi", 0], ["multiphoton", "g1-e1", "pi", 0],
        ["multiphoton", "e1-f1", "pi", 0], ["multiphoton", "f1-g2", "pi", 0],
        ["multiphoton", "g2-e2", "pi", 0], ["multiphoton", "e2-f2", "pi", 0],
        ["wait", 5.0]]


def test_point_keys_stay_out_of_the_saved_config(mock_hold):
    for key in ("pre_sweep_pulse", "pulse_e", "pulse_f", "prepulse"):
        assert key not in mock_hold.cfg.expt
    with h5py.File(mock_hold.fname, "r") as f:
        assert f["populations"].shape == (3, 2, 3)
        assert f["idata"].shape[0] == 9


def test_display_draws_the_fit(mock_hold):
    t = np.array([0, 5.0])
    data = dict(mock_hold.data)
    data.update(hold_ns=np.array([1]), hold_waits=t, nu=np.array([11967.05]),
                populations=np.array([[[0.0, 0.1, 0.9], [0.4, 0.1, 0.5]]]),
                decay_rate=np.array([0.1]), decay_amplitude=np.array([0.9]),
                decay_offset=np.array([0.0]), share_to_g=np.array([0.95]),
                rate_to_g=np.array([0.095]))
    fig = mock_hold.display(data=data)
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert labels == ["P(g)", "P(e)", "P(f)", "P(f) fit"]
