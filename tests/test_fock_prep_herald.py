"""FockPrepHeraldExperiment (experiments/qsim/fock_prep_herald.py) and fitting/qsim/f_postselect.py.

The estimators on synthetic shots; every herald mode compiled and run on the mock station with the
expt settings of jonginn's 11-12 population jobs (JOB-20261112-00178..00182), at cycle 0.
"""
import matplotlib
import numpy as np
import pytest
from slab import AttrDict

matplotlib.use("Agg")

from experiments.characterization_runner import CharacterizationRunner  # noqa: E402
from experiments.qsim.fock_prep_herald import (  # noqa: E402
    HERALD_MODES, FockPrepHeraldExperiment, FockPrepHeraldProgram, prep_lanes)
from experiments.qsim.mbr_campaign import mock_station  # noqa: E402
from fitting.qsim import f_postselect  # noqa: E402

# cfg.expt of JOB-20261112-00181 without the sweep and the confusion matrix.
POPULATION_EXPT = dict(
    expts=1, reps=2, rounds=1, qubits=[0], f0g1_cavity=1, normalize=False,
    active_reset=True, man_reset=True, storage_reset=[1, 2, 4, 5], pre_relax_delay=100,
    relax_delay=200, postpulse=True, init_fock=False, init_stor=0, readout="multiparity",
    parity_fast=False, cond_sec_phase=-90, phase_second_pulse=180, swap_stors=[1, 2, 4, 5],
    update_phases=True, detunings=[0.0] * 4, reset_dump_mode=2, dump_reset_iter_num=1,
    include_10cycles_buffer=True, include_10cycles_buffer_in_pi_half=True,
    floquet_waveform="preload_flattop", floquet_hardware_loop=True, scramble_sync_cycles=2,
    palindrome_scramble=False, prepulse=False, floquet_cycle=0)


def test_lanes_and_decoding():
    # two points, three shots, read_num 3: lanes (reset, parity 1, parity 2)
    i = np.array([[0, 10, -10, 0, -10, 10, 0, 10, 10], [0, -10, -10, 0, -10, -10, 0, 10, -10]])
    q = np.zeros_like(i)
    shots = f_postselect.lanes(i, q, 3)
    assert shots.shape == (2, 3, 3)
    n = f_postselect.decode_n_mod4(shots[..., 1], shots[..., 2], threshold=0)
    assert n.tolist() == [[1, 2, 3], [0, 0, 1]]
    p = f_postselect.n_distribution(n, keep=np.array([[1, 1, 0], [1, 1, 1]], bool))
    assert p.tolist() == [[0, 0.5, 0.5, 0], [2 / 3, 1 / 3, 0, 0]]
    assert np.isnan(f_postselect.n_distribution(n, np.zeros_like(n, bool))).all()


def test_corrected_inverts_the_confusion_matrix():
    C = np.array([[0.9, 0.05, 0.03, 0.02], [0.05, 0.85, 0.05, 0.05],
                  [0.06, 0.06, 0.8, 0.08], [0.06, 0.1, 0.06, 0.78]])
    truth = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(f_postselect.corrected(truth @ C, C), truth)
    assert f_postselect.mean_n(truth) == pytest.approx(2.0)


def test_reads_f_flags_the_f_cloud():
    centers = np.array([-118 - 40j, 24 - 37j, 17 + 40j])
    shots = np.array([-110 - 35j, 30 - 30j, 10 + 45j])
    assert f_postselect.reads_f(shots, centers).tolist() == [False, False, True]


@pytest.mark.parametrize("mode, lanes", [("none", 0), ("end", 1), ("each", 3), ("each_reset", 9)])
def test_prep_lanes(mode, lanes):
    expt = AttrDict(dict(occupations=[3, 0, 0, 0, 0], prep_herald=mode))
    assert len(prep_lanes(expt)) == lanes
    two = AttrDict(dict(occupations=[1, 2, 0, 0, 0], prep_herald=mode))  # S1 ladder, then M1
    assert len(prep_lanes(two)) == {"none": 0, "end": 2, "each": 3, "each_reset": 9}[mode]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="prep_herald"):
        prep_lanes(AttrDict(dict(occupations=[3, 0, 0, 0, 0], prep_herald="sometimes")))


@pytest.fixture(scope="module")
def runner():
    station = mock_station()
    assert station.is_mock, "refusing to acquire against real instruments"
    defaults = AttrDict(dict(POPULATION_EXPT, occupations=[3, 0, 0, 0, 0], ro_stors=[0, 1]))
    return CharacterizationRunner(station=station, ExptClass=FockPrepHeraldExperiment,
                                  default_expt_cfg=defaults, show=False)


@pytest.mark.parametrize("mode", HERALD_MODES)
def test_every_mode_runs_on_the_mock(runner, mode):
    expt = runner.execute(prep_herald=mode, log=False, show=False, go_kwargs=dict(analyze=True))
    lanes = len(prep_lanes(expt.cfg.expt))
    assert expt.cfg.read_num == 4 + lanes
    assert FockPrepHeraldProgram.readouts_per_shot(expt.cfg) == 4 + lanes
    assert expt.data["p_all"].shape == (2, 4)
    assert np.isnan(expt.data["kept_herald"]).all() == (mode == "none")
    assert np.isnan(expt.data["kept_after_reset"]).all() == (mode != "each_reset")
    labels = expt.shot_labels(centers=np.array([-118 - 40j, 24 - 37j, 17 + 40j]))
    assert labels["n"].shape == labels["no_f"].shape == (2, 2)
    assert (labels["herald"] is None) == (mode == "none")
    steps = {"none": 0, "end": 1, "each": 3, "each_reset": 3}[mode]
    assert labels["step_g"].shape == (2, 2, steps)


def test_display_and_labels_from_h5(runner):
    from experiments.qsim.fock_prep_herald import labels_from_h5
    expt = runner.execute(prep_herald="each_reset", log=False, show=False, go_kwargs=dict(analyze=True))
    fig = expt.display()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[0] == "all shots" and labels[1].startswith("herald (kept")
    assert labels[2].startswith("after reset (kept")
    cfg, lab = labels_from_h5(expt.fname, centers=np.array([-118 - 40j, 24 - 37j, 17 + 40j]))
    assert cfg.expt.prep_herald == "each_reset"
    assert np.array_equal(lab["n"], expt.shot_labels()["n"])


def test_population_cycle_sweep_with_the_herald_program():
    from experiments.qsim.fock_population import FockPopulationExperiment
    from experiments.qsim.fock_prep_herald import labels_from_h5
    station = mock_station()
    assert station.is_mock, "refusing to acquire against real instruments"
    defaults = AttrDict(dict(POPULATION_EXPT, occupations=[3, 0, 0, 0, 0], ro_stor=1,
                             floquet_cycles=[0, 4], prep_herald="end"))
    runner = CharacterizationRunner(station=station, ExptClass=FockPopulationExperiment,
                                    ExptProgram=FockPrepHeraldProgram, default_expt_cfg=defaults, show=False)
    expt = runner.execute(log=False, show=False, go_kwargs=dict(analyze=True))
    assert expt.cfg.read_num == 5  # 2 active reset + 1 herald + 2 parity
    assert "p_mod0" in expt.data and len(expt.data["p_mod0"]) == 2
    cfg, lab = labels_from_h5(expt.fname)
    assert lab["herald"].shape == lab["n"].shape == (2, 2)


def test_prep_lanes_none_needs_no_occupations():
    assert prep_lanes(AttrDict(dict(readout="multiparity"))) == []


def test_select_combines_the_cuts():
    from experiments.qsim.fock_prep_herald import select
    labels = dict(n=np.zeros((1, 4), int), herald=np.array([[1, 1, 0, 0]], bool), after_reset=None,
                  no_f=np.array([[1, 0, 1, 0]], bool))
    assert select(labels).tolist() == [[True] * 4]
    assert select(labels, herald=True, after_reset=True).tolist() == [[True, True, False, False]]
    assert select(labels, herald=True, no_f=True).tolist() == [[True, False, False, False]]
    with pytest.raises(ValueError, match="centers"):
        select(dict(labels, no_f=None), no_f=True)


def test_mean_n_with_error():
    n = np.array([[0, 1, 2, 3], [3, 3, 3, 3]])
    m, s = f_postselect.mean_n_with_error(n, np.array([[1, 1, 1, 1], [1, 1, 0, 0]], bool))
    assert m.tolist() == [1.5, 3.0]
    assert s[0] == pytest.approx(np.std([0, 1, 2, 3]) / 2) and s[1] == 0
