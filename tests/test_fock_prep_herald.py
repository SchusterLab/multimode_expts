"""FockPrepHeraldExperiment (experiments/qsim/fock_prep_herald.py) and fitting/qsim/f_postselect.py.

The estimators on synthetic shots; every herald mode compiled and run on the mock station with the
expt settings of jonginn's 11-12 population jobs (JOB-20261112-00178..00182), at cycle 0.
"""
import h5py
import matplotlib
import numpy as np
import pytest
from slab import AttrDict

matplotlib.use("Agg")

from experiments.characterization_runner import CharacterizationRunner  # noqa: E402
from experiments.qsim.agp import calibration_shots  # noqa: E402
from experiments.qsim.fock_population import FockPopulationExperiment  # noqa: E402
from experiments.qsim.fock_prep_herald import (  # noqa: E402
    HERALD_MODES, FockPrepHeraldExperiment, FockPrepHeraldProgram, HeraldedCalibrationProgram,
    HeraldedConfusionMatrixExperiment, prep_lanes, shot_labels)
from experiments.qsim.mbr_campaign import mock_station  # noqa: E402
from experiments.qsim.multiparity import herald_keep  # noqa: E402
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


# -- the herald cut in the analysis (2026-11-14) ------------------------------------------------

def _herald_shots(labels, herald_e):
    """-> idata (points, shots * 3): lanes (herald, parity 1, parity 2) against threshold 0."""
    labels = np.asarray(labels, int)
    lanes = np.empty((*labels.shape, 3))
    lanes[..., 0] = np.where(np.asarray(herald_e, bool), 100., -100.)
    lanes[..., 1] = 2 * (labels % 2) - 1
    lanes[..., 2] = 2 * (labels // 2) - 1
    return lanes.reshape(len(labels), -1)


def _herald_cfg(prep_herald, **expt):
    """No active reset, so the shot is (herald, parity 1, parity 2): read_num 3."""
    return AttrDict(dict(read_num=3, device=dict(readout=dict(threshold=[0.])), expt=dict(
        qubits=[0], readout="multiparity", active_reset=False, prep_herald=prep_herald, **expt)))


def _population_job(labels, herald_e, prep_herald="end"):
    job = FockPopulationExperiment.__new__(FockPopulationExperiment)
    job.cfg = _herald_cfg(prep_herald, occupations=[3, 0, 0, 0, 0], ro_stor=0,
                          swap_stors=[1, 2, 4, 5], detunings=[0.] * 4,
                          floquet_cycles=list(range(len(labels))), swept_params=["floquet_cycle"])
    idata = _herald_shots(labels, herald_e)
    job.data = dict(xpts=np.arange(len(labels)), idata=idata, qdata=np.zeros_like(idata))
    job.fname = None
    return job


def test_herald_keep_reads_the_lanes_between_reset_and_parity():
    idata = _herald_shots([[0, 1, 2]], [[False, True, False]])
    assert herald_keep(_herald_cfg("end"), idata).tolist() == [[True, False, True]]
    assert herald_keep(_herald_cfg("each"), idata).tolist() == [[True, False, True]]
    for mode in ("none", "each_reset"):
        assert herald_keep(_herald_cfg(mode), idata) is None
    assert herald_keep(AttrDict(dict(expt=dict())), idata) is None  # no prep_herald key
    no_herald_lane = AttrDict(dict(_herald_cfg("end"), read_num=2))  # e.g. Fock 0
    assert herald_keep(no_herald_lane, np.zeros((1, 6))) is None


def test_population_analysis_uses_only_the_kept_shots():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 4, size=(3, 60))
    herald_e = rng.random((3, 60)) < 0.2
    job = _population_job(labels, herald_e)
    job.analyze()
    kept = ~herald_e
    for point in range(3):
        accepted = labels[point, kept[point]]
        np.testing.assert_allclose([job.data[f"p_mod{n}"][point] for n in range(4)],
                                   np.bincount(accepted, minlength=4) / len(accepted))
        assert job.data["n_err"][point] == pytest.approx(accepted.std(ddof=1) / np.sqrt(len(accepted)))
        np.testing.assert_allclose([job.data[f"p_mod{n}_all"][point] for n in range(4)],
                                   np.bincount(labels[point], minlength=4) / labels.shape[1])
    np.testing.assert_array_equal(job.data["shot_counts"], kept.sum(axis=1))
    np.testing.assert_allclose(job.data["kept_fraction"], kept.mean(axis=1))
    np.testing.assert_allclose(job.data["nmod4_mean_all"], labels.mean(axis=1))

    job.cfg.expt.prep_herald = "none"  # the same shots without the cut: all of them, as before
    job.analyze()
    np.testing.assert_array_equal(job.data["shot_counts"], [60, 60, 60])
    np.testing.assert_allclose(job.data["nmod4_mean"], labels.mean(axis=1))
    assert "kept_fraction" not in job.data and "p_mod0_all" not in job.data


def test_heralded_confusion_matrix_rows_come_from_the_kept_shots():
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 4, size=(4, 80))
    herald_e = rng.random((4, 80)) < 0.3
    calibration = HeraldedConfusionMatrixExperiment.__new__(HeraldedConfusionMatrixExperiment)
    calibration.cfg = _herald_cfg("end", swap_stors=[1, 2, 4, 5])
    idata = _herald_shots(labels, herald_e)
    calibration.data = dict(xpts=np.arange(4), idata=idata, qdata=np.zeros_like(idata))
    calibration.analyze()
    kept = ~herald_e
    expected = [np.bincount(labels[n, kept[n]], minlength=4) / kept[n].sum() for n in range(4)]
    np.testing.assert_allclose(calibration.data["multiparity_confusion_matrix"], expected)
    np.testing.assert_array_equal(calibration.data["multiparity_confusion_matrix_shot_counts"],
                                  kept.sum(axis=1))


def _random_lanes(rng, points, shots, read_num, herald_lanes, threshold):
    """-> idata (points, shots * read_num): random g/e lanes; herald lanes read g 80 % of the time."""
    reads_e = rng.random((points, shots, read_num)) < 0.5
    reads_e[..., herald_lanes] = rng.random((points, shots, len(herald_lanes))) < 0.2
    return np.where(reads_e, threshold + 100., threshold - 100.).reshape(points, -1)


def test_herald_cut_matches_the_offline_labels_on_the_mock():
    # The mock returns I = 0 in every lane (all above the threshold), so the cut is checked
    # on random shots put into the mock job's layout.
    station = mock_station()
    assert station.is_mock, "refusing to acquire against real instruments"
    defaults = AttrDict(dict(POPULATION_EXPT, reps=40, occupations=[1, 1, 1, 0, 0], ro_stor=1,
                             floquet_cycles=[0, 4], prep_herald="end"))
    runner = CharacterizationRunner(station=station, ExptClass=FockPopulationExperiment,
                                    ExptProgram=FockPrepHeraldProgram, default_expt_cfg=defaults, show=False)
    expt = runner.execute(log=False, show=False, go_kwargs=dict(analyze=False))
    assert expt.cfg.read_num == 7  # 2 active reset + 3 heralds (S1, S2, M1 ladders) + 2 parity
    threshold = np.ravel(expt.cfg.device.readout.threshold)[0]
    idata = _random_lanes(np.random.default_rng(6), 2, 40, 7, [2, 3, 4], threshold)
    expt.data.update(idata=idata, qdata=np.zeros_like(idata))
    expt.analyze()
    labels = shot_labels(expt.cfg, expt.data)  # guan's offline labels, from prep_lanes()
    keep = herald_keep(expt.cfg, expt.data["idata"])
    np.testing.assert_array_equal(keep, labels["herald"])
    np.testing.assert_array_equal(expt.data["shot_counts"], keep.sum(axis=1))
    np.testing.assert_allclose(np.stack([expt.data[f"p_mod{n}"] for n in range(4)], axis=-1),
                               f_postselect.n_distribution(labels["n"], keep))


def test_heralded_confusion_matrix_is_one_job_on_the_mock():
    station = mock_station()
    assert station.is_mock, "refusing to acquire against real instruments"
    defaults = AttrDict(dict(POPULATION_EXPT, reps=40, prep_herald="end"))
    runner = CharacterizationRunner(station=station, ExptClass=HeraldedConfusionMatrixExperiment,
                                    default_expt_cfg=defaults, show=False)
    expt = runner.execute(log=False, show=False, go_kwargs=dict(analyze=False))
    assert expt.ProgramClass is HeraldedCalibrationProgram
    assert expt.cfg.read_num == 5  # 2 active reset + 1 herald (bare for Fock 0) + 2 parity
    np.testing.assert_array_equal(expt.data["xpts"], np.arange(4))
    assert np.asarray(expt.data["idata"]).shape == (4, 40 * 5)
    threshold = np.ravel(expt.cfg.device.readout.threshold)[0]
    idata = _random_lanes(np.random.default_rng(7), 4, 40, 5, [2], threshold)
    expt.data.update(idata=idata, qdata=np.zeros_like(idata))
    expt.analyze()
    np.testing.assert_allclose(expt.data["multiparity_confusion_matrix"].sum(axis=1), 1)
    np.testing.assert_array_equal(expt.data["multiparity_confusion_matrix_shot_counts"],
                                  herald_keep(expt.cfg, idata).sum(axis=1))


def test_calibration_shots_reads_the_kept_shots_of_each_row(tmp_path):
    path = tmp_path / "heralded_calibration.h5"
    with h5py.File(path, "w") as handle:
        handle["multiparity_confusion_matrix"] = np.eye(4)
        handle["multiparity_confusion_matrix_shot_counts"] = [10, 9, 8, 7]
    assert calibration_shots(dict(fname=str(path))).tolist() == [10, 9, 8, 7]


def test_heralded_confusion_matrix_needs_the_end_herald():
    station = mock_station()
    defaults = AttrDict(dict(POPULATION_EXPT, prep_herald="none"))
    runner = CharacterizationRunner(station=station, ExptClass=HeraldedConfusionMatrixExperiment,
                                    default_expt_cfg=defaults, show=False)
    with pytest.raises(ValueError, match="prep_herald='end'"):
        runner.execute(log=False, show=False, go_kwargs=dict(analyze=False))
