"""Sequential multiparity pulses, paired shots, calibration and raw HDF5 saving."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib
import numpy as np
import pytest

from slab import AttrDict

matplotlib.use("Agg")

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
    FockPopulationSequentialReadoutProgram,
    readout_layout,
)
from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
from tests.population_jobs import COUPLINGS_MHZ, CYCLE_US, KERR_MHZ
from tests.test_fock_population_acquire_mock import population_defaults

CYCLES = [0, 3, 6]
MODES = [0, 4, 7]
MATRICES = np.stack([.8 * np.eye(4) + .05,
                     .9 * np.eye(4) + .025,
                     .72 * np.eye(4) + .07], axis=-1)


def synthetic_job(*, occupations=(1, 1, 0), calibration=None, heralds=True,
                  cycles=CYCLES, labels=None, keep=None):
    """Build paired IQ shots with two independent heralds and distinct mode labels."""
    job = FockPopulationSequentialReadoutExperiment.__new__(
        FockPopulationSequentialReadoutExperiment)
    job.cfg = AttrDict(dict(
        device=dict(readout=dict(threshold=[.75]), manipulate=dict(kerr=[KERR_MHZ])),
        expt=dict(qubits=[0], readout="multiparity", ro_stor=0, ro_stors=MODES.copy(),
                  occupations=list(occupations), swap_stors=[4, 7], detunings=[.01, -.02],
                  floquet_cycles=list(cycles), swept_params=["floquet_cycle"],
                  prepulse=False, postpulse=True, cond_sec_phase=-90,
                  phase_second_pulse=180, active_reset=heralds, ef_reset=False,
                  man_reset=False, storage_reset=False, pre_selection_reset=heralds,
                  pre_selection_parity=heralds, multiparity_confusion_matrix=None,
                  multiparity_confusion_matrices=calibration)))
    layout = readout_layout(job.cfg)
    job.cfg.read_num = int(layout["read_num"])
    if labels is None:
        base = np.array([[0, 3, 1], [1, 0, 2], [2, 1, 3],
                         [3, 2, 0], [0, 1, 2], [2, 3, 0]])
        labels = np.stack([(base + point) % 4 for point in range(len(cycles))])
    labels = np.asarray(labels, dtype=int)
    if keep is None:
        keep = (np.array([[True, True, True, True, True, False],
                          [True, False, True, False, True, True],
                          [False, True, False, True, False, True]]) if heralds
                else np.ones(labels.shape[:2], dtype=bool))
    keep = np.asarray(keep, dtype=bool)
    npoints, shots, _ = labels.shape
    irows = np.full((npoints, shots, job.cfg.read_num), -2.)
    qrows = np.arange(irows.size, dtype=float).reshape(irows.shape) / 100
    pairs = np.asarray(layout["readout_lane_pairs"], dtype=int)
    for step, (first, second) in enumerate(pairs):
        irows[..., first] = .75 + 2 * (labels[..., step] % 2) - 1
        irows[..., second] = .75 + 2 * (labels[..., step] // 2) - 1
    herald_lanes = list(layout["preselection_lanes"])
    for point in range(npoints):
        failed_shots = [shot for shot, accepted in enumerate(keep[point]) if not accepted]
        for failure, shot in enumerate(failed_shots):
            irows[point, shot, herald_lanes[failure % len(herald_lanes)]] = 3.
    points = dict(idata=list(irows.reshape(npoints, -1)),
                  qdata=list(qrows.reshape(npoints, -1)),
                  avgi=[999.] * npoints, avgq=[-999.] * npoints)
    job.data = job.shape_data([("floquet_cycle", list(cycles))], points)
    job.fname = None
    job.derived_params = lambda: dict(floquet_cycle_us=CYCLE_US,
                                     couplings_MHz=COUPLINGS_MHZ)
    return job, labels, keep, irows, qrows


def expected_statistics(labels, keep, matrices=None):
    """Calculate moments directly from each retained shot's measured-label weight."""
    probabilities = np.zeros((len(labels), len(MODES), 4))
    mean = np.zeros(probabilities.shape[:2])
    second = np.zeros(mean.shape)
    error = np.zeros(mean.shape)
    second_error = np.zeros(mean.shape)
    for point, row in enumerate(labels):
        for mode in range(len(MODES)):
            values = row[keep[point], mode]
            probabilities[point, mode] = np.bincount(values, minlength=4) / len(values)
            matrix = np.eye(4) if matrices is None else matrices[:, :, mode]
            number_weights = np.linalg.solve(matrix, np.arange(4.))
            squared_weights = np.linalg.solve(matrix, np.arange(4.)**2)
            measured = number_weights[values]
            measured_second = squared_weights[values]
            mean[point, mode] = measured.mean()
            second[point, mode] = measured_second.mean()
            error[point, mode] = measured.std(ddof=1) / np.sqrt(len(values))
            second_error[point, mode] = measured_second.std(ddof=1) / np.sqrt(len(values))
    return probabilities, mean, second, error, second_error


def save_through_go(job, path, monkeypatch):
    """Use the ordinary Experiment lifecycle without replacing its save implementation."""
    acquired = deepcopy(job.data)
    job.data = {}
    job.fname = str(path)
    monkeypatch.setattr(job, "acquire", lambda progress=False: deepcopy(acquired))
    job.go(save=True, analyze=True)
    return Path(job.fname)


@pytest.mark.parametrize("modes", [[0], [4], [0, 4, 7]])
@pytest.mark.parametrize("fast", [False, True])
def test_program_completes_each_bit_pair_and_only_relaxes_at_the_end(modes, fast):
    events = []
    program = FockPopulationSequentialReadoutProgram.__new__(
        FockPopulationSequentialReadoutProgram)
    program.cfg = AttrDict(dict(
        expt=dict(ro_stors=modes, qubits=[0], phase_second_pulse=180,
                  cond_sec_phase=-90),
        device=dict(readout=dict(threshold=[.75], trig_offset=[450], relax_delay=[1000]),
                    manipulate=dict(revival_time=[1.6], revival_time_fast=[.8],
                                    revival_stark_shift=[.125]))))
    program.qubit_chs, program.res_chs, program.adc_chs = [2], [6], [0]
    program.readout_lengths_adc = [100]
    program.f_ge, program.f_ge_reg, program.f_ge_hpi_fast = 101, [101], 202
    program.pi_ge_gain, program.hpi_ge_gain, program.hpi_ge_gain_fast = 33, 11, 22
    program.us2cycles = lambda duration, **kwargs: round(duration * 1000)
    program.deg2reg = lambda degrees, **kwargs: degrees
    program.ch_page = lambda channel: 1
    program.sreg = lambda channel, register: 16

    def create(gates):
        events.append(("ramsey_gates", deepcopy(gates)))
        return SimpleNamespace(pulse=np.asarray(gates, dtype=object))

    program.get_prepulse_creator = create
    program.custom_pulse = lambda cfg, pulses, prefix: events.append(
        ("custom_pulse", (deepcopy(pulses), prefix)))
    program.safe_regwi = lambda *args: events.append(("register", args))
    program.set_pulse_registers = lambda **kwargs: events.append(("pulse_settings", kwargs))
    program.pulse = lambda **kwargs: events.append(("pulse", kwargs))
    program.read = lambda *args: events.append(("read", args))
    program.condj = lambda *args: events.append(("branch", args))
    program.label = lambda label: events.append(("label", label))
    program.mathi = lambda *args: events.append(("phase_copy", args))
    program.sync_all = lambda cycles=None: events.append(("sync", cycles))
    program.wait_all = lambda cycles: events.append(("feedback_wait", cycles))
    program.measure = lambda **kwargs: events.append(("measure", kwargs))
    program._play_gate_list = lambda gates, prefix: events.append(("swap", (gates, prefix)))
    program.multi_parity_readout(fast=fast)
    completed = len(events)
    program.measure_wrapper()
    assert len(events) == completed  # The inherited body must not measure a third bit.

    measurements = [value for kind, value in events if kind == "measure"]
    assert len(measurements) == 2 * len(modes)
    assert all(value["adcs"] == [0] and value["wait"] for value in measurements)
    assert all(value.get("syncdelay", 0) == 0 for value in measurements[:-1])
    assert measurements[-1]["syncdelay"] == 1_000_000
    final_index = max(index for index, (kind, _) in enumerate(events) if kind == "measure")
    assert events[final_index - 1] == ("sync", 10)
    assert not any(kind in {"read", "pulse"} for kind, _ in events[final_index + 1:])

    # Every nonterminal bit is read and reset before the next measurement.
    measure_indices = [index for index, (kind, _) in enumerate(events) if kind == "measure"]
    for index in range(len(measure_indices) - 1):
        first = measure_indices[index]
        following = measure_indices[index + 1]
        interval = events[first + 1:following]
        resets = [value for kind, value in interval
                  if kind == "pulse_settings" and value["gain"] == program.pi_ge_gain]
        assert len(resets) == 1
        assert ("feedback_wait", 100) in interval and ("sync", 2000) in interval
        assert sum(kind == "read" for kind, _ in interval) == 1
    assert ("register", (0, 11, 75)) in events  # Feedback uses integrated, unnormalized I.
    branch_targets = [value[-1] for kind, value in events if kind == "branch"]
    labels = [value for kind, value in events if kind == "label"]
    assert len(branch_targets) == len(set(branch_targets))
    assert len(labels) == len(set(labels))
    assert set(branch_targets) <= set(labels)

    gates = [value for kind, value in events if kind == "ramsey_gates"]
    first_phase = 180 + 360 * .125 * 1.6 if fast else 180
    first_pulse = ["multiphoton", "g0-e0", "hpi", 0] if fast else ["qubit", "ge", "hpi", 0]
    expected_gates = [first_pulse, ["wait", .8 if fast else 1.6],
                      [*first_pulse[:-1], first_phase]]
    assert gates == [expected_gates] * len(modes)
    prefixes = [value[1] for kind, value in events if kind == "custom_pulse"]
    assert len(prefixes) == len(set(prefixes)) == len(modes)
    second_hpi = [value for kind, value in events
                  if kind == "pulse_settings" and value["gain"] != program.pi_ge_gain]
    assert len(second_hpi) == len(modes)
    for pulse in second_hpi:
        assert pulse["freq"] == (202 if fast else 101)
        assert pulse["gain"] == (22 if fast else 11)
        assert pulse["waveform"] == ("hpi_qubit_ge_fast" if fast else "hpi_qubit_ge")
    second_phase = 180 + 360 * .125 * .4 if fast else 180
    phase_values = [value[2] for kind, value in events
                    if kind == "register" and value[:2] == (1, 8)]
    np.testing.assert_allclose(phase_values, [second_phase, second_phase - 90] * len(modes))
    assert sum(kind == "sync" and value == (400 if fast else 800)
               for kind, value in events) == len(modes)
    swaps = [value for kind, value in events if kind == "swap"]
    assert len({prefix for _, prefix in swaps}) == len(modes) - 1
    assert [gates for gates, _ in swaps] == [
        [["storage", f"M1-S{mode}", "pi", 0]] for mode in modes[1:]]
    assert sum(kind == "sync" and value == 200 for kind, value in events) == len(modes) - 1


def test_layout_decoder_and_raw_moments_share_both_heralds():
    job, labels, keep, irows, qrows = synthetic_job()
    layout = readout_layout(job.cfg)
    assert layout["read_num"] == 9
    np.testing.assert_array_equal(layout["preselection_lanes"], [1, 2])
    np.testing.assert_array_equal(layout["readout_lane_pairs"], [[3, 4], [5, 6], [7, 8]])
    assert FockPopulationSequentialReadoutProgram.readouts_per_shot(job.cfg) == 9
    for point in range(len(CYCLES)):
        decoded = job.decode_readouts(point_idx=point)
        np.testing.assert_array_equal(decoded["keep"], keep[point])
        np.testing.assert_array_equal(decoded["labels"], labels[point, keep[point]])
        np.testing.assert_allclose(decoded["i"], irows[point, keep[point], 3:].reshape(-1, 3, 2))
        np.testing.assert_allclose(decoded["q"], qrows[point, keep[point], 3:].reshape(-1, 3, 2))
    data = job.analyze()
    assert data is job.data
    probabilities, mean, second, error, second_error = expected_statistics(labels, keep)
    for key, expected in [("p_mod", probabilities), ("n_mean", mean), ("n2_mean", second),
                          ("n_err", error), ("n2_err", second_error)]:
        np.testing.assert_allclose(data[key], expected)
    np.testing.assert_array_equal(data["shot_counts"], [[5, 5, 5], [4, 4, 4], [3, 3, 3]])
    np.testing.assert_array_equal(data["ro_stors"], MODES)
    assert int(data["readout_bits"]) == 2
    for point in range(len(CYCLES)):
        np.testing.assert_allclose(data["avgi"][point],
                                   irows[point, keep[point], 3:].reshape(-1, 3, 2).mean(axis=0))


@pytest.mark.parametrize("overrides", [dict(ro_stors=[4, 0]), dict(ro_stors=[0, 4, 4]),
                                       dict(ro_stors=[]), dict(readout="qubit")])
def test_layout_rejects_protocols_that_would_mislabel_modes(overrides):
    job, *_ = synthetic_job()
    job.cfg.expt.update(overrides)
    with pytest.raises(ValueError):
        readout_layout(job.cfg)


def test_empty_accepted_ensemble_keeps_mode_axes_and_nan_statistics():
    job, *_ = synthetic_job(keep=np.zeros((3, 6), dtype=bool))
    decoded = job.decode_readouts()
    assert decoded["labels"].shape == (0, 3)
    assert decoded["i"].shape == (0, 3, 2)
    data = job.analyze()
    np.testing.assert_array_equal(data["shot_counts"], np.zeros((3, 3), dtype=int))
    assert np.isnan(data["p_mod"]).all()
    assert np.isnan(data["n_mean"]).all() and np.isnan(data["n_err"]).all()


@pytest.mark.parametrize("kind", ["tensor", "record", "filename"])
def test_per_step_correction_roundtrips_through_normal_go(tmp_path, monkeypatch, kind):
    calibration_path = tmp_path / "readout_calibration.h5"
    with h5py.File(calibration_path, "w") as handle:
        handle["multiparity_confusion_matrices"] = MATRICES
        handle["ro_stors"] = MODES
    calibration = MATRICES.copy()
    if kind == "record":
        calibration = dict(matrices=MATRICES.tolist(), ro_stors=MODES,
                           fname=str(calibration_path))
    elif kind == "filename":
        calibration = str(calibration_path)
    job, labels, keep, *_ = synthetic_job(calibration=calibration)
    path = save_through_go(job, tmp_path / "sequential.h5", monkeypatch)
    raw, mean, second, error, second_error = expected_statistics(labels, keep, MATRICES)
    for step in range(len(MODES)):
        expected = np.linalg.solve(MATRICES[:, :, step].T, raw[:, step].T).T
        np.testing.assert_allclose(job.data["p_mod_corrected"][:, step], expected)
    for key, expected in [("n_mean_corrected", mean), ("n2_mean_corrected", second),
                          ("n_err", error), ("n2_err", second_error)]:
        np.testing.assert_allclose(job.data[key], expected)
    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == set(job.data)
        assert all(isinstance(handle[key], h5py.Dataset) for key in handle)
        np.testing.assert_allclose(handle["multiparity_confusion_matrices"], MATRICES)
    for load_shots in (True, False):
        loaded = FockPopulationSequentialReadoutExperiment.from_h5file(path, load_shots=load_shots)
        assert ("idata" in loaded.data) == load_shots
        np.testing.assert_allclose(loaded.data["n_mean_corrected"], mean)
        np.testing.assert_array_equal(loaded.data["shot_counts"], job.data["shot_counts"])
        assert loaded.cfg.expt.ro_stors == MODES
        if load_shots:
            np.testing.assert_allclose(loaded.analyze()["n_mean_corrected"], mean)


def test_raw_reanalysis_clears_prior_correction_and_refuses_implicit_broadcast():
    job, labels, keep, *_ = synthetic_job(calibration=MATRICES.copy())
    job.analyze()
    job.cfg.expt.multiparity_confusion_matrices = None
    data = job.analyze()
    assert not any(key.endswith("_corrected") for key in data)
    assert "multiparity_confusion_matrices" not in data
    np.testing.assert_allclose(data["n_mean"], expected_statistics(labels, keep)[1])
    job.cfg.expt.multiparity_confusion_matrix = np.eye(4).tolist()
    with pytest.raises(ValueError, match="(?i)(singular|plural|per.mode|sequential|multiparity_confusion)"):
        job.analyze()


@pytest.mark.parametrize("value", [np.nan, -1., 2.])
def test_invalid_calibration_probabilities_are_not_applied(value):
    matrices = MATRICES.copy()
    matrices[0, 0, 1] = value
    job, *_ = synthetic_job(calibration=matrices)
    with pytest.raises(ValueError, match="row probabilities"):
        job.analyze()


def test_all_modes_compile_acquire_and_save_with_mock_instruments(tmp_path):
    station = mock_station(**pinned_config_set("preload_current"))
    assert station.is_mock, "refusing to acquire against real instruments"
    station.data_path = tmp_path / "data"
    station.data_path.mkdir()
    defaults = population_defaults(list(range(1, 8)))
    defaults.update(ro_stors=list(range(8)), reps=2, floquet_cycles=[0],
                    multiparity_confusion_matrix=None, multiparity_confusion_matrices=None)
    runner = CharacterizationRunner(station=station,
                                  ExptClass=FockPopulationSequentialReadoutExperiment,
                                  default_expt_cfg=defaults, show=False)
    job = runner.execute(use_queue=False, log=False, show=False,
                         go_kwargs=dict(analyze=True, save=True, progress=False))
    assert isinstance(job.prog, FockPopulationSequentialReadoutProgram)
    layout = readout_layout(job.cfg)
    assert job.cfg.read_num == layout["read_num"]
    assert np.asarray(job.data["idata"]).shape == (1, 2 * layout["read_num"])
    assert job.data["p_mod"].shape == (1, 8, 4)
    loaded = FockPopulationSequentialReadoutExperiment.from_h5file(job.fname, load_shots=False)
    np.testing.assert_array_equal(loaded.data["ro_stors"], np.arange(8))
    assert loaded.data["n_mean"].shape == (1, 8)
