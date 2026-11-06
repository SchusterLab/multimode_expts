"""One-job sequential calibration, normal HDF5 saving and mock acquisition."""

import json
from copy import deepcopy

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.agp import calibration_shots
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutExperiment,
    FockPopulationSequentialReadoutProgram,
)
from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
from experiments.qsim.multiparity_confusion_matrix_sequential import (
    MultiParityConfusionMatrixSequentialExperiment,
)
from tests.test_fock_population_acquire_mock import population_defaults
from tests.test_fock_population_sequential import synthetic_job


def test_one_job_selects_each_target_step_and_preserves_raw_shots(tmp_path, monkeypatch):
    labels = np.zeros((12, 6, 3), dtype=int)
    for step in range(3):
        for number in range(4):
            point = 4 * step + number
            labels[point] = np.array([3, 2, 1])
            labels[point, :, step] = [number] * 4 + [(number + 1) % 4, (number + 2) % 4]
    keep_pattern = [[True, True, True, True, True, False],
                    [True, True, True, False, True, False],
                    [True, True, False, True, False, False]]
    keep = []
    for _ in range(4):
        for row in keep_pattern:
            keep.append(row.copy())
    keep = np.asarray(keep, dtype=bool)
    template, *_ = synthetic_job(cycles=list(range(12)), labels=labels, keep=keep)
    calibration = MultiParityConfusionMatrixSequentialExperiment.__new__(
        MultiParityConfusionMatrixSequentialExperiment)
    calibration.cfg = deepcopy(template.cfg)
    calibration.cfg.expt.background_occupations = [1, 0, 2]
    calibration.cfg.expt.prepulse = True
    calibration.cfg.expt.multiparity_confusion_matrix = np.eye(4).tolist()
    calibration.cfg.expt.multiparity_confusion_matrices = np.zeros((4, 4, 3)).tolist()
    calibration.derived_params = template.derived_params
    calibration.fname = str(tmp_path / "sequential_calibration.h5")
    points = dict(idata=template.data["idata"], qdata=template.data["qdata"],
                  avgi=[0.] * 12, avgq=[0.] * 12)

    def acquire_population(self, progress=False, debug=False):
        assert self.cfg.expt.multiparity_confusion_matrix is None
        assert self.cfg.expt.multiparity_confusion_matrices is None
        assert self.cfg.expt.prepulse is False
        assert self.cfg.expt.floquet_cycle == 0
        assert self.cfg.expt.floquet_cycles == [0]
        assert self.cfg.expt.swept_params == ["occupations"]
        axes = self.sweep_axes()
        assert len(axes[0][1]) == 12
        for step in range(3):
            for number in range(4):
                expected = [1, 0, 2]
                expected[step] = number
                assert axes[0][1][4 * step + number] == expected
        return self.shape_data(axes, points)

    monkeypatch.setattr(FockPopulationSequentialReadoutExperiment, "acquire", acquire_population)
    calibration.go(analyze=True, save=True)
    data = calibration.data
    expected = np.zeros((4, 4, 3))
    counts = np.zeros((4, 3), dtype=int)
    for step in range(3):
        for number in range(4):
            point = 4 * step + number
            accepted = labels[point, keep[point], step]
            expected[number, :, step] = np.bincount(accepted, minlength=4) / len(accepted)
            counts[number, step] = len(accepted)
    np.testing.assert_allclose(data["multiparity_confusion_matrices"], expected)
    np.testing.assert_array_equal(data["calibration_shot_counts"], counts)
    expected_shot_counts = []
    for row in keep:
        accepted = sum(row)
        expected_shot_counts.append([accepted, accepted, accepted])
    np.testing.assert_array_equal(data["shot_counts"], expected_shot_counts)
    np.testing.assert_array_equal(data["xpts"], np.arange(12))
    np.testing.assert_array_equal(data["calibration_steps"], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(data["prepared_numbers"], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(data["background_occupations"], [1, 0, 2])
    assert "p_mod_corrected" not in data
    assert "time_us" not in data
    with h5py.File(calibration.fname, "r") as handle:
        assert handle["idata"].shape == (12, 6 * calibration.cfg.read_num)
        np.testing.assert_array_equal(handle["idata"], template.data["idata"])
        np.testing.assert_array_equal(handle["prepared_occupations"],
                                      calibration.sweep_axes()[0][1])
        assert set(handle) == set(data)
        config = json.loads(handle.attrs["config"])
        assert config["read_num"] == calibration.cfg.read_num
        assert config["expt"]["multiparity_confusion_matrices"] is None
    assert calibration_shots(dict(fname=calibration.fname)) == 6
    for load_shots in (True, False):
        loaded = MultiParityConfusionMatrixSequentialExperiment.from_h5file(
            calibration.fname, load_shots=load_shots)
        np.testing.assert_allclose(loaded.data["multiparity_confusion_matrices"], expected)
        assert ("idata" in loaded.data) == load_shots
        if load_shots:
            np.testing.assert_allclose(loaded.analyze()["multiparity_confusion_matrices"], expected)
    target, *_ = synthetic_job(calibration=calibration.fname)
    target.analyze()
    for step in range(3):
        corrected = np.linalg.solve(expected[:, :, step].T, target.data["p_mod"][:, step].T).T
        np.testing.assert_allclose(target.data["p_mod_corrected"][:, step], corrected)
    fig, _ = calibration.display()
    matplotlib.pyplot.close(fig)


def test_normal_runner_compiles_all_calibration_settings_in_one_mock_job(tmp_path):
    station = mock_station(**pinned_config_set("preload_current"))
    assert station.is_mock, "refusing to acquire against real instruments"
    station.data_path = tmp_path / "data"
    station.data_path.mkdir()
    defaults = population_defaults([1])
    defaults.update(ro_stors=[0, 1], reps=2, active_reset=False, prepulse=True,
                    multiparity_confusion_matrix=np.eye(4).tolist(),
                    multiparity_confusion_matrices=np.zeros((4, 4, 2)).tolist())
    runner = CharacterizationRunner(station=station,
                                  ExptClass=MultiParityConfusionMatrixSequentialExperiment,
                                  default_expt_cfg=defaults, show=False)
    calibration = runner.execute(use_queue=False, postprocess=False, log=False, show=False,
                                 go_kwargs=dict(analyze=True, save=True, progress=False))
    assert isinstance(calibration.prog, FockPopulationSequentialReadoutProgram)
    assert calibration.data["multiparity_confusion_matrices"].shape == (4, 4, 2)
    assert calibration.data["p_mod"].shape == (8, 2, 4)
    assert np.asarray(calibration.data["idata"]).shape == (8, 2 * calibration.cfg.read_num)
    np.testing.assert_array_equal(calibration.data["calibration_shot_counts"], np.full((4, 2), 2))
    np.testing.assert_array_equal(calibration.data["ro_stors"], [0, 1])
    assert calibration.cfg.expt.floquet_cycle == 0
    assert calibration.cfg.expt.prepulse is False
    assert calibration.cfg.expt.multiparity_confusion_matrix is None
    assert calibration.cfg.expt.multiparity_confusion_matrices is None
    loaded = MultiParityConfusionMatrixSequentialExperiment.from_h5file(
        calibration.fname, load_shots=False)
    np.testing.assert_allclose(loaded.data["multiparity_confusion_matrices"],
                               calibration.data["multiparity_confusion_matrices"])
