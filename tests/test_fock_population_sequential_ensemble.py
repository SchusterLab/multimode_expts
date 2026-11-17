"""Heralded sequential lanes preserve ensemble identities and reuse pulse loops."""

from copy import deepcopy

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.fock_population import FockPopulationExperiment, FockPopulationProgram
from experiments.qsim.fock_population_sequential import (
    FockPopulationSequentialReadoutProgram,
    SequentialMultiparityProgramMixin,
)
from experiments.qsim.fock_population_sequential_ensemble import (
    FockPopulationSequentialEnsembleExperiment,
    FockPopulationSequentialEnsembleProgram,
    FockPopulationSequentialHeraldExperiment,
    FockPopulationSequentialHeraldProgram,
    HeraldedSequentialMultiparityProgramMixin,
    heralded_readout_layout,
)
from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
from experiments.qsim.multiparity_confusion_matrix_sequential import (
    HeraldedMultiParityConfusionMatrixSequentialExperiment,
    HeraldedSequentialCalibrationProgram,
)
from tests.test_fock_population_acquire_mock import population_defaults
from tests.test_fock_population_ensemble import DisorderMemorySoc, bank_commands
from tests.test_fock_population_sequential import synthetic_job


def ancestor_path_count(derived_class, target_class):
    """Count paths only to the application base, without judging QICK's graph."""
    if derived_class is target_class:
        return 1
    return sum(ancestor_path_count(parent, target_class) for parent in derived_class.__bases__)


@pytest.mark.parametrize("program_class", [
    FockPopulationSequentialReadoutProgram, FockPopulationSequentialHeraldProgram,
    FockPopulationSequentialEnsembleProgram, HeraldedSequentialCalibrationProgram,
])
def test_programs_have_one_application_program_chain(program_class):
    assert ancestor_path_count(program_class, FockPopulationProgram) == 1
    assert ancestor_path_count(SequentialMultiparityProgramMixin, FockPopulationProgram) == 0
    assert ancestor_path_count(HeraldedSequentialMultiparityProgramMixin, FockPopulationProgram) == 0


@pytest.mark.parametrize("experiment_class", [
    FockPopulationSequentialHeraldExperiment, FockPopulationSequentialEnsembleExperiment,
    HeraldedMultiParityConfusionMatrixSequentialExperiment,
])
def test_experiments_have_one_application_experiment_chain(experiment_class):
    assert ancestor_path_count(experiment_class, FockPopulationExperiment) == 1


class GroundHeraldSoc(DisorderMemorySoc):
    """Return unambiguous ground readouts instead of the mock's threshold-ambiguous zero."""

    def poll_data(self):
        batches = super().poll_data()
        for _, (channels, _) in batches:
            for channel in channels:
                channel[:, 0] = -10**9
        return batches


def heralded_mock_station(data_path):
    station = mock_station(**pinned_config_set("preload_current"))
    assert station.is_mock
    station.data_path = data_path
    soc = GroundHeraldSoc(station.soccfg)
    station.im[station.hardware_cfg["aliases"]["soc"]] = soc
    return station, soc


@pytest.fixture(autouse=True)
def unavailable_firmware_timing_check(monkeypatch):
    # The installed QICK wheel has no firmware HDL; pulse/readout checks remain real.
    monkeypatch.setattr("experiments.qsim.floquet_train.warn_if_tproc_falls_behind_floquet_train",
                        lambda *args: None)


@pytest.fixture(scope="module")
def ensemble_acquisition(tmp_path_factory):
    station, soc = heralded_mock_station(tmp_path_factory.mktemp("sequential_ensemble"))
    defaults = population_defaults([1, 2])
    defaults.update(
        active_reset=False, ro_stors=[0, 1, 2], prep_herald="end", floquet_cycles=[0, 2],
        occupations=[1, 1, 0], disorder_detunings_mhz=[[.01, -.02], [-.03, .04], [.05, .06]],
        disorder_patterns=[[1., -1.], [-1., 1.], [1., -1.]],
    )
    runner = CharacterizationRunner(
        station=station, ExptClass=FockPopulationSequentialEnsembleExperiment,
        default_expt_cfg=defaults, show=False)
    experiment = runner.execute(use_queue=False, postprocess=False, log=False, show=False,
                                go_kwargs=dict(analyze=True, save=True, progress=False))
    return experiment, soc


def test_full_labels_and_table_survive_sequential_mock_acquisition(ensemble_acquisition):
    experiment, soc = ensemble_acquisition
    assert isinstance(experiment.prog, FockPopulationSequentialEnsembleProgram)
    assert experiment.cfg.read_num == 8  # Two prep heralds and three science pairs.
    assert experiment.data["readout_labels"].shape == (2, 3, 3)
    assert experiment.data["herald_keep"].shape == (2, 3)
    assert experiment.data["p_mod"].shape == (2, 3, 4)
    assert len(soc.uploads) == 2
    np.testing.assert_array_equal(experiment.data["shot_realization_indices"], [[0, 1, 2]] * 2)
    np.testing.assert_array_equal(experiment.data["disorder_patterns"], experiment.cfg.expt.disorder_patterns)
    restored = FockPopulationSequentialEnsembleExperiment.from_h5file(
        experiment.fname, load_shots=False)
    for key in ("readout_labels", "herald_keep", "disorder_patterns", "readout_lane_pairs",
                "shot_realization_indices", "disorder_frequency_words"):
        np.testing.assert_array_equal(restored.data[key], experiment.data[key])
    assert restored.analyze() is restored.data


def test_sequential_readouts_do_not_consume_extra_disorder_rows(ensemble_acquisition):
    experiment, _ = ensemble_acquisition
    program = experiment.prog
    commands, registers = bank_commands(program, program.ensemble_page,
                                        program.disorder_frequency_words)
    expected = [int(word) for row in program.disorder_frequency_words
                for _ in range(2) for word in row]
    assert [command[0] for command in commands] == expected
    assert registers[program.ensemble_page, program.ensemble_pointer_register] == 32 + 3 * 2


def test_combined_program_initializes_common_program_once(ensemble_acquisition, monkeypatch):
    experiment, _ = ensemble_acquisition
    original_initialize = FockPopulationProgram.initialize
    initialized_programs = []

    def record_initialize(program):
        initialized_programs.append(program)
        return original_initialize(program)

    monkeypatch.setattr(FockPopulationProgram, "initialize", record_initialize)
    program = FockPopulationSequentialEnsembleProgram(experiment.soccfg, deepcopy(experiment.cfg))
    assert initialized_programs == [program]
    assert program.cfg.expt.reps == len(program.cfg.expt.disorder_detunings_mhz)


def test_herald_rejection_keeps_all_modes_in_the_same_original_slot():
    template, labels, *_ = synthetic_job(heralds=False)
    experiment = FockPopulationSequentialEnsembleExperiment.__new__(
        FockPopulationSequentialEnsembleExperiment)
    experiment.cfg = deepcopy(template.cfg)
    experiment.cfg.expt.prep_herald = "end"
    layout = heralded_readout_layout(experiment.cfg)
    experiment.cfg.read_num = layout["read_num"]
    original = np.asarray(template.data["idata"]).reshape(3, 6, 6)
    prep_reads = np.full((3, 6, 2), -2.)
    prep_reads[:, 2, 1] = 3.
    shots = np.concatenate((prep_reads, original), axis=-1)
    experiment.data = dict(template.data, idata=shots.reshape(3, -1),
                           qdata=np.zeros((3, 6 * 8)))
    experiment.data.update({key: np.asarray(value, dtype=int) for key, value in layout.items()})
    experiment.fname = None
    experiment.derived_params = template.derived_params
    experiment.analyze()
    np.testing.assert_array_equal(experiment.data["readout_labels"], labels)
    np.testing.assert_array_equal(experiment.data["herald_keep"],
                                  [[True, True, False, True, True, True]] * 3)
    np.testing.assert_array_equal(experiment.data["shot_counts"], np.full((3, 3), 5))
    np.testing.assert_array_equal(experiment.decode_readouts(0)["labels"],
                                  labels[0, [0, 1, 3, 4, 5]])


def test_each_reset_selects_final_heralds_without_rejecting_successful_feedback():
    template, *_ = synthetic_job(heralds=False)
    experiment = FockPopulationSequentialHeraldExperiment.__new__(
        FockPopulationSequentialHeraldExperiment)
    experiment.cfg = deepcopy(template.cfg)
    experiment.cfg.expt.prep_herald = "each_reset"
    layout = experiment.readout_layout()
    assert layout["preselection_lanes"] == [2, 5]
    np.testing.assert_array_equal(layout["readout_lane_pairs"], [[6, 7], [8, 9], [10, 11]])
    experiment.cfg.read_num = layout["read_num"]
    shots = np.full((2, layout["read_num"]), -2.)
    shots[:, [0, 1, 3, 4]] = 3.  # Excited reset inputs are legitimate feedback events.
    shots[1, 5] = 3.  # Only a failed final herald rejects the second shot.
    experiment.data = dict(layout, idata=shots.reshape(1, -1), qdata=np.zeros((1, shots.size)))
    decoded = experiment.decode_readouts(accepted_only=False)
    np.testing.assert_array_equal(decoded["keep"], [True, False])
    assert decoded["labels"].shape == (2, 3)


def test_heralded_sequential_calibration_compiles_vacuum_and_storage_preparations(tmp_path):
    station, _ = heralded_mock_station(tmp_path)
    defaults = population_defaults([1])
    defaults.update(ro_stors=[0, 1], reps=2, active_reset=False, prep_herald="end")
    runner = CharacterizationRunner(
        station=station, ExptClass=HeraldedMultiParityConfusionMatrixSequentialExperiment,
        default_expt_cfg=defaults, show=False)
    calibration = runner.execute(use_queue=False, postprocess=False, log=False, show=False,
                                 go_kwargs=dict(analyze=True, save=True, progress=False))
    assert isinstance(calibration.prog, HeraldedSequentialCalibrationProgram)
    assert calibration.cfg.read_num == 5
    assert np.asarray(calibration.data["idata"]).shape == (8, 2 * 5)
    assert calibration.data["multiparity_confusion_matrices"].shape == (4, 4, 2)
    np.testing.assert_array_equal(calibration.data["readout_lane_pairs"], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(calibration.data["calibration_shot_counts"], np.full((4, 2), 2))
    vacuum = deepcopy(calibration.cfg)
    vacuum.expt.occupations = [0, 0]
    vacuum_program = HeraldedSequentialCalibrationProgram(calibration.soccfg, vacuum)
    heralds = []
    vacuum_program.post_selection_measure = lambda **kwargs: heralds.append(kwargs)
    vacuum_program._prepare_occupation()
    assert len(heralds) == 1


def test_ordinary_heralded_walk_needs_no_disorder_table(tmp_path):
    station, _ = heralded_mock_station(tmp_path)
    defaults = population_defaults([1])
    defaults.update(ro_stors=[0, 1], reps=2, active_reset=False, prep_herald="end")
    runner = CharacterizationRunner(
        station=station, ExptClass=FockPopulationSequentialHeraldExperiment,
        default_expt_cfg=defaults, show=False)
    walk = runner.execute(use_queue=False, postprocess=False, log=False, show=False,
                          go_kwargs=dict(analyze=True, save=True, progress=False))
    assert "disorder_detunings_mhz" not in walk.cfg.expt
    assert walk.cfg.read_num == 5
    assert walk.data["n_mean"].shape == (1, 2)
    np.testing.assert_array_equal(walk.data["readout_lane_pairs"], [[1, 2], [3, 4]])
