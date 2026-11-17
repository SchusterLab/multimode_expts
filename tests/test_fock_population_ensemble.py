"""Disorder rows are fetched on board while pulse phases and shot identities survive."""
from copy import deepcopy

import numpy as np
import pytest
from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner
from experiments.mock_hardware import MockQickSoc
from experiments.qsim.fock_population import FockPopulationProgram
from experiments.qsim.fock_population_ensemble import (
    FockPopulationEnsembleExperiment, FockPopulationEnsembleProgram,
)
from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS


class DisorderMemorySoc(MockQickSoc):
    """Keep uploaded DMEM words without emulating pulse generation."""

    def __init__(self, soccfg):
        super().__init__(soccfg)
        self.uploads = []

    def load_mem(self, data, mem_sel="dmem", addr=0):
        assert mem_sel == "dmem"
        assert addr + len(data) <= self.soccfg["tprocs"][0]["dmem_size"]
        self.uploads.append((addr, np.asarray(data, dtype=np.int32).copy()))


def population_defaults(disorder_vectors):
    defaults = AttrDict(dict(
        expts=1, rounds=1, qubits=[0], f0g1_cavity=1, normalize=False,
        active_reset=True, man_reset=True, storage_reset=[1, 2],
        pre_relax_delay=100, relax_delay=200, prepulse=False, postpulse=True,
        init_fock=False, init_stor=0, ro_stor=2, readout="multiparity", parity_fast=False,
        cond_sec_phase=-90, phase_second_pulse=180, occupations=[1, 0, 0],
        swap_stors=[1, 2], update_phases=True, floquet_cycles=[0, 2, 4],
        swept_params=["floquet_cycle"], disorder_detunings_mhz=disorder_vectors,
        disorder_seed=42,
    ))
    defaults.update(ACTIVE_RESET_DEFAULTS)
    defaults.update(FLOQUET_DEFAULTS)
    return defaults


@pytest.fixture(scope="module")
def acquisition():
    station = mock_station(**pinned_config_set("preload_current"))
    assert station.is_mock
    connection_name = station.hardware_cfg["aliases"]["soc"]
    soc = DisorderMemorySoc(station.soccfg)
    station.im[connection_name] = soc
    disorder_vectors = [[.01, -.02], [-.03, .04], [.05, .06]]
    runner = CharacterizationRunner(
        station=station, ExptClass=FockPopulationEnsembleExperiment,
        default_expt_cfg=population_defaults(disorder_vectors), show=False)
    experiment = runner.execute(log=False, show=False, go_kwargs=dict(analyze=True))
    return experiment, soc


def bank_commands(program, page, frequency_words):
    """Trace the compiled bank operands and real outer/inner loops against DMEM."""
    instructions = program.prog_list
    labels = {instruction["label"]: index for index, instruction in enumerate(instructions)
              if "label" in instruction}
    registers = {(0, 14): 0}
    memory = {32 + index: int(word) for index, word in enumerate(np.ravel(frequency_words))}
    commands = []
    instruction_index = 0
    time_reference = 0
    while instruction_index < len(instructions):
        instruction = instructions[instruction_index]
        instruction_index += 1
        name, arguments = instruction["name"], instruction["args"]
        if name == "synci":
            time_reference += arguments[0]
            continue
        if name not in ("regwi", "bitwi", "mathi", "math", "memr", "loopnz", "set"):
            continue
        instruction_page = arguments[1] if name == "set" else arguments[0]
        outer_counter = name in ("regwi", "loopnz") and arguments[:2] == (0, 14)
        if instruction_page != page and not outer_counter:
            continue
        if name == "regwi":
            registers[arguments[0], arguments[1]] = arguments[2] & 0xffffffff
        elif name == "memr":
            registers[page, arguments[1]] = memory[registers[page, arguments[2]]]
        elif name in ("math", "mathi", "bitwi"):
            destination, source, operation, last = arguments[1:5]
            left = registers.get((page, source), 0)
            right = registers.get((page, last), 0) if name == "math" else last
            value = {"+": lambda: left + right, "-": lambda: left - right,
                     "<<": lambda: left << right, ">>": lambda: left >> right}[operation]()
            registers[page, destination] = value & 0xffffffff
        elif name == "loopnz":
            counter = (arguments[0], arguments[1])
            if registers[counter]:
                registers[counter] -= 1
                instruction_index = labels[arguments[2]]
        elif name == "set":
            pulse_parameters = tuple(registers.get((page, number), 0) for number in arguments[2:7])
            commands.append(pulse_parameters + (time_reference + registers.get((page, arguments[7]), 0),))
    return commands, registers


def test_mock_acquisition_uploads_table_and_preserves_realization_slots(acquisition):
    experiment, soc = acquisition
    assert experiment.cfg.expt.reps == 3
    assert len(soc.uploads) == 3
    assert experiment.data["readout_labels"].shape == (3, 3)
    assert experiment.data["herald_keep"].all()
    assert experiment.data["shot_realization_indices"].tolist() == [[0, 1, 2]] * 3
    for address, payload in soc.uploads:
        assert address == experiment.data["disorder_dmem_address"]
        np.testing.assert_array_equal(payload.view(np.uint32).reshape(3, 2),
                                      experiment.data["disorder_frequency_words"])
    np.testing.assert_array_equal(experiment.data["disorder_detunings_mhz"],
                                  experiment.cfg.expt.disorder_detunings_mhz)


def test_compiled_loop_visits_each_row_and_resets_phase(acquisition):
    experiment, _ = acquisition
    program = experiment.prog
    commands, registers = bank_commands(program, program.ensemble_page, program.disorder_frequency_words)
    expected_frequencies = []
    for frequencies in program.disorder_frequency_words:
        expected_frequencies.extend(list(frequencies) * 4)
    assert [command[0] for command in commands] == expected_frequencies
    phases = np.asarray([command[1] for command in commands]).reshape(3, 4, 2)
    np.testing.assert_array_equal(phases[0], phases[1])
    np.testing.assert_array_equal(phases[1], phases[2])
    assert registers[program.ensemble_page, program.ensemble_pointer_register] == 32 + 3 * 2


def test_zero_time_still_consumes_the_whole_table(acquisition):
    experiment, _ = acquisition
    cfg = deepcopy(experiment.cfg)
    cfg.expt.floquet_cycle = 0
    program = FockPopulationEnsembleProgram(experiment.soccfg, cfg)
    commands, registers = bank_commands(program, program.ensemble_page, program.disorder_frequency_words)
    assert commands == []
    assert registers[program.ensemble_page, program.ensemble_pointer_register] == 32 + 3 * 2


def test_identical_rows_match_existing_fixed_disorder_pulse_commands(acquisition, monkeypatch):
    # This checkout has no firmware HDL for the existing timing-check helper.
    monkeypatch.setattr("experiments.qsim.floquet_train.warn_if_tproc_falls_behind_floquet_train",
                        lambda *args: None)
    experiment, _ = acquisition
    cfg = deepcopy(experiment.cfg)
    cfg.expt.disorder_detunings_mhz = [[.01, -.02]] * 3
    ensemble = FockPopulationEnsembleProgram(experiment.soccfg, cfg)
    cfg.expt.detunings = [.01, -.02]
    fixed = FockPopulationProgram(experiment.soccfg, cfg)
    ensemble_commands, _ = bank_commands(ensemble, ensemble.ensemble_page, ensemble.disorder_frequency_words)
    fixed_commands, _ = bank_commands(fixed, ensemble.ensemble_page, ensemble.disorder_frequency_words)
    assert ensemble_commands == fixed_commands


def test_five_hundred_realizations_fit_without_unrolling_program(acquisition):
    experiment, _ = acquisition
    cfg = deepcopy(experiment.cfg)
    cfg.expt.swap_stors = [1, 2, 4, 5]
    cfg.expt.storage_reset = [1, 2, 4, 5]
    cfg.expt.occupations = [1, 0, 0, 0, 0]
    cfg.expt.disorder_detunings_mhz = [[0.] * 4] * 3
    small_program = FockPopulationEnsembleProgram(experiment.soccfg, cfg)
    cfg.expt.disorder_detunings_mhz = [[0.] * 4] * 500
    large_program = FockPopulationEnsembleProgram(experiment.soccfg, cfg)
    small_program.compile()
    large_program.compile()
    assert len(small_program.binprog) == len(large_program.binprog)
    assert large_program.loop_dims == [500]
    assert large_program.disorder_frequency_words.nbytes == 500 * 4 * 4


def test_table_cannot_overrun_dmem(acquisition):
    experiment, _ = acquisition
    cfg = deepcopy(experiment.cfg)
    cfg.expt.disorder_detunings_mhz = [[0., 0.]] * 2048
    with pytest.raises(ValueError, match="fit DMEM"):
        FockPopulationEnsembleProgram(experiment.soccfg, cfg)


def test_saved_file_reloads_table_and_labels_without_raw_iq(acquisition):
    experiment, _ = acquisition
    restored = FockPopulationEnsembleExperiment.from_h5file(experiment.fname, load_shots=False)
    assert "idata" not in restored.data
    for key in ("disorder_detunings_mhz", "disorder_frequency_words", "readout_labels",
                "herald_keep", "shot_realization_indices", "swap_stors", "occupations", "time_us"):
        np.testing.assert_array_equal(restored.data[key], experiment.data[key])


def test_herald_cut_keeps_original_realization_indices():
    experiment = FockPopulationEnsembleExperiment.__new__(FockPopulationEnsembleExperiment)
    experiment.cfg = AttrDict(dict(read_num=3, device=dict(readout=dict(threshold=[0.])),
        expt=dict(qubits=[0], readout="multiparity", active_reset=False, prep_herald="end",
                  occupations=[1, 0], swap_stors=[1], ro_stor=0)))
    experiment.data = dict(xpts=[0], idata=np.array([[-1, -1, -1, 1, 1, -1, -1, -1, 1]]),
                           qdata=np.zeros((1, 9)))
    experiment.fname = None
    experiment.analyze()
    assert experiment.data["readout_labels"].tolist() == [[0, 1, 2]]
    assert experiment.data["herald_keep"].tolist() == [[True, False, True]]
    assert experiment.data["shot_counts"].tolist() == [2]


