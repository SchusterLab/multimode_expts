"""The tProc timing check for Floquet trains (experiments/floquet_instruction_timing.py).

The clock counts are read from the qick firmware source; the first test pins
what that source says now, so a change in qick's firmware shows up here. The
other tests emit instructions through qick's own methods, so the instruction
format is qick's, and one test compiles a real FockPopulationProgram on the
mock station.
"""
import warnings
from copy import deepcopy

import pandas as pd
import pytest
from qick.asm_v1 import QickProgram
from slab import AttrDict

from experiments.floquet_instruction_timing import (
    floquet_pulse_slots,
    read_tproc_v1_clocks_per_instruction,
    read_tproc_v1_timed_fifo_depth,
    tproc_steps_with_loops_unrolled,
    tproc_v1_firmware_source_folder,
    warn_if_tproc_falls_behind_floquet_train,
)
from experiments.floquet_timing import committed_soccfg
from experiments.qsim.fock_population import FockPopulationProgram
from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
from experiments.qsim.utils import ensure_list_in_cfg
from tests.test_fock_population_acquire_mock import population_defaults

PAGE, PHASE_REGISTER, TIME_REGISTER, LOOP_REGISTER = 1, 10, 11, 12
TPROC_CHANNEL = 2
FAST_PHASE_REGISTER_VALUE = 2**20             # below 2**30: one regwi
SLOW_PHASE_REGISTER_VALUE = 2**31 + 1         # regwi, bitwi and mathi (qick's safe_regwi)


def emit_preloaded_floquet_pulses(program, phase_register_values, slot_clocks):
    """As the preloaded register bank: phase write, one raw set at time 0, sync."""
    for value in phase_register_values:
        program.safe_regwi(PAGE, PHASE_REGISTER, value)
        program.set(TPROC_CHANNEL, PAGE, 1, PHASE_REGISTER, 3, 4, 5, 0)
        program.synci(slot_clocks)


def fails_on_any_warning(program, first_instruction_index=0):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_tproc_falls_behind_floquet_train(program, "test train", first_instruction_index)


def test_firmware_source_gives_the_tproc_v1_clocks_and_fifo_depth():
    firmware_folder = tproc_v1_firmware_source_folder()
    clocks = read_tproc_v1_clocks_per_instruction(firmware_folder)

    assert {name: clocks[name] for name in ("regwi", "synci", "set", "mathi", "math", "bitwi")} == {
        "regwi": {"always": 2}, "synci": {"always": 3}, "set": {"always": 4},
        "mathi": {"always": 5}, "math": {"always": 5}, "bitwi": {"always": 5}}
    assert clocks["loopnz"] == {"if (alu_zero)": 3, "else": 6}
    assert read_tproc_v1_timed_fifo_depth(firmware_folder) == 16


def test_pulses_that_fit_their_slots_give_no_warning():
    program = QickProgram(committed_soccfg())
    emit_preloaded_floquet_pulses(program, [FAST_PHASE_REGISTER_VALUE] * 50, slot_clocks=11)

    fails_on_any_warning(program)


def test_slow_phase_writes_warn_and_name_the_extra_sync():
    program = QickProgram(committed_soccfg())
    emit_preloaded_floquet_pulses(program, [SLOW_PHASE_REGISTER_VALUE] * 50, slot_clocks=11)

    # 2 + 5 + 5 (phase write) + 4 (set) + 3 (synci) = 19 clocks for an 11-clock slot.
    with pytest.warns(UserWarning, match=r"worst: 19 clocks for a 11-clock slot.*Add 8 clocks"):
        warn_if_tproc_falls_behind_floquet_train(program, "test train", 0)


def test_one_slow_slot_after_spare_clocks_gives_no_warning():
    # The frequency-mode error amplification: a phase flip every 40 pulses.
    program = QickProgram(committed_soccfg())
    one_group = [FAST_PHASE_REGISTER_VALUE] * 39 + [SLOW_PHASE_REGISTER_VALUE]
    emit_preloaded_floquet_pulses(program, one_group * 5, slot_clocks=11)

    fails_on_any_warning(program)


def test_only_the_instructions_after_the_first_index_are_checked():
    program = QickProgram(committed_soccfg())
    emit_preloaded_floquet_pulses(program, [SLOW_PHASE_REGISTER_VALUE] * 50, slot_clocks=11)
    first_instruction_index = len(program.prog_list)
    emit_preloaded_floquet_pulses(program, [FAST_PHASE_REGISTER_VALUE] * 50, slot_clocks=11)

    fails_on_any_warning(program, first_instruction_index)


def test_hardware_loop_body_runs_the_loop_count_and_each_jump_costs_its_clocks():
    loop_count = 5
    program = QickProgram(committed_soccfg())
    program.regwi(PAGE, LOOP_REGISTER, loop_count - 1)
    program.label("FLOQUET_LOOP")
    program.set(TPROC_CHANNEL, PAGE, 1, PHASE_REGISTER, 3, 4, 5, 0)
    program.synci(20)
    program.loopnz(PAGE, LOOP_REGISTER, "FLOQUET_LOOP")
    clocks = read_tproc_v1_clocks_per_instruction(tproc_v1_firmware_source_folder())

    scheduled_clocks, instruction_clocks = floquet_pulse_slots(
        *tproc_steps_with_loops_unrolled(program.prog_list, clocks))

    assert scheduled_clocks.tolist() == [20] * (loop_count - 1)
    # synci 3 + loopnz jump 6 + set 4
    assert instruction_clocks.tolist() == [13] * (loop_count - 1)


def test_pulse_starts_from_the_time_register_without_synci():
    # The error amplification with sync 0: no synci, each pulse at its own time.
    program = QickProgram(committed_soccfg())
    for time in (0, 10, 20, 30):
        program.regwi(PAGE, TIME_REGISTER, time)
        program.set(TPROC_CHANNEL, PAGE, 1, PHASE_REGISTER, 3, 4, 5, TIME_REGISTER)
    clocks = read_tproc_v1_clocks_per_instruction(tproc_v1_firmware_source_folder())

    scheduled_clocks, instruction_clocks = floquet_pulse_slots(
        *tproc_steps_with_loops_unrolled(program.prog_list, clocks))

    assert scheduled_clocks.tolist() == [10, 10, 10]
    assert instruction_clocks.tolist() == [6, 6, 6]      # regwi 2 + set 4


SWAP_STORS = [1, 2]


@pytest.fixture(scope="module")
def short_pulse_station(tmp_path_factory):
    """Mock station on the pinned set, with every Floquet plateau cut short."""
    config_set = pinned_config_set("preload_current")
    floquet_table = pd.read_csv(config_set["floquet_file"])
    floquet_table["len (mus)"] = 0.0075
    short_floquet_file = tmp_path_factory.mktemp("floquet") / "CFG-FL-20260908-00001.csv"
    floquet_table.to_csv(short_floquet_file, index=False)
    return mock_station(**dict(config_set, floquet_file=str(short_floquet_file)))


@pytest.fixture(scope="module")
def pinned_station():
    return mock_station(**pinned_config_set("preload_current"))


def compile_fock_population_program(station, **expt_overrides):
    """As tests/program_asm_golden.py builds a Program from a config."""
    cfg = AttrDict(deepcopy(station.hardware_cfg))
    cfg.device.storage._ds_storage = station.ds_storage
    cfg.device.storage._ds_floquet = station.ds_floquet
    cfg.expt = population_defaults(SWAP_STORS)
    cfg.expt.update(expt_overrides)
    cfg.device.readout.relax_delay = [cfg.expt.relax_delay]
    ensure_list_in_cfg(cfg)
    return FockPopulationProgram(soccfg=station.soccfg, cfg=cfg)


@pytest.mark.parametrize("floquet_hardware_loop", [True, False])
def test_short_floquet_pulses_with_one_sync_cycle_warn(short_pulse_station, floquet_hardware_loop):
    with pytest.warns(UserWarning, match="Floquet scramble of"):
        compile_fock_population_program(
            short_pulse_station, floquet_cycle=40, scramble_sync_cycles=1,
            floquet_hardware_loop=floquet_hardware_loop)


def test_pinned_floquet_pulses_give_no_warning(pinned_station):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*tProc issues the pulses slower")
        compile_fock_population_program(pinned_station, floquet_cycle=40, scramble_sync_cycles=1)
