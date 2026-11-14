"""Warn when the tProc issues a Floquet pulse train slower than the train plays.

Why this exists
---------------
The tProc (QICK v1, ``axis_tproc64x32_x8``) does not play pulses itself. Each
``set`` instruction puts "play this pulse at time t" into a queue (the timed
FIFO) of its output channel, and a dispatcher sends it to the generator when
the clock reaches t. Every instruction takes some tProc clocks to run. If the
instructions between two pulses take longer than the time between the two
pulses, the tProc falls behind: the queue empties, the next ``set`` arrives
after its time t, and the dispatcher sends it at once, so the pulse starts
late. Short Floquet pulses with a small sync make this possible.

What it does
------------
1. Takes the instructions that one Floquet train added to ``program.prog_list``.
2. Runs through them in the order the tProc does, following each ``loopnz``.
3. For each pair of neighbouring pulses (a pulse slot) it gets two numbers:
   the scheduled clocks (``synci`` sum plus time register, as qick wrote them)
   and the instruction clocks (the tProc time to issue the next pulse).
4. Keeps a running balance of how far the tProc is ahead of the schedule,
   limited by the depth of the timed FIFO, and warns if a pulse starts late.

No clock counts are typed in here. They are read from the tProc v1 firmware
source in the qick checkout this repo installs (``../qick``, see
``pyproject.toml``):

* ``ctrl.sv``: the control state machine. Each state lasts one clock, so an
  instruction takes as many clocks as there are states on its path from
  ``DECODE_ST`` to the next ``DECODE_ST``.
* ``tproc64x32_x8.v``: the depth ``N`` of the timed FIFO ``fifo_time_i``.

This is a model of the firmware source. It assumes that the FPGA image on the
board was built from that source; it is not a measurement of the board.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import qick
from qick.asm_v1 import QickProgram


def tproc_v1_firmware_source_folder():
    """-> the tProc v1 source folder of the installed qick checkout."""
    qick_checkout = Path(qick.__file__).parents[2]
    return qick_checkout / "firmware" / "ip" / "axis_tproc64x32_x8_v1" / "src"


def _next_state_text_per_state(ctrl_text):
    """-> {state name: its code in the next-state block of ctrl.sv}, comments removed.

    The next-state block is the first ``case (state)`` ... ``endcase`` in ctrl.sv;
    the second one sets the outputs of each state.
    """
    case_block = ctrl_text[ctrl_text.index("case (state)"):]
    case_block = re.sub(r"//.*", "", case_block[:case_block.index("endcase")])
    state_labels = list(re.finditer(r"^\s*(\w+_ST)\s*:", case_block, re.MULTILINE))
    label_ends = [label.start() for label in state_labels[1:]] + [len(case_block)]
    return {label.group(1): case_block[label.end():end]
            for label, end in zip(state_labels, label_ends)}


def _first_state_per_opcode(decode_text):
    """-> {opcode: first state of that instruction}, from the DECODE_ST code."""
    first_state_per_opcode = {}
    for condition, state in re.findall(r"if\s*\((.*?)\)\s*state\s*<=\s*(\w+_ST)",
                                       decode_text, re.DOTALL):
        for opcode in re.findall(r"\d+'b([01]+)", condition):
            first_state_per_opcode[int(opcode, 2)] = state
    return first_state_per_opcode


def _state_transitions(state_text):
    """-> [(condition, next state)] of one state; condition is 'if (...)', 'else' or None."""
    found = re.findall(r"(?:(if|else)\s*(\(.*?\))?\s*)?state\s*<=\s*(\w+_ST)",
                       state_text, re.DOTALL)
    return [(f"{keyword} {condition}".strip() if keyword else None, next_state)
            for keyword, condition, next_state in found]


def _clocks_per_branch(first_state, next_state_text_per_state):
    """-> {branch taken: clocks from DECODE_ST to the next DECODE_ST} of one instruction.

    A state with if / else is a branch, and the key names the branch taken
    ('always' for an instruction without one). A state with a single ``if`` and no
    ``else`` waits until the condition holds; it is counted as one clock, which
    is the case without waiting.
    """
    clocks_per_branch = {}
    clocks_through_first_state = 2      # DECODE_ST, then the first state: one clock each
    open_paths = [(first_state, clocks_through_first_state, "always")]
    while open_paths:
        state, clocks, branch = open_paths.pop()
        transitions = _state_transitions(next_state_text_per_state[state])
        for condition, next_state in transitions:
            path_branch = condition if len(transitions) > 1 else branch
            if next_state == "DECODE_ST":
                clocks_per_branch[path_branch] = clocks
            else:
                open_paths.append((next_state, clocks + 1, path_branch))
    return clocks_per_branch


def read_tproc_v1_clocks_per_instruction(firmware_folder):
    """-> {qick instruction name: {branch taken: clocks}}, read from ctrl.sv.

    The opcode of each instruction comes from qick's own table
    (``QickProgram.instructions[name]['bin']``). For example
    ``{'regwi': {'always': 2}, 'loopnz': {'if (alu_zero)': 3, 'else': 6}, ...}``.
    """
    next_state_text_per_state = _next_state_text_per_state(
        (firmware_folder / "ctrl.sv").read_text())
    first_state_per_opcode = _first_state_per_opcode(next_state_text_per_state["DECODE_ST"])
    return {name: _clocks_per_branch(first_state_per_opcode[spec["bin"]], next_state_text_per_state)
            for name, spec in QickProgram.instructions.items() if "bin" in spec}


def read_tproc_v1_timed_fifo_depth(firmware_folder):
    """-> how many timed instructions one tProc output channel can queue (parameter N of fifo_time_i)."""
    top_level_text = (firmware_folder / "tproc64x32_x8.v").read_text()
    return int(re.search(r"\.N\s*\(\s*(\d+)\s*\)\s*\)\s*fifo_time_i", top_level_text).group(1))


def tproc_steps_with_loops_unrolled(instructions, clocks_per_instruction):
    """-> (clocks, scheduled_starts): one entry per instruction, in the order the tProc runs them.

    instructions: a slice of a qick v1 ``prog_list``. A ``loopnz`` is followed as the
    firmware does: while its counter is not 0, subtract 1 and jump back to its label.
    scheduled_starts: for a ``set``, the scheduled start in tProc clocks from the start
    of the slice (the ``synci`` sum before it plus its time register); NaN otherwise.
    Only ``regwi`` writes are tracked, because the loop counter and the time register
    are written with ``regwi``. Comments are not run by the tProc and are skipped.
    """
    label_index = {instruction["label"]: index
                   for index, instruction in enumerate(instructions) if "label" in instruction}
    register_values = {}
    time_offset = 0
    clocks, scheduled_starts = [], []
    index = 0
    while index < len(instructions):
        name, args = instructions[index]["name"], instructions[index]["args"]
        index += 1
        if name == "comment":
            continue
        branch, scheduled_start = "always", np.nan
        if name == "loopnz":
            page, register, label = args
            branch = "if (alu_zero)"            # the counter is 0: no jump
            if register_values[(page, register)] != 0:
                register_values[(page, register)] -= 1
                branch, index = "else", label_index[label]
        elif name == "regwi":
            page, register, value = args
            register_values[(page, register)] = value
        elif name == "synci":
            time_offset += args[0]
        elif name == "set":
            page, time_register = args[1], args[7]
            scheduled_start = time_offset + register_values.get((page, time_register), 0)
        clocks.append(clocks_per_instruction[name][branch])
        scheduled_starts.append(scheduled_start)
    return np.array(clocks), np.array(scheduled_starts)


def floquet_pulse_slots(clocks, scheduled_starts):
    """-> (scheduled_clocks, instruction_clocks), one entry per pair of neighbouring pulses.

    scheduled_clocks: time from one pulse start to the next, as the program schedules it.
    instruction_clocks: tProc clocks from just after one ``set`` to the end of the next ``set``.
    """
    set_indices = np.flatnonzero(~np.isnan(scheduled_starts))
    clocks_after_each_step = np.cumsum(clocks)
    scheduled_clocks = np.diff(scheduled_starts[set_indices])
    instruction_clocks = np.diff(clocks_after_each_step[set_indices])
    return scheduled_clocks, instruction_clocks


def largest_pulse_lateness_clocks(scheduled_clocks, instruction_clocks, timed_fifo_depth,
                                  extra_sync_clocks=0):
    """-> (largest lateness in tProc clocks, number of that pulse in the train, from 1).

    lead is how far the tProc is ahead of the schedule when it issues a pulse. It
    starts at 0 (on time at the first pulse) and changes by the spare clocks of
    each slot. With ``timed_fifo_depth`` pulses waiting in the queue the tProc
    stops, so lead is at most ``timed_fifo_depth - 1`` slots. A negative lead is
    a late pulse.
    """
    lead, largest_lateness, latest_pulse = 0., 0., 0
    for pulse, (scheduled, needed) in enumerate(
            zip(scheduled_clocks + extra_sync_clocks, instruction_clocks), start=2):
        lead = min(lead + scheduled - needed, (timed_fifo_depth - 1) * scheduled)
        if -lead > largest_lateness:
            largest_lateness, latest_pulse = -lead, pulse
    return largest_lateness, latest_pulse


def smallest_extra_sync_without_late_pulses(scheduled_clocks, instruction_clocks, timed_fifo_depth):
    """-> the smallest number of clocks to add to every slot so that no pulse starts late."""
    extra_sync_clocks = 0
    while largest_pulse_lateness_clocks(scheduled_clocks, instruction_clocks, timed_fifo_depth,
                                        extra_sync_clocks)[0] > 0:
        extra_sync_clocks += 1
    return extra_sync_clocks


def warn_if_tproc_falls_behind_floquet_train(program, train_name, first_instruction_index):
    """Warn if the Floquet train in ``program.prog_list[first_instruction_index:]`` has late pulses.

    Call it right after the code that emits the train, with the length of
    ``prog_list`` from just before it. It only reads ``prog_list``.
    """
    firmware_folder = tproc_v1_firmware_source_folder()
    clocks, scheduled_starts = tproc_steps_with_loops_unrolled(
        program.prog_list[first_instruction_index:],
        read_tproc_v1_clocks_per_instruction(firmware_folder))
    scheduled_clocks, instruction_clocks = floquet_pulse_slots(clocks, scheduled_starts)
    timed_fifo_depth = read_tproc_v1_timed_fifo_depth(firmware_folder)
    largest_lateness, latest_pulse = largest_pulse_lateness_clocks(
        scheduled_clocks, instruction_clocks, timed_fifo_depth)
    if largest_lateness == 0:
        return
    short_slots = instruction_clocks > scheduled_clocks
    worst_slot = np.argmax(instruction_clocks - scheduled_clocks)
    extra_sync_clocks = smallest_extra_sync_without_late_pulses(
        scheduled_clocks, instruction_clocks, timed_fifo_depth)
    warnings.warn(
        f"{train_name}: the tProc issues the pulses slower than they play. "
        f"{short_slots.sum()} of {short_slots.size} pulse slots need more tProc clocks than "
        f"they have (worst: {instruction_clocks[worst_slot]} clocks for a "
        f"{scheduled_clocks[worst_slot]:g}-clock slot). Pulse {latest_pulse} of "
        f"{short_slots.size + 1} starts {largest_lateness:g} clocks "
        f"({1e3 * program.cycles2us(largest_lateness):.0f} ns) late. "
        f"Add {extra_sync_clocks} clocks to the sync between pulses to remove the lateness.",
        stacklevel=2)
