# -*- coding: utf-8 -*-
"""Pulse-level golden for the non-MBR qsim Programs, one expt config each.

The net under the Program tree refactor (``docs/qsim/program_tree_plan.md``,
steps 10B-10D): the two templates (``QsimBaseProgram``, ``DarkBaseProgram``)
merge, the mixins become classes in one chain, and a ``readout`` key replaces
the readout booleans. None of this may change what a Program plays, except
where the plan says so (its section 7), and those changes must show here.

``asm_golden`` pins the MBR job programs, and ``notebook_asm_golden`` pins
what the calibration notebooks compile. This module pins the rest: every
Program whose base changes in the refactor, compiled directly from a config
(no Experiment, no acquisition), on the pinned set ``preload_current``.

Two kinds of cases, both in :data:`CASES`:

* one config per leaf Program, with the values its own methods read;
* the flag matrix: the template bases themselves (their ``core_pulses`` is a
  short wait), under each readout, herald and prepulse option. After the two
  templates merge, both columns compile the same class, and the golden diff
  shows exactly where the two copies differed.

A case marked ``raises`` pins the error it raises now (for example, one
template accepts a config that the other rejects). Any other error fails the
build, so a config mistake cannot be pinned by accident.

Three leaves (``floquet_chevron``, ``floquet_phase_cal``, ``sideband_amp_rabi``)
set their own ``length`` on the Floquet pulse. That works only for native
flat-top pulses; both pinned sets have arb envelopes (``preload_flattop``,
``gauss``), and qick rejects the override. They are pinned as ``raises``.

A key names what is played, not the class that plays it, so a case keeps its
key when its class is renamed or re-parented: the golden then compares the
old class with the new one. The table is data, so step 10D can move the
readout booleans to the ``readout`` key without touching the goldens.

This is a characterization pin: it records what the Programs play now, not
that it is right. Regenerate with ``pixi run python -m tests.program_asm_golden``
and read the diff before committing it -- that diff is the review.
"""
import gzip
import importlib
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

from slab import AttrDict

from experiments.qsim.mbr_campaign import mock_station, pinned_config_set
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS,
    FLOQUET_DEFAULTS,
)
from experiments.qsim.utils import ensure_list_in_cfg
from tests.asm_golden import render

GOLDEN_DIR = Path(__file__).parent / "data" / "program_asm_golden"
CONFIG_SET = "preload_current"


class Case(NamedTuple):
    key: str
    target: str                 # "module:Class"
    overrides: dict             # on BASE
    config_set: str = CONFIG_SET
    raises: bool = False        # pin the error it raises now

SWAP_STORS = [1, 2, 3, 4]

# The template options every case starts from: one photon loaded into S1 and
# read back from S1, no reset, no herald, the default M1 -> qubit readout.
BASE = dict(
    expts=1, reps=10, rounds=1, qubits=[0], normalize=False,
    relax_delay=200,
    active_reset=False, man_reset=False, storage_reset=False,
    prepulse=True, postpulse=True, init_fock=True, init_stor=1, ro_stor=1,
    perform_wigner=False, parity_fast=False, phase_second_pulse=180,
    man_mode_no=1,
    **ACTIVE_RESET_DEFAULTS,
    **FLOQUET_DEFAULTS,
)

# Floquet playback settings shared by the cases that play a Floquet train.
FLOQUET = dict(
    swap_stors=SWAP_STORS, detunings=[0.0] * len(SWAP_STORS),
    update_phases=True, floquet_cycle=3,
)

QSIM = "experiments.qsim"

# The Kerr Ramsey programs have their own body and their own keys.
KERR_RAMSEY = dict(
    start=0.01, step=0.02, ramsey_freq=4.0,
    kerr_gain=0, kerr_detune=0.0, kerr_length=0.5, displace_gain=1000,
    prep_e_first=False, parity_meas=True, man_reset=True, storage_reset=True,
    storage_ramsey=[False, 2, True], man_ramsey=[False, 0], coupler_ramsey=False,
    custom_coupler_pulse=None, echoes=[False, 0], prepulse=False, postpulse=False,
    gate_based=False, pre_sweep_pulse=[], post_sweep_pulse=[],
    user_defined_pulse=[True, 5000.0, 2000, 0.02, 0, 4],
    kerr_drive_type="man-qubit", relax_delay=2500,
)

# The flag matrix: overrides on BASE, played by both template bases.
TEMPLATE_OPTIONS = {
    "default": {},
    "no_prepulse": dict(prepulse=False),
    "no_postpulse": dict(postpulse=False),
    "ro_stor_0": dict(ro_stor=0),
    "init_stor_list": dict(init_stor=[1, 2]),
    "map_to_qubit_ge": dict(map_to_qubit_ge=True),
    "parity_check": dict(parity_check=True, parity_fast=False, phase_second_pulse=180),
    "parity_check_phase_0": dict(parity_check=True, phase_second_pulse=0),
    "active_reset": dict(active_reset=True, man_reset=True, storage_reset=[1, 2],
                         pre_relax_delay=100),
    "parity_readout": dict(parity_readout=True),
    "multiparity_readout": dict(multiparity_readout=True),
    "wigner": dict(perform_wigner=True, wigner_alpha=0.5 + 0.25j),
    "slow_pi_ge_readout": dict(slow_pi_ge_readout=True),
    "init_man_fock_state_1": dict(init_fock=False, init_man_fock_state="1"),
    "init_man_fock_state_plus": dict(init_fock=False, init_man_fock_state="+"),
    "init_man_fock_state_2": dict(init_fock=False, init_man_fock_state="2"),
    "init_alpha": dict(init_fock=False, init_alpha=0.8),
    "crude_comp_gain": dict(init_fock=False, init_man_fock_state="2", do_crude_comp=True),
    "crude_comp_length": dict(init_fock=False, init_man_fock_state="2", do_crude_comp=True,
                              fg_area_comp="length"),
}

# Options that one template rejects now. QsimBaseProgram uses MM_base's
# prep_man_fock_state (0, 1 and superpositions only), and sets up the
# displacement only for perform_wigner.
TEMPLATE_RAISES = {
    ("qsim_template", "init_man_fock_state_2"),
    ("qsim_template", "init_alpha"),
    ("qsim_template", "crude_comp_gain"),
    ("qsim_template", "crude_comp_length"),
}

TEMPLATES = {
    "qsim_template": f"{QSIM}.qsim_base:QsimBaseProgram",
    "dark_template": f"{QSIM}.dark_base:DarkBaseProgram",
}


def _template_cases():
    for template, target in TEMPLATES.items():
        for option, overrides in TEMPLATE_OPTIONS.items():
            yield Case(f"{template}__{option}", target, overrides,
                       raises=(template, option) in TEMPLATE_RAISES)


# One config per leaf.
LEAF_CASES = [Case(*row) for row in [
    # Floquet chevrons and calibrations on the QsimBase template.
    ("floquet_chevron", f"{QSIM}.floquet_chevron:FloquetChevronProgram",
     dict(init_stor=2, ro_stor=2, detune=0.1, length=0.5), CONFIG_SET, True),
    ("floquet_gain_chevron", f"{QSIM}.floquet_gain_chevron:FloquetGainChevronProgram",
     dict(init_stor=2, ro_stor=2, detune=0.1, gain=2000)),
    ("floquet_phase_cal", f"{QSIM}.floquet_offdiag_phasecal:FloquetPhaseCalProgram",
     dict(stor_row=1, stor_col=2, stor_idle=3, length=0.5, advance_phase=30.0), CONFIG_SET, True),
    ("floquet_calibration", f"{QSIM}.sideband_scramble:FloquetCalibrationProgram",
     dict(storA=1, storB=2, storA_advance_phase=10.0, storB_advance_phase=20.0,
          floquet_cycle=3)),
    ("storage_t1", f"{QSIM}.sideband_scramble:StorageT1Program", dict(wait=2.0)),
    ("sideband_scramble", f"{QSIM}.sideband_scramble:SidebandScrambleProgram", FLOQUET),
    ("readout_freq_sweep", f"{QSIM}.diagnose_blob:ReadoutFreqSweepProgram",
     dict(FLOQUET, readout_freq=750.0)),
    ("sideband_scramble_dark_t2", f"{QSIM}.t2_cavity_fluxexcursion:SidebandScrambleDarkProgram",
     dict(FLOQUET, swap_man_dark=False, dark_swap_order=[4, 5], second_rel_phase=180)),
    ("sideband_stark", f"{QSIM}.sideband_stark:SidebandStarkProgram",
     dict(init_stor=2, detune=0.1, wait=1.0, advance_phase=15.0)),
    ("sideband_stark_amplification", f"{QSIM}.sideband_stark:SidebandStarkAmplificationProgram",
     dict(stor_A=1, stor_B=2, n_pulse=3, advance_phase=5.0)),
    ("stark_amplification_old", f"{QSIM}.sideband_stark_shift_cal:SidebandStarkAmplificationModifiedProgram_old",
     dict(stor_A=1, stor_B=2, n_pulse=3, advance_phase=5.0)),
    ("sideband_amp_rabi", f"{QSIM}.sideband_amp_rabi:SidebandAmpRabiProgram",
     dict(init_stor=2, detune=0.1, gain=2000, length=0.5), CONFIG_SET, True),
    # Reads cfg.length_to_sweep: MM_base_initialize copies cfg.expt to the top level.
    ("slow_pi_ge_length_rabi", f"{QSIM}.slow_pi_ge_calibration:SlowPiGeLengthRabiProgram",
     dict(length_to_sweep=1.0)),
    ("slow_pi_ge_ramsey", f"{QSIM}.slow_pi_ge_calibration:SlowPiGeRamseyProgram",
     dict(hpi_length=0.2, ramsey_freq=0.5, wait_time=1.0)),
    ("parity_readout_debugging", f"{QSIM}.parity_readout_debugging:ParityReadoutDebuggingProgram", {}),
    ("active_reset_verification", f"{QSIM}.t2_cavity_fluxexcursion:ActiveResetVerificationProgram", {}),
    ("f0g1_spectroscopy_t2", f"{QSIM}.t2_cavity_fluxexcursion:Qsimf0g1Sepctroscopy",
     dict(freq=2000.0, gain=1000, length=1.0, flux_freq=500.0, flux_drive_gain=500)),
    ("flux_drive_f0g1_spectroscopy", f"{QSIM}.flux_amplitude_calibration:FluxDriveF0g1SpectroscopyProgram",
     dict(freq=2000.0, gain=1000, length=1.0, flux_freq=500.0, flux_drive_gain=500)),
    ("flux_excursion_transition_debugging",
     f"{QSIM}.flux_excursion_transition_debugging:FluxExcursionTransitionDebuggingProgram",
     dict(kerr_freq=300.0, kerr_gain=1000, kerr_length=1.0)),
    ("man_f0g1_flux_excursion_ramsey", f"{QSIM}.man_f0g1_ramsey:ManF0g1FluxExcursionRamseyProgram",
     dict(kerr_freq=300.0, kerr_gain=1000, kerr_length=1.0, virtual_ramsey_freq=0.5)),
    ("cooling_spectroscopy", f"{QSIM}.cooling:CoolingSpectroscopyProgram",
     dict(cooling_freq=800.0, cooling_gain=1000, cooling_length=1.0,
          charge_freq=4000.0, charge_gain=500)),
    ("cooling_probe", f"{QSIM}.cooling:CoolingProbeProgram",
     dict(f0g1_mod_freq=2000.0, f0g1_mod_gain=1000, f0g1_mod_length=1.0,
          probe_freq=800.0, probe_gain=500, probe_length=1.0)),
    ("cooling_f0g1_cal", f"{QSIM}.cooling:CoolingF0g1CalProgram",
     dict(f0g1_mod_freq=2000.0, f0g1_mod_gain=1000, f0g1_mod_length=1.0)),
    ("kerr_wait", f"{QSIM}.kerr:KerrWaitProgram", dict(wait_us_time=1.0)),
    ("kerr_wait_wigner", f"{QSIM}.kerr:KerrWaitProgram",
     dict(wait_us_time=1.0, perform_wigner=True, wigner_alpha=0.5 + 0.25j)),
    ("kerr_eng_base", f"{QSIM}.kerr:KerrEngBaseProgram",
     dict(kerr_gain=1000, kerr_detune=0.1, kerr_length=1.0,
          qubit_drive_pulse=[False, 0.0, 0, 0.0, 0.0])),
    ("kerr_stark", f"{QSIM}.kerr:KerrStarkProgram",
     dict(kerr_gain=1000, kerr_detune=0.1, kerr_length=1.0,
          qubit_drive_pulse=[False, 0.0, 0, 0.0, 0.0])),
    # Own body; the config as guan/qsim_wigner.py's preprocessor resolves it.
    ("kerr_cavity_ramsey", f"{QSIM}.kerr:KerrCavityRamseyProgram", KERR_RAMSEY),
    ("cavity_flux_excursion_ramsey",
     f"{QSIM}.cavity_ramsey_flux_excursion:CavityFluxExcursionRamseyProgram",
     dict(KERR_RAMSEY, kerr_freq=300.0)),
    ("broadband_ge_validation", f"{QSIM}.dark_mode_broadband_ge_validation:BroadbandGeValidationProgram",
     dict(validation_case=2, validation_photon_number=1)),
    # On the DarkBase template.
    ("dark_t1_wait", f"{QSIM}.dark_mode_t1:DarkT1Program",
     dict(swap_stors=SWAP_STORS, wait_length=1.0)),
    ("dark_scramble", f"{QSIM}.mbr_spectroscopy_program:SidebandScrambleDarkProgramNewNew",
     dict(FLOQUET, ro_stor=2, swap_man_dark=False, dark_swap_order=[4, 5],
          second_rel_phase=180, map_to_qubit_ge=True, init_stor=0)),
    ("storage_swap_stark_phase", f"{QSIM}.sideband_stark_shift_cal:SidebandStarkAmplificationModifiedProgram",
     dict(stor_A=1, stor_B=2, n_pulse=3, advance_phase=5.0)),
    ("stark_amplification_newold", f"{QSIM}.sideband_stark_shift_cal:SidebandStarkAmplificationModifiedProgram_newold",
     dict(stor_A=1, stor_B=2, n_pulse=3, advance_phase=5.0)),
    ("storage_swap_phase_accumulation", f"{QSIM}.storage_swap_phase_cal:StorageSwapPhaseAccumulationProgram",
     dict(stor_A=1, stor_B=2, n_pulse=3, advance_phase=5.0)),
    ("floquet_displacement_kerr", f"{QSIM}.floquet_displacement_kerr:FloquetDisplacementKerrProgram",
     dict(swap_stors=SWAP_STORS, displace_gain=1000, n_cycle_pair=2, ramsey_freq=0.5,
          zero_floquet_gain=False)),
    # RAverager: the template body on a hardware sweep (DarkBaseRProgram has
    # no core_pulses of its own, so its leaf stands for it).
    ("multiparity_chevron_r", f"{QSIM}.dark_mode_multiparity_chevron:ManStorMultiparityChevronRProgram",
     dict(start=870.0, step=0.1, expts=3, swap_stor=2, storage_pulse_name="M1-S2",
          custom_scramble_length=0.5, multiparity_readout=True)),
]]

CASES = list(_template_cases()) + LEAF_CASES


def program_class(target):
    module, name = target.split(":")
    return getattr(importlib.import_module(module), name)


def build(station, target, overrides):
    """Compile one Program the way ``CharacterizationRunner.run_local`` sets it up."""
    cfg = AttrDict(deepcopy(station.hardware_cfg))
    cfg.device.storage._ds_storage = station.ds_storage
    cfg.device.storage._ds_floquet = station.ds_floquet
    expt = deepcopy(BASE)
    expt.update(deepcopy(overrides))
    cfg.expt = AttrDict(expt)
    cfg.device.readout.relax_delay = [cfg.expt.relax_delay]
    ensure_list_in_cfg(cfg)
    return program_class(target)(soccfg=station.soccfg, cfg=cfg)


def rendered():
    """Yields ``(key, text)`` for every case: the compiled program, or the
    pinned error of a ``raises`` case."""
    stations = {}
    for case in CASES:
        if case.config_set not in stations:
            stations[case.config_set] = mock_station(**pinned_config_set(case.config_set))
        station = stations[case.config_set]
        # No class name in the text: a renamed class must compare equal.
        header = f"# config set {case.config_set}\n"
        if not case.raises:
            yield case.key, header + render(build(station, case.target, case.overrides))
            continue
        try:
            build(station, case.target, case.overrides)
        except Exception as error:      # the pinned behavior
            yield case.key, header + f"# raises {type(error).__name__}: {error}\n"
        else:
            yield case.key, header + "# expected to raise, but compiled\n"


def path_for(key):
    return GOLDEN_DIR / f"{key}.txt.gz"


def read(key):
    with gzip.open(path_for(key), "rt", encoding="utf-8") as handle:
        return handle.read()


def write(key, text):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    # mtime=0: a regenerated-but-identical file must not show up as a diff.
    with gzip.GzipFile(path_for(key), "wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))


def regenerate():
    written = []
    for key, text in rendered():
        write(key, text)
        written.append(key)
    return written


if __name__ == "__main__":
    for key in regenerate():
        size = path_for(key).stat().st_size
        print(f"wrote {key} ({size / 1024:.1f} KiB gz)")
