"""One Fock preparation per disorder realization, with a DMEM frequency table.

The existing population sweep still uploads one program per evolution time.
Inside that program, each repetition reads the next row of the same disorder
table. Preparation, heralds, and multiparity readout are inherited unchanged.

Data flow: detunings in MHz -> full DDS frequency words -> tProc DMEM ->
frequency registers -> pulse commands. initialize() and core_pulses() build
instructions in Python; the tProc executes those instructions on each shot.
config_all() transfers the table before starting that program.
"""
from copy import deepcopy

import numpy as np

from experiments.qsim.fock_population import FockPopulationExperiment
from experiments.qsim.fock_prep_herald import FockPrepHeraldProgram, shot_labels
from experiments.qsim.floquet_register_bank import (
    _play_preloaded_floquet_register_bank_entry,
    _prepare_preloaded_floquet_register_bank,
)
from experiments.qsim.multiparity import herald_keep


def validate_ensemble_config(program, detunings, dmem_address):
    """Check ensemble settings against the board and calibration before setup."""
    expt = program.cfg.expt
    if (detunings.ndim != 2 or detunings.shape[1] != len(expt.swap_stors)
            or not len(detunings) or not np.isfinite(detunings).all()):
        raise ValueError("disorder_detunings_mhz must be a finite (realizations, storages) table")
    if (expt.get("rounds", 1) != 1 or expt.get("palindrome_scramble", False)
            or int(expt.floquet_cycle) < 0):
        raise ValueError("Ensemble acquisition uses rounds=1 and nonnegative forward Floquet cycles")
    floquet_dataset = program.cfg.device.storage._ds_floquet
    if not expt.get("floquet_hardware_loop", False) or any(
            floquet_dataset.get_waveform(f"M1-S{storage}") != "preload_flattop"
            for storage in expt.swap_stors):
        raise ValueError("Ensemble acquisition requires the preload_flattop hardware loop")
    if (dmem_address < 2 or dmem_address + detunings.size
            > program.tproccfg["dmem_size"]):
        raise ValueError("Disorder table must fit DMEM without its shot-counter address 1")


class FockPopulationEnsembleProgram(FockPrepHeraldProgram):
    """Read one disorder vector per hardware repetition, in saved row order."""

    def initialize(self):
        expt = self.cfg.expt
        # Rows are realizations; columns follow expt.swap_stors. Units are MHz.
        detunings = np.asarray(expt.disorder_detunings_mhz, dtype=float)
        dmem_address = int(expt.get("disorder_dmem_address", 32))
        validate_ensemble_config(self, detunings, dmem_address)
        expt.reps = len(detunings)
        self.cfg.reps = len(detunings)
        expt.rounds = 1
        super().initialize()
        self.disorder_dmem_address = dmem_address
        pulse_arguments = [deepcopy(self.m1s_kwargs[storage - 1]) for storage in expt.swap_stors]
        first_phases, phase_steps = self._floquet_cycle_phases(
            [0.] * len(pulse_arguments), expt.swap_stors, expt.get("update_phases", True))
        self.ensemble_bank, self.ensemble_page, self.ensemble_cycle_register =  _prepare_preloaded_floquet_register_bank(self, 
                                                                                                                         pulse_arguments, 
                                                                                                                         first_phases,
                                                                                                                         phase_steps, 
                                                                                                                         reserved_registers=2)
        # Choose a register NUMBER: use the free slot after the cycle counter.
        # For four storages, the counter is r25 and the pointer is r26.
        # This Python assignment does not increment any hardware register value.
        self.ensemble_pointer_register = self.ensemble_cycle_register + 1
        # This instruction runs before the outer shot loop. Do not reset the
        # pointer per shot: each shot must consume the next row of the table.
        self.safe_regwi(self.ensemble_page, self.ensemble_pointer_register, self.disorder_dmem_address)
        # Use the existing fixed-disorder pulse builder for every realization.
        # Only its resulting frequency words are stored in the DMEM table.
        self.disorder_frequency_words = np.zeros(detunings.shape, dtype=np.uint32)
        for i_realization, detunings_MHz in enumerate(detunings):
            all_pulse_args = self._detuned_floquet_pulse_args(expt.swap_stors, detunings_MHz)
            frequency_words = np.asarray([pulse_args["freq"] for pulse_args in all_pulse_args],
                                         dtype=np.int64)
            # Keep the low 32 bits, as the existing tProc frequency registers do.
            self.disorder_frequency_words[i_realization] = frequency_words

    def config_all(self, soc, **kwargs):
        """Upload the table after program setup and before starting acquisition."""
        super().config_all(soc, **kwargs)
        # Flatten row by row: all storages of realization 0, then realization 1, ...
        frequency_words = self.disorder_frequency_words.reshape(-1)
        # Python uploads the table once per program/time point, before any shots.
        # the `disorder_dmem_address` and `frequency_words` have identical relation to pointer and array in Cpp.
        soc.load_mem(frequency_words, mem_sel="dmem", addr=self.disorder_dmem_address)

    def core_pulses(self):
        """Keep the table pointer between shots; reset the evolution phases.
        Basically, the ensemble_pointer_register is increased by 1 per memr, 
        so each rep the pulse uses different realization.
        """
        # These memr instructions run on the tProc each shot. Copy one row from
        # DMEM into the reused frequency registers, then leave the pointer at
        # the next row. The values stay fixed throughout this shot's evolution.
        for entry in self.ensemble_bank:
            self.memr(self.ensemble_page, entry["frequency_register"], self.ensemble_pointer_register)
            # Unlike choosing a register number above, this instruction changes
            # its stored DMEM address at runtime: R[pointer] = R[pointer] + 1.
            self.mathi(self.ensemble_page, self.ensemble_pointer_register,
                       self.ensemble_pointer_register, "+", 1)
        self._prepare_occupation()
        for entry in self.ensemble_bank:
            self.safe_regwi(self.ensemble_page, entry["phase_register"],
                            entry["first_phase_register_value"])
        self.sync_all()
        if self.cfg.expt.floquet_cycle:
            self._play_ensemble_loop()

    def _play_ensemble_loop(self):
        """Reuse the existing pulse emitter and phase increments without a bank rewrite."""
        self.safe_regwi(self.ensemble_page, self.ensemble_cycle_register,
                        int(self.cfg.expt.floquet_cycle) - 1)
        self.label("ENSEMBLE_FLOQUET")
        for entry in self.ensemble_bank:
            # The helper emits set with this bank's frequency, phase and waveform
            # address registers. The generator reuses the same envelope samples;
            # the frequency register supplies the selected realization's DDS word.
            _play_preloaded_floquet_register_bank_entry(self, entry)
            self.math(self.ensemble_page, entry["phase_register"],
                      entry["phase_register"], "+", entry["phase_step_register"])
            self.sync_all(int(self.cfg.expt.get("scramble_sync_cycles", 10)))
        self.loopnz(self.ensemble_page, self.ensemble_cycle_register, "ENSEMBLE_FLOQUET")


class FockPopulationEnsembleExperiment(FockPopulationExperiment):
    """A cycle sweep whose unfiltered shot index is the disorder realization index."""

    default_program = FockPopulationEnsembleProgram

    def acquire(self, progress=False, debug=False):
        self.cfg.expt.reps = len(self.cfg.expt.disorder_detunings_mhz)
        return super().acquire(progress=progress, debug=debug)

    def shape_data(self, axes, points):
        data = super().shape_data(axes, points)
        detunings = np.asarray(self.cfg.expt.disorder_detunings_mhz, dtype=float)
        realization_indices = np.arange(len(detunings), dtype=int)
        data.update(occupations=np.asarray(self.cfg.expt.occupations, dtype=int),
                    swap_stors=np.asarray(self.cfg.expt.swap_stors, dtype=int),
                    cycles=np.asarray(data["xpts"], dtype=int),
                    time_us=np.asarray(data["xpts"]) * self.prog.calculate_floquet_cycle_us(),
                    disorder_detunings_mhz=detunings,
                    disorder_frequency_words=self.prog.disorder_frequency_words.copy(),
                    realization_indices=realization_indices,
                    shot_realization_indices=np.broadcast_to(
                        realization_indices, (len(data["xpts"]), len(detunings))).copy(),
                    disorder_dmem_address=np.asarray(self.prog.disorder_dmem_address))
        if self.cfg.expt.get("disorder_seed") is not None:
            data["disorder_seed"] = np.asarray(self.cfg.expt.disorder_seed)
        self._store_labels(data)
        return data

    def _store_labels(self, data):
        data["readout_labels"] = np.asarray(shot_labels(self.cfg, data)["n"], dtype=np.uint8)
        accepted = herald_keep(self.cfg, data["idata"])
        data["herald_keep"] = (np.ones(data["readout_labels"].shape, dtype=bool)
                               if accepted is None else accepted)

    def analyze(self, data=None, **kwargs):
        if data is not None:
            self.data = data
        if "idata" in self.data:
            super().analyze(**kwargs)
            self._store_labels(self.data)
        return self.data
