"""Definite Fock preparation and population readout (AGP measurement).

Moved unchanged on 2026-10-15 from the end of ``floquet_dark_mode_readout.py``,
where jonginn added it as a "temporary amendment for adiabatic gauge potential
measurement". The rest of that file was a compatibility shim that guan deleted
on 2026-10-06; this code is the only part kept.
"""

from numbers import Integral

from experiments.dataset import StorageManSwapDataset
from experiments.qsim.floquet_train import FloquetProgram

# Definite Fock preparation and population readout.


def validate_occupations(occupations, swap_stors):
    """Check the supported total-photon <= 3, M1 plus storage input."""
    if len(occupations) != len(swap_stors) + 1:
        raise ValueError("occupations must be [n_M1] followed by swap_stors")
    if len(set(swap_stors)) != len(swap_stors):
        raise ValueError("swap_stors must contain distinct storage modes")
    if any(isinstance(stor, bool) or not isinstance(stor, Integral)
           or stor not in range(1, 8) for stor in swap_stors):
        raise ValueError("storage modes must be in 1..7")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
           for n in occupations):
        raise ValueError("occupations must contain non-negative integers")
    if sum(occupations) > 3:
        raise ValueError("mod-4 outcomes identify photon number only for total N <= 3")


def with_storage_waits(pulses, wait_us):
    """Copy a gate list, adding the existing settling wait after each swap."""
    result = []
    for pulse in pulses:
        result.append(list(pulse))
        if pulse[0] == "storage" and wait_us > 0:
            result.append(["wait", wait_us])
    return result


class FockPopulationProgram(FloquetProgram):
    """The ordinary Qsim reset/readout template with a population experiment core.

    ``occupations`` is [n_M1, n_S...] in ``swap_stors`` order. ``ro_stor`` is
    0 for M1. With ``postpulse=True``, the existing Qsim body swaps
    ``ro_stor`` into M1 before its ordinary multiparity readout.
    ``use_multiphoton_swap`` selects known-N storage PREPARATION rows only.
    """

    @classmethod
    def readouts_per_shot(cls, cfg):
        # The Experiment asks for the buffer size before initialize() runs.
        if cfg.expt.get("readout") != "multiparity":
            raise ValueError("set readout='multiparity' before acquisition")
        return super().readouts_per_shot(cfg)

    def initialize(self):
        self._set_population_defaults()
        self._validate_population_config()
        expt = self.cfg.expt
        self.preparation_pulses = self.build_fock_preparation(
            expt.occupations, expt.swap_stors,
            expt.get("use_multiphoton_swap", False),
            expt.storage_pulse_wait_us)
        super().initialize()

    def _set_population_defaults(self):
        defaults = dict(prepulse=False, postpulse=True, init_stor=0, ro_stor=0,
                        readout="multiparity", parity_fast=False,
                        cond_sec_phase=-90, storage_pulse_wait_us=0.2)
        for name, value in defaults.items():
            self.cfg.expt.setdefault(name, value)

    def _validate_population_config(self):
        expt = self.cfg.expt
        validate_occupations(expt.occupations, expt.swap_stors)
        if expt.prepulse:
            raise ValueError("prepulse must be False: occupations already supplies Fock preparation")
        if expt.ro_stor > 0 and not expt.postpulse:
            raise ValueError("storage readout uses the existing postpulse=True path")
        if expt.readout != "multiparity":
            raise ValueError("FockPopulationProgram requires readout='multiparity'")
        if expt.get("man_mode_no", 1) != 1:
            raise ValueError("the calibrated preparation path uses M1")
        if expt.storage_pulse_wait_us < 0:
            raise ValueError("storage_pulse_wait_us must be non-negative")

    def build_fock_preparation(self, occupations, swap_stors,
                               use_multiphoton_swap=False,
                               storage_pulse_wait_us=0.2):
        """Load initially empty storages once, then load M1 last."""
        validate_occupations(occupations, swap_stors)
        pulses = []
        for storage, number in zip(swap_stors, occupations[1:]):
            if number == 0:
                continue
            pulses.extend(self.prep_man_fock_state(1, str(number)))
            row = f"M1-S{storage}"
            if use_multiphoton_swap:
                row = StorageManSwapDataset.multiphoton_swap_name(row, number)
            pulses.append(["storage", row, "pi", 0.0])
        pulses.extend(self.prep_man_fock_state(1, str(occupations[0])))
        return with_storage_waits(pulses, storage_pulse_wait_us)

    def _play_gate_list(self, pulses, prefix):
        if not pulses:
            return
        creator = self.get_prepulse_creator(deepcopy(pulses))
        self.sync_all()
        self.custom_pulse(self.cfg, creator.pulse, prefix=prefix)
        self.sync_all()

    def core_pulses(self):
        self._play_gate_list(self.preparation_pulses, prefix="population_prep_")
        storages = list(self.cfg.expt.swap_stors)
        self.phase_offsets = [0.0] * len(storages)
        self.disorder_phase_offsets = [0.0] * len(storages)
        self._play_scramble_with_phase_offsets(
            self.phase_offsets, storages, self.disorder_phase_offsets)
        # QsimProgram.body handles postpulse (ro_stor -> man) and multiparity.

