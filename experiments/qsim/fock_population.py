"""Definite Fock preparation and population readout (AGP measurement).

Moved on 2026-10-15 from the end of ``floquet_dark_mode_readout.py``,
where jonginn added it as a "temporary amendment for adiabatic gauge potential
measurement". The rest of that file was a compatibility shim that guan deleted
on 2026-10-06; this code is the only part kept.
"""
from copy import deepcopy
from numbers import Integral

# from experiments.dataset import StorageManSwapDataset  # For the paused builder below.
from experiments.qsim.floquet_train import FloquetProgram

# Definite Fock preparation and population readout.


def validate_occupations(occupations, swap_stors):
    """Check nonnegative Fock occupations in M1 plus storage order."""
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


def set_population_defaults(expt):
    """Fill missing population settings, leaving readout selection to Qsim."""
    defaults = dict(prepulse=False, # Fock ladder is climbed in `corepulse`, so should be false
                    postpulse=True,
                    init_stor=0,
                    ro_stor=0,
                    parity_fast=False,
                    cond_sec_phase=-90) # For multi parity
    for name, value in defaults.items():
        expt.setdefault(name, value)


def validate_population_config(expt):
    """Check preparation and storage routing independently of readout mode."""
    validate_occupations(expt.occupations, expt.swap_stors)
    if expt.prepulse:
        raise ValueError("prepulse must be False: occupations already supplies Fock preparation")
    if expt.ro_stor > 0 and not expt.postpulse:
        raise ValueError("storage readout uses the existing postpulse=True path")
    if expt.get("man_mode_no", 1) != 1:
        raise ValueError("the calibrated preparation path uses M1")


# Retained with the paused gate-list builder below.
# def with_storage_waits(pulses, wait_us):
#     """Copy a gate list, adding a wait after each storage swap."""
#     result = []
#     for pulse in pulses:
#         result.append(list(pulse))
#         if pulse[0] == "storage" and wait_us > 0:
#             result.append(["wait", wait_us])
#     return result


class FockPopulationProgram(FloquetProgram):
    """The ordinary Qsim reset/readout template with a population experiment core.

    ``occupations`` is [n_M1, n_S...] in ``swap_stors`` order. ``ro_stor`` is
    0 for M1. With ``postpulse=True``, the existing Qsim body swaps
    ``ro_stor`` into M1 before the configured readout.
    Preparation uses MM_base.man_stor_swap(), including its fixed swap delay.
    The optional multiphoton gate-list builder is commented out for now;
    use_multiphoton_swap and storage_pulse_wait_us are not used by this path.
    """

    def initialize(self):
        expt = self.cfg.expt
        set_population_defaults(expt)
        validate_population_config(expt)
        super().initialize()

    def _play_gate_list(self, pulses, prefix):
        if not pulses:
            return
        creator = self.get_prepulse_creator(deepcopy(pulses))
        self.sync_all()
        self.custom_pulse(self.cfg, creator.pulse, prefix=prefix)
        self.sync_all() 

    def _prepare_occupation(self):
        expt = self.cfg.expt
        for storage, number in zip(expt.swap_stors, expt.occupations[1:]):
            if number == 0:
                continue
            self._play_gate_list(self.prep_man_fock_state(1, str(number)),
                                 prefix=f"population_prep_S{storage}_")
            self.man_stor_swap(man_idx=1, stor_idx=storage)
        
        self._play_gate_list(self.prep_man_fock_state(1, str(expt.occupations[0])),
                             prefix="population_prep_M1_")
        
    
    def core_pulses(self):
        expt = self.cfg.expt
        self._prepare_occupation()

        storages = list(expt.swap_stors)
        self.phase_offsets = [0.0] * len(storages)
        self.disorder_phase_offsets = [0.0] * len(storages)
        self._play_scramble_with_phase_offsets(
            self.phase_offsets, storages, self.disorder_phase_offsets)
        # QsimProgram.body handles postpulse (ro_stor -> man) and readout.


    # Paused: use MM_base.man_stor_swap() for the current preparation path.
    # def build_fock_preparation(self,
    #                            occupations,
    #                            swap_stors,
    #                            use_multiphoton_swap=False,
    #                            storage_pulse_wait_us=0.2):
    #     """Load initially empty storages once, then load M1 last."""
    #     validate_occupations(occupations, swap_stors)
    #     pulses = []
    #     for storage, number in zip(swap_stors, occupations[1:]):
    #         if number == 0:
    #             continue
    #         pulses.extend(self.prep_man_fock_state(1, str(number)))
    #         row = f"M1-S{storage}"
    #         if use_multiphoton_swap: #TODO: THERE IS NO THIS CONFIG FOR NOW. TO IMPEMENT
    #             row = StorageManSwapDataset.multiphoton_swap_name(row, number)
    #         pulses.append(["storage", row, "pi", 0.0])
    #     pulses.extend(self.prep_man_fock_state(1, str(occupations[0])))
    #     return with_storage_waits(pulses, storage_pulse_wait_us)

