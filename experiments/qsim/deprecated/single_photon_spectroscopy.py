# -*- coding: utf-8 -*-
"""The original N=1 program name, kept as a guard rather than an alias.

``SinglePhotonFloquetSpectroscopyProgram`` predates arbitrary-N spectroscopy.
It is not a forwarding alias: it refuses any photon number other than 1 and
points at the general program, so an old config that happens to carry a
larger N raises instead of quietly measuring something else.

Nothing in the library uses it. It is here, and not deleted, only because
jobs recorded under this class name should still resolve when re-analysed.
"""
import numpy as np

from experiments.qsim.mbr_spectroscopy_program import (
    NPhotonHamiltonianSpectroscopyProgram,
)


class SinglePhotonFloquetSpectroscopyProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Backward-compatible wrapper for the original N=1 program name."""

    def initialize(self):
        ecfg = self.cfg.expt
        photon_number = int(
            ecfg.get("spectroscopy_photon_number", 1))
        if photon_number != 1:
            raise ValueError(
                "SinglePhotonFloquetSpectroscopyProgram only supports N=1; "
                "use NPhotonHamiltonianSpectroscopyProgram for arbitrary N"
            )
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        initial_mode = int(ecfg.get("spectroscopy_initial_mode", 0))
        if initial_mode not in [0] + swap_stors:
            raise ValueError(
                f"spectroscopy_initial_mode={initial_mode} is not in "
                f"[0] + swap_stors={swap_stors}"
            )
        occupations = [0] * (len(swap_stors) + 1)
        occupation_index = (
            0 if initial_mode == 0 else swap_stors.index(initial_mode) + 1
        )
        occupations[occupation_index] = 1
        ecfg.spectroscopy_occupations = occupations
        ecfg.spectroscopy_photon_number = 1
        super().initialize()


