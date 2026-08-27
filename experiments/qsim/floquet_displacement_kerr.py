# -*- coding: utf-8 -*-
"""Floquet displacement Kerr: D(alpha), closed Floquet pairs, D(alpha).

Split out of ``floquet_dark_mode_readout.py`` unchanged. The Program and the
Experiment sat 3,300 lines apart there; the whole acquire/analyze/display
triple is local here.
"""
from copy import deepcopy

import numpy as np
from slab import AttrDict

from fitting.fit_display_classes import CavityRamseyGainSweepFitting
from experiments.qsim.floquet_dark_mode_readout import (
    DarkBaseExperiment,
    DarkBaseProgram,
)

class FloquetDisplacementKerrProgram(DarkBaseProgram):
    """
    D(alpha) -> closed Floquet pairs -> D(alpha) -> vacuum readout using `slow_pi_ge`.
    Now displacement pulse entirely relies on displace_man, which receives complex
    alpha and converts it into magnitude and phase.
    Here, the prepulse and postpulse is turned off, as relying on prepulse and postpulse
    seemed unstraightforward.
    
    """

    def initialize(self):
        ecfg = self.cfg.expt
        ecfg.prepulse = False #Enforcing prepulse to be off
        ecfg.postpulse = False #Enforcing postpulse to be off
        ecfg.init_stor = 0
        ecfg.ro_stor = 0
        ecfg.slow_pi_ge_readout = True
        gain_to_alpha = self.cfg.device.manipulate.gain_to_alpha
        if isinstance(gain_to_alpha, (list, tuple, np.ndarray)):
            gain_to_alpha = gain_to_alpha[ecfg.man_mode_no - 1]
        ecfg.init_alpha = ecfg.displace_gain * gain_to_alpha
        super().initialize()
        self.floquet_cycle_us = self.calculate_floquet_cycle_us()
        ecfg.floquet_cycle_us = self.floquet_cycle_us

    def core_pulses(self):
        ecfg = self.cfg.expt
        swap_stors = list(ecfg.swap_stors)
        phase_offsets = [0.] * len(swap_stors)
        self.displace_man(alpha=ecfg.init_alpha, setup=False, play=True)
        self._play_closed_floquet_cycle_pairs(ecfg.n_cycle_pair, phase_offsets, swap_stors)
        time_us = 2 * ecfg.n_cycle_pair * self.floquet_cycle_us
        # Same gain/sign as the first displacement; only the known Ramsey frame advances.
        second_alpha = ecfg.init_alpha * np.exp(-2j * np.pi * ecfg.ramsey_freq * time_us)
        self.displace_man(alpha=second_alpha, setup=False, play=True)
        self.sync_all()


class FloquetDisplacementKerrExperiment(DarkBaseExperiment):
    """Fit Floquet Kerr with the existing cavity-Ramsey gain-sweep analysis."""

    def acquire(self, progress=False, debug=False):
        data = super().acquire(progress=progress, debug=debug)
        data["floquet_cycle_us"] = self.prog.floquet_cycle_us
        self.cfg.expt.floquet_cycle_us = self.prog.floquet_cycle_us
        return data

    def _ramsey_fitter(self, data):
        cfg = deepcopy(self.cfg)
        gain_to_alpha = cfg.device.manipulate.gain_to_alpha
        if isinstance(gain_to_alpha, (list, tuple, np.ndarray)):
            gain_to_alpha = gain_to_alpha[cfg.expt.man_mode_no - 1]
        cfg.device.manipulate.gain_to_alpha = [gain_to_alpha]
        return CavityRamseyGainSweepFitting(data, config=cfg)

    def analyze(self, data=None, fit=True, **kwargs):
        if data is not None:
            self.data = data
        if not fit:
            return self.data
        cycle_pairs = np.asarray(self.data.get("cycle_pairs", self.data["xpts"]))
        displace_gains = np.asarray(self.data.get("displace_gains", self.data["ypts"]))
        time_us = 2 * cycle_pairs * self.data["floquet_cycle_us"]
        ramsey_data = AttrDict(dict(
            gain_list=displace_gains,
            xpts=np.tile(time_us, (len(displace_gains), 1)),
            g_avgi=self.data["avgi"], g_avgq=self.data["avgq"],
            g_amps=self.data["amps"], g_phases=self.data["phases"],
            e_avgi=self.data["avgi"], e_avgq=self.data["avgq"],
            e_amps=self.data["amps"], e_phases=self.data["phases"],
        ))
        self._ramsey_fitter(ramsey_data).analyze(fit=True, **kwargs)
        self.data.update(ramsey_data)
        self.data["cycle_pairs"] = cycle_pairs
        self.data["displace_gains"] = displace_gains
        return self.data

    def display(self, data=None, **kwargs):
        if data is not None:
            self.data = data
        if "Kerr" not in self.data:
            self.analyze()
        return self._ramsey_fitter(self.data).display(**kwargs)

    def save_data(self, data=None):
        if data is None:
            data = self.data
        peaks = {key: data.pop(key) for key in ["time_peak_g", "time_peak_e"] if key in data}
        fname = super().save_data(data)
        data.update(peaks)
        return fname
