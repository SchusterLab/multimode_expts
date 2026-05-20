import json
import os

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.MM_base import MMAveragerProgram
from experiments.MM_dual_rail_base import MM_dual_rail_base

"""
cooling
"""

class CoolingSpectroscopyProgram(QsimBaseProgram):
    # DRIVE_CHANNEL = 3 # man drive
    # DRIVE_CHANNEL = 6 # stor drive
    FLUX_CHANNEL = 1 # flux low drive
    CHARGE_CHANNEL = 3 # man drive

    def initialize(self):
        super().initialize()
        self.declare_gen(ch=self.FLUX_CHANNEL, nqz=2) # 2.5GHz < fs/2
        self.declare_gen(ch=self.CHARGE_CHANNEL, nqz=2)
        # self.declare_gen(ch=self.DRIVE_CHANNEL, nqz=2) # bad hard coded :(
        # self.declare_gen(ch=self.FLUX_CHANNEL, nqz=2, mixer_freq=7000, mux_freqs=[0])
        self.add_gauss(ch=self.CHARGE_CHANNEL, name="cooling_charge",
                       sigma=self.pi_m1_sigma_low, length=self.pi_m1_sigma_low*6)

    def core_pulses(self):
        qTest = 0
        ecfg = self.cfg.expt
        # spec_pulse = [
        #     [ecfg.cooling_freq],
        #     [ecfg.cooling_gain],
        #     [ecfg.cooling_length],
        #     [0],
        #     [self.DRIVE_CHANNEL],
        #     ['flat_top'],
        #     [self.cfg.device.storage.ramp_sigma],
        # ]
        # self.custom_pulse(self.cfg, spec_pulse, prefix='cool_')
        # [[frequency], [gain], [length (us)], [phases],
        # [drive channel], [shape], [ramp sigma]]
        #

        self.set_pulse_registers(
            ch=self.FLUX_CHANNEL, style="flat_top",
            freq=self.freq2reg(ecfg.cooling_freq, gen_ch=self.FLUX_CHANNEL),
            phase=0, gain=ecfg.cooling_gain,
            length=self.us2cycles(ecfg.cooling_length, gen_ch=self.FLUX_CHANNEL),
            waveform="pi_m1si_low",
        )

        self.set_pulse_registers(
            ch=self.CHARGE_CHANNEL, style="flat_top",
            freq=self.freq2reg(ecfg.charge_freq, gen_ch=self.CHARGE_CHANNEL),
            phase=0, gain=ecfg.charge_gain,
            length=self.us2cycles(ecfg.cooling_length, gen_ch=self.CHARGE_CHANNEL),
            waveform="cooling_charge",
        )

        self.pulse(self.CHARGE_CHANNEL)
        self.pulse(self.FLUX_CHANNEL)
        self.sync_all(self.us2cycles(0.01))


