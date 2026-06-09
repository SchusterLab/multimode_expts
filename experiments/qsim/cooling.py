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
    FLUX_CHANNEL_LOW = 1 # flux low drive
    FLUX_CHANNEL_HIGH = 4 # flux high drive
    CHARGE_CHANNEL = 3 # man drive

    def initialize(self):
        ecfg = self.cfg.expt
        self.FLUX_CHANNEL = self.FLUX_CHANNEL_LOW if ecfg.cooling_freq < 4000 else self.FLUX_CHANNEL_HIGH
        self.FLUX_NQZ = 1 if ecfg.cooling_freq < 4000 else 2

        super().initialize()
        self.declare_gen(ch=self.FLUX_CHANNEL, nqz=self.FLUX_NQZ)
        self.declare_gen(ch=self.CHARGE_CHANNEL, nqz=2)
        # self.declare_gen(ch=self.DRIVE_CHANNEL, nqz=2) # bad hard coded :(
        # self.declare_gen(ch=self.FLUX_CHANNEL, nqz=2, mixer_freq=7000, mux_freqs=[0])

        ramp_sigma = self.cfg.expt.get('ramp_sigma', 0.005)
        flux_ramp_sigma = self.us2cycles(ramp_sigma, gen_ch=self.FLUX_CHANNEL)
        charge_ramp_sigma = self.us2cycles(ramp_sigma, gen_ch=self.CHARGE_CHANNEL)
        self.add_gauss(ch=self.CHARGE_CHANNEL, name="cooling_charge",
                       sigma=charge_ramp_sigma, length=charge_ramp_sigma*6)
        self.add_gauss(ch=self.FLUX_CHANNEL, name="cooling_flux",
                       sigma=flux_ramp_sigma, length=flux_ramp_sigma*6)

    def core_pulses(self):
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
            waveform="cooling_flux",
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


