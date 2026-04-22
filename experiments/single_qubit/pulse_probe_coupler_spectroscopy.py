import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram

"""
Pulse-probe coupler spectroscopy.

Sequence
--------
  1. Prepare manipulate mode (default M1) in Fock |1>.
  2. Play a frequency-swept const probe pulse on the manipulate DAC channel.
  3. Swap the manipulate |1> back to qubit |e> (f0-g1 pi + e0-f0 pi).
  4. Dispersive readout.

The probe tone is currently emitted through the manipulate DAC as a
hardware convenience - it is the path already wired up and calibrated.
The physics target is the coupler (or any feature that couples to the
populated manipulate mode), not the manipulate transition itself. To
retarget this to a dedicated coupler drive channel in the future, only
the `ch=self.man_ch[qTest]` line on the probe pulse needs to change.
"""


class PulseProbeCouplerSpectroscopyProgram(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = cfg.expt.qubits[0]
        man_no = int(cfg.expt.get('man_mode_no', 1))

        # Pre-build the Fock-|1> prep pulse table and the swap-out pulse table.
        # Both are static across the frequency sweep, so we build them once.
        prep_seq = self.prep_man_fock_state(man_no, '1')
        self.prep_pulse = self.get_prepulse_creator(prep_seq).pulse.tolist()

        # Swap manipulate |1> -> qubit |e>: reverse of the last two prep steps.
        swap_seq = [
            ['multiphoton', f'f0-g{man_no}', 'pi', 0],
            ['multiphoton', 'e0-f0', 'pi', 0],
        ]
        self.swap_pulse = self.get_prepulse_creator(swap_seq).pulse.tolist()

        # Optional coupler g->e pi pulse so the probe hits the e-f transition.
        # Mirrors the `qubit_f` flag in pulse_probe_ef_spectroscopy. Params come
        # from the calibrated entry `device.coupler.pulses.pi_ge` in the hardware
        # config.
        self.coupler_f = cfg.expt.get('coupler_f', False)
        if self.coupler_f:
            try:
                pi_cfg = cfg.device.coupler.pulses.pi_ge
                print("Found coupler pi pulse config in hardware config.")
                print(pi_cfg)
            except (AttributeError, KeyError):
                raise KeyError(
                    "coupler_f=True requires device.coupler.pulses.pi_ge in the "
                    "hardware config. Snapshot it with "
                    "station.snapshot_hardware_config(update_main=True)."
                )
            self.coupler_pi_freq = self.freq2reg(pi_cfg.freq[0], gen_ch=self.man_ch[qTest])
            self.coupler_pi_length = self.us2cycles(pi_cfg.length[0], gen_ch=self.man_ch[qTest])
            self.coupler_pi_gain = int(pi_cfg.gain[0])

        # Register setup for the frequency sweep on the manipulate channel.
        self.q_rp = self.ch_page(self.man_ch[qTest])
        self.r_freq = self.sreg(self.man_ch[qTest], "freq")
        self.r_freq2 = 4
        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.man_ch[qTest])
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.man_ch[qTest])

        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start)
        self.synci(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # Optional active reset.
        if cfg.expt.get('active_reset', False):
            self.active_reset()

        self.sync_all()

        # Optional user prepulse (e.g. qubit/storage initialization upstream).
        if cfg.expt.get('prepulse', False):
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

        # Step 1: prepare manipulate |1>.
        self.custom_pulse(cfg, self.prep_pulse, prefix='prep_man_')
        self.sync_all()

        # Optional: drive the coupler g->e before the probe so the sweep
        # targets the e-f transition (analog of qubit_f in the qubit EF probe).
        if self.coupler_f:
            self.setup_and_pulse(
                ch=self.man_ch[qTest],
                style="const",
                freq=self.coupler_pi_freq,
                phase=0,
                gain=self.coupler_pi_gain,
                length=self.coupler_pi_length,
            )
            self.sync_all()

        # Step 2: probe pulse on manipulate channel, frequency swept via register.
        self.set_pulse_registers(
            ch=self.man_ch[qTest],
            style="const",
            freq=0,
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length, gen_ch=self.man_ch[qTest]),
        )
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.man_ch[qTest])
        self.sync_all()

        # Second coupler g-e pi (symmetric to the pre-probe one). Maps
        # coupler |e> -> |g> (off-resonance case) and leaves |f> untouched,
        # so the probe resonance shows up as |g> vs |f> at readout rather
        # than |e> vs |f> (which would be ambiguous for the ancilla).
        if self.coupler_f:
            self.setup_and_pulse(
                ch=self.man_ch[qTest],
                style="const",
                freq=self.coupler_pi_freq,
                phase=180,
                gain=self.coupler_pi_gain,
                length=self.coupler_pi_length,
            )
            self.sync_all()

        # Step 3: swap manipulate |1> -> qubit |e>.
        self.custom_pulse(cfg, self.swap_pulse, prefix='swap_')

        # Step 4: readout.
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest]),
        )

    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step)


class PulseProbeCouplerSpectroscopyExperiment(Experiment):
    """
    Pulse-probe coupler spectroscopy.

    Probes coupler features by driving the manipulate channel at various
    frequencies while the manipulate mode is populated with a single photon,
    then reading out via the qubit after swapping |1> -> |e>.

    Experimental Config:
    expt = dict(
        start:        start probe frequency [MHz]
        step:         probe frequency step [MHz]
        expts:        number of frequency points
        reps:         averages per frequency
        rounds:       sweep repetitions
        length:       probe const pulse length [us]
        gain:         probe const pulse gain [DAC units]
        qubits:       list, e.g. [0]
        man_mode_no:  (optional, default 1) manipulate mode index used
                      for the Fock-|1> prep and swap-out
        prepulse:     (optional) bool, play cfg.expt.pre_sweep_pulse first
        pre_sweep_pulse: (optional) gate-list spec for the custom prepulse
        coupler_f:    (optional, default False) drive the coupler g->e with
                      the calibrated cfg.device.coupler.pulses.pi_ge pulse before
                      the probe sweep, so the probe hits the coupler e-f
                      transition. Mirrors `qubit_f` in pulse_probe_ef_spectroscopy.
        active_reset: (optional) bool, run active reset at start of body
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeCouplerSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)

        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not isinstance(value3, list):
                                value2.update({key3: [value3] * num_qubits_sample})
                elif not isinstance(value, list):
                    subcfg.update({key: [value] * num_qubits_sample})

        read_num = 1
        if self.cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        qspec = PulseProbeCouplerSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = qspec.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            debug=debug,
            readouts_per_experiment=read_num,
        )

        avgi = avgi[0][-1]
        avgq = avgq[0][-1]
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq, 'amps': amps, 'phases': phases}
        self.data = data
        return data

    def analyze(self, data=None, fit=True, signs=[1, 1, 1], **kwargs):
        if data is None:
            data = self.data

        from fitting.fit_display_classes import Spectroscopy
        spec_analysis = Spectroscopy(data, signs=signs, config=self.cfg, station=None)
        spec_analysis.analyze(fit=fit)

        self._spec_analysis = spec_analysis
        return data

    def display(self, data=None, fit=True, signs=[1, 1, 1],
                title='Coupler Spectroscopy (via manipulate channel)', **kwargs):
        if data is None:
            data = self.data

        if hasattr(self, '_spec_analysis'):
            vlines = kwargs.get('vlines', None)
            self._spec_analysis.display(title=title, vlines=vlines, fit=fit)
        else:
            from fitting.fit_display_classes import Spectroscopy
            spec_analysis = Spectroscopy(data, signs=signs, config=self.cfg, station=None)
            vlines = kwargs.get('vlines', None)
            spec_analysis.display(title=title, vlines=vlines, fit=fit)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
