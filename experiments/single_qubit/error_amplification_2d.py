"""2D error amplification: frequency (x) x gain or flat-top length (y) of one pulse, for N repetitions.

The frequency is swept inside the program (as in ErrorAmplificationExperiment, frequency mode); each y value and
each N is one program. N can be one number (one map) or a list (to narrow down, as in Lu et al., Nat. Commun. 14,
5767 (2023), Supplementary Note 8).

Phase rule: the phase of the pulse under test advances by ``phase_step_deg`` after every pulse
(phase of pulse k = pulse_type[3] + k * phase_step_deg):
    0       identical phases (paper protocol a)
    180     pi(0) pi(180) pulses, as the frequency mode of ErrorAmplificationExperiment
    theta_R stepped phases (paper protocol b; their theta_R = 1.3 rad between swaps)

For a pi pulse, an odd N must move the state and an even N must bring it back. ``analyze`` turns the readout into
a success probability per N, takes the product over N, and gives the best point per N (it must not move with N
for a correct calibration) and of the product.

Experiment parameters (cfg.expt):
    pulse_type     ['multiphoton', 'f0-g1', 'pi', 0] etc. (as ErrorAmplificationExperiment)
    start, step, expts   frequency sweep (MHz)
    y_param        'gain' or 'length' ('length' only for flat_top pulses; flat part, us)
    y_values       list of gains or flat lengths
    n_pulses       int or list of int: number of pulses of the pulse under test
    phase_step_deg phase advance after every pulse (default 0)
    start_readout  'e' or 'g': readout of the start state (default: 'e' for multiphoton pulses starting in f,
                   whose post pulse maps f to e; 'g' otherwise)
    reps, rounds, relax_delay, active_reset, ... as ErrorAmplificationExperiment
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict, Experiment
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import MMAveragerProgram, MM_base
from experiments.single_qubit.error_amplification import ErrorAmplificationProgram


def phase_sequence(n_pulses, phase_step_deg, phase0_deg=0.0):
    """Phases (deg, mod 360) of the n_pulses pulses under test."""
    return [float((phase0_deg + k * phase_step_deg) % 360) for k in range(int(n_pulses))]


class ErrorAmplification2DProgram(ErrorAmplificationProgram):
    """ErrorAmplificationProgram in frequency mode, with N single pulses (not pairs), a free phase step, and
    optional gain / flat-length overrides of the pulse under test."""

    def initialize(self):
        super().initialize()
        cfg = AttrDict(self.cfg)
        if cfg.expt.get('override_gain') is not None:
            self.pulse_to_test[1] = int(cfg.expt.override_gain)
        if cfg.expt.get('override_length') is not None:
            if self.pulse_to_test[5] != 'flat_top':
                raise ValueError("y_param='length' needs a flat_top pulse under test")
            self.pulse_to_test[2] = float(cfg.expt.override_length)
            self._length = self.us2cycles(self.pulse_to_test[2], gen_ch=self.pulse_to_test[4])
        self.phases = phase_sequence(self.n_pulses * self.pi_frac, cfg.expt.get('phase_step_deg', 0.0),
                                     self.pulse_to_test[3])

    def body(self):
        cfg = AttrDict(self.cfg)
        self.reset_and_sync()
        if cfg.expt.active_reset:
            self.active_reset(**MM_base.get_active_reset_params(cfg))
        self.sync_all(self.us2cycles(0.2))
        if self.pre_pulse_list is not None:
            self.custom_pulse(cfg, self.pre_pulse_list, prefix='pre_')

        ch = self.pulse_to_test[4]
        kw = dict(ch=ch, style=self.pulse_style, freq=self._freq, gain=int(self.pulse_to_test[1]),
                  phase=self.deg2reg(self.phases[0] if self.phases else 0.0, gen_ch=ch), waveform="pulse_to_test")
        if self.pulse_to_test[5] == "flat_top":
            kw['length'] = self._length
        self.set_pulse_registers(**kw)
        self.mathi(self.channel_page, self.r_freq, self.r_freq2, "+", 0)     # swept frequency from the register

        for k, ph in enumerate(self.phases):
            if k > 0:
                self.safe_regwi(self.channel_page, self.r_phase, self.deg2reg(ph, gen_ch=ch))
            self.pulse(ch=ch)
            if self.floquet_sync_delay != 0:
                self.sync_all(self.floquet_sync_delay)

        self.sync_all()
        if self.post_pulse_list is not None:
            self.custom_pulse(cfg, self.post_pulse_list, prefix='post_')
        self.sync_all()
        self.measure_wrapper()


class ErrorAmplification2DExperiment(Experiment):
    """Frequency x (gain or flat length) error amplification for one or several N. See the module docstring."""

    def __init__(self, soccfg=None, path='', prefix='ErrorAmplification2D', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=None):
        if debug is None:
            debug = self.cfg.expt.get('debug', False)
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

        cfg = deepcopy(self.cfg)
        cfg.expt.parameter_to_test = 'frequency'          # the parent initialize sets up the frequency registers
        adc_ch = cfg.hw.soc.adcs.readout.ch
        n_list = np.atleast_1d(cfg.expt.n_pulses).astype(int)
        y_param = cfg.expt.get('y_param', 'gain')
        if y_param not in ('gain', 'length'):
            raise ValueError("y_param must be 'gain' or 'length'")
        y_values = np.atleast_1d(cfg.expt.y_values).astype(float)
        read_num = 1
        if cfg.expt.get('active_reset', False):
            read_num += MMAveragerProgram.active_reset_read_num(**MM_base.get_active_reset_params(cfg))

        avgi = np.zeros((len(n_list), len(y_values), int(cfg.expt.expts)))
        avgq = np.zeros_like(avgi)
        xpts = None
        for i, n in enumerate(tqdm(n_list, disable=not progress)):
            for j, y in enumerate(y_values):
                c = deepcopy(cfg)
                c.expt.n_pulses = int(n)
                c.expt.override_gain = int(round(y)) if y_param == 'gain' else None
                c.expt.override_length = float(y) if y_param == 'length' else None
                self.prog = ErrorAmplification2DProgram(soccfg=self.soccfg, cfg=c)
                x, ai, aq = self.prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                                              progress=False, debug=debug, readouts_per_experiment=read_num)
                avgi[i, j] = ai[adc_ch[0]][-1]
                avgq[i, j] = aq[adc_ch[0]][-1]
                xpts = x
        self.data = dict(x_pts=np.asarray(xpts), y_pts=y_values, N_pts=n_list, avgi=avgi, avgq=avgq,
                         amp=np.abs(avgi + 1j * avgq), phase=np.angle(avgi + 1j * avgq))
        return self.data

    def _start_readout(self):
        s = self.cfg.expt.get('start_readout')
        if s:
            return s
        pt = self.cfg.expt.pulse_type
        return 'e' if (pt[0] == 'multiphoton' and pt[1][0] == 'f') else 'g'

    def analyze(self, data=None, **kwargs):
        """success[N, y, x]: probability that N pulses did what a perfect pulse does (odd N: state moved, even N:
        state back), from the I quadrature scaled with the config Ig / Ie. Product over N, best (x, y) per N and of
        the product (argmax of a 3x3-smoothed map)."""
        data = self.data if data is None else data
        Ig, Ie = self.cfg.device.readout.Ig[0], self.cfg.device.readout.Ie[0]
        pe = (np.asarray(data['avgi']) - Ig) / (Ie - Ig)                   # fraction read as e
        start_e = self._start_readout() == 'e'
        succ = np.empty_like(pe)
        for i, n in enumerate(data['N_pts']):
            moved = n % 2 == 1
            succ[i] = (1 - pe[i]) if (start_e == moved) else pe[i]       # start e & moved -> want g, ...
        succ = np.clip(succ, 0, 1)
        data['success'] = succ
        data['product'] = np.prod(succ, axis=0)
        x, y = np.asarray(data['x_pts']), np.asarray(data['y_pts'])

        def best(m):
            k = np.ones((3, 3)) / 9.0
            mp = np.pad(m, 1, mode='edge')
            sm = sum(mp[a:a + m.shape[0], b:b + m.shape[1]] * k[a, b] for a in range(3) for b in range(3))
            jj, ii = np.unravel_index(np.argmax(sm), sm.shape)
            return float(x[ii]), float(y[jj]), float(sm[jj, ii])

        data['best_per_N'] = np.array([best(s) for s in succ])          # (x, y, value) per N
        data['best_product'] = np.array(best(data['product']))
        return data

    def display(self, data=None, **kwargs):
        data = self.data if data is None else data
        if 'success' not in data:
            self.analyze(data)
        x, y, N = data['x_pts'], data['y_pts'], data['N_pts']
        n = len(N) + 1
        fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.4), sharey=True)
        ext = [x[0], x[-1], y[0], y[-1]]
        maps = list(data['success']) + [data['product']]
        titles = [f'N = {k}' for k in N] + ['product']
        for ax, m, t in zip(np.atleast_1d(axs), maps, titles):
            ax.imshow(m, origin='lower', aspect='auto', extent=ext, vmin=0, vmax=1, cmap='viridis')
            ax.set_title(t, fontsize=9); ax.set_xlabel('frequency (MHz)')
        for ax, b in zip(np.atleast_1d(axs), list(data['best_per_N']) + [data['best_product']]):
            ax.plot(b[0], b[1], 'r+', ms=10)
        np.atleast_1d(axs)[0].set_ylabel(self.cfg.expt.get('y_param', 'gain'))
        pt = self.cfg.expt.pulse_type
        fig.suptitle(f"2D error amplification {pt[0]} {pt[1]} {pt[2]}, phase step {self.cfg.expt.get('phase_step_deg', 0)} deg",
                     fontsize=10)
        fig.tight_layout()
        plt.show()
