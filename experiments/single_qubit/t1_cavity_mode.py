"""
Cavity Mode T1 Experiment

Measures the energy decay of a Fock state in the manipulate mode or in a
storage mode: prepare |fock>, wait, read out. The wait is a register sweep, so
one program covers all waits.

State preparation is automatic based on the mode:
  - manipulate: MM_base.prep_man_fock_state(str(fock)), the photon ladder
    g_k-e_k, e_k-f_k, f_k-g_k+1 for k < fock (fock = 1 or 2)
  - storage: MM_base.prep_storage_cardinal('1'), i.e. ge pi, ef pi, man pi,
    M1-S<idx> pi (fock = 1)

Readout:
  - 'map': the reverse of the preparation brings |fock> back to the qubit |g>.
    For fock = 1 the only decay product, |0>, ends in |e>, so the signal is
    the population of |1>. For fock = 2 this readout is not valid: the |1>
    product also ends in |g> (the rung pulses are broadband in photon
    number), so 'map' would measure P1 + P2 and is refused.
  - 'parity': the manipulate parity (after the swap back to M1 for storage),
    P0 - P1 + P2 - ... For pure decay (rate k / T1 from |k>) it is
    (1 - 2 exp(-t / T1)) ** fock, so fock = 1 and fock = 2 measure the same T1
    if the decay of |2> is twice that of |1>.

Fit (analyze): 'map': offset + amp * exp(-t / T1). 'parity':
offset + amp * (1 - 2 exp(-t / T1)) ** fock. T1 and its error are stored as
data['T1'], data['T1_err'] [us]; the parameters (offset, amp, T1) as
data['fit_avgi'].
"""

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from scipy.optimize import curve_fit
from slab import AttrDict, Experiment

from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram, warn_step_subcycle
from fitting.decaysin_analysis import h5_safe_data


def exp_decay(t, offset, amp, t1):
    """offset + amp * exp(-t/t1) (three parameters, no degenerate time shift)."""
    return offset + amp * np.exp(-np.asarray(t, float) / t1)


def parity_decay(t, offset, amp, t1, fock=1):
    """Parity signal of |fock> after a wait t for pure decay: offset + amp * (1 - 2 exp(-t/t1)) ** fock."""
    return offset + amp * (1 - 2 * np.exp(-np.asarray(t, float) / t1)) ** fock


class CavityModeT1Program(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.get('rounds', 1)
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]
        warn_step_subcycle(self.soccfg, cfg.expt.step, gen_ch=None, label='step [T1]')

        mode, fock, readout = cfg.expt.mode, int(cfg.expt.get('fock', 1)), cfg.expt.get('readout', 'map')
        if mode == 'manipulate':
            if fock not in (1, 2):
                raise ValueError(f"fock must be 1 or 2 for mode 'manipulate'; got {fock}")
            prep_str = self.prep_man_fock_state(man_no=cfg.expt.get('man_mode_idx', 1), state=str(fock))
            to_man_str = []
        elif mode == 'storage':
            if fock != 1:
                raise ValueError(f"fock must be 1 for mode 'storage'; got {fock}")
            prep_str = self.prep_storage_cardinal('1', cfg.expt.storage_mode_idx)
            to_man_str = prep_str[-1:]                  # the M1-S<idx> swap back to M1
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'manipulate' or 'storage'.")

        if readout == 'map':
            if fock != 1:
                raise ValueError("readout 'map' measures P1 + P2 for fock=2; use readout='parity'")
            post_str = prep_str[::-1]
        elif readout == 'parity':
            post_str = to_man_str + self.get_parity_str(cfg.expt.get('man_mode_no', 1), second_phase=180)
        else:
            raise ValueError(f"Unknown readout '{readout}'. Use 'map' or 'parity'.")
        print(f"T1 {mode} |{fock}>: prep {prep_str}, readout {readout}: {post_str}")
        self.prep_pulse = self.get_prepulse_creator(prep_str).pulse.tolist()
        self.post_pulse = self.get_prepulse_creator(post_str).pulse.tolist()

        # wait register on the qubit page (as T1Program)
        self.wait_page = self.ch_page(self.qubit_chs[qTest])
        self.r_wait = 3
        self.safe_regwi(self.wait_page, self.r_wait, self.us2cycles(cfg.expt.start))
        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        self.reset_and_sync()
        if cfg.expt.get('active_reset', False):
            self.active_reset(**MM_base.get_active_reset_params(cfg))
        self.custom_pulse(cfg, self.prep_pulse, prefix='prep_')
        self.sync_all()
        self.sync(self.wait_page, self.r_wait)
        self.sync_all()
        self.custom_pulse(cfg, self.post_pulse, prefix='post_')
        self.measure_wrapper()

    def update(self):
        self.mathi(self.wait_page, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step))
        self.sync_all(self.us2cycles(0.01))


class CavityModeT1Experiment(Experiment):
    """
    Cavity Mode T1 Experiment

    Experimental Config:
    expt = dict(
        start: wait time start [us]
        step: wait time step [us]
        expts: number of sweep points
        reps: averages per point
        rounds: number of rounds (default 1)
        qubits: e.g. [0]

        mode: 'manipulate' or 'storage'
        fock: 1 (default) or 2 (manipulate only)
        readout: 'map' (default; fock 1 only) or 'parity'

        # for manipulate mode:
        man_mode_idx: manipulate mode index (default 1)
        # for storage mode:
        storage_mode_idx: storage mode index (e.g. 4)
        # for the parity readout:
        man_mode_no: manipulate mode of the parity measurement (default 1)

        active_reset: True/False (default False)
        relax_delay: [us]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CavityModeT1', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        self.format_config_before_experiment(len(self.cfg.device.qubit.f_ge))
        read_num = 1
        if self.cfg.expt.get('active_reset', False):
            read_num += MMAveragerProgram.active_reset_read_num(**MM_base.get_active_reset_params(self.cfg))
        prog = CavityModeT1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                                         progress=progress, readouts_per_experiment=read_num)
        self.prog = prog
        avgi, avgq = avgi[0][-1], avgq[0][-1]
        data = dict(xpts=x_pts, avgi=avgi, avgq=avgq, amps=np.abs(avgi + 1j * avgq), phases=np.angle(avgi + 1j * avgq))
        data['idata'], data['qdata'] = prog.collect_shots()
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        if not fit:
            return data
        t, y = np.asarray(data['xpts'], float), np.asarray(data['avgi'], float)
        fock, readout = int(self.cfg.expt.get('fock', 1)), self.cfg.expt.get('readout', 'map')
        span = max(float(t[-1] - t[0]), 1e-9)
        y_end = float(np.mean(y[-max(1, len(y) // 10):]))
        if readout == 'map':
            model, p0 = exp_decay, [y_end, float(y[0]) - y_end, span / 3]
        else:
            # start values from the shape: fock 1 goes from offset - amp to offset + amp; fock 2 starts and
            # ends at offset + amp, with the extreme (offset) at t = T1 ln 2
            if fock == 1:
                p0 = [(float(y[0]) + y_end) / 2, (y_end - float(y[0])) / 2, span / 3]
            else:
                k = int(np.argmax(np.abs(y - y_end)))
                p0 = [float(y[k]), y_end - float(y[k]), max(float(t[k]), span / 50) / np.log(2)]
            model = lambda tt, offset, amp, t1: parity_decay(tt, offset, amp, t1, fock)
        p, pcov = curve_fit(model, t, y, p0=p0, bounds=([-np.inf, -np.inf, span / 1e3], [np.inf, np.inf, span * 1e2]),
                            maxfev=20000)
        data['fit_avgi'], data['fit_err_avgi'] = np.asarray(p, float), np.asarray(pcov, float)
        data['T1'], data['T1_err'] = float(p[2]), float(np.sqrt(pcov[2, 2]))
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        t, y = np.asarray(data['xpts'], float), np.asarray(data['avgi'], float)
        fock, readout, mode = int(self.cfg.expt.get('fock', 1)), self.cfg.expt.get('readout', 'map'), self.cfg.expt.mode
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, y, 'o', ms=3, label='data')
        if fit and 'fit_avgi' in data:
            tt = np.linspace(t[0], t[-1], 500)
            yy = exp_decay(tt, *data['fit_avgi']) if readout == 'map' else parity_decay(tt, *data['fit_avgi'], fock=fock)
            ax.plot(tt, yy, '-', label=f"T1 = {data['T1']:.1f} +/- {data['T1_err']:.1f} us")
        ax.set_title(f'{mode} |{fock}> T1 ({readout} readout)')
        ax.set_xlabel('wait (us)')
        ax.set_ylabel('I (ADC)')
        ax.legend()
        fig.tight_layout()
        plt.show()
        return fig, ax

    def save_data(self, data=None):
        if data is None:
            data = self.data
        print(f'Saving {self.fname}')
        super().save_data(data=h5_safe_data(data))
        return self.fname
