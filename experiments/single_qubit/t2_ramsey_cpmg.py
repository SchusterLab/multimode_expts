import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import AttrDict, Experiment

import fitting.fitting as fitter
from fitting.decaysin_analysis import (
    fit_decaysin_with_envelope_selection,
    h5_safe_data,
    plot_envelope_overlay,
)
from experiments.MM_base import (
    MM_base, MMAveragerProgram, MMRAveragerProgram, warn_step_subcycle,
)

"""
Ramsey / CPMG on the transmon qubit (ge or ef transition).

This is the qubit analogue of RamseyCouplerExperiment / CavityModeRamseyExperiment,
written specifically for CPMG noise spectroscopy (cpmg_noise_spectroscopy.ipynb).
It deliberately does NOT reuse RamseyExperiment (t2_ramsey.py): that class uses
`echoes=[bool, N]` with *uniform* inter-pulse spacing, an x-axis that is the
*per-segment* wait, and X-axis (phase 0) refocusing pulses played as two hpi's.
That is a CP echo train, not CPMG, and is incompatible with the narrow-filter
PSD inversion (fitting.noise_psd_extraction).

This class instead matches the contract shared by the M1 and coupler sections of
the notebook:
  - scalar `echoes=N` (0 = plain Ramsey), swept by SweepRunner;
  - x-axis = TOTAL free-evolution time tau;
  - standard CPMG timing (Bylander 2011 Eq. 18): end-spacings tau/(2N),
    inter-pi spacings tau/N;
  - pi pulses at phase = 90 deg (Y-axis), refocusing the X-coherence prepared
    by the first phase-0 pi/2.

Sequence
--------
  pi/2(0) -> sync(tau/2N) -> pi(90) -> sync(tau/N) -> ... -> pi(90)
          -> sync(tau/2N) -> pi/2(advance_phase) -> readout.

The pi/2 pulse is the calibrated qubit hpi (hpi_ge, or hpi_ef when checkEF);
the CPMG pi reuses the MM-base full-pi waveform on the same transition
(pi_qubit_ge / pi_qubit_ef) at phase 90. user_defined_freq overrides the pi/2
frequency/gain/width and derives the echo pi as 2x the hpi gain on the
hpi-width waveform (linear-response assumption; warned).
"""


class RamseyCPMGProgram(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        self.checkEF = bool(cfg.expt.get('checkEF', False))
        # qubit_ge_init/after: when probing ef we first populate |e> with a ge
        # pi and map back to |g> before readout. ef_init=False disables both.
        self.do_ge_wrap = self.checkEF and bool(cfg.expt.get('ef_init', True))

        self.num_echoes = int(cfg.expt.get('echoes', 0))
        # Subcycle warning: smallest sub-segment is tau/(2N) under CPMG; tau
        # for plain Ramsey.
        if self.num_echoes >= 1:
            min_step = cfg.expt.step / (2 * self.num_echoes)
            warn_label = f"step/(2*echoes={self.num_echoes}) [CPMG]"
        else:
            min_step = cfg.expt.step
            warn_label = "ramsey step"
        warn_step_subcycle(self.soccfg, min_step,
                           gen_ch=self.qubit_chs[qTest], label=warn_label)

        # ----- pi/2 (Ramsey) pulse + echo pi setup -----
        # Defaults: ge transition. The echo pi reuses the MM-base full-pi
        # waveform already declared on the qubit channel.
        self.pi2sigma = self.us2cycles(
            cfg.device.qubit.pulses.hpi_ge.sigma[qTest],
            gen_ch=self.qubit_chs[qTest])
        self.f_test_reg = self.f_ge_reg[0]
        self.gain_test = cfg.device.qubit.pulses.hpi_ge.gain[qTest]

        self.echo_freq_reg = self.f_test_reg
        self.echo_waveform = "pi_qubit_ge"
        self.echo_gain = self.pi_ge_gain

        if self.checkEF:
            self.pi2sigma = self.us2cycles(
                cfg.device.qubit.pulses.hpi_ef.sigma[qTest],
                gen_ch=self.qubit_chs[qTest])
            self.f_test_reg = self.f_ef_reg[qTest]
            self.gain_test = cfg.device.qubit.pulses.hpi_ef.gain[qTest]
            self.echo_freq_reg = self.f_test_reg
            self.echo_waveform = "pi_qubit_ef"
            self.echo_gain = self.pi_ef_gain

        user_defined = cfg.expt.get('user_defined_freq', [False])
        if user_defined[0]:
            self.f_test_reg = self.freq2reg(
                user_defined[1], gen_ch=self.qubit_chs[qTest])
            self.gain_test = user_defined[2]
            self.pi2sigma = self.us2cycles(
                user_defined[3], gen_ch=self.qubit_chs[qTest])
            # No calibrated full-pi waveform at an arbitrary user frequency:
            # reuse the hpi-width pi/2 waveform and double its gain. Accurate
            # only in the DAC/amp linear regime.
            self.echo_freq_reg = self.f_test_reg
            self.echo_waveform = "pi2_test_ram"
            self.echo_gain = int(round(2 * self.gain_test))
            if self.num_echoes >= 1:
                print("[t2_ramsey_cpmg] WARNING: user_defined_freq CPMG echo pi "
                      "derived as 2x hpi gain on the hpi-width waveform "
                      "(linear-response assumption — verify pulse not near "
                      "DAC/amp saturation).")

        self.add_gauss(ch=self.qubit_chs[qTest], name="pi2_test_ram",
                       sigma=self.pi2sigma, length=self.pi2sigma * 4)

        # ----- wait / phase registers -----
        # r_wait     = end-segment (pi/2 <-> first/last pi) = tau/(2N)
        # r_wait_mid = inter-pi segment                     = tau/N
        # For Ramsey (N=0) only r_wait is used, holding the full tau.
        self.q_rp = self.q_rps[qTest]
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase3 = 5      # int4 scratch for the freq|phase pack
        self.r_wait_mid = 6
        if self.qubit_ch_types[qTest] == 'int4':
            self.r_phase = self.sreg(self.qubit_chs[qTest], "freq")
        else:
            self.r_phase = self.sreg(self.qubit_chs[qTest], "phase")

        if self.num_echoes >= 1:
            self.safe_regwi(
                self.q_rp, self.r_wait,
                self.us2cycles(cfg.expt.start / (2 * self.num_echoes),
                               gen_ch=self.qubit_chs[qTest]))
            self.safe_regwi(
                self.q_rp, self.r_wait_mid,
                self.us2cycles(cfg.expt.start / self.num_echoes,
                               gen_ch=self.qubit_chs[qTest]))
        else:
            self.safe_regwi(
                self.q_rp, self.r_wait,
                self.us2cycles(cfg.expt.start, gen_ch=self.qubit_chs[qTest]))
        self.safe_regwi(self.q_rp, self.r_phase2, 0)

        print('fge is ', cfg.device.qubit.f_ge[qTest])
        print('fef is ', cfg.device.qubit.f_ef[qTest])
        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        self.reset_and_sync()

        if cfg.expt.get('pre_active_reset_pulse', False):
            if cfg.expt.get('gate_based', False):
                creator = self.get_prepulse_creator(
                    cfg.expt.pre_active_reset_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix='pre_ar_')
            else:
                self.custom_pulse(cfg, cfg.expt.pre_active_reset_sweep_pulse,
                                  prefix='pre_ar_')

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        self.sync_all(self.us2cycles(0.1))

        if cfg.expt.get('prepulse', False):
            if cfg.expt.get('gate_based', False):
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix='pre_')
            else:
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

        # ef: populate |e> first
        if self.do_ge_wrap:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb",
                                 freq=self.f_ge_reg[0], phase=0,
                                 gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            self.sync_all(self.us2cycles(0.01))

        # first pi/2 (phase 0)
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb",
                             freq=self.f_test_reg, phase=0,
                             gain=self.gain_test, waveform="pi2_test_ram")
        self.sync_all(self.us2cycles(0.01))

        # advance the swept wait time (end-segment)
        self.sync_all()
        self.sync(self.q_rp, self.r_wait)

        # CPMG echo loop: pi pulses at phase=90 deg (Y-axis).
        # Layout: pi/2 -> sync(r_wait) -> pi_1 -> sync(r_wait_mid) -> ...
        #         -> pi_N -> sync(r_wait) -> second pi/2.
        if self.num_echoes >= 1:
            echo_phase = self.deg2reg(90, gen_ch=self.qubit_chs[qTest])
            for k in range(self.num_echoes):
                self.set_pulse_registers(
                    ch=self.qubit_chs[qTest], style="arb",
                    freq=self.echo_freq_reg, phase=echo_phase,
                    gain=self.echo_gain, waveform=self.echo_waveform)
                self.sync_all(self.us2cycles(0.01))
                self.pulse(ch=self.qubit_chs[qTest])
                self.sync_all()
                is_last = (k == self.num_echoes - 1)
                wait_reg = self.r_wait if is_last else self.r_wait_mid
                self.sync(self.q_rp, wait_reg)
                self.sync_all()

        # second pi/2 with advanced phase (phase reg loaded from r_phase2)
        self.set_pulse_registers(
            ch=self.qubit_chs[qTest], style="arb", freq=self.f_test_reg,
            phase=self.deg2reg(cfg.expt.get('advance_phase', 0),
                               gen_ch=self.qubit_chs[qTest]),
            gain=self.gain_test, waveform="pi2_test_ram")
        if self.qubit_ch_types[qTest] == 'int4':
            self.bitwi(self.q_rp, self.r_phase3, self.r_phase2, '<<', 16)
            self.bitwi(self.q_rp, self.r_phase3, self.r_phase3, '|',
                       self.f_test_reg)
            self.mathi(self.q_rp, self.r_phase, self.r_phase3, "+", 0)
            self.sync_all(self.us2cycles(0.01))
        else:
            self.mathi(self.q_rp, self.r_phase, self.r_phase2, "+", 0)
            self.sync_all(self.us2cycles(0.01))
        self.pulse(ch=self.qubit_chs[qTest])

        self.sync_all()

        # ef: map |e> back to |g> for the standard ge readout
        if self.do_ge_wrap:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb",
                                 freq=self.f_ge_reg[0], phase=0,
                                 gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            self.sync_all(self.us2cycles(0.01))

        self.measure_wrapper()

    def update(self):
        qTest = self.qubits[0]

        if self.num_echoes >= 1:
            step_end = self.us2cycles(
                self.cfg.expt.step / (2 * self.num_echoes),
                gen_ch=self.qubit_chs[qTest])
            step_mid = self.us2cycles(
                self.cfg.expt.step / self.num_echoes,
                gen_ch=self.qubit_chs[qTest])
            self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', step_end)
            self.sync_all(self.us2cycles(0.01))
            self.mathi(self.q_rp, self.r_wait_mid, self.r_wait_mid, '+',
                       step_mid)
        else:
            self.mathi(self.q_rp, self.r_wait, self.r_wait, '+',
                       self.us2cycles(self.cfg.expt.step,
                                      gen_ch=self.qubit_chs[qTest]))
        self.sync_all(self.us2cycles(0.01))
        # phase advance tracks TOTAL tau: per-point step = 360*f_R*step [deg]
        phase_step = self.deg2reg(
            360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step,
            gen_ch=self.qubit_chs[qTest])
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+', phase_step)
        self.sync_all(self.us2cycles(0.01))


class RamseyCPMGExperiment(Experiment):
    """
    Qubit (ge/ef) Ramsey + CPMG, matched to the CPMG noise-spectroscopy
    pipeline (scalar `echoes`, total-tau x-axis, Bylander timing).

    Experimental Config:
    expt = dict(
        start:        wait time start [us]  (TOTAL free-evolution tau)
        step:         wait time step  [us]  (Nyquist: 0.5/step > ramsey_freq)
        expts:        number of tau points
        ramsey_freq:  phase advance rate for the second pi/2 [MHz]
        reps:         averages per tau point
        rounds:       sweep repetitions
        qubits:       list, e.g. [0]
        echoes:       (optional, default 0) number of CPMG pi pulses. 0 = plain
                      Ramsey. >=1 uses standard CPMG timing (Bylander 2011
                      Eq. 18): end-spacings tau/(2N), inter-pi tau/N, pi pulses
                      at phase=90 deg on the same transition as the pi/2.
        checkEF:      (optional, default False) Ramsey on ef instead of ge.
        ef_init:      (optional, default True) when checkEF, wrap with ge pi
                      before/after to populate |e> and map back for readout.
        user_defined_freq: (optional) [enable, freq_MHz, gain_DAC, hpi_sigma_us]
                      override for the pi/2; CPMG pi is derived as 2x gain on
                      the hpi-width waveform (linear-response assumption).
        advance_phase:   (optional, default 0) extra phase on pi/2 #2 [deg]
        active_reset:    (optional) bool, run active reset at start of body
        prepulse:        (optional) bool, play cfg.expt.pre_sweep_pulse first
        pre_sweep_pulse: (optional) gate-list/array spec for the prepulse
        gate_based:      (optional) bool, interpret pulse specs as gate lists
    )
    """

    def __init__(self, soccfg=None, path='', prefix='RamseyCPMG',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        read_num = 1
        if self.cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        ramsey = RamseyCPMGProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = ramsey.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            readouts_per_experiment=read_num,
        )

        # active reset adds readouts per experiment; the Ramsey result is last.
        avgi = avgi[0][-1]
        avgq = avgq[0][-1]
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq,
                'amps': amps, 'phases': phases}
        self.data = data
        return data

    def analyze(self, data=None, fit=True, fitparams=None,
                gauss_ssr_margin=0.05, use_x0=False, **kwargs):
        """Fit Ramsey data with both exp and Gaussian envelopes; pick winner.

        Mirrors RamseyCouplerExperiment.analyze so this class behaves
        identically under the shared notebook fit/plot helpers.
        """
        if data is None:
            data = self.data
        if not fit:
            return data

        xpts = np.asarray(data['xpts'])
        ramsey_freq = self.cfg.expt.ramsey_freq
        for ch in ('avgi', 'avgq', 'amps'):
            r = fit_decaysin_with_envelope_selection(
                xpts, np.asarray(data[ch]),
                fitparams=fitparams, use_x0=use_x0,
                gauss_ssr_margin=gauss_ssr_margin)
            data[f'fit_{ch}']           = r['p']
            data[f'fit_err_{ch}']       = r['cov']
            data[f'fit_{ch}_exp']       = r['p_exp']
            data[f'fit_{ch}_gauss']     = r['p_gauss']
            data[f'fit_err_{ch}_exp']   = r['cov_exp']
            data[f'fit_err_{ch}_gauss'] = r['cov_gauss']
            data[f'fit_ssr_{ch}_exp']   = r['ssr_exp']
            data[f'fit_ssr_{ch}_gauss'] = r['ssr_gauss']
            data[f'fit_envelope_{ch}']  = r['envelope']
            p = r['p']
            if isinstance(p, (list, np.ndarray)):
                data[f'f_adjust_ramsey_{ch}'] = sorted(
                    (ramsey_freq - p[1], ramsey_freq + p[1]), key=abs)
        data['fit_envelope'] = data['fit_envelope_avgi']
        data['fit_gauss_ssr_margin'] = float(gauss_ssr_margin)
        return data

    def _plot_fit_overlay(self, data, channel, label_axis, f_qubit,
                          print_freq_line):
        p = data.get(f'fit_{channel}')
        if not isinstance(p, (list, np.ndarray)):
            return
        envelope = data.get(f'fit_envelope_{channel}', 'exp')
        pCov = data[f'fit_err_{channel}']
        x = data['xpts'][:-1]
        t2_label = '$T_2^*$ (Gauss)' if envelope == 'gauss' else '$T_2$ (Exp)'

        ssr_exp = data.get(f'fit_ssr_{channel}_exp', np.nan)
        ssr_gau = data.get(f'fit_ssr_{channel}_gauss', np.nan)
        try:
            captionStr = (f'{t2_label} fit [us]: {p[3]:.3} $\\pm$ '
                          f'{np.sqrt(pCov[3][3]):.3}\n'
                          f'SSR exp={ssr_exp:.3g} / gauss={ssr_gau:.3g}')
        except (ValueError, TypeError):
            captionStr = f'{t2_label} fit failed'

        plot_envelope_overlay(x, p, envelope=envelope, label=captionStr)
        plt.legend()

        if print_freq_line:
            print(f'Current qubit frequency: {f_qubit}')
        print(f'Envelope chosen ({label_axis}): {envelope}  '
              f'(SSR exp={ssr_exp:.4g}, gauss={ssr_gau:.4g}, '
              f'margin={data.get("fit_gauss_ssr_margin", 0.05):.0%})')
        print(f'Fit frequency from {label_axis} [MHz]: {p[1]} +/- '
              f'{np.sqrt(pCov[1][1])}')
        if p[1] > 2 * self.cfg.expt.ramsey_freq:
            print('WARNING: Fit frequency >2*wR; may be too far from the transition.')
        adj = data[f'f_adjust_ramsey_{channel}']
        print(f'Suggested new qubit frequency from fit {label_axis} [MHz]:\n',
              f'\t{f_qubit + adj[0]}\n',
              f'\t{f_qubit + adj[1]}')
        print(f'T2 from fit {label_axis} [us]: {p[3]}  (envelope={envelope})')

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubits[0]
        checkEF = bool(self.cfg.expt.get('checkEF', False))
        f_qubit = (self.cfg.device.qubit.f_ef[q] if checkEF
                   else self.cfg.device.qubit.f_ge[q])
        title = ('EF ' if checkEF else '') + 'Qubit Ramsey/CPMG'

        plt.figure(figsize=(10, 9))
        plt.subplot(
            211,
            title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
            ylabel="I [ADC level]",
        )
        plt.plot(data["xpts"][:-1], data["avgi"][:-1], 'o-')
        if fit:
            self._plot_fit_overlay(data, channel='avgi', label_axis='I',
                                   f_qubit=f_qubit, print_freq_line=True)
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1], 'o-')
        if fit:
            self._plot_fit_overlay(data, channel='avgq', label_axis='Q',
                                   f_qubit=f_qubit, print_freq_line=False)

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        if data is None:
            data = self.data
        print(f'Saving {self.fname}')
        super().save_data(data=h5_safe_data(data))
        return self.fname
