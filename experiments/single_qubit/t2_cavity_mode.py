"""
Cavity Mode Ramsey / Echo Experiment

Measures T2 coherence of a cavity mode (storage, manipulate, or coupler) by
preparing |0>+|1> in the mode, waiting a variable time, and applying the
second half of the gate with a Ramsey phase.

State preparation is automatic based on the mode:
  - storage: qubit_hpi + ef_pi + man_pi → storage_pi (Ramsey) → wait → ...
  - manipulate: g0-e0_hpi + e0-f0_pi → f0-g1_pi (Ramsey) → wait → ...
    With fock=2 the state is |0>+|2> (the broadband ladder of
    MM_base.prep_man_fock_state(n=2)) and the Ramsey pulse is f1-g2.
  - coupler: custom pulse (user must specify coupler_pulse)

After the Ramsey interaction, the reverse of the state prep brings the
state back for parity readout.

Custom pre/postpulses override the automatic ones (with a warning).

phase_cycle=True measures every wait twice, with the second Ramsey pulse at
phase_offset and phase_offset + 180 deg, and analyses half the difference.
Everything that does not depend on the Ramsey phase cancels. For fock=2 this
removes the background from the photons that decay to |1> or |0> during the
wait: the sideband pulses are not photon-number selective, so the decoder
moves that population in a wait-dependent way, but it cannot make a fringe.

Refactored from t2_cavity.py — Feb 2026
"""

from contextlib import contextmanager
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import AttrDict, Experiment

from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram, warn_step_subcycle
from fitting.fit_display_classes import RamseyFitting
from fitting.decaysin_analysis import h5_safe_data


def fringe(phase_deg, offset, amp, phase0_deg):
    """offset + amp * cos(phase - phase0), phases in degrees."""
    return offset + amp * np.cos(np.radians(np.asarray(phase_deg, float) - phase0_deg))


def fit_fringe(phase_deg, y):
    """Linear least squares of y = B + C cos(phase) + S sin(phase): amplitude A = hypot(C, S) and phase0 =
    atan2(S, C), with errors from the residual variance. Returns the data keys of a sweep='phase' analysis."""
    ph, y = np.radians(np.asarray(phase_deg, float)), np.asarray(y, float)
    X = np.column_stack([np.ones_like(ph), np.cos(ph), np.sin(ph)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b, c, s = beta
    resid = y - X @ beta
    cov = np.sum(resid ** 2) / max(len(y) - 3, 1) * np.linalg.inv(X.T @ X)
    amp = float(np.hypot(c, s))
    amp_err = float(np.sqrt(c ** 2 * cov[1, 1] + s ** 2 * cov[2, 2] + 2 * c * s * cov[1, 2]) / amp) if amp > 0 else np.nan
    phase_err = float(np.degrees(np.sqrt(s ** 2 * cov[1, 1] + c ** 2 * cov[2, 2] - 2 * c * s * cov[1, 2]) / amp ** 2)) if amp > 0 else np.nan
    phase0 = float(np.degrees(np.arctan2(s, c)))
    return dict(fit_avgi=np.array([b, amp, phase0]), fringe_amp=amp, fringe_amp_err=amp_err, fringe_phase_deg=phase0,
                fringe_phase_err_deg=phase_err, fringe_offset=float(b))


class CavityModeRamseyProgram(MMRAveragerProgram):
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

        mode = cfg.expt.mode  # 'storage', 'manipulate', or 'coupler'
        self.mode = mode
        self.num_echoes = cfg.expt.get('echoes', 0)
        self.sweep = cfg.expt.get('sweep', 'wait')
        if self.sweep not in ('wait', 'phase'):
            raise ValueError(f"Unknown sweep '{self.sweep}'. Use 'wait' or 'phase'.")
        # Standard CPMG timing (Bylander 2011 Eq. 18): end-spacings tau/(2N)
        # and inter-pi spacings tau/N. For Hahn echo (N=1) the end spacing is
        # tau/2; for Ramsey (N=0) there are no echoes and the wait is tau.
        # The smallest sub-segment determines the subcycle warning.
        if self.sweep == 'wait':
            if self.num_echoes >= 1:
                min_seg = cfg.expt.step / (2 * self.num_echoes)
                seg_label = f"step/(2*echoes={self.num_echoes}) [CPMG]"
            else:
                min_seg = cfg.expt.step
                seg_label = "step [Ramsey]"
            warn_step_subcycle(
                self.soccfg, min_seg, gen_ch=None, label=seg_label,
            )
        print('Config expt:', cfg.expt)

        self.fock = int(cfg.expt.get('fock', 1))
        if self.fock != 1 and mode != 'manipulate':
            raise ValueError(f"fock={self.fock} is only supported for mode 'manipulate'")
        print(f"f0g1_freq_override: {cfg.expt.get('f0g1_freq', None)}")

        # --- Build state prep, Ramsey pulse, and determine channels ---
        if mode == 'storage':
            man_mode_no = cfg.expt.man_mode_no
            storage_mode_idx = cfg.expt.storage_mode_idx
            full_prep = self.prep_storage_cardinal('+', storage_mode_idx)
            # Last element is the storage sideband pi (the Ramsey interaction)
            self.auto_prep_str = full_prep[:-1]
            ramsey_str = [full_prep[-1]]
            self.auto_post_str = self.auto_prep_str[::-1]
            print(f"Auto prep string: {self.auto_prep_str}")

            self.creator = self.get_prepulse_creator(ramsey_str)
            freq = self.creator.pulse[0][0]
            self.flux_ch_ramsey = (
                self.flux_low_ch if freq < 1800 else self.flux_high_ch)

            phase_on_flux = cfg.expt.get('phase_on_flux', True)
            self.phase_update_channel = (
                self.flux_ch_ramsey if phase_on_flux else self.f0g1_ch)

        elif mode == 'manipulate':
            man_mode_idx = cfg.expt.get('man_mode_idx', 1)
            full_prep = self.prep_man_fock_state(
                man_no=man_mode_idx, state='+', broadband=True, n=self.fock)
            self.auto_prep_str = full_prep[:-1]
            ramsey_str = [full_prep[-1]]       # f0-g1, or f1-g2 for fock=2
            self.auto_post_str = self.auto_prep_str[::-1]

            with self._f0g1_override():
                self.creator = self.get_prepulse_creator(ramsey_str)
            print(f"Ramsey pulse: {self.creator.pulse}")
            self.phase_update_channel = self.f0g1_ch

        elif mode == 'coupler':
            coupler_pulse = cfg.expt.coupler_pulse
            freq = coupler_pulse[0][0]
            self.flux_ch_ramsey = (
                self.flux_low_ch if freq < 1800 else self.flux_high_ch)
            self.phase_update_channel = self.flux_ch_ramsey
            # No auto pre/post for coupler — user must specify
            self.auto_prep_str = []
            self.auto_post_str = []

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                "Use 'storage', 'manipulate', or 'coupler'.")

        # Compile auto pre/post pulses (for fock=2 they contain f0-g1)
        with self._f0g1_override():
            self.auto_prep_pulse = (self.get_prepulse_creator(self.auto_prep_str).pulse.tolist()
                                    if self.auto_prep_str else None)
            self.auto_post_pulse = (self.get_prepulse_creator(self.auto_post_str).pulse.tolist()
                                    if self.auto_post_str else None)

        # Handle custom pre/post overrides
        self.use_custom_prep = cfg.expt.get('prepulse', False)
        self.use_custom_post = cfg.expt.get('postpulse', False)
        if self.use_custom_prep:
            print("NOTE: adding extra custom prep! Auto state prep may or may not also be playing")
            # print('WARNING: Using custom prepulse instead of automatic '
            #       f'state preparation for mode={mode}')
        if self.use_custom_post:
            print("NOTE: adding extra custom post! Auto state teardown may or may not also be playing")
            # print('WARNING: Using custom postpulse instead of automatic '
            #       f'state teardown for mode={mode}')

        self.phase_update_page = self.ch_page(
            self.phase_update_channel[qTest])

        # --- Middle pulse: an operation played after the wait, before the
        # second half-pulse (e.g. swap round trips). The second half reuses
        # the registers of its channel, so the middle pulse must not use it.
        self.mid_pulse = None
        if cfg.expt.get('mid_pulse'):
            if cfg.expt.get('mid_gate_based', True):
                with self._f0g1_override():
                    self.mid_pulse = self.get_prepulse_creator(cfg.expt.mid_pulse).pulse.tolist()
            else:
                self.mid_pulse = cfg.expt.mid_pulse
            if mode in ('storage', 'manipulate'):
                ramsey_ch = self.creator.pulse[4][0]
                if any(ch == ramsey_ch for ch in self.mid_pulse[4]):
                    raise ValueError(f"mid_pulse uses the Ramsey channel {ramsey_ch}; "
                                     "the second half-pulse would play its registers")

        # --- Echo pulse construction ---
        if self.num_echoes > 0:
            if mode == 'coupler':
                raise ValueError(
                    f"Echoes not supported for mode '{mode}'")
            # full_prep is defined for 'storage' and 'manipulate' above.
            # Decode, flip the qubit (the two hpi), encode: a pi on the
            # 0 <-> fock pair. It contains the f0-g1 sideband, so it is
            # compiled with the same f0g1_freq override as the Ramsey pulse.
            echo_str = full_prep[::-1] + full_prep
            with self._f0g1_override():
                self.echo_pulse = self.get_prepulse_creator(
                    echo_str).pulse.tolist()

        # --- Sweep registers ---
        # r_wait     = end-segment (between pi/2 and first/last echo) = tau/(2N)
        # r_wait_mid = inter-pi-pulse segment (between consecutive echoes) = tau/N
        # For Ramsey (N=0) only r_wait is used, holding the full tau.
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase_step = 5
        self.r_wait_mid = 6
        self.r_flux_phase = self.sreg(
            self.phase_update_channel[qTest], "phase")

        if self.num_echoes >= 1:
            self.safe_regwi(self.phase_update_page, self.r_wait,
                            self.us2cycles(cfg.expt.start / (2 * self.num_echoes)))
            self.safe_regwi(self.phase_update_page, self.r_wait_mid,
                            self.us2cycles(cfg.expt.start / self.num_echoes))
        else:
            self.safe_regwi(self.phase_update_page, self.r_wait,
                            self.us2cycles(cfg.expt.start))
        # Phase of the second Ramsey pulse at the first wait: phase_offset
        # (deg; phase_cycle sets it to offset and offset + 180).
        self.safe_regwi(self.phase_update_page, self.r_phase2, self.deg2reg(
            float(cfg.expt.get('phase_offset', 0.0)) % 360,
            gen_ch=self.phase_update_channel[qTest]))

        # Phase step register (register-register math avoids mathi limit).
        # sweep='wait': 360 * ramsey_freq * step per point; sweep='phase':
        # phase_step [deg] per point (default 360 / expts), fixed wait.
        if self.sweep == 'phase':
            step_deg = float(cfg.expt.get('phase_step', 360.0 / cfg.expt.expts))
            self.ramsey_freq_sign = 1 if step_deg >= 0 else -1
        else:
            step_deg = 360 * cfg.expt.ramsey_freq * cfg.expt.step
            self.ramsey_freq_sign = 1 if cfg.expt.ramsey_freq >= 0 else -1
        phase_step_val = self.deg2reg(abs(step_deg) % 360, gen_ch=self.phase_update_channel[qTest])
        self.safe_regwi(
            self.phase_update_page, self.r_phase_step, phase_step_val)

        # --- Parity measurement pulse ---
        man_mode_no = cfg.expt.get('man_mode_no', 1)
        self.parity_meas_pulse = self.get_parity_str(
            man_mode_no, return_pulse=True, second_phase=180, fast=False)

        self.sync_all(200)

    @contextmanager
    def _f0g1_override(self):
        """Compile with expt.f0g1_freq as the f0-g1 sideband frequency (e.g. for
        a per-current calibration). Manipulate mode only; every compilation
        that can contain f0-g1 (Ramsey pulse, auto pre/post, echo) uses it."""
        freq = self.cfg.expt.get('f0g1_freq', None) if self.mode == 'manipulate' else None
        if freq is None:
            yield
            return
        frequencies = self.cfg.device.multiphoton['pi']['fn-gn+1']['frequency']
        saved = frequencies[0]
        frequencies[0] = float(freq)
        try:
            yield
        finally:
            frequencies[0] = saved

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        self.reset_and_sync()

        # Active reset
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # Pre-pulse: custom
        if self.use_custom_prep:
            if cfg.expt.get('gate_based', True):
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(
                    cfg, creator.pulse.tolist(), prefix='pre_')
                print("playing the custom pulse now!", creator.pulse.tolist())
            else:
                self.custom_pulse(
                    cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

        # Pre-pulse: auto --> applies by default unless self.cfg.expt.use_auto_prep=False
        if self.cfg.expt.get('use_auto_prep', True):
            # I don't know how this can even be None tbh since the auto_prep_str is always calculated
            if self.auto_prep_pulse is not None:
                self.custom_pulse(cfg, self.auto_prep_pulse, prefix='auto_prep_')

        # First Ramsey half-pulse
        if self.mode in ('storage', 'manipulate'):
            self.custom_pulse(
                self.cfg, self.creator.pulse,
                prefix=f'{self.mode}_ramsey_')
            self.sync_all(self.us2cycles(0.01))
        elif self.mode == 'coupler':
            self.custom_pulse(
                cfg, cfg.expt.coupler_pulse, prefix='coupler_')
            self.sync_all(self.us2cycles(0.01))

        # Wait
        self.sync_all()
        self.sync(self.phase_update_page, self.r_wait)
        self.sync_all()

        # CPMG echo pulses
        # Layout: prep -> sync(r_wait) -> echo_1 -> sync(r_wait_mid) -> echo_2
        #         -> ... -> sync(r_wait_mid) -> echo_N -> sync(r_wait) -> second-half.
        if self.num_echoes > 0:
            for k in range(self.num_echoes):
                self.custom_pulse(cfg, self.echo_pulse, prefix='echo_')
                self.sync_all()
                is_last = (k == self.num_echoes - 1)
                wait_reg = self.r_wait if is_last else self.r_wait_mid
                self.sync(self.phase_update_page, wait_reg)
                self.sync_all()

        # Middle pulse (after the wait, before the second half-pulse)
        if self.mid_pulse is not None:
            self.custom_pulse(cfg, self.mid_pulse, prefix='mid_')
            self.sync_all()

        # Apply Ramsey phase and second half-pulse
        self.mathi(self.phase_update_page, self.r_flux_phase,
                    self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))

        if self.mode in ('storage', 'coupler'):
            self.pulse(ch=self.flux_ch_ramsey[qTest])
        elif self.mode == 'manipulate':
            self.pulse(ch=self.f0g1_ch[qTest])
        self.sync_all(self.us2cycles(0.01))

        # Post-pulse: auto or custom
        self.sync_all()
        if self.use_custom_post:
            if cfg.expt.get('gate_based', True):
                creator = self.get_prepulse_creator(cfg.expt.post_sweep_pulse)
                self.custom_pulse(
                    cfg, creator.pulse.tolist(), prefix='post_')
            else:
                self.custom_pulse(
                    cfg, cfg.expt.post_sweep_pulse, prefix='post_')
        elif self.auto_post_pulse is not None:
            self.custom_pulse(cfg, self.auto_post_pulse, prefix='auto_post_')

        # Parity measurement
        if cfg.expt.get('parity_meas', True):
            self.custom_pulse(
                self.cfg, self.parity_meas_pulse, prefix='parity_')

        self.measure_wrapper()

    def update(self):
        if self.sweep == 'phase':
            pass                               # fixed wait: only the phase steps
        elif self.num_echoes >= 1:
            step_end_cycles = self.us2cycles(
                self.cfg.expt.step / (2 * self.num_echoes))
            step_mid_cycles = self.us2cycles(
                self.cfg.expt.step / self.num_echoes)
            self.mathi(self.phase_update_page, self.r_wait,
                       self.r_wait, '+', step_end_cycles)
            self.sync_all(self.us2cycles(0.01))
            self.mathi(self.phase_update_page, self.r_wait_mid,
                       self.r_wait_mid, '+', step_mid_cycles)
        else:
            step_cycles = self.us2cycles(self.cfg.expt.step)
            self.mathi(self.phase_update_page, self.r_wait,
                       self.r_wait, '+', step_cycles)
        self.sync_all(self.us2cycles(0.01))
        op = '+' if self.ramsey_freq_sign >= 0 else '-'
        self.math(self.phase_update_page, self.r_phase2,
                  self.r_phase2, op, self.r_phase_step)
        self.sync_all(self.us2cycles(0.01))


class CavityModeRamseyExperiment(Experiment):
    """
    Cavity Mode Ramsey / Echo Experiment

    Experimental Config:
    expt = dict(
        start: wait time start [us]
        step: wait time step [us]
        expts: number of sweep points
        ramsey_freq: virtual detuning frequency [MHz]
        reps: averages per point
        rounds: number of rounds (default 1)

        mode: 'storage', 'manipulate', or 'coupler'

        # for storage mode:
        storage_mode_idx: storage mode index (e.g. 1)
        man_mode_no: manipulate mode to go through (e.g. 1)
        phase_on_flux: if True, phase update on flux ch (default True)

        # for manipulate mode:
        man_mode_idx: manipulate mode index (e.g. 1)
        fock: upper Fock level of the superposition, 1 (|0>+|1>, default)
            or 2 (|0>+|2>; Ramsey pulse f1-g2)
        f0g1_freq: optional f0-g1 frequency override [MHz]

        # for coupler mode:
        coupler_pulse: list of pulse descriptors

        echoes: number of echo pulses (0 = Ramsey, >=1 = CPMG echo).
            Timing is standard CPMG (Bylander 2011 Eq. 18): end-spacings
            tau/(2N), inter-pi spacings tau/N.
        parity_meas: True to use parity measurement (default True)
        phase_offset: phase of the second Ramsey pulse at the first wait
            [deg] (default 0)
        phase_cycle: True to measure at phase_offset and phase_offset + 180
            and analyse half the difference (default False); the raw traces
            are kept as avgi_0 / avgi_180, avgq_0 / avgq_180

        sweep: 'wait' (default) or 'phase'. 'phase': the wait stays at
            start, the phase of the second half steps by phase_step [deg]
            (default 360 / expts) from phase_offset; xpts is that phase.
            analyze() then fits B + A cos(phase - phase0): A (fringe_amp) is
            proportional to the coherence, phase0 (fringe_phase_deg) is the
            phase the sequence adds. Not with phase_cycle.
        mid_pulse: optional pulse sequence played after the wait, before the
            second half-pulse (e.g. swap round trips); it must not use the
            Ramsey channel. mid_gate_based: True (default) for gate-based
            format, False for a compiled 7-row pulse array.

        # optional custom pre/post (overrides auto state prep with warning):
        prepulse: True/False (default False)
        pre_sweep_pulse: pulse sequence
        postpulse: True/False (default False)
        post_sweep_pulse: pulse sequence
        gate_based: True for gate-based format (default True)

        active_reset: True/False
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CavityModeRamsey',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit,
                        self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not isinstance(value3, list):
                                value2.update(
                                    {key3: [value3] * num_qubits_sample})
                elif not isinstance(value, list):
                    subcfg.update({key: [value] * num_qubits_sample})

        read_num = 1
        if self.cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        phase_sweep = self.cfg.expt.get('sweep', 'wait') == 'phase'
        if phase_sweep and self.cfg.expt.get('phase_cycle', False):
            raise ValueError("sweep='phase' already sweeps the phase; do not combine it with phase_cycle")
        if not self.cfg.expt.get('phase_cycle', False):
            x_pts, avgi, avgq, shots = self._acquire_once(self.cfg, read_num, progress)
            if phase_sweep:
                step = float(self.cfg.expt.get('phase_step', 360.0 / self.cfg.expt.expts))
                x_pts = float(self.cfg.expt.get('phase_offset', 0.0)) + step * np.arange(len(avgi))
            data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq}
            data['idata'], data['qdata'] = shots
        else:
            offset = float(self.cfg.expt.get('phase_offset', 0.0))
            data = {}
            for tag, phase in (('0', offset), ('180', offset + 180.0)):
                cfg = deepcopy(self.cfg)
                cfg.expt.phase_offset = phase
                x_pts, data[f'avgi_{tag}'], data[f'avgq_{tag}'], shots = self._acquire_once(cfg, read_num, progress)
                data[f'idata_{tag}'], data[f'qdata_{tag}'] = shots
            data['xpts'] = x_pts
            data['avgi'] = (data['avgi_0'] - data['avgi_180']) / 2
            data['avgq'] = (data['avgq_0'] - data['avgq_180']) / 2
        data['amps'] = np.abs(data['avgi'] + 1j * data['avgq'])
        data['phases'] = np.angle(data['avgi'] + 1j * data['avgq'])
        self.data = data
        return data

    def _acquire_once(self, cfg, read_num, progress):
        """One program over all waits: (xpts, avgi, avgq, (idata, qdata))."""
        ramsey = CavityModeRamseyProgram(soccfg=self.soccfg, cfg=cfg)
        x_pts, avgi, avgq = ramsey.acquire(
            self.im[cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            readouts_per_experiment=read_num)
        self.prog = ramsey
        return x_pts, avgi[0][-1], avgq[0][-1], ramsey.collect_shots()

    def _fitting_cfg(self):
        """Return a config copy with fields that RamseyFitting expects."""
        from copy import deepcopy
        cfg = deepcopy(self.cfg)
        # xpts already represents total wait time (program divides per-segment
        # internally), so tell RamseyFitting NOT to multiply xpts by (1+N).
        cfg.expt.echoes = [False, 0]
        # RamseyFitting.display() reads these qubit-ramsey fields
        cfg.expt.setdefault('checkEF', False)
        cfg.expt.setdefault('f0g1_cavity', 0)
        return cfg

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data = self.data
        if not fit:
            return data
        if self.cfg.expt.get('sweep', 'wait') == 'phase':
            data.update(fit_fringe(data['xpts'], data['avgi']))
            return data
        analysis = RamseyFitting(data, config=self._fitting_cfg())
        analysis.analyze(fitparams=fitparams)
        return analysis.data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        if self.cfg.expt.get('sweep', 'wait') == 'phase':
            phase, y = np.asarray(data['xpts'], float), np.asarray(data['avgi'], float)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(phase, y, 'o', ms=4, label='data')
            if fit and 'fringe_amp' in data:
                pp = np.linspace(phase[0], phase[-1], 400)
                ax.plot(pp, fringe(pp, *data['fit_avgi']), '-',
                        label=f"A = {data['fringe_amp']:.2f} +/- {data['fringe_amp_err']:.2f}, "
                              f"phase0 = {data['fringe_phase_deg']:.0f} deg")
            ax.set_xlabel('phase of the second half (deg)')
            ax.set_ylabel('I (ADC)')
            ax.set_title(f"{self.cfg.expt.mode} phase sweep (wait {self.cfg.expt.start} us)")
            ax.legend()
            fig.tight_layout()
            plt.show()
            return fig, ax
        analysis = RamseyFitting(data, config=self._fitting_cfg())
        analysis.display()

    def save_data(self, data=None):
        if data is None:
            data = self.data
        print(f'Saving {self.fname}')
        super().save_data(data=h5_safe_data(data))
        return self.fname
