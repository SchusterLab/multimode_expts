"""
Cavity Mode Ramsey / Echo Experiment

Measures T2 coherence of a cavity mode (storage, manipulate, or coupler) by
preparing |0>+|1> in the mode, waiting a variable time, and applying the
second half of the gate with a Ramsey phase.

State preparation is automatic based on the mode:
  - storage: qubit_hpi + ef_pi + man_pi → storage_pi (Ramsey) → wait → ...
  - manipulate: g0-e0_hpi + e0-f0_pi → f0-g1_pi (Ramsey) → wait → ...
  - coupler: custom pulse (user must specify coupler_pulse)

After the Ramsey interaction, the reverse of the state prep brings the
state back for parity readout.

Custom pre/postpulses override the automatic ones (with a warning).

Refactored from t2_cavity.py — Feb 2026
"""

import numpy as np
from qick import *
from slab import AttrDict, Experiment

from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram
from fitting.fit_display_classes import RamseyFitting


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
                man_no=man_mode_idx, state='+', broadband=True)
            self.auto_prep_str = full_prep[:-1]
            ramsey_str = [full_prep[-1]]
            self.auto_post_str = self.auto_prep_str[::-1]

            self.creator = self.get_prepulse_creator(ramsey_str)
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

        # Compile auto pre/post pulses
        if self.auto_prep_str:
            creator = self.get_prepulse_creator(self.auto_prep_str)
            self.auto_prep_pulse = creator.pulse.tolist()
        else:
            self.auto_prep_pulse = None

        if self.auto_post_str:
            creator = self.get_prepulse_creator(self.auto_post_str)
            self.auto_post_pulse = creator.pulse.tolist()
        else:
            self.auto_post_pulse = None

        # Handle custom pre/post overrides
        self.use_custom_prep = cfg.expt.get('prepulse', False)
        self.use_custom_post = cfg.expt.get('postpulse', False)
        if self.use_custom_prep:
            print('WARNING: Using custom prepulse instead of automatic '
                  f'state preparation for mode={mode}')
        if self.use_custom_post:
            print('WARNING: Using custom postpulse instead of automatic '
                  f'state teardown for mode={mode}')

        self.phase_update_page = self.ch_page(
            self.phase_update_channel[qTest])

        # --- Echo pulse construction ---
        if self.num_echoes > 0:
            if mode == 'coupler':
                raise ValueError(
                    f"Echoes not supported for mode '{mode}'")
            # full_prep is defined for 'storage' and 'manipulate' above
            echo_str = full_prep[::-1] + full_prep
            self.echo_pulse = self.get_prepulse_creator(
                echo_str).pulse.tolist()

        # --- Sweep registers ---
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase_step = 5
        self.r_flux_phase = self.sreg(
            self.phase_update_channel[qTest], "phase")

        n_wait_segments = 1 + self.num_echoes
        self.safe_regwi(self.phase_update_page, self.r_wait,
                        self.us2cycles(cfg.expt.start / n_wait_segments))
        self.safe_regwi(self.phase_update_page, self.r_phase2, 0)

        # Phase step register (register-register math avoids mathi limit)
        phase_step_val = self.deg2reg(
            360 * abs(cfg.expt.ramsey_freq) * cfg.expt.step,
            gen_ch=self.phase_update_channel[qTest])
        self.safe_regwi(
            self.phase_update_page, self.r_phase_step, phase_step_val)
        self.ramsey_freq_sign = 1 if cfg.expt.ramsey_freq >= 0 else -1

        # --- Parity measurement pulse ---
        man_mode_no = cfg.expt.get('man_mode_no', 1)
        self.parity_meas_pulse = self.get_parity_str(
            man_mode_no, return_pulse=True, second_phase=180, fast=False)

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        self.reset_and_sync()

        # Active reset
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # Pre-pulse: auto or custom
        if self.use_custom_prep:
            if cfg.expt.get('gate_based', True):
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(
                    cfg, creator.pulse.tolist(), prefix='pre_')
            else:
                self.custom_pulse(
                    cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')
        elif self.auto_prep_pulse is not None:
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

        # Echo pulses
        if self.num_echoes > 0:
            for _ in range(self.num_echoes):
                self.custom_pulse(cfg, self.echo_pulse, prefix='echo_')
                self.sync_all()
                self.sync(self.phase_update_page, self.r_wait)
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
        n_wait_segments = 1 + self.num_echoes
        step_cycles = self.us2cycles(self.cfg.expt.step / n_wait_segments)
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

        # for coupler mode:
        coupler_pulse: list of pulse descriptors

        echoes: number of echo pulses (0 = Ramsey, >0 = Echo)
        parity_meas: True to use parity measurement (default True)

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

        ramsey = CavityModeRamseyProgram(soccfg=self.soccfg, cfg=self.cfg)

        x_pts, avgi, avgq = ramsey.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            readouts_per_experiment=read_num)

        avgi = avgi[0][-1]
        avgq = avgq[0][-1]
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {
            'xpts': x_pts,
            'avgi': avgi,
            'avgq': avgq,
            'amps': amps,
            'phases': phases,
        }
        data['idata'], data['qdata'] = ramsey.collect_shots()
        self.data = data
        return data

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
        if fit:
            analysis = RamseyFitting(data, config=self._fitting_cfg())
            analysis.analyze(fitparams=fitparams)
            return analysis.data
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        analysis = RamseyFitting(data, config=self._fitting_cfg())
        analysis.display()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
