"""
Cavity Displacement Ramsey Experiment

Measures T2 coherence after displacing a cavity mode with a user-defined
pulse (Gaussian or flat-top), waiting a variable time, then applying the
second pulse with a Ramsey phase and reading out via a slow pi pulse on
the qubit.

Pulse sequence:
1. (Optional) Active reset
2. (Optional) Prepulse
3. (Optional) Prepare qubit in |e> before displacement
4. Displacement pulse (Gaussian or flat-top) on selected channel
5. Variable wait time (swept in firmware)
6. Second displacement pulse with Ramsey phase
7. (Optional) Postpulse
8. Slow pi_ge pulse on qubit (maps cavity state to qubit for readout)
9. Measure

Also includes CavityDisplacementGainSweepExperiment which sweeps the
displacement gain as an outer loop.

Refactored from t2_cavity.py — Feb 2026
"""

from copy import deepcopy

import numpy as np
from qick import *
from slab import AttrDict, Experiment
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram
from fitting.fit_display_classes import CavityRamseyGainSweepFitting, RamseyFitting

# Channel name → (channel attribute, channel type attribute)
CHANNEL_MAP = {
    'f0g1': ('f0g1_ch', 'f0g1_ch_type'),
    'flux_low': ('flux_low_ch', 'flux_low_ch_type'),
    'qubit': ('qubit_chs', 'qubit_ch_types'),
    'flux_high': ('flux_high_ch', 'flux_high_ch_type'),
    'man': ('man_ch', 'man_ch_type'),
    'storage': ('storage_ch', 'storage_ch_type'),
}


class CavityDisplacementRamseyProgram(MMRAveragerProgram):
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

        # --- Resolve displacement channel from name ---
        channel_name = cfg.expt.channel
        if channel_name not in CHANNEL_MAP:
            raise ValueError(
                f"Unknown channel '{channel_name}'. "
                f"Use one of: {list(CHANNEL_MAP.keys())}")
        ch_attr, ch_type_attr = CHANNEL_MAP[channel_name]
        self.disp_ch = getattr(self, ch_attr)
        self.disp_ch_type = getattr(self, ch_type_attr)

        self.phase_update_channel = self.disp_ch
        self.phase_update_page = self.ch_page(
            self.phase_update_channel[qTest])

        # --- Displacement pulse parameters ---
        self.disp_freq = self.freq2reg(
            cfg.expt.disp_freq, gen_ch=self.disp_ch[qTest])
        self.disp_gain = cfg.expt.disp_gain
        self.disp_sigma = self.us2cycles(
            cfg.expt.disp_sigma, gen_ch=self.disp_ch[qTest])
        self.disp_length = self.us2cycles(
            cfg.expt.get('disp_length', 0), gen_ch=self.disp_ch[qTest])

        self.add_gauss(
            ch=self.disp_ch[qTest], name="disp_gauss",
            sigma=self.disp_sigma, length=self.disp_sigma * 4)

        # --- Slow pi_ge pulse for readout ---
        slow_pi = cfg.device.qubit.pulses.slow_pi_ge
        self.slow_pi_freq = self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])
        self.slow_pi_gain = slow_pi.gain[qTest]
        self.slow_pi_style = slow_pi.type[qTest]
        sigma_cycles = self.us2cycles(
            slow_pi.sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.slow_pi_length = self.us2cycles(
            slow_pi.length[qTest], gen_ch=self.qubit_chs[qTest])
        self.add_gauss(
            ch=self.qubit_chs[qTest], name="slow_pi_ge",
            sigma=sigma_cycles, length=sigma_cycles * 4)

        # --- Sweep registers ---
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase_step = 5
        self.r_disp_phase = self.sreg(
            self.phase_update_channel[qTest], "phase")

        self.safe_regwi(self.phase_update_page, self.r_wait,
                        self.us2cycles(cfg.expt.start))
        self.safe_regwi(self.phase_update_page, self.r_phase2, 0)

        # Phase step register (register-register math avoids mathi limit)
        phase_step_val = self.deg2reg(
            360 * abs(cfg.expt.ramsey_freq) * cfg.expt.step,
            gen_ch=self.phase_update_channel[qTest])
        self.safe_regwi(
            self.phase_update_page, self.r_phase_step, phase_step_val)
        self.ramsey_freq_sign = 1 if cfg.expt.ramsey_freq >= 0 else -1

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        self.reset_and_sync()

        # Active reset
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # Prepulse
        if cfg.expt.get('prepulse', False):
            if cfg.expt.get('gate_based', True):
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(
                    cfg, creator.pulse.tolist(), prefix='pre_')
            else:
                self.custom_pulse(
                    cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

        # Optionally prepare qubit in |e> before displacement
        if cfg.expt.get('prep_e_first', False):
            prep_e = [['qubit', 'ge', 'pi', 0]]
            creator = self.get_prepulse_creator(prep_e)
            self.custom_pulse(cfg, creator.pulse.tolist(), prefix='prep_e_')

        # Displacement pulse
        if self.disp_length == 0:  # Gaussian
            self.setup_and_pulse(
                ch=self.disp_ch[qTest], style="arb",
                freq=self.disp_freq,
                phase=self.deg2reg(0, gen_ch=self.disp_ch[qTest]),
                gain=self.disp_gain,
                waveform="disp_gauss")
        else:  # Flat-top
            self.setup_and_pulse(
                ch=self.disp_ch[qTest], style="flat_top",
                freq=self.disp_freq,
                phase=0,
                gain=self.disp_gain,
                length=self.disp_length,
                waveform="disp_gauss")
        self.sync_all(self.us2cycles(0.01))

        # Wait
        self.sync_all()
        self.sync(self.phase_update_page, self.r_wait)
        self.sync_all()

        # Apply Ramsey phase and second displacement pulse
        self.mathi(self.phase_update_page, self.r_disp_phase,
                    self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))
        self.pulse(ch=self.disp_ch[qTest])
        self.sync_all(self.us2cycles(0.01))

        # Postpulse
        self.sync_all()
        if cfg.expt.get('postpulse', False):
            if cfg.expt.get('gate_based', True):
                creator = self.get_prepulse_creator(cfg.expt.post_sweep_pulse)
                self.custom_pulse(
                    cfg, creator.pulse.tolist(), prefix='post_')
            else:
                self.custom_pulse(
                    cfg, cfg.expt.post_sweep_pulse, prefix='post_')

        # Slow pi_ge readout pulse
        self.setup_and_pulse(
            ch=self.qubit_chs[qTest],
            style=self.slow_pi_style,
            freq=self.slow_pi_freq,
            phase=self.deg2reg(0, gen_ch=self.qubit_chs[qTest]),
            gain=self.slow_pi_gain,
            length=self.slow_pi_length,
            waveform="slow_pi_ge")

        self.measure_wrapper()

    def update(self):
        qTest = self.qubits[0]
        self.mathi(self.phase_update_page, self.r_wait,
                    self.r_wait, '+', self.us2cycles(self.cfg.expt.step))
        self.sync_all(self.us2cycles(0.01))
        op = '+' if self.ramsey_freq_sign >= 0 else '-'
        self.math(self.phase_update_page, self.r_phase2,
                  self.r_phase2, op, self.r_phase_step)
        self.sync_all(self.us2cycles(0.01))


class CavityDisplacementRamseyExperiment(Experiment):
    """
    Cavity Displacement Ramsey Experiment

    Experimental Config:
    expt = dict(
        start: wait time start [us]
        step: wait time step [us]
        expts: number of sweep points
        ramsey_freq: virtual detuning frequency [MHz]
        reps: averages per point
        rounds: number of rounds (default 1)

        channel: 'f0g1', 'flux_low', 'qubit', 'flux_high', 'man', or 'storage'
        disp_freq: displacement frequency [MHz]
        disp_gain: displacement gain [DAC units]
        disp_sigma: Gaussian sigma [us]
        disp_length: flat-top length [us] (0 = Gaussian only)

        prep_e_first: if True, prepare qubit in |e> before displacement

        active_reset: True/False
        prepulse: True/False
        pre_sweep_pulse: pulse sequence
        postpulse: True/False
        post_sweep_pulse: pulse sequence
        gate_based: True for gate-based format
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CavityDispRamsey',
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

        ramsey = CavityDisplacementRamseyProgram(
            soccfg=self.soccfg, cfg=self.cfg)

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

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data = self.data
        if fit:
            analysis = RamseyFitting(data, config=self.cfg)
            analysis.analyze(fitparams=fitparams)
            return analysis.data
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        analysis = RamseyFitting(data, config=self.cfg)
        analysis.display()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


class CavityDisplacementGainSweepExperiment(Experiment):
    """
    Sweeps displacement gain as outer loop, running a cavity displacement
    Ramsey at each gain value.

    Additional Config (on top of CavityDisplacementRamseyExperiment):
    expt = dict(
        gain_start: starting displacement gain
        gain_step: gain increment
        gain_expts: number of gain values
        do_g_and_e: if True, also measure with qubit prepared in |e>
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CavityDispGainSweep',
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

        gain_start = self.cfg.expt.gain_start
        gain_step = self.cfg.expt.gain_step
        gain_expts = self.cfg.expt.gain_expts
        gain_list = np.array(
            [gain_start + i * gain_step for i in range(gain_expts)])
        do_g_and_e = self.cfg.expt.get('do_g_and_e', False)

        n_time_pts = self.cfg.expt.expts
        data = {
            'gain_list': gain_list,
            'xpts': np.zeros((len(gain_list), n_time_pts)),
            'g_avgi': np.zeros((len(gain_list), n_time_pts)),
            'g_avgq': np.zeros((len(gain_list), n_time_pts)),
            'g_amps': np.zeros((len(gain_list), n_time_pts)),
            'g_phases': np.zeros((len(gain_list), n_time_pts)),
            'e_avgi': np.zeros((len(gain_list), n_time_pts)),
            'e_avgq': np.zeros((len(gain_list), n_time_pts)),
            'e_amps': np.zeros((len(gain_list), n_time_pts)),
            'e_phases': np.zeros((len(gain_list), n_time_pts)),
        }

        for i_gain, gain in enumerate(tqdm(gain_list, disable=not progress)):
            # Use a copy to avoid mutating the original config
            sweep_cfg = deepcopy(self.cfg)
            sweep_cfg.expt.disp_gain = int(gain)
            sweep_cfg.expt.prep_e_first = False

            ramsey = CavityDisplacementRamseyProgram(
                soccfg=self.soccfg, cfg=sweep_cfg)
            x_pts, avgi, avgq = ramsey.acquire(
                soc=self.im[self.cfg.aliases.soc],
                threshold=None, load_pulses=True, progress=False,
                readouts_per_experiment=read_num)

            avgi = avgi[0][0]
            avgq = avgq[0][0]
            data['xpts'][i_gain] = x_pts
            data['g_avgi'][i_gain] = avgi
            data['g_avgq'][i_gain] = avgq
            data['g_amps'][i_gain] = np.abs(avgi + 1j * avgq)
            data['g_phases'][i_gain] = np.angle(avgi + 1j * avgq)

            if do_g_and_e:
                sweep_cfg.expt.prep_e_first = True
                ramsey = CavityDisplacementRamseyProgram(
                    soccfg=self.soccfg, cfg=sweep_cfg)
                x_pts, avgi, avgq = ramsey.acquire(
                    soc=self.im[self.cfg.aliases.soc],
                    threshold=None, load_pulses=True, progress=False,
                    readouts_per_experiment=read_num)

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                data['e_avgi'][i_gain] = avgi
                data['e_avgq'][i_gain] = avgq
                data['e_amps'][i_gain] = np.abs(avgi + 1j * avgq)
                data['e_phases'][i_gain] = np.angle(avgi + 1j * avgq)

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        if fit:
            analysis = CavityRamseyGainSweepFitting(
                data, config=self.cfg)
            analysis.analyze(fit=fit, **kwargs)
            return analysis.data
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        analysis = CavityRamseyGainSweepFitting(
            data, config=self.cfg)
        save_fig = kwargs.pop('save_fig', False)
        analysis.display(save_fig=save_fig, **kwargs)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
