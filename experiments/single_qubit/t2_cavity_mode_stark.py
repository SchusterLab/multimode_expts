"""
Cavity Mode Ramsey / Echo Experiment with Stark Drive on Manipulate

Same as t2_cavity_mode.py but each sync-based wait between the two effective
pi/2 pulses (and between each echo segment) is replaced by a flat-top drive
on the manipulate DAC. The drive exactly fills the wait: flat plateau =
wait_per_segment - 2*rise_time, plus 2*rise_time of Gaussian ramps (rise_time
per edge).

Swept parameter: the total wait time between the two half-pulses. The sweep
is a Python-level loop at the Experiment layer (software sweep) — one
program per point. This is required because QICK v1's sync/timestamp
bookkeeping is done at Python compile time from each pulse's declared
length; firmware-side mathi on the flat-top length register changes the DAC
pulse duration but not the scheduled time of subsequent pulses, so an MMR
firmware sweep of the pulse length does not produce the expected Ramsey
evolution. The same software-sweep pattern is used by
KerrCavityRamseyExcursionProgram and length_rabi_f0g1_general.

User-set drive parameters (fixed across the sweep):
  - drive_freq  [MHz]
  - drive_gain  [DAC units]
  - rise_time   [us]  per-edge ramp duration; Gaussian sigma = rise_time / 2

Guard: cfg.expt.start >= 2 * cfg.expt.rise_time (per segment) — enforced
in the Experiment class to ensure a non-negative flat plateau.
"""

import numpy as np
from qick import *
from slab import AttrDict, Experiment
from tqdm.notebook import tqdm

from experiments.MM_base import MM_base, MMAveragerProgram, warn_step_subcycle
from fitting.fit_display_classes import RamseyFitting
from fitting.decaysin_analysis import h5_safe_data


class CavityModeStarkProgram(MMAveragerProgram):
    """One sweep point: fixed wait_time (us), fixed ramsey_phase (deg)."""

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

        mode = cfg.expt.mode
        self.mode = mode
        self.num_echoes = cfg.expt.get('echoes', 0)

        # --- Build state prep, Ramsey pulse, and determine channels ---
        if mode == 'storage':
            storage_mode_idx = cfg.expt.storage_mode_idx
            full_prep = self.prep_storage_cardinal('+', storage_mode_idx)
            self.auto_prep_str = full_prep[:-1]
            ramsey_str = [full_prep[-1]]
            self.auto_post_str = self.auto_prep_str[::-1]
            print(f"Auto prep string: {self.auto_prep_str}")

            self.creator = self.get_prepulse_creator(ramsey_str)
            freq = self.creator.pulse[0][0]
            self.flux_ch_ramsey = (
                self.flux_low_ch if freq < 1800 else self.flux_high_ch)

        elif mode == 'manipulate':
            man_mode_idx = cfg.expt.get('man_mode_idx', 1)
            full_prep = self.prep_man_fock_state(
                man_no=man_mode_idx, state='+', broadband=True)
            self.auto_prep_str = full_prep[:-1]
            ramsey_str = [full_prep[-1]]
            self.auto_post_str = self.auto_prep_str[::-1]
            self.creator = self.get_prepulse_creator(ramsey_str)

        elif mode == 'coupler':
            coupler_pulse = cfg.expt.coupler_pulse
            freq = coupler_pulse[0][0]
            self.flux_ch_ramsey = (
                self.flux_low_ch if freq < 1800 else self.flux_high_ch)
            self.auto_prep_str = []
            self.auto_post_str = []
            self.creator = None

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                "Use 'storage', 'manipulate', or 'coupler'.")

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

        self.use_custom_prep = cfg.expt.get('prepulse', False)
        self.use_custom_post = cfg.expt.get('postpulse', False)
        if self.use_custom_prep:
            print("NOTE: adding extra custom prep! Auto state prep may or may not also be playing")
        if self.use_custom_post:
            print("NOTE: adding extra custom post! Auto state teardown may or may not also be playing")

        # Echo pulse (same train in each echo slot)
        if self.num_echoes > 0:
            if mode == 'coupler':
                raise ValueError(
                    f"Echoes not supported for mode '{mode}'")
            echo_str = full_prep[::-1] + full_prep
            self.echo_pulse = self.get_prepulse_creator(
                echo_str).pulse.tolist()

        # --- Build first and second half-pulse descriptors ---
        # The first half-pulse uses the Ramsey descriptor as-is (phase 0).
        # The second half-pulse uses a copy with phase = ramsey_phase (deg).
        if mode in ('storage', 'manipulate'):
            # self.creator.pulse is a numpy structured-like array; convert to
            # a nested-list form custom_pulse accepts.
            first_pulse = [list(row) for row in self.creator.pulse]
            second_pulse = [list(row) for row in self.creator.pulse]
            second_pulse[3] = [cfg.expt.ramsey_phase
                               for _ in second_pulse[3]]
            self.first_half_pulse = first_pulse
            self.second_half_pulse = second_pulse
        else:  # coupler
            first_pulse = [list(row) for row in np.atleast_2d(
                np.array(cfg.expt.coupler_pulse, dtype=object).T).T]
            # Simpler: just rebuild as list-of-lists matching the expected shape.
            first_pulse = [list(x) if isinstance(x, (list, tuple, np.ndarray))
                           else [x] for x in cfg.expt.coupler_pulse]
            second_pulse = [list(x) for x in first_pulse]
            second_pulse[3] = [cfg.expt.ramsey_phase
                               for _ in second_pulse[3]]
            self.first_half_pulse = first_pulse
            self.second_half_pulse = second_pulse

        # --- Build the Stark drive via the MM_base long-pulse helpers ---
        # The drive is played as arb rise -> N back-to-back const chunks ->
        # arb fall on the manipulate DAC during each wait segment.
        n_wait_segments = 1 + self.num_echoes
        wait_per_seg_us = cfg.expt.wait_time / n_wait_segments
        rise_total_us = 2 * cfg.expt.rise_time  # = 4 * sigma_us, full ramps
        self.stark_flat_us = wait_per_seg_us - rise_total_us
        assert self.stark_flat_us >= 0, (
            f"wait_time per segment ({wait_per_seg_us} us) must be >= "
            f"2*rise_time ({rise_total_us} us)")

        self.stark_handle = self.register_long_pulse(
            ch=self.man_ch[qTest],
            freq_MHz=cfg.expt.drive_freq,
            gain=cfg.expt.drive_gain,
            rise_time_us=cfg.expt.rise_time,
            name='stark_drive')

        # --- Parity measurement pulse ---
        man_mode_no = cfg.expt.get('man_mode_no', 1)
        self.parity_meas_pulse = self.get_parity_str(
            man_mode_no, return_pulse=True, second_phase=180, fast=False)

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)

        self.reset_and_sync()

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        if self.use_custom_prep:
            if cfg.expt.get('gate_based', True):
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(
                    cfg, creator.pulse.tolist(), prefix='pre_')
            else:
                self.custom_pulse(
                    cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

        if cfg.expt.get('use_auto_prep', True):
            if self.auto_prep_pulse is not None:
                self.custom_pulse(
                    cfg, self.auto_prep_pulse, prefix='auto_prep_')

        # First Ramsey half-pulse
        self.custom_pulse(cfg, self.first_half_pulse, prefix='ramsey_first_')
        self.sync_all(self.us2cycles(0.01))

        # Stark drive — segment 1 (replaces the first sync-wait)
        self.play_long_pulse(self.stark_handle, self.stark_flat_us)

        # Echo pulses, each followed by another stark drive segment
        if self.num_echoes > 0:
            for i in range(self.num_echoes):
                self.custom_pulse(cfg, self.echo_pulse, prefix=f'echo_{i}_')
                self.play_long_pulse(self.stark_handle, self.stark_flat_us)

        # Second Ramsey half-pulse (phase = ramsey_phase)
        self.custom_pulse(cfg, self.second_half_pulse,
                          prefix='ramsey_second_')
        self.sync_all(self.us2cycles(0.01))

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

        if cfg.expt.get('parity_meas', True):
            self.custom_pulse(
                cfg, self.parity_meas_pulse, prefix='parity_')

        self.measure_wrapper()


class CavityModeStarkExperiment(Experiment):
    """
    Cavity Mode Ramsey / Echo Experiment with Stark drive on manipulate.

    Software sweep over wait_time: Experiment.acquire runs one program per
    wait-time point.

    Experimental Config:
    expt = dict(
        start: total wait time start [us]  (>= 2 * rise_time per segment)
        step: total wait time step [us]
        expts: number of sweep points
        ramsey_freq: virtual detuning frequency [MHz]
        reps: averages per point
        rounds: number of rounds (default 1)

        mode: 'storage', 'manipulate', or 'coupler'

        # Stark drive on manipulate (fixed across the sweep):
        drive_freq: [MHz]
        drive_gain: [DAC units]
        rise_time: [us]   per-edge ramp duration (Gaussian sigma = rise_time/2)

        # for storage mode:
        storage_mode_idx, man_mode_no

        # for manipulate mode:
        man_mode_idx

        # for coupler mode:
        coupler_pulse

        echoes: number of echo pulses (0 = Ramsey, >0 = Echo)
        parity_meas: True to use parity measurement (default True)

        # optional custom pre/post (overrides auto state prep with warning):
        prepulse, pre_sweep_pulse, postpulse, post_sweep_pulse, gate_based

        active_reset: True/False
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CavityModeStark',
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

        # Build the wait-time grid
        xpts = self.cfg.expt.start + self.cfg.expt.step * np.arange(
            self.cfg.expt.expts)

        # Guard: flat plateau per segment must be non-negative
        n_wait_segments = 1 + self.cfg.expt.get('echoes', 0)
        # Each program quantizes wait_per_seg via sync_all(us2cycles(...));
        # warn against the per-segment step using the default tProc fabric.
        warn_step_subcycle(
            self.soccfg,
            self.cfg.expt.step / n_wait_segments,
            gen_ch=None,
            label=f"step/(1+echoes={self.cfg.expt.get('echoes', 0)})",
        )
        min_wait = 2 * self.cfg.expt.rise_time * n_wait_segments
        if np.min(xpts) < min_wait:
            raise AssertionError(
                f"Minimum wait ({np.min(xpts):.3f} us) < 2*rise_time * "
                f"n_wait_segments ({min_wait:.3f} us). "
                "Increase `start` or reduce `rise_time`/`echoes`.")

        ramsey_freq = self.cfg.expt.ramsey_freq
        avgi_list, avgq_list = [], []
        idata_list, qdata_list = [], []

        for wait in tqdm(xpts, disable=not progress):
            self.cfg.expt.wait_time = float(wait)
            self.cfg.expt.ramsey_phase = (360.0 * ramsey_freq * wait) % 360.0

            prog = CavityModeStarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
                readouts_per_experiment=read_num)

            avgi_list.append(avgi[0][-1])
            avgq_list.append(avgq[0][-1])
            idata, qdata = prog.collect_shots()
            idata_list.append(idata)
            qdata_list.append(qdata)

        avgi = np.array(avgi_list)
        avgq = np.array(avgq_list)
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {
            'xpts': xpts,
            'avgi': avgi,
            'avgq': avgq,
            'amps': amps,
            'phases': phases,
            'idata': np.array(idata_list),
            'qdata': np.array(qdata_list),
        }
        self.data = data
        return data

    def _fitting_cfg(self):
        from copy import deepcopy
        cfg = deepcopy(self.cfg)
        cfg.expt.echoes = [False, 0]
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
        if data is None:
            data = self.data
        print(f'Saving {self.fname}')
        super().save_data(data=h5_safe_data(data))
        return self.fname
