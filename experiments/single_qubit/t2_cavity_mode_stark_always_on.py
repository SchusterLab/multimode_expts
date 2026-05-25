"""
Cavity Mode Ramsey / Echo Experiment with an always-on Stark drive.

Same Ramsey science as t2_cavity_mode_stark.py, but the drive stays at
constant amplitude throughout the entire sequence: it rises *before* any
state preparation, stays steady through prep + Ramsey + parity + readout,
and falls *after* readout. The Ramsey therefore sees only the steady-state
Stark shift — no envelope ramps anywhere near the measurement.

Timing diagram:
    t:  0 -- rise -- guard_pre ------ prep + Ramsey + parity + readout ------ guard_post -- fall -- end
    man_ch: [arb rise][=================== const chunks (drive_hold_time) ===================][arb fall]
    other:              [prep][pi/2]...wait...[pi/2]...[parity]...[readout]

Software sweep over wait_time — one program per point.

Required cfg.expt fields (in addition to the ones already used by
CavityModeStarkExperiment):
  - stark_guard_pre  [us, default 1.0]: idle after rise, before prep.
  - stark_guard_post [us, default 1.0]: idle after readout, before fall.

Optional:
  - drive_hold_time  [us]: total steady-state duration. If omitted, it is
    auto-computed from the cfg as (guard_pre + prep + Ramsey + parity +
    readout + guard_post) plus `drive_hold_margin`, and printed once per
    program instance.
  - drive_hold_margin [us, default 5.0]: safety margin added to the
    auto-computed drive_hold_time.

QICK-level trick: after pre-scheduling all the const chunks on the
manipulate channel up front, we park ``self._gen_ts[stark_ch]`` at 0 so the
sync_all(0.01us) calls inside custom_pulse don't block waiting for the
entire const block. The already-emitted tProc instructions continue to
drive the hardware; only Python-side bookkeeping is adjusted. At the end
the timestamp is restored so sync_all waits for the const to complete
before the fall is scheduled.
"""

import numpy as np
from qick import *
from slab import AttrDict, Experiment
from tqdm.notebook import tqdm

from experiments.MM_base import MM_base, MMAveragerProgram, warn_step_subcycle
from fitting.fit_display_classes import RamseyFitting
from fitting.decaysin_analysis import h5_safe_data


def _estimate_pulse_total_us(pulse_data):
    """Sum the `length` column of a pulse_data descriptor (pulse_data[2])."""
    if pulse_data is None:
        return 0.0
    try:
        return float(np.sum(pulse_data[2]))
    except Exception:
        return 0.0


class CavityModeStarkAlwaysOnProgram(MMAveragerProgram):
    """One sweep point: Stark drive on manipulate stays at constant amplitude
    across prep + Ramsey + readout. Wait time is fixed per program instance.
    """

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

        # Handle f0g1_freq override (e.g., for per-current calibration).
        # The override must be active for *every* compilation that touches the
        # f0-g1 sideband (auto prep, ramsey half-pulse, echo, post), so we
        # install it once at the top of initialize() and restore in a
        # try/finally that wraps the rest of the method.
        f0g1_freq_override = cfg.expt.get('f0g1_freq', None)
        self._f0g1_freq_override = f0g1_freq_override
        _f0g1_saved = None
        if f0g1_freq_override is not None and mode == 'manipulate':
            print(f"f0g1_freq_override: {f0g1_freq_override}")
            mp_entry = cfg.device.multiphoton['pi']['fn-gn+1']
            _f0g1_saved = mp_entry['frequency'][0]
            mp_entry['frequency'][0] = float(f0g1_freq_override)

        try:
            self._initialize_body(cfg, qTest, mode)
        finally:
            if _f0g1_saved is not None:
                cfg.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = (
                    _f0g1_saved)

    def _initialize_body(self, cfg, qTest, mode):
        # --- Build state prep, Ramsey pulse, and determine channels ---
        # (Identical to CavityModeStarkProgram; reproduced here.)
        if mode == 'storage':
            storage_mode_idx = cfg.expt.storage_mode_idx
            full_prep = self.prep_storage_cardinal('+', storage_mode_idx)
            self.auto_prep_str = full_prep[:-1]
            ramsey_str = [full_prep[-1]]
            self.auto_post_str = self.auto_prep_str[::-1]

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

        # Echo pulse
        if self.num_echoes > 0:
            if mode == 'coupler':
                raise ValueError(
                    f"Echoes not supported for mode '{mode}'")
            echo_str = full_prep[::-1] + full_prep
            self.echo_pulse = self.get_prepulse_creator(
                echo_str).pulse.tolist()

        # First and second half-pulse descriptors (second has ramsey_phase)
        if mode in ('storage', 'manipulate'):
            first_pulse = [list(row) for row in self.creator.pulse]
            second_pulse = [list(row) for row in self.creator.pulse]
            second_pulse[3] = [cfg.expt.ramsey_phase
                               for _ in second_pulse[3]]
        else:  # coupler
            first_pulse = [list(x) if isinstance(x, (list, tuple, np.ndarray))
                           else [x] for x in cfg.expt.coupler_pulse]
            second_pulse = [list(x) for x in first_pulse]
            second_pulse[3] = [cfg.expt.ramsey_phase
                               for _ in second_pulse[3]]
        self.first_half_pulse = first_pulse
        self.second_half_pulse = second_pulse

        # --- Always-on Stark drive setup ---
        self.stark_handle = self.register_long_pulse(
            ch=self.man_ch[qTest],
            freq_MHz=cfg.expt.drive_freq,
            gain=cfg.expt.drive_gain,
            rise_time_us=cfg.expt.rise_time,
            name='stark_drive')

        self.guard_pre_us = float(cfg.expt.get('stark_guard_pre', 1.0))
        self.guard_post_us = float(cfg.expt.get('stark_guard_post', 1.0))

        # --- Estimate required drive_hold_time from cfg ---
        ramsey_half_pulse_us = _estimate_pulse_total_us(
            np.atleast_2d(first_pulse)
            if not isinstance(first_pulse, list) else first_pulse)
        echo_us = 0.0
        if self.num_echoes > 0:
            echo_us = self.num_echoes * _estimate_pulse_total_us(
                self.echo_pulse)
        prep_us = _estimate_pulse_total_us(self.auto_prep_pulse)
        post_us = (_estimate_pulse_total_us(self.auto_post_pulse)
                   if self.auto_post_pulse is not None else 0.0)
        parity_us = (_estimate_pulse_total_us(
            self._parity_pulse_for_estimate(cfg))
            if cfg.expt.get('parity_meas', True) else 0.0)
        readout_us = float(
            cfg.device.readout.readout_length[qTest]
            if isinstance(cfg.device.readout.readout_length, list)
            else cfg.device.readout.readout_length)

        est_min_us = (
            self.guard_pre_us
            + prep_us
            + 2 * ramsey_half_pulse_us
            + cfg.expt.wait_time
            + echo_us
            + post_us
            + parity_us
            + readout_us
            + self.guard_post_us)

        # Expose the breakdown for callers (Experiment.acquire prints these once).
        self.hold_time_breakdown = {
            'guard_pre': self.guard_pre_us,
            'prep': prep_us,
            'pi2_x2': 2 * ramsey_half_pulse_us,
            'wait_time': float(cfg.expt.wait_time),
            'echo': echo_us,
            'post': post_us,
            'parity': parity_us,
            'readout': readout_us,
            'guard_post': self.guard_post_us,
            'est_min': est_min_us,
        }

        # drive_hold_time is auto-computed by default (estimate + margin);
        # user can override by setting cfg.expt.drive_hold_time explicitly.
        drive_hold_margin_us = float(
            cfg.expt.get('drive_hold_margin', 5.0))
        drive_hold_override = cfg.expt.get('drive_hold_time', None)
        if drive_hold_override is None:
            self.drive_hold_us = est_min_us + drive_hold_margin_us
        else:
            self.drive_hold_us = float(drive_hold_override)

        assert self.drive_hold_us >= est_min_us, (
            f"drive_hold_time ({self.drive_hold_us:.2f} us) < estimated "
            f"minimum ({est_min_us:.2f} us). Components: "
            f"guard_pre={self.guard_pre_us}, prep={prep_us:.2f}, "
            f"pi/2 x2={2*ramsey_half_pulse_us:.2f}, "
            f"wait={cfg.expt.wait_time:.2f}, echo={echo_us:.2f}, "
            f"post={post_us:.2f}, parity={parity_us:.2f}, "
            f"readout={readout_us:.2f}, guard_post={self.guard_post_us}.")

        # --- Parity measurement pulse ---
        man_mode_no = cfg.expt.get('man_mode_no', 1)
        self.parity_meas_pulse = self.get_parity_str(
            man_mode_no, return_pulse=True, second_phase=180, fast=False)

        self.sync_all(200)

    def _parity_pulse_for_estimate(self, cfg):
        """Build the parity meas pulse once up-front so we can estimate its
        duration before the rest of initialize() finishes."""
        man_mode_no = cfg.expt.get('man_mode_no', 1)
        return self.get_parity_str(
            man_mode_no, return_pulse=True, second_phase=180, fast=False)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        self.reset_and_sync()

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # Sync all channels before the stark drive starts.
        self.sync_all()

        # --- Rise: play on manipulate alone ---
        self.play_long_pulse_rise(self.stark_handle)

        # Advance tProc reference by guard_pre so subsequent non-manipulate
        # pulses start only after the drive has settled.
        self.sync_all(self.us2cycles(self.guard_pre_us))

        # --- Pre-schedule the entire const middle on the manipulate channel ---
        # These instructions are emitted to the tProc now; the DAC will play
        # them in parallel with the non-manipulate pulses we schedule next.
        self.play_long_pulse_const(self.stark_handle, self.drive_hold_us)
        stark_ch = self.stark_handle.ch

        # Park the manipulate's Python-side timestamp at 0 so the internal
        # sync_all(0.01us) calls that custom_pulse makes don't wait for the
        # whole const block. The hardware is unaffected — only bookkeeping.
        ts_end_of_const = self.get_timestamp(gen_ch=stark_ch)
        self.set_timestamp(0, gen_ch=stark_ch)

        # --- Non-manipulate sequence: runs in parallel with the const ---
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

        # Wait (Ramsey evolution while drive stays on)
        self.sync_all(self.us2cycles(
            cfg.expt.wait_time / (1 + self.num_echoes)))

        if self.num_echoes > 0:
            for i in range(self.num_echoes):
                self.custom_pulse(cfg, self.echo_pulse, prefix=f'echo_{i}_')
                self.sync_all(self.us2cycles(
                    cfg.expt.wait_time / (1 + self.num_echoes)))

        # Second Ramsey half-pulse
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

        # --- Restore manipulate timestamp so sync_all waits for the const ---
        self.set_timestamp(ts_end_of_const, gen_ch=stark_ch)
        # Guard-post: keep the drive on a bit longer past readout, then fall.
        self.sync_all(self.us2cycles(self.guard_post_us))

        # Fall
        self.play_long_pulse_fall(self.stark_handle)
        self.sync_all()


class CavityModeStarkAlwaysOnExperiment(Experiment):
    """
    Cavity Mode Ramsey with an always-on Stark drive on manipulate.

    The drive is rung up before state prep, stays at constant amplitude
    through prep + Ramsey + readout, and rings down afterwards. Only the
    steady-state Stark shift affects the Ramsey fringe.

    Experimental Config:
    expt = dict(
        # Ramsey sweep (software-swept per point)
        start: total wait time start [us]
        step: total wait time step [us]
        expts: number of sweep points
        ramsey_freq: virtual detuning frequency [MHz]
        reps: averages per point
        rounds: number of rounds (default 1)

        mode: 'storage', 'manipulate', or 'coupler'

        # Stark drive on manipulate (fixed across the sweep)
        drive_freq: [MHz]
        drive_gain: [DAC units]
        rise_time: [us]   per-edge ramp duration (Gaussian sigma = rise_time/2)
        stark_guard_pre:  [us, default 1.0] idle after rise, before prep
        stark_guard_post: [us, default 1.0] idle after readout, before fall
        drive_hold_time:  [us] total steady-state duration; must be >=
                          guard_pre + prep + Ramsey + parity + readout
                          + guard_post. Asserted at initialize().

        # for storage mode:
        storage_mode_idx, man_mode_no

        # for manipulate mode:
        man_mode_idx

        # for coupler mode:
        coupler_pulse

        echoes: number of echo pulses (0 = Ramsey, >0 = Echo)
        parity_meas: True to use parity measurement (default True)

        prepulse, pre_sweep_pulse, postpulse, post_sweep_pulse, gate_based

        active_reset: True/False
    )
    """

    def __init__(self, soccfg=None, path='', prefix='CavityModeStarkAlwaysOn',
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

        xpts = self.cfg.expt.start + self.cfg.expt.step * np.arange(
            self.cfg.expt.expts)

        # Each program quantizes wait_per_seg via sync_all(us2cycles(...));
        # warn against the per-segment step using the default tProc fabric.
        n_wait_segments = 1 + self.cfg.expt.get('echoes', 0)
        warn_step_subcycle(
            self.soccfg,
            self.cfg.expt.step / n_wait_segments,
            gen_ch=None,
            label=f"step/(1+echoes={self.cfg.expt.get('echoes', 0)})",
        )

        ramsey_freq = self.cfg.expt.ramsey_freq

        # Build one probe program at the worst-case (longest) wait to
        # compute drive_hold_time once, print the budget, and freeze it for
        # every subsequent point.
        max_wait = float(np.max(xpts))
        self.cfg.expt.wait_time = max_wait
        self.cfg.expt.ramsey_phase = 0.0
        probe = CavityModeStarkAlwaysOnProgram(
            soccfg=self.soccfg, cfg=self.cfg)
        b = probe.hold_time_breakdown
        drive_hold_us = probe.drive_hold_us
        margin_us = drive_hold_us - b['est_min']
        print(
            f"[CavityModeStarkAlwaysOn] drive_hold_time = "
            f"{drive_hold_us:.2f} us  "
            f"(estimate {b['est_min']:.2f} + margin {margin_us:.2f}) "
            f"at longest wait = {max_wait:.2f} us\n"
            f"  breakdown [us]: "
            f"guard_pre={b['guard_pre']}, prep={b['prep']:.2f}, "
            f"pi/2 x2={b['pi2_x2']:.2f}, wait={b['wait_time']:.2f}, "
            f"echo={b['echo']:.2f}, post={b['post']:.2f}, "
            f"parity={b['parity']:.2f}, readout={b['readout']:.2f}, "
            f"guard_post={b['guard_post']}")
        # Freeze the value so per-point programs don't recompute (and so
        # the drive always covers the worst-case duration).
        self.cfg.expt.drive_hold_time = drive_hold_us

        avgi_list, avgq_list = [], []
        idata_list, qdata_list = [], []

        for wait in tqdm(xpts, disable=not progress):
            self.cfg.expt.wait_time = float(wait)
            self.cfg.expt.ramsey_phase = (360.0 * ramsey_freq * wait) % 360.0

            prog = CavityModeStarkAlwaysOnProgram(
                soccfg=self.soccfg, cfg=self.cfg)
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
