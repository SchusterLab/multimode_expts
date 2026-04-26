"""
f0g1 sideband spectroscopy with an always-on AC Stark drive on the
manipulate channel.

Same probe + measure body as PulseProbeF0g1SpectroscopyProgram, but a
constant-amplitude tone is rung up on `man_ch` before the measurement,
held steady through the entire body, and rung down at the end. Active
reset (if enabled) runs *before* the rise -- the drive only covers the
main shot.

Channel layout (verified):
- Stark drive: self.man_ch[qTest]   (cfg.hw.soc.dacs.manipulate_in.ch)
- f0g1 probe:  self.f0g1_ch[qTest]  (cfg.hw.soc.dacs.sideband.ch)
These are physically distinct DACs, so the const drive and the probe
do not collide.

Two classes:
- PulseProbeF0g1StarkAlwaysOnProgram      (MMRAveragerProgram, freq sweep)
- PulseProbeF0g1StarkAlwaysOnExperiment   (1D wrapper, single flux point)

For 2D vs flux current, do a notebook-side queued-job loop using
CharacterizationRunner.run(coupler_current=...) -- the worker auto-ramps
station.yoko_coupler. See do_coupler_current_sweep in the
T2_ac_stark_shift notebook for the canonical pattern.
"""

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import AttrDict, Experiment

from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram


def _estimate_pulse_total_us(pulse_data):
    """Sum the `length` column of a pulse_data descriptor (pulse_data[2])."""
    if pulse_data is None:
        return 0.0
    try:
        return float(np.sum(pulse_data[2]))
    except Exception:
        return 0.0


class PulseProbeF0g1StarkAlwaysOnProgram(MMRAveragerProgram):
    """f0g1 frequency sweep with an always-on Stark drive on man_ch."""

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

        # --- Frequency sweep registers (same as the no-drive program) ---
        self.q_rp = self.ch_page(self.f0g1_ch[qTest])
        self.r_freq = self.sreg(self.f0g1_ch[qTest], "freq")
        self.r_freq2 = 4
        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_chs[qTest])
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_chs[qTest])
        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start)

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
        prepulse_us = 0.0
        if cfg.expt.get('prepulse', False):
            prepulse_us = _estimate_pulse_total_us(cfg.expt.pre_sweep_pulse)

        qubit_f_us = 0.0
        if cfg.expt.get('qubit_f', False):
            pi_ge_sigma_us = float(cfg.device.qubit.pulses.pi_ge.sigma[qTest])
            pi_ef_sigma_us = float(cfg.device.qubit.pulses.pi_ef.sigma[qTest])
            qubit_f_us = 4.0 * (pi_ge_sigma_us + pi_ef_sigma_us)

        probe_us = float(cfg.expt.length)
        readout_length = cfg.device.readout.readout_length
        readout_us = float(
            readout_length[qTest] if isinstance(readout_length, list)
            else readout_length)

        est_min_us = (
            self.guard_pre_us
            + prepulse_us
            + qubit_f_us
            + probe_us
            + 0.05  # the sync_all(0.05us) before measure
            + readout_us
            + self.guard_post_us)

        self.hold_time_breakdown = {
            'guard_pre': self.guard_pre_us,
            'prepulse': prepulse_us,
            'qubit_f': qubit_f_us,
            'probe': probe_us,
            'readout': readout_us,
            'guard_post': self.guard_post_us,
            'est_min': est_min_us,
        }

        drive_hold_margin_us = float(cfg.expt.get('drive_hold_margin', 5.0))
        drive_hold_override = cfg.expt.get('drive_hold_time', None)
        if drive_hold_override is None:
            self.drive_hold_us = est_min_us + drive_hold_margin_us
        else:
            self.drive_hold_us = float(drive_hold_override)

        assert self.drive_hold_us >= est_min_us, (
            f"drive_hold_time ({self.drive_hold_us:.2f} us) < estimated "
            f"minimum ({est_min_us:.2f} us). Components: "
            f"guard_pre={self.guard_pre_us}, prepulse={prepulse_us:.2f}, "
            f"qubit_f={qubit_f_us:.2f}, probe={probe_us:.2f}, "
            f"readout={readout_us:.2f}, guard_post={self.guard_post_us}.")

        self.synci(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # Active reset runs before the drive turns on (drive covers main shot only).
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        self.sync_all()

        # --- Rise: play on manipulate alone ---
        self.play_long_pulse_rise(self.stark_handle)
        self.sync_all(self.us2cycles(self.guard_pre_us))

        # --- Pre-schedule the entire const middle on man_ch ---
        self.play_long_pulse_const(self.stark_handle, self.drive_hold_us)
        stark_ch = self.stark_handle.ch
        ts_end_of_const = self.get_timestamp(gen_ch=stark_ch)
        self.set_timestamp(0, gen_ch=stark_ch)

        # --- Original probe + measure body ---
        if cfg.expt.get('prepulse', False):
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre_ar_')

        if cfg.expt.get('qubit_f', False):
            self.setup_and_pulse(
                ch=self.qubit_chs[qTest], style="arb",
                freq=self.f_ge_reg[0], phase=0,
                gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            self.setup_and_pulse(
                ch=self.qubit_chs[qTest], style="arb",
                freq=self.f_ef_reg[0], phase=0,
                gain=self.pi_ef_gain, waveform="pi_qubit_ef")

        self.set_pulse_registers(
            ch=self.f0g1_ch[qTest],
            style="const",
            freq=0,  # set by mathi from r_freq2
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length, gen_ch=self.f0g1_ch[qTest]))
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.f0g1_ch[qTest])

        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest]))

        # --- Restore manipulate timestamp so sync_all waits for the const ---
        self.set_timestamp(ts_end_of_const, gen_ch=stark_ch)
        self.sync_all(self.us2cycles(self.guard_post_us))

        self.play_long_pulse_fall(self.stark_handle)
        self.sync_all()

    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step)


def _print_hold_time_breakdown(prog, prefix='[PulseProbeF0g1StarkAlwaysOn]'):
    b = prog.hold_time_breakdown
    margin = prog.drive_hold_us - b['est_min']
    print(
        f"{prefix} drive_hold_time = {prog.drive_hold_us:.2f} us  "
        f"(estimate {b['est_min']:.2f} + margin {margin:.2f})\n"
        f"  breakdown [us]: guard_pre={b['guard_pre']}, "
        f"prepulse={b['prepulse']:.2f}, qubit_f={b['qubit_f']:.2f}, "
        f"probe={b['probe']:.2f}, readout={b['readout']:.2f}, "
        f"guard_post={b['guard_post']}")


class PulseProbeF0g1StarkAlwaysOnExperiment(Experiment):
    """
    1D f0g1 spectroscopy at a single flux point with an always-on Stark
    drive on the manipulate channel.

    Experimental Config
    expt = dict(
        # Frequency axis
        start, step, expts, reps, rounds, length, gain, qubits

        # Optional probe-side fields (carried over from the no-drive program)
        qubit_f, prepulse, pre_sweep_pulse, active_reset

        # Stark drive (fixed across the freq sweep)
        drive_freq          [MHz]
        drive_gain          [DAC]
        rise_time           [us]   per-edge ramp; sigma = rise_time/2
        stark_guard_pre     [us, default 1.0]
        stark_guard_post    [us, default 1.0]
        drive_hold_time     [us, optional] override for auto-computed value
        drive_hold_margin   [us, default 5.0]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeF0g1StarkAlwaysOn',
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

        prog = PulseProbeF0g1StarkAlwaysOnProgram(
            soccfg=self.soccfg, cfg=self.cfg)
        _print_hold_time_breakdown(prog)

        x_pts, avgi, avgq = prog.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None, load_pulses=True,
            progress=progress, debug=debug,
            readouts_per_experiment=read_num)

        avgi = avgi[0][-1]
        avgq = avgq[0][-1]
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq,
                'amps': amps, 'phases': phases}
        self.data = data
        return data

    def analyze(self, data=None, fit=True, signs=[1, 1, 1], **kwargs):
        if data is None:
            data = self.data
        from fitting.fit_display_classes import Spectroscopy
        spec_analysis = Spectroscopy(data, signs=signs, config=self.cfg, station=None)
        spec_analysis.analyze(fit=fit)
        return data

    def display(self, data=None, fit=True, signs=[1, 1, 1],
                title='f0g1 Spectroscopy (Stark drive on)', **kwargs):
        if data is None:
            data = self.data
        from fitting.fit_display_classes import Spectroscopy
        spec_analysis = Spectroscopy(data, signs=signs, config=self.cfg, station=None)
        vlines = kwargs.get('vlines', None)
        spec_analysis.display(title=title, vlines=vlines, fit=fit)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
