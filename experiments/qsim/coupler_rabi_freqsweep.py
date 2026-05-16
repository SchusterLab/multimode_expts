"""
Coupler-drive Rabi chevron with a hardware frequency sweep.

Difference vs amplitude_rabi_coupler / length_rabi_coupler
---------------------------------------------------------
Those run pure software sweeps over a single axis (gain or length) and use a
coupler-state-selective f0-g{man_no} readout. This module instead:
  - sweeps the drive frequency on the manipulate channel as a hardware
    NDAverager axis (one acquire() call returns a freq trace),
  - software-loops the second axis (gain for the amplitude variant, flat
    pulse length for the length variant) in Python,
  - maps the cavity population to a g/e qubit readout via the qsim_base-style
    ge -> ef -> f0g1 pi chain (not the coupler-state-selective f0-gn).

Pulse sequence
--------------
  1. Flat-top pulse on the manipulate channel:
       - frequency: NDAverager hardware sweep (innermost / fastest axis),
       - gain: fixed per-acquire (set by the outer Python loop in the
         AmplitudeRabi variant; fixed at cfg.expt.gain in the Length variant),
       - flat length: fixed per-acquire (set by the outer Python loop in the
         Length variant; fixed at cfg.expt.flat_length in the Amplitude
         variant),
       - ramps: gauss envelope of sigma cfg.expt.ramp_sigma (us). Total pulse
         time on a ch in {0, 1, 3} is 6*sigma + flat_length (matches
         MM_base.custom_pulse flat_top accounting).
  2. ge pi (qubit), ef pi (qubit), f0g1 pi (M1) via get_prepulse_creator +
     custom_pulse, identical mechanism to qsim_base's prepulse chain.
  3. Standard dispersive readout via measure_wrapper.

No active reset and no pre-selection: the program assumes thermalized starting
state between shots (relax_delay) and writes only the swept pulse + pi chain.
"""

from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from qick.averager_program import QickSweep
from slab import AttrDict, Experiment
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.MM_base import MMNDAveragerProgram


class CouplerRabiFreqsweepProgram(MMNDAveragerProgram):
    """
    Flat-top pulse on the manipulate channel with hardware freq sweep, then
    qubit ge pi -> ef pi -> f0g1 pi, then readout.

    Required cfg.expt fields:
        qubits:        list of qubit indices, e.g. [0]
        reps:          averages per inner-axis point
        rounds:        soft averages of the whole hardware sweep
        gain:          DAC gain of the swept manipulate pulse (int)
        flat_length:   flat-segment duration of the swept pulse [us]
        ramp_sigma:    gauss ramp sigma of the swept pulse [us]
        freq_start:    inner-axis (freq) start [MHz]
        freq_stop:     inner-axis (freq) stop  [MHz]
        freq_expts:    inner-axis (freq) number of points
    """

    def __init__(self, soccfg: QickConfig, cfg: AttrDict):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.get('rounds', 1)

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = cfg.expt.qubits[0]

        # Post-sweep pi chain: ge -> ef -> f0g1, same gate spec as the
        # qsim_base prepulse that maps |g, n_cav> against the qubit f-manifold.
        postchain_seq = [
            ['qubit', 'ge', 'pi', 0],
            ['qubit', 'ef', 'pi', 0],
            ['man', 'M1', 'pi', 0],
        ]
        self.post_pulse = self.get_prepulse_creator(postchain_seq).pulse.tolist()

        # Swept flat-top pulse on the manipulate channel
        man_ch = self.man_ch[qTest]
        self.man_ch_used = man_ch

        gain = int(cfg.expt.gain)
        flat_length_us = float(cfg.expt.flat_length)
        ramp_sigma_us = float(cfg.expt.ramp_sigma)

        ramp_sigma_cycles = self.us2cycles(ramp_sigma_us, gen_ch=man_ch)
        flat_length_cycles = self.us2cycles(flat_length_us, gen_ch=man_ch)
        self.ramp_sigma_cycles = ramp_sigma_cycles
        self.flat_length_cycles = flat_length_cycles

        # Manipulate ch (gen 0/1/3) flat_top envelopes use sigma*6 in MM_base's
        # custom_pulse; mirror that here so the ramp shape matches everywhere.
        self.add_gauss(
            ch=man_ch, name="chevron_ramp",
            sigma=ramp_sigma_cycles, length=ramp_sigma_cycles * 6,
        )

        freq_start_MHz = float(cfg.expt.freq_start)
        freq_stop_MHz = float(cfg.expt.freq_stop)
        freq_npts = int(cfg.expt.freq_expts)

        self.set_pulse_registers(
            ch=man_ch, style="flat_top",
            freq=self.freq2reg(freq_start_MHz, gen_ch=man_ch),
            phase=0, gain=gain,
            length=flat_length_cycles,
            waveform="chevron_ramp",
        )

        # NDAverager sweep on the manipulate gen's freq register. First-added
        # is innermost / fastest; only one axis here.
        self.swept_freq_reg = self.get_gen_reg(man_ch, "freq")
        self.add_sweep(QickSweep(
            self, self.swept_freq_reg,
            freq_start_MHz, freq_stop_MHz, freq_npts,
        ))

        self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = AttrDict(self.cfg)

        if self.flat_length_cycles > 0 or self.ramp_sigma_cycles > 0:
            self.pulse(ch=self.man_ch_used)
            self.sync_all()

        self.custom_pulse(cfg, self.post_pulse, prefix='post_')

        self.sync_all(self.us2cycles(0.05))
        self.measure_wrapper()


def _broadcast_device_cfg(cfg):
    """Broadcast scalar device entries into per-qubit lists (mirrors the
    boilerplate at the top of LengthRabiCouplerExperiment.acquire)."""
    num_qubits_sample = len(cfg.device.qubit.f_ge)
    for subcfg in (cfg.device.readout, cfg.device.qubit, cfg.hw.soc):
        for key, value in subcfg.items():
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        if not isinstance(value3, list):
                            value2.update({key3: [value3] * num_qubits_sample})
            elif not isinstance(value, list):
                subcfg.update({key: [value] * num_qubits_sample})


def _run_freq_line(soccfg, im, cfg, progress=False):
    """Build the program with current cfg and run one hardware freq sweep.
    Returns (freq_pts_MHz, avgi_line, avgq_line)."""
    prog = CouplerRabiFreqsweepProgram(soccfg=soccfg, cfg=cfg)
    expt_pts, avg_di, avg_dq = prog.acquire(
        im[cfg.aliases.soc],
        threshold=None,
        load_pulses=True,
        progress=progress,
    )
    # expt_pts: list per sweep axis; we registered exactly one axis (freq).
    # avg_di/avg_dq: list per ro_ch; we use ro_ch 0. Last "save_experiments"
    # axis is the readout index; we only do 1 readout.
    freq_pts = np.asarray(expt_pts[0])
    avgi = np.asarray(avg_di[0]).squeeze()
    avgq = np.asarray(avg_dq[0]).squeeze()
    return freq_pts, avgi, avgq


class AmplitudeRabiCouplerFreqsweepExperiment(Experiment):
    """
    Amplitude / frequency chevron on the manipulate-channel coupler drive.

    Inner axis (hardware, NDAverager): pulse frequency.
    Outer axis (software, Python loop): pulse gain.

    Experimental Config:
    expt = dict(
        qubits:       list, e.g. [0]
        reps:         averages per (gain, freq) point
        rounds:       soft averages of each freq sweep
        flat_length:  flat-segment duration of the swept pulse [us]
        ramp_sigma:   gauss ramp sigma of the swept pulse [us]
        freq_start:   inner-axis freq start [MHz]
        freq_stop:    inner-axis freq stop  [MHz]
        freq_expts:   inner-axis number of freq points
        start:        outer-axis (gain) start [DAC]
        step:         outer-axis (gain) step  [DAC]
        expts:        outer-axis (gain) number of points
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiCouplerFreqsweep',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        _broadcast_device_cfg(self.cfg)

        gains = (self.cfg.expt.start
                 + self.cfg.expt.step * np.arange(self.cfg.expt.expts))

        avgi_2d, avgq_2d = [], []
        freq_pts = None
        for gain in tqdm(gains, disable=not progress):
            self.cfg.expt.gain = int(round(gain))
            f, avgi_line, avgq_line = _run_freq_line(
                self.soccfg, self.im, self.cfg, progress=False,
            )
            if freq_pts is None:
                freq_pts = f
            avgi_2d.append(avgi_line)
            avgq_2d.append(avgq_line)

        avgi = np.asarray(avgi_2d)
        avgq = np.asarray(avgq_2d)
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {
            'xpts': freq_pts,
            'ypts': gains,
            'avgi': avgi, 'avgq': avgq,
            'amps': amps, 'phases': phases,
        }
        self.data = data
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        fig, axs = plt.subplots(2, 1, figsize=(10, 9))
        axs[0].set_title('Coupler amplitude/freq chevron')
        mesh = axs[0].pcolormesh(data['xpts'], data['ypts'], data['avgi'],
                                 shading='auto')
        fig.colorbar(mesh, ax=axs[0], label='I [ADC level]')
        mesh = axs[1].pcolormesh(data['xpts'], data['ypts'], data['avgq'],
                                 shading='auto')
        fig.colorbar(mesh, ax=axs[1], label='Q [ADC level]')
        for ax in axs:
            ax.set_xlabel('Drive freq [MHz]')
            ax.set_ylabel('Gain [DAC]')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


class LengthRabiCouplerFreqsweepExperiment(Experiment):
    """
    Length / frequency chevron on the manipulate-channel coupler drive.

    Inner axis (hardware, NDAverager): pulse frequency.
    Outer axis (software, Python loop): flat-segment length [us].
    Ramp sigma stays fixed at cfg.expt.ramp_sigma across the sweep.

    Experimental Config:
    expt = dict(
        qubits:       list, e.g. [0]
        reps:         averages per (length, freq) point
        rounds:       soft averages of each freq sweep
        gain:         DAC gain of the swept pulse (fixed)
        ramp_sigma:   gauss ramp sigma of the swept pulse [us] (fixed)
        freq_start:   inner-axis freq start [MHz]
        freq_stop:    inner-axis freq stop  [MHz]
        freq_expts:   inner-axis number of freq points
        start:        outer-axis (flat length) start [us]
        step:         outer-axis (flat length) step  [us]
        expts:        outer-axis number of length points
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiCouplerFreqsweep',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        _broadcast_device_cfg(self.cfg)

        lengths = (self.cfg.expt.start
                   + self.cfg.expt.step * np.arange(self.cfg.expt.expts))

        avgi_2d, avgq_2d = [], []
        freq_pts = None
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.flat_length = float(length)
            f, avgi_line, avgq_line = _run_freq_line(
                self.soccfg, self.im, self.cfg, progress=False,
            )
            if freq_pts is None:
                freq_pts = f
            avgi_2d.append(avgi_line)
            avgq_2d.append(avgq_line)

        avgi = np.asarray(avgi_2d)
        avgq = np.asarray(avgq_2d)
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {
            'xpts': freq_pts,
            'ypts': lengths,
            'avgi': avgi, 'avgq': avgq,
            'amps': amps, 'phases': phases,
        }
        self.data = data
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        fig, axs = plt.subplots(2, 1, figsize=(10, 9))
        axs[0].set_title('Coupler length/freq chevron')
        mesh = axs[0].pcolormesh(data['xpts'], data['ypts'], data['avgi'],
                                 shading='auto')
        fig.colorbar(mesh, ax=axs[0], label='I [ADC level]')
        mesh = axs[1].pcolormesh(data['xpts'], data['ypts'], data['avgq'],
                                 shading='auto')
        fig.colorbar(mesh, ax=axs[1], label='Q [ADC level]')
        for ax in axs:
            ax.set_xlabel('Drive freq [MHz]')
            ax.set_ylabel('Flat length [us]')
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
