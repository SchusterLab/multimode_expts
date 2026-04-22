import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.MM_base import MM_base, MMAveragerProgram
from experiments.single_qubit.ramsey_coupler import (
    SUPPORTED_PULSE_TYPES,
    _resolve_pulse_params,
    _validate_pulse_type,
)

"""
Length (time) Rabi on the coupler, driven via the manipulate channel.

Sequence
--------
  1. Prepare manipulate mode (default M1) in Fock |1>.
  2. Play a pulse on the manipulate DAC at a fixed drive frequency and
     gain, sweeping the pulse duration (the rabi knob).
  3. Swap manipulate |1> back to qubit |e> (f0-g1 pi + e0-f0 pi).
  4. Dispersive readout.

Uses the same Python-level length loop pattern as length_rabi_f0g1_general:
each point is its own compiled program, driven by `cfg.expt.length_placeholder`.

Drive frequency and pulse gain default to cfg.device.coupler.pulses.hpi,
matching amplitude_rabi_coupler / ramsey_coupler; cfg.expt overrides win.
"""


class LengthRabiCouplerProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        qTest = cfg.expt.qubits[0]
        man_no = int(cfg.expt.get('man_mode_no', 1))

        pulse_type = cfg.expt.get('pulse_type', 'const')
        _validate_pulse_type(pulse_type)
        self.pulse_type = pulse_type

        prep_seq = self.prep_man_fock_state(man_no, '1')
        self.prep_pulse = self.get_prepulse_creator(prep_seq).pulse.tolist()

        swap_seq = [
            ['multiphoton', f'f0-g{man_no}', 'pi', 0],
            ['multiphoton', 'e0-f0', 'pi', 0],
        ]
        self.swap_pulse = self.get_prepulse_creator(swap_seq).pulse.tolist()

        freq_MHz, gain = _resolve_pulse_params(cfg, names=('freq', 'gain'))
        self.drive_gain = gain
        self.f_drive = self.freq2reg(freq_MHz, gen_ch=self.man_ch[qTest])

        length_us = float(cfg.expt.length_placeholder)
        self.length_us = length_us
        if pulse_type == 'gauss':
            # For a gauss pulse, total length = sigma*4, so sigma = length/4.
            sigma_cycles = self.us2cycles(length_us / 4, gen_ch=self.man_ch[qTest])
            self.drive_sigma = sigma_cycles
            if sigma_cycles > 0:
                self.add_gauss(ch=self.man_ch[qTest], name="coupler_len_rabi",
                               sigma=sigma_cycles, length=sigma_cycles * 4)
        elif pulse_type == 'const':
            self.drive_length = self.us2cycles(length_us, gen_ch=self.man_ch[qTest])
        else:
            raise ValueError(f"unreachable: pulse_type={pulse_type!r}")

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        self.sync_all()

        if cfg.expt.get('prepulse', False):
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

        self.custom_pulse(cfg, self.prep_pulse, prefix='prep_man_')
        self.sync_all()

        # Skip the drive at length=0 (identity point -> baseline readout).
        if self.length_us > 0:
            if self.pulse_type == 'gauss':
                self.setup_and_pulse(
                    ch=self.man_ch[qTest],
                    style="arb",
                    freq=self.f_drive,
                    phase=0,
                    gain=self.drive_gain,
                    waveform="coupler_len_rabi",
                )
            elif self.pulse_type == 'const':
                self.setup_and_pulse(
                    ch=self.man_ch[qTest],
                    style="const",
                    freq=self.f_drive,
                    phase=0,
                    gain=self.drive_gain,
                    length=self.drive_length,
                )
            else:
                raise ValueError(f"unreachable: pulse_type={self.pulse_type!r}")
            self.sync_all()

        self.custom_pulse(cfg, self.swap_pulse, prefix='swap_')

        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest]),
        )


class LengthRabiCouplerExperiment(Experiment):
    """
    Length (time) Rabi on the coupler, driven via the manipulate channel.

    Prepares |1> in the manipulate mode, plays a pulse on the manipulate
    DAC at a fixed drive frequency and gain with swept duration, then
    swaps |1> -> |e> on the qubit for readout. Length is swept in a
    Python-level loop (one compiled QICK program per length point).

    Experimental Config:
    expt = dict(
        start:        pulse length start [us]
        step:         pulse length step  [us]
        expts:        number of length points
        reps:         averages per length point
        rounds:       sweep repetitions
        qubits:       list, e.g. [0]
        man_mode_no:  (optional, default 1) manipulate mode index used
                      for Fock-|1> prep and swap-out
        pulse_type:   (optional, default 'const') pulse envelope. Supported:
                      'const' (flat pulse of swept duration),
                      'gauss' (sigma = length/4; total length = length).
                      Other values raise.
        freq:         (optional) override cfg.device.coupler.pulses.hpi.freq[0]  [MHz]
        gain:         (optional) override cfg.device.coupler.pulses.hpi.gain[0]  [DAC]
        prepulse:        (optional) bool, play cfg.expt.pre_sweep_pulse first
        pre_sweep_pulse: (optional) gate-list spec for the custom prepulse
        active_reset:    (optional) bool, run active reset at start of body
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiCoupler',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        _validate_pulse_type(self.cfg.expt.get('pulse_type', 'const'))

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

        read_num = 1
        if self.cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        lengths = (self.cfg.expt.start
                   + self.cfg.expt.step * np.arange(self.cfg.expt.expts))

        avgi_list, avgq_list = [], []
        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            prog = LengthRabiCouplerProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
                debug=debug,
                readouts_per_experiment=read_num,
            )
            avgi_list.append(avgi[0][-1])
            avgq_list.append(avgq[0][-1])

        avgi = np.array(avgi_list)
        avgq = np.array(avgq_list)
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {'xpts': lengths, 'avgi': avgi, 'avgq': avgq,
                'amps': amps, 'phases': phases}
        self.data = data
        return data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data = self.data

        if fit:
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'], data['avgi'], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'], data['avgq'], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'], data['amps'], fitparams=fitparams)
            data['fit_avgi'] = p_avgi
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            # p = [amp, freq_MHz, phase_deg, tau, offset, slope]
            for key, p in (('avgi', p_avgi), ('avgq', p_avgq), ('amps', p_amps)):
                if isinstance(p, (list, np.ndarray)):
                    period = 1.0 / p[1]                     # us (since xpts are in us)
                    phase_rad = p[2] * np.pi / 180
                    pi_time = (0.5 - phase_rad / np.pi) * period
                    hpi_time = (0.25 - phase_rad / np.pi) * period
                    data[f'pi_time_{key}'] = pi_time
                    data[f'hpi_time_{key}'] = hpi_time
        return data

    def display(self, data=None, fit=True, vline=None, **kwargs):
        if data is None:
            data = self.data

        plt.figure(figsize=(10, 9))
        plt.subplot(
            211,
            title='Coupler Length Rabi (via manipulate channel)',
            ylabel="I [ADC units]",
        )
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1], 'o-')
        if fit:
            p = data.get('fit_avgi')
            if isinstance(p, (list, np.ndarray)):
                plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
                pi_time = data['pi_time_avgi']
                hpi_time = data['hpi_time_avgi']
                print(f'Pi time from avgi data [us]: {pi_time}')
                print(f'\tPi/2 time from avgi data [us]: {hpi_time}')
                plt.axvline(pi_time, color='0.2', linestyle='--')
                plt.axvline(hpi_time, color='0.2', linestyle='--')
                if vline is not None:
                    plt.axvline(vline, color='0.2', linestyle='--')

        plt.subplot(212, xlabel="Pulse length [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1], 'o-')
        if fit:
            p = data.get('fit_avgq')
            if isinstance(p, (list, np.ndarray)):
                plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
                pi_time = data['pi_time_avgq']
                hpi_time = data['hpi_time_avgq']
                print(f'Pi time from avgq data [us]: {pi_time}')
                print(f'\tPi/2 time from avgq data [us]: {hpi_time}')
                plt.axvline(pi_time, color='0.2', linestyle='--')
                plt.axvline(hpi_time, color='0.2', linestyle='--')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
