import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import Experiment, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram
from experiments.single_qubit.ramsey_coupler import (
    SUPPORTED_PULSE_TYPES,
    _resolve_pulse_params,
    _validate_pulse_type,
)

"""
Amplitude Rabi on the coupler, driven via the manipulate channel.

Sequence
--------
  1. Prepare manipulate mode (default M1) in Fock |1>.
  2. Play a pulse on the manipulate DAC at a fixed drive frequency,
     sweeping the pulse gain (the rabi knob). Pulse shape defaults to
     'gauss' and is validated up front - any other shape raises.
  3. Swap manipulate |1> back to qubit |e> (f0-g1 pi + e0-f0 pi).
  4. Dispersive readout.

Drive frequency and pulse sigma default to cfg.device.coupler.pulses.hpi
(same hardware entry used by ramsey_coupler), so once the Ramsey calibrates
the drive, the same settings flow into this rabi. cfg.expt.{freq, sigma}
may override ad-hoc during bringup.
"""


class AmplitudeRabiCouplerProgram(MMRAveragerProgram):
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
        man_no = int(cfg.expt.get('man_mode_no', 1))

        pulse_type = cfg.expt.get('pulse_type', 'gauss')
        _validate_pulse_type(pulse_type)
        self.pulse_type = pulse_type

        prep_seq = self.prep_man_fock_state(man_no, '1')
        self.prep_pulse = self.get_prepulse_creator(prep_seq).pulse.tolist()

        swap_seq = [
            ['multiphoton', f'f0-g{man_no}', 'pi', 0],
            ['multiphoton', 'e0-f0', 'pi', 0],
        ]
        self.swap_pulse = self.get_prepulse_creator(swap_seq).pulse.tolist()

        if pulse_type == 'gauss':
            freq_MHz, sigma_us = _resolve_pulse_params(cfg, names=('freq', 'sigma'))
            self.drive_sigma = self.us2cycles(sigma_us, gen_ch=self.man_ch[qTest])
            self.add_gauss(ch=self.man_ch[qTest], name="coupler_rabi",
                           sigma=self.drive_sigma, length=self.drive_sigma * 4)
        elif pulse_type == 'const':
            freq_MHz, length_us = _resolve_pulse_params(cfg, names=('freq', 'length'))
            self.drive_length = self.us2cycles(length_us, gen_ch=self.man_ch[qTest])
        else:
            raise ValueError(f"unreachable: pulse_type={pulse_type!r}")
        self.f_drive = self.freq2reg(freq_MHz, gen_ch=self.man_ch[qTest])

        # manipulate DAC is 'full'-type, so direct gain register works.
        self.man_rp = self.ch_page(self.man_ch[qTest])
        self.r_gain = self.sreg(self.man_ch[qTest], "gain")
        self.r_gain3 = 4
        self.safe_regwi(self.man_rp, self.r_gain3, int(cfg.expt.start))

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

        if self.pulse_type == 'gauss':
            self.set_pulse_registers(
                ch=self.man_ch[qTest],
                style="arb",
                freq=self.f_drive,
                phase=0,
                gain=0,  # overwritten by mathi from r_gain3
                waveform="coupler_rabi",
            )
        elif self.pulse_type == 'const':
            self.set_pulse_registers(
                ch=self.man_ch[qTest],
                style="const",
                freq=self.f_drive,
                phase=0,
                gain=0,  # overwritten by mathi from r_gain3
                length=self.drive_length,
            )
        else:
            raise ValueError(f"unreachable: pulse_type={self.pulse_type!r}")

        self.mathi(self.man_rp, self.r_gain, self.r_gain3, "+", 0)
        self.pulse(ch=self.man_ch[qTest])
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

    def update(self):
        self.mathi(self.man_rp, self.r_gain3, self.r_gain3, '+', int(self.cfg.expt.step))


class AmplitudeRabiCouplerExperiment(Experiment):
    """
    Amplitude Rabi on the coupler, driven via the manipulate channel.

    Prepares |1> in the manipulate mode, plays a pulse on the manipulate
    DAC at a fixed drive frequency with swept gain, then swaps
    |1> -> |e> on the qubit for readout.

    Experimental Config:
    expt = dict(
        start:        gain start [DAC units]
        step:         gain step  [DAC units]
        expts:        number of gain points
        reps:         averages per gain point
        rounds:       sweep repetitions
        qubits:       list, e.g. [0]
        man_mode_no:  (optional, default 1) manipulate mode index used
                      for Fock-|1> prep and swap-out
        pulse_type:   (optional, default 'gauss') pulse envelope. Supported:
                      'gauss' (uses sigma; length = sigma*4),
                      'const' (uses length).
                      Other values raise.
        freq:         (optional) override cfg.device.coupler.pulses.hpi.freq[0]   [MHz]
        sigma:        (optional, pulse_type='gauss') override
                      cfg.device.coupler.pulses.hpi.sigma[0]  [us]
        length:       (optional, pulse_type='const') override
                      cfg.device.coupler.pulses.hpi.length[0] [us]
        prepulse:        (optional) bool, play cfg.expt.pre_sweep_pulse first
        pre_sweep_pulse: (optional) gate-list spec for the custom prepulse
        active_reset:    (optional) bool, run active reset at start of body
    )
    """

    def __init__(self, soccfg=None, path='', prefix='AmplitudeRabiCoupler',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        _validate_pulse_type(self.cfg.expt.get('pulse_type', 'gauss'))

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

        amprabi = AmplitudeRabiCouplerProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = amprabi.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            debug=debug,
            readouts_per_experiment=read_num,
        )

        avgi = avgi[0][-1]
        avgq = avgq[0][-1]
        amps = np.abs(avgi + 1j * avgq)
        phases = np.angle(avgi + 1j * avgq)

        data = {'xpts': x_pts, 'avgi': avgi, 'avgq': avgq,
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

            for key, p in (('avgi', p_avgi), ('avgq', p_avgq), ('amps', p_amps)):
                if isinstance(p, (list, np.ndarray)):
                    period = 1.0 / p[1]
                    phase_rad = p[2] * np.pi / 180
                    pi_gain = (0.5 - phase_rad / np.pi) * period
                    hpi_gain = (0.25 - phase_rad / np.pi) * period
                    data[f'pi_gain_{key}'] = pi_gain
                    data[f'hpi_gain_{key}'] = hpi_gain
        return data

    def display(self, data=None, fit=True, vline=None, **kwargs):
        if data is None:
            data = self.data

        plt.figure(figsize=(10, 9))
        plt.subplot(
            211,
            title='Coupler Amplitude Rabi (via manipulate channel)',
            ylabel="I [ADC units]",
        )
        plt.plot(data["xpts"][1:-1], data["avgi"][1:-1], 'o-')
        if fit:
            p = data.get('fit_avgi')
            if isinstance(p, (list, np.ndarray)):
                plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
                pi_gain = data['pi_gain_avgi']
                hpi_gain = data['hpi_gain_avgi']
                print(f'Pi gain from avgi data [DAC units]: {pi_gain}')
                print(f'\tPi/2 gain from avgi data [DAC units]: {hpi_gain}')
                plt.axvline(pi_gain, color='0.2', linestyle='--')
                plt.axvline(hpi_gain, color='0.2', linestyle='--')
                if vline is not None:
                    plt.axvline(vline, color='0.2', linestyle='--')

        plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][1:-1], data["avgq"][1:-1], 'o-')
        if fit:
            p = data.get('fit_avgq')
            if isinstance(p, (list, np.ndarray)):
                plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
                pi_gain = data['pi_gain_avgq']
                hpi_gain = data['hpi_gain_avgq']
                print(f'Pi gain from avgq data [DAC units]: {pi_gain}')
                print(f'\tPi/2 gain from avgq data [DAC units]: {hpi_gain}')
                plt.axvline(pi_gain, color='0.2', linestyle='--')
                plt.axvline(hpi_gain, color='0.2', linestyle='--')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
