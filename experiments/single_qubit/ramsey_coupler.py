import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import Experiment, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram

"""
Ramsey on the coupler, driven via the manipulate channel.

Sequence
--------
  1. Prepare manipulate mode (default M1) in Fock |1>.
  2. First pi/2 gaussian pulse on the manipulate DAC at the coupler drive freq.
  3. Variable wait tau (with register-stepped phase advance).
  4. Second pi/2 gaussian pulse on the manipulate DAC with phase = 360 * ramsey_freq * tau.
  5. Swap manipulate |1> back to qubit |e> (f0-g1 pi + e0-f0 pi).
  6. Dispersive readout.

Like pulse_probe_coupler_spectroscopy, the Ramsey tones ride the manipulate
DAC because that is the path already wired and calibrated. The pi/2 pulse
parameters (freq, sigma, gain) default to cfg.device.coupler.pulses.hpi
- added to the versioned hardware config via station.snapshot_hardware_config -
and can be overridden ad-hoc from cfg.expt during bringup.
"""


SUPPORTED_PULSE_TYPES = ('gauss', 'const')


def _validate_pulse_type(pulse_type):
    if pulse_type not in SUPPORTED_PULSE_TYPES:
        raise ValueError(
            f"coupler experiments: pulse_type={pulse_type!r} not supported. "
            f"Currently implemented: {SUPPORTED_PULSE_TYPES}."
        )


def _resolve_pulse_params(cfg, names=('freq', 'sigma', 'gain')):
    """Return pulse params for the coupler pi/2 gaussian, one per name.

    Priority: cfg.expt overrides > cfg.device.coupler.pulses.hpi defaults.
    `names` selects which fields to resolve (e.g. Rabi sweeps gain, so it
    only asks for ('freq', 'sigma')).
    """
    coupler_hpi = None
    device = cfg.get('device') if isinstance(cfg, dict) else getattr(cfg, 'device', None)
    if device is not None:
        coupler = device.get('coupler') if isinstance(device, dict) else getattr(device, 'coupler', None)
        if coupler is not None:
            pulses = coupler.get('pulses') if isinstance(coupler, dict) else getattr(coupler, 'pulses', None)
            if pulses is not None:
                coupler_hpi = pulses.get('hpi') if isinstance(pulses, dict) else getattr(pulses, 'hpi', None)

    expt = cfg.expt

    def _pick(name):
        if name in expt and expt[name] is not None:
            return expt[name]
        if coupler_hpi is None:
            raise KeyError(
                f"ramsey_coupler: '{name}' missing. Either pass cfg.expt.{name}, "
                "or add device.coupler.pulses.hpi to the hardware config via "
                "station.snapshot_hardware_config(update_main=True)."
            )
        return coupler_hpi[name][0]

    return tuple(_pick(n) for n in names)


class RamseyCouplerProgram(MMRAveragerProgram):
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

        prep_seq = self.prep_man_fock_state(man_no, '1')
        self.prep_pulse = self.get_prepulse_creator(prep_seq).pulse.tolist()

        swap_seq = [
            ['multiphoton', f'f0-g{man_no}', 'pi', 0],
            ['multiphoton', 'e0-f0', 'pi', 0],
        ]
        self.swap_pulse = self.get_prepulse_creator(swap_seq).pulse.tolist()

        pulse_type = cfg.expt.get('pulse_type', 'const')
        _validate_pulse_type(pulse_type)
        self.pulse_type = pulse_type

        if pulse_type == 'gauss':
            freq_MHz, sigma_us, gain = _resolve_pulse_params(
                cfg, names=('freq', 'sigma', 'gain'))
            self.drive_sigma = self.us2cycles(sigma_us, gen_ch=self.man_ch[qTest])
            self.add_gauss(ch=self.man_ch[qTest], name="coupler_pi2_ram",
                           sigma=self.drive_sigma, length=self.drive_sigma * 4)
        elif pulse_type == 'const':
            freq_MHz, length_us, gain = _resolve_pulse_params(
                cfg, names=('freq', 'length', 'gain'))
            self.drive_length = self.us2cycles(length_us, gen_ch=self.man_ch[qTest])
        else:
            raise ValueError(f"unreachable: pulse_type={pulse_type!r}")

        self.drive_gain = gain
        self.f_drive = self.freq2reg(freq_MHz, gen_ch=self.man_ch[qTest])

        # manipulate DAC is 'full'-type; simple phase reg works, no int4 shift needed.
        self.man_rp = self.ch_page(self.man_ch[qTest])
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase = self.sreg(self.man_ch[qTest], "phase")

        self.safe_regwi(self.man_rp, self.r_wait,
                        self.us2cycles(cfg.expt.start, gen_ch=self.man_ch[qTest]))
        self.safe_regwi(self.man_rp, self.r_phase2, 0)

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
            self.setup_and_pulse(
                ch=self.man_ch[qTest],
                style="arb",
                freq=self.f_drive,
                phase=0,
                gain=self.drive_gain,
                waveform="coupler_pi2_ram",
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
        self.sync(self.man_rp, self.r_wait)

        advance_phase_reg = self.deg2reg(cfg.expt.get('advance_phase', 0),
                                         gen_ch=self.man_ch[qTest])
        if self.pulse_type == 'gauss':
            self.set_pulse_registers(
                ch=self.man_ch[qTest],
                style="arb",
                freq=self.f_drive,
                phase=advance_phase_reg,
                gain=self.drive_gain,
                waveform="coupler_pi2_ram",
            )
        elif self.pulse_type == 'const':
            self.set_pulse_registers(
                ch=self.man_ch[qTest],
                style="const",
                freq=self.f_drive,
                phase=advance_phase_reg,
                gain=self.drive_gain,
                length=self.drive_length,
            )
        else:
            raise ValueError(f"unreachable: pulse_type={self.pulse_type!r}")
        self.mathi(self.man_rp, self.r_phase, self.r_phase2, "+", 0)
        self.sync_all(self.us2cycles(0.01))
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
        qTest = self.qubits[0]

        phase_step = self.deg2reg(
            360 * self.cfg.expt.ramsey_freq * self.cfg.expt.step,
            gen_ch=self.man_ch[qTest],
        )

        self.mathi(self.man_rp, self.r_wait, self.r_wait, '+',
                   self.us2cycles(self.cfg.expt.step, gen_ch=self.man_ch[qTest]))
        self.sync_all(self.us2cycles(0.01))
        self.mathi(self.man_rp, self.r_phase2, self.r_phase2, '+', phase_step)
        self.sync_all(self.us2cycles(0.01))


class RamseyCouplerExperiment(Experiment):
    """
    Ramsey on the coupler, driven via the manipulate channel.

    Prepares |1> in the manipulate mode, plays a two-pi/2-pulse Ramsey
    sequence on the manipulate DAC at a fixed drive frequency, sweeps the
    wait time tau, then swaps |1> -> |e> on the qubit for readout.

    Experimental Config:
    expt = dict(
        start:        wait time start [us]
        step:         wait time step  [us]  (Nyquist: 0.5/step > ramsey_freq)
        expts:        number of wait-time points
        ramsey_freq:  phase advance rate for the second pi/2 [MHz]
        reps:         averages per wait-time point
        rounds:       sweep repetitions
        qubits:       list, e.g. [0]
        man_mode_no:  (optional, default 1) manipulate mode index used
                      for Fock-|1> prep and swap-out
        pulse_type:   (optional, default 'const') pulse envelope. Supported:
                      'gauss' (uses sigma; length = sigma*4),
                      'const' (uses length).
                      Other values raise.
        freq:         (optional) override cfg.device.coupler.pulses.hpi.freq[0]   [MHz]
        sigma:        (optional, pulse_type='gauss') override
                      cfg.device.coupler.pulses.hpi.sigma[0]  [us]
        length:       (optional, pulse_type='const') override
                      cfg.device.coupler.pulses.hpi.length[0] [us]
        gain:         (optional) override cfg.device.coupler.pulses.hpi.gain[0]   [DAC]
        advance_phase:   (optional, default 0) extra phase on pi/2 #2 [deg]
        prepulse:        (optional) bool, play cfg.expt.pre_sweep_pulse first
        pre_sweep_pulse: (optional) gate-list spec for the custom prepulse
        active_reset:    (optional) bool, run active reset at start of body
    )
    """

    def __init__(self, soccfg=None, path='', prefix='RamseyCoupler',
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

        ramsey = RamseyCouplerProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = ramsey.acquire(
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

            ramsey_freq = self.cfg.expt.ramsey_freq
            if isinstance(p_avgi, (list, np.ndarray)):
                data['f_adjust_ramsey_avgi'] = sorted(
                    (ramsey_freq - p_avgi[1], ramsey_freq + p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)):
                data['f_adjust_ramsey_avgq'] = sorted(
                    (ramsey_freq - p_avgq[1], ramsey_freq + p_avgq[1]), key=abs)
            if isinstance(p_amps, (list, np.ndarray)):
                data['f_adjust_ramsey_amps'] = sorted(
                    (ramsey_freq - p_amps[1], ramsey_freq + p_amps[1]), key=abs)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        f_drive, = _resolve_pulse_params(self.cfg, names=('freq',))
        title = 'Coupler Ramsey (via manipulate channel)'

        plt.figure(figsize=(10, 9))
        plt.subplot(
            211,
            title=f"{title} (Ramsey Freq: {self.cfg.expt.ramsey_freq} MHz)",
            ylabel="I [ADC level]",
        )
        plt.plot(data["xpts"][:-1], data["avgi"][:-1], 'o-')
        if fit:
            p = data.get('fit_avgi')
            if isinstance(p, (list, np.ndarray)):
                pCov = data['fit_err_avgi']
                try:
                    captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\\pm$ {np.sqrt(pCov[3][3]):.3}'
                except ValueError:
                    captionStr = 'fit failed'
                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1],
                         fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1],
                         fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.legend()
                print(f'Current coupler drive frequency: {f_drive}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2 * self.cfg.expt.ramsey_freq:
                    print('WARNING: Fit frequency >2*wR; may be too far from the real transition.')
                print('Suggested new drive frequency from fit I [MHz]:\n',
                      f'\t{f_drive + data["f_adjust_ramsey_avgi"][0]}\n',
                      f'\t{f_drive + data["f_adjust_ramsey_avgi"][1]}')
                print(f'T2 Ramsey from fit I [us]: {p[3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1], 'o-')
        if fit:
            p = data.get('fit_avgq')
            if isinstance(p, (list, np.ndarray)):
                pCov = data['fit_err_avgq']
                try:
                    captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\\pm$ {np.sqrt(pCov[3][3]):.3}'
                except ValueError:
                    captionStr = 'fit failed'
                plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
                plt.plot(data["xpts"][:-1],
                         fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.plot(data["xpts"][:-1],
                         fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]),
                         color='0.2', linestyle='--')
                plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2 * self.cfg.expt.ramsey_freq:
                    print('WARNING: Fit frequency >2*wR; may be too far from the real transition.')
                print('Suggested new drive frequency from fit Q [MHz]:\n',
                      f'\t{f_drive + data["f_adjust_ramsey_avgq"][0]}\n',
                      f'\t{f_drive + data["f_adjust_ramsey_avgq"][1]}')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
