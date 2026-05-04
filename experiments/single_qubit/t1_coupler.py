import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import Experiment, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram, warn_step_subcycle
from experiments.single_qubit.ramsey_coupler import _validate_pulse_type

"""
T1 on the coupler, driven via the manipulate channel.

Sequence
--------
  1. Play the calibrated coupler pi pulse on the manipulate DAC to drive the
     coupler g -> e. Supports 'const' and 'gauss' envelopes.
  2. Wait variable tau (register-stepped).
  3. Coupler-state-selective readout on the qubit:
       g0-e0 pi  (qubit g -> e)
       e0-f0 pi  (qubit e -> f)
       f0-g{man_no} pi  (on-res if coupler in g -> qubit |g>;
                         detuned by chi if coupler in e -> qubit stays |f>)
       e0-f0 pi  (remap |f> -> |e> so g/e readout applies)
  4. Dispersive readout.

M1 is kept empty throughout so it does not chi-shift the coupler or decay
during the wait. The coupler's g-e population shows up as an exponential in
the readout signal versus tau, from which T1 is fit. Pulse params for the
coupler pi come from `device.coupler.pulses.pi_ge` (via
station.snapshot_hardware_config); cfg.expt overrides win ad-hoc.
"""


class T1CouplerProgram(MMRAveragerProgram):
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
        warn_step_subcycle(self.soccfg, cfg.expt.step,
                           gen_ch=self.man_ch[qTest], label="t1 step")
        man_no = int(cfg.expt.get('man_mode_no', 1))

        # Coupler-state-selective readout prep: prepare qubit |f>, then play
        # f0-g{man_no} pi (on-res if coupler |g>, detuned if coupler |e>),
        # then remap |f> -> |e> for the standard g/e readout.
        readout_seq = [
            ['multiphoton', 'g0-e0', 'pi', 0],
            ['multiphoton', 'e0-f0', 'pi', 0],
            ['multiphoton', f'f0-g{man_no}', 'pi', 0],
            ['multiphoton', 'e0-f0', 'pi', 0],
        ]
        f0g1_freq_override = cfg.expt.get('f0g1_freq', None)
        if f0g1_freq_override is not None:
            mp_entry = cfg.device.multiphoton['pi']['fn-gn+1']
            _saved_f0g1 = mp_entry['frequency'][0]
            mp_entry['frequency'][0] = float(f0g1_freq_override)
            try:
                self.swap_pulse = self.get_prepulse_creator(readout_seq).pulse.tolist()
            finally:
                mp_entry['frequency'][0] = _saved_f0g1
        else:
            self.swap_pulse = self.get_prepulse_creator(readout_seq).pulse.tolist()

        # Coupler pi pulse params from device.coupler.pulses.pi_ge (calibrated).
        # cfg.expt.{freq, length, sigma, gain, pulse_type} optionally override ad-hoc.
        try:
            pi_cfg = cfg.device.coupler.pulses.pi_ge
            print("Using coupler pi pulse params from cfg.device.coupler.pulses.pi_ge:")
            print(pi_cfg)
        except (AttributeError, KeyError):
            pi_cfg = None

        def _pick(name):
            if name in cfg.expt and cfg.expt[name] is not None:
                return cfg.expt[name]
            if pi_cfg is None:
                raise KeyError(
                    f"t1_coupler: '{name}' missing. Either pass cfg.expt.{name}, "
                    "or add device.coupler.pulses.pi_ge to the hardware config via "
                    "station.snapshot_hardware_config(update_main=True)."
                )
            return pi_cfg[name][0]

        # Resolve pulse_type: cfg.expt override wins, else pi_ge.type, else 'const'.
        if cfg.expt.get('pulse_type') is not None:
            pulse_type = cfg.expt['pulse_type']
        elif pi_cfg is not None and 'type' in pi_cfg:
            pulse_type = pi_cfg['type'][0]
        else:
            pulse_type = 'const'
        _validate_pulse_type(pulse_type)
        self.pulse_type = pulse_type

        freq_MHz = _pick('freq')
        gain = _pick('gain')
        self.coupler_pi_freq = self.freq2reg(freq_MHz, gen_ch=self.man_ch[qTest])
        self.coupler_pi_gain = int(gain)

        if pulse_type == 'const':
            length_us = _pick('length')
            self.coupler_pi_length = self.us2cycles(length_us, gen_ch=self.man_ch[qTest])
            print(f"Coupler pi pulse (const): freq={freq_MHz} MHz, length={length_us} us, gain={gain}")
        elif pulse_type == 'gauss':
            sigma_us = _pick('sigma')
            sigma_cycles = self.us2cycles(sigma_us, gen_ch=self.man_ch[qTest])
            self.coupler_pi_sigma = sigma_cycles
            self.add_gauss(ch=self.man_ch[qTest], name="coupler_pi",
                           sigma=sigma_cycles, length=sigma_cycles * 4)
            print(f"Coupler pi pulse (gauss): freq={freq_MHz} MHz, sigma={sigma_us} us, gain={gain}")
        else:
            raise ValueError(f"unreachable: pulse_type={pulse_type!r}")

        # Wait-time sweep register on the manipulate channel page.
        self.man_rp = self.ch_page(self.man_ch[qTest])
        self.r_wait = 3
        self.safe_regwi(self.man_rp, self.r_wait,
                        self.us2cycles(cfg.expt.start, gen_ch=self.man_ch[qTest]))

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

        # Step 1: coupler pi pulse.
        if self.pulse_type == 'gauss':
            self.setup_and_pulse(
                ch=self.man_ch[qTest],
                style="arb",
                freq=self.coupler_pi_freq,
                phase=0,
                gain=self.coupler_pi_gain,
                waveform="coupler_pi",
            )
        elif self.pulse_type == 'const':
            self.setup_and_pulse(
                ch=self.man_ch[qTest],
                style="const",
                freq=self.coupler_pi_freq,
                phase=0,
                gain=self.coupler_pi_gain,
                length=self.coupler_pi_length,
            )
        else:
            raise ValueError(f"unreachable: pulse_type={self.pulse_type!r}")

        # Step 2: variable wait for coupler T1 decay.
        self.sync_all()
        self.sync(self.man_rp, self.r_wait)
        self.sync_all(self.us2cycles(0.05))

        # Step 3: coupler-state-selective readout prep
        # (g0-e0 pi + e0-f0 pi + f0-g{man_no} pi + e0-f0 pi).
        self.custom_pulse(cfg, self.swap_pulse, prefix='swap_')

        # Step 4: readout.
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest]),
        )

    def update(self):
        self.mathi(self.man_rp, self.r_wait, self.r_wait, '+',
                   self.us2cycles(self.cfg.expt.step, gen_ch=self.man_ch[self.qubits[0]]))
        self.sync_all(self.us2cycles(0.01))


class T1CouplerExperiment(Experiment):
    """
    T1 on the coupler, driven via the manipulate channel.

    Experimental Config:
    expt = dict(
        start:        wait time start [us]
        step:         wait time step  [us]
        expts:        number of wait-time points
        reps:         averages per wait-time point
        rounds:       sweep repetitions
        qubits:       list, e.g. [0]
        man_mode_no:  (optional, default 1) manipulate mode index whose
                      f0-g{man_no} transition is used for coupler-state
                      selective readout. M1 stays empty throughout.
        pulse_type:   (optional) pulse envelope. Resolution priority:
                      cfg.expt.pulse_type > pi_ge.type[0] > 'const'.
                      Supported: 'const' (uses length),
                                 'gauss' (uses sigma; total length = sigma*4).
                      Other values raise.
        freq:         (optional) override cfg.device.coupler.pulses.pi_ge.freq[0]   [MHz]
        length:       (optional, pulse_type='const') override
                      cfg.device.coupler.pulses.pi_ge.length[0] [us]
        sigma:        (optional, pulse_type='gauss') override
                      cfg.device.coupler.pulses.pi_ge.sigma[0]  [us]
        gain:         (optional) override cfg.device.coupler.pulses.pi_ge.gain[0]   [DAC]
        f0g1_freq:    (optional) override cfg.device.multiphoton['pi']['fn-gn+1'].frequency[0]
                      [MHz] for the readout swap pulse only. Used when characterizing the
                      coupler at a flux current where the calibrated f0g1 readout pulse
                      differs from the cfg-default value.
        prepulse:        (optional) bool, play cfg.expt.pre_sweep_pulse first
        pre_sweep_pulse: (optional) gate-list spec for the custom prepulse
        active_reset:    (optional) bool, run active reset at start of body
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1Coupler',
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

        t1 = T1CouplerProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(
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
            # fitparams = [y-offset, amp, x-offset, decay rate (T1)]
            data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(
                data['xpts'][:-1], data['avgi'][:-1], fitparams=fitparams)
            data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(
                data['xpts'][:-1], data['avgq'][:-1], fitparams=fitparams)
            data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(
                data['xpts'][:-1], data['amps'][:-1], fitparams=fitparams)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        plt.figure(figsize=(10, 9))
        plt.subplot(
            211,
            title='Coupler T1 (via manipulate channel)',
            ylabel="I [ADC units]",
        )
        plt.plot(data["xpts"][:-1], data["avgi"][:-1], 'o-')
        if fit:
            p = data.get('fit_avgi')
            if isinstance(p, (list, np.ndarray)):
                pCov = data['fit_err_avgi']
                try:
                    captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\\pm$ {np.sqrt(pCov[3][3]):.3}'
                except ValueError:
                    captionStr = 'fit failed'
                plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.legend()
                print(f'T1 from avgi [us]: {p[3]}')

        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1], 'o-')
        if fit:
            p = data.get('fit_avgq')
            if isinstance(p, (list, np.ndarray)):
                pCov = data['fit_err_avgq']
                try:
                    captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\\pm$ {np.sqrt(pCov[3][3]):.3}'
                except ValueError:
                    captionStr = 'fit failed'
                plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *p), label=captionStr)
                plt.legend()
                print(f'T1 from avgq [us]: {p[3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
