import matplotlib.pyplot as plt
import numpy as np
from qick import *
from slab import Experiment, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import MM_base, MMAveragerProgram, MMRAveragerProgram, warn_step_subcycle

"""
Ramsey / CPMG on the coupler, driven via the manipulate channel.

Sequence
--------
  1. First pi/2 pulse on the manipulate DAC at the coupler drive freq.
  2. Variable wait tau, optionally interrupted by `echoes` CPMG pi pulses
     (Bylander 2011 timing: end-spacings tau/(2N), inter-pi tau/N, pi
     pulses at phase=90 deg for X-coherence refocusing).
  3. Second pi/2 pulse on the manipulate DAC with phase = 360 * ramsey_freq * tau.
  4. Coupler-state-selective readout prep on the qubit:
       g0-e0 pi + e0-f0 pi + f0-g{man_no} pi + e0-f0 pi.
     Coupler |g> -> qubit |g>; coupler |e> -> qubit |e>.
  5. Dispersive readout.

M1 is kept empty throughout so it does not chi-shift the coupler or decay
during the wait. The pi/2 pulse defaults to cfg.device.coupler.pulses.hpi;
the CPMG pi pulse to cfg.device.coupler.pulses.pi. Both pulses share the
same drive frequency (the coupler transition is the same regardless of
rotation angle), taken from the hpi entry / `freq` override. If only one
pulse is calibrated, the other is derived by scaling gain by 0.5x (hpi
from pi) or 2x (pi from hpi), with a warning — accurate only in the
DAC/amplifier linear regime. cfg.expt overrides win: bare keys (`freq`,
`sigma`/`length`, `gain`) for the hpi pulse, `pi_`-prefixed keys
(`pi_sigma`/`pi_length`, `pi_gain`) for the CPMG pi pulse envelope and
amplitude.
"""


SUPPORTED_PULSE_TYPES = ('gauss', 'const')


def _validate_pulse_type(pulse_type):
    if pulse_type not in SUPPORTED_PULSE_TYPES:
        raise ValueError(
            f"coupler experiments: pulse_type={pulse_type!r} not supported. "
            f"Currently implemented: {SUPPORTED_PULSE_TYPES}."
        )


def _read_coupler_pulse(cfg, pulse_kind):
    """Return the cfg.device.coupler.pulses[pulse_kind] AttrDict, or None."""
    device = cfg.get('device') if isinstance(cfg, dict) else getattr(cfg, 'device', None)
    if device is None:
        return None
    coupler = device.get('coupler') if isinstance(device, dict) else getattr(device, 'coupler', None)
    if coupler is None:
        return None
    pulses = coupler.get('pulses') if isinstance(coupler, dict) else getattr(coupler, 'pulses', None)
    if pulses is None:
        return None
    return pulses.get(pulse_kind) if isinstance(pulses, dict) else getattr(pulses, pulse_kind, None)


def _resolve_pulse_params(cfg, names=('freq', 'sigma', 'gain'), pulse_kind='hpi'):
    """Return pulse params for the coupler hpi (pi/2) or pi pulse.

    Priority:
      1. cfg.expt overrides — bare names for pulse_kind='hpi', 'pi_'-prefixed
         names (e.g. cfg.expt.pi_gain) for pulse_kind='pi'.
      2. cfg.device.coupler.pulses.{pulse_kind}
      3. cfg.device.coupler.pulses.{other_kind} with `gain` rescaled
         (0.5x for hpi-from-pi, 2x for pi-from-hpi). Warns on stderr.
      4. cfg.expt overrides for the OTHER kind with `gain` rescaled the
         same way. Lets a bringup notebook drive both pulses off a single
         set of hpi overrides without snapshotting the hardware config.
         Warns on stderr.

    The 3x/4x fallbacks rely on the DAC/amp linear-response assumption —
    accurate only when both pulses are well inside the linear regime.

    `names` selects which fields to resolve (e.g. Rabi sweeps gain, so it only
    asks for ('freq', 'sigma')). Raises KeyError if no source supplies the
    requested field.
    """
    if pulse_kind not in ('hpi', 'pi'):
        raise ValueError(f"pulse_kind must be 'hpi' or 'pi', got {pulse_kind!r}")
    other_kind = 'pi' if pulse_kind == 'hpi' else 'hpi'
    override_prefix = '' if pulse_kind == 'hpi' else 'pi_'
    other_override_prefix = 'pi_' if pulse_kind == 'hpi' else ''
    gain_scale = 0.5 if pulse_kind == 'hpi' else 2.0

    primary = _read_coupler_pulse(cfg, pulse_kind)
    fallback = _read_coupler_pulse(cfg, other_kind) if primary is None else None
    if fallback is not None:
        print(
            f"[ramsey_coupler] WARNING: device.coupler.pulses.{pulse_kind} missing; "
            f"deriving {pulse_kind} from coupler.pulses.{other_kind} with "
            f"gain * {gain_scale} (linear-response assumption — verify pulse not "
            "near DAC/amp saturation)."
        )

    expt = cfg.expt
    _override_fallback_warned = [False]

    def _pick(name):
        override_key = f'{override_prefix}{name}'
        if override_key in expt and expt[override_key] is not None:
            return expt[override_key]
        if primary is not None:
            return primary[name][0]
        if fallback is not None:
            val = fallback[name][0]
            if name == 'gain':
                val = int(round(val * gain_scale))
            return val
        # Last-resort: derive from cfg.expt overrides for the other kind
        # (e.g. CPMG pi from bare hpi overrides during bringup).
        other_key = f'{other_override_prefix}{name}'
        if other_key in expt and expt[other_key] is not None:
            if not _override_fallback_warned[0]:
                print(
                    f"[ramsey_coupler] WARNING: device.coupler.pulses.{{{pulse_kind},"
                    f"{other_kind}}} both missing; deriving {pulse_kind} from "
                    f"cfg.expt {other_kind} overrides with gain * {gain_scale} "
                    "(linear-response assumption — verify pulse not near DAC/amp "
                    "saturation)."
                )
                _override_fallback_warned[0] = True
            val = expt[other_key]
            if name == 'gain':
                val = int(round(val * gain_scale))
            return val
        raise KeyError(
            f"ramsey_coupler ({pulse_kind}): '{name}' missing. Provide "
            f"cfg.expt.{override_key}, cfg.expt.{other_key}, or add "
            f"device.coupler.pulses.{pulse_kind} "
            f"(or .{other_kind} as a fallback) to the hardware config via "
            "station.snapshot_hardware_config(update_main=True)."
        )

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
        print(
        f"man_ch={self.man_ch[qTest]} "
        f"freq_reg={self.sreg(self.man_ch[qTest], 'freq')} "
        f"phase_reg={self.sreg(self.man_ch[qTest], 'phase')} "
        f"gain_reg={self.sreg(self.man_ch[qTest], 'gain')}"
        )

        self.num_echoes = int(cfg.expt.get('echoes', 0))
        # Subcycle warning: smallest sub-segment is tau/(2N) under CPMG; tau
        # for plain Ramsey.
        if self.num_echoes >= 1:
            min_step = cfg.expt.step / (2 * self.num_echoes)
            warn_label = f"step/(2*echoes={self.num_echoes}) [CPMG]"
        else:
            min_step = cfg.expt.step
            warn_label = "ramsey step"
        warn_step_subcycle(self.soccfg, min_step,
                           gen_ch=self.man_ch[qTest], label=warn_label)
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

        pulse_type = cfg.expt.get('pulse_type', 'const')
        _validate_pulse_type(pulse_type)
        self.pulse_type = pulse_type

        if pulse_type == 'gauss':
            freq_MHz, sigma_us, gain = _resolve_pulse_params(
                cfg, names=('freq', 'sigma', 'gain'), pulse_kind='hpi')
            print(f"drive freq={freq_MHz} MHz, sigma={sigma_us} us, gain={gain} DAC ")
            self.drive_sigma = self.us2cycles(sigma_us, gen_ch=self.man_ch[qTest])
            self.add_gauss(ch=self.man_ch[qTest], name="coupler_pi2_ram",
                           sigma=self.drive_sigma, length=self.drive_sigma * 4)
        elif pulse_type == 'const':
            freq_MHz, length_us, gain = _resolve_pulse_params(
                cfg, names=('freq', 'length', 'gain'), pulse_kind='hpi')
            self.drive_length = self.us2cycles(length_us, gen_ch=self.man_ch[qTest])
        else:
            raise ValueError(f"unreachable: pulse_type={pulse_type!r}")

        self.drive_gain = gain
        self.f_drive = self.freq2reg(freq_MHz, gen_ch=self.man_ch[qTest])

        # CPMG pi pulse setup (only needed when echoes >= 1). Echo pulse
        # always uses the same drive frequency as the pi/2 (same coupler
        # transition); only envelope and gain are pi-specific.
        if self.num_echoes >= 1:
            if pulse_type == 'gauss':
                pi_sigma_us, pi_gain = _resolve_pulse_params(
                    cfg, names=('sigma', 'gain'), pulse_kind='pi')
                self.echo_sigma = self.us2cycles(
                    pi_sigma_us, gen_ch=self.man_ch[qTest])
                self.add_gauss(ch=self.man_ch[qTest], name="coupler_pi_echo",
                               sigma=self.echo_sigma,
                               length=self.echo_sigma * 4)
            else:  # const
                pi_length_us, pi_gain = _resolve_pulse_params(
                    cfg, names=('length', 'gain'), pulse_kind='pi')
                self.echo_length = self.us2cycles(
                    pi_length_us, gen_ch=self.man_ch[qTest])
            self.echo_gain = pi_gain

        # manipulate DAC is 'full'-type; simple phase reg works, no int4 shift needed.
        self.man_rp = self.ch_page(self.man_ch[qTest])
        # r_wait     = end-segment (between pi/2 and first/last pi) = tau/(2N)
        # r_wait_mid = inter-pi segment (between consecutive pi's)  = tau/N
        # For Ramsey (N=0) only r_wait is used, holding the full tau.
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase_step = 5
        self.r_wait_mid = 6
        self.r_phase = self.sreg(self.man_ch[qTest], "phase")

        if self.num_echoes >= 1:
            self.safe_regwi(
                self.man_rp, self.r_wait,
                self.us2cycles(cfg.expt.start / (2 * self.num_echoes),
                               gen_ch=self.man_ch[qTest]))
            self.safe_regwi(
                self.man_rp, self.r_wait_mid,
                self.us2cycles(cfg.expt.start / self.num_echoes,
                               gen_ch=self.man_ch[qTest]))
        else:
            self.safe_regwi(
                self.man_rp, self.r_wait,
                self.us2cycles(cfg.expt.start, gen_ch=self.man_ch[qTest]))
        self.safe_regwi(self.man_rp, self.r_phase2, 0)

        # Phase step register (register-register math avoids mathi 31-bit limit
        # when ramsey_freq*step wraps deg2reg into [2^31, 2^32)).
        phase_step_val = self.deg2reg(
            360 * abs(cfg.expt.ramsey_freq) * cfg.expt.step,
            gen_ch=self.man_ch[qTest])
        self.safe_regwi(self.man_rp, self.r_phase_step, phase_step_val)
        self.ramsey_freq_sign = 1 if cfg.expt.ramsey_freq >= 0 else -1

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # Phase-reset every channel's DDS accumulator to a common reference
        # (phrst=1 zero-gain pulse on each ch). Without this the manipulate
        # channel's accumulator drifts shot-to-shot and the per-pulse `phase`
        # parameter on the second pi/2 has no fixed reference, so phase sweeps
        # collapse to a constant readout. Every other Ramsey in this repo
        # (t2_ramsey, t2_cavity_*, t2_cavity_displacement, ...) calls this.
        self.reset_and_sync()

        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        self.sync_all()

        if cfg.expt.get('prepulse', False):
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre_')

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

        # CPMG echo loop: pi pulses at phase=90 deg (Y-axis rotation, refocuses
        # X-axis coherence prepared by the first phase=0 pi/2).
        # Layout: pi/2 -> sync(r_wait) -> pi_1 -> sync(r_wait_mid) -> pi_2
        #         -> ... -> sync(r_wait_mid) -> pi_N -> sync(r_wait) -> pi/2.
        if self.num_echoes >= 1:
            echo_phase = self.deg2reg(90, gen_ch=self.man_ch[qTest])
            for k in range(self.num_echoes):
                if self.pulse_type == 'gauss':
                    self.set_pulse_registers(
                        ch=self.man_ch[qTest],
                        style="arb",
                        freq=self.f_drive,
                        phase=echo_phase,
                        gain=self.echo_gain,
                        waveform="coupler_pi_echo",
                    )
                else:  # const
                    self.set_pulse_registers(
                        ch=self.man_ch[qTest],
                        style="const",
                        freq=self.f_drive,
                        phase=echo_phase,
                        gain=self.echo_gain,
                        length=self.echo_length,
                    )
                self.sync_all(self.us2cycles(0.01))
                self.pulse(ch=self.man_ch[qTest])
                self.sync_all()
                is_last = (k == self.num_echoes - 1)
                wait_reg = self.r_wait if is_last else self.r_wait_mid
                self.sync(self.man_rp, wait_reg)
                self.sync_all()

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

        if self.num_echoes >= 1:
            step_end_cycles = self.us2cycles(
                self.cfg.expt.step / (2 * self.num_echoes),
                gen_ch=self.man_ch[qTest])
            step_mid_cycles = self.us2cycles(
                self.cfg.expt.step / self.num_echoes,
                gen_ch=self.man_ch[qTest])
            self.mathi(self.man_rp, self.r_wait, self.r_wait, '+',
                       step_end_cycles)
            self.sync_all(self.us2cycles(0.01))
            self.mathi(self.man_rp, self.r_wait_mid, self.r_wait_mid, '+',
                       step_mid_cycles)
        else:
            self.mathi(self.man_rp, self.r_wait, self.r_wait, '+',
                       self.us2cycles(self.cfg.expt.step,
                                      gen_ch=self.man_ch[qTest]))
        self.sync_all(self.us2cycles(0.01))
        op = '+' if self.ramsey_freq_sign >= 0 else '-'
        self.math(self.man_rp, self.r_phase2, self.r_phase2, op, self.r_phase_step)
        self.sync_all(self.us2cycles(0.01))


class RamseyCouplerExperiment(Experiment):
    """
    Ramsey on the coupler, driven via the manipulate channel.

    Plays a two-pi/2-pulse Ramsey sequence on the manipulate DAC at a fixed
    drive frequency, sweeps the wait time tau, then does a coupler-state-
    selective readout prep (g0-e0 pi + e0-f0 pi + f0-g{man_no} pi + e0-f0
    pi) that maps coupler |g> -> qubit |g> and coupler |e> -> qubit |e>.
    M1 is kept empty throughout.

    Experimental Config:
    expt = dict(
        start:        wait time start [us]
        step:         wait time step  [us]  (Nyquist: 0.5/step > ramsey_freq)
        expts:        number of wait-time points
        ramsey_freq:  phase advance rate for the second pi/2 [MHz]
        reps:         averages per wait-time point
        rounds:       sweep repetitions
        qubits:       list, e.g. [0]
        man_mode_no:  (optional, default 1) manipulate mode index whose
                      f0-g{man_no} transition is used for coupler-state
                      selective readout. M1 stays empty throughout.
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
        echoes:       (optional, default 0) number of CPMG pi pulses inserted
                      between the two pi/2 pulses. 0 = plain Ramsey. >=1 uses
                      standard CPMG timing (Bylander 2011 Eq. 18): end-spacings
                      tau/(2N), inter-pi tau/N, pi pulses at phase=90 deg.
                      The echo pi pulse reuses the pi/2 drive frequency
                      (`freq` above or coupler.pulses.hpi.freq[0]); only its
                      envelope and gain are pi-specific.
        pi_sigma:     (optional, pulse_type='gauss') override
                      cfg.device.coupler.pulses.pi.sigma[0]   [us]
        pi_length:    (optional, pulse_type='const') override
                      cfg.device.coupler.pulses.pi.length[0]  [us]
        pi_gain:      (optional) override cfg.device.coupler.pulses.pi.gain[0]    [DAC]
                      If coupler.pulses.pi is missing entirely, the pi params
                      fall back to coupler.pulses.hpi (with gain * 2). If both
                      are missing, the pi pulse derives from the bare hpi
                      overrides (`sigma`/`length`, `gain`) with gain * 2.
                      Warning printed; assumes DAC/amp linear response.
        f0g1_freq:    (optional) override cfg.device.multiphoton['pi']['fn-gn+1'].frequency[0]
                      [MHz] for the readout swap pulse only. Used when characterizing the
                      coupler at a flux current where the calibrated f0g1 readout pulse
                      differs from the cfg-default value.
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
        if data is None:
            data = self.data
        print(f'Saving {self.fname}')
        super().save_data(data=data)
