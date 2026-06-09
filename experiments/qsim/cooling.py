import json
import os

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.MM_base import MMAveragerProgram
from experiments.MM_dual_rail_base import MM_dual_rail_base

"""
cooling
"""

class CoolingSpectroscopyProgram(QsimBaseProgram):
    # DRIVE_CHANNEL = 3 # man drive
    # DRIVE_CHANNEL = 6 # stor drive
    FLUX_CHANNEL_LOW = 1 # flux low drive
    FLUX_CHANNEL_HIGH = 4 # flux high drive
    CHARGE_CHANNEL = 3 # man drive

    def initialize(self):
        ecfg = self.cfg.expt
        self.FLUX_CHANNEL = self.FLUX_CHANNEL_LOW if ecfg.cooling_freq < 4000 else self.FLUX_CHANNEL_HIGH
        self.FLUX_NQZ = 1 if ecfg.cooling_freq < 4000 else 2

        super().initialize()
        self.declare_gen(ch=self.FLUX_CHANNEL, nqz=self.FLUX_NQZ)
        self.declare_gen(ch=self.CHARGE_CHANNEL, nqz=2)
        # self.declare_gen(ch=self.DRIVE_CHANNEL, nqz=2) # bad hard coded :(
        # self.declare_gen(ch=self.FLUX_CHANNEL, nqz=2, mixer_freq=7000, mux_freqs=[0])

        ramp_sigma = self.cfg.expt.get('ramp_sigma', 0.005)
        flux_ramp_sigma = self.us2cycles(ramp_sigma, gen_ch=self.FLUX_CHANNEL)
        charge_ramp_sigma = self.us2cycles(ramp_sigma, gen_ch=self.CHARGE_CHANNEL)
        self.add_gauss(ch=self.CHARGE_CHANNEL, name="cooling_charge",
                       sigma=charge_ramp_sigma, length=charge_ramp_sigma*6)
        self.add_gauss(ch=self.FLUX_CHANNEL, name="cooling_flux",
                       sigma=flux_ramp_sigma, length=flux_ramp_sigma*6)

    def core_pulses(self):
        ecfg = self.cfg.expt
        # spec_pulse = [
        #     [ecfg.cooling_freq],
        #     [ecfg.cooling_gain],
        #     [ecfg.cooling_length],
        #     [0],
        #     [self.DRIVE_CHANNEL],
        #     ['flat_top'],
        #     [self.cfg.device.storage.ramp_sigma],
        # ]
        # self.custom_pulse(self.cfg, spec_pulse, prefix='cool_')
        # [[frequency], [gain], [length (us)], [phases],
        # [drive channel], [shape], [ramp sigma]]
        #

        self.set_pulse_registers(
            ch=self.FLUX_CHANNEL, style="flat_top",
            freq=self.freq2reg(ecfg.cooling_freq, gen_ch=self.FLUX_CHANNEL),
            phase=0, gain=ecfg.cooling_gain,
            length=self.us2cycles(ecfg.cooling_length, gen_ch=self.FLUX_CHANNEL),
            waveform="cooling_flux",
        )

        self.set_pulse_registers(
            ch=self.CHARGE_CHANNEL, style="flat_top",
            freq=self.freq2reg(ecfg.charge_freq, gen_ch=self.CHARGE_CHANNEL),
            phase=0, gain=ecfg.charge_gain,
            length=self.us2cycles(ecfg.cooling_length, gen_ch=self.CHARGE_CHANNEL),
            waveform="cooling_charge",
        )

        self.pulse(self.CHARGE_CHANNEL)
        self.pulse(self.FLUX_CHANNEL)
        self.sync_all(self.us2cycles(0.01))


class _CoolingBase(QsimBaseProgram):
    """
    Shared building blocks for the cooling probe sequence and its f0g1
    calibration. Both run with cfg.expt.prepulse = cfg.expt.postpulse = False:
    the whole sequence lives in core_pulses(), and the base body() still
    provides reset_and_sync(), optional active_reset, and the final readout via
    measure_wrapper().

    The "modified" f0g1 (its freq/gain/length supplied via cfg.expt) sits on the
    same sideband channel and uses the same flat_top shape as the bare f0g1; the
    calibration program sweeps those params, the probe program reuses the
    calibrated values verbatim.

    Modified-f0g1 cfg.expt keys:
        f0g1_mod_freq   : MHz   (modified f0g1 carrier)
        f0g1_mod_gain   : DAC   (modified f0g1 gain)
        f0g1_mod_length : us    (modified f0g1 flat-top length)
    """

    def _play_prep_chain(self, prefix='cool1_'):
        """
        ge pi, ef pi, BARE f0g1 pi, M1-C pi, ge pi, ef pi.

        All dataset-driven gates: qubit pulses from cfg.device.qubit.pulses,
        the bare f0g1 ('man','M1') and the M1->coupler swap ('storage','M1-C')
        from the man1-storage dataset rows of the same name.
        """
        seq = self.get_prepulse_creator([
            ['qubit', 'ge', 'pi', 0],
            ['qubit', 'ef', 'pi', 0],
            ['man', 'M1', 'pi', 0],        # bare f0g1
            ['storage', 'M1-C', 'pi', 0],  # M1 -> coupler
            ['qubit', 'ge', 'pi', 0],
            ['qubit', 'ef', 'pi', 0],
        ])
        self.custom_pulse(self.cfg, seq.pulse, prefix=prefix)
        self.sync_all()

    def _f0g1_mod_pulse(self):
        """
        Build the modified-f0g1 pulse array from cfg.expt. Same sideband channel
        and flat_top shape as the bare f0g1; only carrier/gain/length differ.
        Returns a fresh list each call (custom_pulse mutates the phase row in
        place, so the probe program must not share one instance across plays).
        """
        ecfg = self.cfg.expt
        return [
            [ecfg.f0g1_mod_freq],
            [ecfg.f0g1_mod_gain],
            [ecfg.f0g1_mod_length],
            [0],
            [self.f0g1_ch[0]],
            ['flat_top'],
            [self.cfg.device.manipulate.ramp_sigma],
        ]


class CoolingProbeProgram(_CoolingBase):
    """
    Sequence under investigation:

        ge pi, ef pi, f0g1 pi (BARE),
        M1-C pi,
        ge pi, ef pi, f0g1 pi (MODIFIED),
        probe pulse (const on flux-low),
        f0g1 (MODIFIED),
        readout

    The first f0g1 is the bare f0g1; the second and third are an identical
    *modified* f0g1 (calibrate it first with CoolingF0g1CalProgram).

    Expected cfg.expt keys: the modified-f0g1 keys (see _CoolingBase) plus
        probe_freq      : MHz   (pulse under investigation)
        probe_gain      : DAC
        probe_length    : us
        probe_phase     : deg   (optional, default 0)
    """

    def core_pulses(self):
        cfg = self.cfg
        ecfg = cfg.expt

        # --- pulse under investigation: constant pulse on flux-low ---
        probe = [
            [ecfg.probe_freq],
            [ecfg.probe_gain],
            [ecfg.probe_length],
            [ecfg.get('probe_phase', 0)],
            [self.flux_low_ch[0]],
            ['const'],
            [0],
        ]

        # 1) ge, ef, BARE f0g1, M1-C swap, ge, ef
        self._play_prep_chain(prefix='cool1_')

        # 2) MODIFIED f0g1
        self.custom_pulse(cfg, self._f0g1_mod_pulse(), prefix='f0g1mod_a_')
        self.sync_all()

        # 3) probe pulse under investigation
        self.custom_pulse(cfg, probe, prefix='probe_')
        self.sync_all()

        # 4) MODIFIED f0g1 again (identical params to step 2)
        self.custom_pulse(cfg, self._f0g1_mod_pulse(), prefix='f0g1mod_b_')
        self.sync_all()

        # readout handled by base body() -> measure_wrapper()


class CoolingF0g1CalProgram(_CoolingBase):
    """
    Calibration for the modified f0g1: the CoolingProbeProgram sequence
    truncated right after the *first* modified f0g1, then readout:

        ge pi, ef pi, f0g1 pi (BARE),
        M1-C pi,
        ge pi, ef pi, f0g1 pi (MODIFIED),
        readout

    Sweep f0g1_mod_freq and f0g1_mod_gain (software loops via
    QsimBaseExperiment) to find the carrier/gain that maximize the f0g1
    transfer back to the qubit. f0g1_mod_length is held fixed at the value the
    probe sequence will use.

    Expected cfg.expt keys: the modified-f0g1 keys (see _CoolingBase). No probe
    keys are needed.
    """

    def core_pulses(self):
        cfg = self.cfg

        # 1) ge, ef, BARE f0g1, M1-C swap, ge, ef
        self._play_prep_chain(prefix='cool1_')

        # 2) MODIFIED f0g1 (the pulse being calibrated)
        self.custom_pulse(cfg, self._f0g1_mod_pulse(), prefix='f0g1mod_cal_')
        self.sync_all()

        # readout handled by base body() -> measure_wrapper()


# -----------------------------------------------------------------------------
# Example: run CoolingProbeProgram via QsimBaseExperiment (software loops).
#
# QsimBaseExperiment.acquire() requires a non-empty swept_params, so even a
# "single shot" is expressed as a 1-point sweep. Sweep whatever you want to
# characterize by listing it in swept_params and providing the plural list
# (e.g. swept_params=['probe_length'] + probe_lengths=[...]). 2D sweeps:
# swept_params=['outer','inner'] with both plural lists present.
#
#   expt_params = dict(
#       expts = 1,
#       reps = 1000,
#       rounds = 1,
#       qubits = [0],
#       init_stor = 0,          # read unconditionally by base body()
#       ro_stor = 0,            # "
#       prepulse = False,       # whole sequence lives in core_pulses()
#       postpulse = False,
#       active_reset = False,
#       normalize = False,      # accessed directly in acquire()
#
#       # modified f0g1 (#2 and #3, identical) -- calibrated separately
#       f0g1_mod_freq = 1234.5,     # MHz
#       f0g1_mod_gain = 12000,      # DAC units
#       f0g1_mod_length = 0.5,      # us (flat-top length)
#
#       # pulse under investigation: const on flux-low
#       probe_freq = 800.0,         # MHz
#       probe_gain = 5000,          # DAC units
#       probe_length = 1.0,         # us
#       probe_phase = 0,            # deg (optional)
#
#       # single shot -> 1-point sweep (swap for a real sweep later)
#       swept_params = ['probe_length'],
#       probe_lengths = [1.0],
#   )
#
#   expt = QsimBaseExperiment(
#       soccfg=soc, path=expt_path, config_file=config_path,
#       prefix="CoolingProbe", expt_params=expt_params,
#       program=CoolingProbeProgram, progress=True)
#   expt.go(analyze=False, display=True, progress=True, save=True)
# -----------------------------------------------------------------------------
#
# Example: calibrate the modified f0g1 (2D freq x gain software sweep).
#
#   cal_params = dict(
#       expts = 1,
#       reps = 1000,
#       rounds = 1,
#       qubits = [0],
#       init_stor = 0,
#       ro_stor = 0,
#       prepulse = False,
#       postpulse = False,
#       active_reset = False,
#       normalize = False,
#
#       # length fixed to the value the probe sequence will use; freq/gain swept
#       f0g1_mod_length = 0.5,      # us
#       f0g1_mod_freq = 1234.5,     # MHz  (overwritten each point)
#       f0g1_mod_gain = 12000,      # DAC  (overwritten each point)
#
#       # 2D sweep: outer (y) = freq, inner (x) = gain
#       swept_params = ['f0g1_mod_freq', 'f0g1_mod_gain'],
#       f0g1_mod_freqs = list(np.linspace(1230, 1240, 31)),
#       f0g1_mod_gains = list(range(8000, 16001, 250)),
#   )
#
#   cal = QsimBaseExperiment(
#       soccfg=soc, path=expt_path, config_file=config_path,
#       prefix="CoolingF0g1Cal", expt_params=cal_params,
#       program=CoolingF0g1CalProgram, progress=True)
#   cal.go(analyze=False, display=True, progress=True, save=True)
# -----------------------------------------------------------------------------


