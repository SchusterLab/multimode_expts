import json
import os
from copy import deepcopy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from qick import QickConfig
from qick.helpers import gauss
from fitting.fit_display_classes import GeneralFitting
from slab import AttrDict, Experiment, dsfit
from slab.experiment import NpEncoder
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.dataset import FloquetStorageSwapDataset
from experiments.qsim.utils import (
    ensure_list_in_cfg,
    post_select_raverager_data,
)
from fitting.fit_utils import guess_freq
from experiments.MM_base import MMAveragerProgram, MMRAveragerProgram


# The last step before the measurement, chosen by cfg.expt.readout
# (docs/qsim/program_tree_plan.md, section 4): mode -> final readouts per shot.
# It plays with or without postpulse. The postpulse decodes: it swaps ro_stor
# into M1, and for 'qubit' it also maps M1 to the qubit (f0-g1, then ef if
# map_to_qubit_ge); the other modes read M1, so the photon stays there.
READOUT_MODES = {
    "qubit": 1,
    "parity": 1,
    "multiparity": 2,       # two parity readouts
    "wigner": 1,            # displace by wigner_alpha, then parity
    "slow_pi_ge": 1,        # number-selective qubit pi: M1 in vacuum or not
}

# The booleans that cfg.expt.readout replaced in step 10D. New configs may not
# set them, not even to False: one key, one way to say it.
RETIRED_READOUT_FLAGS = ("perform_wigner", "parity_readout",
                         "multiparity_readout", "slow_pi_ge_readout")


def readout_mode(expt):
    """-> the final readout of a new expt config; raises if it cannot be played.

    ``post_select_pre_pulse`` is an MM_base key that other experiments play;
    no qsim program does, so it is refused when set (False is allowed, for
    shared default dicts).
    """
    retired = [key for key in RETIRED_READOUT_FLAGS if key in expt]
    if retired:
        raise ValueError(
            f"{retired} were replaced by cfg.expt.readout, one of "
            f"{sorted(READOUT_MODES)} (default 'qubit'); see "
            "docs/qsim/program_tree_plan.md, section 4")
    if expt.get("post_select_pre_pulse", False):
        raise ValueError("post_select_pre_pulse=True: no qsim program plays that readout")
    mode = expt.get("readout", "qubit")
    if mode not in READOUT_MODES:
        raise ValueError(f"readout={mode!r}; expected one of {sorted(READOUT_MODES)}")
    return mode


def saved_readout_mode(expt):
    """-> the final readout of a saved expt config, also one saved before step 10D.

    Old jobs record the retired booleans instead of ``readout``. They are read
    in the order the template played them: ``perform_wigner`` won over the
    parity flags, and ``multiparity_readout`` over ``parity_readout``. Read
    saved configs only here, never in a Program.
    """
    if "readout" in expt:
        return expt["readout"]
    if expt.get("perform_wigner", False):
        return "wigner"
    if expt.get("multiparity_readout", False):
        return "multiparity"
    if expt.get("parity_readout", False):
        return "parity"
    if expt.get("slow_pi_ge_readout", False):
        return "slow_pi_ge"
    return "qubit"


def _herald_readouts(cfg):
    """-> the readouts a shot plays before its final readout."""
    count = 0
    if cfg.expt.get('parity_check', False):
        count += 1
    if cfg.expt.get('active_reset', False):
        params = MMAveragerProgram.get_active_reset_params(cfg)
        count += MMAveragerProgram.active_reset_read_num(**params)
    return count


def readout_lane_count(cfg):
    """-> how many readouts one shot of ``cfg`` produces, for a saved ``cfg`` too.

    A shot is one science measurement plus whatever heralds precede it, so
    the raw single-shot arrays are interleaved with this period and the
    science lane is the last one.

    One definition, because two agreeing copies is one copy plus a
    liability: ``acquire`` writes this into ``cfg.read_num`` at acquisition
    time, and the shot subsampler has to recover the same number from jobs
    saved before that field existed. If the two ever disagree, subsampling
    reads the wrong lane and silently returns other readouts' shots. For a new
    config it equals ``QsimProgram.readouts_per_shot``.

    A config saved before step 10D (no ``readout`` key) gets the count of that
    time: one more for ``multiparity_readout``, whatever else is set. That is
    what the driver asked for, even where the program played fewer (with
    ``perform_wigner`` too, it played only the Wigner readout).
    """
    if "readout" in cfg.expt:
        final = READOUT_MODES[cfg.expt.readout]
    else:
        final = 2 if cfg.expt.get('multiparity_readout', False) else 1
    return _herald_readouts(cfg) + final


def classify_two_parity_readouts(expt, point_idx=0, threshold=None, e_is_high_I=True):
    rn = expt.cfg.read_num
    qTest = expt.cfg.expt.qubits[0]

    if threshold is None:
        threshold = expt.cfg.device.readout.threshold[qTest]

    idata = np.asarray(expt.data['idata'][point_idx])
    qdata = np.asarray(expt.data['qdata'][point_idx])

    i_first  = idata[rn-2::rn]
    q_first  = qdata[rn-2::rn]
    i_second = idata[rn-1::rn]
    q_second = qdata[rn-1::rn]

    if e_is_high_I:
        first_e = i_first > threshold
        second_e = i_second > threshold
    else:
        first_e = i_first < threshold
        second_e = i_second < threshold

    b0 = first_e.astype(int)
    b1 = second_e.astype(int)

    # cond_sec_phase = -90 convention:
    # (g,g)->0, (e,g)->1, (g,e)->2, (e,e)->3
    n_mod4 = b0 + 2*b1

    out = {
        'i_first': i_first,
        'q_first': q_first,
        'i_second': i_second,
        'q_second': q_second,

        'first_e': first_e,
        'second_e': second_e,

        # parity expectation values:
        # +1 means bit=0, -1 means bit=1.
        # first parity = (-1)^n
        # second parity = +1 for n=0,1 mod 4 and -1 for n=2,3 mod 4.
        'parity_first': 1 - 2*b0,
        'parity_second': 1 - 2*b1,

        'n_mod4': n_mod4,

        'p_first_e': np.mean(first_e),
        'p_second_e': np.mean(second_e),

        'p_gg': np.mean((~first_e) & (~second_e)),
        'p_eg': np.mean(( first_e) & (~second_e)),
        'p_ge': np.mean((~first_e) & ( second_e)),
        'p_ee': np.mean(( first_e) & ( second_e)),
    }

    out['p_mod0'] = np.mean(n_mod4 == 0)
    out['p_mod1'] = np.mean(n_mod4 == 1)
    out['p_mod2'] = np.mean(n_mod4 == 2)
    out['p_mod3'] = np.mean(n_mod4 == 3)

    out['mean_parity_first'] = np.mean(out['parity_first'])
    out['mean_parity_second'] = np.mean(out['parity_second'])
    out['mean_n_mod4'] = np.mean(n_mod4)

    return out


class QsimProgram(MMAveragerProgram):
    """
    The qsim template (QsimBaseProgram until step 10F; DarkBaseProgram's copy
    of it merged in at step 10B).

    First initialize a photon into man1 by qubit ge, qubit ef, f0g1 
    Then (optionally) swap into init_stor
    Then do whatever in the core_pulses() that you override
    Finally swap ro_stor back into man and then man into qb and readout
    """
    _pre_selection_filtering = True

    def __init__(self, soccfg: QickConfig, cfg: AttrDict):
        readout_mode(cfg.expt)  # refuse a config it cannot play, before compiling
        super().__init__(soccfg, cfg)

    @classmethod
    def readouts_per_shot(cls, cfg):
        """-> how many readouts one shot of a new ``cfg`` plays: heralds, then the final readout."""
        return _herald_readouts(cfg) + READOUT_MODES[readout_mode(cfg.expt)]


    def retrieve_swap_parameters(self):
        """
        retrieve pulse parameters for the M1-Sx swap
        """
        qTest = self.qubits[0]

        flux_high_threshold = 1800

        stor_names = [f'M1-S{stor_no}' for stor_no in range(1,8)]
        self.m1s_pi_fracs = [self.swap_ds.get_pi_frac(stor_name) for stor_name in stor_names]
        self.m1s_freq_MHz = [self.swap_ds.get_freq(stor_name) for stor_name in stor_names]
        self.m1s_is_low_freq = [self.m1s_freq_MHz[i_stor] < flux_high_threshold for i_stor in range(7)]

        self.m1s_ch = [self.flux_low_ch[qTest] if self.m1s_is_low_freq[i_stor] else self.flux_high_ch[qTest] for i_stor in range(7)]
        self.m1s_freq = [self.freq2reg(freq_MHz, gen_ch=ch) for freq_MHz, ch in zip(self.m1s_freq_MHz, self.m1s_ch)]
        self.m1s_length = [self.us2cycles(self.swap_ds.get_len(stor_name), gen_ch=ch)
            for stor_name, ch in zip(stor_names, self.m1s_ch)]
        self.m1s_gain = [self.swap_ds.get_gain(stor_name) for stor_name in stor_names]
        # Use each mode's calibrated waveform from the dataset.
        self.m1s_waveform_mode = []
        for stor_name in stor_names:
            waveform = self.swap_ds.get_waveform(stor_name)
            if waveform in ('gauss', 'gaussian', 'arb'):
                waveform = 'gauss'
            elif waveform != 'preload_flattop':
                waveform = 'flat_top'
            self.m1s_waveform_mode.append(waveform)

        self.m1s_style = ['flat_top' if mode == 'flat_top' else 'arb'
                          for mode in self.m1s_waveform_mode]

        self.m1s_wf_name = []
        for index, waveform_mode in enumerate(self.m1s_waveform_mode):
            suffix = 'low' if self.m1s_is_low_freq[index] else 'high'
            if waveform_mode == 'gauss':
                waveform_name = f"pi_m1s{index + 1}_gauss_{suffix}"
            elif waveform_mode == 'preload_flattop':
                waveform_name = f"pi_m1s{index + 1}_preload_flattop_{suffix}"
            else:
                waveform_name = f"pi_m1si_{suffix}"
            self.m1s_wf_name.append(waveform_name)
            
    


    def displace_man(self, alpha=None, setup=False, play=False):
        # This function must be called first with setup before calling it with play
        if setup:
            self.gain2alpha = self.cfg.device.manipulate.gain_to_alpha[self.man_mode_idx]
            self.f_man = self.freq2reg(
                self.cfg.device.manipulate.f_ge[self.man_mode_idx],
                gen_ch=self.man_ch[self.man_mode_idx]
                )

            self.displace_sigma = self.us2cycles(
                self.cfg.device.manipulate.displace_sigma[self.man_mode_idx],
                gen_ch=self.man_ch[self.man_mode_idx]
                )
            self.add_gauss(
                ch=self.man_ch[self.man_mode_idx],
                name="displace",
                sigma=self.displace_sigma,
                length=self.displace_sigma*4
                )
            
        if play:
            assert alpha is not None

            _alpha = np.conj(alpha) # convert to conjugate to respect qick convention
            gain =  int(np.abs(_alpha)/self.gain2alpha)
            phase = np.angle(_alpha)/np.pi*180 - 90 # 90 is needed since da/dt = -i*drive
            self.setup_and_pulse(
                ch=self.man_ch[self.man_mode_idx],
                style="arb",
                freq=self.f_man,
                phase=self.deg2reg(phase, gen_ch=self.man_ch[self.man_mode_idx]),
                gain=gain,
                waveform="displace")


    def _initialize_floquet_pulses(self):
        """Register Floquet waveforms and build their pulse arguments."""
        # Register one complete arb envelope for gauss and preload_flattop modes.
        # Native flat_top modes keep reusing MM_base's pi_m1si_low/high ramps.
        for i_stor in range(7):
            stor_name = f"M1-S{i_stor+1}"
            ch = self.m1s_ch[i_stor]
            waveform_mode = self.m1s_waveform_mode[i_stor]
            
            if waveform_mode == 'gauss':
                sig_us = self.cfg.expt.get("floquet_gauss_sigma", None)
                if sig_us is None:
                    sig_us = self.swap_ds.get_gauss_sigma(stor_name)
                n_sig = self.swap_ds.get_gauss_n_sigma(stor_name)
                sigma = self.us2cycles(sig_us, gen_ch=ch)
                self.add_gauss(ch=ch,
                               name=self.m1s_wf_name[i_stor],
                               sigma=sigma,
                               length=sigma * n_sig)
                
            elif waveform_mode == 'preload_flattop':
                self.add_preloaded_flat_top(ch=ch,
                                            name=self.m1s_wf_name[i_stor],
                                            flat_length_us=self.swap_ds.get_len(stor_name),
                                            ramp_sigma_us=self.swap_ds.get_ramp_sigma(stor_name))

        self.m1s_kwargs = []
        for stor in range(7):
            kw = {
                'ch': self.m1s_ch[stor],
                'style': self.m1s_style[stor],
                'freq': self.m1s_freq[stor],
                'phase': 0,
                'gain': self.m1s_gain[stor],
                'waveform': self.m1s_wf_name[stor],
            }
            if self.m1s_style[stor] != 'arb':   # flat_top / const need the plateau length
                kw['length'] = self.m1s_length[stor]
            self.m1s_kwargs.append(kw)


    def initialize(self):
        """
        MM_base_init to pull basic info
        Retrieves ch, freq, length, gain from csv for M1-Sx π/2 pulses
        """
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        #TODO: this should use a config key to determine whether
        # to use floquet or gate (pi or pi/2) datasets
        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1  # using first manipulate channel index needs to be fixed at some point

        self._initialize_floquet_pulses()

        if readout_mode(self.cfg.expt) == "wigner" or ('init_alpha' in self.cfg.expt):
            self.displace_man(setup=True, play=False)

        self.sync_all(200)


    def core_pulses(self):
        """
        Override this method to control what happens in between pre and post pulses
        so that for most experiments we don't need to override the body method
        """
        # eg:
        # self.setup_and_pulse(**self.m1s_kwargs[0])
        self.sync_all(self.us2cycles(0.1))
    

    def body(self):
        cfg=AttrDict(self.cfg)
        readout = readout_mode(cfg.expt)

        # initializations as necessary
        self.reset_and_sync()

        if self.cfg.expt.get('active_reset', False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            if self.cfg.expt.get('pre_relax_delay', 0) > 0:
                self.sync_all(self.us2cycles(self.cfg.expt.pre_relax_delay))

        init_stor = self.cfg.expt.init_stor
        ro_stor = self.cfg.expt.ro_stor
        if self.cfg.expt.get("parity_check", False):
            self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.phase_second_pulse, fast=self.cfg.expt.parity_fast)
            qTest = self.cfg.expt.qubits[0]
            self.sync_all()
            self.measure(
                pulse_ch=self.res_chs[qTest],
                adcs=[self.adc_chs[qTest]],
                adc_trig_offset=self.cfg.device.readout.trig_offset[qTest],
                wait=True
            )
            if np.abs(self.cfg.expt.get("phase_second_pulse", 180))  <  90:
                self.sync_all(self.us2cycles(2.0))
                reset_pulse_creator = self.get_prepulse_creator([['qubit', 'ge', 'pi', 0]])
                cfg = AttrDict(self.cfg)
                self.custom_pulse(cfg, reset_pulse_creator.pulse, prefix = 'pre_parity_check_reset_')
            self.sync_all(self.us2cycles(2.0))
            self.reset_and_sync()

        # prepulse: ge -> ef -> f0g1
        # TODO: make this overridable from cfg
        if cfg.expt.prepulse:

            if type(init_stor) is int:
                init_stor = [init_stor]
            if type(init_stor) is not list:
                raise ValueError("init_stor must be int or list of int")

            if cfg.expt.init_fock:

                prepulse_cfg = []
                for each_init_stor in init_stor:
                    prepulse_cfg += [
                        ['qubit', 'ge', 'pi', 0,],
                        ['qubit', 'ef', 'pi', 0,], # qubit in f
                        ['man', 'M1', 'pi', 0,], # f0-g1 --> man in 1
                    ]
                    if each_init_stor > 0:
                        prepulse_cfg.append(['storage', f'M1-S{each_init_stor}', 'pi', 0,])

                pulse_creator = self.get_prepulse_creator(prepulse_cfg)
                self.sync_all()
                self.custom_pulse(cfg, pulse_creator.pulse, prefix='pre_')
                self.sync_all()

            elif cfg.expt.get("init_man_fock_state", None) is not None:
                _init_state = cfg.expt.init_man_fock_state
                _man_no = getattr(cfg.expt, 'man_mode_no', 1) #currently not used
                prepulse_cfg = []
                for each_init_stor in init_stor:
                    prepulse_cfg += self.prep_man_fock_state(_man_no,
                                                             _init_state,
                                                             broadband=False) #Check
                    if each_init_stor > 0:
                        prepulse_cfg.append(['storage', f'M1-S{each_init_stor}', 'pi', 0,])
                pulse_creator = self.get_prepulse_creator(prepulse_cfg)
                if not self.cfg.expt.get("do_crude_comp", False):
                    self.sync_all()
                    self.custom_pulse(cfg, pulse_creator.pulse, prefix = 'pre_')
                    self.sync_all()
                else:
                    pulse_data = np.array(pulse_creator.pulse, dtype=object).copy()

                    # Compensate f_n -> g_{n+1} sideband matrix element.
                    # If you are using repeated bare 'f0-g1', set scale by occurrence.
                    fg_count = 0
                    scale_by_occurrence = self.cfg.expt.get("fg_scale_by_occurrence", True)
                    fg_area_comp = self.cfg.expt.get("fg_area_comp", "gain")  # "gain" or "length"

                    for k, p in enumerate(prepulse_cfg):
                        if len(p) < 2:
                            continue

                        is_multiphoton = (p[0] == "multiphoton")
                        transition = p[1]

                        is_fg_sideband = (
                            is_multiphoton
                            and isinstance(transition, str)
                            and transition.startswith("f")
                            and "-g" in transition
                        )

                        if not is_fg_sideband:
                            continue

                        if scale_by_occurrence:
                            # Works even if the logical string repeats bare 'f0-g1':
                            # first f-g pulse -> n=0, second -> n=1, third -> n=2.
                            n = fg_count
                        else:
                            # Works if the string is f0-g1, f1-g2, f2-g3.
                            n = int(transition.split("-")[0][1:])
                        factor = np.sqrt(n + 1)

                        old_gain = pulse_data[1, k]
                        old_length = pulse_data[2, k]

                        if fg_area_comp == "gain":
                            pulse_data[1, k] = int(round(old_gain / factor))

                        elif fg_area_comp == "length":
                            pulse_data[2, k] = old_length / factor

                        else:
                            raise ValueError("fg_area_comp must be either 'gain' or 'length'.")

                        if self.cfg.expt.get("debug", False):
                            print(
                                f"f-g compensation pulse {k}: {transition}, "
                                f"n={n}, factor=sqrt({n+1})={factor:.3f}, "
                                f"gain={old_gain}->{pulse_data[1, k]}, "
                                f"length={old_length}->{pulse_data[2, k]}"
                            )

                        fg_count += 1

                    if self.cfg.expt.get("debug", False):
                        print("final compensated prep pulse table:")
                        for k, row in enumerate(pulse_data.T):
                            label = prepulse_cfg[k] if k < len(prepulse_cfg) else None
                            print(f"{k:02d}", label, "->", row)
                    self.sync_all()
                    self.custom_pulse(cfg, pulse_data, prefix='pre_')
                    self.sync_all()
            else:  # init in coherent state

                assert 'init_alpha' in cfg.expt and cfg.expt.init_alpha

                for each_init_stor in init_stor:
                    self.displace_man(
                        alpha=cfg.expt.init_alpha,
                        setup=False,
                        play=True,
                    )

                    if each_init_stor > 0:
                        prepulse_cfg = [['storage', f'M1-S{each_init_stor}', 'pi', 0]]

                        pulse_creator = self.get_prepulse_creator(prepulse_cfg)
                        self.sync_all()
                        self.custom_pulse(cfg, pulse_creator.pulse, prefix=f'pre_{each_init_stor}_')
                        self.sync_all()

        # core pulses: override the method to define your own expeirment
        self.core_pulses()

        # postpulse: decode, i.e. swap ro_stor into M1, and for the qubit
        # readout map M1 to the qubit. The other readouts read M1.
        if cfg.expt.postpulse:

            # Move ro_stor to man
            postpulse_cfg = [ ['storage', f'M1-S{ro_stor}', 'pi', 0,] ] if ro_stor > 0 else []

            if readout == "qubit":
                # Move man to qubit for population measurement
                postpulse_cfg.append(['man', 'M1', 'pi', 0,])
                if self.cfg.expt.get('map_to_qubit_ge', False):
                    postpulse_cfg.append(['qubit', 'ef', 'pi', 0,])

            pulse_creator = self.get_prepulse_creator(postpulse_cfg)
            self.sync_all()
            self.custom_pulse(cfg, pulse_creator.pulse, prefix='post_')
            self.sync_all()

        # readout: the last step before the measurement, with or without postpulse
        if readout in ("parity", "multiparity"):

            if readout == "parity":
                if self.cfg.expt.get("debug", False):
                    print("Performing parity readout with parity pulse")
                self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.get("phase_second_pulse", 180), fast=self.cfg.expt.parity_fast)
            if readout == "multiparity":
                if self.cfg.expt.get("debug", False):
                    print("Performing multiparity readout with parity pulse")
                self.multi_parity_readout(fast = self.cfg.expt.get("parity_fast", False))
            self.sync_all()

        if readout == "wigner":
            # Population is still in man, perform displacement + parity measurement

            # Displacement
            self.displace_man(
                alpha=cfg.expt.wigner_alpha,
                setup=False,
                play=True,
                )

            # Parity pulse on qubit
            self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.phase_second_pulse, fast=self.cfg.expt.parity_fast)

        if readout == "slow_pi_ge":
            qTest = self.cfg.expt.qubits[0]
            slow_pi_ge = cfg.device.qubit.pulses.slow_pi_ge
            slow_pi_ge_pulse = [
                [cfg.device.qubit.f_ge[qTest]],
                [slow_pi_ge.gain[qTest]],
                [slow_pi_ge.length[qTest]],
                [0.0],
                [self.qubit_chs[qTest]],
                [slow_pi_ge.type[qTest]],
                [slow_pi_ge.sigma[qTest]],
            ]
            self.custom_pulse(
                cfg,
                slow_pi_ge_pulse,
                prefix="slow_pi_ge_readout_",
            )

        self.measure_wrapper()

    # ---------------------------------------------------------------------
    # Manipulate-mode pulses. man_reset and prep_man_fock_state override
    # MM_base's; multi_parity_readout is new. Until step 10B they were the
    # ManipulateModePulses mixin of the dark-mode base, so only the DarkBase
    # and MBR programs had them. The differences from MM_base: man_reset
    # repeats each dump pulse cfg.expt.dump_reset_iter_num times (default 1,
    # the same pulses), and prep_man_fock_state accepts any photon number.
    # ---------------------------------------------------------------------

    def multi_parity_readout(self, 
                             name='multiparity_readout', 
                             register_label='mpreadout', 
                             man_idx=1, 
                             final_sync=False,
                             fast = False):
        # fast = self.cfg.expt.get('parity_fast', False)
        # import the config and set qubit number, by default 0 since we have only one, but should be done better
        cfg=AttrDict(self.cfg)
        qTest = self.cfg.expt.qubits[0]
        self.r_cond_phase = 8
        self.r_read_q = 9
        self.r_thresh_q = 11 
        wait_after_readout = 0.10 # in us
        wait_after_reset = 2.0
        
        second_phase = self.cfg.expt.get("phase_second_pulse", 180) #if 180, maps even to ground
        cond_sec_phase = self.cfg.expt.get("cond_sec_phase", 90)
        cond_op = "<" if second_phase > 90 else ">"

        self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]))
        # check if final sync is needed (only if last readout)
        mid_sync_delay = self.us2cycles(wait_after_reset)
        if final_sync:
            final_sync_delay = self.us2cycles(self.cfg.device.readout.relax_delay[qTest])
        else: 
            if self.cfg.expt.get("debug", False):
                print("needs a pretty long sync here due to the measurement")
            final_sync_delay = self.us2cycles(wait_after_reset)

        # parity pulses, for now I will do something hacky, 
        # i.e. will only load the waveform once, should rewrite custom_pulse to be more general

        parity_str = self.get_parity_str(man_idx, return_pulse=True, second_phase=second_phase, fast=fast)
        self.custom_pulse(cfg, parity_str, prefix=name)
        
        
        # # measurement
        # self.sync_all(self.us2cycles(0.1))
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            t='auto',
            wait=True)
        # I dont exactly get why I need a wait instead of sync here, but ok, this is the minimal wait for read to be done    
        self.wait_all(self.us2cycles(wait_after_readout))        
        # # syntax is read(input_ch, page, upper/lower, reg) where lower is I, upper is Q
        self.read(0, 0, "lower", self.r_read_q) # stores I in (0,0) into r_read_q
        # # first if 
        self.condj(0, self.r_read_q, "<", self.r_thresh_q,
                   register_label+"LABEL1")  # compare the value recorded above to the value stored in threshold.
        self.set_pulse_registers(ch=self.qubit_chs[qTest],
                                 freq=self.f_ge_reg[qTest],
                                 style="arb",
                                 phase=self.deg2reg(0),
                                 gain=self.pi_ge_gain,
                                 waveform='pi_qubit_ge')
        self.pulse(ch=self.qubit_chs[qTest])
        self.label(register_label+"LABEL1")  # location to be jumped to
        self.sync_all(mid_sync_delay)
        
        ##Second parity pulse
        if fast:
            revival_time = cfg.device.manipulate.revival_time_fast[man_idx-1] / 2
        else:
            revival_time = cfg.device.manipulate.revival_time[man_idx-1] / 2
        revival_cycles = self.us2cycles(revival_time)
        reg_page = self.ch_page(self.qubit_chs[qTest])
        reg_phase =self.sreg(self.qubit_chs[qTest], "phase")
        if fast: 
            freq_pi = self.f_ge_hpi_fast
            gain_pi = self.hpi_ge_gain_fast
            waveform_pi = 'hpi_qubit_ge_fast'
            freq_AC = self.cfg.device.manipulate.revival_stark_shift[man_idx-1]
            theta_2 = second_phase + 2*np.pi*freq_AC * revival_time * 180/np.pi
            theta_2 = theta_2 % 360
        else:
            freq_pi = self.f_ge
            gain_pi = self.hpi_ge_gain
            theta_2 = second_phase
            waveform_pi = 'hpi_qubit_ge'
            
        # self.safe_regwi(reg_page, self.r_cond_phase, self.deg2reg(theta_2))
        # self.condj(0, self.r_read_q, cond_op, self.r_thresh_q,
        #            register_label+"LABEL2")  # compare the value recorded above to the value stored in threshold.
        # self.mathi(reg_page, self.r_cond_phase, self.r_cond_phase, "+",  self.deg2reg(cond_sec_phase))
        # self.label(register_label+"LABEL2")
        

        
        theta_skip = theta_2 % 360
        theta_corr = (theta_2 + cond_sec_phase) % 360
        theta_skip_reg = self.deg2reg(theta_skip, gen_ch=self.qubit_chs[qTest])
        theta_corr_reg = self.deg2reg(theta_corr, gen_ch=self.qubit_chs[qTest])
        self.safe_regwi(reg_page, self.r_cond_phase, theta_skip_reg)
        self.condj(0, self.r_read_q, cond_op, self.r_thresh_q, register_label+"LABEL2")
        self.safe_regwi(reg_page, self.r_cond_phase, theta_corr_reg)
        self.label(register_label+"LABEL2")
        
        #first pi/2 pulse
        self.set_pulse_registers(ch=self.qubit_chs[qTest],
                                 freq=freq_pi,
                                 style="arb",
                                 phase=self.deg2reg(0),
                                 gain=gain_pi,
                                 waveform=waveform_pi)
        self.pulse(ch=self.qubit_chs[qTest])
        self.sync_all()
        # wait based on revival time 
        self.sync_all(revival_cycles)
        # second pi/2 pulse, if fast take into account AC stark phase
        # here we can just update the phase of the waveform
        self.mathi(reg_page, reg_phase, self.r_cond_phase, "+", 0)
        self.pulse(ch=self.qubit_chs[qTest])
        # self.sync_all()
        # self.measure(
        #     pulse_ch=self.res_chs[qTest],
        #     adcs=[self.adc_chs[qTest]],
        #     adc_trig_offset=cfg.device.readout.trig_offset[qTest],
        #     t='auto',
        #     wait=True)
        # # I dont exactly get why I need a wait instead of sync here, but ok, this is the minimal wait for read to be done    
        # self.wait_all(self.us2cycles(wait_after_readout))        
        # # # syntax is read(input_ch, page, upper/lower, reg) where lower is I, upper is Q
        # self.read(0, 0, "lower", self.r_read_q) # stores I in (0,0) into r_read_q
        # # # first if 
        # self.condj(0, self.r_read_q, "<", self.r_thresh_q,
        #            register_label+"LABEL3")  # compare the value recorded above to the value stored in threshold.
        # self.set_pulse_registers(ch=self.qubit_chs[qTest],
        #                          freq=self.f_ge_reg[qTest],
        #                          style="arb",
        #                          phase=self.deg2reg(0),
        #                          gain=self.pi_ge_gain,
        #                          waveform='pi_qubit_ge')
        # self.pulse(ch=self.qubit_chs[qTest])
        # self.label(register_label+"LABEL3")  # location to be jumped to
        # self.sync_all(final_sync_delay)
        
        
    def prep_man_fock_state(self, man_no, state, broadband=False):
        r"""
        Override the one in MMbase, just for the debugging purpose. 
        The program is curretly not perfect, as it simply divides the pulse length by \sqrt{n}
        -----------
        Build a gate-based pulse string to prepare a Fock state (or
        superposition of two adjacent Fock states) in the manipulate mode.

        Args:
            man_no: Manipulate mode number.
            state: Which state to prepare.
                '0'  → |0> (vacuum, returns empty list)
                'n'  → |n> (single Fock state, e.g., '1', '2', '3')
                '+'  → |0> + |1>
                '-'  |0> - |1>
                '+i' → |0> + i|1>
                '-i' → |0> - i|1>
            broadband: If True, use broadband preparation (drives through
                g0-e0 transition for all steps).

        Returns:
            List of gate-string descriptors suitable for get_prepulse_creator().
        """
        if self.cfg.expt.get("debug", False):
            print("RUNNING MULTIFOCK PREP")
        STATE_MAP = {
            '+': ([0, 1], 0),    # |0> + |1>
            '-': ([0, 1], 180),  # |0> - |1>
            '+i': ([0, 1], 90),  # |0> + i|1>
            '-i': ([0, 1], -90), # |0> - i|1>
        }
        
        if state == '0':
            return []

        if state in STATE_MAP:
            fock_spec, phase = STATE_MAP[state]
        elif isinstance(state, str) and state.isdigit():
            fock_spec, phase = int(state), None
        else:
            raise ValueError(
                f"Unknown state '{state}'. "
                f"Use a positive integer (e.g., '1', '2') or one of: {list(STATE_MAP.keys())}"
            )

        # 2. Single Fock state |n>
        if isinstance(fock_spec, int):
            pulse_seq = []
            for i in range(fock_spec):
                # pulse_seq += [['multiphoton', 'g0-e0', 'pi', 0]]
                # pulse_seq += [['multiphoton', 'e0-f0', 'pi', 0]]
                # pulse_seq += [['multiphoton', 'f0-g1', 'pi', 0]]
                pulse_seq += [['multiphoton', f'g{i}-e{i}', 'pi', 0]]
                pulse_seq += [['multiphoton', f'e{i}-f{i}', 'pi', 0]]
                pulse_seq += [['multiphoton', f'f{i}-g{i + 1}', 'pi', 0]]
            if self.cfg.expt.get("debug", False):
                print("single Fock prep pulse_seq:")
                for p in pulse_seq:
                    print("  ", p)
            return pulse_seq

        # 3. Superposition |n> + e^(i*phase)|m>
        state_1, state_2 = fock_spec
        pulse_seq = []
        for i in range(state_1):
            pulse_seq += [['multiphoton', f'g{i}-e{i}', 'pi', 0]]
            pulse_seq += [['multiphoton', f'e{i}-f{i}', 'pi', 0]]
            pulse_seq += [['multiphoton', f'f{i}-g{i + 1}', 'pi', 0]]

        start_idx = 0 if broadband else state_1
        pulse_seq += [
            ['multiphoton', f'g{start_idx}-e{start_idx}', 'hpi', phase]
        ]

        diff = state_2 - state_1
        shelving = 0
        for k in range(diff):
            n = state_1 + k
            pulse_seq += [['multiphoton', f'e{n}-f{n}', 'pi', 0]]
            if shelving < diff - 1:
                pulse_seq += [
                    ['multiphoton', f'g{start_idx}-e{start_idx}', 'pi', 0]
                ]
            pulse_seq += [['multiphoton', f'f{n}-g{n + 1}', 'pi', 0]]
            if shelving < diff - 1:
                pulse_seq += [
                    ['multiphoton', f'g{start_idx}-e{start_idx}', 'pi', 0]
                ]
            shelving += 1

        return pulse_seq
        
    def man_reset(self, man_idx=1, dump_mode_idx=2, chi_dressed=True):
        '''
        Reset manipulate mode by swapping it to lossy mode

        chi_dressed: if man freq shifted due to pop in qubit e, f states.
        using_qubit: if True, we do g1-f0/ef/qubit reset instead of using the dump, which is not indeal since it remove only the fock 1 population but can be usefull if dump cannot be found 
        '''
        if self.cfg.expt.get("debug", False):
            print("overrided man reset is called")
        qTest = 0
        cfg=AttrDict(self.cfg)

        MiDj_freq = self.dataset.get_freq(f'M{man_idx}-D{dump_mode_idx}')
        MiDj_gain = self.dataset.get_gain(f'M{man_idx}-D{dump_mode_idx}')
        MiDj_length = self.dataset.get_pi(f'M{man_idx}-D{dump_mode_idx}')
        N = 2 if chi_dressed else 0
        chi_ge = cfg.device.manipulate.chi_ge[qTest]
        chi_ef = cfg.device.manipulate.chi_ef[qTest]

        self.sideband_sigma_high = self.us2cycles(self.cfg.device.storage.ramp_sigma, gen_ch=self.flux_high_ch[qTest])
        self.add_gauss(ch=self.flux_high_ch[qTest],
                    name="ramp_high",# + str(man_idx),
                    sigma=self.sideband_sigma_high,
                    length=self.sideband_sigma_high*6) # M1-x flat tops use 6 sigma
        # self.wait_all(self.us2cycles(0.1))
        self.sync_all(self.us2cycles(0.1))

        chis = [chi_ge, chi_ge+chi_ef] if chi_dressed else [0]
        ch = self.flux_high_ch[qTest]
        iter_num = self.cfg.expt.get("dump_reset_iter_num", 1)
        for n in range(0, N+1): # works when MiDj freq goes down (chi<0, bare freq+chi*n)
            for chi in chis:
                for _ in range(iter_num):
                    freq_chi_shifted = MiDj_freq + (n * chi)
                    # if cfg.expt.get("man_reset_print", True):
                    #     print(ch, freq_chi_shifted, MiDj_length, MiDj_gain)
                    self.set_pulse_registers(
                        ch=ch,
                        freq=self.freq2reg(freq_chi_shifted, gen_ch=ch),
                        style="flat_top",
                        phase=self.deg2reg(0),
                        length=self.us2cycles(MiDj_length, gen_ch=ch),
                        gain=MiDj_gain,
                        waveform="ramp_high"
                        )
                    self.pulse(ch=ch)
                    self.sync_all()
                # self.sync_all(self.us2cycles(0.025))
        # self.wait_all(self.us2cycles(0.25))
        self.sync_all(self.us2cycles(2))


class QsimRProgram(MMRAveragerProgram):
    """RAverager counterpart of ``QsimProgram``, for hardware depth sweeps.

    ``DarkBaseRProgram`` in ``dark_base`` until step 10F. The pulse-building methods below do not depend on the AveragerProgram
    software loop, so both program types use the same implementations.
    Concrete RAverager programs only need to define ``core_pulses`` and
    ``update``.
    """

    _pre_selection_filtering = True

    retrieve_swap_parameters = QsimProgram.retrieve_swap_parameters #borrowing methods
    _initialize_floquet_pulses = QsimProgram._initialize_floquet_pulses
    # Two of the three manipulate-mode methods, by assignment rather than
    # inheritance: inheriting from QsimProgram would also shadow MM_base's
    # ``man_reset``, which is the one ``active_reset`` plays here.
    prep_man_fock_state = QsimProgram.prep_man_fock_state
    multi_parity_readout = QsimProgram.multi_parity_readout
    body = QsimProgram.body #borrowing methods

    def __init__(self, soccfg, cfg):
        readout_mode(cfg.expt)  # as QsimProgram: refuse what it cannot play
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg)

    readouts_per_shot = QsimProgram.readouts_per_shot

    def initialize(self):
        self.MM_base_initialize()

        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        man_mode_no = self.cfg.expt.get("man_mode_no", 1)
        self.man_mode_idx = man_mode_no - 1

        self._initialize_floquet_pulses()

        self.sync_all(200)


class QsimExperiment(Experiment):
    """
    The one qsim sweep driver (QsimBaseExperiment until step 10F; also
    DarkBaseExperiment, which was the same driver with another name).

    Sweep 1 or 2 parameters in cfg.expt
    Experimental Config:
    expt = dict(
        expts: number experiments should be 1 here as we do soft loops
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qubits: this is just 0 for the purpose of the currrent multimode sample
        init_stor: storage to initialize the photon into (0-7)
        ro_stor: storage to readout the photon from (0-7)
        active_reset: bool (uses get_active_reset_params for ef_reset, man_reset, storage_reset, etc.)
        swept_params: list of parameters to sweep, e.g. ['detune', 'gain']
    )
    In principle this overlaps with qick.NDAveragerProgram, but this allows you to
    skip writing new expeirment classes or at least acquire() while doing 
    more general sweeps than just a qick register, incl nonlinear steps.
    Consider doing NDAverager or RAverager if there's speed advantage.

    Usage: if you want to sweep cfg.expt.paramName, 
    include paramName here in this list 
    AND include cfg.expt.paramNames (note the s) as a list of values to step thru.
    (You want a list instead of numpy array for better yaml export.)
    Currently handles 1D and 2D sweeps and plots only.
    For 2D, order is [outer (y), inner (x)].
    """
    # The Program a subclass runs when the caller names none.
    default_program = None

    def __init__(self, soccfg=None, path='', prefix=None,
                 config_file=None, expt_params=None,
                 program=None, progress=None, **kwargs):
        """
        program can be:
        - A class object (the class you imported, not an instance)
        - A tuple of (module_path, class_name) strings
        - None (the class's ``default_program``, else QsimProgram)
        """
        if not prefix:
            prefix = self.__class__.__name__
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress, **kwargs)
        self.cfg.expt = AttrDict(expt_params)

        program = program or self.default_program
        # Store program class info as strings (pickle-safe)
        if program is None:
            # Default to QsimProgram
            self.program_module = QsimProgram.__module__
            self.program_class = QsimProgram.__name__
        elif isinstance(program, tuple) and len(program) == 2:
            # Program passed as (module, class_name) tuple
            self.program_module, self.program_class = program
        else:
            # Program passed as class object - extract module and name
            self.program_module = program.__module__
            self.program_class = program.__name__

        self.cfg.expt.QickProgramName = self.program_class

        # ProgramClass is loaded lazily when needed
        self._ProgramClass = None

    @property
    def ProgramClass(self):
        """Lazy load the program class when first accessed."""
        if self._ProgramClass is None:
            import importlib
            module = importlib.import_module(self.program_module)
            self._ProgramClass = getattr(module, self.program_class)
        return self._ProgramClass


    def sweep_axes(self):
        """-> ``[(cfg.expt key, values), ...]``, outermost first: the points acquire visits.

        The default: each key in ``cfg.expt.swept_params`` (outer first), with its
        values in ``cfg.expt[key + 's']``. Subclasses add axes (Wigner) or fix them.
        """
        return [(key, self.cfg.expt[key + 's']) for key in self.cfg.expt.swept_params]

    def acquire(self, progress=False, debug=False):
        """Build, compile and acquire one Program per sweep point; keep the raw shots.

        The one sweep driver of the qsim Experiments (step 10E,
        ``docs/qsim/program_tree_plan.md``). The Program says how many readouts
        a shot has (``readouts_per_shot``), so the lanes cannot drift from the
        pulses. The science lane is the last one; with active reset and
        ``pre_selection_reset``, a point's average keeps only the shots whose
        herald found the qubit in g.
        """
        ensure_list_in_cfg(self.cfg)
        read_num = self.ProgramClass.readouts_per_shot(self.cfg)
        self.cfg.read_num = read_num

        axes = self.sweep_axes()
        self.outer_param = axes[0][0]
        self.inner_param = axes[1][0] if len(axes) > 1 else 'dummy'

        pre_select = (self.cfg.expt.get('active_reset', False)
                      and self.cfg.expt.get('pre_selection_reset', False))
        threshold = self.cfg.device.readout.threshold[self.cfg.expt.qubits[0]]

        points = dict(avgi=[], avgq=[], idata=[], qdata=[])
        for outer in tqdm(axes[0][1], disable=not progress):
            for inner in product(*(values for _, values in axes[1:])):
                for (key, _), value in zip(axes, (outer,) + inner):
                    self.cfg.expt[key] = value
                self.prog = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = self.prog.acquire(self.im[self.cfg.aliases.soc],
                                               threshold=None,
                                               load_pulses=True,
                                               progress=False,
                                               debug=debug,
                                               readouts_per_experiment=read_num)
                idata, qdata = self.prog.collect_shots()
                if pre_select:
                    avgi, avgq = GeneralFitting.filter_shots_per_point(
                        idata, qdata, read_num, threshold=threshold, pre_selection=True)
                else:
                    avgi, avgq = avgi[0][-1], avgq[0][-1]
                points['avgi'].append(avgi)
                points['avgq'].append(avgq)
                points['idata'].append(idata)
                points['qdata'].append(qdata)

        data = self.shape_data(axes, points)

        if self.cfg.expt.get('parity_check', False):
            # The parity herald is the first lane after the active-reset lanes.
            start = 0
            if self.cfg.expt.get('active_reset', False):
                params = MMAveragerProgram.get_active_reset_params(self.cfg)
                start += MMAveragerProgram.active_reset_read_num(**params)
            data['parity_idata'] = np.asarray(data['idata'])[..., start::read_num]
            data['parity_qdata'] = np.asarray(data['qdata'])[..., start::read_num]

        if self.cfg.expt.get('normalize', False):
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.data = data
        return data

    def shape_data(self, axes, points):
        """-> the data dict from the per-point lists, in the 1D/2D layout.

        1D: ``xpts`` is the axis. 2D: ``ypts`` is the outer axis, ``xpts`` the
        inner one, and the averages have shape (outer, inner). ``idata`` and
        ``qdata`` stay lists of per-point shot arrays.
        """
        assert len(axes) in {1, 2}, "the default layout handles 1D and 2D sweeps"
        shape = [len(values) for _, values in axes]
        avgi = np.reshape(np.array(points['avgi']), shape)
        avgq = np.reshape(np.array(points['avgq']), shape)
        data = dict(
            avgi=avgi, avgq=avgq,
            amps=np.abs(avgi + 1j * avgq),
            phases=np.angle(avgi + 1j * avgq),
            idata=points['idata'], qdata=points['qdata'],
        )
        if len(axes) == 2:
            data['xpts'] = axes[1][1]
            data['ypts'] = axes[0][1]
        else:
            data['xpts'] = axes[0][1]
        return data

    def analyze_multiparity(self):
        """Classify the two parity readouts of every point (``readout='multiparity'``)."""
        keys = [
            'mean_parity_first', 'mean_parity_second',
            'p_first_e', 'p_second_e',
            'p_mod0', 'p_mod1', 'p_mod2', 'p_mod3',
            'p_gg', 'p_eg', 'p_ge', 'p_ee',
            'mean_n_mod4',
        ]

        xpts = np.asarray(self.data['xpts']).reshape(-1)
        ypts = np.asarray(self.data.get('ypts', [])).reshape(-1)
        is_2d = len(ypts) > 0

        if is_2d:
            data_shape = (len(ypts), len(xpts))
            point_count = len(ypts) * len(xpts)
        else:
            data_shape = (len(xpts),)
            point_count = len(xpts)

        if len(self.data['idata']) != point_count:
            raise ValueError(
                'multiparity point count does not match the sweep axes: '
                f"{len(self.data['idata'])} shots rows for shape {data_shape}"
            )

        out = {'xpts': xpts}
        if is_2d:
            out['ypts'] = ypts
        for key in keys:
            out[key] = []

        for j in range(point_count):
            r = classify_two_parity_readouts(self, point_idx=j)

            for key in keys:
                out[key].append(r[key])

        for key in keys:
            out[key] = np.asarray(out[key]).reshape(data_shape)

        # Same quantity as p1 + 2*p2 + 3*p3.
        # Kept as a convenient explicit name.
        out['nmod4_mean'] = out['mean_n_mod4']

        self.data['multiparity'] = out
        return out


    # def analyze(self, data=None, fit=True, fitparams = None, **kwargs):
    #     pass


    def display(self, data=None, fit=True, **kwargs):
        #TODO: might want to add capability to plot custom keys
        # such as extra_data_keys=['best_fit']
        if data is None:
            data=self.data

        title = self.fname.split(os.path.sep)[-1] if isinstance(self.fname, str) else self.fname.name

        if 'ypts' in data.keys(): # guess if this is 2D or 1D
            fig, axs = plt.subplots(2, 1, figsize=(10, 9))
            axs[0].set_title(title)
            mesh = axs[0].pcolormesh(data['xpts'], data['ypts'], data['avgi'])
            fig.colorbar(mesh, ax=axs[0], label='I [ADC level]')
            mesh = axs[1].pcolormesh(data['xpts'], data['ypts'], data['avgq'])
            fig.colorbar(mesh, ax=axs[1], label='Q [ADC level]')
            try:
                xlabel, ylabel = self.inner_param, self.outer_param
            except AttributeError:
                try:
                    ylabel, xlabel = self.cfg.swept_params
                except AttributeError:
                    try:
                        ylabel, xlabel = self.cfg.expt.swept_params
                    except Exception as e:
                        print("Couldn't get x and y labels automatically:", e)
                        xlabel, ylabel = None, None
            for ax in axs:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
        else:
            try:
                xlabel = self.outer_param
            except AttributeError:
                xlabel = self.cfg.expt.swept_params[0]
            except Exception as e:
                print("Couldn't get x label automatially", e)
            fig, axs = plt.subplots(2, 1, figsize=(10, 9))
            ax = axs[0]
            ax.set_title(title)
            ax.set_ylabel("I [ADC level]")
            ax.plot(data["xpts"], data["avgi"],'o-')
            ax = axs[1]
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Q [ADC level]")
            ax.plot(data["xpts"], data["avgq"],'o-')
        return fig, axs


    # Provenance recorded beside `config`, deliberately not inside it.
    #
    # `cfg.expt` is the *input* to a run: the values a notebook sets by hand,
    # and copies into the next submission to build on. The Floquet cycle time
    # is generated, never consumed. Parking it in `cfg.expt` would make a
    # derived value indistinguishable from a chosen one, and it would ride
    # along into the next job's config looking like a manual override.
    #
    # It is worth saving at all because it is the one fact offline analysis
    # cannot otherwise recover from the file: it existed only on the compiled
    # program, which lives in the job pickle, and pickles are ephemeral.
    # Offline analysis reads it back through `recorded_derived_params`. Old
    # files that predate it reach the analysis only after conversion
    # (tools/migrate_mbr_jobs.py), which writes it.
    DERIVED_PARAMS_ATTR = "derived_params"

    def derived_params(self):
        """-> what this run computed that its config does not already say.

        None when the compiled program has no Floquet timing to report: not
        every Qsim program plays a Floquet train, and an absent attribute is
        exactly what the reader expects in that case.

        One job records one cycle time, which is the assumption the aggregate
        analysis has always made (`_saved_parameters` reads the timing off a
        single child's program). A sweep over something that changes the cycle
        time -- `floquet_gauss_sigma`, say -- would record only its last
        value; no such sweep exists today.
        """
        prog = getattr(self, "prog", None)
        if prog is None or not all(hasattr(prog, name) for name in
                                   ("calculate_floquet_cycle_us", "m1s_pi_fracs")):
            return None

        cycle_us = float(prog.calculate_floquet_cycle_us())
        pi_fracs = [int(frac) for frac in prog.m1s_pi_fracs]
        params = dict(floquet_cycle_us=cycle_us,
                      m1s_pi_fracs=pi_fracs,
                      source=f"compiled {type(prog).__name__}")

        # Redundant with the two above -- 1/(4 * pi_frac * T) -- and recorded
        # anyway so the reader can check the file against itself. Omitted
        # rather than guessed if any entry would not be finite, since a
        # placeholder here is worse than an absent key.
        if cycle_us > 0. and all(frac > 0 for frac in pi_fracs):
            params["couplings_MHz"] = [1. / (4. * frac * cycle_us)
                                       for frac in pi_fracs]
        return params

    def recorded_derived_params(self):
        """-> :meth:`derived_params` of this job, live or as saved. None if neither.

        A job that holds its compiled program reports it directly; one loaded
        with ``from_h5file`` reads the attribute its file carries.
        """
        live = self.derived_params()
        if live:
            return live
        raw = getattr(self, "data", {}).get("attrs", {}).get(self.DERIVED_PARAMS_ATTR)
        if raw is None:
            return None
        return json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)

    def save_derived_params(self):
        """Write :meth:`derived_params` into the data file as an attribute."""
        params = self.derived_params()
        if not params:
            return None
        with self.datafile() as f:
            f.attrs[self.DERIVED_PARAMS_ATTR] = json.dumps(params, cls=NpEncoder)
        return params

    def save_data(self, data=None):
        # do we really need to ovrride this?
        # TODO: at least make this save line-by-line
        temp_cfg = deepcopy(self.cfg)
        if "ds_floquet" in self.cfg:
            self.cfg.pop('ds_floquet')  # remove the dataset object from cfg before saving otherwise json gets mad
        if "ds_floquet" in self.cfg.expt:
            self.cfg.expt.pop('ds_floquet')  # remove the dataset object from cfg before saving otherwise json gets mad
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        self.save_derived_params()
        self.cfg = temp_cfg
        return self.fname
