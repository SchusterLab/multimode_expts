# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.signal import find_peaks
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.fit_display_classes import (
    CavityRamseyGainSweepFitting,
    GeneralFitting,
    RamseyFitting,
)
from experiments.MM_base import *
from experiments.characterization_runner import CharacterizationRunner
from experiments.qsim.qsim_base import *
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display import *

from experiments.qsim.kerr import *

from experiments.qsim.qsim_base import QsimBaseExperiment, QsimBaseProgram
from experiments.qsim.sideband_scramble import SidebandScrambleProgram

from copy import copy, deepcopy
from itertools import product

from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view

def flatten_exp_lists(items, container_types=(list, tuple, set)):
    for x in items:
        if isinstance(x, container_types):
            yield from flatten_exp_lists(x, container_types)
        else:
            yield x

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


CENTRAL_RETURN_PRIMARY_PHOTONS = 3
CENTRAL_RETURN_ALLOWED_TOTAL_PHOTONS = (1, 3)


def validate_central_return_occupations(
        initial_occupations,
        swap_stors,
        expected_total_photons=None,
        require_collision_free=True):
    """Validate the product-Fock convention used by the local-return program.

    ``initial_occupations`` is ordered as ``[n_M1] + [n_S for S in
    swap_stors]``.  The N=3 pilot deliberately defaults to collision-free
    states: they use only the already-calibrated one-photon M1/storage swaps
    and avoid adding a multi-photon state-preparation uncertainty to the
    correlation-hole measurement.
    """
    swap_stors = list(swap_stors)
    occupations = list(initial_occupations)

    if not swap_stors:
        raise ValueError("swap_stors must contain at least one storage mode")
    if any(isinstance(stor, (bool, np.bool_))
           or not isinstance(stor, (int, np.integer))
           for stor in swap_stors):
        raise TypeError("swap_stors entries must be integer storage numbers")
    swap_stors = [int(stor) for stor in swap_stors]
    if len(set(swap_stors)) != len(swap_stors):
        raise ValueError(f"swap_stors entries must be distinct, got {swap_stors}")
    if any(stor < 1 or stor > 7 for stor in swap_stors):
        raise ValueError(f"swap_stors must be in S1..S7, got {swap_stors}")

    expected_len = 1 + len(swap_stors)
    if len(occupations) != expected_len:
        raise ValueError(
            "initial_occupations must be ordered as [n_M1] + occupations "
            f"for swap_stors={swap_stors}; expected {expected_len} entries, "
            f"got {len(occupations)}"
        )
    if any(isinstance(n, (bool, np.bool_))
           or not isinstance(n, (int, np.integer))
           for n in occupations):
        raise TypeError("initial_occupations entries must be non-negative integers")
    occupations = [int(n) for n in occupations]
    if any(n < 0 for n in occupations):
        raise ValueError("initial_occupations entries must be non-negative")
    total_photons = sum(occupations)
    if total_photons not in CENTRAL_RETURN_ALLOWED_TOTAL_PHOTONS:
        raise ValueError(
            "central-return experiment supports the primary N=3 protocol "
            "and the N=1 self-Kerr-inactive control; got occupations "
            f"{occupations} with N={total_photons}"
        )
    if expected_total_photons is not None \
            and total_photons != int(expected_total_photons):
        raise ValueError(
            f"expected total occupation N={int(expected_total_photons)}, "
            f"got occupations {occupations} with N={total_photons}"
        )
    if require_collision_free and any(n > 1 for n in occupations):
        raise ValueError(
            "the decisive pilot uses collision-free product Fock states "
            f"(occupations 0 or 1), got {occupations}"
        )

    return occupations, swap_stors


def configure_central_return_metadata(expt_cfg):
    """Normalize and preserve the invariants for direct M1 local return."""
    if "initial_occupations" not in expt_cfg:
        raise ValueError(
            "initial_occupations is required and must be ordered as "
            "[n_M1] + [n_S for S in swap_stors]"
        )

    expected_total_photons = expt_cfg.get(
        "expected_total_occupation", expt_cfg.get("total_photons", None))
    if not expt_cfg.get("require_collision_free_initial_state", True):
        raise ValueError(
            "CentralBosonLocalReturnProgram currently supports only "
            "collision-free initial occupations; the multi-Fock opt-in does "
            "not share the calibrated compensation path"
        )
    occupations, swap_stors = validate_central_return_occupations(
        expt_cfg.initial_occupations,
        expt_cfg.swap_stors,
        expected_total_photons=expected_total_photons,
        require_collision_free=True,
    )
    total_photons = sum(occupations)

    if int(expt_cfg.get("man_mode_no", 1)) != 1:
        raise ValueError(
            "CentralBosonLocalReturnProgram is hard-wired to M1; "
            "man_mode_no must be 1"
        )
    if not expt_cfg.get("prepulse", False):
        raise ValueError(
            "prepulse must be True to explicitly enable the custom central "
            "product-Fock preparation"
        )
    for custom_flag in ("custom_prepulse", "custom_postpulse"):
        if expt_cfg.get(custom_flag, False):
            raise ValueError(
                f"{custom_flag}=True is incompatible with the fixed "
                "central-return pulse body"
            )
    if expt_cfg.get("map_to_qubit_ge", False):
        raise ValueError(
            "map_to_qubit_ge is not used: central return ends in direct M1 "
            "multiparity readout"
        )

    for legacy_flag in (
            "load_man_dark", "swap_man_dark", "swap_man_large_dark"):
        if expt_cfg.get(legacy_flag, False):
            raise ValueError(
                f"{legacy_flag}=True is incompatible with direct central "
                "return; no dark-mode load or analyzer is used"
            )
    if expt_cfg.get("perform_wigner", False):
        raise ValueError("perform_wigner must be False for direct multiparity readout")
    if expt_cfg.get("parity_check", False):
        raise ValueError(
            "parity_check is not part of the direct central-return pulse body"
        )
    if expt_cfg.get("parity_readout", False):
        raise ValueError(
            "parity_readout must be False; use multiparity_readout for the "
            "four-outcome M1 measurement"
        )
    if "ro_stor" in expt_cfg and int(expt_cfg.ro_stor) != 0:
        raise ValueError("ro_stor must be 0: M1 is read out directly")
    if not expt_cfg.get("multiparity_readout", False):
        raise ValueError("multiparity_readout must be True for central return")
    if not expt_cfg.get("postpulse", False):
        raise ValueError(
            "postpulse must be True to make the final direct M1 readout "
            "explicit in the experiment config"
        )

    cond_sec_phase = float(expt_cfg.get("cond_sec_phase", -90.0))
    if not np.isclose((cond_sec_phase + 90.0) % 360.0, 0.0):
        raise ValueError(
            "cond_sec_phase must be -90 deg (mod 360) so p_mod0..p_mod3 "
            "use the classifier convention"
        )
    second_phase = float(expt_cfg.get("phase_second_pulse", 180.0))
    if not np.isclose(second_phase % 360.0, 180.0):
        raise ValueError(
            "phase_second_pulse must be 180 deg (mod 360) for the direct "
            "multiparity classifier convention"
        )

    target = occupations[0] % 4
    for key, expected in (
            ("initial_central_occupation", occupations[0]),
            ("return_target_nmod4", target)):
        if key in expt_cfg and int(expt_cfg[key]) != expected:
            raise ValueError(
                f"{key}={expt_cfg[key]} conflicts with initial_occupations "
                f"(expected {expected})"
            )

    # These fields live in cfg.expt, so they are saved with every hardware job.
    expt_cfg.initial_occupations = occupations
    expt_cfg.initial_occupation_modes = ["M1"] + [
        f"S{stor}" for stor in swap_stors]
    expt_cfg.total_photons = total_photons
    expt_cfg.expected_total_occupation = total_photons
    expt_cfg.primary_total_photons = CENTRAL_RETURN_PRIMARY_PHOTONS
    expt_cfg.is_primary_n3 = (total_photons == CENTRAL_RETURN_PRIMARY_PHOTONS)
    expt_cfg.is_kerr_inactive_n1_control = (total_photons == 1)
    expt_cfg.initial_central_occupation = occupations[0]
    expt_cfg.return_target_nmod4 = target
    expt_cfg.return_target_key = f"p_mod{target}"
    expt_cfg.return_observable = "P(n_M1(t) mod 4 == return_target_nmod4)"
    expt_cfg.desired_local_return_observable = "P(n_M1(t) == n_M1(0))"
    expt_cfg.mod4_number_mapping_exact_in_fixed_N_le_3 = True
    expt_cfg.mod4_mapping_requires_no_upward_leakage = True
    expt_cfg.readout_mode = "M1_direct_multiparity"
    expt_cfg.initial_state_preparation = "custom_central_product_fock"
    expt_cfg.legacy_base_preparation_bypassed = True
    expt_cfg.require_collision_free_initial_state = True
    expt_cfg.man_mode_no = 1
    expt_cfg.prepulse = True
    expt_cfg.custom_prepulse = False
    expt_cfg.custom_postpulse = False
    expt_cfg.map_to_qubit_ge = False
    for ignored_key in ("init_fock", "init_man_fock_state", "init_stor"):
        expt_cfg.pop(ignored_key, None)
    expt_cfg.ro_stor = 0
    expt_cfg.multiparity_readout = True
    expt_cfg.cond_sec_phase = -90.0
    expt_cfg.phase_second_pulse = 180.0
    expt_cfg.perform_wigner = False
    expt_cfg.parity_check = False
    expt_cfg.parity_readout = False
    expt_cfg.load_man_dark = False
    expt_cfg.swap_man_dark = False
    expt_cfg.swap_man_large_dark = False

    return occupations, swap_stors



class DarkBaseExperiment(QsimBaseExperiment):
    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        read_num = 1
        if self.cfg.expt.get('parity_check', False):
            read_num += 1
        if self.cfg.expt.get('active_reset', False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)
        if self.cfg.expt.get('multiparity_readout', False):
            read_num += 1
        self.cfg.read_num = read_num
        assert len(self.cfg.expt.swept_params) in {1,2}, "can only handle 1D and 2D sweeps for now"
        sweep_dim = 2 if len(self.cfg.expt.swept_params) == 2 else 1

        if 'perform_wigner' not in self.cfg.expt:
            self.cfg.expt.perform_wigner = False

        outer_param = self.cfg.expt.swept_params[0]
        outer_params = self.cfg.expt[outer_param+'s']
        if sweep_dim == 2:
            inner_param = self.cfg.expt.swept_params[1]
            inner_params = self.cfg.expt[inner_param+'s']
        else:
            inner_param = 'dummy'
            inner_params = [None]  # Dummy value for single parameter sweep
        self.outer_param, self.inner_param = outer_param, inner_param

        data = {
            'avgi': [], 'avgq': [],
            'amps': [], 'phases': [],
            'idata': [], 'qdata': [],
        }
        if sweep_dim == 2:
            data['xpts'] = inner_params
            data['ypts'] = outer_params
        else:
            data['xpts'] = outer_params

        for self.cfg.expt[outer_param] in tqdm(outer_params, disable=not progress):
            for self.cfg.expt[inner_param] in inner_params:
                self.prog = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)

                avgi, avgq = self.prog.acquire(self.im[self.cfg.aliases.soc],
                                                threshold=None,
                                                load_pulses=True,
                                                progress=False,
                                                debug=debug,
                                                readouts_per_experiment=read_num)

                idata, qdata = self.prog.collect_shots()
                data['idata'].append(idata)
                data['qdata'].append(qdata)

                if self.cfg.expt.active_reset and self.cfg.expt.get('pre_selection_reset', False):
                    avgi_val, avgq_val = GeneralFitting.filter_shots_per_point(
                        idata, qdata, read_num,
                        threshold=self.cfg.device.readout.threshold[self.cfg.expt.qubits[0]],
                        pre_selection=True)
                else:
                    avgi_val = avgi[0][-1]
                    avgq_val = avgq[0][-1]

                avgi, avgq = avgi_val, avgq_val
                data['avgi'].append(avgi)
                data['avgq'].append(avgq)
                data['amps'].append(np.abs(avgi+1j*avgq)) # Calculating the magnitude
                data['phases'].append(np.angle(avgi+1j*avgq)) # Calculating the phase

        for key in 'avgi avgq amps phases'.split():
            data[key] = np.array(data[key])
            if sweep_dim == 2:
                data[key] = np.reshape(data[key], (len(outer_params), len(inner_params)))

        if self.cfg.expt.get('parity_check', False):
            idata_all = np.array(data['idata'])
            qdata_all = np.array(data['qdata'])
            _parity_start_idx= 0  
            if self.cfg.expt.get('active_reset', False):
                params = MMAveragerProgram.get_active_reset_params(self.cfg)
                _parity_start_idx += MMAveragerProgram.active_reset_read_num(**params)
            
            data['parity_idata'] = idata_all[..., _parity_start_idx::read_num]
            data['parity_qdata'] = qdata_all[..., _parity_start_idx::read_num]

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.data=data
        return data

    def analyze_multiparity(self):
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
        

class DarkBaseProgram(QsimBaseProgram):
    
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

        # Optional in-memory matrix for the ds_storage swaps. Rows are
        # affected modes and columns are the modes whose swap pulse is played.
        self.storage_phase_matrix = self.cfg.expt.get(
            "storage_phase_matrix", None)
        if self.storage_phase_matrix is not None:
            self.storage_phase_matrix = np.asarray(
                self.storage_phase_matrix, dtype=float)

        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1  # using first manipulate channel index needs to be fixed at some point

        
        # Register a gaussian envelope for each mode flagged 'arb'; per-mode sigma
        # comes from the dataset (cfg.expt.floquet_gauss_sigma overrides if given).
        # flat_top modes reuse MM_base's pi_m1si_low/high ramp waveforms (no buffer cost).
        for i_stor in range(7):
            if self.m1s_style[i_stor] != 'arb':
                continue
            stor_name = f"M1-S{i_stor+1}"
            ch = self.m1s_ch[i_stor]
            sig_us = self.cfg.expt.get("floquet_gauss_sigma", None)
            if sig_us is None:
                sig_us = self.swap_ds.get_gauss_sigma(stor_name)
            n_sig = self.swap_ds.get_gauss_n_sigma(stor_name)
            sigma = self.us2cycles(sig_us, gen_ch=ch)
            self.add_gauss(ch=ch, name=self.m1s_wf_name[i_stor], sigma=sigma, length=sigma * n_sig)

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
            
        if self.cfg.expt.perform_wigner or ('init_alpha' in self.cfg.expt):
            self.displace_man(setup=True, play=False)

        self.sync_all(200)

    def calculate_floquet_cycle_us(self):
        ecfg = self.cfg.expt
        cycle_us = len(ecfg.swap_stors) * self.cycles2us(ecfg.get("scramble_sync_cycles", 10))
        for stor in ecfg.swap_stors:
            index = stor - 1
            ch = self.m1s_ch[index]
            if self.m1s_style[index] == "arb": #I think naming this as 'arb' is not really a good convention
                sigma_us = ecfg.get("floquet_gauss_sigma", None) # I suspect this is needed now. Could be deleted
                if sigma_us is None:
                    sigma_us = self.swap_ds.get_gauss_sigma(f"M1-S{stor}")
                pulse_cycles = self.us2cycles(sigma_us, gen_ch=ch) * self.swap_ds.get_gauss_n_sigma(f"M1-S{stor}")
            else:
                ramp_cycles = self.pi_m1_sigma_low if self.m1s_is_low_freq[index] else self.pi_m1_sigma_high
                pulse_cycles = self.m1s_length[index] + 6 * ramp_cycles
            cycle_us += self.cycles2us(pulse_cycles, gen_ch=ch)
        return cycle_us

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
        """
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
        
    def _apply_dark_wait_phase_tracking(self, phase_offsets, wait_length):

        ecfg = self.cfg.expt

        if not ecfg.get("track_dark_wait_phase", True):
            return

        (
            swap_stors,
            stor_first,
            stor_last,
            idx_first,
            idx_last,
            n_first_full,
            n_last_half,
        ) = self._get_dark_swap_params()

        # MHz * us = cycles, so multiply by 360 to get degrees.
        rate_MHz = ecfg.get("dark_wait_phase_rate_MHz", 0.0)
        phase_deg = 360.0 * rate_MHz * wait_length

        # Optional static offset for fine tuning.
        phase_deg += ecfg.get("dark_wait_phase_offset_deg", 0.0)

        # Usually shift the last-storage half-swap phase, because that sets
        # the relative phase between stor_first and stor_last in the dark readout.
        phase_offsets[idx_last] = (phase_offsets[idx_last] + phase_deg) % 360.0

        if ecfg.get("debug", False):
            print(
                f"[DarkT1] wait phase tracking: wait={wait_length} us, "
                f"rate={rate_MHz} MHz, added={phase_deg % 360:.3f} deg, "
                f"phase_offsets={phase_offsets}"
            )
        
    def _mod360(self, phase_deg):
        return phase_deg % 360.0

    def _get_dark_swap_params(self):
        ecfg = self.cfg.expt

        swap_stors = list(ecfg.swap_stors)
        stor_first, stor_last = ecfg.dark_swap_order

        if stor_first not in swap_stors:
            raise ValueError(f"stor_first={stor_first} is not in swap_stors={swap_stors}")
        if stor_last not in swap_stors:
            raise ValueError(f"stor_last={stor_last} is not in swap_stors={swap_stors}")
        if stor_first == stor_last:
            raise ValueError("stor_first and stor_last must be different")

        idx_first = swap_stors.index(stor_first)
        idx_last = swap_stors.index(stor_last)

        # Existing convention:
        # m1s_pi_fracs[stor-1] fractional pulses = full pi swap.
        # half of that = pi/2 area in the old language, i.e. the half-swap used for dark readout.
        n_first_full = int(self.m1s_pi_fracs[stor_first - 1])
        n_last_half = int(self.m1s_pi_fracs[stor_last - 1] // 2)

        if self.cfg.expt.get("debug", False):
            print(
                f"[DarkT1] stor_first={stor_first}, stor_last={stor_last}, "
                f"n_first_full={n_first_full}, n_last_half={n_last_half}"
            )

        return swap_stors, stor_first, stor_last, idx_first, idx_last, n_first_full, n_last_half

    def _advance_phase_offsets(self, phase_offsets, swap_stors, pulsed_stor):
        """
        Preserve the existing phase tracking between Floquet pulses.

        This uses only the Floquet-to-Floquet matrix in ds_floquet:

            phase[stor_B] += get_phase_from("M1-S{stor_B}", "M1-S{pulsed_stor}")

        The separate decoder_phase_matrix is directional and must never be
        used here. It is applied only to the decoder storage swaps and f0g1.
        """
        pulsed_name = f"M1-S{pulsed_stor}"

        for j_stor, stor_B in enumerate(swap_stors):
            if stor_B == pulsed_stor:
                continue

            stor_B_name = f"M1-S{stor_B}"
            phase_shift = self.swap_ds.get_phase_from(
                stor_B_name, pulsed_name)

            phase_offsets[j_stor] += phase_shift
            phase_offsets[j_stor] = self._mod360(phase_offsets[j_stor])

    def _play_closed_floquet_cycle_pairs(
            self, n_cycle_pair, phase_offsets, swap_stors,
            extra_forward=False):
        update_phases = bool(self.cfg.expt.get("update_phases", True))
        sync_cycles = int(
            self.cfg.expt.get("scramble_sync_cycles", 10))
        pulse_args_by_stor = {
            stor: deepcopy(self.m1s_kwargs[stor - 1])
            for stor in swap_stors
        }
        if self.cfg.expt.zero_floquet_gain:
            for pulse_args in pulse_args_by_stor.values():
                pulse_args["gain"] = 0

        self.sync_all()

        for pair_index in range(n_cycle_pair):
            forward_phases = []
            for stor in swap_stors:
                stor_index = swap_stors.index(stor)
                phase_deg = self._mod360(
                    phase_offsets[stor_index])
                forward_phases.append((stor, phase_deg))

                pulse_args = pulse_args_by_stor[stor]
                pulse_args["phase"] = self.deg2reg(
                    phase_deg, gen_ch=pulse_args["ch"])
                self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)

                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

            # Replay the actual forward control phases in reverse order and
            # add 180 degrees.  Recomputing the inverse phases from the live
            # tracker would not be the adjoint of the emitted forward cycle.
            for stor, forward_phase_deg in reversed(forward_phases):
                inverse_phase_deg = self._mod360(
                    forward_phase_deg + 180.0
                )
                pulse_args = pulse_args_by_stor[stor]
                pulse_args["phase"] = self.deg2reg(
                    inverse_phase_deg, gen_ch=pulse_args["ch"])
                self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)

                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

            if self.cfg.expt.get("debug", False):
                print(
                    "[EntireCyclePhase] pair=",
                    pair_index,
                    "zero Floquet gain=",
                    self.cfg.expt.zero_floquet_gain,
                    "forward phases=",
                    forward_phases,
                )

        if extra_forward:
            for stor in swap_stors:
                stor_index = swap_stors.index(stor)
                phase_deg = self._mod360(phase_offsets[stor_index])
                pulse_args = pulse_args_by_stor[stor]
                pulse_args["phase"] = self.deg2reg(
                    phase_deg, gen_ch=pulse_args["ch"])
                self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)
                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )
        self.sync_all()

    def _advance_storage_phase_offsets(
            self, phase_offsets, swap_stors, pulsed_stor):
        """Advance later ds_storage swap phases after one ds_storage swap.

        Unlike the legacy Floquet matrix, this matrix may have a calibrated
        diagonal: ``matrix[i, i]`` is the active-access phase of mode i.
        """
        if self.storage_phase_matrix is None:
            return

        pulsed_index = swap_stors.index(pulsed_stor)
        for affected_index in range(len(swap_stors)):
            phase_offsets[affected_index] = self._mod360(
                phase_offsets[affected_index]
                + self.storage_phase_matrix[affected_index, pulsed_index]
            )

    def _advance_decoder_phase_offsets(
            self, decoder_phase_offsets, swap_stors, pulsed_stor):
        """Advance every decoder axis after one physical Floquet pulse."""
        pulse_column = swap_stors.index(pulsed_stor)
        for decoder_axis in range(len(decoder_phase_offsets)):
            decoder_phase_offsets[decoder_axis] = self._mod360(
                decoder_phase_offsets[decoder_axis]
                + self.decoder_phase_matrix[decoder_axis, pulse_column]
            )

    def _play_scramble_with_phase_offsets(
        self,
        phase_offsets,
        swap_stors,
        disorder_phase_offsets=None,
        decoder_phase_offsets=None,
    ):
        """
        Play the calibrated ordered Floquet pulse train while using the
        caller-provided phase_offsets as the live frame tracker.

        If cfg.expt.palindrome_scramble is True, consecutive Floquet cycles
        alternate direction:

            cycle 0: swap_stors
            cycle 1: reversed(swap_stors)
            cycle 2: swap_stors
            ...

        This keeps the number of pulses per floquet_cycle unchanged. With an
        even floquet_cycle, each forward cycle has a reverse partner.
        Otherwise it preserves the configured swap_stors order.

        Correct continuous sequence:

            load dark mode
                updates phase_offsets

            scramble
                emits pulses with current phase_offsets
                updates the same phase_offsets after every pulse

            read dark mode
                consumes the final phase_offsets

        Keeping this implementation on DarkBaseProgram makes the Floquet path
        independent of the implementation details in sideband_scramble.py.
        If ``decoder_phase_offsets`` is supplied, it is updated after every
        physical Floquet pulse using ``decoder_phase_matrix``.
        """
        ecfg = self.cfg.expt
        swap_stors = list(swap_stors)

        if len(phase_offsets) != len(swap_stors):
            raise ValueError(
                f"phase_offsets length {len(phase_offsets)} does not match "
                f"swap_stors length {len(swap_stors)}"
            )
        if decoder_phase_offsets is not None \
                and len(decoder_phase_offsets) != len(swap_stors) + 1:
            raise ValueError(
                "decoder_phase_offsets must be ordered as "
                "[M1 photon lowering, M1-S4, ...]"
            )
        raw_detunings = ecfg.get("detunings", None)
        if raw_detunings is None or raw_detunings is False:
            detunings = [0.0] * len(swap_stors)
        else:
            detunings = list(raw_detunings)
            if len(detunings) == 0:
                detunings = [0.0] * len(swap_stors)
        if len(detunings) != len(swap_stors):
            raise AssertionError(
                "length of detunings doesn't match that of swap_stors"
            )
        detunings = [float(d) for d in detunings]

        if disorder_phase_offsets is None:
            disorder_phase_offsets = [0.0] * len(swap_stors)
        if len(disorder_phase_offsets) != len(swap_stors):
            raise ValueError(
                f"disorder_phase_offsets length {len(disorder_phase_offsets)} "
                f"does not match swap_stors length {len(swap_stors)}"
            )

        update_phases = ecfg.get("update_phases", True)
        scramble_sync_cycles = int(ecfg.get("scramble_sync_cycles", 10))
        palindrome_scramble = bool(ecfg.get("palindrome_scramble", False))
        floquet_hardware_loop = bool(ecfg.get("floquet_hardware_loop", False))
        floquet_cycle = int(ecfg.floquet_cycle)

        if floquet_cycle < 0:
            raise ValueError("floquet_cycle must be non-negative")
        if floquet_hardware_loop and palindrome_scramble:
            raise ValueError(
                "floquet_hardware_loop does not support palindrome_scramble"
            )

        # Deep copy the calibrated Floquet pulse parameters before applying
        # the per-storage detunings for this experiment.
        all_pulse_args = []
        for i_stor, stor in enumerate(swap_stors):
            pulse_args = deepcopy(self.m1s_kwargs[stor - 1])
            pulse_args["freq"] += self.freq2reg(
                detunings[i_stor],
                gen_ch=pulse_args["ch"],
            )
            all_pulse_args.append(pulse_args)

        pulse_us_by_stor = []
        for i_stor, stor in enumerate(swap_stors):
            stor_name = f"M1-S{stor}"
            if self.m1s_style[stor - 1] == "arb":
                sig_us = ecfg.get("floquet_gauss_sigma", None)
                if sig_us is None:
                    sig_us = self.swap_ds.get_gauss_sigma(stor_name)
                pulse_us = float(sig_us) * float(
                    self.swap_ds.get_gauss_n_sigma(stor_name)
                )
            else:
                # QICK flat_top length is only the plateau.  The registered
                # pi_m1si_low/high Gaussian ramp is 6 sigma in total and is
                # part of the physical time over which a detuned frame
                # advances.
                ramp_sigma_us = float(
                    self.cfg.device.manipulate.ramp_sigma)
                pulse_us = (
                    float(self.swap_ds.get_len(stor_name))
                    + 6.0 * ramp_sigma_us
                )
            pulse_us_by_stor.append(pulse_us)

        forward_sequence = list(range(len(swap_stors)))
        reverse_sequence = list(reversed(forward_sequence))

        scramble_sync_us = float(self.cycles2us(scramble_sync_cycles))
        scramble_elapsed_us = floquet_cycle * (
            sum(pulse_us_by_stor) + len(swap_stors) * scramble_sync_us
        )

        self.sync_all()

        if ecfg.get("debug", False):
            print("[DarkScramble] using shared phase_offsets for scramble")
            print("[DarkScramble] initial phase_offsets:", phase_offsets)
            print("[DarkScramble] initial disorder_phase_offsets:", disorder_phase_offsets)
            print("[DarkScramble] detunings MHz:", detunings)
            print("[DarkScramble] scramble sync cycles:", scramble_sync_cycles)
            print("[DarkScramble] palindrome scramble:", palindrome_scramble)
            print("[DarkScramble] hardware loop:", floquet_hardware_loop)
            print("[DarkScramble] forward sequence:", swap_stors)
            if palindrome_scramble:
                print("[DarkScramble] reverse sequence:", list(reversed(swap_stors)))
                if floquet_cycle % 2:
                    print(
                        "[DarkScramble] odd floquet_cycle leaves one unpaired "
                        "forward cycle"
                    )
            print("[DarkScramble] scramble elapsed us:", scramble_elapsed_us)
            print("[DarkScramble] pulse args:", all_pulse_args)

        if floquet_hardware_loop and floquet_cycle > 0 and swap_stors:
            first_cycle_phases = [0.0] * len(swap_stors)
            phase_offsets_after_cycle = list(phase_offsets)

            # Later pulses include the shifts from earlier pulses in cycle 0.
            for i_stor in forward_sequence:
                stor = swap_stors[i_stor]
                first_cycle_phases[i_stor] = self._mod360(
                    phase_offsets_after_cycle[i_stor]
                )
                if update_phases:
                    self._advance_phase_offsets(
                        phase_offsets=phase_offsets_after_cycle,
                        swap_stors=swap_stors,
                        pulsed_stor=stor,
                    )

            phase_step_per_cycle = [
                self._mod360(phase_after - phase_before)
                for phase_before, phase_after in zip(
                    phase_offsets,
                    phase_offsets_after_cycle,
                )
            ]

            phase_registers = []
            next_register_by_page = {}
            # Raw registers must be on the same page as the generator phase.
            for pulse_args in all_pulse_args:
                ch = pulse_args["ch"]
                page = self.ch_page(ch)
                if page == 0:
                    raise RuntimeError(
                        "floquet_hardware_loop cannot use page 0 scratch registers"
                    )

                phase_register = next_register_by_page.get(page, 1)
                phase_step_register = phase_register + 1
                next_register_by_page[page] = phase_step_register + 1
                phase_registers.append(
                    (page, phase_register, phase_step_register)
                )

            loop_page = phase_registers[0][0]
            loop_register = next_register_by_page.get(loop_page, 1)
            next_register_by_page[loop_page] = loop_register + 1

            register_maps = list(self._gen_regmap.values()) + list(
                self._ro_regmap.values()
            )
            for page, next_register in next_register_by_page.items():
                first_special_register = min(
                    register
                    for register_page, register in register_maps
                    if register_page == page and register > 0
                )
                if next_register > first_special_register:
                    raise RuntimeError(
                        "floquet_hardware_loop does not have enough scratch "
                        f"registers on page {page}"
                    )

            for i_stor, pulse_args in enumerate(all_pulse_args):
                ch = pulse_args["ch"]
                gen_manager_name = self._gen_mgrs[ch].__class__.__name__
                if gen_manager_name != "FullSpeedGenManager":
                    raise RuntimeError(
                        "floquet_hardware_loop requires a full-speed generator; "
                        f"channel {ch} uses {gen_manager_name}"
                    )

                page, phase_register, phase_step_register = \
                    phase_registers[i_stor]
                self.safe_regwi(
                    page,
                    phase_register,
                    self.deg2reg(first_cycle_phases[i_stor], gen_ch=ch),
                )
                self.safe_regwi(
                    page,
                    phase_step_register,
                    self.deg2reg(phase_step_per_cycle[i_stor], gen_ch=ch),
                )

            self.safe_regwi(loop_page, loop_register, floquet_cycle - 1)

            floquet_loop_number = getattr(self, "_floquet_loop_number", 0)
            self._floquet_loop_number = floquet_loop_number + 1
            floquet_loop_label = f"FLOQUET_LOOP_{floquet_loop_number}"

            # Configure the next waveform while the current pulse is playing.
            # This leaves the original 10-cycle setup margin unchanged.
            first_pulse_args = all_pulse_args[forward_sequence[0]]
            first_pulse_args["phase"] = 0
            self.set_pulse_registers(**first_pulse_args)
            self.label(floquet_loop_label)

            for step_idx, i_stor in enumerate(forward_sequence):
                stor = swap_stors[i_stor]
                pulse_args = all_pulse_args[i_stor]
                ch = pulse_args["ch"]
                page, phase_register, phase_step_register = \
                    phase_registers[i_stor]

                self.mathi(
                    page,
                    self.sreg(ch, "phase"),
                    phase_register,
                    "+",
                    0,
                )

                if ecfg.get("debug", False):
                    print(
                        f"[DarkScramble] hardware step={step_idx}, "
                        f"stor={stor}, "
                        f"first_phase_deg={first_cycle_phases[i_stor]:.3f}, "
                        f"phase_step_deg={phase_step_per_cycle[i_stor]:.3f}"
                    )

                self.pulse(ch)
                self.math(
                    page,
                    phase_register,
                    phase_register,
                    "+",
                    phase_step_register,
                )

                next_i_stor = forward_sequence[
                    (step_idx + 1) % len(forward_sequence)
                ]
                next_pulse_args = all_pulse_args[next_i_stor]
                next_pulse_args["phase"] = 0
                self.set_pulse_registers(**next_pulse_args)
                self.sync_all(scramble_sync_cycles)

            self.loopnz(loop_page, loop_register, floquet_loop_label)

            for i_stor in range(len(swap_stors)):
                phase_offsets[i_stor] = self._mod360(
                    phase_offsets[i_stor]
                    + floquet_cycle * phase_step_per_cycle[i_stor]
                )

            if decoder_phase_offsets is not None:
                for _ in range(floquet_cycle):
                    for stor in swap_stors:
                        self._advance_decoder_phase_offsets(
                            decoder_phase_offsets=decoder_phase_offsets,
                            swap_stors=swap_stors,
                            pulsed_stor=stor,
                        )
        else:
            for kk in range(floquet_cycle):
                if palindrome_scramble and kk % 2:
                    cycle_sequence = reverse_sequence
                else:
                    cycle_sequence = forward_sequence

                for step_idx, i_stor in enumerate(cycle_sequence):
                    stor = swap_stors[i_stor]
                    pulse_args = all_pulse_args[i_stor]

                    phase_deg = self._mod360(phase_offsets[i_stor])
                    pulse_args["phase"] = self.deg2reg(
                        phase_deg,
                        gen_ch=pulse_args["ch"],
                    )

                    if ecfg.get("debug", False) and kk == 0:
                        print(
                            f"[DarkScramble] cycle={kk}, step={step_idx}, "
                            f"stor={stor}, phase_deg={phase_deg:.3f}, "
                            f"stark_phase={phase_offsets[i_stor]:.3f}"
                        )

                    self.setup_and_pulse(**pulse_args)
                    self.sync_all(scramble_sync_cycles)

                    if decoder_phase_offsets is not None:
                        self._advance_decoder_phase_offsets(
                            decoder_phase_offsets=decoder_phase_offsets,
                            swap_stors=swap_stors,
                            pulsed_stor=stor,
                        )

                    if update_phases:
                        self._advance_phase_offsets(
                            phase_offsets=phase_offsets,
                            swap_stors=swap_stors,
                            pulsed_stor=stor,
                        )

        for j_stor, detuning_MHz in enumerate(detunings):
            disorder_phase_offsets[j_stor] = self._mod360(
                disorder_phase_offsets[j_stor]
                + 360.0 * detuning_MHz * scramble_elapsed_us
            )

        if ecfg.get("debug", False):
            print("[DarkScramble] final phase_offsets:", phase_offsets)
            print("[DarkScramble] final disorder_phase_offsets:", disorder_phase_offsets)

        self.sync_all()

    def _play_m1s_frac_train(
        self,
        stor,
        n_frac,
        phase_offsets,
        swap_stors,
        disorder_phase_offsets=None,
        logical_phase_deg=0.0,
        logical_phase_step_deg=0.0,
        inverse=False,
        update_phases=True,
        label="",
    ):
        """
        Play n_frac copies of the calibrated M1-S{stor} fractional pulse.

        logical_phase_deg:
            Desired logical phase of this beam-splitter pulse.

        logical_phase_step_deg:
            Phase added after every physical fractional pulse.  A value of
            180 degrees emits the sequence +, -, +, -, ... in one train.

        inverse:
            If True, implements U^\dagger by adding 180 degrees to the pulse phase.
            This uses U(-theta, phi) = U(theta, phi + 180 deg).

        phase_offsets:
            Mutable list tracking calibrated Stark/off-resonant frame
            corrections from previous pulses.

        disorder_phase_offsets:
            Optional synthetic-disorder rotating-frame phases. These are added
            to the dark load/readout pulse axes but are not advanced inside the
            load/readout sequence; they represent the frame accumulated before
            the analyzer starts.
        """
        n_frac = int(n_frac)
        if n_frac <= 0:
            return

        idx = swap_stors.index(stor)
        pulse_args = deepcopy(self.m1s_kwargs[stor - 1])

        disorder_phase_deg = 0.0
        if disorder_phase_offsets is not None:
            disorder_phase_deg = disorder_phase_offsets[idx]

        inverse_phase = 180.0 if inverse else 0.0
        first_phase_deg = self._mod360(
            phase_offsets[idx]
            + disorder_phase_deg
            + logical_phase_deg
            + inverse_phase
        )
        phase_step_deg = self._mod360(logical_phase_step_deg)
        sync_cycles = int(
            self.cfg.expt.get("scramble_sync_cycles", 10))
        hardware_loop = bool(
            self.cfg.expt.get("floquet_hardware_loop", False))

        if self.cfg.expt.get("debug", False):
            direction = "inverse" if inverse else "forward"
            print(
                f"[DarkT1] {label}: stor={stor}, {direction}, "
                f"n_frac={n_frac}, phase_deg={first_phase_deg:.3f}, "
                f"phase_step_deg={phase_step_deg:.3f}, "
                f"phase_offset={phase_offsets[idx]:.3f}, "
                f"disorder_phase={disorder_phase_deg:.3f}, "
                f"logical_phase={logical_phase_deg:.3f}, "
                f"hardware_loop={hardware_loop}"
            )

        if hardware_loop:
            ch = pulse_args["ch"]
            page = self.ch_page(ch)
            if page == 0: #<--- This part should be reviewed; not sure if page 0 is fully forbidden, but seems this is added to avoid
                          #the collision with other registers assigned for RAverager or hardware loop
                raise RuntimeError(
                    "floquet_hardware_loop cannot use page 0 scratch registers"
                )
            if self._gen_mgrs[ch].__class__.__name__ != "FullSpeedGenManager": #This is done because only full speed generator has separate phase sreg.
                raise RuntimeError(
                    "floquet_hardware_loop requires a full-speed generator; "
                    f"channel {ch} uses "
                    f"{self._gen_mgrs[ch].__class__.__name__}"
                )

            phase_register = 1
            phase_step_register = 2
            loop_register = 3
            register_maps = list(self._gen_regmap.values()) + list(
                self._ro_regmap.values()) #In general, _gen_regmap[(ch, "freg")] = (page, register), so values() collapses those into
                                          #list, which is used to get the lowest value of sreg for the page. 
            first_special_register = min(
                register
                for register_page, register in register_maps
                if register_page == page and register > 0
            ) #This is done as QICK allocates sregs from the highest numbers
            if loop_register >= first_special_register:
                raise RuntimeError(
                    "floquet_hardware_loop does not have enough scratch "
                    f"registers on page {page}"
                )

            self.safe_regwi(
                page,
                phase_register,
                self.deg2reg(first_phase_deg, gen_ch=ch),
            )
            self.safe_regwi(
                page,
                phase_step_register,
                self.deg2reg(phase_step_deg, gen_ch=ch),
            )
            self.safe_regwi(page, loop_register, n_frac - 1)

            pulse_args["phase"] = 0
            self.set_pulse_registers(**pulse_args)

            floquet_loop_number = getattr(
                self, "_floquet_loop_number", 0)
            self._floquet_loop_number = floquet_loop_number + 1
            loop_label = f"FLOQUET_FRAC_LOOP_{floquet_loop_number}"
            self.label(loop_label)

            self.mathi(
                page,
                self.sreg(ch, "phase"),
                phase_register,
                "+",
                0,
            )
            self.pulse(ch)
            self.math(
                page,
                phase_register,
                phase_register,
                "+",
                phase_step_register,
            )
            self.sync_all(sync_cycles)
            self.loopnz(page, loop_register, loop_label)
        else:
            for kk in range(n_frac):
                phase_deg = self._mod360(
                    first_phase_deg + kk * phase_step_deg)
                pulse_args["phase"] = self.deg2reg(
                    phase_deg,
                    gen_ch=pulse_args["ch"],
                )
                self.setup_and_pulse(**pulse_args)
                self.sync_all(sync_cycles)

        if update_phases:
            for _ in range(n_frac):
                self._advance_phase_offsets(
                    phase_offsets=phase_offsets,
                    swap_stors=swap_stors,
                    pulsed_stor=stor,
                )
    
    def _accumulate_scramble_phases(self, phase_offsets, swap_stors):
        if not self.cfg.expt.get("update_phases", True):
            return
        for _ in range(self.cfg.expt.floquet_cycle):
            for stor in swap_stors:
                self._advance_phase_offsets(
                    phase_offsets=phase_offsets,
                    swap_stors=swap_stors,
                    pulsed_stor=stor,
                )

    def _prepare_dark_mode(self, phase_offsets, disorder_phase_offsets=None):
        """
        Prepare the same dark/normal mode that the old readout block measures.

        Old readout map:

            first full swap, then last half swap

        As an operator:

            R_read = U_last_half U_first_full

        Therefore prepare is inverse:

            R_prep = R_read^\dagger
                   = U_first_full^\dagger U_last_half^\dagger

        In actual time order:

            last half inverse first,
            first full inverse second.
        """
        (
            swap_stors,
            stor_first,
            stor_last,
            idx_first,
            idx_last,
            n_first_full,
            n_last_half,
        ) = self._get_dark_swap_params()

        update_phases = self.cfg.expt.get("update_phases", True)
        second_rel_phase = self.cfg.expt.get("second_rel_phase", 0.0)

        # 1. inverse of the old second pulse: last half-swap inverse
        self._play_m1s_frac_train(
            stor=stor_last,
            n_frac=n_last_half,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
            logical_phase_deg=second_rel_phase,
            inverse=True,
            update_phases=update_phases,
            label="prepare: inverse last half-swap",
        )

        # 2. inverse of the old first pulse: first full-swap inverse
        self._play_m1s_frac_train(
            stor=stor_first,
            n_frac=n_first_full,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
            logical_phase_deg=0.0,
            inverse=True,
            update_phases=update_phases,
            label="prepare: inverse first full-swap",
        )

        self.sync_all()

    def _read_dark_mode(self, phase_offsets, disorder_phase_offsets=None):
        """
        Original dark readout block:

            first full swap,
            then last half swap.

        This maps the selected dark/normal mode back into M1.
        """
        (
            swap_stors,
            stor_first,
            stor_last,
            idx_first,
            idx_last,
            n_first_full,
            n_last_half,
        ) = self._get_dark_swap_params()

        update_phases = self.cfg.expt.get("update_phases", True)
        second_rel_phase = self.cfg.expt.get("second_rel_phase", 0.0)

        # 1. old first pulse: first full swap
        self._play_m1s_frac_train(
            stor=stor_first,
            n_frac=n_first_full,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
            logical_phase_deg=0.0,
            inverse=False,
            update_phases=update_phases,
            label="readout: first full-swap",
        )
        virtual_ramsey_phase = 0.0
        if self.cfg.expt.get("dark_virtual_ramsey", False):
            virtual_ramsey_phase = (
                self.cfg.expt.get("dark_virtual_ramsey_phase_per_cycle_deg", 0.0)
                * self.cfg.expt.get("floquet_cycle", 0)
            )
            virtual_ramsey_phase += self.cfg.expt.get("virtual_ramsey_phase_offset_deg", 0.0)
            virtual_ramsey_phase = self._mod360(virtual_ramsey_phase)
            if self.cfg.expt.get("debug", False):
                print(f"Dark fixed second half swap ramsey: {virtual_ramsey_phase}")

        second_logical_phase = self._mod360(second_rel_phase + virtual_ramsey_phase)
                        
        # 2. old second pulse: last half swap
        self._play_m1s_frac_train(
            stor=stor_last,
            n_frac=n_last_half,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
            logical_phase_deg=second_logical_phase,
            inverse=False,
            update_phases=update_phases,
            label="readout: last half-swap",
        )

        self.sync_all()

    def get_dark_swap_params_large_support(self):
        """
        Length-4 analog of _get_dark_swap_params.

        Expects cfg.expt.dark_swap_order = [m1, m2, m3, m4], where m1 is the
        storage that ends up holding the final amplitude after the readout
        sequence (i.e. the storage that R_m1(-pi) is applied to last).

        Returns:
            swap_stors: list of all storages participating in scrambling
            stors:      [m1, m2, m3, m4] storage indices
            idxs:       positions of stors[i] inside swap_stors
            n_full:     per-storage full pi-swap fractional-pulse counts
            n_half:     per-storage half pi-swap fractional-pulse counts
        """
        ecfg = self.cfg.expt

        swap_stors = list(ecfg.swap_stors)
        stors = list(ecfg.dark_swap_order)

        if len(stors) != 4:
            raise ValueError(
                f"dark_swap_order must have length 4 for large support, got {stors}"
            )
        if len(set(stors)) != 4:
            raise ValueError(f"dark_swap_order entries must be distinct, got {stors}")
        for s in stors:
            if s not in swap_stors:
                raise ValueError(f"stor={s} is not in swap_stors={swap_stors}")

        idxs = [swap_stors.index(s) for s in stors]
        n_full = [int(self.m1s_pi_fracs[s - 1]) for s in stors]
        n_half = [int(self.m1s_pi_fracs[s - 1] // 2) for s in stors]

        if ecfg.get("debug", False):
            print(
                f"[DarkLarge] stors={stors}, idxs={idxs}, "
                f"n_full={n_full}, n_half={n_half}"
            )

        return swap_stors, stors, idxs, n_full, n_half

    # def _read_large_dark(self, phase_offsets):
    #     """
    #     Length-4 dark/normal-mode readout:

    #         R_m2(+pi) -> R_m1(pi/2) -> R_m2(-pi)
    #         -> R_m4(+pi) -> R_m3(pi/2) -> R_m4(-pi)
    #         -> R_m3(+pi) -> R_m1(pi/2) -> R_m3(-pi)
    #         -> R_m1(-pi)

    #     Maps the selected length-4 dark/normal mode back into M1.
    #     """
    #     swap_stors, stors, _idxs, n_full, n_half = (
    #         self.get_dark_swap_params_large_support()
    #     )
    #     m1, m2, m3, m4 = stors
    #     n_full_1, n_full_2, n_full_3, n_full_4 = n_full
    #     n_half_1, _n_half_2, n_half_3, _n_half_4 = n_half

    #     update_phases = self.cfg.expt.get("update_phases", True)

    #     # (stor, n_frac, inverse, label) -- in time order
    #     sequence = [
    #         (m2, n_full_2, False, "large: R_m2(+pi)"),
    #         (m1, n_half_1, False, "large: R_m1(pi/2) #1"),
    #         (m2, n_full_2, True,  "large: R_m2(-pi)"),
    #         (m4, n_full_4, False, "large: R_m4(+pi)"),
    #         (m3, n_half_3, False, "large: R_m3(pi/2)"),
    #         (m4, n_full_4, True,  "large: R_m4(-pi)"),
    #         (m3, n_full_3, False, "large: R_m3(+pi)"),
    #         (m1, n_half_1, False, "large: R_m1(pi/2) #2"),
    #         (m3, n_full_3, True,  "large: R_m3(-pi)"),
    #         (m1, n_full_1, True,  "large: R_m1(-pi)"),
    #     ]

    #     for stor, n_frac, inverse, label in sequence:
    #         self._play_m1s_frac_train(
    #             stor=stor,
    #             n_frac=n_frac,
    #             phase_offsets=phase_offsets,
    #             swap_stors=swap_stors,
    #             logical_phase_deg=0.0,
    #             inverse=inverse,
    #             update_phases=update_phases,
    #             label=label,
    #         )

    #     self.sync_all()

    def _get_large_dark_read_sequence(self):
        """
        Return the length-4 dark/normal-mode readout sequence in actual
        time order.

        Tuple convention:
            (stor, n_frac, logical_phase_deg, inverse, label)

        inverse=False means +area with the requested logical phase.
        inverse=True means the adjoint of that area, implemented in
        _play_m1s_frac_train by adding 180 deg to the pulse phase.
        """
        swap_stors, stors, _idxs, n_full, n_half = (
            self.get_dark_swap_params_large_support()
        )
        m1, m2, m3, m4 = stors
        n_full_1, n_full_2, n_full_3, n_full_4 = n_full
        n_half_1, _n_half_2, n_half_3, _n_half_4 = n_half

        # This is the readout sequence from the comment in _read_large_dark.
        # It maps the selected length-4 dark/normal mode back into M1.
        sequence = [
            (m2, n_full_2, 0.0, False, "large read: R_m2(+pi)"),
            (m1, n_half_1, 0.0, False, "large read: R_m1(pi/2) #1"),
            (m2, n_full_2, 0.0, True,  "large read: R_m2(-pi)"),
            (m4, n_full_4, 0.0, False, "large read: R_m4(+pi)"),
            (m3, n_half_3, 0.0, False, "large read: R_m3(pi/2)"),
            (m4, n_full_4, 0.0, True,  "large read: R_m4(-pi)"),
            (m3, n_full_3, 0.0, False, "large read: R_m3(+pi)"),
            (m1, n_half_1, 0.0, False, "large read: R_m1(pi/2) #2"),
            (m3, n_full_3, 0.0, True,  "large read: R_m3(-pi)"),
            (m1, n_full_1, 0.0, True,  "large read: R_m1(-pi)"),
        ]
        return swap_stors, sequence

    def _large_dark_fraction_to_n_frac(
        self,
        stor,
        numerator,
        denominator,
        n_full,
        label,
    ):
        """
        Convert an exact rational multiple of pi to fractional-pulse count.

        The floquet dataset convention is:

            n_full fractional pulses = pi beamsplitter area.

        Therefore (numerator / denominator) * pi requires
        n_full * numerator / denominator fractional pulses.  This helper is
        exact and never rounds.
        """
        n_frac_num = int(n_full) * int(numerator)
        if n_frac_num % int(denominator) != 0:
            raise ValueError(
                f"{label}: exact large-dark direct sequence needs "
                f"{numerator}/{denominator} of a pi pulse on M1-S{stor}, "
                f"but n_full={n_full} does not make an integer fractional "
                "pulse count."
            )
        n_frac = n_frac_num // int(denominator)

        if self.cfg.expt.get("debug", False):
            print(
                f"[DarkLargeDirect] {label}: stor={stor}, "
                f"target_angle/pi={numerator}/{denominator}, "
                f"n_frac={n_frac}"
            )

        return n_frac

    def _get_large_dark_direct_read_sequence(self):
        """
        Five-pulse direct readout for the same length-4 mode targeted by
        _get_large_dark_read_sequence(), up to the beamsplitter phase
        convention used by _play_m1s_frac_train.

        The old 10-pulse sequence synthesizes pairwise storage-storage
        rotations through M1.  If we only need to map the selected mode

            (m1 - m2 - m3 + m4) / 2

        back to M1, the following exact direct sequence is enough:

            read:  R_m1(+pi,   phase=180)
                -> R_m2(+pi/2, phase=180)
                -> R_m3(+pi,   phase=0)
                -> R_m4(-pi/2, phase=0)
                -> R_m2(+pi/2, phase=0)

        Its inverse is used for load by _invert_large_dark_sequence().
        This is not the same full unitary on the orthogonal storage modes;
        it is equivalent for loading/readout of the selected large-dark mode.

        This path only uses pi and pi/2 areas, so it is exact whenever the
        relevant n_full values make pi/2 an integer fractional-pulse count.
        """
        swap_stors, stors, _idxs, n_full, _n_half = (
            self.get_dark_swap_params_large_support()
        )
        m1, m2, m3, m4 = stors
        n_full_1, n_full_2, n_full_3, n_full_4 = n_full

        n_m1_full = self._large_dark_fraction_to_n_frac(
            m1, 1, 1, n_full_1, "direct read: R_m1(pi)"
        )
        n_m2_half = self._large_dark_fraction_to_n_frac(
            m2, 1, 2, n_full_2, "direct read: R_m2(pi/2)"
        )
        n_m3_full = self._large_dark_fraction_to_n_frac(
            m3, 1, 1, n_full_3, "direct read: R_m3(pi)"
        )
        n_m4_half = self._large_dark_fraction_to_n_frac(
            m4, 1, 2, n_full_4, "direct read: R_m4(pi/2)"
        )

        sequence = [
            (m1, n_m1_full, 180.0, False, "large direct read: R_m1(pi)"),
            (m2, n_m2_half, 180.0, False, "large direct read: R_m2(pi/2) #1"),
            (m3, n_m3_full, 0.0, False, "large direct read: R_m3(pi)"),
            (m4, n_m4_half, 0.0, True, "large direct read: R_m4(-pi/2)"),
            (m2, n_m2_half, 0.0, False, "large direct read: R_m2(pi/2) #2"),
        ]
        return swap_stors, sequence

    def _invert_large_dark_sequence(self, sequence, label_prefix="large load"):
        """
        Build the adjoint sequence.

        For load, we need R_read^dagger.  Therefore we reverse the readout
        time order and invert each pulse.  The calibrated dynamic phase
        tracking is still performed in the actual time order by
        _play_m1s_frac_train; do not reverse or subtract phase_offsets here.
        """
        inv_sequence = []
        for stor, n_frac, logical_phase_deg, inverse, label in reversed(sequence):
            inv_sequence.append((
                stor,
                n_frac,
                logical_phase_deg,
                not inverse,
                f"{label_prefix}: inverse of [{label}]",
            ))
        return inv_sequence

    def _play_large_dark_sequence(
        self,
        phase_offsets,
        swap_stors,
        sequence,
        disorder_phase_offsets=None,
    ):
        """
        Play a length-4 dark/load/read sequence while tracking all storage
        pulse-frame offsets after every fractional M1-S pulse.
        """
        update_phases = self.cfg.expt.get("update_phases", True)

        for stor, n_frac, logical_phase_deg, inverse, label in sequence:
            self._play_m1s_frac_train(
                stor=stor,
                n_frac=n_frac,
                phase_offsets=phase_offsets,
                swap_stors=swap_stors,
                disorder_phase_offsets=disorder_phase_offsets,
                logical_phase_deg=logical_phase_deg,
                inverse=inverse,
                update_phases=update_phases,
                label=label,
            )

        self.sync_all()

    def _read_large_dark(self, phase_offsets, disorder_phase_offsets=None):
        """
        Length-4 dark/normal-mode readout:

            R_m2(+pi) -> R_m1(pi/2) -> R_m2(-pi)
            -> R_m4(+pi) -> R_m3(pi/2) -> R_m4(-pi)
            -> R_m3(+pi) -> R_m1(pi/2) -> R_m3(-pi)
            -> R_m1(-pi)

        Maps the selected length-4 dark/normal mode back into M1.
        """
        if self.cfg.expt.get("large_dark_direct_sequence", False):
            swap_stors, sequence = self._get_large_dark_direct_read_sequence()
        else:
            swap_stors, sequence = self._get_large_dark_read_sequence()
        self._play_large_dark_sequence(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            sequence=sequence,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def _load_large_dark(self, phase_offsets, disorder_phase_offsets=None):
        """
        Length-4 dark/normal-mode load:

            R_load = R_read^dagger

        In actual time order this is the readout sequence reversed, with
        each pulse replaced by its inverse.  This maps an M1 excitation into
        the same selected length-4 dark/normal mode that _read_large_dark()
        later maps back to M1.
        """
        if self.cfg.expt.get("large_dark_direct_sequence", False):
            swap_stors, read_sequence = self._get_large_dark_direct_read_sequence()
        else:
            swap_stors, read_sequence = self._get_large_dark_read_sequence()
        load_sequence = self._invert_large_dark_sequence(
            read_sequence,
            label_prefix="large load",
        )
        self._play_large_dark_sequence(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            sequence=load_sequence,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def _prepare_large_dark_mode(self, phase_offsets, disorder_phase_offsets=None):
        """Alias kept for consistency with _prepare_dark_mode()."""
        self._load_large_dark(
            phase_offsets,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def body(self):
        cfg=AttrDict(self.cfg)
        slow_pi_ge_readout = bool(
            cfg.expt.get("slow_pi_ge_readout", False)
        )

        if slow_pi_ge_readout and (
                cfg.expt.get("parity_readout", False)
                or cfg.expt.get("multiparity_readout", False)):
            raise ValueError(
                "slow_pi_ge_readout cannot be combined with parity readout"
            )

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
                print("running")
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

        # postpulse
        if cfg.expt.postpulse:

            # Move ro_stor to man
            postpulse_cfg = [ ['storage', f'M1-S{ro_stor}', 'pi', 0,] ] if ro_stor > 0 else []
            
            skip_default_m1_to_qubit_readout = (
                self.cfg.expt.get("parity_readout", False)
                or self.cfg.expt.get("multiparity_readout", False)
                or slow_pi_ge_readout
            )

            if not self.cfg.expt.perform_wigner and not skip_default_m1_to_qubit_readout:
                # Move man to qubit for population measurement
                postpulse_cfg.append(['man', 'M1', 'pi', 0,])
                if self.cfg.expt.get('map_to_qubit_ge', False):
                    postpulse_cfg.append(['qubit', 'ef', 'pi', 0,])

            pulse_creator = self.get_prepulse_creator(postpulse_cfg)
            self.sync_all()
            self.custom_pulse(cfg, pulse_creator.pulse, prefix='post_')
            self.sync_all()

            if not self.cfg.expt.perform_wigner and (self.cfg.expt.get("parity_readout", False) or self.cfg.expt.get("multiparity_readout", False)):
                
                if not self.cfg.expt.get("multiparity_readout", False):
                    if self.cfg.expt.get("debug", False):
                        print("Performing parity readout with parity pulse")
                    self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.get("phase_second_pulse", 180), fast=self.cfg.expt.parity_fast)
                if self.cfg.expt.get("multiparity_readout", False):
                    if self.cfg.expt.get("debug", False):
                        print("Performing multiparity readout with parity pulse")
                    self.multi_parity_readout(fast = self.cfg.expt.get("parity_fast", False))
                self.sync_all()
                
            if self.cfg.expt.perform_wigner:
                # Population is still in man, perform displacement + parity measurement

                # Displacement
                self.displace_man(
                    alpha=cfg.expt.wigner_alpha,
                    setup=False,
                    play=True,
                    )
                
                # Parity pulse on qubit
                self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.phase_second_pulse, fast=self.cfg.expt.parity_fast)

        if slow_pi_ge_readout:
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


class DarkBaseRProgram(MMRAveragerProgram):
    """RAverager counterpart of ``DarkBaseProgram``.

    The pulse-building methods below do not depend on the AveragerProgram
    software loop, so both program types use the same implementations.
    Concrete RAverager programs only need to define ``core_pulses`` and
    ``update``.
    """

    _pre_selection_filtering = True

    retrieve_swap_parameters = QsimBaseProgram.retrieve_swap_parameters #borrowing methods
    prep_man_fock_state = DarkBaseProgram.prep_man_fock_state #borrowing methods
    multi_parity_readout = DarkBaseProgram.multi_parity_readout #borrowing methods
    body = DarkBaseProgram.body #borrowing methods

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()

        self.swap_ds = self.cfg.device.storage._ds_floquet
        self.retrieve_swap_parameters()

        self.storage_phase_matrix = self.cfg.expt.get(
            "storage_phase_matrix", None)
        if self.storage_phase_matrix is not None:
            self.storage_phase_matrix = np.asarray(
                self.storage_phase_matrix, dtype=float)

        man_mode_no = self.cfg.expt.get("man_mode_no", 1)
        self.man_mode_idx = man_mode_no - 1

        for stor in range(7):
            if self.m1s_style[stor] != "arb":
                continue

            stor_name = f"M1-S{stor + 1}"
            ch = self.m1s_ch[stor]
            sigma_us = self.cfg.expt.get("floquet_gauss_sigma", None)
            if sigma_us is None:
                sigma_us = self.swap_ds.get_gauss_sigma(stor_name)
            n_sigma = self.swap_ds.get_gauss_n_sigma(stor_name)
            sigma = self.us2cycles(sigma_us, gen_ch=ch)

            self.add_gauss(
                ch=ch,
                name=self.m1s_wf_name[stor],
                sigma=sigma,
                length=sigma * n_sigma,
            )

        self.m1s_kwargs = []
        for stor in range(7):
            pulse_kwargs = {
                "ch": self.m1s_ch[stor],
                "style": self.m1s_style[stor],
                "freq": self.m1s_freq[stor],
                "phase": 0,
                "gain": self.m1s_gain[stor],
                "waveform": self.m1s_wf_name[stor],
            }
            if self.m1s_style[stor] != "arb":
                pulse_kwargs["length"] = self.m1s_length[stor]
            self.m1s_kwargs.append(pulse_kwargs)

        self.sync_all(200)


class ManStorMultiparityChevronRProgram(DarkBaseRProgram):
    """Hardware frequency sweep for one fixed M1-storage pulse length."""

    def initialize(self):
        super().initialize()

        cfg = AttrDict(self.cfg)
        swap_stor = int(cfg.expt.swap_stor)
        pulse_creator = self.get_prepulse_creator([
            ["storage", f"M1-S{swap_stor}", "pi", 0.0],
        ])
        pulse = pulse_creator.pulse

        self.chevron_ch = int(pulse[4][0])
        self.chevron_page = self.ch_page(self.chevron_ch)
        self.chevron_freq_register = self.sreg(
            self.chevron_ch, "freq")
        self.chevron_freq_sweep_register = 4
        self.chevron_start_freq = self.freq2reg(
            cfg.expt.start, gen_ch=self.chevron_ch)
        self.chevron_step_freq = self.freq2reg(
            cfg.expt.step, gen_ch=self.chevron_ch)
        self.safe_regwi(
            self.chevron_page,
            self.chevron_freq_sweep_register,
            self.chevron_start_freq,
        )

        self.chevron_gain = int(cfg.expt.get(
            "custom_scramble_gain", pulse[1][0]))
        self.chevron_phase = self.deg2reg(
            cfg.expt.get("custom_scramble_phase", 0.0),
            gen_ch=self.chevron_ch,
        )
        self.chevron_length = self.us2cycles(
            cfg.expt.custom_scramble_length,
            gen_ch=self.chevron_ch,
        )

        if self.chevron_ch == self.flux_low_ch[0]:
            self.chevron_waveform = "pi_m1si_low"
        elif self.chevron_ch == self.flux_high_ch[0]:
            self.chevron_waveform = "pi_m1si_high"
        else:
            raise ValueError(
                f"unexpected M1-storage channel: {self.chevron_ch}")

        self.sync_all(200)

    def core_pulses(self):
        # Prep/reset pulses may use the same flux channel, so restore every
        # fixed pulse register immediately before the swept swap.
        self.set_pulse_registers(
            ch=self.chevron_ch,
            style="flat_top",
            freq=0,
            phase=self.chevron_phase,
            gain=self.chevron_gain,
            length=self.chevron_length,
            waveform=self.chevron_waveform,
        )
        self.mathi(
            self.chevron_page,
            self.chevron_freq_register,
            self.chevron_freq_sweep_register,
            "+",
            0,
        )
        self.pulse(ch=self.chevron_ch)
        self.sync_all(self.us2cycles(0.2))

    def update(self):
        self.mathi(
            self.chevron_page,
            self.chevron_freq_sweep_register,
            self.chevron_freq_sweep_register,
            "+",
            self.chevron_step_freq,
        )


class ManStorMultiparityChevronRExperiment(DarkBaseExperiment):
    """Software length sweep with a hardware frequency sweep on every line."""

    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        lengths = np.asarray(
            self.cfg.expt.custom_scramble_lengths, dtype=float)
        frequency_count = int(self.cfg.expt.expts)
        reps = int(self.cfg.expt.reps)
        rounds = int(self.cfg.expt.get("rounds", 1))
        if rounds != 1:
            raise ValueError(
                "ManStorMultiparityChevronRExperiment requires rounds=1 "
                "so every multiparity shot remains associated with its "
                "length point"
            )

        if not self.cfg.expt.get("perform_wigner", False):
            self.cfg.expt.perform_wigner = False

        read_num = 1
        if self.cfg.expt.get("active_reset", False):
            params = MMRAveragerProgram.get_active_reset_params(self.cfg)
            read_num += MMRAveragerProgram.active_reset_read_num(**params)
        if self.cfg.expt.get("multiparity_readout", False):
            read_num += 1
        self.cfg.read_num = read_num

        avgi_lines = []
        avgq_lines = []
        idata_lines = []
        qdata_lines = []
        frequency_points = None

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.custom_scramble_length = float(length)
            self.prog = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)

            xpts, avgi, avgq = self.prog.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
                debug=debug,
                readouts_per_experiment=read_num,
            )
            frequency_points = np.asarray(xpts, dtype=float)
            avgi_lines.append(np.asarray(avgi[0][-1], dtype=float))
            avgq_lines.append(np.asarray(avgq[0][-1], dtype=float))

            idata, qdata = self.prog.collect_shots()
            expected_values = rounds * frequency_count * reps * read_num
            if np.asarray(idata).size != expected_values:
                raise RuntimeError(
                    "unexpected RAverager I-buffer size: "
                    f"got {np.asarray(idata).size}, expected "
                    f"{expected_values}"
                )
            if np.asarray(qdata).size != expected_values:
                raise RuntimeError(
                    "unexpected RAverager Q-buffer size: "
                    f"got {np.asarray(qdata).size}, expected "
                    f"{expected_values}"
                )

            idata = np.asarray(idata).reshape(
                rounds, frequency_count, reps, read_num)
            qdata = np.asarray(qdata).reshape(
                rounds, frequency_count, reps, read_num)
            idata_lines.append(
                idata.transpose(1, 0, 2, 3).reshape(frequency_count, -1)
            )
            qdata_lines.append(
                qdata.transpose(1, 0, 2, 3).reshape(frequency_count, -1)
            )

        avgi = np.asarray(avgi_lines).T
        avgq = np.asarray(avgq_lines).T
        idata_lines = np.asarray(idata_lines)
        qdata_lines = np.asarray(qdata_lines)

        data = {
            "xpts": lengths,
            "ypts": frequency_points,
            "avgi": avgi,
            "avgq": avgq,
            "amps": np.abs(avgi + 1j * avgq),
            "phases": np.angle(avgi + 1j * avgq),
            "idata": [],
            "qdata": [],
        }

        # DarkBaseExperiment.analyze_multiparity expects frequency-major rows.
        for frequency_index in range(frequency_count):
            for length_index in range(len(lengths)):
                data["idata"].append(
                    idata_lines[length_index, frequency_index])
                data["qdata"].append(
                    qdata_lines[length_index, frequency_index])

        self.data = data
        return data




class DarkT1Program(DarkBaseProgram):

    def core_pulses(self):
        ecfg = self.cfg.expt

        wait_length = ecfg.get("wait_length", ecfg.get("wait", 0.0))

        if not ecfg.get("swap_man_dark", False):
            self.sync_all(self.us2cycles(wait_length))
            return

        swap_stors = list(ecfg.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)

        self.sync_all()

        # 1. M1 photon -> dark/normal mode
        self._prepare_dark_mode(phase_offsets)

        # 2. wait
        self.sync_all(self.us2cycles(wait_length))

        # 3. compensate dark-mode relative phase during wait
        self._apply_dark_wait_phase_tracking(phase_offsets, wait_length)

        # 4. dark/normal mode -> M1
        self._read_dark_mode(phase_offsets)

        self.sync_all()

class DarkT1Experiment(QsimBaseExperiment):
    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)

        T1 = data['fit_avgi'][3]  # decay rate
        T1_err = np.sqrt(data['fit_err_avgi'][3][3])
        kappa = 1/T1/2/ np.pi  # kappa = 1/T1/2/pi in unit of freq
        kappa_err = T1_err/T1**2 # kappa_err = T1_err/T1**2 * kappa

        data['T1'] = T1
        data['T1_err'] = T1_err
        data['kappa_in_freq'] = kappa
        data['kappa_err_in_freq'] = kappa_err


        return data
    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        T1 = data['T1']
        T1_err = data['T1_err']
        kappa = data['kappa_in_freq']
        kappa_err = data['kappa_err_in_freq']

        text = f"$T_1$ = {T1:.3f} $\pm$ {T1_err:.3f} us\n"
        text += f"$\kappa$ = {kappa*1e3:.3f} $\pm$ {kappa_err*1e3:.3f}KHz *2$\pi$\n"


        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'

            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')

        # add the text box with T1 and kappa values
        plt.gcf().text(0.15, 0.8, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

        plt.show()


class SidebandScrambleDarkProgramNewNew(SidebandScrambleProgram, DarkBaseProgram):
    # MRO: this -> SidebandScrambleProgram -> DarkBaseProgram -> QsimBaseProgram
    # Dark load, scramble, and readout share one mutable phase_offsets list.

    def _prepare_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        """
        Dispatch dark-mode load without modifying existing dark helpers.
        """
        if self.cfg.expt.get("swap_man_large_dark", False):
            self._prepare_large_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )
        else:
            self._prepare_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )

    def _read_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        """
        Dispatch dark-mode readout without modifying existing dark helpers.
        """
        if self.cfg.expt.get("swap_man_large_dark", False):
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with four supports")
            self._read_large_dark(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )
        else:
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with two supports")
            self._read_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )

    def core_pulses(self):
        swap_stors = list(self.cfg.expt.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)
        disorder_phase_offsets = [0.0] * len(swap_stors)

        # 1. Optional load:
        # M1/man excitation -> selected dark/normal mode.
        #
        # This mutates phase_offsets.  Those offsets must be the initial
        # frame for the following scramble.
        if self.cfg.expt.get("load_man_dark", False):
            self._prepare_selected_dark_mode(
                phase_offsets,
                disorder_phase_offsets=disorder_phase_offsets,
            )

        # 2. Scramble with the same live phase tracker.
        #
        # Do NOT call super().core_pulses() here.  That method creates a
        # fresh local swap_stor_phases = [0, 0, ...] and physically emits
        # the scramble with the wrong initial frame after dark load.
        self._play_scramble_with_phase_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
        )

        if not self.cfg.expt.get("swap_man_dark", False):
            return

        # 3. Readout with the final tracker.
        #
        # Do NOT call _accumulate_scramble_phases() here.  The actual scramble
        # above already updated phase_offsets pulse-by-pulse.  Calling
        # _accumulate_scramble_phases() again would double-count.
        self._read_selected_dark_mode(
            phase_offsets,
            disorder_phase_offsets=disorder_phase_offsets,
        )


class BroadbandGeValidationProgram(QsimBaseProgram):
    """Measure the configured broadband ge pi on |g,n> and |e,n>.

    ``validation_case`` selects one of six direct-readout experiments:

      0: |g,n> reference, no broadband pulse
      1: |e,n> reference, no broadband pulse
      2: |g,n> followed by pi_ge_broadband
      3: |e,n> followed by pi_ge_broadband
      4: |g,n> followed by repeated B(0) B(180) inverse pairs
      5: |e,n> followed by repeated B(0) B(180) inverse pairs

    Sweeping photon number and these six cases gives a separate IQ axis for
    every n, so photon-number-dependent readout shifts are not mistaken for a
    broadband-pulse error.
    """

    def initialize(self):
        # No storage-swap pulse is played here.  The runner still supplies the
        # storage/Floquet dataset handles required by the generic pulse creator.
        self.MM_base_initialize()
        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)
        photon_number = int(cfg.expt.validation_photon_number)
        validation_case = int(cfg.expt.validation_case)

        if photon_number not in (0, 1, 2, 3):
            raise ValueError(
                "validation_photon_number must be 0, 1, 2, or 3; "
                f"got {photon_number}"
            )
        if validation_case not in (0, 1, 2, 3, 4, 5):
            raise ValueError(
                "validation_case must be 0 (g ref), 1 (e ref), "
                "2 (g->e), 3 (e->g), 4 (g inverse pairs), or "
                "5 (e inverse pairs)"
            )

        self.reset_and_sync()
        if cfg.expt.get('active_reset', False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = cfg.expt.get('pre_relax_delay', 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        pulse_seq = self.prep_man_photon(photon_number)
        if validation_case in (1, 3, 5):
            pulse_seq.append([
                'multiphoton',
                f'g{photon_number}-e{photon_number}',
                'pi',
                0.0,
            ])
        if validation_case in (2, 3):
            pulse_seq.append(['qubit', 'ge_broadband', 'pi', 0.0])
        elif validation_case in (4, 5):
            inverse_pairs = int(
                cfg.expt.get('validation_inverse_pairs', 4))
            if inverse_pairs < 1:
                raise ValueError(
                    "validation_inverse_pairs must be at least 1")
            for _ in range(inverse_pairs):
                pulse_seq.append(
                    ['qubit', 'ge_broadband', 'pi', 0.0])
                pulse_seq.append(
                    ['qubit', 'ge_broadband', 'pi', 180.0])

        if pulse_seq:
            pulse = self.get_prepulse_creator(pulse_seq)
            self.sync_all()
            self.custom_pulse(
                cfg, pulse.pulse, prefix='ge_broadband_validation_')
            self.sync_all()

        self.measure_wrapper()


class NPhotonHamiltonianSpectroscopyProgram(
        SidebandScrambleDarkProgramNewNew):
    """Measure the complex diagonal return for one occupation string.

    ``spectroscopy_prep_phase`` is theta on the first qubit half-pi pulse, and
    ``spectroscopy_analyzer_phase`` is phi on the final qubit half-pi pulse.
    The measured return uses theta = 0, 180 degrees and phi = 0, 90 degrees.
    With QICK's raw DDS-phase convention, ``Q_phi = Re[A exp(+i phi)]``;
    therefore the complex return is ``A = Q_0 - i Q_90``.
    A complete fixed-N basis is summed in the notebook.

    ``decoder_phase_matrix[row, column]`` tracks decoder control axes. Row 0
    is the common f_n-g_(n+1) axis. The remaining rows are the M1-storage
    decoder axes in ``swap_stors`` order. Columns are the physical Floquet
    pulses, also in ``swap_stors`` order.

    Every played Floquet pulse advances all measured decoder-axis slopes.
    During decoding the accumulated M1 slope is subtracted from every inverse
    f_n-g_(n+1) pulse.  The storage rows are defined as
    ``mode_path_storage - mode_path_M1`` and are added to the inverse
    M1-storage pulses because those pulse phases enter the recovered return
    amplitude with the opposite sign.

    ``spectroscopy_phase_correction_mode='decoder'`` keeps this original
    decoder-pulse correction.  ``'final_analyzer'`` skips the decoder matrix
    and subtracts
    ``floquet_cycle * final_analyzer_phase_per_cycle_deg`` from the final
    qubit half-pi instead.
    """

    @staticmethod
    def _get_encoder_pulses(occupations, swap_stors):
        """
        Returning encoding pulse sequence as per `prepulse_creator2`.
        So each element in the list follows
            - output = [pulse1, pulse2, ...]
            - pulse1 = ['transition_name', 'which_transition', 'pi or hpi', relative_phase]
        The input of occupation should be n-string in the order of [n_M1] + [n_S for S in swap_stors].
        The shelving is done except for the final step, which is when last_mode and last_photon is true
        """

        occupied_modes = [(stor, occupation) for stor, occupation in zip(swap_stors, occupations[1:]) if occupation > 0] 
        if occupations[0] > 0:
            # M1 is loaded last because storage loading passes through M1.
            occupied_modes.append((0, occupations[0]))

        encoder_pulses = []
        for mode_index, (mode, photon_number) in enumerate(occupied_modes):
            last_mode = mode_index == len(occupied_modes) - 1

            for n in range(photon_number):
                last_photon = n == photon_number - 1
                last_ladder_step = last_mode and last_photon

                encoder_pulses.append(["multiphoton", f"e{n}-f{n}", "pi", 0.0,])
                if not last_ladder_step:
                    encoder_pulses.append(["qubit", "ge_broadband", "pi", 0.0,]) 
                encoder_pulses.append(["multiphoton", f"f{n}-g{n + 1}", "pi", 0.0,])
                if mode > 0 and last_photon:
                    encoder_pulses.append(["storage", f"M1-S{mode}", "pi", 0.0,]) 
                if not last_ladder_step:
                    encoder_pulses.append(["qubit", "ge_broadband", "pi", 0.0,])

        return encoder_pulses

    @staticmethod
    def _get_inverse_pulses(encoder_pulses): 
        """
        The input is list of the pulse that follows the syntax of `prepulse_creator2`.
            - input = [pulse1, pulse2, ...]
            - pulse1 = ['transition_name', 'which_transition', 'pi or hpi', relative_phase]
        The function gives the inverse of the input pulse list, by reversing the order and 
        adding relative phase of 180 deg to each pulse.
        Note that the phase is changed by modifying pulse[3] by the aforementioned syntax
            
        """
        
        inverse_pulses = []
        for pulse in reversed(encoder_pulses):
            pulse = list(pulse)
            pulse[3] = (float(pulse[3]) + 180.0) % 360.0
            inverse_pulses.append(pulse)
        return inverse_pulses

    def _add_wait_after_storage_pulses(self, pulses):
        """
        This is to add wait after manipulate storage swap pulses, and 
        this follows the reasoning of `man_stor_swap` method in the MMbase.
        The default is 0.2 us, but this can be modified by specifying `storage_pulse_wait_us` in the 
        experimental config when the program is executed.
        """
        
        wait_us = float(self.cfg.expt.get(
            "storage_pulse_wait_us", 0.2))
        if wait_us <= 0.0:
            return pulses

        pulses_with_wait = []
        for pulse in pulses:
            pulses_with_wait.append(pulse)
            if pulse[0] == "storage":
                pulses_with_wait.append(["wait", wait_us])

        return pulses_with_wait

    def initialize(self):
        """
        The primary purpose of overriding `initialize` method is for the initialization of phase calibration matrix.
        For now, `storage_phase_matrix` is not really being used, as the phase accumulating during floquet cycle matters.
        There are two mode of calibration:  `decoder` and `final_analyzer`
        1. `decoder`
        - uses `decoder_phase_matrix` for the calibration of ac stark shift on every sideband transition including fn_gn+1.
        2. `final_analyzer`
        - uses `expt.cfg.final_analyzer_phase_per_cycle_deg` at the final hpi to calibrate ac stark shift out collectively.
        """
        
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        if len(set(swap_stors)) != len(swap_stors):
            raise ValueError(f"swap_stors must be distinct; got {swap_stors}")
        if any(stor < 1 or stor > 7 for stor in swap_stors):
            raise ValueError(f"swap_stors entries must be in 1..7; got {swap_stors}")

        matrix_shape = (len(swap_stors), len(swap_stors))
        storage_phase_matrix = ecfg.get("storage_phase_matrix", None)
        if storage_phase_matrix is not None:
            storage_phase_matrix = np.asarray(
                storage_phase_matrix, dtype=float)
            if storage_phase_matrix.shape != matrix_shape:
                raise ValueError(f"storage_phase_matrix must have shape {matrix_shape}; got {storage_phase_matrix.shape}")
            ecfg.storage_phase_matrix = storage_phase_matrix

        decoder_matrix_shape = (len(swap_stors) + 1,len(swap_stors),)
        
        decoder_phase_matrix = ecfg.get("decoder_phase_matrix", None)
        self.decoder_phase_matrix_is_calibrated = decoder_phase_matrix is not None

        if decoder_phase_matrix is None:
            self.decoder_phase_matrix = np.zeros(decoder_matrix_shape)
        else:
            decoder_phase_matrix = np.asarray(decoder_phase_matrix,dtype=float,)
            if decoder_phase_matrix.shape != decoder_matrix_shape:
                raise ValueError(f"decoder_phase_matrix must have shape {decoder_matrix_shape}; got {decoder_phase_matrix.shape}")
            self.decoder_phase_matrix = decoder_phase_matrix.copy()

        if "spectroscopy_occupations" in ecfg:
            occupations = list(ecfg.spectroscopy_occupations)
        else:
            initial_mode = int(ecfg.spectroscopy_initial_mode)
            photon_number = int(ecfg.get("spectroscopy_photon_number", 1))
            if initial_mode not in [0] + swap_stors:
                raise ValueError(f"spectroscopy_initial_mode must be M1 (0) or one of swap_stors={swap_stors}; got {initial_mode}")
            occupations = [0] * (len(swap_stors) + 1)
            occupation_index = (0 if initial_mode == 0 else swap_stors.index(initial_mode) + 1)
            occupations[occupation_index] = photon_number

        if len(occupations) != len(swap_stors) + 1:
            raise ValueError(f"spectroscopy_occupations must be [n_M1] followed by the occupations of swap_stors={swap_stors}; got {occupations}")
        if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) for n in occupations):
            raise TypeError("spectroscopy_occupations entries must be non-negative integers")
        occupations = [int(n) for n in occupations]
        if any(n < 0 for n in occupations):
            raise ValueError("spectroscopy_occupations entries must be non-negative")

        photon_number = sum(occupations)
        if photon_number < 1:
            raise ValueError("spectroscopy_occupations must contain photons")
        if max(occupations) > 9:
            raise ValueError("The current multiphoton transition parser supports local occupations only through n=9")

        self.encoder_pulses = self._get_encoder_pulses(
            occupations, swap_stors)
        final_occupations = ecfg.get("spectroscopy_final_occupations", occupations)
        self.decoder_encoder_pulses = self._get_encoder_pulses(final_occupations, swap_stors)

        if any(pulse[1] == "ge_broadband" for pulse in self.encoder_pulses):
            pulse_key = "pi_ge_broadband"
            if pulse_key not in self.cfg.device.qubit.pulses:
                raise KeyError("This occupation-string encoder requires device.qubit.pulses.pi_ge_broadband")

            broadband_cfg = self.cfg.device.qubit.pulses[pulse_key]
            for field in ("frequency", "gain", "sigma", "length", "type"):
                if field not in broadband_cfg or np.asarray(
                        broadband_cfg[field]).size == 0:
                    raise RuntimeError(
                        "device.qubit.pulses.pi_ge_broadband."
                        f"{field} must contain one value"
                    )
            gain = np.asarray(broadband_cfg.get("gain", []), dtype=float).reshape(-1)
            
            if gain.size == 0 or gain[0] <= 0:
                raise RuntimeError("device.qubit.pulses.pi_ge_broadband.gain must be a configured nonzero value")

        max_local_occupation = max(occupations)
        multiphoton_pi = self.cfg.device.multiphoton.pi
        for transition in ("en-fn", "fn-gn+1"):
            if transition not in multiphoton_pi:
                raise KeyError(f"device.multiphoton.pi.{transition} is missing")
            
            for field in ("frequency", "gain", "length", "type", "sigma"):
                values = np.asarray(multiphoton_pi[transition].get(field, [])).reshape(-1)
                if len(values) < max_local_occupation:
                    raise RuntimeError(f"device.multiphoton.pi.{transition}.{field} needs at least {max_local_occupation} entries for spectroscopy_occupations={occupations}")

        storage_dataset = self.cfg.device.storage._ds_storage
        for stor, occupation in zip(swap_stors, occupations[1:]):
            if occupation == 0:
                continue
            stor_name = f"M1-S{stor}"
            if storage_dataset.get_gain(stor_name) <= 0:
                raise RuntimeError(f"{stor_name} needs a calibrated nonzero ds_storage gain")
            if storage_dataset.get_pi(stor_name) <= 0:
                raise RuntimeError(f"{stor_name} needs a calibrated ds_storage pi length")

        if ecfg.get("palindrome_scramble", False) \
                and int(ecfg.floquet_cycle) % 2:
            raise ValueError("palindrome spectroscopy uses an even number of nominal cycles; one symmetric sample is a forward/reverse pair")
        for flag in ("load_man_dark", "swap_man_dark", "swap_man_large_dark","perform_wigner", "parity_readout", "multiparity_readout"):
            if ecfg.get(flag, False):
                raise ValueError(f"{flag}=True is incompatible with vacuum-referenced Hamiltonian spectroscopy")

        prep_phase = float(ecfg.get("spectroscopy_prep_phase", 0.0))
        analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        phase_correction_mode = str(ecfg.get("spectroscopy_phase_correction_mode", "decoder"))
        if phase_correction_mode not in ("decoder", "final_analyzer"):
            raise ValueError("spectroscopy_phase_correction_mode must be 'decoder' or 'final_analyzer'")

        ecfg.spectroscopy_occupations = occupations
        ecfg.spectroscopy_prep_phase = prep_phase % 360.0
        ecfg.spectroscopy_analyzer_phase = analyzer_phase % 360.0
        ecfg.spectroscopy_phase_correction_mode = phase_correction_mode
        ecfg.final_analyzer_phase_per_cycle_deg = float(ecfg.get("final_analyzer_phase_per_cycle_deg", 0.0))
        ecfg.final_analyzer_phase_application_sign = -1.
        ecfg.spectroscopy_photon_number = photon_number
        ecfg.init_stor = 0
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        phase_correction_mode = ecfg.spectroscopy_phase_correction_mode
        update_decoder_phases = (ecfg.get("update_phases", True) and phase_correction_mode == "decoder")
        storage_phase_offsets = [0.0] * len(swap_stors)
        phase_offsets = [0.0] * len(swap_stors)
        disorder_phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        # (|g,0> + exp(i theta)|e,0>) / sqrt(2) ->
        # (|g,0> + exp(i theta)|g,n>) / sqrt(2)
        encoder_pulses = deepcopy(self.encoder_pulses)
        for pulse in encoder_pulses:
            if pulse[0] != "storage":
                continue

            stor = int(pulse[1].split("-S")[1])
            stor_index = swap_stors.index(stor)
            pulse[3] = self._mod360(pulse[3] + storage_phase_offsets[stor_index])
            self._advance_storage_phase_offsets(phase_offsets=storage_phase_offsets,swap_stors=swap_stors,pulsed_stor=stor)

        prepulse_cfg = [["qubit", "ge", "hpi", ecfg.spectroscopy_prep_phase],] + encoder_pulses
        prepulse_cfg = self._add_wait_after_storage_pulses(prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(cfg, prepulse.pulse, prefix="floquet_spec_pre_")
        self.sync_all()

        decoder_phase_deg = [0.0] * (len(swap_stors) + 1)
        if update_decoder_phases and not self.decoder_phase_matrix_is_calibrated:
            raise RuntimeError("decoder_phase_matrix is missing. Run the exact-path Floquet phase calibration before spectroscopy.")

        # U(t)|n>. Each physical Floquet pulse advances the measured decoder
        # phase slopes in decoder_phase_deg. The inverse decoder later uses
        # the opposite sign to cancel those slopes.
        self._play_scramble_with_phase_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
            decoder_phase_offsets=(decoder_phase_deg if update_decoder_phases else None),
        )

        # Decode |n> to |e,0>, then interfere it with |g,0>.
        postpulse_cfg = self._get_inverse_pulses(self.decoder_encoder_pulses)
        for pulse in postpulse_cfg:
            # Every f_n-g_(n+1) pulse transfers one M1 photon, so all n use
            # the same M1-frame correction. N ladder steps then give N times
            # that phase without an explicit photon-number multiplier.
            if pulse[0] == "multiphoton" and pulse[1].startswith("f") and "-g" in pulse[1]:
                pulse[3] = self._mod360(pulse[3] - decoder_phase_deg[0])

            elif pulse[0] == "storage":
                stor = int(pulse[1].split("-S")[1])
                stor_index = swap_stors.index(stor)
                pulse[3] = self._mod360(
                    pulse[3]
                    + storage_phase_offsets[stor_index]
                    + decoder_phase_deg[stor_index + 1]
                    + disorder_phase_offsets[stor_index]
                )
                self._advance_storage_phase_offsets(
                    phase_offsets=storage_phase_offsets,
                    swap_stors=swap_stors,
                    pulsed_stor=stor,
                )

        analyzer_phase = float(ecfg.spectroscopy_analyzer_phase)
        if phase_correction_mode == "final_analyzer":
            # Q_phi = Re[A exp(+i phi)], so a measured +Gamma phase is
            # removed by shifting the final analyzer by -Gamma.
            analyzer_phase -= (int(ecfg.floquet_cycle)* float(ecfg.final_analyzer_phase_per_cycle_deg))

        postpulse_cfg.append(["qubit", "ge", "hpi",self._mod360(analyzer_phase),])
        postpulse_cfg = self._add_wait_after_storage_pulses(postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="floquet_spec_post_")
        self.sync_all()
        self.measure_wrapper()



class EncodingOrthogonalityProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure coherent cross-return amplitudes between encoder paths.

    One job fixes ``spectroscopy_occupations`` (the encoded column). Its outer
    software sweep packs decoder occupation and analyzer phase into
    ``decoder_analyzer_row``; the inner sweep is the usual preparation phase
    ``0/180``. Only zero Floquet cycles are accepted, so this probes the access
    paths rather than Floquet time evolution.
    """
    def initialize(self):
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        decoder_occupations = [list(occupation) for occupation in ecfg.orthogonality_decoder_occupations]
        analyzer_phases = ecfg.orthogonality_analyzer_phases
        decoder_analyzer_row = int(ecfg.decoder_analyzer_row)
        decoder_index = decoder_analyzer_row // 2
        analyzer_phase_index = decoder_analyzer_row % 2
        decoder_occupation = decoder_occupations[decoder_index]
        ecfg.spectroscopy_analyzer_phase = float(
            analyzer_phases[analyzer_phase_index])
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = 0.
        ecfg.floquet_cycle = 0

        super().initialize()
        self.decoder_encoder_pulses = self._get_encoder_pulses(
            decoder_occupation, swap_stors)

    def _get_inverse_pulses(self, _):
        """
        Overrides ``_get_inverse_pulses`` in the parent ``NPhotonHamiltonianSpectroscopyProgram``
        for a given decoder_occupation

        """

        # Parent body asks for inverse(encoder); use the selected decoder here.
        return super()._get_inverse_pulses(self.decoder_encoder_pulses)


class EncodingPropagatorProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure one raw column of the short-time propagator."""

    def initialize(self):
        ecfg = self.cfg.expt
        cycle_decoder_analyzer = list(ecfg.cycle_decoder_analyzer)
        decoder_occupation = [
            int(n) for n in cycle_decoder_analyzer[1:-1]
        ]

        ecfg.floquet_cycle = int(cycle_decoder_analyzer[0])
        ecfg.spectroscopy_analyzer_phase = float(
            cycle_decoder_analyzer[-1])
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.final_analyzer_phase_per_cycle_deg = 0.

        super().initialize()
        self.decoder_encoder_pulses = self._get_encoder_pulses(
            decoder_occupation,
            [int(stor) for stor in ecfg.swap_stors],
        )

    def _get_inverse_pulses(self, _):
        # The parent body requests inverse(encoder); decode the selected row.
        return super()._get_inverse_pulses(self.decoder_encoder_pulses)


class EncodingStarkShiftCalibrationProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure the raw Floquet-pulse phase of one occupation basis state.

    ``spectroscopy_occupations`` selects the encoded occupation string and
    ``stor_B`` selects the physical M1-storage Floquet pulse.  An even number
    of pulses is played with alternating 0/180-degree phase, so the intended
    exchange closes while any repeatable diagonal phase remains measurable.

    No previously measured decoder or ds_storage phase correction is applied
    here.  The fixed encoder/decoder access phase is removed in the notebook
    by dividing the complex return by its zero-pulse value.  A nonzero
    ``final_analyzer_phase_per_pulse_deg`` is multiplied by ``n_pulse`` and
    subtracted only from the final qubit half-pi for an end-to-end sign check.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        n_pulse = int(ecfg.n_pulse)

        if stor_B not in swap_stors:
            raise ValueError(
                f"stor_B must be one of {swap_stors}; got {stor_B}"
            )
        if n_pulse < 0 or n_pulse % 2:
            raise ValueError(
                "n_pulse must be a non-negative even integer so the "
                "alternating Floquet train closes before decoding"
            )
        if "spectroscopy_occupations" not in ecfg:
            raise ValueError(
                "EncodingStarkShiftCalibrationProgram requires "
                "spectroscopy_occupations"
            )

        ecfg.spectroscopy_prep_phase = float(
            ecfg.get("spectroscopy_prep_phase", 0.0))
        ecfg.spectroscopy_analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        ecfg.final_analyzer_phase_per_pulse_deg = float(ecfg.get(
            "final_analyzer_phase_per_pulse_deg", 0.0
        ))
        ecfg.floquet_cycle = 0
        ecfg.palindrome_scramble = False
        ecfg.update_phases = False
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        prepulse_cfg = [[
            "qubit", "ge", "hpi",
            float(ecfg.spectroscopy_prep_phase),
        ]] + deepcopy(self.encoder_pulses)
        prepulse_cfg = self._add_wait_after_storage_pulses(
            prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="encoding_stark_pre_")
        self.sync_all()

        self._play_m1s_frac_train(
            stor=stor_B,
            n_frac=int(ecfg.n_pulse),
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            logical_phase_deg=0.0,
            logical_phase_step_deg=180.0,
            update_phases=False,
            label="occupation-resolved encoding Stark calibration",
        )

        postpulse_cfg = self._get_inverse_pulses(
            self.encoder_pulses)
        analyzer_phase = (
            float(ecfg.spectroscopy_analyzer_phase)
            # The same Q_phi convention as spectroscopy is used here.
            - int(ecfg.n_pulse)
            * float(ecfg.final_analyzer_phase_per_pulse_deg)
        )

        postpulse_cfg.append([
            "qubit", "ge", "hpi",
            self._mod360(analyzer_phase),
        ])
        postpulse_cfg = self._add_wait_after_storage_pulses(
            postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="encoding_stark_post_")
        self.sync_all()
        self.measure_wrapper()


class EntireFloquetCyclePhaseCalibrationProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure one occupation's phase from the entire Floquet cycle.

    One closed pair is

        ordered Floquet cycle -> reverse-order inverse Floquet cycle.

    The inverse cycle uses the same tracked logical axes as spectroscopy and
    adds 180 degrees to every beam-splitter pulse.  It therefore cancels the
    intended exchange motion while repeatable diagonal phase can accumulate.
    ``n_cycle_pair`` pairs contain ``2 * n_cycle_pair`` physical entire
    cycles.  The notebook fits against that physical-cycle count, so its
    slope is directly in degrees per entire Floquet cycle.

    ``final_analyzer_phase_per_cycle_deg`` is used only for the end-to-end
    sign check.  It is multiplied by ``2 * n_cycle_pair`` and subtracted from
    the final qubit half-pi.

    ``n_physical_cycle`` also permits odd guide points.  An odd point plays
    all complete forward/inverse pairs followed by one forward cycle.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        if ecfg.get("phase_unwrap_mode", "pair") == "odd_guide":
            n_physical_cycle = int(ecfg.n_physical_cycle)
        else:
            n_physical_cycle = 2 * int(ecfg.n_cycle_pair)

        if n_physical_cycle < 0:
            raise ValueError("n_physical_cycle must be non-negative")
        if "spectroscopy_occupations" not in ecfg:
            raise ValueError(
                "EntireFloquetCyclePhaseCalibrationProgram requires "
                "spectroscopy_occupations"
            )
        if ecfg.get("floquet_hardware_loop", False):
            raise ValueError(
                "The exact multi-mode forward/inverse calibration currently "
                "uses the software-emitted pulse sequence; set "
                "floquet_hardware_loop=False"
            )

        detunings = ecfg.get("detunings", None)
        if detunings is not None and detunings is not False \
                and np.asarray(detunings).size > 0 \
                and not np.allclose(detunings, 0.0):
            raise ValueError(
                "Entire-cycle phase calibration uses zero detuning.  "
                "Disorder is part of the target Hamiltonian and must not be "
                "calibrated out."
            )

        ecfg.spectroscopy_prep_phase = float(
            ecfg.get("spectroscopy_prep_phase", 0.0))
        ecfg.spectroscopy_analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        ecfg.final_analyzer_phase_per_cycle_deg = float(ecfg.get(
            "final_analyzer_phase_per_cycle_deg", 0.0
        ))
        ecfg.zero_floquet_gain = bool(ecfg.get(
            "zero_floquet_gain", False
        ))
        ecfg.spectroscopy_phase_correction_mode = "final_analyzer"
        ecfg.n_physical_cycle = n_physical_cycle
        ecfg.floquet_cycle = 0
        ecfg.palindrome_scramble = False
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        n_physical_cycle = int(ecfg.n_physical_cycle)
        n_cycle_pair = n_physical_cycle // 2
        phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        prepulse_cfg = [[
            "qubit", "ge", "hpi",
            float(ecfg.spectroscopy_prep_phase),
        ]] + deepcopy(self.encoder_pulses)
        prepulse_cfg = self._add_wait_after_storage_pulses(
            prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="entire_cycle_phase_pre_")
        self.sync_all()

        self._play_closed_floquet_cycle_pairs(
            n_cycle_pair=n_cycle_pair,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            extra_forward=n_physical_cycle % 2,
        )

        postpulse_cfg = self._get_inverse_pulses(
            self.encoder_pulses)
        analyzer_phase = (
            float(ecfg.spectroscopy_analyzer_phase)
            - n_physical_cycle
            * float(ecfg.final_analyzer_phase_per_cycle_deg)
        )
        postpulse_cfg.append([
            "qubit", "ge", "hpi", self._mod360(analyzer_phase),
        ])
        postpulse_cfg = self._add_wait_after_storage_pulses(
            postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="entire_cycle_phase_post_")
        self.sync_all()
        self.measure_wrapper()


class FloquetDisplacementKerrProgram(DarkBaseProgram):
    """
    D(alpha) -> closed Floquet pairs -> D(alpha) -> vacuum readout using `slow_pi_ge`.
    Now displacement pulse entirely relies on displace_man, which receives complex
    alpha and converts it into magnitude and phase.
    Here, the prepulse and postpulse is turned off, as relying on prepulse and postpulse
    seemed unstraightforward.
    
    """

    def initialize(self):
        ecfg = self.cfg.expt
        ecfg.prepulse = False #Enforcing prepulse to be off
        ecfg.postpulse = False #Enforcing postpulse to be off
        ecfg.init_stor = 0
        ecfg.ro_stor = 0
        ecfg.slow_pi_ge_readout = True
        gain_to_alpha = self.cfg.device.manipulate.gain_to_alpha
        if isinstance(gain_to_alpha, (list, tuple, np.ndarray)):
            gain_to_alpha = gain_to_alpha[ecfg.man_mode_no - 1]
        ecfg.init_alpha = ecfg.displace_gain * gain_to_alpha
        super().initialize()
        self.floquet_cycle_us = self.calculate_floquet_cycle_us()
        ecfg.floquet_cycle_us = self.floquet_cycle_us

    def core_pulses(self):
        ecfg = self.cfg.expt
        swap_stors = list(ecfg.swap_stors)
        phase_offsets = [0.] * len(swap_stors)
        self.displace_man(alpha=ecfg.init_alpha, setup=False, play=True)
        self._play_closed_floquet_cycle_pairs(ecfg.n_cycle_pair, phase_offsets, swap_stors)
        time_us = 2 * ecfg.n_cycle_pair * self.floquet_cycle_us
        # Same gain/sign as the first displacement; only the known Ramsey frame advances.
        second_alpha = ecfg.init_alpha * np.exp(-2j * np.pi * ecfg.ramsey_freq * time_us)
        self.displace_man(alpha=second_alpha, setup=False, play=True)
        self.sync_all()


class SinglePhotonFloquetSpectroscopyProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Backward-compatible wrapper for the original N=1 program name."""

    def initialize(self):
        ecfg = self.cfg.expt
        photon_number = int(
            ecfg.get("spectroscopy_photon_number", 1))
        if photon_number != 1:
            raise ValueError(
                "SinglePhotonFloquetSpectroscopyProgram only supports N=1; "
                "use NPhotonHamiltonianSpectroscopyProgram for arbitrary N"
            )
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        initial_mode = int(ecfg.get("spectroscopy_initial_mode", 0))
        if initial_mode not in [0] + swap_stors:
            raise ValueError(
                f"spectroscopy_initial_mode={initial_mode} is not in "
                f"[0] + swap_stors={swap_stors}"
            )
        occupations = [0] * (len(swap_stors) + 1)
        occupation_index = (
            0 if initial_mode == 0 else swap_stors.index(initial_mode) + 1
        )
        occupations[occupation_index] = 1
        ecfg.spectroscopy_occupations = occupations
        ecfg.spectroscopy_photon_number = 1
        super().initialize()


class FloquetPhaseAccumulationProgram(
        NPhotonHamiltonianSpectroscopyProgram):
    """Measure the phase of one closed N=1 access path.

    ``spectroscopy_occupations`` selects M1 or one storage access path and
    ``stor_B`` selects one physical Floquet pulse column. ``n_pulse`` copies
    of that pulse are played with alternating 0/180-degree drive phase.
    ``spectroscopy_prep_phase`` is theta on the initial qubit half-pi and
    ``spectroscopy_analyzer_phase`` is phi on the final qubit half-pi, exactly
    as in active spectroscopy. Their phase cycle reconstructs the complex
    return.

    The M1 path gives the common f_n-g_(n+1) row. A storage path contains that
    common row plus its M1-storage row, so the notebook subtracts the measured
    M1 path before constructing ``decoder_phase_matrix``.
    """

    def initialize(self):
        ecfg = self.cfg.expt
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        if stor_B not in swap_stors:
            raise ValueError(
                f"stor_B must be one of {swap_stors}; got {stor_B}"
            )

        if "spectroscopy_occupations" not in ecfg:
            raise ValueError(
                "FloquetPhaseAccumulationProgram requires the exact "
                "spectroscopy_occupations row"
            )
        if int(ecfg.n_pulse) % 2:
            raise ValueError(
                "n_pulse must be even so the alternating Floquet train "
                "closes before decoding"
            )

        ecfg.spectroscopy_prep_phase = float(
            ecfg.get("spectroscopy_prep_phase", 0.0))
        ecfg.spectroscopy_analyzer_phase = float(
            ecfg.get("spectroscopy_analyzer_phase", 0.0))
        ecfg.floquet_cycle = 0
        ecfg.palindrome_scramble = False
        ecfg.ro_stor = 0
        super().initialize()

    def body(self):
        ecfg = self.cfg.expt
        cfg = AttrDict(self.cfg)
        swap_stors = [int(stor) for stor in ecfg.swap_stors]
        stor_B = int(ecfg.stor_B)
        phase_offsets = [0.0] * len(swap_stors)
        storage_phase_offsets = [0.0] * len(swap_stors)

        self.reset_and_sync()
        if ecfg.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = ecfg.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        encoder_pulses = deepcopy(self.encoder_pulses)
        for pulse in encoder_pulses:
            if pulse[0] != "storage":
                continue

            stor = int(pulse[1].split("-S")[1])
            stor_index = swap_stors.index(stor)
            pulse[3] = self._mod360(
                pulse[3] + storage_phase_offsets[stor_index]
            )
            self._advance_storage_phase_offsets(
                phase_offsets=storage_phase_offsets,
                swap_stors=swap_stors,
                pulsed_stor=stor,
            )

        prepulse_cfg = [
            [
                "qubit", "ge", "hpi",
                float(ecfg.spectroscopy_prep_phase),
            ],
        ] + encoder_pulses
        prepulse_cfg = self._add_wait_after_storage_pulses(
            prepulse_cfg)
        prepulse = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="floquet_phase_pre_")
        self.sync_all()

        # The notebook uses even pulse counts so every 0/180 pair closes the
        # intended beam-splitter transfer before its complex phase is read.
        self._play_m1s_frac_train(
            stor=stor_B,
            n_frac=int(ecfg.n_pulse),
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            logical_phase_deg=0.0,
            logical_phase_step_deg=180.0,
            update_phases=False,
            label="exact-path Floquet phase calibration",
        )

        postpulse_cfg = self._get_inverse_pulses(
            self.encoder_pulses)
        for pulse in postpulse_cfg:
            if pulse[0] != "storage":
                continue

            stor = int(pulse[1].split("-S")[1])
            stor_index = swap_stors.index(stor)
            pulse[3] = self._mod360(
                pulse[3] + storage_phase_offsets[stor_index]
            )
            self._advance_storage_phase_offsets(
                phase_offsets=storage_phase_offsets,
                swap_stors=swap_stors,
                pulsed_stor=stor,
            )

        postpulse_cfg.append([
            "qubit", "ge", "hpi",
            float(ecfg.spectroscopy_analyzer_phase),
        ])
        postpulse_cfg = self._add_wait_after_storage_pulses(
            postpulse_cfg)
        postpulse = self.get_prepulse_creator(postpulse_cfg)
        self.sync_all()
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="floquet_phase_post_")
        self.sync_all()
        self.measure_wrapper()


class CentralBosonLocalReturnProgram(SidebandScrambleDarkProgramNewNew):
    """Direct M1 local-return protocol for the central-boson Floquet model.

    The pulse body is intentionally separate from ``DarkBaseProgram.body``:
    it prepares one product Fock state, plays only the calibrated sequential
    Floquet scramble, and measures M1 in place.  In particular, it never calls
    a dark-mode load or analyzer.  For the primary fixed-N=3 experiment (and
    the N=1 self-Kerr-inactive control), M1 multiparity resolves n_M1=0..3.
    Thus ``p_mod{n_M1(0)}`` represents local return within the fixed N<=3
    model; upward leakage to n_M1>=4 would alias under modulo-4 readout.
    """

    def initialize(self):
        self.initial_occupations, self.central_return_swap_stors = \
            configure_central_return_metadata(self.cfg.expt)
        super().initialize()

    def _prepare_initial_product_fock(self):
        """Prepare ``[n_M1] + [n_S for S in swap_stors]`` without dark load.

        Each occupied storage is prepared by making its Fock occupation in M1
        and applying the existing calibrated full M1-S swap.  The central
        occupation is prepared last.  The decisive protocol defaults to
        collision-free occupations, for which every nonzero preparation is a
        calibrated one-photon transfer.  Multi-Fock product-state preparation
        is deliberately rejected until it shares the calibrated compensation
        path used by the legacy program.
        """
        occupations, swap_stors = validate_central_return_occupations(
            self.cfg.expt.initial_occupations,
            self.cfg.expt.swap_stors,
            expected_total_photons=self.cfg.expt.total_photons,
            require_collision_free=True,
        )

        prepulse_cfg = []
        for stor, occupation in zip(swap_stors, occupations[1:]):
            if occupation == 0:
                continue
            prepulse_cfg += self.prep_man_fock_state(
                self.cfg.expt.get("man_mode_no", 1),
                str(occupation),
                broadband=False,
            )
            prepulse_cfg.append([
                "storage", f"M1-S{stor}", "pi", 0,
            ])

        central_occupation = occupations[0]
        if central_occupation:
            prepulse_cfg += self.prep_man_fock_state(
                self.cfg.expt.get("man_mode_no", 1),
                str(central_occupation),
                broadband=False,
            )

        if not prepulse_cfg:
            return

        pulse_creator = self.get_prepulse_creator(prepulse_cfg)
        self.sync_all()
        self.custom_pulse(
            AttrDict(self.cfg),
            pulse_creator.pulse,
            prefix="central_return_pre_",
        )
        self.sync_all()

        if self.cfg.expt.get("debug", False):
            print(
                "[CentralReturn] prepared",
                dict(zip(
                    self.cfg.expt.initial_occupation_modes,
                    occupations,
                )),
            )

    def core_pulses(self):
        """Play the calibrated scramble with no dark-mode load/readout."""
        swap_stors = list(self.central_return_swap_stors)
        phase_offsets = [0.0] * len(swap_stors)
        disorder_phase_offsets = [0.0] * len(swap_stors)

        self._play_scramble_with_phase_offsets(
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def body(self):
        """Prepare -> Floquet scramble -> direct M1 multiparity -> measure."""
        self.reset_and_sync()

        if self.cfg.expt.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = self.cfg.expt.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        # This custom body is the only preparation path; it deliberately does
        # not call DarkBaseProgram.body and therefore cannot double-run its
        # legacy init_stor/init_man_fock_state preparation.
        self._prepare_initial_product_fock()
        self.core_pulses()

        # ro_stor is validated to be zero.  Population remains in M1 and is
        # measured directly; no storage-to-M1 or dark-mode analyzer is emitted.
        self.multi_parity_readout(
            man_idx=self.man_mode_idx + 1,
            fast=self.cfg.expt.get("parity_fast", False),
        )
        self.sync_all()
        self.measure_wrapper()


class CentralBosonLocalReturnExperiment(DarkBaseExperiment):
    """Experiment wrapper that stores and extracts the direct local return."""

    def _store_raw_central_return(self):
        """Flatten threshold-classified mod-4 probabilities for safe saving."""
        target = int(self.cfg.expt.return_target_nmod4)
        multiparity = self.analyze_multiparity()
        p_mod4_raw = np.column_stack([
            np.asarray(multiparity[f"p_mod{n}"], dtype=float)
            for n in range(4)
        ])

        # analyze_multiparity() caches a nested dictionary in self.data. Slab's
        # HDF5 writer expects numeric arrays, so never leave that cache in the
        # payload that CharacterizationRunner saves.
        self.data.pop("multiparity", None)
        self.data["p_mod4_raw"] = p_mod4_raw
        self.data["central_return_probability_raw"] = p_mod4_raw[:, target]
        return self.data

    def acquire(self, progress=False, debug=False):
        occupations, _ = configure_central_return_metadata(self.cfg.expt)
        if len(self.cfg.expt.swept_params) != 1:
            raise ValueError(
                "CentralBosonLocalReturnExperiment expects one sweep axis; "
                "run initial states and disorder realizations as separate "
                "jobs so their metadata stay unambiguous"
            )

        data = super().acquire(progress=progress, debug=debug)
        target = int(self.cfg.expt.return_target_nmod4)

        # Store the provenance in the data payload as well as cfg.expt, so it
        # survives both config-aware and data-only aggregation workflows.
        data["initial_occupations"] = np.asarray(occupations, dtype=int)
        data["initial_central_occupation"] = int(occupations[0])
        data["total_photons"] = int(sum(occupations))
        data["return_target_nmod4"] = target

        self.data = data
        return self._store_raw_central_return()

    def analyze(self, data=None, **kwargs):
        if data is not None:
            self.data = data
        return self._store_raw_central_return()


class SidebandScrambleDarkProgramNew(SidebandScrambleProgram, DarkBaseProgram):
    # MRO: this -> SidebandScrambleProgram -> DarkBaseProgram -> QsimBaseProgram
    # so super().core_pulses() plays the scrambling pulses, while the dark-mode
    # helpers (_read_dark_mode, _accumulate_scramble_phases, man_reset, ...) are
    # inherited from DarkBaseProgram.

    def core_pulses(self):
        swap_stors = list(self.cfg.expt.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)

        if self.cfg.expt.get("load_man_dark", False):
            self._prepare_large_dark_mode(phase_offsets)

        super().core_pulses()  # SidebandScrambleProgram.core_pulses(): plays scrambling

        if not self.cfg.expt.get("swap_man_dark", False):
            return

        # Replay the phase bookkeeping in the calibrated frame to match what
        # the (just-played) scrambling left behind. SidebandScrambleProgram
        # keeps its phase tracker local, so we reconstruct it here.
        self._accumulate_scramble_phases(phase_offsets, swap_stors)
        if self.cfg.expt.get("swap_man_dark", False) and not self.cfg.expt.get("swap_man_large_dark", False):
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with two supports")
            # Map the selected dark/normal mode back into M1.
            self._read_dark_mode(phase_offsets)
        elif self.cfg.expt.get("swap_man_large_dark", False):
            if self.cfg.expt.get("debug", False):
                print("reading out dark mode with four supports")
            self._read_large_dark(phase_offsets)


class ManStorScrambleProgram(SidebandScrambleProgram, DarkBaseProgram):
    # MRO: this -> SidebandScrambleProgram -> DarkBaseProgram -> QsimBaseProgram
    # so super().core_pulses() plays the scrambling pulses, while the dark-mode
    # helpers (_read_dark_mode, _accumulate_scramble_phases, man_reset, ...) are
    # inherited from DarkBaseProgram.

    def core_pulses(self):
        self.sync_all()
        swap_stor = self.cfg.expt.swap_stor
        _pulse_cfg = [ ['storage', f'M1-S{swap_stor}', 'pi', 0,] ] 
        _pulse_creator = self.get_prepulse_creator(_pulse_cfg)
        _pulse = _pulse_creator.pulse
        _pulse[2][0] = self.cfg.expt.length
        if self.cfg.expt.get("custom_scramble_gain", None) is not None:
            _pulse[1][0] = self.cfg.expt.custom_scramble_gain
        if self.cfg.expt.get("custom_scramble_freq", None) is not None:
            _pulse[0][0] = self.cfg.expt.custom_scramble_freq
        if self.cfg.expt.get("custom_scramble_phase", None) is not None:
            _pulse[3][0] = self.cfg.expt.custom_scramble_phase
        
        self.custom_pulse(self.cfg, _pulse, prefix = 'swap_pulse___manstorscram')
        self.sync_all(0.2)



class SidebandScrambleDarkProgramDebug(SidebandScrambleDarkProgramNewNew):
    # Debug variant for repeated load/readout checks.

    def _prepare_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        return super()._prepare_selected_dark_mode(
            phase_offsets=phase_offsets,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def _read_selected_dark_mode(
        self,
        phase_offsets,
        disorder_phase_offsets=None,
    ):
        return super()._read_selected_dark_mode(
            phase_offsets=phase_offsets,
            disorder_phase_offsets=disorder_phase_offsets,
        )

    def core_pulses(self):
        swap_stors = list(self.cfg.expt.swap_stors)
        phase_offsets = [0.0] * len(swap_stors)

        # 1. Optional load:
        # M1/man excitation -> selected dark/normal mode.
        #
        # This mutates phase_offsets.  Those offsets must be the initial
        # frame for the following scramble.
        for _ in range(self.cfg.expt.get("number_of_load_unload")):
            if self.cfg.expt.get("debug", False):
                print(
                    "doing debugging with "
                    f"{self.cfg.expt.get('number_of_load_unload')}"
                )
            self._prepare_selected_dark_mode(phase_offsets)
            self._read_selected_dark_mode(phase_offsets)
        # if self.cfg.expt.get("load_man_dark", False):
        #     self._prepare_selected_dark_mode(phase_offsets)

        # # 2. Scramble with the same live phase tracker.
        # #
        # # Do NOT call super().core_pulses() here.  That method creates a
        # # fresh local swap_stor_phases = [0, 0, ...] and physically emits
        # # the scramble with the wrong initial frame after dark load.
        # self._play_scramble_with_phase_offsets(
        #     phase_offsets=phase_offsets,
        #     swap_stors=swap_stors,
        # )

        # if not self.cfg.expt.get("swap_man_dark", False):
        #     return

        # # 3. Readout with the final tracker.
        # #
        # # Do NOT call _accumulate_scramble_phases() here.  The actual scramble
        # # above already updated phase_offsets pulse-by-pulse.  Calling
        # # _accumulate_scramble_phases() again would double-count.
        # self._read_selected_dark_mode(phase_offsets)



class KerrWaitProgramDark(DarkBaseProgram):
    def core_pulses(self):
        # print("Adding man-dump pulse")
        # self.man_reset(man_idx=1, dump_mode_idx=2, chi_dressed=True)

        self.sync_all(self.us2cycles(self.cfg.expt.wait_us_time))

'''
============================================================================
============================================================================
============================================================================
========   OLD PROGRAMS: TO BE DELETED AFTER SOME TIME======================
============================================================================
============================================================================
============================================================================
'''
class SidebandScrambleDarkProgram(SidebandScrambleProgram):
    
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
    
    
    def core_pulses(self):
        super().core_pulses() #already has sync_all at the last
        if self.cfg.expt.get("swap_man_dark", False):
            swap_stors = self.cfg.expt.swap_stors
            swap_stor_phases = [0.0] * len(swap_stors)

            if self.cfg.expt.update_phases:
                for _ in range(self.cfg.expt.floquet_cycle):
                    for i_stor, stor in enumerate(swap_stors):
                        for j_stor, stor_B in enumerate(swap_stors):
                            if stor_B != stor:
                                stor_B_name = f"M1-S{stor_B}"
                                stor_name = f"M1-S{stor}"
                                swap_stor_phases[j_stor] += self.swap_ds.get_phase_from(stor_B_name, stor_name)
                                swap_stor_phases[j_stor] = swap_stor_phases[j_stor] % 360
            
            stor_first, stor_last = self.cfg.expt.dark_swap_order
            list_index_start = swap_stors.index(stor_first)
            list_index_last = swap_stors.index(stor_last)

            n_first = self.m1s_pi_fracs[stor_first - 1] 
            n_last = self.m1s_pi_fracs[stor_last - 1] // 2
            first_stor_name = f"M1-S{stor_first}"
            last_stor_name = f"M1-S{stor_last}"

            if self.cfg.expt.get("second_rel_phase", 0) != 0:
                swap_stor_phases[list_index_last] += self.cfg.expt.second_rel_phase
                swap_stor_phases[list_index_last] = swap_stor_phases[list_index_last] % 360

            first_pulse_args = deepcopy(self.m1s_kwargs[stor_first - 1])
            second_pulse_args = deepcopy(self.m1s_kwargs[stor_last - 1])

            for _ in range(n_first): # full swap and phase update
                first_pulse_args['phase'] = self.deg2reg(swap_stor_phases[list_index_start], gen_ch=first_pulse_args['ch'])
                self.setup_and_pulse(**first_pulse_args)
                swap_stor_phases[list_index_last] += self.swap_ds.get_phase_from(last_stor_name, first_stor_name)
                swap_stor_phases[list_index_last] = swap_stor_phases[list_index_last] % 360
                self.sync_all(10)

            for _ in range(n_last):
                second_pulse_args['phase'] = self.deg2reg(swap_stor_phases[list_index_last], gen_ch=second_pulse_args['ch'])
                self.setup_and_pulse(**second_pulse_args)
                self.sync_all(10)
            
            self.sync_all()
            
            


class SidebandStarkAmplificationModifiedProgram_old(QsimBaseProgram):
    """
    Original phase-accumulation calibration sequence.

    Kept for comparing against the DarkBaseProgram/setup_and_pulse-matched
    version below.
    
    THIS IS STILL BEING USED; MUST BE KEPT IN THE REFACTORING.
    """

    def core_pulses(self):
        _scramble_sync_cycles = self.cfg.expt.get("scramble_sync_cycles", 10)
        i_storA = self.cfg.expt.stor_A - 1
        i_storB = self.cfg.expt.stor_B - 1
        m1s_kwarg_A = self.m1s_kwargs[i_storA]
        m1s_kwarg_B = self.m1s_kwargs[i_storB]

        n_pulse_B = self.cfg.expt.n_pulse
        pi_frac_A = self.m1s_pi_fracs[i_storA]
        pi_frac_B = self.m1s_pi_fracs[i_storB]

        ch_A = m1s_kwarg_A['ch']
        ch_B = m1s_kwarg_B['ch']
        channel_page_B = self.ch_page(ch_B)
        r_phase_B= self.sreg(ch_B, "phase")

        # Apply pi/2 pulse on stor_A
        self.set_pulse_registers(**m1s_kwarg_A)
        for i in range(pi_frac_A // 2):
            self.pulse(ch_A)
            if self.cfg.expt.get("include_10cycles_buffer", False) and self.cfg.expt.get("include_10cycles_buffer_in_pi_half", False):
                self.sync_all(_scramble_sync_cycles)
        self.sync_all()

        # # Apply a 2pi * n_pulse gate on stor_B
        # self.set_pulse_registers(**m1s_kwarg_B)
        # for i in range(n_pulse_B * 2 * pi_frac_B):
        #     self.pulse(ch_B)
        # advance_phase_A = self.deg2reg(n_pulse_B * pi_frac * self.cfg.expt.advance_phase)
        # self.sync_all()

        # Apply a (pi/12, -pi/12) * n_pulse gate on stor_B
        phase = 0
        self.set_pulse_registers(**m1s_kwarg_B)
        for i in range(n_pulse_B):
            for j in range(2):
                self.pulse(ch_B)
                # update the phase modulo 360
                phase += 180
                phase = phase % 360
                _phase_reg = self.deg2reg(phase, gen_ch=ch_B)
                self.safe_regwi(channel_page_B, r_phase_B, _phase_reg)
                if self.cfg.expt.get("include_10cycles_buffer", False):
                    self.sync_all(_scramble_sync_cycles)
        advance_phase_A = self.deg2reg(2 * n_pulse_B * self.cfg.expt.advance_phase)
        self.sync_all()

        # Apply -pi/2 pulse on stor_A with advanced phase
        m1s_kwarg_A_advanced = deepcopy(m1s_kwarg_A)
        m1s_kwarg_A_advanced['phase'] = advance_phase_A
        self.set_pulse_registers(**m1s_kwarg_A_advanced)
        for i in range(pi_frac_A // 2):
            self.pulse(m1s_kwarg_A_advanced['ch'])
            if self.cfg.expt.get("include_10cycles_buffer", False) and self.cfg.expt.get("include_10cycles_buffer_in_pi_half", False):
                self.sync_all(_scramble_sync_cycles)
        self.sync_all()


class StorageSwapPhaseAccumulationProgram(DarkBaseProgram):
    """Measure the phase matrix of the ds_storage swap pulses.

    ``stor_A`` is the affected Ramsey mode and ``stor_B`` is the pulsed mode.
    The first and last A pulses use exactly half of the calibrated full-swap
    plateau time.  Every B full/inverse-full pair closes its population action
    while retaining the phase accumulated by A.  ``advance_phase`` is the
    compensation in degrees per physical B full-swap pulse.
    """

    def _play_storage_pulse(self, stor, length_us, phase_deg):
        storage_ds = self.cfg.device.storage._ds_storage
        stor_name = f"M1-S{stor}"
        freq = storage_ds.get_freq(stor_name)

        if freq < 1800:
            ch = self.flux_low_ch[0]
            waveform = "pi_m1si_low"
        else:
            ch = self.flux_high_ch[0]
            waveform = "pi_m1si_high"

        self.setup_and_pulse(
            ch=ch,
            style="flat_top",
            freq=self.freq2reg(freq, gen_ch=ch),
            phase=self.deg2reg(self._mod360(phase_deg), gen_ch=ch),
            gain=storage_ds.get_gain(stor_name),
            length=self.us2cycles(length_us, gen_ch=ch),
            waveform=waveform,
        )
        self.sync_all(self.us2cycles(0.01))

    def core_pulses(self):
        stor_A = int(self.cfg.expt.stor_A)
        stor_B = int(self.cfg.expt.stor_B)
        n_pulse_B = int(self.cfg.expt.n_pulse)
        advance_phase = float(self.cfg.expt.advance_phase)

        storage_ds = self.cfg.device.storage._ds_storage
        pi_A = float(storage_ds.get_pi(f"M1-S{stor_A}"))
        pi_B = float(storage_ds.get_pi(f"M1-S{stor_B}"))

        # Ramsey preparation on A.  This intentionally uses pi_A / 2 rather
        # than the separately calibrated ds_storage h_pi entry.
        self._play_storage_pulse(
            stor=stor_A,
            length_us=pi_A / 2.0,
            phase_deg=0.0,
        )

        # A 0/180 pair is a full swap followed by its physical inverse.
        # Both pulses produce the same off-target Stark phase on A.
        for _ in range(n_pulse_B):
            self._play_storage_pulse(
                stor=stor_B,
                length_us=pi_B,
                phase_deg=0.0,
            )
            self._play_storage_pulse(
                stor=stor_B,
                length_us=pi_B,
                phase_deg=180.0,
            )

        self._play_storage_pulse(
            stor=stor_A,
            length_us=pi_A / 2.0,
            phase_deg=180.0 + 2.0 * n_pulse_B * advance_phase,
        )
        self.sync_all()

    def body(self):
        cfg = AttrDict(self.cfg)

        self.reset_and_sync()
        if cfg.expt.get("active_reset", False):
            params = MMAveragerProgram.get_active_reset_params(self.cfg)
            self.active_reset(**params)
            pre_relax_delay = cfg.expt.get("pre_relax_delay", 0)
            if pre_relax_delay > 0:
                self.sync_all(self.us2cycles(pre_relax_delay))

        # Load one photon into M1, close the storage Ramsey sequence, and
        # unload the photon.  The measured state is therefore g at n_pulse=0.
        prepulse = self.get_prepulse_creator([
            ["qubit", "ge", "pi", 0.0],
            ["qubit", "ef", "pi", 0.0],
            ["man", "M1", "pi", 0.0],
        ])
        self.custom_pulse(
            cfg, prepulse.pulse, prefix="storage_phase_pre_")
        self.sync_all()

        self.core_pulses()

        postpulse = self.get_prepulse_creator([
            ["man", "M1", "pi", 180.0],
            ["qubit", "ef", "pi", 180.0],
            ["qubit", "ge", "pi", 180.0],
        ])
        self.custom_pulse(
            cfg, postpulse.pulse, prefix="storage_phase_post_")
        self.sync_all()
        self.measure_wrapper()


class SidebandStarkAmplificationModifiedProgram(DarkBaseProgram):
    """
    Measure how a Floquet pulse on B shifts the later ds_storage swap on A.

    The Ramsey preparation/readout pulses on A use exactly half of the
    calibrated ds_storage full-swap length. Between them, the program applies
    (+Floquet B, -Floquet B) pairs. The final A half-swap phase is advanced by
    ``2 * n_pulse * advance_phase``, so the fitted ``advance_phase`` is the
    compensation per physical Floquet B pulse.
    """

    def _play_storage_half_swap(self, stor, phase_deg):
        storage_ds = self.cfg.device.storage._ds_storage
        stor_name = f"M1-S{stor}"
        freq = storage_ds.get_freq(stor_name)

        if freq < 1800:
            ch = self.flux_low_ch[0]
            waveform = "pi_m1si_low"
        else:
            ch = self.flux_high_ch[0]
            waveform = "pi_m1si_high"

        self.setup_and_pulse(
            ch=ch,
            style="flat_top",
            freq=self.freq2reg(freq, gen_ch=ch),
            phase=self.deg2reg(self._mod360(phase_deg), gen_ch=ch),
            gain=storage_ds.get_gain(stor_name),
            length=self.us2cycles(
                storage_ds.get_pi(stor_name) / 2.0,
                gen_ch=ch,
            ),
            waveform=waveform,
        )
        self.sync_all(self.us2cycles(0.01))

    def core_pulses(self):
        stor_A = int(self.cfg.expt.stor_A)
        stor_B = int(self.cfg.expt.stor_B)
        swap_stors = [stor_A, stor_B]
        phase_offsets = [0.0, 0.0]

        n_pulse_B = int(self.cfg.expt.n_pulse)
        advance_phase = float(self.cfg.expt.advance_phase)

        self._play_storage_half_swap(stor=stor_A, phase_deg=0.0)

        # This is the same weak physical Floquet pulse and sync gap used by
        # spectroscopy. The 180-degree alternation closes population transfer.
        self._play_m1s_frac_train(
            stor=stor_B,
            n_frac=2 * n_pulse_B,
            phase_offsets=phase_offsets,
            swap_stors=swap_stors,
            logical_phase_deg=0.0,
            logical_phase_step_deg=180.0,
            update_phases=False,
            label="phase calibration: alternating B train",
        )

        self._play_storage_half_swap(
            stor=stor_A,
            phase_deg=2.0 * n_pulse_B * advance_phase,
        )
        self.sync_all()


class SidebandStarkAmplificationModifiedProgram_newold(DarkBaseProgram):
    """
    1. Apply pi/2 swap pulse made of floquet pulses on stor_A
    2. Apply another floquet 2pi pulse on stor_B to calibrate the matrix element for. Do this xN times for error amplification
    3. Apply a -pi/2 swap pulse of floquet pulses on stor_A, with advanced phase
    
    Parameters in cfg.expt (sweepable):
    stor_A
    stor_B
    n_pulse: Nx pulses on stor B 
    advance_phase: phase of the last pulse on stor_A
    """

    def core_pulses(self):
        i_storA = self.cfg.expt.stor_A - 1
        i_storB = self.cfg.expt.stor_B - 1
        m1s_kwarg_A = self.m1s_kwargs[i_storA]
        m1s_kwarg_B = self.m1s_kwargs[i_storB]

        n_pulse_B = self.cfg.expt.n_pulse
        pi_frac_A = self.m1s_pi_fracs[i_storA]

        ch_A = m1s_kwarg_A['ch']
        ch_B = m1s_kwarg_B['ch']

        # Apply pi/2 pulse on stor_A
        self.set_pulse_registers(**m1s_kwarg_A)
        for i in range(pi_frac_A // 2):
            self.pulse(ch_A)
        self.sync_all()

        # Apply a (pi/12, -pi/12) * n_pulse gate on stor_B
        m1s_kwarg_B = deepcopy(m1s_kwarg_B)
        for i in range(n_pulse_B):
            for phase in (0, 180):
                m1s_kwarg_B['phase'] = self.deg2reg(phase, gen_ch=ch_B)
                self.setup_and_pulse(**m1s_kwarg_B)
                self.sync_all(10)
        advance_phase_A = self.deg2reg(
            2 * n_pulse_B * self.cfg.expt.advance_phase,
            gen_ch=ch_A,
        )
        
        # Apply -pi/2 pulse on stor_A with advanced phase
        m1s_kwarg_A_advanced = deepcopy(m1s_kwarg_A)
        m1s_kwarg_A_advanced['phase'] = advance_phase_A
        self.set_pulse_registers(**m1s_kwarg_A_advanced)
        for i in range(pi_frac_A // 2):
            self.pulse(ch_A)


class BatchRunner(CharacterizationRunner):
    """CharacterizationRunner with bounded parallel queue submission."""

    @staticmethod
    def _plain(obj):
        """
        Convert config values only at the queue's JSON boundary.
        Non JSON compatible objects are converted into compatible ones.
        """
        if isinstance(obj, dict):
            return {key: BatchRunner._plain(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BatchRunner._plain(value) for value in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def execute(self,
                  configs, 
                  batch_size=10, 
                  postprocess=True, 
                  priority=0,
                  poll_interval=2., 
                  timeout=None, 
                  log=None,
                  show=None):
        """Submit at most batch_size jobs, then collect them in config order."""
        if self.job_client is None:
            raise ValueError("job_client is required")
        if (isinstance(batch_size, (bool, np.bool_))
                or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")

        configs = list(configs) #list of config dictionary that is overrided in the submitted job
        if not configs:
            raise ValueError("configs cannot be empty")
        expts = []
        self.last_job_ids = []
        program_module = None
        program_class = None
        if self.program is not None:
            program_module = self.program.__module__
            program_class = self.program.__name__

        for start in range(0, len(configs), batch_size):
            pending = []
            batch_configs = [self.preprocessor(self.station, self.default_expt_cfg, **overrides) for overrides in configs[start:start + batch_size]]
            station_config = self._serialize_station_config()
            print(f"batch {start // batch_size + 1}: {len(batch_configs)} jobs")
            try:
                for cfg in batch_configs:
                    job_id = self.job_client.submit_job(
                        experiment_class=self.ExptClass.__name__,
                        experiment_module=self.ExptClass.__module__,
                        expt_config=self._plain(dict(cfg)), 
                        station_config=station_config,
                        user=self.station.user, 
                        priority=priority,
                        program_class=program_class, 
                        program_module=program_module,
                    )
                    pending.append(job_id)
                    self.last_job_ids.append(job_id)

                for job_id in list(pending):
                    result = self.job_client.wait_for_completion(
                        job_id, poll_interval=poll_interval, timeout=timeout, verbose=False)
                    pending.pop(0)
                    self.last_job_result = result
                    if not result.is_successful():
                        raise RuntimeError(
                            f"Job {job_id} {result.status}: {result.error_message or 'No details'}")
                    expt = result.load_expt()
                    if postprocess:
                        self.postprocessor(self.station, expt)
                    self._render_log_show(expt, show=show, log=log, display_kwargs=None)
                    expts.append(expt)
            except BaseException: #BaseException is inherited by Exception, KeyboardInterrupt, SystemExit, GeneratorExit
                # Below is to cancel the running job when there is KeyboardInterrupt
                for job_id in pending:
                    try:
                        self.job_client.cancel_job(job_id)
                    except Exception:
                        pass
                raise
        if self.program is not None:
            batch_expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=f"{self.ExptClass.__name__}_batch",
                config_file=self.station.hardware_config_file,
                program=self.program,
            )
        else:
            batch_expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=f"{self.ExptClass.__name__}_batch",
                config_file=self.station.hardware_config_file,
            )
        batch_expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))
        batch_expt.cfg.expt = deepcopy(self.default_expt_cfg)
        batch_expt.data = AttrDict()
        batch_expt.batch_expts = expts
        batch_expt.batch_job_ids = list(self.last_job_ids)
        batch_expt._analysis_station = self.station
        return batch_expt


class EncodingHamiltonianSpectroscopyExperiment(DarkBaseExperiment):
    """Per-job and aggregate analysis for encoding-calibrated spectroscopy."""

    @classmethod
    def _from_expts(cls, expts, job_ids=None, station=None):
        """
        Reconstruct 'mother' experiment instance by collecting all the experiments
        to be analyzed. The `sister` experiment should have identical following params;
        otherwise, `_saved_parameters` will return error
            - swap_stors
            - detunings
            - physical_kerr_MHz
            - scramble_sync_cycles
            - floquet_gauss_sigma
            - floquet_waveform
        The current version also assumes reconstructing the data from pickle; 
        if this is the case,
            - floquet_cycle_us
            - couplings_MHz
        which is to ensure that the same floquet pulse sequence is used.
        Otherwise, the station should be input as well to calculate physical params
        """
        
        expts = list(flatten_exp_lists(expts))
        if not expts:
            raise ValueError("experiment jobs cannot be empty")
        aggregate = cls.__new__(cls)
        aggregate.cfg = expts[0].cfg
        aggregate.data = AttrDict()
        aggregate.batch_expts = expts
        aggregate.batch_job_ids = list(job_ids or [])
        aggregate._analysis_station = station
        return aggregate

    @classmethod
    def from_job_files(cls, job_files, station=None):
        """
        Load child experiment pickle/H5 files into one analysis object.
        """
        import pickle
        from pathlib import Path

        if isinstance(job_files, (str, Path)):
            job_files = [job_files]
        expts = []
        for job_file in flatten_exp_lists(job_files):
            if not isinstance(job_file, (str, Path)):
                expts.append(job_file)
                continue
            path = Path(job_file)
            if path.suffix.lower() in (".h5", ".hdf5"):
                expts.append(cls.from_h5file(str(path)))
            else:
                with path.open("rb") as handle:
                    expts.append(pickle.load(handle))
        return cls._from_expts(expts, station=station)

    @classmethod
    def from_job_ids(cls, job_ids, client=None, station=None):
        """
        Load completed queue jobs without rebuilding a BatchRunner.
        If station is properly specified (along with project name and directory (which is hardcoded)),
        a list of job_ids is okay. Otherwise, a complete directory is necessary.
        """
        if isinstance(job_ids, (str, int, np.integer)):
            job_ids = [job_ids]
        job_ids = [str(job_id) for job_id in flatten_exp_lists(job_ids)]
        if not job_ids:
            raise ValueError("job_ids cannot be empty")
        if client is None:
            if station is None:
                raise ValueError("client or station is required to resolve job IDs")
            job_files = [station.expt_objs_path / f"{job_id}_expt.pkl" for job_id in job_ids]
            aggregate = cls.from_job_files(job_files, station=station)
            aggregate.batch_job_ids = job_ids
            return aggregate
        expts = []
        for job_id in job_ids:
            result = client.get_status(job_id)
            if not result.is_successful():
                raise RuntimeError(f"Job {job_id} {result.status}: {result.error_message or 'No details'}")
            expts.append(result.load_expt())
        return cls._from_expts(expts, job_ids=job_ids, station=station)

    @staticmethod
    def _first_scalar(value):
        """
        This is to avoid a conflict due to different data type of 
        self Kerr. Sometimes it is stored as a list, sometimes as a float...
        """
        
        values = np.asarray(value).reshape(-1)
        if len(values) == 0:
            raise ValueError("saved scalar config is empty")
        return float(values[0])

    @staticmethod
    def _saved_detunings(ecfg, mode_count):
        """
        This is to avoid a conflict due to different data type of 
        detuning. Sometimes it is stored as a list, sometimes as an np array...
        """
        
        detunings = ecfg.get("detunings", None)
        if detunings is None or detunings is False or np.asarray(detunings).size == 0:
            detunings = [0.] * mode_count
        return np.asarray(detunings, dtype=float)

    @classmethod
    def _saved_parameters(cls, expts, station=None):
        """
        Check whether all the sister expts have the same params,
        and return the params as an AttrDict.
        If the class is called for post processing using hdf5 files,
        one should specify station with proper config as well.
        
        Returning params are:
            - `swap_stors`
            - `detunings`
            - `mode_labels`
            - `hardware_parameters`
                - `floquet_cycles_us`
                - `couplings_MHz`
                - `physical_kerr_MHz`
        """
        
        first_cfg = expts[0].cfg
        first_expt_cfg = first_cfg.expt
        swap_stors = [int(stor) for stor in first_expt_cfg.swap_stors]
        detunings = cls._saved_detunings(first_expt_cfg, len(swap_stors))
        physical_kerr_MHz = -abs(cls._first_scalar(first_cfg.device.manipulate.kerr))
        if len(detunings) != len(swap_stors) or not np.all(np.isfinite(detunings)):
            raise ValueError("saved detunings do not match swap_stors")

        program_hardware = []
        sync_cycles = int(first_expt_cfg.get("scramble_sync_cycles", 10))
        floquet_gauss_sigma = first_expt_cfg.get("floquet_gauss_sigma", None)
        floquet_waveform = first_expt_cfg.get("floquet_waveform", None)
        for expt in expts:
            cfg = expt.cfg
            ecfg = cfg.expt
            if [int(stor) for stor in ecfg.swap_stors] != swap_stors:
                raise ValueError("saved jobs use different swap_stors")
            if len(ecfg.spectroscopy_occupations) != len(swap_stors) + 1:
                raise ValueError("saved spectroscopy_occupations do not match swap_stors")
            saved_detunings = cls._saved_detunings(ecfg, len(swap_stors))
            if saved_detunings.shape != detunings.shape or not np.allclose(saved_detunings, detunings):
                raise ValueError("saved jobs use different detunings")
            saved_kerr_MHz = -abs(cls._first_scalar(cfg.device.manipulate.kerr))
            if not np.isclose(saved_kerr_MHz, physical_kerr_MHz):
                raise ValueError("saved jobs use different M1 self-Kerr")
            if int(ecfg.get("scramble_sync_cycles", 10)) != sync_cycles or ecfg.get("floquet_gauss_sigma", None) != floquet_gauss_sigma or ecfg.get("floquet_waveform", None) != floquet_waveform:
                raise ValueError("saved jobs use different Floquet timing configs")
            prog = getattr(expt, "prog", None)
            if prog is not None and hasattr(prog, "calculate_floquet_cycle_us") and hasattr(prog, "m1s_pi_fracs"):
                floquet_cycle_us = float(prog.calculate_floquet_cycle_us())
                pi_fracs = np.asarray([prog.m1s_pi_fracs[stor - 1] for stor in swap_stors], dtype=float)
                couplings_MHz = 1. / (4. * pi_fracs * floquet_cycle_us)
                program_hardware.append((floquet_cycle_us, couplings_MHz))

        if program_hardware:
            floquet_cycle_us, couplings_MHz = program_hardware[0]
            for saved_cycle_us, saved_couplings_MHz in program_hardware[1:]:
                if not np.isclose(saved_cycle_us, floquet_cycle_us) or not np.allclose(saved_couplings_MHz, couplings_MHz):
                    raise ValueError("saved jobs use different Floquet hardware parameters")
            hardware_source = "saved program"
        elif station is not None:
            hardware = cls.hardware_parameters(station, 
                                               swap_stors, 
                                               sync_cycles, 
                                               floquet_gauss_sigma, 
                                               floquet_waveform)
            floquet_cycle_us, couplings_MHz = hardware.floquet_cycle_us, hardware.couplings_MHz
            hardware_source = "current station (H5 fallback)"
        else:
            raise RuntimeError("Floquet cycle time and couplings need the job pickle or station when loading H5 files")
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0. or not np.all(np.isfinite(couplings_MHz)) or np.min(couplings_MHz) <= 0.:
            raise ValueError("saved Floquet hardware parameters must be finite and positive")
        hardware = AttrDict(dict(floquet_cycle_us=float(floquet_cycle_us), couplings_MHz=np.asarray(couplings_MHz), physical_kerr_MHz=physical_kerr_MHz, source=hardware_source))
        return AttrDict(dict(swap_stors=swap_stors, 
                             detunings=detunings, 
                             mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors], 
                             hardware=hardware))

    @staticmethod
    def _cycle_branches(occupations, cycle_branches=0):
        """
        Specfy a branch for an accumulated phase for a encoding pulse per a floquet cycle
        default is 0, and possible inputs are
            - int: fix branch to int (ideally 0 or 1) for all encoding pulses
            - list: select branch per occupation; should match the length of occupation
            - dictionary: select branch for a specified occupation.
            
        Example:
        1. cycle_branches = 1
        2. occupations = [
                (2, 0, 0),
                (1, 1, 0),
                (1, 0, 1),
            ]
            cycle_branches = [0, 1, 0]
        3. cycle_branches = {
                (2, 0, 0): 0,
                (1, 1, 0): 1,
                (1, 0, 1): 0,
            }
        
        Recommendation is 3
        """
        
        if isinstance(cycle_branches, dict):
            branches = np.asarray([cycle_branches.get(tuple(occupation), 0) for occupation in occupations], dtype=float)
        elif np.isscalar(cycle_branches):
            branches = np.full(len(occupations), cycle_branches, dtype=float)
        else:
            branches = np.asarray(cycle_branches, dtype=float)
        if branches.shape != (len(occupations),) or not np.all(np.isfinite(branches)) or not np.allclose(branches, np.round(branches)):
            raise ValueError("cycle_branches must give one integer branch per occupation")
        return branches.astype(int)

    @classmethod
    def _calibration_data(cls, calibration, station=None):
        if calibration is None:
            return None
        from pathlib import Path
        if isinstance(calibration, (str, Path, list, tuple)):
            calibration = cls.from_job_files(calibration, station=station)
        if hasattr(calibration, "data"):
            if "phase_mod180" not in calibration.data:
                if hasattr(calibration, "batch_expts"):
                    cls.analyze(calibration, stage="calibration")
                else:
                    raise ValueError("calibration must contain the aggregated calibration jobs")
            return calibration.data
        return AttrDict(calibration)

    @classmethod
    def phase_correction_from_calibration(cls, 
                                          calibration, 
                                          cycle_branches=0, 
                                          second_branch=False,
                                          station=None):
        """
        Prepare the phase calibration list, which is returned as `phase_by_occupation` 
        by `build_phase_correction` method.
        """
        
        calibration = cls._calibration_data(calibration, station=station)
        if calibration is None:
            raise ValueError("calibration is required")
        if "hardware" not in calibration:
            raise ValueError("calibration hardware is unavailable; analyze the calibration experiment first")
        branches = cls._cycle_branches(calibration.occupations, cycle_branches)
        if second_branch:
            if np.any(branches):
                raise ValueError("use either cycle_branches or second_branch, not both")
            branches += 1
        return cls.build_phase_correction(calibration.occupations, calibration.phase_mod180, branches, calibration.hardware.physical_kerr_MHz, calibration.hardware.floquet_cycle_us)

    @classmethod
    def _saved_correction(cls, expts):
        """
        Reconstruct `phase_by_occupation` from the stored experiments.
        For now, `decoder` is the only supported
        
        """
        
        phase_by_occupation = {}
        modes = set()
        application_signs = set()
        missing_application_sign = False
        for expt in expts:
            ecfg = expt.cfg.expt
            occupation = tuple(ecfg.get("spectroscopy_final_occupations", ecfg.spectroscopy_occupations))
            phase = float(ecfg.get("final_analyzer_phase_per_cycle_deg", 0.))
            if occupation in phase_by_occupation and not np.isclose(phase, phase_by_occupation[occupation]):
                raise ValueError(f"{occupation} spectroscopy chunks used different analyzer corrections")
            phase_by_occupation[occupation] = phase
            modes.add(str(ecfg.get("spectroscopy_phase_correction_mode", 
                                   "decoder")))
            application_sign = ecfg.get("final_analyzer_phase_application_sign", None)
            if application_sign is None:
                missing_application_sign = True
            else:
                application_sign = float(application_sign)
                if application_sign not in (-1., 1.):
                    raise ValueError("saved analyzer phase application sign must be +1 or -1")
                application_signs.add(application_sign)
        if len(application_signs) > 1:
            raise ValueError("saved jobs use different analyzer phase application signs")
        if len(modes) != 1:
            raise ValueError("saved jobs use different spectroscopy phase-correction modes")
        nonzero_correction = any(not np.isclose(phase, 0.) for phase in phase_by_occupation.values())
        if nonzero_correction and missing_application_sign and application_signs:
            raise ValueError("saved jobs mix marked and unmarked analyzer phase conventions")
        application_sign = next(iter(application_signs)) if len(application_signs) == 1 and not missing_application_sign else None
        return AttrDict(dict(phase_by_occupation=phase_by_occupation, 
                             modes=modes, 
                             application_sign=application_sign))

    @classmethod
    def _postprocess_reconstruction(cls, 
                                    reconstruction, 
                                    saved_correction, 
                                    calibration, 
                                    hardware, 
                                    phase_frame, 
                                    manual_kerr_MHz, 
                                    cycle_branches, 
                                    legacy):
        """
        The postprocessing got a bit complicated as the previous experiment
        did not designate `application_sign`. The current convention is 
        `application_sign` = -1, whereas previously it was +1.
        The designation of +1 to application_sign is done by setting legacy = True.
        
        
        
        """
        
        if manual_kerr_MHz is not None and phase_frame == "as_acquired":
            phase_frame = "manual_kerr"
        if phase_frame == "zero_kerr":
            if manual_kerr_MHz is not None:
                raise ValueError("zero_kerr does not take manual_kerr_MHz")
            manual_kerr_MHz = 0.
        if phase_frame not in ("as_acquired", "uncorrected", "zero_kerr", "manual_kerr"):
            raise ValueError("phase_frame must be 'as_acquired', 'uncorrected', 'zero_kerr', or 'manual_kerr'")
        occupations = reconstruction.occupations
        final_occupations = reconstruction.get("final_occupations", occupations)
        branches = cls._cycle_branches(final_occupations, cycle_branches)
        A = reconstruction.A.copy()
        target_correction = None
        application_sign = saved_correction.application_sign
        legacy_migration = False

        if phase_frame == "as_acquired":
            if legacy is not None:
                raise ValueError("legacy is only used with uncorrected/zero_kerr/manual_kerr rephasing")
            for row, branch in enumerate(branches):
                A[row] *= np.exp(-1j * np.deg2rad(180. * branch) * reconstruction.cycles)
            physical_kerr_MHz = hardware.physical_kerr_MHz
        else:
            if saved_correction.modes != {"final_analyzer"}:
                raise ValueError("uncorrected/zero_kerr/manual_kerr rephasing requires spectroscopy_phase_correction_mode='final_analyzer'")
            if application_sign is None:
                nonzero_correction = any(not np.isclose(phase, 0.) for phase in saved_correction.phase_by_occupation.values())
                if nonzero_correction and legacy is None:
                    raise ValueError("saved jobs do not record the analyzer sign; use legacy=True for old +correction jobs or legacy=False for -correction jobs")
                application_sign = 1. if legacy else -1.
                legacy_migration = bool(legacy)
            elif legacy is not None and application_sign != (1. if legacy else -1.):
                raise ValueError("legacy disagrees with the saved analyzer phase application sign")
            legacy_migration = application_sign == 1.

            if phase_frame == "uncorrected":
                if manual_kerr_MHz is not None:
                    raise ValueError("uncorrected does not take manual_kerr_MHz")
                for row, occupation in enumerate(final_occupations):
                    saved_phase = saved_correction.phase_by_occupation[tuple(occupation)]
                    A[row] *= np.exp(-1j * np.deg2rad(application_sign * saved_phase + 180. * branches[row]) * reconstruction.cycles)
                physical_kerr_MHz = hardware.physical_kerr_MHz
            else:
                if manual_kerr_MHz is None or not np.isfinite(manual_kerr_MHz):
                    raise ValueError("phase_frame='manual_kerr' requires a finite signed manual_kerr_MHz")
                if calibration is None:
                    raise ValueError("zero_kerr/manual_kerr rephasing requires calibration")
                calibration_phase = {tuple(occupation): phase for occupation, phase in zip(calibration.occupations, calibration.phase_mod180)}
                missing = [occupation for occupation in final_occupations if tuple(occupation) not in calibration_phase]
                if missing:
                    raise ValueError(f"calibration is missing occupations {missing}")
                target_correction = cls.build_phase_correction(final_occupations, [calibration_phase[tuple(occupation)] for occupation in final_occupations], branches, float(manual_kerr_MHz), hardware.floquet_cycle_us)
                for row, occupation in enumerate(final_occupations):
                    saved_phase = saved_correction.phase_by_occupation[tuple(occupation)]
                    target_phase = target_correction.phase_by_occupation[tuple(occupation)]
                    A[row] *= np.exp(-1j * np.deg2rad(application_sign * saved_phase + target_phase) * reconstruction.cycles)
                physical_kerr_MHz = float(manual_kerr_MHz)
        normalized_A = np.asarray([row / row[0] if tuple(initial) == tuple(final) else row for row, initial, final in zip(A, occupations, final_occupations)])
        return AttrDict(dict(reconstruction=AttrDict(dict(occupations=occupations, 
                                                          final_occupations=final_occupations,
                                                          cycles=reconstruction.cycles,
                                                          A=A,
                                                          A_norm=normalized_A)), 
                             target_correction=target_correction, 
                             physical_kerr_MHz=physical_kerr_MHz, 
                             phase_frame=phase_frame, 
                             cycle_branches=branches, 
                             analyzer_phase_application_sign=application_sign, 
                             legacy_analyzer_migration=legacy_migration))

    @classmethod
    def subsample_spectroscopy_shots(cls,
                                     spectroscopy_expts,
                                     shots_per_point,
                                     seed=None):
        """
        Rebuild saved spectroscopy averages from fewer final-readout shots.

        Every saved sweep point contains interleaved readout lanes in ``idata``
        and ``qdata``. The final science measurement is the last lane of each
        repetition. This method draws ``shots_per_point`` paired I/Q samples
        from that lane without replacement and replaces only ``avgi``,
        ``avgq``, ``amps``, and ``phases`` in lightweight experiment copies.
        The original experiments and their raw-shot arrays are not modified.

        QICK's saved averaged values and ``collect_shots`` may use different ADC
        offsets or averaging rounds. For each point, the subset fluctuation
        relative to the complete raw-shot mean is therefore added to the saved
        average. Selecting every available shot then reproduces the saved
        ``avgi`` and ``avgq`` exactly without assuming a particular QICK offset.

        ``shots_per_point`` is the number of final-readout shots used for each
        saved ``(Floquet cycle, preparation phase)`` point. ``seed`` makes the
        random subset reproducible. Pre-selected acquisitions are rejected
        because their saved average is conditioned on herald lanes and cannot
        be reproduced by unconditioned final-lane sampling.

        Returns ``(subsampled_expts, metadata)``. The metadata records the
        requested shot count, seed, available-shot range, and raw-versus-saved
        averaging diagnostics for every child job.
        """

        if isinstance(shots_per_point, (bool, np.bool_)):
            raise ValueError("shots_per_point must be a positive integer")
        if not isinstance(shots_per_point, (int, np.integer)) or shots_per_point < 1:
            raise ValueError("shots_per_point must be a positive integer")
        shots_per_point = int(shots_per_point)
        spectroscopy_expts = list(flatten_exp_lists(spectroscopy_expts))
        if not spectroscopy_expts:
            raise ValueError("spectroscopy_expts cannot be empty")

        rng = np.random.default_rng(seed)
        subsampled_expts = []
        job_summaries = []
        available_shots = []

        for job_index, expt in enumerate(spectroscopy_expts):
            if (expt.cfg.expt.get("active_reset", False)
                    and expt.cfg.expt.get("pre_selection_reset", False)):
                raise ValueError(
                    "shot subsampling does not support pre_selection_reset; "
                    "the saved average is conditioned on herald readouts"
                )
            if "idata" not in expt.data or "qdata" not in expt.data:
                raise ValueError(f"spectroscopy job {job_index} has no saved single-shot IQ data")
            if "avgi" not in expt.data or "avgq" not in expt.data:
                raise ValueError(f"spectroscopy job {job_index} has no saved averaged IQ data")

            saved_avgi = np.asarray(expt.data["avgi"], dtype=float)
            saved_avgq = np.asarray(expt.data["avgq"], dtype=float)
            if saved_avgi.shape != saved_avgq.shape:
                raise ValueError(f"spectroscopy job {job_index} has mismatched avgi/avgq shapes")
            point_count = saved_avgi.size

            def point_rows(values, name):
                try:
                    array = np.asarray(values)
                except ValueError:
                    array = None
                if array is not None and array.ndim >= 2 and array.shape[0] == point_count:
                    return [np.asarray(array[index], dtype=float).reshape(-1)
                            for index in range(point_count)]
                if point_count == 1 and array is not None and array.dtype != object:
                    return [np.asarray(array, dtype=float).reshape(-1)]
                if len(values) == point_count:
                    return [np.asarray(values[index], dtype=float).reshape(-1)
                            for index in range(point_count)]
                raise ValueError(
                    f"spectroscopy job {job_index} has {name} that does not "
                    f"match its {point_count} sweep points"
                )

            idata_rows = point_rows(expt.data["idata"], "idata")
            qdata_rows = point_rows(expt.data["qdata"], "qdata")

            read_num = int(expt.cfg.get("read_num", 0))
            if read_num < 1:
                read_num = 1
                if expt.cfg.expt.get("parity_check", False):
                    read_num += 1
                if expt.cfg.expt.get("active_reset", False):
                    reset_params = MMAveragerProgram.get_active_reset_params(expt.cfg)
                    read_num += MMAveragerProgram.active_reset_read_num(**reset_params)
                if expt.cfg.expt.get("multiparity_readout", False):
                    read_num += 1
            final_lane = read_num - 1

            final_i_rows = []
            final_q_rows = []
            for point_index, (idata, qdata) in enumerate(zip(idata_rows, qdata_rows)):
                if len(idata) != len(qdata):
                    raise ValueError(
                        f"spectroscopy job {job_index}, point {point_index} has "
                        "different I/Q shot counts"
                    )
                if len(idata) % read_num:
                    raise ValueError(
                        f"spectroscopy job {job_index}, point {point_index} raw "
                        f"length {len(idata)} is not divisible by read_num={read_num}"
                    )
                final_i = idata[final_lane::read_num]
                final_q = qdata[final_lane::read_num]
                if len(final_i) < shots_per_point:
                    raise ValueError(
                        f"spectroscopy job {job_index}, point {point_index} has "
                        f"only {len(final_i)} final-readout shots; requested "
                        f"{shots_per_point}"
                    )
                final_i_rows.append(final_i)
                final_q_rows.append(final_q)
                available_shots.append(len(final_i))

            saved_avgi_flat = saved_avgi.reshape(-1)
            saved_avgq_flat = saved_avgq.reshape(-1)
            full_i_mean = np.asarray([np.mean(values) for values in final_i_rows])
            full_q_mean = np.asarray([np.mean(values) for values in final_q_rows])
            full_raw_minus_saved_avgi = full_i_mean - saved_avgi_flat
            full_raw_minus_saved_avgq = full_q_mean - saved_avgq_flat

            sampled_avgi = np.empty(point_count, dtype=float)
            sampled_avgq = np.empty(point_count, dtype=float)
            for point_index, (final_i, final_q) in enumerate(zip(final_i_rows, final_q_rows)):
                selected_indices = rng.choice(len(final_i),
                                              size=shots_per_point,
                                              replace=False)
                sampled_avgi[point_index] = (
                    saved_avgi_flat[point_index]
                    + np.mean(final_i[selected_indices])
                    - full_i_mean[point_index]
                )
                sampled_avgq[point_index] = (
                    saved_avgq_flat[point_index]
                    + np.mean(final_q[selected_indices])
                    - full_q_mean[point_index]
                )

            sampled_avgi = sampled_avgi.reshape(saved_avgi.shape)
            sampled_avgq = sampled_avgq.reshape(saved_avgq.shape)
            sampled_data = AttrDict(dict(expt.data))
            sampled_data["avgi"] = sampled_avgi
            sampled_data["avgq"] = sampled_avgq
            sampled_data["amps"] = np.abs(sampled_avgi + 1j * sampled_avgq)
            sampled_data["phases"] = np.angle(sampled_avgi + 1j * sampled_avgq)
            sampled_data.pop("Pe", None)
            sampled_data.pop("return_quadrature", None)

            sampled_expt = copy(expt)
            sampled_expt.data = sampled_data
            subsampled_expts.append(sampled_expt)
            job_summaries.append(AttrDict(dict(
                job_index=job_index,
                read_num=read_num,
                point_count=point_count,
                minimum_available_shots=min(len(values) for values in final_i_rows),
                maximum_available_shots=max(len(values) for values in final_i_rows),
                median_full_raw_minus_saved_avgi=float(
                    np.median(full_raw_minus_saved_avgi)
                ),
                median_full_raw_minus_saved_avgq=float(
                    np.median(full_raw_minus_saved_avgq)
                ),
                maximum_full_raw_minus_saved_avgi_scatter=float(
                    np.max(np.abs(
                        full_raw_minus_saved_avgi
                        - np.median(full_raw_minus_saved_avgi)
                    ))
                ),
                maximum_full_raw_minus_saved_avgq_scatter=float(
                    np.max(np.abs(
                        full_raw_minus_saved_avgq
                        - np.median(full_raw_minus_saved_avgq)
                    ))
                ),
            )))

        metadata = AttrDict(dict(
            shots_per_point=shots_per_point,
            seed=seed,
            replace=False,
            minimum_available_shots=min(available_shots),
            maximum_available_shots=max(available_shots),
            job_summaries=job_summaries,
        ))
        return subsampled_expts, metadata

    def analyze(self, 
                data=None, 
                **kwargs):
        """
        Analyze a single job or an aggregate calibration/spectroscopy experiment.

        ``stage`` selects one of four paths:

        1. ``None`` calls ``_quadrature`` for one saved job. It converts the two
           preparation-phase measurements into the real return quadrature
           ``Q_phi`` at that job's analyzer phase. It does not combine the
           ``phi=0`` and ``phi=90`` jobs into a complex return.
        2. ``'calibration'`` calls ``analyze_calibration`` on ``batch_expts`` and
           returns the measured phase per physical Floquet cycle modulo 180
           degrees, together with the reconstructed hardware and mode labels.
        3. ``'orthogonality'`` reconstructs the raw zero-cycle cross-return
           matrix between independently selected encoder and decoder paths.
        4. ``'spectrum'`` reconstructs ``A_alpha=Q_0-iQ_90`` from all chunked
           spectroscopy jobs, transforms it to the requested phase frame, and
           calculates the measured FFT and the matching fixed-N Hamiltonian,
           eigenenergies, LDOS weights, and theory spectrum.

        The spectrum result stores both ``acquired_reconstruction`` and
        ``reconstruction``. The former is exactly the complex return built from
        the saved quadratures. The latter is the return after the selected
        phase-frame transformation and is passed to ``analyze_spectrum``.
        ``_postprocess_reconstruction`` is therefore called even for
        ``phase_frame='as_acquired'``; with branch zero it is a pass-through that
        also records the Kerr and phase-frame metadata.

        Spectrum keyword arguments:

        - ``occupations`` optionally fixes the reconstruction row order and must
          exactly match the occupations represented by the jobs.
        - ``calibration`` accepts the aggregate calibration experiment, its data,
          or an equivalent mapping. It is required for ``zero_kerr`` and
          ``manual_kerr`` because those frames must remove the saved analyzer
          correction and construct a new occupation-dependent correction.
        - ``phase_frame`` is ``'as_acquired'`` by default. ``'uncorrected'``
          undoes the saved final-analyzer correction. ``'zero_kerr'`` rephases
          to a Hamiltonian with zero M1 self-Kerr. ``'manual_kerr'`` rephases to
          the signed value supplied through ``manual_kerr_MHz``. Supplying
          ``manual_kerr_MHz`` with the default frame selects ``'manual_kerr'``.
        - ``cycle_branches`` is an integer applied to every occupation, one
          integer per occupation, or a dictionary keyed by occupation. A branch
          ``b`` changes the chosen phase slope by ``180*b`` degrees per cycle.
          ``second_branch=True`` is shorthand for adding one branch everywhere
          and cannot be combined with a nonzero ``cycle_branches`` value.
        - ``legacy`` resolves old jobs that did not save the analyzer-phase
          application sign. Use ``True`` for the old ``+correction`` convention
          and ``False`` for the current ``-correction`` convention. It is only
          meaningful when undoing or replacing the acquired correction.
        - ``fft_window`` is ``'raw'``, ``'hann'``, ``'hamming'``, or ``'blackman'``;
          ``zero_padding`` is an integer FFT-length multiplier.
        - ``spectrum_method`` is ``'fft'`` by default. ``'matrix_pencil'`` or
          ``'mpm'`` additionally performs the rowwise Matrix-Pencil analysis and
          stores it in ``data.matrix_pencil``. The raw FFT remains in
          ``data.spectrum`` and is used as the unfiltered reference.
        - ``shots_per_point`` optionally reconstructs every spectroscopy job
          from that many randomly selected final-readout shots at each saved
          cycle and preparation phase. ``shot_seed`` makes the without-
          replacement subsets reproducible. The raw jobs are not modified, and
          the returned data records the selection in ``shot_subsampling``.
        - Matrix-Pencil heuristic kwargs use the ``mpm_`` prefix. The defaults
          reproduce the notebook values: 
          ``mpm_requested_max_modes=35``,
          ``mpm_minimum_consecutive_ranks=3``,
          ``mpm_minimum_supporting_rows=1``,
          ``mpm_track_frequency_tolerance_bins=1.5``,
          ``mpm_noise_singular_value_factor=2.858``, ; Note: this is math-based heuristic value based on REF: IEEE Transactions on Information Theory, 60(8), 5040-5053.
          ``mpm_numerical_floor=1e-10``, and pole radii from 0.2 to 1.05.
          
          
          Track/merge/dedup frequency tolerances, decay tolerances, pencil
          length, rank-sweep extent, growth clipping, and least-squares rcond
          can also be overridden; see ``analyze_matrix_pencil``.

        Self-Kerr convention:

        The standard acquisition preserves the physical M1 self-Kerr. For an
        M1 occupation ``n``, ``analyze_spectrum`` uses

            ``E_K/h=(K/2)*n*(n-1)``

        with signed ``K=physical_kerr_MHz``. The corresponding return-phase
        slope per Floquet cycle is

            ``Gamma_K=-360*(E_K/h)*T_cycle`` degrees/cycle.

        If calibration measures ``Gamma_meas=Gamma_unwanted+Gamma_K``, then
        ``build_phase_correction`` sends

            ``Gamma_correction=Gamma_meas-Gamma_K``

        to the final analyzer. Removing Kerr from the correction does not remove
        Kerr from the measured return: after the analyzer correction the residual
        phase is ``Gamma_meas-Gamma_correction=Gamma_K``. Consequently the
        default ``phase_frame='as_acquired'`` leaves ``A_alpha`` unchanged and
        compares it with a Hamiltonian containing the signed physical Kerr.

        ``phase_frame='zero_kerr'`` instead removes that residual physical-Kerr
        evolution and compares against ``K=0``. ``phase_frame='manual_kerr'``
        removes the acquired correction and applies the correction appropriate
        to the requested signed Kerr. If a custom acquisition canceled the Kerr
        itself by sending the full ``Gamma_meas`` to the analyzer, its acquired
        return is already in the zero-Kerr frame and must not be interpreted by
        the default physical-Kerr assumption.

        ``data``, when supplied, replaces ``self.data`` before analysis. Every
        path returns ``self.data``; the calibration and spectrum paths replace it
        with their aggregate result.
        """
        
        if data is not None:
            self.data = data
        stage = kwargs.pop("stage", None)
        if stage is None:
            self._quadrature(self)
        elif stage == "calibration":
            self.data = self.analyze_calibration(
                self.batch_expts, 
                kwargs.get("occupations"), 
                kwargs.get("cycle_pairs"),
                repeats=kwargs.get("repeats"),
            )
            saved = self._saved_parameters(self.batch_expts, 
                                           getattr(self, "_analysis_station", None))
            self.data.hardware = saved.hardware
            self.data.mode_labels = saved.mode_labels
        elif stage == "orthogonality":
            self.data = self.reconstruct_orthogonality(
                self.batch_expts,
                kwargs.get("occupations"),
            )
        elif stage == "propagator":
            self.data = self.reconstruct_propagator(
                self.batch_expts,
                kwargs.get("occupations"),
            )
        elif stage == "spectrum":
            spectrum_method = str(kwargs.get("spectrum_method", "fft")).lower()
            if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
                spectrum_method = "matrix_pencil"
            if spectrum_method not in ("fft", "matrix_pencil"):
                raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
            analysis_expts = self.batch_expts
            shot_subsampling = None
            shots_per_point = kwargs.get("shots_per_point", None)
            if shots_per_point is not None:
                analysis_expts, shot_subsampling = self.subsample_spectroscopy_shots(
                    self.batch_expts,
                    shots_per_point,
                    seed=kwargs.get("shot_seed", None),
                )
            elif kwargs.get("shot_seed", None) is not None:
                raise ValueError("shot_seed requires shots_per_point")
            saved = self._saved_parameters(analysis_expts, 
                                           getattr(self, "_analysis_station", None))
            acquired_reconstruction = self.reconstruct_spectroscopy(analysis_expts, 
                                                                    kwargs.get("occupations"))
            photon_numbers = {sum(occupation) for occupation in acquired_reconstruction.occupations}
            if len(photon_numbers) != 1:
                raise ValueError("spectroscopy jobs must belong to one fixed-photon-number sector")
            photon_number = photon_numbers.pop()
            calibration_arg = kwargs.get("calibration", None)
            calibration = self._calibration_data(calibration_arg, getattr(self, "_analysis_station", None))
            if calibration is not None and "mode_labels" in calibration and list(calibration.mode_labels) != list(saved.mode_labels):
                raise ValueError("calibration and spectroscopy use different modes")
            if calibration is not None and "hardware" in calibration:
                calibration_couplings = np.asarray(calibration.hardware.couplings_MHz)
                if calibration_couplings.shape != saved.hardware.couplings_MHz.shape or not np.isclose(calibration.hardware.floquet_cycle_us, saved.hardware.floquet_cycle_us) or not np.allclose(calibration_couplings, saved.hardware.couplings_MHz):
                    raise ValueError("calibration and spectroscopy used different Floquet hardware")
            cycle_branches = kwargs.get("cycle_branches", 0)
            if kwargs.get("second_branch", False):
                cycle_branches = self._cycle_branches(acquired_reconstruction.final_occupations,
                                                      cycle_branches)
                if np.any(cycle_branches):
                    raise ValueError("use either cycle_branches or second_branch, not both")
                cycle_branches += 1
            saved_correction = self._saved_correction(analysis_expts)
            postprocessed = self._postprocess_reconstruction(
                acquired_reconstruction, 
                saved_correction, 
                calibration, 
                saved.hardware,
                kwargs.get("phase_frame", "as_acquired"), 
                kwargs.get("manual_kerr_MHz", None), 
                cycle_branches, 
                kwargs.get("legacy", None))
            spectrum = self.analyze_spectrum(
                postprocessed.reconstruction, 
                photon_number, 
                saved.detunings,
                saved.hardware.couplings_MHz, 
                saved.hardware.floquet_cycle_us,
                postprocessed.physical_kerr_MHz,
                kwargs.get("fft_window", "raw"),
                kwargs.get("zero_padding", 1),
            )
            self.data = AttrDict(dict(
                calibration=calibration, 
                correction=saved_correction,
                saved_correction=saved_correction, 
                target_correction=postprocessed.target_correction,
                acquired_reconstruction=acquired_reconstruction, 
                reconstruction=postprocessed.reconstruction,
                spectrum=spectrum,
                hardware=saved.hardware, 
                photon_number=photon_number, 
                detunings=saved.detunings,
                mode_labels=saved.mode_labels, 
                phase_frame=postprocessed.phase_frame,
                cycle_branches=postprocessed.cycle_branches,
                analyzer_phase_application_sign=postprocessed.analyzer_phase_application_sign,
                legacy_analyzer_migration=postprocessed.legacy_analyzer_migration,
                spectrum_method=spectrum_method,
            ))
            if spectrum_method == "matrix_pencil":
                self.data.matrix_pencil = self.analyze_matrix_pencil(
                    postprocessed.reconstruction,
                    spectrum,
                    requested_max_modes=kwargs.get("mpm_requested_max_modes", None),
                    pencil_length=kwargs.get("mpm_pencil_length", None),
                    minimum_consecutive_ranks=kwargs.get("mpm_minimum_consecutive_ranks", 3),
                    minimum_supporting_rows=kwargs.get("mpm_minimum_supporting_rows", 1),
                    track_frequency_tolerance_bins=kwargs.get("mpm_track_frequency_tolerance_bins", 1.5),
                    merge_frequency_tolerance_bins=kwargs.get("mpm_merge_frequency_tolerance_bins", None),
                    dedup_frequency_tolerance_bins=kwargs.get("mpm_dedup_frequency_tolerance_bins", None),
                    track_decay_tolerance_per_us=kwargs.get("mpm_track_decay_tolerance_per_us", None),
                    dedup_decay_tolerance_per_us=kwargs.get("mpm_dedup_decay_tolerance_per_us", None),
                    match_decay=kwargs.get("mpm_match_decay", True),
                    numerical_floor=kwargs.get("mpm_numerical_floor", 1e-10),
                    noise_singular_value_factor=kwargs.get("mpm_noise_singular_value_factor", 2.858), # this is based on the paper; IEEE Transactions on Information Theory 60.8 (2014): 5040-5053.
                    minimum_pole_radius=kwargs.get("mpm_minimum_pole_radius", 0.2),
                    maximum_pole_radius=kwargs.get("mpm_maximum_pole_radius", 1.05),
                    require_early_start=kwargs.get("mpm_require_early_start", True),
                    rank_sweep_extra=kwargs.get("mpm_rank_sweep_extra", None),
                    clip_growth=kwargs.get("mpm_clip_growth", True),
                    least_squares_rcond=kwargs.get("mpm_least_squares_rcond", None),
                    store_rank_sweeps=kwargs.get("mpm_store_rank_sweeps", False),
                )
            if shot_subsampling is not None:
                self.data.shot_subsampling = shot_subsampling
            if hasattr(calibration_arg, "batch_job_ids"):
                self.calibration_job_ids = list(calibration_arg.batch_job_ids)
        else:
            raise ValueError(
                "stage must be 'calibration', 'orthogonality', "
                "'propagator', or 'spectrum'"
            )
        return self.data

    @staticmethod
    def _quadrature(expt):
        if "return_quadrature" in expt.data:
            return np.asarray(expt.data["return_quadrature"])
        cycles = expt.data["ypts"]
        theta = expt.data["xpts"]
        signal = np.asarray(expt.data["avgi"]).reshape(len(cycles), len(theta))
        q = expt.cfg.expt.qubits[0]
        Ig = expt.cfg.device.readout.Ig[q]
        Ie = expt.cfg.device.readout.Ie[q]
        if np.isclose(Ig, Ie):
            raise ValueError("Ig and Ie are identical; recalibrate readout")
        expt.data["Pe"] = (signal - Ig) / (Ie - Ig)
        expt.data["return_quadrature"] = expt.data["Pe"][:, 0] - expt.data["Pe"][:, 1]
        return expt.data["return_quadrature"]

    @staticmethod
    def _unwrap_cycle_phase(complex_return, 
                            physical_cycles, 
                            valid_mask, 
                            closed_mask, 
                            guide_weight=0.2):
        raw_phase = np.rad2deg(np.angle(complex_return))
        fit_mask = valid_mask & closed_mask
        guide_mask = valid_mask & ~closed_mask

        # Closed-pair points are two physical cycles apart, so their slope is unique modulo 180 deg/cycle.
        slopes = np.linspace(-90., 90., 18001)
        fit_return = complex_return[fit_mask] / np.abs(complex_return[fit_mask])
        rotation = np.exp(-1j * np.deg2rad(np.outer(slopes, physical_cycles[fit_mask])))
        score = np.abs(rotation @ fit_return) / len(fit_return)

        # Odd points may have a different offset; only their internal phase progression is used as a weak guide.
        if np.count_nonzero(guide_mask) > 1:
            guide_return = complex_return[guide_mask] / np.abs(complex_return[guide_mask])
            rotation = np.exp(-1j * np.deg2rad(np.outer(slopes, physical_cycles[guide_mask])))
            score += guide_weight * np.abs(rotation @ guide_return) / len(guide_return)

        slope = slopes[np.argmax(score)]
        phase = np.full(len(complex_return), np.nan)
        for mask in [fit_mask, guide_mask]:
            if not np.any(mask):
                continue
            unit_return = complex_return[mask] / np.abs(complex_return[mask])
            rotated_return = unit_return * np.exp(-1j * np.deg2rad(slope * physical_cycles[mask]))
            intercept = np.rad2deg(np.angle(np.sum(rotated_return)))
            phase_model = intercept + slope * physical_cycles[mask]
            phase[mask] = raw_phase[mask] + 360. * np.round((phase_model - raw_phase[mask]) / 360.)
        return phase

    @classmethod
    def analyze_cycle_phase(cls, #batch_expt is plugged in
                            phi0_expts, 
                            phi90_expts, 
                            occupation, 
                            cycle_pairs, 
                            radius_fraction=0.1,
                            unwrap_mode="pair"):
        """
        The method recieves phi0 and phi90 experiments to
            1. reconstruct Q0-iQ90
            2. Unwrap the phase using np.unwrap
            2. fit phase per physical entire cycle.
        
        Return: AttrDict containing
            - occupation: tuple(occupation), #state string tuple
            - physical_cycles: physical_cycles, #floquet cycles
            - complex_return: complex_return,
            - relative_return: relative_return,
            - return_phase: phase,
            - phase_fit: phase_fit,
            - fnames: fnames,
            - phase_per_cycle: parameters[0],
            - phase_error: np.sqrt(covariance[0, 0])}
        """
        if isinstance(phi0_expts, (list, tuple)):
            phi0_expts = list(phi0_expts)
        else:
            phi0_expts = [phi0_expts]
        if isinstance(phi90_expts, (list, tuple)):
            phi90_expts = list(phi90_expts)
        else:
            phi90_expts = [phi90_expts]
        if len(phi0_expts) != len(phi90_expts):
            raise ValueError("phi=0 and phi=90 repeat counts differ")

        if unwrap_mode not in ("pair", "odd_guide"):
            raise ValueError("unwrap_mode must be 'pair' or 'odd_guide'")
        cycle_pairs = np.asarray(cycle_pairs)
        complex_returns = []
        fnames = []

        for expt_phi0, expt_phi90 in zip(phi0_expts, phi90_expts):
            if not np.array_equal(expt_phi0.data["ypts"], cycle_pairs):
                raise ValueError("saved cycle-pair sweep changed")
            if not np.array_equal(expt_phi90.data["ypts"], cycle_pairs):
                raise ValueError("saved cycle-pair sweep changed")
            if not np.allclose(expt_phi0.data["xpts"], [0., 180.]):
                raise ValueError("saved preparation phases changed")
            if not np.allclose(expt_phi90.data["xpts"], [0., 180.]):
                raise ValueError("saved preparation phases changed")
            complex_returns.append(cls._quadrature(expt_phi0) - 1j * cls._quadrature(expt_phi90))
            for expt in [expt_phi0, expt_phi90]:
                if hasattr(expt, "fname"):
                    fname = str(expt.fname).replace("\\", "/").split("/")[-1]
                    fnames.append(fname)

        complex_returns = np.asarray(complex_returns)
        # Rows are repeated jobs; average the same cycle point across repeats.
        complex_return = complex_returns.mean(axis=0)
        magnitude = np.abs(complex_return)
        positive = magnitude[np.isfinite(magnitude) & (magnitude > 0.)]
        radius_floor = 0.
        if len(positive):
            radius_floor = radius_fraction * np.median(positive)
        valid_mask = np.isfinite(complex_return) & (magnitude > radius_floor)
        if unwrap_mode == "odd_guide":
            physical_cycles = cycle_pairs
            closed_mask = physical_cycles % 2 == 0
        else:
            physical_cycles = 2 * cycle_pairs
            closed_mask = np.ones(len(cycle_pairs), dtype=bool)
        fit_mask = valid_mask & closed_mask
        if np.count_nonzero(fit_mask) < 3:
            raise RuntimeError(f"{tuple(occupation)} has too few valid IQ points")

        phase = cls._unwrap_cycle_phase(complex_return, physical_cycles, valid_mask, closed_mask)
        parameters, covariance = np.polyfit(physical_cycles[fit_mask], 
                                            phase[fit_mask], 
                                            1, 
                                            cov=True)
        phase_fit = parameters[0] * physical_cycles + parameters[1]
        relative_return = magnitude.copy()
        if magnitude[0] > 1e-12:
            relative_return = magnitude / magnitude[0]
        return AttrDict(dict(
            occupation=tuple(occupation),
            physical_cycles=physical_cycles,
            complex_return=complex_return,
            relative_return=relative_return,
            return_phase=phase,
            phase_fit=phase_fit,
            closed_mask=closed_mask,
            unwrap_mode=unwrap_mode,
            fnames=fnames,
            phase_per_cycle=parameters[0],
            phase_error=np.sqrt(covariance[0, 0]),
        ))

    @staticmethod
    def build_phase_correction(occupations, 
                               phase_mod180, 
                               cycle_branches, 
                               physical_kerr_MHz,
                               floquet_cycle_us,
                               correction_sign=1.):
        """
        Compute the phase correction that the pulse program subtracts from the
        final analyzer, excluding the always-on M1 self-Kerr phase.
        """
        phase_mod180 = np.asarray(phase_mod180)
        cycle_branches = np.asarray(cycle_branches)
        if correction_sign not in (-1., 1.):
            raise ValueError("correction_sign must be +1 or -1")
        if phase_mod180.shape != (len(occupations),) or cycle_branches.shape != (len(occupations),):
            raise ValueError("phase_mod180 and cycle_branches must match occupations")
        physical_kerr_MHz = float(physical_kerr_MHz)
        if not np.isfinite(physical_kerr_MHz):
            raise ValueError("physical_kerr_MHz must be finite")
        n_M1 = np.asarray([occupation[0] for occupation in occupations])
        kerr_energy_MHz = 0.5 * physical_kerr_MHz * n_M1 * (n_M1 - 1)
        kerr_phase = -360. * kerr_energy_MHz * floquet_cycle_us
        measured_phase = phase_mod180 + 180. * cycle_branches
        analyzer_phase = correction_sign * (measured_phase - kerr_phase) 

        phase_by_occupation = {}
        for occupation, phase in zip(occupations, analyzer_phase):
            phase_by_occupation[tuple(occupation)] = float(phase)
        return AttrDict(dict(
            measured_phase=measured_phase, 
            kerr_phase=kerr_phase,
            cycle_branches=cycle_branches,
            physical_kerr_MHz=physical_kerr_MHz,
            phase_by_occupation=phase_by_occupation,
        ))

    @classmethod
    def reconstruct_orthogonality(cls,
                                  orthogonality_expts,
                                  occupations=None):
        """Reconstruct the zero-cycle encoder-to-decoder cross-return matrix.

        Rows are decoder occupations and columns are encoder occupations. Each
        encoder job contains outer rows ``(decoder, phi=0/90)`` and inner
        preparation phases ``theta=0/180``. With the QICK phase convention,
        ``M[j, i] = Q_0 - i Q_90``. The raw matrix is deliberately not divided
        by its zero-cycle values because those values are the diagnostic.
        """
        first_cfg = orthogonality_expts[0].cfg.expt
        swap_stors = [int(stor) for stor in first_cfg.swap_stors]
        decoder_order = [
            tuple(occupation)
            for occupation in first_cfg.orthogonality_decoder_occupations
        ]
        columns = {}
        for expt in orthogonality_expts:
            cfg = expt.cfg.expt
            encoder = tuple(cfg.spectroscopy_occupations)
            quadrature = np.asarray(
                cls._quadrature(expt), dtype=float
            ).reshape(len(decoder_order), 2)
            columns[encoder] = quadrature[:, 0] - 1j * quadrature[:, 1]

        occupation_order = decoder_order if occupations is None else [
            tuple(occupation) for occupation in occupations
        ]

        decoder_indices = [decoder_order.index(occ) for occ in occupation_order]
        matrix = np.column_stack([
            columns[occupation][decoder_indices]
            for occupation in occupation_order
        ]).astype(complex, copy=False)
        amplitude = np.abs(matrix)
        power = amplitude ** 2
        diagonal_amplitude = np.abs(np.diag(matrix))
        denominator = np.outer(diagonal_amplitude, diagonal_amplitude)
        normalized_power = power / denominator
        normalized_amplitude = matrix / np.sqrt(denominator)
        offdiagonal_normalized_power = normalized_power.copy()
        np.fill_diagonal(offdiagonal_normalized_power, 0.)
        column_leakage = np.sum(offdiagonal_normalized_power, axis=0)

        return AttrDict(dict(
            occupations=occupation_order,
            mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors],
            matrix=matrix,
            amplitude=amplitude,
            power=power,
            diagonal_amplitude=diagonal_amplitude,
            normalized_amplitude=normalized_amplitude,
            normalized_power=normalized_power,
            offdiagonal_normalized_power=offdiagonal_normalized_power,
            column_leakage=column_leakage,
            matrix_orientation="rows=decoder, columns=encoder",
        ))

    @classmethod
    def reconstruct_propagator(cls,
                               propagator_expts,
                               occupations=None):
        """Reconstruct raw and Kerr-preserving ``M_q[j, i]`` matrices."""
        first_cfg = propagator_expts[0].cfg.expt
        swap_stors = [int(stor) for stor in first_cfg.swap_stors]
        cycles = [int(cycle) for cycle in first_cfg.propagator_cycles]
        decoder_order = [
            tuple(occupation)
            for occupation in first_cfg.propagator_occupations
        ]

        columns = {}
        for expt in propagator_expts:
            encoder = tuple(expt.cfg.expt.spectroscopy_occupations)
            quadrature = np.asarray(
                cls._quadrature(expt), dtype=float
            ).reshape(len(cycles), len(decoder_order), 2)
            columns[encoder] = (
                quadrature[:, :, 0] - 1j * quadrature[:, :, 1]
            )

        occupation_order = decoder_order if occupations is None else [
            tuple(occupation) for occupation in occupations
        ]
        decoder_indices = [
            decoder_order.index(occupation)
            for occupation in occupation_order
        ]
        raw_matrices = np.stack([
            columns[occupation][:, decoder_indices]
            for occupation in occupation_order
        ], axis=2).astype(complex, copy=False)
        phase_correction = np.asarray(
            first_cfg.propagator_decoder_phase_correction_deg,
            dtype=float,
        )[decoder_indices]
        matrices = raw_matrices * np.exp(
            -1j * np.deg2rad(
                np.asarray(cycles)[:, None, None]
                * phase_correction[None, :, None]
            )
        )

        return AttrDict(dict(
            cycles=np.asarray(cycles, dtype=int),
            occupations=occupation_order,
            mode_labels=["M1"] + [f"S{stor}" for stor in swap_stors],
            raw_matrices=raw_matrices,
            matrices=matrices,
            decoder_phase_correction_deg=phase_correction,
            matrix_orientation="rows=decoder, columns=encoder",
        ))

    @classmethod
    def reconstruct_spectroscopy(cls, 
                                 spectroscopy_expts, 
                                 occupations=None):
        """
        Combine chunked spectroscopy jobs into the complex return amplitudes in
        the phase frame used during acquisition.

        For each initial occupation ``alpha``, the saved jobs must contain the
        analyzer settings ``phi=0`` and ``phi=90``. ``_quadrature`` first forms
        ``Q_phi=Pe(theta=0)-Pe(theta=180)`` from the two preparation phases.
        With the QICK convention ``Q_phi=Re[A_alpha exp(+i phi)]``, the two
        analyzer quadratures give

            ``A_alpha=Q_0-i Q_90=<alpha|U|alpha>``.

        The analyzer phase already contains any correction played by the pulse
        program. This method only reconstructs what was acquired: it does not
        undo or replace that correction, select a 180-degree phase branch,
        change the self-Kerr frame, or perform an FFT. Those operations belong
        to ``analyze(stage='spectrum')`` after this reconstruction.

        Jobs are grouped using the saved ``spectroscopy_occupations`` and
        ``spectroscopy_analyzer_phase``. Cycle chunks are concatenated and
        sorted, and every occupation and analyzer quadrature must cover the same
        non-overlapping cycle points. If ``occupations`` is supplied, it sets
        the returned row order and must contain exactly the occupations present
        in the saved jobs.

        Returns an AttrDict with ``occupations``, the common sorted ``cycles``,
        and a complex array ``A`` of shape ``(n_occupations, n_cycles)``. This
        result is called ``acquired_reconstruction`` by ``analyze`` to distinguish
        it from the reconstruction after an optional phase-frame transformation.
        """
        if not spectroscopy_expts:
            raise ValueError("spectroscopy_expts cannot be empty")
        grouped = {}
        for expt in spectroscopy_expts:
            cfg = expt.cfg.expt
            occupation = tuple(cfg.spectroscopy_occupations)
            final_occupation = tuple(cfg.get("spectroscopy_final_occupations", occupation))
            state = (final_occupation, occupation)
            phi = cfg.spectroscopy_analyzer_phase
            if phi not in (0., 90.):
                raise ValueError(f"{occupation} has analyzer phase {phi}; expected 0 or 90")
            if "floquet_cycles" not in cfg:
                raise ValueError(f"{occupation}, phi={phi}: this is not a spectroscopy job")
            if not np.allclose(expt.data["xpts"], [0., 180.]):
                raise ValueError(f"{occupation}, phi={phi}: saved preparation phases changed")
            if not np.array_equal(expt.data["ypts"], cfg.floquet_cycles):
                raise ValueError(f"{occupation}, phi={phi}: saved cycles do not match its config")
            if state not in grouped:
                grouped[state] = {0.: [], 90.: []}
            grouped[state][phi].append(expt)

        if occupations is None:
            state_order = list(grouped)
        else:
            occupation_order = [tuple(occupation) for occupation in occupations]
            state_order = [next(state for state in grouped if state[1] == occupation) for occupation in occupation_order]
        if len(state_order) != len(grouped) or set(state_order) != set(grouped):
            raise ValueError("spectroscopy occupations do not match the saved configs")
        expected_cycles = None
        rows = []

        for state in state_order:
            quadratures = []
            for phi in [0., 90.]:
                expts = grouped[state][phi]
                if not expts:
                    raise ValueError(f"{occupation} is missing phi={phi} data")
                cycles = np.concatenate([np.asarray(expt.data["ypts"]) for expt in expts])
                quadrature = np.concatenate([cls._quadrature(expt) for expt in expts])
                order = np.argsort(cycles)
                cycles = cycles[order]
                if len(np.unique(cycles)) != len(cycles):
                    raise ValueError(f"{occupation}, phi={phi}: spectroscopy cycles overlap")
                if expected_cycles is None:
                    expected_cycles = cycles
                elif not np.array_equal(cycles, expected_cycles):
                    raise ValueError(f"{occupation}, phi={phi}: spectroscopy cycles are incomplete")
                quadratures.append(quadrature[order])
            rows.append(quadratures[0] - 1j * quadratures[1])
        A = np.asarray(rows, dtype = complex)
        occupation_order = [state[1] for state in state_order]
        final_occupations = [state[0] for state in state_order]
        normalized_A = np.asarray([row / row[0] if initial == final else row for row, (final, initial) in zip(A, state_order)])
        return AttrDict(dict(occupations=occupation_order,
                             final_occupations=final_occupations,
                             cycles=expected_cycles, 
                             A= A,
                             A_norm= normalized_A))

    @staticmethod
    def analyze_spectrum(reconstruction, 
                         photon_number, 
                         detunings, 
                         couplings_MHz,
                         floquet_cycle_us, 
                         physical_kerr_MHz, 
                         fft_window="raw", 
                         zero_padding=1):
        """
        Build the fixed-N Hamiltonian, LDOS weights, and measured/theory spectra.
        
        Returns AttrDict with:
            - time_us=time_us, 
            - energy_MHz=energy_MHz, 
            - measured_local=measured_local, 
            - theory_local=theory_local,
            - measured=measured, 
            - theory=theory, 
            - energies_MHz=energies_MHz,
            - fock_basis=fock_basis,
            - basis_eigenstate_weights=np.abs(states) ** 2,
            - eigenstate_weights=eigenstate_weights, 
            - physical_kerr_MHz=physical_kerr_MHz,
            - complete_basis=complete_basis, 
            - energy_limit_MHz=energy_limit_MHz,
            - fft_window=fft_window, 
            - zero_padding=zero_padding,
            - fft_resolution_MHz=1. / (len(cycles) * sample_time_us),
        
        energies_MHz is the list of energy eigenvalue of the hamiltonian.
        theory_local is the fft result expected from theory, and 
        measured_local is the measured fft result.
        """
        cycles = reconstruction.cycles
        A = reconstruction.A
        final_occupations = reconstruction.get("final_occupations", reconstruction.occupations)
        detunings = np.asarray(detunings)
        physical_kerr_MHz = float(physical_kerr_MHz)
        if not np.isfinite(physical_kerr_MHz):
            raise ValueError("physical_kerr_MHz must be finite")
        if len(cycles) < 2:
            raise ValueError("spectroscopy requires at least two cycle points")
        time_us = cycles * floquet_cycle_us
        sample_time_us = time_us[1] - time_us[0]
        if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
            raise ValueError("spectroscopy time samples are not uniform")

        if fft_window is None:
            fft_window = "raw"
        windows = {"raw": np.ones, "hann": np.hanning,
                   "hamming": np.hamming, "blackman": np.blackman}
        if fft_window not in windows:
            raise ValueError("fft_window must be 'raw', 'hann', 'hamming', or 'blackman'")
        window = windows[fft_window](len(cycles))
        if not isinstance(zero_padding, (int, np.integer)) or zero_padding < 1:
            raise ValueError("zero_padding must be an integer >= 1")
        if np.sum(window) <= 0.:
            raise ValueError(f"{fft_window} window needs more cycle points")

        n_fft = zero_padding * len(cycles)
        energy_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_time_us))
        # One Trotter step is one complete pulse+sync Floquet cycle.
        # couplings_MHz already includes each pulse's share of that cycle:
        # g = 1/(4*pi_frac*T_cycle). Always-on self-Kerr enters without scaling.
        
        
        
        #Here, the Hamiltonian is directly calculated as a matrix in a Fock basis
        #First, product makes the all possible product states within photon_number
        #and then those are conditionally stored in fock_basis if the number = photon number
        mode_count = len(reconstruction.occupations[0])
        fock_basis = [
            list(occupation) for occupation in product(range(photon_number + 1), repeat=mode_count)
            if sum(occupation) == photon_number
        ]
        #Storing index of each fock basis
        fock_index = {tuple(occupation): index for index, occupation in enumerate(fock_basis)}
        #Making Hamiltonian matrix in a fock basis
        H_MHz = np.zeros((len(fock_basis), len(fock_basis)))
        # The pulse program adds detuning to the positive storage-M1 sideband, so the rotating-frame onsite energy is -detuning.
        onsite_MHz = np.concatenate(([0.], -detunings))
        # updating Hamiltonian indices by estimating
        # <n_i|H_{diag}|n_j> = \delta_{ij}(delta_i n_i+Kerr/2*n_M*(n_M-1) 
        #Specifically, the algorithm is
        #   1. Multiply self Kerr times n_M1
        #   2. Multiply onsize detuning times n_i
        # <n_i|H_{coupling}|n_j>  = g \delta_{n_M+1  n_i-1}\sqrt{n_M+1 n_i}+
        #                           g \delta_{n_M-1  n_i+1}\sqrt{n_M   n_i+1}
        #Specifically, the algorithm is
        #For each column occupation,
        #   1. Loop the iteraction on storage mode index i
        #   2. Find the state with n_M increased by 1 and n_i decreased by 1 
        #      using fock_index dictionary
        #   3. Add g * \sqrt{n_M+1 n_i}
        #   4. Do the same for the state iwth n_M-1 and n_i+1
        for column, occupation in enumerate(fock_basis):
            n_M1 = occupation[0]
            H_MHz[column, column] = np.dot(onsite_MHz, occupation) + 0.5 * physical_kerr_MHz * n_M1 * (n_M1 - 1)
            for mode_index, coupling_MHz in enumerate(couplings_MHz, start=1):
                if n_M1 == 0:
                    continue
                final_occupation = occupation.copy()
                final_occupation[0] -= 1
                final_occupation[mode_index] += 1
                row = fock_index[tuple(final_occupation)]
                matrix_element = coupling_MHz * np.sqrt(n_M1 * (occupation[mode_index] + 1))
                H_MHz[row, column] += matrix_element
                H_MHz[column, row] += matrix_element
        #Using np.linalg.eigh, get the eigenvalue of the Hamiltonian Matrix
        #Returns matrix with the index of (f, k), where k being eigenstate index
        #And f being fock state index.
        #So each column is an eigen state in a fock basis
        energies_MHz, states = np.linalg.eigh(H_MHz)
        #For each occupations for the experiment, calculate its index in the basis
        #that is used for the matrix setup
        basis_rows = [fock_index[tuple(occupation)] for occupation in reconstruction.occupations]
        final_rows = [fock_index[tuple(occupation)] for occupation in final_occupations]
        #Pick rows in the eigenstate matrix
        spectral_weights = states[final_rows] * states[basis_rows].conj()
        eigenstate_weights = np.abs(spectral_weights)
        #List of "Theory phase", which is the list of e^{-i2 * pi * f_{eigen} t}
        theory_phase = np.exp(-2j * np.pi * np.outer(energies_MHz, time_us))
        #Do the matrix multiplication, which will give sum_n <n|U|n> 
        #as a function of time
        theory_A = spectral_weights @ theory_phase
        
        
        #############################################################
        #######   FFT of measured dataset   #########################
        #############################################################
        
        fft_scale = n_fft / np.sum(window)
        measured_local = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(A * window, n=n_fft, axis=1), axes=1))
        diagonal = np.asarray([tuple(initial) == tuple(final) for initial, final in zip(reconstruction.occupations, final_occupations)])
        fft_normalization = np.where(diagonal, np.maximum(np.abs(A[:, 0]), 1e-12), 1.)
        measured_local /= fft_normalization[:, None]
        theory_local = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(theory_A * window, n=n_fft, axis=1), axes=1))
        measured = np.sum(measured_local, axis=0)
        theory = np.sum(theory_local, axis=0)
        if np.max(theory) > 0.:
            theory_local *= np.max(measured) / np.max(theory)
            theory = np.sum(theory_local, axis=0)
        complete_basis = np.all(diagonal) and set(map(tuple, reconstruction.occupations)) == set(map(tuple, fock_basis))
        energy_limit_MHz = min(np.max(np.abs(energy_MHz)), max(0.6, 1.2 * np.max(np.abs(energies_MHz))))

        return AttrDict(dict(
            time_us=time_us, 
            energy_MHz=energy_MHz, 
            measured_local=measured_local, 
            theory_local=theory_local,
            measured=measured, 
            theory=theory, 
            theory_A=theory_A,
            energies_MHz=energies_MHz,
            fock_basis=fock_basis,
            basis_eigenstate_weights=np.abs(states) ** 2,
            eigenstate_weights=eigenstate_weights, 
            spectral_weights=spectral_weights,
            fft_normalization=fft_normalization,
            physical_kerr_MHz=physical_kerr_MHz,
            complete_basis=complete_basis, 
            energy_limit_MHz=energy_limit_MHz,
            fft_window=fft_window, 
            zero_padding=zero_padding,
            fft_resolution_MHz=1. / (len(cycles) * sample_time_us),
        ))

    @staticmethod
    def analyze_matrix_pencil(reconstruction,
                              spectrum,
                              requested_max_modes=None,
                              pencil_length=None,
                              minimum_consecutive_ranks=3,
                              minimum_supporting_rows=1,
                              track_frequency_tolerance_bins=1.5,
                              merge_frequency_tolerance_bins=None,
                              dedup_frequency_tolerance_bins=None,
                              track_decay_tolerance_per_us=None,
                              dedup_decay_tolerance_per_us=None,
                              match_decay=True,
                              numerical_floor=1e-10,
                              noise_singular_value_factor=2.858,
                              minimum_pole_radius=0.2,
                              maximum_pole_radius=1.05,
                              require_early_start=True,
                              rank_sweep_extra=None,
                              clip_growth=True,
                              least_squares_rcond=None,
                              store_rank_sweeps=False):
        """
        Find shared damped-exponential poles independently in each occupation.

        The measured return is modeled as

            A_i[n] = sum_m c_im z_m**n,
            z_m = exp((-gamma_m - 2j*pi*f_m)*dt).

        Every occupation receives its own Hankel rank sweep. Poles are tracked
        between adjacent assumed ranks, stable rowwise candidates are merged by
        frequency across occupations, and the merged frequencies/decays are held
        fixed while every ``c_im`` is refit for every occupation by least squares,
        including occupations that did not initially detect that pole.
        No Hamiltonian eigenenergy or theory spectrum is used for pole selection.

        The returned ``reconstructed_local`` is the finite-time FFT of the fitted
        complex returns, evaluated with the same window, padding, sign convention,
        and measured-A(0) normalization as ``spectrum.measured_local``. The
        continuous pole weights are returned separately as ``pole_DOS_weights``.
        For a complete basis their ideal value is the trace of each spectral
        projector, so a degenerate pole has its multiplicity as its DOS weight.
        The DOS uses the linear sum of A_i(0)-normalized complex amplitudes; the
        imaginary part and the sum of amplitude magnitudes are kept as diagnostics.

        Defaults reproduce the exploratory notebook algorithm. They are analysis
        choices rather than calibrated confidence levels. ``rank_sweep_extra=None``
        keeps the original full algebraic sweep. Setting it to 2 tests only through
        ``estimated_signal_rank + 2``, which is sufficient for a three-rank
        persistence requirement. ``match_decay=False`` tracks poles by frequency
        only and leaves decay as a fitted-pole diagnostic.
        """

        A = np.asarray(reconstruction.A, dtype=complex)
        time_us = np.asarray(spectrum.time_us, dtype=float)
        occupations = [tuple(occupation) for occupation in reconstruction.occupations]
        final_occupations = [tuple(occupation) for occupation in reconstruction.get(
            "final_occupations", occupations)]
        diagonal = np.asarray([initial == final for initial, final in zip(occupations, final_occupations)])
        if requested_max_modes is None:
            requested_max_modes = len(spectrum.fock_basis)
        
        
        
        ################################################################################
        ####   A Bunch of safety checks           ######################################
        ################################################################################
        if A.ndim != 2 or A.shape[0] != len(occupations) or A.shape[1] != len(time_us):
            raise ValueError("reconstruction.A must have shape (occupation, time point)")
        if len(time_us) < 5:
            raise ValueError("Matrix Pencil requires at least five time points")
        if not np.isclose(time_us[0], 0.):
            raise ValueError("Matrix Pencil DOS reconstruction requires the zero-time point")
        sample_time_us = time_us[1] - time_us[0]
        if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
            raise ValueError("Matrix Pencil requires uniformly spaced time points")
        if not isinstance(requested_max_modes, (int, np.integer)) or isinstance(requested_max_modes, (bool, np.bool_)) or requested_max_modes < 1:
            raise ValueError("requested_max_modes must be a positive integer")
        if not isinstance(minimum_consecutive_ranks, (int, np.integer)) or isinstance(minimum_consecutive_ranks, (bool, np.bool_)) or minimum_consecutive_ranks < 1:
            raise ValueError("minimum_consecutive_ranks must be a positive integer")
        if not isinstance(minimum_supporting_rows, (int, np.integer)) or isinstance(minimum_supporting_rows, (bool, np.bool_)) or minimum_supporting_rows < 1:
            raise ValueError("minimum_supporting_rows must be a positive integer")
        if pencil_length is None:
            pencil_length = len(time_us) // 2
        if not isinstance(pencil_length, (int, np.integer)) or isinstance(pencil_length, (bool, np.bool_)) or pencil_length < 1 or pencil_length >= len(time_us):
            raise ValueError("pencil_length must be an integer between 1 and sample_count - 1")
        maximum_algebraic_rank = min(requested_max_modes, pencil_length, len(time_us) - pencil_length)
        if maximum_algebraic_rank < minimum_consecutive_ranks:
            raise ValueError(f"this Matrix-Pencil configuration permits at most {maximum_algebraic_rank} ranks, fewer than minimum_consecutive_ranks={minimum_consecutive_ranks}")
        if rank_sweep_extra is not None and (not isinstance(rank_sweep_extra, (int, np.integer)) or isinstance(rank_sweep_extra, (bool, np.bool_)) or rank_sweep_extra < 0):
            raise ValueError("rank_sweep_extra must be None or a nonnegative integer")
        if not np.isfinite(track_frequency_tolerance_bins) or track_frequency_tolerance_bins <= 0.:
            raise ValueError("track_frequency_tolerance_bins must be finite and positive")
        if merge_frequency_tolerance_bins is None:
            merge_frequency_tolerance_bins = track_frequency_tolerance_bins
        if dedup_frequency_tolerance_bins is None:
            dedup_frequency_tolerance_bins = track_frequency_tolerance_bins
        if not np.isfinite(merge_frequency_tolerance_bins) or merge_frequency_tolerance_bins <= 0.:
            raise ValueError("merge_frequency_tolerance_bins must be finite and positive")
        if not np.isfinite(dedup_frequency_tolerance_bins) or dedup_frequency_tolerance_bins <= 0.:
            raise ValueError("dedup_frequency_tolerance_bins must be finite and positive")
        if not np.isfinite(numerical_floor) or numerical_floor <= 0.:
            raise ValueError("numerical_floor must be finite and positive")
        row_normalization = np.asarray([A[row, 0] if diagonal[row] else 1. for row in range(len(A))])[:, None]
        if np.any(np.abs(row_normalization) <= numerical_floor):
            raise ValueError("Matrix Pencil DOS reconstruction requires nonzero A_i(0) for every occupation")
        normalized_A = A / row_normalization
        if not np.isfinite(noise_singular_value_factor) or noise_singular_value_factor <= 0.:
            raise ValueError("noise_singular_value_factor must be finite and positive")
        if not np.isfinite(minimum_pole_radius) or not np.isfinite(maximum_pole_radius) or minimum_pole_radius <= 0. or maximum_pole_radius <= minimum_pole_radius:
            raise ValueError("pole radii must satisfy 0 < minimum_pole_radius < maximum_pole_radius")
        if least_squares_rcond is not None and (not np.isfinite(least_squares_rcond) or least_squares_rcond < 0.):
            raise ValueError("least_squares_rcond must be None or finite and nonnegative")

        sample_count = len(time_us)
        sampling_frequency_MHz = 1. / sample_time_us
        nyquist_MHz = 0.5 * sampling_frequency_MHz
        fft_resolution_MHz = 1. / (sample_count * sample_time_us)
        track_frequency_tolerance_MHz = track_frequency_tolerance_bins * fft_resolution_MHz
        merge_frequency_tolerance_MHz = merge_frequency_tolerance_bins * fft_resolution_MHz
        dedup_frequency_tolerance_MHz = dedup_frequency_tolerance_bins * fft_resolution_MHz
        if track_decay_tolerance_per_us is None:
            track_decay_tolerance_per_us = 2. * np.pi * track_frequency_tolerance_MHz
        if dedup_decay_tolerance_per_us is None:
            dedup_decay_tolerance_per_us = track_decay_tolerance_per_us
        if not np.isfinite(track_decay_tolerance_per_us) or track_decay_tolerance_per_us <= 0.:
            raise ValueError("track_decay_tolerance_per_us must be finite and positive")
        if not np.isfinite(dedup_decay_tolerance_per_us) or dedup_decay_tolerance_per_us <= 0.:
            raise ValueError("dedup_decay_tolerance_per_us must be finite and positive")
        ################################################################################
        ####   Safety checks end                  ######################################
        ################################################################################
        
        def wrap_frequency(frequency_MHz):
            """
            Nested function to place estimated frequency within the first Nyquist zone.
            This is done by first add, and then get the quotient value, and then
            subtract the added value again.
            """
            return (np.asarray(frequency_MHz) + nyquist_MHz) % sampling_frequency_MHz - nyquist_MHz

        def frequency_distance(first_MHz, second_MHz):
            """
            Calculate thre frequency diffrence between first_MHz and second MHz.
            The first_MHz can be a list or an array, when the return is the 
            list of difference. 
            The function is used to compare the estimated frequency 
            from \Sigma_r^{-1} U_r^\dagger H_1 V_r and 
            \Sigma_{r-1}^{-1} U_{r-1}^\dagger H_1 V_{r-1}
            """
            return np.abs(wrap_frequency(np.asarray(first_MHz) - second_MHz))

        def circular_frequency_center(frequencies_MHz, weights):
            """
            Convert angle in to a circle, which is basically a quotient space on R1 with
            the sampling frequency. This is to avoid the pathologial average due to finite
            sampling frequency; e.g., For 100MHz f_s, -49 MHz and 49 MHz gives 0.
            
            The basic algorithm is first convert frequency into phase,
            calculate the vector on IQ plane, average the vector and then
            return the angle of the averaged vector.
            """
            frequencies_MHz = np.asarray(frequencies_MHz, dtype=float)
            weights = np.asarray(weights, dtype=float)
            phases = 2. * np.pi * frequencies_MHz / sampling_frequency_MHz
            weighted_vector = np.sum(weights * np.exp(1j * phases))
            if np.abs(weighted_vector) < np.finfo(float).eps: 
                # If the vectors are all compensated to give zero, which is highly unlikely,
                # then gives the median of frequencies as an alternative.
                return float(wrap_frequency(np.median(frequencies_MHz)))
            return float(wrap_frequency(np.angle(weighted_vector) * sampling_frequency_MHz / (2. * np.pi)))

        row_candidates = []
        row_diagnostics = []
        #--- A. Row MPM iteraction initiation--------------------------------------------------
        for row_index, row in enumerate(normalized_A):
            trace_analysis = EncodingHamiltonianSpectroscopyExperiment.analyze_matrix_pencil_trace(
                row,
                time_us,
                requested_max_modes=requested_max_modes,
                pencil_length=pencil_length,
                minimum_consecutive_ranks=minimum_consecutive_ranks,
                track_frequency_tolerance_bins=track_frequency_tolerance_bins,
                dedup_frequency_tolerance_bins=dedup_frequency_tolerance_bins,
                track_decay_tolerance_per_us=track_decay_tolerance_per_us,
                dedup_decay_tolerance_per_us=dedup_decay_tolerance_per_us,
                match_decay=match_decay,
                numerical_floor=numerical_floor,
                noise_singular_value_factor=noise_singular_value_factor,
                minimum_pole_radius=minimum_pole_radius,
                maximum_pole_radius=maximum_pole_radius,
                require_early_start=require_early_start,
                rank_sweep_extra=rank_sweep_extra,
                clip_growth=clip_growth,
                least_squares_rcond=least_squares_rcond,
                store_rank_sweeps=store_rank_sweeps)

            for candidate in trace_analysis.candidates:
                candidate.row_index = row_index
                candidate.occupation = occupations[row_index]
                row_candidates.append(candidate)
            diagnostic = trace_analysis.diagnostic
            diagnostic.occupation = occupations[row_index]
            row_diagnostics.append(diagnostic)
        #--- B. Sortitng and MPM iteraction initiation--------------------------------------------------
        #--- 1. Order row_candidates in the order of confidence and then merge them
        #       when the frequency distance is smaller than merge_frequency_tolerance_MHz
        ordered_candidates = sorted(row_candidates, 
                                    key=lambda candidate: -candidate.confidence)
        clusters = []
        for candidate in ordered_candidates:
            compatible_clusters = []
            for cluster_index, cluster in enumerate(clusters):
                existing_rows = {member.row_index for member in cluster.members}
                if candidate.row_index in existing_rows:
                    continue
                distance_MHz = frequency_distance(candidate.frequency_MHz, 
                                                  cluster.frequency_MHz)
                if distance_MHz <= merge_frequency_tolerance_MHz:
                    compatible_clusters.append((distance_MHz, cluster_index))
            if not compatible_clusters:
                clusters.append(AttrDict(dict(frequency_MHz=candidate.frequency_MHz, 
                                              members=[candidate])))
                continue
            _, nearest_cluster_index = min(compatible_clusters)
            cluster = clusters[nearest_cluster_index]
            cluster.members.append(candidate)
            cluster.frequency_MHz = circular_frequency_center([member.frequency_MHz for member in cluster.members], 
                                                              [member.confidence for member in cluster.members])

        #--- 2. Discard candidates which appeared less than `minimum_supporting_rows`; 
        #       e.g. if if `minimum_supporting_rows` = 2 and the pole has appeard in only one row, it is discarded
        merged_candidates = []
        for cluster in clusters:
            members = cluster.members
            supporting_rows = sorted({member.row_index for member in members})
            if len(supporting_rows) < minimum_supporting_rows:
                continue
            member_weights = np.asarray([member.confidence for member in members])
            frequency_MHz = circular_frequency_center([member.frequency_MHz for member in members], member_weights)
            frequency_scatter_MHz = float(np.max([frequency_distance(member.frequency_MHz, frequency_MHz) for member in members]))
            decay_values = np.asarray([member.decay_per_us for member in members])
            raw_decay_per_us = float(np.median(decay_values))
            decay_per_us = max(0., raw_decay_per_us) if clip_growth else raw_decay_per_us
            rank_spans = np.asarray([member.rank_span for member in members])
            confidence = len(supporting_rows) * np.median(rank_spans) / (1. + frequency_scatter_MHz / merge_frequency_tolerance_MHz)
            merged_candidates.append(AttrDict(dict(frequency_MHz=frequency_MHz,
                                                   raw_decay_per_us=raw_decay_per_us,
                                                   decay_per_us=decay_per_us,
                                                   implied_growth=raw_decay_per_us < 0.,
                                                   supporting_rows=supporting_rows,
                                                   supporting_occupations=[occupations[row] for row in supporting_rows],
                                                   median_rank_span=float(np.median(rank_spans)),
                                                   frequency_scatter_MHz=frequency_scatter_MHz,
                                                   decay_scatter_per_us=float(np.max(np.abs(decay_values - raw_decay_per_us))),
                                                   confidence=float(confidence),
                                                   members=members)))
            
        #--- 3. Select upto `requested_max_modes`
        merged_candidates.sort(key=lambda candidate: (-candidate.confidence, -len(candidate.supporting_rows), candidate.frequency_scatter_MHz))
        selected_candidates = merged_candidates[:min(requested_max_modes, sample_count - 1)]
        selected_candidates.sort(key=lambda candidate: candidate.frequency_MHz)
        if not selected_candidates:
            raise RuntimeError("no stable rowwise Matrix-Pencil candidates were found")

        
        #--- 4. Extract amplitude of each pole using lstsq, which estimates x for A x = b with least square difference.
        selected_frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in selected_candidates])
        selected_decay_per_us = np.asarray([candidate.decay_per_us for candidate in selected_candidates])
        shared_poles = np.exp((-selected_decay_per_us - 2j * np.pi * selected_frequencies_MHz) * sample_time_us) # The list of poles; [z_1, z_2, ..., z_K]
        
        sample_index = np.arange(sample_count) #The arange upto N-1, where N is the number of time samples
        design = shared_poles[None, :] ** sample_index[:, None] #broadcast z into row, time into column, to make (transposed) vandermonde matrix. Shape: [N, K]
        normalized_amplitudes = np.zeros((len(occupations), len(selected_candidates)), dtype=complex)
        normalized_fitted_return = np.zeros_like(normalized_A)
         
        for row_index, normalized_row in enumerate(normalized_A):
            row_amplitudes, _, _, _ = np.linalg.lstsq(design, 
                                                      normalized_row,
                                                      rcond=least_squares_rcond) #Finds design * x = normalized_row. The solution is the amplitude for each row.
            normalized_amplitudes[row_index] = row_amplitudes
            normalized_fitted_return[row_index] = design @ row_amplitudes #Finds design * row_amp = fitted row
            
        
        amplitudes = normalized_amplitudes * row_normalization
        fitted_return = normalized_fitted_return * row_normalization
        residual = A - fitted_return
        residual_norm_by_row = np.linalg.norm(residual, axis=1)
        signal_norm_by_row = np.linalg.norm(A, axis=1)
        relative_residual_by_row = residual_norm_by_row / np.maximum(signal_norm_by_row, np.finfo(float).eps)
        relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(A))

        #--- Below are redoing fft; but I think the below can be deleted. The dependency got too complicated, so i am leaving it as is.
        fft_window = spectrum.get("fft_window", "raw")
        windows = {"raw": np.ones, 
                   "hann": np.hanning, 
                   "hamming": np.hamming, 
                   "blackman": np.blackman}
        if fft_window not in windows:
            raise ValueError("Matrix-Pencil display requires a supported spectrum.fft_window")
        zero_padding = spectrum.get("zero_padding", None)
        if not isinstance(zero_padding, (int, np.integer)) or zero_padding < 1:
            raise ValueError("Matrix Pencil requires an unmerged spectrum with integer zero_padding")
        n_fft = zero_padding * sample_count
        energy_MHz = np.asarray(spectrum.energy_MHz)
        expected_energy_MHz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=sample_time_us))
        if energy_MHz.shape != expected_energy_MHz.shape or not np.allclose(energy_MHz, expected_energy_MHz):
            raise ValueError("Matrix Pencil requires the original uniform FFT energy grid")
        window = windows[fft_window](sample_count)
        fft_scale = n_fft / np.sum(window)
        normalization = np.asarray(spectrum.fft_normalization)[:, None]
        reconstructed_local = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(fitted_return * window, n=n_fft, axis=1), axes=1))
        reconstructed_local /= normalization
        reconstructed = np.sum(reconstructed_local, axis=0)
        spectral_amplitudes = np.where(diagonal[:, None], normalized_amplitudes, amplitudes)
        pole_local_weights = np.real(spectral_amplitudes)
        pole_complex_DOS_weights = np.sum(spectral_amplitudes, axis=0)
        pole_DOS_weights = np.real(pole_complex_DOS_weights)
        pole_DOS_imaginary_weights = np.imag(pole_complex_DOS_weights)
        pole_local_magnitude_weights = np.abs(spectral_amplitudes)
        pole_amplitude_magnitude_sums = np.sum(pole_local_magnitude_weights, axis=0)
        row_weight_sums = np.sum(pole_local_weights, axis=1)
        total_DOS_weight = float(np.sum(pole_DOS_weights))
        coherent_trace_weights = np.abs(pole_complex_DOS_weights)
        supporting_row_counts = np.asarray([len(candidate.supporting_rows) for candidate in selected_candidates])

        settings = AttrDict(dict(requested_max_modes=int(requested_max_modes),
                                 pencil_length=int(pencil_length),
                                 minimum_consecutive_ranks=int(minimum_consecutive_ranks),
                                 minimum_supporting_rows=int(minimum_supporting_rows),
                                 track_frequency_tolerance_bins=float(track_frequency_tolerance_bins),
                                 merge_frequency_tolerance_bins=float(merge_frequency_tolerance_bins),
                                 dedup_frequency_tolerance_bins=float(dedup_frequency_tolerance_bins),
                                 track_frequency_tolerance_MHz=float(track_frequency_tolerance_MHz),
                                 merge_frequency_tolerance_MHz=float(merge_frequency_tolerance_MHz),
                                 dedup_frequency_tolerance_MHz=float(dedup_frequency_tolerance_MHz),
                                 track_decay_tolerance_per_us=float(track_decay_tolerance_per_us),
                                 dedup_decay_tolerance_per_us=float(dedup_decay_tolerance_per_us),
                                 match_decay=bool(match_decay),
                                 numerical_floor=float(numerical_floor),
                                 noise_singular_value_factor=float(noise_singular_value_factor),
                                 minimum_pole_radius=float(minimum_pole_radius),
                                 maximum_pole_radius=float(maximum_pole_radius),
                                 require_early_start=bool(require_early_start),
                                 rank_sweep_extra=rank_sweep_extra,
                                 clip_growth=bool(clip_growth),
                                 least_squares_rcond=least_squares_rcond,
                                 fft_window=fft_window,
                                 zero_padding=int(zero_padding)))
        modes = AttrDict(dict(frequencies_MHz=selected_frequencies_MHz,
                              decay_per_us=selected_decay_per_us,
                              poles=shared_poles,
                              supporting_row_counts=supporting_row_counts,
                              supporting_rows=[candidate.supporting_rows for candidate in selected_candidates],
                              supporting_occupations=[candidate.supporting_occupations for candidate in selected_candidates],
                              local_complex_amplitudes=spectral_amplitudes,
                              local_weights=pole_local_weights,
                              local_magnitude_weights=pole_local_magnitude_weights,
                              complex_DOS_weights=pole_complex_DOS_weights,
                              DOS_weights=pole_DOS_weights,
                              DOS_imaginary_weights=pole_DOS_imaginary_weights,
                              amplitude_magnitude_sums=pole_amplitude_magnitude_sums,
                              row_weight_sums=row_weight_sums,
                              total_DOS_weight=total_DOS_weight,
                              coherent_trace_weights=coherent_trace_weights))
        fit = AttrDict(dict(amplitudes=amplitudes,
                            normalized_amplitudes=normalized_amplitudes,
                            fitted_return=fitted_return,
                            normalized_fitted_return=normalized_fitted_return,
                            residual=residual,
                            relative_residual=relative_residual,
                            relative_residual_by_row=relative_residual_by_row,
                            design_condition_number=float(np.linalg.cond(design))))
        spectra = AttrDict(dict(energy_MHz=energy_MHz,
                                measured_local=np.asarray(spectrum.measured_local),
                                reconstructed_local=reconstructed_local,
                                measured=np.asarray(spectrum.measured),
                                reconstructed=reconstructed,
                                complete_basis=bool(spectrum.complete_basis)))
        return AttrDict(dict(method="matrix_pencil",
                             occupations=occupations,
                             row_normalization=row_normalization[:, 0],
                             settings=settings,
                             sampling=AttrDict(dict(time_us=time_us,
                                                    sample_time_us=sample_time_us,
                                                    sampling_frequency_MHz=sampling_frequency_MHz,
                                                    nyquist_MHz=nyquist_MHz,
                                                    fft_resolution_MHz=fft_resolution_MHz,
                                                    frequency_branch_note=f"frequencies are principal aliases modulo {sampling_frequency_MHz:.6g} MHz")),
                             modes=modes,
                             fit=fit,
                             spectra=spectra,
                             candidates=AttrDict(dict(per_row=row_candidates,
                                                      merged=merged_candidates,
                                                      selected=selected_candidates)),
                             row_diagnostics=row_diagnostics,
                             selected_frequencies_MHz=selected_frequencies_MHz,
                             selected_decay_per_us=selected_decay_per_us,
                             selected_candidates=selected_candidates,
                             amplitudes=amplitudes,
                             normalized_amplitudes=normalized_amplitudes,
                             fitted_return=fitted_return,
                             normalized_fitted_return=normalized_fitted_return,
                             residual=residual,
                             relative_residual=relative_residual,
                             relative_residual_by_row=relative_residual_by_row,
                             design_condition_number=float(np.linalg.cond(design)),
                             energy_MHz=energy_MHz,
                             reconstructed_local=reconstructed_local,
                             reconstructed=reconstructed,
                             pole_local_weights=pole_local_weights,
                             pole_local_magnitude_weights=pole_local_magnitude_weights,
                             pole_complex_DOS_weights=pole_complex_DOS_weights,
                             pole_DOS_weights=pole_DOS_weights,
                             pole_DOS_imaginary_weights=pole_DOS_imaginary_weights,
                             pole_amplitude_magnitude_sums=pole_amplitude_magnitude_sums,
                             row_weight_sums=row_weight_sums,
                             total_DOS_weight=total_DOS_weight,
                             complete_basis=bool(spectrum.complete_basis)))

    @staticmethod
    def analyze_matrix_pencil_trace(trace,
                                    time_us,
                                    requested_max_modes=None,
                                    pencil_length=None,
                                    minimum_consecutive_ranks=3,
                                    track_frequency_tolerance_bins=1.5,
                                    dedup_frequency_tolerance_bins=None,
                                    track_decay_tolerance_per_us=None,
                                    dedup_decay_tolerance_per_us=None,
                                    match_decay=True,
                                    numerical_floor=1e-10,
                                    noise_singular_value_factor=2.858,
                                    minimum_pole_radius=0.2,
                                    maximum_pole_radius=1.05,
                                    require_early_start=True,
                                    rank_sweep_extra=None,
                                    clip_growth=True,
                                    least_squares_rcond=None,
                                    store_rank_sweeps=False):
        """
        Apply the rowwise Matrix-Pencil analysis to one complex time trace.

        This is the independent analysis performed inside each iteration of
        :meth:`analyze_matrix_pencil`: Hankel construction, SVD and rank sweep,
        pole-history tracking, stability filtering, within-trace deduplication,
        and a final least-squares amplitude fit. It requires only one complex
        trace and uniformly spaced time points; it does not require a spectrum,
        reconstruction, experiment, or calculated Hamiltonian.

        The input is divided by ``trace[0]`` for conditioning. The returned
        ``normalized_amplitudes`` fit that normalized trace, while
        ``amplitudes`` and ``fitted_return`` are restored to the original
        input scale. Therefore, for

            trace = sum_i A_i(t) / A_i(0),

        the ideal real parts of ``amplitudes`` are the delta-functional DOS
        weights, including degeneracy multiplicities.
        """

        trace = np.asarray(trace, dtype=complex)
        time_us = np.asarray(time_us, dtype=float)
        if trace.ndim != 1 or time_us.ndim != 1 or len(trace) != len(time_us):
            raise ValueError("trace and time_us must be one-dimensional arrays of equal length")
        if len(time_us) < 5:
            raise ValueError("Matrix Pencil requires at least five time points")
        if not np.isclose(time_us[0], 0.):
            raise ValueError("Matrix Pencil trace analysis requires the zero-time point")
        sample_time_us = time_us[1] - time_us[0]
        if sample_time_us <= 0. or not np.allclose(np.diff(time_us), sample_time_us):
            raise ValueError("Matrix Pencil requires uniformly spaced time points")
        if not np.isfinite(numerical_floor) or numerical_floor <= 0.:
            raise ValueError("numerical_floor must be finite and positive")
        initial_return = trace[0]
        if np.abs(initial_return) <= numerical_floor:
            initial_return = 1.
        normalized_return = trace / initial_return

        if pencil_length is None:
            pencil_length = len(time_us) // 2
        if not isinstance(pencil_length, (int, np.integer)) or isinstance(pencil_length, (bool, np.bool_)) or pencil_length < 1 or pencil_length >= len(time_us):
            raise ValueError("pencil_length must be an integer between 1 and sample_count - 1")
        if requested_max_modes is None:
            requested_max_modes = min(pencil_length, len(time_us) - pencil_length)
        if not isinstance(requested_max_modes, (int, np.integer)) or isinstance(requested_max_modes, (bool, np.bool_)) or requested_max_modes < 1:
            raise ValueError("requested_max_modes must be a positive integer")
        if not isinstance(minimum_consecutive_ranks, (int, np.integer)) or isinstance(minimum_consecutive_ranks, (bool, np.bool_)) or minimum_consecutive_ranks < 1:
            raise ValueError("minimum_consecutive_ranks must be a positive integer")
        maximum_algebraic_rank = min(requested_max_modes, pencil_length, len(time_us) - pencil_length)
        if maximum_algebraic_rank < minimum_consecutive_ranks:
            raise ValueError(f"this Matrix-Pencil configuration permits at most {maximum_algebraic_rank} ranks, fewer than minimum_consecutive_ranks={minimum_consecutive_ranks}")
        if rank_sweep_extra is not None and (not isinstance(rank_sweep_extra, (int, np.integer)) or isinstance(rank_sweep_extra, (bool, np.bool_)) or rank_sweep_extra < 0):
            raise ValueError("rank_sweep_extra must be None or a nonnegative integer")
        if not np.isfinite(track_frequency_tolerance_bins) or track_frequency_tolerance_bins <= 0.:
            raise ValueError("track_frequency_tolerance_bins must be finite and positive")
        if dedup_frequency_tolerance_bins is None:
            dedup_frequency_tolerance_bins = track_frequency_tolerance_bins
        if not np.isfinite(dedup_frequency_tolerance_bins) or dedup_frequency_tolerance_bins <= 0.:
            raise ValueError("dedup_frequency_tolerance_bins must be finite and positive")
        if not np.isfinite(noise_singular_value_factor) or noise_singular_value_factor <= 0.:
            raise ValueError("noise_singular_value_factor must be finite and positive")
        if not np.isfinite(minimum_pole_radius) or not np.isfinite(maximum_pole_radius) or minimum_pole_radius <= 0. or maximum_pole_radius <= minimum_pole_radius:
            raise ValueError("pole radii must satisfy 0 < minimum_pole_radius < maximum_pole_radius")
        if least_squares_rcond is not None and (not np.isfinite(least_squares_rcond) or least_squares_rcond < 0.):
            raise ValueError("least_squares_rcond must be None or finite and nonnegative")

        sample_count = len(time_us)
        sampling_frequency_MHz = 1. / sample_time_us
        nyquist_MHz = 0.5 * sampling_frequency_MHz
        fft_resolution_MHz = 1. / (sample_count * sample_time_us)
        track_frequency_tolerance_MHz = track_frequency_tolerance_bins * fft_resolution_MHz
        dedup_frequency_tolerance_MHz = dedup_frequency_tolerance_bins * fft_resolution_MHz
        if track_decay_tolerance_per_us is None:
            track_decay_tolerance_per_us = 2. * np.pi * track_frequency_tolerance_MHz
        if dedup_decay_tolerance_per_us is None:
            dedup_decay_tolerance_per_us = track_decay_tolerance_per_us
        if not np.isfinite(track_decay_tolerance_per_us) or track_decay_tolerance_per_us <= 0.:
            raise ValueError("track_decay_tolerance_per_us must be finite and positive")
        if not np.isfinite(dedup_decay_tolerance_per_us) or dedup_decay_tolerance_per_us <= 0.:
            raise ValueError("dedup_decay_tolerance_per_us must be finite and positive")

        def wrap_frequency(frequency_MHz):
            return (np.asarray(frequency_MHz) + nyquist_MHz) % sampling_frequency_MHz - nyquist_MHz

        def frequency_distance(first_MHz, second_MHz):
            return np.abs(wrap_frequency(np.asarray(first_MHz) - second_MHz))

        def circular_frequency_center(frequencies_MHz, weights):
            frequencies_MHz = np.asarray(frequencies_MHz, dtype=float)
            weights = np.asarray(weights, dtype=float)
            phases = 2. * np.pi * frequencies_MHz / sampling_frequency_MHz
            weighted_vector = np.sum(weights * np.exp(1j * phases))
            if np.abs(weighted_vector) < np.finfo(float).eps:
                return float(wrap_frequency(np.median(frequencies_MHz)))
            return float(wrap_frequency(np.angle(weighted_vector) * sampling_frequency_MHz / (2. * np.pi)))

        row_candidates = []
        row_diagnostics = []
        occupations = [("trace",)]
        #--- A. Row MPM iteraction initiation--------------------------------------------------
        for row_index, row in enumerate([normalized_return]):
            row_norm = np.linalg.norm(row)
            if row_norm <= numerical_floor: 
            #Disregard the row if there is no peak; the criteria is numerical_floor, which is adhoc
                row_diagnostics.append(AttrDict(dict(
                    occupation=occupations[row_index],
                    estimated_signal_rank=0,
                    maximum_rank=0, 
                    singular_values=np.array([]),
                    candidates=[])))
                continue
            
            
            scaled_row = row / row_norm
            
            #--- 1. Hankel Matrix Formation-----------------------------------------------------
            windows = sliding_window_view(scaled_row, pencil_length + 1) # primitive for Hankel 
            unshifted = windows[:, :-1] # H_0 by discarding the last columns
            shifted = windows[:, 1:] # H_1 by discarding the first columns
            
            #--- 2. Thin SVD of unshifted (H_0) matrix.
            #--- singular_values is 1d array. others are 2d array matrix.
            #--- Thesholding can be done in a different way.
            left_vectors, singular_values, right_vectors_h = np.linalg.svd(unshifted, 
                                                                           full_matrices=False)
            relative_singular_values = singular_values / singular_values[0]       
            singular_value_threshold = noise_singular_value_factor * np.median(singular_values) #as we do not know the singular value of noise, it is replaced by median value.
            
            #--- 3. Prepare iterative MPM after setting up signal rank, numerical rank.
            estimated_signal_rank = max(1, int(np.count_nonzero(singular_values > singular_value_threshold)))
            numerical_rank = int(np.count_nonzero(relative_singular_values > numerical_floor))
            maximum_rank = min(requested_max_modes, unshifted.shape[0], unshifted.shape[1], numerical_rank)
            if rank_sweep_extra is not None:
                maximum_rank = min(maximum_rank, estimated_signal_rank + rank_sweep_extra)
        
            #--- 4. Execute MPM by sweeping rank from 1 to maximum_rank
            rank_solutions = []
            for rank in range(1, maximum_rank + 1):
                left = left_vectors[:, :rank]
                right = right_vectors_h[:rank].conj().T
                shifted_reduced = left.conj().T @ shifted @ right
                # np.linalg.solve(A, B) does A^{-1} B
                reduced_pencil = np.linalg.solve(np.diag(singular_values[:rank]), shifted_reduced)
                poles = np.linalg.eigvals(reduced_pencil)
                pole_radii = np.abs(poles)
                # I am not really sure if this masking is necessary. There are already tons of
                # other validation steps, including signal rank and things like that
                # plus, minimum/maximum_pole_radius is somewhat ad hoc
                valid = np.isfinite(poles) & np.isfinite(pole_radii) & (pole_radii >= minimum_pole_radius) & (pole_radii <= maximum_pole_radius)
                poles = poles[valid]
                pole_radii = pole_radii[valid]
                frequencies_MHz = -np.angle(poles) / (2. * np.pi * sample_time_us)
                decay_per_us = -np.log(pole_radii) / sample_time_us
                order = np.argsort(frequencies_MHz) # sort in an ascending order
                rank_solutions.append(AttrDict(dict(rank=rank,
                                                    frequencies_MHz=frequencies_MHz[order],
                                                    decay_per_us=decay_per_us[order],
                                                    pole_radii=pole_radii[order])))
            #--- 5. Classification of poles; the poles are grouped when they are within tolerance
            pole_histories = []
            first_solution = rank_solutions[0]
            for pole_index, frequency_MHz in enumerate(first_solution.frequencies_MHz):
                pole_histories.append(AttrDict(dict(ranks=[first_solution.rank],
                                                     frequencies_MHz=[float(frequency_MHz)],
                                                     decay_per_us=[float(first_solution.decay_per_us[pole_index])],
                                                     pole_radii=[float(first_solution.pole_radii[pole_index])])))

            for solution in rank_solutions[1:]: #Note: solution is AttrDict
                current_rank = solution.rank
                previous_rank = current_rank - 1

                # Record every compatible (previous history, current pole) pair and its distance.
                candidate_matches = []
                for pole_history in pole_histories:
                    # This loop is to pick poles that can be thought of as the same pole
                    # found previously. If there is a pole that is "same", the AttrDict of
                    # distance, pole_history, and index is stored.
                    # The reason of collecting candidate_matches first is to consider
                    # all possible combinations of distance and find the best track
                    last_rank_where_pole_was_found = pole_history.ranks[-1]
                    if last_rank_where_pole_was_found != previous_rank:
                        continue

                    previous_frequency_MHz = pole_history.frequencies_MHz[-1]
                    previous_decay_per_us = pole_history.decay_per_us[-1]
                    for current_pole_index, current_frequency_MHz in enumerate(solution.frequencies_MHz):
                        # This loop is to pick pole index of the frequency
                        # that appeared previously.
                        frequency_difference_MHz = frequency_distance(current_frequency_MHz, 
                                                                      previous_frequency_MHz)
                        decay_difference_per_us = np.abs(solution.decay_per_us[current_pole_index] - previous_decay_per_us)
                        frequency_is_close = frequency_difference_MHz <= track_frequency_tolerance_MHz
                        decay_is_close = decay_difference_per_us <= track_decay_tolerance_per_us
                        if not frequency_is_close or (match_decay and not decay_is_close):
                            continue

                        match_distance = frequency_difference_MHz / track_frequency_tolerance_MHz
                        if match_decay:
                            match_distance += decay_difference_per_us / track_decay_tolerance_per_us
                        candidate_matches.append(AttrDict(dict(distance=match_distance,
                                                               pole_history=pole_history,
                                                               current_pole_index=current_pole_index)))

                candidate_matches.sort(key=lambda element: element.distance) #sort candidates in the order of match_distance
                assigned_current_pole_indices = set()
                for candidate_match in candidate_matches:
                    pole_history = candidate_match.pole_history #Note: dictionary variable name also acts as a reference in C++
                    current_pole_index = candidate_match.current_pole_index
                    
                    history_already_extended = pole_history.ranks[-1] == current_rank
                    pole_already_assigned = current_pole_index in assigned_current_pole_indices
                    
                    if history_already_extended or pole_already_assigned:
                        continue
                    pole_history.ranks.append(current_rank)
                    pole_history.frequencies_MHz.append(float(solution.frequencies_MHz[current_pole_index]))
                    pole_history.decay_per_us.append(float(solution.decay_per_us[current_pole_index]))
                    pole_history.pole_radii.append(float(solution.pole_radii[current_pole_index]))
                    assigned_current_pole_indices.add(current_pole_index)

                # A current pole that matched no existing history starts a new history at this rank.
                for current_pole_index, current_frequency_MHz in enumerate(solution.frequencies_MHz):
                    if current_pole_index in assigned_current_pole_indices:
                        continue
                    pole_histories.append(AttrDict(dict(ranks=[current_rank],
                                                         frequencies_MHz=[float(current_frequency_MHz)],
                                                         decay_per_us=[float(solution.decay_per_us[current_pole_index])],
                                                         pole_radii=[float(solution.pole_radii[current_pole_index])])))
            
            #--- 6. Filter poles that did not appear consecutively or appeared after signal rank; this is to rule out poles from noise
            candidates = []
            for pole_history in pole_histories:
                rank_span = len(pole_history.ranks)
                #below only retains poles with minimum_consecutive ranks
                if rank_span < minimum_consecutive_ranks:
                    continue
                #below only retains poles appears before signal rank
                if require_early_start and pole_history.ranks[0] > estimated_signal_rank:
                    continue
                frequencies_MHz = np.asarray(pole_history.frequencies_MHz)
                decay_per_us = np.asarray(pole_history.decay_per_us)
                frequency_MHz = circular_frequency_center(frequencies_MHz, 
                                                          np.ones(rank_span))
                frequency_scatter_MHz = float(np.max(frequency_distance(frequencies_MHz, 
                                                                        frequency_MHz)))
                median_decay_per_us = float(np.median(decay_per_us))
                decay_scatter_per_us = float(np.max(np.abs(decay_per_us - median_decay_per_us)))
                confidence_denominator = 1. + frequency_scatter_MHz / track_frequency_tolerance_MHz
                if match_decay:
                    confidence_denominator += decay_scatter_per_us / track_decay_tolerance_per_us
                candidates.append(AttrDict(dict(row_index=row_index,
                                                occupation=occupations[row_index],
                                                frequency_MHz=frequency_MHz,
                                                decay_per_us=median_decay_per_us,
                                                pole_radius=float(np.median(pole_history.pole_radii)),
                                                first_rank=pole_history.ranks[0],
                                                last_rank=pole_history.ranks[-1],
                                                rank_span=rank_span,
                                                frequency_scatter_MHz=frequency_scatter_MHz,
                                                decay_scatter_per_us=decay_scatter_per_us,
                                                confidence=float(rank_span / confidence_denominator))))
            
            
            candidates.sort(key=lambda candidate: (-candidate.rank_span, #to sort large values first
                                                   candidate.frequency_scatter_MHz, 
                                                   candidate.decay_scatter_per_us))
            
            #--- 7. Deduplicate frequency poles from candidates; the criteria is dedup_frequency_tolerance_MHz, which is identical to the track_frequency_tolerance_MHz by defaultt
            unique_candidates = []
            for candidate in candidates:
                # This loop is to discard duplicated peaked
                duplicate = False
                for existing in unique_candidates:
                    duplicate = frequency_distance(candidate.frequency_MHz, 
                                                   existing.frequency_MHz) <= dedup_frequency_tolerance_MHz
                    if match_decay:
                        duplicate = duplicate and np.abs(candidate.decay_per_us - existing.decay_per_us) <= dedup_decay_tolerance_per_us
                    if duplicate:
                        break
                if not duplicate:
                    unique_candidates.append(candidate)
            row_candidates.extend(unique_candidates)
            diagnostic = AttrDict(dict(occupation=occupations[row_index],
                                       estimated_signal_rank=estimated_signal_rank,
                                       maximum_rank=maximum_rank,
                                       singular_values=singular_values,
                                       relative_singular_values=relative_singular_values,
                                       candidates=unique_candidates))
            if store_rank_sweeps:
                diagnostic.rank_solutions = rank_solutions
                diagnostic.tracks = pole_histories
            row_diagnostics.append(diagnostic)
            
        selected_candidates = row_candidates[:min(requested_max_modes, sample_count - 1)]
        selected_candidates.sort(key=lambda candidate: candidate.frequency_MHz)
        selected_frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in selected_candidates])
        raw_decay_per_us = np.asarray([candidate.decay_per_us for candidate in selected_candidates])
        selected_decay_per_us = np.maximum(raw_decay_per_us, 0.) if clip_growth else raw_decay_per_us.copy()
        shared_poles = np.exp((-selected_decay_per_us - 2j * np.pi * selected_frequencies_MHz) * sample_time_us)
        sample_index = np.arange(sample_count)
        design = shared_poles[None, :] ** sample_index[:, None]

        if len(selected_candidates):
            normalized_amplitudes, _, _, _ = np.linalg.lstsq(design,
                                                             normalized_return,
                                                             rcond=least_squares_rcond)
            normalized_fitted_return = design @ normalized_amplitudes
            design_condition_number = float(np.linalg.cond(design))
        else:
            normalized_amplitudes = np.array([], dtype=complex)
            normalized_fitted_return = np.zeros_like(normalized_return)
            design_condition_number = np.nan
        amplitudes = normalized_amplitudes * initial_return
        fitted_return = normalized_fitted_return * initial_return
        residual = trace - fitted_return
        relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(trace))
        diagnostic = row_diagnostics[0]

        return AttrDict(dict(method="matrix_pencil_trace",
                             trace=trace,
                             time_us=time_us,
                             initial_return=initial_return,
                             normalized_return=normalized_return,
                             candidates=row_candidates,
                             selected_candidates=selected_candidates,
                             selected_frequencies_MHz=selected_frequencies_MHz,
                             selected_raw_decay_per_us=raw_decay_per_us,
                             selected_decay_per_us=selected_decay_per_us,
                             poles=shared_poles,
                             normalized_amplitudes=normalized_amplitudes,
                             amplitudes=amplitudes,
                             DOS_weights=np.real(amplitudes),
                             DOS_imaginary_weights=np.imag(amplitudes),
                             amplitude_magnitudes=np.abs(amplitudes),
                             normalized_fitted_return=normalized_fitted_return,
                             fitted_return=fitted_return,
                             residual=residual,
                             relative_residual=relative_residual,
                             design_condition_number=design_condition_number,
                             diagnostic=diagnostic,
                             sampling=AttrDict(dict(sample_time_us=sample_time_us,
                                                    sampling_frequency_MHz=sampling_frequency_MHz,
                                                    nyquist_MHz=nyquist_MHz,
                                                    fft_resolution_MHz=fft_resolution_MHz,
                                                    frequency_branch_note=f"frequencies are principal aliases modulo {sampling_frequency_MHz:.6g} MHz")),
                             settings=AttrDict(dict(requested_max_modes=int(requested_max_modes),
                                                    pencil_length=int(pencil_length),
                                                    minimum_consecutive_ranks=int(minimum_consecutive_ranks),
                                                    track_frequency_tolerance_bins=float(track_frequency_tolerance_bins),
                                                    dedup_frequency_tolerance_bins=float(dedup_frequency_tolerance_bins),
                                                    track_frequency_tolerance_MHz=float(track_frequency_tolerance_MHz),
                                                    dedup_frequency_tolerance_MHz=float(dedup_frequency_tolerance_MHz),
                                                    track_decay_tolerance_per_us=float(track_decay_tolerance_per_us),
                                                    dedup_decay_tolerance_per_us=float(dedup_decay_tolerance_per_us),
                                                    match_decay=bool(match_decay),
                                                    numerical_floor=float(numerical_floor),
                                                    noise_singular_value_factor=float(noise_singular_value_factor),
                                                    minimum_pole_radius=float(minimum_pole_radius),
                                                    maximum_pole_radius=float(maximum_pole_radius),
                                                    require_early_start=bool(require_early_start),
                                                    rank_sweep_extra=rank_sweep_extra,
                                                    clip_growth=bool(clip_growth),
                                                    least_squares_rcond=least_squares_rcond))))


    def analyze_matrix_pencil_occupation(self,
                                         occupation,
                                         data=None,
                                         matrix_pencil=None,
                                         least_squares_rcond=None):
        """
        Refit one occupation using only the poles detected independently in that row.

        This is different from taking one row of the global Matrix-Pencil fit. The
        global fit uses every shared pole selected after candidates from all rows are
        merged. This wrapper instead uses ``row_diagnostics[row].candidates`` and
        therefore shows the Matrix-Pencil result supported by the selected occupation
        before the cross-row merge.
        """
        data = self.data if data is None else data
        if "reconstruction" not in data or "spectrum" not in data:
            raise ValueError("occupation Matrix-Pencil analysis requires analyzed spectroscopy data")
        if data.get("spectrum_only", False):
            raise ValueError("occupation Matrix Pencil requires the original occupation time traces")
        if matrix_pencil is None:
            matrix_pencil = data.get("matrix_pencil", None)
        if matrix_pencil is None:
            raise ValueError("Matrix-Pencil analysis is unavailable; analyze with spectrum_method='matrix_pencil'")

        occupations = [tuple(value) for value in data.reconstruction.occupations]
        if [tuple(value) for value in matrix_pencil.occupations] != occupations:
            raise ValueError("Matrix-Pencil occupations do not match the spectroscopy reconstruction")
        if isinstance(occupation, (int, np.integer)):
            row = int(occupation)
            if row < 0 or row >= len(occupations):
                raise IndexError("occupation row is outside the spectroscopy data")
        else:
            occupation = tuple(occupation)
            if occupation not in occupations:
                raise ValueError(f"{occupation} is not in the spectroscopy data")
            row = occupations.index(occupation)
        occupation = occupations[row]

        diagnostic = matrix_pencil.row_diagnostics[row]
        row_candidates = sorted(list(diagnostic.candidates), key=lambda candidate: candidate.frequency_MHz)
        frequencies_MHz = np.asarray([candidate.frequency_MHz for candidate in row_candidates], dtype=float)
        raw_decay_per_us = np.asarray([candidate.decay_per_us for candidate in row_candidates], dtype=float)
        if matrix_pencil.settings.clip_growth:
            decay_per_us = np.maximum(raw_decay_per_us, 0.)
        else:
            decay_per_us = raw_decay_per_us.copy()

        time_us = np.asarray(matrix_pencil.sampling.time_us, dtype=float)
        sample_time_us = float(matrix_pencil.sampling.sample_time_us)
        measured_return = np.asarray(data.reconstruction.A[row], dtype=complex)
        initial_return = matrix_pencil.row_normalization[row]
        normalized_return = measured_return / initial_return
        sample_index = np.arange(len(time_us))
        poles = np.exp((-decay_per_us - 2j * np.pi * frequencies_MHz) * sample_time_us)
        design = poles[None, :] ** sample_index[:, None]
        if least_squares_rcond is None:
            least_squares_rcond = matrix_pencil.settings.least_squares_rcond
        if len(row_candidates):
            normalized_amplitudes, _, _, _ = np.linalg.lstsq(design, normalized_return, rcond=least_squares_rcond)
            normalized_fitted_return = design @ normalized_amplitudes
            design_condition_number = float(np.linalg.cond(design))
        else:
            normalized_amplitudes = np.array([], dtype=complex)
            normalized_fitted_return = np.zeros_like(normalized_return)
            design_condition_number = np.nan
        amplitudes = normalized_amplitudes * initial_return
        fitted_return = normalized_fitted_return * initial_return
        residual = measured_return - fitted_return
        relative_residual = float(np.linalg.norm(residual) / np.linalg.norm(measured_return))

        windows = {"raw": np.ones, "hann": np.hanning, "hamming": np.hamming, "blackman": np.blackman}
        fft_window = matrix_pencil.settings.fft_window
        zero_padding = matrix_pencil.settings.zero_padding
        window = windows[fft_window](len(time_us))
        n_fft = zero_padding * len(time_us)
        fft_scale = n_fft / np.sum(window)
        reconstructed_spectrum = fft_scale * np.abs(np.fft.fftshift(np.fft.ifft(fitted_return * window, n=n_fft))) / data.spectrum.fft_normalization[row]
        is_diagonal = occupation == tuple(data.reconstruction.final_occupations[row])
        spectral_amplitudes = normalized_amplitudes if is_diagonal else amplitudes

        return AttrDict(dict(method="matrix_pencil_occupation",
                             row_index=row,
                             occupation=occupation,
                             diagnostic=diagnostic,
                             candidates=row_candidates,
                             time_us=time_us,
                             measured_return=measured_return,
                             normalized_return=normalized_return,
                             fitted_return=fitted_return,
                             normalized_fitted_return=normalized_fitted_return,
                             residual=residual,
                             relative_residual=relative_residual,
                             frequencies_MHz=frequencies_MHz,
                             raw_decay_per_us=raw_decay_per_us,
                             decay_per_us=decay_per_us,
                             poles=poles,
                             normalized_amplitudes=normalized_amplitudes,
                             local_weights=np.real(spectral_amplitudes),
                             local_magnitude_weights=np.abs(spectral_amplitudes),
                             design_condition_number=design_condition_number,
                             energy_MHz=np.asarray(data.spectrum.energy_MHz),
                             measured_spectrum=np.asarray(data.spectrum.measured_local[row]),
                             reconstructed_spectrum=reconstructed_spectrum))

    @staticmethod
    def merge_spectra(data):
        """
        This method is to join occupation rows from separately analyzed spectra on 
        one common energy grid.

        Each input must already contain ``reconstruction`` and ``spectrum`` 
        from ``analyze(stage='spectrum')``. The photon number, mode order, phase frame, 
        and FFT window must agree, but the record length, zero padding, and energy grids 
        may differ. A source Hamiltonian may differ from the first source only when its
        largest eigenenergy change is smaller than the worse of their physical FFT 
        resolutions. The first source defines the Hamiltonian used for all theory and 
        LDOS rows. Measured FFT magnitudes are linearly interpolated only over the shared 
        Nyquist range. Interpolation aligns coordinates; it does not improve a row's 
        physical resolution, which is retained in ``row_fft_resolution_MHz``.

        The returned object is spectrum-only because its rows do not share one complex time
        grid. It can be passed to ``display`` and to complete-basis level-statistics 
        analysis, but not to SFF or occupation time-trace displays.
        """
        sources = list(data) if isinstance(data, (list, tuple)) else [data]
        sources = [source.data if not isinstance(source, dict) and hasattr(source, "data") else source for source in sources]
        if len(sources) < 2:
            raise ValueError("merge_spectra requires at least two analyzed spectra")
        for source in sources:
            if "reconstruction" not in source or "spectrum" not in source:
                raise ValueError("every merge source must contain reconstruction and spectrum")
            if source.get("spectrum_only", False):
                raise ValueError("merge_spectra does not accept an already merged spectrum")

        reference = sources[0]
        reference_energies_MHz = np.asarray(reference.spectrum.energies_MHz)
        if "fock_basis" not in reference.spectrum or "basis_eigenstate_weights" not in reference.spectrum:
            raise ValueError("reanalyze the first spectrum with the current code before merging")
        reference_fock_basis = [tuple(occupation) for occupation in reference.spectrum.fock_basis]
        reference_basis_eigenstate_weights = np.asarray(reference.spectrum.basis_eigenstate_weights)
        if reference_basis_eigenstate_weights.shape != (len(reference_fock_basis), len(reference_energies_MHz)):
            raise ValueError("the first spectrum has inconsistent full-basis eigenstate weights")
        reference_weights_by_occupation = dict(zip(reference_fock_basis, reference_basis_eigenstate_weights))
        occupations = []
        measured_rows = []
        theory_rows = []
        row_fft_resolution_MHz = []
        row_physical_kerr_MHz = []
        hamiltonian_mismatch_MHz = []
        energy_steps_MHz = []
        energy_minima_MHz = []
        energy_maxima_MHz = []
        windows = {"raw": np.ones, 
                   "hann": np.hanning, 
                   "hamming": np.hamming, 
                   "blackman": np.blackman}
        # Collect the spectrum and its resolution information
        for source in sources:
            spectrum = source.spectrum
            if source.photon_number != reference.photon_number or list(source.mode_labels) != list(reference.mode_labels):
                raise ValueError("merged spectra must use the same photon number and mode order")
            if source.phase_frame != reference.phase_frame:
                raise ValueError("merged spectra must use the same physical phase frame")
            if spectrum.fft_window != reference.spectrum.fft_window:
                raise ValueError("merged spectra must use the same FFT window")
            source_energies_MHz = np.asarray(spectrum.energies_MHz)
            if source_energies_MHz.shape != reference_energies_MHz.shape:
                raise ValueError("merged spectra describe different Hilbert-space dimensions")
            energy_mismatch_MHz = float(np.max(np.abs(source_energies_MHz - reference_energies_MHz)))
            allowed_mismatch_MHz = max(float(spectrum.fft_resolution_MHz), float(reference.spectrum.fft_resolution_MHz))
            if energy_mismatch_MHz > allowed_mismatch_MHz:
                raise ValueError(f"source Hamiltonian differs by {energy_mismatch_MHz:.6g} MHz, larger than the {allowed_mismatch_MHz:.6g} MHz FFT resolution")
            hamiltonian_mismatch_MHz.append(energy_mismatch_MHz)

            source_occupations = [tuple(occupation) for occupation in source.reconstruction.occupations]
            if set(occupations).intersection(source_occupations):
                raise ValueError("merged spectra contain duplicate occupations")
            occupations.extend(source_occupations)
            energy_MHz = np.asarray(spectrum.energy_MHz)
            energy_steps_MHz.append(float(np.median(np.diff(energy_MHz))))
            energy_minima_MHz.append(float(energy_MHz[0]))
            energy_maxima_MHz.append(float(energy_MHz[-1]))
            row_fft_resolution_MHz.extend([float(spectrum.fft_resolution_MHz)] * len(source_occupations))
            row_physical_kerr_MHz.extend([float(spectrum.physical_kerr_MHz)] * len(source_occupations))
        # Choose the coarsest binning and range
        energy_step_MHz = max(energy_steps_MHz)
        energy_min_MHz = max(energy_minima_MHz)
        energy_max_MHz = min(energy_maxima_MHz)
        

        first_bin = int(np.ceil(energy_min_MHz / energy_step_MHz - 1e-10))
        last_bin = int(np.floor(energy_max_MHz / energy_step_MHz + 1e-10))
        energy_MHz = np.arange(first_bin, last_bin + 1, dtype=float) * energy_step_MHz
        if len(energy_MHz) < 2:
            raise ValueError("merged spectra have no usable common energy range")
        if np.any(reference_energies_MHz < energy_MHz[0]) or np.any(reference_energies_MHz > energy_MHz[-1]):
            raise ValueError("Hamiltonian eigenenergies lie outside the shared Nyquist range")
        eigenstate_weights = np.asarray([reference_weights_by_occupation[occupation] for occupation in occupations])

        for source in sources:
            spectrum = source.spectrum
            source_energy_MHz = np.asarray(spectrum.energy_MHz)
            measured_rows.extend([np.interp(energy_MHz, source_energy_MHz, row) for row in spectrum.measured_local])

            time_us = np.asarray(spectrum.time_us)
            window = windows[spectrum.fft_window](len(time_us))
            theory_phase = np.exp(-2j * np.pi * np.outer(reference_energies_MHz, time_us))
            source_weights = np.asarray([reference_weights_by_occupation[tuple(occupation)] for occupation in source.reconstruction.occupations])
            theory_A = source_weights @ theory_phase
            fourier_kernel = np.exp(2j * np.pi * np.outer(time_us, energy_MHz))
            theory_rows.extend(np.abs((theory_A * window) @ fourier_kernel) / np.sum(window))

        measured_local = np.asarray(measured_rows)
        theory_local = np.asarray(theory_rows)
        measured = np.sum(measured_local, axis=0)
        theory = np.sum(theory_local, axis=0)
        if np.max(theory) > 0.:
            theory_local *= np.max(measured) / np.max(theory)
            theory = np.sum(theory_local, axis=0)

        complete_basis = set(occupations) == set(reference_fock_basis)
        row_fft_resolution_MHz = np.asarray(row_fft_resolution_MHz)
        row_physical_kerr_MHz = np.asarray(row_physical_kerr_MHz)
        energy_limit_MHz = min(np.max(np.abs(energy_MHz)), max(0.6, 1.2 * np.max(np.abs(reference_energies_MHz))))
        spectrum = AttrDict(dict(
            energy_MHz=energy_MHz, 
            measured_local=measured_local, 
            theory_local=theory_local,
            measured=measured, 
            theory=theory, 
            energies_MHz=reference_energies_MHz,
            fock_basis=[list(occupation) for occupation in reference_fock_basis], 
            basis_eigenstate_weights=reference_basis_eigenstate_weights,
            eigenstate_weights=eigenstate_weights, 
            physical_kerr_MHz=float(reference.spectrum.physical_kerr_MHz),
            complete_basis=complete_basis, 
            energy_limit_MHz=energy_limit_MHz,
            fft_window=reference.spectrum.fft_window, 
            zero_padding=None,
            fft_resolution_MHz=float(np.max(row_fft_resolution_MHz)), 
            row_fft_resolution_MHz=row_fft_resolution_MHz,
            mixed_resolution=not np.allclose(row_fft_resolution_MHz, row_fft_resolution_MHz[0]),
            row_physical_kerr_MHz=row_physical_kerr_MHz,
            hamiltonian_mismatch_MHz=float(np.max(hamiltonian_mismatch_MHz)),
            mixed_hamiltonian=not np.isclose(np.max(hamiltonian_mismatch_MHz), 0.), 
            energy_grid_step_MHz=energy_step_MHz,
        ))
        return AttrDict(dict(
            reconstruction=AttrDict(dict(occupations=occupations)), 
            spectrum=spectrum,
            hardware=reference.hardware, 
            photon_number=reference.photon_number,
            detunings=reference.detunings, 
            mode_labels=reference.mode_labels,
            phase_frame=reference.phase_frame,
            source_phase_frames=[source.get("source_phase_frame", source.phase_frame) for source in sources],
            spectrum_only=True, # To disable SFF analysis
        ))

    def analyze_level_statistics(self,
                                 data=None,
                                 peak_prominence=None,
                                 peak_prominence_fraction=None,
                                 minimum_peak_distance_MHz=None,
                                 energy_limit_MHz=None):
        """Analyze measured DOS peaks and level spacings without imposing multiplicities.

        ``spectrum.measured`` is the sum of the occupation-resolved FFT magnitudes. Its
        normalization gives an ideal isolated one-level line height one, so each detected
        peak height is rounded independently to estimate its multiplicity. The rounded
        values are never rescaled or adjusted to make their sum equal the Hilbert-space
        dimension D; disagreement with D is retained as a measurement diagnostic.

        Exact degeneracy is determined only from the Hamiltonian eigenenergies. If an exact
        degeneracy is present, the result retains the raw zero gaps, the defined ratios, and
        the number of undefined 0/0 ratios. If the Hamiltonian is nondegenerate, experimental
        gap ratios are calculated only when all D measured peaks are resolved.
        """
        if data is None:
            data = self.data
        if "spectrum" not in data:
            raise ValueError("level statistics requires analyzed spectroscopy data")
        spectrum = data.spectrum
        if not spectrum.complete_basis:
            raise ValueError("level statistics requires the complete fixed-N occupation basis")
        window_resolution_factors = {"raw": 1., "hann": 2., "hamming": 2., "blackman": 3.}
        default_prominence_fractions = {"raw": 0.05, "hann": 0.05, "hamming": 0.05, "blackman": 0.05}
        if spectrum.fft_window not in window_resolution_factors:
            raise ValueError(f"unknown FFT window {spectrum.fft_window}")
        if peak_prominence_fraction is None:
            peak_prominence_fraction = default_prominence_fractions[spectrum.fft_window]
        if not np.isfinite(peak_prominence_fraction) or peak_prominence_fraction < 0. or peak_prominence_fraction >= 1.:
            raise ValueError("peak_prominence_fraction must be finite and in [0, 1)")

        energy_MHz = np.asarray(spectrum.energy_MHz, dtype=float)
        measured_DOS = np.asarray(spectrum.measured, dtype=float)
        theory_DOS = np.asarray(spectrum.theory, dtype=float)
        if energy_MHz.ndim != 1 or measured_DOS.shape != energy_MHz.shape or theory_DOS.shape != energy_MHz.shape:
            raise ValueError("measured DOS, theory DOS, and FFT energy axis must be one-dimensional and have the same shape")
        if not np.all(np.isfinite(energy_MHz)) or not np.all(np.isfinite(measured_DOS)) or not np.all(np.isfinite(theory_DOS)):
            raise ValueError("measured DOS, theory DOS, and FFT energy axis must be finite")

        if energy_limit_MHz is None:
            energy_limit_MHz = min(float(np.max(np.abs(energy_MHz))), float(spectrum.get("energy_limit_MHz", np.max(np.abs(energy_MHz)))))
        if not np.isfinite(energy_limit_MHz) or energy_limit_MHz <= 0.:
            raise ValueError("energy_limit_MHz must be finite and positive")
        inside_energy_limit = np.abs(energy_MHz) <= energy_limit_MHz
        energy_MHz = energy_MHz[inside_energy_limit]
        measured_DOS = measured_DOS[inside_energy_limit]
        theory_DOS = theory_DOS[inside_energy_limit]
        if len(energy_MHz) < 3:
            raise ValueError("the selected energy range has fewer than three FFT bins")

        energy_bin_widths_MHz = np.diff(energy_MHz)
        energy_step_MHz = float(energy_bin_widths_MHz[0])
        if energy_step_MHz <= 0. or not np.allclose(energy_bin_widths_MHz, energy_step_MHz):
            raise ValueError("FFT energy grid must be uniformly increasing")

        measured_DOS_range = float(np.max(measured_DOS) - np.min(measured_DOS))
        if measured_DOS_range <= 0.:
            raise ValueError("measured DOS has no peak contrast")
        if peak_prominence is None:
            peak_prominence = peak_prominence_fraction * measured_DOS_range
        elif not np.isfinite(peak_prominence) or peak_prominence < 0.:
            raise ValueError("peak_prominence must be finite and nonnegative")
        else:
            peak_prominence = float(peak_prominence)
            peak_prominence_fraction = peak_prominence / measured_DOS_range

        row_fft_resolution_MHz = spectrum.get("row_fft_resolution_MHz", [spectrum.fft_resolution_MHz])
        fft_resolution_MHz = float(np.max(row_fft_resolution_MHz))
        window_resolution_factor = window_resolution_factors[spectrum.fft_window]
        effective_resolution_MHz = window_resolution_factor * fft_resolution_MHz
        if minimum_peak_distance_MHz is None:
            minimum_peak_distance_MHz = effective_resolution_MHz
        if not np.isfinite(minimum_peak_distance_MHz) or minimum_peak_distance_MHz <= 0.:
            raise ValueError("minimum_peak_distance_MHz must be finite and positive")
        minimum_peak_distance_bins = max(1, int(np.ceil(minimum_peak_distance_MHz / energy_step_MHz - 1e-10)))

        peak_indices, peak_properties = find_peaks(
            measured_DOS,
            prominence=peak_prominence,
            distance=minimum_peak_distance_bins,
        )
        peak_prominences = peak_properties["prominences"]
        peak_energies_MHz = energy_MHz[peak_indices]
        peak_heights = measured_DOS[peak_indices]

        basis_dimension = len(data.reconstruction.occupations)
        detected_peak_count = len(peak_energies_MHz)
        if detected_peak_count == 0:
            raise ValueError("no experimental DOS peaks were detected; reduce peak_prominence")

        # The FFT normalization makes an isolated one-level peak height one. Round each measured DOS peak independently and leave any mismatch with D visible.
        rounded_peak_multiplicities = np.floor(peak_heights + 0.5).astype(int)
        rounded_multiplicity_sum = int(np.sum(rounded_peak_multiplicities))
        multiplicities_are_positive = bool(np.all(rounded_peak_multiplicities >= 1))
        multiplicity_sum_matches_D = rounded_multiplicity_sum == basis_dimension

        # Determine exact Hamiltonian degeneracies from eigenenergies rather than from finite-resolution measured peaks.
        theory_energies_MHz = np.sort(np.asarray(spectrum.energies_MHz, dtype=float))
        degeneracy_tolerance_MHz = 1e-10
        theory_raw_gaps_MHz = np.diff(theory_energies_MHz)
        theory_zero_gap_mask = np.isclose(theory_raw_gaps_MHz, 0., rtol=0., atol=degeneracy_tolerance_MHz)
        theory_raw_gaps_MHz[theory_zero_gap_mask] = 0.
        has_exact_degeneracy = bool(np.any(theory_zero_gap_mask))
        theory_distinct_energies_MHz = []
        theory_multiplicities = []
        for energy_index, energy_MHz_value in enumerate(theory_energies_MHz):
            if energy_index == 0 or not theory_zero_gap_mask[energy_index - 1]:
                theory_distinct_energies_MHz.append(float(energy_MHz_value))
                theory_multiplicities.append(1)
            else:
                theory_multiplicities[-1] += 1
        theory_distinct_energies_MHz = np.asarray(theory_distinct_energies_MHz)
        theory_multiplicities = np.asarray(theory_multiplicities, dtype=int)

        detected_peak_gaps_MHz = np.diff(peak_energies_MHz)
        inferred_degenerate_energies_MHz = np.asarray([], dtype=float)
        inferred_degenerate_gaps_MHz = np.asarray([], dtype=float)
        inferred_degenerate_gap_ratios = []
        inferred_undefined_gap_ratios = 0
        if multiplicities_are_positive:
            inferred_degenerate_energies_MHz = np.repeat(peak_energies_MHz, rounded_peak_multiplicities)
            inferred_degenerate_gaps_MHz = np.diff(inferred_degenerate_energies_MHz)
            for left_gap_MHz, right_gap_MHz in zip(inferred_degenerate_gaps_MHz[:-1], inferred_degenerate_gaps_MHz[1:]):
                smaller_gap_MHz = min(left_gap_MHz, right_gap_MHz)
                larger_gap_MHz = max(left_gap_MHz, right_gap_MHz)
                if larger_gap_MHz == 0.:
                    inferred_undefined_gap_ratios += 1
                else:
                    inferred_degenerate_gap_ratios.append(smaller_gap_MHz / larger_gap_MHz)
        inferred_degenerate_gap_ratios = np.asarray(inferred_degenerate_gap_ratios, dtype=float)

        full_spectrum_inside_selected_energy_range = bool(np.all(np.abs(theory_energies_MHz) <= energy_limit_MHz + degeneracy_tolerance_MHz))
        all_D_peaks_resolved = detected_peak_count == basis_dimension
        mixed_hamiltonian = bool(spectrum.get("mixed_hamiltonian", False))
        gap_ratio_available = not has_exact_degeneracy and all_D_peaks_resolved and basis_dimension >= 3 and not mixed_hamiltonian and full_spectrum_inside_selected_energy_range
        if has_exact_degeneracy:
            gap_ratio_unavailable_reason = "the exact Hamiltonian contains degenerate levels"
        elif basis_dimension < 3:
            gap_ratio_unavailable_reason = "at least three levels are required"
        elif mixed_hamiltonian:
            gap_ratio_unavailable_reason = "the merged rows use different Hamiltonians"
        elif not full_spectrum_inside_selected_energy_range:
            gap_ratio_unavailable_reason = "the selected energy range does not contain the full spectrum"
        elif not all_D_peaks_resolved:
            gap_ratio_unavailable_reason = f"{detected_peak_count} of {basis_dimension} measured peaks are resolved"
        else:
            gap_ratio_unavailable_reason = None

        experimental_level_energies_MHz = np.asarray([], dtype=float)
        gaps_MHz = np.asarray([], dtype=float)
        gap_ratios = []
        if gap_ratio_available:
            experimental_level_energies_MHz = np.asarray(peak_energies_MHz, dtype=float)
            gaps_MHz = np.diff(experimental_level_energies_MHz)
            for left_gap_MHz, right_gap_MHz in zip(gaps_MHz[:-1], gaps_MHz[1:]):
                smaller_gap_MHz = min(left_gap_MHz, right_gap_MHz)
                larger_gap_MHz = max(left_gap_MHz, right_gap_MHz)
                gap_ratios.append(smaller_gap_MHz / larger_gap_MHz)
        gap_ratios = np.asarray(gap_ratios, dtype=float)

        theory_gap_ratios = []
        theory_undefined_gap_ratios = 0
        for left_gap_MHz, right_gap_MHz in zip(theory_raw_gaps_MHz[:-1], theory_raw_gaps_MHz[1:]):
            smaller_gap_MHz = min(left_gap_MHz, right_gap_MHz)
            larger_gap_MHz = max(left_gap_MHz, right_gap_MHz)
            if larger_gap_MHz == 0.:
                theory_undefined_gap_ratios += 1
            else:
                theory_gap_ratios.append(smaller_gap_MHz / larger_gap_MHz)
        theory_gap_ratios = np.asarray(theory_gap_ratios, dtype=float)

        if "photon_number" in data:
            photon_number = data.photon_number
        else:
            photon_number = sum(data.reconstruction.occupations[0])
        return AttrDict(dict(
            energy_MHz=energy_MHz,
            measured_DOS=measured_DOS,
            theory_DOS=theory_DOS,
            peak_indices=peak_indices,
            peak_energies_MHz=peak_energies_MHz,
            peak_heights=peak_heights,
            peak_prominences=peak_prominences,
            multiplicities=rounded_peak_multiplicities,
            rounded_peak_multiplicities=rounded_peak_multiplicities,
            rounded_multiplicity_sum=rounded_multiplicity_sum,
            multiplicities_are_positive=multiplicities_are_positive,
            multiplicity_sum_matches_D=multiplicity_sum_matches_D,
            experimental_level_energies_MHz=experimental_level_energies_MHz,
            gaps_MHz=gaps_MHz,
            gap_ratios=gap_ratios,
            detected_peak_gaps_MHz=detected_peak_gaps_MHz,
            inferred_degenerate_energies_MHz=inferred_degenerate_energies_MHz,
            inferred_degenerate_gaps_MHz=inferred_degenerate_gaps_MHz,
            inferred_degenerate_gap_ratios=inferred_degenerate_gap_ratios,
            inferred_undefined_gap_ratios=inferred_undefined_gap_ratios,
            theory_energies_MHz=theory_energies_MHz,
            theory_distinct_energies_MHz=theory_distinct_energies_MHz,
            theory_multiplicities=theory_multiplicities,
            theory_gaps_MHz=theory_raw_gaps_MHz,
            theory_raw_gaps_MHz=theory_raw_gaps_MHz,
            theory_zero_gap_mask=theory_zero_gap_mask,
            theory_gap_ratios=theory_gap_ratios,
            theory_undefined_gap_ratios=theory_undefined_gap_ratios,
            has_exact_degeneracy=has_exact_degeneracy,
            degeneracy_tolerance_MHz=degeneracy_tolerance_MHz,
            basis_dimension=basis_dimension,
            detected_peak_count=detected_peak_count,
            gap_ratio_count=len(gap_ratios),
            all_D_peaks_resolved=all_D_peaks_resolved,
            gap_ratio_available=gap_ratio_available,
            gap_ratio_unavailable_reason=gap_ratio_unavailable_reason,
            sufficient_for_gap_ratios=gap_ratio_available,
            full_spectrum_inside_selected_energy_range=full_spectrum_inside_selected_energy_range,
            photon_number=photon_number,
            peak_prominence=peak_prominence,
            peak_prominence_fraction=float(peak_prominence_fraction),
            minimum_peak_distance_MHz=float(minimum_peak_distance_MHz),
            energy_limit_MHz=float(energy_limit_MHz),
            fft_resolution_MHz=fft_resolution_MHz,
            effective_resolution_MHz=effective_resolution_MHz,
            window_resolution_factor=window_resolution_factor,
            fft_window=spectrum.fft_window,
            mixed_resolution=bool(spectrum.get("mixed_resolution", False)),
            raw_window_peak_ambiguity=spectrum.fft_window == "raw",
            poisson_mean=2. * np.log(2.) - 1.,
            goe_mean=4. - 2. * np.sqrt(3.),
        ))

    def analyze_sff(self, 
                    data=None, 
                    row_normalize=True):
        """
        Analyzes SFF based. Should be run after running `analyze_spectrum`.
        This dependency can be lifted in fugure. 
        """
        
        data = self.data if data is None else data
        if "reconstruction" not in data or "spectrum" not in data:
            raise ValueError("SFF analysis requires analyzed spectroscopy data")
        if data.get("spectrum_only", False):
            raise ValueError("SFF requires one common complex time grid and is unavailable for merged spectra")
        reconstruction = data.reconstruction
        spectrum = data.spectrum
        if not spectrum.complete_basis:
            raise ValueError("SFF requires the complete fixed-N occupation basis; this data gives only a projected trace")

        cycles = np.asarray(reconstruction.cycles)
        A = np.asarray(reconstruction.A, dtype=complex).copy()
        if A.ndim != 2 or A.shape[1] != len(spectrum.time_us):
            raise ValueError("measured return and spectroscopy time dimensions differ")
        if row_normalize:
            if len(cycles) == 0 or not np.isclose(cycles[0], 0.):
                raise ValueError("row normalization requires the zero-cycle point")
            if np.any(np.abs(A[:, 0]) < 1e-12):
                raise ValueError("at least one occupation has zero return at t=0")
            A /= A[:, :1]

        energies_MHz = np.asarray(spectrum.energies_MHz)
        dimension = len(energies_MHz)
        if dimension == 0 or A.shape[0] != dimension:
            raise ValueError("measured basis dimension and theory Hilbert-space dimension differ")
        time_us = np.asarray(spectrum.time_us)
        Z_exp = np.sum(A, axis=0)
        Z_theory = np.sum(np.exp(-2j * np.pi * np.outer(energies_MHz, time_us)), axis=0)
        SFF_exp = np.abs(Z_exp / dimension) ** 2
        SFF_theory = np.abs(Z_theory / dimension) ** 2
        degenerate_pairs = np.isclose(energies_MHz[:, None], energies_MHz[None, :], rtol=0., atol=1e-10)
        plateau_reference = np.count_nonzero(degenerate_pairs) / dimension ** 2
        return AttrDict(dict(
            time_us=time_us, 
            A=A, 
            Z_exp=Z_exp,
            Z_theory=Z_theory,
            SFF_exp=SFF_exp, 
            SFF_theory=SFF_theory, 
            dimension=dimension,
            plateau_reference=plateau_reference, 
            nondegenerate_reference=1. / dimension,
            row_normalized=bool(row_normalize), 
            photon_number=data.get("photon_number", sum(reconstruction.occupations[0])),
            phase_frame=data.get("phase_frame", "as_acquired"), 
            physical_kerr_MHz=spectrum.physical_kerr_MHz,
        ))

    @staticmethod
    def display_cycle_phase(result, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[4., 1.])
        iq_axis = fig.add_subplot(grid[0, 0])
        relative_axis = fig.add_subplot(grid[1, 0])
        phase_axis = fig.add_subplot(grid[:, 1])

        iq_axis.plot(result.complex_return.real, result.complex_return.imag, "o--", color="black", linewidth=1.2, markersize=4, label="raw return")
        points = iq_axis.scatter(result.complex_return.real, result.complex_return.imag, c=result.physical_cycles, cmap="viridis", s=28, zorder=3)
        if result.get("unwrap_mode", "pair") == "odd_guide":
            iq_axis.scatter(result.complex_return.real[result.closed_mask], result.complex_return.imag[result.closed_mask], color="tab:red", s=30, zorder=4, label="closed pair")
        iq_limit = 1.15 * np.nanmax(np.abs(result.complex_return))
        if iq_limit <= 0.:
            iq_limit = 1.
        iq_axis.axhline(0., color="0.85")
        iq_axis.axvline(0., color="0.85")
        iq_axis.set(xlim=(-iq_limit, iq_limit), ylim=(-iq_limit, iq_limit), xlabel=r"$Q_0$", ylabel=r"$-Q_{90}$", title="raw complex return")
        iq_axis.set_aspect("equal", adjustable="box")
        iq_axis.legend()
        fig.colorbar(points, ax=iq_axis, label="number of physical entire Floquet cycles")

        relative_axis.plot(result.physical_cycles, result.relative_return, "o-")
        if result.get("unwrap_mode", "pair") == "odd_guide":
            relative_axis.plot(result.physical_cycles[result.closed_mask], result.relative_return[result.closed_mask], "o", color="tab:red")
        relative_axis.axhline(1., color="0.7")
        relative_axis.set(xlabel="number of physical entire Floquet cycles", ylabel=r"$|A|/|A(0)|$", title="relative return")

        phase_axis.plot(result.physical_cycles, result.return_phase, "o", label="measured")
        if result.get("unwrap_mode", "pair") == "odd_guide":
            phase_axis.plot(result.physical_cycles[result.closed_mask], result.return_phase[result.closed_mask], "o", color="tab:red", label="closed pair (fit)")
        phase_axis.plot(result.physical_cycles, result.phase_fit, label="fit")
        phase_axis.set(xlabel="number of physical entire Floquet cycles", ylabel="return phase (deg)")
        phase_axis.legend()

        title = f"{result.occupation}: {result.phase_per_cycle:.4f} +/- {result.phase_error:.4f} deg / cycle"
        if result.fnames:
            title += "\n" + "\n".join(result.fnames)
        fig.suptitle(title)
        return fig

    @staticmethod
    def display_calibration_results(calibration, ncols=None):
        results = calibration.results
        if len(results) == 0:
            raise ValueError("calibration results cannot be empty")
        if ncols is None:
            nrows = max(1, int(np.floor(np.sqrt(len(results)))))
            ncols = int(np.ceil(len(results) / nrows))
        elif not isinstance(ncols, (int, np.integer)) or ncols < 1:
            raise ValueError("ncols must be a positive integer")
        else:
            ncols = min(int(ncols), len(results))
            nrows = int(np.ceil(len(results) / ncols))

        fig = plt.figure(figsize=(7 * ncols, 4 * nrows), constrained_layout=True)
        subfigures = fig.subfigures(nrows, ncols, squeeze=False)
        for result, subfigure in zip(results, subfigures.flat):
            EncodingHamiltonianSpectroscopyExperiment.display_cycle_phase(result, subfigure)
        for subfigure in subfigures.flat[len(results):]:
            subfigure.set_visible(False)
        return fig

    @staticmethod
    def display_calibration_summary(calibration):
        rows = np.arange(len(calibration.occupations))
        labels = [str(occupation) for occupation in calibration.occupations]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(rows) + 2)), constrained_layout=True)
        ax.errorbar(calibration.phase_mod180, rows, xerr=calibration.phase_error, fmt="o")
        ax.axvline(0., color="0.7")
        ax.set(xlabel="measured phase mod 180 (deg / entire cycle)", ylabel="occupation", title="entire-cycle calibration summary")
        ax.set_yticks(rows)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        return fig

    @staticmethod
    def display_local_density_of_states(spectrum, occupations, ax=None):
        """Plot rho_i(E) = sum_a |<i|E_a>|^2 delta(E-E_a)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        else:
            fig = ax.figure
        ldos_energies_MHz, energy_indices = np.unique(np.round(spectrum.energies_MHz, 10), return_inverse=True)
        ldos_weights = np.asarray([np.bincount(energy_indices, weights=weights, minlength=len(ldos_energies_MHz)) for weights in spectrum.eigenstate_weights])
        for row, weights in enumerate(ldos_weights):
            height = 0.8 * weights
            ax.hlines(row, -spectrum.energy_limit_MHz, spectrum.energy_limit_MHz, color="0.85")
            ax.vlines(ldos_energies_MHz, row, row + height, color="tab:blue")
        ax.set_yticks(np.arange(len(occupations)))
        ax.set_yticklabels([str(occupation) for occupation in occupations])
        ax.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="eigenenergy E/h (MHz)", ylabel="initial occupation", title="exact local density-of-states weights")
        return fig

    @staticmethod
    def display_occupation(reconstruction, 
                           spectrum, 
                           occupation, 
                           phase_frame=None, 
                           ldos_weight_cutoff=0,
                           axes = None,
                           figsize = None,
                           plot_abs_A = False):
        """
        The lowest-level function that displays
        - time trace
        - FFT result
        - LDOS result
        
        """
        
        if isinstance(occupation, (int, np.integer)):
            row = int(occupation)
            if row < 0 or row >= len(reconstruction.occupations):
                raise IndexError("occupation row is outside the spectroscopy data")
        else:
            occupation = tuple(occupation)
            if occupation not in reconstruction.occupations:
                raise ValueError(f"{occupation} is not in the spectroscopy data")
            row = reconstruction.occupations.index(occupation)
        occupation = tuple(reconstruction.occupations[row])

        measured = spectrum.measured_local[row]
        theory = spectrum.theory_local[row].copy()
        if np.max(theory) > 0.:
            theory *= np.max(measured) / np.max(theory)
        if not np.isfinite(ldos_weight_cutoff) or ldos_weight_cutoff < 0.:
            raise ValueError("ldos_weight_cutoff must be finite and nonnegative")
        ldos_energies_MHz, energy_indices = np.unique(np.round(spectrum.energies_MHz, 10), return_inverse=True)
        ldos_weights = np.bincount(energy_indices, weights=spectrum.eigenstate_weights[row], minlength=len(ldos_energies_MHz))
        keep = ldos_weights >= ldos_weight_cutoff
        if axes is None:
            if figsize is None:
                figsize = (18, 4.8)
            fig, axes = plt.subplots(1, 3, 
                                     figsize=figsize, 
                                     constrained_layout=True)
        axes[0].plot(spectrum.time_us, reconstruction.A[row].real, label="Re A")
        axes[0].plot(spectrum.time_us, reconstruction.A[row].imag, label="Im A")
        if plot_abs_A:
            axes[0].plot(spectrum.time_us, np.abs(reconstruction.A[row]), "--", color="0.5", label="|A|")
        axes[0].set(xlabel="time (us)", 
                    ylabel="return amplitude", 
                    title="oscillation trace")
        axes[0].legend()

        axes[1].plot(spectrum.energy_MHz, measured, color="black", label="measured")
        axes[1].plot(spectrum.energy_MHz, theory, color="tab:orange", label="theory (shape scaled)")
        axes[1].set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude", title="finite-time FFT")
        axes[1].legend()

        axes[2].vlines(ldos_energies_MHz[keep], 0., ldos_weights[keep], color="tab:blue")
        axes[2].plot(ldos_energies_MHz[keep], ldos_weights[keep], "o", color="tab:blue", markersize=4)
        axes[2].set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="eigenenergy E/h (MHz)", ylabel="spectral weight", title="exact LDOS weights")
        title = str(occupation)
        if phase_frame is not None:
            title += f"; frame={phase_frame}"
        fig.suptitle(f"{title}; Kerr={spectrum.physical_kerr_MHz:.6g} MHz")
        return fig

    def display_occupations(self, 
                            data=None, 
                            occupations=None, 
                            ldos_weight_cutoff=1e-3,
                            spectrum_method=None,
                            show_mpm_magnitude_weights=False):
        data = self.data if data is None else data
        if "reconstruction" not in data or "spectrum" not in data:
            raise ValueError("occupation display requires analyzed spectroscopy data")
        if data.get("spectrum_only", False):
            raise ValueError("occupation time traces are unavailable for merged spectra with different time grids")
        if spectrum_method is None:
            spectrum_method = data.get("spectrum_method", "fft")
        spectrum_method = str(spectrum_method).lower()
        if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
            spectrum_method = "matrix_pencil"
        if spectrum_method not in ("fft", "matrix_pencil"):
            raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
        if occupations is None:
            selections = data.reconstruction.occupations
        elif isinstance(occupations, (int, np.integer)):
            selections = [occupations]
        else:
            selections = list(occupations)
            if selections and all(np.isscalar(value) for value in selections):
                selections = [selections]

        figures = {}
        for occupation in selections:
            if isinstance(occupation, (int, np.integer)):
                key = tuple(data.reconstruction.occupations[int(occupation)])
            else:
                key = tuple(occupation)
            if spectrum_method == "matrix_pencil":
                figures[key] = self.display_matrix_pencil_occupation(data=data, occupation=occupation, show_magnitude_weights=show_mpm_magnitude_weights)
            else:
                figures[key] = self.display_occupation(data.reconstruction, data.spectrum, occupation, data.get("phase_frame", None), ldos_weight_cutoff)
        return figures

    def display_level_statistics(self,
                                 data=None,
                                 level_statistics=None,
                                 peak_prominence=None,
                                 peak_prominence_fraction=None,
                                 minimum_peak_distance_MHz=None,
                                 energy_limit_MHz=None):
        """
        Plot the measured complete-basis DOS and the spacing information justified by
        the data. Exact degeneracy gives raw gaps and all defined ratios, with Poisson/GOE
        means retained only as visual references. A nondegenerate Hamiltonian gives a
        gap-ratio plot only when all D measured levels are resolved; otherwise it gives a
        raw detected-gap diagnostic.
        
        """
        if level_statistics is None:
            level_statistics = self.analyze_level_statistics(
                data=data,
                peak_prominence=peak_prominence,
                peak_prominence_fraction=peak_prominence_fraction,
                minimum_peak_distance_MHz=minimum_peak_distance_MHz,
                energy_limit_MHz=energy_limit_MHz,
            )
        panel_count = 3 if level_statistics.has_exact_degeneracy else 2
        figure_width = 20 if level_statistics.has_exact_degeneracy else 14
        fig, axes = plt.subplots(1, panel_count, figsize=(figure_width, 5.2), constrained_layout=True)
        axes[0].plot(level_statistics.energy_MHz, level_statistics.theory_DOS, color="0.65", linewidth=1.5, alpha=0.6, label="exact H shape (scaled)")
        axes[0].plot(level_statistics.energy_MHz, level_statistics.measured_DOS, color="black", linewidth=2., label="experiment")
        axes[0].errorbar(level_statistics.peak_energies_MHz, level_statistics.peak_heights, xerr=0.5 * level_statistics.effective_resolution_MHz, fmt="o", color="tab:red", capsize=3, label="detected peaks (bar = resolution)")
        for peak_energy_MHz, peak_height, multiplicity in zip(level_statistics.peak_energies_MHz, level_statistics.peak_heights, level_statistics.multiplicities):
            axes[0].annotate(f"{peak_height:.2f} -> m~{multiplicity}", (peak_energy_MHz, peak_height), xytext=(0, 7), textcoords="offset points", ha="center", color="tab:red")
        multiplicity_title = f"m = rounded measured peak height; sum(m)={level_statistics.rounded_multiplicity_sum}, D={level_statistics.basis_dimension} (not forced)"
        axes[0].set(xlabel="energy E/h (MHz)", ylabel="summed FFT magnitude", title=f"measured DOS\n{multiplicity_title}")
        axes[0].set_ylim(bottom=0.)
        axes[0].legend()

        if level_statistics.has_exact_degeneracy:
            theory_gap_indices = np.arange(len(level_statistics.theory_raw_gaps_MHz))
            theory_zero_gap_count = int(np.count_nonzero(level_statistics.theory_zero_gap_mask))
            axes[1].scatter(theory_gap_indices, level_statistics.theory_raw_gaps_MHz, s=55, color="0.7", alpha=0.5, label=f"exact H: {theory_zero_gap_count}/{len(theory_gap_indices)} zero gaps")
            if len(level_statistics.inferred_degenerate_gaps_MHz):
                measured_gap_indices = np.arange(len(level_statistics.inferred_degenerate_gaps_MHz))
                measured_zero_gap_count = int(np.count_nonzero(np.isclose(level_statistics.inferred_degenerate_gaps_MHz, 0., rtol=0., atol=level_statistics.degeneracy_tolerance_MHz)))
                measured_label = f"experiment from rounded heights: {measured_zero_gap_count}/{len(measured_gap_indices)} zero gaps, sum(m)={level_statistics.rounded_multiplicity_sum}"
                if not level_statistics.multiplicity_sum_matches_D or level_statistics.detected_peak_count != len(level_statistics.theory_distinct_energies_MHz):
                    measured_label += " (incomplete)"
                axes[1].scatter(measured_gap_indices, level_statistics.inferred_degenerate_gaps_MHz, s=90, color="black", label=measured_label)
            else:
                axes[1].text(0.5, 0.5, "measured peak heights do not give positive multiplicities", transform=axes[1].transAxes, ha="center", va="center")
            axes[1].axhline(0., color="0.8")
            axes[1].set(xlabel="gap index", ylabel=r"adjacent gap $E_{n+1}-E_n$ (MHz)", title="raw adjacent gaps\nexact zeros retained")
            if len(level_statistics.theory_gap_ratios):
                theory_ratio_indices = np.arange(len(level_statistics.theory_gap_ratios))
                axes[2].scatter(theory_ratio_indices, level_statistics.theory_gap_ratios, s=55, color="0.7", alpha=0.5, label=f"exact H: n={len(theory_ratio_indices)}, 0/0 omitted={level_statistics.theory_undefined_gap_ratios}")
            else:
                axes[2].plot([], [], "o", color="0.7", label=f"exact H: all ratios undefined, 0/0 omitted={level_statistics.theory_undefined_gap_ratios}")
            if len(level_statistics.inferred_degenerate_gap_ratios):
                measured_ratio_indices = np.arange(len(level_statistics.inferred_degenerate_gap_ratios))
                axes[2].scatter(measured_ratio_indices, level_statistics.inferred_degenerate_gap_ratios, s=90, color="black", label=f"experiment: n={len(measured_ratio_indices)}, 0/0 omitted={level_statistics.inferred_undefined_gap_ratios}")
            else:
                axes[2].plot([], [], "o", color="black", label=f"experiment: no defined ratios, 0/0 omitted={level_statistics.inferred_undefined_gap_ratios}")
            axes[2].axhline(level_statistics.poisson_mean, color="tab:blue", linestyle="--", label=f"Poisson mean={level_statistics.poisson_mean:.3f}")
            axes[2].axhline(level_statistics.goe_mean, color="tab:orange", linestyle="--", label=f"GOE mean={level_statistics.goe_mean:.3f}")
            axes[2].set(xlabel="defined gap-ratio sample", ylabel=r"adjacent-gap ratio $\tilde r$", ylim=(-0.03, 1.03), title="defined ratios including exact zeros\nreference lines are not a fit")
            axes[2].legend()
            figure_title = f"N={level_statistics.photon_number}, D={level_statistics.basis_dimension}; exact multiplicity greater than one"
        elif not level_statistics.gap_ratio_available:
            theory_gap_indices = np.arange(len(level_statistics.theory_raw_gaps_MHz))
            detected_gap_indices = np.arange(len(level_statistics.detected_peak_gaps_MHz))
            axes[1].scatter(theory_gap_indices, level_statistics.theory_raw_gaps_MHz, s=55, color="0.7", alpha=0.5, label="exact H gaps (background)")
            if len(level_statistics.detected_peak_gaps_MHz):
                axes[1].scatter(detected_gap_indices, level_statistics.detected_peak_gaps_MHz, s=90, color="black", label="gaps between detected peaks")
            axes[1].text(0.5, 0.95, level_statistics.gap_ratio_unavailable_reason, transform=axes[1].transAxes, ha="center", va="top")
            axes[1].set(xlabel="gap index", ylabel=r"adjacent gap $E_{n+1}-E_n$ (MHz)", title="measured spectrum incomplete: raw gaps only\nPoisson/GOE comparison not used")
            figure_title = f"N={level_statistics.photon_number}, D={level_statistics.basis_dimension}; {level_statistics.detected_peak_count}/{level_statistics.basis_dimension} measured peaks detected"
        else:
            theory_ratio_indices = np.arange(len(level_statistics.theory_gap_ratios))
            experimental_ratio_indices = np.arange(len(level_statistics.gap_ratios))
            axes[1].scatter(theory_ratio_indices, level_statistics.theory_gap_ratios, s=55, color="0.7", alpha=0.5, label=f"exact H (background): mean={np.mean(level_statistics.theory_gap_ratios):.3f}")
            axes[1].scatter(experimental_ratio_indices, level_statistics.gap_ratios, s=90, color="black", label=f"experiment: mean={np.mean(level_statistics.gap_ratios):.3f}")
            axes[1].axhline(level_statistics.poisson_mean, color="tab:blue", linestyle="--", label=f"Poisson mean={level_statistics.poisson_mean:.3f}")
            axes[1].axhline(level_statistics.goe_mean, color="tab:orange", linestyle="--", label=f"GOE mean={level_statistics.goe_mean:.3f}")
            axes[1].set(xlabel="gap-ratio sample", ylabel=r"adjacent-gap ratio $\tilde r$", ylim=(-0.03, 1.03), title="adjacent-gap ratios")
            figure_title = f"N={level_statistics.photon_number}, D={level_statistics.basis_dimension}; all measured levels resolved"
        axes[1].legend()
        fig.suptitle(figure_title)
        return fig

    def display_sff(self, 
                    data=None, 
                    sff=None, 
                    row_normalize=True,
                    plot_theory_limit = False):
        if sff is None:
            sff = self.analyze_sff(data=data, row_normalize=row_normalize)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        axes[0].plot(sff.time_us, sff.SFF_exp, color="black", label="experiment")
        axes[0].plot(sff.time_us, sff.SFF_theory, color="tab:orange", label="theory")
        if plot_theory_limit:
            axes[0].axhline(sff.plateau_reference, color="0.6", linestyle=":", label="theory infinite-time average")
        axes[0].set(xlabel="time (us)", ylabel=r"$K(t)=|\mathrm{Tr}\,U(t)/D|^2$", title="spectral form factor")
        axes[0].legend()

        axes[1].semilogy(sff.time_us, np.maximum(sff.SFF_exp, 1e-12), color="black", label="experiment")
        axes[1].semilogy(sff.time_us, np.maximum(sff.SFF_theory, 1e-12), color="tab:orange", label="theory")
        if plot_theory_limit:
            axes[1].axhline(sff.plateau_reference, color="0.6", linestyle=":", label="theory infinite-time average")
        axes[1].set(xlabel="time (us)", ylabel=r"$K(t)$", title="spectral form factor (log scale)")
        axes[1].legend()
        fig.suptitle(f"N={sff.photon_number}, D={sff.dimension}; frame={sff.phase_frame}; Kerr={sff.physical_kerr_MHz:.6g} MHz")
        return fig

    def display_matrix_pencil_occupation(self,
                                         data=None,
                                         occupation=None,
                                         result=None,
                                         show_magnitude_weights=False):
        """Plot the independent rowwise Matrix-Pencil result for one occupation."""
        data = self.data if data is None else data
        if result is None:
            if occupation is None:
                raise ValueError("occupation must be supplied when result is None")
            result = self.analyze_matrix_pencil_occupation(occupation, data=data)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
        axes[0].plot(result.time_us, result.measured_return.real, color="tab:blue", label="Re A")
        axes[0].plot(result.time_us, result.measured_return.imag, color="tab:orange", label="Im A")
        axes[0].plot(result.time_us, result.fitted_return.real, "--", color="tab:blue", label="Re MPM fit")
        axes[0].plot(result.time_us, result.fitted_return.imag, "--", color="tab:orange", label="Im MPM fit")
        axes[0].set(xlabel="time (us)", ylabel="return amplitude", title="oscillation trace and rowwise MPM fit")
        axes[0].legend()

        axes[1].plot(result.energy_MHz, result.measured_spectrum, color="black", label="measured FFT")
        axes[1].plot(result.energy_MHz, result.reconstructed_spectrum, "--", color="tab:blue", label="rowwise MPM reconstruction")
        for frequency_index, frequency_MHz in enumerate(result.frequencies_MHz):
            label = "rowwise MPM poles" if frequency_index == 0 else None
            axes[1].axvline(frequency_MHz, color="cyan", linewidth=0.8, alpha=0.55, label=label)
        axes[1].set(xlim=(-data.spectrum.energy_limit_MHz, data.spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude", title="finite-time FFT")
        axes[1].legend()

        if len(result.frequencies_MHz):
            axes[2].vlines(result.frequencies_MHz, 0., result.local_weights, color="tab:blue")
            axes[2].plot(result.frequencies_MHz, result.local_weights, "o", color="tab:blue", label="linear local weights")
            if show_magnitude_weights:
                axes[2].plot(result.frequencies_MHz, result.local_magnitude_weights, "x", color="0.4", label="amplitude magnitudes")
        else:
            axes[2].text(0.5, 0.5, "no stable rowwise candidates", transform=axes[2].transAxes, ha="center", va="center")
        axes[2].axhline(0., color="0.8", linewidth=0.8)
        axes[2].set(xlim=(-data.spectrum.energy_limit_MHz, data.spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="local pole weight", title="individual-occupation MPM weights")
        if len(result.frequencies_MHz):
            axes[2].legend()

        frame = data.get("phase_frame", "as_acquired")
        fig.suptitle(f"{result.occupation}; rowwise poles={len(result.frequencies_MHz)}; estimated signal rank={result.diagnostic.estimated_signal_rank}; relative residual={result.relative_residual:.3f}; frame={frame}")
        return fig

    def display_matrix_pencil(self,
                              data=None,
                              matrix_pencil=None,
                              show_poles=True):
        """
        Compare the measured and theoretical occupation-resolved FFTs and DOS.

        The upper panels show the measured and theoretical occupation-resolved
        finite-time FFTs. The lower-left panel compares the summed measured FFT
        with the Matrix-Pencil reconstruction and delta-function DOS. The
        lower-right panel compares the summed theoretical FFT with the exact
        Hamiltonian delta-function DOS.
        """
        data = self.data if data is None else data
        if "reconstruction" not in data or "spectrum" not in data:
            raise ValueError("Matrix-Pencil display requires analyzed spectroscopy data")
        if data.get("spectrum_only", False):
            raise ValueError("Matrix Pencil requires occupation traces on one common time grid")
        if matrix_pencil is None:
            matrix_pencil = data.get("matrix_pencil", None)
        if matrix_pencil is None:
            raise ValueError("Matrix-Pencil analysis is unavailable; analyze with spectrum_method='matrix_pencil'")

        reconstruction = data.reconstruction
        spectrum = data.spectrum
        rows = np.arange(len(reconstruction.occupations))
        labels = [str(occupation) for occupation in reconstruction.occupations]
        energy_MHz = np.asarray(spectrum.energy_MHz)
        measured_local = np.asarray(spectrum.measured_local)
        reconstructed_local = np.asarray(matrix_pencil.reconstructed_local)
        theory_local = np.asarray(spectrum.theory_local)
        if measured_local.shape != reconstructed_local.shape or measured_local.shape != theory_local.shape:
            raise ValueError("measured, Matrix-Pencil, and theory spectra use different grids")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
        measured_axis = axes[0, 0]
        theory_axis = axes[0, 1]
        measured_DOS_axis = axes[1, 0]
        theory_DOS_axis = axes[1, 1]
        extent = [energy_MHz[0], energy_MHz[-1], -0.5, len(rows) - 0.5]
        vmax = max(np.max(measured_local), np.max(theory_local))
        for axis, local, title in zip((measured_axis, theory_axis),
                                      (measured_local, theory_local),
                                      ("measured finite-time FFT", "theory finite-time FFT")):
            image = axis.imshow(local, origin="lower", aspect="auto", interpolation="nearest", extent=extent, cmap="magma", vmin=0., vmax=vmax)
            if show_poles:
                for frequency_MHz in matrix_pencil.selected_frequencies_MHz:
                    axis.axvline(frequency_MHz, color="cyan", linewidth=0.7, alpha=0.45)
            axis.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", title=title)
            axis.set_yticks(rows)
            axis.set_yticklabels(labels)
        if show_poles:
            rowwise_frequencies_MHz = [candidate.frequency_MHz for candidate in matrix_pencil.candidates.per_row]
            rowwise_indices = [candidate.row_index for candidate in matrix_pencil.candidates.per_row]
            measured_axis.scatter(rowwise_frequencies_MHz, rowwise_indices, s=20, facecolors="none", edgecolors="cyan", linewidths=0.8, label="rowwise Matrix-Pencil poles")
            measured_axis.legend()
        measured_axis.set_ylabel(f"occupation {data.mode_labels}")
        fig.colorbar(image, ax=(measured_axis, theory_axis), label="spectral magnitude")

        measured_DOS_axis.plot(energy_MHz, spectrum.measured, color="black", linewidth=1.5, label="measured FFT sum")
        measured_DOS_axis.plot(energy_MHz, matrix_pencil.reconstructed, color="tab:blue", linestyle="--", linewidth=1.5, label="Matrix-Pencil finite-time reconstruction")
        measured_DOS_axis.vlines(matrix_pencil.selected_frequencies_MHz, 0., matrix_pencil.pole_DOS_weights, color="tab:blue", alpha=0.7, label="Matrix-Pencil linear pole DOS weights")
        measured_DOS_axis.plot(matrix_pencil.selected_frequencies_MHz, matrix_pencil.pole_DOS_weights, "o", color="tab:blue", markersize=5)
        measured_DOS_title = "measured FFT sum and Matrix-Pencil DOS" if spectrum.complete_basis else "measured projected FFT sum and Matrix-Pencil weights"
        measured_DOS_axis.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude / pole weight", title=measured_DOS_title)
        measured_DOS_axis.legend()

        exact_energies_MHz, exact_energy_indices = np.unique(np.round(np.asarray(spectrum.energies_MHz), 10), return_inverse=True)
        eigenstate_weights = np.asarray(spectrum.eigenstate_weights)
        if eigenstate_weights.ndim != 2 or eigenstate_weights.shape[1] != len(spectrum.energies_MHz):
            raise ValueError("exact eigenstate weights and energies have different dimensions")
        exact_state_weights = np.sum(eigenstate_weights, axis=0)
        exact_DOS_weights = np.bincount(exact_energy_indices, weights=exact_state_weights, minlength=len(exact_energies_MHz))
        theory_DOS_axis.plot(energy_MHz, spectrum.theory, color="tab:orange", linewidth=1.5, label="theory FFT sum")
        theory_DOS_axis.vlines(exact_energies_MHz, 0., exact_DOS_weights, color="tab:orange", alpha=0.7, label="exact Hamiltonian DOS weights")
        theory_DOS_axis.plot(exact_energies_MHz, exact_DOS_weights, "o", color="tab:orange", markersize=5)
        theory_DOS_title = "theory FFT sum and exact DOS" if spectrum.complete_basis else "theory projected FFT sum and exact spectral weights"
        theory_DOS_axis.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude / DOS weight", title=theory_DOS_title)
        theory_DOS_axis.legend()
        if show_poles:
            for frequency_MHz in matrix_pencil.selected_frequencies_MHz:
                measured_DOS_axis.axvline(frequency_MHz, color="cyan", linewidth=0.7, alpha=0.35)
                theory_DOS_axis.axvline(frequency_MHz, color="cyan", linewidth=0.7, alpha=0.35)
        fig.suptitle(f"K={len(matrix_pencil.selected_frequencies_MHz)} shared poles; global relative residual={matrix_pencil.relative_residual:.3f}; frequencies modulo fs={matrix_pencil.sampling.sampling_frequency_MHz:.6g} MHz")
        return fig

    def display_orthogonality(self, 
                              data=None,
                              **kwargs):
        """Plot raw cross return and raw/normalized off-diagonal leakage."""
        data = self.data if data is None else data
        if "matrix" not in data:
            raise ValueError(
                "orthogonality display requires stage='orthogonality' data"
            )
        matrix = np.asarray(data.matrix, dtype=complex)
        labels = [str(tuple(occupation)) for occupation in data.occupations]
        size = len(labels)
        if matrix.shape != (size, size):
            raise ValueError("orthogonality matrix and labels have different sizes")

        raw_offdiagonal = np.abs(matrix).copy()
        np.fill_diagonal(raw_offdiagonal, 0.)
        panels = [
            (np.abs(matrix), r"raw $|M_{j i}|$ (diagonal contrast retained)"),
            (raw_offdiagonal, r"raw off-diagonal $|M_{j i}|$"),
            (
                np.asarray(data.offdiagonal_normalized_power, dtype=float),
                r"normalized off-diagonal $|M_{j i}|^2/(|M_{ii}||M_{jj}|)$",
            ),
        ]
        
        figsize = kwargs.get("figsize", (max(16, 1.35 * size + 10), 6))
        fig, axes = plt.subplots(
            1, 3, figsize=figsize,
            constrained_layout=True,
        )
        for axis, (values, title) in zip(axes, panels):
            image = axis.imshow(
                values, origin="upper", aspect="equal", cmap="magma", vmin=0.
            )
            axis.set_title(title)
            axis.set_xlabel("encoder occupation i")
            axis.set_ylabel("decoder occupation j")
            axis.set_xticks(np.arange(size))
            axis.set_yticks(np.arange(size))
            axis.set_xticklabels(labels, rotation=55, ha="right")
            axis.set_yticklabels(labels)
            fig.colorbar(image, ax=axis)
            if size <= 6:
                for row in range(size):
                    for column in range(size):
                        value = values[row, column]
                        text = "nan" if not np.isfinite(value) else f"{value:.3f}"
                        axis.text(
                            column, row, text,
                            ha="center", va="center", color="cyan", fontsize=8,
                        )

        finite_offdiagonal = np.asarray(
            data.offdiagonal_normalized_power, dtype=float).copy()
        np.fill_diagonal(finite_offdiagonal, np.nan)
        max_leakage = (
            float(np.nanmax(finite_offdiagonal))
            if np.any(np.isfinite(finite_offdiagonal)) else np.nan
        )
        fig.suptitle(
            "zero-cycle encoder/decoder cross return; "
            f"min diagonal |M|={np.min(data.diagonal_amplitude):.3f}; "
            f"max normalized off-diagonal power={max_leakage:.3g}"
        )
        return fig

    @staticmethod
    def display_result(reconstruction, 
                       spectrum, 
                       mode_labels):
        rows = np.arange(len(reconstruction.occupations))
        labels = [str(occupation) for occupation in reconstruction.occupations]
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

        extent = [spectrum.energy_MHz[0], spectrum.energy_MHz[-1], -0.5, len(rows) - 0.5]
        vmax = max(np.max(spectrum.measured_local), np.max(spectrum.theory_local))
        for ax, local, title in zip(axes[0], [spectrum.measured_local, spectrum.theory_local], ["experiment", "theory"]):
            image = ax.imshow(local, origin="lower", aspect="auto", interpolation="nearest", extent=extent, cmap="magma", vmin=0., vmax=vmax)
            fft_label = f"{spectrum.fft_window}, pad x{spectrum.zero_padding}"
            if spectrum.get("zero_padding", None) is None:
                fft_label = f"{spectrum.fft_window}, merged grid"
            elif spectrum.get("mixed_resolution", False):
                fft_label = f"{spectrum.fft_window}, mixed resolution"
            ax.set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", title=f"{title}: {fft_label}")
            ax.set_yticks(rows)
            ax.set_yticklabels(labels)
        axes[0, 0].set_ylabel(f"occupation {mode_labels}")
        fig.colorbar(image, ax=axes[0], label="spectral magnitude")

        EncodingHamiltonianSpectroscopyExperiment.display_local_density_of_states(spectrum, reconstruction.occupations, axes[1, 0])
        axes[1, 1].plot(spectrum.energy_MHz, spectrum.measured, color="black", label="experiment")
        axes[1, 1].plot(spectrum.energy_MHz, spectrum.theory, color="tab:orange", label="theory")
        title = "projected spectrum"
        if spectrum.complete_basis:
            title = "complete-basis DOS"
        axes[1, 1].set(xlim=(-spectrum.energy_limit_MHz, spectrum.energy_limit_MHz), xlabel="energy E/h (MHz)", ylabel="spectral magnitude", title=title)
        axes[1, 1].legend()
        resolution_label = f"FFT resolution: {spectrum.fft_resolution_MHz:.6g} MHz"
        if spectrum.get("mixed_resolution", False):
            resolution_label = f"FFT resolution range: {np.min(spectrum.row_fft_resolution_MHz):.6g}-{np.max(spectrum.row_fft_resolution_MHz):.6g} MHz"
        if spectrum.get("mixed_hamiltonian", False):
            resolution_label += f"; source-H mismatch: {spectrum.hamiltonian_mismatch_MHz:.6g} MHz"
        fig.suptitle(f"Kerr used in plotted Hamiltonian: {spectrum.physical_kerr_MHz:.6g} MHz; {resolution_label}")
        return fig

    def display(self, data=None, occupation=None, **kwargs):
        if data is not None:
            self.data = data
        if "matrix" in self.data:
            return self.display_orthogonality(self.data)
        if "spectrum" in self.data:
            spectrum_method = str(kwargs.get("spectrum_method", self.data.get("spectrum_method", "fft"))).lower()
            if spectrum_method in ("mpm", "rowwise_matrix_pencil"):
                spectrum_method = "matrix_pencil"
            if spectrum_method not in ("fft", "matrix_pencil"):
                raise ValueError("spectrum_method must be 'fft' or 'matrix_pencil'")
            if occupation is not None:
                if self.data.get("spectrum_only", False):
                    raise ValueError("occupation time traces are unavailable for merged spectra with different time grids")
                if spectrum_method == "matrix_pencil":
                    return self.display_matrix_pencil_occupation(data=self.data,
                                                                 occupation=occupation,
                                                                 show_magnitude_weights=kwargs.get("show_mpm_magnitude_weights", False))
                return self.display_occupation(self.data.reconstruction, self.data.spectrum, occupation, self.data.get("phase_frame", None), kwargs.get("ldos_weight_cutoff", 1e-3))
            if spectrum_method == "matrix_pencil":
                return self.display_matrix_pencil(data=self.data,
                                                  show_poles=kwargs.get("show_mpm_poles", True))
            fig = self.display_result(self.data.reconstruction,
                                      self.data.spectrum,
                                      self.data.mode_labels)
            if self.data.spectrum.complete_basis and kwargs.get("level_statistics", True):
                self.display_level_statistics(
                    data=self.data,
                    peak_prominence=kwargs.get("level_peak_prominence", None),
                    peak_prominence_fraction=kwargs.get("level_peak_prominence_fraction", None),
                    minimum_peak_distance_MHz=kwargs.get("level_minimum_peak_distance_MHz", None),
                    energy_limit_MHz=kwargs.get("level_energy_limit_MHz", None),
                )
            return fig
        if "phase_mod180" in self.data:
            if "results" not in self.data:
                self.data = self.analyze_calibration(self.batch_expts)
            self.display_calibration_results(self.data, kwargs.get("ncols", None))
            return self.display_calibration_summary(self.data)
        return super().display(data=self.data, **kwargs)

    @staticmethod
    def hardware_parameters(station, 
                            swap_stors, 
                            sync_cycles, 
                            floquet_gauss_sigma=None,
                            floquet_waveform=None):
        """
        Returns hardware related physical paramters such as
            - floquet_cycle_us: time for a single floquet cycle in a microsecond
            - couplings_MHz: an array of effective BS coupling between man and stor
            - physical_kerr_MHz: self Kerr on a central mode (manipulate)
        All the values are calculated from the config/expt_cfg input
        """
        
        if (isinstance(sync_cycles, (bool, np.bool_))
                or not isinstance(sync_cycles, (int, np.integer)) or sync_cycles < 0):
            raise ValueError("sync_cycles must be a nonnegative integer")
        ramp_sigma = station.hardware_cfg.device.manipulate.ramp_sigma
        if isinstance(ramp_sigma, (list, tuple, np.ndarray)):
            ramp_sigma = ramp_sigma[0]
        pulse_us = []
        pi_fracs = []
        for stor in swap_stors:
            pulse_name = f"M1-S{stor}"
            waveform = floquet_waveform if floquet_waveform is not None else station.ds_floquet.get_waveform(pulse_name)
            if waveform in ("gauss", "gaussian", "arb"):
                sigma = floquet_gauss_sigma
                if sigma is None:
                    sigma = station.ds_floquet.get_gauss_sigma(pulse_name)
                length = sigma * station.ds_floquet.get_gauss_n_sigma(pulse_name)
            else:
                length = station.ds_floquet.get_len(pulse_name) + 6. * ramp_sigma
            pulse_us.append(length)
            pi_fracs.append(station.ds_floquet.get_pi_frac(pulse_name))

        sync_us = station.soccfg.cycles2us(sync_cycles)
        # Actual hardware time of one entire Floquet/Trotter cycle.
        floquet_cycle_us = sum(pulse_us) + len(swap_stors) * sync_us
        if not np.all(np.isfinite(pulse_us + pi_fracs)) or min(pulse_us + pi_fracs) <= 0.:
            raise ValueError("Floquet pulse lengths and pi fractions must be finite and positive")
        if not np.isfinite(floquet_cycle_us) or floquet_cycle_us <= 0.:
            raise ValueError("Floquet cycle duration must be finite and positive")

        # 2 * pi * g * t_swap = pi / 2 -> g = 1/ 4/ t_swap
        # pi_frac repetitions of each pulse+sync block make a full swap:
        # g_{bare} = 1/ 4 / n_frac / (t_pulse + t_sync)
        # len(swap_stors) -> g_{eff} * T_F = g_{bare} * (t_pulse+t_sync) = 1/4/n_frac
        # So g_{eff} = 1/4/n_frac/T_F
        couplings_MHz = [1. / (4. * pi_frac * floquet_cycle_us) for pi_frac in pi_fracs]
        physical_kerr_MHz = station.hardware_cfg.device.manipulate.kerr
        if isinstance(physical_kerr_MHz, (list, tuple, np.ndarray)):
            physical_kerr_MHz = physical_kerr_MHz[0]
        physical_kerr_MHz = -abs(physical_kerr_MHz)
        if not np.isfinite(physical_kerr_MHz):
            raise ValueError("physical Kerr must be finite")
        return AttrDict(dict(floquet_cycle_us=floquet_cycle_us,
                             couplings_MHz=np.asarray(couplings_MHz),
                             physical_kerr_MHz=physical_kerr_MHz))

    @classmethod
    def analyze_calibration(cls, 
                            expts, 
                            occupations=None, 
                            cycle_pairs=None, 
                            repeats=None):
        """
        The method first groups input experiments in a pair of encoding/decoding
        calibration with phi 0 and 90. For each pair, extract phase accumulation
        using `analyze_cycle_phase` and return the collective dictionary in the following form.
        
        Return: Attrdict containing
            - "occupations": occupation_order,
            - "results": results,
            - "phase_mod180": np.asarray([result.phase_per_cycle for result in results]),
            - "phase_error": np.asarray([result.phase_error for result in results])
        
        """
        
        if not expts:
            raise ValueError("calibration expts cannot be empty")
        grouped = {}
        for expt in expts:
            cfg = expt.cfg.expt
            occupation = tuple(cfg.spectroscopy_occupations)
            phi = cfg.spectroscopy_analyzer_phase
            if phi not in (0., 90.):
                raise ValueError(f"{occupation} has analyzer phase {phi}; expected 0 or 90")
            if occupation not in grouped:
                grouped[occupation] = {0.: [], 90.: []}
            grouped[occupation][phi].append(expt)

        if occupations is None:
            occupation_order = list(grouped)
        else:
            occupation_order = [tuple(occupation) for occupation in occupations]
        if len(occupation_order) != len(grouped) or set(occupation_order) != set(grouped):
            raise ValueError("calibration occupations do not match the saved configs")
        results = []
        for occupation in occupation_order:
            phi0_expts = grouped[occupation][0.]
            phi90_expts = grouped[occupation][90.]
            if not phi0_expts or len(phi0_expts) != len(phi90_expts):
                raise ValueError(f"{occupation} needs the same nonzero number of phi=0 and phi=90 jobs")
            if repeats is not None and len(phi0_expts) != repeats:
                raise ValueError(f"{occupation} has {len(phi0_expts)} repeats; expected {repeats}")
            saved_cycle_pairs = np.asarray(phi0_expts[0].cfg.expt.n_cycle_pairs)
            if cycle_pairs is not None and not np.array_equal(saved_cycle_pairs, cycle_pairs):
                raise ValueError(f"{occupation}: calibration cycles do not match the requested cycles")
            saved_unwrap_mode = phi0_expts[0].cfg.expt.get("phase_unwrap_mode", "pair")
            if saved_unwrap_mode == "odd_guide":
                saved_cycle_counts = np.asarray(phi0_expts[0].cfg.expt.n_physical_cycles)
            else:
                saved_cycle_counts = saved_cycle_pairs
            for expt in phi0_expts + phi90_expts:
                if not np.array_equal(expt.cfg.expt.n_cycle_pairs, saved_cycle_pairs):
                    raise ValueError(f"{occupation}: calibration configs use different cycles")
                if expt.cfg.expt.get("phase_unwrap_mode", "pair") != saved_unwrap_mode:
                    raise ValueError(f"{occupation}: calibration configs use different unwrap modes")
                if saved_unwrap_mode == "odd_guide" and not np.array_equal(expt.cfg.expt.n_physical_cycles, saved_cycle_counts):
                    raise ValueError(f"{occupation}: calibration configs use different physical cycles")
            results.append(cls.analyze_cycle_phase(phi0_expts,
                                                   phi90_expts,
                                                   occupation,
                                                   saved_cycle_counts,
                                                   unwrap_mode=saved_unwrap_mode,))
        return AttrDict(dict(
            occupations=occupation_order,
            results=results,
            phase_mod180=np.asarray([result.phase_per_cycle for result in results]),
            phase_error=np.asarray([result.phase_error for result in results]),
        ))

    @staticmethod
    def calibration_batch(default_expt_cfg, 
                          swap_stors, 
                          occupations, 
                          cycle_pairs,
                          sync_cycles=10, 
                          repeats=1, 
                          reps=1500,
                          unwrap_mode="pair"):
        """
        Returns dictionary of 
            - default_expt_cfg
            - list of config to be overrided in each job
            - repeats (usually 1)
        The list of config is then used to make and batch jobs in a chunk.
        For now, other paramters such as `update_phase`, `palindrome_scramble`, 
        `spectroscopy_prep_phases`, `floquet_hardware_loop`, `swept_params`
        are fixed.
        In a nearest term, the program should migrate to using `floquet_hardware_loop`
        
        The actual batch is done by plugging the output to the BatchRunner.
        Example:
            calibration_batch = EncSpec.calibration_batch()
            calibration_expt = calibration_runner.execute(calibration_batch.configs)
        """
        if unwrap_mode not in ("pair", "odd_guide"):
            raise ValueError("unwrap_mode must be 'pair' or 'odd_guide'")

        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=reps, 
            storage_reset=swap_stors, 
            swap_stors=swap_stors,
            n_cycle_pairs=cycle_pairs.tolist(),
            scramble_sync_cycles=sync_cycles,
            
            floquet_hardware_loop=False,
            detunings=[0.] * len(swap_stors), #detuning must be 0 for the calibration
            update_phases=True, 
            palindrome_scramble=False, 
            phase_unwrap_mode=unwrap_mode,
            spectroscopy_prep_phases=[0., 180.],
        ))
        if unwrap_mode == "odd_guide":
            physical_cycles = []
            for cycle_pair in cycle_pairs.tolist():
                physical_cycles.extend([2 * cycle_pair, 2 * cycle_pair + 1])
            defaults.update(dict(
                n_physical_cycles=physical_cycles,
                swept_params=["n_physical_cycle", "spectroscopy_prep_phase"],
            ))
        else:
            defaults.update(dict(
                swept_params=["n_cycle_pair", "spectroscopy_prep_phase"],
            ))
        configs = [
            dict(spectroscopy_occupations=occupation, 
                 spectroscopy_analyzer_phase=phi,
                 final_analyzer_phase_per_cycle_deg=0.)
            for occupation in occupations for _ in range(repeats) for phi in [0., 90.]
        ]
        return AttrDict(dict(default_expt_cfg=defaults, 
                             configs=configs, 
                             repeats=repeats))

    @staticmethod
    def orthogonality_batch(default_expt_cfg,
                            swap_stors,
                            occupations,
                            sync_cycles=10,
                            reps=300,
                            **kwargs):
        """
        Build one zero-cycle job for each encoder occupation.

        Within that job, measure every decoder occupation at analyzer phases
        0 and 90 degrees, each with preparation phases 0 and 180 degrees.
        
        ``decoder_analyzer_rows`` stores indices. specifically, modulo 2 should
        give the index for analyzer_phase, where as quotient by 2 gives
        which occupation index should be ran.
        """
        swap_stors = [int(stor) for stor in swap_stors]
        occupations = [list(occupation) for occupation in occupations]
        # decoder 0 at 0/90 deg, then decoder 1 at 0/90 deg, and so on.
        decoder_analyzer_rows = list(range(2 * len(occupations)))
        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=int(reps),
            storage_reset=swap_stors,
            swap_stors=swap_stors,
            detunings=[0.] * len(swap_stors),
            scramble_sync_cycles=int(sync_cycles),
            floquet_cycle=0,
            floquet_hardware_loop=False,
            update_phases=False,
            palindrome_scramble=False,
            spectroscopy_phase_correction_mode=kwargs.get("correction_mode","final_analyzer"),
            final_analyzer_phase_per_cycle_deg=0.,
            orthogonality_decoder_occupations=deepcopy(occupations),
            orthogonality_analyzer_phases=[0., 90.],
            decoder_analyzer_rows=decoder_analyzer_rows,
            spectroscopy_prep_phases=[0., 180.],
            swept_params=[
                "decoder_analyzer_row",
                "spectroscopy_prep_phase",
            ],
        ))
        configs = [
            dict(spectroscopy_occupations=list(occupation))
            for occupation in occupations
        ]
        return AttrDict(dict(
            default_expt_cfg=defaults,
            configs=configs,
            occupations=deepcopy(occupations),
            points_per_job=4 * len(occupations),
            total_points=4 * len(occupations) ** 2,
        ))

    @staticmethod
    def propagator_batch(default_expt_cfg,
                         swap_stors,
                         occupations,
                         cycles,
                         phase_by_occupation,
                         sync_cycles=10,
                         reps=300):
        """Build one raw short-time propagator job per encoded occupation."""
        swap_stors = [int(stor) for stor in swap_stors]
        occupations = [list(occupation) for occupation in occupations]
        cycles = [int(cycle) for cycle in cycles]
        decoder_phase_correction = [
            float(phase_by_occupation[tuple(occupation)])
            for occupation in occupations
        ]

        # Every outer sweep value says exactly what is played:
        # [Floquet cycle, decoder occupation..., analyzer phase].
        cycle_decoder_analyzers = [
            [cycle, *decoder_occupation, analyzer_phase]
            for cycle in cycles
            for decoder_occupation in occupations
            for analyzer_phase in (0., 90.)
        ]

        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=int(reps),
            storage_reset=swap_stors,
            swap_stors=swap_stors,
            detunings=[0.] * len(swap_stors),
            scramble_sync_cycles=int(sync_cycles),
            floquet_cycle=0,
            floquet_hardware_loop=False,
            update_phases=True,
            palindrome_scramble=False,
            spectroscopy_phase_correction_mode="final_analyzer",
            final_analyzer_phase_per_cycle_deg=0.,
            propagator_cycles=cycles,
            propagator_occupations=deepcopy(occupations),
            propagator_decoder_phase_correction_deg=(
                decoder_phase_correction
            ),
            cycle_decoder_analyzers=cycle_decoder_analyzers,
            spectroscopy_prep_phases=[0., 180.],
            swept_params=[
                "cycle_decoder_analyzer",
                "spectroscopy_prep_phase",
            ],
        ))
        configs = [
            dict(spectroscopy_occupations=list(occupation))
            for occupation in occupations
        ]
        points_per_job = 4 * len(cycles) * len(occupations)
        return AttrDict(dict(
            default_expt_cfg=defaults,
            configs=configs,
            cycles=cycles,
            occupations=deepcopy(occupations),
            points_per_job=points_per_job,
            total_points=points_per_job * len(occupations),
        ))

    @staticmethod
    def spectroscopy_batch(default_expt_cfg, 
                           swap_stors, 
                           occupations, 
                           cycle_chunks,
                           phase_by_occupation, 
                           detunings=None, 
                           sync_cycles=10, 
                           reps=300,
                           final_occupations=None):
        """
        Returns dictionary of 
            - default_expt_cfg
            - list of config to be overrided in each job
        The list of config is then used to make and batch jobs in a chunk.
        The actual batch is done by plugging the output to the BatchRunner.
        
        Example:
            spectroscopy_batch = EncSpec.spectroscopy_batch()
            spectroscopy_expt = spectroscopy_runner.execute(spectroscopy_batch.configs)
        """
        if detunings is None:
            detunings = [0.] * len(swap_stors)
        else:
            detunings = list(detunings)
        defaults = deepcopy(default_expt_cfg)
        defaults.update(dict(
            reps=reps, 
            storage_reset=swap_stors, 
            swap_stors=swap_stors,
            detunings=detunings, 
            scramble_sync_cycles=sync_cycles,
            
            floquet_hardware_loop=False,
            update_phases=True, 
            palindrome_scramble=False, 
            spectroscopy_phase_correction_mode="final_analyzer",
            spectroscopy_prep_phases=[0., 180.],
            swept_params=["floquet_cycle", "spectroscopy_prep_phase"],
        ))
        final_occupations = occupations if final_occupations is None else final_occupations
        configs = [
            dict(spectroscopy_occupations=occupation,
                 spectroscopy_final_occupations=final_occupation,
                 spectroscopy_analyzer_phase=phi,
                 final_analyzer_phase_per_cycle_deg=phase_by_occupation[tuple(final_occupation)],
                 floquet_cycles=cycles.tolist())
            for occupation, final_occupation in zip(occupations, final_occupations) for cycles in cycle_chunks for phi in [0., 90.]
        ]
        return AttrDict(dict(default_expt_cfg=defaults, 
                             configs=configs))


class FloquetDisplacementKerrExperiment(DarkBaseExperiment):
    """Fit Floquet Kerr with the existing cavity-Ramsey gain-sweep analysis."""

    def acquire(self, progress=False, debug=False):
        data = super().acquire(progress=progress, debug=debug)
        data["floquet_cycle_us"] = self.prog.floquet_cycle_us
        self.cfg.expt.floquet_cycle_us = self.prog.floquet_cycle_us
        return data

    def _ramsey_fitter(self, data):
        cfg = deepcopy(self.cfg)
        gain_to_alpha = cfg.device.manipulate.gain_to_alpha
        if isinstance(gain_to_alpha, (list, tuple, np.ndarray)):
            gain_to_alpha = gain_to_alpha[cfg.expt.man_mode_no - 1]
        cfg.device.manipulate.gain_to_alpha = [gain_to_alpha]
        return CavityRamseyGainSweepFitting(data, config=cfg)

    def analyze(self, data=None, fit=True, **kwargs):
        if data is not None:
            self.data = data
        if not fit:
            return self.data
        cycle_pairs = np.asarray(self.data.get("cycle_pairs", self.data["xpts"]))
        displace_gains = np.asarray(self.data.get("displace_gains", self.data["ypts"]))
        time_us = 2 * cycle_pairs * self.data["floquet_cycle_us"]
        ramsey_data = AttrDict(dict(
            gain_list=displace_gains,
            xpts=np.tile(time_us, (len(displace_gains), 1)),
            g_avgi=self.data["avgi"], g_avgq=self.data["avgq"],
            g_amps=self.data["amps"], g_phases=self.data["phases"],
            e_avgi=self.data["avgi"], e_avgq=self.data["avgq"],
            e_amps=self.data["amps"], e_phases=self.data["phases"],
        ))
        self._ramsey_fitter(ramsey_data).analyze(fit=True, **kwargs)
        self.data.update(ramsey_data)
        self.data["cycle_pairs"] = cycle_pairs
        self.data["displace_gains"] = displace_gains
        return self.data

    def display(self, data=None, **kwargs):
        if data is not None:
            self.data = data
        if "Kerr" not in self.data:
            self.analyze()
        return self._ramsey_fitter(self.data).display(**kwargs)

    def save_data(self, data=None):
        if data is None:
            data = self.data
        peaks = {key: data.pop(key) for key in ["time_peak_g", "time_peak_e"] if key in data}
        fname = super().save_data(data)
        data.update(peaks)
        return fname
