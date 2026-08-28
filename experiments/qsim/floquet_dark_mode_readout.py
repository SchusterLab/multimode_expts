# -*- coding: utf-8 -*-
import importlib

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.signal import find_peaks
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from fitting.qsim import matrix_pencil as matrix_pencil_analysis
from fitting.qsim import level_statistics as level_statistics_analysis
from fitting.qsim import mbr_spectrum as mbr_spectrum_analysis
from fitting.qsim import mbr_phase as mbr_phase_analysis
from fitting.fit_display_classes import (
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


class _StageForwarding(type):
    """Forward class-attribute lookups for methods that moved to a stage class.

    The acquisition notebooks say ``EncSpec.orthogonality_batch``, which is
    attribute access on the *class*. The module-level ``__getattr__`` at the
    bottom of this file handles moved classes and cannot see it, so a stage
    split silently breaks those call sites. A metaclass ``__getattr__`` runs
    only after normal lookup fails, so it costs nothing on real attributes.

    Lazy for the usual reason: every stage module imports this class as its
    base, so resolving at class-creation time would close the cycle. The guard
    against a name the stage class does not actually define matters -- stage
    classes inherit this metaclass, so a blind ``getattr`` would recurse.

    Transitional, and it goes away with the ``stage=`` dispatch, once consumers
    address the owning classes directly.
    """

    def __getattr__(cls, name):
        stage = _MOVED_METHODS.get(name)
        if stage is not None:
            owner = _stage_owner(stage)
            if name in vars(owner):
                return vars(owner)[name].__get__(None, owner)
        raise AttributeError(
            f"type object {cls.__name__!r} has no attribute {name!r}")


class EncodingHamiltonianSpectroscopyExperiment(DarkBaseExperiment,
                                                metaclass=_StageForwarding):
    """Per-job and aggregate analysis for encoding-calibrated spectroscopy."""

    def __getattr__(self, name):
        """Instance-side twin of ``_StageForwarding.__getattr__``.

        The ``stage=`` facade hands ``self`` -- an instance of *this* class --
        to the owning stage class's ``analyze``, whose moved body then calls
        ``self.reconstruct_spectroscopy``. Normal lookup cannot find that: it
        lives on a subclass. The metaclass covers ``EncSpec.<name>`` but not
        ``instance.<name>``, so both halves are needed.

        Called only when normal lookup fails, so real attributes are untouched
        and a missing ``self.data`` still raises AttributeError as before.
        """
        stage = _MOVED_METHODS.get(name)
        if stage is not None:
            owner = _stage_owner(stage)
            attribute = vars(owner).get(name)
            if attribute is not None:
                return attribute.__get__(self, owner)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    @classmethod
    def _from_expts(cls, expts, job_ids=None, station=None):
        """Collect saved jobs for analysis; station supplies missing hardware data."""
        
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
            prog = getattr(expt, "prog", None)
            if prog is not None and hasattr(prog, "calculate_floquet_cycle_us") and hasattr(prog, "m1s_pi_fracs"):
                floquet_cycle_us = float(prog.calculate_floquet_cycle_us())
                pi_fracs = np.asarray([prog.m1s_pi_fracs[stor - 1] for stor in swap_stors], dtype=float)
                couplings_MHz = 1. / (4. * pi_fracs * floquet_cycle_us)
                program_hardware.append((floquet_cycle_us, couplings_MHz))
                break

        if program_hardware:
            floquet_cycle_us, couplings_MHz = program_hardware[0]
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
            self.data = _stage_owner("calibration").analyze_calibration(
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
            self.data = _stage_owner("orthogonality").reconstruct_orthogonality(
                self.batch_expts,
                kwargs.get("occupations"),
            )
        elif stage == "propagator":
            self.data = _stage_owner("propagator").reconstruct_propagator(
                self.batch_expts,
                kwargs.get("occupations"),
            )
        elif stage == "spectrum":
            self.data = _stage_owner("spectrum").analyze(self, **kwargs)
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

    # Analyzer-phase numerics live in fitting/qsim/mbr_phase.py (spec 7.5).
    # Wrappers keep the historical call sites and notebook usage working.
    _cycle_branches = staticmethod(mbr_phase_analysis.cycle_branches)
    build_phase_correction = staticmethod(mbr_phase_analysis.build_phase_correction)
    _unwrap_cycle_phase = staticmethod(mbr_phase_analysis.unwrap_cycle_phase)
    _saved_correction = staticmethod(mbr_phase_analysis.saved_correction)


    # Spectrum and Hamiltonian numerics live in fitting/qsim/mbr_spectrum.py
    # (spec 7.5). Wrapper keeps the historical call sites working.
    analyze_spectrum = staticmethod(mbr_spectrum_analysis.analyze_spectrum)


    # Matrix-Pencil numerics live in fitting/qsim/matrix_pencil.py (spec 7.5).
    # These wrappers keep the historical call sites and notebook usage working.
    analyze_matrix_pencil = staticmethod(matrix_pencil_analysis.analyze_matrix_pencil)
    analyze_matrix_pencil_trace = staticmethod(matrix_pencil_analysis.analyze_matrix_pencil_trace)

    # Spectrum merging, level statistics and SFF numerics live in
    # fitting/qsim/level_statistics.py (spec 7.5). Wrappers keep the historical
    # call sites, notebook usage and self.data defaulting.
    merge_spectra = staticmethod(level_statistics_analysis.merge_spectra)

    def display(self, data=None, occupation=None, **kwargs):
        if data is not None:
            self.data = data
        if "matrix" in self.data:
            # Passing `self` to a subclass method is deliberate scaffolding:
            # display_orthogonality touches `self` only for the `data` default,
            # which is supplied here. It dies with this dispatch.
            return _stage_owner("orthogonality").display_orthogonality(
                self, self.data)
        if "spectrum" in self.data:
            return _stage_owner("spectrum").display(
                self, occupation=occupation, **kwargs)
        if "phase_mod180" in self.data:
            owner = _stage_owner("calibration")
            if "results" not in self.data:
                self.data = owner.analyze_calibration(self.batch_expts)
            owner.display_calibration_results(self.data, kwargs.get("ncols", None))
            return owner.display_calibration_summary(self.data)
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


# ---------------------------------------------------------------------------
# Transitional: analyze(stage=...) for stages that now own their own Experiment.
#
# jonginn's acquisition and post-processing notebooks still enter through
# ``EncodingHamiltonianSpectroscopyExperiment.analyze(stage=...)``, so the string
# dispatch keeps working while the stages move out one at a time.
# ``analysis_notebooks/guan/MBR_analysis.py`` is migrated to the new classes as
# each one lands, and is the worked example to copy from. This whole block goes
# away once no consumer passes ``stage``.
#
# Lazy for the same reason ``__getattr__`` below is: every stage module imports
# its base class from here, so a top-level import would close the cycle.
_STAGE_OWNERS = {
    "spectrum": ("mbr_spectrum", "MBRSpectrumExperiment"),
    "calibration": ("mbr_phase_correction", "MBRPhaseCorrectionExperiment"),
    "orthogonality": ("mbr_orthogonality", "MBROrthogonalityExperiment"),
    "propagator": ("mbr_propagator", "MBRPropagatorExperiment"),
}

# Methods that left the god class, so ``EncSpec.<name>`` keeps resolving.
# One row per moved method, not per class: the notebooks address the methods.
_MOVED_METHODS = {
    "_postprocess_reconstruction": "spectrum",
    "analyze_level_statistics": "spectrum",
    "analyze_matrix_pencil_occupation": "spectrum",
    "analyze_sff": "spectrum",
    "display_level_statistics": "spectrum",
    "display_local_density_of_states": "spectrum",
    "display_matrix_pencil": "spectrum",
    "display_matrix_pencil_occupation": "spectrum",
    "display_occupation": "spectrum",
    "display_occupations": "spectrum",
    "display_result": "spectrum",
    "display_sff": "spectrum",
    "reconstruct_pair_spectroscopy": "spectrum",
    "reconstruct_spectroscopy": "spectrum",
    "spectroscopy_batch": "spectrum",
    "subsample_spectroscopy_shots": "spectrum",
    "_calibration_data": "calibration",
    "analyze_calibration": "calibration",
    "analyze_cycle_phase": "calibration",
    "calibration_batch": "calibration",
    "display_calibration_results": "calibration",
    "display_calibration_summary": "calibration",
    "display_cycle_phase": "calibration",
    "phase_correction_from_calibration": "calibration",
    "display_orthogonality": "orthogonality",
    "orthogonality_batch": "orthogonality",
    "reconstruct_orthogonality": "orthogonality",
    "propagator_batch": "propagator",
    "reconstruct_propagator": "propagator",
}


def _stage_owner(stage):
    module, name = _STAGE_OWNERS[stage]
    return getattr(importlib.import_module(f"experiments.qsim.{module}"), name)


# ---------------------------------------------------------------------------
# Compatibility: measurement families that moved to their own modules.
#
# The acquisition notebooks address programs as
# ``meas.qsim.floquet_dark_mode_readout.<Name>``, so the old attribute has to
# keep resolving. A module-level ``__getattr__`` (PEP 562) does that lazily,
# which matters: every new module imports the still-resident base classes from
# here, so a top-level re-import would be circular.
#
# Deliberately invisible to ``inspect.getmembers``, and therefore to the
# flattening exporter in ``experiments/__init__.py``. Each name is exported to
# the ``experiments`` namespace by its own module instead, exactly once.
_MOVED_TO = {
    "BroadbandGeValidationProgram": "dark_mode_broadband_ge_validation",
    "CENTRAL_RETURN_ALLOWED_TOTAL_PHOTONS": "central_boson_local_return",
    "CENTRAL_RETURN_PRIMARY_PHOTONS": "central_boson_local_return",
    "CentralBosonLocalReturnExperiment": "central_boson_local_return",
    "CentralBosonLocalReturnProgram": "central_boson_local_return",
    "DarkBaseRProgram": "dark_mode_multiparity_chevron",
    "DarkT1Experiment": "dark_mode_t1",
    "DarkT1Program": "dark_mode_t1",
    "FloquetDisplacementKerrExperiment": "floquet_displacement_kerr",
    "FloquetDisplacementKerrProgram": "floquet_displacement_kerr",
    "ManStorMultiparityChevronRExperiment": "dark_mode_multiparity_chevron",
    "ManStorMultiparityChevronRProgram": "dark_mode_multiparity_chevron",
    "SidebandStarkAmplificationModifiedProgram": "sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_newold": "sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_old": "sideband_stark_shift_cal",
    "StorageSwapPhaseAccumulationProgram": "storage_swap_phase_cal",
    "configure_central_return_metadata": "central_boson_local_return",
    "validate_central_return_occupations": "central_boson_local_return",
}


def __getattr__(name):
    module = _MOVED_TO.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(
        importlib.import_module(f"experiments.qsim.{module}"), name)


def __dir__():
    return sorted(list(globals()) + list(_MOVED_TO))
