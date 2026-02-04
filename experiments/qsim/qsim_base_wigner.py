
import os
from copy import deepcopy

import matplotlib.pyplot as plt
from experiments.qsim.qsim_base import QsimBaseExperiment
from slab import AttrDict, Experiment
from fitting.wigner import WignerAnalysis
import qutip as qt
import numpy as np
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.qsim.utils import (
    ensure_list_in_cfg,
)
from slab.datamanagement import AttrDict

class QsimWignerBaseExperiment(QsimBaseExperiment):
    """
    Sweep 1 or 2 parameters in cfg.expt
    Experimental Config:
    expt = dict(
        expts: number experiments should be 1 here as we do soft loops
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        qubits: this is just 0 for the purpose of the currrent multimode sample

        init_stor: int or list, the man/storage(s) to initialize the photon into (0-7)
        init_fock: bool, whether to initialize the man/storage mode in a fock state or in a coherent state
        init_alpha: if init_fock is False, the coherent (complex) state alpha to initialize the mode into

        ro_stor: storage to readout the photon from (0-7)
        active_reset, man_reset, storage_reset: bool
        swept_params: list of parameters to sweep, e.g. ['detune', 'gain']

        wigner parameters:
        perform_wigner: bool
        parity_fast: bool
        pulse_correction: bool — if True, perform two acquisitions per alpha with phase_second_pulse=180 then 0 (non-inline)
        
    )
    In principle this overlaps with qick.NDAveragerProgram, but this allows you to
    skip writing new expeirment classes or at least acquire() while doing 
    more general sweeps than just a qick register, incl nonlinear steps.
    Consider doing NDAverager or RAverager if there's speed advantage.
    See notebook for usage.
    """

    def acquire(self, progress=False, debug=False):
        ensure_list_in_cfg(self.cfg)

        self.pulse_correction = self.cfg.expt.get('pulse_correction', False)
        self.cfg.expt.parity_fast = self.cfg.expt.get('parity_fast', False)
        self.cfg.expt.post_select_pre_pulse = self.cfg.expt.get('post_select_pre_pulse', False)
        self.cfg.expt.active_reset = self.cfg.expt.get('active_reset', False)

        self.cfg.expt.perform_wigner = True

        read_num = 1
        if self.cfg.expt.post_select_pre_pulse:
            read_num += 1
        if self.cfg.expt.active_reset:
            read_num += 3 # ge reset always, ef_reset=True by default, pre_selection_reset=True by default

        # Perform the tomography at different displacements

        # extract displacement list from file path
        if 'alpha_list' in self.cfg.expt:
            alpha_2d = self.cfg.expt.alpha_list  # 2d list 
            # convert list to array 
            alpha_2d = np.array(alpha_2d)
            alpha_list = alpha_2d[:, 0] + 1j * alpha_2d[:, 1]
        else:
            alpha_list = np.load(self.cfg.expt["displacement_path"])  # complex ndarray


        assert len(self.cfg.expt.swept_params) in {1,2}, "can only handle 1D and 2D sweeps for now"
        sweep_dim = 2 if len(self.cfg.expt.swept_params) == 2 else 1

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
            'alpha':alpha_list,
        }
        if sweep_dim == 2:
            data['xpts'] = inner_params
            data['ypts'] = outer_params
        else:
            data['xpts'] = outer_params
        self.outer_params, self.inner_params = outer_params, inner_params

        for self.cfg.expt[outer_param] in tqdm(outer_params, disable=not progress):
            for self.cfg.expt[inner_param] in inner_params:
                for alpha in tqdm(alpha_list, disable=not progress):
                    self.cfg.expt.phase_second_pulse = 180 # reset the phase of the second pulse
                    self.cfg.expt.wigner_alpha = alpha

                    wigner = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)
                    self.prog = wigner

                    avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc],
                                                    threshold=None,
                                                    load_pulses=True,
                                                    progress=False,
                                                    debug=debug,
                                                    readouts_per_experiment=read_num)
                    avgi, avgq = avgi[0][-1], avgq[0][-1]
                    data['avgi'].append(avgi)
                    data['avgq'].append(avgq)
                    data['amps'].append(np.abs(avgi+1j*avgq)) # Calculating the magnitude
                    data['phases'].append(np.angle(avgi+1j*avgq)) # Calculating the phase

                    idata, qdata = wigner.collect_shots()
                    data['idata'].append(idata)
                    data['qdata'].append(qdata)

                    if self.pulse_correction:
                        self.cfg.expt.phase_second_pulse = 0
                        wigner = self.ProgramClass(soccfg=self.soccfg, cfg=self.cfg)
                        avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc],
                                                    threshold=None,
                                                    load_pulses=True,
                                                    progress=False,
                                                    readouts_per_experiment=read_num,
                                                    #  debug=debug
                                                    )
                        avgi, avgq = avgi[0][-1], avgq[0][-1]
                        data['avgi'].append(avgi)
                        data['avgq'].append(avgq)
                        data['amps'].append(np.abs(avgi+1j*avgq)) # Calculating the magnitude
                        data['phases'].append(np.angle(avgi+1j*avgq)) # Calculating the phase

                        idata, qdata = wigner.collect_shots()
                        data['idata'].append(idata)
                        data['qdata'].append(qdata)

        for key in 'avgi avgq amps phases idata qdata'.split():
            data[key] = np.array(data[key])
            dims = [len(outer_params), len(inner_params), len(alpha_list)] # forcing this shape regardless of if len(inner_params)==1
            if self.pulse_correction:
                dims.append(2)
            if key in ['idata', 'qdata']:
                dims.append(-1)
            data[key] = np.reshape(data[key], tuple(dims))

        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.cfg.expt['expts'] = len(data["alpha"]) # this is necessary because of general fitting bin_ss_data which expects the first dimension to come from cfg.expt.expts

        self.data=data
        return data


    def analyze(self, data=None, **kwargs):
        pass

    def analyze_wigner(self, data=None, **kwargs):
        if data is None:
            data = self.data

        mode_state_num = kwargs.get('mode_state_num', 10)

        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1  # using first manipulate channel index needs to be fixed at some point

        read_num = 1
        if self.cfg.expt.post_select_pre_pulse:
            read_num += 1
        if self.cfg.expt.active_reset:
            read_num += 3 # ge reset always, ef_reset=True by default, pre_selection_reset=True by default

        # Calculate the stepping for the final readout
        idx_start = read_num - 1
        idx_step = read_num

        wigner_outputs = dict(
            parity=np.zeros((len(self.outer_params), len(self.inner_params), len(data["alpha"]))),
        )
        if self.pulse_correction:
            wigner_outputs.update(dict(
                pe_plus=np.zeros((len(self.outer_params), len(self.inner_params), len(data["alpha"]))),
                pe_minus=np.zeros((len(self.outer_params), len(self.inner_params), len(data["alpha"]))),
                parity_plus=np.zeros((len(self.outer_params), len(self.inner_params), len(data["alpha"]))),
                parity_minus=np.zeros((len(self.outer_params), len(self.inner_params), len(data["alpha"]))),
            ))
        else:
            wigner_outputs.update(dict(
                pe=np.zeros((len(self.outer_params), len(self.inner_params))),
            ))

        for i_outer, outer_param_val in enumerate(self.outer_params):
            for i_inner, inner_param_val in enumerate(self.inner_params):
                print(outer_param_val, inner_param_val)
                if self.pulse_correction: # shape: (len(outer_params), len(inner_params), len(alpha_list), 2, read_num * num_shots)
                    data_minus = {}
                    data_plus = {}

                    data_minus['idata'] = data['idata'][i_outer, i_inner, :, 0, idx_start::idx_step]
                    data_minus['qdata'] = data['qdata'][i_outer, i_inner, :, 0, idx_start::idx_step]
                    data_plus['idata'] = data['idata'][i_outer, i_inner, :, 1, idx_start::idx_step]
                    data_plus['qdata'] = data['qdata'][i_outer, i_inner, :, 1, idx_start::idx_step]
                    print("shape", data_plus['idata'].shape)

                    if 'post_select_pre_pulse' in self.cfg.expt and self.cfg.expt.post_select_pre_pulse:
                        assert False, "post_select_pre_pulse with pulse_correction not implemented yet"
                    
                    
                    wigner_analysis_minus = WignerAnalysis(data=data_minus,
                                                           config=self.cfg, 
                                                            mode_state_num=mode_state_num,
                                                            alphas=data["alpha"])

                    wigner_analysis_plus = WignerAnalysis(data=data_plus,
                                                          config=self.cfg,
                                                          mode_state_num=mode_state_num,
                                                          alphas=data["alpha"])
            
                    pe_plus = wigner_analysis_plus.bin_ss_data()
                    pe_minus = wigner_analysis_minus.bin_ss_data()
                    parity_plus = (1 - pe_plus) - pe_plus
                    parity_minus = (1 - pe_minus) - pe_minus
                    parity = (parity_minus - parity_plus) / 2

                    # print('pe_plus', pe_plus)
                    # print('parity', parity)

                    # apply scale 
                    scale_parity = self.cfg.device.manipulate.alpha_scale[self.man_mode_idx]
                    # print('scale parity:', scale_parity )
            
                    wigner_outputs["pe_plus"][i_outer, i_inner] = pe_plus
                    wigner_outputs["pe_minus"][i_outer, i_inner] = pe_minus
                    wigner_outputs["parity_plus"][i_outer, i_inner] = parity_plus
                    wigner_outputs["parity_minus"][i_outer, i_inner] = parity_minus
                    wigner_outputs["parity"][i_outer, i_inner] = parity / (scale_parity)
                    print('max parity:', np.max(wigner_outputs["parity"]))
                    print('max parity before scaling:', np.max(parity))


                else:
                    data_wigner = {}
                    idx_start = read_num - 1
                    idx_step = read_num
                    data_wigner["idata"] = data["idata"][i_outer, i_inner, :, idx_start::idx_step]
                    data_wigner["qdata"] = data["qdata"][i_outer, i_inner, :, idx_start::idx_step]

                    wigner_analysis = WignerAnalysis(data=data_wigner,
                                                      config=self.cfg, 
                                                      mode_state_num=mode_state_num,
                                                      alphas=data["alpha"])
                    pe = wigner_analysis.bin_ss_data()
                    wigner_outputs["pe"][i_outer, i_inner] = pe
                    wigner_outputs["parity"][i_outer, i_inner] = (1 - pe) - pe

        data['wigner_outputs'] = wigner_outputs
        return data


    def display(self, data=None, mode_state_num=None, initial_state=None, rotate=None, state_label='', debug_components=False, station=None, save_fig=False, **kwargs):
        if data is None:
            data = self.data

        # Defaults
        if mode_state_num is None:
            mode_state_num = int(getattr(self.cfg.expt, 'display_mode_state_num', 5))
        if rotate is None:
            rotate = bool(getattr(self.cfg.expt, 'display_rotate', False))
        
        wigner_outputs = data['wigner_outputs']

        # Basic validation
        if 'parity' not in wigner_outputs or 'alpha' not in data:
            raise ValueError('Expected keys "parity" and "alpha" in data for Wigner display.')
        parity = wigner_outputs['parity']
        alphas = data['alpha']
        if len(parity[0][0]) != len(alphas):
            raise ValueError(f'Length mismatch: parity ({len(parity[0][0])}) vs alpha ({len(alphas)}).')

        # Default initial state if not provided
        if initial_state is None:
            initial_state = qt.fock(mode_state_num, 0).unit()
        
        wigner_outputs.update(dict(
            rho=[],
            rho_rotated=[],
            fidelity=[],
            W_fit=[],
            W_ideal=[],
            alpha_wigner=[],
            theta_opt=[],
            target_state=initial_state,
        ))

        # Build analysis and plot
        wigner_analysis = WignerAnalysis(data=data, config=self.cfg, mode_state_num=mode_state_num, alphas=alphas, station=station)
        for i_outer, outer_param_val in enumerate(self.outer_params):
            wigner_outputs['rho'].append([])
            wigner_outputs['rho_rotated'].append([])
            wigner_outputs['fidelity'].append([])
            wigner_outputs['W_fit'].append([])
            wigner_outputs['W_ideal'].append([])
            wigner_outputs['alpha_wigner'].append([])
            wigner_outputs['theta_opt'].append([])
            for i_inner, inner_param_val in enumerate(self.inner_params):
                
                print(f'Analyzing for {self.outer_param}={outer_param_val}, {self.inner_param}={inner_param_val}')

                results = wigner_analysis.wigner_analysis_results(parity[i_outer, i_inner], initial_state=initial_state, rotate=rotate)
                fig = wigner_analysis.plot_wigner_reconstruction_results(results, initial_state=initial_state, state_label=state_label)

                wigner_outputs['rho'][-1].append(results['rho'])
                wigner_outputs['rho_rotated'][-1].append(results['rho_rotated'])
                wigner_outputs['fidelity'][-1].append(results['fidelity'])
                wigner_outputs['W_fit'][-1].append(results['W_fit'])
                wigner_outputs['W_ideal'][-1].append(results['W_ideal'])
                wigner_outputs['alpha_wigner'][-1].append(results['x_vec'])
                wigner_outputs['theta_opt'][-1].append(results['theta_max'])


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
        self.cfg = temp_cfg
        return self.fname
