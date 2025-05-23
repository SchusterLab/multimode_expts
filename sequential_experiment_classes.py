import numpy as np
import os
import time
from tqdm import tqdm
import json
from slab.datamanagement import SlabFile
from slab import get_next_filename, AttrDict
from slab.experiment import Experiment
import experiments as meas
from slab.instruments import *
import yaml
from scipy.interpolate import UnivariateSpline
from slab import get_next_filename, get_current_filename

from slab.dsfit import *
from scipy.optimize import curve_fit
import experiments.fitting as fitter
from scipy.fft import fft, fftfreq
from multimode_expts.MM_base import MM_base
from multimode_expts.MM_rb_base import MM_rb_base
from multimode_expts.MM_dual_rail_base import MM_dual_rail_base
from multimode_expts.fit_display import * # for generate combos in MultiRBAM
#import clear_output
from IPython.display import clear_output

class sequential_base_class():
    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        self.soccfg = soccfg
        self.path = path
        self.prefix = prefix
        self.config_file = config_file
        self.exp_param_file = exp_param_file

        #load parameter files
        self.load_config()
        self.load_exp_param()

        pass

    def load_config(self):
        '''Load config file '''
        with open(self.config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        self.yaml_cfg = AttrDict(yaml_cfg)
        return None

    def load_exp_param(self):
        '''Load experiment parameter file '''
        with open(self.exp_param_file, 'r') as file:
            # Load the YAML content
            self.loaded = yaml.safe_load(file)
        return None
    
    def map_sequential_cfg_to_experiment(self): 
        '''Map the sequential config to the experiment config'''
        print(self.experiment_name)
        for keys in self.loaded[self.experiment_name].keys():
            try:
                self.loaded[self.experiment_name][keys] = self.loaded[self.sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass
        return None
    
    def perform_chevron_analysis(self):
        """
        Perform Chevron analysis and live plotting.
        This method performs a Chevron analysis on experimental sweep data 
        and displays the results in a live plot. It assumes that the 
        `self.expt_sweep` attribute is present and contains the necessary 
        data for the analysis. Specifically, `self.expt_sweep.data` must 
        include:
            - 'freq_sweep': Frequencies used in the sweep.
            - 'xpts': Time points for the sweep (expects the first element).
            - 'avgi': Response matrix data for the sweep.
        The method initializes a `ChevronFitting` object with the provided 
        data, performs the analysis, and displays the results. It also 
        clears any previous plots to ensure the display is updated with 
        the latest results.
        Note:
            - This method requires the `multimode_expts.fit_display_classes` 
              module for the `ChevronFitting` class.
            - The `IPython.display.clear_output` function is used to clear 
              the output for live plotting.
            - If any exception occurs during the process, it will print an 
              error message indicating the failure.
        Raises:
            Exception: If any error occurs during the Chevron analysis or 
            plotting process.
        """
        
        try:
            from multimode_expts.fit_display_classes import ChevronFitting

            # Initialize Chevron analysis
            chevron_analysis = ChevronFitting(
                frequencies=self.expt_sweep.data['freq_sweep'],
                time=self.expt_sweep.data['xpts'][0],
                response_matrix=np.array(self.expt_sweep.data['avgi'])
            )

            # Analyze the data
            chevron_analysis.analyze()

            # Close previous plots and display the new one
            #
            from IPython.display import clear_output
            # from multimode_expts.fit_display_classes import SidebandFitting
            clear_output(wait=True)
            plt.close('all')  # Close all existing figures
            chevron_analysis.display_results()
            return chevron_analysis

        except Exception as e:
            print(f"Chevron analysis failed: {e}")
            return None
    

class sidebands_class(sequential_base_class):
    '''Class for sideband experiments; using sideband general experiment'''

    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        super().__init__(soccfg, path, prefix, config_file, exp_param_file)
        self.experiment_class = 'single_qubit.sideband_general'
        self.experiment_name = 'SidebandGeneralExperiment'

    def initialize_expt_sweep(self):
        '''Initialize the experiment sweep data structure'''
        self.expt_sweep = Experiment(
            path=self.path,
            prefix=self.sweep_experiment_name,
            config_file=self.config_file,
        )
        self.expt_sweep.data = dict(freq_sweep=[], gain_sweep=[])

    def sideband_freq_sweep(self):
        '''Frequency sweep for sideband experiments'''
        self.initialize_expt_sweep()
        chevron = None

        for index, freq in enumerate(np.arange(self.loaded[self.sweep_experiment_name]['freq_start'], 
                                                self.loaded[self.sweep_experiment_name]['freq_stop'], 
                                                self.loaded[self.sweep_experiment_name]['freq_step'])):

            print('Index: %s Freq. = %s MHz' % (index, freq))
            self.loaded[self.experiment_name]['flux_drive'][1] = freq

            run_exp = eval(f"meas.{self.experiment_class}.{self.experiment_name}(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file)")

            run_exp.cfg.expt = eval(f"self.loaded['{self.experiment_name}']")

            if run_exp.cfg.expt.active_reset:
                run_exp.cfg.device.readout.relax_delay = 100  # Wait time between experiments [us]
            run_exp.cfg.device.readout.relax_delay = 8000
            print('Waiting for %s us' % run_exp.cfg.device.readout.relax_delay)

            run_exp.go(analyze=False, display=False, progress=False, save=False)

            # Save sweep data
            self.save_sweep_data('freq_sweep', freq, run_exp)

            # Perform sideband analysis and live plotting
            chevron = self.perform_chevron_analysis()
        return chevron

    def sideband_gain_freq_sweep(self):
        '''Gain and frequency sweep for sideband experiments'''
        self.initialize_expt_sweep()

        for index, gain in enumerate(np.arange(self.loaded[self.sweep_experiment_name]['gain_start'],
                                                self.loaded[self.sweep_experiment_name]['gain_stop'],
                                                self.loaded[self.sweep_experiment_name]['gain_step'])):
            print('Index: %s Gain. = %s MHz' % (index, gain))
            self.loaded[self.experiment_name]['flux_drive'][2] = gain

            self.sideband_freq_sweep()

    def sideband_cross_kerr_cancellation(self):
        '''Run two experiments; one with prepulse 1 (no occupied storage) and one with prepulse 2 (occupied storage)'''
        # Prepulse 1
        self.loaded[self.experiment_name]['pre_sweep_pulse'] = self.loaded[self.sweep_experiment_name]['pre_sweep_pulse1']
        self.sideband_gain_freq_sweep()

        # Prepulse 2
        self.loaded[self.experiment_name]['pre_sweep_pulse'] = self.loaded[self.sweep_experiment_name]['pre_sweep_pulse2']
        self.sideband_gain_freq_sweep()


    def run_sweep(self, sweep_experiment_name):
        '''Run the sweep'''
        self.sweep_experiment_name = sweep_experiment_name
        self.map_sequential_cfg_to_experiment()

        if sweep_experiment_name == 'sideband_general_sweep':
            return self.sideband_freq_sweep()

        elif sweep_experiment_name == 'sideband_gain_freq_sweep':
            self.sideband_gain_freq_sweep()

        elif sweep_experiment_name == 'sideband_cross_kerr_cancellation':
            self.sideband_cross_kerr_cancellation()
    

    def save_sweep_data(self, sweep_key, sweep_value, run_exp):
        '''Save sweep data to the experiment file'''
        self.expt_sweep.data[sweep_key].append(sweep_value)
        for data_key in run_exp.data.keys():
            if data_key not in self.expt_sweep.data.keys():
                self.expt_sweep.data[data_key] = []
            self.expt_sweep.data[data_key].append(run_exp.data[data_key])
        self.expt_sweep.save_data()

    


class man_f0g1_class(sequential_base_class):
    '''Class for length Rabi f0g1 sweep experiments'''

    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        super().__init__(soccfg, path, prefix, config_file, exp_param_file)
        self.experiment_class = 'single_qubit.length_rabi_f0g1_general'
        self.experiment_name = 'LengthRabiGeneralF0g1Experiment'

    def freq_sweep(self):
        '''Frequency sweep for length Rabi f0g1'''
        self.expt_sweep = Experiment(
            path=self.path,
            prefix=self.sweep_experiment_name,
            config_file=self.config_file,
        )
        chevron = None

        self.expt_sweep.data = dict(freq_sweep=[])

        for index, freq in enumerate(np.arange(self.loaded[self.sweep_experiment_name]['freq_start'], 
                                                self.loaded[self.sweep_experiment_name]['freq_stop'], 
                                                self.loaded[self.sweep_experiment_name]['freq_step'])):

            print('Index: %s Freq. = %s GHz' % (index, freq))
            self.loaded[self.experiment_name]['freq'] = freq

            run_exp = eval(f"meas.{self.experiment_class}.{self.experiment_name}(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file)")

            run_exp.cfg.expt = eval(f"self.loaded['{self.experiment_name}']")

            if self.loaded[self.experiment_name]['active_reset']:
                print('Doesn’t make sense to active reset in this experiment')
            run_exp.cfg.device.readout.relax_delay = 2500  # Wait time between experiments [us]

            run_exp.go(analyze=False, display=False, progress=False, save=False)

            # Add entry to sweep file
            self.expt_sweep.data['freq_sweep'].append(freq)
            for data_key in run_exp.data.keys():
                if data_key not in self.expt_sweep.data.keys():
                    self.expt_sweep.data[data_key] = []
                self.expt_sweep.data[data_key].append(run_exp.data[data_key])

            # Save the data
            self.expt_sweep.save_data()

            # Perform Chevron analysis and live plotting
            chevron = self.perform_chevron_analysis()
        return chevron

    

    def run_sweep(self, sweep_experiment_name):
        '''Run the sweep'''
        self.sweep_experiment_name = sweep_experiment_name
        self.map_sequential_cfg_to_experiment()

        if sweep_experiment_name == 'length_rabi_f0g1_sweep':
            return self.freq_sweep()


class MM_dual_rail_seq_exp:
    def __init__(self):
        '''Contains sequential experiments for dual rail based sequential experiments'''

    def DualRail_sweep_depth_and_single_spec_and_stor(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        '''
        This performs dual rail rb for a given target mode in presence of a single spectator. 
        This function sweeps the single spectator modes and also internally sweeps all the cardinal states 
        that the spectator mode can be in . 
        This function will also sweep the target modes
        '''
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   
        sweep_experiment_name = 'DualRail_sweep_depth_and_single_spec_and_stor'

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass
        
        start_pair = loaded[sweep_experiment_name]['start_pair']
        target_start, spec_start = start_pair
        
        for kdx, target_mode in enumerate(loaded[sweep_experiment_name]['target_mode_list']):
            if target_mode < target_start: continue

            print('----------------------############---------------------------')
            print('Kndex: %s target mode. = %s ' %(kdx, target_mode))
            loaded[sweep_experiment_name]['target_mode'] = target_mode
            mode_list = [1,2,3,4,5,6,7]
            mode_list.remove(target_mode)
            new_mode_list = mode_list.copy()
            for spec in mode_list: 
                if target_mode == target_start and spec < spec_start: 
                    new_mode_list.remove(spec)
            loaded[sweep_experiment_name]['target_spec_list'] = new_mode_list
            # print(new_mode_list)

            loaded[experiment_name]['bs_para'] = loaded[sweep_experiment_name]['bs_para_list'][kdx]
            print(loaded[experiment_name]['bs_para'])

            self.SingleBeamSplitterRBPostSelection_sweep_depth_and_single_spec(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                        prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name, yaml_cfg])
            





    def SingleBeamSplitterRBPostSelection_sweep_depth_and_single_spec(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, 
                                                                    prep_init = False, prep_params = None):
        '''
        This performs dual rail rb for a given target mode in presence of a single spectator. 
        This function sweeps the single spectator modes and also internally sweeps all the cardinal states 
        that the spectator mode can be in . 
        '''
        if prep_init: 
            config_path, loaded, experiment_class, experiment_name, sweep_experiment_name, yaml_cfg = prep_params
            # target_mode = loaded[sexperiment_name]['target_mode']
        else: 
            #====================================================================#
            config_path = config_file
            print('Config will be', config_path)

            with open(config_file, 'r') as cfg_file:
                yaml_cfg = yaml.safe_load(cfg_file)
            yaml_cfg = AttrDict(yaml_cfg)

            with open(exp_param_file, 'r') as file:
                # Load the YAML content
                loaded = yaml.safe_load(file)
            #===================================================================# 

            experiment_class = 'single_qubit.rb_BSgate_postselection'
            experiment_name = 'SingleBeamSplitterRBPostSelection'   
            sweep_experiment_name = 'SingleBeamSplitterRBPostSelection_sweep_depth_and_single_spec'


            for keys in loaded[experiment_name].keys():
                try:
                    loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
                except:
                    pass
            
        
        target_mode = loaded[sweep_experiment_name]['target_mode']

        #depth_array = np.array([1,2,3,4,5,10,20])
        for jdx, target_spec in enumerate(loaded[sweep_experiment_name]['target_spec_list']):
        #for index, depth in enumerate(depth_array):
            print('-------------------------------------------------')
            print('Jndex: %s target spec. = %s ' %(jdx, target_spec))
            # loaded[experiment_name]['ram_prepulse'][1] = #num_occupied_smodes
            # loaded[experiment_name]['ram_prepulse'][3] = loaded[sweep_experiment_name]['prepulse_vars_list'][jdx] 

            dummy = MM_dual_rail_base(cfg = yaml_cfg)
            prepulse_strs = [dummy.prepulse_str_for_random_ram_state(1, [target_mode], target_spec, i) for i in range(1, 7)]
            print(prepulse_strs)
            loaded[experiment_name]['ram_prepulse_strs'] = prepulse_strs
            self.SingleBeamSplitterRBPostSelection_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                        prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name])
            


    def SingleBeamSplitterRBPostSelection_sweep_depth_and_ram(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   
        sweep_experiment_name = 'SingleBeamSplitterRBPostSelection_sweep_depth_and_ram'

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass

        #depth_array = np.array([1,2,3,4,5,10,20])
        for jdx, num_occupied_smodes in enumerate(loaded[sweep_experiment_name]['num_occupied_smodes_list']):
        #for index, depth in enumerate(depth_array):
            print('-------------------------------------------------')
            print('Jndex: %s depth. = %s ' %(jdx, num_occupied_smodes))
            loaded[experiment_name]['ram_prepulse'][1] = num_occupied_smodes
            loaded[experiment_name]['ram_prepulse'][3] = loaded[sweep_experiment_name]['prepulse_vars_list'][jdx] 

            self.SingleBeamSplitterRBPostSelection_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                        prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name])

    def SingleBeamSplitterRB_stor_ramsey_spec(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        '''
        Depth sweep over all storage-storage pairs  for ramsey inpresence of beamsplitters
        '''
        #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
        #===================================================================# 
        experiment_class = 'single_qubit.rb_BSgate_check_target'
        experiment_name = 'SingleBeamSplitterRB_check_target'   
        sweep_experiment_name = 'SingleBeamSplitterRB_stor_ramsey_spec'

        # for keys in loaded[experiment_name].keys():
        #     try:
        #         loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
        #     except:
        #         pass

        for idx, stor_no in enumerate(loaded[sweep_experiment_name]['stor_list']):
            for jdx, spec_no in enumerate(loaded[sweep_experiment_name]['spec_list']):
                if stor_no == spec_no: # no self -self pair
                    continue
                if [stor_no, spec_no] in loaded[sweep_experiment_name]['skip_pairs']:
                    continue
                print('-------------------------------------------------')
                print('Index: %s Storage = %s, Spectator = %s ' %(idx, stor_no, spec_no))
                # Frequency 
                loaded[sweep_experiment_name]['wait_freq'] = loaded[sweep_experiment_name]['wait_freq_list'][stor_no -1][spec_no -1]

                # update prepulse/post pulse
                loaded[sweep_experiment_name]['pre_sweep_pulse'][-1][1] = 'M1-S' + str(stor_no)
                loaded[sweep_experiment_name]['post_sweep_pulse'][0][1] = 'M1-S' + str(stor_no)

                # update bs_para
                loaded[sweep_experiment_name]['bs_para'] = loaded[sweep_experiment_name]['bs_para_list'][spec_no -1]

                # print(loaded[sweep_experiment_name])

                _ = self.SingleBeamSplitterRB_check_target_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                                  prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name])

    def SingleBeamSplitterRB_stor_ramsey_spec_for_sp_pairs(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        '''
        Depth sweep over all storage-storage pairs  for ramsey inpresence of beamsplitters
        for specific pairs
        '''
        #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
        #===================================================================# 
        experiment_class = 'single_qubit.rb_BSgate_check_target'
        experiment_name = 'SingleBeamSplitterRB_check_target'   
        sweep_experiment_name = 'SingleBeamSplitterRB_stor_ramsey_spec_for_sp_pairs'

        # for keys in loaded[experiment_name].keys():
        #     try:
        #         loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
        #     except:
        #         pass

        for (stor_no, spec_no) in loaded[sweep_experiment_name]['stor_spec_pairs']:
            if stor_no == spec_no: # no self -self pair
                continue
            # if [stor_no, spec_no] in loaded[sweep_experiment_name]['skip_pairs']:
            #     continue
            print('-------------------------------------------------')
            print(' Storage = %s, Spectator = %s ' %( stor_no, spec_no))
            # Frequency 
            loaded[sweep_experiment_name]['wait_freq'] = loaded[sweep_experiment_name]['wait_freq_list'][stor_no -1][spec_no -1]

            # update prepulse/post pulse
            loaded[sweep_experiment_name]['pre_sweep_pulse'][-1][1] = 'M1-S' + str(stor_no)
            loaded[sweep_experiment_name]['post_sweep_pulse'][0][1] = 'M1-S' + str(stor_no)

            # update bs_para
            loaded[sweep_experiment_name]['bs_para'] = loaded[sweep_experiment_name]['bs_para_list'][spec_no -1]

            # print(loaded[sweep_experiment_name])

            _ = self.SingleBeamSplitterRB_check_target_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                                prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name])



    def SingleBeamSplitterRB_check_target_sweep_depth(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, prep_init = False, prep_params = None):
        '''
            Although this function is uses the daughter function SingleBeamSplitterRBPostSelection, 
            the post selection part is unimportant. 

            This is gate based ramsey experiment (instead of time based) for target state in presence
            spectator beamsplitters.
        '''
    # #====================================================================#
    #     config_path = config_file
    #     print('Config will be', config_path)

    #     with open(config_file, 'r') as cfg_file:
    #         yaml_cfg = yaml.safe_load(cfg_file)
    #     yaml_cfg = AttrDict(yaml_cfg)

    #     with open(exp_param_file, 'r') as file:
    #         # Load the YAML content
    #         loaded = yaml.safe_load(file)
    # #===================================================================# 

    #     experiment_class = 'single_qubit.rb_BSgate_check_target'
    #     experiment_name = 'SingleBeamSplitterRB_check_target'   
    #     sweep_experiment_name = 'SingleBeamSplitterRB_check_target_sweep_depth'

    #     for keys in loaded[experiment_name].keys():
    #         try:
    #             loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
    #         except:
    #             pass
        if prep_init: 
            config_path, loaded, experiment_class, experiment_name, sweep_experiment_name = prep_params
        else: 
        #====================================================================#
            config_path = config_file
            print('Config will be', config_path)

            with open(config_file, 'r') as cfg_file:
                yaml_cfg = yaml.safe_load(cfg_file)
            yaml_cfg = AttrDict(yaml_cfg)

            with open(exp_param_file, 'r') as file:
                # Load the YAML content
                loaded = yaml.safe_load(file)
        #===================================================================# 

            experiment_class = 'single_qubit.rb_BSgate_check_target'
            experiment_name = 'SingleBeamSplitterRB_check_target'   
            sweep_experiment_name = 'SingleBeamSplitterRB_check_target_sweep_depth'

        # NOTe the following code is not part of the "prep function"
        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass
        
        loaded[sweep_experiment_name]['depth_list'] = np.arange(loaded[sweep_experiment_name]['depth_start'],
                                                                loaded[sweep_experiment_name]['depth_stop'],
                                                                loaded[sweep_experiment_name]['depth_step'])
        length = len(loaded[sweep_experiment_name]['depth_list'])
        loaded[sweep_experiment_name]['reps_list'] = [loaded[sweep_experiment_name]['repss'] for _ in range(length)] # * len(loaded[sweep_experiment_name]['depth_list'])

        # print(loaded[sweep_experiment_name])
        self.SingleBeamSplitterRBPostSelection_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                    prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name],
                                                    skip_ss = True)
            
    def SingleBeamSplitterRBPostSelection_sweep_depth(self,soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None,
                                                    prep_init = False, prep_params = None, skip_ss = False):
        '''
        Prep_init: True if the config, experiment names are already initialized in some other parent function that calls this as a child
        prep_params: [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name]
        skip_ss: Skip the single shot part of the experiment (True/False) (first depth will have it, later depths will not )
        '''
        if prep_init: 
            config_path, loaded, experiment_class, experiment_name, sweep_experiment_name = prep_params
        else: 
        #====================================================================#
            config_path = config_file
            print('Config will be', config_path)

            with open(config_file, 'r') as cfg_file:
                yaml_cfg = yaml.safe_load(cfg_file)
            yaml_cfg = AttrDict(yaml_cfg)

            with open(exp_param_file, 'r') as file:
                # Load the YAML content
                loaded = yaml.safe_load(file)
        #===================================================================# 

            experiment_class = 'single_qubit.rb_BSgate_postselection'
            experiment_name = 'SingleBeamSplitterRBPostSelection'  
            sweep_experiment_name = 'SingleBeamSplitterRBPostSelection_sweep_depth' 

            for keys in loaded[experiment_name].keys():
                try:
                    loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
                except:
                    pass

        #depth_array = np.array([1,2,3,4,5,10,20])
        for index, depth in enumerate(loaded[sweep_experiment_name]['depth_list']):
        #for index, depth in enumerate(depth_array):
            print('Index: %s depth. = %s ' %(index, depth))
            if index != 0 and skip_ss:
                loaded[experiment_name]['calibrate_single_shot'] = False
            loaded[experiment_name]['rb_depth'] = depth
            loaded[experiment_name]['rb_reps'] = loaded[sweep_experiment_name]['reps_list'][index]
            

            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            print(run_exp.cfg.expt)
            run_exp.go(analyze=False, display=False, progress=False, save=True)

    def SingleBeamSplitterRBPostSelection_sweep_depth_storsweep(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep'][keys]   # overwrite the single experiment file with new paramters
            except:
                pass

        
        for stor_idx, stor_no in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep']['stor_list']): 

            print('-------------------------------------------------')
            print('Storage Index: %s Storage No. = %s ' %(stor_idx, stor_no))
            man_idx = 2   # 1 or 2

            # create prepulse , postpulse, post selection pulse 
            mm_base = MM_base(cfg = yaml_cfg)
            pre_sweep_pulse_str = [['qubit', 'ge', 'pi'],
                            ['qubit', 'ef', 'pi'],
                                ['man', 'M' + str(man_idx) , 'pi']]
            post_sweeep_pulse_str = [['qubit', 'ge', 'hpi'], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx)], 
                        ['qubit', 'ge', 'hpi']]
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no), 'hpi'], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no), 'hpi'],
                            ['qubit', 'ge', 'hpi'], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx)], 
                            ['qubit', 'ge', 'hpi']]
            bs_para_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no), 'hpi']]
            
            creator = mm_base.get_prepulse_creator(pre_sweep_pulse_str)
            loaded[experiment_name]['pre_sweep_pulse'] = creator.pulse.tolist()
            creator = mm_base.get_prepulse_creator(post_sweeep_pulse_str)
            loaded[experiment_name]['post_sweep_pulse'] = creator.pulse.tolist()
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            loaded[experiment_name]['post_selection_pulse'] = creator.pulse.tolist()
            creator = mm_base.get_prepulse_creator(bs_para_str)
            bs_para_pulse = creator.pulse.tolist()
            loaded[experiment_name]['bs_para'] = [bs_para_pulse[0][0], bs_para_pulse[1][0], bs_para_pulse[2][0],  bs_para_pulse[6][0]]

            print('Prepulse: ', loaded[experiment_name]['pre_sweep_pulse'])
            print('Postpulse: ', loaded[experiment_name]['post_sweep_pulse'])
            print('Post Selection Pulse: ', loaded[experiment_name]['post_selection_pulse'])
            print('BS Para: ', loaded[experiment_name]['bs_para'])
            
            
            
            #depth_array = np.array([1,2,3,4,5,10,20])
            for index, depth in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep']['depth_list']):
            #for index, depth in enumerate(depth_array):
                print('Index: %s depth. = %s ' %(index, depth))
                loaded[experiment_name]['rb_depth'] = depth

                loaded[experiment_name]['rb_reps'] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep']['reps_list'][index]
                

                run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


                run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

                # special updates on device_config file
                #run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
                # run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
                # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
                # run_exp.cfg.device.manipulate.readout_length = 5
                # run_exp.cfg.device.storage.readout_length = 5
                run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
                print(run_exp.cfg.expt)
                run_exp.go(analyze=False, display=False, progress=False, save=True)




    def SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep'][keys]   # overwrite the single experiment file with new paramters
            except:
                pass

        
        for stor_idx, stor_no in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['stor_list']): 

            print('-------------------------------------------------')
            print('Storage Index: %s Storage No. = %s ' %(stor_idx, stor_no))
            man_idx = 1   # 1 or 2

            # create prepulse , postpulse, post selection pulse 
            mm_base = MM_base(cfg = yaml_cfg)
            pre_sweep_pulse_str = [['qubit', 'ge', 'pi', 0],
                            ['qubit', 'ef', 'pi', 0],
                                ['man', 'M' + str(man_idx) , 'pi', 0]]
            
            creator = mm_base.get_prepulse_creator(pre_sweep_pulse_str)
            loaded[experiment_name]['pre_sweep_pulse'] = creator.pulse.tolist()
            loaded[experiment_name]['bs_para'] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['bs_para_list'][stor_no-1]

            print('Prepulse: ', loaded[experiment_name]['pre_sweep_pulse'])
            print('BS Para: ', loaded[experiment_name]['bs_para'])

            for index, depth in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['depth_list']):
                print('Index: %s depth. = %s ' %(index, depth))
                loaded[experiment_name]['rb_depth'] = depth

                loaded[experiment_name]['rb_reps'] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['reps_list'][index]
                

                run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


                run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

                # special updates on device_config file
                run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
                print(run_exp.cfg.expt)
                run_exp.go(analyze=False, display=False, progress=False, save=True)
