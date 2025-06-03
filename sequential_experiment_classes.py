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
    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, config_thisrun=None
                 ):
        self.soccfg = soccfg
        self.path = path
        self.prefix = prefix
        self.config_file = config_file
        self.exp_param_file = exp_param_file
        self.config_thisrun = config_thisrun # asnytime you run the expt, copy over this config file!!!

        #load parameter files
        self.load_config()
        self.load_exp_param()

        pass

    def run_with_configthisrun(self, run_expt):
        """
        Run the experiment with a specific configuration for this run.
        
        Parameters:
            config_thisrun (dict): Configuration parameters for this run.
        run_expt (object): The experiment object to run.
        
        Returns:
            None
        """
        
        if self.config_thisrun is not None:
            run_expt.cfg = self.config_thisrun
            print(f"Running experiment with config: configthisrun")
        else:
            print("yoohoo, no config_thisrun provided, using default config")
        return run_expt


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
    
    def initialize_expt_sweep(self, keys=None, create_directory=False):
        """
        Initialize the experiment sweep data structure with specified keys.

        Parameters:
            keys (list of str, optional): List of keys to initialize in the experiment sweep data structure.
                If None, defaults to an empty dictionary.
            create_directory (bool, optional): If True, creates a new directory for storing sweep data files.
                This option is different from usual way of saving all data ins single file. Can be useful if 
                data is inhomogenous.              Thi

        Side Effects:
            - Initializes self.expt_sweep as an Experiment instance.
            - Sets self.expt_sweep.data to a dictionary with the specified keys, each mapped to an empty list.
            - If create_directory is True, creates a directory (named after the experiment file, without .h5 extension)
              for storing sweep data files.
        """
        self.expt_sweep = Experiment(
            path=self.path,
            prefix=self.sweep_experiment_name,
            config_file=self.config_file,
        )
        if keys is None:
            self.expt_sweep.data = {}
        else:
            self.expt_sweep.data = {key: [] for key in keys}
        if create_directory:
            directory = self.expt_sweep.fname
            if directory.lower().endswith('.h5'):
                directory = directory[:-3]
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            self.expt_sweep_dir_name = directory

    def save_sweep_data(self, sweep_key, sweep_value, run_exp, skip_keys=None, 
                        save_data=True):
        """
        Save sweep data and experimental results to the experiment file.

        Parameters:
            sweep_key (str): The key identifying the sweep parameter.
            sweep_value (Any): The value of the sweep parameter for the current run.
            run_exp (object): An object containing experimental data in its `data` attribute.
            skip_keys (list, optional): List of data keys to skip when saving. Defaults to None.

        Side Effects:
            - Appends the sweep value to the corresponding list in `self.expt_sweep.data`.
            - For each key in `run_exp.data`, appends its value to the corresponding list in `self.expt_sweep.data`, unless in skip_keys.
            - Initializes new lists in `self.expt_sweep.data` for any new data keys.
            - Persists the updated data by calling `self.expt_sweep.save_data()`.
        """
        if skip_keys is None:
            skip_keys = []
        self.expt_sweep.data[sweep_key].append(sweep_value)
        
        if save_data:
            for data_key in run_exp.data.keys():
                if data_key in skip_keys:
                    print(f'Skipping key: {data_key}')
                    continue
                if data_key not in self.expt_sweep.data.keys():
                    self.expt_sweep.data[data_key] = []
                self.expt_sweep.data[data_key].append(run_exp.data[data_key])
        print(self.expt_sweep)
        self.expt_sweep.save_data()

   
    
    

class sidebands_class(sequential_base_class):
    '''Class for sideband experiments; using sideband general experiment'''

    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, config_thisrun = None):
        super().__init__(soccfg, path, prefix, config_file, exp_param_file, config_thisrun=config_thisrun)
        self.experiment_class = 'single_qubit.sideband_general'
        self.experiment_name = 'SidebandGeneralExperiment'

    

    def sideband_freq_sweep(self):
        '''Frequency sweep for sideband experiments'''
        self.initialize_expt_sweep(keys = ['freq_sweep'])
        chevron = None

        for index, freq in enumerate(np.arange(self.loaded[self.sweep_experiment_name]['freq_start'], 
                                                self.loaded[self.sweep_experiment_name]['freq_stop'], 
                                                self.loaded[self.sweep_experiment_name]['freq_step'])):

            print('Index: %s Freq. = %s MHz' % (index, freq))
            self.loaded[self.experiment_name]['flux_drive'][1] = freq

            run_exp = eval(f"meas.{self.experiment_class}.{self.experiment_name}(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file)")
            run_exp = self.run_with_configthisrun(run_exp)  # Use the config_thisrun if provided
            
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
    

    

    


class man_f0g1_class(sequential_base_class):
    '''Class for length Rabi f0g1 sweep experiments'''

    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, 
                 config_thisrun=None):
        
        super().__init__(soccfg, path, prefix, config_file, exp_param_file, config_thisrun=config_thisrun)
        self.experiment_class = 'single_qubit.length_rabi_f0g1_general'
        self.experiment_name = 'LengthRabiGeneralF0g1Experiment'

    def freq_sweep(self):
        '''Frequency sweep for length Rabi f0g1'''
        self.initialize_expt_sweep(keys = ['freq_sweep'])
        chevron = None

        for index, freq in enumerate(np.arange(self.loaded[self.sweep_experiment_name]['freq_start'], 
                                                self.loaded[self.sweep_experiment_name]['freq_stop'], 
                                                self.loaded[self.sweep_experiment_name]['freq_step'])):

            print('Index: %s Freq. = %s GHz' % (index, freq))
            self.loaded[self.experiment_name]['freq'] = freq

            run_exp = eval(f"meas.{self.experiment_class}.{self.experiment_name}(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file)")
            run_exp = self.run_with_configthisrun(run_exp)  # Use the config_thisrun if provided
            run_exp.cfg.expt = eval(f"self.loaded['{self.experiment_name}']")

            if self.loaded[self.experiment_name]['active_reset']:
                print('Doesn’t make sense to active reset in this experiment')
            run_exp.cfg.device.readout.relax_delay = 2500  # Wait time between experiments [us]
            print(run_exp.cfg.expt)

            run_exp.go(analyze=False, display=False, progress=False, save=False)
            # return run_exp

            # Add entry to sweep file
            self.save_sweep_data('freq_sweep', freq, run_exp)


            # Perform Chevron analysis and live plotting
            chevron = self.perform_chevron_analysis()
        return chevron

    

    def run_sweep(self, sweep_experiment_name):
        '''Run the sweep'''
        self.sweep_experiment_name = sweep_experiment_name
        self.map_sequential_cfg_to_experiment()

        if sweep_experiment_name == 'length_rabi_f0g1_sweep':
            return self.freq_sweep()



class MM_DualRailRB(sequential_base_class):
    '''Class for dual rail based sequential experiments (RB)'''

    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None,
                 prev_data = None, title = None, config_thisrun=None):
        super().__init__(soccfg, path, prefix, config_file, exp_param_file, config_thisrun=config_thisrun)
        self.experiment_class = 'single_qubit.rb_BSgate_postselection'
        self.experiment_name = 'SingleBeamSplitterRBPostSelection'
        self.prev_data = prev_data
        self.title = title if title is not None else 'Dual Rail RB Post Selection'


    def get_reps(self, depth):
        """
        Rough estimation, need to check with paper and fix this"""
        if depth <250: 
            return 2500
        elif depth < 400:
            return 5000
        else:
            return 10000

    def SingleBeamSplitterRBPostSelection_sweep_depth(self, skip_ss=False):
        '''Sweep over RB depths for dual rail experiment'''
        # all_keys = [
        #     'depth_sweep', 'reps_sweep', 'Idata', 'Qdata', 'Ig', 'Qg', 'Ie', 'Qe',
        #     'fids', 'thresholds', 'angle', 'confusion_matrix', 'sequences',
        # ]
        all_keys = ['filenames']
        self.initialize_expt_sweep(keys=all_keys, create_directory = True)
        path_for_expt = self.expt_sweep_dir_name
        

        for index, depth in enumerate(self.loaded[self.sweep_experiment_name]['depth_list']):
            print('Index: %s depth. = %s ' % (index, depth))
            if index != 0 and skip_ss:
                self.loaded[self.experiment_name]['calibrate_single_shot'] = False
            self.loaded[self.experiment_name]['rb_depth'] = depth
            self.loaded[self.experiment_name]['rb_reps'] = self.get_reps(depth)

            run_exp = eval(f"meas.{self.experiment_class}.{self.experiment_name}(soccfg=self.soccfg, path=path_for_expt, prefix=self.prefix, config_file=self.config_file)")
            run_exp = self.run_with_configthisrun(run_exp)
            run_exp.cfg.expt = eval(f"self.loaded['{self.experiment_name}']")
            # self.expt_sweep.cfg = run_exp.cfg  # Copy the config to the sweep experiment
            #TODO: add expt config to sweep experiment data saving
            print(run_exp.cfg.expt)
    
            run_exp.go(analyze=False, display=False, progress=False, save=True)

            
            # The following is just for making code saving easier!
            run_exp.cfg.expt.running_list = []
            filename = run_exp.fname
            
            # print('run_exp.fname', os.path.basename(run_exp.fname)) # h5py can't store strings 
            # self.save_sweep_data('filenames', os.path.basename(run_exp.fname), run_exp, save_data = False)
            path_for_expt = path_for_expt
            self.perform_RB_analysis(prefix = self.sweep_experiment_name, dir_path = path_for_expt)
        
        return self.sweep_experiment_name, path_for_expt
    
    def perform_RB_analysis(self, prefix = None, dir_path = None):
        """
        Perform RB analysis and live plotting.
        This method performs a Randomized Benchmarking (RB) analysis on the experimental data 
        and displays the results in a live plot. It assumes that the `self.expt_sweep` attribute 
        is present and contains the necessary data for the analysis. Specifically, `self.expt_sweep.data` 
        must include:
            - 'depth_sweep': Depths used in the sweep.
            - 'sequences': Sequences of operations for each depth.
            - Other relevant RB data.
        The method initializes a `RBAnalysis` object with the provided data, performs the analysis, 
        and displays the results. It also clears any previous plots to ensure the display is updated 
        with the latest results.
        Note:
            - This method requires the `multimode_expts.fit_display_classes` module for the `RBAnalysis` class.
            - The `IPython.display.clear_output` function is used to clear the output for live plotting.
            - If any exception occurs during the process, it will print an error message indicating the failure.
        
        Parameters:
            prefix (str, optional): Prefix for the RB expt_sweep file. If None, defaults to "RBAnalysis".
            dir_path (str, optional): Directory path where the data files are stored. If None, uses the current directory.
        Raises:
            Exception: If any error occurs during the RB analysis or plotting process.
        """
        
        try:
            from multimode_expts.fit_display_classes import MM_DualRailRBFitting

            # Initialize RB analysis
            # print all args to rb analysis 
            # Close previous plots and display the new one
            from IPython.display import clear_output

            clear_output(wait=True)
            plt.close('all')  # Close all existing figures
            print("RBAnalysis args:")
            print(f"  filename: None")
            print(f"  file_prefix: {prefix}")
            print(f"  config: {self.yaml_cfg}")
            print(f"  expt_path: {self.path}")
            print(f"  title: {self.title}")
            print(f"  prev_data: {self.prev_data}")
            print(f"  dir_path: {dir_path}")
            rb_analysis = MM_DualRailRBFitting(filename = None, file_prefix = prefix, 
                                   config=self.yaml_cfg, expt_path=self.path, title=self.title, 
                                   prev_data= self.prev_data, dir_path=dir_path)

            

            
            args = rb_analysis.show_rb()
            return None

        except Exception as e:
            print(f"RB analysis failed: {e}")
            return None
    def run_sweep(self, sweep_experiment_name, skip_ss=False):
        '''Run the sweep'''
        self.sweep_experiment_name = sweep_experiment_name
        self.map_sequential_cfg_to_experiment()

        if sweep_experiment_name == 'SingleBeamSplitterRBPostSelection_sweep_depth':
            return self.SingleBeamSplitterRBPostSelection_sweep_depth(skip_ss=skip_ss)

    
    
        
   