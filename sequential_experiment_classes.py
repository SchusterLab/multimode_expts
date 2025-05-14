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
            run_exp.cfg.device.readout.relax_delay = 5000
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