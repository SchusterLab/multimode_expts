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

from dataset import storage_man_swap_dataset

class qsim_base_class():
    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        self.soccfg = soccfg
        self.path = path
        self.prefix = prefix
        self.config_file = config_file
        self.exp_param_file = exp_param_file

        #load parameter files
        self.load_config()
        self.load_exp_param()


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
    
    

class floquet_swap_class(qsim_base_class): 
    '''Base class for floquet swapping experiments '''

    def __init__(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        super().__init__(soccfg, path, prefix, config_file, exp_param_file)
        self.experiment_class = None
        self.experiment_name = None
        # create prepulse , postpulse, post selection pulse 
        self.mm_base = MM_base(cfg = self.yaml_cfg)


    def storage_sweep(self):
        '''Sweep which storage mode to read out'''
        self.experiment_class = 'qsim.floquet_general'
        self.experiment_name = 'FloquetGeneralExperiment'
        self.sweep_experiment_name = 'storage_sweep'
        # self.map_sequential_cfg_to_experiment()

        for floquet_cycles in range(1,501,5):
            self.loaded[self.experiment_name]['floquet_cycles'] = floquet_cycles
            print('Loaded: ', self.loaded[self.experiment_name])
            
            run_exp = eval(f"meas.{self.experiment_class}.{self.experiment_name}(soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file)")
            run_exp.cfg.expt = self.loaded[self.experiment_name]

            # special updates on device_config file
            run_exp.cfg.device.readout.relax_delay = 5000 # Wait time between experiments [us]
            print('Config is: ', run_exp.cfg.expt)
            run_exp.go(analyze=False, display=False, progress=True, save=True)
        

    def run_sweep(self, sweep_experiment_name):
        '''Run the sweep'''
        # self.sweep_experiment_name = sweep_experiment_name
        # self.map_sequential_cfg_to_experiment()

        if sweep_experiment_name == 'storage_sweep':
            self.storage_sweep()

    
