""""
Shared environment setup for notebooks running multimode experiments.
1. Importing the necessary libraries
2. initializing qick 
3. Initializing important paths (data, config, etc)
4. Getting the CSV datasets
"""

import numpy as np
import matplotlib.pyplot as plt

from qick import QickConfig
from tqdm.notebook import tqdm

import time
from copy import deepcopy
import os
import sys
sys.path.append('/home/xilinx/jupyter_notebooks/')
sys.path.append('C:\\_Lib\\python\\rfsoc\\rfsoc_multimode\\example_expts')
# sys.path.append('C:\\_Lib\\python\\multimode')
import scipy as sp
import json

from slab.instruments import *
from slab.experiment import Experiment
from slab.datamanagement import SlabFile
from slab import get_next_filename, get_current_filename, AttrDict
import yaml

from experiments.dataset import *



class MultimodeStation:
    def __init__(self, data_path=None, config_name='hardware_config_202505.yml', exp_param_name='experiment_config.yml', qubit_i=0):
        self.path = data_path or r'H:\Shared drives\SLab\Multimode\experiment\250505_craqm'
        self.expt_path = os.path.join(self.path, 'data')  # Bad labveling here ; this is the data 
        self.data_path = self.expt_path  # This is the data path
        self.mm_expts_path = 'C:\\_Lib\\python\\multimode_expts'
        self.config_file = os.path.join(self.mm_expts_path, 'configs', config_name)
        self.exp_param_file = os.path.join(self.mm_expts_path, 'configs', exp_param_name)
        self.qubit_i = qubit_i

        self._print_paths()

        self.im = self._init_instrument_manager()

        self.expts_path = self._add_expts_path(self.mm_expts_path)
        self.yaml_cfg = self._load_yaml_config(self.config_file)
        self.soc = self._init_qick_config()
        self.meas = self._import_experiments_module()

        # Config for this instance (deepcopy of yaml_cfg)
        self.config_thisrun = AttrDict(deepcopy(self.yaml_cfg))

        # load the multiphoton config
        with open(self.config_thisrun.device.multiphoton_config.file, 'r') as f:
            self.multimode_cfg = yaml.safe_load(f)

        # Initailize the dataset
        ds, ds_thisrun, ds_thisrun_file_path = self.load_storage_man_swap_dataset()
        self.ds_thisrun = ds_thisrun

        # Path for autocalibration plots
        self.autocalib_path = self.create_autocalib_path()

        # For config update logic
        self.updateConfig_bool = False

    def _print_paths(self):
        print("path: ", self.path)
        print('Data will be stored in', self.expt_path)
        print('Hardware configs will be read from', self.config_file)
        print('Experiment params will be read from', self.exp_param_file)

    def _load_yaml_config(self, config_file):
        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)

        return AttrDict(yaml_cfg)

    def _init_instrument_manager(self):
        im = InstrumentManager(ns_address='192.168.137.25')
        print(im['Qick101'])
        return im

    def _init_qick_config(self):
        soc = QickConfig(self.im[self.yaml_cfg['aliases']['soc']].get_cfg())
        print(soc)
        return soc

    def _add_expts_path(self, expts_path):
        sys.path.insert(0, expts_path)
        print('Path added at highest priority')
        print(sys.path)
        return expts_path

    def _import_experiments_module(self):
        self._add_expts_path(self.mm_expts_path)
        import experiments as meas
        print('Importing experiments module from', self.expts_path)
        print(meas.__file__)
        return meas

    def prev_data(self, filename=None, prefix=None):
        if prefix is not None:
            temp_data_file = os.path.join(self.expt_path, get_current_filename(self.expt_path, prefix=prefix, suffix='.h5'))
        else:
            temp_data_file = self.expt_path + '\\' + filename
        with SlabFile(temp_data_file) as a:
            attrs = dict()
            for key in list(a.attrs):
                attrs.update({key: json.loads(a.attrs[key])})
            keys = list(a)
            temp_data = dict()
            for key in keys:
                temp_data.update({key: np.array(a[key])})
        return temp_data, attrs, temp_data_file

    def load_storage_man_swap_dataset(self):
        file_path = os.path.join(self.expts_path, 'man1_storage_swap_dataset.csv')
        ds = StorageManSwapDataset(file_path)
        ds_thisrun = StorageManSwapDataset(ds.create_copy())
        ds_thisrun_file_path = os.path.join(self.expts_path, ds_thisrun.filename)
        return ds, ds_thisrun, ds_thisrun_file_path

    def create_autocalib_path(self):
        """
        Creates a directory inside the data folder for autocalibration plots, named with the current date.
        Returns the path to the created directory.
        """
        autocalib_path = os.path.join(self.expt_path, f'autocalibration_plots_{datetime.now().strftime("%Y-%m-%d")}')
        os.makedirs(autocalib_path, exist_ok=True)
        print('Directory created for autocalibration plots at:', autocalib_path)
        return autocalib_path
        """
        Creates a directory inside the data folder for autocalibration plots, named with the current date.
        Returns the path to the created directory.
        """
        autocalib_path = os.path.join(self.expt_path, f'autocalibration_plots_{datetime.now().strftime("%Y-%m-%d")}')
        os.makedirs(autocalib_path, exist_ok=True)
        print('Directory created for autocalibration plots at:', autocalib_path)
        return autocalib_path

    def convert_attrdict_to_dict(self, attrdict):
        """
        Recursively converts an AttrDict or a nested dictionary into a standard Python dictionary.
        Converts np.float64 values to standard Python float.
        """
        if isinstance(attrdict, AttrDict):
            return {key: self.convert_attrdict_to_dict(value) for key, value in attrdict.items()}
        elif isinstance(attrdict, dict):
            return {key: self.convert_attrdict_to_dict(value) for key, value in attrdict.items()}
        elif isinstance(attrdict, np.float64):
            return float(attrdict)
        else:
            return attrdict

    def convert_numbers_to_float(self, data):
        """
        Recursively converts all numbers in a dictionary to float.
        """
        if isinstance(data, dict):
            return {key: self.convert_numbers_to_float(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_numbers_to_float(item) for item in data]
        elif isinstance(data, float):
            return float(data)
        elif isinstance(data, int):
            return int(data)
        else:
            return data

    def recursive_compare(self, d1, d2, path=""):
        """
        Recursively compares two dictionaries and prints differences.
        """
        for key in d1.keys():
            current_path = f"{path}.{key}" if path else key
            if key not in d2:
                print(f"Key '{current_path}' is missing in config2.")
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                self.recursive_compare(d1[key], d2[key], current_path)
            elif d1[key] != d2[key]:
                print(f"Key '{current_path}' differs:")
                # if isinstance(d1[key], list) and len(d1[key]) == 1:
                #     print(f"  Old value (config1): {d1[key][0]}")
                #     print(f"  New value (config2): {d2[key][0]}")
                # else:
                print(f"  Old value (config1): {d1[key]}")
                print(f"  New value (config2): {d2[key]}")
        for key in d2.keys():
            current_path = f"{path}.{key}" if path else key
            if key not in d1:
                print(f"Key '{current_path}' is missing in config1.")

    def update_yaml_config(self, yaml_cfg, config_thisrun):
        """
        Update the yaml_cfg with values from config_thisrun, excluding the storage_man_file.
        """
        updated_config = deepcopy(config_thisrun)
        updated_config.device.storage.storage_man_file = yaml_cfg.device.storage.storage_man_file
        return updated_config

    def save_configurations(self, yaml_cfg, config_thisrun, autocalib_path, config_path):
        """
        Save the old and updated configurations to their respective files.
        """
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        old_config_path = os.path.join(autocalib_path, f'old_config_{current_time}.yaml')
        old_config = self.convert_numbers_to_float(self.convert_attrdict_to_dict(yaml_cfg))
        with open(old_config_path, 'w') as cfg_file:
            yaml.dump(old_config, cfg_file, default_flow_style=False, indent=4, width=80, canonical=False,
                      explicit_start=True, explicit_end=False, sort_keys=False, line_break=True)

        updated_config = self.convert_numbers_to_float(self.convert_attrdict_to_dict(self.update_yaml_config(yaml_cfg, config_thisrun)))
        with open(config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False, indent=4, width=80, canonical=False,
                      explicit_start=True, explicit_end=False, sort_keys=False, line_break=True)

    def handle_config_update(self, updateConfig_bool=False):
        """
        Main logic for comparing, updating, and saving configuration files.
        Only does config this run 
        """
        print("Comparing configurations:")
        self.recursive_compare(self.yaml_cfg, self.config_thisrun)
        autocalib_path = self.create_autocalib_path()
        config_path = self.config_file
        updated_config = self.update_yaml_config(self.yaml_cfg, self.config_thisrun)
        if updateConfig_bool:
            self.save_configurations(self.yaml_cfg, updated_config, autocalib_path, config_path)
            self.yaml_cfg = updated_config
            print("Configuration updated and saved, excluding storage_man_file. \n!!!!Please set updateConfig to False after this run!!!!!!.")        
# not properly coded 
    # def handle_multiphoton_config_update(self, updateConfig_bool=False):
    #     """
    #     Main logic for comparing, updating, and saving configuration files.
    #     Only does config this run 
    #     """
    #     print("Comparing configurations:")
    #     self.recursive_compare(self.yaml_cfg, self.config_thisrun)
    #     autocalib_path = self.create_autocalib_path()
    #     config_path = self.config_file
    #     updated_config = self.update_yaml_config(self.yaml_cfg, self.config_thisrun)
    #     if updateConfig_bool:
    #         self.save_configurations(self.yaml_cfg, updated_config, autocalib_path, config_path)
    #         self.yaml_cfg = updated_config
    #         print("Configuration updated and saved, excluding storage_man_file. \n!!!!Please set updateConfig to False after this run!!!!!!.")        



