import matplotlib
from copy import deepcopy
from slab import AttrDict

def setup_paths():
    """
    Modifies the Python module search path (sys.path) by appending and inserting specific directories.

    - Appends '/home/xilinx/jupyter_notebooks/' and 'C:\\_Lib\\python\\rfsoc\\rfsoc_multimode\\example_expts' to sys.path.
    - Inserts 'C:\\_Lib\\python\\multimode_expts' at the highest priority (beginning) of sys.path.
    - Prints confirmation messages and the updated sys.path.
    
    Returns:
        sys (module): The sys module with the updated path.
    """
    import sys
    sys.path.append('/home/xilinx/jupyter_notebooks/')
    sys.path.append('C:\\_Lib\\python\\rfsoc\\rfsoc_multimode\\example_expts')
    # sys.path.append('C:\\_Lib\\python\\multimode')
    expts_path = 'C:\\_Lib\\python\\multimode_expts'
    sys.path.insert(0, expts_path)
    print('Path added at highest priority')
    print(sys.path)
    return sys

def setup_plotting():
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [10,6]
    plt.rcParams.update({'font.size': 14})
    return plt

def get_config_paths():
    import os
    curr_path = "C:\_Lib\python\multimode_expts"
    config_file = os.path.join(curr_path, 'configs', 'hardware_config_202505.yml')
    exp_param_file = os.path.join(curr_path, 'configs', 'experiment_config.yml')
    return config_file, exp_param_file

def load_yaml_config(config_file):
    import yaml
    from slab import AttrDict
    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    return AttrDict(yaml_cfg)

def setup_instruments(yaml_cfg):
    from slab.instruments import InstrumentManager
    from qick import QickConfig
    im = InstrumentManager(ns_address='192.168.137.25') # SLAC lab
    print(im['Qick101'])
    soc = QickConfig(im[yaml_cfg['aliases']['soc']].get_cfg())
    print(soc)
    return im, soc

def setup_experiment_data_path():
    import os
    path = r'H:\Shared drives\SLab\Multimode\experiment\250505_craqm'
    print("path: ", path)
    expt_path = os.path.join(path, 'data')
    print('Data will be stored in', expt_path)
    return expt_path

def print_config_paths(config_file, exp_param_file):
    print('Hardware configs will be read from', config_file)
    print('Experiment params will be read from', exp_param_file)

def import_experiments_module():
    import experiments as meas
    print(meas.__file__)
    return meas

def initialize_server():
    sys = setup_paths()
    plt = setup_plotting()
    expt_path = setup_experiment_data_path()
    config_file, exp_param_file = get_config_paths()
    print_config_paths(config_file, exp_param_file)
    yaml_cfg = load_yaml_config(config_file)
    im, soc = setup_instruments(yaml_cfg)
    meas = import_experiments_module()
    return {
        "sys": sys,
        "plt": plt, #???
        "expt_path": expt_path,
        "config_file": config_file,
        "exp_param_file": exp_param_file,
        "yaml_cfg": yaml_cfg,
        "im": im,
        "soc": soc,
        "meas": meas
    }

class run_exp():
    def __init__(self, initialization_dict):
        self.sys = initialization_dict['sys']
        self.plt = initialization_dict['plt']
        self.expt_path = initialization_dict['expt_path']
        self.config_file = initialization_dict['config_file']
        self.exp_param_file = initialization_dict['exp_param_file']
        self.yaml_cfg = initialization_dict['yaml_cfg']
        self.im = initialization_dict['im']
        self.soc = initialization_dict['soc']
        self.meas = initialization_dict['meas']
        print('run_exp initialized with the following parameters:')
    
    def do_single_shot(self, config_thisrun = None, expt_path=None, config_path=None):
        """Run the single shot experiment."""
        if config_thisrun is None:
            config_thisrun = self.yaml_cfg
        if expt_path is None:
            expt_path = self.expt_path
        if config_path is None:
            config_path = self.config_file

        hstgrm = self.meas.single_qubit.single_shot.HistogramExperiment(
            soccfg=self.soc, path=expt_path, prefix='HistogramExperiment', config_file=config_path
        )
        print('Running single shot experiment with the following parameters:')

        hstgrm.cfg = AttrDict(deepcopy(config_thisrun))

        hstgrm.cfg.expt = {
            'qubits': [0],
            'reps': 5000,
            'check_f': False,
            'active_reset': False,
            'man_reset': False,
            'storage_reset': False,
            'qubit': 0,
            'pulse_manipulate': False,
            'cavity_freq': 4984.373226159381,
            'cavity_gain': 800,
            'cavity_length': 2,
            'prepulse': False,
            'pre_sweep_pulse': [
            ['qubit', 'ge', 'pi', 0],
            ['qubit', 'ge', 'pi', 0]
            ],
            'gate_based': True,
        }

        hstgrm.cfg.device.readout.relax_delay = [2500]  # Wait time between experiments [us]
        try:
            hstgrm.go(analyze=False, display=False, progress=True, save=True)
        except Exception as e:
            print(f"An error occurred during hstgrm.go: {e}")
            return None
        

        threshold = config_thisrun.device.readout.threshold[0]
        hist_analysis = Histogram(
            hstgrm.data, verbose=True, active_reset=False, 
            readout_per_round=4, span=800, threshold=threshold, config=config_thisrun,
        )
        print('Done Histogram')
        hist_analysis.analyze(plot = True)
        return hist_analysis
        
    



# If you want to run initialization on import:
if __name__ == "__main__":
    print('Welcome to the Server')
    return_args = initialize_server()
    print('expt path is ', return_args['expt_path'])
    run_exp_instance = run_exp(return_args)
    run_exp_instance.do_single_shot()

    

