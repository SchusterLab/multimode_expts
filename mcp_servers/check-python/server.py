from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import os

# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server object
mcp = FastMCP()

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


# If you want to run initialization on import:
if __name__ == "__main__":
    os.system("conda activate slab")
    print('Activated slab conda environment')
    return_args = initialize_server()
    

