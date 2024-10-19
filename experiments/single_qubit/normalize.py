import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
import time

import experiments.fitting as fitter


from experiments.single_qubit.length_rabi_general import LengthRabiGeneralExperiment

def normalize_calib(soc, expt_path, config_path):
    """
    Returns two dictionaries containing g and e data respectively 
    """ 

    cfg_expt = {'start': 0.01, # us
                'step': 0.007, # us
                'qubits': [0],
                'expts': 1,
                'reps': 100,
                'rounds': 10,
                'gain': 0 ,# qubit gain [DAC units],
                'ramp_sigma': 0.005 , # us
                'freq': 3566.383307324457,   #2010.5265269004717,   # MHz
                'pi_ge_before': False,
                'pi_ef_before': False,
                'pi_ge_after': False,
                'pre_pulse': False}

    # g state 
    lrabi = LengthRabiGeneralExperiment(
        soccfg=soc,
        path=expt_path,
        config_file=config_path,
    )
    lrabi.cfg.expt = cfg_expt

    #g_state 
    lrabi.go(analyze=False, display=False, progress=False, save=False)
    g_data = lrabi.data

    # e state 

    lrabi.cfg.expt['pi_ge_before'] = 'True'
    lrabi.go(analyze=False, display=False, progress=False, save=False)
    e_data = lrabi.data

    # # f state 
    lrabi.cfg.expt['pi_ef_before'] = 'True'
    lrabi.go(analyze=False, display=False, progress=False, save=False)
    f_data = lrabi.data
    return g_data, e_data, f_data