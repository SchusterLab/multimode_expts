# %load_ext autoreload
# %autoreload 2

import os
import sys
import time
from copy import deepcopy

import numpy as np
import yaml
from qick import QickConfig
from slab import AttrDict, get_current_filename, get_next_filename
from slab.instruments import InstrumentManager
from tqdm.auto import tqdm

expts_path = os.getcwd()
if expts_path not in sys.path:
    sys.path.insert(0, expts_path)
    print("Path added at highest priority")

# Import the experiments module from multimode
import multimode_expts.experiments as meas
# Verify the module is imported from the correct path
print(meas.__file__)

from dataset import FloquetStorageSwapDataset, StorageManSwapDataset
from MM_dual_rail_base import MM_dual_rail_base


expt_path = "/mnt/d/experiments/251031_qsim/data"
print("Data will be stored in", expt_path)

curr_path = os.getcwd()
config_file = os.path.join(curr_path, "configs", "hardware_config_202505.yml")
print("Config will be", config_file)

exp_param_file = os.path.join(curr_path, "configs", "experiment_config.yml")
exp_param_path = exp_param_file
print("Exp Param will be", exp_param_path)

qubit_i = 0

with open(config_file, "r") as cfg_file:
    yaml_cfg = yaml.safe_load(cfg_file)
yaml_cfg = AttrDict(yaml_cfg)

im = InstrumentManager(ns_address="192.168.137.25")  # SLAC lab
soc = QickConfig(im[yaml_cfg["aliases"]["soc"]].get_cfg())
print(soc)


# %%

ds = FloquetStorageSwapDataset(
    os.path.join(expts_path, "floquet_storage_swap_dataset.csv")
)

ds_thisrun = FloquetStorageSwapDataset(ds.create_copy())
ds_thisrun_file_path = os.path.join(expts_path, ds_thisrun.filename)
ds_thisrun.df

# %%

ds_storage = StorageManSwapDataset(
    os.path.join(expts_path, "man1_storage_swap_dataset.csv")
)
ds_storage.df

ds_thisrun.get_columns()

# %%


config_thisrun = AttrDict(deepcopy(yaml_cfg))
config_thisrun.device.storage.floquet_man_stor_file = ds_thisrun_file_path


# %%


def do_kerr_ramsey(
    config_thisrun,
    expt_path,
    config_path,
    start=0.01,  # start delay
    step=0.02,  # step size
    expts=100,  # number of experiments
    ramsey_freq=3.7,  # Ramsey frequency
    kerr_gain=2000,
    kerr_detune=-10,
    reps=100,  # repetitions
    rounds=1,  # rounds
    qubits=[0],  # qubits
    checkEF=False,  # check EF
    f0g1_cavity=0,  # f0g1 cavity
    init_gf=False,  # initialize gf
    active_reset=False,  # active reset
    man_reset=True,  # manipulate reset
    storage_reset=True,  # storage reset
    user_defined_pulse=None,  # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
    parity_meas=True,  # parity measurement
    man_mode_no=1,
    storage_ramsey=[False, 2, True],  # storage Ramsey
    man_ramsey=None,  # manipulate Ramsey
    coupler_ramsey=False,  # coupler Ramsey
    custom_coupler_pulse=None,  # custom coupler pulse
    echoes=[False, 0],  # echoes
    prepulse=False,  # prepulse
    postpulse=False,  # postpulse
    gate_based=False,  # gate based
    pre_sweep_pulse=None,  # pre sweep pulse
    post_sweep_pulse=None,  # post sweep pulse
    prep_e_first=True,
    relax_delay=2500,  # relax delay
):
    """
    Run the Cavity Ramsey experiment using the specified configuration.
    """
    if user_defined_pulse is None:
        user_defined_pulse = [
            True,
            config_thisrun.device.manipulate.f_ge[man_mode_no - 1],
            2000,  # will be overridden if expt_params.displace_gain is set!
            config_thisrun.device.manipulate.displace_sigma[man_mode_no - 1],
            0,
            4,
        ]

    # [on/off, freq, gain, sigma (mus), length, channel]
    if man_ramsey is None:
        man_ramsey = [False, man_mode_no - 1]
    if custom_coupler_pulse is None:
        custom_coupler_pulse = [
            [944.25],
            [1000],
            [0.316677658],
            [0],
            [1],
            ["flat_top"],
            [0.005],
        ]
    if pre_sweep_pulse is None:
        pre_sweep_pulse = []
    if post_sweep_pulse is None:
        post_sweep_pulse = []

    expt_params = dict(
        start=start,  # start delay
        step=step,  # step size
        expts=expts,  # number of experiments
        ramsey_freq=ramsey_freq,  # Ramsey frequency
        reps=reps,  # repetitions
        rounds=rounds,  # rounds
        qubits=qubits,  # qubits
        checkEF=checkEF,  # check EF
        f0g1_cavity=f0g1_cavity,  # f0g1 cavity
        init_gf=init_gf,  # initialize gf
        active_reset=active_reset,  # active reset
        man_reset=man_reset,  # manipulate reset
        storage_reset=storage_reset,  # storage reset
        user_defined_pulse=user_defined_pulse,  # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
        parity_meas=parity_meas,  # parity measurement
        man_mode_no=man_mode_no,  # manipulate index
        storage_ramsey=storage_ramsey,  # storage Ramsey
        man_ramsey=man_ramsey,  # manipulate Ramsey
        coupler_ramsey=coupler_ramsey,  # coupler Ramsey
        custom_coupler_pulse=custom_coupler_pulse,  # custom coupler pulse
        echoes=echoes,  # echoes
        prepulse=prepulse,  # prepulse
        postpulse=postpulse,  # postpulse
        gate_based=gate_based,  # gate based
        pre_sweep_pulse=pre_sweep_pulse,  # pre sweep pulse
        post_sweep_pulse=post_sweep_pulse,  # post sweep pulse
        prep_e_first=prep_e_first,  # prepare e first
        normalize=False,
        kerr_gain=kerr_gain,
        kerr_detune=kerr_detune,
        # kerr_length = 10,
        # swept_params = ['kerr_length'],
        swept_params=["displace_gain", "kerr_length"],
        kerr_lengths=np.linspace(0.007, 3, 101).tolist(),
        displace_gains=np.arange(2000, 8001, 1000).tolist(),
        # displace_gain = 5000,
        drive_coupler=True,
        ds_thisrun=ds_storage,
    )

    cavity_ramsey = meas.QsimBaseExperiment(
        soccfg=soc,
        path=expt_path,
        prefix="KerrRamseyExperiment",
        config_file=config_path,
        expt_params=expt_params,
        program=meas.KerrCavityRamseyProgram,
        progress=True,
    )

    cavity_ramsey.cfg = AttrDict(deepcopy(config_thisrun))

    cavity_ramsey.cfg.expt = expt_params

    cavity_ramsey.cfg.device.readout.relax_delay = [relax_delay]
    cavity_ramsey.go(analyze=False, display=False, progress=True, save=True)
    return cavity_ramsey


# %%

kerr_ramsey = do_kerr_ramsey(
    config_thisrun=config_thisrun,
    expt_path=expt_path,
    config_path=config_file,
    ramsey_freq=1.5,
    kerr_gain=8000,
    # step = 0.04,
    # expts = 150,
    reps=100,
    prep_e_first=False,
    # active_reset=True,
    # man_reset=True,
    # relax_delay=300,
)

# %%

