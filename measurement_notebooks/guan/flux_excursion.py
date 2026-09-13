# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Flux excursion: Kerr engineering, and recalibrating after a flux move
#
# Split out of `qsim_experiments.py`. The library side of this project is
# `experiments/qsim/cavity_ramsey_flux_excursion.py`,
# `t2_cavity_fluxexcursion.py` and `kerr.py`.
#
# * **Cavity Ramsey**, plain, versus gain, and with Kerr.
# * **Qubit Rabi** in the same setting.
# * **After moving flux**: manipulate parity spectroscopy, gain-to-alpha,
#   monitoring, and nonlinearity versus flux bias.
#
# Historical per the surface map (area 6), kept recoverable.
#
# Its neighbours: `floquet_calibration.py`, `dark_mode.py`, `mbramsey.py`,
# `cooling.py`.

# %% [markdown]
# # Prepare

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy

import experiments as meas
from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner

from job_server import JobClient
from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager

# Initialize database and config manager
db = get_database()
config_dir = 'C:/python/multimode_expts/configs'
config_manager = ConfigVersionManager(config_dir)

# Initialize job client (handle submitting and waiting for jobs)
client = JobClient()

# Check server health
health = client.health_check()
print(f"Server status: {health['status']}")
print(f"Pending jobs: {health['pending_jobs']}")

# %%
# Who is running these experiments??
user = 'guan'

print(f"Welcome {user}!")

# %% editable=true slideshow={"slide_type": ""}
# Initialize station to retrieve soc and configs
station = MultimodeStation(
    user = user,
    experiment_name = "260513_qsim",
    
    hardware_config="CFG-HW-20260709-00019",
    storage_man_file="CFG-M1-20260707-00020",
    floquet_file="CFG-FL-20260707-00053",
    # multiphoton_config="versions/multiphoton_config/CFG-MP-20260115-00001.yml",
    log_measurements=True,
)

# %%
# The swap-parameter reader and the readout-calibration postprocessor used to
# be defined in this cell. Seven notebooks had byte-identical copies, and both
# of them write or read calibration, so they are modules now.
from experiments.qsim.utils import floquet_pulse_parameters
from experiments.readout_calibration import apply_singleshot_calibration

get_floquet_parameters = floquet_pulse_parameters  # old name, same function



# %% [markdown]
# ## Datset for Sidebands

# %%
station.ds_storage.df

# %%
# Override/initalize using the storage dataset
station.ds_floquet.import_from_swap_dataset(station.ds_storage, gain_div=1, 
                                    pi_div=15)
station.snapshot_floquet_storage_swap(update_main=False)

# %%
station.ds_floquet.df

# %% [markdown]
# # Single shot

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
singleshot_defaults = AttrDict(dict(    
    reps=5000,
    relax_delay=500,
    check_f=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    qubit=0,
    pulse_manipulate=False,
    # cavity_freq=4984.373226159381,
    # cavity_gain=400,
    # cavity_length=2,
    prepulse=False,
    pre_sweep_pulse=None,
    gate_based=True,
    qubits=[0],
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

# `apply_singleshot_calibration` replaces the local
# `singleshot_postproc` this cell used to define.


# %%
# Execute
# =================================
ss_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.HistogramExperiment,
    default_expt_cfg = singleshot_defaults,
    postprocessor = apply_singleshot_calibration,
    job_client=client,
)

ss = ss_runner.execute(
    check_f=False,
    active_reset=False, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
    relax_delay=2000,
    # active_reset=True,
    # relax_delay=200,
    priority=1,
)
# ss.display(station)

# %%
station.hardware_cfg.data_management['vault_root'] = 'H:/Shared drives/Slab/Multimode'

# %%
station.snapshot_hardware_config(update_main=False)
station.preview_config_update()

# %% [markdown]
# # Kerr engineering

# %%
expt_params = dict(
    expts = 1,
    reps = 200,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    init_stor = 0, # storage mode number to initialize to n=1 Fock state
    ro_stor = 0, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    echoes = [False, 0], # [on/off, number of echoes]
    # === new class
    # kerr_gain = 1000,
    kerr_detune = -10,
    kerr_length = 10,
    swept_params = ['kerr_gain'],
    # kerr_lengths = np.linspace(0.007,5,21).tolist(),
    kerr_gains = np.arange(0,2000,10).tolist(),
    prepulse = False,
    postpulse = False,
)

qbe = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"KerrQBHeating",
    config_file=config_path,
    expt_params=expt_params,
    program=meas.KerrEngBaseProgram,
    progress=True)

qbe.cfg.device.readout.relax_delay = [2000]  # Wait time between experiments [us]
qbe.go(analyze=False, display=True, progress=True, save=True)


# %% [markdown]
# ## Cavity Ramsey 
#
# This is to find out $\chi$, $\chi'$, $\Delta$ and $K_c$

# %%
def do_cavity_ramsey(
    config_thisrun,
    expt_path,
    config_path,
    start=0.01,           # start delay
    step=0.05,            # step size
    expts=200,            # number of experiments
    ramsey_freq=3.7,      # Ramsey frequency
    reps=100,              # repetitions
    rounds=1,             # rounds
    qubits=[0],           # qubits
    checkEF=False,        # check EF
    f0g1_cavity=0,        # f0g1 cavity
    init_gf=False,        # initialize gf
    active_reset=False,   # active reset
    man_reset=True,       # manipulate reset
    storage_reset=True,   # storage reset
    user_defined_pulse=None, # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
    parity_meas=True,     # parity measurement
    man_mode_no=1,            
    storage_ramsey=[False, 2, True], # storage Ramsey
    man_ramsey=None,      # manipulate Ramsey
    coupler_ramsey=False, # coupler Ramsey
    custom_coupler_pulse=None, # custom coupler pulse
    echoes=[False, 0],    # echoes
    prepulse=False,       # prepulse
    postpulse=False,      # postpulse
    gate_based=False,     # gate based
    pre_sweep_pulse=None, # pre sweep pulse
    post_sweep_pulse=None,# post sweep pulse
    prep_e_first = True,
    relax_delay=2500      # relax delay
):
    """
    Run the Cavity Ramsey experiment using the specified configuration.
    """
    if user_defined_pulse is None:
        user_defined_pulse = [True, config_thisrun.device.manipulate.f_ge[man_mode_no-1], 10, 
                                config_thisrun.device.manipulate.displace_sigma[man_mode_no-1], 0,
                                  4]
        
    #[on/off, freq, gain, sigma (mus), phase, channel] 
    if man_ramsey is None:
        man_ramsey = [False, man_mode_no -1]
    if custom_coupler_pulse is None:
        custom_coupler_pulse = [[944.25], [1000], [0.316677658], [0], [1], ['flat_top'], [0.005]]
    if pre_sweep_pulse is None:
        pre_sweep_pulse = []
    if post_sweep_pulse is None:
        post_sweep_pulse = []

    cavity_ramsey = meas.single_qubit.t2_cavity.CavityRamseyExperiment(
        soccfg=soc, path=expt_path, prefix='CavityRamseyExperiment', config_file=config_path
    )

    cavity_ramsey.cfg = AttrDict(deepcopy(config_thisrun))

    cavity_ramsey.cfg.expt = dict(
        start=start,                    # start delay
        step=step,                      # step size
        expts=expts,                    # number of experiments
        ramsey_freq=ramsey_freq,        # Ramsey frequency
        reps=reps,                      # repetitions
        rounds=rounds,                  # rounds
        qubits=qubits,                  # qubits
        checkEF=checkEF,                # check EF
        f0g1_cavity=f0g1_cavity,        # f0g1 cavity
        init_gf=init_gf,                # initialize gf
        active_reset=active_reset,      # active reset
        man_reset=man_reset,            # manipulate reset
        storage_reset=storage_reset,    # storage reset
        user_defined_pulse=user_defined_pulse, # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
        parity_meas=parity_meas,        # parity measurement
        man_mode_no=man_mode_no,                # manipulate index
        storage_ramsey=storage_ramsey,  # storage Ramsey
        man_ramsey=man_ramsey,          # manipulate Ramsey
        coupler_ramsey=coupler_ramsey,  # coupler Ramsey
        custom_coupler_pulse=custom_coupler_pulse, # custom coupler pulse
        echoes=echoes,                  # echoes
        prepulse=prepulse,              # prepulse
        postpulse=postpulse,            # postpulse
        gate_based=gate_based,          # gate based
        pre_sweep_pulse=pre_sweep_pulse,# pre sweep pulse
        post_sweep_pulse=post_sweep_pulse, # post sweep pulse
        prep_e_first=prep_e_first,  # prepare e first
    )

    cavity_ramsey.cfg.device.readout.relax_delay = [relax_delay]
    cavity_ramsey.go(analyze=False, display=False, progress=True, save=True)
    return cavity_ramsey


# %%
cavity_ramsey = do_cavity_ramsey(
    config_thisrun=config_thisrun,
    expt_path=expt_path,
    config_path=config_file,
    man_mode_no=1, 
    ramsey_freq=1,
    step = 0.02, 
    expts = 100, 
    reps = 100,
    prep_e_first = False,
    # user_defined_pulse=[True, expts_base_inst.config_thisrun.device.manipulate.f_ge[0], 1500, 
    #                             expts_base_inst.config_thisrun.device.manipulate.displace_sigma[0],
    #                               0, 4])
    user_defined_pulse=[True, config_thisrun.device.manipulate.f_ge[0], 2000, 
                                config_thisrun.device.manipulate.displace_sigma[0],
                                  0, 4])


    #user defined pulse [on/off, freq, gain, sigma (mus), phase, channel] )

# %%
cavity_ramsey.analyze()
cavity_ramsey.display()


# %% [markdown]
# ### vs gain

# %%
def do_cavity_ramsey_gain_sweep(
    config_thisrun,
    expt_path,
    config_path,
    start=0.01,           # start delay
    step=0.05,            # step size
    expts=200,            # number of experiments
    ramsey_freq=3.7,      # Ramsey frequency
    gain_start = 1000,  # start gain
    gain_step = 1000,      # step size for gain
    gain_expts = 5,        # number of experiments for gain
    reps=100,              # repetitions
    rounds=1,             # rounds
    qubits=[0],           # qubits
    checkEF=False,        # check EF
    f0g1_cavity=0,        # f0g1 cavity
    init_gf=False,        # initialize gf
    active_reset=False,   # active reset
    man_reset=True,       # manipulate reset
    storage_reset=True,   # storage reset
    user_defined_pulse=None, # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
    parity_meas=True,     # parity measurement
    man_mode_no=1,            
    storage_ramsey=[False, 2, True], # storage Ramsey
    man_ramsey=None,      # manipulate Ramsey
    coupler_ramsey=False, # coupler Ramsey
    custom_coupler_pulse=None, # custom coupler pulse
    echoes=[False, 0],    # echoes
    prepulse=False,       # prepulse
    postpulse=False,      # postpulse
    gate_based=False,     # gate based
    pre_sweep_pulse=None, # pre sweep pulse
    post_sweep_pulse=None,# post sweep pulse
    relax_delay=2500,      # relax delay
    do_g_and_e=False, # do e-f first
):
    """
    Run the Cavity Ramsey experiment using the specified configuration.
    """
    if user_defined_pulse is None:
        user_defined_pulse = [True,
                              config_thisrun.device.manipulate.f_ge[man_mode_no-1], 
                              1000, 
                              config_thisrun.device.manipulate.displace_sigma[man_mode_no-1], 
                              0,
                              4]
    #[on/off, freq, gain, sigma (mus), length, channel] 
    if man_ramsey is None:
        man_ramsey = [False, man_mode_no -1]
    if custom_coupler_pulse is None:
        custom_coupler_pulse = [[944.25], [1000], [0.316677658], [0], [1], ['flat_top'], [0.005]]
    if pre_sweep_pulse is None:
        pre_sweep_pulse = []
    if post_sweep_pulse is None:
        post_sweep_pulse = []

    cavity_ramsey = meas.single_qubit.t2_cavity.CavityRamseyGainSweepExperiment(
        soccfg=soc, path=expt_path, prefix='CavityRamseyGainSweepExperiment', config_file=config_path
    )

    cavity_ramsey.cfg = AttrDict(deepcopy(config_thisrun))

    cavity_ramsey.cfg.expt = dict(
        start=start,                    # start delay
        step=step,                      # step size
        expts=expts,                    # number of experiments
        ramsey_freq=ramsey_freq,        # Ramsey frequency
        gain_start=gain_start,          # start gain
        gain_step=gain_step,            # step size for gain
        gain_expts=gain_expts,          # number of experiments for gain
        reps=reps,                      # repetitions
        rounds=rounds,                  # rounds
        qubits=qubits,                  # qubits
        checkEF=checkEF,                # check EF
        f0g1_cavity=f0g1_cavity,        # f0g1 cavity
        init_gf=init_gf,                # initialize gf
        active_reset=active_reset,      # active reset
        man_reset=man_reset,            # manipulate reset
        storage_reset=storage_reset,    # storage reset
        user_defined_pulse=user_defined_pulse, # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
        parity_meas=parity_meas,        # parity measurement
        man_mode_no=man_mode_no,                # manipulate index
        storage_ramsey=storage_ramsey,  # storage Ramsey
        man_ramsey=man_ramsey,          # manipulate Ramsey
        coupler_ramsey=coupler_ramsey,  # coupler Ramsey
        custom_coupler_pulse=custom_coupler_pulse, # custom coupler pulse
        echoes=echoes,                  # echoes
        prepulse=prepulse,              # prepulse
        postpulse=postpulse,            # postpulse
        gate_based=gate_based,          # gate based
        pre_sweep_pulse=pre_sweep_pulse,# pre sweep pulse
        post_sweep_pulse=post_sweep_pulse, # post sweep pulse
        do_g_and_e=do_g_and_e,  # do e-f first

        qubit_drive_pulse=[True],
    )

    cavity_ramsey.cfg.device.readout.relax_delay = [relax_delay]
    cavity_ramsey.go(analyze=False, display=False, progress=True, save=True)
    return cavity_ramsey



# %%
gain_start = 1000
gain_stop = 7000
gain_step = 1000
gain_expts = int((gain_stop - gain_start) / gain_step) + 1
print(f'Gain start: {gain_start}, Gain stop: {gain_stop}, Gain step: {gain_step}, Gain expts: {gain_expts}')

# %%
cavity_ramsey_sweep = do_cavity_ramsey_gain_sweep(
    config_thisrun=config_thisrun,
    expt_path=expt_path,
    config_path=config_file,
    gain_start=gain_start,  # start gain
    gain_step=gain_step,      # step size for gain
    gain_expts=gain_expts,        # number of experiments for gain
    ramsey_freq=0.8,
    step = 0.05, 
    expts = 100, 
    reps = 100,
    do_g_and_e=False,
)

# %%
cavity_ramsey_sweep.analyze()
cavity_ramsey_sweep.display()
delta_g = cavity_ramsey_sweep.data['detuning_g']

# config_thisrun.device.manipulate.f_ge[0] -= delta_g
# print(f"Updated f_ge frequency: {config_thisrun.device.manipulate.f_ge[0]} MHz")

# %%

# %%

# %% [markdown]
# ## with kerr

# %%
kerr_ramsey_defaults = AttrDict(dict(
    start=0.01,           # start delay
    step=0.02,            # step size
    expts=100,            # number of experiments
    ramsey_freq=3.7,      # Ramsey frequency
    kerr_gain=2000,
    kerr_detune=-10,
    reps=100,              # repetitions
    rounds=1,             # rounds
    qubits=[0],           # qubits
    checkEF=False,        # check EF
    f0g1_cavity=0,        # f0g1 cavity
    init_gf=False,        # initialize gf
    active_reset=False,   # active reset
    man_reset=True,       # manipulate reset
    storage_reset=True,   # storage reset
    user_defined_pulse=None, # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
    parity_meas=True,     # parity measurement
    man_mode_no=1,
    storage_ramsey=[False, 2, True], # storage Ramsey
    man_ramsey=None,      # manipulate Ramsey
    coupler_ramsey=False, # coupler Ramsey
    custom_coupler_pulse=None, # custom coupler pulse
    echoes=[False, 0],    # echoes
    prepulse=False,       # prepulse
    postpulse=False,      # postpulse
    gate_based=False,     # gate based
    pre_sweep_pulse=None, # pre sweep pulse
    post_sweep_pulse=None,# post sweep pulse
    prep_e_first=True,
    normalize=False,
    swept_params=['displace_gain', 'kerr_detune'],
    kerr_detunes = np.linspace(-100, 100, 5).tolist(),
    displace_gains = np.arange(2000, 8001, 1000).tolist(),
    kerr_drive_type='man-qubit', # 'man-coupler', 'qubit'
    relax_delay=2500,
))

def kerr_ramsey_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    man_mode_no = expt_cfg.man_mode_no
    hw = station.hardware_cfg

    if expt_cfg.user_defined_pulse is None:
        expt_cfg.user_defined_pulse = [
            True,
            hw.device.manipulate.f_ge[man_mode_no - 1], # freq
            2000,  # will be overridden if expt_cfg.displace_gain is set! # gain
            hw.device.manipulate.displace_sigma[man_mode_no - 1], # sigma
            0, # length
            4, # proxy for ch for displacement, 4 = man
        ]
    
    # [on/off, freq, gain, sigma (mus), length, channel]
    if expt_cfg.man_ramsey is None:
        expt_cfg.man_ramsey = [False, man_mode_no - 1]
    if expt_cfg.custom_coupler_pulse is None and expt_cfg.kerr_drive_type == 'man-coupler':
        expt_cfg.custom_coupler_pulse = [[944.25], [1000], [0.316677658], [0], [1], ['flat_top'], [0.005]]
    if expt_cfg.pre_sweep_pulse is None:
        expt_cfg.pre_sweep_pulse = []
    if expt_cfg.post_sweep_pulse is None:
        expt_cfg.post_sweep_pulse = []

    print(expt_cfg)
    return expt_cfg

# def kerr_ramsey_postproc(station, expt):
#     pass


# %%
kerr_ramsey_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.qsim.kerr.KerrCavityRamseyProgram,
    default_expt_cfg=kerr_ramsey_defaults,
    preprocessor=kerr_ramsey_preproc,
    # postprocessor=kerr_ramsey_postproc,
    job_client=client,
)

kerr_detunes = np.linspace(-50, -10, 5).tolist()
kerr_gain = station.hardware_cfg.device.multiphoton['pi']['fn-gn+1']['gain'][0]

kerr_lengths = np.linspace(0.010, 4, 101).tolist()
displace_gains = np.arange(2000, 7001, 1000).tolist()
print("kerr_detunes", kerr_detunes)
print("kerr_gain", kerr_gain)
print("kerr_lengths", kerr_lengths)
print("displace_gains", displace_gains)

kerr_expts = []

for kerr_detune in kerr_detunes:
    print("kerr detune:", kerr_detune)
    kerr_ramsey = kerr_ramsey_runner.execute(
        ramsey_freq=1,
        kerr_gain=kerr_gain, # gain for kerr pulse
        kerr_detune=kerr_detune,
        reps=50,
        prep_e_first=False,
        active_reset=False,
        # man_reset=True,

        swept_params = ['displace_gain', 'kerr_length'],
        kerr_lengths = kerr_lengths,
        displace_gains = displace_gains,
        # displace_gain = 5000,
        kerr_drive_type='man-qubit', # 'man-coupler', 'qubit
    )
    kerr_expts.append(kerr_ramsey)
    kerr_ramsey.display()
    plt.show()

# %%
# Reopen old data
expt_objs = [
    "D:\\experiments\\260130_qsim_kerr_engineering\\expt_objs\\JOB-20260213-00296_expt.pkl",
    "D:\\experiments\\260130_qsim_kerr_engineering\\expt_objs\\JOB-20260213-00297_expt.pkl",
    "D:\\experiments\\260130_qsim_kerr_engineering\\expt_objs\\JOB-20260213-00298_expt.pkl",
    "D:\\experiments\\260130_qsim_kerr_engineering\\expt_objs\\JOB-20260213-00299_expt.pkl",
]

# h5_files = [
# "D:\\experiments\\260128_qsim_wigner\\data\\JOB-20260204-00129_QsimWignerBaseExperiment.h5",
# "D:\\experiments\\260128_qsim_wigner\\data\\JOB-20260204-00131_QsimWignerBaseExperiment.h5",
# "D:\\experiments\\260128_qsim_wigner\\data\\JOB-20260204-00133_QsimWignerBaseExperiment.h5",
# "D:\\experiments\\260128_qsim_wigner\\data\\JOB-20260204-00134_QsimWignerBaseExperiment.h5",
# "D:\\experiments\\260128_qsim_wigner\\data\\JOB-20260204-00136_QsimWignerBaseExperiment.h5",
# "D:\\experiments\\260128_qsim_wigner\\data\\JOB-20260204-00138_QsimWignerBaseExperiment.h5"
# ]

import pickle
for i, expt_obj in enumerate(expt_objs):
    print("kerr detune:", kerr_detunes[i])
    with open(expt_obj, "rb") as f:
        kerr_ramsey = pickle.load(f)
        kerr_ramsey.display()
        plt.show()

# %%
# def do_kerr_ramsey(
#     config_thisrun,
#     expt_path,
#     config_path,
#     start=0.01,           # start delay
#     step=0.02,            # step size
#     expts=100,            # number of experiments
#     ramsey_freq=3.7,      # Ramsey frequency
#     kerr_gain=2000,
#     kerr_detune=-10,
#     reps=100,              # repetitions
#     rounds=1,             # rounds
#     qubits=[0],           # qubits
#     checkEF=False,        # check EF
#     f0g1_cavity=0,        # f0g1 cavity
#     init_gf=False,        # initialize gf
#     active_reset=False,   # active reset
#     man_reset=True,       # manipulate reset
#     storage_reset=True,   # storage reset
#     user_defined_pulse=None, # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
#     parity_meas=True,     # parity measurement
#     man_mode_no=1,            
#     storage_ramsey=[False, 2, True], # storage Ramsey
#     man_ramsey=None,      # manipulate Ramsey
#     coupler_ramsey=False, # coupler Ramsey
#     custom_coupler_pulse=None, # custom coupler pulse
#     echoes=[False, 0],    # echoes
#     prepulse=False,       # prepulse
#     postpulse=False,      # postpulse
#     gate_based=False,     # gate based
#     pre_sweep_pulse=None, # pre sweep pulse
#     post_sweep_pulse=None,# post sweep pulse
#     prep_e_first = True,
#     relax_delay=2500      # relax delay
# ):
#     """
#     Run the Cavity Ramsey experiment using the specified configuration.
#     """
#     if user_defined_pulse is None:
#         user_defined_pulse = [True,
#                               config_thisrun.device.manipulate.f_ge[man_mode_no-1], 
#                               2000, # will be overridden if expt_params.displace_gain is set! 
#                               config_thisrun.device.manipulate.displace_sigma[man_mode_no-1], 
#                               0,
#                               4]
        
#     #[on/off, freq, gain, sigma (mus), length, channel] 
#     if man_ramsey is None:
#         man_ramsey = [False, man_mode_no -1]
#     if custom_coupler_pulse is None:
#         custom_coupler_pulse = [[944.25], [1000], [0.316677658], [0], [1], ['flat_top'], [0.005]]
#     if pre_sweep_pulse is None:
#         pre_sweep_pulse = []
#     if post_sweep_pulse is None:
#         post_sweep_pulse = []

#     expt_params = dict(
#         start=start,                    # start delay
#         step=step,                      # step size
#         expts=expts,                    # number of experiments
#         ramsey_freq=ramsey_freq,        # Ramsey frequency
#         reps=reps,                      # repetitions
#         rounds=rounds,                  # rounds
#         qubits=qubits,                  # qubits
#         checkEF=checkEF,                # check EF
#         f0g1_cavity=f0g1_cavity,        # f0g1 cavity
#         init_gf=init_gf,                # initialize gf
#         active_reset=active_reset,      # active reset
#         man_reset=man_reset,            # manipulate reset
#         storage_reset=storage_reset,    # storage reset
#         user_defined_pulse=user_defined_pulse, # [on/off, freq, gain, sigma (mus), 0, 4] # if off, use config freq
#         parity_meas=parity_meas,        # parity measurement
#         man_mode_no=man_mode_no,                # manipulate index
#         storage_ramsey=storage_ramsey,  # storage Ramsey
#         man_ramsey=man_ramsey,          # manipulate Ramsey
#         coupler_ramsey=coupler_ramsey,  # coupler Ramsey
#         custom_coupler_pulse=custom_coupler_pulse, # custom coupler pulse
#         echoes=echoes,                  # echoes
#         prepulse=prepulse,              # prepulse
#         postpulse=postpulse,            # postpulse
#         gate_based=gate_based,          # gate based
#         pre_sweep_pulse=pre_sweep_pulse,# pre sweep pulse
#         post_sweep_pulse=post_sweep_pulse, # post sweep pulse
#         prep_e_first=prep_e_first,  # prepare e first
#         normalize = False,
#         kerr_gain = kerr_gain,
#         kerr_detune = kerr_detune,
#         # kerr_length = 10,
#         # swept_params = ['kerr_length'],
#         swept_params = ['displace_gain', 'kerr_detune'],
#         # swept_params = ['displace_gain', 'kerr_length'],
#         # kerr_lengths = np.linspace(0.007,3,101).tolist(),
#         kerr_detunes = np.linspace(-100, 100, 5).tolist()
#         displace_gains = np.arange(2000,8001,1000).tolist(),
#         # displace_gain = 5000,
#         kerr_drive_type='man-qubit', # 'man-coupler', 'qubit
#     )

#     cavity_ramsey = QsimBaseExperiment(
#     soccfg=soc,
#     path=expt_path,
#     prefix=f"KerrRamseyExperiment",
#     config_file=config_path,
#     expt_params=expt_params,
#     program=meas.qsim.kerr.KerrCavityRamseyProgram,
#     progress=True)

#     cavity_ramsey.cfg = AttrDict(deepcopy(config_thisrun))

#     cavity_ramsey.cfg.expt = expt_params

#     cavity_ramsey.cfg.device.readout.relax_delay = [relax_delay]
#     cavity_ramsey.go(analyze=False, display=False, progress=True, save=True)
#     return cavity_ramsey


# %%
# kerr_ramsey = do_kerr_ramsey(
#     config_thisrun=config_thisrun,
#     expt_path=expt_path,
#     config_path=config_file,
#     ramsey_freq=1,
#     kerr_gain=0,
#     kerr_detune=-30,
#     # step = 0.04, 
#     # expts = 150, 
#     reps = 100,
#     prep_e_first=False,
#     # active_reset=True,
#     # man_reset=True,
#     # relax_delay=300,
# )

# %%

# %%
for kerr_detune in [-30,-20,-10,-5,0,5,10,20,30]:
    print(f'detune {kerr_detune}')
    for kerr_gain in range(200,2901,300):
        # print(f"Running Kerr Ramsey with kerr_gain = {kerr_gain}")
        kerr_ramsey = do_kerr_ramsey(
            config_thisrun=config_thisrun,
            expt_path=expt_path,
            config_path=config_file,
            ramsey_freq=1.5,
            kerr_gain=kerr_gain,
            kerr_detune=kerr_detune,
            # step = 0.04, 
            # expts = 150, 
            reps = 100,
            prep_e_first=False,
            # active_reset=True,
            # man_reset=True,
            # relax_delay=300,
        )

# %%

# %% [markdown]
# ## Qubit rabi

# %%
from multimode_expts.experiments.single_qubit.amplitude_rabi import AmplitudeRabiExperiment, AmplitudeRabiChevronExperiment
from multimode_expts.experiments.single_qubit.length_rabi import LengthRabiExperiment

# %%
rabi = LengthRabiExperiment(
        soccfg=soc, 
        path=expt_path, 
        prefix='LengthRabiExperiment', 
        config_file=config_path
    )

rabi.cfg = AttrDict(deepcopy(config_thisrun))
rabi.cfg.expt = dict(
    start=0,                    # start delay
    step=0.007,                      # step size
    expts=100,                    # number of experiments
    reps=100,                      # repetitions
    rounds=1,                  # rounds
    qubits=[0],                  # qubits
    checkEF=False,                # check EF
    checkZZ=False,               # check ZZ
    pulse_type='const',        # pulse type
    repeat_time = 1,
    pre_pulse=False,       # pre pulse
    post_pulse=False,            # post pulse
)

rabi.cfg.device.readout.relax_delay = [500]
rabi.go(analyze=True, display=True, progress=True, save=True)

# %%
rabi = AmplitudeRabiChevronExperiment(
        soccfg=soc, 
        path=expt_path, 
        prefix='AmplitudeRabiChevronExperiment', 
        config_file=config_path
    )

rabi.cfg = AttrDict(deepcopy(config_thisrun))
rabi.cfg.expt = dict(
    start_f=3569,                # start frequency
    step_f=0.1,                 # f step size
    expts_f=21,                 # number of experiments
    start_gain=0,                    # start delay
    step_gain=10,                      # step size
    expts_gain=100,                    # number of experiments
    reps=100,                      # repetitions
    rounds=1,                  # rounds
    flat_length=0,
    sigma_test=1,
    user_defined_freq=[False,0],  # [on/off, freq]
    qubits=[0],                  # qubits
    checkEF=False,                # check EF
    checkZZ=False,                # check ZZ
    pulse_type='const',        # pulse type
    prepulse=False,       # pre pulse
    postpulse=False,            # post pulse
    pulse_ge_init=False,
    pulse_ge_after=False,
)

rabi.cfg.device.readout.relax_delay = [1500]
rabi.go(analyze=True, display=True, progress=True, save=True)

# %%

# %% [markdown]
# # Calibration after moving flux

# %%
# coupler
dcflux = YokogawaGS200(address="192.168.137.148")
dcflux.set_output(True)
dcflux.set_mode('current')
dcflux.ramp_current(0.5e-3, sweeprate=0.0001)

# %%
# jpa
dcflux = YokogawaGS200(address="192.168.137.149")
dcflux.ramp_current(-0.00454, sweeprate=0.002)


# %%

# %% [markdown]
# ## Manipulate parity spectroscopy
#

# %%
def do_parity_freq_experiment(
    start=4960,
    stop=5020,
    step=0.60,
    reps=100,
    rounds=1,
    qubit=[0],
    normalize=False,
    single_shot=False,
    singleshot_reps=10000,
    span=20,
    manipulate_no=1,
    displace=(True, 0.1, 1000),
    const_pulse=(False, 1),
    f0g1_cavity=0,
    prepulse=False,
    pre_sweep_pulse=None,
    relax_delay=2500
):
    """
    Run the Parity Frequency Experiment with configurable parameters.
    """
    
    expt_cfg = {
        'start': start,
        'stop': stop,
        'step': step,
        'reps': reps,
        'rounds': rounds,
        'qubits': qubit,
        'normalize': normalize,
        'single_shot': single_shot,
        'singleshot_reps': singleshot_reps,
        'span': span,
        'manipulate': manipulate_no,
        'displace': list(displace),
        'const_pulse': list(const_pulse),
        'f0g1_cavity': f0g1_cavity,
        'prepulse': prepulse,
        'pre_sweep_pulse': pre_sweep_pulse
    }
    # Example usage of relax_delay in experiment config:
    parity_freq_exp = meas.single_qubit.parity_freq.ParityFreqExperiment(
        soccfg=soc, path=expt_path, 
        prefix='ParityFreqExperiment', config_file=config_file
    )
    parity_freq_exp.cfg = AttrDict(deepcopy(config_thisrun))
    parity_freq_exp.cfg.expt = expt_cfg
    parity_freq_exp.cfg.device.readout.relax_delay = [relax_delay]
    parity_freq_exp.go(analyze=False, display=False, progress=True, save=True)
    return parity_freq_exp



# %%
parity_freq_exp = do_parity_freq_experiment()

# %%
from fitting.fit_display_classes import Spectroscopy
spec = Spectroscopy(parity_freq_exp.data, config=parity_freq_exp.cfg)
spec.analyze()
spec.display()

# %%
config_thisrun.device.manipulate.f_ge[0] = spec.data['fit_avgi'][2]


# %%

# %% [markdown]
# ## Gain to alpha

# %%
def do_parity_gain_experiment(
    config_thisrun,
    expt_path,
    config_path,
    start=0,
    step=100,
    expts=40,
    reps=250,
    rounds=1,
    qubit=0,
    qubits=[0],
    normalize=False,
    single_shot=False,
    singleshot_reps=1000,
    singleshot_active_reset=False,
    singleshot_man_reset=True,
    singleshot_storage_reset=True,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    span=1000,
    prep_e=False,
    manipulate=1,
    displace=(True, 0.05), # [enable, sigma] (gaussian length is 4sigma)
    const_pulse=(False, 1), # [enable, length]
    f0g1_cavity=0,
    prepulse=False,
    pre_sweep_pulse=None,
    relax_delay=2500, 
    pulse_correction=False
):
    """
    Run the Parity Gain Experiment with configurable parameters.
    """
    expt_cfg = {
        'start': start,
        'step': step,
        'expts': expts,
        'reps': reps,
        'rounds': rounds,
        'qubit': qubit,
        'qubits': qubits,
        'normalize': normalize,
        'single_shot': single_shot,
        'singleshot_reps': singleshot_reps,
        'singleshot_active_reset': singleshot_active_reset,
        'singleshot_man_reset': singleshot_man_reset,
        'singleshot_storage_reset': singleshot_storage_reset,
        'active_reset': active_reset,
        'man_reset': man_reset,
        'storage_reset': storage_reset,
        'span': span,
        'prep_e': prep_e,
        'manipulate': manipulate,
        'displace': list(displace),
        'const_pulse': list(const_pulse),
        'f0g1_cavity': f0g1_cavity,
        'prepulse': prepulse,
        'pre_sweep_pulse': pre_sweep_pulse if pre_sweep_pulse is not None else [],
        'pulse_correction': pulse_correction
    }
    parity_gain_exp = meas.single_qubit.parity_gain.ParityGainExperiment(
        soccfg=soc, path=expt_path, prefix='ParityGainExperiment', config_file=config_path
    )
    parity_gain_exp.cfg = AttrDict(deepcopy(config_thisrun))
    parity_gain_exp.cfg.expt = expt_cfg
    parity_gain_exp.cfg.device.readout.relax_delay = [relax_delay]
    
    parity_gain_exp.go(analyze=False, display=False, progress=True, save=True)
    return parity_gain_exp



# %%

# %%
parity_gain_expt = do_parity_gain_experiment(
    config_thisrun=config_thisrun,
    expt_path=expt_path,
    config_path=config_file,
    pulse_correction=True,
    reps=200,
)

parity_gain_expt.analyze()

# %%
#update device
gain_to_alpha = parity_gain_expt.data['gain_to_alpha']
print(f'Gain to alpha: {gain_to_alpha}')
config_thisrun.device.manipulate.gain_to_alpha[0] = gain_to_alpha

# %% [markdown]
# ### monitoring

# %%
from time import sleep

all_gtas = []

for i in range(100):
    parity_gain_expt = do_parity_gain_experiment(
        config_thisrun=config_thisrun,
        expt_path=expt_path,
        config_path=config_file,
        pulse_correction=True,
        reps=200,
    )

    parity_gain_expt.analyze(plot=False)
    gain_to_alpha = parity_gain_expt.data['gain_to_alpha']
    all_gtas.append(gain_to_alpha)
    sleep(300)

plt.plot(all_gtas)

# %%

# %% [markdown]
# ## nonlinearity vs flux bias

# %%
np.linspace(0.75e-3, 0.1e-3, 66)[:40]

# %%

# %%
flux_currents = np.linspace(0, 0.15e-3, 6)
for flux_current in flux_currents:
    # set current
    dcflux.ramp_current(flux_current, sweeprate=1e-4)
    parity_freq_exp = do_parity_freq_experiment()

    # find manipulate f_ge
    spec = Spectroscopy(parity_freq_exp.data, config=parity_freq_exp.cfg)
    spec.analyze()
    config_thisrun.device.manipulate.f_ge[0] = spec.data['fit_avgi'][2]

    # calibrate parity to gain
    parity_gain_expt = do_parity_gain_experiment(
        config_thisrun=config_thisrun,
        expt_path=expt_path,
        config_path=config_file,
        pulse_correction=True,
        reps=200,
    )
    parity_gain_expt.analyze()
    gain_to_alpha = parity_gain_expt.data['gain_to_alpha']
    print(f'Gain to alpha: {gain_to_alpha}')
    config_thisrun.device.manipulate.gain_to_alpha[0] = gain_to_alpha
    
    # do cavity ramsey
    kerr_ramsey = do_kerr_ramsey(
        config_thisrun=config_thisrun,
        expt_path=expt_path,
        config_path=config_file,
        ramsey_freq=1.5,
        kerr_gain=0,
        # step = 0.04, 
        # expts = 150, 
        reps = 100,
        prep_e_first=False,
        # active_reset=True,
        # man_reset=True,
        # relax_delay=300,
    )

