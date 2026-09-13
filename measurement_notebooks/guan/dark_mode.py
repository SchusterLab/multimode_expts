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
# # Dark mode: scrambling, and reading a collective mode
#
# Split out of `qsim_experiments.py`. This is the photon-number scrambling
# work and the dark/normal-mode readout built on it -- the project the
# `dark_mode_*` modules and `dark_mode_encoding.py` came from.
#
# * **Sideband Ramsey** and **sideband scramble** -- the drive itself.
# * **Phase calibration using a quantum walk**.
# * **Scramble using DarkBase** -- the current path, through
#   `experiments/qsim/dark_base.py`.
# * Amplitude Rabi, the Ramsey phase offset, a general 2D sweep, and T1.
#
# Mostly historical: the surface map lists bright/dark basis preparation and
# readout under area 6. It is kept runnable because the shared pulse layer
# underneath it is what MBR uses.
#
# Its neighbours: `floquet_calibration.py`, `flux_excursion.py`,
# `mbramsey.py`, `cooling.py`.

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
# # Photon number scrambling

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Sideband Ramsey

# %% [markdown]
# This is starting to test our channel phases: when we switch from one channel generator freq to diff freq (activating different storage swaps), does our code preserve phase coherence between the successive (partial) pulses. 
#
# We start from a simple M1-Sx Ramsey: qubit ge, qubit ef, f0g1 to initialize man1 into |1>, then do pi/2 on the beam splitter and wait and another pi/2.

# %%
expt_params = dict(
    start = 0.01, # wait time tau [us]
    step = 0.1,  # [us] 1 cycle is 0.0023251488095238095 [us], 2.7901785714285716 # [us]=1200 cycles
    expts = 100,
    ramsey_freq = 0.2, # [MHz]
    detune = 0,
    ac_stark=0.5, # [MHz] not sure how to define sign yet
    # but in any case this needs to be smaller than ramsey_freq
    # because qick can't handle negative numbers
    reps = 100,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    stor_no = 1, # storage mode number, 1 to 7
    normalize = False,
    active_reset = True,
    man_reset = True, 
    storage_reset = True, 
    advance_phase = 0, # advance phase of second pi/2 by this much [degrees]
    echoes = [False, 0], # [on/off, number of echoes]
)

sbr = SidebandRamseyExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"SidebandRamsey_M1S{expt_params['stor_no']}",
    config_file=config_path,
    expt_params = expt_params,
    progress=True)

sbr.cfg.device.readout.relax_delay = [200]  # Wait time between experiments [us]
# sbr.acquire()
sbr.go(analyze=True, display=True, progress=True, save=True)

# %%
idata = sbr.data['idata']
idata = idata.reshape((len(idata)//4,4))

qdata = sbr.data['qdata']
qdata = qdata.reshape((len(qdata)//4,4))

fig, axs = plt.subplots(nrows=4,ncols=2, figsize=(8,8))
for kk in range(4):
    axs[kk,0].hist(idata[:,kk], bins=100)
    axs[kk,1].hist(qdata[:,kk], bins=100)
None

# %%

# %% [markdown]
# ## Sideband scramble

# %%
sideband_scramble_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end

    init_fock=True,

    normalize=False,
    post_select_pre_pulse=False,
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    prepulse=True,
    postpulse=True,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def sideband_scramble_preproc(station, default_expt_cfg, **kwargs):
    assert 'swept_params' in kwargs
    assert len(kwargs['swept_params']) > 0

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    assert 'init_stor' in kwargs
    if not expt_cfg.init_fock:
        assert 'init_alpha' in kwargs
        
    # print(expt_cfg)
    return expt_cfg



# %%
scramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleProgram,
    default_expt_cfg=sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

floquet_cycles = np.arange(0, 201, step=4)

meas_stors = [0,4,5]
swap_stors = [4,5]
detunings = [0,0] # None/False/unspecified all default to all zeros

scramble_expts = []

for update_phases in [True]:
    for meas_stor in meas_stors:
        scramble = scramble_runner.execute(
            reps=50,
            init_fock=True,
            init_stor=0,
            ro_stor=meas_stor,
            relax_delay=8000,

            swap_stors=swap_stors,
            update_phases=update_phases,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
        )
        scramble_expts.append(scramble)
        # scramble.display()

# %%
meas_stors = [0,1,7]

for update_phases in [True]: # [False, True]:
    for meas_stor in meas_stors:
        expt_params = dict(
            expts = 1,
            reps = 100,
            rounds = 1,
            qubits = [0],
            f0g1_cavity = 1,  #  name of manipulate cavity (1 or 2)
            init_stor = 0, # storage mode number to initialize to n=1 Fock state
            ro_stor = meas_stor, # storage mode number that gets read out in the end
            # if 0, this means to read out man instead
            floquet_cycles = list(range(0,201)),
            detune=0,
            normalize = False,
            active_reset = False,
            man_reset = True, 
            storage_reset = True, 
            # advance_phase=3,
            # swept_params = ['advance_phase', 'floquet_cycle'],
            swept_params = ['floquet_cycle'],
            swap_stors = [1,7],
            # floquet_dataset_filename = 'floquet_storage_2Derramp.csv',
            update_phases = update_phases, 
            echoes = [False, 0], # [on/off, number of echoes]
            prepulse=True,
            postpulse=True,
        )
    
        sbs = meas.QsimBaseExperiment(
            soccfg=soc,
            path=expt_path,
            prefix=f"SidebandScramble_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
            config_file=config_path,
            expt_params = expt_params,
            program = meas.SidebandScrambleProgram,
            progress=True)
    
        sbs.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
        sbs.go(analyze=False, display=True, progress=True, save=True)

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Phase calibration using quantum walk

# %%
from multimode_expts.experiments.qsim.sideband_scramble import FloquetCalibrationProgram

# %%
storA = 2
storB = 3

expt_params = dict(
    expts = 1,
    reps = 50,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    storA = storA,
    storB = storB,
    init_stor = storA, # storage mode number to initialize to n=1 Fock state
    ro_stor = storB, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    floquet_cycles = list(range(1,201,2)), 
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    storA_advance_phases = np.linspace(-20,20,81).tolist(), # advance phase of each successive pulse for the init mode[degrees]
    storB_advance_phase = ds_thisrun.get_phase_from(f'M1-S{storB}', f'M1-S{storA}')*2,
    # ro_advance_phases = np.linspace(-10,10,101).tolist(), # advance phase of each successive pulse for the ro mode [degrees]
    swept_params = ['storA_advance_phase', 'floquet_cycle'],
    # swept_params = ['floquet_cycle'],
    echoes = [False, 0], # [on/off, number of echoes]
    prepulse=True,
    postpulse=True
)


fce = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"FloquetCalibration_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
    config_file=config_path,
    expt_params = expt_params,
    program = FloquetCalibrationProgram,
    progress=True)

fce.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
fce.go(analyze=False, display=True, progress=True, save=True)

# %%
for storA, storB in [(2,6),(2,7),(3,2),(3,6),(3,7),(6,2),(6,3),(6,7),(7,2),(7,3),(7,6)]:
    expt_params = dict(
        expts = 1,
        reps = 50,
        rounds = 1,
        qubits = [0],
        f0g1_cavity = 1,  #  1/2 name of manipulate cavity
        storA = storA,
        storB = storB,
        init_stor = storA, # storage mode number to initialize to n=1 Fock state
        ro_stor = storB, # storage mode number that gets read out in the end
        # if 0, this means to read out man instead
        floquet_cycles = list(range(1,201,5)), 
        normalize = False,
        active_reset = False,
        man_reset = True, 
        storage_reset = True, 
        storA_advance_phases = np.linspace(-20,20,41).tolist(), # advance phase of each successive pulse for the init mode[degrees]
        storB_advance_phase = ds_thisrun.get_phase_from(f'M1-S{storB}', f'M1-S{storA}')*2,
        # ro_advance_phases = np.linspace(-10,10,101).tolist(), # advance phase of each successive pulse for the ro mode [degrees]
        swept_params = ['storA_advance_phase', 'floquet_cycle'],
        # swept_params = ['floquet_cycle'],
        echoes = [False, 0], # [on/off, number of echoes]
        prepulse=True,
        postpulse=True
    )
    
    
    fce = QsimBaseExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"FloquetCalibration_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
        config_file=config_path,
        expt_params = expt_params,
        program = FloquetCalibrationProgram,
        progress=True)
    
    fce.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
    fce.go(analyze=False, display=True, progress=True, save=True)

# %%
fce.display()

# %%
expt_params = dict(
    expts = 1,
    reps = 200,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    init_stor = 3, # storage mode number to initialize to n=1 Fock state
    ro_stor = 1, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    floquet_cycles = list(range(1,101)),
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    init_advance_phases = np.linspace(-20,20,101).tolist(), # advance phase of each successive pulse for the init mode[degrees]
    ro_advance_phase = -4,
    # ro_advance_phases = np.linspace(-10,10,101).tolist(), # advance phase of each successive pulse for the ro mode [degrees]
    swept_params = ['init_advance_phase', 'floquet_cycle'],
    # swept_params = ['floquet_cycle'],
    echoes = [False, 0], # [on/off, number of echoes]
)

fce = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"FloquetCalibration_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
    config_file=config_path,
    expt_params = expt_params,
    program = FloquetCalibrationProgram,
    progress=True)

fce.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
fce.go(analyze=False, display=True, progress=True, save=True)

# %%
expt_params = dict(
    expts = 1,
    reps = 200,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    init_stor = 2, # storage mode number to initialize to n=1 Fock state
    ro_stor = 3, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    floquet_cycles = list(range(1,101)),
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    init_advance_phases = np.linspace(-20,20,101).tolist(), # advance phase of each successive pulse for the init mode[degrees]
    ro_advance_phase = 6,
    # ro_advance_phases = np.linspace(-10,10,101).tolist(), # advance phase of each successive pulse for the ro mode [degrees]
    swept_params = ['init_advance_phase', 'floquet_cycle'],
    # swept_params = ['floquet_cycle'],
    echoes = [False, 0], # [on/off, number of echoes]
)

fce = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"FloquetCalibration_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
    config_file=config_path,
    expt_params = expt_params,
    program = FloquetCalibrationProgram,
    progress=True)

fce.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
fce.go(analyze=False, display=True, progress=True, save=True)

# %%
expt_params = dict(
    expts = 1,
    reps = 200,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    init_stor = 3, # storage mode number to initialize to n=1 Fock state
    ro_stor = 2, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    floquet_cycles = list(range(1,101)),
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    init_advance_phases = np.linspace(-20,20,101).tolist(), # advance phase of each successive pulse for the init mode[degrees]
    ro_advance_phase = -8,
    # ro_advance_phases = np.linspace(-10,10,101).tolist(), # advance phase of each successive pulse for the ro mode [degrees]
    swept_params = ['init_advance_phase', 'floquet_cycle'],
    # swept_params = ['floquet_cycle'],
    echoes = [False, 0], # [on/off, number of echoes]
)

fce = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"FloquetCalibration_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
    config_file=config_path,
    expt_params = expt_params,
    program = FloquetCalibrationProgram,
    progress=True)

fce.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
fce.go(analyze=False, display=True, progress=True, save=True)

# %%

# %%

# %%

# %%

# %%
expt_params = dict(
    expts = 1,
    reps = 200,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    init_stor = 1, # storage mode number to initialize to n=1 Fock state
    ro_stor = 3, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    floquet_cycles = list(range(1,101)),
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    init_advance_phases = list(np.arange(-10,10,0.5)), # advance phase of each successive pulse [degrees]
    ro_advance_phases = list(range(-10,30,1)), # advance phase of each successive pulse [degrees]
    # advance_phase=0,
    swept_params = ['advance_phase', 'floquet_cycle'],
    # swept_params = ['floquet_cycle'],
    # swap_stors = [1, 2],
    # update_phases = True, 
    echoes = [False, 0], # [on/off, number of echoes]
)

sbs = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"SidebandScramble_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
    config_file=config_path,
    expt_params = expt_params,
    program = SidebandScrambleProgram,
    progress=True)

sbs.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
sbs.go(analyze=False, display=True, progress=True, save=True)

# %%
expt_params = dict(
    expts = 1,
    reps = 200,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    init_stor = 1, # storage mode number to initialize to n=1 Fock state
    ro_stor = 2, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    floquet_cycles = list(range(1,101)),
    # gain_div = 3,
    # length_div = 2,
    # detune=0,
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    advance_phases = list(range(-30,30,1)), # advance phase of each successive pulse [degrees]
    # advance_phase=0,
    swept_params = ['advance_phase', 'floquet_cycle'],
    # swept_params = ['floquet_cycle'],
    # swap_stors = [1, 2],
    # update_phases = True, 
    echoes = [False, 0], # [on/off, number of echoes]
)

sbs = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"SidebandScramble_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
    config_file=config_path,
    expt_params = expt_params,
    program = SidebandScrambleProgram,
    progress=True)

sbs.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
sbs.go(analyze=False, display=True, progress=True, save=True)

# %%

# %%

# %%

# %%
idata = np.array(sbs.data['idata']).ravel()
idata = idata.reshape((len(idata)//4,4))

qdata = np.array(sbs.data['qdata']).ravel()
qdata = qdata.reshape((len(qdata)//4,4))

fig, axs = plt.subplots(nrows=4,ncols=2, figsize=(8,8))
for kk in range(4):
    axs[kk,0].hist(idata[:,kk], bins=100)
    axs[kk,1].hist(qdata[:,kk], bins=100)
None

# %%

# %%

# %%

# %%
for init_stor in range(3,8):
    # for ro_stor in range(8):
    ro_stor = 0
    for detune in np.linspace(-0.2,0.2,21):
        expt_params = dict(
            expts = 1,
            reps = 1000,
            rounds = 1,
            qubits = [0],
            f0g1_cavity = 1,  #  1/2 name of manipulate cavity
            init_stor = init_stor, # storage mode number to initialize to n=1 Fock state
            ro_stor = ro_stor, # storage mode number that gets read out in the end
            # if 0, this means to read out man instead
            floquet_cycles = list(range(1,101)),
            detune=detune,
            normalize = False,
            active_reset = True,
            man_reset = True, 
            storage_reset = True, 
            advance_phase = 0, # advance phase of second pi/2 by this much [degrees]
            echoes = [False, 0], # [on/off, number of echoes]
        )
        
        sbs = SidebandScrambleExperiment(
            soccfg=soc,
            path=expt_path,
            prefix=f"SidebandScramble_S{expt_params['init_stor']}_to_S{expt_params['ro_stor']}",
            config_file=config_path,
            expt_params = expt_params,
            progress=True)
        
        sbs.cfg.device.readout.relax_delay = [200]  # Wait time between experiments [us]
        sbs.go(analyze=False, display=False, progress=False, save=True)

# %%
plt.plot(ss[9].data['avgi'])

# %%

# %%

# %%

# %% [markdown]
# ## Scramble using DarkBase 
#
# for initial coherent state prep and parity readout 

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
dark_scramble_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end

    init_fock=True,

    normalize=False,
    post_select_pre_pulse=False,
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    prepulse=True,
    postpulse=True,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def dark_scramble_preproc(station, default_expt_cfg, **kwargs):
    assert 'swept_params' in kwargs
    assert len(kwargs['swept_params']) > 0

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    assert 'init_stor' in kwargs
    if not expt_cfg.init_fock:
        assert 'init_alpha' in kwargs
        
    # print(expt_cfg)
    return expt_cfg



# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"} jupyter={"outputs_hidden": true}
dark_scramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.qsim.floquet_dark_mode_readout.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dark_scramble_defaults,   # + the keys above
    postprocessor=None,
    job_client=client,
)


floquet_cycles = np.arange(0, 200, step=1)

meas_stors = [0,4,5]
swap_stors = [4,5]
detunings = [0,0] # None/False/unspecified all default to all zeros

scramble_expts = []

init_alphas = np.linspace(0.1,5,50)

for init_alpha in init_alphas:
    for meas_stor in meas_stors:
        scramble = dark_scramble_runner.execute(
            reps=400,
            init_fock=False,
            init_alpha=init_alpha,
            init_stor=0,
            ro_stor=meas_stor,
            relax_delay=8000,
    
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
    
            parity_readout = True,
            parity_fast = False,
            perform_wigner = False,
        )
        scramble_expts.append(scramble)

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
station.use_real_instruments()

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
dark_scramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment,
    ExptProgram=meas.qsim.floquet_dark_mode_readout.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dark_scramble_defaults,   # + the keys above
    postprocessor=None,
    job_client=client,
)


floquet_cycles = np.arange(0, 2001, step=5)

meas_stors = [0,4,5]
swap_stors = [4,5]
detunings = [0,0] # None/False/unspecified all default to all zeros


scramble = dark_scramble_runner.execute(
            reps=200,
            init_fock=False,
            init_alpha=init_alpha,
            init_stor=0,
            ro_stor=0,
            relax_delay=80,
    
            swap_stors=[4,5],
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
    
            parity_readout = True,
            parity_fast = False,
            perform_wigner = False,
        )

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
m1is = np.array([se.data['avgi'] for se in scramble_expts])
plt.pcolormesh(m1is)

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
m1is = np.array([se.data['avgi'] for se in scramble_expts[0::3]])
s4is = np.array([se.data['avgi'] for se in scramble_expts[1::3]])
s5is = np.array([se.data['avgi'] for se in scramble_expts[2::3]])

plt.pcolormesh(floquet_cycles, init_alphas, m1is)
plt.colorbar(label='avgi')
plt.xlabel('cycles')
plt.ylabel('alpha')
plt.title('M1 parity')


# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
plt.pcolormesh(floquet_cycles, init_alphas, s4is)

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}
plt.pcolormesh(floquet_cycles, init_alphas, s5is)

# %% jupyterlab_notify.notify={"mode": "default", "defaultThreshold": "30s"}

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Amplitude Rabi

# %%
from multimode_expts.experiments.qsim.sideband_amp_rabi import SidebandAmpRabiExperiment

# %%
for init_stor in range(1,8):
    ro_stor = 0
    expt_params = dict(
        expts = 1,
        reps = 500,
        rounds = 1,
        qubits = [0],
        f0g1_cavity = 1,  #  1/2 name of manipulate cavity
        init_stor = init_stor, # storage mode number to initialize to n=1 Fock state
        ro_stor = ro_stor, # storage mode number that gets read out in the end
        # if 0, this means to read out man instead
        detunes=np.linspace(-2,2,101).tolist(),
        gains=list(range(0,20000,100)),
        length=3, # us rabi pulse legnth
        normalize = False,
        active_reset = True,
        man_reset = True, 
        storage_reset = True, 
        advance_phase = 0, # advance phase of second pi/2 by this much [degrees]
        echoes = [False, 0], # [on/off, number of echoes]
    )
    
    sare = SidebandAmpRabiExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"SidebandAmpRabi_S{expt_params['init_stor']}",
        config_file=config_path,
        expt_params = expt_params,
        progress=True)
    
    sare.cfg.device.readout.relax_delay = [200]  # Wait time between experiments [us]
    sare.go(analyze=False, display=False, progress=True, save=True)

# %%

# %%

# %% [markdown]
# ## Find phase offset for ramsey

# %%
from multimode_expts.experiments.qsim.sideband_stark import SidebandStarkExperiment, SidebandStarkProgram

# %%
# for init_stor in range(1, 8):
for init_stor in range(1, 3):
    ro_stor = 0
    expt_params = dict(
        expts = 1,
        reps = 100,
        rounds = 1,
        qubits = [0],
        f0g1_cavity = 1,  #  1/2 name of manipulate cavity
        init_stor = init_stor, # storage mode number to initialize to n=1 Fock state
        ro_stor = ro_stor, # storage mode number that gets read out in the end
        # if 0, this means to read out man instead
        # detunes=np.linspace(-0.1,0.1,101).tolist(),
        detune = 0,
        advance_phases=np.linspace(-90,90,31).tolist(),
        # wait=10, # wait time between two hpi pulses in us
        waits = np.linspace(0,30,31).tolist(),
        swept_params = ['advance_phase', 'wait'],
        # usage: if you want to sweep cfg.expt.paramName, 
        # include paramName here in this list 
        # AND include cfg.expt.paramNames (note the s) as a list of values to step thru.
        # (You want a list instead of numpy array for better yaml export.)
        # Currently handles 1D and 2D sweeps and plots only.
        # For 2D, order is [outer, inner].
        normalize = False,
        active_reset = False,
        man_reset = True, 
        storage_reset = True, 
        # advance_phase = 0, # advance phase of second pi/2 by this much [degrees]
        echoes = [False, 0], # [on/off, number of echoes]
    )
    
    sta = SidebandStarkExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"SidebandStark_S{expt_params['init_stor']}",
        config_file=config_path,
        expt_params = expt_params,
        program=SidebandStarkProgram,
        progress=True)
    
    sta.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
    sta.go(analyze=False, display=False, progress=True, save=True)

# %%
sta.analyze()

# %%
for init_stor in [1]: #range(1,8):
    ro_stor = 0
    expt_params = dict(
        expts = 1,
        reps = 500,
        rounds = 1,
        qubits = [0],
        f0g1_cavity = 1,  #  1/2 name of manipulate cavity
        init_stor = init_stor, # storage mode number to initialize to n=1 Fock state
        ro_stor = ro_stor, # storage mode number that gets read out in the end
        # if 0, this means to read out man instead
        # detunes=np.linspace(-0.1,0.1,101).tolist(),
        detune = 0,
        phases=np.linspace(-90,90,91).tolist(),
        # wait=10, # wait time between two hpi pulses in us
        waits = np.linspace(0,2,51).tolist(),
        swept_params = ['phase', 'wait'],
        # usage: if you want to sweep cfg.expt.paramName, 
        # include paramName here in this list 
        # AND include cfg.expt.paramNames (note the s) as a list of values to step thru.
        # (You want a list instead of numpy array for better yaml export.)
        # Currently handles 1D and 2D sweeps and plots only.
        # For 2D, order is [outer, inner].
        normalize = False,
        active_reset = True,
        man_reset = True, 
        storage_reset = True, 
        advance_phase = 0, # advance phase of second pi/2 by this much [degrees]
        echoes = [False, 0], # [on/off, number of echoes]
    )
    
    sta = SidebandStarkExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"SidebandStark_S{expt_params['init_stor']}",
        config_file=config_path,
        expt_params = expt_params,
        progress=True)
    
    sta.cfg.device.readout.relax_delay = [200]  # Wait time between experiments [us]
    sta.go(analyze=False, display=False, progress=True, save=True)

# %% [markdown]
# ### new base class general 2D

# %%
from multimode_expts.experiments.qsim.sideband_stark import SidebandStarkProgram
from multimode_expts.experiments.qsim.qsim_base import QsimBaseExperiment

# %%
for init_stor in [2]: #range(1,8):
    ro_stor = 0
    expt_params = dict(
        expts = 1,
        reps = 100,
        rounds = 1,
        qubits = [0],
        f0g1_cavity = 1,  #  1/2 name of manipulate cavity
        init_stor = init_stor, # storage mode number to initialize to n=1 Fock state
        ro_stor = ro_stor, # storage mode number that gets read out in the end
        # if 0, this means to read out man instead
        detune = 0,
        # detunes=np.linspace(-0.1,0.1,101).tolist(),
        advance_phase = 0, # advance phase of second pi/2 by this much [degrees]
        # advance_phases=np.linspace(-90,90,91).tolist(),
        # wait=10, # wait time between two hpi pulses in us
        waits = np.linspace(0,20,101).tolist(),
        swept_params = ['wait'],
        # usage: if you want to sweep cfg.expt.paramName, 
        # include paramName here in this list 
        # AND include cfg.expt.paramNames (note the s) as a list of values to step thru.
        # (You want a list instead of numpy array for better yaml export.)
        # Currently handles 1D and 2D sweeps and plots only.
        # For 2D, order is [outer, inner].
        normalize = False,
        active_reset = False,
        man_reset = True, 
        storage_reset = True, 
        echoes = [False, 0], # [on/off, number of echoes]
    )
    
    qbe = QsimBaseExperiment(
        soccfg=soc,
        path=expt_path,
        prefix=f"SidebandStark_S{expt_params['init_stor']}",
        config_file=config_path,
        expt_params=expt_params,
        program=SidebandStarkProgram,
        progress=True)
    
    qbe.cfg.device.readout.relax_delay = [8000]  # Wait time between experiments [us]
    qbe.go(analyze=False, display=True, progress=True, save=True)

# %%

# %% [markdown]
# ### T1

# %%
from multimode_expts.experiments.qsim.sideband_scramble import StorageT1Program

# %%
init_stor = 0
expt_params = dict(
    expts = 1,
    reps = 200,
    rounds = 1,
    qubits = [0],
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    init_stor = init_stor, # storage mode number to initialize to n=1 Fock state
    ro_stor = init_stor, # storage mode number that gets read out in the end
    # if 0, this means to read out man instead
    # wait=10, # wait time between two hpi pulses in us
    waits = np.linspace(0,500,51).tolist(),
    swept_params = ['wait'],
    # usage: if you want to sweep cfg.expt.paramName, 
    # include paramName here in this list 
    # AND include cfg.expt.paramNames (note the s) as a list of values to step thru.
    # (You want a list instead of numpy array for better yaml export.)
    # Currently handles 1D and 2D sweeps and plots only.
    # For 2D, order is [outer, inner].
    normalize = False,
    active_reset = False,
    man_reset = True, 
    storage_reset = True, 
    prepulse = True,
    postpulse = True,
    echoes = [False, 0], # [on/off, number of echoes]
)

qbe = QsimBaseExperiment(
    soccfg=soc,
    path=expt_path,
    prefix=f"StorageT1_S{expt_params['init_stor']}",
    config_file=config_path,
    expt_params=expt_params,
    program=StorageT1Program,
    progress=True)

qbe.cfg.device.readout.relax_delay = [5000]  # Wait time between experiments [us]
qbe.go(analyze=False, display=True, progress=True, save=True)

