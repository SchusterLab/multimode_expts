# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Multimode (local)
#     language: python
#     name: multimode-local
# ---

# %% [markdown]
# # Prepare

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import qutip as qt
from copy import deepcopy

import experiments as meas
from experiments.MM_dual_rail_base import MM_dual_rail_base
from fitting.fit_display_classes import GeneralFitting
from fitting.wigner import WignerAnalysis

from slab import AttrDict
from experiments import MultimodeStation, CharacterizationRunner, SweepRunner

from job_server import JobClient
from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager

# %%
# Initialize database and config manager
db = get_database()
config_dir = 'C:/python/multimode_expts/configs'
config_manager = ConfigVersionManager(config_dir)

# Initialize job client (handle submitting and waiting for jobs)
client = JobClient()

# Check server health
health = client.health_check()
client.print_queue()

# %%
# Who is running these experiments??
user = 'Jonginn'

print(f"Welcome {user}!")

# %%
config_dict = {'hardware_config': 'CFG-HW-20261120-00001',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20261119-00011',
 'floquet_storage_swap': 'CFG-FL-20261119-00022'}

# Initialize station to retrieve soc and configs
station = MultimodeStation(
    user = user,
    # experiment_name = "260130_qsim_wigner",
    # experiment_name = "260206_dump_mode",
    experiment_name = "1111_recalibration_after_recooldown",
    project = "QubitCalibration",
    
    storage_man_file = config_dict['man1_storage_swap'],
    hardware_config= config_dict['hardware_config'],
    # hardware_config = "CFG-HW-20260402-00005",
    floquet_file=config_dict['floquet_storage_swap'],

    # storage_man_file="CFG-M1-20260218-00025",
    # hardware_config="CFG-HW-20260211-00039",
    # floquet_file="versions/floquet_storage_swap/CFG-FL-20260520-00002.csv",
    # storage_man_file="versions/man1_storage_swap/CFG-M1-20260519-00015.csv",
    # multiphoton_config="versions/multiphoton_config/CFG-MP-20260115-00001.yml",
    
    # experiment_name = "251111_qec",
    
    log_measurements=True,
)

# %% [markdown]
# ## Experiments to run
#
# Depending on stage of cooldown, we will run a different sequence of calibration experiments. For example, amplitude rabi don't need to be updated every time, but the frequency correction from T2 is important to do every day. In the dictionary experiments to run, we will speciy the experiments we want to run. 

# %%
expts_to_run = {# readout 
                'res_spec': False, # Readout spectroscopy
                'single_shot': True, 
                # qubit ge 
                'pulse_probe_ge': True,
                't2_ge': True, 
                'amplitude_ge': True,
                't1_ge': True,
                # qubit ef
                'pulse_probe_ef': True,
                't2_ef': True,
                'amplitude_ef': True,
                't1_ef': True,

                # manipulate 
                'man_modes': [2], # [2] for M2, [1, 2] for both
                'pulse_probe_f0g1': True,
                'length_rabi_sweep':True,
                'length_rabi':False, # this will run automatically if the length_rabi_sweep is set to True
                'chi_ge': True, 
                'chi_ef': True,
                'man_parity_spectroscopy': False,
                'man_gain_to_alpha': False,
                'man_self_kerr': False,
                'RB': False,

                #storage
                'stor_modes': [1,2,3,4,5,6,7], # [1,2, .., 7] if also want to run  all modes 
                # 'stor_modes': [3, 4, 5], # [1,2, .., 7] if also want to run  all modes 
                'stor_spectroscopy': True,
                'sideband_freq_sweep': True,
                'sideband_length_rabi': True,
                # 'storage_t1': True
                }

man_modes = tuple(expts_to_run['man_modes'])
if any(mode not in (1, 2) for mode in man_modes):
    raise ValueError("man_modes supports M1 and M2")
stor_modes = tuple(expts_to_run['stor_modes'])
reset_dump_modes = {1: 2, 2: 2}


def m1_only(stage):
    if 2 in man_modes:
        print(f'{stage}: skipped M2 because the existing pulse program uses M1.')
    return tuple(mode for mode in man_modes if mode == 1)

# %% [markdown]
# # Qubit characterization

# %% [markdown]
# ## Update main configs

# %%
# When ready uncomment:
# station.update_all_station_snapshots(update_main=True)

# %% [markdown]
# ## Resonator spectroscopy

# %% [markdown]
# Fitting parameters are wrong because of using the hanger function (more or less reflection/2), instead of transmission. Is this an easy fix?

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
resspec_defaults = AttrDict(dict(
    # start = center - span / 2, 
    # step = span / expts, # min step ~1 Hz
    # center and span can be user supplied or use defaults in preproc
    expts = 250, # Number experiments stepping from start
    reps = 500, # Number averages per point
    relax_delay = 50, # us
    pulse_e = False, # add ge pi pulse prior to measurement
    pulse_f = False, # add ef pi pulse prior to measurement
    pulse_cavity = False,  # prepulse on cavity prior to measurement (False also disables next line)
    cavity_pulse = [4984.373226159381, 800, 2, 0], # [frequency, gain, length, phase]  const pulse
    qubit = 0,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def resspec_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)

    span = kwargs.pop('span', 1.5)  # MHz
    center = kwargs.pop('center', station.hardware_cfg.device.readout.frequency[0])
    center = 749.9
    expts = kwargs.get('expts', default_expt_cfg.expts)

    expt_cfg.start = center - span / 2
    expt_cfg.step = span / expts
    expt_cfg.update(kwargs)
    station.snapshot_hardware_config(update_main=False)
    return expt_cfg

def resspec_postproc(station, expt):
    old_freq = station.hardware_cfg.device.readout.frequency[0]
    station.hardware_cfg.device.readout.frequency = [expt.data['fit'][0]]
    print(f'Updated readout frequency from {old_freq} to {expt.data["fit"][0]}!')



# %%
# Execute
# =================================
rspec_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.ResonatorSpectroscopyExperiment,
    default_expt_cfg = resspec_defaults,
    preprocessor = resspec_preproc,
    postprocessor = None, # resspec_postproc, # uncomment if you want to update the readout freq from this expt
    job_client=client,
)

if expts_to_run['res_spec']:
    rspec = rspec_runner.execute()
    rspec.display()

# %%
# station.preview_config_update()

# %% [markdown]
# ## Single Shot

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
    cavity_freq=4984.373226159381,
    cavity_gain=400,
    cavity_length=2,
    prepulse=False,
    pre_sweep_pulse=None,
    gate_based=True,
    qubits=[0],
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def singleshot_postproc(station, expt):
    expt.analyze(station=station, subdir=station.autocalib_path)
    station.hardware_cfg.device.readout.update(expt.readout_cfg)


# %%
# Execute
# =================================
ss_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.HistogramExperiment,
    default_expt_cfg = singleshot_defaults,
    postprocessor = singleshot_postproc,
    job_client=client,
)

if expts_to_run['single_shot']:
    ss = ss_runner.execute(
        check_f=False,
        active_reset=True, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
        relax_delay=2000,
        ef_reset = False,


        reps=5000,
        # active_reset=True,
        # relax_delay=200,
    )
    ss.display(station)

# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
num_col = 2
if ss.cfg.expt.ef_reset:
    num_col = 3


fig, ax = plt.subplots(2, num_col, figsize = (3*num_col, 5))

size = 0.5
alpha = 0.5

for i in range(num_col):
    ax[0, i].scatter(ss.data['Ig'][i::num_col], ss.data['Qg'][i::num_col], label = 'ground', s = size, alpha = alpha)
    ax[0, i].scatter(ss.data['Ie'][i::num_col], ss.data['Qe'][i::num_col], label = 'excited', s = size, alpha = alpha)
    ax[1, i].hist(ss.data['Ig'][i::num_col], bins = 128, alpha = alpha)
    ax[1, i].hist(ss.data['Ie'][i::num_col], bins = 128, alpha = alpha)
fig.tight_layout()

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## Qubit ge

# %% [markdown]
# ### Pulse-probe

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
gespec_defaults = AttrDict(dict(    
    # start=center-span/2,  # [MHz]
    # step=span/expts,  # min step ~1 MHz
    expts=200,  # Number of experiments stepping from start
    reps=100,  # Number of averages per point
    rounds=1,  # Number of start to finish sweeps to average over
    length=1,  # Qubit probe constant pulse length [us]
    gain=100,  # Qubit pulse gain
    sigma=0.1,  # Qubit flat top sigma
    qubit=[0],
    qubits=[0],
    prepulse=False,
    pre_sweep_pulse=[],
    gate_based=False,
    relax_delay=250,  # Wait time between experiments [us]
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values


def gespec_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)

    span = kwargs.pop('span', 10)  # MHz
    center = kwargs.pop('center', station.hardware_cfg.device.qubit.f_ge[0])
    expts = kwargs.pop('expts', default_expt_cfg.expts)

    expt_cfg.start = center - span / 2
    expt_cfg.step = span / expts
    expt_cfg.expts = expts
    
    expt_cfg.update(kwargs)
    return expt_cfg

def gespec_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ge[0]
    station.hardware_cfg.device.qubit.f_ge = [expt.data['fit_avgi'][2]]
    print(f'Updated qubit frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ge[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
gespec_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.PulseProbeSpectroscopyExperiment,
    default_expt_cfg = gespec_defaults,
    preprocessor = gespec_preproc,
    postprocessor = gespec_postproc,
    job_client=client,
)

if expts_to_run['pulse_probe_ge']:
    gespec = gespec_runner.execute(
        # coupler_current=coupler_current,
        span = 10,
        expts = 200
    )
    gespec.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# ### T2 Ramsey

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
from sqlalchemy import false


geramsey_defaults = AttrDict(dict(
    ramsey_freq=0.2,  # [MHz]
    start=0.01, # us
    step=0.5,
    expts=101,
    reps=200,
    rounds=1,
    pre_sweep_pulse=None,
    post_sweep_pulse=None,
    if_ef=False,
    ef_init=True, # redundant
    qubits=[0],
    user_defined_freq=[False, 3568.2038290468167, 5304, 0.035],
    f0g1_cavity=0,
    normalize=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    prepulse=None,
    postpulse=None,
    pre_active_reset_pulse=False,
    gate_based=False,
    advance_phase=0,
    echoes=[False, 0],
    relax_delay=200
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def geramsey_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    # Copied over but what's all this below????
    # Which ones are actually used? 
    # Can we straighten out all the nested boolean logic???
    checkEF = False
    qubit_ge_init = False
    qubit_ge_after = False
    if expt_cfg.if_ef:
        checkEF = True
        qubit_ge_init = True if expt_cfg.ef_init else False
        qubit_ge_after = True if expt_cfg.ef_init else False
    expt_cfg.checkEF = checkEF
    expt_cfg.qubit_ge_init = qubit_ge_init
    expt_cfg.qubit_ge_after = qubit_ge_after

    expt_cfg.prepulse = False if expt_cfg.pre_sweep_pulse is None else True if expt_cfg.prepulse is None else expt_cfg.prepulse
    expt_cfg.postpulse = False if expt_cfg.post_sweep_pulse is None else True if expt_cfg.postpulse is None else expt_cfg.postpulse
    
    return expt_cfg

def geramsey_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ge[0]
    station.hardware_cfg.device.qubit.f_ge = [
        station.hardware_cfg.device.qubit.f_ge[0] + min(expt.data['f_adjust_ramsey_avgi'])
    ]
    print(f'Updated qubit frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ge[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
geramsey_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.RamseyExperiment,
    default_expt_cfg = geramsey_defaults,
    preprocessor = geramsey_preproc,
    postprocessor = geramsey_postproc,
    job_client=client,
)

if expts_to_run['t2_ge']:
    geramsey = geramsey_runner.execute(
        ramsey_freq = 0.2, #0.2
        step=0.4, #0.4
        active_reset=False,
        relax_delay=2500,
        postprocess=True,
        if_ef=False,
        expts = 100
    )
    geramsey.display()

# %% [markdown]
# ### Amplitude Rabi

# %% [markdown]
# We should probably use a cosine fit with fixed phase=0 instead of decaying sine with varying phase?

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
amprabi_defaults = AttrDict(dict(
    start=0,
    step=60,
    expts=151,
    reps=200,
    rounds=1,
    sigma_test=None,
    qubit=0,
    pulse_type='gauss',
    drag_beta=0.0,
    pulse_ge_init=False,
    pulse_ge_after=False,
    checkZZ=False,
    checkEF=False,
    qubits=[0],
    flat_length=0,
    normalize=False,
    single_shot=False,
    singleshot_reps=10000,
    span=50,
    user_defined_freq=[False, 3568.203829046816],
    prepulse=False,
    postpulse=False, 
    if_ef=False,  # If true, will check ef frequency and update it
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

def amprabi_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    # Copied over but what's all this below????
    # Which ones are actually used? 
    # Can we straighten out all the nested boolean logic???
    pulse_ge = station.hardware_cfg.device.qubit.pulses.pi_ge
    if expt_cfg.sigma_test is None:
        expt_cfg.sigma_test = pulse_ge.sigma[0]
    if expt_cfg.step is None:
        expt_cfg.step = int(pulse_ge.gain[0] / (expt_cfg.expts - 1))
    
    expt_cfg.checkEF = False
    expt_cfg.pulse_ge_init = False
    expt_cfg.pulse_ge_after = False
    if expt_cfg.if_ef:
        expt_cfg.checkEF = True
        expt_cfg.pulse_ge_init = True
        expt_cfg.pulse_ge_after = True
    
    return expt_cfg

def amprabi_postproc(station, expt):
    station.hardware_cfg.device.qubit.pulses.pi_ge.gain = [expt.data['pi_gain_avgi']]
    station.hardware_cfg.device.qubit.pulses.hpi_ge.gain = [expt.data['hpi_gain_avgi']]
    print('Updated qubit ge pi and hpi gaussian gain!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
amprabi_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.AmplitudeRabiExperiment,
    default_expt_cfg = amprabi_defaults,
    preprocessor = amprabi_preproc,
    postprocessor = amprabi_postproc,
    job_client=client,
)

if expts_to_run['amplitude_ge']:
    amprabi = amprabi_runner.execute(
        relax_delay=2500,
        expts=301,
        postprocess=True,
    )
    amprabi.display()

    t2_ramsey_ge_after_amp = geramsey_runner.execute(
        ramsey_freq = 0.2, #0.2,
        step = 0.4, #0.4,
        active_reset = True,
        relax_delay=250,
    )
    t2_ramsey_ge_after_amp.display()


# %%
station.preview_config_update()
station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# ### Error Amplification

# %%
# Error Amplification - Runner Pattern with CharacterizationRunner
# =====================================
error_amp_defaults = AttrDict(dict(
    reps=100,
    rounds=1,
    qubit=0,
    n_pulses=10,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500,
    start=0,
    expts=10,
    step=100,
    parameter_to_test='gain',  # 'gain' or 'frequency'
    pulse_type=['qubit', 'ge', 'pi', 0],
    # Multiphoton-specific parameters
    man_no=1,  # Manipulate mode number
    pulse_name='pi',  # 'pi' or 'hpi'
))


def error_amp_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    expt_cfg.qubits = [expt_cfg.qubit]
    
    return expt_cfg

def error_amp_postproc(station, expt, state_fin='g'):
    """Update config based on error amplification results.
    
    For multiphoton transitions, updates the multiphoton config.
    For standard qubit calibration, updates the qubit pulse params.
    """
    expt_cfg = expt.cfg.expt
    parameter = expt_cfg.parameter_to_test
    pulse_name = expt_cfg.pulse_type[2]  # 'pi' or 'hpi'
    transition_key = expt_cfg.pulse_type[1] # 'ge' or 'ef'
    print(f'Processing error amplification for parameter: {parameter}, pulse: {pulse_name}')


    if state_fin == 'e':
        expt.analyze(data=expt.data, state_fin='e')
        expt.display(data=expt.data, state_fin='e')

    
    # Get the optimal value from the experiment data
    # Error amplification typically finds the value that minimizes error
    if 'optimal_value' in expt.data:
        optimal_val = expt.data['optimal_value']

    if 'fit_avgi' in expt.data:
        print('parameter:', parameter)
        if parameter == 'frequency':
            optimal_val = expt.data['fit_avgi'][2]
        elif parameter == 'gain':
            optimal_val = int(expt.data['fit_avgi'][2])
        else: 
            raise ValueError(f"Unknown parameter to test: {parameter}")

    else:
        # If no automatic extraction, user needs to analyze manually
        print('No automatic optimal value extraction. Please analyze the data manually.')
        return
    
    # Standard qubit update
    pulse_key = f'{pulse_name}_{transition_key}'
    freq_key = f"f_{transition_key}"
    if parameter == 'frequency':
        old_val = station.hardware_cfg.device.qubit[freq_key][0]
        station.hardware_cfg.device.qubit[freq_key][0] = optimal_val
        print(f'Updated qubit {freq_key} {parameter}: {old_val} -> {optimal_val}')
    elif parameter == 'gain':
        old_val = station.hardware_cfg.device.qubit.pulses[pulse_key]['gain'][0]
        station.hardware_cfg.device.qubit.pulses[pulse_key]['gain'][0] = optimal_val
        print(f'Updated qubit {pulse_key} {parameter}: {old_val} -> {optimal_val}')
        


# %%
error_amp_runner_qubit = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_defaults,
    preprocessor=error_amp_preproc,
    postprocessor=error_amp_postproc,
    job_client=client,
    use_queue=True,
)

# %% [markdown]
# #### Frequency

# %%
expts = 50 #50
band = 1 #2
freq_start = station.hardware_cfg.device.qubit['f_ge'][0] - band
# freq_start = 3569 - band
step = 2 * band / expts
print(f'Frequency start: {freq_start}, Step: {step}')



error_amp = error_amp_runner_qubit.execute(
    parameter_to_test='frequency',
    step=2 * band / expts, 
    start=freq_start,
    expts=expts,
    reps=50,
    pulse_type=['qubit', 'ge', 'pi', 0],
    # n_pulses=20,
)


# %% [markdown]
# #### Amplitude

# %%
band = 1000
expts = 25
gain_start = station.hardware_cfg.device.qubit.pulses.pi_ge.gain[0] - band
# gain_start = 10000
step = 2 * band / expts
print(f'Gain start: {gain_start}, Step: {step}')

error_amp = error_amp_runner_qubit.execute(
    parameter_to_test='gain',
    step=step,
    start=gain_start,
    expts=expts,
    reps=50,
    pulse_type=['qubit', 'ge', 'pi', 0],
)

# %% [markdown]
# ### T1

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
t1_ge_defaults = AttrDict(dict(
    start=0,
    step=20,
    expts=100,
    reps=100,
    rounds=1,
    qubit=0,
    qubit_ef=False,
    normalize=False,
    relax_delay=2500,
))

def t1_ge_postproc(station, expt):
    station.hardware_cfg.device.qubit.T1 = [expt.data['fit_avgi'][3]]
    print('Updated qubit T1!')
    station.snapshot_hardware_config(update_main=False)



# %%
# Execute
# =================================
t1_ge_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.t1.T1Experiment,
    default_expt_cfg=t1_ge_defaults,
    postprocessor=t1_ge_postproc,
    job_client=client,
)

if expts_to_run['t1_ge']:
    t1_ge = t1_ge_runner.execute()
    t1_ge.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# ## Qubit ef

# %% [markdown]
# ### Pulse-probe

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
efspec_defaults = AttrDict(dict(
    start=3415,
    step=0.05,
    expts=500,
    reps=200,
    rounds=1,
    length=1,
    gain=100,
    qubit_f=False,
    qubit=0,
    cavity_drive=False,
    wait_qubit=False,
    relax_delay=500,
))

def efspec_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ef[0]
    station.hardware_cfg.device.qubit.f_ef = [expt.data['fit_avgi'][2]]
    print(f'Updated qubit ef frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ef[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
efspec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.PulseProbeEFSpectroscopyExperiment,
    default_expt_cfg=efspec_defaults,
    postprocessor=efspec_postproc,
    job_client=client,
)

if expts_to_run['pulse_probe_ef']:
    qspec_ef = efspec_runner.execute(
    expts=500)
    qspec_ef.display()

# %% [markdown]
# ### T2 Ramsey

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
# Reuse ge defaults but with if_ef=True
eframsey_defaults = AttrDict(deepcopy(geramsey_defaults))
eframsey_defaults.if_ef = True
eframsey_defaults.ef_init = True
eframsey_defaults.ramsey_freq = 3  # Typical ef ramsey frequency

def eframsey_postproc(station, expt):
    old_freq = station.hardware_cfg.device.qubit.f_ef[0]
    station.hardware_cfg.device.qubit.f_ef = [
        station.hardware_cfg.device.qubit.f_ef[0] + min(expt.data['f_adjust_ramsey_avgi'])
    ]
    print(f'Updated qubit ef frequency from {old_freq} to {station.hardware_cfg.device.qubit.f_ef[0]}!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
eframsey_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.RamseyExperiment,
    default_expt_cfg=eframsey_defaults,
    preprocessor=geramsey_preproc,  # Reuse ge preprocessor
    postprocessor=eframsey_postproc,
    job_client=client,
)

if expts_to_run['t2_ef']:
    t2ramsey_ef = eframsey_runner.execute(
        ramsey_freq=0.1, #0.2
        step = 0.2, #0.4 0.2
        relax_delay=2500,
        active_reset=True,
        expts = 100
    )
    t2ramsey_ef.display()

# %% [markdown]
# ### Amplitude Rabi

# %% [markdown]
# We should probably use a cosine fit with fixed phase=0 instead of decaying sine with varying phase?

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
# Reuse ge defaults but with if_ef=True
efamprabi_defaults = AttrDict(deepcopy(amprabi_defaults))
efamprabi_defaults.if_ef = True
efamprabi_defaults.step = 40  # Typical ef amplitude step
efamprabi_defaults.expts = 150

def efamprabi_postproc(station, expt):
    station.hardware_cfg.device.qubit.pulses.pi_ef.gain = [expt.data['pi_gain_avgi']]
    station.hardware_cfg.device.qubit.pulses.hpi_ef.gain = [expt.data['hpi_gain_avgi']]
    print('Updated qubit ef pi and hpi gaussian gain!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
efamprabi_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.AmplitudeRabiExperiment,
    default_expt_cfg=efamprabi_defaults,
    preprocessor=amprabi_preproc,  # Reuse ge preprocessor
    postprocessor=efamprabi_postproc,
    job_client=client,
)

if expts_to_run['amplitude_ef']:
    amprabi_ef = efamprabi_runner.execute(
                relax_delay=2500,
                expts = 500
    )           
    amprabi_ef.display()
    
    # After amplitude calibration, do another T2 Ramsey to fine tune frequency
    t2_ramsey_ef_after_amp = eframsey_runner.execute(
        ramsey_freq=0.5,
        step = 0.1,
        relax_delay=2500,
    )
    t2_ramsey_ef_after_amp.display()

# %% [markdown]
# ### Error Amplification

# %%
# Error Amplification - Runner Pattern with CharacterizationRunner
# =====================================
error_amp_defaults = AttrDict(dict(
    reps=100,
    rounds=1,
    qubit=0,
    n_pulses=10,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500,
    start=0,
    expts=10,
    step=100,
    parameter_to_test='gain',  # 'gain' or 'frequency'
    pulse_type=['qubit', 'ge', 'pi', 0],
    # Multiphoton-specific parameters
    man_no=1,  # Manipulate mode number
    pulse_name='pi',  # 'pi' or 'hpi'
))


def error_amp_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    expt_cfg.qubits = [expt_cfg.qubit]
    
    return expt_cfg

def error_amp_postproc(station, expt, state_fin='g'):
    """Update config based on error amplification results.
    
    For multiphoton transitions, updates the multiphoton config.
    For standard qubit calibration, updates the qubit pulse params.
    """
    expt_cfg = expt.cfg.expt
    parameter = expt_cfg.parameter_to_test
    pulse_name = expt_cfg.pulse_type[2]  # 'pi' or 'hpi'
    transition_key = expt_cfg.pulse_type[1] # 'ge' or 'ef'
    print(f'Processing error amplification for parameter: {parameter}, pulse: {pulse_name}')


    if state_fin == 'e':
        expt.analyze(data=expt.data, state_fin='e')
        expt.display(data=expt.data, state_fin='e')

    
    # Get the optimal value from the experiment data
    # Error amplification typically finds the value that minimizes error
    if 'optimal_value' in expt.data:
        optimal_val = expt.data['optimal_value']

    if 'fit_avgi' in expt.data:
        print('parameter:', parameter)
        if parameter == 'frequency':
            optimal_val = expt.data['fit_avgi'][2]
        elif parameter == 'gain':
            optimal_val = int(expt.data['fit_avgi'][2])
        else: 
            raise ValueError(f"Unknown parameter to test: {parameter}")

    else:
        # If no automatic extraction, user needs to analyze manually
        print('No automatic optimal value extraction. Please analyze the data manually.')
        return
    
    # Standard qubit update
    pulse_key = f'{pulse_name}_{transition_key}'
    freq_key = f"f_{transition_key}"
    if parameter == 'frequency':
        old_val = station.hardware_cfg.device.qubit[freq_key][0]
        station.hardware_cfg.device.qubit[freq_key][0] = optimal_val
        print(f'Updated qubit {freq_key} {parameter}: {old_val} -> {optimal_val}')
    elif parameter == 'gain':
        old_val = station.hardware_cfg.device.qubit.pulses[pulse_key]['gain'][0]
        station.hardware_cfg.device.qubit.pulses[pulse_key]['gain'][0] = optimal_val
        print(f'Updated qubit {pulse_key} {parameter}: {old_val} -> {optimal_val}')
        
error_amp_runner_qubit = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_defaults,
    preprocessor=error_amp_preproc,
    postprocessor=error_amp_postproc,
    job_client=client,
    use_queue=True,
)

# %% [markdown]
# #### Frequency

# %%
expts = 50 #50
band = 1 #2
freq_start = station.hardware_cfg.device.qubit['f_ef'][0] - band
# freq_start = 3569 - band
step = 2 * band / expts
print(f'Frequency start: {freq_start}, Step: {step}')



error_amp = error_amp_runner_qubit.execute(
    parameter_to_test='frequency',
    step=2 * band / expts, 
    start=freq_start,
    expts=expts,
    reps=50,
    pulse_type=['qubit', 'ef', 'pi', 0],
    # n_pulses=20,
)


# %% [markdown]
# #### Amplitude

# %%
band = 500
expts = 25
gain_start = station.hardware_cfg.device.qubit.pulses.pi_ef.gain[0] - band
# gain_start = 10000
step = 2 * band / expts
print(f'Gain start: {gain_start}, Step: {step}')

error_amp = error_amp_runner_qubit.execute(
    parameter_to_test='gain',
    step=step,
    start=gain_start,
    expts=expts,
    reps=50,
    pulse_type=['qubit', 'ef', 'pi', 0],
)

# %% [markdown]
# ### T1

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
t1_ef_defaults = AttrDict(dict(
    start=0,
    step=5,
    expts=100,
    reps=50,
    rounds=1,
    qubit=0,
    qubit_ef=True,
    normalize=False,
    relax_delay=2500,
))

def t1_ef_postproc(station, expt):
    station.hardware_cfg.device.qubit.T1_ef = [expt.data['fit_avgq'][3]]
    print('Updated qubit T1 ef!')
    station.snapshot_hardware_config(update_main=False)


# %%
# Execute
# =================================
t1_ef_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.T1Experiment,
    default_expt_cfg=t1_ef_defaults,
    postprocessor=t1_ef_postproc,
    job_client=client,
)

if expts_to_run['t1_ef']:
    t1_ef = t1_ef_runner.execute()
    t1_ef.display()

# %%
station.preview_config_update()
# station.snapshot_hardware_config(update_main=True)

# %%
station.snapshot_hardware_config()

# %% [markdown]
# ## Slow pi ge calibration
#
# Run the fixed-gain length Rabi first, then the slow-pulse Ramsey. Both postprocessors update only the in-memory station config. Save YAML manually after the calibration is finished.

# %%
from pathlib import Path
import fitting.fitting as fitter
from fitting.fit_display_classes import LengthRabiFitting, RamseyFitting
from experiments.qsim.slow_pi_ge_calibration import (
    SlowPiGeLengthRabiProgram,
    SlowPiGeRamseyProgram,
)


slow_pi_ge_length_defaults = AttrDict(dict(
    reps=200,
    rounds=1,
    qubits=[0],
    swept_params=['length_to_sweep'],
    length_to_sweeps=np.linspace(0.01, 30., 151).tolist(),
    active_reset=False,
    prepulse=False,
    postpulse=False,
    init_stor=0,
    ro_stor=0,
    normalize=False,
    relax_delay=2500,
))


def slow_pi_ge_length_preproc(
        station, default_expt_cfg, fixed_gain=None, lengths=None, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    pulse = station.hardware_cfg.device.qubit.pulses.slow_pi_ge
    if pulse.type[0] != 'flat_top':
        raise ValueError(
            f"slow_pi_ge must be flat_top, got {pulse.type[0]!r}"
        )

    if fixed_gain is not None:
        pulse.gain[0] = int(fixed_gain)
    if lengths is not None:
        expt_cfg.length_to_sweeps = np.asarray(
            lengths, dtype=float
        ).tolist()

    print(
        'slow_pi_ge fixed gain =', pulse.gain[0],
        ', ramp sigma =', pulse.sigma[0], 'us'
    )
    return expt_cfg


def slow_pi_ge_length_postproc(station, expt):
    analysis = LengthRabiFitting(
        data=expt.data,
        config=expt.cfg,
        station=station,
    )
    analysis.analyze()

    i_range = np.ptp(np.asarray(expt.data['avgi'], dtype=float))
    q_range = np.ptp(np.asarray(expt.data['avgq'], dtype=float))
    fit_channel = 'avgi' if i_range >= q_range else 'avgq'
    fit = np.asarray(
        expt.data[f'fit_{fit_channel}'], dtype=float
    ).copy()

    pi_length, hpi_length = analysis._calculate_pi_lengths(fit)
    period = 1. / abs(fit[1])
    first_length = float(np.min(expt.data['xpts']))
    while hpi_length < first_length:
        pi_length += period
        hpi_length += period

    last_length = float(np.max(expt.data['xpts']))
    if pi_length > last_length:
        raise RuntimeError(
            f'Fitted pi length {pi_length:.6f} us is outside '
            f'the measured range ending at {last_length:.6f} us'
        )

    pulse = station.hardware_cfg.device.qubit.pulses.slow_pi_ge
    old_length = float(pulse.length[0])
    pulse.length[0] = float(pi_length)

    expt.data['slow_pi_ge_fit_channel'] = fit_channel
    expt.data['pi_length'] = float(pi_length)
    expt.data['hpi_length'] = float(hpi_length)
    print(
        f'Updated in-memory slow_pi_ge length: '
        f'{old_length:.6f} -> {pi_length:.6f} us'
    )
    print(f'slow_pi_ge hpi length: {hpi_length:.6f} us')
    print('YAML was not written.')



# %%
slow_pi_ge_length_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimExperiment,
    ExptProgram=SlowPiGeLengthRabiProgram,
    default_expt_cfg=slow_pi_ge_length_defaults,
    preprocessor=slow_pi_ge_length_preproc,
    postprocessor=slow_pi_ge_length_postproc,
    job_client=client,
    use_queue=True,
)


# %%
# Keep the gain low and calibrate the pulse by changing its length.
# slow_pi_ge_fixed_gain = 15
slow_pi_ge_lengths = np.linspace(0.01, 30., 151)

slow_pi_ge_length_rabi = slow_pi_ge_length_runner.execute(
    # fixed_gain=slow_pi_ge_fixed_gain,
    lengths=slow_pi_ge_lengths,
    postprocess=True,
)

x = np.asarray(slow_pi_ge_length_rabi.data['xpts'], dtype=float)
x_fit = np.linspace(x.min(), x.max(), 1000)
pi_length = float(slow_pi_ge_length_rabi.data['pi_length'])
hpi_length = float(slow_pi_ge_length_rabi.data['hpi_length'])
file_name = Path(slow_pi_ge_length_rabi.fname).name

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
for ax, channel, label in zip(
        axes, ['avgi', 'avgq'], ['I', 'Q']
):
    ax.plot(x, slow_pi_ge_length_rabi.data[channel], 'o-', label=label)
    fit = slow_pi_ge_length_rabi.data[f'fit_{channel}']
    ax.plot(x_fit, fitter.decaysin(x_fit, *fit), label='fit')
    ax.axvline(hpi_length, color='0.3', ls='--', label='hpi')
    ax.axvline(pi_length, color='0.3', ls=':', label='pi')
    ax.set_ylabel(f'{label} [ADC units]')
    ax.legend()

axes[0].set_title(file_name)
axes[-1].set_xlabel('slow pi ge flat-top length (us)')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Slow-pulse Ramsey

# %%
slow_pi_ge_ramsey_defaults = AttrDict(dict(
    ramsey_freq=0.2,
    wait_times=(
        0.01 + 0.2 * np.arange(101)
    ).tolist(),
    hpi_length=None,
    reps=200,
    rounds=1,
    qubits=[0],
    swept_params=['wait_time'],
    active_reset=False,
    prepulse=False,
    postpulse=False,
    init_stor=0,
    ro_stor=0,
    normalize=False,
    relax_delay=2500,
    checkEF=False,
    echoes=[False, 0],
))


def slow_pi_ge_ramsey_preproc(
        station, default_expt_cfg, wait_times=None, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)

    if wait_times is not None:
        expt_cfg.wait_time_sweeps = np.asarray(
            wait_times, dtype=float
        ).tolist()
    if expt_cfg.hpi_length is None:
        raise ValueError(
            'Pass the hpi_length measured by the slow length Rabi'
        )

    return expt_cfg


def slow_pi_ge_ramsey_postproc(station, expt):
    analysis = RamseyFitting(
        data=expt.data,
        config=expt.cfg,
        station=station,
    )
    analysis.analyze()

    i_range = np.ptp(np.asarray(expt.data['avgi'], dtype=float))
    q_range = np.ptp(np.asarray(expt.data['avgq'], dtype=float))
    fit_channel = 'avgi' if i_range >= q_range else 'avgq'
    frequency_correction = float(min(
        expt.data[f'f_adjust_ramsey_{fit_channel}'],
        key=abs,
    ))

    old_frequency = float(
        station.hardware_cfg.device.qubit.f_ge[0]
    )
    new_frequency = old_frequency + frequency_correction
    station.hardware_cfg.device.qubit.f_ge[0] = new_frequency

    expt.data['slow_pi_ge_fit_channel'] = fit_channel
    expt.data['drive_frequency'] = old_frequency
    expt.data['frequency_correction'] = frequency_correction
    print(
        f'Updated in-memory qubit f_ge: '
        f'{old_frequency:.9f} -> {new_frequency:.9f} MHz'
    )
    print('YAML was not written.')



# %%
slow_pi_ge_ramsey_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimExperiment,
    ExptProgram=SlowPiGeRamseyProgram,
    default_expt_cfg=slow_pi_ge_ramsey_defaults,
    preprocessor=slow_pi_ge_ramsey_preproc,
    postprocessor=slow_pi_ge_ramsey_postproc,
    job_client=client,
    use_queue=True,
)


# %%
# Repeat length Rabi -> Ramsey as needed.
slow_pi_ge_ramsey_freq = 0.2
slow_pi_ge_ramsey_wait_times = (
    0.01 + 0.2 * np.arange(101)
)
slow_pi_ge_hpi_length = float(
    slow_pi_ge_length_rabi.data['hpi_length']
)

slow_pi_ge_ramsey = slow_pi_ge_ramsey_runner.execute(
    ramsey_freq=slow_pi_ge_ramsey_freq,
    wait_times=slow_pi_ge_ramsey_wait_times,
    hpi_length=slow_pi_ge_hpi_length,
    postprocess=True,
)



# %%

slow_pi_ge_ramsey_analysis = RamseyFitting(
    data=slow_pi_ge_ramsey.data,
    config=slow_pi_ge_ramsey.cfg,
    station=station,
)



slow_pi_ge_ramsey_analysis.display(
    f_test=slow_pi_ge_ramsey.data['drive_frequency'],
    title_str=Path(slow_pi_ge_ramsey.fname).stem,
)

# %%
station.snapshot_hardware_config(update_main = True)

# %%
station.hardware_cfg.hw.yoko_coupler.current #= 0.00025

# %% [markdown]
# # Manipulate

# %%
from experiments.MM_base import multiphoton_slot, multiphoton_table

if man_modes:  # Both ladders start from the same vacuum qubit transitions.
    mp = station.hardware_cfg.device.multiphoton
    q = station.hardware_cfg.device.qubit
    mp.pi['gn-en'].frequency[0] = q.f_ge[0]
    mp.pi['gn-en'].gain[0] = q.pulses.pi_ge.gain[0]
    mp.pi['en-fn'].frequency[0] = q.f_ef[0]
    mp.pi['en-fn'].gain[0] = q.pulses.pi_ef.gain[0]


def f0g1_values(station, mode):
    if mode == 1:
        pulse = station.hardware_cfg.device.multiphoton.pi['fn-gn+1']
        return pulse.frequency[0], pulse.gain[0], pulse.length[0]
    row = f'M{mode}'
    ds = station.ds_storage
    return ds.get_freq(row), ds.get_gain(row), ds.get_pi(row)


def man_photon_pulses(station, mode):
    base = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
    if mode == 1:
        prep = base.prep_man_photon(mode)
    else:
        prep = [['qubit', 'ge', 'pi', 0], ['qubit', 'ef', 'pi', 0], ['man', f'M{mode}', 'pi', 0]]
    return (
        base.get_prepulse_creator(prep).pulse.tolist(),
        base.get_prepulse_creator(prep[-1:-3:-1]).pulse.tolist(),
    )


def ensure_mode_slot(manipulate, parameter, index):
    values = manipulate.setdefault(parameter, [])
    if len(values) <= index:
        values.extend([None] * (index + 1 - len(values)))

# %% [markdown]
# ## Spectroscopy

# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
station.ds_storage.df

# %%
f0g1spec_defaults = AttrDict(dict(
    start=None,  # Will be computed in preprocessor from ds_storage
    step=0.2,
    expts=200,
    reps=100,
    rounds=1,
    length=1,
    gain=3000,
    pulse_type='gaussian',
    qubit_f=True,
    qubits=[0],
    prepulse=False,
    relax_delay=200,
))

def f0g1spec_preproc(station, default_expt_cfg, man_mode_no=1, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.man_mode_no = man_mode_no  # Store for postprocessor
    expt_cfg.update(kwargs)
    
    # Compute start frequency from dataset if not provided
    if expt_cfg.start is None:
        frequency, _, _ = f0g1_values(station, man_mode_no)
        expt_cfg.start = frequency - 20
    
    return expt_cfg

def f0g1spec_postproc(station, expt):
    man_mode_no = expt.cfg.expt.man_mode_no
    man_name = f'M{man_mode_no}'
    frequency = float(expt.data['fit_avgi'][2])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f'{man_name} fitted f0-g1 frequency must be positive and finite, got {frequency}')
    frequencies, slot = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', 'frequency', 0)
    station.ds_storage.update_freq(man_name, frequency)
    frequencies[slot] = frequency
    if man_mode_no == 1:
        station.multimode_cfg.pulses['pi_f0-g1']['frequency'] = frequency
    print(f"Updated {man_name} f0g1 freq to: {station.ds_storage.get_freq(man_name)}")
    station.snapshot_hardware_config(update_main=False)
    if man_mode_no == 1:
        station.snapshot_multiphoton_config(update_main=False)
    station.snapshot_man1_storage_swap(update_main=False)

# %%
# Execute
# =================================
f0g1spec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.PulseProbeF0g1SpectroscopyExperiment,
    default_expt_cfg=f0g1spec_defaults,
    preprocessor=f0g1spec_preproc,
    postprocessor=f0g1spec_postproc,
    job_client=client,
)

man_specs = {}
f0g1_start_mhz = {2: 1800}

for i in man_modes:
    if expts_to_run['pulse_probe_f0g1']:
        print(f'Running pulse probe f0g1 for mode {i}')
        man_specs[i] = f0g1spec_runner.execute(
            man_mode_no=i,
            go_kwargs=dict(progress=True),
            relax_delay=2500,
            start=f0g1_start_mhz.get(i),
            expts=400,
        )
        man_specs[i].display()


# %% [markdown]
# ## Find Frequency (Chevron)
#

# %%
chevron_defaults = AttrDict(dict(
    start=0, # time start in us
    step=0.1, # time step in us
    expts=25, # number of time points
    reps=100,
    rounds=1,
    qubits=[0],
    gain=None, # Leave as None to use current value in dataset
    ramp_sigma=0.005,
    use_arb_waveform=False,
    pi_ge_before=True,
    pi_ef_before=True,
    pi_ge_after=False,
    normalize=False,
    active_reset=False,
    check_man_reset=[False, 0],
    check_man_reset_pi=[],
    prepulse=False,
    pre_sweep_pulse=[],
    err_amp_reps=0,
    swap_lossy=False,
))


def f0g1_chevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    mode = expt_cfg.man_mode_no
    stor_name = f'M{mode}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=mother_expt.data['freq_sweep'],
        time=mother_expt.data['xpts'][0],
        response_matrix=mother_expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()
    
    best_freq = chevron_analysis.results.get('best_frequency_contrast')
        
    if best_freq is not None and np.isfinite(best_freq) and best_freq > 0:
        pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
        gain = expt_cfg.get('gain', station.ds_storage.get_gain(stor_name))
        if not all(np.isfinite(value) and value > 0 for value in (pi_len, gain)):
            raise ValueError(f'{stor_name} chevron requires a positive finite pi length and gain')
        gain = int(gain)
        print(f"Best frequency found: {best_freq:.4f} MHz")
        station.ds_storage.update_freq(stor_name, best_freq)
        frequencies, slot = multiphoton_slot(station.hardware_cfg, mode, 'pi', 'fn-gn+1', 'frequency', 0)
        frequencies[slot] = float(best_freq)
        if mode == 1:
            station.multimode_cfg.pulses['pi_f0-g1']['frequency'] = float(best_freq)
        print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz")

        station.ds_storage.update_pi(stor_name, pi_len)
        lengths, slot = multiphoton_slot(station.hardware_cfg, mode, 'pi', 'fn-gn+1', 'length', 0)
        lengths[slot] = float(pi_len)
        if mode == 1:
            station.multimode_cfg.pulses['pi_f0-g1']['length'] = float(pi_len)
        print('Updated the pi length to:', pi_len)

        station.ds_storage.update_h_pi(stor_name, pi_len / 2)
        print('Updated the h_pi length to:', pi_len / 2, '(ds_storage)')

        station.ds_storage.update_gain(stor_name, gain)
        gains, slot = multiphoton_slot(station.hardware_cfg, mode, 'pi', 'fn-gn+1', 'gain', 0)
        gains[slot] = gain
        if mode == 1:
            station.multimode_cfg.pulses['pi_f0-g1']['gain'] = gain
        else:
            for field, value in (('sigma', float(mother_expt.cfg.device.manipulate.ramp_sigma)), ('type', 'flat_top')):
                settings, slot = multiphoton_slot(station.hardware_cfg, mode, 'pi', 'fn-gn+1', field, 0)
                settings[slot] = value
        print('Updated gain to:', gain)
        station.snapshot_man1_storage_swap(update_main=False)
        station.snapshot_hardware_config(update_main=False)
        if mode == 1:
            station.snapshot_multiphoton_config(update_main=False)
    mother_expt.analysis = chevron_analysis
    chevron_analysis.display_results()


f0g1_chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,
    default_expt_cfg=chevron_defaults,
    sweep_param='freq',
    live_plot=False,
    # preprocessor=my_preproc,
    # postprocessor=f0g1_chevron_postproc,
    job_client=client,
)

# %%
# coarse sweep
f0g1_chevron = {}
for mode in man_modes:
    frequency, gain, _ = f0g1_values(station, mode)
    f0g1_chevron[mode] = f0g1_chevron_runner.execute(
        man_mode_no=mode,
        sweep_start=frequency - 3,
        sweep_stop=frequency + 3,
        sweep_npts=11,
        gain=gain,
        start = 1, # time start in us
        batch=True,
    )


# %%
for expt in f0g1_chevron.values():
    f0g1_chevron_postproc(station, expt)

# %%
# fine sweep
f0g1_chevron = {}
for mode in man_modes:
    frequency, gain, _ = f0g1_values(station, mode)
    f0g1_chevron[mode] = f0g1_chevron_runner.execute(
        man_mode_no=mode,
        sweep_start=frequency - 0.5,
        sweep_stop=frequency + 0.5,
        sweep_npts=11,
        gain=gain,
        # start = 1, # time start in us
        # coupler_current=coupler_current,
        reps=75,
        batch=True,
    )


# %%
for expt in f0g1_chevron.values():
    f0g1_chevron_postproc(station, expt)

# %%
# station.hard

# %% [markdown]
# ## Error amplification

# %%
# Manipulate/Error Amplification - New Pattern with CharacterizationRunner
span = 0.5
expts = 40


steps = span/expts


# Configuration defaults
error_amp_defaults = AttrDict(dict(
    start=None,
    expts=expts,
    step=steps, # 0.025
    # start=1984,
    # expts=40,
    # step=0.025, # 0.025
    reps=100,
    rounds=1,
    qubit=0,
    qubits=[0],
    n_pulses=7,
    sideband = 'f0-g1', #should be in the format of 'fn-gn+1' 
    parameter_to_test='frequency',
    pulse_type= None, # if this is None, will be set to ['multiphoton', sideband, 'pi', 0] in preproc
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500,
))

def error_amp_preproc(station, 
                      default_expt_cfg, 
                      **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    photon_number = int(expt_cfg.sideband[1])
    expt_cfg.multiphoton_mode = expt_cfg.man_mode_no
    if expt_cfg.start is None:
        if photon_number == 0:
            center, _, _ = f0g1_values(station, expt_cfg.man_mode_no)
        else:
            pulse = multiphoton_table(station.hardware_cfg, expt_cfg.man_mode_no, num_photon_levels=photon_number + 1)
            center = pulse.pi['fn-gn+1'].frequency[photon_number]
        expt_cfg.start = center - span / 2
    if expt_cfg.pulse_type is None:
        if expt_cfg.man_mode_no == 2 and photon_number == 0:
            expt_cfg.pulse_type = ['man', 'M2', 'pi', 0]
        else:
            expt_cfg.pulse_type = ['multiphoton', expt_cfg.sideband, 'pi', 0]
    return expt_cfg

def error_amp_postproc(station, expt):
    """
    Postprocessor for error amplification.
    Analyze with custom parameters.
    """
    sideband = expt.cfg.expt.sideband
    _sideband = sideband[0] + 'n' + '-' + sideband[3] + 'n+1'
    photon_number = int(sideband[1])
    # Perform analysis with state_fin='e' as in original code
    expt.analyze(data=expt.data, state_fin='e')
    # expt.display(data=expt.data, state_fin='e')
    print('Error amplification analysis complete')

    man_mode_no = expt.cfg.expt.man_mode_no
    man_name = f'M{man_mode_no}'
    frequency = float(expt.data['fit_avgi'][2])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f'{man_name} fitted {sideband} frequency must be positive and finite, got {frequency}')
    frequencies, slot = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', _sideband, 'frequency',
                                       m1_photons=photon_number if man_mode_no == 1 else 0,
                                       m2_photons=photon_number if man_mode_no == 2 else 0)
    print(f"{man_name} {sideband} pi frequency before update:", frequencies[slot])
    frequencies[slot] = frequency
    print(f"{man_name} {sideband} pi frequency after update:", frequency)
    if photon_number == 0:
        if man_mode_no == 1:
            station.multimode_cfg.pulses['pi_f0-g1']['frequency'] = frequency
        station.ds_storage.update_freq(man_name, frequency)
        print(f"Updated the {man_name} ds_storage frequency to:", station.ds_storage.get_freq(man_name))
    station.snapshot_hardware_config(update_main=False)
    if man_mode_no == 1:
        station.snapshot_multiphoton_config(update_main=False)
    if photon_number == 0:
        station.snapshot_man1_storage_swap(update_main=False)

# Create runner
error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_defaults,
    preprocessor=error_amp_preproc,
    postprocessor=error_amp_postproc,
    job_client=client,
)

# Example execution

# Run with analyze=False, display=False initially
# Postprocessor will handle custom analysis
error_amp_exp = {}
for mode in man_modes:
    error_amp_exp[mode] = error_amp_runner.execute(
        man_mode_no=mode,
        reps = 100, #100
        go_kwargs = dict(analyze=False, display=False),
        postprocess=True  # This will call postprocessor which does the custom analysis
    )

    error_amp_exp[mode].display()

# %% [markdown]
# ## Length Rabi f0g1 (Update time)

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
lenrabi_f0g1_defaults = AttrDict(dict(
    start=None,  # Will be computed in preprocessor (soc.cycles2us(3))
    step=0.01,
    qubits=[0],
    expts=150,
    reps=100,
    rounds=1,
    gain=8000,
    freq=None,  # Will be set from ds_storage in preprocessor
    use_arb_waveform=False,
    pi_ge_before=True,
    pi_ef_before=True,
    pi_ge_after=True,
    normalize=False,
    active_reset=False,
    man_reset=True,
    stor_reset=True,
    check_man_reset=[False, 0],
    swap_lossy=False,
    check_man_reset_pi=[],
    prepulse=False,
    pre_sweep_pulse=[],
    err_amp_reps=0,
    relax_delay=5000,
))

def lenrabi_f0g1_preproc(station, default_expt_cfg, man_mode_no=1, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.man_mode_no = man_mode_no  # Store for postprocessor
    expt_cfg.update(kwargs)
    
    # Compute start and freq from soc and ds_storage if not provided
    if expt_cfg.start is None:
        expt_cfg.start = station.soccfg.cycles2us(3)
    if expt_cfg.freq is None:
        expt_cfg.freq, _, _ = f0g1_values(station, man_mode_no)
    
    return expt_cfg

def lenrabi_f0g1_postproc(station, expt):
    man_mode_no = expt.cfg.expt.man_mode_no
    
    # Get analysis results from the LengthRabiFitting object stored in expt
    if hasattr(expt, '_length_rabi_analysis'):
        analysis = expt._length_rabi_analysis
        pi_length = analysis.results['pi_length']
        pi2_length = analysis.results['pi2_length']
        gain = expt.cfg.expt.gain
        freq = expt.cfg.expt.freq

        man_name = f'M{man_mode_no}'
        if not all(np.isfinite(value) and value > 0 for value in (freq, pi_length, pi2_length, gain)):
            raise ValueError(f'{man_name} length Rabi requires positive finite frequency, lengths, and gain')
        gain = int(gain)
        pulse_fields = dict(frequency=float(freq), length=float(pi_length), gain=gain)
        if man_mode_no == 2:
            pulse_fields.update(sigma=float(expt.cfg.device.manipulate.ramp_sigma), type='flat_top')
        station.ds_storage.update_freq(man_name, freq)
        station.ds_storage.update_pi(man_name, pi_length)
        station.ds_storage.update_h_pi(man_name, pi2_length)
        station.ds_storage.update_gain(man_name, gain)
        for field, value in pulse_fields.items():
            settings, slot = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', field, 0)
            settings[slot] = value
        if man_mode_no == 1:
            station.multimode_cfg.pulses['pi_f0-g1']['frequency'] = float(freq)
            station.multimode_cfg.pulses['pi_f0-g1']['length'] = float(pi_length)
            station.multimode_cfg.pulses['pi_f0-g1']['gain'] = gain
        print(f'Updated dataset {man_name}: pi_length={pi_length:.4f}, pi2_length={pi2_length:.4f}, gain={gain}')
        station.snapshot_hardware_config(update_main=False)
        if man_mode_no == 1:
            station.snapshot_multiphoton_config(update_main=False)
        station.snapshot_man1_storage_swap(update_main=False)

# %%
# Execute
# =================================
lenrabi_f0g1_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,
    default_expt_cfg=lenrabi_f0g1_defaults,
    preprocessor=lenrabi_f0g1_preproc,
    postprocessor=lenrabi_f0g1_postproc,
    job_client=client,
)

len_rabis_mans = {}

for mode in man_modes:
    if expts_to_run['length_rabi'] or expts_to_run['length_rabi_sweep']:
        print(f'Running length rabi for mode {mode}')
        len_rabis_mans[mode] = lenrabi_f0g1_runner.execute(
            man_mode_no=mode,
            go_kwargs=dict(progress=True),
            step=0.01,
        )
        len_rabis_mans[mode].display()


# %% [markdown]
# ## Chi between qubit and manipulate

# %%
def ramsey_adjustment(expt):
    return min(expt.data['f_adjust_ramsey_avgi'], key=abs)


def run_chi_ramsey(runner, mode, **kwargs):
    expt = runner.execute(
        man_mode_no=mode, step=0.1, expts=100, relax_delay=2500,
        active_reset=False, postprocess=False, **kwargs,
    )
    expt.display()
    return expt

# %% [markdown]
# ### ge

# %%
chi_ge_ramsey = {}
if expts_to_run['chi_ge']:
    reference = run_chi_ramsey(geramsey_runner, man_modes[0])
    for mode in man_modes:
        prep, _ = man_photon_pulses(station, mode)
        chi_ge_ramsey[mode] = run_chi_ramsey(
            geramsey_runner, mode, pre_sweep_pulse=prep,
        )
        chi_ge = ramsey_adjustment(chi_ge_ramsey[mode]) - ramsey_adjustment(reference)
        if not np.isfinite(chi_ge) or chi_ge == 0:
            raise ValueError(f'M{mode} chi_ge fit cannot set a parity delay: {chi_ge}')
        manipulate = station.hardware_cfg.device.manipulate
        index = mode - 1
        ensure_mode_slot(manipulate, 'chi_ge', index)
        manipulate.chi_ge[index] = float(chi_ge)
        manipulate.revival_time[index] = float(abs(1 / (2 * chi_ge)))
        station.snapshot_hardware_config(update_main=False)

# %% [markdown]
# ### ef

# %%
chi_ef_ramsey = {}
if expts_to_run['chi_ef']:
    reference = run_chi_ramsey(eframsey_runner, man_modes[0], ramsey_freq=0.2)
    for mode in man_modes:
        prep, _ = man_photon_pulses(station, mode)
        chi_ge = station.hardware_cfg.device.manipulate.chi_ge[mode - 1]
        base = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
        ge = base.get_prepulse_creator([['qubit', 'ge', 'pi', 0]]).pulse.tolist()
        ge[0][0] += chi_ge
        chi_ef_ramsey[mode] = run_chi_ramsey(
            eframsey_runner, mode, ramsey_freq=0.2, ef_init=False,
            pre_sweep_pulse=[left + right for left, right in zip(prep, ge)],
            post_sweep_pulse=ge,
        )
        chi_ef = ramsey_adjustment(chi_ef_ramsey[mode]) - ramsey_adjustment(reference)
        if not np.isfinite(chi_ef):
            raise ValueError(f'M{mode} chi_ef fit is invalid: {chi_ef}')
        manipulate = station.hardware_cfg.device.manipulate
        ensure_mode_slot(manipulate, 'chi_ef', mode - 1)
        manipulate.chi_ef[mode - 1] = float(chi_ef)
        station.snapshot_hardware_config(update_main=False)

# %% [markdown]
# ## Parity Delay
# NOT Implemented yet: We can use chi to estimate parity waiting time pi/chi but can also fine tune it using this experiment
#
# Basically Length rabi analysis 

# %%
parity_delay_defaults = AttrDict(dict(
    start=0.01,
    step=0.05,
    expts=50,
    reps=250,
    rounds=1,
    qubits=[0],
    parity_fast=False,
    normalize=False,
    active_reset=False,
    man_reset=True,
    gate_based=False,
    storage_reset=True,
    man_mode_no=1,
    relax_delay=2500, 
    param_to_calib='length', # length or ac_stark_shift 
    shift_sign=1, # 1 or -1 for ac stark shift direction
    ))

mm_base_calib = MM_dual_rail_base(station.hardware_cfg, station.soccfg)


def parity_delay_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    if 'param_to_calib' in expt_cfg and expt_cfg.param_to_calib =='ac_stark_shift':
        expt_cfg.prepulse = False
        expt_cfg.pre_sweep_pulse = []
    elif 'param_to_calib' in expt_cfg and expt_cfg.param_to_calib =='length':
        expt_cfg.prepulse = True
        expt_cfg.pre_sweep_pulse, _ = man_photon_pulses(station, expt_cfg.man_mode_no)
    return expt_cfg

def parity_delay_postproc(station, expt):
    from fitting.fit_display_classes import LengthRabiFitting
    # Analyze the data
    length_rabi_analysis = LengthRabiFitting(expt.data, config = expt.cfg)
    length_rabi_analysis.analyze(fitparams = [100, 1, 0, None, None, None])
    length_rabi_analysis.display(title_str='Length Rabi General F0g1')
    expt_cfg = expt.cfg.expt
    man_mode_no = expt_cfg.man_mode_no
    shift_sign = expt_cfg.shift_sign


    if expt_cfg.param_to_calib == 'length':
        delay = length_rabi_analysis.results['pi_length']
        print(f'Fitted revival time: {delay} us')
        old_delay = station.hardware_cfg.device.manipulate.revival_time[man_mode_no - 1]
        station.hardware_cfg.device.manipulate.revival_time[man_mode_no - 1] = delay
        print(f'Updated revival time for mode {man_mode_no}: {old_delay} -> {station.hardware_cfg.device.manipulate.revival_time[man_mode_no - 1]}')

    elif expt_cfg.param_to_calib == 'ac_stark_shift': 
        freq = length_rabi_analysis.data['fit_avgi'][1]
        print(f'Fitted ac Stark shift: {freq} MHz')
        old_freq = station.hardware_cfg.device.manipulate.revival_stark_shift[man_mode_no - 1]
        station.hardware_cfg.device.manipulate.revival_stark_shift[man_mode_no - 1] += shift_sign * freq
        print(f'Updated ac Stark shift for mode {man_mode_no}: {old_freq} -> {station.hardware_cfg.device.manipulate.revival_stark_shift[man_mode_no - 1]}')

    


# %% [markdown]
# ### Calibrate first the AC stark shift
#
# Manipulate in 0, qubit should not rotate
#
# this is calibrating the virtual Z to correct from rotating frame to bare frame

# %%
parity_delay_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityDelayExperiment,
    default_expt_cfg=parity_delay_defaults,
    postprocessor=parity_delay_postproc,
    preprocessor=parity_delay_preproc,
    job_client=client,
    use_queue=True,
)

parity_delay_ac_stark = {}
for mode in man_modes:
    parity_delay_ac_stark[mode] = parity_delay_runner.execute(
        man_mode_no=mode,
        param_to_calib='ac_stark_shift',
        go_kwargs=dict(analyze=False, display=False),
        step=0.2,
        expts=100,
        reps=100,
        shift_sign=1,
        # coupler_current=0.0,
    )


# %% [markdown]
# Confirming the correction is in the right direction

# %%
parity_delay_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityDelayExperiment,
    default_expt_cfg=parity_delay_defaults,
    preprocessor=parity_delay_preproc,
    job_client=client,
    use_queue=True,
)

parity_stark_check = {}
for mode in man_modes:
    parity_stark_check[mode] = parity_delay_runner.execute(
        man_mode_no=mode,
        param_to_calib='ac_stark_shift',
        go_kwargs=dict(analyze=False, display=False),
        step=0.2,
        expts=100,
        reps=100,
        shift_sign=1,
        # coupler_current=0.0,
    )


# %%
for expt in parity_stark_check.values():
    expt.display(fit=False)

# %% [markdown]
# ### Now parity length after the AC stark shift is calibrated

# %%
parity_delay_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityDelayExperiment,
    default_expt_cfg=parity_delay_defaults,
    postprocessor=parity_delay_postproc,
    preprocessor=parity_delay_preproc,
    job_client=client,
    use_queue=True,
)

parity_delay_length = {}
for mode in man_modes:
    parity_delay_length[mode] = parity_delay_runner.execute(
        man_mode_no=mode,
        param_to_calib='length',
        go_kwargs=dict(analyze=False, display=False),
        step=0.04,
        expts=100,
        reps=200,
        parity_fast = False
    )


# %% [markdown]
# ## T1

# %%
t1_manipulate_defaults = AttrDict(dict(
    start=0,
    step=8,
    expts=120,
    reps=500,
    rounds=1,
    man_mode_no=1,
    f0g1_param=None,
    cavity_prepulse=[False, 300, 1.5],
    f0g1_prep=True,
    resolved_pi=False,
    qubit=0,
    normalize=False,
))


def t1_manipulate_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    if expt_cfg.f0g1_param is None:
        mode = expt_cfg.man_mode_no
        frequency, gain, length = f0g1_values(station, mode)
        if mode == 1:
            sigma = station.hardware_cfg.device.multiphoton.pi['fn-gn+1'].sigma[0]
        else:
            sigma = station.hardware_cfg.device.manipulate.ramp_sigma
        expt_cfg.f0g1_param = [frequency, gain, length, sigma]

    expt_cfg.cavity = expt_cfg.man_mode_no

    return expt_cfg


t1_manipulate_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.t1_cavity.T1CavityExperiment,
    default_expt_cfg=t1_manipulate_defaults,
    preprocessor=t1_manipulate_preproc,
    job_client=client,
)

# %%
t1_manipulate = {
    mode: t1_manipulate_runner.execute(man_mode_no=mode)
    for mode in man_modes
}

# %%
for expt in t1_manipulate.values():
    expt.analyze()
    expt.display()

# %% [markdown]
# ## T2
#
# prepares man in + and does ramsey with this

# %%
cross_kerr_stor = 4
cross_kerr_init_fock = 0

def prep_cross_kerr(mode):
    cfg = dict(parity_meas=False, use_auto_prep=True, prepulse=False)
    if cross_kerr_init_fock == 1:
        base = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
        prep, _ = man_photon_pulses(station, mode)
        swap = base.get_prepulse_creator(
            [['storage', f'M{mode}-S{cross_kerr_stor}', 'pi', 0]],
        ).pulse.tolist()
        cfg.update(
            prepulse=True, gate_based=False,
            pre_sweep_pulse=[left + right for left, right in zip(prep, swap)],
        )
    return cfg


# %% [markdown]
# ### First calibrate the stark shift and residual detuning on the + state during the f0-g1

# %%
t2_manipulate_defaults = AttrDict(dict(
    start=0,
    step=0.005,
    expts=300,
    reps=200,
    rounds=1,
    mode='manipulate',
    man_mode_no=1,
    qubits=[0],
    ramsey_freq=20,
))


t2_manipulate_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.t2_cavity_mode.CavityModeRamseyExperiment,
    default_expt_cfg=t2_manipulate_defaults,
    # preprocessor=t2_manipulate_preproc,
    job_client=client,
)

# %%
t2_manipulate = {
    mode: t2_manipulate_runner.execute(
        man_mode_no=mode, man_mode_idx=mode, **prep_cross_kerr(mode),
    ) for mode in man_modes
}

# %%
fit_ac_freq = {}
for mode, expt in t2_manipulate.items():
    expt.analyze()
    expt.display()
    fit_ac_freq[mode] = expt.data['f_adjust_ramsey_avgi'][0]
    print(f'M{mode} fit freq MHz', fit_ac_freq[mode])


# %% [markdown]
# ### Ac stark shift adjustment
#
# Slight correction of the ac stark shift correction at long times

# %%
t2_manipulate_2_defaults = AttrDict(dict(
    start=0,
    step=0.1,
    expts=100,
    reps=200,
    rounds=1,
    mode='manipulate',
    man_mode_no=1,
    qubits=[0],
    ramsey_freq=2,  # Set per mode below using its previous fit.
))


t2_manipulate_2_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.t2_cavity_mode.CavityModeRamseyExperiment,
    default_expt_cfg=t2_manipulate_2_defaults,
    # preprocessor=t2_manipulate_preproc,
    job_client=client,
)

# %%
t2_manipulate_2 = {}
for mode in man_modes:
    expt = t2_manipulate_2_runner.execute(
        man_mode_no=mode, man_mode_idx=mode, ramsey_freq=fit_ac_freq[mode] + 2,
        **prep_cross_kerr(mode),
    )
    t2_manipulate_2[mode] = expt
    expt.analyze()
    expt.display()


# %%
fit_ac_freq_fine = {}
for mode in man_modes:
    effective_ramsey_freq = t2_manipulate_2[mode].cfg.expt.ramsey_freq - fit_ac_freq[mode]
    fitted_freq = t2_manipulate_2[mode].data['fit_avgi'][1]
    freq_update = sorted(
        (effective_ramsey_freq - fitted_freq, effective_ramsey_freq + fitted_freq), key=abs)
    fit_ac_freq_fine[mode] = fit_ac_freq[mode] + freq_update[0] # sign could be + or -...
    # fit_ac_freq_fine = t2_manipulate_2[mode].data['f_adjust_ramsey_avgi'][0]
    print("Make sure to double check the adjustment sign above is correct by running the next cell with 0 additional virtual ramsey!")
    print(f"M{mode} fit freq MHz updated", fit_ac_freq_fine[mode])


# %% [markdown]
# ### Proper t2

# %%
t2_manipulate_3_defaults = AttrDict(dict(
    start=0,
    step=5,
    expts=100,
    reps=200,
    rounds=1,
    mode='manipulate',
    man_mode_no=1,
    qubits=[0],
    ramsey_freq=0.01,
))


t2_manipulate_3_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.t2_cavity_mode.CavityModeRamseyExperiment,
    default_expt_cfg=t2_manipulate_3_defaults,
    # preprocessor=t2_manipulate_3_preproc,
    job_client=client,
)

# %%
print({mode: prep_cross_kerr(mode) for mode in man_modes})

# %%
t2_manipulate_3 = {}
for mode in man_modes:
    t2_manipulate_3[mode] = t2_manipulate_3_runner.execute(
        man_mode_no=mode,
        man_mode_idx=mode,

        # high kerr point
        start=0,
        step=5,
        expts=100,
        ramsey_freq=fit_ac_freq_fine[mode] + 0.010,

        **prep_cross_kerr(mode)
        )
    t2_manipulate_3[mode].analyze()
    t2_manipulate_3[mode].display()


# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
station.update_all_station_snapshots()

# %% [markdown]
# ## Manipulate parity spectroscopy
#
# The self-Kerr sequence uses the selected mode's cavity drive frequency and
# displacement calibration. Enable the three stages above to refresh all of them.

# %%
from fitting.fit_display_classes import Spectroscopy


def man_parity_spectroscopy_postproc(station, expt):
    mode = expt.cfg.expt.man_mode_no
    spec = Spectroscopy(expt.data, config=expt.cfg)
    spec.analyze()
    spec.display(title=f'M{mode} parity spectroscopy')
    frequency = spec.data['fit_avgi'][2]
    if not min(expt.data['xpts']) <= frequency <= max(expt.data['xpts']):
        raise ValueError(f'M{mode} cavity frequency fit {frequency} is outside the sweep')
    station.hardware_cfg.device.manipulate.f_ge[mode - 1] = float(frequency)
    station.snapshot_hardware_config(update_main=False)
    print(f'M{mode} cavity drive frequency: {frequency:.6f} MHz')


man_parity_spectroscopy_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityFreqExperiment,
    default_expt_cfg=AttrDict(dict(
        step=0.6, reps=500, rounds=1, qubits=[0], normalize=False,
        single_shot=False, displace=(True, 0.1, 1000), const_pulse=(False, 1),
        f0g1_cavity=0, prepulse=False, pre_sweep_pulse=None, parity_fast=False,
        relax_delay=2500,
    )),
    postprocessor=man_parity_spectroscopy_postproc,
    job_client=client,
)

man_parity_spectra = {}
if expts_to_run['man_parity_spectroscopy']:
    for mode in man_modes:
        center = station.hardware_cfg.device.manipulate.f_ge[mode - 1]
        man_parity_spectra[mode] = man_parity_spectroscopy_runner.execute(
            man_mode_no=mode, start=center - 30, stop=center + 30,
            go_kwargs=dict(analyze=False, display=False),
        )

# %% [markdown]
# ## Gain to alpha calibration
#
# The vacuum parity follows $e^{-2|\alpha|^2}$.
# Use `gain = abs(alpha) / gain_to_alpha` for the calibrated Gaussian displacement.

# %%
from experiments.single_qubit.parity_gain import ParityGainExperiment


def man_gain_to_alpha_postproc(station, expt):
    mode = expt.cfg.expt.man_mode_no
    expt.analyze(scale=True, plot=False)
    gain_to_alpha = expt.data['gain_to_alpha']
    scale = expt.data['scale']
    if not np.isfinite(gain_to_alpha) or gain_to_alpha <= 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'M{mode} gain-to-alpha fit is invalid: {gain_to_alpha}, scale {scale}')
    expt.display(fit=False)
    manipulate = station.hardware_cfg.device.manipulate
    manipulate.gain_to_alpha[mode - 1] = float(gain_to_alpha)
    manipulate.alpha_scale[mode - 1] = float(scale)
    station.snapshot_hardware_config(update_main=False)
    print(f'M{mode} gain to alpha: {gain_to_alpha:.6g}')


man_gain_to_alpha_runner = CharacterizationRunner(
    station=station,
    ExptClass=ParityGainExperiment,
    default_expt_cfg=AttrDict(dict(
        start=0, step=100, expts=40, reps=250, rounds=1, qubit=0, qubits=[0],
        normalize=False, single_shot=False, prep_e=False, active_reset=False,
        man_reset=True, storage_reset=True, const_pulse=(False, 1),
        f0g1_cavity=0, prepulse=False, pre_sweep_pulse=None, relax_delay=2500,
        pulse_correction=True, parity_fast=False, scaling_amplitude=True,
    )),
    postprocessor=man_gain_to_alpha_postproc,
    job_client=client,
)

man_gain_to_alpha = {}
if expts_to_run['man_gain_to_alpha']:
    for mode in man_modes:
        sigma = station.hardware_cfg.device.manipulate.displace_sigma[mode - 1]
        man_gain_to_alpha[mode] = man_gain_to_alpha_runner.execute(
            man_mode_no=mode, manipulate=mode, displace=(True, sigma),
            go_kwargs=dict(analyze=False, display=False),
            display_kwargs=dict(fit=False),
        )

# %% [markdown]
# ## Self-Kerr from cavity Ramsey versus gain
#
# Use the frequency and gain-to-alpha calibration above to fit Kerr from the
# frequency shift versus displacement. The result is stored in MHz.

# %%
from experiments.single_qubit.t2_cavity import CavityRamseyGainSweepExperiment


man_self_kerr_runner = CharacterizationRunner(
    station=station,
    ExptClass=CavityRamseyGainSweepExperiment,
    default_expt_cfg=AttrDict(dict(
        start=0.01, step=0.02, expts=150, ramsey_freq=4, reps=100, rounds=1,
        qubits=[0], checkEF=False, f0g1_cavity=0, init_gf=False, active_reset=True,
        man_reset=True, storage_reset=False, parity_meas=True, parity_fast=False,
        storage_ramsey=[False, 1, True], coupler_ramsey=False, echoes=[False, 0],
        prepulse=False, postpulse=False, gate_based=False, pre_sweep_pulse=[],
        post_sweep_pulse=[], relax_delay=200, do_g_and_e=False,
    )),
    job_client=client,
)

self_kerr_gains = {1: (1000, 4000, 500), 2: (1000, 4000, 500)}  # start, stop, step
self_kerr_fit_indices = {}  # Optional mode-to-index-list overrides for the g-state traces.
# The legacy dump reset targets M1. For M2 use qubit reset and a longer wait.
man_self_kerr = {}
if expts_to_run['man_self_kerr']:
    manipulate = station.hardware_cfg.device.manipulate
    for mode in man_modes:
        dump = reset_dump_modes[mode]
        if mode == 1:
            if not station.ds_storage.has_row(f'M{mode}-D{dump}'):
                raise ValueError(f'Cannot run M{mode} self-Kerr: missing M{mode}-D{dump} reset row')
        gain_start, gain_stop, gain_step = self_kerr_gains[mode]
        index = mode - 1
        drive = [True, manipulate.f_ge[index], gain_start, manipulate.displace_sigma[index], 0, 4]
        result = man_self_kerr_runner.execute(
            man_mode_no=mode, man_ramsey=[False, index], user_defined_pulse=drive,
            gain_to_alpha=manipulate.gain_to_alpha[index],
            gain_start=gain_start, gain_step=gain_step,
            gain_expts=(gain_stop - gain_start) // gain_step + 1,
            reset_dump_mode=dump, man_reset=mode == 1,
            relax_delay=200 if mode == 1 else 2500,
            go_kwargs=dict(analyze=False, display=False),
            show=False, log=False,
        )
        man_self_kerr[mode] = result
        result.data = result.analyze(track_peaks=True, fit_indices_g=self_kerr_fit_indices.get(mode))
        result.display(save_fig=False)
        kerr = result.data['Kerr']
        if not np.isfinite(kerr):
            raise ValueError(f'M{mode} self-Kerr fit is invalid: {kerr}')
        ensure_mode_slot(manipulate, 'kerr', index)
        manipulate.kerr[index] = float(kerr)
        station.snapshot_hardware_config(update_main=False)
        print(f'M{mode} self-Kerr: {kerr * 1e3:.3f} kHz')

# %% [markdown]
# ## Multiphoton calibration
#
# Enter the starting photon count for each mode. Use 0 for vacuum or leave both as None to skip these cells.
# Calibrate M1 with M2 at 0. Then select M2, hold M1 at the desired photon count, and step M2 up from 0.
# At (n, 0), skip the ge/ef cells if those qubit pulses were already calibrated in the M1 pass.
# For example, at (2, 1) the M2 swap adds a second M2 photon. Preparing this point uses the M2 swap at (2, 0).
# Calibrate ge frequency and gain, then ef frequency and gain, then the swap.
# Run swap spectroscopy or a chevron to locate the line, fine frequency to refine
# it, and length Rabi to set the pi length. Each cell reads the current settings.
# Set `save_multiphoton_fits = False` to measure and inspect without saving a fit.

# %%
from experiments.MM_base import multiphoton_prep
from experiments.autocal import _checks as fit_checks
from experiments.autocal._util import fit_line, plot_fit
from fitting.fitting import lorfunc

initial_m1_photon_number = None  # 0–9 photons in M1
initial_m2_photon_number = None  # 0–9 photons in M2
save_multiphoton_fits = True
multiphoton_swap_modes = [mode for mode in man_modes if mode == 2 or initial_m2_photon_number in (None, 0)]
multiphoton_qubit_modes = multiphoton_swap_modes[:1]  # The qubit transitions are shared.

# %%
def run_multiphoton_stage(mode, transition_kind, stage, field, *, center=None, half_span=None, pi_span=None, **overrides):
    m1_photons = initial_m1_photon_number
    m2_photons = initial_m2_photon_number
    if m1_photons is None and m2_photons is None:
        return None
    if any(isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 0 <= count <= 9
           for count in (m1_photons, m2_photons)):
        raise ValueError('Initial photon counts must be integers from 0 to 9')
    photon_number = m1_photons if mode == 1 else m2_photons
    other_mode_photon_number = m2_photons if mode == 1 else m1_photons
    transition = {'ge': f'g{photon_number}-e{photon_number}', 'ef': f'e{photon_number}-f{photon_number}',
                  'swap': f'f{photon_number}-g{photon_number + 1}'}[transition_kind]
    transition_key = {'ge': 'gn-en', 'ef': 'en-fn', 'swap': 'fn-gn+1'}[transition_kind]
    pulse = multiphoton_table(station.hardware_cfg, mode, other_mode_photon_number=other_mode_photon_number,
                              num_photon_levels=photon_number + 1).pi[transition_key]
    kwargs = dict(avoid_yoko=False) | overrides
    # The f0-g1 postprocessors save at zero photons. Save the selected transition below.
    kwargs.update(transition=transition, man_mode_no=mode, man_no=mode, multiphoton_mode=mode,
                  multiphoton_other_mode_photon_number=other_mode_photon_number, postprocess=False)
    if stage != 'chevron':
        kwargs.setdefault('go_kwargs', dict(analyze=False, display=False))
        kwargs.setdefault('show', False)
    if half_span is not None:
        if center is None:
            center = pulse[field][photon_number]
        if stage == 'chevron':
            kwargs.update(sweep_start=center - half_span, sweep_stop=center + half_span)
        else:
            kwargs.update(start=center - half_span, step=2 * half_span / kwargs['expts'])
    if stage == 'error_amplification':
        state_fin = 'e' if transition_kind == 'swap' else 'g'
        kwargs.setdefault('reps', 50 if field == 'frequency' and transition_kind != 'swap' else 100)
        kwargs.setdefault('n_pulses', 10)
        kwargs.update(sideband=transition, pulse_type=['multiphoton', transition, 'pi', 0],
                      pulse_name='pi', parameter_to_test=field, state_fin=state_fin)
        expt = error_amp_runner.execute(**kwargs)
        expt.analyze(state_fin=state_fin, periodic=False)
        data = expt.data
        fit = data.get('fit_avgi')
        ok, info = fit_checks.gaussian(data['x_pts'], data['prod_avgi'], fit)
        value = fit[2] if fit is not None else None
        expt.display(fit=fit is not None)
    else:
        # These sideband scans need |f, M1 photons, M2 photons> before the swept pulse.
        prep = multiphoton_prep(station.hardware_cfg, m1_photons, m2_photons)
        for qubit_transition in (f'g{photon_number}-e{photon_number}', f'e{photon_number}-f{photon_number}'):
            prep.append(['multiphoton', qubit_transition, 'pi', 0, mode, other_mode_photon_number])
        base = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
        kwargs.update(prepulse=True, pre_sweep_pulse=base.get_prepulse_creator(prep).pulse.tolist())
        if stage == 'spectroscopy':
            kwargs.setdefault('reps', 500)
            kwargs.setdefault('gain', 1000)
            kwargs['qubit_f'] = False
            expt = f0g1spec_runner.execute(**kwargs)
            x, y = expt.data['xpts'], expt.data['avgi']
            fit = fit_line(x, y)
            ok, info = fit_checks.lorentzian(x, y, fit, max_width_frac=0.05, min_snr=5 if photon_number == 0 else 3.5)
            value = fit[2]
            _, ax = plt.subplots()
            plot_fit(x, y, lorfunc, fit, vline=value, xlabel='sideband frequency (MHz)')(ax)
            plt.show()
        else:
            kwargs.update(pi_ge_before=False, pi_ef_before=False, pi_ge_after=False, postpulse=False)
            kwargs.setdefault('gain', pulse.gain[photon_number])
            if stage == 'chevron':
                kwargs.setdefault('batch', True)
                kwargs['analyze'] = False
                expt = f0g1_chevron_runner.execute(**kwargs)
                expt.analyze(station=station)
                value = expt._chevron_analysis.results.get('best_frequency_contrast')
                frequencies = expt.data['freq_sweep']
                ok = value is not None and min(frequencies) < value < max(frequencies)
                info = {'reason': 'no chevron center inside the sweep'}
            else:  # length_rabi
                kwargs['step'] = pi_span * pulse.length[photon_number] / kwargs['expts']
                kwargs.setdefault('freq', pulse.frequency[photon_number])
                kwargs.setdefault('relax_delay', 2500)
                expt = lenrabi_f0g1_runner.execute(**kwargs)
                expt.analyze()
                results = expt._length_rabi_analysis.results
                value = results.get('pi_length')
                ok, info = fit_checks.rabi(expt.data['xpts'], expt.data['avgi'], results.get('fit_avgi'), value)
            if value is not None:
                expt.display()
    print(f'M{mode} {transition} at M1={m1_photons}, M2={m2_photons}: '
          f'{field} fit = {value}')
    if not ok:
        print(f"Fit not saved: {info.get('reason', 'fit check failed')}")
    elif save_multiphoton_fits:
        value = int(round(value)) if field == 'gain' else float(value)
        pulse_fields = {field: value}
        if transition_kind == 'swap' and field == 'length':
            pulse_fields['gain'] = int(expt.cfg.expt.gain)
            if mode == 2 or m2_photons:
                pulse_fields.update(sigma=float(expt.cfg.device.manipulate.ramp_sigma), type='flat_top')
        for parameter, setting in pulse_fields.items():
            settings, slot = multiphoton_slot(station.hardware_cfg, mode, 'pi', transition_key, parameter,
                                               m1_photons, m2_photons)
            settings[slot] = setting
        if transition_kind == 'swap' and m1_photons == m2_photons == 0:
            if field == 'frequency':
                station.ds_storage.update_freq(f'M{mode}', value)
            elif field == 'length':
                station.ds_storage.update_pi(f'M{mode}', value)
                station.ds_storage.update_h_pi(f'M{mode}', value / 2)
                station.ds_storage.update_gain(f'M{mode}', pulse_fields['gain'])
            station.snapshot_man1_storage_swap(update_main=False)
            if mode == 1:
                station.multimode_cfg.pulses['pi_f0-g1'].update(pulse_fields)
                station.snapshot_multiphoton_config(update_main=False)
        station.snapshot_hardware_config(update_main=False)
    return expt

# %% [markdown]
# ### Qubit ge frequency

# %%
multiphoton_ge_freq = {mode: run_multiphoton_stage(mode, 'ge', 'error_amplification', 'frequency', half_span=1, expts=50)
                       for mode in multiphoton_qubit_modes}

# %% [markdown]
# ### Qubit ge gain

# %%
multiphoton_ge_gain = {mode: run_multiphoton_stage(mode, 'ge', 'error_amplification', 'gain', half_span=500, expts=25)
                       for mode in multiphoton_qubit_modes}

# %% [markdown]
# ### Qubit ef frequency

# %%
multiphoton_ef_freq = {mode: run_multiphoton_stage(mode, 'ef', 'error_amplification', 'frequency', half_span=1, expts=50)
                       for mode in multiphoton_qubit_modes}

# %% [markdown]
# ### Qubit ef gain

# %%
multiphoton_ef_gain = {mode: run_multiphoton_stage(mode, 'ef', 'error_amplification', 'gain', half_span=500, expts=25)
                       for mode in multiphoton_qubit_modes}

# %% [markdown]
# ### Sideband spectroscopy

# %%
sideband_spec_center_mhz = {}  # Optional sweep center for each selected mode.
multiphoton_sideband_spec = {mode: run_multiphoton_stage(mode, 'swap', 'spectroscopy', 'frequency',
                                                        center=sideband_spec_center_mhz.get(mode),
                                                        half_span=25, expts=500, length=1) for mode in multiphoton_swap_modes}

# %% [markdown]
# ### Sideband frequency and length chevron

# %%
multiphoton_sideband_chevron = {mode: run_multiphoton_stage(mode, 'swap', 'chevron', 'frequency', half_span=8,
                                                           sweep_npts=32, start=1) for mode in multiphoton_swap_modes}

# %% [markdown]
# ### Sideband fine frequency

# %%
multiphoton_sideband_freq = {mode: run_multiphoton_stage(mode, 'swap', 'error_amplification', 'frequency', half_span=0.3,
                                                        expts=50, n_pulses=7, n_step=1) for mode in multiphoton_swap_modes}

# %% [markdown]
# ### Sideband length Rabi

# %%
multiphoton_sideband_rabi = {mode: run_multiphoton_stage(mode, 'swap', 'length_rabi', 'length', pi_span=3, expts=200)
                            for mode in multiphoton_swap_modes}

# %% [markdown]
# # Storage

# %%
# Storage Spectroscopy - New Pattern with CharacterizationRunner
from experiments.MM_dual_rail_base import MM_dual_rail_base


def get_storage_mode_parameters(ds_storage, config_thisrun, man_mode_no, stor_mode_no):
    """Get pulse parameters for a given storage mode."""
    stor_name = 'M' + str(man_mode_no) + '-S' + str(stor_mode_no)
    freq = ds_storage.get_freq(stor_name)
    gain = ds_storage.get_gain(stor_name)
    pi_len = ds_storage.get_pi(stor_name)
    h_pi_len = ds_storage.get_h_pi(stor_name)
    # flux_low_ch = config_thisrun.hw.soc.dacs.flux_low.ch
    # flux_high_ch = config_thisrun.hw.soc.dacs.flux_high.ch
    # ch = flux_low_ch if freq < 1000 else flux_high_ch
    ch = 'low' if freq < 1800 else 'high'

    prepulse, postpulse = man_photon_pulses(station, man_mode_no)

    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse


# %% [markdown]
# ## Man-Stor Spectroscopy

# %%
# Configuration defaults
stor_spec_defaults = AttrDict(dict(
    step=0.08,  # bw/expts = 20/250
    expts=250,
    reps=200,
    qubit=[0],
    qubits=[0], # oh well
    man_mode_no=1, 
    stor_mode_no=1, 
    bw=20, 
    flux_drive_ch = None, # if None, will default to value in ds_storage
    flux_drive_gain = None, # if None, will default to value in ds_storage
    flux_drive_duration=5,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    relax_delay=500,
))

def stor_spec_preproc(station, default_expt_cfg, **kwargs):
    """Preprocessor for storage spectroscopy."""
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get storage mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_storage_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.stor_mode_no
    )
    # # manually chaning freq for testing
    if kwargs.get("center_freq", 0) != 0:
        print(" manually chaning freq for testing")
        freq = expt_cfg.center_freq
    
    
    # Set sweep range centered on current frequency
    expt_cfg.start = freq - expt_cfg.bw/2
    expt_cfg.step = expt_cfg.bw / expt_cfg.expts
    # Override flux drive channel and gain if specified
    if expt_cfg.flux_drive_gain:
        gain = expt_cfg.flux_drive_gain
    if expt_cfg.flux_drive_ch:
        ch = expt_cfg.flux_drive_ch
    
    gain = int(0.5 * gain)
    expt_cfg.flux_drive = [ch, 1, gain, expt_cfg.flux_drive_duration]
    # [ch, freq_placeholder, gain, length(us)]

    # and prepulse/postpulse
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Storage spectroscopy for M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}: freq={freq:.3f} MHz, bw={expt_cfg.bw} MHz')
    
    return expt_cfg

def stor_spec_postproc(station, expt):
    """Postprocessor for storage spectroscopy - updates frequency."""
    man_mode_no = expt.cfg.expt.man_mode_no
    stor_mode_no = expt.cfg.expt.stor_mode_no
    stor_name = f'M{man_mode_no}-S{stor_mode_no}'
    new_freq = expt.data['fit_avgi'][2]
    station.ds_storage.update_freq(stor_name, new_freq)
    print(f'Updated frequency for {stor_name}: {new_freq:.4f} MHz')

# Create runner
stor_spec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.FluxSpectroscopyF0g1Experiment,
    default_expt_cfg=stor_spec_defaults,
    preprocessor=stor_spec_preproc,
    postprocessor=stor_spec_postproc,
    job_client=client,
)


# %%
# Execute for all storage modes
storage_spec_gain = {(1, 1): 6000}
save_storage_spec = False
if expts_to_run.get('stor_spectroscopy', False):
    stor_specs = {}
    # for stor_mode_no in range(1, 8):  # modes 1-7
    for mode, stor_mode_no in [(mode, storage) for mode in man_modes for storage in stor_modes]:
        print(f'Running storage spectroscopy for mode {stor_mode_no}')
        stor_specs[mode, stor_mode_no] = stor_spec_runner.execute(
            man_mode_no=mode,
            dump_mode_no=reset_dump_modes[mode],
            stor_mode_no=stor_mode_no,
            # coupler_current=coupler_current,
            flux_drive_gain=storage_spec_gain.get((mode, stor_mode_no)),
            flux_drive_duration=2,
            postprocess=save_storage_spec,
            # expts = 800,
            reps = 400,
            # relax_delay = 800,
            bw = 20, #80
            # center_freq = 522
        )
        stor_specs[mode, stor_mode_no].display()

# %%
station.update_all_station_snapshots()


# %% [markdown]
# ## Man-dump

# %%
def get_dump_mode_parameters(ds_storage, config_thisrun, man_mode_no, dump_mode_no):
    """Get pulse parameters for a given dump mode."""
    dump_name = 'M' + str(man_mode_no) + '-D' + str(dump_mode_no)
    freq = ds_storage.get_freq(dump_name)
    gain = ds_storage.get_gain(dump_name)
    pi_len = ds_storage.get_pi(dump_name)
    h_pi_len = ds_storage.get_h_pi(dump_name)
    flux_low_ch = config_thisrun.hw.soc.dacs.flux_low.ch
    flux_high_ch = config_thisrun.hw.soc.dacs.flux_high.ch
    # ch = flux_low_ch if freq < 1000 else flux_high_ch
    ch = 4 
    print('Flux channel manually set to 4')

    prepulse, postpulse = man_photon_pulses(station, man_mode_no)

    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse


# %%
if 1 in man_modes:  # M1-only hand-tuned dump check
    station.ds_storage.get_freq('M1-D1')

# %% [markdown]
# ### Spectroscopy

# %%
# Configuration defaults
dump_spec_defaults = AttrDict(dict(
    # step=0.5,  # span/expts = 100/200
    # step=0.2,  # span/expts = 100/200
    span=20,
    expts=200,
    reps=200,
    qubit=[0],
    flux_drive=[0, 1, 5000, 5],  # [ch, freq_placeholder, gain, length(us)]
    flux_gain=5000,
    flux_length=5,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    relax_delay=2500,
))

def dump_spec_preproc(station, default_expt_cfg, man_mode_no=1, dump_mode_no=1, **kwargs):
    """Preprocessor for dump spectroscopy."""
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(man_mode_no=man_mode_no, dump_mode_no=dump_mode_no, **kwargs)
    
    # Get dump mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_dump_mode_parameters(
        station.ds_storage, station.hardware_cfg, man_mode_no, dump_mode_no
    )
    print(f'Using dump mode parameters: freq={freq}, gain={gain}, pi_len={pi_len}, h_pi_len={h_pi_len}, ch={ch}')

    # freq = 2313.0 # Temporary fix for testing
    # freq = 2313
    # freq = 1426.889688
    c_freq = expt_cfg.get("center_freq", None)
    if c_freq is not None:
        freq = c_freq
    # Set sweep range centered on current frequency
    expt_cfg.step = expt_cfg.span / expt_cfg.expts
    expt_cfg.start = freq - expt_cfg.span/2
    print(f'Set sweep start to {expt_cfg.start} MHz with step {expt_cfg.step} MHz')
    
    # Set flux drive channel and prepulse/postpulse
    expt_cfg.flux_drive = ['high', expt_cfg.start, expt_cfg.flux_gain, expt_cfg.flux_length]
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Dump spectroscopy for M{man_mode_no}-D{dump_mode_no}: freq={freq:.3f} MHz, span={expt_cfg.span} MHz, gain={expt_cfg.flux_gain}, length={expt_cfg.flux_length} us')
    
    return expt_cfg

def dump_spec_postproc(station, expt):
    """Postprocessor for dump spectroscopy - updates frequency."""
    dump_name = f'M{expt.cfg.expt.man_mode_no}-D{expt.cfg.expt.dump_mode_no}'
    new_freq = expt.data['fit_avgi'][2]
    expt.display()
    station.ds_storage.update_freq(dump_name, new_freq)
    print(f'Updated frequency for {dump_name}: {new_freq:.4f} MHz')
    station.snapshot_man1_storage_swap(update_main=False)



# %%
# Create runner
dump_spec_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.FluxSpectroscopyF0g1Experiment,
    default_expt_cfg=dump_spec_defaults,
    preprocessor=dump_spec_preproc,
    # postprocessor=dump_spec_postproc,
    job_client=client,
)

# Execute for dump modes
dump_specs = {}
if expts_to_run.get('dump_spectroscopy', True):
    for mode in man_modes:
        dump_mode_no = reset_dump_modes[mode]  # specify which dump modes to run
        print(f'Running dump spectroscopy for M{mode}-D{dump_mode_no}')
        dump_specs[mode, dump_mode_no] = dump_spec_runner.execute(
            man_mode_no=mode,
            dump_mode_no=dump_mode_no,
            flux_gain=3000,
            flux_length=20,
            # step=0.05,  # bw/expts = 100/200
            # span=30.0,
            # expts=600,
            span=40, #originally 20
            expts=200,
            # reps=300,
            # center_freq = 2265,
            # span=100, #originally 20
            # expts=400,
            reps=100,
            # coupler_current=coupler_current,
        )
        dump_specs[mode, dump_mode_no].display()

# %%
for expt in dump_specs.values():
    dump_spec_postproc(station, expt)

# %%
fig, ax = plt.subplots(figsize = (12, 4))
for (mode, dump), expt in dump_specs.items():
    ax.plot(expt.data['xpts'], expt.data['avgi'], label=f'M{mode}-D{dump}')
if dump_specs:
    ax.legend()
# ax.set_xlim([2100, 2160])

# %%
if 1 in man_modes:  # M1-only hand-tuned dump check
    station.ds_storage.update_freq('M1-D2', 2272)

# %%
if 1 in man_modes:  # M1-only hand-tuned dump check
    station.ds_storage.get_freq('M1-D2')

# %%
if 1 in man_modes:  # M1-only hand-tuned dump check
    station.ds_storage.update_freq('M1-D2', 2254.2140925381445)
    station.ds_storage.update_pi('M1-D2', 40.0)
    station.ds_storage.update_h_pi('M1-D2', 20.0)
    station.ds_storage.update_gain('M1-D2', 20000)


    # station.ds_storage.update_pi('M1-S5', 0.0)
    # station.ds_storage.update_h_pi('M1-S5', 0.0)
    station.ds_storage.df
    station.snapshot_man1_storage_swap(update_main=False)

# %% [markdown]
# ### Dump chevron
#
# Note that this experiment plays a flat top pulse by default while the spectroscopy plays a constant pulse.
# The man reset code uses a flat top, so this is the one that should be used for the dump reset calibration!

# %%
from fitting.fit_display_classes import ChevronFitting
from datetime import datetime

# Configuration defaults for dump sideband sweep
dump_sideband_chevron_defaults = AttrDict(dict(
    start=0.007, # start time in us
    pi_len_sweep=2.0, # total sweep length in us
    expts=30, # num steps of time
    reps=50,
    rounds=1,
    qubit=0,
    qubits=[0],
    man_mode_no=1,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    update_post_pulse_phase=[False, 0],
    relax_delay=2500,
))

def dump_sideband_chevron_preproc(station, default_expt_cfg, **kwargs):
    assert 'dump_mode_no' in kwargs
    
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get dump mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_dump_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.dump_mode_no
    )

    print('Prepulse:', prepulse)
    print('Postpulse:', postpulse)

    dump_name = f'M{expt_cfg.man_mode_no}-D{expt_cfg.dump_mode_no}'
    
    pi_len_sweep = expt_cfg.pi_len_sweep
    expt_cfg.step = pi_len_sweep / (expt_cfg.expts - 1)
    if 'gain' in expt_cfg and expt_cfg.gain is not None:
        gain = expt_cfg.gain  # Override gain if provided
    
    # expt_cfg.flux_drive = [ch, freq, gain, 0]
    expt_cfg.flux_drive = ['high', freq, gain, 0]
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Dump sideband chevron for {dump_name}: freq={freq:.3f} MHz, gain={gain}')
    
    return expt_cfg

def dump_sideband_chevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    stor_name = f'M{expt_cfg.man_mode_no}-D{expt_cfg.dump_mode_no}'

    from fitting.fit_display_classes import ChevronFitting, Spectroscopy
    import fitting.fitting as fitter
    times = mother_expt.data['xpts'][0]
    freqs = mother_expt.data['freq_sweep']

    chevron_analysis = ChevronFitting(
        frequencies=freqs,
        time=times,
        response_matrix=mother_expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )
    # chevron_analysis.analyze() # we don't actually use the results of the analysis, just for plotting
    chevron_analysis.display_results()

    # Fit to spectroscopy sice decay dominates over rabi
    print("Fitting assumes single shot blobs are rotated correctly")
    signs = [1, -1, 1]
    fitting_data = dict(
        avgi=mother_expt.data['avgi'][:, -1],
        avgq=mother_expt.data['avgq'][:, -1],
        amps=mother_expt.data['amps'][:, -1],
        xpts=freqs
    )
    spec_analysis = Spectroscopy(fitting_data, signs=signs, config=station.hardware_cfg, station=station)
    spec_analysis.analyze(fit=True)
    mother_expt.analysis = spec_analysis
    t = times[-1]
    spec_analysis.display(title=f'Dump Mode Spectroscopy (t={t} us)')

    best_freq = spec_analysis.data["fit_avgi"][2]

    plt.figure(figsize=(5, 4))
    best_freq_cut_idx = np.argmin(np.abs(freqs - best_freq))
    best_freq_cut = mother_expt.data['avgi'][best_freq_cut_idx, :]
    plt.plot(times, best_freq_cut, '.-')

    # Fit to an exponential decay
    spec_analysis.data["fit_T1_avgi"], spec_analysis.data["fit_err_T1_avgi"] = fitter.fitexp(times, best_freq_cut, fitparams=None)
    best_freq_cut_fit = fitter.expfunc(times, *spec_analysis.data["fit_T1_avgi"])
    p = spec_analysis.data['fit_T1_avgi']
    pCov = spec_analysis.data['fit_err_T1_avgi']
    t1 = p[3]
    captionStr = f'$T_1$ fit [us]: {t1:.3} $\\pm$ {np.sqrt(pCov[3][3]):.3}'
    print("final avgi value at best frequency:", best_freq_cut[-1])
    plt.plot(times, best_freq_cut_fit, label=captionStr)
    plt.legend()
    plt.xlabel("Time (us)")
    plt.ylabel("avgi")
    plt.title(f"Response at Freq {freqs[best_freq_cut_idx]} MHz")
    plt.show()


    print(f"Best frequency found: {best_freq:.4f} MHz")
    station.ds_storage.update_freq(stor_name, best_freq)
    print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz (ds_storage)")

    pi_len = 3 * t1
    station.ds_storage.update_pi(stor_name, pi_len)
    print('Updated the pi length to 3 T1:', pi_len, "(ds_storage)")

    station.ds_storage.update_h_pi(stor_name, pi_len / 2)
    print('Updated the h_pi length to:', pi_len / 2, "(ds_storage)")

    gain = expt_cfg.get('gain', station.ds_storage.get_gain(stor_name))
    station.ds_storage.update_gain(stor_name, gain)
    print('Updated gain to:', gain)

    station.snapshot_man1_storage_swap(update_main=False)


dump_sideband_chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=dump_sideband_chevron_defaults,
    sweep_param='freq',
    # sweep_param='gain',
    preprocessor=dump_sideband_chevron_preproc,
    postprocessor=dump_sideband_chevron_postproc,
    job_client=client,
)


# %%
dump_chevron = {}
for man_mode_no in man_modes:
    # Run dump chevron for specified dump mode
    dump_mode_no = reset_dump_modes[man_mode_no]

    print(f'Running dump sideband chevron for M{man_mode_no}-D{dump_mode_no}')

    # Get current frequency for sweep range
    dump_name = f'M{man_mode_no}-D{dump_mode_no}'
    center_freq = station.ds_storage.get_freq(dump_name)
    # center_freq = 2263.309186693316
    freq_bw = 2 #10  # MHz bandwidth for frequency sweep
    # freq_bw = 30  # MHz bandwidth for frequency sweep

    dump_chevron[man_mode_no] = dump_sideband_chevron_runner.execute(
        dump_mode_no=dump_mode_no,
        man_mode_no=man_mode_no,
        sweep_start=center_freq - freq_bw/2,
        sweep_stop=center_freq + freq_bw/2,
        sweep_npts=11,
        pi_len_sweep=60.0,#100.0,
        expts=20,
        reps=100,
        gain=6000,
        # coupler_current=coupler_current,
        batch=True,
        # relax_delay=8000,
    )


# %%
for expt in dump_chevron.values():
    dump_sideband_chevron_postproc(station, expt)

# %%
if 1 in man_modes:  # M1-only hand-tuned dump check
    station.ds_storage.update_pi('M1-D2', 40)
    station.ds_storage.update_h_pi('M1-D2', 20)

# %%
station.update_all_station_snapshots()

# %%
station.ds_storage.df

# %%
station.update_all_station_snapshots()

# %%

# %%
for mode in dump_chevron:
    print(f'M{mode} dump chevron:', dump_chevron[mode].cfg.expt.sweep)

# %%
station.update_all_station_snapshots(update_main=False)

# %%
station.ds_storage.df

# %% [markdown]
# ### Check man-dump reset with single shot (M1 program)

# %%
from experiments.qsim.man_reset_verification import ManActiveResetVerificationProgram

default_cfg = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end
    init_fock=False,
    parity_fast=False,
    dummys=[0],
    swept_params = ['dummy'],
    normalize=False,
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    parity_check = False,
    prepulse=False,
    postpulse=True,
    init_alpha = 0
))

man_dump_ver_relax_delay = 2000

def verification_preproc(station, 
                         default_expt_cfg, 
                         man_mode_no = 1,
                         **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    mm_base_dummy = MM_dual_rail_base(station.hardware_cfg, soccfg=station.soccfg)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no) #pulse cfg for preparing man into 1: pi_ge -> pi_ef -> M_pi
    # prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist()
    # expt_cfg.pre_sweep_pulse = prepulse
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist() # M_pi-> pi_ef (mapping man 1 to e state)
    expt_cfg.post_sweep_pulse = postpulse
    return expt_cfg

debug_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimExperiment,
    # ExptProgram=meas.qsim.t2_cavity_fluxexcursion.ActiveResetVerificationProgram,
    ExptProgram=ManActiveResetVerificationProgram,
    default_expt_cfg=default_cfg,
    preprocessor=verification_preproc,
    postprocessor=None,
    job_client=client,
)

if m1_only('Dump-reset verification'):
    debug = debug_runner.execute(
        init_stor = 0,
        reps = 3000,
        relax_delay = man_dump_ver_relax_delay,
        active_reset=True,
        man_reset=False,
        storage_reset=False
    )

    debug1 = debug_runner.execute(
        init_stor = 0,
        reps = 3000,
        relax_delay = man_dump_ver_relax_delay,
        active_reset=True,
        man_reset=True,
        storage_reset=False,
        reset_dump_mode = 1
    )

# %%
if 1 in man_modes:
    plt.plot(np.array(debug.data['idata'])[0, 2::3], label = 'without active reset')
    plt.plot(np.array(debug1.data['idata'])[0, 2::3], label = 'with active reset')
    plt.legend()

# %%
if 1 in man_modes:
    import numpy as np
    import matplotlib.pyplot as plt

    i_wo = np.array(debug.data['idata'])[0, 2::3]
    q_wo = np.array(debug.data['qdata'])[0, 2::3]

    i_w = np.array(debug1.data['idata'])[0, 2::3]
    q_w = np.array(debug1.data['qdata'])[0, 2::3]

    i_th = -25
    q_th = -20

    def calc_populations(i_data, q_data, i_th, q_th):
        total = len(i_data)

        cnt_left = np.sum(i_data < i_th)
        cnt_top_right = np.sum((i_data >= i_th) & (q_data >= q_th))
        cnt_bot_right = np.sum((i_data >= i_th) & (q_data < q_th))

        p_left = cnt_left / total * 100
        p_top_right = cnt_top_right / total * 100
        p_bot_right = cnt_bot_right / total * 100

        return p_left, p_top_right, p_bot_right

    pop_wo = calc_populations(i_wo, q_wo, i_th, q_th)
    pop_w = calc_populations(i_w, q_w, i_th, q_th)

    title_wo = f"g: {pop_wo[0]:.1f}%, f: {pop_wo[1]:.1f}%, e: {pop_wo[2]:.1f}%"
    title_w = f"g: {pop_w[0]:.1f}%, f: {pop_w[1]:.1f}%, e: {pop_w[2]:.1f}%"

    fig, ax = plt.subplots(1, 3 , figsize = (24,8))
    color_list = ['red', 'black']
    transp = 0.5

    ax[0].scatter(i_wo, q_wo, label = 'without active man reset', color = color_list[0], alpha = transp)
    ax[0].scatter(i_w, q_w, label = 'with active man reset', color = color_list[1], alpha = transp)
    ax[0].set_xlabel("I (ADC unt)")
    ax[0].set_ylabel("Q (ADC unt)")
    ax[0].set_title("Combined View")
    ax[0].legend()

    ax[1].scatter(i_wo,
                  q_wo,
                #   label = 'without active reset',
                  color = color_list[0],
                  alpha = transp)
    ax[1].set_xlabel("I (ADC unt)")
    ax[1].set_ylabel("Q (ADC unt)")
    ax[1].set_title(f"Without Man Reset\n{title_wo}")

    ax[2].scatter(i_w,
                  q_w,
                #   label = 'with active reset',
                  color = color_list[1],
                  alpha = transp)
    ax[2].set_title(f"With Man Reset\n{title_w}")
    ax[2].set_xlabel("I (ADC unt)")
    ax[2].set_ylabel("Q (ADC unt)")

    for axis in ax:
        axis.axvline(i_th, color='gray', linestyle='--', alpha=0.8)
        axis.hlines(q_th, xmin=i_th, xmax=axis.get_xlim()[1], color='gray', linestyle='--', alpha=0.8)


    fig.tight_layout()
    plt.show()

# %% [markdown]
# ### Man-Dump reset parameter optimization

# %%
from experiments.qsim.man_reset_verification import ManActiveResetVerificationProgram

default_cfg = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    ro_stor=0, # storage mode number that gets read out in the end
    init_fock=False,
    parity_fast=False,
    dummys=[0],
    swept_params = ['dummy'],
    normalize=False,
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    parity_check = False,
    prepulse=False,
    postpulse=True,
    init_alpha = 0
))

man_dump_ver_relax_delay = 0.2
if 1 in man_modes:
    station.ds_storage.update_pi('M1-D2', 40)

def verification_preproc(station, 
                         default_expt_cfg, 
                         man_mode_no = 1,
                         **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    mm_base_dummy = MM_dual_rail_base(station.hardware_cfg, soccfg=station.soccfg)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no) #pulse cfg for preparing man into 1: pi_ge -> pi_ef -> M_pi
    # prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist()
    # expt_cfg.pre_sweep_pulse = prepulse
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist() # M_pi-> pi_ef (mapping man 1 to e state)
    expt_cfg.post_sweep_pulse = postpulse
    return expt_cfg

debug_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimExperiment,
    # ExptProgram=meas.qsim.t2_cavity_fluxexcursion.ActiveResetVerificationProgram,
    ExptProgram=ManActiveResetVerificationProgram,
    default_expt_cfg=default_cfg,
    preprocessor=verification_preproc,
    postprocessor=None,
    job_client=client,
)

if m1_only('Dump-reset verification'):
    debug1 = debug_runner.execute(
        init_stor = 0,
        reps = 30000,
        relax_delay = man_dump_ver_relax_delay,
        active_reset=True,
        man_reset=True,
        storage_reset=False
    )

# %%
if 1 in man_modes:
    import numpy as np
    import matplotlib.pyplot as plt

    i_1 = np.array(debug1.data['idata'])[0, 0::3]
    q_1 = np.array(debug1.data['qdata'])[0, 0::3]

    i_2 = np.array(debug1.data['idata'])[0, 1::3]
    q_2 = np.array(debug1.data['qdata'])[0, 1::3]

    i_3 = np.array(debug1.data['idata'])[0, 2::3]
    q_3 = np.array(debug1.data['qdata'])[0, 2::3]

    i_list = [i_1, i_2, i_3]
    q_list = [q_1, q_2, q_3]

    i_th = -100
    q_th = -200

    def calc_populations(i_data, q_data, i_th, q_th):
        total = len(i_data)

        cnt_left = np.sum(i_data < i_th)
        cnt_top_right = np.sum((i_data >= i_th) & (q_data >= q_th))
        cnt_bot_right = np.sum((i_data >= i_th) & (q_data < q_th))

        p_left = cnt_left / total * 100
        p_top_right = cnt_top_right / total * 100
        p_bot_right = cnt_bot_right / total * 100

        return p_left, p_top_right, p_bot_right

    # pop_1 = calc_populations(i_1, q_1, i_th, q_th)
    # pop_2 = calc_populations(i_2, q_2, i_th, q_th)
    # pop_3 = calc_populations(i_3, q_3, i_th, q_th)
    pop_list = [calc_populations(i_list[i], q_list[i], i_th, q_th) for i in range(3)]
    title_list = [f"g: {pop_list[i][0]:.1f}%, f: {pop_list[i][1]:.1f}%, e: {pop_list[i][2]:.1f}%" for i in range(3)]
    label_list = [
        "After $\\pi_{ge}\\rightarrow\\pi_{ef}\\rightarrow M_{\\pi}$",
        "After ge active reset",
        "After $M_{\\pi}\\rightarrow\\pi_{ef}$"
    ]
    # title_1 = f"g: {pop_1[0]:.1f}%, f: {pop_1[1]:.1f}%, e: {pop_1[2]:.1f}%"
    # title_2 = f"g: {pop_2[0]:.1f}%, f: {pop_2[1]:.1f}%, e: {pop_2[2]:.1f}%"
    # title_3 = f"g: {pop_3[0]:.1f}%, f: {pop_3[1]:.1f}%, e: {pop_3[2]:.1f}%"



    fig, ax = plt.subplots(1, 3 , figsize = (24,8))
    color_list = ['red', 'black']
    transp = 0.05
    for i in range(3):
        ax[i].scatter(i_list[i],
                      q_list[i],
                    label = label_list[i],
                    color = color_list[0],
                    alpha = transp)
        ax[i].set_xlabel("I (ADC unt)")
        ax[i].set_ylabel("Q (ADC unt)")
        ax[i].set_title(label_list[i]+"\n"+title_list[i])
        ax[i].legend()

    # ax[1].scatter(i_wo,
    #               q_wo,
    #             #   label = 'without active reset',
    #               color = color_list[0],
    #               alpha = transp)
    # ax[1].set_xlabel("I (ADC unt)")
    # ax[1].set_ylabel("Q (ADC unt)")
    # ax[1].set_title(f"Without Man Reset\n{title_wo}")

    # ax[2].scatter(i_w,
    #               q_w,
    #             #   label = 'with active reset',
    #               color = color_list[1],
    #               alpha = transp)
    # ax[2].set_title(f"With Man Reset\n{title_w}")
    # ax[2].set_xlabel("I (ADC unt)")
    # ax[2].set_ylabel("Q (ADC unt)")

    for axis in ax:
        axis.axvline(i_th, color='gray', linestyle='--', alpha=0.8)
        axis.hlines(q_th, xmin=i_th, xmax=axis.get_xlim()[1], color='gray', linestyle='--', alpha=0.8)


    fig.tight_layout()
    plt.show()

# %% [markdown]
# ### Ringdown Analysis (to be deleted..)

# %%
if 1 in man_modes:
    rlx_del_com_mda_list = np.linspace(0, 0.5, 51)
    debug1_list = []
    for rlx_del_com_mda in tqdm(rlx_del_com_mda_list):
        debug1 = debug_runner.execute(
            init_stor = 0,
            reps = 30000,
            relax_delay = rlx_del_com_mda,
            active_reset=True,
            man_reset=True,
            storage_reset=False
        )
        debug1_list.append(debug1)

# %%
if 1 in man_modes:
    import numpy as np
    import matplotlib.pyplot as plt

    i_th = -100
    q_th = -200

    g_population_first_stage = []
    for debug1 in debug1_list:
        i_1 = np.array(debug1.data['idata'])[0, 0::3]
        q_1 = np.array(debug1.data['qdata'])[0, 0::3]

        i_2 = np.array(debug1.data['idata'])[0, 1::3]
        q_2 = np.array(debug1.data['qdata'])[0, 1::3]

        i_3 = np.array(debug1.data['idata'])[0, 2::3]
        q_3 = np.array(debug1.data['qdata'])[0, 2::3]

        i_list = [i_1, i_2, i_3]
        q_list = [q_1, q_2, q_3]

        pop_list = [calc_populations(i_list[i], q_list[i], i_th, q_th) for i in range(3)]
        title_list = [f"g: {pop_list[i][0]:.1f}%, f: {pop_list[i][1]:.1f}%, e: {pop_list[i][2]:.1f}%" for i in range(3)]
        label_list = [
            "After $\\pi_{ge}\\rightarrow\\pi_{ef}\\rightarrow M_{\\pi}$",
            "After ge active reset",
            "After $M_{\\pi}\\rightarrow\\pi_{ef}$"
        ]
        g_population_first_stage.append(pop_list[0][0])

    fig, ax = plt.subplots()
    ax.scatter(rlx_del_com_mda_list,
               g_population_first_stage)
    fig.tight_layout()
    plt.show()

# %%
if 1 in man_modes:
    if 1 in man_modes:  # M1-only hand-tuned dump check
        station.ds_storage.get_pi('M1-D2')

# %%
if 1 in man_modes:
    station.update_all_station_snapshots(update_main = False)

# %% [markdown]
# ## Man-coupler (to be updated)

# %%
def get_coupler_parameters(ds_storage, config_thisrun, man_mode_no):
    """
    Get pulse parameters for a given storage mode. 
    Also returns prepulse and postpulse (single photon prep and meas for ge meas)

    Args:
        ds_storage: Dataset object for managing frequency data.
        config_thisrun: Configuration dictionary for the current run.
        man_mode_no: Manipulation mode number.
        dump_mode_no: Dump mode number.

    Returns:
        A tuple containing freq, gain, ch, prepulse, and postpulse.
    """
    stor_name = 'M' + str(man_mode_no) + '-C'
    freq = ds_storage.get_freq(stor_name)
    gain = ds_storage.get_gain(stor_name)
    pi_len = ds_storage.get_pi(stor_name)
    h_pi_len = ds_storage.get_h_pi(stor_name)
    ch = 'low' if freq < 1000 else 'high'

    from experiments.MM_dual_rail_base import MM_dual_rail_base
    mm_base_dummy = MM_dual_rail_base(config_thisrun, soccfg=soc)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist() # for ge meas, only do f0g1 and ef pi

    prepulse_overwrite = [['multiphoton', 'g0-e0', 'pi', 0],
                            ['multiphoton', 'e0-f0', 'pi', 0],
                            ['multiphoton', 'f0-g1', 'pi', 0]
                        ]
    postpulse_overwrite = [ ['multiphoton', 'f0-g1', 'pi', 0],
                            ['multiphoton', 'e0-f0', 'pi', 0]]
    prepulse = mm_base_dummy.get_prepulse_creator(prepulse_overwrite).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(postpulse_overwrite).pulse.tolist()

    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse


def do_coupler_spectroscopy(config_thisrun, 
                         ds_storage, 
                         expt_path, 
                         config_path, 
                         man_mode_no = 1, 
                         dump_no = 1,
                         flux_gain = 5000,
                         flux_length = 1):
    """
    Run the Flux Spectroscopy F0g1 Experiment.

    This function performs a flux spectroscopy experiment to measure the transition frequency
    between the f0 and g1 states of a qubit. It configures the experiment parameters, executes
    the experiment, and saves the results.

    Args:
        config_thisrun (AttrDict): Configuration dictionary for the current run.
        ds_storage (dataset.storage_man_swap_dataset): Dataset object for managing frequency data.
        expt_path (str): Path to save the experiment results.
        config_path (str): Path to the configuration file.
        man_mode_no (int, optional): Manipulation mode number (default is 1).
        dump_no (int, optional): Storage mode number (default is 1).

    Returns:
        FluxSpectroscopyF0g1Experiment: The experiment object containing the results.
    """
    flux_spec = meas.single_qubit.rf_flux_spectroscopy_f0g1.FluxSpectroscopyF0g1Experiment(
        soccfg=soc, path=expt_path, prefix='FluxSpectroscopyF0g1Experiment', config_file=config_path
    )

    flux_spec.cfg = AttrDict(deepcopy(config_thisrun))

    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_coupler_parameters(ds_storage, config_thisrun,
                                                                      man_mode_no)

    # freq = 420

    flux_spec.cfg.expt = dict(
        start=freq - 50,  # Start RF frequency [MHz]
        step=1,  # Step size [MHz]
        expts=100,  # Number of experiments
        reps=200,  # Number of averages per point
        qubit=[0],
        flux_drive=[ch, 1, flux_gain, flux_length],  # RF flux modulation parameters [low/high (ch), freq (will be overwritten), gain, length(us)]
        prepulse=True,
        postpulse=True,
        active_reset=False,
        man_reset=False,
        storage_reset=False,
        pre_sweep_pulse= prepulse,
        post_sweep_pulse= postpulse,
    )

    flux_spec.cfg.device.readout.relax_delay = [2500]  # Wait time between experiments [us]
    flux_spec.go(analyze=False, display=False, progress=True, save=True)
    return flux_spec


def update_dump_spectroscopy(flux_spec, ds_storage, man_mode_no = 1):
    """Update the configuration based on Flux Spectroscopy F0g1 experiment results."""
    # Update the dataset with the new frequency
    ds_storage.update_freq('M' + str(man_mode_no) + '-C', flux_spec.data['fit'][2])
    print(f"Updated frequency for M{man_mode_no}-C: {flux_spec.data['fit'][2]}")

if 1 in man_modes:
    ds_storage.update_freq('M1-C', 919)

# %%
if m1_only('Legacy coupler spectroscopy'):
    spec = do_coupler_spectroscopy(config_thisrun, ds_storage, expt_path, config_path, man_mode_no=1,
                                flux_gain=10000, flux_length=2)
    # analyze_and_display_stor_spectroscopy(spec)
    # update_dump_spectroscopy(spec, ds_storage, 1, 1)
    spec.analyze(fit=True)
    spec.display()


# %% [markdown]
# ## Freq Chevron

# %%
from datetime import datetime

# Configuration defaults for sideband sweep
sideband_chevron_defaults = AttrDict(dict(
    start=0.007, # start time in us
    pi_len_sweep=2.0, # total sweep length in us
    expts=25, # num steps of time
    reps=50,
    rounds=1,
    qubit=0,
    qubits=[0],
    man_mode_no=1,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    update_post_pulse_phase=[False, 0],
    relax_delay=2500,
))

def sideband_chevron_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get storage mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_storage_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.stor_mode_no
    )

    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    
    pi_len_sweep = expt_cfg.pi_len_sweep
    expt_cfg.step = pi_len_sweep / (expt_cfg.expts - 1)
    if 'gain' in expt_cfg and expt_cfg.gain is not None:
        gain = expt_cfg.gain  # Override gain if provided
    if expt_cfg.get("freq", None) is not None:
        freq = expt_cfg.freq
        print(f"frequency overrided to {freq} MHz")
    expt_cfg.flux_drive = [ch, freq, gain, 0]
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Sideband chevron for {stor_name}: freq={freq:.3f} MHz, gain={gain}')
    
    return expt_cfg

def sideband_chevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    print('mom', expt_cfg)
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=mother_expt.data['freq_sweep'],
        time=mother_expt.data['xpts'][0],
        response_matrix=mother_expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()
    
    best_freq = chevron_analysis.results.get('best_frequency_contrast')
        
    if best_freq and not expt_cfg.get("skip_update", False):
        print(f"Best frequency found: {best_freq:.4f} MHz")
        station.ds_storage.update_freq(stor_name, best_freq)
        print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz")
        pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
        station.ds_storage.update_pi(stor_name, pi_len)
        print('Updated the pi length to:', pi_len)
        station.ds_storage.update_h_pi(stor_name, pi_len / 2)
        print('Updated the h_pi length to:', pi_len / 2)
        station.ds_storage.update_gain(stor_name, expt_cfg.flux_drive[2])
        print('Updated gain to:', expt_cfg.flux_drive[2])
    mother_expt.analysis = chevron_analysis
    # station.snapshot_man1_storage_swap(update_main=False)

sideband_chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=sideband_chevron_defaults,
    sweep_param='freq',
    preprocessor=sideband_chevron_preproc,
    postprocessor=sideband_chevron_postproc,
    job_client=client
)

# %%
storage_chevrons = {}
if expts_to_run['sideband_freq_sweep']:
    for mode in man_modes:
        for stor_i in stor_modes:
            stor_name = f'M{mode}-S{stor_i}'
            rd = 3000


            manual_center_freq = None#None # #880.9
            manual_gain = None #None #8000
            prefactor = 5
            pi_len_sweep = 10 #10

            freq_span = 0.5*prefactor  # MHz
            freq_step = 0.05*prefactor

            # freq_span = 0.25*prefactor  # MHz
            # freq_step = 0.025*prefactor


            if manual_center_freq is None:
                sweep_start = station.ds_storage.get_freq(stor_name) - freq_span/2
                sweep_stop=station.ds_storage.get_freq(stor_name) + freq_span/2
            else:
                sweep_start=manual_center_freq - freq_span/2
                sweep_stop=manual_center_freq + freq_span/2
            if manual_gain is None:
                gain = station.ds_storage.get_gain(stor_name)
            else:
                gain = manual_gain

            print(f'Running sideband chevron for {stor_name}')
            storage_chevrons[mode, stor_i] = sideband_chevron_runner.execute(
                man_mode_no=mode,
                reps=50,
                stor_mode_no=stor_i,

                sweep_start=sweep_start,
                sweep_stop=sweep_stop,
                sweep_npts=int(freq_span//freq_step + 1),
                pi_len_sweep=pi_len_sweep,

                expts=50, # num steps of time
                gain=gain,
                # freq = 1242.8,
                skip_update = True,
                batch = True,

                relax_delay = rd

            )
            expt = storage_chevrons[mode, stor_i]
            expt.analysis.display_results(title=expt.fname)


# %%
station.ds_storage.df

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## Storage Length Rabi

# %%
# Storage length Rabi with n photons in M1. Verification only: no config update.
# Prep |g, n> in M1 -> M1-Sx flux pulse of swept length -> map M1 |n> to qubit e
# with f{n-1}-g{n} then e{n-1}-f{n-1}.
# Qubit in e means M1 still has n photons. At the swap pi length the signal should go to g.
stor_rabi_defaults = AttrDict(dict(
    start=0.0,            # us; length 0 plays no swap, a reference point
    length_sweep=None,    # us; None -> 3 x the dataset pi length
    expts=51,
    reps=200,
    rounds=1,
    qubit=0,
    qubits=[0],
    man_mode_no=1,
    man_photon_no=1,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    update_post_pulse_phase=[False, 0],
    relax_delay=2500,
))

def stor_rabi_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    n = int(expt_cfg.man_photon_no)
    assert n >= 1, 'man_photon_no must be >= 1'

    freq, gain, pi_len, h_pi_len, ch, _, _ = get_storage_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.stor_mode_no
    )
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'

    # The Program reads freq/gain from cfg.expt when the keys exist, so always set real values
    if expt_cfg.get('freq', None) is not None:
        freq = expt_cfg.freq
    if expt_cfg.get('gain', None) is not None:
        gain = expt_cfg.gain
    expt_cfg.freq = freq
    expt_cfg.gain = gain
    expt_cfg.flux_drive = [ch, freq, gain, 0]

    length_sweep = expt_cfg.length_sweep if expt_cfg.length_sweep is not None else 3 * pi_len
    expt_cfg.step = length_sweep / (expt_cfg.expts - 1)

    if expt_cfg.man_mode_no == 2:
        if n != 1:
            raise ValueError('M2 storage length Rabi requires the one-photon pulse')
        expt_cfg.pre_sweep_pulse, expt_cfg.post_sweep_pulse = man_photon_pulses(station, 2)
    else:
        mm_base_dummy = MM_dual_rail_base(station.hardware_cfg, soccfg=station.soccfg)
        prep = mm_base_dummy.prep_man_fock_state(expt_cfg.man_mode_no, str(n))
        post = prep[-1:-3:-1]  # f{n-1}-g{n}, then e{n-1}-f{n-1}
        expt_cfg.pre_sweep_pulse = mm_base_dummy.get_prepulse_creator(prep).pulse.tolist()
        expt_cfg.post_sweep_pulse = mm_base_dummy.get_prepulse_creator(post).pulse.tolist()

    print(f'Storage length Rabi {stor_name}, n={n}: freq={freq:.3f} MHz, gain={gain}, '
          f'sweep 0..{length_sweep:.3f} us (ds pi = {pi_len:.3f} us)')
    print('  post pulse:', expt_cfg.post_sweep_pulse)
    return expt_cfg

stor_rabi_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=stor_rabi_defaults,
    preprocessor=stor_rabi_preproc,
    postprocessor=None,  # verification only: nothing is written back
    job_client=client,
    show=False,
)

# %%
stor_rabis = {}
if expts_to_run['sideband_length_rabi']:
    for mode in man_modes:
        photon_numbers = [1, 2, 3] if mode == 1 else [1]
        for stor_i in stor_modes:
            stor_rabis[mode, stor_i] = stor_rabi_runner.execute(
                man_mode_no=mode,
                overrides=[dict(man_photon_no=n) for n in photon_numbers],
                stor_mode_no=stor_i,
                reps=200,
                expts=51,
                # length_sweep=3.0,
                postprocess=False,
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            for n, expt in zip(photon_numbers, stor_rabis[mode, stor_i]):
                ax.plot(expt.data['xpts'], expt.data['avgi'], 'o-', ms=3, label=f'n = {n}')
                print(f'n = {n}: {expt.fname}')
            ax.axvline(station.ds_storage.get_pi(f'M{mode}-S{stor_i}'), color='0.3', ls='--', label='ds pi length')
            ax.set(xlabel=f'M{mode}-S{stor_i} swap length [us]', ylabel='avg I [ADC]',
                   title=f'Storage length Rabi M{mode}-S{stor_i}: qubit e = M{mode} still has n photons')
            ax.legend()
            plt.show()


# %%
station.hardware_cfg.device.readout

# %%
from datetime import datetime

# Configuration defaults for sideband sweep
sideband_chevron_defaults = AttrDict(dict(
    start=0.007, # start time in us
    pi_len_sweep=2.0, # total sweep length in us
    expts=25, # num steps of time
    reps=50,
    rounds=1,
    qubit=0,
    qubits=[0],
    man_mode_no=1,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    update_post_pulse_phase=[False, 0],
    relax_delay=2500,
))

def sideband_chevron_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get storage mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_storage_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.stor_mode_no
    )

    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    
    pi_len_sweep = expt_cfg.pi_len_sweep
    expt_cfg.step = pi_len_sweep / (expt_cfg.expts - 1)
    if 'gain' in expt_cfg and expt_cfg.gain is not None:
        gain = expt_cfg.gain  # Override gain if provided
    if expt_cfg.get("freq", None) is not None:
        freq = expt_cfg.freq
        print(f"frequency overrided to {freq} MHz")
    expt_cfg.flux_drive = [ch, freq, gain, 0]
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Sideband chevron for {stor_name}: freq={freq:.3f} MHz, gain={gain}')
    
    return expt_cfg

def sideband_chevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    print('mom', expt_cfg)
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=mother_expt.data['freq_sweep'],
        time=mother_expt.data['xpts'][0],
        response_matrix=mother_expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()
    
    # best_freq = chevron_analysis.results.get('best_frequency_contrast')
        
    # if best_freq and not expt_cfg.get("skip_update", False):
    #     print(f"Best frequency found: {best_freq:.4f} MHz")
    #     station.ds_storage.update_freq(stor_name, best_freq)
    #     print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz")
    #     pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
    #     station.ds_storage.update_pi(stor_name, pi_len)
    #     print('Updated the pi length to:', pi_len)
    #     station.ds_storage.update_h_pi(stor_name, pi_len / 2)
    #     print('Updated the h_pi length to:', pi_len / 2)
    #     station.ds_storage.update_gain(stor_name, expt_cfg.flux_drive[2])
    #     print('Updated gain to:', expt_cfg.flux_drive[2])
    mother_expt.analysis = chevron_analysis
    # station.snapshot_man1_storage_swap(update_main=False)

sideband_length_Rabi_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.sideband_general.SidebandGeneralExperiment,
    default_expt_cfg=sideband_chevron_defaults,
    preprocessor=sideband_chevron_preproc,
    postprocessor=sideband_chevron_postproc,
    job_client=client
)

# %%

# %% [markdown]
# ## Freq - Amplitude Chevron

# %%
from experiments.qsim.sideband_amplitude_sweep import SidebandAmplitudeSweepExperiment


from datetime import datetime

# Configuration defaults for sideband sweep
sideband_chevron_defaults = AttrDict(dict(
    start=0, 
    expts=25, # num steps of time
    reps=50,
    rounds=1,
    qubit=0,
    qubits=[0],
    man_mode_no=1,
    prepulse=True,
    postpulse=True,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    update_post_pulse_phase=[False, 0],
    relax_delay=2500,
))

def sideband_ampchevron_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get storage mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_storage_mode_parameters(
        station.ds_storage, station.hardware_cfg, expt_cfg.man_mode_no, expt_cfg.stor_mode_no
    )

    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'
    
    # pi_len_sweep = expt_cfg.pi_len_sweep
    # expt_cfg.step = pi_len_sweep / (expt_cfg.expts - 1)
    expt_cfg.step = int(pi_gain_sweep/(expt_cfg.expts -1))
    
    expt_cfg.flux_drive = [ch, freq, 0, pi_len]
    expt_cfg.pre_sweep_pulse = prepulse
    expt_cfg.post_sweep_pulse = postpulse
    
    print(f'Sideband chevron for {stor_name}: freq={freq:.3f} MHz, gain={gain}')
    
    return expt_cfg

def sideband_ampchevron_postproc(station, mother_expt):
    expt_cfg = mother_expt.cfg.expt
    print('mom', expt_cfg)
    stor_name = f'M{expt_cfg.man_mode_no}-S{expt_cfg.stor_mode_no}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=mother_expt.data['freq_sweep'],
        time=mother_expt.data['xpts'][0],
        response_matrix=mother_expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()
    
    best_freq = chevron_analysis.results.get('best_frequency_contrast')
        
    # if best_freq:
    #     print(f"Best frequency found: {best_freq:.4f} MHz")
    #     station.ds_storage.update_freq(stor_name, best_freq)
    #     print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz")
    #     pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
    #     station.ds_storage.update_pi(stor_name, pi_len)
    #     print('Updated the pi length to:', pi_len)
    #     station.ds_storage.update_h_pi(stor_name, pi_len / 2)
    #     print('Updated the h_pi length to:', pi_len / 2)
    #     station.ds_storage.update_gain(stor_name, expt_cfg.flux_drive[2])
    #     print('Updated gain to:', expt_cfg.flux_drive[2])
    mother_expt.analysis = chevron_analysis
    # station.snapshot_man1_storage_swap(update_main=False)


sideband_amp_chevron_runner = SweepRunner(
    station=station,
    ExptClass=SidebandAmplitudeSweepExperiment,
    default_expt_cfg=sideband_chevron_defaults,
    sweep_param='freq',
    preprocessor=sideband_ampchevron_preproc,
    postprocessor=sideband_ampchevron_postproc,
    job_client=client,
)

# %%
storage_amp_chevrons = {}
for mode in man_modes:
    for stor_i in stor_modes:
        stor_name = f'M{mode}-S{stor_i}'

        freq_span = 4  # MHz
        freq_step = 0.1
        pi_gain_sweep = 20000
        gain = station.ds_storage.get_gain(stor_name)

        print(f'Running sideband chevron for {stor_name}')
        storage_amp_chevrons[mode, stor_i] = sideband_amp_chevron_runner.execute(
            man_mode_no=mode,
            reps=50,
            stor_mode_no=stor_i,
            sweep_start=station.ds_storage.get_freq(stor_name) - freq_span/2,
            sweep_stop=station.ds_storage.get_freq(stor_name) + freq_span/2,
            sweep_npts=int(freq_span//freq_step + 1),
            pi_gain_sweep=pi_gain_sweep,
            expts = 50,
        )
        storage_amp_chevrons[mode, stor_i].analysis.display_results()


# %%
idxes = [15, 16, 17, 18, 19, 20, 21, 22, 23]

if storage_amp_chevrons:
    mode, storage = next(reversed(storage_amp_chevrons))  # Defaults to the last measured pair.
    chevron_data = storage_amp_chevrons[mode, storage].data
    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_title(f'M{mode}-S{storage}')
    x = chevron_data['xpts'][0]
    for idx in idxes:
        y = chevron_data['freq_sweep'][idx]
        ax.plot(x, chevron_data['avgi'][idx], label=f"detuning = {y}")
    ax.legend()

# %%
if 1 in man_modes:  # Manual M1 frequency override
    station.ds_storage.update_freq('M1-S1', 347.57)

# %%
if 1 in man_modes:  # Manual M1 frequency override
    station.ds_storage.update_freq('M1-S6', 1244.536721 - 1.261634)

# %%
np.round(2266.797314 - 2265.535680, 6)

# %%
station.ds_storage.df

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## Error amplification

# %%
error_amp_gain1 = {}
error_amp_freq1 = {}
error_amp_gain2 = {}
error_amp_freq2 = {}

# %%
error_amp_stor_defaults = AttrDict(dict(
    reps=50,
    rounds=1,
    qubits=[0],
    man_mode_no=1,
    stor_is_dump=False,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500, 
    expts=30,
)) # Shouldn't be modifying this on the fly!
# You can use kwargs in the run function to override these values

error_amp_gain_stor_coarse_defaults = AttrDict(dict(
    n_start=0,
    n_step=3,
    n_pulses=15,
    expts=50,
))

error_amp_freq_stor_coarse_defaults = AttrDict(dict(
    n_start=0,
    n_step=1,
    n_pulses=5,
    expts=50,
))


def error_amp_stor_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    assert 'parameter_to_test' in kwargs 

    # construct the defaults
    expt_cfg = deepcopy(default_expt_cfg)
    if kwargs['parameter_to_test'] == 'gain':
        expt_cfg.update(error_amp_gain_stor_coarse_defaults)
    elif kwargs['parameter_to_test'] == 'frequency':
        expt_cfg.update(error_amp_freq_stor_coarse_defaults)

    # override with the passed kwargs
    expt_cfg.update(kwargs)

    stor_mode_no = expt_cfg.stor_mode_no    
    man_mode_no = expt_cfg.man_mode_no

    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_storage_mode_parameters(
        station.ds_storage, station.hardware_cfg, man_mode_no, stor_mode_no
        )
    print(f"Previous freq {freq}, gain {gain}, pi_len {pi_len}")

    if 'span' not in expt_cfg:
        if expt_cfg.parameter_to_test == 'gain':
            expt_cfg.span = int(gain * 0.3)
        elif expt_cfg.parameter_to_test == 'frequency':
            expt_cfg.span = 0.15 # MHz
  
    stor_name = f'M{man_mode_no}-S{stor_mode_no}'
    pulse_type = ['storage', stor_name, 'pi', 0]

    if expt_cfg.parameter_to_test == 'frequency':
        start = freq - expt_cfg.span / 2
        step = expt_cfg.span / (expt_cfg.expts - 1)
    elif expt_cfg.parameter_to_test == 'gain':
        start = int(gain - expt_cfg.span / 2)
        step = int(expt_cfg.span / (expt_cfg.expts - 1))
    else:
        raise ValueError("parameter_to_test must be either 'frequency' or 'gain'.")
    expt_cfg.start = start
    expt_cfg.step = step
    expt_cfg.pulse_type = pulse_type 
    return expt_cfg

def error_amp_stor_postproc(station, expt):
    expt.analyze(data=expt.data, state_fin='e')

    opt_gain = expt.data['fit_avgi'][2]
    stor_name = f'M{expt.cfg.expt.man_mode_no}-S{expt.cfg.expt.stor_mode_no}'
    if expt.cfg.expt.parameter_to_test == 'gain':
        station.ds_storage.update_gain(stor_name, opt_gain)
    elif expt.cfg.expt.parameter_to_test == 'frequency':
        station.ds_storage.update_freq(stor_name, opt_gain)
    station.snapshot_man1_storage_swap(update_main=False)


error_amp_stor_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_stor_defaults,
    preprocessor=error_amp_stor_preproc,
    postprocessor=error_amp_stor_postproc,
    job_client=client,
)

# %% [markdown]
# ### Coarse

# %%
span_list        = [0.2 ,  0.2,  0.2,  0.2,  0.2,  0.2, 0.2]
gain_pref_list   = [None, None,  0.3,  0.3,  0.2, 0.15, 0.1] # should be less than 1.0


n_step_list_freq = [None, None, None, None, None, None, None]
n_step_list_gain = [None, None, None, None,    2,    2,    2]

do_freq_erroramp = True
do_gain_erroramp = True

stors_to_run_erramp = [1]

dump_mode_no = 2

for mode in man_modes:
    for i, stor_i in enumerate(stors_to_run_erramp):
        if stor_i not in stor_modes:
            continue
        stor_idx = stor_i - 1

        prefactor = gain_pref_list[stor_idx]
        if prefactor is None:
            prefactor = 0.3

        n_step_freq = n_step_list_freq[stor_idx]
        n_step_gain = n_step_list_gain[stor_idx]
        if n_step_freq is None:
            n_step_freq = 1
        if n_step_gain is None:
            n_step_gain = 1
        n_pulses_freq = 5 * n_step_freq
        n_pulses_gain = 5 * n_step_gain
        stor_name = f'M{mode}-S{stor_i}'
        print("Running", stor_name)
        if do_freq_erroramp == True:
            error_amp_freq1[mode, stor_i] = error_amp_stor_runner.execute(
                man_mode_no=mode,
                stor_mode_no=stor_i,
                parameter_to_test='frequency',
                span=span_list[stor_idx], #0.1
                expts=100,
                n_step = n_step_freq,
                n_pulses = n_pulses_freq,
                active_reset=mode == 1,
                man_reset=True,
                storage_reset=[stor_i],
                relax_delay = 200,
                reset_dump_mode=reset_dump_modes[mode]
            )
            if not station.log_measurements:
                error_amp_freq1[mode, stor_i].display()
        if do_gain_erroramp == True:
            error_amp_gain1[mode, stor_i] = error_amp_stor_runner.execute(
                man_mode_no=mode,
                stor_mode_no=stor_i,
                parameter_to_test='gain',
                span = int(station.ds_storage.get_gain(stor_name) * prefactor),
                n_step = n_step_gain,
                n_pulses=n_pulses_gain,
                active_reset=mode == 1,
                man_reset=True,
                storage_reset=[stor_i],
                relax_delay = 200,
                reset_dump_mode=reset_dump_modes[mode]
            )
            if not station.log_measurements:
                error_amp_gain1[mode, stor_i].display()


# %%
station.ds_storage.df

# %%
station.update_all_station_snapshots()

# %%
# for i, stor_i in enumerate(expts_to_run['stor_modes']):
#     if stor_i != 4: continue
#     # if stor_i == 5: continue

#     stor_name = 'M1-S' + str(stor_i)
#     print(stor_name)
#     ds.update_gain(stor_name, ds_storage.get_gain(stor_name))
#     ds.update_freq(stor_name, ds_storage.get_freq(stor_name))
#     ds.update_pi(stor_name, ds_storage.get_pi(stor_name))
#     ds.update_h_pi(stor_name, ds_storage.get_h_pi(stor_name))

# %%
for mode in man_modes:
    for storage in stor_modes:
        row = f'M{mode}-S{storage}'
        print(f'Frequency for {row}:', station.ds_storage.get_freq(row))

# %%
# #Update sweep data and analyze results

# temp_data, attrs, filename = prev_data(expt_path, prefix='sideband_general_sweep')
# print('File saved at:', filename)

# from fitting.fit_display_classes import ChevronFitting
# from datetime import datetime
# chevron_analysis = ChevronFitting(
#     frequencies=temp_data['freq_sweep'],
#     time=temp_data['xpts'][0],
#     response_matrix=temp_data['avgi'],
#     config = config_thisrun,
# )
# chevron_analysis.analyze()

# current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# chevron_analysis.display_results(save_fig=True,  title=f'M{1}-S{7}_{current_time}')

# %%
# i = 5
# print(f'Running fine sideband sweep for storage mode {i + 1}')
# freq_start = ds_storage.get_freq('M1-S' + str(i + 1)) - 1.5
# freq_stop = ds_storage.get_freq('M1-S' + str(i + 1)) + 1.5
# freq_step = 0.2
# sideband_chevrons_fine = do_sideband_general_sweep(
#     config_thisrun, ds_storage, expt_path, config_path, freq_start, freq_stop, freq_step, 
#     man_mode_no=1, stor_mode_no=i + 1, start_time=0.1, use_flux_low=True)
# # update_sideband_general_sweep(expt_path, config_thisrun, ds_storage, man_mode_no=1, stor_mode_no=i + 1, use_flux_low=True)

# %%
# i = 5
# print(f'Running fine sideband sweep for storage mode {i + 1}')
# freq_start = ds_storage.get_freq('M1-S' + str(i + 1)) - 0.5
# freq_stop = ds_storage.get_freq('M1-S' + str(i + 1)) + 0.5
# freq_step = 0.2
# sideband_chevrons_fine = do_sideband_general_sweep(
#     config_thisrun, ds_storage, expt_path, config_path, freq_start, freq_stop, freq_step, 
#     man_mode_no=1, stor_mode_no=i + 1, start_time=0.1, use_flux_low=False)
# # update_sideband_general_sweep(expt_path, config_thisrun, ds_storage, man_mode_no=1, stor_mode_no=i + 1, use_flux_low=True)

# %% [markdown]
# ### Fine

# %%
for mode in man_modes:
    for i, stor_i in enumerate([1, 2, 4, 5, 7]):
        if stor_i not in stor_modes:
            continue
        stor_idx = stor_i - 1

        prefactor = gain_pref_list[stor_idx]
        if prefactor is None:
            prefactor = 0.3
        prefactor *= 0.5
        n_step_freq = n_step_list_freq[stor_idx]
        n_step_gain = n_step_list_gain[stor_idx]
        if n_step_freq is None:
            n_step_freq = 1
        if n_step_gain is None:
            n_step_gain = 1
        n_pulses_freq = 5 * n_step_freq
        n_pulses_gain = 5 * n_step_gain
        stor_name = f'M{mode}-S{stor_i}'
        print("Running", stor_name)
        if do_freq_erroramp == True:
            error_amp_freq1[mode, stor_i] = error_amp_stor_runner.execute(
                man_mode_no=mode,
                stor_mode_no=stor_i,
                parameter_to_test='frequency',
                span=span_list[stor_idx], #0.1
                expts=100,
                n_step = n_step_freq,
                n_pulses = n_pulses_freq,
                active_reset=mode == 1,
                man_reset=True,
                storage_reset=[stor_i],
                relax_delay = 200,
                reset_dump_mode=reset_dump_modes[mode]
            )
            if not station.log_measurements:
                error_amp_freq1[mode, stor_i].display()
        if do_gain_erroramp == True:
            error_amp_gain1[mode, stor_i] = error_amp_stor_runner.execute(
                man_mode_no=mode,
                stor_mode_no=stor_i,
                parameter_to_test='gain',
                span = int(station.ds_storage.get_gain(stor_name) * prefactor),
                n_step = n_step_gain,
                n_pulses=n_pulses_gain,
                active_reset=mode == 1,
                man_reset=True,
                storage_reset=[stor_i],
                relax_delay = 200,
                reset_dump_mode=reset_dump_modes[mode]
            )
            if not station.log_measurements:
                error_amp_gain1[mode, stor_i].display()


# %%
station.update_all_station_snapshots()

# %% [markdown]
# ### Manually get the frequencies from the chevrons

# %%
# file_list = [34,36,38,40,42,44,46]
# chevrons_stor_objects = [None]* len(file_list)
# name = '_sideband_general_sweep.h5'
# # name = '_storage_sideband_sweep.h5'
# for idx, file_no in enumerate(file_list):
#     full_name = str(file_no).zfill(5)+name
#     expt_path_mod = r"H:\Shared drives\SLab\Multimode\experiment\250505_craqm\data"
#     temp_data, attrs, filename = prev_data(expt_path_mod, full_name)  # ef

#     from fitting.fit_display_classes import ChevronFitting
#     chevron_analysis = ChevronFitting(
#         frequencies=temp_data['freq_sweep'],
#         time=temp_data['xpts'][0],
#         response_matrix=temp_data['avgi']
#     )
#     chevrons_stor_objects[idx] = chevron_analysis

# %%
# i = 6
# # guess_freq = 1422.66
# anls = chevrons_stor_objects[i]
# anls.analyze()
# anls.display_results(save_fig=False, 
#                     directory=autocalib_path, 
#                     title=f'M1-S{i+1}')
#                 #  hlines = [guess_freq])
# ds_storage.update_freq('M1-S' + str(i+1), anls.results['best_frequency_period'])
# # or override manually 
# # ds_storage.update_freq('M1-S' + str(i+1), 1052.8026)

# %%
# ds_storage.df

# %%

# ds_storage.update_freq('M1-S' + str(i), guess_freq)

# %%
# ds_storage.get_freq('M1-S3')

# %%
# stor_mode_no = 1
# man_mode_no = 1
# len_rabi_storage = do_sideband_general(config_thisrun, expt_path, config_path, man_mode_no, stor_mode_no)
# update_sideband_general(len_rabi_storage, config_thisrun, ds_storage, man_mode_no, stor_mode_no)

# %% [markdown]
# ## Storage and Manipulate Cavity T2 using QsimProg

# %%
qsim_t1_default = AttrDict(dict(
        expts=1,
        rounds=1,
        qubits=[0],
        man_mode_no=1,
        swept_params=["wait"],
        prepulse=True,
        postpulse=True,
        init_fock=True,
        readout="qubit",
        map_to_qubit_ge=True,
        normalize=False,
        active_reset=True,
        man_reset=True,
        reset_dump_mode=2,
        dump_reset_iter_num=1,
    ))


t1_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.qsim.qsim_base.QsimExperiment,
    ExptProgram=meas.qsim.sideband_scramble.StorageT1Program,
    job_client=client,
    default_expt_cfg=qsim_t1_default,
)

# %%
from experiments.qsim.qsim_cavity_t2 import QsimCavityT2Experiment, QsimCavityT2Program


qsim_t2_default = AttrDict(dict(
    qubits=[0],
    man_mode_no=1,
    expts=1,
    rounds=1,
    swept_params=["wait"],
    prepulse=True,
    init_fock=False,
    init_man_fock_state="+",
    postpulse=False,
    readout="qubit",
    active_reset=True,
    man_reset=True,
    reset_dump_mode=2,
    dump_reset_iter_num=1,
))

t2_runner = CharacterizationRunner(
    station=station,
    ExptClass=QsimCavityT2Experiment,
    ExptProgram=QsimCavityT2Program,
    default_expt_cfg=qsim_t2_default,
    use_queue=False,
)

# %%
storage_modes = [4]
run_storage_t1 = True
run_storage_t2 = True

t1_storage = {}
t2_storage = {}
T1_storage = {}
T2_storage = {}

for mode in man_modes:
    for storage in storage_modes:
        if storage not in stor_modes:
            continue
        if mode == 2:
            # The Qsim programs below build M1 swaps internally. The cavity-mode
            # programs can prepare and read this storage through the M2 swap.
            cfg = AttrDict(dict(mode='storage', storage_mode_idx=storage, storage_man_mode=mode,
                                man_mode_no=mode, qubits=[0], reps=300, rounds=1, start=0,
                                active_reset=False, relax_delay=2500))
            if run_storage_t1:
                from experiments.single_qubit.t1_cavity_mode import CavityModeT1Experiment
                t1_storage[mode, storage] = CharacterizationRunner(
                    station=station, ExptClass=CavityModeT1Experiment, default_expt_cfg=cfg, job_client=client,
                ).execute(step=50, expts=101, readout='map', postprocess=False, show=False, log=False)
            if run_storage_t2:
                from experiments.single_qubit.t2_cavity_mode import CavityModeRamseyExperiment
                t2_storage[mode, storage] = CharacterizationRunner(
                    station=station, ExptClass=CavityModeRamseyExperiment, default_expt_cfg=cfg, job_client=client,
                ).execute(step=20, expts=151, ramsey_freq=0.01, parity_meas=False,
                          postprocess=False, show=False, log=False)
            continue
        if run_storage_t1 == True:
            t1_storage[mode, storage] = t1_runner.execute(
                man_mode_no=mode,
                init_stor=storage,
                ro_stor=storage,
                storage_reset=[storage],
                waits=np.arange(0, 5001, 50).tolist(),  # us
                reps=300,
                postprocess=False, show=False, log=False,
            )
        if run_storage_t2 == True:
            t2_storage[mode, storage] = t2_runner.execute(
                use_queue=False,
                man_mode_no=mode,
                init_stor=storage,
                ro_stor=storage,
                storage_reset=[storage],
                waits=np.arange(0, 3001, 20).tolist(),  # us
                ramsey_freq=0.01,                     # MHz
                reps=300,
                go_kwargs={"analyze": False},
                postprocess=False,
                show=False,
                log=False,
            )


# %%
for expt in t1_storage.values():
    expt.analyze()
    expt.display()

# %%
for expt in t2_storage.values():
    expt.analyze()
    expt.display()

# %% [markdown]
# ## Randomized Benchmarking

# %%
len([1, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 400, 500, 600, 700, 850, 1000, 1250, 1500])


# %%
# config_thisrun.device.storage.ramp_sigma
# man_mode_no = 1
# stor_mode_no = 7
# req = ds_storage.get_freq('M' + str(man_mode_no) + '-S' + str(stor_mode_no))
# gain = ds_storage.get_gain('M' + str(man_mode_no) + '-S' + str(stor_mode_no))
# hpi_length = ds_storage.get_h_pi('M' + str(man_mode_no) + '-S' + str(stor_mode_no))
# print("ramp_sigma:", config_thisrun.device.storage.ramp_sigma)
# print("man_mode_no:", man_mode_no)
# print("stor_mode_no:", stor_mode_no)
# print("freq:", req)
# print("gain:", gain)
# print("hpi_length:", hpi_length)

# %%

def do_single_beam_splitter_rb_postselection_sweep_depth(config_thisrun, ds_storage, expt_path, config_path, exp_param_file, man_mode_no=1, stor_mode_no=1,
                                                         prev_data=None):
    """
    Run the SingleBeamSplitterRBPostSelection_sweep_depth experiment.
    """
    from multimode_expts.sequential_experiment_classes import MM_DualRailRB
    experiment_class = MM_DualRailRB
    sweep_experiment_name = 'SingleBeamSplitterRBPostSelection_sweep_depth'
    class_for_exp = experiment_class(
        soccfg=soc, path=expt_path, prefix=sweep_experiment_name, config_file=config_path, exp_param_file=exp_param_file, 
        prev_data=prev_data
    )

    class_for_exp.yaml_cfg = AttrDict(deepcopy(config_thisrun))
    # Customize bs_para for the given manipulation and storage mode using ds_storage directly
    freq = ds_storage.get_freq('M' + str(man_mode_no) + '-S' + str(stor_mode_no))
    gain = ds_storage.get_gain('M' + str(man_mode_no) + '-S' + str(stor_mode_no))
    hpi_length = ds_storage.get_h_pi('M' + str(man_mode_no) + '-S' + str(stor_mode_no))
    bs_para = [freq, gain, hpi_length, config_thisrun.device.storage.ramp_sigma]
    print('Beam splitter parameters:', bs_para)
    # Optionally, set up experiment parameters here if needed, e.g.:
    # class_for_exp.loaded[sweep_experiment_name] = dict(...)
    class_for_exp.loaded[sweep_experiment_name] = dict(
        depth_list=[1, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 400, 500, 600, 700, 850, 1000, 1250, 1500, 2000],  # RB sequence depth list
        reps_list = [],
        qubits=[0],
        reps=0,  # doesn't matter

        single_shot_bef_expt=False,  # single shot before experiment
        singleshot_reps=2000,        # single shot measurement repetitions
        span=1000,                   # single shot plot span

        active_reset=False,          # for single shot post selection
        man_reset=True,              # for single shot post selection
        storage_reset=True,          # for single shot post selection
        threshold=None,              # for single shot post selection
        readout_per_round=4,         # for single shot post selection

        rb_active_reset=False,
        rb_man_reset=True,
        rb_storage_reset=True,
        rb_reps=1000,
        gates_per_wait=100000,       # ????
        parity_meas=True,            # If parity measurement is used, set to True; if False the reset arguments below should be false as well
        reset_qubit_after_parity=False,  # True # resetting via second parity str 
        reset_qubit_via_active_reset_after_first_meas=False,  # resetting via active reset after first parity str; the other reset should be false

        rounds=1,                    # always set to 1
        variations=10,                # number of different sequences
        rb_depth=10,                 # rb sequence depth
        IRB_gate_no=-1,              # IRB gate number, -1 means not using
        postselection_delay=2.0,     # in us, gap between two readout pulses
        bs_repeat=1,
        sync=False,
        setup=False,

        bs_para=bs_para,  # at 96  # beam splitter parameters [[frequency], [gain], [length (us)], [sigma]]
        prepulse=False,
        postpulse=False,
        f0g1_offset=0,               # offset phase in deg as a result of f0g1 prepulse/postpulse

        pre_sweep_pulse=[[None]],    # Gate based; prep f0g1 is done automatically ; RAM state prep is also automatic 
        ram_prepulse=[False, 6, [1], 1],  # [True/False, number of storage modes to be populated, [idx of modes to be skipped], variations]
        ram_prepulse_strs=None       # see SingleBeamSplitterRBPostSelection_sweep_depth_and_ram
    )
    

    class_for_exp.yaml_cfg.device.readout.relax_delay = [8000]  # Example, adjust as needed

    prefix, dir_path =eval('class_for_exp.run_sweep')(sweep_experiment_name=sweep_experiment_name)
    print('File saved at:', prefix)
    print('Directory path:', dir_path)

    # from fitting.fit_display_classes import MM_DualRailRBFitting
    # rb_analysis = MM_DualRailRBFitting(
    #     filename=None,
    #     file_prefix=prefix,
    #     config=yaml_cfg,
    #     expt_path=expt_path,
    #     title=f'M{man_mode_no}-S{stor_mode_no}',
    #     prev_data=prev_data,
    #     dir_path=dir_path
    # )
    # rb_analysis.show_rb(save_fig=True)
    # return rb_analysis


# %%
if expts_to_run['RB']:
    storage_rbs = {}
    for mode in m1_only('Storage randomized benchmarking'):
        for storage in stor_modes:
            print(f'Running storage RB postselection sweep depth for storage mode {storage}')
            storage_rbs[mode, storage] = do_single_beam_splitter_rb_postselection_sweep_depth(
                config_thisrun,
                ds_storage,
                expt_path,
                config_path,
                exp_param_file=exp_param_file,
                man_mode_no=mode,
                stor_mode_no=storage
            )


# %%
# from fitting.fit_display_classes import MM_DualRailRBFitting
# dir_no = np.arange(72, 80, 1 )
# for idx, dir_no in enumerate(dir_no):
#     prefix = f"SingleBeamSplitterRBPostSelection_sweep_depth"
#     dir_name = f"{str(dir_no).zfill(5)}_SingleBeamSplitterRBPostSelection_sweep_depth" 
#     dir_path = r"H:\\\\Shared drives\\\\SLab\\\\Multimode\\\\experiment\\\\250505_craqm\\\\data\\\\" + dir_name
#     filepath = dir_path
#     # Initialize RB analysis
#     rb_analysis = MM_DualRailRBFitting(
#         filename=None,
#         file_prefix=prefix,
#         config=yaml_cfg,
#         expt_path=expt_path,
#         title=f"M1-S{idx + 1} RB Analysis",
#         prev_data=prev_data,
#         dir_path=filepath
#     )
#     rb_analysis.show_rb(save_fig=True)



# %%
dir_no = 78
dir_name = f"{str(dir_no).zfill(5)}_SingleBeamSplitterRBPostSelection_sweep_depth" 
dir_path = r"H:\\\\Shared drives\\\\SLab\\\\Multimode\\\\experiment\\\\250505_craqm\\\\data\\\\" + dir_name
temp_data, attrs, filename = prev_data(dir_path, prefix='SingleBeamSplitterRBPostSelection_sweep_depth')

# %%
atrrs = AttrDict(attrs)
attrs['config']['device']['qubit']['f_ge']

# %%
# storage_rb

# %%
expt_test = Experiment(
            path=expt_path,
            prefix="yoyoyo",
            config_file=config_path,
        )

# %%
expt_test.data = {}
f = expt_test.save_data()

# %%
if expts_to_run['RB']:
    filename = expt_test.fname
    #create a directory with the filename but no .h5 extension
    import os
    directory = filename
    if directory.lower().endswith('.h5'):
        directory = directory[:-3]
    if not os.path.exists(directory):
        # Only create the directory if it is not the same as the filename (i.e., filename is not a directory itself)
        # Make sure the directory name does not have a .h5 extension
        os.makedirs(directory)

# %%
# directory
# filename_only = os.path.basename(expt_test.fname)
# filename_only

# %%
if expts_to_run['RB']:
    from fitting.fit_display_classes import MM_DualRailRBFitting
    rb_analysis = MM_DualRailRBFitting(file_prefix = "SingleBeamSplitterRBPostSelection_sweep_depth", 
                                    config=config_thisrun, expt_path=expt_path, title='M1_S1', 
                                    prev_data= prev_data)
    rb_analysis.show_rb()

# %%
if expts_to_run['RB']:
    temp_data, attrs, filename = prev_data(expt_path, '00036_SingleBeamSplitterRBPostSelection_sweep_depth.h5')

# %%
attrs['config'].keys()

# %%
temp_data.keys()

# %%
num_entries = len(temp_data['Idata'])
print(f"Number of entries in 'Idata': {num_entries}")

# %%
temp_data['sequences']

# %% [markdown]
# # Shock TLS Function

# %%
from multimode_expts.experiments.single_qubit.pulse_probe_f0g1_spectroscopy import PulseProbeF0g1SpectroscopyExperiment

def do_pulse_probe_f0g1_spectroscopy(config_thisrun, expt_path, config_path, 
                                     start=2007, step=0.02, expts=300, reps=100,
                                       rounds=1, length=1, gain=5000, pulse_type='gaussian',
                                         qubit_f=True, qubits=[0], prepulse=False, pre_sweep_pulse=None):
    """
    Run the PulseProbeF0g1SpectroscopyExperiment with specified parameters.
    """
    
    expt = PulseProbeF0g1SpectroscopyExperiment(
        soccfg=soc, path=expt_path, prefix='PulseProbeF0g1SpectroscopyExperiment', config_file=config_path
    )
    expt.cfg = AttrDict(deepcopy(config_thisrun))
    expt.cfg.expt = dict(
        start=start,
        step=step,
        expts=expts,
        reps=reps,
        rounds=rounds,
        length=length,
        gain=gain,
        pulse_type=pulse_type,
        qubit_f=qubit_f,
        qubits=qubits,
        prepulse=prepulse,
        pre_sweep_pulse=pre_sweep_pulse
    )
    expt.cfg.device.readout.relax_delay = [5]  # Wait time between experiments [us]
    expt.go(analyze=True, display=True, progress=True, save=True)
    return expt

# %%
# t2ramsey_ge_check = None
# while True: 
#     #close previous plots 
#     # import matplotlib.pyplot as plt
#     #each iteration is 5 minutes
#     # do_pulse_probe_f0g1_spectroscopy(
#     #     config_thisrun, expt_path, config_path,
#     #     start=3300, step=0.04, expts=10000, reps=100,
#     #     rounds=5, length=50, gain=30000, pulse_type='gaussian',
#     #     qubit_f=False, qubits=[0], prepulse=False, pre_sweep_pulse=None
#     # )

#     do_pulse_probe_ge(config_thisrun, start = 3300, 
#                       step = 0.04, expts = 10000, reps = 100, rounds = 5,
#                       length = 50, gain = 30000)
#     from IPython.display import clear_output
#     # from fitting.fit_display_classes import SidebandFitting
#     clear_output(wait=True)
#     plt.close('all')  # Close all existing figures
#     t2ramsey_ge_check = do_t2_ramsey_ge(config_thisrun, expt_path, config_path)
#     t2ramsey_ge_check.analyze()
#     t2ramsey_ge_check.display(title_str='T2_ge_TLS')


# %%
from experiments.MM_base import MMAveragerProgram
from multimode_expts.experiments.qsim.utils import ensure_list_in_cfg

class ShockTLSProgram(MMAveragerProgram):
    def initialize(self):
        self.MM_base_initialize()
        
        
    def body(self):
        self.reset_and_sync()

        self.setup_and_pulse(
            ch=0,
            style='const',
            freq=self.freq2reg(1918, gen_ch=0),
            length=50,
            phase=0,
            gain=30000,
        )
    def update(self):
        print('Updating frequency...')
        print('Current frequency (MHz):', self.reg2freq(self.r_freq2, gen_ch=0))
        print('Frequency step (MHz):', self.f_step)
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index


class ShockTLSExperiment(Experiment):
    def acquire(self, progress):
        ensure_list_in_cfg(self.cfg)
        
        self.prog = ShockTLSProgram(soccfg=self.soccfg, cfg=self.cfg)
        self.prog.acquire(self.im[self.cfg.aliases.soc],
                         threshold=None,
                         load_pulses=True,
                         progress=progress,
                         debug=False)
    


# %% [markdown]
# ### ShockTLS !!!

# %%
shockTLS_bool = False # Set to FALSE to avoid running!!!!!!
if shockTLS_bool:    
    scktls = ShockTLSExperiment(soccfg=soc, path=expt_path, prefix='shockTLS',
                            config_file=config_path)
    scktls.cfg = AttrDict(deepcopy(config_thisrun))

    scktls.cfg.expt = dict(expts=2000, reps=2000, rounds=100000, qubit=[0])

    scktls.cfg.device.readout.relax_delay = [50] # Wait time between experiments [us]
    for kk in range(50):
        scktls.go(analyze=False, display=False, progress=True, save=False)


# %% [markdown]
# # Update Config and Dataset

# %%
def convert_attrdict_to_dict(attrdict):
    """
    Recursively converts an AttrDict or a nested dictionary into a standard Python dictionary.
    Converts np.float64 values to standard Python float.

    Args:
        attrdict (AttrDict or dict): The input AttrDict or dictionary to be converted.

    Returns:
        dict: A standard Python dictionary with all nested AttrDicts or dictionaries converted.
    """
    if isinstance(attrdict, AttrDict):
        return {key: convert_attrdict_to_dict(value) for key, value in attrdict.items()}
    elif isinstance(attrdict, dict):
        return {key: convert_attrdict_to_dict(value) for key, value in attrdict.items()}
    elif isinstance(attrdict, np.float64):
        return float(attrdict)
    else:
        return attrdict


def convert_numbers_to_float(data):
    """
    Recursively converts all numbers in a dictionary to float.

    Args:
        data (dict): The input dictionary.

    Returns:
        dict: A dictionary with all numbers converted to float.
    """
    if isinstance(data, dict):
        return {key: convert_numbers_to_float(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numbers_to_float(item) for item in data]
    elif isinstance(data, float):
        return float(data)
    elif isinstance(data, int):
        return int(data)
    else:
        return data


# %% [markdown]
# ## Dataset comparison for man1_storage_swap_dataset

# %%
#compare the two datsets
differences = ds_storage.compare_with(ds)
# Filter out differences related to 'last_update' column
filtered_differences = [diff for diff in differences if diff['column'] != 'last_update']
# Update labels and calculate differences
for diff in filtered_differences:
    if np.isnan(diff['self_value']) and np.isnan(diff['other_value']):
        diff['difference'] = None
        continue
    diff['label'] = f"New value: {diff['self_value']}, Old value: {diff['other_value']}"
    diff['difference'] = diff['self_value'] - diff['other_value'] if isinstance(diff['self_value'], (int, float)) and isinstance(diff['other_value'], (int, float)) else None

# Group differences by storage name
grouped_differences = {}
for diff in filtered_differences:
    stor_name = diff['stor_name']
    if stor_name not in grouped_differences:
        grouped_differences[stor_name] = []
    grouped_differences[stor_name].append(diff)

# Print grouped differences
for stor_name, diffs in grouped_differences.items():
    print(f"In {stor_name}:")
    for diff in diffs:
        if np.isnan(diff['self_value']) and np.isnan(diff['other_value']):
            continue
        print(f"  Column '{diff['column']}': new_value = {diff['self_value']}, old_value = {diff['other_value']}, difference = {diff['difference']}")
    grouped_differences[stor_name].append(diff)


# %%
ds_storage.df["gain (DAC units)"] = ds_storage.df["gain (DAC units)"].astype(int)

# %%
ds_storage.df

# %% [markdown]
# ### set TRUE to update dataset for man1_storage_swap_dataset

# %%
# stor new datset at old file path
updateDataset = True # Set to FALSE after this run!!!!!!
if updateDataset:
    # store old dataset  into autocalib_path with date-time in the name
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    old_dataset_path = os.path.join(autocalib_path, f'old_dataset_{current_time}.h5')
    ds_storage.save_to_file(old_dataset_path)
    print('Old dataset saved to:', old_dataset_path)

    # save new dataset to current_dataset path
    ds_storage.save_to_file(ds.filename)
    # ds = copy(ds_storage)
    print('New dataset saved to old file path!')

# %% [markdown]
# ## Config comparison for hardware_config_202505

# %%
from pprint import pprint
print("Comparing configurations:")

def recursive_compare(d1, d2, path=""):
    for key in d1.keys():
        current_path = f"{path}.{key}" if path else key
        if key not in d2:
            print(f"Key '{current_path}' is missing in config2.")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            recursive_compare(d1[key], d2[key], current_path)
        elif d1[key] != d2[key]:
            print(f"Key '{current_path}' differs:")
            if isinstance(d1[key], list) and len(d1[key]) == 1:
                print(f"  Old value (config1): {d1[key][0]}")
                print(f"  New value (config2): {d2[key][0]}")
            else:
                print(f"  Old value (config1): {d1[key]}")
                print(f"  New value (config2): {d2[key]}")
    for key in d2.keys():
        current_path = f"{path}.{key}" if path else key
        if key not in d1:
            print(f"Key '{current_path}' is missing in config1.")

recursive_compare(yaml_cfg, config_thisrun)


# %%
# Copy contents from config_thisrun to yaml_cfg except for the storage_man_file
def update_yaml_config(yaml_cfg, config_thisrun):
    """
    Update the yaml_cfg with values from config_thisrun, excluding the storage_man_file.
    """
    updated_config = deepcopy(config_thisrun)
    updated_config.device.storage.storage_man_file = yaml_cfg.device.storage.storage_man_file
    return updated_config

def save_configurations(yaml_cfg, config_thisrun, autocalib_path, config_path):
    """
    Save the old and updated configurations to their respective files.
    """
    # Save old configuration
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    old_config_path = os.path.join(autocalib_path, f'old_config_{current_time}.yaml')
    old_config = convert_numbers_to_float(convert_attrdict_to_dict(yaml_cfg) )
    with open(old_config_path, 'w') as cfg_file:
        yaml.dump(old_config, cfg_file, default_flow_style=False, indent=4, width=80, canonical=False,
                  explicit_start=True, explicit_end=False, sort_keys=False, line_break=True)

    # Save updated configuration
    updated_config = convert_numbers_to_float(convert_attrdict_to_dict(update_yaml_config(yaml_cfg, config_thisrun)))
    with open(config_path, 'w') as f:
        yaml.dump(updated_config, f, default_flow_style=False, indent=4, width=80, canonical=False,
                  explicit_start=True, explicit_end=False, sort_keys=False, line_break=True)


# %% [markdown]
# ### set TRUE to update hardware_config!!

# %%
# Main logic
updateConfig_bool = True # Set to FALSE after this run!!!!!!
updated_config = update_yaml_config(yaml_cfg, config_thisrun)
if updateConfig_bool:
    save_configurations(yaml_cfg, updated_config, autocalib_path, config_path)
    yaml_cfg = updated_config
    updateConfig_bool = False
    print("Configuration updated and saved, excluding storage_man_file. \n!!!!Please set updateConfig to False after this run!!!!!!.")


# %% [markdown]
# # Loopbackprogram

# %%
class LoopbackProgram(AveragerProgram):
    def initialize(self):
        cfg=self.cfg   
        res_ch = cfg["res_ch"]

        # set the nyquist zone
        self.declare_gen(ch=cfg["res_ch"], nqz=1)
        
        # configure the readout lengths and downconversion frequencies (ensuring it is an available DAC frequency)
        for ch in cfg["ro_chs"]:
            self.declare_readout(ch=ch, length=self.cfg["readout_length"],
                                 freq=self.cfg["pulse_freq"], gen_ch=cfg["res_ch"])

        # convert frequency to DAC frequency (ensuring it is an available ADC frequency)
        freq = self.freq2reg(cfg["pulse_freq"],gen_ch=res_ch, ro_ch=cfg["ro_chs"][0])
        phase = self.deg2reg(cfg["res_phase"], gen_ch=res_ch)
        gain = cfg["pulse_gain"]
        self.default_pulse_registers(ch=res_ch, freq=freq, phase=phase, gain=gain)

        style=self.cfg["pulse_style"]

        if style in ["flat_top","arb"]:
            sigma = cfg["sigma"]
            self.add_gauss(ch=res_ch, name="measure", sigma=sigma, length=sigma*5)
            
        if style == "const":
            self.set_pulse_registers(ch=res_ch, style=style, length=cfg["length"])
        elif style == "flat_top":
            # The first half of the waveform ramps up the pulse, the second half ramps down the pulse
            self.set_pulse_registers(ch=res_ch, style=style, waveform="measure", length=cfg["length"])
        elif style == "arb":
            self.set_pulse_registers(ch=res_ch, style=style, waveform="measure")
        
        self.synci(200)  # give processor some time to configure pulses
    
    def body(self):
        # fire the pulse
        # trigger all declared ADCs
        # pulse PMOD0_0 for a scope trigger
        # pause the tProc until readout is done
        # increment the time counter to give some time before the next measurement
        # (the syncdelay also lets the tProc get back ahead of the clock)
        self.measure(pulse_ch=self.cfg["res_ch"], 
                     adcs=self.ro_chs,
                     pins=[0], 
                     adc_trig_offset=self.cfg["adc_trig_offset"],
                     wait=True,
                     syncdelay=self.us2cycles(self.cfg["relax_delay"]))


# %%
GEN_CH = 5
RO_CH = 0
config={"res_ch":GEN_CH, # --Fixed
        "ro_chs":[RO_CH], # --Fixed
        "reps":1, # --Fixed
        "relax_delay":1.0, # --us
        "res_phase":0, # --degrees
        "pulse_style": "const", # --Fixed
        
        "length":20, # [Clock ticks]
        # Try varying length from 10-100 clock ticks
        
        "readout_length":100, # [Clock ticks]
        # Try varying readout_length from 50-1000 clock ticks

        "pulse_gain":3000, # [DAC units]
        # Try varying pulse_gain from 500 to 30000 DAC units

        "pulse_freq": 750, # [MHz]
        # In this program the signal is up and downconverted digitally so you won't see any frequency
        # components in the I/Q traces below. But since the signal gain depends on frequency, 
        # if you lower pulse_freq you will see an increased gain.

        "adc_trig_offset": 100, # [Clock ticks]
        # Try varying adc_trig_offset from 100 to 220 clock ticks

        "soft_avgs":100
        # Try varying soft_avgs from 1 to 200 averages

       }

###################
# Try it yourself !
###################

prog =LoopbackProgram(soc, config)
iq_list = prog.acquire_decimated( soc = im[config_thisrun.aliases.soc], progress=True)

# %%
iq_list

# %%
# Plot results.
plt.figure(1)
for ii, iq in enumerate(iq_list):
    plt.plot(iq[0], label="I value, ADC %d"%(config['ro_chs'][ii]))
    plt.plot(iq[1], label="Q value, ADC %d"%(config['ro_chs'][ii]))
    plt.plot(np.abs(iq[0]+1j*iq[1]), label="mag, ADC %d"%(config['ro_chs'][ii]))
plt.ylabel("a.u.")
plt.xlabel("Clock ticks")
plt.title("Averages = " + str(config["soft_avgs"]))
plt.legend();

# %% [markdown]
# ## Example: Initialize and run pulse-probe f0g1 DC flux sweep
#

# %% [markdown]
# ## f0g1 spectroscopy vs flux (2D)
# This section wraps the new `PulseProbeF0g1SpectroscopyFluxSweepExperiment` into a convenience function using the same config approach as the 1D f0g1 example.

# %%
# Helper: do_f0g1_versus_flux_spectroscopy()
from pathlib import Path
from typing import Optional
from copy import deepcopy
import yaml
from slab import AttrDict
from experiments.single_qubit.pulse_probe_f0g1_spectroscopy import (
    PulseProbeF0g1SpectroscopyFluxSweepExperiment,
)

def do_f0g1_versus_flux_spectroscopy(
    *,
    config_thisrun,
    ds_storage,
    man_mode_no: int = 1,
    prefix: str = "PulseProbeF0g1SpectroscopyFluxSweep",
    # DC current sweep
    curr_start: float = -1,
    curr_step: float = 0.001,
    curr_expts: int = 21,
    yokogawa_address: str = "192.168.137.148",
    sweeprate: float = 2,
    safety_limit: float = 10,
    # frequency sweep overrides (optional, follow the 1D example)
    freq_start: Optional[float] = None,
    freq_step: Optional[float] = None,
    freq_expts: Optional[int] = None,
    # book-keeping
    save: bool = True,
    progress: bool = False,
    qubit_f: bool = True,
    qubits: list = [0],
    prepulse: bool = False,
    pre_sweep_pulse: Optional[list] = None,
):
    """
    Configure and run the 2D f0g1 spectroscopy vs flux using the same conventions
    as the 1D PulseProbeF0g1SpectroscopyExperiment above. Returns the saved H5 filename.
    """
    # Resolve frequency defaults from dataset if not provided
    if freq_start is None:
        try:
            base_f = ds_storage.get_freq('M' + str(man_mode_no))
        except Exception:
            base_f = ds_storage.get_freq('M1')
        if base_f is None:
            raise ValueError("Could not resolve base frequency from ds_storage.")
        freq_start = base_f - 5  # MHz offset similar to 1D convention
    if freq_step is None:
        freq_step = 1.0
    if freq_expts is None:
        freq_expts = 51

    # Build optional prepulse if requested and not provided
    print(pre_sweep_pulse, prepulse)
    if prepulse and pre_sweep_pulse is None:
        # mm_base_calib is prepared earlier in the notebook alongside config_thisrun/ds_storage
        mm_base_dummy = MM_dual_rail_base(config_thisrun, soccfg=soc)
        pre = mm_base_dummy.prep_man_photon(man_no=man_mode_no, photon_no=0)
        pre.append(['multiphoton', 'g0-e0', 'pi', 0])
        pre.append(['multiphoton', 'e0-f0', 'pi', 0])
        pre_sweep_pulse = mm_base_dummy.get_prepulse_creator(pre, config_thisrun).pulse.tolist()
        print('prep pulse', pre_sweep_pulse)


    # Instantiate the 2D experiment. Use global paths/config used elsewhere in the notebook
    exp = PulseProbeF0g1SpectroscopyFluxSweepExperiment(
        soccfg=soc if 'soc' in globals() else None,
        path=str(expt_path),
        prefix=prefix,
        config_file=str(config_file),
    )

    # Attach this-run configuration and expt overrides
    exp.cfg = AttrDict(deepcopy(config_thisrun))

    ex_overrides = dict(
        # Frequency sweep (inner axis)
        start=freq_start,
        step=freq_step,
        expts=freq_expts,
        reps=250,
        rounds=1,
        length=1,
        gain=5000,
        pulse_type='gaussian',
        qubit_f=bool(qubit_f),
        qubits=list(qubits),
        prepulse=bool(prepulse),
        pre_sweep_pulse=(pre_sweep_pulse if prepulse else []),
        # Current sweep (outer axis)
        curr_start=curr_start,
        curr_step=curr_step,
        curr_expts=curr_expts,
        yokogawa_address=yokogawa_address,
        sweeprate=sweeprate,
        safety_limit=safety_limit,
    )

    # Ensure an expt section exists and populate overrides
    if not hasattr(exp.cfg, 'expt') or exp.cfg.expt is None:
        exp.cfg.expt = AttrDict()
    for k, v in ex_overrides.items():
        try:
            exp.cfg.expt[k] = v
        except Exception:
            setattr(exp.cfg.expt, k, v)

    exp.go(analyze=False, display=True, progress=progress, save=save)
    return exp

# %%
# Call example: run the 2D f0g1 vs flux experiment

f_start = 1970
f_stop = 2025
f_expt = 500
freq_step = (f_stop - f_start)/f_expt

c_start = 0.30
c_stop = 1
# c_stop = 1 
c_expt = 10
c_step = np.abs(c_stop - c_start)/c_expt


prefix_2d = "PulseProbeF0g1SpectroscopyFluxSweep"
fname_2d = do_f0g1_versus_flux_spectroscopy(
    config_thisrun=config_thisrun,
    ds_storage=ds_storage,
    man_mode_no=1,
    prefix=prefix_2d,
    # DC current sweep
    curr_start=c_start,  # -10 mA
    curr_step=c_step,    # 1 mA steps
    curr_expts=c_expt,      # total points
    # Frequency sweep (mirror the 1D example defaults; override if needed)
    # freq_start=..., freq_step=..., freq_expts=...,
    save=True,
    progress=True,
    qubit_f=True,
    qubits=[0],
    prepulse=True,
    freq_start=f_start,
    freq_step = freq_step,
    freq_expts=f_expt,
)
# print("Saved:", fname_2d)

# %% [markdown]
# # JPA Calibration

# %%
def do_jpa_current_sweep( config_thisrun,
    expt_path,
    config_path,
    jpa_current_start=-8,
    jpa_current_step=0.1,
    jpa_current_stop = -2,
    qubits=[0],
    reps=5000,
    check_f=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    qubit=0,
    pulse_manipulate=False,
    cavity_freq=4984.373226159381,
    cavity_gain=800,
    cavity_length=2,
    prepulse=False,
    pre_sweep_pulse=None,
    gate_based=True,
    relax_delay=2500
):
    """Run the single shot experiment with configurable parameters."""
    from multimode_expts.sequential_experiment_classes import histogram_sweep_class
    experiment_class = histogram_sweep_class
    sweep_experiment_name = 'histogram_jpa_current_sweep'
    class_for_exp = experiment_class(soccfg=soc, path=expt_path, prefix=sweep_experiment_name, config_file=config_path, exp_param_file=exp_param_file)

    class_for_exp.yaml_cfg = AttrDict(deepcopy(config_thisrun))
    
    class_for_exp.loaded[sweep_experiment_name] =  {
        'qubits': qubits,
        'reps': reps,
        'check_f': check_f,
        'active_reset': active_reset,
        'man_reset': man_reset,
        'storage_reset': storage_reset,
        'qubit': qubit,
        'pulse_manipulate': pulse_manipulate,
        'cavity_freq': cavity_freq,
        'cavity_gain': cavity_gain,
        'cavity_length': cavity_length,
        'prepulse': prepulse,
        'pre_sweep_pulse': pre_sweep_pulse,
        'gate_based': gate_based,
        'jpa_current_start': jpa_current_start,
        'jpa_current_step': jpa_current_step,
        'jpa_current_stop': jpa_current_stop,
        'relax_delay': relax_delay
    }

   
    class_for_exp.yaml_cfg.device.readout.relax_delay = [relax_delay]  # Wait time between experiments [us]

    return eval('class_for_exp.run_sweep')( sweep_experiment_name = sweep_experiment_name)

# %%
do_jpa_current_sweep(config_thisrun, expt_path,
                                   config_path, reps=1000,
                                   jpa_current_start=-5.0,
                                   jpa_current_stop=-4.0,
                                   jpa_current_step=0.05,
                                    #  jpa_current_start=0., jpa_current_step=0.2,
                                    #    jpa_current_stop=10., 
                                      #  active_reset=True,
                                         relax_delay=5)

# %%
from slab.instruments import YokogawaGS200
dcflux = YokogawaGS200(address="192.168.137.149")
dcflux.set_output(True)
dcflux.set_mode('current')

jpa_current = -1.875 # mA
jpa_current = 0 # mA
current = jpa_current * 1e-3  # Convert from mA to A
dcflux.set_current(current)

print(1e3 * dcflux.get_current(), 'mA')


# %% [markdown]
# #### Sweep both gain and current

# %%
def do_jpa_current_gain_sweep( config_thisrun,
    expt_path,
    config_path,
    jpa_current_start=3.7,
    jpa_current_step=0.005,
    jpa_current_stop = 3.8,
    jpa_gain_start=-15,
    jpa_gain_step=0.5,
    jpa_gain_stop = -11,
    qubits=[0],
    reps=1000,
    check_f=False,
    active_reset=False,
    man_reset=False,
    storage_reset=False,
    qubit=0,
    pulse_manipulate=False,
    cavity_freq=4984.373226159381,
    cavity_gain=800,
    cavity_length=2,
    prepulse=False,
    pre_sweep_pulse=None,
    gate_based=True,
    relax_delay=2500
):
    """Run the single shot experiment with configurable parameters."""
    from multimode_expts.sequential_experiment_classes import histogram_sweep_class
    experiment_class = histogram_sweep_class
    sweep_experiment_name = 'histogram_jpa_gain_current_sweep'
    class_for_exp = experiment_class(soccfg=soc, path=expt_path, prefix=sweep_experiment_name, config_file=config_path, exp_param_file=exp_param_file)

    class_for_exp.yaml_cfg = AttrDict(deepcopy(config_thisrun))
    
    class_for_exp.loaded[sweep_experiment_name] =  {
        'qubits': qubits,
        'reps': reps,
        'check_f': check_f,
        'active_reset': active_reset,
        'man_reset': man_reset,
        'storage_reset': storage_reset,
        'qubit': qubit,
        'pulse_manipulate': pulse_manipulate,
        'cavity_freq': cavity_freq,
        'cavity_gain': cavity_gain,
        'cavity_length': cavity_length,
        'prepulse': prepulse,
        'pre_sweep_pulse': pre_sweep_pulse,
        'gate_based': gate_based,
        'jpa_current_start': jpa_current_start,
        'jpa_current_step': jpa_current_step,
        'jpa_current_stop': jpa_current_stop,
        'jpa_gain_start': jpa_gain_start,
        'jpa_gain_step': jpa_gain_step,
        'jpa_gain_stop': jpa_gain_stop,
    }

    print('relax delay set to:', relax_delay)
    class_for_exp.yaml_cfg.device.readout.relax_delay = [relax_delay]  # Wait time between experiments [us]

    return eval('class_for_exp.run_sweep')( sweep_experiment_name = sweep_experiment_name)


# %%
sweep_func = do_jpa_current_gain_sweep(config_thisrun, expt_path,
                                        config_path, reps=1000, 
                                        jpa_current_start=-5.5,
                                        jpa_current_step=-0.1,
                                        jpa_current_stop=-4.5,
                                        jpa_gain_start=-15,
                                        jpa_gain_step=0.5,
                                        jpa_gain_stop=-5,
                                        # active_reset=True, relax_delay=5
                                        )

# %% [markdown]
# # Save Config

# %%
station.update_all_station_snapshots()

# %% [markdown]
# # Change Current (Be Careful)

# %%
RESET_CURRENT = False
CURRENT_SAFETY_THRESHOLD = 1e-3

current_to_set = 0.0005
if RESET_CURRENT:
    print(f"CURRENT THEN: {station.hardware_cfg.hw.yoko_coupler.current}")
    if not current_to_set > CURRENT_SAFETY_THRESHOLD:
        station.hardware_cfg.hw.yoko_coupler.current = current_to_set
    else:
        print("CURRENT TO HIGH")
        
print(f"CURRENT NOW: {station.hardware_cfg.hw.yoko_coupler.current}")

# %%
