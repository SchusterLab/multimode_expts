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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Prepare

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import experiments as meas

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
config_dict = {'hardware_config': 'CFG-HW-20261120-00001',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20261119-00011',
 'floquet_storage_swap': 'CFG-FL-20261119-00022'}

# Initialize station to retrieve soc and configs
station = MultimodeStation(
    user = 'Jonginn',
    experiment_name = "261113_calibration_after_recooldown_for_overtheweekend_batch",
    project = "AutoCalibrate",
    
    storage_man_file = config_dict['man1_storage_swap'],
    hardware_config= config_dict['hardware_config'],
    floquet_file=config_dict['floquet_storage_swap'],

    log_measurements=True,
)

def man_photon_pulses(station, man_mode_no):
    from experiments.MM_dual_rail_base import MM_dual_rail_base

    base = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
    if man_mode_no == 1:
        one_photon_prep_sequence = base.prep_man_photon(man_mode_no)
    else:
        one_photon_prep_sequence = [['qubit', 'ge', 'pi', 0], ['qubit', 'ef', 'pi', 0], ['man', f'M{man_mode_no}', 'pi', 0]]
    return (
        base.get_prepulse_creator(one_photon_prep_sequence).pulse.tolist(),
        base.get_prepulse_creator(one_photon_prep_sequence[-1:-3:-1]).pulse.tolist(),
    )

# %%
STORAGE_SIDEBAND_FREQ_SPAN= [0.2 ,  0.4,  0.4,  0.3,  0.3,  0.3, 0.2] #[0.2 ,  0.2,  0.2,  0.1,  0.15,  0.3, 0.2]
STORAGE_SIDEBAND_GAIN_SPAN= [None,  0.2,  0.3,  0.1,  0.1, 0.15, 0.1] #[None, None,  0.3,  0.3,  0.1, 0.15, 0.1] # should be less than 1.0
STORAGE_SIDEBAND_MAX_GAIN=  [None, 11000, None, None,None,None,None,]

active_reset_default_dict = {
    "reset_dump_mode": 2,
    "dump_reset_iter_num": 1,
}

floquet_default_dict = {
    # For phase accumulation
    "include_10cycles_buffer": True,
    "include_10cycles_buffer_in_pi_half": True,
    "floquet_hardware_loop": True,
    "scramble_sync_cycles": 2,
    "palindrome_scramble": False,
}

calibration_cfg_default_dict = {
    "fine_phase_calibration_expts": 51,
}

USE_QUEUE = True

expts_to_run = {# readout 
                'res_spec': False, # Readout spectroscopy
                'single_shot': True, 
                
                
                # qubit ge 
                'pulse_probe_ge': False,
                't2_ge': True, 
                'amplitude_ge': True,
                't1_ge': False,
                'ge_freq_error_amplification': False,
                'ge_gain_error_amplification': False,
                
                
                # qubit ef
                'pulse_probe_ef': False,
                't2_ef': True,
                'amplitude_ef': True,
                't1_ef': False,
                'ef_freq_error_amplification': False,
                'ef_gain_error_amplification': False,
                
                # manipulate 
                'man_modes': [1], # [2] for M2, [1, 2] for both
                'pulse_probe_f0g1': False,
                'man_freq_coarse_chevron': False,
                'man_freq_fine_chevron': False,
                'man_error_amplification': True,
                'length_rabi_sweep':True,
                'length_rabi_expts': 150,  # Time points in the 4-pi manipulate sweep.
                'length_rabi':False, # this will run automatically if the length_rabi_sweep is set to True
                'chi_ge': False, 
                'chi_ef': False,
                'RB': False,
                'parity_delay_slow': True,
                'parity_delay_fast': False,
                't1_man': False,
                't2_man': False,
                

                #storage
                'stor_modes': [4, 5, 6, 7], # [1,2, .., 7] if also want to run  all modes 
                # 'stor_modes': [3, 4, 5], # [1,2, .., 7] if also want to run  all modes 
                'stor_spectroscopy': False,

                'dump_spectroscopy': False, #dump should be reivewed
                'dump_chevron': False, #dump should be reivewed
                'dump_modes': [1, 2],
                'dump_for_reset': active_reset_default_dict["reset_dump_mode"],

                'sideband_freq_sweep': False,
                'sideband_chevron': False,
                'sideband_erroramp': True,
                # 'storage_t1': True


                #Floquet
                'floquet_stor_modes': [4, 5, 6, 7],
                'floquet_chevron_coarse': False,
                'floquet_chevron_fine': False,
                'floquet_error_amplificaton_coarse': False,
                'floquet_error_amplificaton_fine': True,
                'floquet_phase_accumulation_matrix_coarse': False,
                'floquet_phase_accumulation_matrix_fine': True,
                'floquet_phase_accumulation_matrix_fine_bw': 4,
                
                
                
                #Multiphoton
                'multiphoton_fetch_from_singlequbit': False,
                # Final photon numbers (M1, M2). With man_modes=[1, 2], e.g.
                # [(1, 3), (2, 3), (3, 2)] calibrates M1 first, then each M2 ladder.
                # With [2], M1 preparation uses its existing calibrations.
                # None uses the transition list below; [] skips the multiphoton scans.
                # A rejected fit leaves the saved pulse in place for subsequent scans.
                'multiphoton_target_states': None,
                'multiphoton_transitions': [
                                            'g1-e1', 
                                            'e1-f1', 
                                            'f1-g2',
                                            'g2-e2',
                                            'e2-f2',
                                            'f2-g3'],
                'sideband_pulse_probe_spectroscopy_multiphoton' : False,
                'sideband_quick_chevron_and_length_rabi_multiphoton': False,
                }

man_modes = tuple(expts_to_run['man_modes'])
if any(man_mode_no not in (1, 2) for man_mode_no in man_modes):
    raise ValueError("man_modes supports M1 and M2")
man_storage_mode_pairs = [
    (man_mode_no, stor_mode_no) for man_mode_no in man_modes for stor_mode_no in expts_to_run['stor_modes']
]

# %% [markdown]
# # Qubit characterization

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
        active_reset=False,  # Calibrate the readout before using its threshold for reset.
        relax_delay=2000,


        reps=5000,
        # active_reset=True,
        # relax_delay=200,
    )
    # ss.display(station)

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
    )
    gespec.display()

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


# %% [markdown]
# ### Error Amplification

# %%
# Error Amplification - Runner Pattern with CharacterizationRunner
# =====================================
ge_error_amp_defaults = AttrDict(dict(
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


def ge_error_amp_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    expt_cfg.qubits = [expt_cfg.qubit]
    
    return expt_cfg

def ge_error_amp_postproc(station, expt, state_fin='g'):
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
expts = 50 #50
band = 1 #2
freq_start = station.hardware_cfg.device.qubit['f_ge'][0] - band
step = 2 * band / expts
print(f'Frequency start: {freq_start}, Step: {step}')


if expts_to_run['ge_freq_error_amplification']:
    ge_error_amp_runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.ErrorAmplificationExperiment,
        default_expt_cfg=ge_error_amp_defaults,
        preprocessor=ge_error_amp_preproc,
        postprocessor=ge_error_amp_postproc,
        job_client=client,
        use_queue=True,
    )
    bare_ge_frequency_error_amp = ge_error_amp_runner.execute(
        parameter_to_test='frequency',
        step=2 * band / expts, 
        start=freq_start,
        expts=expts,
        reps=50,
        pulse_type=['qubit', 'ge', 'pi', 0],
        # n_pulses=20,
    )


# %%
band = 500
expts = 25
gain_start = station.hardware_cfg.device.qubit.pulses.pi_ge.gain[0] - band
step = 2 * band / expts
print(f'Gain start: {gain_start}, Step: {step}')

if expts_to_run['ge_gain_error_amplification']:
    ge_error_amp_runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.ErrorAmplificationExperiment,
        default_expt_cfg=ge_error_amp_defaults,
        preprocessor=ge_error_amp_preproc,
        postprocessor=ge_error_amp_postproc,
        job_client=client,
        use_queue=True,
    )
    bare_ge_gain_error_amp = ge_error_amp_runner.execute(
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
eframsey_defaults = AttrDict(dict(
    ramsey_freq=3, start=0.01, step=0.5, expts=101, reps=200, rounds=1,
    if_ef=True, ef_init=True, qubits=[0], relax_delay=200,
    user_defined_freq=[False, 3568.2038290468167, 5304, 0.035],
    f0g1_cavity=0, normalize=False, active_reset=False, man_reset=False, storage_reset=False,
    prepulse=None, postpulse=None, pre_sweep_pulse=None, post_sweep_pulse=None,
    pre_active_reset_pulse=False, gate_based=False, advance_phase=0, echoes=[False, 0],
))

def eframsey_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    expt_cfg.checkEF = bool(expt_cfg.if_ef)
    expt_cfg.qubit_ge_init = bool(expt_cfg.if_ef and expt_cfg.ef_init)
    expt_cfg.qubit_ge_after = expt_cfg.qubit_ge_init
    if expt_cfg.prepulse is None or expt_cfg.pre_sweep_pulse is None:
        expt_cfg.prepulse = expt_cfg.pre_sweep_pulse is not None
    if expt_cfg.postpulse is None or expt_cfg.post_sweep_pulse is None:
        expt_cfg.postpulse = expt_cfg.post_sweep_pulse is not None
    return expt_cfg

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
    preprocessor=eframsey_preproc,
    postprocessor=eframsey_postproc,
    job_client=client,
)

if expts_to_run['t2_ef']:
    t2ramsey_ef = eframsey_runner.execute(
        ramsey_freq=0.2,
        step = 0.2,
        relax_delay=2500,
        active_reset=True,
    )
    t2ramsey_ef.display()

# %% [markdown]
# ### Amplitude Rabi

# %% [markdown]
# We should probably use a cosine fit with fixed phase=0 instead of decaying sine with varying phase?

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
efamprabi_defaults = AttrDict(dict(
    start=0, step=40, expts=150, reps=200, rounds=1, sigma_test=None,
    qubit=0, qubits=[0], if_ef=True, checkEF=True, pulse_ge_init=True, pulse_ge_after=True,
    pulse_type='gauss', drag_beta=0.0, flat_length=0, checkZZ=False,
    normalize=False, single_shot=False, singleshot_reps=10000, span=50,
    user_defined_freq=[False, 3568.203829046816], prepulse=False, postpulse=False,
))

def efamprabi_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    qubit_pulses = station.hardware_cfg.device.qubit.pulses
    test_pulse = qubit_pulses.pi_ef if expt_cfg.if_ef else qubit_pulses.pi_ge
    if expt_cfg.sigma_test is None:
        expt_cfg.sigma_test = test_pulse.sigma[0]
    if expt_cfg.step is None:
        expt_cfg.step = int(test_pulse.gain[0] / (expt_cfg.expts - 1))
    expt_cfg.checkEF = bool(expt_cfg.if_ef)
    expt_cfg.pulse_ge_init = expt_cfg.checkEF
    expt_cfg.pulse_ge_after = expt_cfg.checkEF
    return expt_cfg

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
    preprocessor=efamprabi_preproc,
    postprocessor=efamprabi_postproc,
    job_client=client,
)

if expts_to_run['amplitude_ef']:
    amprabi_ef = efamprabi_runner.execute(
        relax_delay=2500,
        expts=500,
    )
    amprabi_ef.display()

# %%
# Verify the EF frequency after the gain update.
if expts_to_run['amplitude_ef']:
    ef_verification_ramsey_runner = CharacterizationRunner(
        station=station, ExptClass=meas.RamseyExperiment, default_expt_cfg=eframsey_defaults,
        preprocessor=eframsey_preproc, postprocessor=eframsey_postproc, job_client=client,
    )
    t2_ramsey_ef_after_amp = ef_verification_ramsey_runner.execute(
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
ef_error_amp_defaults = AttrDict(dict(
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


def ef_error_amp_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    expt_cfg.qubits = [expt_cfg.qubit]
    
    return expt_cfg

def ef_error_amp_postproc(station, expt, state_fin='g'):
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
expts = 50 #50
band = 1 #2
freq_start = station.hardware_cfg.device.qubit['f_ef'][0] - band
step = 2 * band / expts
print(f'Frequency start: {freq_start}, Step: {step}')


if expts_to_run['ef_freq_error_amplification']:
    ef_error_amp_runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.ErrorAmplificationExperiment,
        default_expt_cfg=ef_error_amp_defaults,
        preprocessor=ef_error_amp_preproc,
        postprocessor=ef_error_amp_postproc,
        job_client=client,
        use_queue=True,
    )
    bare_ef_frequency_error_amp = ef_error_amp_runner.execute(
        parameter_to_test='frequency',
        step=2 * band / expts, 
        start=freq_start,
        expts=expts,
        reps=50,
        pulse_type=['qubit', 'ef', 'pi', 0],
        # n_pulses=20,
    )

# %%
band = 500
expts = 25
gain_start = station.hardware_cfg.device.qubit.pulses.pi_ef.gain[0] - band
step = 2 * band / expts
print(f'Gain start: {gain_start}, Step: {step}')

if expts_to_run['ef_gain_error_amplification']:
    ef_error_amp_runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.ErrorAmplificationExperiment,
        default_expt_cfg=ef_error_amp_defaults,
        preprocessor=ef_error_amp_preproc,
        postprocessor=ef_error_amp_postproc,
        job_client=client,
        use_queue=True,
    )
    bare_ef_gain_error_amp = ef_error_amp_runner.execute(
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

# %% [markdown]
# # Manipulate

# %%
from experiments.MM_base import multiphoton_slot, multiphoton_table
from experiments.MM_dual_rail_base import MM_dual_rail_base


def f0g1_values(station, man_mode_no):
    if man_mode_no == 1:
        pulse = station.hardware_cfg.device.multiphoton.pi['fn-gn+1']
        return pulse.frequency[0], pulse.gain[0], pulse.length[0]
    manipulate_row_name = f'M{man_mode_no}'
    storage_dataset = station.ds_storage
    return storage_dataset.get_freq(manipulate_row_name), storage_dataset.get_gain(manipulate_row_name), storage_dataset.get_pi(manipulate_row_name)


def ensure_mode_slot(manipulate, parameter, man_mode_index):
    values = manipulate.setdefault(parameter, [])
    if len(values) <= man_mode_index:
        values.extend([None] * (man_mode_index + 1 - len(values)))

# %% [markdown]
# ## Spectroscopy

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
    calibration_values, calibration_key = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', 'frequency', 0)
    calibration_values[calibration_key] = frequency
    station.ds_storage.update_freq(man_name, frequency)
    print(f"Updated {man_name} f0g1 freq to: {station.ds_storage.get_freq(man_name)}")
    station.snapshot_hardware_config(update_main=False)
    station.snapshot_man1_storage_swap(update_main=False)


# %%
# station.yoko_coupler.ramp_current(0.5e-3, sweeprate=0.0001)

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

for man_mode_no in man_modes:
    if expts_to_run['pulse_probe_f0g1']:
        print(f'Running pulse probe f0g1 for M{man_mode_no}')
        man_specs[man_mode_no] = f0g1spec_runner.execute(
            man_mode_no=man_mode_no,
            go_kwargs=dict(progress=True),
            relax_delay=2500,
            start=1980 if man_mode_no == 1 else None,
        )


# %% [markdown]
# ## Find Frequency (Chevron)
#

# %%
f0g1_chevron_defaults = AttrDict(dict(
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
    man_mode_no = expt_cfg.man_mode_no
    manipulate_row_name = f'M{man_mode_no}'

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
        pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_contrast']['omega'])
        gain = expt_cfg.gain  # The gain was fixed across all frequency traces.
        if not all(np.isfinite(value) and value > 0 for value in (pi_len, gain)):
            raise ValueError(f'{manipulate_row_name} chevron requires a positive finite pi length and gain')
        gain = int(gain)
        print(f"Best frequency found: {best_freq:.4f} MHz")
        station.ds_storage.update_freq(manipulate_row_name, best_freq)
        calibration_values, calibration_key = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', 'frequency', 0)
        calibration_values[calibration_key] = float(best_freq)
        print(f"Updated {manipulate_row_name} frequency to {best_freq:.4f} MHz")

        station.ds_storage.update_pi(manipulate_row_name, pi_len)
        calibration_values, calibration_key = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', 'length', 0)
        calibration_values[calibration_key] = float(pi_len)
        print('Updated the pi length to:', pi_len)

        station.ds_storage.update_h_pi(manipulate_row_name, pi_len / 2)
        print('Updated the h_pi length to:', pi_len / 2, '(ds_storage)')

        station.ds_storage.update_gain(manipulate_row_name, gain)
        calibration_values, calibration_key = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', 'gain', 0)
        calibration_values[calibration_key] = gain
        print('Updated gain to:', gain)
        station.snapshot_man1_storage_swap(update_main=False)
        station.snapshot_hardware_config(update_main=False)
    mother_expt.analysis = chevron_analysis
    chevron_analysis.display_results()


f0g1_chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,
    default_expt_cfg=f0g1_chevron_defaults,
    sweep_param='freq',
    live_plot=False,
    # preprocessor=my_preproc,
    postprocessor=f0g1_chevron_postproc,
    job_client=client,
)

# %%
# coarse sweep
f0g1_chevron_coarse = {}
if expts_to_run["man_freq_coarse_chevron"]:
    for man_mode_no in man_modes:
        frequency, gain, _ = f0g1_values(station, man_mode_no)
        f0g1_chevron_coarse[man_mode_no] = f0g1_chevron_runner.execute(
            man_mode_no=man_mode_no,
            sweep_start=frequency - 3,
            sweep_stop=frequency + 3,
            sweep_npts=11,
            gain=gain,
            start=1,  # time start in us
            batch=True,
        )

# %%
# fine sweep

f0g1_chevron_fine = {}
if expts_to_run["man_freq_fine_chevron"]:
    for man_mode_no in man_modes:
        frequency, gain, _ = f0g1_values(station, man_mode_no)
        f0g1_chevron_fine[man_mode_no] = f0g1_chevron_runner.execute(
            man_mode_no=man_mode_no,
            sweep_start=frequency - 0.5,
            sweep_stop=frequency + 0.5,
            sweep_npts=11,
            gain=gain,
            reps=75,
            batch=True,
        )

# %% [markdown]
# ## Error amplification

# %%
# Manipulate/Error Amplification - New Pattern with CharacterizationRunner
span = 0.5
expts = 40


steps = span/expts


# Configuration defaults
man_f0g1_error_amp_defaults = AttrDict(dict(
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

def man_f0g1_error_amp_preproc(station,
                      default_expt_cfg, 
                      **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    if expt_cfg.pulse_type is None:
        if expt_cfg.man_mode_no == 2 and expt_cfg.sideband == 'f0-g1':
            expt_cfg.pulse_type = ['man', 'M2', 'pi', 0]
        else:
            expt_cfg.pulse_type = ['multiphoton', expt_cfg.sideband, 'pi', 0]
    expt_cfg.multiphoton_mode = expt_cfg.man_mode_no
    if expt_cfg.start is None:
        if expt_cfg.man_mode_no == 2 and expt_cfg.sideband == 'f0-g1':
            frequency, _, _ = f0g1_values(station, 2)
        else:
            photon_number = int(expt_cfg.sideband[1])
            pulses = multiphoton_table(station.hardware_cfg, expt_cfg.man_mode_no, num_photon_levels=photon_number + 1)
            frequency = pulses.pi['fn-gn+1'].frequency[photon_number]
        expt_cfg.start = frequency - span / 2
    return expt_cfg

def man_f0g1_error_amp_postproc(station, expt):
    """
    Postprocessor for error amplification.
    Analyze with custom parameters.
    """
    # Perform analysis with state_fin='e' as in original code
    expt.analyze(data=expt.data, state_fin='e')
    print('Error amplification analysis complete')
    cfg = expt.cfg.expt
    man_name = f'M{cfg.man_mode_no}'
    frequency = float(expt.data['fit_avgi'][2])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f'{man_name} fitted {cfg.sideband} frequency must be positive and finite, got {frequency}')
    photon_no = int(cfg.sideband[1])
    calibration_values, calibration_key = multiphoton_slot(station.hardware_cfg, cfg.man_mode_no, 'pi', 'fn-gn+1', 'frequency',
                                m1_photons=photon_no if cfg.man_mode_no == 1 else 0,
                                m2_photons=photon_no if cfg.man_mode_no == 2 else 0)
    print(f"{man_name} {cfg.sideband} pi frequency before update:", calibration_values[calibration_key])
    calibration_values[calibration_key] = frequency
    print(f"{man_name} {cfg.sideband} pi frequency after update:", frequency)
    if cfg.sideband == 'f0-g1':
        station.ds_storage.update_freq(man_name, frequency)
        print(f"Updated the {man_name} ds_storage frequency to:", station.ds_storage.get_freq(man_name))
    station.snapshot_hardware_config(update_main=False)
    if cfg.sideband == 'f0-g1':
        station.snapshot_man1_storage_swap(update_main=False)

# Create runner
man_f0g1_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=man_f0g1_error_amp_defaults,
    preprocessor=man_f0g1_error_amp_preproc,
    postprocessor=man_f0g1_error_amp_postproc,
    job_client=client,
)

# Example execution

# Run with analyze=False, display=False initially
# Postprocessor will handle custom analysis

man_error_amp = {}
if expts_to_run["man_error_amplification"]:
    for man_mode_no in man_modes:
        man_error_amp[man_mode_no] = man_f0g1_error_amp_runner.execute(
            man_mode_no=man_mode_no,
            reps = 100, #100
            go_kwargs = dict(analyze=False, display=False),
            postprocess=True  # This will call postprocessor which does the custom analysis
        )


# %% [markdown]
# ## Length Rabi f0g1 (Update time)

# %%
# Define defaults, smart config preprocessing and post-measurement updates
# =====================================
lenrabi_f0g1_defaults = AttrDict(dict(
    start=0.0,
    step=None,  # Set from the saved pi length and number of points.
    qubits=[0],
    expts=expts_to_run['length_rabi_expts'],
    reps=100,
    rounds=1,
    gain=None,
    freq=None,  # Use the selected mode's saved pulse settings.
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
    expt_cfg.man_mode_no = man_mode_no
    if man_mode_no == 2:
        expt_cfg.pi_ge_after = False
        expt_cfg.relax_delay = station.hardware_cfg.device.readout.relax_delay[0]
    expt_cfg.update(kwargs)

    frequency, gain, pi_length_us = f0g1_values(station, man_mode_no)
    if expt_cfg.freq is None:
        expt_cfg.freq = frequency
    if expt_cfg.gain is None:
        expt_cfg.gain = gain
    if expt_cfg.start is None:
        expt_cfg.start = 0.0
    if expt_cfg.step is None:
        if expt_cfg.expts < 2:
            raise ValueError("A 4-pi length sweep needs at least two points.")
        if not np.isfinite(pi_length_us) or pi_length_us <= 0:
            raise ValueError(f"M{man_mode_no} needs a positive saved pi length for this sweep.")
        stop_us = 4 * pi_length_us
        if expt_cfg.start == 0:
            sideband_ch = int(np.atleast_1d(station.hardware_cfg.hw.soc.dacs.sideband.ch)[0])
            first_step_us = stop_us / (expt_cfg.expts - 1)
            # Zero skips the pulse. A positive flat top needs at least three generator cycles.
            if station.soccfg.us2cycles(first_step_us, gen_ch=sideband_ch) < 3:
                expt_cfg.start = station.soccfg.cycles2us(3, gen_ch=sideband_ch)
        if expt_cfg.start >= stop_us:
            raise ValueError("The sweep start must be below four times the saved pi length.")
        expt_cfg.step = (stop_us - expt_cfg.start) / (expt_cfg.expts - 1)
    return expt_cfg

# def lenrabi_f0g1_postproc(station, expt):
#     man_mode_no = expt.cfg.expt.man_mode_no
    
#     # Get analysis results from the LengthRabiFitting object stored in expt
#     if hasattr(expt, '_length_rabi_analysis'):
#         analysis = expt._length_rabi_analysis
#         pi_length = analysis.results['pi_length']
#         pi2_length = analysis.results['pi2_length']
#         gain = expt.cfg.expt.gain
#         freq = expt.cfg.expt.freq

       
#         station.ds_storage.update_all('M' + str(man_mode_no), freq, None, pi_length, pi2_length, gain)
#         print(f'Updated dataset M{man_mode_no}: pi_length={pi_length:.4f}, pi2_length={pi2_length:.4f}, gain={gain}')

#         station.snapshot_hardware_config(update_main=False)
#         station.snapshot_multiphoton_config(update_main=False)

def lenrabi_f0g1_postproc(station, expt):
    man_mode_no = expt.cfg.expt.man_mode_no
    
    # Get analysis results from the LengthRabiFitting object stored in expt
    if hasattr(expt, '_length_rabi_analysis'):
        analysis = expt._length_rabi_analysis
        pi_length = analysis.results.get('pi_length')
        if man_mode_no == 2:
            from experiments.autocal._checks import rabi
            ok, info = rabi(expt.data['xpts'], expt.data['avgi'], analysis.results.get('fit_avgi'), pi_length)
            if not ok:
                print(f"M2 length Rabi not saved: {info['reason']}")
                return
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
            calibration_values, calibration_key = multiphoton_slot(station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', field, 0)
            calibration_values[calibration_key] = value
        print(f'Updated dataset {man_name}: pi_length={pi_length:.4f}, pi2_length={pi2_length:.4f}, gain={gain}')
        station.snapshot_hardware_config(update_main=False)
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

man_f0g1_length_rabi = {}

for man_mode_no in man_modes:
    if expts_to_run['length_rabi'] or expts_to_run['length_rabi_sweep']:
        print(f'Running length rabi for M{man_mode_no}')
        man_f0g1_length_rabi[man_mode_no] = lenrabi_f0g1_runner.execute(
            man_mode_no=man_mode_no,
            go_kwargs=dict(progress=True),
        )
        man_f0g1_length_rabi[man_mode_no].display()


# %% [markdown]
# ## Chi between qubit and Manipulate

# %%
chi_ramsey_defaults = AttrDict(dict(
    start=0.01, step=0.1, expts=100, reps=200, rounds=1, ramsey_freq=0.2,
    qubits=[0], relax_delay=2500, checkEF=False, qubit_ge_init=False, qubit_ge_after=False,
    prepulse=False, postpulse=False, pre_sweep_pulse=None, post_sweep_pulse=None,
    active_reset=False, man_reset=False, storage_reset=False, pre_active_reset_pulse=False,
    gate_based=False, f0g1_cavity=0, normalize=False, advance_phase=0, echoes=[False, 0],
    user_defined_freq=[False, 3568.2038290468167, 5304, 0.035],
))
chi_fit_params = [300, None, None, None, None, None]


def fitted_chi(reference, dressed):
    return dressed.data['f_adjust_ramsey_avgi'][0] - reference.data['f_adjust_ramsey_avgi'][0]


# %% [markdown]
# ### ge

# %%
chi_ge_runs = {}
if expts_to_run['chi_ge']:
    chi_ge_runner = CharacterizationRunner(
        station, meas.RamseyExperiment, chi_ramsey_defaults, job_client=client, show=False,
    )
    for man_mode_no in man_modes:
        prepulse, _ = man_photon_pulses(station, man_mode_no)
        reference = chi_ge_runner.execute(man_mode_no=man_mode_no, go_kwargs=dict(analyze=False, display=False))
        reference.analyze(fitparams=chi_fit_params.copy())
        reference.display()
        dressed = chi_ge_runner.execute(man_mode_no=man_mode_no, prepulse=True, pre_sweep_pulse=prepulse,
                                        go_kwargs=dict(analyze=False, display=False))
        dressed.analyze(fitparams=chi_fit_params.copy())
        dressed.display()
        chi_ge_runs[man_mode_no] = reference, dressed
        chi = fitted_chi(reference, dressed)
        if not np.isfinite(chi) or chi == 0:
            raise ValueError(f'M{man_mode_no} chi_ge fit must be finite and nonzero')
        manipulate = station.hardware_cfg.device.manipulate
        man_mode_index = man_mode_no - 1
        ensure_mode_slot(manipulate, 'chi_ge', man_mode_index)
        manipulate.chi_ge[man_mode_index] = float(chi)
        manipulate.revival_time[man_mode_index] = float(abs(1 / (2 * chi)))
        station.snapshot_hardware_config(update_main=False)


# %% [markdown]
# ### ef

# %%
chi_ef_runs = {}
if expts_to_run['chi_ef']:
    chi_ef_runner = CharacterizationRunner(
        station, meas.RamseyExperiment, chi_ramsey_defaults, job_client=client, show=False,
    )
    for man_mode_no in man_modes:
        hw = station.hardware_cfg
        chi_ge_values = hw.device.manipulate.get('chi_ge', [])
        man_mode_index = man_mode_no - 1
        if man_mode_index >= len(chi_ge_values) or chi_ge_values[man_mode_index] is None or not np.isfinite(chi_ge_values[man_mode_index]):
            raise ValueError(f'M{man_mode_no} chi_ge is required for chi_ef calibration')

        prepulse, _ = man_photon_pulses(station, man_mode_no)
        qubit = hw.device.qubit
        dressed_ge = float(qubit.f_ge[0]) + chi_ge_values[man_mode_index]
        hpi_ge = qubit.pulses.hpi_ge
        ge_check = chi_ef_runner.execute(
            man_mode_no=man_mode_no, prepulse=True, pre_sweep_pulse=prepulse,
            user_defined_freq=[True, dressed_ge, hpi_ge.gain[0], hpi_ge.sigma[0]],
            go_kwargs=dict(analyze=False, display=False),
        )
        ge_check.analyze(fitparams=chi_fit_params.copy())
        ge_check.display()
        base = MM_dual_rail_base(hw, station.soccfg)
        ge_pi = base.get_prepulse_creator([['qubit', 'ge', 'pi', 0]]).pulse.tolist()
        ge_pi[0][0] = dressed_ge
        dressed_prep = [row + pulse for row, pulse in zip(prepulse, ge_pi)]

        reference = chi_ef_runner.execute(
            man_mode_no=man_mode_no, checkEF=True, qubit_ge_init=True, qubit_ge_after=True,
            ramsey_freq=3, go_kwargs=dict(analyze=False, display=False),
        )
        reference.analyze(fitparams=chi_fit_params.copy())
        reference.display()
        dressed = chi_ef_runner.execute(
            man_mode_no=man_mode_no, checkEF=True, prepulse=True, pre_sweep_pulse=dressed_prep,
            postpulse=True, post_sweep_pulse=ge_pi, ramsey_freq=3, go_kwargs=dict(analyze=False, display=False),
        )
        dressed.analyze(fitparams=chi_fit_params.copy())
        dressed.display()
        chi_ef_runs[man_mode_no] = reference, ge_check, dressed
        chi_ef = fitted_chi(reference, dressed)
        if not np.isfinite(chi_ef):
            raise ValueError(f'M{man_mode_no} chi_ef fit is invalid: {chi_ef}')
        manipulate = station.hardware_cfg.device.manipulate
        ensure_mode_slot(manipulate, 'chi_ef', man_mode_index)
        manipulate.chi_ef[man_mode_index] = float(chi_ef)
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
    for key in ['xpts', 'avgi', 'avgq']:
        if key in length_rabi_analysis.data:
            length_rabi_analysis.data[key] = np.array(length_rabi_analysis.data[key])
    length_rabi_analysis.analyze() #fitparams = [100, 0.1, 0, None, None, None])
    length_rabi_analysis.display(title_str='Length Rabi General F0g1')
    expt_cfg = expt.cfg.expt
    man_mode_no = expt_cfg.man_mode_no
    shift_sign = expt_cfg.shift_sign
    manipulate = station.hardware_cfg.device.manipulate


    if expt_cfg.param_to_calib == 'length':
        delay = length_rabi_analysis.results['pi_length']
        print(f'Fitted revival time: {delay} us')
        parameter = 'revival_time_fast' if expt_cfg.parity_fast else 'revival_time'
        man_mode_index = man_mode_no - 1
        ensure_mode_slot(manipulate, parameter, man_mode_index)
        manipulate[parameter][man_mode_index] = float(delay)
        station.snapshot_hardware_config(update_main=False)

    elif expt_cfg.param_to_calib == 'ac_stark_shift': 
        freq = length_rabi_analysis.data['fit_avgi'][1]
        print(f'Fitted ac Stark shift: {freq} MHz')
        manipulate.revival_stark_shift[man_mode_no - 1] += float(shift_sign * freq)
        station.snapshot_hardware_config(update_main=False)


# %% [markdown]
# ### Calibrate first the AC stark shift
#
# Manipulate in 0, qubit should not rotate
#
# this is calibrating the virtual Z to correct from rotating frame to bare frame

# %%
parity_stark_shift_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityDelayExperiment,
    default_expt_cfg=parity_delay_defaults,
    postprocessor=parity_delay_postproc,
    preprocessor=parity_delay_preproc,
    job_client=client,
    use_queue=True,
)

parity_delay_ac_stark = {}
if expts_to_run['parity_delay_fast']:
    for man_mode_no in man_modes:
        parity_delay_ac_stark[man_mode_no] = parity_stark_shift_runner.execute(
            man_mode_no=man_mode_no,
            parity_fast=True,
            param_to_calib='ac_stark_shift',
            go_kwargs=dict(analyze=False, display=False),
            step=0.2,
            expts=100,
            reps=100,
            shift_sign=1,
        )

# %% [markdown]
# Confirming the correction is in the right direction

# %%
parity_stark_shift_check_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityDelayExperiment,
    default_expt_cfg=parity_delay_defaults,
    preprocessor=parity_delay_preproc,
    job_client=client,
    use_queue=True,
)

parity_delay_ac_stark_check = {}
if expts_to_run['parity_delay_fast']:
    for man_mode_no in man_modes:
        parity_delay_ac_stark_check[man_mode_no] = parity_stark_shift_check_runner.execute(
            man_mode_no=man_mode_no,
            parity_fast=True,
            param_to_calib='ac_stark_shift',
            go_kwargs=dict(analyze=False, display=False),
            step=0.2,
            expts=100,
            reps=100,
            shift_sign=1,
        )
        parity_delay_ac_stark_check[man_mode_no].display(fit=False)

# %% [markdown]
# ### Now parity length after the AC stark shift is calibrated

# %%
parity_revival_time_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ParityDelayExperiment,
    default_expt_cfg=parity_delay_defaults,
    postprocessor=parity_delay_postproc,
    preprocessor=parity_delay_preproc,
    job_client=client,
    use_queue=True,
)

parity_delays = {}
if expts_to_run['parity_delay_fast'] or expts_to_run['parity_delay_slow']:
    for man_mode_no in man_modes:
        parity_delays[man_mode_no] = parity_revival_time_runner.execute(
            man_mode_no=man_mode_no,
            param_to_calib='length',
            parity_fast=expts_to_run['parity_delay_fast'],
            go_kwargs=dict(analyze=False, display=False),
            step=0.04,
            expts=100,
            reps=100,
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
        frequency, gain, length = f0g1_values(station, expt_cfg.man_mode_no)
        sigma = station.hardware_cfg.device.manipulate.ramp_sigma
        if expt_cfg.man_mode_no == 1:
            sigma = station.hardware_cfg.device.multiphoton.pi['fn-gn+1'].sigma[0]
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

t1_manipulate = {}
if expts_to_run['t1_man']:
    for man_mode_no in man_modes:
        t1_manipulate[man_mode_no] = t1_manipulate_runner.execute(man_mode_no=man_mode_no)
        t1_manipulate[man_mode_no].analyze()

# %% [markdown]
# ## T2
#
# prepares man in + and does ramsey with this

# %%
cross_kerr_storage_mode_no = 7
cross_kerr_storage_photon_number = 0

def cross_kerr_prep(man_mode_no):
    cfg = dict(parity_meas=True, use_auto_prep=True, prepulse=False)
    if cross_kerr_storage_photon_number == 1:
        stor_name = f'M{man_mode_no}-S{cross_kerr_storage_mode_no}'
        if not station.ds_storage.has_row(stor_name):
            raise ValueError(f'Missing storage swap row {stor_name}')
        cfg.update(prepulse=True, pre_sweep_pulse=[
            ['qubit', 'ge', 'pi', 0],
            ['qubit', 'ef', 'pi', 0],
            ['man', f'M{man_mode_no}', 'pi', 0],
            ['storage', stor_name, 'pi', 0],
        ])
    return cfg

# %% [markdown]
# ### First calibrate the stark shift and residual detuning on the + state during the f0-g1

# %%
t2_manipulate_defaults = AttrDict(dict(
    start=0,
    step=0.01,
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


t2_manipulate = {}
if expts_to_run['t2_man']:
    for man_mode_no in man_modes:
        result = t2_manipulate_runner.execute(
            man_mode_no=man_mode_no, man_mode_idx=man_mode_no, **cross_kerr_prep(man_mode_no),
        )
        t2_manipulate[man_mode_no] = result
        result.analyze()
        print(f'M{man_mode_no} fit freq MHz', result.data['f_adjust_ramsey_avgi'][0])


# %% [markdown]
# # Storage

# %%
# Storage Spectroscopy - New Pattern with CharacterizationRunner
from experiments.MM_dual_rail_base import MM_dual_rail_base


def get_storage_mode_parameters(station, man_mode_no, stor_mode_no):
    """Get pulse parameters for a given storage mode."""
    stor_name = 'M' + str(man_mode_no) + '-S' + str(stor_mode_no)
    freq = station.ds_storage.get_freq(stor_name)
    gain = station.ds_storage.get_gain(stor_name)
    pi_len = station.ds_storage.get_pi(stor_name)
    h_pi_len = station.ds_storage.get_h_pi(stor_name)
    # flux_low_ch = station.hardware_cfg.hw.soc.dacs.flux_low.ch
    # flux_high_ch = station.hardware_cfg.hw.soc.dacs.flux_high.ch
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
        station, expt_cfg.man_mode_no, expt_cfg.stor_mode_no
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
    expt_cfg.flux_drive = [ch, 1, gain, expt_cfg.flux_drive_duration]  # Auto uses the full spectroscopy gain.
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
    if not np.isfinite(new_freq) or new_freq <= 0:
        raise ValueError(f'Invalid spectroscopy frequency for {stor_name}: {new_freq}')
    station.ds_storage.update_freq(stor_name, new_freq)
    station.snapshot_man1_storage_swap(update_main=False)
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
storage_spectra_by_man_storage_pair = {}
if expts_to_run.get('stor_spectroscopy', False):
    for man_mode_no in man_modes:
        for stor_mode_no in expts_to_run['stor_modes']:
            man_storage_pair = (man_mode_no, stor_mode_no)
            storage_spectra_by_man_storage_pair[man_storage_pair] = stor_spec_runner.execute(
                man_mode_no=man_mode_no,
                stor_mode_no=stor_mode_no,
                flux_drive_duration=1,
                expts=800,
                reps=100,
                relax_delay=800,
                bw=40,
            )
            storage_spectra_by_man_storage_pair[man_storage_pair].display()


# %% [markdown]
# ## Man-dump

# %%
def get_dump_mode_parameters(station, man_mode_no, dump_mode_no):
    """Get pulse parameters for a given dump mode."""
    dump_name = 'M' + str(man_mode_no) + '-D' + str(dump_mode_no)
    freq = station.ds_storage.get_freq(dump_name)
    gain = station.ds_storage.get_gain(dump_name)
    pi_len = station.ds_storage.get_pi(dump_name)
    h_pi_len = station.ds_storage.get_h_pi(dump_name)
    flux_low_ch = station.hardware_cfg.hw.soc.dacs.flux_low.ch
    flux_high_ch = station.hardware_cfg.hw.soc.dacs.flux_high.ch
    # ch = flux_low_ch if freq < 1000 else flux_high_ch
    ch = 4 
    print('Flux channel manually set to 4')

    # prepulse_spec = [['multiphoton', 'g0-e0', 'pi', 0],
    #                  ['multiphoton', 'e0-f0', 'pi', 0],
    #                  ['multiphoton', 'f0-g1', 'pi', 0]]
    # postpulse_spec = [['multiphoton', 'f0-g1', 'pi', 0],
    #                   ['multiphoton', 'e0-f0', 'pi', 0]]
    # prepulse = mm_base_dummy.get_prepulse_creator(prepulse_spec).pulse.tolist()
    # postpulse = mm_base_dummy.get_prepulse_creator(postpulse_spec).pulse.tolist()

    prepulse, postpulse = man_photon_pulses(station, man_mode_no)

    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse


def available_dump_pairs(station):
    dataset = station.ds_storage
    rows = set(dataset.df[dataset.indexing_row_name])
    for man_mode_no in man_modes:
        for dump_mode_no in expts_to_run['dump_modes']:
            row = f'M{man_mode_no}-D{dump_mode_no}'
            if row in rows:
                yield man_mode_no, dump_mode_no
            else:
                print(f'Skipping {row} dump calibration. No row in the selected storage CSV.')


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
    expt_cfg.update(kwargs)
    expt_cfg.man_mode_no = man_mode_no
    expt_cfg.dump_mode_no = dump_mode_no
    
    # Get dump mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_dump_mode_parameters(
        station, man_mode_no, dump_mode_no
    )
    print(f'Using dump mode parameters: freq={freq}, gain={gain}, pi_len={pi_len}, h_pi_len={h_pi_len}, ch={ch}')

    freq = expt_cfg.get('center_freq', freq)
    
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
    cfg = expt.cfg.expt
    dump_name = f'M{cfg.man_mode_no}-D{cfg.dump_mode_no}'
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
    postprocessor=dump_spec_postproc,
    job_client=client,
)

# Execute for dump modes
dump_specs = {}
if expts_to_run.get('dump_spectroscopy', False):
    for man_mode_no, dump_mode_no in available_dump_pairs(station):
        print(f'Running dump spectroscopy for M{man_mode_no}-D{dump_mode_no}')
        dump_specs[man_mode_no, dump_mode_no] = dump_spec_runner.execute(
            man_mode_no=man_mode_no,
            dump_mode_no=dump_mode_no,
            flux_gain=3000,
            flux_length=60,
            # step=0.05,  # bw/expts = 100/200
            # span=30.0,
            # expts=600,
            span=5.0, #originally 20
            expts=100,
            reps=300,
            # coupler_current=coupler_current,
        )
        # dump_specs[dump_mode_no].display()
    # dump_spec_postproc(station, dump_specs[1], man_mode_no=1,
    #     dump_mode_no=dump_mode_no)

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
        station, expt_cfg.man_mode_no, expt_cfg.dump_mode_no
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
    dump_row_name = f'M{expt_cfg.man_mode_no}-D{expt_cfg.dump_mode_no}'

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
    station.ds_storage.update_freq(dump_row_name, best_freq)
    print(f"Updated {dump_row_name} frequency to {best_freq:.4f} MHz (ds_storage)")

    pi_len = 3 * t1
    station.ds_storage.update_pi(dump_row_name, pi_len)
    print('Updated the pi length to 3 T1:', pi_len, "(ds_storage)")

    station.ds_storage.update_h_pi(dump_row_name, pi_len / 2)
    print('Updated the h_pi length to:', pi_len / 2, "(ds_storage)")

    gain = expt_cfg.get('gain', station.ds_storage.get_gain(dump_row_name))
    station.ds_storage.update_gain(dump_row_name, gain)
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
# Run dump chevron for specified dump mode
frequency_bandwidth_by_dump_mode = [3, 2]

# Get current frequency for sweep range
# center_freq = 2265.387857	
# freq_bw = 30  # MHz bandwidth for frequency sweep

dump_chevrons = {}
if expts_to_run.get('dump_chevron', False):
    for man_mode_no, dump_mode_no in available_dump_pairs(station):
        dump_name = f'M{man_mode_no}-D{dump_mode_no}'
        dump_mode_index = dump_mode_no - 1
        center_freq = station.ds_storage.get_freq(dump_name)
        print(f'Running dump sideband chevron for M{man_mode_no}-D{dump_mode_no}')
         # MHz bandwidth for frequency sweep
        freq_bw = frequency_bandwidth_by_dump_mode[dump_mode_index]
        dump_chevrons[man_mode_no, dump_mode_no] = dump_sideband_chevron_runner.execute(
            dump_mode_no=dump_mode_no,
            man_mode_no=man_mode_no,
            sweep_start=center_freq - freq_bw/2,
            sweep_stop=center_freq + freq_bw/2,
            sweep_npts=15,
            pi_len_sweep=60.0,#100.0,
            expts=20,
            reps=100,
            gain=6000 if man_mode_no == 1 else station.ds_storage.get_gain(dump_name),
            # coupler_current=coupler_current,
            batch=True,
        )

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
    reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
    dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
    update_post_pulse_phase=[False, 0],
    relax_delay=2500,
))

def sideband_chevron_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    
    # Get storage mode parameters
    freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse = get_storage_mode_parameters(
        station, expt_cfg.man_mode_no, expt_cfg.stor_mode_no
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
        pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_contrast']['omega'])
        if not all(np.isfinite(value) and value > 0 for value in (best_freq, pi_len, expt_cfg.flux_drive[2])):
            raise ValueError(f'{stor_name} chevron requires a positive finite frequency, pi length, and gain')
        print(f"Best frequency found: {best_freq:.4f} MHz")
        station.ds_storage.update_freq(stor_name, best_freq)
        print(f"Updated {stor_name} frequency to {best_freq:.4f} MHz")
        station.ds_storage.update_pi(stor_name, pi_len)
        print('Updated the pi length to:', pi_len)
        station.ds_storage.update_h_pi(stor_name, pi_len / 2)
        print('Updated the h_pi length to:', pi_len / 2)
        station.ds_storage.update_gain(stor_name, expt_cfg.flux_drive[2])
        print('Updated gain to:', expt_cfg.flux_drive[2])
        station.snapshot_man1_storage_swap(update_main=False)
    mother_expt.analysis = chevron_analysis

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
if expts_to_run['sideband_chevron']:
    for man_mode_no, stor_mode_no in man_storage_mode_pairs:
        stor_name = f'M{man_mode_no}-S{stor_mode_no}'
        freq_span = 0.5*3  # MHz
        freq_step = 0.05*3
        pi_len_sweep = 10
        gain = station.ds_storage.get_gain(stor_name)
        # gain = 30000
        
        print(f'Running sideband chevron for {stor_name}')
        chevron_analysis = sideband_chevron_runner.execute(
            reps=50,
            man_mode_no=man_mode_no,
            stor_mode_no=stor_mode_no,
            sweep_start=station.ds_storage.get_freq(stor_name) - freq_span/2,
            sweep_stop=station.ds_storage.get_freq(stor_name) + freq_span/2,
            sweep_npts=int(freq_span//freq_step + 1),
            pi_len_sweep=pi_len_sweep,
            
            expts=50, # num steps of time
            gain=gain,
            # freq = 1242.8,
            skip_update = False,
            batch = True
        )
        storage_chevrons[man_mode_no, stor_mode_no] = chevron_analysis
        chevron_analysis.analysis.display_results(title = chevron_analysis.fname)

# %% [markdown]
# ## Error amplification

# %%
storage_error_amp_gain = {}
storage_error_amp_freq = {}

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
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    dump_reset_iter_num=active_reset_default_dict["dump_reset_iter_num"],
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
        station, man_mode_no, stor_mode_no
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
        if expt_cfg.get("max_gain", None) is not None:
            if (start + step * expt_cfg.expts > expt_cfg.max_gain):
                start = int(expt_cfg.max_gain - expt_cfg.span)
                print(f"Max gain is limitedd by {expt_cfg.max_gain}")
    else:
        raise ValueError("parameter_to_test must be either 'frequency' or 'gain'.")
    expt_cfg.start = start
    expt_cfg.step = step
    expt_cfg.pulse_type = pulse_type 
    return expt_cfg


def error_amp_stor_postproc(station, expt):
    expt.analyze(data=expt.data, state_fin='e')

    cfg = expt.cfg.expt
    stor_name = f'M{cfg.man_mode_no}-S{cfg.stor_mode_no}'
    fit = expt.data.get('fit_avgi')
    center = fit[2] if fit is not None else np.nan
    x_pts = expt.data['x_pts']
    if not (np.isfinite(center) and center > 0 and len(x_pts) and np.min(x_pts) <= center <= np.max(x_pts)):
        print(f'{stor_name} {cfg.parameter_to_test} not saved: no positive fitted center inside the sweep')
        return
    if cfg.parameter_to_test == 'gain':
        station.ds_storage.update_gain(stor_name, center)
    elif cfg.parameter_to_test == 'frequency':
        station.ds_storage.update_freq(stor_name, center)
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
frequency_span_mhz_by_storage_mode = STORAGE_SIDEBAND_FREQ_SPAN #[0.2 ,  0.4,  0.4,  0.3,  0.3,  0.3, 0.2]
gain_span_fraction_by_storage_mode = STORAGE_SIDEBAND_GAIN_SPAN #[None,  0.2,  0.3,  0.1,  0.1, 0.15, 0.2] # should be less than 1.0
max_gain_by_storage_mode    = STORAGE_SIDEBAND_MAX_GAIN #[None, 11000, None, None,None,None,None,]


freq_scan_pulse_count_step_by_storage_mode = [None, None, None, None, None, None, None]
gain_scan_pulse_count_step_by_storage_mode = [   5,    5,    5,    5,    5,    5,    5]

do_freq_erroramp = True
do_gain_erroramp = True


if expts_to_run['sideband_erroramp']:
    for man_mode_no, stor_mode_no in man_storage_mode_pairs:
        storage_mode_index = stor_mode_no - 1
        
        gain_span_fraction = gain_span_fraction_by_storage_mode[storage_mode_index]
        if gain_span_fraction is None:
            gain_span_fraction = 0.3
        
        n_step_freq = freq_scan_pulse_count_step_by_storage_mode[storage_mode_index]
        n_step_gain = gain_scan_pulse_count_step_by_storage_mode[storage_mode_index]
        if n_step_freq is None:
            n_step_freq = 1
        if n_step_gain is None:
            n_step_gain = 1
        n_pulses_freq = 5 * n_step_freq
        n_pulses_gain = 5 * n_step_gain
        man_storage_pair = (man_mode_no, stor_mode_no)
        stor_name = f'M{man_mode_no}-S{stor_mode_no}'
        # The existing active-reset sequence starts with the M1 dump.
        use_active_reset = man_mode_no == 1
        print("Running", stor_name)
        if do_freq_erroramp == True:
            storage_error_amp_freq[man_storage_pair] = error_amp_stor_runner.execute(
                man_mode_no=man_mode_no,
                stor_mode_no=stor_mode_no,
                parameter_to_test='frequency',
                span=frequency_span_mhz_by_storage_mode[storage_mode_index], #0.1
                expts=100,
                n_step = n_step_freq,
                n_pulses = n_pulses_freq,
                active_reset=use_active_reset,
                man_reset=use_active_reset,
                storage_reset=[stor_mode_no] if use_active_reset else False,
                relax_delay = 1000
            )
            if not station.log_measurements:
                storage_error_amp_freq[man_storage_pair].display()
        if do_gain_erroramp == True:
            storage_error_amp_gain[man_storage_pair] = error_amp_stor_runner.execute(
                man_mode_no=man_mode_no,
                stor_mode_no=stor_mode_no,
                parameter_to_test='gain',
                span = int(station.ds_storage.get_gain(stor_name) * gain_span_fraction),
                n_step = n_step_gain,
                n_pulses=n_pulses_gain,
                
                max_gain = max_gain_by_storage_mode[storage_mode_index], #add to autocalib nb

                
                active_reset=use_active_reset,
                man_reset=use_active_reset,
                storage_reset=[stor_mode_no] if use_active_reset else False,
                relax_delay = 4000
            )
            if not station.log_measurements:
                storage_error_amp_gain[man_storage_pair].display()

# %% [markdown]
# # Floquet

# %%
floquet_storage_modes = expts_to_run['floquet_stor_modes'] if 1 in man_modes else []
if 2 in man_modes and any(expts_to_run[flag] for flag in (
    'floquet_chevron_coarse',
    'floquet_chevron_fine',
    'floquet_error_amplificaton_coarse',
    'floquet_error_amplificaton_fine',
    'floquet_phase_accumulation_matrix_coarse',
    'floquet_phase_accumulation_matrix_fine',
)):
    print('Skipping M2 Floquet calibration. The Floquet train and phase calibration use M1.')

# %%
from experiments.MM_dual_rail_base import MM_dual_rail_base

def get_floquet_parameters(station, man_mode_no, stor_mode_no):
    """
    Get pulse parameters for a given storage mode. 
    Also returns prepulse and postpulse (single photon prep and meas for ge meas)

    Args:
        station: MultimodeStation object for managing frequency data.
        man_mode_no: Manipulation mode number.
        stor_mode_no: Storage mode number.

    Returns:
        A tuple containing freq, gain, ch, prepulse, and postpulse.
    """
    stor_name = 'M' + str(man_mode_no) + '-S' + str(stor_mode_no)
    freq = station.ds_floquet.get_freq(stor_name)
    gain = station.ds_floquet.get_gain(stor_name)
    length = station.ds_floquet.get_len(stor_name)
    pi_frac = station.ds_floquet.get_pi_frac(stor_name)
    ch = 'low' if freq < 1000 else 'high'

    mm_base_dummy = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist() # for ge meas, only do f0g1 and ef pi

    return freq, gain, length, pi_frac, ch, prepulse, postpulse



# %% [markdown]
# ## Chevron

# %%
floquet_freq_chev_defaults = AttrDict(dict(
    expts = 1,
    reps = 100,
    rounds = 1,
    qubits = [0],
    ro_stor = 0, # storage mode number that gets read out in the end
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    # if 0, this means to read out man instead
    detunes=np.linspace(-0.3, 0.3, 11).tolist(),
    swept_params = ['detune', 'length'],
    normalize = False,
    active_reset = False,
    man_reset = False, 
    storage_reset = False, 
    prepulse=True,
    postpulse=True,
    init_fock=True,
)) # Shouldn't be modifying this on the fly!
floquet_freq_chev_defaults.update(active_reset_default_dict)
floquet_freq_chev_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

def floquet_freq_chev_preproc(station, default_expt_cfg, **kwargs):
    assert 'init_stor' in kwargs
    expt_cfg = deepcopy(default_expt_cfg)
    ds_floquet = station.ds_floquet
    init_stor = kwargs.pop('init_stor') # storage mode number to initialize to n=1 Fock state
    lengths = np.linspace(0.01, 3.0 * ds_floquet.get_len(f'M1-S{init_stor}'), 10).tolist()

    expt_cfg.init_stor = init_stor
    expt_cfg.lengths = lengths
    expt_cfg.update(kwargs)
    print(expt_cfg)
    return expt_cfg

def floquet_freq_chev_postproc(station, expt):
    expt_cfg = expt.cfg.expt
    stor_name = f'M{expt_cfg.f0g1_cavity}-S{expt_cfg.init_stor}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=np.array(expt.data['ypts']),
        time=np.array(expt.data['xpts']),
        response_matrix=expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()
    
    best_detune = chevron_analysis.results.get('best_frequency_contrast')

    if best_detune is not None:
        pi_frac = station.ds_floquet.get_pi_frac(stor_name)
        print(f"Best detune found: {best_detune:.4f} MHz")
        current_freq = station.ds_floquet.get_freq(stor_name)
        new_freq = current_freq + best_detune
        station.ds_floquet.update_freq(stor_name, new_freq)
        print(f"Updated {stor_name} frequency to {new_freq:.4f} MHz")
        frac_pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
        station.ds_floquet.update_len(stor_name, frac_pi_len)
        print(f'Updated the pi/{pi_frac} length from {station.ds_floquet.get_len(stor_name):.4f} to {frac_pi_len:.4f}')
    
    chevron_analysis.display_results()
    expt.analysis = chevron_analysis
    station.snapshot_floquet_storage_swap(update_main=False)


# %% [markdown]
# ### Coarse

# %%
floquet_coarse_chevron_experiments = [None] * len(floquet_storage_modes)

floquet_coarse_chevron_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.FloquetChevronExperiment,
    ExptProgram=meas.FloquetChevronProgram,
    default_expt_cfg=floquet_freq_chev_defaults,
    preprocessor=floquet_freq_chev_preproc,
    postprocessor=floquet_freq_chev_postproc,
    job_client=client,
)

for selected_storage_index, storage_mode_no in enumerate(floquet_storage_modes):
    if expts_to_run['floquet_chevron_coarse']:
        # ds_floquet.update_gain(f'M1-S{storage_mode_no}', ds_floquet.get_gain(f'M1-S{storage_mode_no}') * 0.75)
        # print(f'Updating gain for M1-S{storage_mode_no} to {ds_floquet.get_gain(f"M1-S{storage_mode_no}")}')

        print(f'Running Floquet Frequency vs Length Chevron for Storage Mode {storage_mode_no}')
        floquet_coarse_chevron_experiments[selected_storage_index] = floquet_coarse_chevron_runner.execute(
            init_stor=storage_mode_no,
            # detunes=np.linspace(-0.2, 0.2, 51).tolist(),
            detunes=np.linspace(-1, 1, 51).tolist(),
            reps=100,
            
            # relax_delay=8000,
            relax_delay = 200,
            active_reset=True,
            man_reset=True, 
            storage_reset=[storage_mode_no],
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
        )
        # floquet_coarse_chevron_experiments[selected_storage_index].display()
        # clear_output(wait=True)

# %% [markdown]
# ### Fine

# %%
floquet_fine_chevron_experiments  = [None] * len(floquet_storage_modes)
chevron_half_span_mhz_by_storage_index = [ 0.5,  0.3, None, None,  0.2,  0.2,  0.2]



floquet_fine_chevron_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.FloquetChevronExperiment,
    ExptProgram=meas.FloquetChevronProgram,
    default_expt_cfg=floquet_freq_chev_defaults,
    preprocessor=floquet_freq_chev_preproc,
    postprocessor=floquet_freq_chev_postproc,
    job_client=client,
)

for selected_storage_index, storage_mode_no in enumerate(floquet_storage_modes):
    if expts_to_run['floquet_chevron_fine']:
        # ds_floquet.update_gain(f'M1-S{storage_mode_no}', ds_floquet.get_gain(f'M1-S{storage_mode_no}') * 0.75)
        # print(f'Updating gain for M1-S{storage_mode_no} to {ds_floquet.get_gain(f"M1-S{storage_mode_no}")}')
        storage_config_index = storage_mode_no - 1
        detuning_half_span_mhz = chevron_half_span_mhz_by_storage_index[storage_config_index]
        if detuning_half_span_mhz is None:
            detuning_half_span_mhz = 1.0
        print(f'Running Floquet Frequency vs Length Chevron for Storage Mode {storage_mode_no}')
        floquet_fine_chevron_experiments[selected_storage_index] = floquet_fine_chevron_runner.execute(
            init_stor=storage_mode_no,
            detunes=np.linspace(-detuning_half_span_mhz, detuning_half_span_mhz, 51).tolist(),
            reps=100,
            
            relax_delay = 200, #8000
            active_reset=True,
            man_reset=True, 
            storage_reset=[storage_mode_no],
            reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
        )
        if not station.log_measurements:
            floquet_fine_chevron_experiments[selected_storage_index].display()
        # clear_output(wait=True)

# %% [markdown]
# ## Error amplification on floquet pulses

# %%
error_amp_floquet_defaults = AttrDict(dict(
    reps=100,
    rounds=1,
    qubits=[0],
    active_reset=False,
    man_mode_no=1,
    stor_is_dump=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500, 
    expts=25,
    qubit_start_storage='g',
    floquet_hardware_loop=floquet_default_dict["floquet_hardware_loop"],
    floquet_sync_delay=floquet_default_dict["scramble_sync_cycles"]
)) # Shouldn't be modifying this on the fly!
error_amp_floquet_defaults.update(active_reset_default_dict)
error_amp_floquet_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

floquet_gain_error_amp_defaults = AttrDict(dict(
    n_pulses=8,
    span=4000,#4000
    expts=30,#30
))

floquet_frequency_error_amp_defaults = AttrDict(dict(
    n_pulses=7,
    span=0.25,
    expts=50,
))


def error_amp_floquet_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_mode_no' in kwargs
    assert 'parameter_to_test' in kwargs 

    # construct the defaults
    expt_cfg = deepcopy(default_expt_cfg)
    if kwargs['parameter_to_test'] == 'gain':
        expt_cfg.update(floquet_gain_error_amp_defaults)
    elif kwargs['parameter_to_test'] == 'frequency':
        expt_cfg.update(floquet_frequency_error_amp_defaults)
    # override with the passed kwargs
    expt_cfg.update(kwargs)

    freq, gain, length, pi_frac, ch, prepulse, postpulse = get_floquet_parameters(station, expt_cfg.man_mode_no, expt_cfg.stor_mode_no)
    pulse_type = ['floquet', f'M{expt_cfg.man_mode_no}-{"D" if expt_cfg.stor_is_dump else "S"}{expt_cfg.stor_mode_no}', f'pi/{pi_frac}', 0]
    # freq = 695.7
    # gain = 10000
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

def error_amp_floquet_postproc(station, expt):
    expt.analyze(data=expt.data, state_fin='e')

    fitted_pulse_parameter = expt.data['fit_avgi'][2]
    floquet_dataset_row = 'M1-S' + str(expt.cfg.expt.stor_mode_no)
    if expt.cfg.expt.parameter_to_test == 'gain':
        station.ds_floquet.update_gain(floquet_dataset_row, fitted_pulse_parameter)
        print(f'Updated gain for {floquet_dataset_row} to {fitted_pulse_parameter}')
    elif expt.cfg.expt.parameter_to_test == 'frequency':
        station.ds_floquet.update_freq(floquet_dataset_row, fitted_pulse_parameter)
        print(f'Updated frequency for {floquet_dataset_row} to {fitted_pulse_parameter}')
    station.snapshot_floquet_storage_swap(update_main=False)


error_amp_floquet_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.error_amplification.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_floquet_defaults,
    preprocessor=error_amp_floquet_preproc,
    postprocessor=error_amp_floquet_postproc,
    job_client=client,
)

# %%
floquet_coarse_gain_error_amp_experiments = [None] * len(floquet_storage_modes)
floquet_coarse_frequency_error_amp_experiments = [None] * len(floquet_storage_modes)
floquet_fine_gain_error_amp_experiments = [None] * len(floquet_storage_modes)
floquet_fine_frequency_error_amp_experiments = [None] * len(floquet_storage_modes)


frequency_span_mhz_by_storage_index = [0.1,  0.1,  None,  0.1,   0.1, None, None] # set for the Coarse
gain_span_fraction_by_storage_index = [None, None, None, None,  0.15, None, None]

# %% [markdown]
# ### Coarse

# %%
default_frequency_span_mhz = 0.1
default_gain_span_fraction = 0.3

do_freq_erroramp = True
do_gain_erroramp = True

if expts_to_run['floquet_error_amplificaton_coarse']:
    for selected_storage_index, storage_mode_no in enumerate(floquet_storage_modes):
        storage_config_index = storage_mode_no - 1
        floquet_dataset_row = 'M1-S' + str(storage_mode_no)
        
        frequency_span_mhz = default_frequency_span_mhz if frequency_span_mhz_by_storage_index[storage_config_index] is None else frequency_span_mhz_by_storage_index[storage_config_index]
        gain_span_fraction = default_gain_span_fraction if gain_span_fraction_by_storage_index[storage_config_index] is None else gain_span_fraction_by_storage_index[storage_config_index]
        
        if do_freq_erroramp:
            floquet_coarse_frequency_error_amp_experiments[selected_storage_index] = error_amp_floquet_runner.execute(
                stor_mode_no=storage_mode_no,
                parameter_to_test='frequency',
                go_kwargs=dict(analyze=False, progress=True, display=False),
                span = frequency_span_mhz, # 0.2 is default, but we want to be more precise since we are already close
                
                # relax_delay=8000,
                relax_delay = 200,
                active_reset=True,
                man_reset=True, 
                storage_reset=[storage_mode_no],
                reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
                dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            )
            if not station.log_measurements:
                floquet_coarse_frequency_error_amp_experiments[selected_storage_index].display()
        if do_gain_erroramp:
            floquet_coarse_gain_error_amp_experiments[selected_storage_index] = error_amp_floquet_runner.execute(
                stor_mode_no=storage_mode_no,
                parameter_to_test='gain',   
                go_kwargs=dict(analyze=False, progress=True, display=False),
                span=int(station.ds_floquet.get_gain(floquet_dataset_row) * gain_span_fraction), #0.7 is default
                # span = 59,
                expts = 60,
                # expts=60,
                
                # relax_delay=8000,
                relax_delay = 200,
                active_reset=True,
                man_reset=True, 
                storage_reset=[storage_mode_no],
                reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
                dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            )
            if not station.log_measurements:
                floquet_coarse_gain_error_amp_experiments[selected_storage_index].display()

# %% [markdown]
# ### Fine

# %%
default_frequency_span_mhz = 0.1
default_gain_span_fraction = 0.3

do_freq_erroramp = True
do_gain_erroramp = True

if expts_to_run['floquet_error_amplificaton_fine']:
    for selected_storage_index, storage_mode_no in enumerate(floquet_storage_modes):
        storage_config_index = storage_mode_no - 1
        floquet_dataset_row = 'M1-S' + str(storage_mode_no)
        
        frequency_span_mhz = default_frequency_span_mhz if frequency_span_mhz_by_storage_index[storage_config_index] is None else frequency_span_mhz_by_storage_index[storage_config_index]
        gain_span_fraction = default_gain_span_fraction if gain_span_fraction_by_storage_index[storage_config_index] is None else gain_span_fraction_by_storage_index[storage_config_index]
        
        if do_freq_erroramp:
            floquet_fine_frequency_error_amp_experiments[selected_storage_index] = error_amp_floquet_runner.execute(
                stor_mode_no=storage_mode_no,
                parameter_to_test='frequency',
                go_kwargs=dict(analyze=False, progress=True, display=False),
                span = frequency_span_mhz / 2, # 0.2 is default, but we want to be more precise since we are already close
                
                # relax_delay=8000,
                relax_delay = 200,
                active_reset=True,
                man_reset=True, 
                storage_reset=[storage_mode_no],
                reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
                dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            )
            if not station.log_measurements:
                floquet_fine_frequency_error_amp_experiments[selected_storage_index].display()
        if do_gain_erroramp:
            floquet_fine_gain_error_amp_experiments[selected_storage_index] = error_amp_floquet_runner.execute(
                stor_mode_no=storage_mode_no,
                parameter_to_test='gain',   
                go_kwargs=dict(analyze=False, progress=True, display=False),
                span=int(station.ds_floquet.get_gain(floquet_dataset_row) * gain_span_fraction / 2), #0.7 is default
                # span = 59,
                expts = 60,
                # expts=60,
                
                # relax_delay=8000,
                relax_delay = 200,
                active_reset=True,
                man_reset=True, 
                storage_reset=[storage_mode_no],
                reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
                dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
            )
            if not station.log_measurements:
                floquet_fine_gain_error_amp_experiments[selected_storage_index].display()

# %% [markdown]
# ## Phase Accumulation

# %%
coarse_phase_experiments_by_target_and_source = [[None for _ in range(7)] for _ in range(7)]
fine_phase_experiments_by_target_and_source = [[None for _ in range(7)] for _ in range(7)]

# %%
sideband_stark_error_amp_defaults = AttrDict(dict(
    expts=1,
    reps=100,
    rounds=1,
    qubits=[0],
    f0g1_cavity=1,  #  1/2 name of manipulate cavity
    init_stor=0, # storage mode number to initialize to n=1 Fock state (0=man)
    ro_stor=0, # storage mode number that gets read out in the end (0=man)
    advance_phases=np.linspace(-15, 15, 51).tolist(),
    n_pulses=np.arange(0, 24, 4).tolist(),
    swept_params=['n_pulse', 'advance_phase'],
    normalize=False, # not sure what this does
    active_reset=False,
    man_reset=False, 
    storage_reset=False, 
    prepulse=True,
    postpulse=True,
    init_fock=True,
))
sideband_stark_error_amp_defaults.update(active_reset_default_dict)
sideband_stark_error_amp_defaults.update(floquet_default_dict)

def sideband_stark_error_amp_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_A' in kwargs
    assert 'stor_B' in kwargs

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    print(expt_cfg)
    return expt_cfg

def sideband_stark_error_amp_postproc(station, expt):
    phase_target_storage_mode_no = expt.cfg.expt.stor_A
    phase_source_storage_mode_no = expt.cfg.expt.stor_B
    phase_target_dataset_row = 'M1-S' + str(phase_target_storage_mode_no)
    phase_source_dataset_row = 'M1-S' + str(phase_source_storage_mode_no)
    expt.analyze(fit=True)
    expt.display(fit=True)
    # fitted_phase_degrees = expt.data['fit_avgi'][2] / 2 # divide by 2 since did pi/12, -pi/12 on the from_stor swap
    fitted_phase_degrees = expt.data['fit_avgi'][2] # not dividing seems to give the correct result somehow? TODO: figure out why
    print("Opt phase on", phase_target_dataset_row, "from", phase_source_dataset_row, ":", fitted_phase_degrees)
    station.ds_floquet.update_phase_from(phase_target_dataset_row, phase_source_dataset_row, fitted_phase_degrees)
    station.snapshot_floquet_storage_swap(update_main=False)

sideband_stark_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.SidebandStarkAmplificationExperiment,
    ExptProgram=meas.SidebandStarkAmplificationProgram, #meas.SidebandStarkAmplificationProgram,
    default_expt_cfg=sideband_stark_error_amp_defaults,
    preprocessor=sideband_stark_error_amp_preproc,
    postprocessor=sideband_stark_error_amp_postproc,
    job_client=client,
)

# %%
station.ds_floquet.df if station.ds_floquet is not None else None

# %% [markdown]
# ### Coarse

# %%


if expts_to_run['floquet_phase_accumulation_matrix_coarse']:
    for phase_target_storage_mode_no in floquet_storage_modes:
        for phase_source_storage_mode_no in floquet_storage_modes:
            if phase_target_storage_mode_no == phase_source_storage_mode_no:
                continue
            print("Starting experiment for storage modes:", phase_target_storage_mode_no, "from", phase_source_storage_mode_no)

            phase_experiment = sideband_stark_error_amp_runner.execute(
                stor_A=phase_target_storage_mode_no,
                stor_B=phase_source_storage_mode_no,
                # relax_delay=8000,
                relax_delay = 100,
                active_reset=True,
                man_reset=True, 
                storage_reset=[phase_target_storage_mode_no, phase_source_storage_mode_no],
                reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
                dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
                reps=100,
                include_10cycles_buffer = floquet_default_dict["include_10cycles_buffer"],
                include_10cycles_buffer_in_pi_half = floquet_default_dict["include_10cycles_buffer_in_pi_half"],
                advance_phases=np.linspace(-20, 20, 51).tolist(),
                # n_pulses=np.arange(0, 42, 7).tolist(),
            )
            coarse_phase_experiments_by_target_and_source[phase_target_storage_mode_no - 1][phase_source_storage_mode_no - 1] = phase_experiment

# %% [markdown]
# ### Fine

# %%

if expts_to_run['floquet_phase_accumulation_matrix_fine']:
    for phase_target_storage_mode_no in floquet_storage_modes:
        for phase_source_storage_mode_no in floquet_storage_modes:
            if phase_target_storage_mode_no == phase_source_storage_mode_no:
                continue
            print("Starting experiment for storage modes:", phase_target_storage_mode_no, "from", phase_source_storage_mode_no)
            saved_phase_degrees = station.ds_floquet.get_phase_from(f'M1-S{phase_target_storage_mode_no}', f"M1-S{phase_source_storage_mode_no}")
            phase_sweep_start_degrees = saved_phase_degrees - expts_to_run['floquet_phase_accumulation_matrix_fine_bw']/2
            phase_sweep_stop_degrees   = saved_phase_degrees + expts_to_run['floquet_phase_accumulation_matrix_fine_bw']/2
            
            phase_experiment = sideband_stark_error_amp_runner.execute(
                stor_A=phase_target_storage_mode_no,
                stor_B=phase_source_storage_mode_no,
                # relax_delay=8000,
                relax_delay = 100,
                active_reset=True,
                man_reset=True, 
                storage_reset=[phase_target_storage_mode_no, phase_source_storage_mode_no],
                reset_dump_mode = active_reset_default_dict["reset_dump_mode"],
                dump_reset_iter_num = active_reset_default_dict["dump_reset_iter_num"],
                reps=100,
                include_10cycles_buffer = floquet_default_dict["include_10cycles_buffer"],
                include_10cycles_buffer_in_pi_half = floquet_default_dict["include_10cycles_buffer_in_pi_half"],
                advance_phases=np.linspace(phase_sweep_start_degrees,
                                           phase_sweep_stop_degrees,
                                           calibration_cfg_default_dict["fine_phase_calibration_expts"]).tolist(),
                n_pulses=np.arange(0, 48, 8).tolist(),
            )
            fine_phase_experiments_by_target_and_source[phase_target_storage_mode_no - 1][phase_source_storage_mode_no - 1] = phase_experiment


# %% [markdown]
# # MultiPhoton

# %%
import numpy as np

# Seed the legacy M1 pulses from the current qubit and M1 vacuum calibrations.
if expts_to_run['multiphoton_fetch_from_singlequbit'] and 1 in man_modes:
    qubit_config = station.hardware_cfg.device.qubit
    m1_pi_pulses = station.hardware_cfg.device.multiphoton.pi
    m1_sideband_pi_pulse = m1_pi_pulses['fn-gn+1']
    for initial_m1_photon_number in range(len(m1_pi_pulses['gn-en']['frequency'])):
        m1_pi_pulses['gn-en']['frequency'][initial_m1_photon_number] = qubit_config.f_ge[0]
        m1_pi_pulses['gn-en']['gain'][initial_m1_photon_number] = qubit_config.pulses.pi_ge.gain[0]
        m1_pi_pulses['en-fn']['frequency'][initial_m1_photon_number] = qubit_config.f_ef[0]
        m1_pi_pulses['en-fn']['gain'][initial_m1_photon_number] = qubit_config.pulses.pi_ef.gain[0]
        for pulse_parameter in ('frequency', 'gain', 'length'):
            m1_sideband_pi_pulse[pulse_parameter][initial_m1_photon_number] = m1_sideband_pi_pulse[pulse_parameter][0]
    station.snapshot_hardware_config(update_main=False)

# %% [markdown]
# ### Multiphoton error amplification

# %%
from copy import deepcopy

from experiments import CharacterizationRunner
from experiments.MM_base import multiphoton_slot, multiphoton_table
from experiments.autocal import _checks as fit_checks
from experiments.single_qubit.error_amplification import ErrorAmplificationExperiment
from slab import AttrDict

multiphoton_error_amp_defaults = AttrDict(dict(
    reps=100, rounds=1, qubit=0, qubits=[0], n_pulses=10, pulse_name='pi',
    active_reset=False, man_reset=True, storage_reset=True, relax_delay=2500,
    avoid_yoko=False,
))


def multiphoton_error_amp_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    man_mode_no = expt_cfg.man_mode_no
    initial_m1_photon_number = expt_cfg.initial_m1_photon_number
    initial_m2_photon_number = expt_cfg.initial_m2_photon_number
    photon_number_in_selected_mode = initial_m1_photon_number if man_mode_no == 1 else initial_m2_photon_number
    photon_number_in_other_mode = initial_m2_photon_number if man_mode_no == 1 else initial_m1_photon_number
    pulse_transition_name = {
        'gn-en': f'g{photon_number_in_selected_mode}-e{photon_number_in_selected_mode}',
        'en-fn': f'e{photon_number_in_selected_mode}-f{photon_number_in_selected_mode}',
        'fn-gn+1': f'f{photon_number_in_selected_mode}-g{photon_number_in_selected_mode + 1}',
    }[expt_cfg.transition_key]
    expt_cfg.man_no = expt_cfg.multiphoton_mode = man_mode_no
    expt_cfg.multiphoton_other_mode_photon_number = photon_number_in_other_mode
    expt_cfg.transition = expt_cfg.sideband = pulse_transition_name
    expt_cfg.pulse_type = ['multiphoton', pulse_transition_name, 'pi', 0]
    expt_cfg.state_fin = 'e' if expt_cfg.transition_key == 'fn-gn+1' else 'g'

    pulse_table = multiphoton_table(
        station.hardware_cfg, man_mode_no, other_mode_photon_number=photon_number_in_other_mode,
        num_photon_levels=photon_number_in_selected_mode + 1,
    )
    sweep_center = pulse_table.pi[expt_cfg.transition_key][expt_cfg.parameter_to_test][photon_number_in_selected_mode]
    expt_cfg.start = sweep_center - expt_cfg.half_span
    expt_cfg.step = 2 * expt_cfg.half_span / expt_cfg.expts
    return expt_cfg


def multiphoton_error_amp_postproc(station, expt, *, save=True):
    expt_cfg = expt.cfg.expt
    man_mode_no = expt_cfg.man_mode_no
    initial_m1_photon_number = expt_cfg.initial_m1_photon_number
    initial_m2_photon_number = expt_cfg.initial_m2_photon_number
    swept_parameter = expt_cfg.parameter_to_test
    gaussian_fit = expt.data.get('fit_avgi')
    if gaussian_fit is None:
        print(f'M{man_mode_no} {expt_cfg.transition} at ({initial_m1_photon_number}, {initial_m2_photon_number}): no {swept_parameter} fit')
        return
    print(f'M{man_mode_no} {expt_cfg.transition} at ({initial_m1_photon_number}, {initial_m2_photon_number}): {swept_parameter} fit = {gaussian_fit[2]}')
    fit_accepted, fit_check = fit_checks.gaussian(expt.data['x_pts'], expt.data['prod_avgi'], gaussian_fit)
    if not fit_accepted:
        print(f"Fit not saved: {fit_check['reason']}")
        return
    if save:
        fitted_pulse_setting = int(round(gaussian_fit[2])) if swept_parameter == 'gain' else float(gaussian_fit[2])
        calibration_values, photon_index_or_pair_key = multiphoton_slot(
            station.hardware_cfg, man_mode_no, 'pi', expt_cfg.transition_key, swept_parameter,
            initial_m1_photon_number, initial_m2_photon_number,
        )
        calibration_values[photon_index_or_pair_key] = fitted_pulse_setting
        if expt_cfg.transition_key == 'fn-gn+1' and initial_m1_photon_number == initial_m2_photon_number == 0 and swept_parameter == 'frequency':
            station.ds_storage.update_freq(f'M{man_mode_no}', fitted_pulse_setting)
            station.snapshot_man1_storage_swap(update_main=False)
        station.snapshot_hardware_config(update_main=False)

# %% [markdown]
# ### Sideband spectroscopy

# %%
from experiments.MM_base import multiphoton_prep
from experiments.MM_dual_rail_base import MM_dual_rail_base
from experiments.autocal._util import fit_line
from experiments.single_qubit.pulse_probe_f0g1_spectroscopy import PulseProbeF0g1SpectroscopyExperiment

multiphoton_spec_defaults = AttrDict(dict(
    step=0.1, expts=500, reps=500, rounds=1, length=1, gain=1000,
    pulse_type='gaussian', qubit_f=False, qubits=[0], relax_delay=200, avoid_yoko=False,
))


def multiphoton_sideband_pulse_parameters(station, man_mode_no, initial_m1_photon_number, initial_m2_photon_number):
    photon_number_in_selected_mode = initial_m1_photon_number if man_mode_no == 1 else initial_m2_photon_number
    photon_number_in_other_mode = initial_m2_photon_number if man_mode_no == 1 else initial_m1_photon_number
    pulse_table = multiphoton_table(
        station.hardware_cfg, man_mode_no, other_mode_photon_number=photon_number_in_other_mode,
        num_photon_levels=photon_number_in_selected_mode + 1,
    )
    sideband_pi_pulses = pulse_table.pi['fn-gn+1']
    return (
        sideband_pi_pulses.frequency[photon_number_in_selected_mode],
        sideband_pi_pulses.gain[photon_number_in_selected_mode],
        sideband_pi_pulses.length[photon_number_in_selected_mode],
    )


def multiphoton_sideband_preproc(station, default_expt_cfg, **kwargs):
    """Prepare |f, n_M1, n_M2> before sweeping the selected sideband."""
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    man_mode_no = expt_cfg.man_mode_no
    initial_m1_photon_number = expt_cfg.initial_m1_photon_number
    initial_m2_photon_number = expt_cfg.initial_m2_photon_number
    photon_number_in_selected_mode = initial_m1_photon_number if man_mode_no == 1 else initial_m2_photon_number
    photon_number_in_other_mode = initial_m2_photon_number if man_mode_no == 1 else initial_m1_photon_number
    expt_cfg.man_no = expt_cfg.multiphoton_mode = man_mode_no
    expt_cfg.multiphoton_other_mode_photon_number = photon_number_in_other_mode
    expt_cfg.transition = f'f{photon_number_in_selected_mode}-g{photon_number_in_selected_mode + 1}'
    state_preparation_pulses = multiphoton_prep(station.hardware_cfg, initial_m1_photon_number, initial_m2_photon_number)
    ge_transition = f'g{photon_number_in_selected_mode}-e{photon_number_in_selected_mode}'
    ef_transition = f'e{photon_number_in_selected_mode}-f{photon_number_in_selected_mode}'
    state_preparation_pulses += [
        ['multiphoton', ge_transition, 'pi', 0, man_mode_no, photon_number_in_other_mode],
        ['multiphoton', ef_transition, 'pi', 0, man_mode_no, photon_number_in_other_mode],
    ]
    pulse_builder = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
    expt_cfg.pre_sweep_pulse = pulse_builder.get_prepulse_creator(state_preparation_pulses).pulse.tolist()
    expt_cfg.prepulse = True
    return expt_cfg


def multiphoton_spec_postproc(station, expt, *, save=True):
    expt_cfg = expt.cfg.expt
    man_mode_no = expt_cfg.man_mode_no
    initial_m1_photon_number = expt_cfg.initial_m1_photon_number
    initial_m2_photon_number = expt_cfg.initial_m2_photon_number
    photon_number_in_selected_mode = initial_m1_photon_number if man_mode_no == 1 else initial_m2_photon_number
    lorentzian_fit = expt.data['fit_avgi']
    fit_accepted, fit_check = fit_checks.lorentzian(
        expt.data['xpts'], expt.data['avgi'], lorentzian_fit,
        max_width_frac=0.05, min_snr=5 if photon_number_in_selected_mode == 0 else 3.5,
    )
    if not fit_accepted:
        print(f"Fit not saved: {fit_check['reason']}")
        return
    sideband_frequency_mhz = lorentzian_fit[2]
    print(f'M{man_mode_no} {expt_cfg.transition} at ({initial_m1_photon_number}, {initial_m2_photon_number}): spectroscopy frequency = {sideband_frequency_mhz} MHz')
    if save:
        calibration_values, photon_index_or_pair_key = multiphoton_slot(
            station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', 'frequency',
            initial_m1_photon_number, initial_m2_photon_number,
        )
        calibration_values[photon_index_or_pair_key] = float(sideband_frequency_mhz)
        if initial_m1_photon_number == initial_m2_photon_number == 0:
            station.ds_storage.update_freq(f'M{man_mode_no}', float(sideband_frequency_mhz))
            station.snapshot_man1_storage_swap(update_main=False)
        station.snapshot_hardware_config(update_main=False)


# %% [markdown]
# ### Sideband frequency and length chevron

# %%
from experiments import SweepRunner
from experiments.single_qubit.length_rabi_f0g1_general import LengthRabiGeneralF0g1Experiment

multiphoton_chevron_defaults = AttrDict(dict(
    start=1, step=0.1, expts=25, reps=100, rounds=1, qubits=[0],
    ramp_sigma=0.005, use_arb_waveform=False, pi_ge_before=False, pi_ef_before=False,
    pi_ge_after=False, postpulse=False, normalize=False, active_reset=False,
    check_man_reset=[False, 0], check_man_reset_pi=[], err_amp_reps=0, swap_lossy=False, avoid_yoko=False,
))


def multiphoton_chevron_postproc(station, expt, *, save=True):
    expt_cfg = expt.cfg.expt
    man_mode_no = expt_cfg.man_mode_no
    initial_m1_photon_number = expt_cfg.initial_m1_photon_number
    initial_m2_photon_number = expt_cfg.initial_m2_photon_number
    best_contrast_frequency_mhz = expt._chevron_analysis.results.get('best_frequency_contrast')
    swept_frequencies_mhz = expt.data['freq_sweep']
    if best_contrast_frequency_mhz is None or not min(swept_frequencies_mhz) < best_contrast_frequency_mhz < max(swept_frequencies_mhz):
        print('Fit not saved: no chevron center inside the sweep')
        return
    print(f'M{man_mode_no} {expt_cfg.transition} at ({initial_m1_photon_number}, {initial_m2_photon_number}): chevron frequency = {best_contrast_frequency_mhz} MHz')
    if save:
        calibration_values, photon_index_or_pair_key = multiphoton_slot(
            station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', 'frequency',
            initial_m1_photon_number, initial_m2_photon_number,
        )
        calibration_values[photon_index_or_pair_key] = float(best_contrast_frequency_mhz)
        if initial_m1_photon_number == initial_m2_photon_number == 0:
            station.ds_storage.update_freq(f'M{man_mode_no}', float(best_contrast_frequency_mhz))
            station.snapshot_man1_storage_swap(update_main=False)
        station.snapshot_hardware_config(update_main=False)


# %% [markdown]
# ### Sideband length Rabi setup and optional preliminary scan

# %%
multiphoton_rabi_defaults = AttrDict(dict(
    expts=200, reps=500, rounds=1, qubits=[0], relax_delay=2500, use_arb_waveform=False,
    pi_ge_before=False, pi_ef_before=False, pi_ge_after=False, postpulse=False, normalize=False,
    active_reset=False, man_reset=True, stor_reset=True, check_man_reset=[False, 0],
    swap_lossy=False, check_man_reset_pi=[], err_amp_reps=0, avoid_yoko=False,
))


def multiphoton_rabi_postproc(station, expt, *, save=True):
    expt_cfg = expt.cfg.expt
    man_mode_no = expt_cfg.man_mode_no
    initial_m1_photon_number = expt_cfg.initial_m1_photon_number
    initial_m2_photon_number = expt_cfg.initial_m2_photon_number
    rabi_fit_results = expt._length_rabi_analysis.results
    pi_pulse_length_us = rabi_fit_results.get('pi_length')
    fit_accepted, fit_check = fit_checks.rabi(
        expt.data['xpts'], expt.data['avgi'], rabi_fit_results.get('fit_avgi'), pi_pulse_length_us,
    )
    if not fit_accepted:
        print(f"Fit not saved: {fit_check['reason']}")
        return
    print(f'M{man_mode_no} {expt_cfg.transition} at ({initial_m1_photon_number}, {initial_m2_photon_number}): pi length = {pi_pulse_length_us} us')
    if save:
        sideband_pulse_settings = dict(length=float(pi_pulse_length_us), gain=int(expt_cfg.gain))
        if man_mode_no == 2:
            sideband_pulse_settings.update(sigma=float(expt.cfg.device.manipulate.ramp_sigma), type='flat_top')
        for pulse_setting_name, pulse_setting_value in sideband_pulse_settings.items():
            calibration_values, photon_index_or_pair_key = multiphoton_slot(
                station.hardware_cfg, man_mode_no, 'pi', 'fn-gn+1', pulse_setting_name,
                initial_m1_photon_number, initial_m2_photon_number,
            )
            calibration_values[photon_index_or_pair_key] = pulse_setting_value
        if initial_m1_photon_number == initial_m2_photon_number == 0:
            station.ds_storage.update_pi(f'M{man_mode_no}', float(pi_pulse_length_us))
            station.ds_storage.update_h_pi(f'M{man_mode_no}', float(pi_pulse_length_us) / 2)
            station.ds_storage.update_gain(f'M{man_mode_no}', sideband_pulse_settings['gain'])
            station.snapshot_man1_storage_swap(update_main=False)
        station.snapshot_hardware_config(update_main=False)


# %% [markdown]
# ### Run multiphoton calibration

# %%
# Bind every runner to the station used for this run.
multiphoton_error_amp_runner = CharacterizationRunner(
    station, ErrorAmplificationExperiment, multiphoton_error_amp_defaults,
    preprocessor=multiphoton_error_amp_preproc, postprocessor=multiphoton_error_amp_postproc,
    job_client=client, use_queue=USE_QUEUE, show=False,
)
multiphoton_spec_runner = CharacterizationRunner(
    station, PulseProbeF0g1SpectroscopyExperiment, multiphoton_spec_defaults,
    preprocessor=multiphoton_sideband_preproc, postprocessor=multiphoton_spec_postproc,
    job_client=client, use_queue=USE_QUEUE, show=False,
)
multiphoton_chevron_runner = SweepRunner(
    station, LengthRabiGeneralF0g1Experiment, multiphoton_chevron_defaults, sweep_param='freq',
    preprocessor=multiphoton_sideband_preproc, postprocessor=multiphoton_chevron_postproc,
    job_client=client, use_queue=USE_QUEUE, live_plot=False,
)
multiphoton_rabi_runner = CharacterizationRunner(
    station, LengthRabiGeneralF0g1Experiment, multiphoton_rabi_defaults,
    preprocessor=multiphoton_sideband_preproc, postprocessor=multiphoton_rabi_postproc,
    job_client=client, use_queue=USE_QUEUE, show=False,
)

target_fock_states = expts_to_run.get('multiphoton_target_states')
transitions_to_calibrate = []
if target_fock_states is None:
    for man_mode_no in man_modes:
        for transition_name in expts_to_run['multiphoton_transitions']:
            if man_mode_no != man_modes[0] and transition_name in ('g0-e0', 'e0-f0'):
                continue  # Both modes share the vacuum qubit pulses.
            photon_number_in_selected_mode = int(transition_name.split('-')[0][1:])
            if not 0 <= photon_number_in_selected_mode <= 9:
                raise ValueError(f'Initial photon numbers must be between 0 and 9: {transition_name}')
            initial_m1_photon_number = photon_number_in_selected_mode if man_mode_no == 1 else 0
            initial_m2_photon_number = photon_number_in_selected_mode if man_mode_no == 2 else 0
            transitions_to_calibrate.append(
                (man_mode_no, initial_m1_photon_number, initial_m2_photon_number, transition_name)
            )
else:
    # Fill M1 first. Each requested M1 occupation has its own M2 swaps.
    max_target_m2_photons_by_m1 = {}
    for target_m1_photons, target_m2_photons in target_fock_states:
        if any(isinstance(photon_count, bool) or not isinstance(photon_count, (int, np.integer))
               or not 0 <= photon_count <= 10 for photon_count in (target_m1_photons, target_m2_photons)):
            raise ValueError(f'Target photon numbers must be integers from 0 to 10: {(target_m1_photons, target_m2_photons)}')
        max_target_m2_photons_by_m1[target_m1_photons] = max(
            max_target_m2_photons_by_m1.get(target_m1_photons, 0), target_m2_photons
        )

    man_mode_and_initial_photon_numbers_for_sideband = []
    if 1 in man_modes:
        max_target_m1_photons = max(max_target_m2_photons_by_m1.keys(), default=0)
        for initial_m1_photon_number in range(max_target_m1_photons):
            man_mode_and_initial_photon_numbers_for_sideband.append((1, initial_m1_photon_number, 0))
    if 2 in man_modes:
        for initial_m1_photon_number, target_m2_photons in max_target_m2_photons_by_m1.items():
            for initial_m2_photon_number in range(target_m2_photons):
                man_mode_and_initial_photon_numbers_for_sideband.append(
                    (2, initial_m1_photon_number, initial_m2_photon_number)
                )

    photon_pairs_with_qubit_transitions_scheduled = set()
    for man_mode_no, initial_m1_photon_number, initial_m2_photon_number in man_mode_and_initial_photon_numbers_for_sideband:
        initial_photon_pair = (initial_m1_photon_number, initial_m2_photon_number)
        photon_number_in_selected_mode = initial_m1_photon_number if man_mode_no == 1 else initial_m2_photon_number
        if initial_photon_pair not in photon_pairs_with_qubit_transitions_scheduled:
            ge_transition = f'g{photon_number_in_selected_mode}-e{photon_number_in_selected_mode}'
            ef_transition = f'e{photon_number_in_selected_mode}-f{photon_number_in_selected_mode}'
            transitions_to_calibrate.append(
                (man_mode_no, initial_m1_photon_number, initial_m2_photon_number, ge_transition)
            )
            transitions_to_calibrate.append(
                (man_mode_no, initial_m1_photon_number, initial_m2_photon_number, ef_transition)
            )
            photon_pairs_with_qubit_transitions_scheduled.add(initial_photon_pair)
        sideband_transition = f'f{photon_number_in_selected_mode}-g{photon_number_in_selected_mode + 1}'
        transitions_to_calibrate.append(
            (man_mode_no, initial_m1_photon_number, initial_m2_photon_number, sideband_transition)
        )

# Seed the shared vacuum GE/EF pulses once before the first selected measurement.
if transitions_to_calibrate:
    multiphoton_cfg = station.hardware_cfg.device.multiphoton
    qubit_cfg = station.hardware_cfg.device.qubit
    multiphoton_cfg.pi['gn-en'].frequency[0] = qubit_cfg.f_ge[0]
    multiphoton_cfg.pi['gn-en'].gain[0] = qubit_cfg.pulses.pi_ge.gain[0]
    multiphoton_cfg.pi['en-fn'].frequency[0] = qubit_cfg.f_ef[0]
    multiphoton_cfg.pi['en-fn'].gain[0] = qubit_cfg.pulses.pi_ef.gain[0]

for man_mode_no, initial_m1_photon_number, initial_m2_photon_number, transition_name in transitions_to_calibrate:
    print(f'Calibrating M{man_mode_no} {transition_name} at (M1, M2) = '
          f'({initial_m1_photon_number}, {initial_m2_photon_number})')
    man_mode_and_initial_photon_numbers = dict(
        man_mode_no=man_mode_no,
        initial_m1_photon_number=initial_m1_photon_number,
        initial_m2_photon_number=initial_m2_photon_number,
    )
    if transition_name[0] in ('g', 'e'):
        qubit_transition_key = 'gn-en' if transition_name[0] == 'g' else 'en-fn'
        qubit_frequency_error_amp = multiphoton_error_amp_runner.execute(
            **man_mode_and_initial_photon_numbers, transition_key=qubit_transition_key, parameter_to_test='frequency',
            half_span=1, expts=50, reps=50,
            postprocess=False, go_kwargs=dict(analyze=False, display=False),
        )
        qubit_frequency_error_amp.analyze(state_fin='g', periodic=False)
        qubit_frequency_error_amp.display(fit='fit_avgi' in qubit_frequency_error_amp.data)
        multiphoton_error_amp_postproc(station, qubit_frequency_error_amp)

        qubit_gain_error_amp = multiphoton_error_amp_runner.execute(
            **man_mode_and_initial_photon_numbers, transition_key=qubit_transition_key, parameter_to_test='gain',
            half_span=500, expts=25,
            postprocess=False, go_kwargs=dict(analyze=False, display=False),
        )
        qubit_gain_error_amp.analyze(state_fin='g', periodic=False)
        qubit_gain_error_amp.display(fit='fit_avgi' in qubit_gain_error_amp.data)
        multiphoton_error_amp_postproc(station, qubit_gain_error_amp)
        continue

    if expts_to_run['sideband_pulse_probe_spectroscopy_multiphoton']:
        sideband_frequency_mhz, _, _ = multiphoton_sideband_pulse_parameters(
            station, **man_mode_and_initial_photon_numbers
        )
        sideband_spectroscopy = multiphoton_spec_runner.execute(
            **man_mode_and_initial_photon_numbers,
            start=sideband_frequency_mhz - multiphoton_spec_defaults.step * multiphoton_spec_defaults.expts / 2,
            postprocess=False, go_kwargs=dict(analyze=False, display=False),
        )
        sideband_spectroscopy.data['fit_avgi'] = fit_line(
            sideband_spectroscopy.data['xpts'], sideband_spectroscopy.data['avgi']
        )
        sideband_spectroscopy.display()
        multiphoton_spec_postproc(station, sideband_spectroscopy)

    if expts_to_run['sideband_quick_chevron_and_length_rabi_multiphoton']:
        sideband_frequency_mhz, sideband_gain, _ = multiphoton_sideband_pulse_parameters(
            station, **man_mode_and_initial_photon_numbers
        )
        sideband_chevron = multiphoton_chevron_runner.execute(
            **man_mode_and_initial_photon_numbers,
            sweep_start=sideband_frequency_mhz - 8, sweep_stop=sideband_frequency_mhz + 8,
            sweep_npts=32, gain=sideband_gain, batch=True, analyze=False, postprocess=False, log=False,
        )
        sideband_chevron.analyze(station=station)
        sideband_chevron.display()
        multiphoton_chevron_runner._maybe_log_measurement(sideband_chevron, log=None, render_display=True)
        multiphoton_chevron_postproc(station, sideband_chevron)

        sideband_frequency_mhz, sideband_gain, sideband_pi_length_us = multiphoton_sideband_pulse_parameters(
            station, **man_mode_and_initial_photon_numbers
        )
        preliminary_sideband_length_rabi = multiphoton_rabi_runner.execute(
            **man_mode_and_initial_photon_numbers, freq=sideband_frequency_mhz, gain=sideband_gain,
            start=station.soccfg.cycles2us(3), step=3 * sideband_pi_length_us / multiphoton_rabi_defaults.expts,
            postprocess=False, go_kwargs=dict(analyze=False, display=False),
        )
        preliminary_sideband_length_rabi.analyze()
        preliminary_sideband_length_rabi._length_rabi_analysis.fit = (
            preliminary_sideband_length_rabi.data.get('fit_avgi') is not None and
            preliminary_sideband_length_rabi.data.get('fit_avgq') is not None
        )
        preliminary_sideband_length_rabi.display()
        multiphoton_rabi_postproc(station, preliminary_sideband_length_rabi)

    sideband_frequency_error_amp = multiphoton_error_amp_runner.execute(
        **man_mode_and_initial_photon_numbers, transition_key='fn-gn+1', parameter_to_test='frequency',
        half_span=0.3, expts=50, n_pulses=7, n_step=1,
        postprocess=False, go_kwargs=dict(analyze=False, display=False),
    )
    sideband_frequency_error_amp.analyze(state_fin='e', periodic=False)
    sideband_frequency_error_amp.display(fit='fit_avgi' in sideband_frequency_error_amp.data)
    multiphoton_error_amp_postproc(station, sideband_frequency_error_amp)

    sideband_frequency_mhz, sideband_gain, sideband_pi_length_us = multiphoton_sideband_pulse_parameters(
        station, **man_mode_and_initial_photon_numbers
    )
    final_sideband_length_rabi = multiphoton_rabi_runner.execute(
        **man_mode_and_initial_photon_numbers, freq=sideband_frequency_mhz, gain=sideband_gain,
        start=station.soccfg.cycles2us(3), step=3 * sideband_pi_length_us / multiphoton_rabi_defaults.expts,
        postprocess=False, go_kwargs=dict(analyze=False, display=False),
    )
    final_sideband_length_rabi.analyze()
    final_sideband_length_rabi._length_rabi_analysis.fit = (
        final_sideband_length_rabi.data.get('fit_avgi') is not None and
        final_sideband_length_rabi.data.get('fit_avgq') is not None
    )
    final_sideband_length_rabi.display()
    multiphoton_rabi_postproc(station, final_sideband_length_rabi)

# %% [markdown]
# # Config Update

# %%
station.update_all_station_snapshots(update_main=False)

# %%
# station.hardware_cfg.device.multiphoton.pi['fn-gn+1']['gain'][2] = 10000
