# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Multimode (direct remote)
#     language: python
#     name: multimode-direct
# ---

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
from experiments import MultimodeStation, CharacterizationRunner, SweepRunner

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
user = 'guan'

print(f"Welcome {user}!")

# %% editable=true slideshow={"slide_type": ""}
# Initialize station to retrieve soc and configs
config_dict = {'hardware_config': 'CFG-HW-20260726-00021',
 'multiphoton_config': 'CFG-MP-20260121-00001',
 'man1_storage_swap': 'CFG-M1-20260726-00010',
 'floquet_storage_swap': 'CFG-FL-20260722-00001'}


station = MultimodeStation(
    user = user,
    experiment_name = "260728_qsim",
    project = "ManyBodyRamsey",
    log_measurements=True,
    
    storage_man_file = config_dict['man1_storage_swap'],
    hardware_config= config_dict['hardware_config'],
    floquet_file=config_dict['floquet_storage_swap'],
    

    # storage_man_file="CFG-M1-20260218-00025",
    # hardware_config="CFG-HW-20260211-00039",
    # floquet_file="versions/floquet_storage_swap/CFG-FL-20260520-00002.csv",
    # multiphoton_config="versions/multiphoton_config/CFG-MP-20260115-00001.yml",
)


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

def singleshot_postproc(station, expt):
    expt.analyze(plot=False, station=station, subdir=station.autocalib_path)
    fids = expt.data['fids']
    confusion_matrix = expt.data['confusion_matrix']
    thresholds_new = expt.data['thresholds']
    angle = expt.data['angle']
    print(fids)

    hardware_cfg = station.hardware_cfg
    hardware_cfg.device.readout.phase = [hardware_cfg.device.readout.phase[0] + angle]
    hardware_cfg.device.readout.threshold = thresholds_new
    hardware_cfg.device.readout.threshold_list = [thresholds_new]
    hardware_cfg.device.readout.Ie = [np.median(expt.data['Ie_rot'])]
    hardware_cfg.device.readout.Ig = [np.median(expt.data['Ig_rot'])]
    if expt.cfg.expt.active_reset:
        hardware_cfg.device.readout.confusion_matrix_with_active_reset = confusion_matrix
    else:
        hardware_cfg.device.readout.confusion_matrix_without_reset = confusion_matrix
    print('Updated readout!')


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

ss = ss_runner.execute(
    check_f=False,
    active_reset=False, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
    relax_delay=2000,
    # active_reset=True,
    # relax_delay=200,
    priority=1,
)
ss.display(station)


# %% [markdown]
# # N-photon Hamiltonian spec with encoding pulse calibration under entire Floquet cycle
#
# Run calibration, analyzer sign check, and spectroscopy in order.

# %%
# 1. Calibrate the phase of each occupation.
from itertools import product


EncSpec = meas.EncodingHamiltonianSpectroscopyExperiment


encspec_modes = [4, 5, 6, 7]
encspec_mode_labels = ['M1'] + [f'S{stor}' for stor in encspec_modes]

# Build the complete fixed-N sector once.
encspec_N = 3
encspec_sector_occupations = [
    list(occupation)
    for occupation in product(
        range(encspec_N + 1),
        repeat=len(encspec_mode_labels),
    )
    if sum(occupation) == encspec_N
]
encspec_sector_occupations.sort(reverse=True)

# Edit this short list for a POC. These four states test distinct loading paths.
encspec_poc_occupations = [
    [2, 0, 0, 0, 1],
    [3, 0, 0, 0, 0],
    # [0, 3, 0, 0, 0],
    # [0, 0, 3, 0, 0],
    # [0, 0, 0, 3, 0],
]

# Change only this flag to run the complete fixed-N sector.
encspec_use_full_sector = False
encspec_occupations = [
    list(occupation)
    for occupation in (
        encspec_sector_occupations
        if encspec_use_full_sector
        else encspec_poc_occupations
    )
]

encspec_cycle_pairs = np.arange(0, 17, dtype=int)
encspec_physical_cycle_counts = 2 * encspec_cycle_pairs
encspec_theta_list = np.array([0., 180.])
encspec_phi_list = np.array([0., 90.])
encspec_hardware_loop = False
encspec_scramble_sync_cycles = 10
encspec_priority = 0


encspec_defaults = AttrDict(dict(
    expts=1,
    reps=300,
    rounds=1,
    qubits=[0],
    normalize=False,
    active_reset=True,
    man_reset=True,
    storage_reset=encspec_modes,
    pre_relax_delay=100,
    relax_delay=200,
    reset_dump_mode=2,
    dump_reset_iter_num=1,
    use_qubit_man_reset=False,
    prepulse=False,
    postpulse=False,
    init_fock=False,
    perform_wigner=False,
    parity_readout=False,
    multiparity_readout=False,
    load_man_dark=False,
    swap_man_dark=False,
    swap_man_large_dark=False,
    update_phases=True,
    floquet_cycle=0,
    palindrome_scramble=False,
    scramble_sync_cycles=encspec_scramble_sync_cycles,
    floquet_hardware_loop=encspec_hardware_loop,
    swap_stors=encspec_modes,
    detunings=np.zeros(len(encspec_modes)).tolist(),
    spectroscopy_occupations=encspec_occupations[0],
    spectroscopy_analyzer_phase=0.,
    final_analyzer_phase_per_cycle_deg=0.,
    n_cycle_pairs=encspec_cycle_pairs.tolist(),
    spectroscopy_prep_phases=encspec_theta_list.tolist(),
    swept_params=['n_cycle_pair', 'spectroscopy_prep_phase'],
))

encspec_cycle_branches = [0] * len(encspec_occupations)  # 180-deg/cycle branch
encspec_correction_sign = 1.  # overwritten by the sign-check cell
encspec_hardware = EncSpec.hardware_parameters(
    station, 
    encspec_modes, 
    encspec_scramble_sync_cycles,
    encspec_defaults.get('floquet_gauss_sigma', None),
)

calibration_batch = EncSpec.calibration_batch(
    encspec_defaults, 
    encspec_modes, 
    encspec_occupations, 
    encspec_cycle_pairs,
    sync_cycles=encspec_scramble_sync_cycles, 
    repeats=1, 
    reps=500,
)
calibration_runner = meas.BatchRunner(
    station=station, 
    ExptClass=EncSpec,
    ExptProgram=meas.EntireFloquetCyclePhaseCalibrationProgram,
    default_expt_cfg=calibration_batch.default_expt_cfg,
    job_client=client, 
    show=False,
)
calibration_expt = calibration_runner.execute(
    calibration_batch.configs, 
    batch_size=10, 
    log=True, 
    show=False,
)
calibration_expt.analyze(
    stage='calibration', occupations=encspec_occupations,
    cycle_pairs=encspec_cycle_pairs, repeats=calibration_batch.repeats,
)

# %%
calibration_expt.display()

plt.show()

# %% [markdown]
# ## 2. Final-analyzer sign check
#
# Run this after calibration to test both correction signs on one occupation. This does not resolve the 180-deg/cycle branch.

# %%
# Test which analyzer sign makes the measured phase slope closest to zero.
encspec_sign_occupation = encspec_occupations[0]  # choose a state with good contrast and nonzero phase
encspec_sign_index = encspec_occupations.index(encspec_sign_occupation)
encspec_sign_raw_phase = calibration_expt.data.phase_mod180[encspec_sign_index]
encspec_sign_residual_phase = {}

for correction_sign in [1., -1.]:
    sign_batch = EncSpec.calibration_batch(
        encspec_defaults, encspec_modes, [encspec_sign_occupation], encspec_cycle_pairs,
        sync_cycles=encspec_scramble_sync_cycles, repeats=1, reps=1500,
    )
    correction_per_cycle = correction_sign * encspec_sign_raw_phase
    for config in sign_batch.configs:
        config['final_analyzer_phase_per_cycle_deg'] = correction_per_cycle

    sign_expt = calibration_runner.execute(
        sign_batch.configs, batch_size=2, log=True, show=False,
    )
    sign_expt.analyze(
        stage='calibration', occupations=[encspec_sign_occupation],
        cycle_pairs=encspec_cycle_pairs, repeats=sign_batch.repeats,
    )
    residual_phase = sign_expt.data.phase_mod180[0]
    encspec_sign_residual_phase[correction_sign] = residual_phase
    print('correction sign:', correction_sign, 'residual phase:', residual_phase, 'deg / cycle')

if abs(encspec_sign_residual_phase[1.]) <= abs(encspec_sign_residual_phase[-1.]):
    encspec_correction_sign = 1.
else:
    encspec_correction_sign = -1.
print('selected correction sign:', encspec_correction_sign)


# %% [markdown]
# ## 3. N-photon spectroscopy

# %%
def floquet_cycle_list_gen(start, 
                           stop, 
                           chunk,
                           step = 1):
    _to_return = []
    _q = (stop-start)//chunk
    _start_idx = start
    for _ in range(_q):
        _to_return.append(np.arange(_start_idx, 
                                    _start_idx+chunk,
                                    step))
        _start_idx += chunk
    if _start_idx < stop:
        
        _to_return.append(np.arange(_start_idx, 
                                    stop, 
                                    step))
    return _to_return


# %%
# Build the Kerr-preserving correction, acquire spectroscopy, analyze, and display.
encspec_cycle_chunks = floquet_cycle_list_gen(0, 400, 400, 20)
encspec_detunings = [0.] * len(encspec_modes)
encspec_correction = EncSpec.build_phase_correction(
    encspec_occupations, calibration_expt.data.phase_mod180,
    encspec_cycle_branches, encspec_hardware.physical_kerr_MHz,
    encspec_hardware.floquet_cycle_us, correction_sign=encspec_correction_sign,
)

spectroscopy_batch = EncSpec.spectroscopy_batch(
    encspec_defaults, encspec_modes, encspec_occupations, encspec_cycle_chunks,
    encspec_correction.phase_by_occupation, detunings=encspec_detunings,
    sync_cycles=encspec_scramble_sync_cycles, reps=300,
)
spectroscopy_runner = floquet_dark_mode_readout.BatchRunner(
    station=station, 
    ExptClass=EncSpec,
    ExptProgram=floquet_dark_mode_readout.NPhotonHamiltonianSpectroscopyProgram,
    default_expt_cfg=spectroscopy_batch.default_expt_cfg,
    job_client=client, show=False,
)
spectroscopy_expt = spectroscopy_runner.execute(
    spectroscopy_batch.configs, batch_size=8, log=True, show=False,
)
spectroscopy_expt.analyze(
    stage='spectrum', occupations=encspec_occupations,
    photon_number=encspec_N,
    detunings=encspec_detunings, couplings_MHz=encspec_hardware.couplings_MHz,
    floquet_cycle_us=encspec_hardware.floquet_cycle_us,
    physical_kerr_MHz=encspec_hardware.physical_kerr_MHz,
    fft_window='raw', zero_padding=1,
    calibration=calibration_expt.data, correction=encspec_correction,
    mode_labels=['M1'] + [f'S{stor}' for stor in encspec_modes],
)
spectroscopy_expt.calibration_job_ids = calibration_expt.batch_job_ids

# %%
spectroscopy_expt.display()
plt.show()

# %%
