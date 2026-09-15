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
# # Floquet pulse calibration
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 58-60, 69-122 and 150-158 by the stage-2 notebook decomposition. On the
# surface map this is a "prepare and calibrate" section: it is what the MBR
# measurement campaigns need in place before they can run.
#
# The source calibrated the same three quantities twice over, once per pulse
# envelope:
#
# | | flat-top (legacy) | Gaussian / preloaded |
# |---|---|---|
# | Chevron | frequency-length, cells 71-76 | frequency-gain, cells 100-102 |
# | Error amplification | cells 78-83 | cells 107-113 |
# | Phase accumulation | cells 88-91 | cells 115-121 |
#
# The measurement bodies of those two columns were copy-paste duplicates of
# each other. They are now single functions in
# `experiments/qsim/notebook_helpers/floquet_calibration.py`, called twice.
# The module docstring records exactly which cells collapsed into which
# function and, for each, which differences were real and therefore became
# arguments. The largest collapses: four copies of the error-amplification
# loop (cells 81, 83, 109, 113) and five of the phase-accumulation double loop
# (cells 89, 91, 116, 119, 121).
#
# **The defaults dicts deliberately stayed here.** They are what a calibration
# session edits, and the two halves genuinely disagree in them -- cell 107
# applies `floquet_default_dict` to the error-amplification defaults where cell
# 78 set `floquet_waveform` alone -- so folding them together would have merged
# a real difference.
#
# The bare-readout check at the end (cells 150-158) is in
# `floquet_bare_readout.py`. **Read the TODO on `floquet_cycle_to_us` there
# before trusting its time axis:** it is the naive exact-sum cycle duration,
# which `experiments/floquet_timing.py` documents as running ~1.2% long
# against the clock-quantized definition. It is preserved as found rather than
# corrected, because fixing it here would have been a silent physics change.
#
# Setup is `experiments/qsim/notebook_helpers/qsim_session.py` (cells 2-6).
# `ds_floquet` dataset initialization is the tail of
# `multiphoton_calibration.py`, which the surface map puts there.
#
# Its neighbours: `multiphoton_calibration.py`, `mbr.py`, `mbr_disorder.py`,
# `mbr_tomography.py`, `mbr_sff.py`, `floquet_displacement_kerr.py`.

# %%
# %load_ext autoreload
# %autoreload 2

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, SweepRunner

# Imported under the names the relocated cells already use, so their bodies
# did not have to be edited. Definitions are in qsim_session.py.
from experiments.qsim.notebook_helpers.qsim_session import (
    open_session,
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.floquet_calibration import (
    error_amp_floquet_postproc,
    error_amp_floquet_preproc,
    floquet_cycle_list_gen,
    floquet_freq_chev_postproc,
    floquet_freq_chev_preproc,
    floquet_gain_chev_postproc,
    floquet_gain_chev_preproc,
    get_floquet_parameters,
    run_floquet_error_amp_sweep,
    run_freq_chevron_sweep,
    run_gain_chevron_sweep,
    run_phase_accumulation_pairs,
    sideband_stark_error_amp_postproc,
    sideband_stark_error_amp_preproc,
)
from experiments.qsim.notebook_helpers.floquet_bare_readout import (
    flatten_exp_lists,
    plot_bare_scramble,
    run_bare_scramble_sweep,
    sideband_scramble_preproc,
)

# %%
# The config versions this campaign ran against. A scientific choice, so it
# stays written down here rather than defaulting inside open_session.
config_dict = {
    "hardware_config": "CFG-HW-20260904-00019",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260904-00014",
    "floquet_storage_swap": "CFG-FL-20260904-00042",
}

session = open_session(
    user="jonginn",
    experiment_name="260818_qsim_spectroscopy",
    project="EncSpec",
    config_dict=config_dict,
)
station = session.station
client = session.client
db = session.db
config_manager = session.config_manager

# %% [markdown]
# # Calling Floquet parameter helper and optional resetting
#
# `get_floquet_parameters` (source cell 60) is now imported from the helper
# module; the error-amplification preprocessor is its main caller.

# %%
# Review first. Nothing is written to YAML by the cells above.
station.preview_config_update()

# Save a new non-main hardware config only after the plots look acceptable.
# broadband_config_id = station.snapshot_hardware_config(update_main=False)
# print(broadband_config_id)
#
# Put the returned CFG-HW-... in config_dict at the top after restarting.
# Only make it the main config intentionally:
# station.snapshot_hardware_config(update_main=True)

# %% [markdown]
# # Floquet pulse calibrations
#
# ## Freq chevron

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

floquet_freq_chev_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.FloquetChevronExperiment,
    ExptProgram=meas.FloquetChevronProgram,
    default_expt_cfg=floquet_freq_chev_defaults,
    preprocessor=floquet_freq_chev_preproc,
    postprocessor=floquet_freq_chev_postproc,
    job_client=client,
)

# %%
# Source cell 72: one fixed +/-1 MHz span for every mode.
stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))

freq_len_expt = run_freq_chevron_sweep(
    runner=floquet_freq_chev_runner,
    station=station,
    stor_modes=stor_modes_to_run,
    default_span=1.0,
    reps=100,
    relax_delay=200,
    active_reset=True,
    reset_dump_mode=1,
    always_display=True,
)

# %%
for i in range(len(stor_modes_to_run)):
    floquet_freq_chev_postproc(station, freq_len_expt[i])

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ### Fine
#
# Source cell 76. Same loop as above with a per-mode span table.

# %%
stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
freq_span_expt = [ 0.5,  0.3, None, None,  0.2,  0.2,  0.2]

freq_len_expt = run_freq_chevron_sweep(
    runner=floquet_freq_chev_runner,
    station=station,
    stor_modes=stor_modes_to_run,
    freq_spans=freq_span_expt,
    default_span=1.0,
    reps=100,
    relax_delay=200,  # 8000 without active reset
    active_reset=True,
    reset_dump_mode=1,
)

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
    floquet_waveform=floquet_default_dict["floquet_waveform"],
    floquet_hardware_loop=floquet_default_dict["floquet_hardware_loop"],
    scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
)) # Shouldn't be modifying this on the fly!
error_amp_floquet_defaults.update(active_reset_default_dict)
# You can use kwargs in the run function to override these values

error_amp_gain_floquet_coarse_defaults = AttrDict(dict(
    n_pulses=8,
    span=4000,#4000
    expts=30,#30
))

error_amp_freq_floquet_coarse_defaults = AttrDict(dict(
    n_pulses=7,
    span=0.25,
    expts=50,
))

# The preprocessor reads those two span blocks. They are arguments to the hook
# rather than module constants, so they stay editable here; partial binds them
# before the runner ever sees the hook.
error_amp_floquet_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.error_amplification.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_floquet_defaults,
    preprocessor=partial(
        error_amp_floquet_preproc,
        gain_coarse_defaults=error_amp_gain_floquet_coarse_defaults,
        freq_coarse_defaults=error_amp_freq_floquet_coarse_defaults,
    ),
    postprocessor=error_amp_floquet_postproc,
    job_client=client,
)

# %%
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
stor_modes_to_run = [4, 5]

freq_span_list = [0.1, 0.1, None, None, 0.15, 0.1, 0.05] # set for the Coarse
gain_span_list = [None, None, None, None, None, None, None]

# %% [markdown]
# ### Coarse

# %%
error_amp_freq1, error_amp_gain1 = run_floquet_error_amp_sweep(
    runner=error_amp_floquet_runner,
    station=station,
    stor_modes=stor_modes_to_run,
    freq_span_list=freq_span_list,
    gain_span_list=gain_span_list,
    freq_span_default=0.1,
    gain_span_default=0.3,
    span_divisor=1,
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    do_freq_erroramp=True,
    do_gain_erroramp=True,
    active_reset=True,
    relax_delay=200,
)

# %% [markdown]
# ### Fine
#
# Source cell 83: the same loop with both spans halved. Note it also used a
# literal `reset_dump_mode=1` rather than the shared default, which is
# preserved here rather than quietly normalized.

# %%
stor_modes_to_run = [7]

error_amp_freq1, error_amp_gain1 = run_floquet_error_amp_sweep(
    runner=error_amp_floquet_runner,
    station=station,
    stor_modes=stor_modes_to_run,
    freq_span_list=freq_span_list,
    gain_span_list=gain_span_list,
    freq_span_default=0.1,
    gain_span_default=0.3,
    span_divisor=2,
    reset_dump_mode=1,
    do_freq_erroramp=True,
    do_gain_erroramp=True,
    active_reset=True,
    relax_delay=200,
)

# %%
station.update_all_station_snapshots()

# %%
station.ds_floquet.df

# %% [markdown]
# ## Phase accumulation matrix from stark shifts

# %%
phase_expts = [[None for _ in range(7)] for _ in range(7)]

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

sideband_stark_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.SidebandStarkAmplificationExperiment,
    ExptProgram=meas.SidebandStarkAmplificationProgram,
    default_expt_cfg=sideband_stark_error_amp_defaults,
    preprocessor=sideband_stark_error_amp_preproc,
    postprocessor=sideband_stark_error_amp_postproc,
    job_client=client,
)

# %%
# Source cell 89: all ordered pairs among these modes. It did not forward the
# pi/2 buffer flag or scramble_sync_cycles, unlike the Gaussian half below.
stor_modes_to_run = [2, 5] #list(range(1,8))

phase_expts = run_phase_accumulation_pairs(
    runner=sideband_stark_error_amp_runner,
    stor_modes_to=stor_modes_to_run,
    stor_modes_from=stor_modes_to_run,
    advance_phases=np.linspace(-10, 10, 101).tolist(),
    reset_dump_mode=1,
    floquet_settings=floquet_default_dict,
    phase_expts=phase_expts,
    reps=100,
    relax_delay=100,
    active_reset=True,
    forward_pi_half_buffer=False,
    forward_sync_cycles=False,
)

# %%
station.update_all_station_snapshots()

# %%
# Source cell 91: one directed pair only.
phase_expts = run_phase_accumulation_pairs(
    runner=sideband_stark_error_amp_runner,
    stor_modes_to=[5],
    stor_modes_from=[3],
    advance_phases=np.linspace(-10, 10, 101).tolist(),
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    floquet_settings=floquet_default_dict,
    phase_expts=phase_expts,
    reps=100,
    relax_delay=100,
    active_reset=True,
    forward_pi_half_buffer=False,
    forward_sync_cycles=False,
)

# %%
station.snapshot_floquet_storage_swap(update_main=False)

# %%
station.ds_floquet.update_len('M1-S4', 0.065348)

# %%
station.snapshot_floquet_storage_swap()

# %%
station.update_all_station_snapshots()

# %% [markdown]
# # Gaussian/Preloadd floquet pulse calibratons
#
# The same three calibrations again, for the Gaussian/preloaded envelope.

# %%
station.ds_floquet.df

# %% [markdown]
# ## Gain Chevron

# %%
floquet_gain_chev_defaults = AttrDict(dict(
    expts = 1,
    reps = 100,
    rounds = 1,
    qubits = [0],
    ro_stor = 0, # storage mode number that gets read out in the end
    f0g1_cavity = 1,  #  1/2 name of manipulate cavity
    # if 0, this means to read out man instead
    detunes=np.linspace(-0.3, 0.3, 11).tolist(),
    swept_params = ['detune', 'gain'],
    normalize = False,
    active_reset = False,
    man_reset = False,
    storage_reset = False,
    prepulse=True,
    postpulse=True,
    init_fock=True,
)) # Shouldn't be modifying this on the fly!
floquet_gain_chev_defaults.update(active_reset_default_dict)
floquet_gain_chev_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

# %%
from experiments.qsim.floquet_gain_chevron import FloquetGainChevronExperiment
from experiments.qsim.floquet_gain_chevron import FloquetGainChevronProgram

stor_modes_to_run = [4] #list(range(1,8))
you_have_to_do_amp_chev = True
if you_have_to_do_amp_chev == True:
    # Note the postprocessor is intentionally not attached here; cell 102
    # below applies it by hand after inspecting the result.
    floquet_gain_chev_runner = CharacterizationRunner(
        station=station,
        ExptClass=FloquetGainChevronExperiment,
        ExptProgram=FloquetGainChevronProgram,
        default_expt_cfg=floquet_gain_chev_defaults,
        preprocessor=floquet_gain_chev_preproc,
        # postprocessor=floquet_gain_chev_postproc,
        job_client=client,
    )

    freq_len_expt = run_gain_chevron_sweep(
        runner=floquet_gain_chev_runner,
        station=station,
        stor_modes=stor_modes_to_run,
        detune_span=0.5,
        reps=50,
        gain_expts=21,
        max_gain=14000,
        relax_delay=200,  # 8000 without active reset
        active_reset=True,
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
        debug=False,
    )

# %%
floquet_gain_chev_postproc(station, freq_len_expt[0])

# %%
# station.ds_floquet.update_gain("M1-S6", 13648)
station.ds_floquet.update_gain("M1-S2", 3700)
# station.ds_floquet.update_freq("M1-S4", 878.2990264499939)

# station.ds_floquet.update_freq("M1-S4", 873.5242589230924)

# %%
station.ds_floquet.df

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## Error Amplification for Gaussian pulses
#
# Source cell 107 redefined the defaults and both hooks. The hooks were
# byte-identical to cell 78's, so only the defaults are rebuilt here -- and
# they do differ: this block applies the whole `floquet_default_dict` where
# cell 78 set `floquet_waveform` alone.

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
    scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
)) # Shouldn't be modifying this on the fly!
error_amp_floquet_defaults.update(active_reset_default_dict)
error_amp_floquet_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

error_amp_gain_floquet_coarse_defaults = AttrDict(dict(
    n_pulses=8,
    span=4000,#4000
    expts=30,#30
))

error_amp_freq_floquet_coarse_defaults = AttrDict(dict(
    n_pulses=7,
    span=0.25,
    expts=50,
))

error_amp_floquet_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.single_qubit.error_amplification.ErrorAmplificationExperiment,
    default_expt_cfg=error_amp_floquet_defaults,
    preprocessor=partial(
        error_amp_floquet_preproc,
        gain_coarse_defaults=error_amp_gain_floquet_coarse_defaults,
        freq_coarse_defaults=error_amp_freq_floquet_coarse_defaults,
    ),
    postprocessor=error_amp_floquet_postproc,
    job_client=client,
)

# %%
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
# stor_modes_to_run = [4, 5, 6, 7] #list(range(1,8))

stor_modes_to_run = [1, 2, 3, 4, 5, 7]

freq_span_list = [0.1,  0.07, 0.07, 0.1,  0.07,   0.1, 0.05] # set for the Coarse
gain_span_list = [None, None, None, None,  0.25,  0.25, 0.25]

# %%
# Source cell 109. Unlike the flat-top coarse pass, this one derives
# relax_delay from whether active reset is on.
do_active_reset = True
relax_delay = 200
if not do_active_reset:
    relax_delay = 8000

stor_modes_to_run = [4]

error_amp_freq1, error_amp_gain1 = run_floquet_error_amp_sweep(
    runner=error_amp_floquet_runner,
    station=station,
    stor_modes=stor_modes_to_run,
    freq_span_list=freq_span_list,
    gain_span_list=gain_span_list,
    freq_span_default=0.1,
    gain_span_default=0.3,
    span_divisor=1,
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    do_freq_erroramp=True,
    do_gain_erroramp=True,
    active_reset=do_active_reset,
    relax_delay=relax_delay,
)

# %%
# station.ds_floquet.update_freq("M1-S4", 878.27)

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## Fine Error Amplification for Gaussian pulses
#
# Source cell 113: spans halved, and it reuses whatever `stor_modes_to_run`
# the cell above left bound.

# %%
error_amp_freq1, error_amp_gain1 = run_floquet_error_amp_sweep(
    runner=error_amp_floquet_runner,
    station=station,
    stor_modes=stor_modes_to_run,
    freq_span_list=freq_span_list,
    gain_span_list=gain_span_list,
    freq_span_default=0.1,
    gain_span_default=0.3,
    span_divisor=2,
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    do_freq_erroramp=True,
    do_gain_erroramp=True,
    active_reset=True,
    relax_delay=200,
)

# %% [markdown]
# ## Phase Accumulation

# %%
phase_expts = [[None for _ in range(7)] for _ in range(7)]

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

sideband_stark_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.SidebandStarkAmplificationExperiment,
    # ExptProgram=meas.qsim.floquet_dark_mode_readout.SidebandStarkAmplificationModifiedProgram,
    ExptProgram=meas.SidebandStarkAmplificationProgram, #meas.SidebandStarkAmplificationProgram,
    default_expt_cfg=sideband_stark_error_amp_defaults,
    preprocessor=sideband_stark_error_amp_preproc,
    postprocessor=sideband_stark_error_amp_postproc,
    job_client=client,
)

# %%
# Source cell 116. From here on the buffer flag and sync cycles are forwarded.
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
stor_modes_to_run = [1, 2, 3, 4] #list(range(1,8))

do_active_reset = True
relax_delay = 200
if not do_active_reset:
    relax_delay = 8000

phase_expts = run_phase_accumulation_pairs(
    runner=sideband_stark_error_amp_runner,
    stor_modes_to=stor_modes_to_run,
    stor_modes_from=stor_modes_to_run,
    advance_phases=np.linspace(-10, 10, 51).tolist(),
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    floquet_settings=floquet_default_dict,
    phase_expts=phase_expts,
    reps=100,
    relax_delay=relax_delay,
    active_reset=do_active_reset,
)

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ### Redo or add calibration
#
# Source cell 119: one directed pair, on a wider and finer phase grid.

# %%
phase_expts = run_phase_accumulation_pairs(
    runner=sideband_stark_error_amp_runner,
    stor_modes_to=[2],
    stor_modes_from=[6],
    advance_phases=np.linspace(-30, 30, 301).tolist(),
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    floquet_settings=floquet_default_dict,
    phase_expts=phase_expts,
    reps=100,
    relax_delay=100,
    active_reset=True,
)

# %% [markdown]
# ### Add modes
#
# Source cell 121: measure only the pairs that involve a newly added mode,
# skipping those where both modes were already calibrated.

# %%
stor_modes_to_add = [2] #list(range(1,8))
pre_existing_store_modes = [4, 5, 6]

total_stor_modes = list(set(pre_existing_store_modes + stor_modes_to_add))

phase_expts = run_phase_accumulation_pairs(
    runner=sideband_stark_error_amp_runner,
    stor_modes_to=total_stor_modes,
    stor_modes_from=total_stor_modes,
    advance_phases=np.linspace(-20, 20, 101).tolist(),
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    floquet_settings=floquet_default_dict,
    phase_expts=phase_expts,
    reps=100,
    relax_delay=100,
    active_reset=True,
    pre_existing_modes=pre_existing_store_modes,
)

# %%
station.update_all_station_snapshots()

# %% [markdown]
# # Bare dark-mode readout check
#
# Source cells 150-158, the "Bare" subsection. This is the prerequisite check
# the MBR campaigns need; the displace/multiparity dark-mode work that follows
# it in the source (cells 159-262) is dormant, in
# `dormant/dark_mode.py`.

# %%
dm_sideband_scramble_defaults = AttrDict(dict(
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
dm_sideband_scramble_defaults.update(active_reset_default_dict)
dm_sideband_scramble_defaults.update(floquet_default_dict)
# You can use kwargs in the run function to override these values

dmscramble_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.QsimBaseExperiment,
    ExptProgram=meas.SidebandScrambleDarkProgramNewNew,
    default_expt_cfg=dm_sideband_scramble_defaults,
    preprocessor=sideband_scramble_preproc,
    postprocessor=None,
    job_client=client,
)

# %%
floquet_cycles_list = floquet_cycle_list_gen(0, 2000, 2000, 15)

swap_stors = [1, 2, 3, 4]
# meas_stors = [0, 2, 6]
# meas_stors = [0]
meas_stors = [0] + swap_stors
# meas_stors = [0, 1, 2, 3, 4]
dark_swaps = [4, 5]

scramble_expts = run_bare_scramble_sweep(
    runner=dmscramble_runner,
    station=station,
    floquet_settings=floquet_default_dict,
    active_reset_settings=active_reset_default_dict,
    swap_stors=swap_stors,
    meas_stors=meas_stors,
    floquet_cycles_list=floquet_cycles_list,
    dark_swaps=dark_swaps,
    reps=300,
    active_reset=True,
)

# %%
fname_list = []
for exp in flatten_exp_lists(scramble_expts):
    fname_list.append(exp.fname)

# %%
# The time axis comes from floquet_cycle_to_us. See the TODO on that function:
# it is the naive exact-sum cycle duration, ~1.2% long against
# experiments/floquet_timing.floquet_cycle_us. Pass plot_time=False for a
# Floquet-cycle axis that does not depend on it.
fig, ax, combined_data, cycle_us = plot_bare_scramble(
    scramble_expts,
    station=station,
    fname_list=fname_list,
    plot_time=True,
    plot_q=False,
    scatter=False,
)

# %%
station.update_all_station_snapshots()
