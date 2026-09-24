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
# # Multiphoton calibration
#
# Split out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
# 7-57 and 61-68 by the stage-2 notebook decomposition. On the surface map this
# is one of the "prepare and calibrate" sections: it is what the MBR
# measurement campaigns need in place before they can run.
#
# Three things in order:
#
# 1. **Broadband qubit ge calibration.** Makes a separate `pi_ge_broadband`;
#    the ordinary `pi_ge` and `hpi_ge` are untouched.
# 2. **N-photon M1-storage full-swap calibration.** One direct
#    $|N,0\rangle \leftrightarrow |0,N\rangle$ swap row at a time.
# 3. **Floquet dataset setup and single-shot readout calibration.** Cells
#    61-68 of the source, which the map assigns here.
#
# Long procedural cells moved to
# `experiments/qsim/notebook_helpers/multiphoton_calibration.py`. That module's
# docstring records which source cells each function came from, including four
# places where the source had copy-paste duplicate cells differing only in a
# scan width -- coarse/fine frequency and coarse/fine gain -- which are now one
# function called twice.
#
# **What deliberately stayed here:** every accept cell. Deciding whether to
# take a fitted candidate into `ds_storage` or into the `pi_ge_broadband`
# config after looking at a plot is the scientific choice, so it is still
# written out cell by cell. None of the helper functions writes to the station.
#
# The defaults of source cells 2-6 are in
# `experiments/qsim/notebook_helpers/defaults.py`.
#
# Its neighbours: `floquet_calibration.py`, `mbr.py`, `mbr_disorder.py`,
# `mbr_tomography.py`, `mbr_sff.py`, `floquet_displacement_kerr.py`.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from copy import deepcopy

import experiments as meas
from slab import AttrDict
from experiments import CharacterizationRunner, SweepRunner, MultimodeStation

from job_server import JobClient
# Imported under the names the relocated cells already use.
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.run_mode import run_settings

# Set by tools/run_qsim_suite.py. Unset: through the queue in the main
# checkout, directly on this kernel in a worktree (see run_mode.py).
RUN = run_settings()
from experiments.qsim.notebook_helpers.multiphoton_calibration import (
    analyze_swap_chevron,
    broadband_amprabi_preproc,
    build_swap_pulse_sequences,
    even_gain_grid,
    fit_broadband_frequency,
    broadband_gain_grid,
    fit_broadband_gain,
    plot_broadband_rabi_transfer,
    plot_broadband_validation,
    plot_iq_endpoints,
    score_return_error,
    singleshot_postproc,
)

# %%
# The config versions this campaign ran against.
config_dict = {
    "hardware_config": "CFG-HW-20260904-00019",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260904-00014",
    "floquet_storage_swap": "CFG-FL-20260904-00042",
}

station = MultimodeStation(
    user="jonginn",
    experiment_name="260818_qsim_spectroscopy",
    project="EncSpec",
    log_measurements=not RUN.smoke,
    **RUN.station_configs(config_dict),
)
client = JobClient()

# %%
station.ds_storage.df

# %% [markdown]
# # Broadband qubit ge calibration
#
# This makes a separate `pi_ge_broadband`; the ordinary `pi_ge` and `hpi_ge` are not changed.
#
# Run the cells in order: add the runtime config row, run the same amplitude-Rabi sweep at (n=0,1,2,3), choose one common gain, run the exact pulse-path validation, and then check zero-photon phase accumulation. The plotted values are population-transfer coordinates obtained from each (n)'s own g/e IQ references, not leakage-resolved gate fidelities.
#
# Start with the ordinary `pi_ge` sigma. If the pi points separate with photon number, shorten `broadband_sigma` and repeat. Save a config snapshot only after inspecting the plots.

# %% [markdown]
# ## Resetting Config files

# %%
from experiments.qsim.floquet_dark_mode_readout import (
    DarkBaseExperiment,
    BroadbandGeValidationProgram,
)


broadband_gain_limit = 30000
photon_numbers = [0]

RESET_GE_BROADBAND = False
SET_SIGMA_FRACTION = 3

print(station.hardware_cfg.device.qubit.pulses.pi_ge_broadband)

# %%

ge_frequencies = np.asarray(
    station.hardware_cfg.device.multiphoton.pi['gn-en'].frequency[:4],
    dtype=float,
)

pi_ge = station.hardware_cfg.device.qubit.pulses.pi_ge
broadband_frequency = 0.5 * (
    ge_frequencies.min() + ge_frequencies.max()
)
broadband_sigma = float(pi_ge.sigma[0] / SET_SIGMA_FRACTION)
broadband_gain = int(np.clip(
    round(pi_ge.gain[0] * SET_SIGMA_FRACTION), 0, broadband_gain_limit
))

# If the four pi points do not overlap, shorten this and rerun from here.
# broadband_sigma = 0.020
if RESET_GE_BROADBAND:
    station.hardware_cfg.device.qubit.pulses['pi_ge_broadband'] = AttrDict(dict(
        frequency=[broadband_frequency],
        gain=[broadband_gain],
        sigma=[broadband_sigma],
        length=[0.0],
        type=['gauss'],
    ))
    print('gN-eN frequencies:', ge_frequencies)
    print('broadband frequency:', broadband_frequency)

station.hardware_cfg.device.qubit.pulses.pi_ge_broadband

# %%
# Define the amplitude-Rabi runner. `broadband_amprabi_preproc` is now in the
# helper module.
# =====================================
broadband_amprabi_defaults = AttrDict(dict(
    start=0,
    step=RUN.pick(200, smoke=1000),
    expts=RUN.pick(151, smoke=31),       # gain = 0 ... 30000
    reps=RUN.pick(200, smoke=100),
    rounds=1,
    sigma_test=broadband_sigma,
    qubit=0,
    qubits=[0],
    pulse_type='gauss',
    flat_length=0,
    user_defined_freq=[True, broadband_frequency],
    checkZZ=False,
    checkEF=False,
    pulse_ge_init=False,
    pulse_ge_after=False,
    normalize=False,
    single_shot=False,
    prepulse=False,
    pre_sweep_pulse=[],
    postpulse=False,
    post_sweep_pulse=[],
    gate_based=True,
    active_reset=False,
    relax_delay=2500,
))

broadband_amprabi_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.AmplitudeRabiExperiment,
    default_expt_cfg=broadband_amprabi_defaults,
    preprocessor=broadband_amprabi_preproc,
    job_client=client,
    use_queue=RUN.use_queue,
)

# %%
# Execute amplitude Rabi from |g,n> and |e,n> (two jobs per photon number).
# These are pure-state preparations, so conditional shelving is not needed.
broadband_rabi_from_g = [None] * len(photon_numbers)
broadband_rabi_from_e = [None] * len(photon_numbers)

for photon_number in photon_numbers:
    prep_gN = []
    for n in range(photon_number):
        prep_gN += [
            ['multiphoton', f'g{n}-e{n}', 'pi', 0.0],
            ['multiphoton', f'e{n}-f{n}', 'pi', 0.0],
            ['multiphoton', f'f{n}-g{n + 1}', 'pi', 0.0],
        ]
    prep_eN = prep_gN + [[
        'multiphoton', f'g{photon_number}-e{photon_number}', 'pi', 0.0
    ]]

    print(f'Running broadband Rabi from |g,{photon_number}>')
    broadband_rabi_from_g[photon_number] = broadband_amprabi_runner.execute(
        sigma_test=broadband_sigma,
        user_defined_freq=[True, broadband_frequency],
        pre_sweep_pulse=prep_gN,
        show=True,
        log=True,
    )

    print(f'Running broadband Rabi from |e,{photon_number}>')
    broadband_rabi_from_e[photon_number] = broadband_amprabi_runner.execute(
        sigma_test=broadband_sigma,
        user_defined_freq=[True, broadband_frequency],
        pre_sweep_pulse=prep_eN,
        show=True,
        log=True,
    )

# %%
# Put every photon number on its own g/e readout axis.
gain, g_to_e, e_to_g = plot_broadband_rabi_transfer(
    broadband_rabi_from_g, broadband_rabi_from_e, photon_numbers
)

# %%
# Choose one gain where all eight curves above are near 1.
# The N=0 fit is only a starting value; overwrite it after looking at the plot.
broadband_gain = int(np.clip(
    round(broadband_rabi_from_g[0].data['pi_gain_avgi']),
    0,
    broadband_gain_limit,
))
# broadband_gain = 12345

station.hardware_cfg.device.qubit.pulses.pi_ge_broadband.gain = [
    broadband_gain
]
print('pi_ge_broadband gain:', broadband_gain)
station.hardware_cfg.device.qubit.pulses.pi_ge_broadband

# %% [markdown]
# ## Error Amplification

# %%
from experiments.single_qubit.error_amplification import (
    ErrorAmplificationExperiment,
)

bb = station.hardware_cfg.device.qubit.pulses.pi_ge_broadband


broadband_error_amp_defaults = AttrDict(dict(
    reps=50,
    rounds=1,
    qubit=0,
    qubits=[0],
    n_start=1,
    n_step=1,
    n_pulses=10,
    active_reset=False,
    man_reset=True,
    storage_reset=True,
    relax_delay=2500,
    parameter_to_test='frequency',
    pulse_type=['qubit', 'ge_broadband', 'pi', 0.0],
))

broadband_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=ErrorAmplificationExperiment,
    default_expt_cfg=broadband_error_amp_defaults,
    job_client=client,
    use_queue=RUN.use_queue,
    show=False,
)


# %% [markdown]
# ### Coarse
#
# Source cells 18 and 19. They are the same bodies as the fine pair below,
# differing only in scan width, so both pairs now call one function each.

# %%
bb_freq_center = float(bb.frequency[0])
bb_freq_band = 5.0
bb_freq_points = 51

bb_freq_error_amp = broadband_error_amp_runner.execute(
    parameter_to_test='frequency',
    start=bb_freq_center - bb_freq_band,
    step=2 * bb_freq_band / (bb_freq_points - 1),
    expts=bb_freq_points,
    reps=50,
    n_pulses=10,
    pulse_type=['qubit', 'ge_broadband', 'pi', 0],
    postprocess=False,
    show=False,
    log=True,
)

# %%
best_frequency, best_frequency_err = fit_broadband_frequency(
    bb_freq_error_amp, center=bb_freq_center
)
# Accept the coarse frequency fit.
bb.frequency[0] = best_frequency

# %%
bb_gain_center = int(round(float(bb.gain[0])))
bb_gain_start, bb_gain_stop, bb_gain_step = broadband_gain_grid(
    current_gain=bb_gain_center,
    half_band=3000,
    gain_limit=broadband_gain_limit,
    points=26,
)
print(f'broadband gain scan: {bb_gain_start} ... {bb_gain_stop} '
      f'(step {bb_gain_step}, limit {broadband_gain_limit})')

bb_gain_error_amp = broadband_error_amp_runner.execute(
    parameter_to_test='gain',
    start=bb_gain_start,
    step=bb_gain_step,
    expts=26,
    reps=50,
    n_pulses=10,
    pulse_type=['qubit', 'ge_broadband', 'pi', 0],
    postprocess=False,
    show=False,
    # Source cell 19 passed log=False here, deferring the log until
    # after the refit; cell 22 logged immediately.
    log=False,
)

# %%
best_gain, best_gain_err, fitted_gain = fit_broadband_gain(
    bb_gain_error_amp,
    current_gain=bb_gain_center,
    gain_start=bb_gain_start,
    gain_stop=bb_gain_stop,
)
# Accept the coarse gain fit.
bb.gain[0] = best_gain

# %% [markdown]
# ### Fine

# %% tags=["suite-skip"]
# Suite: skipped. A second pass of the coarse scan above, narrower.
bb_freq_center = float(bb.frequency[0])
bb_freq_band = 1.0
bb_freq_points = 51

bb_freq_error_amp = broadband_error_amp_runner.execute(
    parameter_to_test='frequency',
    start=bb_freq_center - bb_freq_band,
    step=2 * bb_freq_band / (bb_freq_points - 1),
    expts=bb_freq_points,
    reps=50,
    n_pulses=10,
    pulse_type=['qubit', 'ge_broadband', 'pi', 0],
    postprocess=False,
    show=False,
    log=True,
)

# %% tags=["suite-skip"]
best_frequency, best_frequency_err = fit_broadband_frequency(
    bb_freq_error_amp, center=bb_freq_center
)
# Accept the fine frequency fit.
bb.frequency[0] = best_frequency

# %% tags=["suite-skip"]
# Suite: skipped. A second pass of the coarse scan above, narrower.
bb_gain_center = int(round(float(bb.gain[0])))
bb_gain_start, bb_gain_stop, bb_gain_step = broadband_gain_grid(
    current_gain=bb_gain_center,
    half_band=1500,
    gain_limit=broadband_gain_limit,
    points=26,
)
print(f'broadband gain scan: {bb_gain_start} ... {bb_gain_stop} '
      f'(step {bb_gain_step}, limit {broadband_gain_limit})')

bb_gain_error_amp = broadband_error_amp_runner.execute(
    parameter_to_test='gain',
    start=bb_gain_start,
    step=bb_gain_step,
    expts=26,
    reps=50,
    n_pulses=10,
    pulse_type=['qubit', 'ge_broadband', 'pi', 0],
    postprocess=False,
    show=False,
    log=True,
)

# %% tags=["suite-skip"]
best_gain, best_gain_err, fitted_gain = fit_broadband_gain(
    bb_gain_error_amp,
    current_gain=bb_gain_center,
    gain_start=bb_gain_start,
    gain_stop=bb_gain_stop,
)
# Accept the fine gain fit.
bb.gain[0] = best_gain

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## Validation

# %%
# Exact-path validation
# case 0: |g,n> reference
# case 1: |e,n> reference
# case 2: |g,n> --pi_ge_broadband--> |e,n>
# case 3: |e,n> --pi_ge_broadband--> |g,n>
broadband_validation_cases = [0, 1, 2, 3]

broadband_validation_defaults = AttrDict(dict(
    expts=1,
    reps=500,
    rounds=1,
    qubits=[0],
    swept_params=['validation_photon_number', 'validation_case'],
    validation_photon_numbers=photon_numbers,
    validation_cases=broadband_validation_cases,
    validation_photon_number=0,
    validation_case=0,
    active_reset=False,
    pre_relax_delay=0,
    relax_delay=2500,
    normalize=False,
    perform_wigner=False,
    dedupe_waveforms=True,
    prepulse=False,
    postpulse=False,
    init_stor=0,
    ro_stor=0,
))

broadband_validation_runner = CharacterizationRunner(
    station=station,
    ExptClass=DarkBaseExperiment,
    ExptProgram=BroadbandGeValidationProgram,
    default_expt_cfg=broadband_validation_defaults,
    job_client=client,
    use_queue=RUN.use_queue,
)

# %%
# Execute and inspect the configured ['qubit', 'ge_broadband', 'pi', phase]
broadband_validation = broadband_validation_runner.execute(
    reps=RUN.pick(500, smoke=100),
    show=False,
    log=True,
)

# %%
validation_transfer = plot_broadband_validation(
    broadband_validation, photon_numbers, broadband_validation_cases
)

# %%
station.update_all_station_snapshots()

# %% [markdown]
# # N-photon M1-storage full-swap calibration
#
# This section calibrates one direct \( |N,0\rangle \leftrightarrow |0,N\rangle \) M1-storage swap at a time. The N=1 row remains M1-Sj; additional rows are named M1-Sj@N2, M1-Sj@N3, and so on.
#
# Run the cells in order. Measurement cells do not update ds_storage. A result is written only by the separate accept cell after its plot, and the snapshot is a separate final action.

# %% [markdown]
# ## 1. Select and initialize one row
#
# Choose one storage mode and one photon number. A missing @N row is copied from the legacy N=1 row. For an existing row, list only the fields to reset; leave the tuple empty to preserve it.

# %%
import importlib

from experiments.single_qubit import error_amplification
from experiments.single_qubit.sideband_general import SidebandGeneralExperiment

# Source cell 30 also re-imported numpy, pyplot, deepcopy, AttrDict, the two
# runners, MM_dual_rail_base and ChevronFitting. The preamble now covers the
# first six, and the last two moved into the helper module with the cells that
# used them (build_swap_pulse_sequences, analyze_swap_chevron).
importlib.reload(error_amplification)
ErrorAmplificationExperiment = error_amplification.ErrorAmplificationExperiment

multiphoton_swap_storage_mode = 2
multiphoton_swap_photon_number = 2
if multiphoton_swap_photon_number < 1:
    raise ValueError('multiphoton_swap_photon_number must be at least one')
multiphoton_swap_gain_limit = 30000
multiphoton_swap_storage_wait_us = 0.2
# Start conservatively: the legacy storage active-reset pulse is only N=1 calibrated.
# Enable active reset only after verifying that it empties the N-photon residue.
multiphoton_swap_active_reset = False
multiphoton_swap_relax_delay_us = 2500.0

# Choose from: 'freq', 'precision', 'pi', 'h_pi', 'gain'.
# Example: ('freq', 'pi', 'h_pi')
multiphoton_swap_reset_fields = ()

multiphoton_swap_base_name = f'M1-S{multiphoton_swap_storage_mode}'
multiphoton_swap_pulse_name = station.ds_storage.multiphoton_swap_name(
    multiphoton_swap_base_name, multiphoton_swap_photon_number
)

if not station.ds_storage.has_row(multiphoton_swap_pulse_name):
    station.ds_storage.initialize_multiphoton_swap(
        multiphoton_swap_base_name, multiphoton_swap_photon_number
    )
    print('initialized missing row from', multiphoton_swap_base_name)
elif multiphoton_swap_reset_fields:
    station.ds_storage.initialize_multiphoton_swap(
        multiphoton_swap_base_name,
        multiphoton_swap_photon_number,
        overwrite=True,
        fields=multiphoton_swap_reset_fields,
    )
    print('reset fields:', multiphoton_swap_reset_fields)
else:
    print('preserving existing row')

print('calibrating:', multiphoton_swap_pulse_name)
print('frequency:', station.ds_storage.get_freq(multiphoton_swap_pulse_name), 'MHz')
print('gain:', station.ds_storage.get_gain(multiphoton_swap_pulse_name))
print('pi length:', station.ds_storage.get_pi(multiphoton_swap_pulse_name), 'us')

# %% [markdown]
# ## 2. State preparation and endpoint decoder
#
# The Chevron and error amplification use the same preparation. The decoder uses only the top two transitions,
#
# \[
# f_{N-1}\leftrightarrow g_N,\qquad e_{N-1}\leftrightarrow f_{N-1},
# \]
#
# so only an exact return to \( |g,N\rangle \) is marked bright. Lower M1 occupations remain dark instead of being mistaken for a successful return.

# %%
multiphoton_swap_sequences = build_swap_pulse_sequences(
    station, multiphoton_swap_photon_number
)
multiphoton_swap_prep_descriptions = multiphoton_swap_sequences['prep_descriptions']
multiphoton_swap_extra_prep = multiphoton_swap_sequences['extra_prep']
multiphoton_swap_endpoint_decoder_descriptions = multiphoton_swap_sequences[
    'endpoint_decoder_descriptions'
]
multiphoton_swap_prep_pulse = multiphoton_swap_sequences['prep_pulse']
multiphoton_swap_endpoint_decoder = multiphoton_swap_sequences['endpoint_decoder']

# %% [markdown]
# ## 3. Frequency-length Chevron
#
# This is the same SidebandGeneralExperiment and SweepRunner flow used by the ordinary M1-storage calibration. The length sweep includes a true zero-pulse reference.

# %%
chevron_frequency_span_MHz = 0.40
chevron_frequency_points = RUN.pick(25, smoke=7)
chevron_length_points = RUN.pick(41, smoke=21)
chevron_length_stop_us = 2.2 * station.ds_storage.get_pi(multiphoton_swap_pulse_name)
chevron_gain = station.ds_storage.get_gain(multiphoton_swap_pulse_name)
chevron_reps = RUN.pick(100, smoke=50)

chevron_center_MHz = float(station.ds_storage.get_freq(multiphoton_swap_pulse_name))
chevron_channel = 'low' if chevron_center_MHz < 1800 else 'high'

multiphoton_swap_chevron_defaults = AttrDict(dict(
    start=0.0,
    step=chevron_length_stop_us / (chevron_length_points - 1),
    expts=chevron_length_points,
    reps=chevron_reps,
    rounds=1,
    qubit=0,
    qubits=[0],
    flux_drive=[chevron_channel, chevron_center_MHz, chevron_gain, 0.0],
    length_placeholder=0.0,
    prepulse=True,
    pre_sweep_pulse=multiphoton_swap_prep_pulse,
    postpulse=True,
    post_sweep_pulse=multiphoton_swap_endpoint_decoder,
    update_post_pulse_phase=[False, 0.0],
    active_reset=multiphoton_swap_active_reset,
    man_reset=True,
    storage_reset=[multiphoton_swap_storage_mode],
    reset_dump_mode=active_reset_default_dict['reset_dump_mode'],
    dump_reset_iter_num=active_reset_default_dict['dump_reset_iter_num'],
    relax_delay=multiphoton_swap_relax_delay_us,
))

multiphoton_swap_chevron_runner = SweepRunner(
    station=station,
    ExptClass=SidebandGeneralExperiment,
    default_expt_cfg=multiphoton_swap_chevron_defaults,
    sweep_param='freq',
    postprocessor=None,
    job_client=client,
    use_queue=RUN.use_queue,
)


# %%
multiphoton_swap_chevron = multiphoton_swap_chevron_runner.execute(
    sweep_start=chevron_center_MHz - chevron_frequency_span_MHz / 2,
    sweep_stop=chevron_center_MHz + chevron_frequency_span_MHz / 2,
    sweep_npts=chevron_frequency_points,
    gain=chevron_gain,
    batch=True,
    show=False,
    log=True,
)
print('jobs:', multiphoton_swap_chevron_runner.last_job_ids)


# %%
(
    multiphoton_swap_chevron_analysis,
    chevron_frequency_candidate_MHz,
    chevron_pi_candidate_us,
) = analyze_swap_chevron(
    multiphoton_swap_chevron, station, multiphoton_swap_pulse_name
)

# %% [markdown]
# ### Accept the Chevron result
#
# Edit the two assignments if the fit selected the wrong branch. This is the first cell that changes the in-memory row.

# %%
selected_frequency_MHz = chevron_frequency_candidate_MHz
selected_pi_us = chevron_pi_candidate_us

station.ds_storage.update_freq(multiphoton_swap_pulse_name, selected_frequency_MHz)
station.ds_storage.update_pi(multiphoton_swap_pulse_name, selected_pi_us)
station.ds_storage.update_h_pi(multiphoton_swap_pulse_name, selected_pi_us / 2)
station.ds_storage.update_gain(multiphoton_swap_pulse_name, chevron_gain)
station.ds_storage.update_precision(
    multiphoton_swap_pulse_name,
    chevron_frequency_span_MHz / (chevron_frequency_points - 1),
)
print('accepted Chevron values for', multiphoton_swap_pulse_name)


# %% [markdown]
# ## 4. Error amplification
#
# Every depth uses an even number of physical swaps. The depth-zero row is an in-situ preparation/readout reference. The plotted score is the mean IQ distance from that row, and measurement cells only print a candidate.
#
# All four scans below -- coarse/fine frequency and coarse/fine gain, source
# cells 42, 45, 48 and 51 -- are the same `runner.execute` with different
# settings, and share the scoring in `score_return_error`.

# %%
storage_wait_cycles = int(station.soccfg.us2cycles(multiphoton_swap_storage_wait_us))

multiphoton_swap_error_amp_defaults = AttrDict(dict(
    reps=75,
    rounds=1,
    qubit=0,
    qubits=[0],
    n_start=0,
    n_step=1,
    n_pulses=6,
    active_reset=multiphoton_swap_active_reset,
    man_mode_no=1,
    man_reset=True,
    storage_reset=[multiphoton_swap_storage_mode],
    reset_dump_mode=active_reset_default_dict['reset_dump_mode'],
    dump_reset_iter_num=active_reset_default_dict['dump_reset_iter_num'],
    relax_delay=multiphoton_swap_relax_delay_us,
    expts=31,
    parameter_to_test='frequency',
    pulse_type=['storage', multiphoton_swap_pulse_name, 'pi', 0.0],
    pre_sweep_pulse=multiphoton_swap_extra_prep,
    post_sweep_pulse=multiphoton_swap_endpoint_decoder_descriptions,
    floquet_sync_delay=storage_wait_cycles,
))

multiphoton_swap_error_amp_runner = CharacterizationRunner(
    station=station,
    ExptClass=ErrorAmplificationExperiment,
    default_expt_cfg=multiphoton_swap_error_amp_defaults,
    job_client=client,
    use_queue=RUN.use_queue,
    show=False,
)


# %% [markdown]
# ### 4-1. Coarse frequency

# %%
coarse_frequency_center_MHz = float(
    station.ds_storage.get_freq(multiphoton_swap_pulse_name)
)
coarse_frequency_half_span_MHz = 0.10
coarse_frequency_points = RUN.pick(31, smoke=11)

multiphoton_swap_coarse_frequency = multiphoton_swap_error_amp_runner.execute(
    parameter_to_test='frequency',
    start=coarse_frequency_center_MHz - coarse_frequency_half_span_MHz,
    step=2 * coarse_frequency_half_span_MHz / (coarse_frequency_points - 1),
    expts=coarse_frequency_points,
    n_start=0,
    n_step=1,
    n_pulses=6,
    reps=75,
    postprocess=False,
    show=True,
    log=True,
    display_kwargs=dict(fit=False),
)

# %%
_x, _return_error, coarse_frequency_candidate_MHz = score_return_error(
    multiphoton_swap_coarse_frequency,
    xlabel='frequency (MHz)',
    title='coarse frequency',
)

# %%
selected_frequency_MHz = coarse_frequency_candidate_MHz
station.ds_storage.update_freq(multiphoton_swap_pulse_name, selected_frequency_MHz)
station.ds_storage.update_precision(
    multiphoton_swap_pulse_name,
    2 * coarse_frequency_half_span_MHz / (coarse_frequency_points - 1),
)
print('accepted coarse frequency:', selected_frequency_MHz)


# %% [markdown]
# ### 4-2. Coarse gain
#
# Set an even half-span and step. The last point is capped at 30000 because the flat-top program also uses a half-gain register.

# %%
coarse_gain_center = station.ds_storage.get_gain(multiphoton_swap_pulse_name)
coarse_gain_start, coarse_gain_stop, coarse_gain_step, coarse_gain_points = (
    even_gain_grid(
        center=coarse_gain_center,
        half_span=2000,
        step=RUN.pick(200, smoke=800),
        gain_limit=multiphoton_swap_gain_limit,
    )
)
print('coarse gain sweep:', coarse_gain_start, '...', coarse_gain_stop,
      'step', coarse_gain_step)

multiphoton_swap_coarse_gain = multiphoton_swap_error_amp_runner.execute(
    parameter_to_test='gain',
    start=coarse_gain_start,
    step=coarse_gain_step,
    expts=coarse_gain_points,
    n_start=0,
    n_step=1,
    n_pulses=6,
    reps=75,
    postprocess=False,
    show=True,
    log=True,
    display_kwargs=dict(fit=False),
)

# %%
_x, _return_error, coarse_gain_candidate = score_return_error(
    multiphoton_swap_coarse_gain,
    xlabel='gain',
    title='coarse gain',
    as_int=True,
)

# %%
selected_gain = int(np.clip(coarse_gain_candidate, 0, multiphoton_swap_gain_limit))
station.ds_storage.update_gain(multiphoton_swap_pulse_name, selected_gain)
print('accepted coarse gain:', selected_gain)


# %% [markdown]
# ### 4-3. Fine gain

# %% tags=["suite-skip"]
# Suite: skipped. The fine pass repeats the coarse scan and its accept step above.
fine_gain_center = station.ds_storage.get_gain(multiphoton_swap_pulse_name)
fine_gain_start, fine_gain_stop, fine_gain_step, fine_gain_points = even_gain_grid(
    center=fine_gain_center,
    half_span=600,
    step=40,
    gain_limit=multiphoton_swap_gain_limit,
)
print('fine gain sweep:', fine_gain_start, '...', fine_gain_stop,
      'step', fine_gain_step)

multiphoton_swap_fine_gain = multiphoton_swap_error_amp_runner.execute(
    parameter_to_test='gain',
    start=fine_gain_start,
    step=fine_gain_step,
    expts=fine_gain_points,
    n_start=0,
    n_step=1,
    n_pulses=10,
    reps=100,
    postprocess=False,
    show=True,
    log=True,
    display_kwargs=dict(fit=False),
)

# %% tags=["suite-skip"]
_x, _return_error, fine_gain_candidate = score_return_error(
    multiphoton_swap_fine_gain,
    xlabel='gain',
    title='fine gain',
    as_int=True,
)

# %% tags=["suite-skip"]
# Suite: skipped. The fine pass repeats the coarse scan and its accept step above.
selected_gain = int(np.clip(fine_gain_candidate, 0, multiphoton_swap_gain_limit))
station.ds_storage.update_gain(multiphoton_swap_pulse_name, selected_gain)
print('accepted fine gain:', selected_gain)


# %% [markdown]
# ### 4-4. Fine frequency

# %% tags=["suite-skip"]
# Suite: skipped. The fine pass repeats the coarse scan and its accept step above.
fine_frequency_center_MHz = float(
    station.ds_storage.get_freq(multiphoton_swap_pulse_name)
)
fine_frequency_half_span_MHz = 0.02
fine_frequency_points = 31

multiphoton_swap_fine_frequency = multiphoton_swap_error_amp_runner.execute(
    parameter_to_test='frequency',
    start=fine_frequency_center_MHz - fine_frequency_half_span_MHz,
    step=2 * fine_frequency_half_span_MHz / (fine_frequency_points - 1),
    expts=fine_frequency_points,
    n_start=0,
    n_step=1,
    n_pulses=10,
    reps=100,
    postprocess=False,
    show=True,
    log=True,
    display_kwargs=dict(fit=False),
)

# %% tags=["suite-skip"]
_x, _return_error, fine_frequency_candidate_MHz = score_return_error(
    multiphoton_swap_fine_frequency,
    xlabel='frequency (MHz)',
    title='fine frequency',
)

# %% tags=["suite-skip"]
# Suite: skipped. The fine pass repeats the coarse scan and its accept step above.
selected_frequency_MHz = fine_frequency_candidate_MHz
station.ds_storage.update_freq(multiphoton_swap_pulse_name, selected_frequency_MHz)
station.ds_storage.update_precision(
    multiphoton_swap_pulse_name,
    2 * fine_frequency_half_span_MHz / (fine_frequency_points - 1),
)
print('accepted fine frequency:', selected_frequency_MHz)


# %% [markdown]
# ## 5. Validation
#
# The odd check compares zero and one swap: the endpoint marker should change from bright to dark. The even check compares zero and two swaps: the endpoint marker should return to the same IQ point. Together they distinguish coherent transfer from simple photon loss.

# %%
validation_frequency_MHz = float(
    station.ds_storage.get_freq(multiphoton_swap_pulse_name)
)
validation_gain = station.ds_storage.get_gain(multiphoton_swap_pulse_name)
validation_pi_us = float(station.ds_storage.get_pi(multiphoton_swap_pulse_name))

odd_validation_defaults = deepcopy(multiphoton_swap_chevron_defaults)
odd_validation_defaults.update(dict(
    start=0.0,
    step=validation_pi_us,
    expts=2,
    reps=RUN.pick(300, smoke=100),
    flux_drive=[
        'low' if validation_frequency_MHz < 1800 else 'high',
        validation_frequency_MHz,
        validation_gain,
        0.0,
    ],
))

odd_validation_runner = CharacterizationRunner(
    station=station,
    ExptClass=SidebandGeneralExperiment,
    default_expt_cfg=odd_validation_defaults,
    job_client=client,
    use_queue=RUN.use_queue,
    show=False,
)
multiphoton_swap_odd_validation = odd_validation_runner.execute(
    postprocess=False, show=False, log=True
)

# %%
odd_z, odd_separation = plot_iq_endpoints(
    multiphoton_swap_odd_validation,
    labels=('zero swaps', 'one swap'),
    title=f'{multiphoton_swap_pulse_name}: odd-swap validation',
)
print('zero-to-one-swap IQ separation:', odd_separation)

# %%
multiphoton_swap_even_validation = multiphoton_swap_error_amp_runner.execute(
    parameter_to_test='frequency',
    start=validation_frequency_MHz,
    step=0.0,
    expts=1,
    n_start=0,
    n_step=1,
    n_pulses=1,
    reps=RUN.pick(300, smoke=100),
    postprocess=False,
    show=False,
    log=True,
)

# %%
even_z, even_error = plot_iq_endpoints(
    multiphoton_swap_even_validation,
    labels=('zero swaps', 'two swaps'),
    title=f'{multiphoton_swap_pulse_name}: even-return validation',
)
print('zero-to-two-swap IQ error:', even_error)

# %% [markdown]
# ## 6. Save the accepted row
#
# Run this only after inspecting both validation plots. The production flag stays False until it is changed explicitly in measurement_config_default_dict. After enabling it, rerun the exact-path phase calibration.

# %%
multiphoton_swap_snapshot_id = station.snapshot_man1_storage_swap(update_main=False)
print('saved non-main ds_storage snapshot:', multiphoton_swap_snapshot_id)
print('production flag:', measurement_config_default_dict['use_multiphoton_swap'])

# %% [markdown]
# ## Datset for Sidebands
#
# Source cells 61-64. The surface map puts these here rather than in
# `floquet_calibration.py`, even though they set up `ds_floquet`: they are
# dataset initialization that runs once before any Floquet calibration.

# %%
# True reinitializes frequency/gain from ds_storage; False keeps the calibrated dataset.
RESET_FLOQUET_DATASET = True
WAVEFORM_TO_USE = 'preload_flattop'  # 'gauss' keeps the original Gaussian initialization.

# Requested envelope lengths below are quantized to DAC clocks; the next cell includes sync timing too.
preload_flat_length_us = 0.037
preload_ramp_sigma_us = 0.002
preload_pi_frac = 40

if RESET_FLOQUET_DATASET:
    if WAVEFORM_TO_USE == 'gauss':
        station.ds_floquet.import_from_swap_dataset(
            station.ds_storage, gain_div=1, pi_div=40)
        for i in range(7):
            station.ds_floquet.update_gauss_n_sigma(f"M1-S{i+1}", 4)
            station.ds_floquet.update_waveform(f"M1-S{i+1}", 'gauss')
            station.ds_floquet.update_gauss_sigma(f"M1-S{i+1}", 0.02)
    elif WAVEFORM_TO_USE == 'preload_flattop':
        station.ds_floquet.import_from_swap_dataset(
            station.ds_storage, gain_div=1, pi_div=preload_pi_frac)
        for i in range(7):
            station.ds_floquet.update_waveform(f"M1-S{i+1}", 'preload_flattop')
            station.ds_floquet.update_len(f"M1-S{i+1}", preload_flat_length_us)
            station.ds_floquet.update_ramp_sigma(f"M1-S{i+1}", preload_ramp_sigma_us)
    else:
        raise ValueError("WAVEFORM_TO_USE must be 'gauss' or 'preload_flattop'")

# Keep copied experiment defaults consistent; None uses the saved dataset when not resetting.
floquet_default_dict["floquet_waveform"] = WAVEFORM_TO_USE if RESET_FLOQUET_DATASET else None
# station.snapshot_floquet_storage_swap(update_main=False)

# %%
import importlib
from experiments.qsim import floquet_dark_mode_readout
importlib.reload(floquet_dark_mode_readout)

TROTTER_MODES_IN_USE = [1, 2, 3, 4]  # Use the same modes as the intended scramble.
# Same DAC-envelope and integer tProc sync timing as the spectroscopy program.
floquet_timing = floquet_dark_mode_readout.EncodingHamiltonianSpectroscopyExperiment.hardware_parameters(
    station, TROTTER_MODES_IN_USE, floquet_default_dict['scramble_sync_cycles'],
    floquet_waveform=floquet_default_dict.get('floquet_waveform'),
)
floquet_cycle_us = floquet_timing.floquet_cycle_us
effective_g_kHz = (1000 * floquet_timing.couplings_MHz).tolist()
print(f"Modes: {TROTTER_MODES_IN_USE}; scheduled Floquet cycle: {floquet_cycle_us:.9f} us")
print("Clock-quantized effective g (kHz):", effective_g_kHz)

# %%
# Keep the legacy manual adjustment, but do not break the common preload envelope.
if station.ds_floquet.get_waveform('M1-S6') != 'preload_flattop':
    station.ds_floquet.update_len('M1-S6', 0.01)

station.ds_floquet.df


# %% [markdown]
# # Single shot

# %%
# Defaults and post-measurement readout update. `singleshot_postproc` is now
# in the helper module.
# =====================================
singleshot_defaults = AttrDict(dict(
    reps=RUN.pick(5000, smoke=1000),
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

# %%
# Execute
# =================================
ss_runner = CharacterizationRunner(
    station = station,
    ExptClass = meas.HistogramExperiment,
    default_expt_cfg = singleshot_defaults,
    postprocessor = singleshot_postproc,
    job_client=client,
    use_queue=RUN.use_queue,
)

ss = ss_runner.execute(
    check_f=False,
    active_reset=False, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
    relax_delay=2000,
    # active_reset=True,
    # relax_delay=200,
    priority=1,
    # coupler_current = 0
    avoid_yoko = measurement_config_default_dict['avoid_yoko'],


)

# %%
station.update_all_station_snapshots()
