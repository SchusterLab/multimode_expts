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
# The *hooks* of those two columns were copy-paste duplicates, and they are
# single functions in `experiments/qsim/notebook_helpers/floquet_calibration.py`
# now -- preprocessors, postprocessors and `floquet_cycle_list_gen`. The
# module docstring records which cells collapsed into which function.
#
# **The defaults dicts, the runners and the sweep loops stay here**, in the
# canonical defaults -> runner -> execute shape. An earlier pass had also
# folded the loops into `run_*_sweep()` helpers; that hid `runner.execute`
# behind a closed keyword list, so a cell could no longer pass `use_queue`,
# `priority`, `go_kwargs` or any other expt_cfg override. The loops are back
# inline: a few lines of `for` around `runner.execute(...)` is transparent,
# and every kwarg lands where a calibration session expects it.
#
# The two halves genuinely disagree in their defaults -- cell 107 applies
# `floquet_default_dict` to the error-amplification defaults where cell 78 set
# `floquet_waveform` alone -- so they are written out twice, not merged.
#
# The bare-readout check at the end (cells 150-158) is in
# `floquet_bare_readout.py`. **Read the TODO on `floquet_cycle_to_us` there
# before trusting its time axis:** it is the naive exact-sum cycle duration,
# which `experiments/floquet_timing.py` documents as running ~1.2% long
# against the clock-quantized definition. It is preserved as found rather than
# corrected, because fixing it here would have been a silent physics change.
#
# The defaults of cells 2-6 are in `experiments/qsim/notebook_helpers/defaults.py`.
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
from experiments import CharacterizationRunner, SweepRunner, MultimodeStation

from job_server import JobClient
# Imported under the names the relocated cells already use.
from experiments.qsim.notebook_helpers.defaults import (
    ACTIVE_RESET_DEFAULTS as active_reset_default_dict,
    FLOQUET_DEFAULTS as floquet_default_dict,
    MEASUREMENT_CONFIG_DEFAULTS as measurement_config_default_dict,
)
from experiments.qsim.notebook_helpers.run_mode import run_settings

# Set by tools/run_qsim_suite.py. Unset: a normal run through the queue.
RUN = run_settings()
from experiments.qsim.notebook_helpers.floquet_calibration import (
    error_amp_floquet_postproc,
    error_amp_floquet_preproc,
    floquet_cycle_list_gen,
    floquet_freq_chev_postproc,
    floquet_freq_chev_preproc,
    floquet_gain_chev_postproc,
    floquet_gain_chev_preproc,
    get_floquet_parameters,
    sideband_stark_error_amp_postproc,
    sideband_stark_error_amp_preproc,
)
from experiments.qsim.notebook_helpers.floquet_bare_readout import (
    flatten_exp_lists,
    plot_bare_scramble,
    sideband_scramble_preproc,
)

# %%
# The config versions this campaign ran against.
config_dict = {
    "hardware_config": "CFG-HW-20260915-00001",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260909-00032",
    "floquet_storage_swap": "CFG-FL-20260909-00043",
}

station = MultimodeStation(
    user="guan",
    experiment_name="260915_qsim_migration",
    project="test_migration",
    log_measurements=not RUN.smoke,
    mock=RUN.mock,
    **RUN.station_configs(config_dict),
)
client = JobClient()

# %% [markdown] jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
# # Single shot

# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
# Define defaults, smart config preprocessing and post-measurement updates
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


# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
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
    go_kwargs=dict(analyze=False, display=False),
    check_f=False,
    active_reset=False, # on recalibration of readout, turn off active reset because it will be wrong for selecting when to apply the qubit pulse
    relax_delay=2000,
    # active_reset=True,
    # relax_delay=200,
    # coupler_current=coupler_current,
    # priority=1,
    use_queue=False
)
# ss.display()

# %% jupyterlab_notify.notify={"defaultThreshold": "30s", "mode": "default"}
station.update_all_station_snapshots()

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
    use_queue=RUN.use_queue,
)

# %%
# Source cell 72: one fixed +/-1 MHz span for every mode.
stor_modes_to_run = RUN.pick([1, 2, 5, 6, 7], smoke=[1]) #list(range(1,8))
detune_span = 1.0

freq_len_expt = [None] * len(stor_modes_to_run)
for i, init_stor in enumerate(stor_modes_to_run):
    print(f'Running Floquet Frequency vs Length Chevron for Storage Mode {init_stor}')
    freq_len_expt[i] = floquet_freq_chev_runner.execute(
        init_stor=init_stor,
        detunes=np.linspace(-detune_span, detune_span, RUN.pick(51, smoke=11)).tolist(),
        reps=100,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[init_stor],
        reset_dump_mode=1,
        use_queue=False,
    )
    freq_len_expt[i].display()

# %%
for i in range(len(stor_modes_to_run)):
    floquet_freq_chev_postproc(station, freq_len_expt[i])

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ### Fine
#
# Source cell 76. Same loop as above with a per-mode span table.

# %% tags=["suite-skip"]
# Suite: skipped. A second pass of the coarse chevron loop above, with narrower spans.
stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
freq_span_expt = [ 0.5,  0.3, None, None,  0.2,  0.2,  0.2] # by stor-1, None -> default
default_span = 1.0

freq_len_expt = [None] * len(stor_modes_to_run)
for i, init_stor in enumerate(stor_modes_to_run):
    span = freq_span_expt[init_stor - 1]
    if span is None:
        span = default_span
    print(f'Running Floquet Frequency vs Length Chevron for Storage Mode {init_stor}')
    freq_len_expt[i] = floquet_freq_chev_runner.execute(
        init_stor=init_stor,
        detunes=np.linspace(-span, span, 51).tolist(),
        reps=100,
        relax_delay=200,  # 8000 without active reset
        active_reset=True,
        man_reset=True,
        storage_reset=[init_stor],
        reset_dump_mode=1,
    )
    if not station.log_measurements:
        freq_len_expt[i].display()

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
    use_queue=RUN.use_queue,
)

# %%
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
stor_modes_to_run = RUN.pick([4, 5], smoke=[4])

# Both span lists are indexed by stor-1; None falls back to the default below.
freq_span_list = [0.1, 0.1, None, None, 0.15, 0.1, 0.05] # set for the Coarse
gain_span_list = [None, None, None, None, None, None, None]
freq_span_default = 0.1
gain_span_default = 0.3

# %% [markdown]
# ### Coarse
#
# Source cell 81. Comment out either execute block to skip that scan.

# %%
span_divisor = 1
error_amp_freq1 = [None] * len(stor_modes_to_run)
error_amp_gain1 = [None] * len(stor_modes_to_run)
for i, stor_i in enumerate(stor_modes_to_run):
    stor_name = f'M1-S{stor_i}'
    freq_span = freq_span_list[stor_i - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor_i - 1]
    if gain_span is None:
        gain_span = gain_span_default

    error_amp_freq1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='frequency',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.2 is the program default, but we are already close.
        span=freq_span / span_divisor,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )
    if not station.log_measurements:
        error_amp_freq1[i].display()

    error_amp_gain1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='gain',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.7 is the program default.
        span=int(station.ds_floquet.get_gain(stor_name) * gain_span / span_divisor),
        expts=60,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )
    if not station.log_measurements:
        error_amp_gain1[i].display()

# %% [markdown]
# ### Fine
#
# Source cell 83: the same loop with both spans halved. Note it also used a
# literal `reset_dump_mode=1` rather than the shared default, which is
# preserved here rather than quietly normalized.

# %% tags=["suite-skip"]
# Suite: skipped. A second pass of the coarse loop above, with halved spans.
stor_modes_to_run = [7]

span_divisor = 2
error_amp_freq1 = [None] * len(stor_modes_to_run)
error_amp_gain1 = [None] * len(stor_modes_to_run)
for i, stor_i in enumerate(stor_modes_to_run):
    stor_name = f'M1-S{stor_i}'
    freq_span = freq_span_list[stor_i - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor_i - 1]
    if gain_span is None:
        gain_span = gain_span_default

    error_amp_freq1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='frequency',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.2 is the program default, but we are already close.
        span=freq_span / span_divisor,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=1,
    )
    if not station.log_measurements:
        error_amp_freq1[i].display()

    error_amp_gain1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='gain',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.7 is the program default.
        span=int(station.ds_floquet.get_gain(stor_name) * gain_span / span_divisor),
        expts=60,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=1,
    )
    if not station.log_measurements:
        error_amp_gain1[i].display()

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
    use_queue=RUN.use_queue,
)

# %%
# Source cell 89: all ordered pairs among these modes. It did not forward the
# pi/2 buffer flag or scramble_sync_cycles, unlike the Gaussian half below.
stor_modes_to_run = [2, 5] #list(range(1,8))

for init_storA in stor_modes_to_run:
    for init_storB in stor_modes_to_run:
        if init_storA == init_storB:
            continue
        print("Starting experiment for storage modes:", init_storA, "from", init_storB)
        phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
            stor_A=init_storA,
            stor_B=init_storB,
            reps=100,
            relax_delay=100,
            active_reset=True,
            man_reset=True,
            storage_reset=[init_storA, init_storB],
            reset_dump_mode=1,
            include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
            advance_phases=np.linspace(-10, 10, 101).tolist(),
        )

# %%
station.update_all_station_snapshots()

# %% tags=["suite-skip"]
# Suite: skipped. A redo of one pair from the loop above.
# Source cell 91: one directed pair only.
init_storA, init_storB = 5, 3

print("Starting experiment for storage modes:", init_storA, "from", init_storB)
phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
    stor_A=init_storA,
    stor_B=init_storB,
    reps=100,
    relax_delay=100,
    active_reset=True,
    man_reset=True,
    storage_reset=[init_storA, init_storB],
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
    advance_phases=np.linspace(-10, 10, 101).tolist(),
)

# %%
station.snapshot_floquet_storage_swap(update_main=False)

# %% tags=["suite-skip"]
# Suite: skipped. A hand-entered value.
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
        use_queue=RUN.use_queue,
    )

    detune_span = 0.5

    freq_len_expt = [None] * len(stor_modes_to_run)
    for i, init_stor in enumerate(stor_modes_to_run):
        print(f'Running Floquet Frequency vs Gain Chevron for Storage Mode {init_stor}')
        freq_len_expt[i] = floquet_gain_chev_runner.execute(
            init_stor=init_stor,
            detunes=np.linspace(-detune_span, detune_span, 21).tolist(),
            reps=50,
            gain_expts=21,
            max_gain=14000,
            relax_delay=200,  # 8000 without active reset
            active_reset=True,
            man_reset=True,
            storage_reset=[init_stor],
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            debug=False,
        )
        if not station.log_measurements:
            freq_len_expt[i].display()

# %%
floquet_gain_chev_postproc(station, freq_len_expt[0])

# %% tags=["suite-skip"]
# Suite: skipped. Hand-entered values.
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
    use_queue=RUN.use_queue,
)

# %%
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
# stor_modes_to_run = [4, 5, 6, 7] #list(range(1,8))

stor_modes_to_run = [1, 2, 3, 4, 5, 7]

# Both span lists are indexed by stor-1; None falls back to the default below.
freq_span_list = [0.1,  0.07, 0.07, 0.1,  0.07,   0.1, 0.05] # set for the Coarse
gain_span_list = [None, None, None, None,  0.25,  0.25, 0.25]
freq_span_default = 0.1
gain_span_default = 0.3

# %%
# Source cell 109. Unlike the flat-top coarse pass, this one derives
# relax_delay from whether active reset is on.
do_active_reset = True
relax_delay = 200
if not do_active_reset:
    relax_delay = 8000

stor_modes_to_run = [4]

span_divisor = 1
error_amp_freq1 = [None] * len(stor_modes_to_run)
error_amp_gain1 = [None] * len(stor_modes_to_run)
for i, stor_i in enumerate(stor_modes_to_run):
    stor_name = f'M1-S{stor_i}'
    freq_span = freq_span_list[stor_i - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor_i - 1]
    if gain_span is None:
        gain_span = gain_span_default

    error_amp_freq1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='frequency',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.2 is the program default, but we are already close.
        span=freq_span / span_divisor,
        relax_delay=relax_delay,
        active_reset=do_active_reset,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )
    if not station.log_measurements:
        error_amp_freq1[i].display()

    error_amp_gain1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='gain',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.7 is the program default.
        span=int(station.ds_floquet.get_gain(stor_name) * gain_span / span_divisor),
        expts=60,
        relax_delay=relax_delay,
        active_reset=do_active_reset,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )
    if not station.log_measurements:
        error_amp_gain1[i].display()

# %%
# station.ds_floquet.update_freq("M1-S4", 878.27)

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ## Fine Error Amplification for Gaussian pulses
#
# Source cell 113: spans halved, and it reuses whatever `stor_modes_to_run`
# the cell above left bound.

# %% tags=["suite-skip"]
# Suite: skipped. A second pass of the coarse loop above, with halved spans.
span_divisor = 2
error_amp_freq1 = [None] * len(stor_modes_to_run)
error_amp_gain1 = [None] * len(stor_modes_to_run)
for i, stor_i in enumerate(stor_modes_to_run):
    stor_name = f'M1-S{stor_i}'
    freq_span = freq_span_list[stor_i - 1]
    if freq_span is None:
        freq_span = freq_span_default
    gain_span = gain_span_list[stor_i - 1]
    if gain_span is None:
        gain_span = gain_span_default

    error_amp_freq1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='frequency',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.2 is the program default, but we are already close.
        span=freq_span / span_divisor,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )
    if not station.log_measurements:
        error_amp_freq1[i].display()

    error_amp_gain1[i] = error_amp_floquet_runner.execute(
        stor_mode_no=stor_i,
        parameter_to_test='gain',
        go_kwargs=dict(analyze=False, progress=True, display=False),
        # 0.7 is the program default.
        span=int(station.ds_floquet.get_gain(stor_name) * gain_span / span_divisor),
        expts=60,
        relax_delay=200,
        active_reset=True,
        man_reset=True,
        storage_reset=[stor_i],
        reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    )
    if not station.log_measurements:
        error_amp_gain1[i].display()

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
    use_queue=RUN.use_queue,
)

# %%
# Source cell 116. From here on the buffer flag and sync cycles are forwarded.
# stor_modes_to_run = [1, 2, 5, 6, 7] #list(range(1,8))
stor_modes_to_run = RUN.pick([1, 2, 3, 4], smoke=[1, 2]) #list(range(1,8))

do_active_reset = True
relax_delay = 200
if not do_active_reset:
    relax_delay = 8000

for init_storA in stor_modes_to_run:
    for init_storB in stor_modes_to_run:
        if init_storA == init_storB:
            continue
        print("Starting experiment for storage modes:", init_storA, "from", init_storB)
        phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
            stor_A=init_storA,
            stor_B=init_storB,
            reps=100,
            relax_delay=relax_delay,
            active_reset=do_active_reset,
            man_reset=True,
            storage_reset=[init_storA, init_storB],
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
            include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
            scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
            advance_phases=np.linspace(-10, 10, 51).tolist(),
        )

# %%
station.update_all_station_snapshots()

# %% [markdown]
# ### Redo or add calibration
#
# Source cell 119: one directed pair, on a wider and finer phase grid.

# %% tags=["suite-skip"]
# Suite: skipped. A redo of one pair from the loop above.
init_storA, init_storB = 2, 6

print("Starting experiment for storage modes:", init_storA, "from", init_storB)
phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
    stor_A=init_storA,
    stor_B=init_storB,
    reps=100,
    relax_delay=100,
    active_reset=True,
    man_reset=True,
    storage_reset=[init_storA, init_storB],
    reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
    include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
    include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
    scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
    advance_phases=np.linspace(-30, 30, 301).tolist(),
)

# %% [markdown]
# ### Add modes
#
# Source cell 121: measure only the pairs that involve a newly added mode,
# skipping those where both modes were already calibrated.

# %% tags=["suite-skip"]
# Suite: skipped. Extends a campaign already calibrated by hand.
stor_modes_to_add = [2] #list(range(1,8))
pre_existing_store_modes = [4, 5, 6]

total_stor_modes = list(set(pre_existing_store_modes + stor_modes_to_add))

for init_storA in total_stor_modes:
    for init_storB in total_stor_modes:
        if init_storA == init_storB:
            continue
        # Skip the pairs that were already calibrated before the new modes.
        if (init_storA in pre_existing_store_modes
                and init_storB in pre_existing_store_modes):
            continue
        print("Starting experiment for storage modes:", init_storA, "from", init_storB)
        phase_expts[init_storA - 1][init_storB - 1] = sideband_stark_error_amp_runner.execute(
            stor_A=init_storA,
            stor_B=init_storB,
            reps=100,
            relax_delay=100,
            active_reset=True,
            man_reset=True,
            storage_reset=[init_storA, init_storB],
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            include_10cycles_buffer=floquet_default_dict["include_10cycles_buffer"],
            include_10cycles_buffer_in_pi_half=floquet_default_dict["include_10cycles_buffer_in_pi_half"],
            scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
            advance_phases=np.linspace(-20, 20, 101).tolist(),
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
    use_queue=RUN.use_queue,
)

# %%
floquet_cycles_list = floquet_cycle_list_gen(
    0, RUN.pick(2000, smoke=300), RUN.pick(2000, smoke=300), 15
)

swap_stors = [1, 2, 3, 4]
# meas_stors = [0, 2, 6]
# meas_stors = [0]
meas_stors = RUN.pick([0] + swap_stors, smoke=[0, 1])
# meas_stors = [0, 1, 2, 3, 4]
dark_swaps = [4, 5]

detunings = [0] * len(swap_stors)
# The storages that get reset are meas_stors without the leading 0, which
# reads out the manipulate mode.
reset_stors = meas_stors[1:]

scramble_expts = []
for meas_stor in tqdm(meas_stors):
    scramble_sub_expts = []
    for floquet_cycles in floquet_cycles_list:
        scramble_sub_expts.append(dmscramble_runner.execute(
            reps=RUN.pick(300, smoke=100),
            init_fock=True,
            init_stor=0,
            ro_stor=meas_stor,
            relax_delay=200,  # 8000 without active reset
            active_reset=True,
            pre_relax_delay=100,
            man_reset=True,
            storage_reset=reset_stors,
            reset_dump_mode=active_reset_default_dict["reset_dump_mode"],
            dump_reset_iter_num=active_reset_default_dict["dump_reset_iter_num"],
            swap_stors=swap_stors,
            update_phases=True,
            detunings=detunings,
            floquet_cycles=floquet_cycles,
            swept_params=['floquet_cycle'],
            custom_prepulse=False,
            custom_postpulse=False,
            debug=False,
            swap_man_dark=False,
            dark_swap_order=dark_swaps,
            second_rel_phase=180,
            map_to_qubit_ge=True,
            prepulse=True,   # for debugging. Should always be true
            postpulse=True,  # for debugging. Should always be true
            palindrome_scramble=floquet_default_dict["palindrome_scramble"],
            scramble_sync_cycles=floquet_default_dict["scramble_sync_cycles"],
        ))
    scramble_expts.append(scramble_sub_expts)

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
