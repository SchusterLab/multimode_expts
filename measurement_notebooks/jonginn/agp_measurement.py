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
#     display_name: Multimode (direct remote)
#     language: python
#     name: multimode-direct
# ---

# %% [markdown]
# # Multiparity calibration and central-boson population measurements
#
# Run the acquisition cells you need. Each runner returns the measured Experiment;
# its analyze() calculates the multiparity results before the worker saves the JOB.
# Pass the calibration HDF5 filename so corrected populations and moments are saved
# with each measurement. Plot the saved results directly.
#
# One calibration JOB prepares Fock 0..3. One AGP JOB sweeps all selected occupations
# and cycles for one disorder strength and initial-state direction. The final cells
# collect those AGP files into a disorder scan with complete child data.
# ro_stor=0 selects M1; positive values are physical storage IDs.

# %% tags=["imports"]
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.qsim.agp import CentralBosonAGPDisorderScan, CentralBosonAGPExperiment
from experiments.qsim.fock_population import FockPopulationProgram
from experiments.qsim.multiparity import MultiparityQsimExperiment
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS
from fitting.qsim.mbr_hamiltonian import fixed_n_hamiltonian
from job_server import JobClient

# %% [markdown]
# ## 1. Station and worker
#
# These are the configuration versions used by the existing measurement notebook.
# Constructing the runners below does not submit measurements.

# %% tags=["station"]
user = "jonginn"
config_dict = {
    "hardware_config": "CFG-HW-20261019-00058",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261019-00001",
    "floquet_storage_swap": "CFG-FL-20261017-00034",
}

client = JobClient()
health = client.health_check()
print(f"Server status: {health['status']}")
client.print_queue()

station = MultimodeStation(
    user=user,
    experiment_name="agp_k44_first_scan",
    project="AGP",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)

# %% tags=["defaults"]
STORAGE_MODES = [4, 5, 6, 7]
common_defaults = AttrDict(dict(
    expts=1, reps=1, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORAGE_MODES, pre_relax_delay=100, relax_delay=200,
    postpulse=True, init_fock=False, init_stor=0, ro_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    swap_stors=STORAGE_MODES, update_phases=True,
    detunings=[0.0, 0.0, 0.0, 0.0],
))
common_defaults.update(ACTIVE_RESET_DEFAULTS)
common_defaults.update(FLOQUET_DEFAULTS)

# Calibration uses Qsim's prepulse; FockPopulationProgram prepares in core_pulses.
calibration_defaults = AttrDict(dict(common_defaults, prepulse=True))
population_defaults = AttrDict(dict(
    common_defaults, prepulse=False, occupations=[3, 0, 0, 0, 0],
    floquet_cycles=[0], swept_params=["floquet_cycle"],
))
agp_defaults = AttrDict(dict(common_defaults, prepulse=False))

# %% tags=["runners"]
confusion_matrix_runner = CharacterizationRunner(
    station=station,
    ExptClass=MultiParityConfusionMatrixExperiment,
    default_expt_cfg=calibration_defaults,
    job_client=client,
    use_queue=True,
    show=False,
)

population_runner = CharacterizationRunner(
    station=station,
    ExptClass=MultiparityQsimExperiment,
    ExptProgram=FockPopulationProgram,
    default_expt_cfg=population_defaults,
    job_client=client,
    use_queue=True,
    show=False,
)

agp_runner = CharacterizationRunner(
    station=station,
    ExptClass=CentralBosonAGPExperiment,
    default_expt_cfg=agp_defaults,
    job_client=client,
    use_queue=True,
    show=False,
)

# %% [markdown]
# ## 2. Fock 0..3 readout calibration: one JOB
#
# The Experiment owns the prepared-state sweep and stores C[prepared, measured]
# in data["multiparity_confusion_matrix"]. No additional population JOBs or
# manual matrix construction are needed.

# %% tags=["calibration"]
calibration_expt = confusion_matrix_runner.execute(reps=6000, postprocess=False)
print(calibration_expt.fname)

# %% tags=["calibration_display"]
calibration_expt.display()
print("Confusion matrix condition number:",
      np.linalg.cond(calibration_expt.data["multiparity_confusion_matrix"]))

# %% tags=["calibration_reference"]
CONFUSION_MATRIX_FNAME = str(calibration_expt.fname)
# To reuse a saved calibration, replace the assignment above with its full HDF5 filename.

# %% [markdown]
# ## 3. Population evolution from one initial occupation
#
# Each measured mode uses a fresh preparation and a separate JOB. The worker
# saves raw and corrected populations, mean n mod 4, and shot errors through
# MultiparityQsimExperiment.analyze(). postpulse=True swaps the selected storage
# into M1 before readout.

# %% tags=["walk_settings"]
swap_stors = list(STORAGE_MODES)
meas_stors = [0] + swap_stors
occupations = [3, 0, 0, 0, 0]
detunings = [0.0, 0.0, 0.0, 0.0]  # MHz
floquet_cycles = list(range(0, 100, 2))

# %% tags=["walk_acquire"]
walk_expts = []
for meas_stor in meas_stors:
    walk_expt = population_runner.execute(
        reps=300,
        occupations=occupations,
        swap_stors=swap_stors,
        storage_reset=swap_stors,
        ro_stor=meas_stor,
        detunings=detunings,
        floquet_cycles=floquet_cycles,
        multiparity_confusion_matrix=CONFUSION_MATRIX_FNAME,
        postprocess=False, show=False, log=False,
    )
    walk_expts.append(walk_expt)
    print(walk_expt.fname)

# %% tags=["walk_display"]
walk_fig, walk_axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for walk_expt in walk_expts:
    data = walk_expt.data
    mode = walk_expt.cfg.expt.ro_stor
    walk_axes[0].plot(data["xpts"], data["nmod4_mean"], label=f"Raw, Mode {mode}")
    walk_axes[1].plot(data["xpts"], data["nmod4_mean_corrected"],
                      label=f"Corrected, Mode {mode}")
for axis in walk_axes:
    axis.set(xlabel="Floquet cycles", ylabel="Mean n mod 4")
    axis.legend()
walk_fig.tight_layout()
plt.show()

# %% [markdown]
# ### Optional: compare population evolution with the fixed-N Hamiltonian
#
# Use each JOB's recorded Floquet timing and couplings. The physical Kerr is
# supplied separately. This comparison uses H/h in MHz with onsite storage
# energies -detuning. For the N=3 input above, number and number modulo 4 agree.

# %% tags=["theory_settings"]
KERR_KHZ = -27.0  # K / (2*pi), including its sign.

# %% tags=["walk_theory"]
theory_axis = walk_axes[1]
for line in list(theory_axis.lines):
    if line.get_label().startswith("Theory,"):
        line.remove()
measured_colors = {line.get_label(): line.get_color() for line in theory_axis.lines}

for plot_index, walk_expt in enumerate(walk_expts):
    settings = walk_expt.cfg.expt
    storages = list(settings.swap_stors)
    initial_occupation = tuple(settings.occupations)
    readout_storage = settings.ro_stor
    readout_index = [0, *storages].index(readout_storage)
    timing = walk_expt.recorded_derived_params()
    cycle_us = float(timing["floquet_cycle_us"])
    couplings_mhz = np.array([timing["couplings_MHz"][storage - 1] for storage in storages])
    if settings.get("zero_floquet_gain", False):
        couplings_mhz[:] = 0.0
    detunings_mhz = settings.get("detunings")
    if detunings_mhz is None or detunings_mhz is False or len(detunings_mhz) == 0:
        detunings_mhz = np.zeros(len(storages))

    model = fixed_n_hamiltonian(
        photon_number=sum(initial_occupation),
        mode_count=1 + len(storages),
        detunings=detunings_mhz,
        couplings_MHz=couplings_mhz,
        physical_kerr_MHz=KERR_KHZ / 1000,
    )
    cycles = np.asarray(walk_expt.data["xpts"], dtype=float)
    theory_cycles = np.linspace(cycles.min(), cycles.max(), max(1000, len(cycles)))
    time_us = theory_cycles * cycle_us
    initial_index = model.fock_index[initial_occupation]
    eigenstate_amplitudes = model.states[initial_index, :].conj()
    phases = np.exp(-2j * np.pi * model.energies_MHz[:, None] * time_us[None, :])
    state = model.states @ (eigenstate_amplitudes[:, None] * phases)
    probability = np.abs(state) ** 2
    theory_mean_number = np.asarray(model.fock_basis)[:, readout_index] @ probability
    color = measured_colors.get(f"Corrected, Mode {readout_storage}", f"C{plot_index % 10}")
    theory_axis.plot(theory_cycles, theory_mean_number, color=color, alpha=0.45,
                     linewidth=3, label=f"Theory, Mode {readout_storage}")
    print(f"Mode {readout_storage}: input={initial_occupation}, K={KERR_KHZ:g} kHz, "
          f"cycle={cycle_us:.6f} us, g={np.round(1000 * couplings_mhz, 3)} kHz")

theory_axis.legend(ncol=2)
walk_fig.tight_layout()
display(walk_fig)

# %% [markdown]
# ## 4. Central-boson occupation-by-cycle measurements
#
# One JOB per disorder strength and direction. CentralBosonAGPExperiment generates
# the fixed-photon-number occupations internally and excludes states with zero
# photons in the selected direction. Here ro_stor=direction measures autocorrelations;
# direction=0 selects M1 and other values are physical IDs in STORAGE_MODES.
#
# analyze() saves first and second moments and their ensemble-averaged two-time
# correlations. No Matrix Pencil fit or AGP norm is calculated in this notebook.

# %% tags=["agp_settings"]
TOTAL_PHOTONS = 3
AGP_DIRECTIONS = [7]
DISORDER_RMS_KHZ = [0]
AGP_REPS = 500
AGP_CYCLES = list(range(0, 400, 1))

DISORDER_PATTERN = np.array([-0.70675335, 0.19463085, -1.03457499, 1.54669749])
DISORDER_PATTERN -= np.mean(DISORDER_PATTERN)
pattern_rms = np.sqrt(np.mean(DISORDER_PATTERN**2))
if pattern_rms > 0:
    DISORDER_PATTERN /= pattern_rms
if any(value != 0 for value in DISORDER_RMS_KHZ) and pattern_rms == 0:
    raise ValueError("Nonzero disorder needs a nonconstant DISORDER_PATTERN.")

# %% tags=["agp_acquire"]
scan_expts = {direction: [] for direction in AGP_DIRECTIONS}
for direction in AGP_DIRECTIONS:
    for disorder_khz in DISORDER_RMS_KHZ:
        scan_expt = agp_runner.execute(
            reps=AGP_REPS,
            photon_sector_number=TOTAL_PHOTONS,
            direction=direction,
            ro_stor=direction,
            swap_stors=STORAGE_MODES,
            storage_reset=STORAGE_MODES,
            detunings=(0.001 * disorder_khz * DISORDER_PATTERN).tolist(),
            disorder_khz=float(disorder_khz),
            floquet_cycles=AGP_CYCLES,
            multiparity_confusion_matrix=CONFUSION_MATRIX_FNAME,
            postprocess=False, show=False, log=False,
        )
        scan_expts[direction].append(scan_expt)
        print(f"Direction {direction}, W={disorder_khz:g} kHz: {scan_expt.fname}")
        print("Occupations:", scan_expt.data["occupations"].shape,
              "cycles:", len(scan_expt.data["xpts"]))

# %% tags=["agp_display"]
raw = False  # True selects uncorrected correlations.
suffix = "" if raw else "_corrected"
for direction, expts in scan_expts.items():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for axis, key, moment in [
        (axes[0], "nmod4_ensemble_averaged_correlation", 1),
        (axes[1], "nmod4_second_moment_ensemble_averaged_correlation", 2),
    ]:
        for expt in expts:
            axis.plot(expt.data["xpts"], expt.data[key + suffix],
                      label=f"W={expt.cfg.expt.disorder_khz:g} kHz")
        axis.set(xlabel="Floquet cycles", ylabel="Ensemble-averaged correlation",
                 title=f"Direction {direction}, moment {moment} ({'raw' if raw else 'corrected'})")
        axis.legend()
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Assemble disorder scans
#
# Each direction has a separate bundle. The assembler checks cycle axes and
# measurement settings, keeps each JOB's correction, and copies complete AGP
# children into a new HDF5. The root contains axes and results without duplicate
# idata/qdata. save_data() writes a new file under the experiment's assembled_data.
#
# To inspect a saved bundle later, use
# CentralBosonAGPDisorderScan.from_h5file("path/to/assembled.h5").

# %% tags=["assemble"]
disorder_scans = {}
for direction, expts in scan_expts.items():
    scan = CentralBosonAGPDisorderScan(
        [expt.fname for expt in expts], notes=f"Central-boson correlations, direction {direction}",
    )
    scan.analyze()
    scan.save_data()
    disorder_scans[direction] = scan
    print(f"Direction {direction}: {scan.fname}")

# %% tags=["disorder_display"]
for scan in disorder_scans.values():
    for moment in (1, 2):
        fig, axis = scan.display(moment=moment, raw=False)
        plt.show()
        plt.close(fig)
