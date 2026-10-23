# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AGP layers on the device: a logic check at low depth (guan + Claude, 2026-10-23)
#
# Checks that the three population layers acquire through the queue as designed
# (experiments/qsim/fock_population.py, agp.py): FockPopulationExperiment jobs
# (one state, one read mode, a few seconds each), a CentralBosonAGPExperiment per W,
# and a CentralBosonAGPDisorderScan. Few reps, few cycles: the numbers are not physics.
# Interleaves with other users' jobs (batch_size=3). avoid_yoko=True keeps the coupler
# at the current the worker has (seb's overnight runs: 0.30 mA), so the S4-S7 swaps,
# calibrated at 0.25 mA, are off; that does not matter for the logic.

# %% Cell 1: station on the s4s7 snapshots (measurement_notebooks/guan/mbr_s4s7_n2.py, the
# config versions its 10-22 jobs ran with), and the population runner (its cell 8).
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.qsim.agp import CentralBosonAGPDisorderScan
from experiments.qsim.fock_population import FockPopulationExperiment, FockPopulationSetExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS
from job_server import JobClient

config_dict = {
    "hardware_config": "CFG-HW-20261022-00017",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261020-00016",
    "floquet_storage_swap": "CFG-FL-20261017-00034",
}

client = JobClient()
client.print_queue()

station = MultimodeStation(
    user="guan",
    experiment_name="261023_agp_layers_check",
    project="AGP",
    log_measurements=False,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)

STORS = [4, 5, 6, 7]
population_defaults = AttrDict(dict(
    expts=1, reps=100, rounds=1, qubits=[0], f0g1_cavity=1,
    normalize=False, active_reset=True, man_reset=True,
    storage_reset=STORS, pre_relax_delay=100, relax_delay=200,
    prepulse=False, postpulse=True, init_fock=False, init_stor=0,
    readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180,
    occupations=[1, 0, 0, 0, 0], swap_stors=STORS,
    use_multiphoton_swap=False, storage_pulse_wait_us=0.2, ro_stor=0,
    update_phases=True, detunings=[0.0] * len(STORS),
    floquet_cycles=[0], swept_params=["floquet_cycle"],
    avoid_yoko=True,
))
population_defaults.update(ACTIVE_RESET_DEFAULTS)
population_defaults.update(FLOQUET_DEFAULTS)

runner = CharacterizationRunner(
    station=station, ExptClass=FockPopulationExperiment,
    default_expt_cfg=population_defaults, job_client=client, use_queue=True, show=False,
)

SCRATCH = os.environ.get("AGP_CHECK_FIGURES", ".")


def save_figs(tag):
    paths = []
    for k, num in enumerate(plt.get_fignums()):
        path = os.path.join(SCRATCH, f"{tag}_{k}.png")
        plt.figure(num).savefig(path, dpi=80)
        paths.append(path)
    plt.close("all")
    print("saved", paths)


# %% Cell 2: readout confusion as a population set: Fock 0..3 in M1, zero cycles, read M1. Ran
# 2026-10-23 00:59 (JOB-20261023-00011..00014): not diagonal at 0.30 mA (1 reads as 3, 3 as 0;
# condition number 31). The settings are for 0.25 mA; the cause is not investigated.
# (4 jobs). C[n] is the raw label distribution of job n.
calibration_set = FockPopulationSetExperiment(
    [([number, 0, 0, 0, 0], 0) for number in range(4)], [0], STORS, [0.] * 4,
    notes="readout confusion for the AGP layers check")
calibration_set.acquire(runner, batch_size=3, reps=600, log=False, show=False)
calibration_set.analyze()
calibration_manifest = calibration_set.save()
confusion = calibration_set.data.p_mod[:, 0, :]
print("C[prepared, measured]:\n", np.round(confusion, 3), "\ncondition number", np.linalg.cond(confusion))
calibration = dict(matrix=confusion.tolist(), fname=str(calibration_manifest),
                   job_id=",".join(calibration_set.job_ids))

# %% Cell 3: a two-W scan, N=2 prepared and read in S7 (5 states per W), cycles 0..90 by 10.
# Ran 2026-10-23 01:00-01:02: JOB-20261023-00015..00027, interleaved with seb 00018/22/26;
# scan manifest assembled_data/261023_010214_CentralBosonAGPDisorderScan.yaml.
# The detuning pattern is the 10-17 scans' (unit RMS); W = 0 and 60 kHz. 10 jobs.
PATTERN = [-0.7067, 0.1947, -1.0346, 1.5467]
scan = CentralBosonAGPDisorderScan.from_pattern(
    [0, 60], PATTERN, photon_sector_number=2, direction=7, swap_stors=STORS,
    cycles=list(range(0, 100, 10)), calibration=calibration, notes="AGP layers check, low depth")
scan.acquire(runner, batch_size=3, reps=100, log=False, show=False)
scan.analyze()
scan_manifest = scan.save()
print("scan manifest:", scan_manifest)
print("parts:", [str(part.manifest_path) for part in scan.children])
print("job IDs:", [part.job_ids for part in scan.children])

# %% Cell 4: look at each layer from its own file, without the layer below.
job = scan.children[1].children[0]
loaded = FockPopulationExperiment.from_h5file(job.fname, load_shots=False)
loaded.display()
part = type(scan.children[1]).from_h5file(scan.children[1].manifest_path.with_suffix(".h5"))
part.display()
view = CentralBosonAGPDisorderScan.from_h5file(scan_manifest.with_suffix(".h5"))
view.display(moment=1)
calibration_set.display()
save_figs("agp_layers_check")
for path in [job.fname, part.fname, view.fname]:
    print(f"{os.path.getsize(path) / 1e3:8.1f} kB  {path}")
