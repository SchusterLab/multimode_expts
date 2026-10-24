# %% [markdown]
# # Confusion matrix after seb's 10-23 readout optimization (guan + Claude, 2026-10-23)
#
# CFG-HW-20261023-00039 = CFG-HW-20261023-00016 (0.25 mA, phase refreshed 12:2x) with
# device.readout copied from seb's CFG-HW-20261023-00036 (750.1 MHz, gain 2300, 1.5 us).
# Checks the readout at 0.25 mA with a histogram, then compares the one-job
# MultiParityConfusionMatrixExperiment with the 4-job FockPopulationSetExperiment.

# %%
import time

import h5py
import numpy as np
from slab import AttrDict

import experiments as meas
from experiments import CharacterizationRunner, MultimodeStation
from experiments.qsim.fock_population import FockPopulationExperiment, FockPopulationSetExperiment
from experiments.qsim.multiparity_confusion_matrix import MultiParityConfusionMatrixExperiment
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS
from job_server import JobClient

client = JobClient()
station = MultimodeStation(
    user="guan", experiment_name="261023_agp_layers_check", project="AGP", log_measurements=True,
    hardware_config="CFG-HW-20261023-00039", multiphoton_config="CFG-MP-20260121-00001",
    storage_man_file="CFG-M1-20261020-00016", floquet_file="CFG-FL-20261017-00034",
)
assert abs(station.hardware_cfg.hw.yoko_coupler.current - 0.25e-3) < 1e-9
STORS = [4, 5, 6, 7]
common = AttrDict(dict(
    reps=600, rounds=1, qubits=[0], f0g1_cavity=1, normalize=False, active_reset=True,
    man_reset=True, storage_reset=STORS, pre_relax_delay=100, relax_delay=500, postpulse=True,
    init_fock=False, init_stor=0, ro_stor=0, readout="multiparity", parity_fast=False,
    cond_sec_phase=-90, phase_second_pulse=180, swap_stors=STORS, update_phases=True,
    detunings=[0.0] * len(STORS),
))
common.update(ACTIVE_RESET_DEFAULTS)
common.update(FLOQUET_DEFAULTS)
population_defaults = AttrDict(dict(
    common, expts=1, prepulse=False, occupations=[1, 0, 0, 0, 0], use_multiphoton_swap=False,
    storage_pulse_wait_us=0.2, floquet_cycles=[0], swept_params=["floquet_cycle"]))
calibration_defaults = AttrDict(dict(common, prepulse=True))


def eg_direction(fname):
    """-> (angle of median(e) - median(g) in raw I/Q, in degrees; separation)."""
    with h5py.File(fname, "r") as f:
        g = np.median(f["Ig"][()]) + 1j * np.median(f["Qg"][()])
        e = np.median(f["Ie"][()]) + 1j * np.median(f["Qe"][()])
    return np.degrees(np.angle(e - g)), abs(e - g)


# %% Histogram check at 0.25 mA: JOB-20261023-00441 was submitted on this config.
CHECK_JOB = "JOB-20261023-00441"
while (status := client.get_status(CHECK_JOB)).status in ("pending", "running"):
    time.sleep(5)
print(CHECK_JOB, status.status, status.data_file_path)
angle, separation = eg_direction(status.data_file_path)
print(f"CHECK e-g direction {angle:.2f} deg, separation {separation:.1f}")

# %% One-job confusion matrix, then the 4-job set as the hardware cross-check.
cm_runner = CharacterizationRunner(
    station=station, ExptClass=MultiParityConfusionMatrixExperiment,
    default_expt_cfg=calibration_defaults, job_client=client, use_queue=True, show=False)
cm = cm_runner.execute()
one_job = np.asarray(cm.data["multiparity_confusion_matrix"])
print("ONEJOB", cm.fname)
print(np.round(one_job, 3), "cond", np.linalg.cond(one_job))

population_runner = CharacterizationRunner(
    station=station, ExptClass=FockPopulationExperiment,
    default_expt_cfg=population_defaults, job_client=client, use_queue=True, show=False)
calibration_set = FockPopulationSetExperiment(
    [([number, 0, 0, 0, 0], 0) for number in range(4)], [0], STORS, [0.] * 4,
    notes="readout confusion with seb's 10-23 readout (CFG-HW-20261023-00039), "
          "cross-check of the one-job class")
calibration_set.acquire(population_runner, batch_size=4, show=False)
calibration_set.analyze()
manifest = calibration_set.save()
four_jobs = calibration_set.data.p_mod[:, 0, :]
print("SET", calibration_set.job_ids, manifest)
print(np.round(four_jobs, 3), "cond", np.linalg.cond(four_jobs))
print("MAXDIFF", np.abs(one_job - four_jobs).max())
