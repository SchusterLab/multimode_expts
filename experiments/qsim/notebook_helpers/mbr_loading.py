"""Job-number-to-file loading for the MBR analysis notebooks.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cell 4 by
the stage-2 notebook split. That cell defined all 71 of the notebook's helpers
at once; these six are the only ones the active MBR analysis themes call, and
all six do the same job: turn a job date and number into a path, then into a
loaded experiment.

Callers: `analysis_notebooks/202609_qsim_migration/mbr.py`,
`mbr_disorder.py`, `mbr_sampling.py`, `mbr_spectral_validation.py`.

This is a temporary home. The stage-2 instructions are explicit that these are
not to be reconciled with the existing library loaders yet -- notably
`EncodingHamiltonianSpectroscopyExperiment.from_h5file`, which is the canonical
path and which `load_encoding_spectroscopy` here wraps. Deciding which one
survives is a later, theme-specific task.

`tqdm`, `EncodingHamiltonianSpectroscopyExperiment` and
`MBRPhaseCorrectionExperiment` were notebook globals that these bodies relied
on; they are imported properly here, since the move is what broke them.
"""

import os

import numpy as np
from tqdm.notebook import tqdm

import experiments as meas
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.mbr_phase_correction import MBRPhaseCorrectionExperiment

# Notebook globals in cell 2 of data_postprocess.ipynb. Kept as defaults so
# these functions stay callable, but every function still takes `basedir`
# explicitly so a notebook can point elsewhere without editing this file.
REPO_ROOT = os.path.join("C:", os.sep, "python", "multimode_expts")
BASE_DIR = os.path.join("C:", os.sep, "experiments")


def check_program_class(obj,
                        class_name):
    if obj.prog.__class__.__name__ == class_name:
        return True
    return False


def job_id_generator(job_date, job_start_num, job_finish_num, step=1):
    starts = [job_start_num] if isinstance(job_start_num, (int, np.integer)) else list(job_start_num)
    finishes = [job_finish_num] if isinstance(job_finish_num, (int, np.integer)) else list(job_finish_num)
    dates = [job_date] * len(starts) if isinstance(job_date, (str, int, np.integer)) else list(job_date)
    steps = [step] * len(starts) if isinstance(step, (int, np.integer)) else list(step)
    if not (len(dates) == len(starts) == len(finishes) == len(steps)): raise ValueError('job range arguments must have matching lengths')
    return [f'JOB-{date}-{job:05d}' for date, start, finish, stride in zip(dates, starts, finishes, steps) for job in range(int(start), int(finish) + 1, int(stride))]


def hdf5_path_generator(project_name,
                        job_date,
                        job_start_num,
                        job_finish_num, 
                        experiement_name,
                        basedir = BASE_DIR,
                        verbose = False):
    object_directory = os.path.join(basedir, project_name, "data")
    path = {}
    hdf5_path_to_return = []
    for job_id in job_id_generator(job_date, job_start_num, job_finish_num):
        filepath = os.path.join(object_directory, f"{job_id}_{experiement_name}.h5")
        if os.path.exists(filepath):
            hdf5_path_to_return.append(filepath)
            if verbose: print(f"[O] Found: {filepath}")
        elif verbose: print(f"[X] Missing: {filepath}")
    path[experiement_name] = hdf5_path_to_return
    return path


def path_to_experiment(hdf5_path, ExpClass):
    
    exp_name = list(hdf5_path.keys())[0]
    path_list = hdf5_path[exp_name]

    exp_list = []

    for fname in tqdm(path_list):
        obj = ExpClass.from_h5file(fname)

        obj.path = os.path.dirname(fname)
        obj.config_file = fname
        obj.prefix = os.path.splitext(os.path.basename(fname))[0]

        exp_list.append(obj)

    return exp_list


def load_encoding_spectroscopy(EncSpec, calibration_job_ids, spectroscopy_job_ids, client=None, calibration_program_name='EntireFloquetCyclePhaseCalibrationProgram', spectroscopy_program_name='NPhotonHamiltonianSpectroscopyProgram'):
    if client is None:
        from job_server import JobClient
        client = JobClient()
    def load_selected(job_ids, selector):
        expts, loaded_job_ids, skipped_job_ids = [], [], []
        for job_id in job_ids:
            try:
                result = client.get_status(job_id)
                if not result.is_successful():
                    skipped_job_ids.append(job_id)
                    continue
                expt = result.load_expt()
                keep = selector(expt)
            except Exception:
                skipped_job_ids.append(job_id)
                continue
            if keep:
                expts.append(expt)
                loaded_job_ids.append(job_id)
            else:
                skipped_job_ids.append(job_id)
        return expts, loaded_job_ids, skipped_job_ids
    calibration_expts, calibration_loaded_job_ids, calibration_skipped_job_ids = load_selected(calibration_job_ids, lambda expt: check_program_class(expt, calibration_program_name))
    spectroscopy_expts, spectroscopy_loaded_job_ids, spectroscopy_skipped_job_ids = load_selected(spectroscopy_job_ids, lambda expt: isinstance(expt, EncodingHamiltonianSpectroscopyExperiment) or check_program_class(expt, spectroscopy_program_name))
    calibration_expt = MBRPhaseCorrectionExperiment._from_expts(calibration_expts, job_ids=calibration_loaded_job_ids)
    spectroscopy_expt = EncSpec._from_expts(spectroscopy_expts, job_ids=spectroscopy_loaded_job_ids)
    calibration_expt.skipped_job_ids = calibration_skipped_job_ids
    spectroscopy_expt.skipped_job_ids = spectroscopy_skipped_job_ids
    calibration_expt.analyze()
    return calibration_expt, spectroscopy_expt


def load_dark_experiments(project, job_date_list, start_num_list, finish_num_list,
                          ExpClass=meas.qsim.floquet_dark_mode_readout.DarkBaseExperiment):
    """Reconstruct (and dedup) an experiment dataset from *.h5 files.

    Defaults to DarkBaseExperiment; pass e.g. meas.qsim.qsim_base.QsimBaseExperiment
    to load Qsim runs instead. The .h5 filename tag is taken from ExpClass.__name__,
    so it must match how the files were saved: JOB-<date>-<num>_<ExpClass>.h5."""
    path = hdf5_path_generator(project, job_date_list, start_num_list,
                               finish_num_list, ExpClass.__name__)
    expts = path_to_experiment(path, ExpClass)
    return list(dict.fromkeys(expts))
