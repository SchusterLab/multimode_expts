"""Job-number-to-experiment loading for the MBR analysis notebooks.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cell 4 by
the stage-2 notebook split, then reduced to two functions when the loading
path was purged of the job server.

Callers: `analysis_notebooks/202609_qsim_migration/mbr.py`,
`mbr_sampling.py`, `mbr_spectral_validation.py`, and
`notebook_helpers/mbr_n3_reprocess.py`.

What changed and why
--------------------
`load_encoding_spectroscopy` used to ask a `JobClient` for each job's status
and then unpickle a whole Experiment. It now goes through
:mod:`experiments.saved_jobs`, which reads HDF5 only. Two consequences worth
knowing at the call site:

* Filtering is by the program class **recorded in the provenance sidecar**,
  not by the class of an unpickled `expt.prog`. A mixed job range is therefore
  filtered before any file is opened, and the answer no longer depends on the
  acquisition revision still being importable.
* There is no `client` argument. Nothing here talks to the job server, the job
  database, or a station.

The `hdf5_path_generator`/`path_to_experiment`/`load_dark_experiments` trio
that used to live here is gone: it globbed a hard-coded `C:\\experiments` and
loaded children without their Floquet timing.
:func:`experiments.job_paths.resolve_job_paths` does the path resolution
properly, and works off the acquisition workstation. The dormant notebooks
under `analysis_notebooks/202609_qsim_migration/dormant/` keep their own inline
copies, so they are unaffected.
"""

import numpy as np

from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.saved_jobs import load_aggregate

CALIBRATION_PROGRAM = "EntireFloquetCyclePhaseCalibrationProgram"
SPECTROSCOPY_PROGRAM = "NPhotonHamiltonianSpectroscopyProgram"


def job_id_generator(job_date, job_start_num, job_finish_num, step=1):
    """-> the `JOB-<date>-<number>` IDs for one or more inclusive ranges.

    Every argument broadcasts: pass scalars for a single range, or matching
    lists to union several. Unchanged from cell 4.
    """
    starts = [job_start_num] if isinstance(job_start_num, (int, np.integer)) else list(job_start_num)
    finishes = [job_finish_num] if isinstance(job_finish_num, (int, np.integer)) else list(job_finish_num)
    dates = [job_date] * len(starts) if isinstance(job_date, (str, int, np.integer)) else list(job_date)
    steps = [step] * len(starts) if isinstance(step, (int, np.integer)) else list(step)
    if not (len(dates) == len(starts) == len(finishes) == len(steps)):
        raise ValueError('job range arguments must have matching lengths')
    return [f'JOB-{date}-{job:05d}'
            for date, start, finish, stride in zip(dates, starts, finishes, steps)
            for job in range(int(start), int(finish) + 1, int(stride))]


def load_encoding_spectroscopy(EncSpec,
                               calibration_job_ids,
                               spectroscopy_job_ids,
                               timing=None,
                               calibration_program_name=CALIBRATION_PROGRAM,
                               spectroscopy_program_name=SPECTROSCOPY_PROGRAM):
    """-> (calibration_expt, spectroscopy_expt) for one encoding-spectroscopy set.

    Both aggregates come from HDF5 alone. The calibration is analyzed before
    it is returned, since the spectroscopy analysis takes it as an input; the
    spectroscopy aggregate is not, because its analysis takes parameters that
    are a per-notebook choice.

    Args:
        EncSpec: the stage class that reassembles the spectroscopy jobs,
            e.g. `MBRSpectrumExperiment`.
        calibration_job_ids, spectroscopy_job_ids: job IDs, nested lists fine.
        timing: historical `dict(floquet_cycle_us=..., m1s_pi_fracs=[...])`,
            for files that carry neither a `derived_params` attribute nor a
            sidecar entry. Omit it whenever provenance is available.
        calibration_program_name, spectroscopy_program_name: the recorded
            program classes to keep. The spectroscopy range for these datasets
            was submitted interleaved with other programs, which is why
            filtering exists at all.

    Both returned aggregates carry `skipped_job_ids` (and `skipped_jobs`, with
    the reason per job) so a notebook can see what was dropped.
    """
    calibration_expt = load_aggregate(
        calibration_job_ids,
        owner=MBRPhaseCorrectionExperiment,
        program_class=calibration_program_name,
        timing=timing,
        analyze=True,
    )
    spectroscopy_expt = load_aggregate(
        spectroscopy_job_ids,
        owner=EncSpec,
        program_class=spectroscopy_program_name,
        timing=timing,
    )
    return calibration_expt, spectroscopy_expt
