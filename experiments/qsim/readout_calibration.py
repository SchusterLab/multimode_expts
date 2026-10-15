"""Single-shot calibration for long qsim campaigns.

Use with CharacterizationRunner: each subsequent job snapshots the updated
station readout settings. Active reset is disabled while measuring the clouds.
"""
from experiments.characterization_runner import CharacterizationRunner
from experiments.readout_calibration import apply_singleshot_calibration as singleshot_postproc
from experiments.single_qubit.single_shot import HistogramExperiment
from slab import AttrDict


def singleshot_runner(station, client=None, use_queue=False, reps=5000,
                      avoid_yoko=True):
    """A g/e Histogram runner with the standard calibration postprocessor."""
    return CharacterizationRunner(
        station=station, ExptClass=HistogramExperiment,
        default_expt_cfg=AttrDict(dict(
            reps=reps, relax_delay=2000, check_f=False, active_reset=False,
            man_reset=False, storage_reset=False, qubit=0, qubits=[0],
            pulse_manipulate=False, prepulse=False, pre_sweep_pulse=None,
            gate_based=True, avoid_yoko=avoid_yoko)),
        postprocessor=singleshot_postproc, job_client=client, use_queue=use_queue)


def recalibrate_before_batch(runner, **execute_kwargs):
    """Callback for Spectrum.acquire; persist calibration files and queue IDs.

    Mock runners skip calibration postprocessing by their normal policy.
    On hardware, failure to fit the Histogram prevents the next batch.
    """
    def callback(spectrum, start):
        expt = runner.execute(active_reset=False, **execute_kwargs)
        return dict(readout_calibration=dict(
            files=[str(expt.fname)],
            job_ids=[str(job) for job in runner.last_job_ids],
            occupation_start=int(start)))
    return callback
