# -*- coding: utf-8 -*-
"""Queue submission for a whole batch of jobs, a bounded number at a time.

``CharacterizationRunner`` submits one job and waits. A campaign submits
hundreds -- one per disorder realization per occupation -- and waiting for
each in turn wastes the queue. ``BatchRunner`` submits ``batch_size`` jobs,
waits for that group, then submits the next, so the worker always has work
without the whole campaign being queued at once.

It is infrastructure, not physics, which is why it is a top-level module
beside ``characterization_runner`` rather than under ``qsim/``: nothing in it
knows what is being measured. It does record what was measured --
``experiment_class``/``experiment_module`` and ``program_class``/
``program_module`` go into the job with the config -- and that recording is
the reason class names and modules in this tree count as provenance.

``execute`` needs a ``job_client`` and so only runs where the job server is
reachable.
"""
from copy import deepcopy

import numpy as np
from slab import AttrDict

from experiments.characterization_runner import CharacterizationRunner


def _is_real_job_client(client):
    """Does this client talk to the actual job server?

    Imported lazily so batch_runner does not pull in job_server at import
    time, and returns False if job_server is unavailable -- in which case
    nothing could reach a real queue anyway.
    """
    try:
        from job_server import JobClient
    except Exception:
        return False
    return isinstance(client, JobClient)


class BatchRunner(CharacterizationRunner):
    """CharacterizationRunner with bounded parallel queue submission."""

    @staticmethod
    def _plain(obj):
        """
        Convert config values only at the queue's JSON boundary.
        Non JSON compatible objects are converted into compatible ones.
        """
        if isinstance(obj, dict):
            return {key: BatchRunner._plain(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BatchRunner._plain(value) for value in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    def execute(self,
                  configs,
                  batch_size=10,
                  postprocess=True,
                  priority=0,
                  poll_interval=2.,
                  timeout=None,
                  log=None,
                  show=None,
                  allow_queue_in_mock=False):
        """Submit at most batch_size jobs, then collect them in config order."""
        if self.job_client is None:
            raise ValueError("job_client is required")

        # This override is queue-only: unlike CharacterizationRunner.execute,
        # which switches to run_local() when the station is mock, there is no
        # local batch path. So a mock session would submit real jobs to the
        # real queue, where the worker runs whatever is checked out at
        # C:\python\multimode_expts -- not the caller's worktree, and against
        # real hardware unless that worker was itself started with --mock.
        #
        # The hazard is specifically mock station + real queue. A test driving
        # a fake client is not at risk, so check the client's type rather than
        # is_mock alone.
        if (getattr(self.station, "is_mock", False)
                and not allow_queue_in_mock
                and _is_real_job_client(self.job_client)):
            raise RuntimeError(
                "BatchRunner.execute submits to the job queue and has no "
                "local mock path, but this station has mock instruments. "
                "The queue worker would run the main checkout against real "
                "hardware unless it was started with --mock. Start a mock "
                "worker and pass allow_queue_in_mock=True if that is what "
                "you want."
            )
        if (isinstance(batch_size, (bool, np.bool_))
                or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")

        configs = list(configs) #list of config dictionary that is overrided in the submitted job
        if not configs:
            raise ValueError("configs cannot be empty")
        expts = []
        self.last_job_ids = []
        program_module = None
        program_class = None
        if self.program is not None:
            program_module = self.program.__module__
            program_class = self.program.__name__

        for start in range(0, len(configs), batch_size):
            pending = []
            batch_configs = [self.preprocessor(self.station, self.default_expt_cfg, **overrides) for overrides in configs[start:start + batch_size]]
            station_config = self._serialize_station_config()
            print(f"batch {start // batch_size + 1}: {len(batch_configs)} jobs")
            try:
                for cfg in batch_configs:
                    job_id = self.job_client.submit_job(
                        experiment_class=self.ExptClass.__name__,
                        experiment_module=self.ExptClass.__module__,
                        expt_config=self._plain(dict(cfg)), 
                        station_config=station_config,
                        user=self.station.user, 
                        priority=priority,
                        program_class=program_class, 
                        program_module=program_module,
                    )
                    pending.append(job_id)
                    self.last_job_ids.append(job_id)

                for job_id in list(pending):
                    result = self.job_client.wait_for_completion(
                        job_id, poll_interval=poll_interval, timeout=timeout, verbose=False)
                    pending.pop(0)
                    self.last_job_result = result
                    if not result.is_successful():
                        raise RuntimeError(
                            f"Job {job_id} {result.status}: {result.error_message or 'No details'}")
                    expt = result.load_expt()
                    if postprocess:
                        self.postprocessor(self.station, expt)
                    self._render_log_show(expt, show=show, log=log, display_kwargs=None)
                    expts.append(expt)
            except BaseException: #BaseException is inherited by Exception, KeyboardInterrupt, SystemExit, GeneratorExit
                # Below is to cancel the running job when there is KeyboardInterrupt
                for job_id in pending:
                    try:
                        self.job_client.cancel_job(job_id)
                    except Exception:
                        pass
                raise
        if self.program is not None:
            batch_expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=f"{self.ExptClass.__name__}_batch",
                config_file=self.station.hardware_config_file,
                program=self.program,
            )
        else:
            batch_expt = self.ExptClass(
                soccfg=self.station.soccfg,
                path=self.station.data_path,
                prefix=f"{self.ExptClass.__name__}_batch",
                config_file=self.station.hardware_config_file,
            )
        batch_expt.cfg = AttrDict(deepcopy(self.station.hardware_cfg))
        batch_expt.cfg.expt = deepcopy(self.default_expt_cfg)
        batch_expt.data = AttrDict()
        batch_expt.batch_expts = expts
        batch_expt.batch_job_ids = list(self.last_job_ids)
        batch_expt._analysis_station = self.station
        return batch_expt
