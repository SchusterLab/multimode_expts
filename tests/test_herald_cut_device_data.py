"""The herald cut of the population analysis on guan's saved 11-13 jobs (0.45 mA).

FockPopulationExperiment.analyze() must keep exactly the shots of guan's offline cut
(fock_prep_herald.labels_from_h5 + select(herald=True)), and give his corrected walk numbers
(vault entry ANA-20261113-142706). Needs the job files under C:\\experiments (pippin); skipped
elsewhere. Read only: nothing is saved.
"""
import glob
import json

import h5py
import numpy as np
import pytest

from experiments.qsim.fock_population import FockPopulationExperiment
from experiments.qsim.fock_prep_herald import labels_from_h5, select
from fitting.qsim import f_postselect

WALK_END = [f"JOB-20261113-{n:05d}" for n in (408, 412, 414, 418, 422)]   # prep_herald="end"
ROWS = {(0, "none"): "JOB-20261113-00398", (1, "end"): "JOB-20261113-00402",
        (2, "end"): "JOB-20261113-00403", (3, "end"): "JOB-20261113-00404"}


def _path(job_id):
    paths = glob.glob(rf"C:\experiments\*\data\{job_id}_*.h5")
    return paths[0] if len(paths) == 1 else None


pytestmark = pytest.mark.skipif(
    any(_path(job) is None for job in WALK_END + list(ROWS.values())),
    reason="guan's 11-13 job files are not on this machine")


def test_analysis_keeps_exactly_guans_shots():
    for job_id in WALK_END:
        path = _path(job_id)
        with h5py.File(path, "r") as handle:
            assert json.loads(handle.attrs["config"])["expt"]["prep_herald"] == "end"
        job = FockPopulationExperiment.from_h5file(path)
        job.cfg.expt.multiparity_confusion_matrix = None  # raw p_mod
        job.analyze()
        _, labels = labels_from_h5(path)
        keep = select(labels, herald=True)
        np.testing.assert_array_equal(
            np.stack([job.data[f"p_mod{n}"] for n in range(4)], axis=-1),
            f_postselect.n_distribution(labels["n"], keep))
        np.testing.assert_array_equal(job.data["shot_counts"], keep.sum(axis=1))
        assert 0.85 < job.data["kept_fraction"].mean() < 0.95


def test_corrected_walk_gives_guans_numbers():
    rows = []
    for (_, _), job_id in sorted(ROWS.items()):
        cfg, labels = labels_from_h5(_path(job_id))
        point = list(cfg.expt.ro_stors).index(0) if "ro_stors" in cfg.expt else 0
        rows.append((labels["n"][point], select(labels, herald=True)[point]))
    matrix = f_postselect.confusion_from_shots([n for n, _ in rows], [k for _, k in rows])
    total, empty = 0, []
    for job_id in WALK_END:
        cfg, labels = labels_from_h5(_path(job_id))
        corrected = f_postselect.corrected_mean_n(labels["n"], matrix, select(labels, herald=True))
        total = total + corrected
        if int(cfg.expt.ro_stor) != 0:
            empty.append(corrected[0])
    assert total[0] == pytest.approx(3.23, abs=0.005)
    assert total.mean() == pytest.approx(2.88, abs=0.005)
    assert 0.01 < min(empty) and max(empty) < 0.09
