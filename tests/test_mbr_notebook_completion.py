"""Physics conventions and calibration ordering for the completed migration."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from slab import AttrDict

from experiments.qsim.mbr_disorder_ensemble import DiagDisorderConfig, plan_diagonal_disorder
from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
from experiments.qsim.mbr_time_trace import MBRTimeTraceExperiment
from experiments.qsim.readout_refit import refit_readout
from experiments.readout_calibration import apply_singleshot_calibration
from fitting.qsim.mbr_disorder import disorder_direction


def calibration():
    return SimpleNamespace(data=AttrDict(dict(
        phase_mod180=True, hardware=AttrDict(dict(
            physical_kerr_MHz=-0.004, couplings_MHz=[0.03]*4,
            floquet_cycle_us=0.4)))))


def test_full_basis_rms_plan_and_historical_norm():
    plan = plan_diagonal_disorder(calibration(), [1, 2, 3, 4],
                                  DiagDisorderConfig(realization_count=2))
    assert plan.total_jobs == 70
    for record in plan.realizations:
        assert len(record['selected_occupations']) == 35
        assert record['state_selection'] == 'full' and record['normalization'] == 'rms'
        assert np.std(record['onsite_MHz']) == pytest.approx(0.05)
        assert np.mean(record['onsite_MHz']) == pytest.approx(0., abs=1e-14)
        np.testing.assert_allclose(record['direction'],
                                   2*disorder_direction(record['seed'], 4))
        spectrum = MBRSpectrumExperiment(
            record['selected_occupations'], plan.cycles, [1, 2, 3, 4],
            detunings=(-np.asarray(record['onsite_MHz'])).tolist(),
            realization_record=record)
        for job in spectrum.job_overrides():
            assert job['disorder_realization_record'] == record
            np.testing.assert_allclose(job['detunings'], -np.asarray(record['onsite_MHz']))


def test_custom_plan_does_not_select_from_theory():
    occupations = [[3, 0, 0, 0, 0], [0, 0, 0, 0, 3]]
    config = DiagDisorderConfig(realization_count=1, state_selection='custom',
                                occupations=occupations, normalization='norm')
    plan = plan_diagonal_disorder(calibration(), [1, 2, 3, 4], config)
    assert plan.total_jobs == 2
    assert plan.realizations[0]['selected_occupations'] == occupations
    assert np.linalg.norm(plan.realizations[0]['direction']) == pytest.approx(1.)
    config.occupations = [[2, 0, 0, 0, 0]]
    with pytest.raises(ValueError, match='fixed-N'):
        plan_diagonal_disorder(calibration(), [1, 2, 3, 4], config)


def test_calibration_finishes_before_next_batch_and_records_all_job_ids():
    spectrum = MBRSpectrumExperiment([[1, 0], [0, 1], [1, 0]], [0, 1], [1])
    events = []
    state = {'version': 0}

    def hook(spectrum, start):
        state['version'] += 1
        events.append(('cal', start))
        return dict(readout_calibration={'version': state['version']})

    class Runner:
        ExptClass = MBRTimeTraceExperiment
        last_job_ids = []

        def execute(self, overrides, **kwargs):
            version = state['version']
            assert all(job['readout_calibration']['version'] == version for job in overrides)
            events.append(('traces', len(overrides), version))
            self.last_job_ids = [f'{version}-{i}' for i in range(len(overrides))]
            return list(overrides)

    runner = Runner()
    spectrum.acquire(runner, batch_size=2, before_batch=hook)
    assert events == [('cal', 0), ('traces', 2, 1), ('cal', 2), ('traces', 1, 2)]
    assert spectrum.job_ids == ['1-0', '1-1', '2-0']

    def broken_hook(spectrum, start):
        raise ValueError('unusable readout')

    with pytest.raises(ValueError, match='unusable'):
        spectrum.acquire(runner, before_batch=broken_hook)
    assert events[-1] == ('traces', 1, 2)


def test_unusable_histogram_does_not_overwrite_readout():
    readout = AttrDict(dict(phase=[10.], Ig=[-1.], Ie=[1.]))
    station = SimpleNamespace(autocalib_path='unused',
                              hardware_cfg=AttrDict(dict(device=dict(readout=readout))))
    expt = SimpleNamespace(
        analyze=lambda **kwargs: None,
        cfg=AttrDict(dict(expt=dict(active_reset=False))),
        data=dict(fids=[1.], angle=5., thresholds=[0.],
                  confusion_matrix=np.eye(2), Ig_rot=[np.nan], Ie_rot=[1.]))
    with pytest.raises(ValueError, match='invalid readout'):
        apply_singleshot_calibration(station, expt)
    assert readout.phase == [10.] and readout.Ig == [-1.]


def test_offline_refit_uses_final_lane_and_does_not_mutate_raw_job(tmp_path):
    rng = np.random.default_rng(21)
    reps, npoints = 1500, 8
    rotation = 0.5
    centers = np.array([[-2., -2*np.tan(rotation)], [2., 2*np.tan(rotation)]])
    populations = np.array([.8, .2, .6, .4, .7, .3, .55, .45])
    labels = rng.random((npoints, reps)) < populations[:, None]
    final = centers[labels.astype(int)] + rng.normal(0., .25, (npoints, reps, 2))
    raw = np.zeros((npoints, reps*2, 2))
    raw[:, 0::2] = 99.  # reset lane must not enter the IQ fit
    raw[:, 1::2] = final
    offset = np.array([.12, -.08])
    means = final.mean(axis=1) + offset
    path = tmp_path/'trace.h5'
    with h5py.File(path, 'w') as handle:
        handle['idata'], handle['qdata'] = raw[..., 0], raw[..., 1]
    job = object.__new__(MBRTimeTraceExperiment)
    job.fname = str(path)
    job.cfg = AttrDict(dict(expt=dict(reps=reps, qubits=[0]),
                           device=dict(readout=dict(Ig=[-2.], Ie=[2.]))))
    job.data = AttrDict(dict(avgi=means[:, 0].reshape(2, 4),
                             avgq=means[:, 1].reshape(2, 4),
                             xpts=[[0., 0.], [180., 0.], [0., 90.], [180., 90.]],
                             ypts=[0, 1], complex_return=np.array([999., 999.])))
    original_file = path.read_bytes()
    corrected, fit = refit_readout(job)
    assert fit['angle_deg'] == pytest.approx(-np.degrees(rotation), abs=1.)
    assert fit['sample_count'] == npoints*reps
    np.testing.assert_allclose(fit['adc_offset'], offset)
    np.testing.assert_allclose(corrected.data['Pe'], labels.mean(axis=1).reshape(2, 4), atol=.02)
    np.testing.assert_array_equal(job.data['complex_return'], [999., 999.])
    np.testing.assert_array_equal(job.data['avgi'], means[:, 0].reshape(2, 4))
    assert job.cfg.device.readout.Ig == [-2.]
    assert path.read_bytes() == original_file
    job.cfg.expt.pre_selection_reset = True
    with pytest.raises(ValueError, match='postselected'):
        refit_readout(job)
