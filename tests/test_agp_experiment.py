"""Central-boson sweeps, number moments, correlations and save lifecycle."""

import json

import h5py
import numpy as np
import pytest
from slab import AttrDict
from slab.experiment import Experiment

from experiments.qsim.agp import CentralBosonAGPExperiment
from experiments.qsim.fock_population import FockPopulationProgram


OCCUPATIONS = [[1, 1], [2, 0]]
CYCLES = [0, 3]
POINT_LABELS = {
    ((1, 1), 0): [0, 0, 0, 0],
    ((1, 1), 3): [1, 1, 1, 1],
    ((2, 0), 0): [2, 2, 2, 2],
    ((2, 0), 3): [0, 1, 2, 3],
}
RAW_PROBABILITIES = np.array([
    [[1., 0., 0., 0.], [0., 1., 0., 0.]],
    [[0., 0., 1., 0.], [.25, .25, .25, .25]],
])
RAW_MEAN = np.array([[0., 1.], [2., 1.5]])
RAW_SECOND_MOMENT = np.array([[0., 1.], [4., 3.5]])
# C has 0.85 on its diagonal and 0.05 elsewhere: p_raw = 0.8*p_true + 0.05.
CONFUSION = .8 * np.eye(4) + .05


class ShotProgram:
    """Program protocol only; acquisition/reshaping is the real Qsim driver."""

    visits = []

    @classmethod
    def readouts_per_shot(cls, cfg):
        return 4

    def __init__(self, soccfg, cfg):
        self.point = (tuple(cfg.expt.occupations), cfg.expt.floquet_cycle)
        self.visits.append((self.point, cfg.expt.ro_stor, list(cfg.expt.detunings)))
        labels = np.asarray(POINT_LABELS[self.point])
        lanes = np.full((len(labels), 4), 100.)
        lanes[:, -2] = 2 * (labels % 2) - 1
        lanes[:, -1] = 2 * (labels // 2) - 1
        self.idata = lanes.reshape(-1)

    def acquire(self, soc, **kwargs):
        # The final lane's average is what QsimExperiment.acquire extracts.
        return np.array([[0., 0., 0., np.mean(POINT_LABELS[self.point])]]), np.zeros((1, 4))

    def collect_shots(self):
        return self.idata, np.zeros_like(self.idata)


@pytest.fixture
def experiment():
    expt = CentralBosonAGPExperiment.__new__(CentralBosonAGPExperiment)
    expt.cfg = AttrDict({
        'aliases': {'soc': 'synthetic'},
        'device': {'readout': {'threshold': [0.]}, 'qubit': {'f_ge': [5000.]}},
        'hw': {'soc': {}},
        'expt': {
            'photon_sector_number': 2, 'direction': 0,
            'floquet_cycles': list(CYCLES),
            'qubits': [0], 'readout': 'multiparity',
            'ro_stor': 7, 'swap_stors': [7], 'detunings': [.02],
        },
    })
    expt.soccfg = object()
    expt.im = {'synthetic': object()}
    expt._ProgramClass = ShotProgram
    expt.data = {}
    ShotProgram.visits = []
    return expt


def test_acquire_visits_every_occupation_and_cycle_with_raw_shots(experiment):
    data = experiment.acquire()
    assert ShotProgram.visits == [
        ((tuple(occupation), cycle), 7, [.02])
        for occupation in OCCUPATIONS for cycle in CYCLES
    ]
    np.testing.assert_array_equal(data['occupations'], OCCUPATIONS)
    assert data['occupations'].dtype.kind in 'iu'
    np.testing.assert_array_equal(data['xpts'], CYCLES)
    np.testing.assert_array_equal(data['ypts'], [0, 1])
    np.testing.assert_array_equal(data['avgi'], RAW_MEAN)
    assert np.asarray(data['idata']).shape == (4, 16)
    assert np.asarray(data['qdata']).shape == (4, 16)
    assert experiment.cfg.expt.photon_sector_number == 2
    assert experiment.generate_fock_strings() == OCCUPATIONS
    assert experiment.cfg.expt.floquet_cycles == CYCLES


@pytest.mark.parametrize('photons, storages, direction, expected_count', [
    (0, [], 0, 0), (0, [4, 5, 6, 7], 7, 0),
    (1, [4, 7], 0, 1), (1, [4, 7], 7, 1),
    (3, [4, 5, 6, 7], 0, 15), (3, [4, 5, 6, 7], 7, 15),
])
def test_generated_direction_sector_is_complete_and_json_serializable(
        experiment, photons, storages, direction, expected_count):
    experiment.cfg.expt.photon_sector_number = photons
    experiment.cfg.expt.swap_stors = storages
    experiment.cfg.expt.direction = direction
    modes = 1 + len(storages)
    selected_column = [0, *storages].index(direction)

    states = experiment.generate_fock_strings()

    assert len(states) == expected_count
    assert len(set(map(tuple, states))) == len(states)
    assert all(len(state) == modes and sum(state) == photons for state in states)
    assert all(isinstance(n, int) and n >= 0 for state in states for n in state)
    assert all(state[selected_column] > 0 for state in states)
    assert json.loads(json.dumps(states)) == states


def test_generated_sector_keeps_state_and_storage_column_order(experiment):
    assert experiment.generate_fock_strings() == [[1, 1], [2, 0]]
    experiment.cfg.expt.direction = 7
    assert experiment.generate_fock_strings() == [[0, 2], [1, 1]]


def test_direction_uses_physical_storage_id_with_reordered_storages(experiment):
    experiment.cfg.expt.photon_sector_number = 1
    experiment.cfg.expt.swap_stors = [7, 4]
    experiment.cfg.expt.direction = 4
    assert experiment.generate_fock_strings() == [[0, 0, 1]]


def test_direction_must_be_m1_or_a_configured_storage(experiment):
    experiment.cfg.expt.direction = 4  # Only Storage 7 is in this job.
    with pytest.raises(ValueError):
        experiment.generate_fock_strings()


@pytest.mark.parametrize('corrected', [False, True])
def test_analyze_classifies_and_corrects_each_occupation_cycle(experiment, corrected):
    acquired = experiment.acquire()
    if corrected:
        experiment.cfg.expt.multiparity_confusion_matrix = CONFUSION.tolist()
    data = experiment.analyze(acquired)
    assert data is experiment.data
    np.testing.assert_array_equal(data['mean_n_mod4'], RAW_MEAN)
    for n in range(4):
        np.testing.assert_array_equal(data[f'p_mod{n}'], RAW_PROBABILITIES[..., n])
        assert data[f'p_mod{n}'].shape == (2, 2)
        if corrected:
            np.testing.assert_allclose(
                data[f'p_mod{n}_corrected'], (RAW_PROBABILITIES[..., n] - .05) / .8,
            )
        else:
            assert f'p_mod{n}_corrected' not in data
    if corrected:
        np.testing.assert_allclose(data['nmod4_mean_corrected'], (RAW_MEAN - .3) / .8)
    else:
        assert 'nmod4_mean_corrected' not in data
    np.testing.assert_array_equal(data['nmod4_second_moment'], RAW_SECOND_MOMENT)
    assert data['nmod4_second_moment'].shape == (2, 2)
    assert data['sector_dimension'] == 3  # Full sector; only two states were measured.
    np.testing.assert_allclose(data['nmod4_ensemble_averaged_correlation'], [4 / 3, 4 / 3])
    np.testing.assert_allclose(data['nmod4_second_moment_ensemble_averaged_correlation'], [16 / 3, 5])
    if corrected:
        np.testing.assert_allclose(
            data['nmod4_second_moment_corrected'], (RAW_SECOND_MOMENT - .7) / .8,
        )
        np.testing.assert_allclose(data['nmod4_ensemble_averaged_correlation_corrected'], [31 / 24, 31 / 24])
        np.testing.assert_allclose(
            data['nmod4_second_moment_ensemble_averaged_correlation_corrected'], [125 / 24, 115 / 24],
        )
    else:
        for key in ['nmod4_second_moment', 'nmod4_ensemble_averaged_correlation',
                    'nmod4_second_moment_ensemble_averaged_correlation']:
            assert f'{key}_corrected' not in data
    expected_error = np.array([[0., 0.], [0., np.sqrt(5 / 12)]])
    if corrected:
        expected_error /= .8
    np.testing.assert_allclose(data['n_err'], expected_error, atol=1e-7)


def test_correlations_match_operator_trace_for_unitary_dynamics(experiment):
    # Two photons in two modes; select states with Storage 7 initially occupied.
    experiment.cfg.expt.direction = 7
    experiment.cfg.read_num = 4
    occupations = experiment.generate_fock_strings()
    assert occupations == [[0, 2], [1, 1]]
    # Each state's shots at t=0, then after a balanced beamsplitter.
    labels = np.array([[2, 2, 2, 2], [0, 1, 1, 2],
                       [1, 1, 1, 1], [0, 0, 2, 2]])
    lanes = np.full((*labels.shape, 4), 100.)
    lanes[..., -2] = 2 * (labels % 2) - 1
    lanes[..., -1] = 2 * (labels // 2) - 1
    data = experiment.analyze({
        'occupations': np.array(occupations), 'xpts': np.array(CYCLES),
        'ypts': np.arange(2), 'idata': lanes.reshape(4, -1),
        'qdata': np.zeros((4, 16)),
    })

    # Independent complete-basis operator calculation, including the omitted n=0 state.
    unitary = np.array([[.5, -np.sqrt(.5), .5],
                        [np.sqrt(.5), 0., -np.sqrt(.5)],
                        [.5, np.sqrt(.5), .5]])
    number = np.diag([0., 1., 2.])
    for power, key in [(1, 'nmod4_ensemble_averaged_correlation'),
                       (2, 'nmod4_second_moment_ensemble_averaged_correlation')]:
        operator = np.linalg.matrix_power(number, power)
        expected = [np.trace(operator @ operator) / 3,
                    np.trace(unitary.T @ operator @ unitary @ operator) / 3]
        np.testing.assert_allclose(data[key], expected)


def test_reanalysis_without_calibration_clears_corrected_moments_and_correlations(experiment):
    experiment.acquire()
    experiment.cfg.expt.multiparity_confusion_matrix = CONFUSION.tolist()
    experiment.analyze()
    for suffix in ['', '_corrected']:
        experiment.data[f'nmod4_correlation{suffix}'] = np.full(2, -999.)
        experiment.data[f'nmod4_second_moment_correlation{suffix}'] = np.full(2, -999.)
    experiment.cfg.expt.multiparity_confusion_matrix = None
    data = experiment.analyze()
    np.testing.assert_array_equal(data['nmod4_second_moment'], RAW_SECOND_MOMENT)
    np.testing.assert_allclose(data['nmod4_ensemble_averaged_correlation'], [4 / 3, 4 / 3])
    np.testing.assert_allclose(data['nmod4_second_moment_ensemble_averaged_correlation'], [16 / 3, 5])
    for key in ['nmod4_second_moment', 'nmod4_ensemble_averaged_correlation',
                'nmod4_second_moment_ensemble_averaged_correlation']:
        assert f'{key}_corrected' not in data
    for suffix in ['', '_corrected']:
        assert f'nmod4_correlation{suffix}' not in data
        assert f'nmod4_second_moment_correlation{suffix}' not in data


@pytest.mark.parametrize('storages, direction, cycles, shape, dimension', [
    ([7, 4, 5, 6], 4, [0, 2, 7], (15, 3), 35),
    ([7, 4, 5, 6], 4, [0], (15, 1), 35),
    ([], 0, [0, 2, 7], (1, 3), 1),
    ([], 0, [0], (1, 1), 1),
])
@pytest.mark.parametrize('corrected', [False, True])
def test_2d_analysis_preserves_each_point_through_save_and_reload(
        experiment, tmp_path, storages, direction, cycles, shape, dimension, corrected):
    experiment.cfg.expt.update(
        photon_sector_number=3, swap_stors=storages, direction=direction,
        floquet_cycles=cycles, ro_stor=0, detunings=[0.] * len(storages),
    )
    occupations = experiment.generate_fock_strings()
    counts = np.empty((*shape, 4), dtype=int)
    for state in range(shape[0]):
        for cycle in range(shape[1]):
            first_three = [state + 1, cycle + 1, (state + 2 * cycle) % 5]
            counts[state, cycle] = [*first_three, 32 - sum(first_three)]

    class GridShotProgram(ShotProgram):
        @classmethod
        def readouts_per_shot(cls, cfg):
            return 6

        def __init__(self, soccfg, cfg):
            state = occupations.index(cfg.expt.occupations)
            cycle = cycles.index(cfg.expt.floquet_cycle)
            self.labels = np.repeat(np.arange(4), counts[state, cycle])
            lanes = np.full((32, 6), 100.)
            lanes[:, -2] = 2 * (self.labels % 2) - 1
            lanes[:, -1] = 2 * (self.labels // 2) - 1
            self.idata = lanes.reshape(-1)

        def acquire(self, soc, **kwargs):
            averages = np.zeros((1, 6))
            averages[0, -1] = self.labels.mean()
            return averages, np.zeros_like(averages)

    experiment._ProgramClass = GridShotProgram
    if corrected:
        experiment.cfg.expt.multiparity_confusion_matrix = CONFUSION.tolist()
    experiment.fname = str(tmp_path / 'grid.h5')
    experiment.go(analyze=True, save=True)
    data = experiment.data

    np.testing.assert_array_equal(data['occupations'], occupations)
    np.testing.assert_array_equal(data['xpts'], cycles)
    np.testing.assert_array_equal(data['ypts'], np.arange(shape[0]))
    assert np.asarray(data['idata']).shape == (shape[0] * shape[1], 32 * 6)
    assert data['sector_dimension'] == dimension
    initial_n = np.asarray(occupations)[:, [0, *storages].index(direction)]
    raw_probability = counts / 32
    raw_mean = raw_probability @ np.arange(4)
    expected_error = np.sqrt((raw_probability @ np.array([0, 1, 4, 9]) - raw_mean**2) / 31)
    if corrected:
        expected_error /= .8
    assert data['n_err'].shape == shape
    np.testing.assert_allclose(data['n_err'], expected_error)

    for suffix in ['', '_corrected'] if corrected else ['']:
        probability = (raw_probability - .05) / .8 if suffix else raw_probability
        for n in range(4):
            assert data[f'p_mod{n}{suffix}'].shape == shape
            np.testing.assert_allclose(data[f'p_mod{n}{suffix}'], probability[..., n])
        mean = probability @ np.arange(4)
        second_moment = probability @ np.array([0, 1, 4, 9])
        assert data[f'nmod4_mean{suffix}'].shape == shape
        assert data[f'nmod4_second_moment{suffix}'].shape == shape
        np.testing.assert_allclose(data[f'nmod4_mean{suffix}'], mean)
        np.testing.assert_allclose(data[f'nmod4_second_moment{suffix}'], second_moment)
        for key, moment, weights in [
            ('nmod4_ensemble_averaged_correlation', mean, initial_n),
            ('nmod4_second_moment_ensemble_averaged_correlation', second_moment, initial_n**2),
        ]:
            assert data[f'{key}{suffix}'].shape == (shape[1],)
            np.testing.assert_allclose(
                data[f'{key}{suffix}'], np.sum(weights[:, None] * moment, axis=0) / dimension,
            )

    with h5py.File(experiment.fname, 'r') as saved:
        for key in data:
            np.testing.assert_array_equal(saved[key][()], data[key])
    loaded = CentralBosonAGPExperiment.from_h5file(experiment.fname)
    reanalyzed = loaded.analyze()
    for key in data:
        np.testing.assert_array_equal(reanalyzed[key], data[key])


def test_go_saves_calibration_and_occupation_table_for_reanalysis(
        experiment, tmp_path, monkeypatch):
    calibration_path = tmp_path / 'calibration.h5'
    with h5py.File(calibration_path, 'w') as saved:
        saved['multiparity_confusion_matrix'] = CONFUSION
    experiment.cfg.expt.multiparity_confusion_matrix = str(calibration_path)
    experiment.fname = str(tmp_path / 'central_boson.h5')

    experiment.go(analyze=True, save=True)
    with h5py.File(experiment.fname, 'r') as saved:
        config = json.loads(saved.attrs['config'])['expt']
        assert config['photon_sector_number'] == 2
        assert config['direction'] == 0
        assert config['floquet_cycles'] == CYCLES
        assert config['swept_params'] == ['occupations', 'floquet_cycle']
        assert config['multiparity_confusion_matrix']['fname'] == str(calibration_path)
        assert config['multiparity_confusion_matrix']['matrix'] == CONFUSION.tolist()
        np.testing.assert_array_equal(saved['occupations'][()], OCCUPATIONS)
        np.testing.assert_array_equal(saved['ypts'][()], [0, 1])
        for n in range(4):
            assert saved[f'p_mod{n}_corrected'].shape == (2, 2)
        assert saved['sector_dimension'][()] == 3
        for suffix in ['', '_corrected']:
            assert saved[f'nmod4_second_moment{suffix}'].shape == (2, 2)
            assert saved[f'nmod4_ensemble_averaged_correlation{suffix}'].shape == (2,)
            assert saved[f'nmod4_second_moment_ensemble_averaged_correlation{suffix}'].shape == (2,)
        assert saved['n_err'].shape == (2, 2)
        assert saved['confusion_matrix_source'][()].decode() == str(calibration_path)

    def unexpected_calibration_lookup(fname):
        raise AssertionError('reanalysis should use the saved calibration snapshot')

    monkeypatch.setattr(
        'experiments.qsim.multiparity.multiparity_calibration_from_file',
        unexpected_calibration_lookup,
    )
    loaded = CentralBosonAGPExperiment.from_h5file(experiment.fname)
    result = loaded.analyze()
    for key, expected in experiment.data.items():
        np.testing.assert_array_equal(result[key], expected)
    assert loaded.cfg.expt.photon_sector_number == 2
    assert loaded.cfg.expt.direction == 0
    assert loaded.generate_fock_strings() == OCCUPATIONS


def test_default_program_is_fock_population_without_hardware_initialization(monkeypatch):
    def initialize_base(self, **kwargs):
        self.cfg = AttrDict({})

    monkeypatch.setattr(Experiment, '__init__', initialize_base)
    expt = CentralBosonAGPExperiment(expt_params={
        'photon_sector_number': 2, 'direction': 0, 'floquet_cycles': list(CYCLES),
    })
    assert expt.ProgramClass is FockPopulationProgram
    assert expt.cfg.expt.QickProgramName == 'FockPopulationProgram'
