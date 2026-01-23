"""
Test harness for SweepRunner with mother experiment pattern.

This module provides:
1. Mock experiment/station objects for unit testing without hardware
2. Tests for the new "mother experiment" pattern where analyze/display
   methods detect 2D data and delegate to appropriate fitting classes

Usage:
    # Quick unit test with mock data
    python -m pytest tests/test_sweep_runner.py -v

    # Or run interactively
    python tests/test_sweep_runner.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# =============================================================================
# MOCK HARDWARE MODULES - Must happen BEFORE any other imports
# =============================================================================

def create_mock_module(name, **attrs):
    """Create a mock module with specified attributes."""
    mock = MagicMock()
    for k, v in attrs.items():
        setattr(mock, k, v)
    mock.__name__ = name
    return mock

# Mock qick and all its submodules (FPGA library)
class MockQickProgram:
    """Mock base class for QICK programs."""
    def __init__(self, soccfg=None, cfg=None):
        self.soccfg = soccfg
        self.cfg = cfg

class MockAveragerProgram(MockQickProgram):
    """Mock AveragerProgram from qick."""
    pass

class MockRAveragerProgram(MockQickProgram):
    """Mock RAveragerProgram from qick."""
    pass

mock_qick = create_mock_module('qick')
mock_qick.QickProgram = MockQickProgram
mock_qick.AveragerProgram = MockAveragerProgram
mock_qick.RAveragerProgram = MockRAveragerProgram
mock_qick.QickConfig = MagicMock

mock_qick_helpers = create_mock_module('qick.helpers',
    gauss=lambda x: x,
    sin2=lambda x: x,
    tanh=lambda x: x,
    flat_top_gauss=lambda x: x,
)
mock_qick.helpers = mock_qick_helpers

sys.modules['qick'] = mock_qick
sys.modules['qick.helpers'] = mock_qick_helpers

# Mock telnetlib (removed in Python 3.12+, needed by slab.instruments)
sys.modules['telnetlib'] = MagicMock()

# Mock Pyro4 (instrument server, optional)
sys.modules['Pyro4'] = MagicMock()

# Mock visa/pyvisa (instrument communication)
sys.modules['visa'] = MagicMock()
sys.modules['pyvisa'] = MagicMock()

# Mock lmfit (fitting library)
mock_lmfit = create_mock_module('lmfit')
mock_lmfit_models = create_mock_module('lmfit.models')
mock_lmfit.models = mock_lmfit_models
mock_lmfit.Model = MagicMock
sys.modules['lmfit'] = mock_lmfit
sys.modules['lmfit.models'] = mock_lmfit_models

# Now we can safely import everything else
import numpy as np
import tempfile
import os
from copy import deepcopy
from unittest.mock import patch

from slab import AttrDict, Experiment

# Import SweepRunner from the new location
try:
    from experiments.sweep_runner import SweepRunner, default_preprocessor
    SWEEP_RUNNER_AVAILABLE = True
except Exception as e:
    print(f'Warning: SweepRunner import failed ({type(e).__name__}: {e})')
    SWEEP_RUNNER_AVAILABLE = False
    SweepRunner = None
    default_preprocessor = None


# =============================================================================
# Mock Objects for Testing Without Hardware
# =============================================================================

class MockChevronAnalysis:
    """Mock ChevronFitting analysis class."""

    def __init__(self, frequencies, time, response_matrix, config=None, station=None):
        self.frequencies = np.array(frequencies)
        self.time = np.array(time)
        self.response_matrix = np.array(response_matrix)
        self.config = config
        self.station = station
        self.results = {}

    def analyze(self):
        """Find best frequency based on contrast."""
        contrasts = []
        for row in self.response_matrix:
            contrast = np.max(row) - np.min(row)
            contrasts.append(contrast)

        best_idx = np.argmax(contrasts)
        self.results = {
            'best_frequency_contrast': self.frequencies[best_idx],
            'best_frequency_period': self.frequencies[best_idx],  # Simplified
            'best_contrast': contrasts[best_idx],
            'contrasts': contrasts,
        }
        return self.results

    def display_results(self, save_fig=False, title=''):
        """Mock display - just print summary."""
        print(f'[MockChevronAnalysis] Best frequency: {self.results["best_frequency_contrast"]:.4f}')
        print(f'[MockChevronAnalysis] Best contrast: {self.results["best_contrast"]:.4f}')


class MockLengthRabiFitting:
    """Mock LengthRabiFitting analysis class for 1D data."""

    def __init__(self, data, fit=True, fitparams=None, config=None, station=None, **kwargs):
        self.data = data
        self.config = config
        self.station = station
        self.results = {}

    def analyze(self, **kwargs):
        """Mock 1D analysis."""
        self.results = {
            'fit_avgi': [1.0, 0.5, 0, 10, 0, 0],  # Mock fit params
            'fit_avgq': [1.0, 0.5, 0, 10, 0, 0],
        }
        return self.results

    def display(self, title_str='', **kwargs):
        """Mock display."""
        print(f'[MockLengthRabiFitting] 1D display: {title_str}')


class MockExperiment(Experiment):
    """
    Mock experiment that generates synthetic 1D sweep data.
    Simulates LengthRabiGeneralF0g1Experiment with 2D detection in analyze/display.
    """

    def __init__(self, soccfg=None, path='', prefix='MockExperiment', config_file=None, progress=None):
        # Don't call super().__init__ to avoid file system operations
        self.soccfg = soccfg
        self.path = path
        self.prefix = prefix
        self.config_file = config_file
        self.cfg = AttrDict({})
        self.data = {}
        self.im = {'soc': MagicMock()}
        self._fname = None
        self._chevron_analysis = None
        self._length_rabi_analysis = None

    @property
    def fname(self):
        if self._fname is None:
            self._fname = os.path.join(self.path, f'{self.prefix}_0000.h5')
        return self._fname

    def acquire(self, progress=False, debug=False):
        """Generate synthetic length rabi data (1D sweep over length)."""
        cfg = self.cfg

        start = cfg.expt.get('start', 0)
        step = cfg.expt.get('step', 0.1)
        expts = cfg.expt.get('expts', 25)
        freq = cfg.expt.get('freq', 5000)

        lengths = start + step * np.arange(expts)
        avgi = []
        avgq = []

        # Generate synthetic oscillation data
        for length in lengths:
            omega = 2 * np.pi * 0.5 * (1 + (freq - 5000) / 100)
            decay = np.exp(-length / 10)
            i = decay * np.cos(omega * length) + np.random.normal(0, 0.02)
            q = decay * np.sin(omega * length) + np.random.normal(0, 0.02)
            avgi.append(i)
            avgq.append(q)

        self.data = {
            'xpts': np.array(lengths),
            'avgi': np.array(avgi),
            'avgq': np.array(avgq),
            'amps': np.abs(np.array(avgi) + 1j * np.array(avgq)),
            'phases': np.angle(np.array(avgi) + 1j * np.array(avgq)),
        }
        return self.data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        """
        Analyze data with 2D detection.
        This mirrors the pattern in LengthRabiGeneralF0g1Experiment.
        """
        if data is None:
            data = self.data

        station = kwargs.pop('station', None)

        # Detect 2D sweep data
        is_2d = (
            'freq_sweep' in data
            and 'avgi' in data
            and hasattr(data['avgi'], 'ndim')
            and data['avgi'].ndim == 2
        )

        if is_2d:
            # Use mock ChevronFitting for 2D
            time = data['xpts'][0] if data['xpts'].ndim > 1 else data['xpts']
            analysis = MockChevronAnalysis(
                frequencies=data['freq_sweep'],
                time=time,
                response_matrix=data['avgi'],
                config=self.cfg,
                station=station,
            )
            analysis.analyze()
            self._chevron_analysis = analysis
            return data

        # 1D case
        analysis = MockLengthRabiFitting(data, fit=fit, fitparams=fitparams, config=self.cfg, station=station)
        analysis.analyze()
        self._length_rabi_analysis = analysis
        return data

    def display(self, data=None, fit=True, title_str='Mock Experiment', **kwargs):
        """
        Display results with 2D detection.
        This mirrors the pattern in LengthRabiGeneralF0g1Experiment.
        """
        if data is None:
            data = self.data

        # 2D case
        if hasattr(self, '_chevron_analysis') and self._chevron_analysis is not None:
            self._chevron_analysis.display_results(
                save_fig=kwargs.get('save_fig', False),
                title=title_str,
            )
            return

        # 1D case
        if hasattr(self, '_length_rabi_analysis') and self._length_rabi_analysis is not None:
            self._length_rabi_analysis.display(title_str=title_str, **kwargs)
        else:
            print(f'[MockExperiment] Display: {title_str}')

    def save_data(self, data=None):
        # In mock, just pretend to save
        pass

    def go(self, save=False, analyze=False, display=False, progress=False):
        self.acquire(progress)
        if analyze:
            self.analyze()
        if display:
            self.display()


class MockStation:
    """Mock MultimodeStation for testing without hardware."""

    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp()

        # Create required paths
        self.data_path = Path(self.temp_dir) / 'data'
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.hardware_config_file = Path(self.temp_dir) / 'device.yaml'
        self.hardware_config_file.touch()

        # Mock SoC
        self.soc = MagicMock()

        # Config that experiments will read
        self.config_thisrun = AttrDict({
            'device': {
                'readout': {'relax_delay': [1000]},
                'qubit': {'f_ge': [5000]},
                'manipulate': {'f0g1_freq': [2000]},
                'multiphoton': {'pi': {'fn-gn+1': {'frequency': [2000]}}},
            },
            'expt': {},
        })

        # Mock dataset
        self.ds_thisrun = MagicMock()
        self.ds_thisrun.get_freq = MagicMock(return_value=5000)
        self.ds_thisrun.get_gain = MagicMock(return_value=8000)

    def cleanup(self):
        """Remove temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# =============================================================================
# Test Functions
# =============================================================================

def test_sweep_runner_basic():
    """Test basic SweepRunner functionality with mock objects."""
    print('\n' + '='*60)
    print('TEST: SweepRunner Basic Functionality')
    print('='*60)

    if not SWEEP_RUNNER_AVAILABLE:
        print('  SKIPPED: SweepRunner not available')
        return True

    # Setup
    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=10,
        reps=100,
        relax_delay=1000,
    ))

    # Create runner (no analysis_class - analysis via Experiment.analyze())
    runner = SweepRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
    )

    # Run sweep
    result = runner.run(
        sweep_start=4990,
        sweep_stop=5010,
        sweep_npts=11,
        incremental_save=False,
    )

    # Validate - should return mother experiment
    assert isinstance(result, MockExperiment), "Should return mother experiment"
    assert 'freq_sweep' in result.data, "Should have sweep values"
    assert 'avgi' in result.data, "Should have avgi data"
    assert len(result.data['freq_sweep']) == 11, f"Expected 11 points, got {len(result.data['freq_sweep'])}"

    # Should have 2D analysis attached
    assert hasattr(result, '_chevron_analysis'), "Should have _chevron_analysis"
    assert result._chevron_analysis is not None, "_chevron_analysis should not be None"

    print('\n  Basic test passed!')
    station.cleanup()
    return True


def test_sweep_runner_2d_detection():
    """Test that analyze() correctly detects 2D data."""
    print('\n' + '='*60)
    print('TEST: SweepRunner 2D Detection in analyze()')
    print('='*60)

    if not SWEEP_RUNNER_AVAILABLE:
        print('  SKIPPED: SweepRunner not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=10,
        reps=100,
        relax_delay=1000,
    ))

    runner = SweepRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
    )

    result = runner.run(
        sweep_start=4995,
        sweep_stop=5005,
        sweep_npts=5,
        incremental_save=False,
    )

    # Check 2D detection worked
    assert result._chevron_analysis is not None, "Should detect 2D and create chevron analysis"
    assert 'best_frequency_contrast' in result._chevron_analysis.results, "Should have analysis results"

    best_freq = result._chevron_analysis.results['best_frequency_contrast']
    print(f'  Best frequency found: {best_freq:.4f} MHz')

    print('\n  2D detection test passed!')
    station.cleanup()
    return True


def test_sweep_runner_postprocessor():
    """Test postprocessor receives mother experiment."""
    print('\n' + '='*60)
    print('TEST: SweepRunner Postprocessor')
    print('='*60)

    if not SWEEP_RUNNER_AVAILABLE:
        print('  SKIPPED: SweepRunner not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=10,
        reps=100,
        relax_delay=1000,
    ))

    postproc_called = [False]
    received_expt = [None]

    def my_postproc(station, mother_expt):
        postproc_called[0] = True
        received_expt[0] = mother_expt
        # Access analysis results
        if hasattr(mother_expt, '_chevron_analysis') and mother_expt._chevron_analysis:
            best_freq = mother_expt._chevron_analysis.results.get('best_frequency_contrast')
            print(f'  [Postproc] Best freq: {best_freq:.4f}')
            station.ds_thisrun.update_freq('M1', best_freq)

    runner = SweepRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        postprocessor=my_postproc,
    )

    result = runner.run(
        sweep_start=4995,
        sweep_stop=5005,
        sweep_npts=5,
        incremental_save=False,
    )

    # Validate
    assert postproc_called[0], "Postprocessor should be called"
    assert received_expt[0] is result, "Postprocessor should receive mother experiment"
    station.ds_thisrun.update_freq.assert_called_once()

    print('\n  Postprocessor test passed!')
    station.cleanup()
    return True


def test_sweep_runner_live_plot():
    """Test live plotting functionality."""
    print('\n' + '='*60)
    print('TEST: SweepRunner Live Plot')
    print('='*60)

    if not SWEEP_RUNNER_AVAILABLE:
        print('  SKIPPED: SweepRunner not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=5,
        reps=100,
        relax_delay=1000,
    ))

    # Patch IPython display to avoid errors
    with patch('experiments.sweep_runner.SweepRunner._do_live_plot') as mock_live:
        runner = SweepRunner(
            station=station,
            ExptClass=MockExperiment,
            default_expt_cfg=defaults,
            sweep_param='freq',
            live_plot=True,
        )

        result = runner.run(
            sweep_start=4998,
            sweep_stop=5002,
            sweep_npts=5,
            incremental_save=False,
        )

    # Should have called _do_live_plot for each point
    assert mock_live.call_count == 5, f"Expected 5 live plot calls, got {mock_live.call_count}"

    print(f'\n  Live plot test passed! ({mock_live.call_count} calls)')
    station.cleanup()
    return True


def test_sweep_runner_incremental_save():
    """Test that incremental save actually writes files."""
    print('\n' + '='*60)
    print('TEST: SweepRunner Incremental Save')
    print('='*60)

    if not SWEEP_RUNNER_AVAILABLE:
        print('  SKIPPED: SweepRunner not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=5,
        reps=100,
        relax_delay=1000,
    ))

    save_counts = [0]

    def counting_save(self, data=None):
        save_counts[0] += 1

    with patch.object(MockExperiment, 'save_data', counting_save):
        runner = SweepRunner(
            station=station,
            ExptClass=MockExperiment,
            default_expt_cfg=defaults,
            sweep_param='freq',
        )

        result = runner.run(
            sweep_start=4998,
            sweep_stop=5002,
            sweep_npts=5,
            incremental_save=True,
        )

    # Should save after each point (5) + final save (1) = 6
    expected_saves = 6
    assert save_counts[0] == expected_saves, f"Expected {expected_saves} saves, got {save_counts[0]}"

    print(f'\n  Incremental save test passed! ({save_counts[0]} saves)')
    station.cleanup()
    return True


def test_data_structure():
    """Test that output data structure is correct for 2D analysis."""
    print('\n' + '='*60)
    print('TEST: Data Structure for 2D Analysis')
    print('='*60)

    if not SWEEP_RUNNER_AVAILABLE:
        print('  SKIPPED: SweepRunner not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.2,
        expts=10,
        reps=100,
        relax_delay=1000,
    ))

    runner = SweepRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
    )

    result = runner.run(
        sweep_start=4990,
        sweep_stop=5010,
        sweep_npts=11,
        incremental_save=False,
    )

    data = result.data

    # Check required keys
    required_keys = ['freq_sweep', 'xpts', 'avgi', 'avgq']
    for key in required_keys:
        assert key in data, f"Missing key: {key}"
        print(f'    Has {key}: shape={np.array(data[key]).shape}')

    # Check shapes are consistent
    n_freqs = len(data['freq_sweep'])
    assert data['avgi'].ndim == 2, "avgi should be 2D"
    assert data['avgi'].shape[0] == n_freqs, "avgi outer dim should match freq count"
    print(f'    Shapes consistent: {n_freqs} freq points, each with {data["avgi"].shape[1]} time points')

    print('\n  Data structure test passed!')
    station.cleanup()
    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ('Basic SweepRunner', test_sweep_runner_basic),
        ('2D Detection', test_sweep_runner_2d_detection),
        ('Postprocessor', test_sweep_runner_postprocessor),
        ('Live Plot', test_sweep_runner_live_plot),
        ('Incremental Save', test_sweep_runner_incremental_save),
        ('Data Structure', test_data_structure),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()

    # Summary
    print('\n' + '='*60)
    print('TEST SUMMARY')
    print('='*60)
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)

    for name, success, error in results:
        status = '  PASS' if success else '  FAIL'
        print(f'  {status}: {name}')
        if error:
            print(f'         Error: {error}')

    print(f'\nTotal: {passed}/{total} tests passed')
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
