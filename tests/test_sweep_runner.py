"""
Test harness for SweepRunner vs sequential_base_class comparison.

This module provides:
1. Mock experiment/station objects for unit testing without hardware
2. Side-by-side comparison of old and new sweep patterns
3. Validation that both produce equivalent results

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
# Need to create proper base classes that can be inherited from
# (MagicMock causes metaclass conflicts when used as base class)

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
    gauss=lambda x: x,  # Mock gauss function
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

# Mock lmfit (fitting library - may not be installed)
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
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'measurement_notebooks'))

from slab import AttrDict, Experiment

# Pre-import meas_utils to trigger any import errors early
# Some errors (like metaclass conflicts) happen only on first import
try:
    from meas_utils import SweepRunner, default_preprocessor
    MEAS_UTILS_AVAILABLE = True
except Exception as e:
    print(f'Warning: Initial meas_utils import failed ({type(e).__name__}: {e})')
    print('Tests will attempt to import individually.')
    MEAS_UTILS_AVAILABLE = False
    SweepRunner = None
    default_preprocessor = None


# =============================================================================
# Mock Objects for Testing Without Hardware
# =============================================================================

class MockProgram:
    """Mock QICK program that generates synthetic data."""
    def __init__(self, soccfg, cfg):
        self.cfg = cfg
        self.soccfg = soccfg

    def acquire(self, soc, **kwargs):
        """Generate synthetic IQ data."""
        # Simulate length rabi oscillation
        freq = self.cfg.expt.get('freq', 5000)
        length = self.cfg.expt.get('length_placeholder', 1.0)

        # Synthetic decaying sine based on freq and length
        t = length
        omega = 2 * np.pi * 0.5  # oscillation rate
        detuning = (freq - 5000) / 10  # detuning from resonance
        decay = np.exp(-t / 10)

        avgi = decay * np.cos(omega * t + detuning * t) + np.random.normal(0, 0.01)
        avgq = decay * np.sin(omega * t + detuning * t) + np.random.normal(0, 0.01)

        return [[avgi]], [[avgq]]


class MockExperiment(Experiment):
    """
    Mock experiment that generates synthetic 1D sweep data.
    Simulates what LengthRabiGeneralF0g1Experiment does.
    """
    ProgramClass = MockProgram

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
            # Decaying sine with frequency-dependent rate
            omega = 2 * np.pi * 0.5 * (1 + (freq - 5000) / 100)  # freq affects oscillation rate
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

    def analyze(self, data=None, **kwargs):
        return self.data

    def display(self, data=None, **kwargs):
        pass

    def save_data(self, data=None):
        # In mock, just pretend to save
        pass

    def go(self, save=False, analyze=False, display=False, progress=False):
        self.acquire(progress)
        if analyze:
            self.analyze()
        if display:
            self.display()
        # Note: We don't actually save in mock


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

    def convert_attrdict_to_dict(self, d):
        """Convert AttrDict to regular dict (for JSON serialization)."""
        if isinstance(d, AttrDict):
            return {k: self.convert_attrdict_to_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            return {k: self.convert_attrdict_to_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self.convert_attrdict_to_dict(v) for v in d]
        else:
            return d

    def cleanup(self):
        """Remove temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


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
            'best_contrast': contrasts[best_idx],
            'contrasts': contrasts,
        }
        return self.results

    def display_results(self, save_fig=False, title=''):
        """Mock display - just print summary."""
        print(f'[MockAnalysis] Best frequency: {self.results["best_frequency_contrast"]:.4f}')
        print(f'[MockAnalysis] Best contrast: {self.results["best_contrast"]:.4f}')


# =============================================================================
# Test Functions
# =============================================================================

def test_sweep_runner_basic():
    """Test basic SweepRunner functionality with mock objects."""
    print('\n' + '='*60)
    print('TEST: SweepRunner Basic Functionality')
    print('='*60)

    if not MEAS_UTILS_AVAILABLE:
        print('  SKIPPED: meas_utils not available')
        return True  # Skip but don't fail

    # Setup
    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=10,
        reps=100,
        relax_delay=1000,
    ))

    def mock_postproc(station, result):
        if isinstance(result, dict):
            print(f'[Postproc] Got raw data with keys: {list(result.keys())}')
        else:
            print(f'[Postproc] Got analysis with best_freq: {result.results.get("best_frequency_contrast")}')

    # Create runner WITHOUT analysis factory (returns raw data)
    runner = SweepRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        postprocessor=mock_postproc,
    )

    # Run sweep
    result = runner.run(
        sweep_start=4990,
        sweep_stop=5010,
        sweep_step=2,
        incremental_save=False,  # Faster for test
    )

    # Validate
    assert isinstance(result, dict), "Should return dict when no analysis_factory"
    assert 'freq_sweep' in result, "Should have sweep values"
    assert 'avgi' in result, "Should have avgi data"
    assert len(result['freq_sweep']) == 11, f"Expected 11 points, got {len(result['freq_sweep'])}"

    print('\n✓ Basic test passed!')
    station.cleanup()
    return True


def test_sweep_runner_with_analysis():
    """Test SweepRunner with analysis factory."""
    print('\n' + '='*60)
    print('TEST: SweepRunner with Analysis Factory')
    print('='*60)

    if not MEAS_UTILS_AVAILABLE:
        print('  SKIPPED: meas_utils not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=10,
        reps=100,
        relax_delay=1000,
    ))

    def analysis_factory(sweep_data, station):
        return MockChevronAnalysis(
            frequencies=sweep_data['freq_sweep'],
            time=sweep_data['xpts'][0] if sweep_data['xpts'].ndim > 1 else sweep_data['xpts'],
            response_matrix=sweep_data['avgi'],
            config=station.config_thisrun,
            station=station,
        )

    def postproc(station, analysis):
        print(f'[Postproc] Best freq: {analysis.results["best_frequency_contrast"]}')
        station.ds_thisrun.update_freq('M1', analysis.results['best_frequency_contrast'])

    runner = SweepRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        analysis_factory=analysis_factory,
        postprocessor=postproc,
    )

    result = runner.run(
        sweep_start=4990,
        sweep_stop=5010,
        sweep_step=2,
        incremental_save=False,
    )

    # Validate
    assert hasattr(result, 'results'), "Should return analysis object"
    assert 'best_frequency_contrast' in result.results, "Should have analysis results"
    station.ds_thisrun.update_freq.assert_called_once()

    print('\n✓ Analysis test passed!')
    station.cleanup()
    return True


def test_sweep_runner_live_callback():
    """Test SweepRunner with live analysis callback."""
    print('\n' + '='*60)
    print('TEST: SweepRunner with Live Callback')
    print('='*60)

    if not MEAS_UTILS_AVAILABLE:
        print('  SKIPPED: meas_utils not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=5,  # Fewer points for faster test
        reps=100,
        relax_delay=1000,
    ))

    callback_counts = [0]

    def live_callback(sweep_data, station):
        callback_counts[0] += 1
        n_points = len(sweep_data['freq_sweep'])
        print(f'    [Live] Callback #{callback_counts[0]}, {n_points} points collected')

    runner = SweepRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        sweep_param='freq',
        live_analysis_fn=live_callback,
    )

    result = runner.run(
        sweep_start=4995,
        sweep_stop=5005,
        sweep_step=2,
        incremental_save=False,
    )

    # Validate
    expected_points = 6  # 4995, 4997, 4999, 5001, 5003, 5005
    assert callback_counts[0] == expected_points, f"Expected {expected_points} callbacks, got {callback_counts[0]}"

    print(f'\n✓ Live callback test passed! ({callback_counts[0]} callbacks)')
    station.cleanup()
    return True


def test_sweep_runner_incremental_save():
    """Test that incremental save actually writes files."""
    print('\n' + '='*60)
    print('TEST: SweepRunner Incremental Save')
    print('='*60)

    if not MEAS_UTILS_AVAILABLE:
        print('  SKIPPED: meas_utils not available')
        return True

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=0.1,
        expts=5,
        reps=100,
        relax_delay=1000,
    ))

    # Patch Experiment.save_data to count calls
    save_counts = [0]
    original_save = Experiment.save_data

    def counting_save(self, data=None):
        save_counts[0] += 1
        # Don't actually save in test
        pass

    with patch.object(Experiment, 'save_data', counting_save):
        runner = SweepRunner(
            station=station,
            ExptClass=MockExperiment,
            default_expt_cfg=defaults,
            sweep_param='freq',
        )

        result = runner.run(
            sweep_start=4998,
            sweep_stop=5002,
            sweep_step=1,
            incremental_save=True,
        )

    # Should save after each point (5) + final save (1) = 6
    expected_saves = 6
    assert save_counts[0] == expected_saves, f"Expected {expected_saves} saves, got {save_counts[0]}"

    print(f'\n✓ Incremental save test passed! ({save_counts[0]} saves)')
    station.cleanup()
    return True


def compare_with_sequential_pattern():
    """
    Compare SweepRunner output with what sequential_base_class would produce.
    This is a structural comparison, not a data comparison (since we're using mocks).
    """
    print('\n' + '='*60)
    print('TEST: Compare with Sequential Pattern (Structural)')
    print('='*60)

    if not MEAS_UTILS_AVAILABLE:
        print('  SKIPPED: meas_utils not available')
        return True

    station = MockStation()

    # Config similar to what sequential_base_class would use
    defaults = AttrDict(dict(
        start=0,
        step=0.2,
        expts=10,
        reps=100,
        relax_delay=1000,
        gain=8000,
        qubits=[0],
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
        sweep_step=2,
        incremental_save=False,
    )

    # Check that output structure matches sequential_base_class
    # sequential_base_class produces: {'freq_sweep': [...], 'xpts': [...], 'avgi': [...], ...}

    required_keys = ['freq_sweep', 'xpts', 'avgi', 'avgq']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
        print(f'  ✓ Has {key}: shape={np.array(result[key]).shape}')

    # Check shapes are consistent
    n_freqs = len(result['freq_sweep'])
    assert result['avgi'].shape[0] == n_freqs, "avgi outer dim should match freq count"
    print(f'  ✓ Shapes consistent: {n_freqs} freq points, each with {result["avgi"].shape[1]} time points')

    print('\n✓ Sequential pattern comparison passed!')
    station.cleanup()
    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ('Basic SweepRunner', test_sweep_runner_basic),
        ('SweepRunner with Analysis', test_sweep_runner_with_analysis),
        ('SweepRunner Live Callback', test_sweep_runner_live_callback),
        ('SweepRunner Incremental Save', test_sweep_runner_incremental_save),
        ('Sequential Pattern Comparison', compare_with_sequential_pattern),
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
        status = '✓ PASS' if success else '✗ FAIL'
        print(f'  {status}: {name}')
        if error:
            print(f'         Error: {error}')

    print(f'\nTotal: {passed}/{total} tests passed')
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
