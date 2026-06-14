"""
Test harness for CharacterizationRunner.

This module provides:
1. Mock experiment/station objects for unit testing without hardware
2. Tests for the CharacterizationRunner pattern including preprocessors,
   postprocessors, and smart use_queue defaults

Note on Mock Hardware:
    This test file uses lightweight, self-contained mocks (MockExperiment, MockStation)
    designed specifically for unit testing without any file system dependencies.

    For integration testing or manual development, use the centralized mock hardware
    in experiments/mock_hardware.py via MultimodeStation(mock=True). The centralized
    mocks are more realistic and load actual config files from disk.

    Test mocks (this file):
    - MockStation: Minimal, creates temp directories, no real config files
    - MockExperiment: Generates synthetic data, no QICK dependencies

    Centralized mocks (experiments/mock_hardware.py):
    - MockQickConfig, MockQickSoc: Realistic QICK hardware simulation
    - MockInstrumentManager: Simulates instrument access
    - MockYokogawa: Mock voltage source
    - Used by MultimodeStation when mock=True

Usage:
    # Quick unit test with mock data
    python -m pytest tests/test_characterization_runner.py -v

    # Or run interactively
    python tests/test_characterization_runner.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# =============================================================================
# Imports
# =============================================================================
# NOTE: This file used to inject fake `qick`/`lmfit`/`Pyro4`/`visa`/`telnetlib`
# modules into sys.modules here, before learning that qick can be imported for
# real (mock mode only stubs QickSoc, not QickConfig -- see
# experiments/mock_hardware.py). That global injection permanently polluted
# sys.modules for the rest of the pytest process and broke any later test that
# imported the real qick. It is gone now: the runner and its real deps import
# cleanly, and these tests exercise pure runner logic via the lightweight
# Mock{Experiment,Station} below (no qick contact at all).
import numpy as np
import tempfile
import os
from copy import deepcopy
from unittest.mock import patch

from slab import AttrDict, Experiment

# Import CharacterizationRunner from the experiments module. This is a HARD
# import on purpose: it used to be wrapped in try/except with a RUNNER_AVAILABLE
# flag, which (combined with the fake-qick sys.modules injection above) silently
# turned every test into a no-op skip when the import broke. If this import ever
# fails again, the whole module should error loudly at collection time.
from experiments.characterization_runner import (
    CharacterizationRunner,
    default_preprocessor,
    default_postprocessor,
)


# =============================================================================
# Mock Objects for Testing Without Hardware
# =============================================================================

class MockAmplitudeRabiFitting:
    """Mock fitting class for amplitude rabi experiments."""

    def __init__(self, data, fit=True, fitparams=None, config=None, **kwargs):
        self.data = data
        self.config = config
        self.results = {}

    def analyze(self, **kwargs):
        """Mock analysis - find pi pulse gain."""
        xpts = self.data.get('xpts', np.linspace(0, 9000, 151))
        avgi = self.data.get('avgi', np.zeros_like(xpts))

        # Find the first minimum (pi pulse)
        pi_idx = len(xpts) // 4  # Approximate pi at 1/4 of sweep
        hpi_idx = pi_idx // 2  # Half-pi at half that

        self.results = {
            'pi_gain_avgi': xpts[pi_idx],
            'hpi_gain_avgi': xpts[hpi_idx],
            'fit_avgi': [1.0, 0.5, 0, pi_idx * 60, 0, 0],  # Mock fit params
        }
        return self.results

    def display(self, title_str='', **kwargs):
        """Mock display."""
        print(f'[MockAmplitudeRabiFitting] Display: {title_str}')


class MockExperiment(Experiment):
    """
    Mock experiment that generates synthetic amplitude rabi data.
    Simulates AmplitudeRabiExperiment behavior.
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
        self._analysis = None

    @property
    def fname(self):
        if self._fname is None:
            self._fname = os.path.join(self.path, f'{self.prefix}_0000.h5')
        return self._fname

    def acquire(self, progress=False, debug=False):
        """Generate synthetic amplitude rabi data."""
        cfg = self.cfg

        start = cfg.expt.get('start', 0)
        step = cfg.expt.get('step', 60)
        expts = cfg.expt.get('expts', 151)

        # Generate x points (gain values)
        xpts = start + step * np.arange(expts)

        # Generate synthetic Rabi oscillation
        # Simulates oscillation in gain space
        pi_gain = 4500  # Typical pi pulse gain
        omega = np.pi / pi_gain  # Period = 2 * pi_gain

        avgi = np.cos(omega * xpts) + np.random.normal(0, 0.05, len(xpts))
        avgq = np.sin(omega * xpts) + np.random.normal(0, 0.05, len(xpts))

        self.data = {
            'xpts': np.array(xpts),
            'avgi': np.array(avgi),
            'avgq': np.array(avgq),
            'amps': np.abs(np.array(avgi) + 1j * np.array(avgq)),
            'phases': np.angle(np.array(avgi) + 1j * np.array(avgq)),
            'pi_gain_avgi': pi_gain,
            'hpi_gain_avgi': pi_gain // 2,
        }
        return self.data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        """Analyze data with mock fitting."""
        if data is None:
            data = self.data

        analysis = MockAmplitudeRabiFitting(data, fit=fit, fitparams=fitparams, config=self.cfg)
        analysis.analyze()
        self._analysis = analysis
        return data

    def display(self, data=None, fit=True, title_str='Mock Experiment', **kwargs):
        """Display results."""
        if self._analysis is not None:
            self._analysis.display(title_str=title_str, **kwargs)
        else:
            print(f'[MockExperiment] Display: {title_str}')

    def save_data(self, data=None):
        """Mock save - does nothing."""
        pass

    def go(self, save=False, analyze=False, display=False, progress=False):
        """Run the experiment."""
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

        # Mock QickConfig (renamed from .soc -> .soccfg; runner reads .soccfg)
        self.soccfg = MagicMock()
        # Instrument manager the runner/experiment reach through
        self.im = {'soc': MagicMock()}

        # Mock mode flag (test mock station is always mock)
        self._is_mock = True

        # User for job submission
        self.user = 'test_user'

        # Config that experiments will read
        self.hardware_cfg = AttrDict({
            'device': {
                'readout': {'relax_delay': [1000]},
                'qubit': {
                    'f_ge': [5000],
                    'pulses': {
                        'pi_ge': {'gain': [4500], 'sigma': [0.05]},
                        'hpi_ge': {'gain': [2250], 'sigma': [0.05]},
                        'pi_ef': {'gain': [4000], 'sigma': [0.05]},
                        'hpi_ef': {'gain': [2000], 'sigma': [0.05]},
                    },
                },
                'storage': {
                    '_ds_storage': None,
                    '_ds_floquet': None,
                },
            },
            'expt': {},
        })

        # Mock datasets
        self.ds_storage = MagicMock()
        self.ds_floquet = None

    @property
    def is_mock(self) -> bool:
        """Test mock station is always in mock mode."""
        return self._is_mock

    def cleanup(self):
        """Remove temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


# =============================================================================
# Test Functions
# =============================================================================

def test_runner_basic():
    """Test basic CharacterizationRunner functionality."""
    print('\n' + '='*60)
    print('TEST: CharacterizationRunner Basic Functionality')
    print('='*60)

    # Setup
    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=151,
        reps=200,
        relax_delay=1000,
    ))

    # Create runner
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
    )

    # Run experiment locally
    expt = runner.run_local(postprocess=False)

    # Validate
    assert isinstance(expt, MockExperiment), "Should return experiment instance"
    assert 'xpts' in expt.data, "Should have xpts data"
    assert 'avgi' in expt.data, "Should have avgi data"
    assert len(expt.data['xpts']) == 151, f"Expected 151 points, got {len(expt.data['xpts'])}"

    print('\n  Basic test passed!')
    station.cleanup()


def test_runner_preprocessor():
    """Test that preprocessor transforms config correctly."""
    print('\n' + '='*60)
    print('TEST: CharacterizationRunner Preprocessor')
    print('='*60)

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=151,
        reps=200,
        sigma_test=None,
        if_ef=False,
    ))

    preproc_called = [False]
    received_kwargs = [{}]

    def my_preproc(station, default_expt_cfg, **kwargs):
        preproc_called[0] = True
        received_kwargs[0] = kwargs

        expt_cfg = deepcopy(default_expt_cfg)
        expt_cfg.update(kwargs)

        # Custom logic: if sigma_test is None, use from config
        if expt_cfg.sigma_test is None:
            expt_cfg.sigma_test = station.hardware_cfg.device.qubit.pulses.pi_ge.sigma[0]

        return expt_cfg

    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        preprocessor=my_preproc,
    )

    # Run with custom kwargs
    expt = runner.run_local(
        step=100,
        expts=50,
        postprocess=False,
    )

    # Validate preprocessor was called
    assert preproc_called[0], "Preprocessor should be called"
    assert received_kwargs[0].get('step') == 100, "Should receive step kwarg"
    assert received_kwargs[0].get('expts') == 50, "Should receive expts kwarg"

    # Validate config was transformed
    assert expt.cfg.expt.step == 100, "Config should have updated step"
    assert expt.cfg.expt.expts == 50, "Config should have updated expts"
    assert expt.cfg.expt.sigma_test == 0.05, "sigma_test should be filled from config"

    print('\n  Preprocessor test passed!')
    station.cleanup()


def test_runner_postprocessor():
    """Test that postprocessor updates station config."""
    print('\n' + '='*60)
    print('TEST: CharacterizationRunner Postprocessor')
    print('='*60)

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=151,
        reps=200,
    ))

    postproc_called = [False]
    received_expt = [None]

    def my_postproc(station, expt):
        postproc_called[0] = True
        received_expt[0] = expt

        # Update station config with fitted values (like amprabi_postproc)
        station.hardware_cfg.device.qubit.pulses.pi_ge.gain = [expt.data['pi_gain_avgi']]
        station.hardware_cfg.device.qubit.pulses.hpi_ge.gain = [expt.data['hpi_gain_avgi']]
        print(f'  [Postproc] Updated pi_ge gain to {expt.data["pi_gain_avgi"]}')

    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        postprocessor=my_postproc,
    )

    # Check initial values
    initial_pi_gain = station.hardware_cfg.device.qubit.pulses.pi_ge.gain[0]
    print(f'  Initial pi_ge gain: {initial_pi_gain}')

    # Run with postprocessing
    expt = runner.run_local(postprocess=True)

    # Validate
    assert postproc_called[0], "Postprocessor should be called"
    assert received_expt[0] is expt, "Postprocessor should receive experiment"

    # Check that station config was updated
    new_pi_gain = station.hardware_cfg.device.qubit.pulses.pi_ge.gain[0]
    print(f'  New pi_ge gain: {new_pi_gain}')
    assert new_pi_gain == expt.data['pi_gain_avgi'], "Station config should be updated"

    print('\n  Postprocessor test passed!')
    station.cleanup()


def test_runner_use_queue_default():
    """Test that use_queue defaults to True (independent of station.is_mock)."""
    print('\n' + '='*60)
    print('TEST: CharacterizationRunner use_queue Default')
    print('='*60)

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=10,
    ))

    # Test 1: Default should be use_queue=True (regardless of mock mode)
    runner1 = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
    )
    assert runner1.use_queue == True, "Default should be use_queue=True"
    print(f'  Default (no use_queue specified): use_queue={runner1.use_queue}')

    # Test 2: Explicit use_queue=True should work
    runner2 = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        use_queue=True,
    )
    assert runner2.use_queue == True, "Explicit use_queue=True should be honored"
    print(f'  Explicit use_queue=True: use_queue={runner2.use_queue}')

    # Test 3: Explicit use_queue=False should work
    runner3 = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
        use_queue=False,
    )
    assert runner3.use_queue == False, "Explicit use_queue=False should be honored"
    print(f'  Explicit use_queue=False: use_queue={runner3.use_queue}')

    print('\n  use_queue default test passed!')
    station.cleanup()


def test_runner_execute_dispatches_correctly():
    """Test that execute() dispatches to run() or run_local() based on use_queue."""
    print('\n' + '='*60)
    print('TEST: CharacterizationRunner execute() Dispatch')
    print('='*60)

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=10,
    ))

    # Create runner with use_queue=False (mock station default)
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
    )

    # execute() should call run_local() for mock station
    expt = runner.execute(postprocess=False)
    assert isinstance(expt, MockExperiment), "execute() should return experiment"
    print(f'  execute() with use_queue={runner.use_queue} completed successfully')

    # Test override: execute(use_queue=False) explicitly
    expt2 = runner.execute(use_queue=False, postprocess=False)
    assert isinstance(expt2, MockExperiment), "execute(use_queue=False) should work"
    print(f'  execute(use_queue=False) override completed successfully')

    print('\n  execute() dispatch test passed!')
    station.cleanup()


def test_runner_go_kwargs():
    """Test that go_kwargs are passed to expt.go()."""
    print('\n' + '='*60)
    print('TEST: CharacterizationRunner go_kwargs')
    print('='*60)

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=10,
    ))

    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
    )

    # Run with custom go_kwargs
    expt = runner.run_local(
        postprocess=False,
        go_kwargs={'analyze': False, 'display': False, 'save': False},
    )

    # The experiment should still have data (from acquire)
    assert 'xpts' in expt.data, "Should have acquired data"
    # But _analysis should be None since analyze=False
    assert expt._analysis is None, "Should not have analyzed (analyze=False)"

    print('\n  go_kwargs test passed!')
    station.cleanup()


def test_default_preprocessor():
    """Test the default preprocessor behavior."""
    print('\n' + '='*60)
    print('TEST: Default Preprocessor')
    print('='*60)

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=151,
        reps=200,
    ))

    # Use default preprocessor (by not specifying one)
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
    )

    # Run with kwargs that should be merged
    expt = runner.run_local(
        step=100,
        custom_param='test_value',
        postprocess=False,
    )

    # Check that kwargs were merged into config
    assert expt.cfg.expt.step == 100, "step should be updated from kwargs"
    assert expt.cfg.expt.expts == 151, "expts should be from defaults"
    assert expt.cfg.expt.custom_param == 'test_value', "custom_param should be added"

    print('\n  Default preprocessor test passed!')
    station.cleanup()


def test_default_postprocessor():
    """Test the default postprocessor behavior (does nothing)."""
    print('\n' + '='*60)
    print('TEST: Default Postprocessor')
    print('='*60)

    station = MockStation()

    defaults = AttrDict(dict(
        start=0,
        step=60,
        expts=10,
    ))

    # Store original config value
    original_gain = station.hardware_cfg.device.qubit.pulses.pi_ge.gain[0]

    # Use default postprocessor (by not specifying one)
    runner = CharacterizationRunner(
        station=station,
        ExptClass=MockExperiment,
        default_expt_cfg=defaults,
    )

    # Run with postprocessing (but default postprocessor does nothing)
    expt = runner.run_local(postprocess=True)

    # Check that station config was NOT changed
    current_gain = station.hardware_cfg.device.qubit.pulses.pi_ge.gain[0]
    assert current_gain == original_gain, "Default postprocessor should not modify config"

    print('\n  Default postprocessor test passed!')
    station.cleanup()
