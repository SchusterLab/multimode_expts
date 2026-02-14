"""
Regression tests for the lmfit-based fitting infrastructure.

Tests run against real experiment HDF5 files when available, and skip
gracefully when data files are not configured.

Configure data file paths in tests/test_data_config.yaml.

Run with: pixi run pytest tests/test_fitting.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

import fitting.fitting as fitter
from fitting.models import PARAM_ORDER

# ====================================================================== #
# Load test data config
# ====================================================================== #

_CONFIG_PATH = Path(__file__).parent / 'test_data_config.yaml'

def _load_config():
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}

_CONFIG = _load_config()


def _get_data_path(section, key):
    """Get a data file path from config, or None if not configured."""
    path = (_CONFIG.get(section) or {}).get(key)
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return path


def _load_h5_data(path):
    """Load data dict from an HDF5 file using SlabFile."""
    from slab.datamanagement import SlabFile
    with SlabFile(str(path), 'r') as f:
        data = f.get_group_data(group='/')
    return data


def _load_h5_config(path):
    """Load config from an HDF5 file attrs."""
    from slab.datamanagement import SlabFile
    from slab import AttrDict
    with SlabFile(str(path), 'r') as f:
        data = f.get_group_data(group='/')
    config_str = data.get('attrs', {}).get('config')
    if config_str:
        return AttrDict(yaml.safe_load(config_str))
    return None


# ====================================================================== #
# Unit tests (no data files needed)
# ====================================================================== #

class TestFitResultBackwardCompat:
    """Test that FitResult behaves like the old (pOpt, pCov) tuple."""

    def _make_result(self):
        x = np.linspace(0, 10, 100)
        y = 0.5 * np.sin(2 * np.pi * 2 * x + 0.5) * np.exp(-x / 3) + 0.1
        return fitter.fitdecaysin(x, y)

    def test_tuple_unpacking(self):
        p, pCov = self._make_result()
        assert isinstance(p, np.ndarray)
        assert isinstance(pCov, np.ndarray)
        assert p.shape == (6,)
        assert pCov.shape == (6, 6)

    def test_len(self):
        result = self._make_result()
        assert len(result) == 2

    def test_integer_indexing(self):
        result = self._make_result()
        p, _ = result
        for i in range(6):
            assert result[i] == p[i]

    def test_named_access(self):
        result = self._make_result()
        p, _ = result
        for i, name in enumerate(PARAM_ORDER['decaysin']):
            assert result[name] == p[i]

    def test_stderr(self):
        result = self._make_result()
        for name in PARAM_ORDER['decaysin']:
            err = result.stderr(name)
            assert isinstance(err, (float, np.floating))

    def test_splat_evaluation(self):
        x = np.linspace(0, 10, 100)
        y = 0.5 * np.sin(2 * np.pi * 2 * x + 0.5) * np.exp(-x / 3) + 0.1
        p, _ = fitter.fitdecaysin(x, y)
        y_eval = fitter.decaysin(x, *p)
        assert y_eval.shape == x.shape
        assert np.all(np.isfinite(y_eval))

    def test_repr(self):
        result = self._make_result()
        s = repr(result)
        assert 'FitResult' in s
        assert 'decay' in s


class TestBareModelFunctions:
    """Test that all bare model functions are accessible and evaluate correctly."""

    def test_expfunc(self):
        x = np.linspace(0, 10, 50)
        y = fitter.expfunc(x, 0.3, 0.7, 0, 5)
        assert y.shape == (50,)
        assert np.isclose(y[0], 1.0, atol=0.01)

    def test_expfunc1(self):
        x = np.linspace(0, 10, 50)
        y = fitter.expfunc1(x, 0.3, 0.7, 5)
        assert y.shape == (50,)

    def test_sinfunc(self):
        x = np.linspace(0, 5, 50)
        y = fitter.sinfunc(x, 0.5, 2, 0, 0.1)
        assert y.shape == (50,)

    def test_decaysin(self):
        x = np.linspace(0, 10, 50)
        y = fitter.decaysin(x, 0.5, 2, 45, 3, 0.1, 0)
        assert y.shape == (50,)

    def test_decaysin1(self):
        x = np.linspace(0, 10, 50)
        y = fitter.decaysin1(x, 0.5, 2, 45, 3, 0.1)
        assert y.shape == (50,)

    def test_decaysin_dualrail(self):
        x = np.linspace(0, 10, 50)
        y = fitter.decaysin_dualrail(x, 0.5, 1, 0, 5, 5, 0.3, 0, 0)
        assert y.shape == (50,)

    def test_twofreq_decaysin(self):
        x = np.linspace(0, 10, 50)
        y = fitter.twofreq_decaysin(x, 0.5, 1, 0, 5, 0.1, 2, 0, 0.3)
        assert y.shape == (50,)

    def test_lorfunc(self):
        x = np.linspace(4000, 5000, 50)
        y = fitter.lorfunc(x, 0.1, 2, 4500, 50)
        assert y.shape == (50,)

    def test_gaussianfunc(self):
        x = np.linspace(4000, 5000, 50)
        y = fitter.gaussianfunc(x, 0.05, 1.5, 4500, 80)
        assert y.shape == (50,)

    def test_hangerfunc(self):
        x = np.array([5000.0])
        y = fitter.hangerfunc(x, 5000, 5000, 1000, 0, 1, 0)
        assert y.shape == (1,)

    def test_hangerS21func(self):
        x = np.array([5000.0])
        y = fitter.hangerS21func(x, 5000, 5000, 1000, 0, 1, 0)
        assert y.shape == (1,)

    def test_hangerS21func_sloped(self):
        x = np.array([5000.0])
        y = fitter.hangerS21func_sloped(x, 5000, 5000, 1000, 0, 1, 0, 0.001)
        assert y.shape == (1,)

    def test_rb_func(self):
        y = fitter.rb_func(10, 0.99, 0.8, 0.1)
        assert np.isfinite(y)

    def test_rb_error(self):
        assert np.isfinite(fitter.rb_error(0.99, 2))

    def test_rb_gate_fidelity(self):
        assert np.isfinite(fitter.rb_gate_fidelity(0.99, 0.985, 2))


class TestFitFunctions:
    """Test all fit* functions with simple synthetic data."""

    def test_fitexp(self):
        x = np.linspace(0, 20, 100)
        y = 0.3 + 0.7 * np.exp(-(x - 0) / 5) + np.random.normal(0, 0.005, 100)
        p, pCov = fitter.fitexp(x, y)
        assert p.shape == (4,)
        assert abs(p[3] - 5) < 1

    def test_fitexp1(self):
        x = np.linspace(0, 20, 100)
        y = 0.3 + 0.7 * np.exp(-x / 5) + np.random.normal(0, 0.005, 100)
        p, pCov = fitter.fitexp1(x, y)
        assert p.shape == (3,)
        assert abs(p[2] - 5) < 1

    def test_fitsin(self):
        x = np.linspace(0, 5, 200)
        y = 0.4 * np.sin(2 * np.pi * 3 * x + 0.5) + 0.5 + np.random.normal(0, 0.01, 200)
        p, pCov = fitter.fitsin(x, y)
        assert p.shape == (4,)
        assert abs(p[1] - 3) < 0.5

    def test_fitdecaysin(self):
        x = np.linspace(0, 10, 200)
        y = 0.5 * np.sin(2 * np.pi * 2 * x + 0.8) * np.exp(-x / 3) + 0.1 + np.random.normal(0, 0.01, 200)
        p, pCov = fitter.fitdecaysin(x, y)
        assert p.shape == (6,)
        assert abs(p[1] - 2) < 0.5

    def test_fitdecaysin_with_fitparams(self):
        x = np.linspace(0, 10, 200)
        y = 0.5 * np.sin(2 * np.pi * 2 * x + 0.8) * np.exp(-x / 3) + 0.1 + np.random.normal(0, 0.01, 200)
        p, pCov = fitter.fitdecaysin(x, y, fitparams=[0.5, 2, None, None, None, None])
        assert p.shape == (6,)

    def test_fitdecaysin1(self):
        x = np.linspace(0, 10, 200)
        y = 0.5 * np.sin(2 * np.pi * 2 * x + 0.8) * np.exp(-x / 3) + 0.1 + np.random.normal(0, 0.01, 200)
        p, pCov = fitter.fitdecaysin1(x, y)
        assert p.shape == (6,)

    def test_fitlor(self):
        x = np.linspace(4000, 5000, 200)
        y = 0.1 + 2.0 / (1 + (x - 4500) ** 2 / 50 ** 2) + np.random.normal(0, 0.01, 200)
        p, pCov = fitter.fitlor(x, y)
        assert p.shape == (4,)
        assert abs(p[2] - 4500) < 10

    def test_fitgaussian(self):
        x = np.linspace(4000, 5000, 200)
        y = 0.05 + 1.5 * np.exp(-((x - 4500) / 80) ** 2) + np.random.normal(0, 0.01, 200)
        p, pCov = fitter.fitgaussian(x, y)
        assert p.shape == (4,)
        assert abs(p[2] - 4500) < 10

    def test_fithanger(self):
        x = np.linspace(4900, 5100, 200)
        y = fitter.hangerS21func_sloped(x, 5000, 5000, 1000, 0, 0.5, 0.2, 0.0001)
        y += np.random.normal(0, 0.001, 200)
        p, pCov = fitter.fithanger(x, y)
        assert p.shape == (7,)
        assert abs(p[0] - 5000) < 10

    def test_fitrb(self):
        depths = np.array([1, 5, 10, 20, 50, 100, 200, 500])
        y = 0.8 * 0.99 ** depths + 0.1 + np.random.normal(0, 0.005, len(depths))
        p, pCov = fitter.fitrb(depths, y)
        assert p.shape == (3,)
        assert abs(p[0] - 0.99) < 0.02


# ====================================================================== #
# Interface 1: fit_display_classes with real data
# ====================================================================== #

class TestFitDisplayClasses:
    """Tests using fit_display_classes directly with real HDF5 data."""

    def _check_fit_result(self, data, key_prefix, model_func, param_order_key):
        """Common checks for fit results stored in data dict."""
        fit_key = f'fit_{key_prefix}'
        err_key = f'fit_err_{key_prefix}'
        assert fit_key in data, f"Missing {fit_key} in data"
        assert err_key in data, f"Missing {err_key} in data"
        p = data[fit_key]
        pCov = data[err_key]
        assert isinstance(p, np.ndarray), f"{fit_key} should be ndarray"
        assert isinstance(pCov, np.ndarray), f"{err_key} should be ndarray"
        n = len(PARAM_ORDER[param_order_key])
        assert p.shape == (n,), f"{fit_key} shape {p.shape} != ({n},)"
        assert pCov.shape == (n, n), f"{err_key} shape {pCov.shape} != ({n}, {n})"
        assert np.any(np.isfinite(p)), f"All params in {fit_key} are non-finite"
        xpts = data['xpts']
        y_eval = model_func(xpts, *p)
        assert y_eval.shape == xpts.shape

    def test_ramsey_fitting(self):
        path = _get_data_path('fit_display_classes', 'RamseyFitting')
        if path is None:
            pytest.skip("No data file for RamseyFitting")
        data = _load_h5_data(path)
        cfg = _load_h5_config(path)
        if cfg is None:
            pytest.skip("No config in HDF5 file for RamseyFitting")
        from fitting.fit_display_classes import RamseyFitting
        rf = RamseyFitting(data=data, config=cfg)
        data = rf.analyze(data=data, fit=True)
        self._check_fit_result(data, 'avgi', fitter.decaysin, 'decaysin')
        self._check_fit_result(data, 'avgq', fitter.decaysin, 'decaysin')

    def test_amplitude_rabi_fitting(self):
        path = _get_data_path('fit_display_classes', 'AmplitudeRabiFitting')
        if path is None:
            pytest.skip("No data file for AmplitudeRabiFitting")
        data = _load_h5_data(path)
        cfg = _load_h5_config(path)
        if cfg is None:
            pytest.skip("No config in HDF5 file for AmplitudeRabiFitting")
        from fitting.fit_display_classes import AmplitudeRabiFitting
        arf = AmplitudeRabiFitting(data=data, config=cfg)
        data = arf.analyze(data=data, fit=True)
        self._check_fit_result(data, 'avgi', fitter.decaysin, 'decaysin')
        self._check_fit_result(data, 'avgq', fitter.decaysin, 'decaysin')

    def test_spectroscopy(self):
        path = _get_data_path('fit_display_classes', 'Spectroscopy')
        if path is None:
            pytest.skip("No data file for Spectroscopy")
        data = _load_h5_data(path)
        cfg = _load_h5_config(path)
        if cfg is None:
            pytest.skip("No config in HDF5 file for Spectroscopy")
        from fitting.fit_display_classes import Spectroscopy
        sp = Spectroscopy(data=data, config=cfg)
        sp.analyze(fit=True)
        self._check_fit_result(sp.data, 'avgi', fitter.lorfunc, 'lor')

    def test_length_rabi_fitting(self):
        path = _get_data_path('fit_display_classes', 'LengthRabiFitting')
        if path is None:
            pytest.skip("No data file for LengthRabiFitting")
        data = _load_h5_data(path)
        cfg = _load_h5_config(path)
        if cfg is None:
            pytest.skip("No config in HDF5 file for LengthRabiFitting")
        from fitting.fit_display_classes import LengthRabiFitting
        lrf = LengthRabiFitting(data=data, config=cfg)
        lrf.analyze()
        self._check_fit_result(lrf.data, 'avgi', fitter.decaysin, 'decaysin')


# ====================================================================== #
# Interface 2: Experiment.from_h5file() -> analyze()
# ====================================================================== #

class TestExperimentAnalyze:
    """Tests using Experiment.from_h5file() to load and analyze."""

    def test_t1_experiment(self):
        path = _get_data_path('experiments', 'T1Experiment')
        if path is None:
            pytest.skip("No data file for T1Experiment")
        from experiments.single_qubit.t1 import T1Experiment
        expt = T1Experiment.from_h5file(str(path))
        expt.analyze()

    def test_t2_ramsey_experiment(self):
        path = _get_data_path('experiments', 'T2RamseyExperiment')
        if path is None:
            pytest.skip("No data file for T2RamseyExperiment")
        from experiments.single_qubit.t2_ramsey import T2RamseyExperiment
        expt = T2RamseyExperiment.from_h5file(str(path))
        expt.analyze()

    def test_amplitude_rabi_experiment(self):
        path = _get_data_path('experiments', 'AmplitudeRabiExperiment')
        if path is None:
            pytest.skip("No data file for AmplitudeRabiExperiment")
        from experiments.single_qubit.amplitude_rabi import AmplitudeRabiExperiment
        expt = AmplitudeRabiExperiment.from_h5file(str(path))
        expt.analyze()

    def test_pulse_probe_spectroscopy_experiment(self):
        path = _get_data_path('experiments', 'PulseProbeSpectroscopyExperiment')
        if path is None:
            pytest.skip("No data file for PulseProbeSpectroscopyExperiment")
        from experiments.single_qubit.pulse_probe_spectroscopy import PulseProbeSpectroscopyExperiment
        expt = PulseProbeSpectroscopyExperiment.from_h5file(str(path))
        expt.analyze()
