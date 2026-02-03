"""
Centralized mock hardware implementations for testing without real instruments.

This module provides mock versions of hardware components for:
- Integration testing on development machines
- Local rapid iteration without hardware access
- CI/CD pipeline testing

Mock classes:
- MockQickConfig: Simulates QICK SoC configuration
- MockQickSoc: Mock QICK proxy with fake Rabi oscillation data
- MockInstrumentManager: Simulates instrument access
- MockYokogawa: Mock voltage source

Usage:
    # These are used internally by MultimodeStation when mock=True
    # Direct usage is not typically needed, but available:
    from experiments.mock_hardware import MockQickConfig, MockInstrumentManager

    config = MockQickConfig()
    im = MockInstrumentManager()
"""

import numpy as np
from typing import Optional, Dict


class MockQickConfig:
    """
    Mock QICK SoC configuration for testing.

    Simulates the QickConfig interface without real hardware.
    Provides conversion methods that return realistic values.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        """
        Initialize mock QICK config.

        Args:
            cfg: Optional config dict. If None, creates default mock config.
        """
        self._cfg = cfg or self._create_default_cfg()
        print("[MOCK] QickConfig initialized")

    def _create_default_cfg(self) -> dict:
        """Create a realistic mock QICK config structure."""
        return {
            "gens": [
                {
                    "maxv": 32767,
                    "maxv_scale": 1.0,
                    "samps_per_clk": 16,
                    "f_fabric": 430.08,
                    "type": "full",
                }
                for _ in range(7)
            ],
            "readouts": [
                {
                    "freq": 100.0,
                    "f_fabric": 430.08,
                }
            ],
            "tprocs": [
                {
                    "f_time": 384.0,
                }
            ],
        }

    def __getitem__(self, key):
        return self._cfg[key]

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def freq2reg(self, f, gen_ch=0, ro_ch=None):
        """Mock frequency to register conversion."""
        print(f"[MOCK] freq2reg: f={f} MHz, gen_ch={gen_ch}")
        return int(f * 1000)  # Simplified conversion

    def reg2freq(self, reg, gen_ch=0, ro_ch=None):
        """Mock register to frequency conversion."""
        return reg / 1000.0

    def us2cycles(self, us, gen_ch=0, ro_ch=None):
        """Mock microseconds to cycles conversion."""
        cycles = int(us * 430.08)  # Approximate conversion
        print(f"[MOCK] us2cycles: {us} us -> {cycles} cycles")
        return cycles

    def cycles2us(self, cycles, gen_ch=0, ro_ch=None):
        """Mock cycles to microseconds conversion."""
        return cycles / 430.08

    def deg2reg(self, deg, gen_ch=0):
        """Mock degrees to register conversion."""
        return int(deg * 65536 / 360)

    def reg2deg(self, reg, gen_ch=0):
        """Mock register to degrees conversion."""
        return reg * 360 / 65536

    def __repr__(self):
        return "<MockQickConfig: Simulated QICK hardware>"


class MockQickSoc:
    """Mock QICK SoC proxy for InstrumentManager."""

    def __init__(self):
        self._cfg = MockQickConfig()._cfg
        print("[MOCK] QickSoc proxy initialized")

    def get_cfg(self) -> dict:
        """Return mock configuration."""
        return self._cfg

    def acquire(self, *args, **kwargs):
        """
        Mock data acquisition.

        Returns simulated Rabi oscillation data:
        - 100 points on x-axis over 1 us (0 to 1.0 us)
        - Oscillation frequency: 5 MHz
        - Decay time constant: 10 us (T2)

        Returns:
            Tuple of (xpts, [[avgi]], [[avgq]]) matching real QICK format
        """
        n_expts = kwargs.get("expts", 100)
        print(f"[MOCK] acquire: generating {n_expts} data points")

        # Generate x points: 100 points over 1 us
        xpts = np.linspace(0, 1.0, n_expts)  # in microseconds

        # Simulation parameters
        osc_freq_mhz = 5.0  # 5 MHz oscillation
        decay_time_us = 10.0  # T2 = 10 us decay constant

        # Generate simulated Rabi oscillation with decay
        # I = A * cos(2*pi*f*t) * exp(-t/T2) + noise
        # Q = A * sin(2*pi*f*t) * exp(-t/T2) + noise
        omega = 2 * np.pi * osc_freq_mhz  # angular frequency (rad/us since t is in us)
        decay = np.exp(-xpts / decay_time_us)
        noise_amplitude = 5.0

        avgi = np.cos(omega * xpts) * decay * 100 + np.random.randn(n_expts) * noise_amplitude
        avgq = np.sin(omega * xpts) * decay * 100 + np.random.randn(n_expts) * noise_amplitude

        print(f"[MOCK] acquire: returning simulated I/Q data (f={osc_freq_mhz} MHz, T2={decay_time_us} us)")
        return xpts, [[avgi]], [[avgq]]


class MockInstrumentManager(dict):
    """
    Mock InstrumentManager that prints operations instead of executing.

    Inherits from dict to match the real InstrumentManager interface.
    """

    def __init__(self):
        super().__init__()
        self["Qick101"] = MockQickSoc()  # Default QICK alias
        print("[MOCK] InstrumentManager initialized")

    def keys(self):
        return ["Qick101"]

    def __repr__(self):
        return "<MockInstrumentManager: Simulated hardware access>"


class MockYokogawa:
    """Mock Yokogawa voltage source."""

    def __init__(self, name: str, address: str):
        self.name = name
        self.address = address
        self._voltage = 0.0
        self._output_enabled = False
        print(f"[MOCK] Yokogawa {name} initialized at {address}")

    def set_voltage(self, v: float):
        print(f"[MOCK] {self.name}.set_voltage({v})")
        self._voltage = v

    def get_voltage(self) -> float:
        return self._voltage

    def output_on(self):
        print(f"[MOCK] {self.name}.output_on()")
        self._output_enabled = True

    def output_off(self):
        print(f"[MOCK] {self.name}.output_off()")
        self._output_enabled = False

    def ramp_current(self, current: float, sweeprate: float=0.001, channel: int=1):
        print(f"[MOCK] {self.name}.ramp_current({current})")
        self._current = current
