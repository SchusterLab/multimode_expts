"""
Backwards-compatible re-exports from experiments module.

This module is DEPRECATED. Please import directly from:
    - experiments.station (MultimodeStation)
    - experiments.characterization_runner (CharacterizationRunner)
    - experiments.sweep_runner (SweepRunner)

Example:
    # Old (deprecated):
    from meas_utils import MultimodeStation, SweepRunner

    # New (preferred):
    from experiments import MultimodeStation, SweepRunner
    # or
    from experiments.station import MultimodeStation
    from experiments.sweep_runner import SweepRunner
"""

import warnings

# Show deprecation warning on import
warnings.warn(
    "meas_utils is deprecated. Import from 'experiments' instead:\n"
    "  from experiments import MultimodeStation, SweepRunner, CharacterizationRunner",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backwards compatibility
from experiments.station import MultimodeStation
from experiments.characterization_runner import (
    CharacterizationRunner,
    PreProcessor,
    PostProcessor,
    default_preprocessor,
    default_postprocessor,
)
from experiments.sweep_runner import SweepRunner, register_analysis_class

# For any code that imported these directly
__all__ = [
    'MultimodeStation',
    'CharacterizationRunner',
    'SweepRunner',
    'PreProcessor',
    'PostProcessor',
    'default_preprocessor',
    'default_postprocessor',
    'register_analysis_class',
]
