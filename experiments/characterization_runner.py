"""
CharacterizationRunner: Simple runner for single-point experiments.

This module provides a clean pattern for running characterization experiments
with minimal boilerplate in notebooks. Just define:
- default_expt_cfg: Default experiment parameters
- preprocessor: Optional function to transform config (e.g., span/center -> start/step)
- postprocessor: Optional function to extract results and update station config

Usage:
    from experiments.station import MultimodeStation
    from experiments.characterization_runner import CharacterizationRunner
    import experiments as meas

    station = MultimodeStation(experiment_name="241215_calibration")

    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.ResonatorSpectroscopyExperiment,
        default_expt_cfg=defaults,
        preprocessor=my_preproc,  # Optional
        postprocessor=my_postproc,  # Optional
    )

    result = runner.run(some_param=123)
"""

from copy import deepcopy
from typing import Optional, Callable, Protocol, TYPE_CHECKING

from slab import AttrDict
from slab.experiment import Experiment

if TYPE_CHECKING:
    from experiments.station import MultimodeStation


class PreProcessor(Protocol):
    """Protocol for preprocessor functions."""

    def __call__(
        self, station: "MultimodeStation", default_expt_cfg: AttrDict, **kwargs
    ) -> AttrDict:
        """
        Transform default config with user kwargs into final expt config.

        Args:
            station: MultimodeStation instance
            default_expt_cfg: Default experiment config template
            **kwargs: User-provided overrides

        Returns:
            Final AttrDict config for the experiment
        """
        ...


class PostProcessor(Protocol):
    """Protocol for postprocessor functions."""

    def __call__(self, station: "MultimodeStation", expt: Experiment) -> None:
        """
        Extract results from experiment and update station config.

        Args:
            station: MultimodeStation instance
            expt: Completed experiment object with results

        Returns:
            None (mutates station.config_thisrun in place)
        """
        ...


def default_preprocessor(station, default_expt_cfg, **kwargs):
    """
    Default preprocessor: simply update default config with user kwargs.

    If your preprocessor just needs to merge kwargs into the default config,
    you don't need to write one - leave preprocessor=None and this is used.

    For custom logic (e.g., converting span/center to start/stop), write your
    own preprocessor following this pattern.
    """
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    return expt_cfg


def default_postprocessor(station, expt):
    """
    Default postprocessor: does nothing.

    Override this to extract fit results and update station.config_thisrun.
    """
    return


class CharacterizationRunner:
    """
    Manages execution of single-point characterization experiments.

    Encapsulates the boilerplate of:
    - Creating experiment instance
    - Setting up configuration
    - Running the experiment
    - Extracting results to update config

    This keeps notebooks clean with only ephemeral settings visible.
    """

    def __init__(
        self,
        station: "MultimodeStation",
        ExptClass: type,
        default_expt_cfg: AttrDict,
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
        ExptProgram: Optional[type] = None,
    ):
        """
        Initialize the runner.

        Args:
            station: MultimodeStation instance for hardware access
            ExptClass: Experiment class to instantiate (e.g., meas.SomeExperiment)
            default_expt_cfg: AttrDict template for expt.cfg.expt
            preprocessor: Function to generate expt.cfg.expt from defaults + kwargs
            postprocessor: Function to extract results and update station.config_thisrun
            ExptProgram: for QsimBaseExperiment, this is the program class to use
        """
        self.station = station
        self.ExptClass = ExptClass
        self.default_expt_cfg = default_expt_cfg
        self.preprocessor = preprocessor or default_preprocessor
        self.postprocessor = postprocessor or default_postprocessor
        self.program = ExptProgram

    def run(
        self, postprocess: bool = True, go_kwargs: Optional[dict] = None, **kwargs
    ) -> Experiment:
        """
        Run the experiment.

        Args:
            postprocess: Whether to run postprocessor after experiment
            go_kwargs: Dict passed to expt.go() (analyze, display, progress, save)
            **kwargs: Passed to preprocessor to modify config

        Returns:
            Completed Experiment object
        """
        go_kwargs = go_kwargs or {}

        # Create experiment instance
        if self.program is not None:
            expt = self.ExptClass(
                soccfg=self.station.soc,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
                program=self.program, 
            )
        else:
            expt = self.ExptClass(
                soccfg=self.station.soc,
                path=self.station.data_path,
                prefix=self.ExptClass.__name__,
                config_file=self.station.hardware_config_file,
            )

        # Setup config
        expt.cfg = AttrDict(deepcopy(self.station.config_thisrun))
        expt.cfg.expt = self.preprocessor(self.station, self.default_expt_cfg, **kwargs)

        # Handle relax_delay if present
        if hasattr(expt.cfg.expt, "relax_delay"):
            expt.cfg.device.readout.relax_delay = [expt.cfg.expt.relax_delay]

        # Run with sensible defaults
        go_defaults = {"analyze": True, "display": True, "progress": True, "save": True}
        go_defaults.update(go_kwargs)
        expt.go(**go_defaults)

        # Run postprocessor
        if postprocess:
            self.postprocessor(self.station, expt)

        return expt
