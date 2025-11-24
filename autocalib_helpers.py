"""
Helper classes and functions to reduce boilerplate in autocalibration notebooks.

This module provides utilities to eliminate repetitive patterns in experiment
setup, execution, analysis, and configuration updates.
"""

from copy import deepcopy
from typing import Optional, Dict, Any, Callable, List, Tuple
import numpy as np
from slab import AttrDict
import experiments as meas


class ExperimentRunner:
    """
    Helper class to eliminate boilerplate in experiment setup and execution.
    
    This class handles the common pattern of:
    1. Creating an experiment object
    2. Setting cfg from config_thisrun
    3. Setting cfg.expt parameters
    4. Setting relax_delay
    5. Running the experiment
    
    Usage:
        runner = ExperimentRunner(soc, expt_path, config_path, config_thisrun)
        expt = runner.run(
            experiment_class=meas.single_qubit.resonator_spectroscopy.ResonatorSpectroscopyExperiment,
            prefix='ResonatorSpectroscopyExperiment',
            expt_params={'start': 749, 'step': 0.01, 'expts': 250, 'reps': 500},
            relax_delay=50,
            analyze=False,
            display=False,
            progress=True,
            save=True
        )
    """
    
    def __init__(self, soc, expt_path: str, config_path: str, config_thisrun: AttrDict):
        self.soc = soc
        self.expt_path = expt_path
        self.config_path = config_path
        self.config_thisrun = config_thisrun
    
    def run(self,
            experiment_class,
            prefix: str,
            expt_params: Dict[str, Any],
            relax_delay: Optional[float] = None,
            analyze: bool = False,
            display: bool = False,
            progress: bool = True,
            save: bool = True,
            **kwargs) -> Any:
        """
        Run an experiment with standardized setup.
        
        Args:
            experiment_class: The experiment class to instantiate
            prefix: Experiment prefix for file naming
            expt_params: Dictionary of experiment parameters (goes into cfg.expt)
            relax_delay: Relax delay in microseconds (if None, uses config default)
            analyze: Whether to analyze results
            display: Whether to display results
            progress: Whether to show progress bar
            save: Whether to save data
            **kwargs: Additional arguments passed to experiment constructor
        
        Returns:
            The experiment object
        """
        # Create experiment object
        expt = experiment_class(
            soccfg=self.soc,
            path=self.expt_path,
            prefix=prefix,
            config_file=self.config_path,
            **kwargs
        )
        
        # Set config from config_thisrun
        expt.cfg = AttrDict(deepcopy(self.config_thisrun))
        
        # Set experiment parameters
        expt.cfg.expt = dict(expt_params)
        
        # Set relax delay if provided
        if relax_delay is not None:
            expt.cfg.device.readout.relax_delay = [relax_delay]
        
        # Run experiment
        expt.go(analyze=analyze, display=display, progress=progress, save=save)
        
        return expt


class ConfigUpdater:
    """
    Helper class to standardize configuration update patterns.
    
    This class provides a registry of update functions that can be called
    to update config_thisrun based on experiment results.
    
    Usage:
        updater = ConfigUpdater(config_thisrun)
        updater.update_readout_frequency(rspec.data['fit'][0])
        updater.update_qubit_frequency_ge(qspec.data['fit_avgi'][2])
    """
    
    def __init__(self, config_thisrun: AttrDict):
        self.config = config_thisrun
    
    def update_readout_frequency(self, frequency: float):
        """Update readout frequency in config."""
        self.config.device.readout.frequency = [frequency]
        print(f'Updated readout frequency to {frequency} MHz!')
    
    def update_qubit_frequency_ge(self, frequency: float):
        """Update qubit ge transition frequency."""
        self.config.device.qubit.f_ge = [frequency]
        print(f'Updated qubit ge frequency to {frequency} MHz!')
    
    def update_qubit_frequency_ef(self, frequency: float):
        """Update qubit ef transition frequency."""
        self.config.device.qubit.f_ef = [frequency]
        print(f'Updated qubit ef frequency to {frequency} MHz!')
    
    def update_qubit_pi_gain_ge(self, pi_gain: float, hpi_gain: float):
        """Update qubit ge pi and half-pi gains."""
        self.config.device.qubit.pulses.pi_ge.gain = [pi_gain]
        self.config.device.qubit.pulses.hpi_ge.gain = [hpi_gain]
        print(f'Updated qubit ge pi gain to {pi_gain} and hpi gain to {hpi_gain}!')
    
    def update_qubit_pi_gain_ef(self, pi_gain: float, hpi_gain: float):
        """Update qubit ef pi and half-pi gains."""
        self.config.device.qubit.pulses.pi_ef.gain = [pi_gain]
        self.config.device.qubit.pulses.hpi_ef.gain = [hpi_gain]
        print(f'Updated qubit ef pi gain to {pi_gain} and hpi gain to {hpi_gain}!')
    
    def update_qubit_t1(self, t1: float):
        """Update qubit T1."""
        self.config.device.qubit.T1 = [t1]
        print(f'Updated qubit T1 to {t1} us!')
    
    def update_qubit_t1_ef(self, t1_ef: float):
        """Update qubit ef T1."""
        self.config.device.qubit.T1_ef = [t1_ef]
        print(f'Updated qubit ef T1 to {t1_ef} us!')
    
    def update_single_shot(self, hist_analysis):
        """Update readout parameters from single shot histogram analysis."""
        hist_analysis.analyze(plot=True)
        fids = hist_analysis.results['fids']
        confusion_matrix = hist_analysis.results['confusion_matrix']
        thresholds_new = hist_analysis.results['thresholds']
        angle = hist_analysis.results['angle']
        
        self.config.device.readout.phase = [self.config.device.readout.phase[0] + angle]
        self.config.device.readout.threshold = thresholds_new
        self.config.device.readout.threshold_list = [thresholds_new]
        self.config.device.readout.Ie = [np.median(hist_analysis.data['Ie_rot'])]
        self.config.device.readout.Ig = [np.median(hist_analysis.data['Ig_rot'])]
        
        if hist_analysis.cfg.expt.active_reset:
            self.config.device.readout.confusion_matrix_with_active_reset = confusion_matrix
        else:
            self.config.device.readout.confusion_matrix_without_reset = confusion_matrix
        print('Updated readout parameters!')


class ExperimentExecutor:
    """
    Helper class to handle conditional execution, analysis, and updates.
    
    This class combines ExperimentRunner and ConfigUpdater to provide
    a complete workflow: run experiment -> analyze -> display -> update config.
    
    Usage:
        executor = ExperimentExecutor(soc, expt_path, config_path, config_thisrun)
        
        # Simple execution
        expt = executor.execute_if(
            condition=expts_to_run['res_spec'],
            experiment_class=meas.single_qubit.resonator_spectroscopy.ResonatorSpectroscopyExperiment,
            prefix='ResonatorSpectroscopyExperiment',
            expt_params={'start': 749, 'step': 0.01, 'expts': 250, 'reps': 500},
            relax_delay=50
        )
        
        # With analysis and update
        if expt:
            executor.analyze_and_update(
                expt,
                analysis_class=SomeAnalysisClass,
                update_func=lambda analysis: updater.update_readout_frequency(analysis.results['freq'])
            )
    """
    
    def __init__(self, soc, expt_path: str, config_path: str, config_thisrun: AttrDict, 
                 monitor=None):
        """
        Args:
            soc: QickConfig object
            expt_path: Path to save experiment data
            config_path: Path to config file
            config_thisrun: Current run configuration
            monitor: Optional ExperimentMonitor instance for persistent logging
        """
        self.runner = ExperimentRunner(soc, expt_path, config_path, config_thisrun)
        self.updater = ConfigUpdater(config_thisrun)
        self.config_thisrun = config_thisrun
        self.monitor = monitor
    
    def execute_if(self,
                   condition: bool,
                   experiment_class,
                   prefix: str,
                   expt_params: Dict[str, Any],
                   relax_delay: Optional[float] = None,
                   analyze: bool = False,
                   display: bool = False,
                   progress: bool = True,
                   save: bool = True,
                   experiment_name: Optional[str] = None,
                   **kwargs) -> Optional[Any]:
        """
        Execute experiment only if condition is True.
        
        Args:
            condition: Whether to run the experiment
            experiment_class: Experiment class to instantiate
            prefix: Experiment prefix for file naming
            expt_params: Experiment parameters
            relax_delay: Relax delay in microseconds
            analyze: Whether to analyze results
            display: Whether to display results
            progress: Whether to show progress bar
            save: Whether to save data
            experiment_name: Optional name for monitoring (defaults to prefix)
            **kwargs: Additional arguments for experiment
        
        Returns:
            Experiment object if condition is True, None otherwise
        """
        if not condition:
            if self.monitor:
                self.monitor.log(f"Skipping {experiment_name or prefix} (condition=False)")
            return None
        
        exp_name = experiment_name or prefix
        if self.monitor:
            self.monitor.log(f"Starting {exp_name}...")
        
        try:
            expt = self.runner.run(
                experiment_class=experiment_class,
                prefix=prefix,
                expt_params=expt_params,
                relax_delay=relax_delay,
                analyze=analyze,
                display=display,
                progress=progress,
                save=save,
                **kwargs
            )
            
            if self.monitor:
                self.monitor.log(f"{exp_name} completed successfully")
            
            return expt
        except Exception as e:
            if self.monitor:
                self.monitor.log(f"Error in {exp_name}: {str(e)}", level='error')
            raise
    
    def analyze_and_display(self, expt, analysis_class=None, title_str: Optional[str] = None, **analysis_kwargs):
        """
        Analyze and display experiment results.
        
        Args:
            expt: Experiment object
            analysis_class: Optional analysis class to use (if None, uses expt's built-in analysis)
            title_str: Optional title for display
            **analysis_kwargs: Additional arguments for analysis
        """
        if analysis_class:
            analysis = analysis_class(expt.data, config=expt.cfg, **analysis_kwargs)
            analysis.analyze()
            analysis.display(title_str=title_str)
            return analysis
        else:
            if hasattr(expt, 'analyze'):
                expt.analyze(**analysis_kwargs)
            if hasattr(expt, 'display'):
                expt.display(title_str=title_str)
            return expt
    
    def analyze_and_update(self,
                           expt,
                           update_func: Callable,
                           analysis_class=None,
                           title_str: Optional[str] = None,
                           **analysis_kwargs):
        """
        Analyze, display, and update config based on results.
        
        Args:
            expt: Experiment object
            update_func: Function that takes analysis/expt and updates config
            analysis_class: Optional analysis class
            title_str: Optional title for display
            **analysis_kwargs: Additional arguments for analysis
        """
        analysis = self.analyze_and_display(expt, analysis_class, title_str, **analysis_kwargs)
        update_func(analysis if analysis_class else expt)
        return analysis if analysis_class else expt


class ParameterExtractor:
    """
    Helper class to extract common parameters from dataset and config.
    
    This reduces repetition in getting frequencies, gains, pulse lengths, etc.
    """
    
    def __init__(self, ds_thisrun, config_thisrun: AttrDict):
        self.ds = ds_thisrun
        self.config = config_thisrun
    
    def get_storage_params(self, man_mode_no: int, stor_mode_no: int) -> Dict[str, Any]:
        """
        Get all parameters for a storage mode.
        
        Returns dict with: freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse
        """
        stor_name = f'M{man_mode_no}-S{stor_mode_no}'
        freq = self.ds.get_freq(stor_name)
        gain = self.ds.get_gain(stor_name)
        pi_len = self.ds.get_pi(stor_name)
        h_pi_len = self.ds.get_h_pi(stor_name)
        
        # Determine channel
        flux_low_ch = self.config.hw.soc.dacs.flux_low.ch
        flux_high_ch = self.config.hw.soc.dacs.flux_high.ch
        ch = flux_low_ch if freq < 1000 else flux_high_ch
        
        # Get prepulse and postpulse
        from MM_dual_rail_base import MM_dual_rail_base
        mm_base = MM_dual_rail_base(self.config, self.config.hw.soc)
        prep_man_pi = mm_base.prep_man_photon(man_mode_no)
        prepulse = mm_base.get_prepulse_creator(prep_man_pi).pulse.tolist()
        postpulse = mm_base.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist()
        
        return {
            'freq': freq,
            'gain': gain,
            'pi_len': pi_len,
            'h_pi_len': h_pi_len,
            'ch': ch,
            'prepulse': prepulse,
            'postpulse': postpulse
        }
    
    def get_manipulate_params(self, man_mode_no: int) -> Dict[str, Any]:
        """Get parameters for a manipulation mode."""
        freq = self.ds.get_freq(f'M{man_mode_no}')
        gain = self.ds.get_gain(f'M{man_mode_no}')
        pi_len = self.ds.get_pi(f'M{man_mode_no}')
        h_pi_len = self.ds.get_h_pi(f'M{man_mode_no}')
        
        return {
            'freq': freq,
            'gain': gain,
            'pi_len': pi_len,
            'h_pi_len': h_pi_len
        }


def create_experiment_workflow(soc, expt_path: str, config_path: str, config_thisrun: AttrDict, 
                              ds_thisrun, monitor=None):
    """
    Create all helper objects for a complete experiment workflow.
    
    Args:
        soc: QickConfig object
        expt_path: Path to save experiment data
        config_path: Path to config file
        config_thisrun: Current run configuration
        ds_thisrun: Dataset object
        monitor: Optional ExperimentMonitor instance for persistent logging
    
    Returns:
        Tuple of (executor, updater, extractor)
    """
    executor = ExperimentExecutor(soc, expt_path, config_path, config_thisrun, monitor=monitor)
    updater = ConfigUpdater(config_thisrun)
    extractor = ParameterExtractor(ds_thisrun, config_thisrun)
    
    return executor, updater, extractor

