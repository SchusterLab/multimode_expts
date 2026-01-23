"""
Compare old sequential_base_class vs new SweepRunner for M1 chevron sweep.

This script runs both methods with identical parameters and compares:
1. HDF5 data structure and contents
2. Analysis results from ChevronFitting
3. Performance and file sizes

Usage:
    pixi run python test_sweep_comparison.py [--method old|new|both]
"""

import sys
import argparse
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'measurement_notebooks'))

from slab import AttrDict
from slab.datamanagement import SlabFile
import experiments as meas
from fitting.fit_display_classes import ChevronFitting
from meas_utils import MultimodeStation, SweepRunner
from experiments.sequential_experiment_classes import man_f0g1_class


def load_h5_data(filepath):
    """Load data from HDF5 file for comparison."""
    with SlabFile(filepath) as f:
        data = {}
        for key in list(f):
            data[key] = np.array(f[key])
    return data


def compare_h5_files(old_file, new_file):
    """Compare two HDF5 files and report differences."""
    print("\n" + "="*70)
    print("HDF5 FILE COMPARISON")
    print("="*70)

    old_data = load_h5_data(old_file)
    new_data = load_h5_data(new_file)

    print(f"\nOld file: {old_file.name}")
    print(f"  Keys: {sorted(old_data.keys())}")
    print(f"  Size: {old_file.stat().st_size / 1024:.1f} KB")

    print(f"\nNew file: {new_file.name}")
    print(f"  Keys: {sorted(new_data.keys())}")
    print(f"  Size: {new_file.stat().st_size / 1024:.1f} KB")

    # Compare keys
    old_keys = set(old_data.keys())
    new_keys = set(new_data.keys())

    if old_keys == new_keys:
        print(f"\n✓ Keys match: {len(old_keys)} keys")
    else:
        print(f"\n✗ Keys differ!")
        if old_keys - new_keys:
            print(f"  Only in old: {old_keys - new_keys}")
        if new_keys - old_keys:
            print(f"  Only in new: {new_keys - old_keys}")

    # Compare array shapes and values for common keys
    common_keys = old_keys & new_keys
    print(f"\nComparing {len(common_keys)} common arrays:")

    all_match = True
    for key in sorted(common_keys):
        old_arr = old_data[key]
        new_arr = new_data[key]

        shape_match = old_arr.shape == new_arr.shape
        if shape_match and old_arr.size > 0:
            # Check if arrays are close (allow small numerical differences)
            values_match = np.allclose(old_arr, new_arr, rtol=1e-5, atol=1e-8)
            max_diff = np.max(np.abs(old_arr - new_arr)) if old_arr.size > 0 else 0
        else:
            values_match = False
            max_diff = np.nan

        status = "✓" if (shape_match and values_match) else "✗"
        print(f"  {status} {key:20s} shape={old_arr.shape} -> {new_arr.shape}, max_diff={max_diff:.2e}")

        if not (shape_match and values_match):
            all_match = False

    if all_match:
        print("\n✓✓✓ All arrays match! Data is identical.")
    else:
        print("\n⚠ Some arrays differ (see above)")

    return all_match


def run_old_method(station, freq_start, freq_stop, freq_step):
    """Run chevron sweep using old man_f0g1_class."""
    print("\n" + "="*70)
    print("RUNNING OLD METHOD (man_f0g1_class)")
    print("="*70)

    sweep_experiment_name = 'length_rabi_f0g1_sweep'

    # Create old-style class
    class_for_exp = man_f0g1_class(
        soccfg=station.soc,
        path=station.data_path,
        prefix=sweep_experiment_name,
        config_file=station.hardware_config_file,
        exp_param_file=station.config_dir / "experiment_config.yml",
        config_thisrun=station.config_thisrun
    )

    # Set sweep parameters (matching notebook exactly)
    class_for_exp.loaded[sweep_experiment_name] = {
        'freq_start': freq_start,
        'freq_stop': freq_stop,
        'freq_step': freq_step,
        'start': 2,
        'step': 0.1,
        'qubits': [0],
        'expts': 25,
        'reps': 100,
        'rounds': 1,
        'gain': 8000,
        'ramp_sigma': 0.005,
        'use_arb_waveform': False,
        'pi_ge_before': True,
        'pi_ef_before': True,
        'pi_ge_after': False,
        'normalize': False,
        'active_reset': False,
        'check_man_reset': [False, 0],
        'check_man_reset_pi': [],
        'prepulse': False,
        'pre_sweep_pulse': [],
        'err_amp_reps': 0,
    }

    print(f"\nSweep parameters:")
    print(f"  Frequency: {freq_start} to {freq_stop} MHz (step {freq_step})")
    print(f"  Length: 2 to {2 + 0.1*25:.1f} us (25 points)")
    print(f"  Reps: 100")

    # Run sweep
    print(f"\nRunning old method sweep...")
    start_time = datetime.now()
    chevron = class_for_exp.run_sweep(sweep_experiment_name)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"✓ Old method complete in {elapsed:.1f}s")

    # Get the saved file
    old_file = class_for_exp.expt_sweep.fname
    print(f"  Saved to: {old_file}")

    return old_file, chevron, elapsed


def run_new_method(station, freq_start, freq_stop, freq_step):
    """Run chevron sweep using new SweepRunner."""
    print("\n" + "="*70)
    print("RUNNING NEW METHOD (SweepRunner)")
    print("="*70)

    # Load experiment defaults from experiment_config.yml (like old method)
    exp_config_file = station.config_dir / "experiment_config.yml"
    with open(exp_config_file, 'r') as f:
        exp_config = yaml.safe_load(f)

    # Get defaults for LengthRabiGeneralF0g1Experiment
    length_rabi_config = exp_config.get('LengthRabiGeneralF0g1Experiment', {})

    # Create default config matching old method exactly
    chevron_defaults = AttrDict(dict(
        start=2,
        step=0.1,
        expts=25,
        reps=100,
        rounds=1,
        qubits=[0],
        gain=8000,
        ramp_sigma=0.005,
        use_arb_waveform=False,
        pi_ge_before=True,
        pi_ef_before=True,
        pi_ge_after=False,
        normalize=False,
        active_reset=False,
        check_man_reset=[False, 0],
        check_man_reset_pi=[],
        prepulse=False,
        pre_sweep_pulse=[],
        err_amp_reps=0,
        swap_lossy=False,
    ))

    print(f"\nSweep parameters:")
    print(f"  Frequency: {freq_start} to {freq_stop} MHz (step {freq_step})")
    print(f"  Length: 2 to {2 + 0.1*25:.1f} us (25 points)")
    print(f"  Reps: 100")

    # Preprocessor
    def chevron_preproc(station, default_expt_cfg, **kwargs):
        expt_cfg = deepcopy(default_expt_cfg)
        expt_cfg.update(kwargs)
        return expt_cfg

    # Analysis factory
    def chevron_analysis_factory(sweep_data, station):
        time = sweep_data['xpts'][0] if sweep_data['xpts'].ndim > 1 else sweep_data['xpts']
        return ChevronFitting(
            frequencies=sweep_data['freq_sweep'],
            time=time,
            response_matrix=sweep_data['avgi'],
            config=station.config_thisrun,
            station=station,
        )

    # Postprocessor (matches update_length_rabi_f0g1_sweep)
    def chevron_postproc(station, analysis):
        best_freq = analysis.results.get('best_frequency_contrast')
        station.ds_thisrun.update_freq('M1', best_freq)
        station.config_thisrun.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = best_freq
        print(f"\n  [Postproc] Updated frequency to: {best_freq:.4f} MHz")

    # Create SweepRunner
    chevron_runner = SweepRunner(
        station=station,
        ExptClass=meas.LengthRabiGeneralF0g1Experiment,
        default_expt_cfg=chevron_defaults,
        sweep_param='freq',
        preprocessor=chevron_preproc,
        postprocessor=chevron_postproc,
        analysis_factory=chevron_analysis_factory,
        live_analysis_fn=None,  # Disable live plotting for fair comparison
    )

    # Run sweep
    print(f"\nRunning new method sweep...")
    start_time = datetime.now()
    result = chevron_runner.run(
        sweep_start=freq_start,
        sweep_stop=freq_stop,
        sweep_step=freq_step,
        postprocess=True,
        incremental_save=True,
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"✓ New method complete in {elapsed:.1f}s")

    # Get the saved file - need to find most recent .h5 file with the right prefix
    import glob
    sweep_files = sorted(
        glob.glob(str(station.data_path / "*LengthRabiGeneralF0g1Experiment_sweep.h5")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True
    )
    if sweep_files:
        new_file = sweep_files[0]  # Most recent
        print(f"  Saved to: {new_file}")
    else:
        raise FileNotFoundError("Could not find sweep file!")

    return new_file, result, elapsed


def compare_analysis_results(old_chevron, new_result):
    """Compare analysis results between old and new methods."""
    print("\n" + "="*70)
    print("ANALYSIS RESULTS COMPARISON")
    print("="*70)

    # Check if both have results
    old_has_results = old_chevron is not None and hasattr(old_chevron, 'results')
    new_has_results = hasattr(new_result, 'results')

    print(f"\nOld method has results: {old_has_results}")
    print(f"New method has results: {new_has_results}")

    if not (old_has_results and new_has_results):
        print("\n⚠ Cannot compare - one or both methods don't have analysis results")
        return

    # Compare key results
    old_res = old_chevron.results
    new_res = new_result.results

    print("\nKey metrics:")
    print(f"{'Metric':<30s} {'Old':<20s} {'New':<20s} {'Match':<10s}")
    print("-" * 70)

    for key in ['best_frequency_contrast', 'best_contrast', 'period', 'linewidth']:
        if key in old_res and key in new_res:
            old_val = old_res[key]
            new_val = new_res[key]

            if old_val is not None and new_val is not None:
                match = np.isclose(old_val, new_val, rtol=1e-4)
                status = "✓" if match else "✗"
                print(f"{key:<30s} {old_val:<20.6f} {new_val:<20.6f} {status:<10s}")
            else:
                print(f"{key:<30s} {str(old_val):<20s} {str(new_val):<20s} {'N/A':<10s}")


def main():
    parser = argparse.ArgumentParser(description='Compare old vs new chevron sweep methods')
    parser.add_argument('--method', choices=['old', 'new', 'both'], default='both',
                        help='Which method to run (default: both)')
    parser.add_argument('--sweep', choices=['fine', 'coarse'], default='fine',
                        help='Sweep range: coarse (±3 MHz) or fine (±0.5 MHz)')
    args = parser.parse_args()

    print("="*70)
    print("CHEVRON SWEEP COMPARISON: Old vs New")
    print("="*70)

    # Initialize station
    print("\nInitializing station...")
    station = MultimodeStation(experiment_name="260101_qsim")

    # Get M1 frequency
    freq_m1 = station.ds_thisrun.get_freq('M1')
    print(f"Current M1 frequency: {freq_m1:.2f} MHz")

    # Set sweep range (matching notebook)
    if args.sweep == 'coarse':
        freq_start = freq_m1 - 3.0
        freq_stop = freq_m1 + 3.0
        freq_step = 0.3
        sweep_type = "COARSE"
    else:  # fine
        freq_start = freq_m1 - 0.5
        freq_stop = freq_m1 + 0.5
        freq_step = 0.1
        sweep_type = "FINE"

    n_freq = int((freq_stop - freq_start) / freq_step) + 1
    print(f"\nRunning {sweep_type} sweep:")
    print(f"  Range: {freq_start:.2f} to {freq_stop:.2f} MHz")
    print(f"  Step: {freq_step} MHz")
    print(f"  Points: {n_freq} freq × 25 length = {n_freq * 25} total")

    # Run methods
    old_file = None
    new_file = None
    old_chevron = None
    new_result = None
    old_time = 0
    new_time = 0

    if args.method in ['old', 'both']:
        old_file, old_chevron, old_time = run_old_method(
            station, freq_start, freq_stop, freq_step
        )

    if args.method in ['new', 'both']:
        new_file, new_result, new_time = run_new_method(
            station, freq_start, freq_stop, freq_step
        )

    # Compare results
    if args.method == 'both':
        compare_h5_files(Path(old_file), Path(new_file))
        compare_analysis_results(old_chevron, new_result)

        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        print(f"Old method: {old_time:.1f}s")
        print(f"New method: {new_time:.1f}s")
        print(f"Difference: {abs(new_time - old_time):.1f}s ({100*(new_time-old_time)/old_time:+.1f}%)")

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
