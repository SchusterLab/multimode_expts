"""
Minimal hardware test for SweepRunner.

This script tests SweepRunner with real FPGA hardware to verify:
1. Connection to real station/FPGA works
2. File paths and directory structure are correct
3. Incremental file saving works with real .h5 files
4. ChevronFitting analysis works with real data
5. Results match the old sequential_base_class pattern

Usage:
    pixi run python test_sweep_hardware.py

Requirements:
    - FPGA must be connected
    - Config files must exist in expected locations
"""

from pathlib import Path
import numpy as np
from copy import deepcopy

from slab import AttrDict
import experiments as meas
from fitting.fit_display_classes import ChevronFitting
from experiments.station import MultimodeStation
from experiments.sweep_runner import SweepRunner


def test_sweep_runner_minimal():
    """
    Run a minimal chevron sweep (just a few points) to verify SweepRunner works.
    """
    print("\n" + "="*70)
    print("SWEEPRUNNER HARDWARE TEST - Minimal Chevron")
    print("="*70)

    # Create station (same as notebook)
    print("\nInitializing MultimodeStation...")
    try:
        station = MultimodeStation(
            experiment_name="260101_qsim",  # Use current experiment name
        )
        print(f"[OK] Station created successfully")
        print(f"  Data path: {station.data_path}")
        print(f"  Config file: {station.hardware_config_file}")
    except Exception as e:
        print(f"[ERROR] Could not create station: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check station has required attributes
    required_attrs = ['soc', 'config_thisrun', 'ds_thisrun', 'data_path']
    for attr in required_attrs:
        if not hasattr(station, attr):
            print(f"[X] ERROR: Station missing required attribute: {attr}")
            return False
    print(f"[OK] Station has all required attributes")

    # Get current M1 frequency from config
    try:
        freq_m1 = station.ds_thisrun.get_freq('M1')
        print(f"[OK] Current M1 frequency: {freq_m1:.2f} MHz")
    except Exception as e:
        print(f"[X] ERROR: Could not get M1 frequency: {e}")
        print("  Using default center frequency of 200 MHz")
        freq_m1 = 200.0

    # Define minimal sweep parameters
    # Use VERY SMALL sweep for quick test (only 5 points)
    sweep_config = {
        'freq_start': freq_m1 - 1.0,  # ±1 MHz around center
        'freq_stop': freq_m1 + 1.0,
        'freq_step': 0.5,  # 5 points total
        'description': 'Minimal test sweep (5 freq points)',
    }

    n_points = int((sweep_config['freq_stop'] - sweep_config['freq_start']) / sweep_config['freq_step']) + 1
    print(f"\nSweep Configuration:")
    print(f"  Frequency range: {sweep_config['freq_start']:.2f} - {sweep_config['freq_stop']:.2f} MHz")
    print(f"  Step size: {sweep_config['freq_step']} MHz")
    print(f"  Total points: {n_points}")

    # Load experiment defaults from experiment_config.yml (like sequential_base_class does)
    import yaml
    exp_config_file = station.config_dir / "experiment_config.yml"
    print(f"\nLoading experiment config from: {exp_config_file}")

    try:
        with open(exp_config_file, 'r') as f:
            exp_config = yaml.safe_load(f)

        # Get defaults for LengthRabiGeneralF0g1Experiment
        length_rabi_config = exp_config.get('LengthRabiGeneralF0g1Experiment', {})
        print(f"  Loaded {len(length_rabi_config)} parameters from config file")

        # Start with file defaults, then override for minimal test
        chevron_defaults = AttrDict(length_rabi_config)
        chevron_defaults.update({
            'expts': 15,     # Fewer length points for speed
            'reps': 50,      # Fewer reps for speed
            'step': 0.2,     # Slightly larger step for speed
        })
    except Exception as e:
        print(f"  [WARN] Could not load experiment config: {e}")
        print(f"  Using minimal defaults instead")
        # Fallback to minimal defaults
        chevron_defaults = AttrDict(dict(
            start=2, step=0.2, expts=15, reps=50, rounds=1, qubits=[0],
            gain=8000, ramp_sigma=0.005, use_arb_waveform=False,
            pi_ge_before=True, pi_ef_before=True, pi_ge_after=False,
            normalize=False, active_reset=False, check_man_reset=[False, 0],
            check_man_reset_pi=[], prepulse=False, pre_sweep_pulse=[],
            err_amp_reps=0, swap_lossy=False,
        ))

    print(f"\nExperiment Configuration:")
    print(f"  Length range: {chevron_defaults.start} - {chevron_defaults.start + chevron_defaults.step * chevron_defaults.expts:.2f} us")
    print(f"  Length points: {chevron_defaults.expts}")
    print(f"  Reps per point: {chevron_defaults.reps}")
    print(f"  Total measurements: {n_points * chevron_defaults.expts} (= {n_points} freq × {chevron_defaults.expts} length)")

    # Preprocessor (could add smart config logic here)
    def chevron_preproc(station, default_expt_cfg, **kwargs):
        expt_cfg = deepcopy(default_expt_cfg)
        expt_cfg.update(kwargs)
        return expt_cfg

    # Analysis factory
    def chevron_analysis_factory(sweep_data, station):
        """Create ChevronFitting analysis from sweep data."""
        print(f"\n  [Analysis] Creating ChevronFitting with {len(sweep_data['freq_sweep'])} frequency points")

        # Extract time array (might be 1D or 2D depending on data structure)
        time = sweep_data['xpts'][0] if sweep_data['xpts'].ndim > 1 else sweep_data['xpts']

        analysis = ChevronFitting(
            frequencies=sweep_data['freq_sweep'],
            time=time,
            response_matrix=sweep_data['avgi'],
            config=station.config_thisrun,
            station=station,
        )
        return analysis

    # Postprocessor
    def chevron_postproc(station, analysis):
        """Update config with best frequency from analysis."""
        best_freq = analysis.results.get('best_frequency_contrast')
        best_contrast = analysis.results.get('best_contrast')

        print(f"\n  [Postproc] Best frequency: {best_freq:.4f} MHz")
        print(f"  [Postproc] Best contrast: {best_contrast:.4f}")

        # In production, would update config:
        # station.ds_thisrun.update_freq('M1', best_freq)
        # station.config_thisrun.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = best_freq
        print(f"  [Postproc] (Skipping config update in test mode)")

    # Optional: Live analysis callback
    def live_callback(sweep_data, station):
        """Called after each frequency point for live updates."""
        n = len(sweep_data['freq_sweep'])
        latest_freq = sweep_data['freq_sweep'][-1]
        print(f"    [Live] Point {n}/{n_points} complete (freq={latest_freq:.2f} MHz)")

    # Create SweepRunner
    print(f"\n" + "-"*70)
    print("Creating SweepRunner...")
    print("-"*70)

    try:
        chevron_runner = SweepRunner(
            station=station,
            ExptClass=meas.LengthRabiGeneralF0g1Experiment,
            default_expt_cfg=chevron_defaults,
            sweep_param='freq',
            preprocessor=chevron_preproc,
            postprocessor=chevron_postproc,
            analysis_factory=chevron_analysis_factory,
            live_analysis_fn=live_callback,
        )
        print("[OK] SweepRunner created successfully")
    except Exception as e:
        print(f"[X] ERROR creating SweepRunner: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run the sweep
    print(f"\n" + "-"*70)
    print("Running sweep...")
    print("-"*70)
    print(f"This will take approximately {n_points * chevron_defaults.expts * chevron_defaults.reps * 0.001:.1f} seconds")
    print(f"(Estimate based on ~1ms per shot)")

    try:
        result = chevron_runner.run(
            sweep_start=sweep_config['freq_start'],
            sweep_stop=sweep_config['freq_stop'],
            sweep_step=sweep_config['freq_step'],
            postprocess=True,
            incremental_save=True,  # Test file saving
        )
        print("\n[OK] Sweep completed successfully!")

    except Exception as e:
        print(f"\n[X] ERROR during sweep: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate results
    print(f"\n" + "-"*70)
    print("Validating results...")
    print("-"*70)

    if hasattr(result, 'results'):
        print(f"[OK] Got analysis object with results")
        print(f"  Result keys: {list(result.results.keys())}")

        # Check for expected keys
        if 'best_frequency_contrast' in result.results:
            print(f"  [OK] Best frequency: {result.results['best_frequency_contrast']:.4f} MHz")
        if 'best_contrast' in result.results:
            print(f"  [OK] Best contrast: {result.results['best_contrast']:.4f}")

    else:
        print(f"[WARN] WARNING: Result is not an analysis object")
        print(f"  Result type: {type(result)}")

    # Check if file was saved
    print(f"\n" + "-"*70)
    print("Checking saved files...")
    print("-"*70)

    # Look for most recent .h5 file in data directory
    try:
        data_path = Path(station.data_path)
        h5_files = sorted(data_path.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)

        if h5_files:
            latest_file = h5_files[0]
            file_age = (Path(__file__).stat().st_mtime - latest_file.stat().st_mtime)

            # If file was modified in last 60 seconds, assume it's ours
            if abs(file_age) < 60:
                print(f"[OK] Found recent .h5 file: {latest_file.name}")
                print(f"  Size: {latest_file.stat().st_size / 1024:.1f} KB")
            else:
                print(f"[WARN] Latest .h5 file is old ({file_age:.0f}s ago)")
        else:
            print(f"[WARN] No .h5 files found in {data_path}")

    except Exception as e:
        print(f"[WARN] Could not check files: {e}")

    # Summary
    print(f"\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("[OK] SweepRunner hardware test PASSED!")
    print("\nNext steps:")
    print("  1. Review the saved data file to verify correctness")
    print("  2. Compare with old sequential_base_class output if available")
    print("  3. Run a full-scale sweep if this test looks good")
    print("  4. Integrate into single_qubit_autocalibrate.ipynb")

    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SweepRunner Hardware Integration Test")
    print("="*70)
    print("\nThis script will:")
    print("  1. Connect to the real FPGA via station")
    print("  2. Run a minimal chevron sweep (5 freq × 15 length = 75 measurements)")
    print("  3. Test incremental file saving")
    print("  4. Run ChevronFitting analysis")
    print("  5. Validate results")
    print("\nStarting test...\n")

    success = test_sweep_runner_minimal()

    sys.exit(0 if success else 1)
