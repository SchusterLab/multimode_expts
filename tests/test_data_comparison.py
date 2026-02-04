"""
Run old and new methods with identical parameters, no analysis/plotting.
Just save HDF5 files for manual comparison.
"""

from pathlib import Path
from copy import deepcopy
import numpy as np
import yaml

from slab import AttrDict
import experiments as meas
from experiments.station import MultimodeStation
from experiments.sweep_runner import SweepRunner
from experiments.sequential_experiment_classes import man_f0g1_class


# Common sweep parameters
FREQ_RANGE = 0.3  # MHz around center
FREQ_STEP = 0.1   # MHz
LENGTH_START = 2
LENGTH_STEP = 0.1
LENGTH_POINTS = 25
REPS = 100


def run_old_method(station, freq_m1):
    """Run old man_f0g1_class method with analysis disabled."""
    print("\n" + "="*70)
    print("RUNNING OLD METHOD")
    print("="*70)

    freq_start = freq_m1 - FREQ_RANGE
    freq_stop = freq_m1 + FREQ_RANGE

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

    # Set sweep parameters
    class_for_exp.loaded[sweep_experiment_name] = {
        'freq_start': freq_start,
        'freq_stop': freq_stop,
        'freq_step': FREQ_STEP,
        'start': LENGTH_START,
        'step': LENGTH_STEP,
        'qubits': [0],
        'expts': LENGTH_POINTS,
        'reps': REPS,
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

    n_freq = int((freq_stop - freq_start) / FREQ_STEP) + 1
    print(f"Sweep: {n_freq} freq × {LENGTH_POINTS} length = {n_freq * LENGTH_POINTS} measurements")
    print(f"Freq: {freq_start:.2f} to {freq_stop:.2f} MHz (step {FREQ_STEP})")
    print(f"Length: {LENGTH_START} to {LENGTH_START + LENGTH_STEP * LENGTH_POINTS:.1f} us ({LENGTH_POINTS} points)")
    print(f"Reps: {REPS}")
    print(f"Expected time: ~{n_freq * 10:.0f}s ({n_freq} points × 10s)")

    # HACK: Disable live analysis by replacing perform_chevron_analysis with no-op
    class_for_exp.perform_chevron_analysis = lambda: None

    print("\nRunning old method (live analysis disabled)...")
    import time
    start_time = time.time()

    # Run sweep
    class_for_exp.run_sweep(sweep_experiment_name)

    elapsed = time.time() - start_time

    old_file = class_for_exp.expt_sweep.fname
    print(f"\nOld method complete in {elapsed:.1f}s")
    print(f"Saved to: {old_file}")

    return old_file, elapsed


def run_new_method(station, freq_m1):
    """Run new SweepRunner method."""
    print("\n" + "="*70)
    print("RUNNING NEW METHOD")
    print("="*70)

    freq_start = freq_m1 - FREQ_RANGE
    freq_stop = freq_m1 + FREQ_RANGE

    # Create config matching old method exactly
    chevron_defaults = AttrDict(dict(
        start=LENGTH_START,
        step=LENGTH_STEP,
        expts=LENGTH_POINTS,
        reps=REPS,
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

    n_freq = int((freq_stop - freq_start) / FREQ_STEP) + 1
    print(f"Sweep: {n_freq} freq × {LENGTH_POINTS} length = {n_freq * LENGTH_POINTS} measurements")
    print(f"Freq: {freq_start:.2f} to {freq_stop:.2f} MHz (step {FREQ_STEP})")
    print(f"Length: {LENGTH_START} to {LENGTH_START + LENGTH_STEP * LENGTH_POINTS:.1f} us ({LENGTH_POINTS} points)")
    print(f"Reps: {REPS}")
    print(f"Expected time: ~{n_freq * 10:.0f}s ({n_freq} points × 10s)")

    # Preprocessor
    def chevron_preproc(station, default_expt_cfg, **kwargs):
        expt_cfg = deepcopy(default_expt_cfg)
        expt_cfg.update(kwargs)
        return expt_cfg

    # Create SweepRunner (no analysis, no live callbacks)
    chevron_runner = SweepRunner(
        station=station,
        ExptClass=meas.LengthRabiGeneralF0g1Experiment,
        default_expt_cfg=chevron_defaults,
        sweep_param='freq',
        preprocessor=chevron_preproc,
        postprocessor=None,
        analysis_factory=None,  # No analysis
        live_analysis_fn=None,  # No live callbacks
    )

    print("\nRunning new method (no analysis)...")
    import time
    start_time = time.time()

    # Run sweep
    result = chevron_runner.run(
        sweep_start=freq_start,
        sweep_stop=freq_stop,
        sweep_step=FREQ_STEP,
        postprocess=False,
        incremental_save=True,
    )

    elapsed = time.time() - start_time

    # Find the saved file
    import glob
    sweep_files = sorted(
        glob.glob(str(station.data_path / "*LengthRabiGeneralF0g1Experiment_sweep.h5")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True
    )
    new_file = sweep_files[0] if sweep_files else None

    print(f"\nNew method complete in {elapsed:.1f}s")
    print(f"Saved to: {new_file}")

    return new_file, elapsed


def compare_files_quick(old_file, new_file):
    """Quick comparison of two HDF5 files."""
    from slab.datamanagement import SlabFile

    print("\n" + "="*70)
    print("QUICK COMPARISON")
    print("="*70)

    with SlabFile(old_file) as f_old, SlabFile(new_file) as f_new:
        old_keys = set(f_old.keys())
        new_keys = set(f_new.keys())

        print(f"\nKeys:")
        print(f"  Old: {sorted(old_keys)}")
        print(f"  New: {sorted(new_keys)}")

        if old_keys == new_keys:
            print("  Status: MATCH")
        else:
            print("  Status: DIFFER")
            if old_keys - new_keys:
                print(f"    Only in old: {old_keys - new_keys}")
            if new_keys - old_keys:
                print(f"    Only in new: {new_keys - old_keys}")

        print(f"\nArray shapes:")
        for key in sorted(old_keys & new_keys):
            old_arr = np.array(f_old[key])
            new_arr = np.array(f_new[key])
            match = "MATCH" if old_arr.shape == new_arr.shape else "DIFFER"
            print(f"  {key:20s} old={str(old_arr.shape):15s} new={str(new_arr.shape):15s} [{match}]")

    print(f"\nFiles ready for manual inspection:")
    print(f"  Old: {old_file}")
    print(f"  New: {new_file}")


def main():
    print("="*70)
    print("DATA COMPARISON: Old vs New (No Analysis)")
    print("="*70)

    # Initialize station
    print("\nInitializing station...")
    station = MultimodeStation(experiment_name="260101_qsim")
    freq_m1 = station.ds_thisrun.get_freq('M1')
    print(f"M1 frequency: {freq_m1:.2f} MHz")

    # Run both methods
    old_file, old_time = run_old_method(station, freq_m1)
    new_file, new_time = run_new_method(station, freq_m1)

    # Quick comparison
    compare_files_quick(old_file, new_file)

    # Performance summary
    print("\n" + "="*70)
    print("PERFORMANCE")
    print("="*70)
    print(f"Old method: {old_time:.1f}s")
    print(f"New method: {new_time:.1f}s")
    print(f"Difference: {abs(new_time - old_time):.1f}s ({100*(new_time-old_time)/old_time:+.1f}%)")

    print("\n" + "="*70)
    print("DONE - Ready for manual file comparison")
    print("="*70)


if __name__ == '__main__':
    main()
