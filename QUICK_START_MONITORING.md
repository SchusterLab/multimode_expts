# Quick Start: Persistent Monitoring

## The Problem
When running overnight experiments in Jupyter, if you disconnect:
- ❌ Progress bars disappear
- ❌ Print statements are lost
- ❌ Plots don't show up
- ❌ No way to check progress remotely

## The Solution (3 Steps)

### Step 1: Add to Your Notebook

At the top of your notebook, after setting up paths:

```python
from persistent_monitoring import create_monitor
from autocalib_helpers import create_experiment_workflow

# Create monitor (one line!)
monitor = create_monitor(expt_path, "autocalibration")

# Add monitor to workflow (optional but recommended)
executor, updater, extractor = create_experiment_workflow(
    soc, expt_path, config_path, config_thisrun, ds_thisrun, monitor=monitor
)
```

### Step 2: Replace Print/TQDM

**Before:**
```python
print("Starting experiment...")
for i in tqdm(range(100)):
    # do work
```

**After:**
```python
monitor.log("Starting experiment...")
for i in monitor.tqdm(range(100), desc="Running experiments"):
    # do work
```

### Step 3: Start Dashboard (On Server PC)

```bash
python monitor_dashboard.py --status-dir /path/to/experiments/data/status --port 8888
```

Then access from any device: `http://server-ip:8888`

## That's It!

Now you can:
- ✅ Disconnect from Jupyter
- ✅ Monitor progress via web browser
- ✅ See all logs and messages
- ✅ Check progress from phone/tablet/laptop
- ✅ No RDP needed!

## Full Example

```python
# Setup
from persistent_monitoring import create_monitor
from autocalib_helpers import create_experiment_workflow

monitor = create_monitor(expt_path, "autocalibration")
executor, updater, extractor = create_experiment_workflow(
    soc, expt_path, config_path, config_thisrun, ds_thisrun, monitor=monitor
)

# Run experiments
monitor.log("Starting autocalibration sequence...")

if expts_to_run['res_spec']:
    monitor.log("Running resonator spectroscopy...")
    rspec = executor.execute_if(
        condition=True,
        experiment_class=meas.single_qubit.resonator_spectroscopy.ResonatorSpectroscopyExperiment,
        prefix='ResonatorSpectroscopyExperiment',
        expt_params={'start': 749, 'step': 0.01, 'expts': 250, 'reps': 500},
        relax_delay=50
    )
    updater.update_readout_frequency(rspec.data['fit'][0])
    monitor.log("Resonator spectroscopy completed!")

monitor.complete("All experiments done!")
```

## Files Created

All monitoring data is saved to:
- `expt_path/monitoring/logs/` - Log files
- `expt_path/monitoring/status/` - Status JSON files
- `expt_path/monitoring/plots/` - Saved plots

These persist even if you disconnect!









