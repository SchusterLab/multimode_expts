# Persistent Monitoring Guide

This guide explains how to use the persistent monitoring system to monitor long-running experiments even when the Jupyter client disconnects.

## Problem

When running long experiments in Jupyter:
- If the client disconnects, all outputs (progress bars, print statements, plots) are lost
- No way to monitor progress remotely
- Must use RDP to check on experiments

## Solution

The persistent monitoring system provides:
1. **File-based logging** - All output persists to log files
2. **Progress tracking** - Status written to JSON files
3. **Web dashboard** - Monitor progress via HTTP from any device
4. **Persistent progress bars** - TQDM progress bars that survive disconnects

## Setup

### 1. In Your Notebook

At the beginning of your notebook, after setting up paths:

```python
from persistent_monitoring import create_monitor

# Create monitor (logs and status will be saved to expt_path/monitoring/)
monitor = create_monitor(expt_path, "autocalibration_run_20250101")
```

### 2. Replace Print Statements

**Before:**
```python
print("Starting experiment...")
print(f"Running step {i} of {total}")
```

**After:**
```python
monitor.log("Starting experiment...")
monitor.log(f"Running step {i} of {total}")
```

### 3. Replace TQDM Progress Bars

**Before:**
```python
from tqdm.notebook import tqdm
for i in tqdm(range(100), desc="Running experiments"):
    # do work
    pass
```

**After:**
```python
for i in monitor.tqdm(range(100), desc="Running experiments"):
    # do work
    monitor.log(f"Completed experiment {i}")
```

### 4. Update Progress Manually

For complex experiments with multiple steps:

```python
monitor.update_progress(
    step_name="Resonator Spectroscopy",
    step_number=1,
    total_steps=10,
    message="Finding readout frequency..."
)

# Later...
monitor.update_progress(
    step_name="Qubit Calibration",
    step_number=2,
    total_steps=10,
    message="Calibrating qubit frequency..."
)
```

### 5. Save Plots Persistently

**Before:**
```python
plt.figure()
plt.plot(data)
plt.show()  # Lost on disconnect!
```

**After:**
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data)
monitor.save_plot(fig, "resonator_spectrum.png")
plt.close(fig)  # Close to free memory
```

### 6. Mark Experiment Complete

```python
monitor.complete("All experiments completed successfully!")
```

## Running the Web Dashboard

### On the Server PC

Start the dashboard server (can run in background):

```bash
# Basic usage
python monitor_dashboard.py --status-dir /path/to/experiments/data/status --port 8888

# With all directories
python monitor_dashboard.py \
    --status-dir /path/to/experiments/data/status \
    --log-dir /path/to/experiments/data/logs \
    --plots-dir /path/to/experiments/data/plots \
    --host 0.0.0.0 \
    --port 8888
```

### Access from Any Device

Once the server is running, access the dashboard from any device on the network:

```
http://server-ip-address:8888
```

The dashboard will:
- Auto-refresh every 5 seconds
- Show current progress percentage
- Display recent messages and logs
- Show any errors
- Display recent plots (if plots directory is configured)

## Integration with Autocalibration Helpers

You can combine the monitoring system with the autocalibration helpers:

```python
from autocalib_helpers import create_experiment_workflow
from persistent_monitoring import create_monitor

# Setup
monitor = create_monitor(expt_path, "autocalibration")
executor, updater, extractor = create_experiment_workflow(
    soc, expt_path, config_path, config_thisrun, ds_thisrun
)

# Run experiments with monitoring
monitor.update_progress("Resonator Spectroscopy", 1, 10)
if expts_to_run['res_spec']:
    monitor.log("Starting resonator spectroscopy...")
    rspec = executor.execute_if(
        condition=True,
        experiment_class=meas.single_qubit.resonator_spectroscopy.ResonatorSpectroscopyExperiment,
        prefix='ResonatorSpectroscopyExperiment',
        expt_params={'start': 749, 'step': 0.01, 'expts': 250, 'reps': 500},
        relax_delay=50
    )
    updater.update_readout_frequency(rspec.data['fit'][0])
    monitor.log("Resonator spectroscopy completed!")
    monitor.update_progress("Resonator Spectroscopy", 1, 10, "Completed")
```

## File Structure

The monitoring system creates the following structure:

```
experiment_data/
├── data/              # Your experiment data files
├── monitoring/
│   ├── logs/
│   │   └── experiment_20250101_120000.log
│   ├── status/
│   │   └── experiment_status.json
│   └── plots/
│       ├── resonator_spectrum.png
│       └── qubit_rabi.png
```

## Status JSON Format

The status file contains:

```json
{
  "experiment_name": "autocalibration",
  "start_time": "2025-01-01T12:00:00",
  "last_update": "2025-01-01T12:30:00",
  "current_step": 5,
  "total_steps": 10,
  "progress_percent": 50.0,
  "status": "running",
  "steps": [
    {
      "name": "Resonator Spectroscopy",
      "step_number": 1,
      "timestamp": "2025-01-01T12:05:00",
      "message": "Finding readout frequency..."
    }
  ],
  "messages": [
    {
      "timestamp": "2025-01-01T12:05:00",
      "level": "info",
      "message": "Starting resonator spectroscopy..."
    }
  ],
  "errors": []
}
```

## Tips

1. **Run dashboard in background**: Use `nohup` or `screen`/`tmux` to keep it running
   ```bash
   nohup python monitor_dashboard.py --status-dir /path/to/status --port 8888 > dashboard.log 2>&1 &
   ```

2. **Check logs directly**: If dashboard is down, you can still check:
   - Status: `cat /path/to/status/experiment_status.json`
   - Logs: `tail -f /path/to/logs/experiment_*.log`

3. **Multiple experiments**: Each experiment gets its own status file, so you can monitor multiple experiments

4. **Firewall**: Make sure port 8888 (or your chosen port) is open if accessing from remote network

5. **Auto-start dashboard**: Add to startup script or systemd service

## Troubleshooting

**Dashboard shows "No status files found"**
- Check that status directory path is correct
- Verify that monitor is actually writing status files
- Check file permissions

**Can't access dashboard from remote device**
- Check firewall settings
- Verify server is bound to `0.0.0.0` not `127.0.0.1`
- Check network connectivity

**Progress not updating**
- Check that monitor.update_progress() is being called
- Verify status file is being written (check file modification time)
- Check for errors in notebook output

## Example: Full Integration

```python
# Setup
from persistent_monitoring import create_monitor
from autocalib_helpers import create_experiment_workflow

monitor = create_monitor(expt_path, "autocalibration_full")
executor, updater, extractor = create_experiment_workflow(
    soc, expt_path, config_path, config_thisrun, ds_thisrun
)

# Define experiment sequence
experiments = [
    ("Resonator Spectroscopy", 'res_spec'),
    ("Single Shot", 'single_shot'),
    ("Pulse Probe GE", 'pulse_probe_ge'),
    ("T2 Ramsey GE", 't2_ge'),
    # ... more experiments
]

monitor.update_progress("Starting", 0, len(experiments))

# Run experiments
for i, (name, key) in enumerate(monitor.tqdm(experiments, desc="Running calibration")):
    monitor.update_progress(name, i+1, len(experiments), f"Starting {name}...")
    
    if expts_to_run.get(key):
        monitor.log(f"Running {name}...")
        # Run experiment using executor
        # ... experiment code ...
        monitor.log(f"{name} completed!")
    else:
        monitor.log(f"Skipping {name} (not in expts_to_run)")

monitor.complete("All experiments completed!")
```

Now you can:
- Disconnect from Jupyter
- Check progress via web dashboard
- View logs and plots remotely
- No need for RDP!









