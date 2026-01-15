# Job Server for Multi-User Experiment Scheduling

This package provides a job queue system that allows multiple users to submit experiment jobs from separate SSH connections. Jobs are scheduled and executed sequentially to ensure hardware exclusivity.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Users (via SSH)                             │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │   Jupyter    │  │   Jupyter    │  │   Jupyter    │         │
│   │  (User A)    │  │  (User B)    │  │  (User C)    │         │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│          │                 │                 │                  │
│          └─────────────────┼─────────────────┘                  │
│                            │                                    │
│                            ▼                                    │
│                   ┌─────────────────┐                           │
│                   │   JobClient     │  Python library           │
│                   └────────┬────────┘                           │
│                            │ HTTP requests                      │
│                            ▼                                    │
│              ┌──────────────────────────┐                       │
│              │   FastAPI Job Server     │  (port 8000)          │
│              │                          │                       │
│              │  ┌────────┐ ┌─────────┐  │                       │
│              │  │ SQLite │ │   ID    │  │                       │
│              │  │  Queue │ │Generator│  │                       │
│              │  └────────┘ └─────────┘  │                       │
│              └────────────┬─────────────┘                       │
│                           │                                     │
│                           │ polls for jobs                      │
│                           ▼                                     │
│              ┌──────────────────────────┐                       │
│              │      Job Worker          │  (single instance)    │
│              │                          │                       │
│              │  ┌────────────────────┐  │                       │
│              │  │  MultimodeStation  │  │                       │
│              │  │  (or MockStation)  │  │                       │
│              │  └────────────────────┘  │                       │
│              └────────────┬─────────────┘                       │
│                           │                                     │
│                           ▼                                     │
│              ┌──────────────────────────┐                       │
│              │     Hardware (QICK)      │                       │
│              └──────────────────────────┘                       │
│                           │                                     │
│                           ▼                                     │
│              ┌──────────────────────────┐                       │
│              │   HDF5 Data Files        │                       │
│              │  JOB-20260113-00001.h5   │                       │
│              └──────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Job Server (`server.py`)

A FastAPI HTTP server that:
- Receives job submissions from clients
- Generates unique job IDs (format: `JOB-YYYYMMDD-NNNNN`)
- Stores jobs in a SQLite database
- Provides endpoints for status queries and queue management

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and queue statistics |
| `/jobs/submit` | POST | Submit a new job |
| `/jobs/queue` | GET | List pending and running jobs |
| `/jobs/history` | GET | List recent job history |
| `/jobs/{job_id}` | GET | Get status of specific job |
| `/jobs/{job_id}` | DELETE | Cancel a pending job |

### 2. Job Worker (`worker.py`)

A daemon process that:
- Polls the database for pending jobs
- Executes jobs one at a time (ensuring hardware exclusivity)
- Updates job status (pending → running → completed/failed)
- Saves experiment data to HDF5 files with job-based naming

**Execution Order:**
1. Jobs with higher priority run first
2. Within same priority, jobs run in FIFO order (first submitted, first executed)

### 3. Job Client (`client.py`)

A Python library for users to interact with the job server:
- Submit jobs with explicit parameters
- Check job status
- Wait for job completion
- List queue and cancel jobs

### 4. Config Versioning (`config_versioning.py`)

Manages immutable snapshots of configuration files:
- Creates copies of config files before each job runs
- Uses SHA256 checksums to avoid duplicate copies
- Links config versions to job IDs for reproducibility
- Tracks "main" (canonical, most up-to-date) version for each config type
- Provides push/pull functions to update and retrieve main configs

### 5. Mock Hardware (`mock_hardware.py`)

Provides simulated hardware for testing:
- `MockStation`: Mimics MultimodeStation without real hardware
- `MockQickConfig`: Simulates QICK SoC configuration
- Generates synthetic Rabi oscillation data for testing

## Database Schema

### Jobs Table
| Column | Type | Description |
|--------|------|-------------|
| job_id | STRING | Unique ID (JOB-YYYYMMDD-NNNNN) |
| user | STRING | Username of submitter |
| experiment_class | STRING | Class name (e.g., "AmplitudeRabiExperiment") |
| experiment_module | STRING | Module path for dynamic import |
| experiment_config | JSON | Experiment-specific parameters |
| status | ENUM | pending, running, completed, failed, cancelled |
| priority | INT | Higher = runs sooner (default: 0) |
| created_at | DATETIME | Submission timestamp |
| started_at | DATETIME | Execution start timestamp |
| completed_at | DATETIME | Completion timestamp |
| data_file_path | STRING | Path to output HDF5 file |
| error_message | TEXT | Error details if failed |

### Config Versions Table
| Column | Type | Description |
|--------|------|-------------|
| version_id | STRING | Unique ID (CFG-{TYPE}-YYYYMMDD-NNNNN) |
| config_type | ENUM | hardware_config, multiphoton_config, etc. |
| original_filename | STRING | Original file name |
| snapshot_path | STRING | Path to versioned copy |
| checksum | STRING | SHA256 hash for deduplication |
| created_by_job_id | STRING | Job that triggered this snapshot |

### Main Configs Table
| Column | Type | Description |
|--------|------|-------------|
| config_type | ENUM | hardware_config, multiphoton_config, etc. |
| version_id | STRING | Main (canonical) version |
| updated_at | DATETIME | When this was set as main |
| updated_by | STRING | Username who set this as main |

## Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn sqlalchemy pydantic requests
```

### 2. Start the Server

```bash
cd /path/to/multimode_expts
pixi run python -m uvicorn job_server.server:app --host 0.0.0.0 --port 8000
```

### 3. Start the Worker

**For testing (mock hardware):**
```bash
pixi run python -m job_server.worker --mock
```

**For real hardware:**
```bash
pixi run python -m job_server.worker
```

## Usage in Jupyter Notebooks

### Basic Example: Submit and Wait

```python
from job_server import JobClient

# Create client (connects to localhost:8000 by default)
client = JobClient()

# Check server health
health = client.health_check()
print(f"Server status: {health['status']}")
print(f"Pending jobs: {health['pending_jobs']}")

# Submit an experiment job
job_id = client.submit_job(
    experiment_class="AmplitudeRabiExperiment",
    experiment_module="experiments.single_qubit.amplitude_rabi",
    expt_config={
        "start": 0,
        "step": 100,
        "expts": 50,
        "reps": 1000,
        "rounds": 1,
        "qubits": [0],
        "sigma_test": 0.035,
        "pulse_type": "gauss",
    },
    user="Claude",  # Your username
    priority=0,     # Default priority
)

print(f"Job submitted: {job_id}")

# Wait for completion (blocks until done)
result = client.wait_for_completion(job_id)

if result.is_successful():
    print(f"Success! Data saved to: {result.data_file_path}")
else:
    print(f"Job failed: {result.error_message}")
```

### Example: Submit Multiple Jobs

```python
from job_server import JobClient

client = JobClient()

# Define a list of experiments to run
experiments = [
    {
        "class": "T1Experiment",
        "module": "experiments.single_qubit.t1",
        "config": {"start": 0, "step": 0.5, "expts": 100, "reps": 200},
    },
    {
        "class": "T2RamseyExperiment",
        "module": "experiments.single_qubit.t2_ramsey",
        "config": {"start": 0, "step": 0.1, "expts": 100, "reps": 200},
    },
    {
        "class": "ResonatorSpectroscopyExperiment",
        "module": "experiments.single_qubit.resonator_spectroscopy",
        "config": {"start": 7000, "stop": 7100, "expts": 101, "reps": 500},
    },
]

# Submit all jobs
job_ids = []
for exp in experiments:
    job_id = client.submit_job(
        experiment_class=exp["class"],
        experiment_module=exp["module"],
        expt_config=exp["config"],
        user="Claude",
    )
    job_ids.append(job_id)
    print(f"Submitted {exp['class']}: {job_id}")

# Check queue
client.print_queue()

# Wait for all jobs to complete
results = []
for job_id in job_ids:
    result = client.wait_for_completion(job_id)
    results.append(result)
    print(f"{job_id}: {result.status}")
```

### Example: Priority Scheduling

```python
from job_server import JobClient

client = JobClient()

# Submit a low-priority background calibration
bg_job = client.submit_job(
    experiment_class="ResonatorSpectroscopyExperiment",
    experiment_module="experiments.single_qubit.resonator_spectroscopy",
    expt_config={"start": 7000, "stop": 7100, "expts": 101},
    user="Claude",
    priority=0,  # Low priority
)

# Submit a high-priority urgent experiment
urgent_job = client.submit_job(
    experiment_class="T1Experiment",
    experiment_module="experiments.single_qubit.t1",
    expt_config={"start": 0, "step": 0.5, "expts": 100},
    user="Claude",
    priority=10,  # High priority - runs first!
)

# The urgent_job will run before bg_job even though it was submitted second
client.print_queue()
```

### Example: Non-Blocking Submission

```python
from job_server import JobClient
import time

client = JobClient()

# Submit job without waiting
job_id = client.submit_job(
    experiment_class="AmplitudeRabiExperiment",
    experiment_module="experiments.single_qubit.amplitude_rabi",
    expt_config={"start": 0, "step": 100, "expts": 50, "reps": 1000},
    user="Claude",
)

print(f"Submitted: {job_id}")
print("Doing other work while job runs...")

# Do other work here
time.sleep(5)

# Check status periodically
status = client.get_status(job_id)
print(f"Current status: {status.status}")

# When ready, wait for completion
if not status.is_done():
    result = client.wait_for_completion(job_id)
```

### Example: Cancel a Job

```python
from job_server import JobClient

client = JobClient()

# Submit a job
job_id = client.submit_job(
    experiment_class="AmplitudeRabiExperiment",
    experiment_module="experiments.single_qubit.amplitude_rabi",
    expt_config={"start": 0, "step": 100, "expts": 50, "reps": 1000},
    user="Claude",
)

# Changed your mind? Cancel it (only works if still pending)
try:
    client.cancel_job(job_id)
    print(f"Job {job_id} cancelled")
except Exception as e:
    print(f"Could not cancel: {e}")
```

### Example: View Job History

```python
from job_server import JobClient

client = JobClient()

# Get recent job history
history = client.get_history(limit=20)
for job in history:
    print(f"{job['job_id']}: {job['status']} - {job['experiment_class']}")

# Filter by user
my_jobs = client.get_history(user="Claude", limit=10)

# Filter by status
completed = client.get_history(status="completed", limit=10)
failed = client.get_history(status="failed", limit=10)
```

### Example: Load Data from Completed Job

```python
from job_server import JobClient
from slab.datamanagement import SlabFile
import numpy as np

client = JobClient()

# Get a completed job's data file path
job_id = "JOB-20260113-00001"
result = client.get_status(job_id)

if result.is_successful() and result.data_file_path:
    # Load the HDF5 data
    with SlabFile(result.data_file_path, 'r') as f:
        # Load config
        config = f.load_config()

        # Load data arrays
        xpts = np.array(f['xpts'])
        avgi = np.array(f['avgi'])
        avgq = np.array(f['avgq'])

    print(f"Loaded {len(xpts)} data points")
```

### Example: Push/Pull Config Versions

The config versioning system tracks the "main" (canonical, most up-to-date) version of each config type. Use push to update the main version, and pull to retrieve it.

```python
from pathlib import Path
from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager
from job_server.models import ConfigType

# Initialize
db = get_database()
config_dir = Path("multimode_expts/configs")
manager = ConfigVersionManager(config_dir)

with db.session() as session:
    # PUSH: Update the main hardware config
    # This creates a snapshot and sets it as the main version
    version_id, snapshot_path = manager.push_hardware_config_to_main(
        source_path=config_dir / "hardware_config_202505.yml",
        session=session,
        updated_by="Claude",  # Your username
    )
    print(f"Pushed new main config: {version_id}")

    # PULL: Get the main hardware config version
    main_version = manager.get_main_version(ConfigType.HARDWARE_CONFIG, session)
    if main_version:
        print(f"Main version: {main_version.version_id}")
        print(f"Snapshot path: {main_version.snapshot_path}")

    # Get path to main config file
    main_path = manager.get_main_config_path(ConfigType.HARDWARE_CONFIG, session)
    print(f"Main config path: {main_path}")

    # Get all main configs at once
    all_main = manager.get_all_main_configs(session)
    print(f"All main versions: {all_main}")
```

### Example: Set Existing Version as Main

```python
from job_server.database import get_database
from job_server.config_versioning import ConfigVersionManager
from job_server.models import ConfigType
from pathlib import Path

db = get_database()
manager = ConfigVersionManager(Path("multimode_expts/configs"))

with db.session() as session:
    # List available versions
    versions = manager.list_versions(ConfigType.HARDWARE_CONFIG, session)
    for v in versions:
        print(f"{v.version_id}: {v.original_filename} ({v.created_at})")

    # Set a specific version as main (without creating a new snapshot)
    manager.set_main_version(
        config_type=ConfigType.HARDWARE_CONFIG,
        version_id="CFG-HW-20260113-00001",
        session=session,
        updated_by="Claude",
    )
```

## File Locations

| Item | Path |
|------|------|
| Database | `multimode_expts/data/jobs.db` |
| Config versions | `multimode_expts/configs/versions/` |
| Data files | Configured in `hardware_config.yml` → `data_management.output_root` |

## Troubleshooting

### Server won't start (port in use)
```bash
# Find and kill existing process
lsof -ti:8000 | xargs kill -9
```

### Worker can't connect to hardware
- Ensure only one worker is running
- Check that InstrumentManager nameserver is accessible
- Use `--mock` flag for testing without hardware

### Jobs stuck in "pending"
- Verify the worker is running
- Check worker logs for errors
- Ensure experiment module path is correct

### Import errors
- Install all dependencies: `pip install fastapi uvicorn sqlalchemy pydantic requests pyvisa pyserial Pyro4 lmfit`
