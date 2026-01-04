# Architecture Analysis: Program, Experiment, and Sequential Wrappers

## Core Architecture (Working Backwards from Hardware)

### Layer 1: Program Classes (FPGA Communication)
**Example:** `LengthRabiF0g1GeneralProgram(MMAveragerProgram)`

**Purpose:** Compile pulse sequences into FPGA instructions

**Key Methods:**
- `initialize()` - Setup channels, frequencies, waveforms
- `body()` - Define pulse sequence (reset, pi pulses, measurement)
- `collect_shots()` - Retrieve raw IQ data from FPGA

**Characteristics:**
- Hardware-specific (talks to QICK/RFSoC FPGA)
- No loops (single execution = single pulse sequence)
- Gets instantiated fresh for each measurement point
- Takes `cfg` dict with all pulse parameters

### Layer 2: Experiment Classes (Software Loop & Analysis)
**Example:** `LengthRabiGeneralF0g1Experiment(Experiment)`

**Purpose:** Loop over ONE parameter, collect data, analyze, display

**Key Methods:**
```python
def acquire(self, progress=False):
    lengths = start + step * np.arange(expts)
    data = {"xpts": [], "avgi": [], "avgq": []}

    for length in lengths:
        self.cfg.expt.length_placeholder = float(length)
        program = LengthRabiF0g1GeneralProgram(soccfg=self.soccfg, cfg=self.cfg)
        avgi, avgq = program.acquire(...)  # Run on FPGA
        data["avgi"].append(avgi)
        data["avgq"].append(avgq)

    self.data = data
    return data

def analyze(self, data=None):
    analysis = LengthRabiFitting(data, config=self.cfg)
    analysis.analyze()
    self._length_rabi_analysis = analysis
    return data

def display(self, data=None):
    self._length_rabi_analysis.display()

def go(self, save=True, analyze=True, display=True, progress=True):
    data = self.acquire(progress)
    if analyze: self.analyze(data)
    if save: self.save_data(data)
    if display: self.display(data)
```

**Characteristics:**
- **1D sweep** - Loops over ONE parameter (length, time, frequency, gain, etc.)
- Software loop (Python for-loop)
- Each iteration creates new Program instance and runs on FPGA
- Returns data dict with xpts, avgi, avgq, etc.
- Has built-in analysis (calls fitting classes)
- Saves to .h5 file

**Examples:**
- `LengthRabiExperiment` - sweeps pulse length
- `T1Experiment` - sweeps wait time
- `AmplitudeRabiExperiment` - sweeps pulse gain
- `PulseProbeSpectroscopy` - sweeps frequency
- `ErrorAmplificationExperiment` - sweeps parameter (freq, gain, etc.)

### Layer 3: Sequential Wrapper Classes (2D Sweeps)
**Example:** `man_f0g1_class(sequential_base_class)`

**Purpose:** Loop over Experiment multiple times to create 2D data

**The Problem They Solve:**
```
Chevron = Frequency × Length sweep
- Outer loop: Frequency (e.g., 100 points)
- Inner loop: Length (handled by LengthRabiExperiment, e.g., 25 points)
- Result: 100 × 25 = 2500 data points in 2D array
```

**How They Work:**
```python
class man_f0g1_class(sequential_base_class):
    def freq_sweep(self):
        self.initialize_expt_sweep(keys=['freq_sweep'])

        for freq in np.arange(freq_start, freq_stop, freq_step):  # OUTER LOOP
            # Create experiment instance
            run_exp = meas.LengthRabiGeneralF0g1Experiment(...)
            run_exp.cfg.expt['freq'] = freq  # Override frequency

            # Run experiment (this does the INNER LOOP over length)
            run_exp.go(analyze=False, display=False, save=False)

            # Save incrementally to sweep file
            self.save_sweep_data('freq_sweep', freq, run_exp)

            # Optional: Live chevron analysis
            chevron = self.perform_chevron_analysis()

        return chevron

    def run_sweep(self, sweep_experiment_name):
        if sweep_experiment_name == 'length_rabi_f0g1_sweep':
            return self.freq_sweep()
```

**Key Infrastructure Methods (from sequential_base_class):**

1. **`initialize_expt_sweep(keys)`**
   - Creates empty sweep data structure
   - Generates .h5 filename for sweep results

2. **`save_sweep_data(sweep_key, sweep_value, run_exp)`**
   - Appends current experiment's data to sweep file
   - Incrementally saves to disk (important for long sweeps)
   - Structure: `{freq_sweep: [f1, f2, ...], xpts: [[l1, l2, ...]], avgi: [[...], [...]], ...}`

3. **`perform_chevron_analysis()`**
   - Creates ChevronFitting object from current sweep data
   - Runs analysis and displays live plot
   - Updates plot after each frequency point

4. **`map_sequential_cfg_to_experiment()`**
   - Maps sweep config from `experiment_config.yml` to experiment config
   - Copies parameters from `loaded['sweep_name']` to `loaded['experiment_name']`

**Characteristics:**
- **2D sweep** - Outer loop over parameter, inner loop is Experiment's 1D sweep
- Incremental file saving (saves after each outer loop iteration)
- Optional live plotting (chevron updates after each frequency)
- Uses separate config file (`experiment_config.yml`)
- Wrapper dispatch pattern: `run_sweep(name)` → specific method

**Current Wrapper Classes:**
1. `man_f0g1_class` - Chevron for manipulate f0g1 transitions
2. `sidebands_class` - Sideband frequency sweeps
3. `histogram_sweep_class` - JPA/histogram parameter sweeps
4. `MM_DualRailRB` - Randomized benchmarking depth sweeps

---

## Dependency Exposure / Blast Radius

### Files Using sequential_base_class Wrappers:

```bash
# Confirmed imports of sequential_experiment_classes:
measurement_notebooks/single_qubit_autocalibrate.ipynb        ← Main target
measurement_notebooks/multiphoton_calibration.ipynb           ← Secondary
measurement_notebooks/single_qubit_experiment.ipynb           ← Likely testing/dev
```

**Only 3 notebooks import sequential wrapper classes!** (Plus checkpoint files)

### Files Using sequential_experiment.py Functions:

```bash
# No imports found
```

**sequential_experiment.py appears to be UNUSED legacy code** (4006 lines!)

### Conclusion on Exposure:
- ✅ **Low blast radius** - Only 2-3 production notebooks affected
- ✅ **No deep dependencies** - Wrappers are leaf nodes (nothing depends on them)
- ✅ **Self-contained** - All sequential code is in `experiments/sequential_experiment*.py`

---

## Why Did They Write This Wrapper Layer?

### Historical Context (Inference):

**Before sequential_base_class:**
```python
# Manual 2D sweep (tedious, error-prone)
sweep_data = {'freq_sweep': [], 'xpts': None, 'avgi': [], 'avgq': []}

for freq in np.arange(2000, 2010, 0.5):  # 20 frequencies
    expt = meas.LengthRabiExperiment(...)
    expt.cfg.expt.freq = freq
    expt.go(analyze=False, display=False, save=False)

    sweep_data['freq_sweep'].append(freq)
    sweep_data['avgi'].append(expt.data['avgi'])
    sweep_data['avgq'].append(expt.data['avgq'])
    # ... more data collection

    # Manually save to file
    with SlabFile(filename, 'w') as f:
        for key, val in sweep_data.items():
            f[key] = val

# Manually run analysis
chevron = ChevronFitting(...)
chevron.analyze()
```

**After sequential_base_class:**
```python
# Cleaner, reusable
class_for_exp = man_f0g1_class(...)
class_for_exp.loaded['sweep'] = {'freq_start': 2000, 'freq_stop': 2010, ...}
chevron = class_for_exp.run_sweep('length_rabi_f0g1_sweep')
# Infrastructure handles: loop, data collection, incremental save, live plotting
```

### What Value Did It Add?

1. **DRY Principle** - Don't repeat outer loop code
2. **Incremental saving** - Don't lose data if sweep crashes
3. **Live plotting** - See chevron evolve during long sweeps
4. **Config management** - Separate file for sweep parameters
5. **Reusable infrastructure** - `initialize_expt_sweep()`, `save_sweep_data()`

### What Problems Did It Create?

1. **Over-abstraction** - `eval('class_for_exp.run_sweep')` is unnecessary
2. **Config indirection** - Three config sources (device.yml, experiment.yml, .loaded dict)
3. **Dispatch complexity** - `run_sweep(name)` → multiple if/elif branches
4. **Not unified** - Different pattern from rest of codebase
5. **Hard to extend** - Need to create new wrapper class for each experiment type

---

## Current State: CharacterizationRunner vs Sequential Classes

### CharacterizationRunner (New Pattern - 1D Sweeps)
```python
runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.LengthRabiExperiment,
    default_expt_cfg=defaults,
    preprocessor=preproc,
    postprocessor=postproc,
)
result = runner.run(freq=2000, ...)
```

**Advantages:**
- Unified pattern across Qubit Characterization section
- Direct experiment class usage (no wrapper)
- Preprocessor/postprocessor pattern (clean separation)
- Config in code (visible, version-controlled)

**Limitation:**
- Only handles 1D sweeps (Experiment's built-in loop)

### sequential_base_class (Old Pattern - 2D Sweeps)
```python
class_for_exp = man_f0g1_class(...)
class_for_exp.loaded['sweep'] = {...}
chevron = class_for_exp.run_sweep('length_rabi_f0g1_sweep')
```

**Advantages:**
- Handles 2D sweeps
- Incremental file saving (tested, works)
- Live plotting during sweep

**Disadvantages:**
- Different pattern from CharacterizationRunner
- Wrapper indirection
- Config complexity
- Not unified with rest of notebook

---

## The Question: Build On Top or Replace?

### If Build On Top:
Keep sequential_base_class, add thin adapter layer:
```python
# Adapter that wraps sequential classes to look like CharacterizationRunner
class SequentialAdapter:
    def __init__(self, sequential_class, ...):
        self.seq_class = sequential_class(...)

    def run(self, **kwargs):
        # Map kwargs to loaded dict
        # Call seq_class.run_sweep()
        # Return results
```

**Pros:**
- Don't break existing code
- Minimal changes

**Cons:**
- Still have two parallel systems
- Technical debt persists
- Doesn't actually simplify anything

### If Replace:
Build enhanced SweepRunner that does what sequential_base_class does:
```python
runner = SweepRunner(
    station=station,
    ExptClass=meas.LengthRabiExperiment,
    AnalysisClass=ChevronFitting,
    default_expt_cfg=defaults,
    sweep_param='freq',
    preprocessor=preproc,
    postprocessor=postproc,
)
chevron = runner.run(
    sweep_start=2000,
    sweep_stop=2010,
    sweep_step=0.5,
)
```

**Pros:**
- Unified pattern (same as CharacterizationRunner)
- Direct experiment usage
- Clean architecture
- Easy to extend

**Cons:**
- Need to reimplement features (incremental save, live plotting)
- Migration effort for 2-3 notebooks

---

## Recommendation

**REPLACE with phased migration:**

### Phase 1: Feature Parity
Enhance SweepRunner to match sequential_base_class features:
1. ✅ Incremental file saving (already has this)
2. ✅ Analysis class creation (already has this)
3. ✅ Postprocessor pattern (already has this)
4. ❌ **Add:** Flexible analysis class initialization
5. ❌ **Add:** Optional live plotting callback

### Phase 2: Test Drive
Migrate Chevron in single_qubit_autocalibrate.ipynb:
- Compare results with original
- Verify incremental saves work
- Test live plotting (if implemented)

### Phase 3: Full Migration
Migrate remaining uses:
- Error amplification (if needed)
- Sidebands (similar to chevron)
- Multiphoton_calibration.ipynb
- Single_qubit_experiment.ipynb

### Phase 4: Deprecate
- Mark sequential_base_class as deprecated
- Delete sequential_experiment.py (appears unused)

**Why This Works:**
1. **Low blast radius** - Only 2-3 notebooks affected
2. **Clear path** - CharacterizationRunner already works for 1D
3. **Feature parity achievable** - No fundamental blockers
4. **Incremental** - Can pause after Phase 2 if issues found
5. **Clean end state** - One unified pattern throughout

**Timeline Estimate:**
- Phase 1: 2-3 days (SweepRunner enhancements)
- Phase 2: 2-3 days (Test and validate)
- Phase 3: 3-5 days (Migrate all uses)
- Phase 4: 1 day (Cleanup)
- **Total: ~2 weeks**

---

## Key Insight

**sequential_base_class exists to turn 1D Experiments into 2D sweeps.**

It's a reasonable solution to a real problem (avoiding manual outer-loop boilerplate), but it predates the CharacterizationRunner pattern and doesn't fit the unified architecture you're building.

Since the blast radius is small (2-3 notebooks) and the pattern is well-understood, replacing it with an enhanced SweepRunner is the right architectural choice for long-term maintainability.
