# SweepRunner: Origins, Purpose, and Comparison

## TL;DR
**SweepRunner was written ~12 days ago (Dec 22, 2025) in a previous Claude session as a proposed replacement for `sequential_base_class`. It has NOT been tested yet - it's a template/prototype.**

---

## Origins and Context

### When Was It Created?
- **Date:** December 22, 2025 (12 days ago)
- **Commit:** `c627ac9` - "claude suggested refactoring template for 2D sweeps"
- **Context:** Previous refactoring session, after CharacterizationRunner was working well
- **Status:** Added to `meas_utils.py` but never tested/used

### Why Was It Created?
Looking at the notebook, the previous session identified the same issue we're discussing:
1. CharacterizationRunner works great for 1D sweeps (Qubit Characterization section)
2. Manipulate section uses old `sequential_base_class` pattern (different from rest of notebook)
3. Need unified pattern for 2D sweeps to match CharacterizationRunner's design

**SweepRunner was proposed as the unified solution for 2D sweeps.**

### Current State in Notebook
The notebook has a "Claude Proposed Refactoring" section (lines 1802-1902) that:
- ✅ Imports SweepRunner
- ✅ Defines defaults, preprocessor, postprocessor (same pattern as CharacterizationRunner)
- ✅ Creates chevron_runner instance
- ❌ **Has an error traceback** - Never successfully run
- ⚠️ **Status:** "Template only - not yet implemented" (per notebook comment)

So you're seeing a **proof-of-concept** that was never tested or completed.

---

## How SweepRunner Differs from sequential_base_class

### Architectural Philosophy

**sequential_base_class:** Object-oriented wrapper pattern
```python
# Create a class instance that wraps the sweep logic
class_for_exp = man_f0g1_class(soccfg, path, prefix, config_file, ...)

# Set config in special .loaded dict
class_for_exp.loaded['length_rabi_f0g1_sweep'] = {
    'freq_start': 2000,
    'freq_stop': 2010,
    'freq_step': 0.5,
    # ... many params
}

# Dispatch to specific sweep method
result = class_for_exp.run_sweep('length_rabi_f0g1_sweep')
# Internally calls: self.freq_sweep()
```

**SweepRunner:** Functional composition pattern (like CharacterizationRunner)
```python
# Create runner with direct experiment class reference
runner = SweepRunner(
    station=station,
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,  # Direct class
    AnalysisClass=ChevronFitting,
    default_expt_cfg=defaults,  # Config in code
    sweep_param='freq',
    preprocessor=preproc_func,
    postprocessor=postproc_func,
)

# Run with explicit sweep parameters
result = runner.run(
    sweep_start=2000,
    sweep_stop=2010,
    sweep_step=0.5,
)
```

---

## Side-by-Side Comparison

### 1. Config Management

**sequential_base_class:**
```python
# THREE config sources:
1. device.yml (hardware config)
2. experiment_config.yml (sweep parameters)
3. .loaded dict (runtime overrides)

# Indirection:
class_for_exp.loaded['sweep_name'] = {...}  # Set here
# ... later in code ...
class_for_exp.map_sequential_cfg_to_experiment()  # Maps sweep → experiment
# ... inside freq_sweep() ...
run_exp.cfg.expt = self.loaded['ExperimentName']  # Use here
```

**SweepRunner:**
```python
# ONE config source:
defaults = AttrDict(dict(
    start=2, step=0.1, expts=25, reps=100,
    gain=8000, # ... all parameters visible in code
))

# Direct:
expt_cfg = preprocessor(station, defaults, **kwargs)
expt.cfg.expt = expt_cfg  # Set directly
```

**Winner:** SweepRunner - simpler, visible, version-controlled

---

### 2. Experiment Class Usage

**sequential_base_class:**
```python
# Wrapper class per experiment type
class man_f0g1_class(sequential_base_class):
    def __init__(...):
        self.experiment_class = 'single_qubit.length_rabi_f0g1_general'
        self.experiment_name = 'LengthRabiGeneralF0g1Experiment'

    def freq_sweep(self):
        # Uses eval() to create experiment
        run_exp = eval(f"meas.{self.experiment_class}.{self.experiment_name}(...)")

# Dispatch mechanism
def run_sweep(self, sweep_experiment_name):
    if sweep_experiment_name == 'length_rabi_f0g1_sweep':
        return self.freq_sweep()
```

**SweepRunner:**
```python
# Direct experiment class reference
runner = SweepRunner(
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,  # Direct import
)

# Inside run():
expt = self.ExptClass(soccfg=..., path=..., ...)  # Direct instantiation
```

**Winner:** SweepRunner - no eval(), no indirection, no wrapper classes needed

---

### 3. Sweep Loop Logic

**sequential_base_class:**
```python
def freq_sweep(self):
    self.initialize_expt_sweep(keys=['freq_sweep'])  # Setup

    for freq in np.arange(freq_start, freq_stop, freq_step):
        run_exp = meas.LengthRabiExperiment(...)
        run_exp.cfg.expt['freq'] = freq
        run_exp.go(analyze=False, display=False, save=False)

        self.save_sweep_data('freq_sweep', freq, run_exp)  # Incremental save
        chevron = self.perform_chevron_analysis()  # Live plotting

    return chevron
```

**SweepRunner:**
```python
def run(self, sweep_start, sweep_stop, sweep_step):
    sweep_vals = np.arange(sweep_start, sweep_stop + sweep_step/2, sweep_step)
    sweep_data = {'freq_sweep': [], 'xpts': None, 'avgi': [], ...}

    for sweep_val in sweep_vals:
        expt = self.ExptClass(...)
        expt.cfg.expt[self.sweep_param] = sweep_val
        expt.go(analyze=False, display=False, save=False)

        # Collect data
        sweep_data['freq_sweep'].append(sweep_val)
        sweep_data['avgi'].append(expt.data['avgi'])
        # ... etc

    # Save to file (at end, not incremental)
    with SlabFile(filename, 'w') as f:
        for key, val in sweep_data.items():
            f[key] = val

    # Create analysis object
    analysis = self.AnalysisClass(...)
    analysis.analyze()

    return analysis
```

**Winner:** Tie - similar logic, but SweepRunner is more generic

---

### 4. Live Plotting

**sequential_base_class:**
```python
def perform_chevron_analysis(self):
    # Called after each frequency point
    chevron = ChevronFitting(
        frequencies=self.expt_sweep.data['freq_sweep'],
        time=self.expt_sweep.data['xpts'][0],
        response_matrix=np.array(self.expt_sweep.data['avgi'])
    )
    chevron.analyze()
    chevron.display()  # Updates plot live
    IPython.display.clear_output(wait=True)
    return chevron

# In freq_sweep():
for freq in frequencies:
    # ... run experiment ...
    chevron = self.perform_chevron_analysis()  # ← Live update
```

**SweepRunner:**
```python
# NOT IMPLEMENTED
# Would need to add:
# 1. Optional live_analysis_fn callback
# 2. Call it after each sweep point
```

**Winner:** sequential_base_class - has working live plotting

---

### 5. Incremental Saving

**sequential_base_class:**
```python
def save_sweep_data(self, sweep_key, sweep_value, run_exp):
    self.expt_sweep.data[sweep_key].append(sweep_value)
    for data_key in run_exp.data.keys():
        self.expt_sweep.data[data_key].append(run_exp.data[data_key])
    self.expt_sweep.save_data()  # ← Saves to disk after EACH point
```

**SweepRunner:**
```python
# Saves at END of sweep
for sweep_val in sweep_vals:
    # ... run and collect data in memory ...

# After loop completes:
with SlabFile(filename, 'w') as f:  # ← Saves ONCE at end
    for key, val in sweep_data.items():
        f[key] = val
```

**Winner:** sequential_base_class - safer (don't lose data if crash)

---

### 6. Analysis Pattern

**sequential_base_class:**
```python
# Separate update function (manual)
def update_length_rabi_f0g1_sweep(expt_path, prefix, config_thisrun):
    # Load data from file
    temp_data, attrs, filename = station.load_data(expt_path, prefix)

    # Create analysis
    chevron = ChevronFitting(
        frequencies=temp_data['freq_sweep'],
        time=temp_data['xpts'][0],
        response_matrix=temp_data['avgi'],
    )
    chevron.analyze()

    # Update config manually
    station.ds_thisrun.update_freq('M1', chevron.results['best_frequency'])

    return chevron

# Usage:
result = do_length_rabi_f0g1_sweep(...)
chevron = update_length_rabi_f0g1_sweep(...)
```

**SweepRunner:**
```python
# Integrated postprocessor
def chevron_postproc(station, analysis):
    analysis.display_results(save_fig=True)
    station.ds_thisrun.update_freq('M1', analysis.results['best_frequency'])

runner = SweepRunner(
    ...,
    AnalysisClass=ChevronFitting,
    postprocessor=chevron_postproc,
)

# Usage:
chevron = runner.run(...)  # Analysis and config update happen automatically
```

**Winner:** SweepRunner - integrated, automatic, follows CharacterizationRunner pattern

---

### 7. Extensibility

**sequential_base_class:**
```python
# To add new sweep type:
1. Create new wrapper class (e.g., new_sweep_class)
2. Inherit from sequential_base_class
3. Define experiment_class and experiment_name
4. Write sweep method (e.g., def my_sweep(self):)
5. Add dispatch in run_sweep()
6. Update experiment_config.yml with new sweep params
```

**SweepRunner:**
```python
# To add new sweep type:
1. Define defaults dict
2. Define preprocessor (if needed)
3. Define postprocessor (if needed)
4. Create runner instance

# That's it - no new classes needed
```

**Winner:** SweepRunner - much simpler to extend

---

## Feature Comparison Table

| Feature | sequential_base_class | SweepRunner | Winner |
|---------|----------------------|-------------|--------|
| **Config management** | 3 sources (device.yml, experiment.yml, .loaded) | 1 source (defaults dict) | SweepRunner |
| **Experiment instantiation** | eval() + string paths | Direct class reference | SweepRunner |
| **Pattern consistency** | Different from CharacterizationRunner | Same as CharacterizationRunner | SweepRunner |
| **Live plotting** | ✅ Working | ❌ Not implemented | sequential_base_class |
| **Incremental saving** | ✅ After each point | ❌ Only at end | sequential_base_class |
| **Analysis integration** | ❌ Separate update function | ✅ Integrated postprocessor | SweepRunner |
| **Code complexity** | High (wrapper classes, dispatch, eval) | Low (functional composition) | SweepRunner |
| **Extensibility** | Need new class per sweep | Just create new runner | SweepRunner |
| **Tested/Working** | ✅ Battle-tested | ❌ Untested prototype | sequential_base_class |

---

## What Would It Take to Use SweepRunner?

### Current Gaps (from prototype):

1. **❌ Incremental saving** - Currently saves at end, should save after each point
   - **Fix:** Add `self.expt_sweep.save_data()` inside loop (copy from sequential_base_class)

2. **❌ Live plotting** - Not implemented
   - **Fix:** Add optional `live_analysis_fn` parameter, call after each point

3. **❌ Analysis class initialization** - Hardcoded to expect specific constructor
   ```python
   # Line 649 in current SweepRunner:
   analysis = self.AnalysisClass(
       frequencies=sweep_data[f'{self.sweep_param}_sweep'],  # ← Hardcoded
       time=sweep_data['xpts'],  # ← Hardcoded
       response_matrix=sweep_data['avgi'],  # ← Hardcoded
       config=self.station.config_thisrun,
       station=self.station,
   )
   ```
   - **Fix:** Make constructor args configurable or use factory function

4. **❌ Never been tested** - Unknown bugs likely
   - **Fix:** Test with real Chevron experiment

---

## Summary: What You're Looking At

**SweepRunner is a ~12-day-old prototype** created in a previous refactoring session as a **proposed replacement** for `sequential_base_class`. It was designed to:
1. Match CharacterizationRunner's pattern (unified architecture)
2. Eliminate wrapper classes and config indirection
3. Simplify 2D sweep implementation

**It has NEVER been tested or used in production.**

The previous session got as far as:
- ✅ Writing the SweepRunner class
- ✅ Adding it to meas_utils.py
- ✅ Creating example usage in notebook
- ❌ Actually running/testing it
- ❌ Fixing the issues discovered

**You're now at the decision point the previous session left off:**
- Do we finish SweepRunner and replace sequential_base_class?
- Or do we keep sequential_base_class and build on top?

---

## My Assessment

**SweepRunner is ~80% complete** and heading in the right direction:
- ✅ Core loop logic correct
- ✅ Analysis pattern better than sequential_base_class
- ✅ Architectural pattern matches CharacterizationRunner
- ❌ Missing incremental save (critical for long sweeps)
- ❌ Missing live plotting (nice-to-have)
- ❌ Hardcoded analysis constructor (fixable)
- ❌ Untested (needs validation)

**Effort to complete:** 2-3 days to add missing features + test
**Effort to keep sequential_base_class:** 0 days (it works)
**Long-term benefit of SweepRunner:** Unified architecture, easier to maintain

Given that you're already refactoring and the blast radius is small (2-3 notebooks), **finishing SweepRunner is the better investment**.
