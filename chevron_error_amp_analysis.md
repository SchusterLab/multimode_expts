# Analysis: Chevron & Error Amplification Refactoring

## Current Architecture Comparison

### Original Pattern (do_ / update_ functions)

**Chevron (Length Rabi F0g1 Sweep):**
```python
def do_length_rabi_f0g1_sweep(config_thisrun, expt_path, config_path, freq_start, freq_stop, freq_step):
    # 1. Create sequential experiment class wrapper
    class_for_exp = man_f0g1_class(soccfg, path, prefix, config_file, exp_param_file, config_thisrun)

    # 2. Set sweep parameters in loaded dict
    class_for_exp.loaded['length_rabi_f0g1_sweep'] = {
        'freq_start': freq_start, 'freq_stop': freq_stop, 'freq_step': freq_step,
        'start': 2, 'step': 0.1, 'expts': 25, 'reps': 100,
        # ... many other params
    }

    # 3. Call run_sweep which dispatches to freq_sweep()
    return class_for_exp.run_sweep(sweep_experiment_name='length_rabi_f0g1_sweep')

def update_length_rabi_f0g1_sweep(expt_path, prefix, config_thisrun, man_mode_no=1):
    # 1. Load sweep data from file
    temp_data, attrs, filename = station.load_data(expt_path, prefix=prefix)

    # 2. Create analysis object
    chevron_analysis = ChevronFitting(
        frequencies=temp_data['freq_sweep'],
        time=temp_data['xpts'][0],
        response_matrix=temp_data['avgi'],
        config=config_thisrun,
        station=station,
    )

    # 3. Run analysis
    chevron_analysis.analyze()

    # 4. Display results
    chevron_analysis.display_results(save_fig=True, title=f'M{man_mode_no}_{timestamp}')

    # 5. Update config
    station.ds_thisrun.update_freq('M' + str(man_mode_no), chevron_analysis.results['best_frequency_contrast'])
    config_thisrun.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = chevron_analysis.results['best_frequency_contrast']

    return chevron_analysis
```

**Error Amplification:**
```python
def do_error_amplification(config_thisrun, expt_path, config_path, reps, rounds, qubit,
                            n_pulses, start, expts, step, parameter_to_test, pulse_type, ...):
    # 1. Build expt_cfg dict manually
    expt_cfg = {
        'reps': reps, 'qubit': qubit, 'qubits': [qubit],
        'start': start, 'expts': expts, 'step': step,
        'n_pulses': n_pulses, 'pulse_type': pulse_type,
        'parameter_to_test': parameter_to_test, 'rounds': rounds,
    }

    # 2. Create experiment directly (no wrapper class)
    error_amp_exp = meas.ErrorAmplificationExperiment(
        soccfg=station.soc, path=expt_path,
        prefix='ErrorAmplificationExperiment', config_file=config_path
    )

    # 3. Set config
    error_amp_exp.cfg = AttrDict(deepcopy(config_thisrun))
    error_amp_exp.cfg.expt = expt_cfg
    error_amp_exp.cfg.expt.relax_delay = relax_delay

    # 4. Run
    error_amp_exp.go(analyze=False, display=False, progress=True, save=True)

    return error_amp_exp

# NO update function exists in original code - analysis is manual
```

### What man_f0g1_class.freq_sweep() Actually Does

```python
def freq_sweep(self):
    # 1. Initialize sweep data structure
    self.initialize_expt_sweep(keys=['freq_sweep'])

    # 2. Loop over frequencies
    for index, freq in enumerate(np.arange(freq_start, freq_stop, freq_step)):
        # 3. Create single experiment instance
        run_exp = meas.LengthRabiGeneralF0g1Experiment(
            soccfg=self.soccfg, path=self.path, prefix=self.prefix, config_file=self.config_file
        )

        # 4. Set config with current freq
        run_exp.cfg = config_thisrun (if provided)
        run_exp.cfg.expt = self.loaded['LengthRabiGeneralF0g1Experiment']
        run_exp.cfg.expt['freq'] = freq  # Override frequency for this point

        # 5. Run experiment (no save, no analyze, no display)
        run_exp.go(analyze=False, display=False, progress=False, save=False)

        # 6. Incrementally save to sweep file
        self.save_sweep_data('freq_sweep', freq, run_exp)

        # 7. Live chevron analysis and plotting (optional)
        chevron = self.perform_chevron_analysis()

    return chevron
```

---

## CharacterizationRunner vs SweepRunner

### CharacterizationRunner (Working Pattern)
- **Use case:** Single 1D sweep experiment
- **Flow:** Create expt → Run once → Analyze → Postprocess → Return expt
- **Data:** Stored in `expt.data` after `.go()`
- **Analysis:** Built into experiment's `.go(analyze=True)`
- **Postprocessor signature:** `postprocessor(station, expt)`
- **Returns:** Experiment object

### SweepRunner (Proposed Pattern)
- **Use case:** 2D sweep (loop over parameter, run 1D expt at each point)
- **Flow:** Loop { Create expt → Run → Save to file } → Load file → Create Analysis → Postprocess
- **Data:** Saved incrementally to .h5 file during sweep
- **Analysis:** Separate analysis class (e.g., ChevronFitting) after sweep completes
- **Postprocessor signature:** `postprocessor(station, analysis)`
- **Returns:** Analysis object (e.g., ChevronFitting)

---

## What Fits in Existing Framework

### ✅ FITS: Chevron with SweepRunner

**Why it fits:**
1. **2D sweep structure:** Frequency × Time sweep maps perfectly to SweepRunner's loop
2. **Analysis class exists:** ChevronFitting is already a separate class
3. **Postprocessor pattern matches:** Takes analysis object, extracts results, updates config
4. **Incremental file saving:** SweepRunner handles this (matches sequential_base_class behavior)

**Current SweepRunner implementation handles:**
- ✅ Looping over sweep parameter
- ✅ Creating experiment at each point
- ✅ Setting sweep parameter in cfg.expt
- ✅ Running with analyze=False, save=False
- ✅ Collecting data incrementally
- ✅ Saving to .h5 file
- ✅ Creating analysis object from file data
- ✅ Running analysis.analyze()
- ✅ Calling postprocessor with (station, analysis)

**What's Missing:**
- ⚠️ **Live plotting during sweep:** sequential_base_class calls `perform_chevron_analysis()` after each point for live updates
- ⚠️ **Flexible analysis class initialization:** Currently hardcoded to expect (frequencies, time, response_matrix, config, station) - may not fit all analysis classes
- ⚠️ **Support for multiple sweep parameters:** Currently only handles single parameter (though this is OK for chevron)

### ✅ PARTIALLY FITS: Error Amplification

**Why it partially fits:**
1. **Has a sweep:** Sweeps over parameter (frequency, gain, etc.)
2. **No wrapper class needed:** Uses experiment directly (simpler than chevron)
3. **No existing analysis class:** Would need to create one OR handle differently

**What DOESN'T fit:**
- ❌ **No standard analysis class:** Unlike chevron (which has ChevronFitting), error amplification doesn't have a dedicated analysis class in the codebase
- ❌ **No update function exists:** In the original code, error amp is run but results aren't automatically analyzed/updated (manual inspection)
- ⚠️ **Analysis requirements unclear:** What analysis should be done? Peak finding? Threshold detection?

**Options for Error Amplification:**
1. **Option A:** Create ErrorAmplificationAnalysis class (like ChevronFitting) → Use SweepRunner
2. **Option B:** Use CharacterizationRunner if it's actually a 1D sweep (not 2D)
3. **Option C:** Keep as do_ function if no automatic analysis is needed (just visual inspection)

---

## Critical Differences: Sequential Wrapper Class vs Direct Experiment

### Chevron uses `man_f0g1_class` (Sequential Wrapper)
```python
class_for_exp = man_f0g1_class(...)  # Wrapper class
class_for_exp.loaded[sweep_name] = {...}  # Config goes in .loaded dict
class_for_exp.run_sweep(sweep_name)  # Dispatches to freq_sweep()
```

**Why wrapper exists:**
- Historical: Predates CharacterizationRunner
- Convenience: Maps sweep config to experiment config via `map_sequential_cfg_to_experiment()`
- Live plotting: Has `perform_chevron_analysis()` built in

**SweepRunner bypasses this:**
- Creates experiment directly: `meas.LengthRabiGeneralF0g1Experiment(...)`
- Sets config directly: `expt.cfg.expt = preprocessor(...)`
- No need for wrapper's dispatch logic

### Error Amplification uses Direct Experiment
```python
error_amp_exp = meas.ErrorAmplificationExperiment(...)  # Direct
error_amp_exp.cfg.expt = expt_cfg  # Direct config setting
error_amp_exp.go(...)  # Direct run
```

**Already matches SweepRunner's pattern** (if we add sweep loop)

---

## Key Architectural Questions

### 1. What does SweepRunner give us that sequential_base_class doesn't?

**SweepRunner advantages:**
- ✅ Unified with CharacterizationRunner pattern (preprocessor, postprocessor, defaults)
- ✅ No need for wrapper classes (man_f0g1_class, sidebands_class, etc.)
- ✅ No need for experiment_config.yml (separate config file for sweep params)
- ✅ Direct experiment class usage (cleaner)
- ✅ Consistent with rest of refactored notebook

**sequential_base_class advantages:**
- ✅ Live plotting during sweep (perform_chevron_analysis)
- ✅ Already exists and works
- ✅ Handles complex sweep types (RB depth sweeps, 2D sweeps)

### 2. Can SweepRunner replace sequential_base_class entirely?

**For Chevron: YES**
- Same loop structure
- Same data saving pattern
- Analysis object pattern fits perfectly

**For Error Amplification: MAYBE**
- Need to decide on analysis approach first
- If no analysis needed → just use do_ function
- If analysis needed → create analysis class → use SweepRunner

**For other sweeps (Sidebands, RB): UNCLEAR**
- Would need to examine each case
- Some may have unique requirements

### 3. What about live plotting?

**SweepRunner currently:** No live plotting (could add as optional feature)

**Options:**
1. **Add to SweepRunner:** Optional `live_analysis_class` parameter, call after each point
2. **Skip for now:** Live plotting is nice-to-have, not critical for autocalibration
3. **Post-sweep plotting:** Load file and plot at end (what update_ function does)

---

## Gap Analysis: What SweepRunner Needs

### Current Implementation Gaps

1. **Analysis Class Flexibility**
   - Hardcoded constructor signature: `(frequencies, time, response_matrix, config, station)`
   - Won't work if analysis class needs different arguments
   - **Solution:** Add `analysis_constructor_fn` parameter OR make analysis class initialization more generic

2. **Data Key Mapping**
   - Hardcoded: `frequencies=sweep_data[f'{sweep_param}_sweep']`
   - Assumes sweep parameter is called "frequencies" in analysis class
   - **Solution:** Add `sweep_param_name_in_analysis` parameter

3. **Live Plotting**
   - Not implemented
   - **Solution:** Add optional `live_analysis_fn(sweep_data_so_far)` callback

4. **Multiple Sweep Parameters**
   - Currently only handles single parameter sweeps
   - **Solution:** Extend to 2D sweeps (nested loops) if needed

5. **Return Value**
   - Returns analysis object, not experiment
   - Consistent with update_ functions but different from CharacterizationRunner
   - **Decision:** Is this OK? (I think YES for sweeps)

---

## Recommended Approach

### For Chevron (Frequency Sweep)

**RECOMMENDED: Use SweepRunner with minor enhancements**

```python
# Config (same as before)
chevron_defaults = AttrDict(dict(
    start=2, step=0.1, expts=25, reps=100, rounds=1,
    gain=8000, qubits=[0], # ... etc
))

def chevron_preproc(station, default_expt_cfg, man_mode_no=1, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    # Smart config logic if needed
    return expt_cfg

def chevron_postproc(station, analysis):
    freq = analysis.results['best_frequency_contrast']
    station.ds_thisrun.update_freq(f'M{man_mode_no}', freq)
    station.config_thisrun.device.multiphoton['pi']['fn-gn+1']['frequency'][0] = freq
    print(f'Updated M{man_mode_no} frequency to {freq}')

# Runner
chevron_runner = SweepRunner(
    station=station,
    ExptClass=meas.LengthRabiGeneralF0g1Experiment,
    AnalysisClass=ChevronFitting,
    default_expt_cfg=chevron_defaults,
    sweep_param='freq',
    preprocessor=chevron_preproc,
    postprocessor=chevron_postproc,
)

# Execute
if expts_to_run['man_chevron']:
    for man_mode_no in expts_to_run['man_modes']:
        analysis = chevron_runner.run(
            sweep_start=freq_start,
            sweep_stop=freq_stop,
            sweep_step=freq_step,
            man_mode_no=man_mode_no,
            postprocess=True,
        )
```

**What needs fixing in SweepRunner:**
1. Analysis class initialization is too rigid (see line 649-655 in meas_utils.py)
2. Need to handle variable analysis class constructors

### For Error Amplification

**OPTIONS:**

**Option A: Create analysis class + use SweepRunner**
```python
# First create fitting/error_amplification_analysis.py with ErrorAmplificationAnalysis class
# Then use SweepRunner similar to chevron
```

**Option B: Use CharacterizationRunner (if it's 1D sweep, not 2D)**
```python
# If error amplification is actually a simple 1D sweep with built-in analysis:
erroramp_runner = CharacterizationRunner(
    station=station,
    ExptClass=meas.ErrorAmplificationExperiment,
    default_expt_cfg=erroramp_defaults,
    preprocessor=erroramp_preproc,
    postprocessor=erroramp_postproc,
)
```

**Option C: Keep as do_ function (if manual inspection is the workflow)**
```python
# If error amplification results are manually inspected, not auto-processed:
# Keep existing do_error_amplification() function
# No update function needed
```

**NEED TO DECIDE:** What is error amplification's role in the calibration workflow?
- Is it auto-analyzed to extract a result and update config?
- Or is it run for visual inspection only?

---

## Summary Table

| Aspect | Chevron (Freq Sweep) | Error Amplification |
|--------|----------------------|---------------------|
| **Current Pattern** | do_ + update_ with sequential wrapper class | do_ only, no update_ |
| **Experiment Type** | 2D sweep (freq × time) | 1D sweep (param sweep) |
| **Analysis Class** | ChevronFitting (exists) | None (need to create?) |
| **Auto-update config?** | Yes (frequency) | Unknown (manual inspection?) |
| **Fits SweepRunner?** | ✅ YES (with minor fixes) | ⚠️ MAYBE (depends on analysis needs) |
| **Fits CharacterizationRunner?** | ❌ NO (2D sweep) | ✅ MAYBE (if 1D sweep) |
| **Live Plotting?** | Used in sequential class | Not used |
| **Recommended Approach** | SweepRunner + fix analysis init | **Decide analysis workflow first** |

---

## Next Steps

1. **Decide on Error Amplification workflow:**
   - What analysis is needed?
   - Should it auto-update config?
   - Create analysis class or keep manual?

2. **Fix SweepRunner analysis initialization:**
   - Make analysis class constructor more flexible
   - Support different parameter names

3. **Test Chevron refactoring:**
   - Implement using SweepRunner
   - Verify results match original
   - Confirm config updates work

4. **Consider live plotting:**
   - Is it needed for autocalibration?
   - If yes, add to SweepRunner
   - If no, skip for now
