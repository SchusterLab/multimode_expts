# Autocalibration Notebook Refactoring Guide

This guide shows how to use the new helper classes to eliminate boilerplate in the autocalibration notebook.

## Setup

At the beginning of your notebook, after setting up `soc`, `expt_path`, `config_path`, `config_thisrun`, and `ds_thisrun`:

```python
from autocalib_helpers import create_experiment_workflow

# Create helper objects
executor, updater, extractor = create_experiment_workflow(
    soc, expt_path, config_path, config_thisrun, ds_thisrun
)
```

## Example Refactorings

### Example 1: Resonator Spectroscopy

**Before (30+ lines):**
```python
def do_res_spec(config_thisrun): 
    rspec = meas.single_qubit.resonator_spectroscopy.ResonatorSpectroscopyExperiment(
        soccfg=soc, path=expt_path, prefix='ResonatorSpectroscopyExperiment', config_file=config_path
    )
    rspec.cfg = AttrDict(deepcopy(config_thisrun))
    span = 1.5
    expts = 150
    start = config_thisrun.device.readout.frequency[0] - span / 2.5
    rspec.cfg.expt = dict(
        start = start,
        step = span / expts,
        expts = 250,
        reps = 500,
        pulse_e = False,
        pulse_f = False,
        pulse_cavity = False,
        cavity_pulse = [4984.373226159381, 800, 2, 0],
        qubit = 0,
    )
    rspec.cfg.device.readout.relax_delay = [50]
    rspec.go(analyze=True, display=True, progress=True, save=True)
    return rspec

def update_res_spec(rspec, config_thisrun):
    config_thisrun.device.readout.frequency = [rspec.data['fit'][0]]
    print('Updated readout frequency!')

if expts_to_run['res_spec']: 
    rspec = do_res_spec(config_thisrun=config_thisrun)
    update_res_spec(rspec, config_thisrun)
    print('Resonator spectroscopy done!')
```

**After (8 lines):**
```python
if expts_to_run['res_spec']:
    span = 1.5
    expts = 250
    start = config_thisrun.device.readout.frequency[0] - span / 2.5
    
    rspec = executor.execute_if(
        condition=True,  # Already checked above
        experiment_class=meas.single_qubit.resonator_spectroscopy.ResonatorSpectroscopyExperiment,
        prefix='ResonatorSpectroscopyExperiment',
        expt_params={
            'start': start,
            'step': span / expts,
            'expts': expts,
            'reps': 500,
            'pulse_e': False,
            'pulse_f': False,
            'pulse_cavity': False,
            'cavity_pulse': [4984.373226159381, 800, 2, 0],
            'qubit': 0,
        },
        relax_delay=50,
        analyze=True,
        display=True
    )
    updater.update_readout_frequency(rspec.data['fit'][0])
    print('Resonator spectroscopy done!')
```

### Example 2: Single Shot

**Before (50+ lines):**
```python
def do_single_shot(
    config_thisrun,
    expt_path,
    config_path,
    qubits=[0],
    reps=5000,
    check_f=False,
    active_reset=True,
    # ... 15 more parameters ...
):
    hstgrm = meas.single_qubit.single_shot.HistogramExperiment(
        soccfg=soc, path=expt_path, prefix='HistogramExperiment', config_file=config_path
    )
    hstgrm.cfg = AttrDict(deepcopy(config_thisrun))
    hstgrm.cfg.expt = {
        'qubits': qubits,
        'reps': reps,
        # ... many more parameters ...
    }
    hstgrm.cfg.device.readout.relax_delay = [relax_delay]
    hstgrm.go(analyze=False, display=False, progress=True, save=True)
    from multimode_expts.fit_display_classes import Histogram
    hist_analysis = Histogram(hstgrm.data, verbose=True, span=None, threshold=None, config=hstgrm.cfg)
    return hstgrm, hist_analysis

def update_single_shot(hist_analysis, config_thisrun):
    hist_analysis.analyze(plot = True)
    fids = hist_analysis.results['fids']
    confusion_matrix = hist_analysis.results['confusion_matrix']
    thresholds_new = hist_analysis.results['thresholds']
    angle = hist_analysis.results['angle']
    config_thisrun.device.readout.phase = [config_thisrun.device.readout.phase[0] + angle]
    config_thisrun.device.readout.threshold = thresholds_new
    config_thisrun.device.readout.threshold_list = [thresholds_new]
    config_thisrun.device.readout.Ie = [np.median(hist_analysis.data['Ie_rot'])]
    config_thisrun.device.readout.Ig = [np.median(hist_analysis.data['Ig_rot'])]
    if hist_analysis.cfg.expt.active_reset:
        config_thisrun.device.readout.confusion_matrix_with_active_reset = confusion_matrix
    else:
        config_thisrun.device.readout.confusion_matrix_without_reset = confusion_matrix
    print('Updated readout!')

hstgrm = None
if expts_to_run['single_shot']: 
    hstgrm = do_single_shot(config_thisrun, expt_path, config_path, 
                            reps = 5000, active_reset=False, relax_delay = 2500)
    update_single_shot(hstgrm[1], config_thisrun)
```

**After (15 lines):**
```python
if expts_to_run['single_shot']:
    hstgrm = executor.execute_if(
        condition=True,
        experiment_class=meas.single_qubit.single_shot.HistogramExperiment,
        prefix='HistogramExperiment',
        expt_params={
            'qubits': [0],
            'reps': 5000,
            'active_reset': False,
            'man_reset': False,
            'storage_reset': False,
            'qubit': 0,
            'pulse_manipulate': False,
            'cavity_freq': 4984.373226159381,
            'cavity_gain': 400,
            'cavity_length': 2,
            'prepulse': False,
            'pre_sweep_pulse': None,
            'gate_based': True,
        },
        relax_delay=2500,
        analyze=False,
        display=False
    )
    
    from multimode_expts.fit_display_classes import Histogram
    hist_analysis = Histogram(hstgrm.data, verbose=True, span=None, threshold=None, config=hstgrm.cfg)
    updater.update_single_shot(hist_analysis)
```

### Example 3: Pulse Probe (GE)

**Before (40+ lines):**
```python
def do_pulse_probe(
    config_thisrun, 
    start=3560,
    step=0.1,
    expts=200,
    reps=2000,
    rounds=1,
    length=1,
    gain=100,
    sigma=0.1,
    qubit=0,
    prepulse=False,
    pre_sweep_pulse=[],
    gate_based=False,
    relax_delay=250,
): 
    qspec = meas.single_qubit.pulse_probe_spectroscopy.PulseProbeSpectroscopyExperiment(
        soccfg=soc, path=expt_path, prefix='PulseProbeSpectroscopyExperiment', 
        config_file=config_file
    )
    qspec.cfg = AttrDict(deepcopy(config_thisrun))
    qspec.cfg.expt = dict(
        qubits = [0],
        start=start,
        step=step,
        expts=expts,
        reps=reps,
        rounds=rounds,
        length=length,
        gain=gain,
        sigma=sigma,
        qubit=qubit,
        prepulse = prepulse, 
        pre_sweep_pulse = pre_sweep_pulse,
        gate_based = gate_based,
    )
    qspec.cfg.device.readout.relax_delay = [relax_delay]
    qspec.go(analyze=True, display=True, progress=True, save=True)
    return qspec

def update_pulse_probe_ge(qspec, config_thisrun):
    config_thisrun.device.qubit.f_ge = [qspec.data['fit_avgi'][2]]
    print('Updated qubit frequency!')

qspec = do_pulse_probe(config_thisrun=config_thisrun, start=3565, step=0.02, expts=500, reps=100)
if expts_to_run['pulse_probe_ge']:
    update_pulse_probe_ge(qspec, config_thisrun)
    print('Pulse probe spectroscopy done!')
```

**After (15 lines):**
```python
qspec = executor.execute_if(
    condition=expts_to_run['pulse_probe_ge'],
    experiment_class=meas.single_qubit.pulse_probe_spectroscopy.PulseProbeSpectroscopyExperiment,
    prefix='PulseProbeSpectroscopyExperiment',
    expt_params={
        'qubits': [0],
        'start': 3565,
        'step': 0.02,
        'expts': 500,
        'reps': 100,
        'rounds': 1,
        'length': 1,
        'gain': 100,
        'sigma': 0.1,
        'qubit': 0,
        'prepulse': False,
        'pre_sweep_pulse': [],
        'gate_based': False,
    },
    relax_delay=300,
    analyze=True,
    display=True
)

if qspec:
    updater.update_qubit_frequency_ge(qspec.data['fit_avgi'][2])
    print('Pulse probe spectroscopy done!')
```

### Example 4: T2 Ramsey

**Before (60+ lines):**
```python
def do_t2_ramsey_ge(
    config_thisrun,
    expt_path,
    config_path,
    pre_sweep_pulse=None,
    post_sweep_pulse=None,
    step_size=soc.cycles2us(8),
    if_ef=False,
    ef_init=True,
    start=0.01,
    expts=201,
    ramsey_freq=3,
    reps=200,
    rounds=1,
    qubits=[0],
    # ... 15 more parameters ...
):
    t2ramsey = meas.single_qubit.t2_ramsey.RamseyExperiment(
        soccfg=soc, path=expt_path, prefix='RamseyExperiment', config_file=config_path
    )
    t2ramsey.cfg = AttrDict(deepcopy(config_thisrun))
    # ... complex logic to set checkEF, qubit_ge_init, etc. ...
    t2ramsey.cfg.expt = {
        'start': start,
        'step': step_size,
        'expts': expts,
        'ramsey_freq': ramsey_freq,
        # ... many more parameters ...
    }
    t2ramsey.cfg.device.readout.relax_delay = [relax_delay]
    t2ramsey.go(analyze=False, display=False, progress=True, save=True)
    from multimode_expts.fit_display_classes import RamseyFitting
    t2ramsey_analysis = RamseyFitting(t2ramsey.data, config=t2ramsey.cfg)
    return t2ramsey_analysis

def update_t2_ramsey_ge(t2ramsey, config_thisrun):
    config_thisrun.device.qubit.f_ge = [config_thisrun.device.qubit.f_ge[0] + min(t2ramsey.data['f_adjust_ramsey_avgi'])]
    print('Updated qubit ge frequency to:', config_thisrun.device.qubit.f_ge[0])

t2ramsey_ge = None
if expts_to_run['t2_ge']:
    t2ramsey_ge = do_t2_ramsey_ge(config_thisrun, expt_path, config_path,
                                  ramsey_freq=0.2, step_size=0.5, expts=100,
                                  pre_sweep_pulse=None, reps=200, active_reset=True, relax_delay=50)
    t2ramsey_ge.analyze()
    t2ramsey_ge.display(title_str='T2_ge')
    update_t2_ramsey_ge(t2ramsey_ge, config_thisrun)
    print('T2 Ramsey done!')
```

**After (20 lines):**
```python
if expts_to_run['t2_ge']:
    t2ramsey = executor.execute_if(
        condition=True,
        experiment_class=meas.single_qubit.t2_ramsey.RamseyExperiment,
        prefix='RamseyExperiment',
        expt_params={
            'start': 0.01,
            'step': 0.5,
            'expts': 100,
            'ramsey_freq': 0.2,
            'reps': 200,
            'rounds': 1,
            'qubits': [0],
            'checkEF': False,
            'user_defined_freq': [False, 3568.2038290468167, 5304, 0.035],
            'f0g1_cavity': 0,
            'normalize': False,
            'active_reset': True,
            'man_reset': False,
            'storage_reset': False,
            'prepulse': False,
            'postpulse': False,
            'pre_active_reset_pulse': False,
            'gate_based': False,
            'advance_phase': 0,
            'echoes': [False, 0],
            'pre_sweep_pulse': None,
            'post_sweep_pulse': None,
        },
        relax_delay=50,
        analyze=False,
        display=False
    )
    
    from multimode_expts.fit_display_classes import RamseyFitting
    t2ramsey_analysis = RamseyFitting(t2ramsey.data, config=t2ramsey.cfg)
    t2ramsey_analysis.analyze()
    t2ramsey_analysis.display(title_str='T2_ge')
    
    # Update frequency
    freq_adjustment = min(t2ramsey_analysis.data['f_adjust_ramsey_avgi'])
    config_thisrun.device.qubit.f_ge = [config_thisrun.device.qubit.f_ge[0] + freq_adjustment]
    print(f'Updated qubit ge frequency to: {config_thisrun.device.qubit.f_ge[0]}')
    print('T2 Ramsey done!')
```

### Example 5: Storage Mode Parameters

**Before (30+ lines repeated in multiple functions):**
```python
def get_storage_mode_parameters(ds_thisrun, config_thisrun, man_mode_no, stor_mode_no):
    stor_name = 'M' + str(man_mode_no) + '-S' + str(stor_mode_no)
    freq = ds_thisrun.get_freq(stor_name)
    gain = ds_thisrun.get_gain(stor_name)
    pi_len = ds_thisrun.get_pi(stor_name)
    h_pi_len = ds_thisrun.get_h_pi(stor_name)
    flux_low_ch = config_thisrun.hw.soc.dacs.flux_low.ch
    flux_high_ch = config_thisrun.hw.soc.dacs.flux_high.ch
    ch = flux_low_ch if freq<1000 else flux_high_ch
    from MM_dual_rail_base import MM_dual_rail_base
    mm_base_dummy = MM_dual_rail_base(config_thisrun, soccfg=soc)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist()
    return freq, gain, pi_len, h_pi_len, ch, prepulse, postpulse
```

**After (1 line):**
```python
params = extractor.get_storage_params(man_mode_no=1, stor_mode_no=1)
# Use params['freq'], params['gain'], params['ch'], params['prepulse'], etc.
```

## Benefits

1. **Reduced code**: ~50-70% reduction in boilerplate
2. **Consistency**: All experiments follow the same pattern
3. **Maintainability**: Changes to experiment setup logic only need to be made in one place
4. **Readability**: Less code means easier to understand what's happening
5. **Type safety**: Helper classes can be extended with type hints and validation

## Migration Strategy

1. Start with simple experiments (resonator spec, single shot)
2. Gradually refactor more complex experiments
3. Keep old functions as wrappers initially for backward compatibility
4. Once all experiments are refactored, remove old wrapper functions

## Additional Improvements

Consider also:
- Creating experiment-specific parameter presets (e.g., `get_ramsey_params()`, `get_rabi_params()`)
- Adding validation for experiment parameters
- Creating a registry of experiments that can be run in sequence
- Adding logging/timing for each experiment step









