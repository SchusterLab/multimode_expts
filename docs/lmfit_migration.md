# lmfit Migration Tracker

Migration from scipy `curve_fit` with positional parameter lists to lmfit with named parameters.

## Status

### Phase 1: Core infrastructure
- [x] `docs/lmfit_migration.md` — this tracking document
- [x] `fitting/models.py` — 8 Model classes with `guess()` methods + `FitResult` wrapper
- [x] `fitting/fitting.py` — Rewritten `fit*` functions returning `FitResult`
- [x] `tests/test_fitting.py` — Unit + real-data regression tests (32 passed, 8 skipped pending data)
- [x] `tests/test_data_config.yaml` — User-provided HDF5 file paths (fill in to enable real-data tests)
- [x] Bug fix: deleted `fitexp_y0fixed` (broken index assignments) — use `ExponentialModel` with `y0` fixed
- [x] Bug fix: removed debug prints from `fitgaussian` and `fithanger`
- [x] Consolidate `expfunc1`/`fitexp1` → `ExponentialModel` with `x0` fixed
- [x] Consolidate `decaysin1`/`fitdecaysin1` → `DecaySineModel` with `x0` fixed

### Phase 2: fit_display_classes.py (future)
- [ ] Verify all `analyze()` methods work unchanged after Phase 1
- [ ] Optionally improve `display()` methods to use named params
- [ ] RamseyFitting
- [ ] AmplitudeRabiFitting
- [ ] Spectroscopy
- [ ] LengthRabiFitting
- [ ] MM_DualRailRBFitting

### Phase 3: Deprecations (future)
- [ ] `fitting/fit_display.py` — Add deprecation warning
- [ ] `slab/dsfit.py` — Add deprecation warning

### Phase 4: Experiment file migration (future, incremental)

#### Tier 1 (single_qubit core)
- [ ] t1.py
- [ ] t2_echo.py
- [ ] t2_ramsey.py
- [ ] amplitude_rabi.py
- [ ] length_rabi.py
- [ ] pulse_probe_spectroscopy.py

#### Tier 2 (qsim)
- [ ] sideband_ramsey.py
- [ ] sideband_scramble.py
- [ ] sequential_experiment.py

#### Tier 3 (qubit_cavity)
- [ ] ecd_read_spec.py
- [ ] amplitude_rabi_f0g1.py
- [ ] displace_amplitude_calibration.py
- [ ] displacement_enhanced_sideband.py
- [ ] out_and_back.py
- [ ] single_mode_cats_wigner_tomography.py

---

## Model Classes (fitting/models.py)

| Model Class | Parameters | Replaces |
|---|---|---|
| `ExponentialModel` | y0, yscale, x0, decay | expfunc/fitexp + expfunc1/fitexp1 (x0 fixed) + expfunc_y0fixed (y0 fixed) |
| `SineModel` | yscale, freq, phase_deg, y0 | sinfunc/fitsin |
| `DecaySineModel` | yscale, freq, phase_deg, decay, y0, x0 | decaysin/fitdecaysin + decaysin1/fitdecaysin1 (x0 fixed) |
| `DecaySineDualRailModel` | yscale, freq, phase_deg, decay, decay_phi, y0, x0_kappa, x0_phi | decaysin_dualrail |
| `TwoFreqDecaySineModel` | yscale0, freq0, phase_deg0, decay0, yscale1, freq1, phase_deg1, y0 | twofreq_decaysin |
| `LorentzianModel` | y0, yscale, x0, xscale | lorfunc/fitlor |
| `GaussianPeakModel` | y0, yscale, x0, sigma | gaussianfunc/fitgaussian |
| `HangerS21SlopedModel` | f0, Qi, Qe, phi, scale, a0, slope | hanger*/fithanger |
| `RBModel` | p, a, b | rb_func/fitrb |

---

## Parameter Name Reference (legacy index → name)

### fitexp / ExponentialModel
| Index | Name | Description |
|---|---|---|
| 0 | y0 | Y offset |
| 1 | yscale | Amplitude |
| 2 | x0 | X offset |
| 3 | decay | Decay time constant |

### fitdecaysin / DecaySineModel
| Index | Name | Description |
|---|---|---|
| 0 | yscale | Oscillation amplitude |
| 1 | freq | Frequency (non-angular, MHz) |
| 2 | phase_deg | Phase in degrees |
| 3 | decay | Decay time constant |
| 4 | y0 | Y offset |
| 5 | x0 | X offset / decay origin |

### fitsin / SineModel
| Index | Name | Description |
|---|---|---|
| 0 | yscale | Amplitude |
| 1 | freq | Frequency |
| 2 | phase_deg | Phase in degrees |
| 3 | y0 | Y offset |

### fitlor / LorentzianModel
| Index | Name | Description |
|---|---|---|
| 0 | y0 | Background offset |
| 1 | yscale | Peak height |
| 2 | x0 | Peak center |
| 3 | xscale | HWHM |

### fitgaussian / GaussianPeakModel
| Index | Name | Description |
|---|---|---|
| 0 | y0 | Background offset |
| 1 | yscale | Peak height |
| 2 | x0 | Peak center |
| 3 | sigma | Width |

### fithanger / HangerS21SlopedModel
| Index | Name | Description |
|---|---|---|
| 0 | f0 | Resonance frequency |
| 1 | Qi | Internal Q |
| 2 | Qe | External Q |
| 3 | phi | Asymmetry phase |
| 4 | scale | Scale |
| 5 | a0 | Amplitude offset |
| 6 | slope | Background slope |

### fitrb / RBModel
| Index | Name | Description |
|---|---|---|
| 0 | p | Depolarizing parameter |
| 1 | a | Amplitude |
| 2 | b | Offset |

---

## Backward Compatibility

All existing code continues to work unchanged:

```python
# Old pattern (still works):
p, pCov = fitter.fitdecaysin(xdata, ydata, fitparams=[...])
fitter.decaysin(xdata, *p)
print(f'T2 = {p[3]}')

# New pattern (additionally available):
result = fitter.fitdecaysin(xdata, ydata)
result['decay']           # named access
result.stderr('decay')    # standard error
```

## New Capabilities After Migration
- Named parameter access: `result['decay']` instead of `p[3]`
- Standard errors: `result.stderr('decay')` instead of `np.sqrt(pCov[3][3])`
- Fix/free parameters: `params['x0'].set(vary=False)` — no more duplicate functions
- Per-parameter bounds: `params['decay'].set(min=0)` — not global bounds tuple
- Expression constraints: `params['y0'].set(expr='yscale / 2')` — derived parameters
- Built-in plotting: `result.lmfit_result.plot()`
- Confidence bands: `result.lmfit_result.eval_uncertainty()`
