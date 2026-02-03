# Mock Mode Architecture Plan

## Problem Statement

Currently, mock mode only partially works:
- ✅ Station correctly creates mock hardware objects (`MockQickConfig`, `MockInstrumentManager`, etc.)
- ✅ Station uses `mock_data/` paths instead of production paths
- ✅ Job server framework (submission, queue, config versioning) works
- ❌ Experiments crash because they inherit from real QICK classes (`RAveragerProgram`, etc.)

The crash happens because:
1. Experiment classes (e.g., `PulseProbeSpectroscopyProgram`) inherit from `MMRAveragerProgram`
2. `MMRAveragerProgram` inherits from real `qick.RAveragerProgram`
3. Real QICK's `__init__` expects complete hardware config (e.g., `tproccfg['type']`)
4. `MockQickConfig` doesn't have all required fields → `KeyError`

This affects both:
- `run_local()` with mock station
- Worker in mock mode

## Design Goals

1. **Station owns mock routing** - Station decides mock vs real, not worker or experiments
2. **Experiments unchanged** - Don't modify every experiment class
3. **Both paths work** - `run_local()` and worker queue should work in mock mode
4. **Testable** - Can test job server framework without hardware

## Proposed Solution: Module-Level QICK Mocking

### Approach

Station's `_initialize_hardware_mock()` patches `sys.modules` with mock QICK classes before experiments are imported.

### Implementation Steps

1. **Create mock QICK base classes** in `experiments/mock_hardware.py`:
   ```python
   class MockQickProgram:
       """Mock base that doesn't require real soccfg."""
       def __init__(self, soccfg, cfg=None):
           self.soccfg = soccfg
           self.cfg = cfg
           # No real QICK initialization

   class MockRAveragerProgram(MockQickProgram):
       """Mock RAveragerProgram that generates simulated data."""
       def acquire(self, soc, **kwargs):
           # Return simulated I/Q data
           return self._generate_mock_data()
   ```

2. **Patch modules in station mock init**:
   ```python
   def _initialize_hardware_mock(self):
       # Patch QICK modules BEFORE any experiment imports
       import sys
       from experiments.mock_hardware import MockRAveragerProgram, ...

       mock_qick = types.ModuleType('qick')
       mock_qick.RAveragerProgram = MockRAveragerProgram
       mock_qick.AveragerProgram = MockAveragerProgram
       # ... etc

       sys.modules['qick'] = mock_qick
       sys.modules['qick.averager_program'] = mock_qick
       # ... then initialize mock hardware objects
   ```

3. **Challenge: Import order**
   - Experiments are often imported at module load time (e.g., `import experiments as meas`)
   - Station is created AFTER imports
   - Solution: Provide a `setup_mock_mode()` function to call BEFORE imports

   ```python
   # In notebook or worker:
   from experiments.mock_hardware import setup_mock_mode
   setup_mock_mode()  # Patches sys.modules

   import experiments as meas  # Now uses mock QICK
   station = MultimodeStation(mock=True)
   ```

4. **Worker changes**:
   ```python
   def main():
       if args.mock:
           from experiments.mock_hardware import setup_mock_mode
           setup_mock_mode()
       # ... rest of worker init
   ```

### Mock Data Generation

The mock `acquire()` should generate realistic-looking data:
- Simulated Rabi oscillations
- Spectroscopy peaks
- Appropriate noise levels

This can be based on experiment type or just generic oscillation data.

## Alternative: Minimal MockQickConfig

Instead of patching modules, make `MockQickConfig` complete enough that real QICK doesn't crash:

```python
def _create_default_cfg(self) -> dict:
    return {
        "tprocs": [{"type": "axis_tproc64x32_x8", "f_time": 384.0, ...}],
        "gens": [{"type": "axis_signal_gen_v6", ...} for _ in range(7)],
        "readouts": [{"tproc_ctrl": 0, ...}],
        # ... all fields QICK expects
    }
```

**Pros:** Simpler, no module patching
**Cons:** Fragile (breaks when QICK updates), still runs real QICK code paths

## Recommendation

Use **Module-Level Mocking** (main solution) because:
- Clean separation of concerns
- Experiments don't need changes
- More robust to QICK library changes
- Can evolve mock behavior independently

## Files to Modify

1. `experiments/mock_hardware.py` - Add mock QICK classes and `setup_mock_mode()`
2. `experiments/station.py` - Call setup in `_initialize_hardware_mock()` or document usage
3. `job_server/worker.py` - Call `setup_mock_mode()` before imports when `--mock`
4. `tests/test_mock_station.py` - Already does module patching, can reuse pattern

## Testing

After implementation:
1. `pixi run python -m job_server.worker --mock` should start without errors
2. Jobs submitted from notebook should complete with mock data
3. `run_local()` with mock station should work
4. Existing tests should still pass
