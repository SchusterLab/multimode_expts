Status: historical user labels; disorder planning superseded by the full-basis RMS flow (2026-10-05, issue 7). Class names below predate the final Program tree renames.

# Notebook Program Labeling

# 1. `floquet_calibration.py`

## Floquet pulse calibrations

### Gain chevron: EXPECTED TO WORK

- **Exp:** `FloquetGainChevronExperiment`
- **Prog:** `FloquetGainChevronProgram`
- EXPECTED TO WORK

### Error amplification / Fine: EXPECTED TO WORK

- **Exp:** `ErrorAmplificationExperiment`
- **Prog:** `ErrorAmplificationProgram`
- EXPECTED TO WORK

### Phase accumulation: EXPECTED TO WORK

- **Exp:** `SidebandStarkAmplificationExperiment`
- **Prog:** `SidebandStarkAmplificationProgram`
- EXPECTED TO WORK

### Bare readout check: EXPECTED TO WORK

- **Exp:** `QsimBaseExperiment`
- **Prog:** `SidebandScrambleDarkProgramNewNew`
- EXPECTED TO WORK
- Note: In the legacy notebook, it was in a dark mote readout section, but it was integrated with the calibration nb in the refactoring process. I think this is good.

# 2. `floquet_displacement_kerr.py`

## Floquet-pulse Kerr from coherent displacement

- **Exp**: `FloquetDisplacementKerrExperiment`
- **Prog:** `FloquetDisplacementKerrProgram`
- NOT EXPECTED TO WORK
- Note: might be useful if the floquet pulse length is reduced, but not sure.

# 3. `multiphoton_calibration.py`

## Broadband qubit ge calibration

### Amplitude Rabi cell after Resetting Config files

- **Exp:** `AmplitudeRabiExperiment`
- **Prog:** `AmplitudeRabiProgram`
- EXPECTED TO WORK

### Error Amplification > Coarse / Fine

- **Exp:** `ErrorAmplificationExperiment`
- **Prog:** `ErrorAmplificationProgram`
- EXPECTED TO WORK

### Validation

- **Exp:** `ErrorAmplificationExperiment`
- **Prog:** `ErrorAmplificationProgram`
- EXPECTED TO WORK
- Note: not used often, but thought would be useful.

## N-photon M1-storage full-swap calibration

### 3. Frequency-length Chevron

- **Exp:** `SidebandGeneralExperiment`
- **Prog:** `SidebandGeneralProgram`
- NOT EXPECTED TO WORK
- Note: This was entirely coded by Agent, which was intended to do pulse shape calibration for multiphoton swaps

### 4. Error amplification

- **Exp:** `ErrorAmplificationExperiment`
- **Prog:** `ErrorAmplificationProgram`
- NOT EXPECTED TO WORK
- Note: `ErrorAmplificationProgram` should work, but its application to this section is NOT expected to work.

### 5. Validation — Odd

- **Exp:** `SidebandGeneralExperiment`
- **Prog:** `SidebandGeneralProgram`
- NOT EXPECTED TO WORK

### 5. Validation — Even

- **Exp:** `ErrorAmplificationExperiment`
- **Prog:** `ErrorAmplificationProgram`
- NOT EXPECTED TO WORK
- Note: `ErrorAmplificationProgram` should work, but its application to this section is NOT expected to work.

## Single shot

- **Exp:** `HistogramExperiment` (`ss_runner`)
- **Prog:** `HistogramProgram`
- EXPECTED TO WORK

# 4. `mbr.py`

## 1. Phase calibration

- **Exp:** `MBRStarkCalExperiment`
- **Prog:** `MBRStarkCalProgram`
- EXPECTED TO WORK

## **2. Zero-cycle encoder/decoder orthogonality**

- **Exp:** `MBROrthoColumnExperiment`, `MBROrthogonalityExperiment`
- **Prog:** `MBROrthoColumnProgram`
- EXPECTED TO WORK

## 3. Diagonal spectroscopy of a fixed-N sector

- **Exp:** `MBRTimeTraceExperiment`, `MBRSpectrumExperiment`
- **Prog:** `MBRTimeTraceProgram`
- EXPECTED TO WORK

## 4. Selected occupations

- **Exp:** `MBRTimeTraceExperiment` and `MBRSpectrumExperiment`
- **Prog:** `MBRTimeTraceProgram`
- EXPECTED TO WORK

## 5. Propagator

- **Exp:** `MBROrthoColumnExperiment` and other related refactored dependents
- **Prog:** `MBROrthoColumnProgram`
- EXPECTED TO WORK

# 5. `mbr_disorder.py`

### Phase calibration

- EXPECTED TO WORK

### 7-1b. Build the theory-selected plan and check time — no jobs

- NOT EXPECTED TO WORK
- Note: As discussed, selecting fock states based on theory is not desired, so this cell should not be retained.
- When I ran notebooks in main branch recently, I did not run this cell as well.
- But might be useful in future, so I did not label it as deprecated.

### 7-1c. Acquire every realization — submits jobs

- **Exp:** `MBRTimeTraceExperiment`
- **Prog:** `MBRTimeTraceProgram`
- EXPECTED TO WORK

### 7-1d. Matrix-Pencil analysis and theory matching — no jobs

- EXPECTED TO WORK
- Note: Might need an elaboration on matching algorithm.

### **7-1e. Pooled level statistics, and save — no jobs**

- EXPECTED TO WORK
- Note: Might need an elaboration on matching algorithm.

# 6. `mbr_tomography.py`

### **8-2. Acquire the three matrices**

- **Exp:** `MBROrthoColumnExperiment`
- **Prog:** `MBROrthoColumnProgram`
- EXPECTED TO WORK
- Note: Considered labeling it as NOT EXPECTED TO WORK, but reverted for it could be useful for some reason. The Hamiltonian tomography did not go quite well even for N =1

---

Other unlabeled one should work.