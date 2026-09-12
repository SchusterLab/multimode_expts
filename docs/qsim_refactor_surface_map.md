# Qsim refactor: one-screen surface map

Source survey: local `main` dd9624f, compared with `guan` 1325e4b, 2026-09-11.
Priority follows Guan's description: many-body Ramsey (MBR) active; literal dark-mode and flux-excursion projects mostly historical. Presence in a notebook does not establish current use or correctness.

| Area | Scientific workflows | Refactor destination / priority |
|---|---|---|
| **1. Prepare and calibrate** | Readout; broadband qubit pulses; N-photon swaps; Floquet frequency, gain and Stark phase | Support active MBR; consolidate only the paths it needs. Floquet gain calibration currently lives in the flux-excursion file. |
| **2. Four core MBR products** | **Phase calibration · orthogonality · spectroscopy · propagator** | Finish these end to end. `guan` has four aggregate classes; acquisition still shares the large pulse base. Diagonal/off-diagonal and photon number are variants, not new families. |
| **3. Spectroscopy campaigns** | Select occupations/channels → acquire disorder realizations → reconstruct spectra → compare levels / pool gap statistics / report plots | The new MBR Experiment classes own series-aware acquisition/analysis orchestration; move that implementation out of cells. Repeated datasets become inputs. |
| **4. Analysis explorations** | Reduced-shot / randomized-occupation replay; alternative pole selection; joint shared-frequency fitting; split-shot reproducibility | Separate experimental estimators from the standard FFT/MPM route. Migrate selected useful methods, not every historical trial by default. |
| **5. Adjacent measurement extensions** | Hamiltonian tomography (from propagators); direct disorder SFF (distinct acquisition); coherent-displacement Floquet Kerr | Explicit support decisions. Preserve their identity; do not silently make all three prerequisites for core MBR. |
| **6. Historical workflows** | Bright/dark basis preparation and readout, multiparity, dark T1 and disorder decay; flux-excursion Ramsey/Kerr and transition scans; Wigner; cooling/debug | Keep recoverable; migrate separately when needed. Extract shared dependencies before isolating old projects. Cooling is Guan's earlier experiment, copied into this notebook; not a new migration target. Wigner activity remains unconfirmed. |

**Shared foundation beneath 1–5:** pulse playback + phase/timing conventions; saved-data loading + provenance + grid validation; reusable numerical routines; normal acquire/analyze/save/display lifecycle.

**Notebook destinations:** {MBR, dark mode, flux excursion} × {measurement, analysis}, preferably Jupytext `.py`; analysis goes under `analysis_notebooks/`. Combine small analysis notebooks if useful. Calibration placement remains open; explicit inputs should make moving cells mechanical. Scratchpads may stay with their neighbors.

**Two stages, within each area:** (1) extract cohesive pieces, preserve behavior and update consumers, aiming for files below ~1 kLOC; (2) improve names, inheritance, duplication and contracts. Choose enough boundaries in stage 1 to avoid hidden globals and circular dependencies. Keep behavioral fixes distinct from moves. Attack order remains undecided.

**Scale:** 680 notebook cells / 23,090 code-cell lines; 8,772-line dark-mode file + 1,503-line flux-excursion file. Rough editing surface: 30–40 kLOC, not unique functionality; do not double-count code already extracted on `guan`. Final file counts and ~200-line modules are aspirations, not commitments.

[Cell-level evidence and gaps](qsim_notebook_surface_inventory.md). This is a source map, not execution coverage or a physics validation.
