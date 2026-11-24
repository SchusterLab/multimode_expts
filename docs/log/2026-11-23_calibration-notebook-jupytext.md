# Jonginn calibration notebook Jupytext pairing

At Jonginn's request, paired `Autocalibrate.ipynb` and
`single_qubit_manualcalibrate.ipynb` in `measurement_notebooks/jonginn/` with
same-name Jupytext `py:percent` scripts for reviewing future calibration edits.

The notebook diffs contain only the pairing metadata and retain their saved
outputs. Round-tripping the scripts reproduced all 144 Autocalibrate cells and
all 287 manual calibration cells in order with the same types and source,
allowing for Jupytext's normalization of trailing whitespace.

The subsequent edits to the scripts pass the selected manipulate mode to
one-photon, storage and multiphoton jobs. The manual notebook adds parity
spectroscopy, displacement gain calibration, a self-Kerr Ramsey sweep and
separate cells for individual ladder transitions. Auto runs each selected axis
with the other mode in vacuum. Manual can also select one joint occupation
`(n,m)`. The qubit ge/ef pulses are shared at each occupation. Manual M1 swaps
run only at `(n,0)` and selected M2 swaps take `(n,m)` to `(n,m+1)`. The existing
M1 data remain in `device.multiphoton`.
Other measured transitions use `device.multiphoton_joint[operation][pulse][field]["n,m"]`.
For example, `["M2"]["pi"]["frequency"]` holds the M2 swap frequencies by starting
occupation `"n,m"` (M1 photons, M2 photons). A vacuum f0-g1 fit also updates the corresponding
CSV row and, for M1, the legacy multiphoton config. Scan estimates do not
overwrite unmeasured fields. The pulse and fit paths were checked offline.

At Jonginn's request, the manual selection cell takes separate values for
`initial_m1_photon_number` and `initial_m2_photon_number` and passes them in
M1, M2 order for state preparation and fit storage. The shared save function
takes M1 and M2 counts directly from either notebook. For an axis calibration,
the caller passes zero for the mode in vacuum.
At Jonginn's request, joint preparation fills M1 first, then M2. Each M2
swap with photons in M1 requires a measured frequency and pi length at its starting occupation.
The eight manual cells use the notebook's error amplification, spectroscopy, chevron and
length Rabi runners directly. Accepted fits update the selected transition. Legacy M1
pulses read their original config arrays at the requested photon index.

Explicit M1 Fock preparation also selects M1 when the experiment defaults to M2.
Manual swap length fits save the gain used during the scan. The storage amplitude
chevron inspection cell reads the last measured manipulate and storage pair.

At Jonginn's request, expanded the nested comprehensions and axis lookups in
`multiphoton_table` and named intermediate values by their source and purpose.
The function docstring explains its return indexing and resolution flags.
Removed the flattened field-list collection. The original dict is traversed
directly to track list length and check whether the existing M1 table can be returned.
All 3,456 before/after comparisons and 17 existing offline storage/preparation
checks passed, including M1 object identity and config immutability. The installed
pixi environment was used because pixi 0.63 cannot read the version 7 lock file.

Jonginn chose to keep the current lookup and estimation behavior. Four private
functions in `MM_base.py` now handle photon coordinates, the M2 vacuum swap,
saved values for one pulse field, and estimates for missing values.

Jonginn requested a review of both notebooks after the manual M2 vacuum length
Rabi ran at a different gain from its fine chevron. Both notebooks now reuse
the selected M2 chevron's gain, time grid, averaging, preparation, readout
mapping and relaxation delay. The frequency comes from the current M2 row.
Without a chevron they use the stored M2 gain and a short time sweep. A failed
M2 Rabi fit leaves the station settings intact. Explicit runner overrides still
take precedence.

Manipulate and storage chevrons now take the saved pi length from the fit at
the selected frequency. M2 dump chevrons read the selected dump row's gain.
The manual parity Stark check uses the fast pulse that applies the correction,
and the storage amplitude sweep reads the gain span supplied for that run.
Storage error amplification skips a write if its fitted center is missing or
outside the measured sweep. Offline checks covered separate and combined mode
selection, explicit overrides, accepted and rejected fits, and QICK compilation.
