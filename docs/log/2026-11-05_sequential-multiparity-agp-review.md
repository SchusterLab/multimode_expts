# Sequential multiparity ownership and AGP review

The user clarified that the new `multiparity()` must define its own pulse and
measurement sequence so it can later support higher parity. Delegating that sequence
to legacy multiparity and measurement wrappers did not meet that requirement.

- Updated only the new sequential source module: both Ramsey stages, first-bit
  conditional phase, both measurements, qubit feedback and terminal relaxation are
  defined locally. A dedicated bit-measurement helper shares the repeated feedback code.
- No calls to legacy `multi_parity_readout`, `parity_active_reset` or
  `measure_wrapper` remain inside the new per-mode block. The first Ramsey preserves
  the existing normal/fast pulse descriptors and calibrated Stark phase convention.
- Replaced the new pulse test's parent-method stubs with primitive pulse/register and
  measurement spies. The sequential test module passes all 23 cases, including
  normal/fast phases, one terminal relaxation, actual mock acquisition, and HDF5 saving.
- Created the requested `measurement_notebooks/jonginn/agp_sequential_measurement.py`
  draft from the reference notebook. No notebook cells were executed.

The user then put notebook work on hold and requested a native sequential AGP design.
The draft still contains the earlier export-to-legacy orchestration; it is not the
approved final workflow. No new AGP classes have been implemented. The next review
should cover new assembled classes that read the original sequential jobs directly,
with numerical estimators reused from the existing fitting/theory modules. Existing
Python source files remain unchanged. No real hardware, job queue or vault was used.

This follow-up supersedes the earlier log's description of delegated multiparity
pulses and its proposed export-based AGP notebook workflow.
