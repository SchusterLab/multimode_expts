
# 1.  Which lines I have deleted

### 1-1. ``subsample_spectroscopy_shots`` in ``EncodingHamiltonianSpectroscopyExperiment``

- What this does: takes a large number of shots and subsample them to construct new experiments with the requested (lower) number of averages.
  - Verified by running real measurements.
- deleted because it was the stale code added for the debug, to see the changes in the number of averages.
  - Verdict upon further discussion: main reason for deletion was because the class it's in looks long. The correct approach is to single out these functionalities to a place that doesn't interfere active development (since that's what we are going to do to the very large files anyway).

### 1-2. ``mpm_candidate_familywise_alpha`` dependent branches in ``analyze_matrix_pencil``

- deleted because the branch was initially added as an ad-hoc degree of freedom to enhance the accuracy of matrix pencil method, But it did not work quite well
  - What it does: vague recollection that it adds one more degree of freedom when selecting poles, based on SVD. There's a certain threshold that decides which poles to abandon. In reality according to agent audits it's more than that.
  - Also, the code was not really perused, as it was one of the random exploration I tried with Codex.
  - Verdict: ok to delete. May reinstate properly after vetting if we need these functionalities in the future.

### 1-3. Deleted ``DisorderSFFExperiment`` class

- it was originally coded to measure the real-time spectral form factor under a lot of different disorder realizations; specifically, it is coded by Codex and never perused, so it was deleted.
  - Re-clarifying: this measures many body Ramsey via the same encoding decoding pulses as our usual experiments except this is hardware loops (over which parameters? didn't read or run) instead of software loops. So this actually sounds like an AveragerProgram vs RAveragerProgram distinction although needs checking.
- Only experiment class deleted, program class still there. See below.
- Verdict after discussion: move to separate module and annotate as untested.

### 1-4. Migrated ``SidebandStarkAmplificationModifiedProgram_old`` to ``SidebandStarkAmplificationProgram``

- both existed before this deletion. Latter was preexisting code from Connie. Former adds sync all cycles between floquet pulses in order to match the sync all cycles in other qsim experiments.
- But the correct pattern is to actually add the functionality of the former to the code of the latter (this deletion does that)
- It also consolidates the experiment config keys that enable the sync all between pulses:
- I deleted the program with the suffix `_old`, as it was coded in the course of debugging, and should have been moved to the ``SidebandStarkAmplificationProgram``. 
	- Also, in the course of migration, I edited the conditional branch criteria, as it depended on three different parameters in ``expt.cfg``. Now, if legacy behavior is turned off, it should be controlled solely by ``scramble_sync_cycles

### 1-5. Deleted unused ``ManStorMultiparityChevronRProgram`` and ``ManStorMultiparityChevronRExperiment``. 

- These programs were deleted because they are used in nowhere. They are originally coded to draw the chevron pattern of multiphoton pulses with multiparity readout when I was doing dark mode readout project.
- Verdict: this did something useful at the time, and it seems correct so we shouldn't delete this but find it a correct home.

# 2. The plans of program deletion & merging

These are Jonginn's tentative plans, not implemented yet.

## 2-1. Stale `subsample_spectroscopy_shots` dependent codes in `EncodingHamiltonianSpectroscopyExperiment`
- Inside `analyze` method of `EncodingHamiltonianSpectroscopyExperiment` there is still a line (which I assume to be 1~3 lines) that uses this deleted method, so these lines should be deleted.

## 2-2. ``DisorderSFFDepthSweepProgram`` and ``DisorderSFFSequenceMixin``
- I did not delete these programs, because as far as I remember some of the methods and attributes are used in other programs; especially hardware loop program. I am uncertain about whether I moved the page and register keeping program to the other class for the hardware loop with `preload_flattop` pulses. 
- The reason of planning to delete these lines is because I have already deleted real-time SFF measurement experiment program (``DisorderSFFExperiment``); I also have not perused all of the program line by line, and these are not used for now, I wanted to delete these lines.

## 2-3.Split ``DarkBaseExperiments`` and ``DarkBaseProgram`` 

The original purpose of these experiments and program was to do basis transformation to readout dark mode while inhering the majority of the functionality from `QsimBaseProgram` and possible Mixin classes. 

### 2-2-1. Migrate some of the methods into ``DarkBaseExperiments`` related to multiparity readout

- `acquire` method of `DarkBaseExperiments` is nearly identical to that of `QsimBaseExperiments`
	- Some of the differences include the `read_num` increment when `multiparity_readout` is turned on, 
- ``analyze_multiparity`` in ``DarkBaseExperiments`` should be moved to ``QsimBaseExperiments``. 

### 2-2-2. Migrate Floquet pulse related methods into a new mixin program; tentative name is `FloquetPulseMixin`
- 

## 2-3. Unify `SidebandScrambleDarkProgram`, `SidebandScrambleDarkProgramNew`, and `SidebandScrambleDarkProgramNewNew`

- The most recent version is `SidebandScrambleDarkProgramNewNew`. However, there are some qsim experiment notebook code that use `SidebandScrambleDarkProgramNew` and `SidebandScrambleDarkProgram`. In principle all of them should be replaced by `SidebandScrambleDarkProgramNewNew`, as the desired functionality is the one in `SidebandScrambleDarkProgramNewNew`, which ultimately uses desired `_play_m1s_frac_train` in `DarkBaseProgram` for floquet pulses.
- The one program I am not really sure about deleting is `SidebandScrambleDarkProgramDebug`. I think this is stale, but need perusal of qsim_experiment notebook. I am now 70 % sure this program can be deleted.

## 2-4. Delete `SidebandStarkAmplificationModifiedProgram` and `SidebandStarkAmplificationModifiedProgram_newold`
- These are stale; I coded these to either use floquet pulse train wrapper (`        self._play_m1s_frac_train`) or manually add `sync_all(10)`, but the former was not really compatible with `RAveragerProgram` and the letter was just used nowhere and therefore should have been deleted.
-
