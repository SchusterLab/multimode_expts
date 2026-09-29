"""Build the fixed synthetic set of the gap score, and cache the fits of slow fitters on it.

    pixi run python tools/pole_fixed_set.py build
    pixi run python tools/pole_fixed_set.py fit C        # then F: it starts from C's cached fits
    pixi run python tools/pole_fixed_set.py fit F --draws 0 --rows 10     # a subset

Files in ``fixed_set.set_folder`` (``<data root>/260818_qsim_spectroscopy/derived_data/pole_finding/fixed_set/``):
``set.h5`` (``fitting.qsim.poles.fixed_set``) and ``fits_<fitter>.h5``. ``fit`` skips cases
already in the cache (fitted on the same rows), so an interrupted run resumes; each fit is written as soon as it is done.
Fits run serially (parallel fitter workers crash on pippin; ``benchmarks`` module docstring).
Plan: docs/qsim/pole_finding_explore.md, T0.
"""
import argparse
import time

from experiments.job_paths import data_root
from fitting.qsim.poles import joint_refined, pursuit
from fitting.qsim.poles.fixed_set import conditions, load_fits, load_set, make_case, save_fit, save_set, set_folder

FOLDER = set_folder(data_root())
SETTINGS = {"C": joint_refined.JointRefinedSettings(), "F": pursuit.PursuitSettings()}


def build():
    FOLDER.mkdir(parents=True, exist_ok=True)
    cases = []
    for i, condition in enumerate(conditions()):
        cases.append(make_case(condition))
        print(f"{i + 1}/80 {condition.key}", flush=True)
    print(save_set(FOLDER / "set.h5", cases, plan="docs/qsim/pole_finding_explore.md T0"))


def fit(fitter, draws=None, rows=None):
    """Fit the cases not yet cached; ``draws`` / ``rows`` (10 or 35): only those, 10-row cases first."""
    cases = load_set(FOLDER / "set.h5")
    path = FOLDER / f"fits_{fitter}.h5"
    done = load_fits(path, cases)
    starts = load_fits(FOLDER / "fits_C.h5", cases) if fitter == "F" else {}
    settings = SETTINGS[fitter]
    chosen = [case for case in cases if (draws is None or case.condition.draw in draws)
              and (rows is None or (case.condition.partial_rows or 35) in rows)]
    chosen.sort(key=lambda case: case.condition.partial_rows is None)
    for i, case in enumerate(chosen):
        key = case.condition.key
        if key in done:
            continue
        start = time.perf_counter()
        if fitter == "C":
            result = joint_refined.fit(case.A, case.time_us, settings)
        else:
            result = pursuit.fit(case.A, case.time_us, settings, start=starts[key][0] if key in starts else None)
        seconds = time.perf_counter() - start
        save_fit(path, case, result, seconds, settings)
        print(f"{i + 1}/{len(chosen)} {key}: {len(result.frequencies_MHz)} poles, {seconds:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["build", "fit"])
    parser.add_argument("fitter", nargs="?", choices=list(SETTINGS))
    parser.add_argument("--draws", type=int, nargs="+", help="only these draws")
    parser.add_argument("--rows", type=int, nargs="+", choices=[10, 35], help="only these row counts")
    args = parser.parse_args()
    build() if args.action == "build" else fit(args.fitter, args.draws, args.rows)


if __name__ == "__main__":
    main()
