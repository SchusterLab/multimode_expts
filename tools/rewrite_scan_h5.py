"""Save the HDF5 of registered AGP disorder scans again, with their analysis inputs.

A scan HDF5 saved before 2026-11-10 has only the arrays of its save, so
``load_scan`` must read every job file (about 0.5 s each over SMB, minutes per
scan). This reads them once per scan (``from_manifest``), analyzes at the default
settings and writes the scan HDF5 again (``CentralBosonAGPDisorderScan.rewrite_h5``).
The old HDF5 is not deleted: it moves to ``<experiment>/assembled_data_old_<YYYYMMDD>``.
Afterwards ``load_scan(key)`` reads that one file and ``analyze()`` works with any
settings. The manifests and the parts' files are not changed.

    pixi run python tools/rewrite_scan_h5.py                 # every registered scan that needs it
    pixi run python tools/rewrite_scan_h5.py kerr s7_fine    # these keys
    pixi run python tools/rewrite_scan_h5.py --force kerr    # also if it has the inputs already

Run it where the data is fast to read when possible. It writes into the data
roots (``experiments/data_paths.py``).
"""
import argparse
import time
import warnings

from experiments.qsim.agp import CentralBosonAGPDisorderScan, load_scan, registered_scans, scan_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("keys", nargs="*", help="registry keys (default: all)")
    parser.add_argument("--force", action="store_true", help="also scans that have the inputs")
    args = parser.parse_args()
    for key in args.keys or registered_scans():
        h5_path = scan_manifest(key).with_suffix(".h5")
        if not args.force and CentralBosonAGPDisorderScan.from_h5file(h5_path).children:
            print(f"{key}: has the inputs, skipped")
            continue
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scan = load_scan(key, jobs=True)
            scan.analyze()
            path = scan.rewrite_h5()
        jobs = sum(len(part.job_ids) for part in scan.children)
        print(f"{key}: {len(scan.children)} W, {jobs} jobs, "
              f"{time.time() - start:.0f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
