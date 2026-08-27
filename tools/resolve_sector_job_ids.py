"""Resolve the published MBR sectors to literal, verified job-ID lists.

The notebook declares them as (date, first, last, step) ranges over a shared,
globally numbered job queue, then removes other users' interleaved jobs either
by a positional stride or by a program-class filter. Both are guesses about
what someone else was doing at the time. This resolves them once, by owner and
program, and prints lists to paste.
"""
import json, sqlite3
from collections import Counter

DB = 'file:///C:/python/multimode_expts/job_server/jobs.db?mode=ro'
CAL = 'EntireFloquetCyclePhaseCalibrationProgram'
SPEC = 'NPhotonHamiltonianSpectroscopyProgram'

RANGES = {
    'N1': dict(calibration=[(20260722, 557, 566, 1)], spectroscopy=[(20260722, 577, 595, 1)]),
    'N2': dict(calibration=[(20260722, 35, 64, 1)],   spectroscopy=[(20260722, 215, 244, 1)]),
    'N3': dict(calibration=[(20260722, 683, 712, 1), (20260723, 1, 40, 1)],
               spectroscopy=[(20260723, 48, 85, 1), (20260723, 87, 149, 2)]),
    'N2_supplement': dict(calibration=[(20260722, 425, 426, 1)],
                          spectroscopy=[(20260722, 452, 454, 1)]),
}

def expand(rs):
    return [f'JOB-{d}-{n:05d}' for d, a, b, s in rs for n in range(a, b + 1, s)]

# Widen to the full span, ignoring the stride, so we can see what it hid.
def span(rs):
    return [f'JOB-{d}-{n:05d}' for d, a, b, _ in rs for n in range(a, b + 1)]

conn = sqlite3.connect(DB, uri=True)

def rows(ids):
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = ','.join('?' * len(chunk))
        for r in conn.execute(
                f'select job_id,user,program_class,status,hardware_config_version_id,'
                f'floquet_storage_version_id,man1_storage_version_id '
                f'from jobs where job_id in ({q})', chunk):
            out[r[0]] = dict(zip(
                ('job_id','user','program_class','status','hw','fl','m1'), r))
    return out

resolved = {}
for sector, kinds in RANGES.items():
    print(f'===== {sector} =====')
    resolved[sector] = {}
    for kind, rs in kinds.items():
        want = CAL if kind == 'calibration' else SPEC
        declared = expand(rs)
        full = span(rs)
        meta = rows(full)
        good = [j for j in full
                if meta.get(j, {}).get('program_class') == want
                and meta[j]['status'] == 'COMPLETED']
        # Did the notebook's own expansion agree with the semantic filter?
        extra = [j for j in declared if j not in good]
        missed = [j for j in good if j not in declared]
        triples = Counter((meta[j]['hw'], meta[j]['fl'], meta[j]['m1']) for j in good)
        print(f'  {kind:13s} declared {len(declared):3d} | span {len(full):3d} | '
              f'owner+program {len(good):3d}')
        if extra:  print(f'      declared but WRONG program/status: {len(extra)} {extra[:4]}')
        if missed: print(f'      MISSED by the declared ranges    : {len(missed)} {missed[:4]}')
        if len(triples) != 1:
            print(f'      !! {len(triples)} config triples: {list(triples.items())[:3]}')
        else:
            print(f'      config triple: {list(triples)[0]}')
        resolved[sector][kind] = good

json.dump(resolved, open(r'C:\python\multimode_expts_guan\tests\data\mbr_sector_job_ids.json','w'),
          indent=1)
print('\nwrote tests/data/mbr_sector_job_ids.json')

# Why literal lists, not ranges
# -----------------------------
# Job IDs are one global counter on a queue shared by every user, and that is
# the intended design: user A measures at set point 1, user B measures at set
# point 2, refits and stores a new config version, A goes again, and so on.
# Everyone's jobs interleave, and each pins its own config, so this is benign.
#
# What is not benign is identifying a dataset by a *numeric range* over that
# counter. The notebook then has to subtract the other user's jobs, and does it
# two different ad-hoc ways: a positional stride (N3: step=2) and a
# program-class filter (N1). The stride assumes strict alternation and correct
# phase; it happened to be exactly right for N3, and would have silently
# selected the wrong jobs had the other user skipped one beat.
#
# Checked 2026-08-26, the declared ranges disagree with the owner+program
# filter in three of four sectors:
#
#     N1 spectroscopy            19 declared ->  10 actually jonginn's
#     N2 spectroscopy            30 declared ->  28
#     N2 supplement spectroscopy  3 declared ->   2
#     N3 spectroscopy            70 declared ->  70   (stride was correct)
#
# Every sector resolves to exactly one config triple, so the underlying data is
# sound; only the selection was loose.
