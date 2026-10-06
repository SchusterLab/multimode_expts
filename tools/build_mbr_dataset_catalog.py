# -*- coding: utf-8 -*-
"""One-off: build ``configs/datasets/mbr_datasets.yaml`` from the two old lists.

Run once on 2026-10-02 (GitHub issue 6, question 1: one master list). Inputs:

- jonginn's catalog ``docs/spectroscopy job id compilation/Only JOB IDs for Agents.md``
  at commit 91da3f7 (branch ``job_id_sorting_out``), read with ``git show``;
- the former test fixture ``tests/data/mbr_datasets.json``.

The script keeps the JSON's data set names where they exist, applies the decisions
jonginn gave on issue 6 (stale Sep 11 jobs dropped, JOB-20260830-00135 dropped, the
quality-concern sets kept and marked) and checks that every list it carries over is
identical to the JSON's. Kept for the record; after the run the JSON and the two
markdown lists are deleted, so it will not run again unchanged.

    pixi run python tools/build_mbr_dataset_catalog.py
"""
import json
import re
import subprocess
import textwrap
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SRC_COMMIT = "91da3f7"
SRC_MD = "docs/spectroscopy job id compilation/Only JOB IDs for Agents.md"
# The JSON is deleted after the first run; read it from the last commit that has it.
JSON_COMMIT = "c3bfeda"
OLD_JSON = "tests/data/mbr_datasets.json"
OUT = REPO / "configs" / "datasets" / "mbr_datasets.yaml"

JOB = r"JOB-\d{8}-\d{5}"


# --------------------------------------------------------------------------
# parse the markdown catalog
# --------------------------------------------------------------------------

def fields(block):
    out = {}
    # "- Key: value"; section 2.1 of the catalog lacks the colon ("- Calibration_job_ids JOB-...").
    for m in re.finditer(r"^- ([\w ]+?)(?::|(?=\s+JOB-))\s*(.*?)(?=^- |\Z)", block,
                         flags=re.M | re.S):
        key, body = m.group(1).strip().lower().replace(" ", "_"), m.group(2)
        if key == "job_ids":
            key = "spec_job_ids"
        if key == "foler":
            key = "folder"
        if key in ("folder", "config"):
            out[key] = re.search(r"[\w-]+", body.strip().strip("'\"\u2019\u201d\u201c")).group(0)
        elif key == "spec_job_ids" and re.search(r"^\s*\d+:\s*\[", body, flags=re.M):
            # realization lines; indented in section 2.2, at column 0 elsewhere
            out["realizations"] = {
                int(k): re.findall(JOB, v)
                for k, v in re.findall(r"^\s*(\d+):\s*\[(.*?)\]", body, flags=re.M)}
        else:
            out[key] = re.findall(JOB, body)
    return out


def parse_catalog(md):
    sections = {}
    for top in re.split(r"^(?=# )", md, flags=re.M)[1:]:
        h1 = re.match(r"# (.*)", top).group(1).strip()
        subs = re.split(r"^(?=## )", top, flags=re.M)
        if len(subs) == 1:
            sections[h1] = dict(label=h1, **fields(top))
        else:
            for sub in subs[1:]:
                h2 = re.match(r"## (.*)", sub).group(1).strip()
                sections[f"{h1} / {h2}"] = dict(label=f"{h1} / {h2}", **fields(sub))
    return sections


def gk(label):
    g = re.search(r"g\s*=\s*([\d.]+)\s*kHz", label)
    k = re.search(r"K\s*=\s*([\d.]+)\s*kHz", label)
    return (float(g.group(1)) if g else None), (float(k.group(1)) if k else None)


def ids(value):
    return set(re.findall(JOB, json.dumps(value)))


# --------------------------------------------------------------------------
# the entries
# --------------------------------------------------------------------------

DM = "260526_qsim_darkmode/assembled_data/"
SP = "260818_qsim_spectroscopy/assembled_data/"


# (name, catalog section prefix)
SECTION_72 = [("sep07_pairs_K52p3_g29p2", "5. "), ("sep05_pairs_K3p6_g30", "6. "),
              ("sep01_pairs_K44_g15", "7. "), ("sep02_pairs_K20_g15", "8. "),
              ("sep02to04_pairs_K3p6_g15", "9. ")]
FORMER_NAMES = {"sep07_pairs_K52p3_g29p2": "d72_Sep07_K52p3_g29p2",
                "sep05_pairs_K3p6_g30": "d72_Sep05_K3p6_g30",
                "sep01_pairs_K44_g15": "d72_Sep01_K44_g15",
                "sep02_pairs_K20_g15": "d72_Sep02_K20_g15",
                "sep02to04_pairs_K3p6_g15": "d72_Sep02-04_K3p6_g15"}
# Sep05 shares its calibration jobs with september_N3, already converted.
SECTION_72_CALIBRATION = {"sep05_pairs_K3p6_g30":
                          "260818_qsim_spectroscopy/assembled_data/"
                          "261002_145528_MBRCalibrationSetExperiment.yaml"}
SECTION_72_NOTES = {
    "sep07_pairs_K52p3_g29p2": "6 realizations x 15 pairs, each pair in 2 time chunks (cycles "
                               "0-398, 400-466): 80 diagonal pairs, 10 off-diagonal.",
    "sep05_pairs_K3p6_g30": "2 realizations x 15 pairs: 23 diagonal, 7 off-diagonal. "
                            "Calibration: september_N3.",
    "sep01_pairs_K44_g15": "5 realizations x 10 pairs: 30 diagonal, 20 off-diagonal. The files "
                           "have no spectroscopy_final_occupations; the decoder is in each "
                           "sweep row (cycle_decoder_analyzers), and the off-diagonal rows "
                           "start at |A| ~ 0.02, as an init != final trace must.",
    "sep02_pairs_K20_g15": "5 realizations x 10 pairs: 33 diagonal, 17 off-diagonal.",
    "sep02to04_pairs_K3p6_g15": "19 realizations x 15 pairs: 161 diagonal, 124 off-diagonal.",
}
SECTION_72_COMMON = (
    "Old notebook section 7-2 ('D72') campaign: one EncodingPropagatorProgram job per "
    "(initial, decoder) pair, theory-selected pairs (d72_selected_pairs), d72_* keys. "
    "Formerly labeled 'off-diagonal' as a whole; about 2/3 of the pairs are diagonal "
    "(checked 2026-10-06 in every file: config and cycle-0 return agree). The diagonal "
    "pairs form the ensemble (manifest); the off-diagonal ones are converted as "
    "TimeTraces in one MBRTimeTraceSetExperiment per realization (offdiag_manifests), not "
    "analyzed (docs/qsim/mbr_step7_plan.md, decision 2). The pulse played no Stark "
    "correction (final_analyzer_phase_per_cycle_deg = 0); the section 7-2 analysis-side "
    "correction is kept in each converted file's converted_from.source_notes. Timing from "
    "the versioned configs; the archived 0.4135 us of the g15 sets was the pre-0d7ea14 "
    "cycle formula. The catalog index is authoritative: the files repeat d72_realization "
    "numbers across sessions. RMS normalization (jonginn, issue 7).")


def build(sections, old):
    entries = {}

    def sec(prefix):
        hits = [v for k, v in sections.items() if k.startswith(prefix)]
        assert len(hits) == 1, (prefix, [k for k in sections if k.startswith(prefix)])
        return hits[0]

    def add(name, s, kind, quality="ok", manifest=None, calibration_manifest=None, notes="",
            **extra):
        g, kerr = gk(s["label"])
        e = dict(label=s["label"], kind=kind, folder=s.get("folder"),
                 floquet_config=s.get("config"), g_kHz=g, K_kHz=kerr, K_source="?",
                 quality=quality, converted=manifest is not None, manifest=manifest,
                 calibration_manifest=calibration_manifest, notes=notes)
        if "calibration_job_ids" in s:
            e["calibration_job_ids"] = s["calibration_job_ids"]
        if "realizations" in s:
            e["realizations"] = s["realizations"]
        elif "spec_job_ids" in s:
            e["job_ids"] = s["spec_job_ids"]
        e.update(extra)
        entries[name] = e

    add("july_N1", sec("1. g = 15 kHz, K = 19.2 kHz / 1."), "spectrum",
        manifest=DM + "260924_163501_MBRSpectrumExperiment.yaml",
        calibration_manifest=DM + "260924_163500_MBRCalibrationSetExperiment.yaml",
        notes="N=1, 5 occupations. The catalog heading says 'Disordered'; the jobs carry no "
              "disorder keys.",
        config_triple=old["july_N1"]["config_triple"])
    add("july_N2", sec("1. g = 15 kHz, K = 19.2 kHz / 2. Disorderless"), "spectrum",
        manifest=DM + "260924_163502_MBRSpectrumExperiment.yaml",
        calibration_manifest=DM + "260924_163502_MBRCalibrationSetExperiment.yaml",
        notes="N=2, 14 of 15 occupations; see july_N2_supplement.",
        config_triple=old["july_N2"]["config_triple"])
    add("july_N2_supplement", sec("1. g = 15 kHz, K = 19.2 kHz / 2-1."), "spectrum",
        manifest=DM + "260924_163503_MBRSpectrumExperiment.yaml",
        calibration_manifest=DM + "260924_163502_MBRCalibrationSetExperiment_2.yaml",
        notes="The missing N=2 occupation of july_N2.",
        config_triple=old["july_N2_supplement"]["config_triple"])
    add("july_N3", sec("1. g = 15 kHz, K = 19.2 kHz / 3."), "spectrum",
        manifest=DM + "260924_163508_MBRSpectrumExperiment.yaml",
        calibration_manifest=DM + "260924_163506_MBRCalibrationSetExperiment.yaml",
        notes="N=3 complete basis, 35 occupations. Legacy analyzer convention "
              "(+cycle*correction).",
        config_triple=old["july_N3"]["config_triple"])
    add("july_N1_early", sec("1. g = 15 kHz, K = 19.2 kHz / 4."), "spectrum",
        notes="N=1, 5 occupations, measured Jul 21-22 before july_N1. Added by jonginn "
              "2026-10-01 (issue 6).")
    add("august_N3", sec("2. g = 8.61 kHz, K = 5.7 kHz / 1."), "spectrum",
        manifest=DM + "260924_163516_MBRSpectrumExperiment.yaml",
        calibration_manifest=DM + "260924_163512_MBRCalibrationSetExperiment.yaml",
        notes="N=3 complete basis.", config_triple=old["august_N3"]["config_triple"])
    add("august_disorder", sec("2. g = 8.61 kHz, K = 5.7 kHz / 2."), "disorder",
        manifest=DM + "260924_195547_MBRDisorderEnsembleExperiment.yaml",
        notes="4 realizations x 10 theory-selected occupations (the old 'pairwise' preview), "
              "disorder_* keys. Converted with the august_N3 calibration set. Former names "
              "august_disorder_r0..r3. Direction normalized by the norm (jonginn, issue 7): "
              "50 kHz here is 25 kHz in the RMS convention.",
        config_triple=old["august_disorder_r0"]["config_triple"])
    add("diagonal_disorder_71", sec("3. g = 15.2 kHz, K = 44.96 kHz / 1."), "disorder",
        manifest=SP + "260924_195637_MBRDisorderEnsembleExperiment.yaml",
        calibration_manifest=SP + "260924_195553_MBRCalibrationSetExperiment.yaml",
        notes="The 7-1 campaign: 20 realizations x 10 occupations, diagonal_disorder_* keys. "
              "r=19 is incomplete (5 of 10 occupations; JOB-20260830-00135 has no phase-90 "
              "partner and is left out, jonginn issue 6). The converted ensemble holds r0-r18. "
              "Direction normalized by the norm (jonginn, issue 7).")
    add("august28_N1", sec("3. g = 15.2 kHz, K = 44.96 kHz / 2."), "spectrum",
        notes="N=1, 5 occupations, Aug 28. Added by jonginn 2026-10-01 (issue 6).")
    add("sep10_full_K3p6_g29p2", sec("4. g = 29.2 kHz, K = 3.6 kHz"), "disorder",
        notes="Sep 10-14: 9 realizations x the complete N=3 basis (35 occupations, 70 jobs "
              "each); diagonal traces (checked 2026-10-02: no offdiag keys, initial == final "
              "in all 630 files). Formerly mislabeled d72_Sep10_K3p6_g29p2. r=0 is "
              "JOB-20260910-00031..72 plus the Sep 11 remeasurement -00046..86; "
              "JOB-20260911-00001..28 are stale (jonginn, issue 6). The files record the "
              "realization under three key spellings (d73_*, then flat realization/seed) and "
              "number r1-r8 as 0-7; the catalog index is authoritative. RMS normalization "
              "(jonginn, issue 7).")
    # Old notebook section 7-2 ("D72") campaigns. Renamed 2026-10-06: each holds both
    # kinds of pair, so the old label "off-diagonal" was wrong (scan of every file;
    # docs/log/2026-10-06_issue5-d72-and-loaders.md).
    for name, prefix in SECTION_72:
        add(name, sec(prefix), "disorder_pairs",
            calibration_manifest=SECTION_72_CALIBRATION.get(name),
            notes=(f"{SECTION_72_NOTES[name]} Former name {FORMER_NAMES[name]}. "
                   f"{SECTION_72_COMMON}"))
    six = sec("6. ")
    entries["september_N3"] = dict(
        label="6. g = 30 kHz, K = 3.6 kHz / calibration set", kind="stark_cal",
        folder=six["folder"], floquet_config=six["config"], g_kHz=30.0, K_kHz=3.6,
        K_source="?", quality="ok", converted=False, manifest=None, calibration_manifest=None,
        notes="N=3 Stark calibration, 35 occupations x 2 phases, preload_flattop swaps. Also "
              "the calibration of sep05_pairs_K3p6_g30.",
        job_ids=old["september_N3"]["calibration"],
        config_triple=old["september_N3"]["config_triple"])
    assert entries["september_N3"]["job_ids"] == six["calibration_job_ids"]
    add("august19_N3_orthogonality_K45", sec("10. "), "orthogonality",
        notes="35 zero-cycle columns, Aug 19. Added by jonginn 2026-10-01 (issue 6).")
    add("august26_N3_orthogonality_K17", sec("11. "), "orthogonality",
        notes="35 zero-cycle columns, Aug 26. Added by jonginn 2026-10-01 (issue 6).")
    add("september_N3_orthogonality", sec("12. "), "orthogonality",
        notes="35 zero-cycle columns, one per initial occupation, modes M1, S2..S5 (given by "
              "the user 2026-09-24).",
        config_triple=old["september_N3_orthogonality"]["config_triple"])
    assert (entries["september_N3_orthogonality"]["job_ids"]
            == old["september_N3_orthogonality"]["orthogonality"])
    add("august26_N3_orthogonality_K17_again", sec("13. "), "orthogonality",
        notes="35 zero-cycle columns, Aug 26, second set. Added by jonginn 2026-10-01 "
              "(issue 6).")
    add("august25_N3", sec("14. "), "spectrum", quality="concern",
        notes="N=3 complete basis, Aug 24-25. jonginn: excluded from his analysis, the "
              "man-stor swap gains looked too high. Added 2026-10-01 (issue 6).")
    add("august19_disorder_K44_g18", sec("15. "), "disorder", quality="concern",
        notes="3 realizations x 10 occupations, disorder_* keys, Aug 19. jonginn: quality "
              "concern. Added 2026-10-01 (issue 6).")
    add("august27_disorder_K20_g15", sec("16. "), "disorder", quality="concern",
        notes="3 realizations x 10 occupations, diagonal_disorder_* keys, Aug 27. jonginn: "
              "quality concern. Added 2026-10-01 (issue 6).")
    entries["august_quickplot"] = dict(
        label="August quick-plot fragment (not in jonginn's catalog)", kind="spectrum",
        folder="260814_qsim_encspec", floquet_config=None, g_kHz=None, K_kHz=None,
        K_source="?", quality="ok", converted=True,
        manifest="260814_qsim_encspec/assembled_data/260924_163516_MBRSpectrumExperiment.yaml",
        calibration_manifest=None,
        notes="8 jobs, 4 occupations: the notebook's data_four_realization. Gates code (golden "
              "baseline), not physics.",
        job_ids=old["august_quickplot"]["spectroscopy"],
        config_triple=old["august_quickplot"]["config_triple"])
    entries["august_N1_propagator"] = dict(
        label="August N=1 propagator (not in jonginn's catalog)", kind="propagator",
        folder="260818_qsim_spectroscopy", floquet_config=None, g_kHz=None, K_kHz=None,
        K_source="?", quality="ok", converted=False, manifest=None, calibration_manifest=None,
        notes=old["august_N1_propagator"]["notes"],
        job_ids=old["august_N1_propagator"]["propagator"],
        config_triple=old["august_N1_propagator"]["config_triple"])
    return entries


# The round of 2026-10-02 (tools/convert_mbr_catalog.py, pippin): name -> (manifest,
# calibration manifest), relative to the data root.
SP_ = "260818_qsim_spectroscopy/assembled_data/"
CONVERTED_2026_10_02 = {
    "july_N1_early": (DM + "261002_145411_MBRSpectrumExperiment.yaml",
                      DM + "261002_145411_MBRCalibrationSetExperiment.yaml"),
    "august28_N1": (SP_ + "261002_145412_MBRSpectrumExperiment.yaml",
                    SP_ + "261002_145412_MBRCalibrationSetExperiment.yaml"),
    "sep10_full_K3p6_g29p2": (SP_ + "261002_145524_MBRDisorderEnsembleExperiment.yaml",
                              SP_ + "261002_145417_MBRCalibrationSetExperiment.yaml"),
    "september_N3": (SP_ + "261002_145528_MBRCalibrationSetExperiment.yaml", None),
    "august19_N3_orthogonality_K45": (SP_ + "261002_145530_MBROrthogonalityExperiment.yaml", None),
    "august26_N3_orthogonality_K17": (SP_ + "261002_145532_MBROrthogonalityExperiment.yaml", None),
    "september_N3_orthogonality": (SP_ + "261002_145535_MBROrthogonalityExperiment.yaml", None),
    "august26_N3_orthogonality_K17_again": (
        SP_ + "261002_145537_MBROrthogonalityExperiment.yaml", None),
    "august25_N3": (SP_ + "261002_145543_MBRSpectrumExperiment.yaml",
                    SP_ + "261002_145541_MBRCalibrationSetExperiment.yaml"),
    "august19_disorder_K44_g18": (SP_ + "261002_145551_MBRDisorderEnsembleExperiment.yaml",
                                  SP_ + "261002_145546_MBRCalibrationSetExperiment.yaml"),
    "august27_disorder_K20_g15": (SP_ + "261002_145559_MBRDisorderEnsembleExperiment.yaml",
                                  SP_ + "261002_145554_MBRCalibrationSetExperiment.yaml"),
    "august_N1_propagator": (SP_ + "261002_145600_MBRHamTomoExperiment.yaml", None),
}


# The round of 2026-10-06 (the section 7-2 pair sets): name -> (ensemble manifest,
# calibration manifest, {realization: off-diagonal MBRTimeTraceSetExperiment manifest}).
CONVERTED_2026_10_06 = {
    "sep05_pairs_K3p6_g30": (
        SP_ + "261006_145820_MBRDisorderEnsembleExperiment.yaml",
        SP_ + "261002_145528_MBRCalibrationSetExperiment.yaml",
        {
        0: SP_ + "261006_145818_MBRTimeTraceSetExperiment.yaml",
        1: SP_ + "261006_145820_MBRTimeTraceSetExperiment.yaml"}),
    "sep01_pairs_K44_g15": (
        SP_ + "261006_145828_MBRDisorderEnsembleExperiment.yaml",
        SP_ + "261006_145823_MBRCalibrationSetExperiment.yaml",
        {
        0: SP_ + "261006_145824_MBRTimeTraceSetExperiment.yaml",
        1: SP_ + "261006_145825_MBRTimeTraceSetExperiment.yaml",
        2: SP_ + "261006_145826_MBRTimeTraceSetExperiment.yaml",
        3: SP_ + "261006_145826_MBRTimeTraceSetExperiment_2.yaml",
        4: SP_ + "261006_145827_MBRTimeTraceSetExperiment.yaml"}),
    "sep02_pairs_K20_g15": (
        SP_ + "261006_145836_MBRDisorderEnsembleExperiment.yaml",
        SP_ + "261006_145831_MBRCalibrationSetExperiment.yaml",
        {
        0: SP_ + "261006_145831_MBRTimeTraceSetExperiment.yaml",
        1: SP_ + "261006_145832_MBRTimeTraceSetExperiment.yaml",
        2: SP_ + "261006_145833_MBRTimeTraceSetExperiment.yaml",
        3: SP_ + "261006_145834_MBRTimeTraceSetExperiment.yaml",
        4: SP_ + "261006_145835_MBRTimeTraceSetExperiment.yaml"}),
    "sep07_pairs_K52p3_g29p2": (
        SP_ + "261006_145858_MBRDisorderEnsembleExperiment.yaml",
        SP_ + "261006_145840_MBRCalibrationSetExperiment.yaml",
        {
        0: SP_ + "261006_145842_MBRTimeTraceSetExperiment.yaml",
        1: SP_ + "261006_145845_MBRTimeTraceSetExperiment.yaml",
        2: SP_ + "261006_145847_MBRTimeTraceSetExperiment.yaml",
        3: SP_ + "261006_145849_MBRTimeTraceSetExperiment.yaml",
        4: SP_ + "261006_145851_MBRTimeTraceSetExperiment.yaml",
        5: SP_ + "261006_145854_MBRTimeTraceSetExperiment.yaml"}),
    "sep02to04_pairs_K3p6_g15": (
        SP_ + "261006_145933_MBRDisorderEnsembleExperiment.yaml",
        SP_ + "261006_145900_MBRCalibrationSetExperiment.yaml",
        {
        0: SP_ + "261006_145901_MBRTimeTraceSetExperiment.yaml",
        1: SP_ + "261006_145903_MBRTimeTraceSetExperiment.yaml",
        2: SP_ + "261006_145904_MBRTimeTraceSetExperiment.yaml",
        3: SP_ + "261006_145906_MBRTimeTraceSetExperiment.yaml",
        4: SP_ + "261006_145907_MBRTimeTraceSetExperiment.yaml",
        5: SP_ + "261006_145909_MBRTimeTraceSetExperiment.yaml",
        6: SP_ + "261006_145910_MBRTimeTraceSetExperiment.yaml",
        7: SP_ + "261006_145911_MBRTimeTraceSetExperiment.yaml",
        8: SP_ + "261006_145913_MBRTimeTraceSetExperiment.yaml",
        9: SP_ + "261006_145915_MBRTimeTraceSetExperiment.yaml",
        10: SP_ + "261006_145916_MBRTimeTraceSetExperiment.yaml",
        11: SP_ + "261006_145917_MBRTimeTraceSetExperiment.yaml",
        12: SP_ + "261006_145919_MBRTimeTraceSetExperiment.yaml",
        13: SP_ + "261006_145921_MBRTimeTraceSetExperiment.yaml",
        14: SP_ + "261006_145922_MBRTimeTraceSetExperiment.yaml",
        15: SP_ + "261006_145923_MBRTimeTraceSetExperiment.yaml",
        16: SP_ + "261006_145925_MBRTimeTraceSetExperiment.yaml",
        17: SP_ + "261006_145926_MBRTimeTraceSetExperiment.yaml",
        18: SP_ + "261006_145928_MBRTimeTraceSetExperiment.yaml"}),
}


def apply_conversions(entries, converted=CONVERTED_2026_10_02):
    for name, (manifest, calibration_manifest) in converted.items():
        entries[name].update(converted=True, manifest=manifest,
                             calibration_manifest=calibration_manifest)
    for name, (manifest, calibration_manifest, offdiag) in CONVERTED_2026_10_06.items():
        entries[name].update(converted=True, manifest=manifest,
                             calibration_manifest=calibration_manifest,
                             offdiag_manifests=offdiag)


def check_against_json(entries, old):
    for n in ("july_N1", "july_N2", "july_N2_supplement", "july_N3", "august_N3"):
        assert entries[n]["job_ids"] == old[n]["spectroscopy"], n
        assert entries[n]["calibration_job_ids"] == old[n]["calibration"], n
    for r in range(4):
        assert (entries["august_disorder"]["realizations"][r]
                == old[f"august_disorder_r{r}"]["spectroscopy"]), r
    assert entries["august_disorder"]["calibration_job_ids"] == old["august_N3"]["calibration"]
    for r in range(19):
        assert (entries["diagonal_disorder_71"]["realizations"][r]
                == old["diagonal_disorder_71"]["realizations"][str(r)]), r
    assert (entries["diagonal_disorder_71"]["calibration_job_ids"]
            == old["diagonal_disorder_71"]["calibration"])
    for n, former in FORMER_NAMES.items():
        assert ({str(k): v for k, v in entries[n]["realizations"].items()}
                == old[former]["realizations"]), n
        assert entries[n]["calibration_job_ids"] == old[former]["calibration"], n
    assert (entries["sep10_full_K3p6_g29p2"]["calibration_job_ids"]
            == old["d72_Sep10_K3p6_g29p2"]["calibration"])


# --------------------------------------------------------------------------
# emit: block mapping, job-ID lists in wrapped flow style
# --------------------------------------------------------------------------

HEADER = """\
# MBR spectroscopy data sets: which job IDs form which data set, and what became of them.
#
# One master list (GitHub issue 6, question 1). It replaces tests/data/mbr_datasets.json (the
# former test fixture), docs/spectroscopy job id compilation/Only JOB IDs for Agents.md and
# docs/job_list_and_nb_labeling/Job_list.md. Built 2026-10-02 by tools/build_mbr_dataset_catalog.py
# from jonginn's catalog at commit {commit} (branch job_id_sorting_out) merged with the JSON; every
# job ID below has an HDF5 file under the stated folder and a record in
# tests/data/job_provenance.json. The human-readable vault with plots stays:
# docs/spectroscopy job id compilation/Human-Friendly Job ID classification doc/.
#
# Job IDs are literal lists, never ranges: the queue is one global counter shared by every user.
#
# Readers: tools/convert_mbr_catalog.py (conversion to the new layout, one block per data set),
# tests/mbr_reference.py (test fixtures). Downstream, analysis_notebooks/pole_finding/registry.yaml
# lists the converted manifests the benchmarks read.
#
# Fields per data set
#   label              jonginn's catalog heading, verbatim (section / subsection)
#   kind               spectrum | disorder | stark_cal | orthogonality | propagator | disorder_pairs
#                      (tools/convert_mbr_catalog.py has one block per kind; disorder_pairs are the old
#                      notebook section 7-2 pair jobs, diagonal and off-diagonal mixed)
#   folder             experiment folder under the data root (C:\\experiments on pippin)
#   floquet_config     the Floquet config version the catalog names (g is computed from its timing)
#   g_kHz, K_kHz       rough labels from the catalog heading. g follows from the swap timing
#                      (a pi/n swap of length T gives the hopping rate after trotterization).
#                      K is either the undriven M1 self-Kerr or a fit of one time trace to theory;
#                      K_source says which when known ("?" = jonginn to fill in). Not inputs to anything.
#   quality            ok | concern (jonginn excluded it from his analysis; kept for completeness)
#   converted          true when the converted files and the manifest below exist on pippin
#   manifest           the assembled manifest, relative to the data root (MBRSpectrumExperiment,
#                      MBRDisorderEnsembleExperiment, ...); calibration_manifest likewise
#   calibration_job_ids  old Stark-calibration jobs (phase 0/90 pairs) of this data set
#   job_ids            the data jobs (spectrum: phase pairs per occupation; orthogonality: one per column)
#   realizations       disorder only: {{index: [job IDs]}}; the index here is authoritative
#   offdiag_manifests  disorder_pairs only: {{realization: MBRTimeTraceSetExperiment manifest}} of the
#                      off-diagonal pairs; the diagonal pairs are in manifest
#   config_triple      config version IDs the JSON resolver recorded (hardware, floquet, man1), where known

datasets:
"""

ORDER = ["label", "kind", "folder", "floquet_config", "g_kHz", "K_kHz", "K_source", "quality",
         "converted", "manifest", "calibration_manifest", "offdiag_manifests", "notes",
         "config_triple", "calibration_job_ids", "job_ids", "realizations"]


def flow(lst, indent):
    pad = " " * indent
    return "\n".join(pad + line for line in textwrap.wrap(
        ", ".join(lst), width=108 - indent, break_long_words=False, break_on_hyphens=False))


def emit(entries):
    out = [HEADER.format(commit=SRC_COMMIT)]
    for name, e in entries.items():
        out.append(f"  {name}:")
        for key in ORDER:
            if e.get(key) is None:
                continue
            v = e[key]
            if key in ("calibration_job_ids", "job_ids"):
                out += [f"    {key}: [  # {len(v)}", flow(v, 6), "    ]"]
            elif key == "realizations":
                total = sum(len(x) for x in v.values())
                out.append(f"    {key}:  # {len(v)} realizations, {total} jobs")
                for r, lst in v.items():
                    out += [f"      {r}: [  # {len(lst)}", flow(lst, 8), "      ]"]
            elif key == "notes":
                out.append("    notes: >-")
                out += ["      " + line for line in textwrap.wrap(v, 94)]
            elif isinstance(v, dict):
                out.append(f"    {key}: {json.dumps(v, ensure_ascii=False)}")
            elif isinstance(v, bool):
                out.append(f"    {key}: {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                out.append(f"    {key}: {v}")
            else:
                out.append(f"    {key}: {json.dumps(v, ensure_ascii=False)}")
        out.append("")
    return "\n".join(out)


def main():
    md = subprocess.run(["git", "show", f"{SRC_COMMIT}:{SRC_MD}"], capture_output=True,
                        text=True, encoding="utf-8", check=True, cwd=REPO).stdout
    old = json.loads(subprocess.run(
        ["git", "show", f"{JSON_COMMIT}:{OLD_JSON}"], capture_output=True, text=True,
        encoding="utf-8", check=True, cwd=REPO).stdout)["datasets"]
    entries = build(parse_catalog(md), old)
    check_against_json(entries, old)
    apply_conversions(entries)
    new_all = set().union(*(ids({k: v for k, v in e.items() if k != "notes"})
                            for e in entries.values()))
    old_all = set().union(*(ids(e) for e in old.values()))
    dropped = old_all - new_all
    print(f"job IDs: yaml {len(new_all)}, json {len(old_all)}, dropped from the json "
          f"{len(dropped)} (expected 29: the 28 stale Sep 11 jobs and JOB-20260830-00135)")
    assert len(dropped) == 29, sorted(dropped)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(emit(entries), encoding="utf-8")
    back = yaml.safe_load(OUT.read_text(encoding="utf-8"))["datasets"]
    assert list(back) == list(entries)
    for n in entries:
        assert ids(back[n]) == ids(entries[n]), n
        for k in ("job_ids", "calibration_job_ids"):
            if k in entries[n]:
                assert back[n][k] == entries[n][k], (n, k)
        if "realizations" in entries[n]:
            assert ({int(k): v for k, v in back[n]["realizations"].items()}
                    == entries[n]["realizations"]), n
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size} bytes), {len(back)} data sets, "
          f"round-trip ok")


if __name__ == "__main__":
    main()
