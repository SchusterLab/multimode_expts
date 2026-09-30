"""Read the curated spectroscopy JOB-ID document without executing its contents."""

from __future__ import annotations

import ast
from pathlib import Path
import re


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_GROUP = re.compile(
    rf"#\s+(\d+)\.\s*g\s*=\s*({_NUMBER})\s*kHz\s*,\s*"
    rf"K\s*=\s*({_NUMBER})\s*kHz\s*$"
)
_SUBGROUP = re.compile(r"##\s+([\d-]+)\.\s*(.+)$")
_FIELD = re.compile(
    r"-\s*(Folder|Foler|Config|Calibration_job_ids|Spec_job_ids)"
    r"(?:\s*:\s*|\s+)(.*)$"
)
_JOB_ID = re.compile(r"JOB-\d{8}-\d{5}$")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _job_list(value, fail, label):
    if not isinstance(value, (list, tuple)) or not value:
        fail(f"{label} must contain a nonempty list of JOB IDs")
    if any(not isinstance(job, str) or not _JOB_ID.fullmatch(job) for job in value):
        fail(f"{label} contains a malformed JOB ID")
    if len(set(value)) != len(value):
        fail(f"{label} contains duplicate JOB IDs")
    return list(value)


def _job_value(text, fail, label):
    text = text.translate(_QUOTES).strip()
    if text.startswith(("[", "{", "(")):
        try:
            node = ast.parse(text, mode="eval")
            value = ast.literal_eval(node)
        except (SyntaxError, ValueError, TypeError) as error:
            fail(f"invalid {label}: {error}")
        if isinstance(node.body, ast.Dict):
            keys = [ast.literal_eval(key) for key in node.body.keys]
            if len(set(keys)) != len(keys):
                fail(f"{label} contains duplicate realization keys")
        return value
    # The first sections use plain comma-separated IDs, without quotes/brackets.
    values = [value.strip() for value in text.split(",")]
    if values and not values[-1]:
        values.pop()
    return _job_list(values, fail, label)


def load_decay_manifest(path):
    """Return one record per documented dataset, in source order.

    Records contain ``dataset_id`` (main/subsection numbers), ``group_title``,
    ``dataset_title``, ``nominal_g_kHz``, ``nominal_K_kHz``, ``folder``,
    ``floquet_config_version``, ``calibration_job_ids``, and
    ``spectroscopy_job_ids_by_r``. Dict keys are bookkeeping display indices;
    a plain JOB list is represented by ``{None: ids}``. They are not asserted
    equal to acquisition realization metadata. ``documented_total_photons``
    and ``documented_disorder`` are None when the title does not specify them.

    ``source_path``, ``source_line``, ``source_end_line``, and
    ``source_field_lines`` retain provenance. The nominal numbers are copied
    from the document, with their written signs; they do not replace HDF5 or
    station/config values in time-axis construction or physical modeling.
    Known typography quirks are accepted; malformed/incomplete blocks raise
    ValueError with the source location instead of being silently skipped.
    """
    path = Path(path).expanduser().resolve()
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    records = []
    group = None
    block = None
    field = None

    def finish(end_line):
        nonlocal block, field
        if block is None:
            return

        def fail(message):
            raise ValueError(f"{path}:{block['source_line']}: {message}")

        fields = block.pop("fields")
        required = {"Folder", "Config", "Calibration_job_ids", "Spec_job_ids"}
        missing = required.difference(fields)
        if missing:
            fail(f"missing required fields: {', '.join(sorted(missing))}")
        values = {key: "\n".join(value).strip() for key, value in fields.items()}
        folder = values["Folder"].translate(_QUOTES).strip("\"' ")
        config = values["Config"].translate(_QUOTES).strip("\"' ")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", folder) or folder in {".", ".."}:
            fail(f"invalid project folder: {folder!r}")
        if not re.fullmatch(r"CFG-FL-\d{8}-\d{5}", config):
            fail(f"invalid Floquet config version: {config!r}")
        calibration = _job_list(
            _job_value(values["Calibration_job_ids"], fail, "Calibration_job_ids"),
            fail, "Calibration_job_ids",
        )
        spectroscopy = _job_value(values["Spec_job_ids"], fail, "Spec_job_ids")
        if isinstance(spectroscopy, dict):
            if not spectroscopy or any(type(r) is not int or r < 0 for r in spectroscopy):
                fail("Spec_job_ids dict requires nonnegative integer display indices")
            spectroscopy = {
                r: _job_list(ids, fail, f"Spec_job_ids[{r}]")
                for r, ids in spectroscopy.items()
            }
        else:
            spectroscopy = {None: _job_list(spectroscopy, fail, "Spec_job_ids")}
        all_spec = [job for ids in spectroscopy.values() for job in ids]
        if len(set(all_spec)) != len(all_spec):
            fail("a spectroscopy JOB ID appears under multiple display indices")
        if set(calibration).intersection(all_spec):
            fail("calibration and spectroscopy JOB IDs overlap")
        title = block["dataset_title"]
        n_match = re.search(r"\bN\s*=\s*(\d+)\b", title)
        disorder = (
            "disorderless" if re.search(r"\bDisorderless\b", title, re.I)
            else "disordered" if re.search(r"\bDisordered\b", title, re.I)
            else None
        )
        block.update(
            folder=folder,
            floquet_config_version=config,
            calibration_job_ids=calibration,
            spectroscopy_job_ids_by_r=spectroscopy,
            documented_total_photons=int(n_match.group(1)) if n_match else None,
            documented_disorder=disorder,
            source_end_line=end_line,
        )
        records.append(block)
        block = None
        field = None

    def start_block(line_number, subgroup_id=None, title=None):
        return dict(
            dataset_id=group["group_id"] + (f".{subgroup_id}" if subgroup_id else ""),
            group_id=group["group_id"],
            group_title=group["group_title"],
            dataset_title=title or group["group_title"],
            nominal_g_kHz=group["nominal_g_kHz"],
            nominal_K_kHz=group["nominal_K_kHz"],
            source_path=str(path),
            source_line=line_number,
            source_field_lines={},
            fields={},
        )

    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        match = _GROUP.fullmatch(line)
        if match:
            finish(line_number - 1)
            group = dict(
                group_id=match[1], group_title=line[2:], source_line=line_number,
                nominal_g_kHz=float(match[2]), nominal_K_kHz=float(match[3]),
            )
            continue
        if group is None:
            if re.match(r"#\s+\d+\.", line):
                raise ValueError(f"{path}:{line_number}: invalid g/K group heading")
            continue  # The introductory explanation is not a dataset.
        match = _SUBGROUP.fullmatch(line)
        if match:
            finish(line_number - 1)
            block = start_block(line_number, match[1], match[2])
            continue
        if line.startswith("#"):
            raise ValueError(f"{path}:{line_number}: unrecognized dataset heading: {line}")
        match = _FIELD.fullmatch(line)
        if match:
            if block is None:
                block = start_block(group["source_line"])
            field = "Folder" if match[1] == "Foler" else match[1]
            if field in block["fields"]:
                raise ValueError(f"{path}:{line_number}: duplicate field {field}")
            block["fields"][field] = [match[2]]
            block["source_field_lines"][field] = line_number
            continue
        if block is None or field is None or line.startswith("-"):
            raise ValueError(f"{path}:{line_number}: unrecognized dataset content: {line}")
        block["fields"][field].append(line)
    finish(len(lines))
    if not records:
        raise ValueError(f"{path}: no documented datasets found")
    if len({record["dataset_id"] for record in records}) != len(records):
        raise ValueError(f"{path}: duplicate dataset section identifiers")
    return records
