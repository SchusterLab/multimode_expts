"""The vault day file: entry ids, notes anchors and a safe note writer.

Day file schema (vault schema 2, docs/lab_workflow_tooling_spec.md item B):

- YAML front matter, then an H1, then one ``##`` section per entry.
- Each entry has an id and a kind (``job``, ``analysis``, ``entry``; ``group``
  is reserved for spec item E). A job entry's id is its job ID; the other ids
  are ``ANA-``/``ENT-`` plus the date and time. Ids are unique in a day file.
- The line under the heading is the id with an Obsidian block id, so another
  note can link to the entry: ``[[guan/AGP/2026/10/2026-10-28#^JOB-20261028-00272]]``.
- The entry ends with its notes block between two anchors::

      <!-- notes:JOB-20261028-00272 -->

      (guan) The first note.

      (Claude) The second note.

      <!-- /notes:JOB-20261028-00272 -->

  A note starts with its author in brackets. A withdrawn note is not deleted;
  it gets ``[withdrawn <date>: <why>]`` after the author.

Entries are only appended (``append_entry``). Notes are written with
``add_note``: the new file is written to a temp file in the same folder and
moved over the day file with ``os.replace``; if the day file changed in the
meantime (a logger append, an Obsidian edit), the write is done again on the
new text. A change in the milliseconds between the last check and the replace
can still be lost. Raw data is in the HDF5 files, not here, so that is
acceptable.
"""

import os
import re
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Optional

SCHEMA_VERSION = 2
PLACEHOLDER = "_add interpretation here_"
JOB_ID_RE = re.compile(r"JOB-\d{8}-\d{5}")
ENTRY_ID_DATE_RE = re.compile(r"-(\d{8})-")


class VaultError(RuntimeError):
    """A vault day file does not have what a write needs."""


def notes_open(entry_id: str) -> str:
    return f"<!-- notes:{entry_id} -->"


def notes_close(entry_id: str) -> str:
    return f"<!-- /notes:{entry_id} -->"


def format_note(text: str, author: str) -> str:
    return f"({author}) {text.strip()}"


def notes_block(entry_id: str, notes: str = "") -> str:
    """-> the ``### Notes`` section of an entry, with its two anchors."""
    body = notes.strip() or PLACEHOLDER
    return f"### Notes\n\n{notes_open(entry_id)}\n\n{body}\n\n{notes_close(entry_id)}\n\n"


def id_line(entry_id: str) -> str:
    """-> the line under the heading: the id, with an Obsidian block id."""
    return f"`{entry_id}` ^{entry_id}\n\n"


def job_id_in(value) -> Optional[str]:
    """-> the first job ID in a path, a string or a list of them, or None."""
    items = value if isinstance(value, (list, tuple)) else [value]
    for item in items:
        match = JOB_ID_RE.search(str(item))
        if match:
            return match.group(0)
    return None


def entry_date(entry_id: str) -> Optional[date]:
    """-> the date in an entry id (``JOB-20261028-00272`` -> 2026-10-28), or None."""
    match = ENTRY_ID_DATE_RE.search(entry_id)
    if not match:
        return None
    digits = match.group(1)
    return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))


def has_entry(text: str, entry_id: str) -> bool:
    return notes_open(entry_id) in text


def unique_entry_id(text: str, base: str) -> str:
    """-> ``base``, or ``base-2``, ``base-3``... if the day file has it already."""
    if not has_entry(text, base):
        return base
    k = 2
    while has_entry(text, f"{base}-{k}"):
        k += 1
    return f"{base}-{k}"


def append_entry(day_file: Path, entry_text: str) -> None:
    """Append one fully built entry to the day file, in one write."""
    with Path(day_file).open("a", encoding="utf-8") as handle:
        handle.write(entry_text)


def _rewrite(day_file: Path, change: Callable[[str], str], retries: int = 20) -> None:
    """Apply ``change`` to the day file text; write again if the file changed meanwhile.

    On Windows, ``os.replace`` fails with PermissionError while another writer
    (the logger, Obsidian, the Drive client) has the file open; that is also
    tried again.
    """
    day_file = Path(day_file)
    for _ in range(retries):
        before = day_file.read_bytes()
        new_text = change(before.decode("utf-8").replace("\r\n", "\n"))
        fd, tmp = tempfile.mkstemp(dir=day_file.parent, prefix=f".{day_file.stem}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(new_text)
            if day_file.read_bytes() != before:
                continue
            try:
                os.replace(tmp, day_file)
            except PermissionError:
                time.sleep(0.05)
                continue
            return
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    raise VaultError(f"{day_file} changed or was busy during {retries} note writes; note not written")


def _block_span(text: str, entry_id: str) -> tuple:
    """-> (start, end) of the text between the two notes anchors of ``entry_id``."""
    opening, closing = notes_open(entry_id), notes_close(entry_id)
    if text.count(opening) != 1 or text.count(closing) != 1:
        raise VaultError(
            f"notes anchor {entry_id}: found {text.count(opening)} opening and "
            f"{text.count(closing)} closing anchors, need one of each"
        )
    start = text.index(opening) + len(opening)
    end = text.index(closing)
    if end < start:
        raise VaultError(f"notes anchor {entry_id}: closing anchor before opening anchor")
    return start, end


def _notes(body: str) -> list:
    """-> the notes (paragraphs) of a notes block body, without the placeholder."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    return [p for p in paragraphs if p != PLACEHOLDER]


def _join(notes: Iterable[str]) -> str:
    notes = list(notes) or [PLACEHOLDER]
    return "\n\n" + "\n\n".join(notes) + "\n\n"


def add_note(day_file: Path, entry_id: str, text: str, author: str) -> Path:
    """Add a note at the end of the notes block of entry ``entry_id``."""
    if not text.strip():
        raise ValueError("empty note")

    def change(old: str) -> str:
        start, end = _block_span(old, entry_id)
        notes = _notes(old[start:end]) + [format_note(text, author)]
        return old[:start] + _join(notes) + old[end:]

    _rewrite(day_file, change)
    return Path(day_file)


def withdraw_note(day_file: Path, entry_id: str, match: str, why: str,
                  on: Optional[date] = None) -> Path:
    """Mark the one note of entry ``entry_id`` that contains ``match`` as withdrawn."""
    stamp = f"[withdrawn {(on or date.today()).isoformat()}: {why.strip()}]"

    def change(old: str) -> str:
        start, end = _block_span(old, entry_id)
        notes = _notes(old[start:end])
        hits = [k for k, note in enumerate(notes) if match in note]
        if len(hits) != 1:
            raise VaultError(f"entry {entry_id}: {len(hits)} notes contain {match!r}, need one")
        note = notes[hits[0]]
        author = re.match(r"\([^)]*\) ", note)
        cut = author.end() if author else 0
        notes[hits[0]] = note[:cut] + stamp + " " + note[cut:]
        return old[:start] + _join(notes) + old[end:]

    _rewrite(day_file, change)
    return Path(day_file)


def _day_files(user_dir: Path, entry_id: str) -> list:
    """-> the day files under ``user_dir`` for the date in the id, one per project folder."""
    day = entry_date(entry_id)
    if day is None:
        return []
    return sorted(Path(user_dir).glob(f"*/{day:%Y}/{day:%m}/{day:%Y-%m-%d}.md"))


def find_entry_file(user_dir: Path, entry_id: str) -> Optional[Path]:
    """-> the day file under ``user_dir`` (``<vault>/Lab/<user>``) that has the entry."""
    for path in _day_files(user_dir, entry_id):
        if has_entry(path.read_text(encoding="utf-8"), entry_id):
            return path
    return None


def job_range(job_ids: list) -> str:
    """-> ``JOB-20261028-00272 to -00281 (10 jobs)`` for consecutive ids of one day, else a list."""
    if len(job_ids) == 1:
        return job_ids[0]
    days = {jid[:12] for jid in job_ids}
    numbers = sorted(int(jid[-5:]) for jid in job_ids)
    if len(days) == 1 and numbers == list(range(numbers[0], numbers[0] + len(numbers))):
        first, last = min(job_ids), max(job_ids)
        return f"{first} to -{last[-5:]} ({len(job_ids)} jobs)"
    return ", ".join(job_ids) + f" ({len(job_ids)} jobs)"


def job_links(lab_dir: Path, user_dir: Path, job_ids: list) -> str:
    """-> one Obsidian link per job, to its entry, or to its day file for an entry
    without anchors (written before vault schema 2), or the bare number if not found."""
    texts: dict = {}
    links = []
    for jid in job_ids:
        target = None
        for path in _day_files(user_dir, jid):
            if path not in texts:
                texts[path] = path.read_text(encoding="utf-8")
            rel = path.relative_to(lab_dir).with_suffix("").as_posix()
            if has_entry(texts[path], jid):
                target = f"{rel}#^{jid}"
                break
            if jid in texts[path] and target is None:
                target = rel
        links.append(f"[[{target}|{jid[-5:]}]]" if target else jid[-5:])
    return " · ".join(links)


def analysis_body(lab_dir: Path, user_dir: Path, job_ids: list, source: Optional[str]) -> str:
    """-> the markdown of an analysis entry between its figures and its notes."""
    lines = []
    if job_ids:
        lines.append(f"**Jobs:** {job_range(job_ids)}\\")
        lines.append(job_links(lab_dir, user_dir, job_ids))
    if source:
        lines.append("")
        lines.append(f"**Source:** {source}")
    return "\n".join(lines)
