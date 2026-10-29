"""Vault day file schema 2: entry ids, notes anchors, add_note, log_analysis (spec items B, C)."""
import threading

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments import vault  # noqa: E402
from experiments.qsim.mbr_campaign import mock_station  # noqa: E402

JOB = "JOB-20261028-00272"
TODAY = f"JOB-{vault.date.today():%Y%m%d}"
TODAY_JOB = f"{TODAY}-00272"


@pytest.fixture
def day_file(tmp_path):
    path = tmp_path / "2026-10-28.md"
    path.write_text("---\ndate: '2026-10-28'\n---\n\n# 2026-10-28\n\n", encoding="utf-8")
    return path


def add_entry(path, entry_id, notes=""):
    vault.append_entry(path, f"## 10:00:00 \u2014 X\n\n{vault.id_line(entry_id)}"
                             f"{vault.notes_block(entry_id, notes)}---\n\n")


def block(path, entry_id):
    text = path.read_text(encoding="utf-8")
    start, end = vault._block_span(text, entry_id)
    return text[start:end].strip()


def test_add_note_replaces_the_placeholder_then_appends(day_file):
    add_entry(day_file, JOB)
    add_entry(day_file, "JOB-20261028-00273")
    assert block(day_file, JOB) == vault.PLACEHOLDER
    vault.add_note(day_file, JOB, "first", "guan")
    vault.add_note(day_file, JOB, "second\n\nwith two paragraphs", "Claude")
    assert block(day_file, JOB) == "(guan) first\n\n(Claude) second\n\nwith two paragraphs"
    assert block(day_file, "JOB-20261028-00273") == vault.PLACEHOLDER


def test_add_note_needs_exactly_one_anchor(day_file):
    with pytest.raises(vault.VaultError, match="found 0 opening"):
        vault.add_note(day_file, JOB, "text", "guan")
    add_entry(day_file, JOB)
    add_entry(day_file, JOB)
    with pytest.raises(vault.VaultError, match="found 2 opening"):
        vault.add_note(day_file, JOB, "text", "guan")


def test_withdraw_note_keeps_the_text(day_file):
    add_entry(day_file, JOB, "(Claude) T1 is 80 us.")
    vault.add_note(day_file, JOB, "Rabi looks clean.", "guan")
    vault.withdraw_note(day_file, JOB, "T1 is", "fit window too short", on=vault.date(2026, 10, 29))
    assert block(day_file, JOB) == ("(Claude) [withdrawn 2026-10-29: fit window too short] T1 is 80 us."
                                    "\n\n(guan) Rabi looks clean.")


def test_add_note_writes_again_after_an_append_in_between(day_file, monkeypatch):
    add_entry(day_file, JOB)
    mkstemp = vault.tempfile.mkstemp
    calls = []

    def append_then_mkstemp(**kwargs):
        if not calls:
            add_entry(day_file, "JOB-20261028-00274")
        calls.append(1)
        return mkstemp(**kwargs)

    monkeypatch.setattr(vault.tempfile, "mkstemp", append_then_mkstemp)
    vault.add_note(day_file, JOB, "note", "guan")
    assert len(calls) == 2
    assert block(day_file, JOB) == "(guan) note"
    assert block(day_file, "JOB-20261028-00274") == vault.PLACEHOLDER
    assert not list(day_file.parent.glob("*.tmp"))


def test_appends_from_a_thread_during_note_writes_are_kept(day_file):
    add_entry(day_file, JOB)
    ids = [f"ENT-20261028-{k:06d}" for k in range(100)]

    def logger():
        for entry_id in ids:
            add_entry(day_file, entry_id)

    thread = threading.Thread(target=logger)
    thread.start()
    for k in range(30):
        vault.add_note(day_file, JOB, f"note {k}", "guan")
    thread.join()
    text = day_file.read_text(encoding="utf-8")
    assert all(vault.has_entry(text, entry_id) for entry_id in ids)
    assert block(day_file, JOB) == "\n\n".join(f"(guan) note {k}" for k in range(30))


def test_ids_and_ranges():
    assert vault.job_id_in(["C:/x/00001_A.h5", f"C:/x/{JOB}_B.h5"]) == JOB
    assert vault.job_id_in("C:/x/00001_A.h5") is None
    assert vault.entry_date("ANA-20261029-141500") == vault.date(2026, 10, 29)
    assert vault.unique_entry_id(vault.notes_open(JOB), JOB) == f"{JOB}-2"
    ladder = [f"JOB-20261028-{n:05d}" for n in range(272, 282)]
    assert vault.job_range(ladder) == "JOB-20261028-00272 to -00281 (10 jobs)"
    assert vault.job_range(ladder[:2] + ladder[5:6]).endswith("(3 jobs)")


@pytest.fixture(scope="module")
def station(tmp_path_factory):
    station = mock_station(user="guan", project="AGP")
    station.vault_root = tmp_path_factory.mktemp("vault")
    return station


def test_log_measurement_writes_schema_2_entries(station):
    station._is_mock = False  # only the vault write; the instruments stay mocks
    try:
        path = station.log_measurement(title="Rabi", parameters={"gain": 1},
                                       data_path=f"C:/data/{TODAY_JOB}_Rabi.h5", notes="looks fine")
        station.log_measurement(title="Rabi", parameters={}, data_path=f"C:/data/{TODAY_JOB}_Rabi.h5")
    finally:
        station._is_mock = True
    text = path.read_text(encoding="utf-8")
    assert "vault_schema: 2" in text
    assert f"`{TODAY_JOB}` ^{TODAY_JOB}" in text and "kind: job" in text
    assert block(path, TODAY_JOB) == "(guan) looks fine"
    assert station.last_vault_entry == (path, f"{TODAY_JOB}-2")
    assert station.log_measurement(title="mock", parameters={}) is None


def test_log_analysis_links_jobs_and_takes_notes(station, monkeypatch):
    station._is_mock = False
    try:
        station.log_measurement(title="Fock 1", parameters={},
                                data_path=f"C:/data/{TODAY}-00273_F.h5")
    finally:
        station._is_mock = True
    import __main__
    monkeypatch.setattr(__main__, "__kernel_exec_source__", "ladder.py, cell 'ladder'", raising=False)
    fig = plt.figure()
    entry_id = station.log_analysis(fig, "Ladder populations", [TODAY_JOB, f"{TODAY}-00273"],
                                    note="P(n) falls as expected.", author="Claude")
    plt.close(fig)
    assert entry_id.startswith("ANA-")
    path = station.last_vault_entry[0]
    text = path.read_text(encoding="utf-8")
    assert f"{TODAY}-00272 to -00273 (2 jobs)" in text
    assert f"[[guan/AGP/{path.parent.parent.name}/{path.parent.name}/{path.stem}#^{TODAY_JOB}|00272]]" in text
    assert "**Source:** ladder.py, cell 'ladder'" in text
    assert "![[figures/" in text
    assert block(path, entry_id) == "(Claude) P(n) falls as expected."
    station.add_note(entry_id, "Agree.", author="guan")
    station.add_note(TODAY_JOB, "From the job.")
    assert block(path, entry_id).endswith("(guan) Agree.")
    assert block(path, TODAY_JOB).endswith("(guan) From the job.")
