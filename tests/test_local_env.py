"""The repo-root .env loader: format, precedence, and absence.

Every test points ``MULTIMODE_ENV_FILE`` at its own tmp_path file and resets
the module's one-shot flag, so none of them read the developer's real .env.
"""
import pytest

from experiments import local_env


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(local_env, "_loaded", False)
    yield


def _file(tmp_path, text, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(text)
    monkeypatch.setenv(local_env.ENV_FILE_VAR, str(path))
    return path


def test_sets_unset_variables(tmp_path, monkeypatch):
    _file(tmp_path, "MULTIMODE_DATA_ROOT=/Volumes/pippin/experiments\n", monkeypatch)
    monkeypatch.delenv("MULTIMODE_DATA_ROOT", raising=False)

    applied = local_env.load_env()

    import os
    assert applied == {"MULTIMODE_DATA_ROOT": "/Volumes/pippin/experiments"}
    assert os.environ["MULTIMODE_DATA_ROOT"] == "/Volumes/pippin/experiments"


def test_real_environment_wins(tmp_path, monkeypatch):
    """The file fills gaps only -- otherwise a one-off override could not work."""
    _file(tmp_path, "MULTIMODE_PATH_BACKEND=vault\n", monkeypatch)
    monkeypatch.setenv("MULTIMODE_PATH_BACKEND", "index")

    assert local_env.load_env() == {}

    import os
    assert os.environ["MULTIMODE_PATH_BACKEND"] == "index"


def test_absent_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv(local_env.ENV_FILE_VAR, str(tmp_path / "nope.env"))
    assert local_env.load_env() == {}


def test_second_call_does_not_reread(tmp_path, monkeypatch):
    path = _file(tmp_path, "MULTIMODE_X=1\n", monkeypatch)
    monkeypatch.delenv("MULTIMODE_X", raising=False)
    monkeypatch.delenv("MULTIMODE_Y", raising=False)
    local_env.load_env()

    path.write_text("MULTIMODE_Y=2\n")
    assert local_env.load_env() == {}
    assert local_env.load_env(force=True) == {"MULTIMODE_Y": "2"}


def test_parses_comments_quotes_and_export(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "  SPACED = spaced value \n"
        'QUOTED="G:/Shared drives/SLab"\n'
        "SINGLE='guan'\n"
        "export EXPORTED=yes\n"
        "URL=https://x/y?a=b\n"
        "junk line without equals\n"
    )
    assert local_env.parse_env_file(path) == {
        "PLAIN": "value",
        "SPACED": "spaced value",
        "QUOTED": "G:/Shared drives/SLab",
        "SINGLE": "guan",
        "EXPORTED": "yes",
        "URL": "https://x/y?a=b",
    }
