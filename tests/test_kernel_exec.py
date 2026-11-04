"""tools/kernel_exec.py: the cell splitter, and a run against a throwaway kernel (spec item A)."""
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kernel_exec  # noqa: E402

SCRIPT = '''# ---
# jupyter:
#   jupytext: {}
# ---

# %% [markdown]
# # Notes

# %% Cell 1: station
x = 1

# %% Cell 1b: run
y = x + 1
# %% Cell 10: last
z = "# %% not a marker inside a string? it is at line start only"
#%% Cell 11
print(z)'''


@pytest.fixture
def cells():
    return kernel_exec.split_cells(SCRIPT)


def test_split_cells(cells):
    assert [(c.index, c.kind, c.title) for c in cells] == [
        (0, "code", "(header)"), (1, "markdown", ""), (2, "code", "Cell 1: station"),
        (3, "code", "Cell 1b: run"), (4, "code", "Cell 10: last"), (5, "code", "Cell 11")]
    assert cells[2].code == "x = 1\n" and cells[2].line == 9
    assert cells[3].code == "y = x + 1\n"
    assert cells[5].code == "print(z)\n"  # the last cell ends the file without a newline


def test_select_cell(cells):
    assert kernel_exec.select_cell(cells, "Cell 1").index == 2
    assert kernel_exec.select_cell(cells, "Cell 1b").index == 3
    assert kernel_exec.select_cell(cells, "Cell 11").index == 5
    assert kernel_exec.select_cell(cells, "4").index == 4
    with pytest.raises(kernel_exec.KernelExecError, match="markdown"):
        kernel_exec.select_cell(cells, "1")
    with pytest.raises(kernel_exec.KernelExecError, match="0 cells match"):
        kernel_exec.select_cell(cells, "Cell 2")
    with pytest.raises(kernel_exec.KernelExecError, match="no cell 9"):
        kernel_exec.select_cell(cells, "9")


@pytest.fixture(scope="module")
def kernel():
    """A throwaway kernel on the live server, deleted after the tests."""
    try:
        server = kernel_exec.Server()
    except kernel_exec.KernelExecError as exc:
        pytest.skip(str(exc))
    name = f"pytest-kernel-exec-{uuid.uuid4().hex[:6]}"
    repo = Path(kernel_exec.__file__).resolve().parents[1]
    assert kernel_exec.main(["start", "--name", name, "--dir", str(repo)]) == 0
    try:
        yield name
    finally:
        session = server.find(name)
        server.api("DELETE", f"api/sessions/{session['id']}")


def test_run_code_and_script_cell(kernel, tmp_path, capsys):
    log = tmp_path / "k.log"
    assert kernel_exec.main(["run", "--kernel", kernel, "--code", "print(6 * 7)",
                             "--log", str(log)]) == 0
    assert "42" in capsys.readouterr().out and "42" in log.read_text(encoding="utf-8")

    script = tmp_path / "s.py"
    script.write_text(SCRIPT + "\n\n# %% Cell 12: source\n"
                      "import __main__\nprint(__main__.__kernel_exec_source__)\n", encoding="utf-8")
    assert kernel_exec.main(["run", "--kernel", kernel, "--script", str(script),
                             "--cell", "Cell 12", "--log", str(log)]) == 0
    assert "s.py, cell Cell 12: source" in capsys.readouterr().out
    kernel_exec.main(["run", "--kernel", kernel, "--code", "print(__kernel_exec_source__)",
                      "--log", str(log)])
    assert capsys.readouterr().out.endswith(": code\nNone\n")


def test_run_error_and_timeout(kernel, tmp_path, capsys):
    log = tmp_path / "k.log"
    assert kernel_exec.main(["run", "--kernel", kernel, "--code", "1/0", "--log", str(log)]) == 1
    assert "ZeroDivisionError" in capsys.readouterr().err
    assert kernel_exec.main(["run", "--kernel", kernel, "--code", "import time; time.sleep(3)",
                             "--timeout", "0.5", "--log", str(log)]) == 2
    assert kernel_exec.main(["interrupt", "--kernel", kernel]) == 0


def test_run_no_wait_writes_the_log(kernel, tmp_path):
    import time

    log = tmp_path / "k.log"
    assert kernel_exec.main(["run", "--kernel", kernel, "--code",
                             "import time; time.sleep(1); print('done later')",
                             "--no-wait", "--log", str(log)]) == 0
    for _ in range(60):
        if log.exists() and "done later" in log.read_text(encoding="utf-8"):
            break
        time.sleep(0.5)
    assert "done later" in log.read_text(encoding="utf-8")
