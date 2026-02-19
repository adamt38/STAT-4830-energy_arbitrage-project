"""Test that the venv exists and script/gd_1d_torch.py runs and produces the figure."""
import os
import pathlib
import subprocess

def _venv_python() -> pathlib.Path:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if os.name == "nt":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def test_gd_1d_torch_runs_and_produces_figure():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    python = _venv_python()
    script = repo_root / "script" / "gd_1d_torch.py"
    figure = repo_root / "figures" / "gd_torch_quadratic_diagnostics.png"

    assert python.exists(), f"Venv Python not found: {python}"
    assert script.exists(), f"Script not found: {script}"

    if figure.exists():
        figure.unlink()

    env = os.environ.copy()
    mpl_dir = repo_root / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)

    result = subprocess.run(
        [str(python), str(script)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr or result.stdout}"

    assert figure.exists(), f"Figure not produced: {figure}"
