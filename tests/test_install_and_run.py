"""Smoke test: core imports resolve (no GPU / network)."""

from pathlib import Path


def test_core_packages_import():
    repo_root = Path(__file__).resolve().parent.parent
    assert (repo_root / "src" / "baseline.py").is_file()
    assert (repo_root / "src" / "polymarket_data.py").is_file()
    assert (repo_root / "src" / "constrained_optimizer.py").is_file()

    import src.baseline  # noqa: F401
    import src.polymarket_data  # noqa: F401
    import src.constrained_optimizer  # noqa: F401
