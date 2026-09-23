from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "UFC-dataset"
DATA_PATHS = {
    "large_set": DATA_ROOT / "Large-set",
    "stats_dir": DATA_ROOT / "Fighter-stats",
}

def verify_integrity() -> list[str]:
    """Return the names of configured data directories that do not exist."""
    return [name for name, path in DATA_PATHS.items() if not path.is_dir()]

def verify_intergrity() -> list[str]:
    """Backward-compatible alias for the misspelled original function."""
    return verify_integrity()