"""Load company-generated die-level CSVs for Model A.

block_readings is never read into memory here: it's a ~2000-value
space-separated string per row and dominates file size (it's the
difference between train.csv being ~800KB/wafer vs a few KB/wafer).
Model A does not use it, so we drop it at the pd.read_csv(usecols=...)
level rather than loading and discarding it.
"""
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"


def load_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _usecols_excluding(path: Path, excluded_columns: list[str]) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return [c for c in header if c not in excluded_columns]


def load_csv(path: Path, excluded_columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    usecols = _usecols_excluding(path, excluded_columns)
    return pd.read_csv(path, usecols=usecols)


def load_train(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    return load_csv(_REPO_ROOT / config["data"]["train_path"],
                     config["model_a"]["excluded_columns"])


def load_test(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    return load_csv(_REPO_ROOT / config["data"]["test_path"],
                     config["model_a"]["excluded_columns"])


def load_validation(config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    return load_csv(_REPO_ROOT / config["data"]["validation_path"],
                     config["model_a"]["excluded_columns"])
