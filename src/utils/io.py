from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write a dataframe.

    Prefers Parquet, but falls back to CSV when Parquet engine isn't available
    (common on Python versions without pyarrow wheels).
    """
    ensure_parent_dir(path)
    try:
        df.to_parquet(path, index=False)
        return
    except Exception:
        # Fall back to CSV with same base filename.
        csv_path = os.path.splitext(path)[0] + ".csv"
        ensure_parent_dir(csv_path)
        df.to_csv(csv_path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    """Read a dataframe.

    Reads Parquet when present; otherwise attempts CSV with the same base name.
    """
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    csv_path = os.path.splitext(path)[0] + ".csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def write_json(obj: Any, path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
