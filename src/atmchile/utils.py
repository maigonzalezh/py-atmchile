from __future__ import annotations

from importlib import resources

import pandas as pd


def convert_str_to_list(value: str | list[str]) -> list[str]:
    """
    Convert a string or list of strings to a list of strings.
    """
    if isinstance(value, str):
        return [value]
    return value


def load_package_csv(
    filename: str,
    encoding: str = "utf-8",
    package: str = "atmchile.data",
) -> pd.DataFrame:
    """
    Load a CSV bundled inside the package using importlib.resources.
    """
    data_path = resources.files(package).joinpath(filename)
    with resources.as_file(data_path) as csv_path:
        return pd.read_csv(csv_path, encoding=encoding)
