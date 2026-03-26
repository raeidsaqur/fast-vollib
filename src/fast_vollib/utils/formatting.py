from __future__ import annotations

import json

import numpy as np
import pandas as pd


def format_named_output(values: np.ndarray, name: str, return_as: str):
    array = np.asarray(values)
    if return_as == "series":
        return pd.Series(array, name=name)
    if return_as == "dataframe":
        return pd.DataFrame(array, columns=[name])
    return array


def format_greeks_output(data: dict[str, np.ndarray], return_as: str):
    if return_as == "dataframe":
        return pd.DataFrame(data)
    if return_as == "json":
        return json.dumps({key: np.asarray(value).tolist() for key, value in data.items()})
    return data
