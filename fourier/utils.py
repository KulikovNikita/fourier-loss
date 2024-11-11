import torch

import numpy as np
import polars as pl
import functools as ft

@ft.lru_cache
def is_to_torch_available(version: str = pl.__version__) -> bool:
    if "0.20.23" <= version:
        dummy: pl.Series = pl.Series(values = [0])
        if hasattr(dummy, "to_torch"):
            return True
    return False

def to_torch(series: pl.Series) -> torch.Tensor:
    result: torch.Tensor
    if is_to_torch_available():
        result = series.to_torch()
    else:
        temp: np.ndarray = series.to_numpy()
        result = torch.asarray(temp)
    assert torch.numel(result) == len(series)
    return result