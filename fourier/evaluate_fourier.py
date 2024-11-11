import torch

import typing

from .custom_typing import FPTensor
from .fourier import SinCos, DEFAULT_FACTOR

def _evaluate_fourier_out(x: FPTensor, 
                          inp: SinCos, out: FPTensor, 
                          factor: float) -> FPTensor:
    assert x.size() == out.size()

    factorized: torch.Tensor = factor * x
    sin_x: FPTensor = torch.sin(factorized)
    cos_x: FPTensor = torch.cos(factorized)

    temp: FPTensor
    sin_term: FPTensor; cos_term: FPTensor
    cos_kx: FPTensor = torch.ones_like(x)
    sin_kx: FPTensor = torch.zeros_like(x)

    n_harmonics: int = inp.n_harmonics
    assert 0 < n_harmonics

    out += (inp.n_samples * 0.5)
    for k in range(1, n_harmonics):
        temp = sin_kx.clone()
        sin_kx = sin_kx * cos_x + cos_kx * sin_x
        cos_kx = cos_kx * cos_x - temp * sin_x

        sin_term = inp.sin[..., k] * sin_kx
        cos_term = inp.cos[..., k] * cos_kx
        out += (sin_term + cos_term)

    return out

def evaluate_fourier(x: FPTensor, inp: SinCos, 
                     out: typing.Optional[FPTensor] = None,
                     factor: float = DEFAULT_FACTOR,) -> FPTensor:
    if out is None:
        out = torch.zeros_like(x)
    return _evaluate_fourier_out(x, inp, out, factor)
