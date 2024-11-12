import torch

import typing

from .custom_typing import FPTensor
from .fourier import SinCos, DEFAULT_FACTOR

def _compute_fourier_out(x: FPTensor, out: SinCos,
                         factor: float) -> SinCos:
    factorized: FPTensor = factor * x
    sin_x: FPTensor = torch.sin(factorized)
    cos_x: FPTensor = torch.cos(factorized)

    temp: FPTensor
    cos_kx: FPTensor = torch.ones_like(x)
    sin_kx: FPTensor = torch.zeros_like(x)

    n_harmonics: int = out.n_harmonics

    for k in range(n_harmonics):
        out.sin[..., k] = torch.sum(sin_kx, dim = 0)
        out.cos[..., k] = torch.sum(cos_kx, dim = 0)

        temp = sin_kx.clone()
        sin_kx = sin_kx * cos_x + cos_kx * sin_x
        cos_kx = cos_kx * cos_x - temp * sin_x

    return out

def _make_sincos(x: FPTensor, n_harmonics: int) -> SinCos:
    dtype: torch.dtype = x.dtype
    device: torch.device = x.device
    parent: typing.Sequence[int] = x.size()[1:]
    size: typing.Sequence[int] = (*parent, n_harmonics)
    def make_empty(size: typing.Sequence[int]) -> FPTensor:
        return torch.empty(size, dtype = dtype, device = device)
    sin: FPTensor = make_empty(size = size)
    cos: FPTensor = make_empty(size = size)
    return SinCos(sin, cos)

def compute_fourier(x: FPTensor, 
                    n_harmonics: typing.Optional[int] = None, 
                    out: typing.Optional[SinCos] = None,
                    factor: float = DEFAULT_FACTOR) -> SinCos:
    if n_harmonics is None:
        assert out is not None
        n_harmonics = out.n_harmonics
    if out is None:
        assert n_harmonics is not None
        out = _make_sincos(x, n_harmonics)
    assert n_harmonics == out.n_harmonics
    return _compute_fourier_out(x, out, factor)

def _compute_fourier(x: FPTensor, n_harmonics: int, factor: float) -> typing.Tuple[FPTensor, FPTensor]:
    result: SinCos = compute_fourier(x = x, n_harmonics = n_harmonics, factor = factor)
    return (result.sin, result.cos)

def compute_fourier_batched(x: FPTensor, 
                            n_harmonics: int, 
                            batch_size: int = 16_384,
                            factor: float = DEFAULT_FACTOR) -> SinCos:
    length: int = x.size(0)
    batch_count: int = length // batch_size
    batch_count += bool(length % batch_size)

    # Shortcut
    if batch_count == 1:
        return compute_fourier(x = x, factor = factor,
                               n_harmonics = n_harmonics,) 

    futures: typing.List[torch.futures.Future[typing.Tuple[FPTensor, FPTensor]]] = []
    for batch in range(batch_count):
        first: int = batch * batch_size
        last: int = min(first + batch_size, length)
        x_slice: FPTensor = x[first : last, ...]
        future: torch.futures.Future[typing.Tuple[FPTensor, FPTensor]] = torch.jit.fork(
            func = _compute_fourier, n_harmonics = n_harmonics, 
            x = x_slice, factor = factor,
        )
        futures.append(future)
    results: typing.List[typing.Tuple[FPTensor, FPTensor]] = torch.futures.wait_all(futures)
    results: typing.List[SinCos] = [SinCos(s, c) for s, c in results]
    return sum(results)
