import torch

from functools import lru_cache

from .fourier import SinCos, DEFAULT_FACTOR
from .custom_typing import DeviceLike, FPTensor

@lru_cache
def _to_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    dummy: torch.Tensor = torch.asarray([0,], dtype = dtype)
    return (1j * dummy).dtype

def make_theta_image(n_harmonics: int, 
                     device: DeviceLike = "cpu",
                     dtype: torch.dtype = torch.float32,
                     factor: float = DEFAULT_FACTOR,) -> SinCos:
    device: torch.device = torch.device(device)
    complex_dtype: torch.dtype = _to_complex_dtype(dtype)
    harmonics: torch.Tensor = torch.arange(1, n_harmonics,
                        dtype = dtype, device = device) * factor
    complex_result: torch.Tensor = torch.ones(size = (n_harmonics,), 
                            dtype = complex_dtype, device = device)
    positive: torch.Tensor = torch.exp(+1.0j * harmonics)
    negative: torch.Tensor = torch.exp(-1.0j * harmonics)
    complex_result[1:] = 0.5j * (positive + negative - 2) / harmonics
    return SinCos(
        sin = complex_result.imag.to(dtype = dtype),
        cos = complex_result.real.to(dtype = dtype),
    )

def integrate(inp: SinCos, factor: float = DEFAULT_FACTOR) -> SinCos:
    n_harmonics: int = inp.n_harmonics
    dtype: torch.dtype = inp.sin.dtype
    device: torch.device = inp.sin.device
    theta: SinCos = make_theta_image(n_harmonics, 
        dtype = dtype, device = device, factor = factor)
    sin: FPTensor = inp.sin * theta.cos - inp.cos * theta.sin
    cos: FPTensor = inp.cos * theta.cos + inp.sin * theta.sin
    return SinCos(sin = sin, cos = cos,)
