import torch

import typing
import dataclasses

from .custom_typing import FPTensor

DEFAULT_FACTOR: float = 0.5 * torch.pi

@dataclasses.dataclass
class SinCos:
    sin: FPTensor
    cos: FPTensor

    def __init__(self, sin: FPTensor, cos: FPTensor) -> None:
        self.__check_dim(sin, cos)
        assert sin.dtype == cos.dtype
        assert sin.device == cos.device
        self.sin, self.cos = sin, cos

    @staticmethod
    def __check_dim(sin: FPTensor, cos: FPTensor) -> None:
        assert sin.size() == cos.size()
        assert 0 < sin.size(-1)

    @property
    def n_harmonics(self) -> int:
        self.__check_dim(self.sin, self.cos)
        return self.sin.size(-1)
    
    def size(self, *args) -> typing.Sequence[int]:
        self.__check_dim(self.sin, self.cos)
        return self.sin.size(*args)

    @property
    def n_samples(self) -> FPTensor:
        self.__check_dim(self.sin, self.cos)
        return self.cos[..., 0]
    
    def __add__(self, other: typing.Union[int, "SinCos"]) -> "SinCos":
        if isinstance(other, int): 
            if other == 0:
                return self
        return SinCos(
            sin = (self.sin + other.sin),
            cos = (self.cos + other.cos),
        )
        
    def __radd__(self, other: typing.Union[int, "SinCos"]) -> "SinCos":
        return self.__add__(other)

def _empty_like(inp: SinCos) -> SinCos:
    empty_sin: FPTensor = torch.zeros_like(inp.sin)
    empty_cos: FPTensor = torch.zeros_like(inp.cos)
    return SinCos(empty_sin, empty_cos)

def normalize_by_samples(inp: SinCos, out: typing.Optional[SinCos] = None) -> SinCos:
    if out is None:
        out = _empty_like(inp)
    n_samples: FPTensor = inp.n_samples
    inv_n_samples: FPTensor = 1.0 / n_samples
    out.sin = inp.sin * inv_n_samples[..., None]
    out.cos = inp.cos * inv_n_samples[..., None]
    return out
