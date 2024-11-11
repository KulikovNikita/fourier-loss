import torch
import polars as pl

from dataclasses import dataclass
from joblib import delayed, Parallel
from typing import Any, List, Tuple, Optional

from integration import integrate
from compute_fourier import compute_fourier
from evaluate_fourier import evaluate_fourier
from custom_typing import DeviceLike, FPTensor
from fourier import normalize_by_samples, SinCos

DEFAULT_HARMONIC_COUNT: int = 128

@dataclass
class FourierProcessor:
    n_harmonics: int
    dtype: torch.dtype
    device: torch.device
    n_jobs: Optional[int]
    
    chunk_size: int = 16_384

    def __init__(self,
                 n_harmonics: int = DEFAULT_HARMONIC_COUNT,
                 n_jobs: Optional[int] = None, 
                 device: DeviceLike = "cpu",
                 dtype = torch.float32,) -> None:
        assert dtype in {torch.float32, torch.float64}
        self.device = torch.device(device)
        self.n_harmonics = n_harmonics
        self.n_jobs = n_jobs
        self.dtype = dtype

    def _asarray(self, obj: Any) -> FPTensor:
        return torch.asarray(
            obj = obj, 
            dtype = self.dtype, 
            device = self.device,
        )

    def _fit_min_max(self, df: pl.DataFrame) -> Tuple[FPTensor, FPTensor]:
        features: List[str] = df.columns
        mins: pl.DataFrame = df.select(pl.col(*features).min())
        maxs: pl.DataFrame = df.select(pl.col(*features).max())
        min_res: FPTensor = self._asarray(obj = mins.row(0))
        max_res: FPTensor = self._asarray(obj = maxs.row(0))
        return (min_res, max_res)

    def _get_epsilon(self) -> FPTensor:
        epsilon: float = torch.finfo(self.dtype).eps
        return self._asarray(epsilon)

    def _normalize_chunk(self, chunk: FPTensor, min_t: FPTensor, max_t: FPTensor) -> FPTensor:
        min_max_diff: FPTensor = max_t - min_t
        epsilon: FPTensor = self._get_epsilon()
        denom: FPTensor = torch.maximum(min_max_diff, epsilon)
        result: FPTensor = (chunk - min_t) / denom
        return result

    def _fit_chunk(self, df: pl.DataFrame, min_t: FPTensor, max_t: FPTensor) -> SinCos:
        chunk: FPTensor = self._asarray(df.to_numpy())
        normalized: FPTensor = self._normalize_chunk(chunk, min_t, max_t)
        return compute_fourier(normalized, n_harmonics = self.n_harmonics)

    def _get_chunks(self, df: pl.DataFrame) -> List[pl.DataFrame]:
        length: int
        length, _ = df.shape
        n_full_chunks: int = length // self.chunk_size
        full_chunks_length: int = n_full_chunks * self.chunk_size
        is_full_length: bool = bool(length - full_chunks_length)
        n_chunks: int = n_full_chunks + is_full_length

        def get_chunk(i: int) -> pl.DataFrame:
            first: int = i * self.chunk_size
            full_last: int = (i + 1) * self.chunk_size
            last: int = min(length, full_last)
            return df.slice(first, last - first)

        return [get_chunk(i) for i in range(n_chunks)]

    def _fit(self, df: pl.DataFrame) -> Tuple[SinCos, FPTensor, FPTensor]:
        min_t: FPTensor; max_t: FPTensor
        min_t, max_t = self._fit_min_max(df)
        chunks: List[df.DataFrame] = self._get_chunks(df)
        parallel: Parallel = Parallel(n_jobs = self.n_jobs)
        proto: Any = lambda c: self._fit_chunk(c, min_t, max_t) 
        chunk_res: List[SinCos] = parallel(delayed(proto)(c) for c in chunks)
        integrated: SinCos = self._integrate_image(sum(chunk_res))
        return (integrated, min_t, max_t)
    
    def _integrate_image(self, image: SinCos) -> SinCos:
        integrated: SinCos = integrate(image)
        return normalize_by_samples(integrated)
    
    def fit(self, df: pl.DataFrame) -> "TrainedFourierProcessor":
        integrated: SinCos; min_t: FPTensor; max_t: FPTensor
        features: List[str] = [str(f) for f in df.schema]
        integrated, min_t, max_t = self._fit(df[features])
        return TrainedFourierProcessor(
            features = features,
            integrated = integrated,
            min_t = min_t,
            max_t = max_t,
            n_harmonics = self.n_harmonics,
            chunk_size = self.chunk_size,
            n_jobs = self.n_jobs,
            device = self.device,
            dtype = self.dtype,
        )
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        transformer: "TrainedFourierProcessor" = self.fit(df)
        return transformer.transform(df)

@dataclass
class TrainedFourierProcessor(FourierProcessor):
    features: Optional[List[str]] = None
    integrated: Optional[SinCos] = None
    min_t: Optional[FPTensor] = None
    max_t: Optional[FPTensor] = None

    def fit(self, df: pl.DataFrame) -> "TrainedFourierProcessor":
        raise RuntimeError("Preprocessor is already trained")
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        selected: df.DataFrame = df[self.features]
        chunks: List[df.DataFrame] = self._get_chunks(selected)
        parallel: Parallel = Parallel(n_jobs = self.n_jobs)
        proto: Any = lambda c: self._transform_chunk(c)
        chunk_res: List[pl.DataFrame] = parallel(
                delayed(proto)(c) for c in chunks)
        return pl.concat(chunk_res)

    def _normalize_chunk(self, chunk: FPTensor) -> FPTensor:
        return super()._normalize_chunk(chunk, self.min_t, self.max_t)

    def _transform_raw_chunk(self, chunk: FPTensor) -> FPTensor:
        normalized: FPTensor = self._normalize_chunk(chunk)
        denormalized: FPTensor =  evaluate_fourier(normalized, self.integrated)
        return (denormalized + 0.5) * 0.5

    def _transform_chunk(self, df: pl.DataFrame) -> pl.DataFrame:
        chunk: FPTensor = self._asarray(df.to_numpy())
        raw: FPTensor = self._transform_raw_chunk(chunk)
        local: Any = raw.detach().cpu().numpy()
        return pl.DataFrame(data = local, schema = self.features)