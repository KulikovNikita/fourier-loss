import torch
import typing

from .integration import integrate
from .custom_typing import FPTensor
from .fourier import normalize_by_samples, SinCos
from .compute_fourier import compute_fourier_batched
from .evaluate_fourier import evaluate_fourier_batched

#@torch.compile
def extract_scores(scores: FPTensor, targets: torch.BoolTensor) -> FPTensor:
    indices: torch.LongTensor = torch.nonzero(targets)[:, 0]
    result: FPTensor = scores.squeeze(-1)[indices]
    return result

#@torch.compile
def split_scores(scores: FPTensor, target: torch.BoolTensor) -> typing.Tuple[FPTensor, FPTensor]:
    pos_scores: FPTensor = extract_scores(scores, target)
    neg_scores: FPTensor = extract_scores(scores, ~target)
    return (neg_scores, pos_scores)

#@torch.compile
def integrate_curve_raw(xs: FPTensor, ys: FPTensor) -> FPTensor:
    return torch.sum(ys[:-1] * torch.diff(xs))

#@torch.compile
def integrate_roc_auc(fpr: FPTensor, tpr: FPTensor) -> FPTensor:
    fpr_first: FPTensor = integrate_curve_raw(fpr, tpr)
    tpr_first: FPTensor = integrate_curve_raw(tpr, fpr)
    return 0.5 * (1.0 + fpr_first - tpr_first)

#@torch.compile
class ColumnRocAuc(torch.nn.Module):
    def __init__(self, n_harmonics: int = 128, 
                 n_points: int = 32_768, batch_size: int = 16_384,) -> None:
        super().__init__()

        self.n_points: int = n_points
        self.batch_size: int = batch_size
        self.n_harmonics: int = n_harmonics
    
    def compute_and_integrate(self, scores: FPTensor) -> SinCos:
        raw_fourier: SinCos = compute_fourier_batched(x = scores, 
            n_harmonics = self.n_harmonics, batch_size = self.batch_size)
        norm_fourier: SinCos = normalize_by_samples(inp = raw_fourier)
        int_fourier: SinCos = integrate(inp = norm_fourier)
        return normalize_by_samples(inp = int_fourier)
    
    def evaluate_rate(self, x: FPTensor, integral: SinCos) -> FPTensor:
        cdf: FPTensor = evaluate_fourier_batched(x = x, inp = integral, batch_size = self.batch_size)
        zero: torch.Tensor = torch.zeros(size = (1,), device = x.device, dtype = x.dtype)
        one: torch.Tensor = torch.ones(size = (1,), device = x.device, dtype = x.dtype)
        return torch.hstack([one, (1.0 - 0.5 * (cdf + 0.5)).to(x.dtype), zero])

    def rate_like(self, like: FPTensor) -> FPTensor:
        return torch.linspace(start = 0.0, end = 1.0, 
            steps = self.n_points, dtype = like.dtype, device = like.device)

    def curve(self, scores: FPTensor, target: torch.BoolTensor) -> typing.Tuple[FPTensor, FPTensor, FPTensor]:
        pos_scores: FPTensor; neg_scores: FPTensor
        pos_scores, neg_scores = split_scores(scores, target)

        pos_integral: SinCos = self.compute_and_integrate(pos_scores)
        neg_integral: SinCos = self.compute_and_integrate(neg_scores)

        rate: FPTensor = self.rate_like(like = scores)
        fpr: FPTensor = self.evaluate_rate(x = rate, integral = neg_integral)
        tpr: FPTensor = self.evaluate_rate(x = rate, integral = pos_integral)
        return (fpr, tpr, rate)

    def forward(self, scores: FPTensor, target: torch.BoolTensor) -> torch.Tensor:
        tpr: FPTensor; fpr: FPTensor
        target_flat: FPTensor = torch.flatten(target).to(torch.bool)[:, None]
        scores_flat: FPTensor = torch.flatten(scores)[:, None]
        fpr, tpr, _ = self.curve(scores_flat, target_flat)
        return integrate_roc_auc(fpr, tpr)
    
#@torch.compile
class RocAuc(torch.nn.Module):
    def __init__(self, statistic: str = "mean", n_harmonics: int = 128, 
                 n_points: int = 32_768, batch_size: int = 16_384,) -> None:
        super().__init__()

        self.statistic: str = str(statistic)
        assert statistic in {"max", "min", "mean", "flatten"}
        self.roc_auc_column: ColumnRocAuc = ColumnRocAuc(
            batch_size = batch_size, n_points = n_points,
            n_harmonics = n_harmonics,
        )

    def normalize_data(self, scores: FPTensor, target: torch.BoolTensor) -> FPTensor:
        #assert scores.dim() == 2
        #assert scores.size() == target.size()

        norm_scores: FPTensor = scores
        norm_target: torch.BoolTensor = target.to(torch.bool)
        if self.statistic == "flatten":
            norm_scores = torch.flatten(norm_scores)[:, None]
            norm_target = torch.flatten(norm_target)[:, None]
        return (norm_scores, norm_target)
        
    def per_column(self, scores: FPTensor, target: torch.BoolTensor) -> FPTensor:
        column_count: int = scores.size(-1)
        futures: typing.List[torch.futures.Future[FPTensor]] = []
        for column in range(column_count):
            future: torch.futures.Future[FPTensor] = torch.jit.fork(
                func = self.roc_auc_column,
                scores = scores[:, column][:, None], 
                target = target[:, column][:, None],
            )
            futures.append(future)
        results: typing.List[FPTensor] = torch.futures.wait_all(futures)
        return torch.hstack(results)
    
    def compute_statistic(self, per_column: FPTensor) -> FPTensor:
        result: FPTensor
        if self.statistic == "flatten":
            assert torch.numel(per_column) == 1
            result = torch.flatten(per_column)[0]
        elif self.statistic == "min":
            result = torch.min(per_column)
        elif self.statistic == "max":
            result = torch.max(per_column)
        elif self.statistic == "mean":
            result = torch.mean(per_column)
        else:
            raise ValueError("Unsupported statistic")
        assert torch.numel(result) == 1
        return result

    def forward(self, scores: FPTensor, target: torch.BoolTensor) -> FPTensor:
        norm_scores: FPTensor; norm_target: torch.BoolTensor
        norm_scores, norm_target = self.normalize_data(scores, target)
        per_column: FPTensor = self.per_column(norm_scores, norm_target)
        return self.compute_statistic(per_column)
