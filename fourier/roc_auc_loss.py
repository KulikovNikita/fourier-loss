import torch
import typing

from .integration import integrate
from .compute_fourier import compute_fourier
from .evaluate_fourier import evaluate_fourier
from .fourier import normalize_by_samples, SinCos

def _norm_or_pass(fourier: SinCos) -> SinCos:
    eps: float = torch.finfo(fourier.sin.dtype).eps
    zeros: torch.BoolTensor = fourier.n_samples < eps

    result: SinCos
    if torch.any(zeros).detach().cpu().item():
        result = SinCos(
            sin = torch.zeros_like(fourier.sin),
            cos = torch.zeros_like(fourier.cos),
        )
    else:
        result = normalize_by_samples(fourier)

    return result

class RocAucLoss(torch.nn.Module):
    def __init__(self, n_harmonics: int = 128, n_steps: int = 1_024, statistic: str = "flatten") -> None:
        super().__init__()

        self.n_steps: int = n_steps
        self.n_harmonics: int = n_harmonics
        self.statistic: str = statistic.lower()
        assert self.statistic in {"flatten", "min", "max", "mean"}

    def extract_scores(self, scores: torch.Tensor, targets: torch.BoolTensor) -> torch.Tensor:
        indices: torch.LongTensor = torch.nonzero(targets).squeeze(-1)
        return scores.squeeze(-1)[indices]

    def split_scores(self, scores: torch.Tensor, targets: torch.BoolTensor):
        assert scores.ndim == 2
        assert scores.size() == targets.size()
        n_columns: int = scores.size(-1)

        pos_scores: typing.List[torch.Tensor] = []
        neg_scores: typing.List[torch.Tensor] = []

        column: int
        for column in range(n_columns):
            scores_slice: torch.Tensor = scores[:, column]
            targets_slice: torch.Tensor = targets[:, column]

            pos_scores.append(self.extract_scores(scores_slice, targets_slice))
            neg_scores.append(self.extract_scores(scores_slice, ~targets_slice))

        assert len(pos_scores) == len(neg_scores)
        assert len(pos_scores) == n_columns

        return (pos_scores, neg_scores)

    def fourier(self, scores: torch.Tensor) -> SinCos:
        fourier: SinCos = compute_fourier(scores, self.n_harmonics)
        result: SinCos = _norm_or_pass(fourier)
        return result
    
    def integrate(self, fourier: SinCos) -> SinCos:
        integrated: SinCos = integrate(fourier)
        result: SinCos = _norm_or_pass(integrated)
        return result

    def chain_fourier(self, scores: torch.Tensor) -> SinCos:
        fourier: SinCos = self.fourier(scores)
        return self.integrate(fourier)
    
    def pos_integrate(self, pos_cdf: torch.Tensor, neg_cdf: torch.Tensor) -> torch.Tensor:
        diff: torch.Tensor = -torch.diff(neg_cdf)
        return torch.sum(pos_cdf[:-1] * diff)

    def neg_integrate(self, pos_cdf: torch.Tensor, neg_cdf: torch.Tensor) -> torch.Tensor:
        diff: torch.Tensor = torch.diff(pos_cdf)
        return 1.0 + torch.sum(neg_cdf[:-1] * diff)

    def normalize_cdf(self, cdf: torch.Tensor) -> torch.Tensor:
        device: torch.device; dtype: torch.dtype
        device, dtype = cdf.device, cdf.dtype

        one: torch.Tensor = torch.ones(size = (1,), device = device, dtype = dtype)
        zero: torch.Tensor = torch.zeros(size = (1,), device = device, dtype = dtype)
        return torch.hstack([one, (1.0 - 0.5 * (cdf + 0.5)).to(dtype), zero])
    
    def score(self, positive: SinCos, negative: SinCos) -> torch.Tensor:
        device: torch.device = positive.sin.device
        thrs: torch.Tensor = torch.linspace(0.0, 1.0, 
                        steps = self.n_steps, device = device)
        pos_cdf: torch.Tensor = evaluate_fourier(thrs, positive)
        neg_cdf: torch.Tensor = evaluate_fourier(thrs, negative)

        pos_norm: torch.Tensor = self.normalize_cdf(pos_cdf)
        neg_norm: torch.Tensor = self.normalize_cdf(neg_cdf)

        pos_auc: torch.Tensor = self.pos_integrate(pos_norm, neg_norm)
        neg_auc: torch.Tensor = self.neg_integrate(pos_norm, neg_norm)
        return 0.5 * (pos_auc + neg_auc)
    
    def score_columns(self, positive: typing.List[SinCos], negative: typing.List[SinCos]) -> typing.List[torch.Tensor]:
        return [self.score(pos, neg) for pos, neg in zip(positive, negative)]

    def fourier_positives(self, pos_scores: torch.Tensor) -> SinCos:
        return [self.chain_fourier(s) for s in pos_scores]
    
    def fourier_negatives(self, neg_scores: torch.Tensor) -> SinCos:
        return [self.chain_fourier(s) for s in neg_scores]
        
    def per_column(self, scores: torch.Tensor, targets: torch.BoolTensor) -> torch.Tensor:
        pos_scores: typing.List[torch.Tensor]; neg_scores: typing.List[torch.Tensor]
        pos_scores, neg_scores = self.split_scores(scores, targets)

        assert len(pos_scores) == len(neg_scores)

        pos_fourier: typing.List[SinCos] = self.fourier_positives(pos_scores)
        neg_fourier: typing.List[SinCos] = self.fourier_negatives(neg_scores)

        assert len(pos_fourier) == len(neg_fourier)

        roc_aucs: typing.List[torch.Tensor] = self.score_columns(pos_fourier, neg_fourier)
        roc_aucs: torch.Tensor = torch.hstack(roc_aucs)

        assert torch.numel(roc_aucs) == len(pos_fourier)
        
        return roc_aucs
    
    def prepare(self, scores: torch.Tensor, targets: torch.BoolTensor) -> typing.Tuple[torch.Tensor, torch.BoolTensor]:
        new_scores: torch.Tensor; new_targets: torch.BoolTensor
        new_scores, new_targets = scores, targets
        if self.statistic == "flatten":
            new_scores = torch.flatten(new_scores)[:, None]
            new_targets = torch.flatten(new_targets)[:, None]
        return (new_scores, new_targets)
    
    def compute_statistic(self, per_column: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor
        if self.statistic == "flatten":
            assert torch.numel(per_column) == 1
            result = torch.flatten(per_column)
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

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scores, targets = self.prepare(scores, targets.to(torch.bool))
        per_column: torch.Tensor = self.per_column(scores, targets)
        result: torch.Tensor = self.compute_statistic(per_column)
        return result
