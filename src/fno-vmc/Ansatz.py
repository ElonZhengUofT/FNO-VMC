from abc import ABC, abstractmethod
import torch.nn as nn
from .fno_model import FNOAnsatz
from .tn_model import TNAnsatz


class BaseAnsatz(ABC, nn.Module):
    """Abstract base class for variational ansatz."""

    @abstractmethod
    def forward(self, x):  # pragma: no cover
        """Compute log ψ for spin configurations x."""
        pass


def make_ansatz(kind: str, dim: int, **kwargs) -> BaseAnsatz:
    """
    Factory to create variational ansatz.
    kind: 'fno' or 'tn'
    dim: lattice dimension (1 or 2)
    kwargs: model-specific parameters
    """
    kind = kind.lower()
    if kind == "fno":
        return FNOAnsatz(dim=dim, **kwargs)
    elif kind == "tn":
        return TNAnsatz(dim=dim, **kwargs)
    else:
        raise ValueError(f"Unknown ansatz kind: {kind}")