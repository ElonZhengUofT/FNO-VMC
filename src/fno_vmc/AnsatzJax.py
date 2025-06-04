from abc import ABC, abstractmethod
import torch.nn as nn
from .fno_model_jax import FNOAnsatzFlax
from .tn_model import TNAnsatz


class BaseAnsatzJax(ABC):
    """Abstract base for variational ansatz for NetKet."""
    @abstractmethod
    def init(self, *args, **kwargs):  # for NetKet
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):  # for NetKet
        pass


def make_ansatz_jax(kind: str, dim: int, **kwargs) -> nn.Module:
    """
    Factory to create JAX/Flax ansatz for NetKet.
    Returns a flax.linen.Module with init and apply.
    """
    kind = kind.lower()
    if kind == "fno":
        # rename legacy key if needed
        if "modes" in kwargs:
            kwargs["modes1"] = kwargs.pop("modes")
        return FNOAnsatzFlax(dim=dim, **kwargs)
    elif kind == "tn":
        return TNAnsatz(dim=dim, **kwargs)
    else:
        raise ValueError(f"Unknown ansatz kind: {kind}")