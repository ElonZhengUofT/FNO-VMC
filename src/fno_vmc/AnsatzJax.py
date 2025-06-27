from abc import ABC, abstractmethod
import flax.linen as nn
from .fno_ansatz_jax import FNOAnsatzFlax
from .tn_model import TNAnsatz
import netket as nk

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
    elif kind == "RBM":
        from netket.models import RBM
        return RBM(n_visible=dim, **kwargs)
    elif kind == "Slater":
        from netket.models import Slater
        return Slater(**kwargs)
    else:
        raise ValueError(f"Unknown ansatz kind: {kind}")

if __name__ == "__main__":
    slater = nk.models.Slater2nd()