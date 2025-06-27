from abc import ABC, abstractmethod
import flax.linen as nn
from src.fno_vmc_nk.ansatz.fno_ansatz_jax import FNOAnsatzFlax
from src.fno_vmc_nk.ansatz.tn_model import TNAnsatz
from src.fno_vmc_nk.ansatz.SlaterFNO import SlaterFNOFlax
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
    elif kind == "rbm":
        from netket.models import RBM
        return RBM(n_visible=dim, **kwargs)
    elif kind == "slater":
        from netket.models import Slater
        return Slater(**kwargs)
    elif kind == "slaterfno":
        return SlaterFNOFlax(dim=dim, **kwargs)
    else:
        raise ValueError(f"Unknown ansatz kind: {kind}")

if __name__ == "__main__":
    slater = nk.models.Slater2nd()