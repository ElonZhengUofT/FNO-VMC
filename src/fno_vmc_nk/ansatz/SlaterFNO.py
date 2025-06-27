import jax
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
from src.fno_vmc_nk.ansatz.fno_ansatz_jax import FNOAnsatzFlax
import netket as nk

class SlaterFNOFlax(nn.Module):
    """
    Flax Module composing FNOAnsatzFlax and NetKet Slater model.
    __call__(x) = Slater(x) * FNO(x)
    """
    dim: int
    modes1: int
    modes2: int = None
    width: int = 32
    channel: int = 1

    def setup(self):
        self.fno = FNOAnsatzFlax(
            dim=self.dim,
            modes1=self.modes1,
            modes2=self.modes2,
            width=self.width,
            channel=self.channel,
        )
        # NetKet Slater is not a Flax module; will initialize separately
        self.slater_model = nk.models.Slater()

    @nn.compact
    def __call__(self, x):
        # Compute FNO orbitals
        phi = self.fno(x)
        # Compute Slater determinant amplitudes
        psi_slater = self.slater_model.apply(self.slater_params, x)
        return psi_slater * phi