import jax
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
from src.fno_vmc_nk.ansatz.fno_ansatz_jax import FNOAnsatzFlax
import netket as nk


class SlaterDetFlax(nn.Module):
    """
    Flax Module for Slater determinant with dynamic dimensions inferred from input.
    """
    dim: int  # 1D or 2D
    channel: int = 2  # number of orbitals per site (for Fermi systems, typically 2 for spin-up and spin-down)

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        batch, n_site = s.shape

        is_up = jnp.logical_or(s == 1, s == 3)  # spin-up 占据位置
        is_down = jnp.logical_or(s == 2, s == 3)  # spin-down 占据位置

        n_up = int(jnp.sum(is_up).item())
        n_down = int(jnp.sum(is_down).item())
        n_particles = n_up + n_down

        # initialize embedding table of shape (n_sites, n_particles)
        embedding = self.param(
            'embedding',
            nn.initializers.lecun_normal(),
            (n_sites, n_particles)
        )
        # gather orbital values -> shape (batch, n_particles, n_particles)
        M = embedding[s]

        # compute determinant for each sample
        return jnp.linalg.det(M)

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
        self.slater_model = nk.models.Slater2nd()

    @nn.compact
    def __call__(self, x):
        # Compute FNO orbitals
        phi = self.fno(x)
        # Compute Slater determinant amplitudes
        psi_slater = self.slater_model(x)
        return psi_slater * phi