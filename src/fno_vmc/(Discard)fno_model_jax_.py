# Compatible with JAX and Flax for NetKet version >= 3.0

import jax.numpy as jnp
import flax.linen as nn
from .fno_jax import FNO2d

class FNOAnsatzFlax(nn.Module):
    """FNO-based variational ansatz in Flax for NetKet MCState."""
    dim: int
    modes1: int
    modes2: int = None
    width: int = 32
    in_channels: int = 1  # input channels, usually 1 for scalar wavefunction

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch, N] values in {-1, +1}
        batch = x.shape[0]
        # reshape to [batch, H, W, 1]
        if self.dim == 1:
            L = x.shape[1]
            u = x.reshape((batch, L, 1, 1)).astype(jnp.float32)
        else:
            L = int(x.shape[1] ** 0.5)
            u = x.reshape((batch, L, L, 1)).astype(jnp.float32)
        # move channels last -> channels first for FNO2d: [batch, 1, H, W]
        u = jnp.transpose(u, (0, 3, 1, 2))
        # Fourier layers
        out = FNO2d(self.in_channels, self.modes1, self.modes2 or 1, self.width)(u)
        # global pooling and head
        out = jnp.mean(out, axis=(2, 3))  # adaptive avg pool
        log_psi = nn.Dense(features=1)(out).squeeze(-1)
        return log_psi