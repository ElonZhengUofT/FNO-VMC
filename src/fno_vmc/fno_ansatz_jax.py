import jax
import jax.numpy as jnp
from .fno_jax import FNO2d
from flax import linen as nn

class FNOAnsatzJax(nn.Module):
    """FNO-based variational ansatz."""
    dim: int
    modes1: int
    modes2: int = None
    width: int = 32

    @nn.compact
    def __call__(self, x):
        batch = x.shape[0]
        # reshape input to image grid
        if self.dim == 1:
            L = x.shape[1]
            u = x.reshape(batch, L, 1, 1)
        else:
            L = int(x.shape[1]**0.5)
            u = x.reshape(batch, L, L, 1)
        # apply FNO (expects NHWC)
        modes2 = self.modes2 or 1
        out = FNO2d(modes1=self.modes1, modes2=modes2, width=self.width)(u)
        # global average pooling over spatial dims
        features = jnp.mean(out, axis=(1,2))  # shape (batch, width)
        # final linear head
        log_psi = nn.Dense(1)(features).squeeze(-1)  # shape (batch,)
        return log_psi