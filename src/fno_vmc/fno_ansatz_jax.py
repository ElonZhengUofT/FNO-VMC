import jax
import jax.numpy as jnp
from .fno_jax import FNO2d
from flax import linen as nn

class FNOAnsatzFlax(nn.Module):
    """FNO-based variational ansatz."""
    dim: int
    modes1: int
    modes2: int = None
    width: int = 32
    channel: int = 1  # number of output channels, default is 1 for scalar output

    @nn.compact
    def __call__(self, x):
        batch = x.shape[0]
        # reshape input to image grid
        batch, features = x.shape[0], x.shape[-1]
        if self.dim == 1:
            L = features // self.channel
            u = x.reshape(batch, L, 1, self.channel)
        else:
            L = int((features / self.channel) ** 0.5)
            u = x.reshape(batch, L, L, self.channel)
        # apply FNO (expects NHWC)
        modes2 = self.modes2 or 1
        out = FNO2d(modes1=self.modes1, modes2=modes2, width=self.width)(u)
        # global average pooling over spatial dims
        features = jnp.mean(out, axis=(1,2))  # shape (batch, width)
        # final linear head
        log_psi = nn.Dense(1, kernel_init=nn.initializers.normal(1e-2))(
            features).squeeze(-1)
        return log_psi