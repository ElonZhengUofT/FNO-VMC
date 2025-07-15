import jax.numpy as jnp
from flax import linen as nn
from .spectral_conv import SpectralConv2d, SpectralConv1d
class FNO2d(nn.Module):
    modes1: int
    modes2: int
    width: int
    depth: int = 4

    @nn.compact
    def __call__(self, x):
        # x: (batch, Nx, Ny, 1)
        # Lift to higher维度
        x = nn.Dense(self.width)(x)

        # Fourier 层 + 点wise 卷积
        for _ in range(self.depth):
            x1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)(x)
            x2 = nn.Conv(self.width, kernel_size=(1,1))(x)
            x = x1 + x2
            x = nn.gelu(x)

        # 投影到目标通道
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x

class FNO1d(nn.Module):
    modes: int
    width: int
    depth: int = 4

    @nn.compact
    def __call__(self, x):
        # x: (batch, N, 1)
        # Lift to higher维度
        x = nn.Dense(self.width)(x)

        # Fourier 层 + 点wise 卷积
        for _ in range(self.depth):
            x1 = SpectralConv1d(self.width, self.width, self.modes)(x)
            x2 = nn.Conv(self.width, kernel_size=(1,))(x)
            x = x1 + x2
            x = nn.gelu(x)

        # 投影到目标通道
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x