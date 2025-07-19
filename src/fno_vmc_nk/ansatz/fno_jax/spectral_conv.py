import jax.numpy as jnp
from flax import linen as nn
from .utils import rfft2d, irfft2d
from jax import random
from jax import config
import jax
config.update("jax_enable_x64", False)

class SpectralConv2d(nn.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    def setup(self):
        # Initialize frequency domain weights
        self.scale = 1 / (self.in_channels * self.out_channels)
        self.weights1 = self.param(
            'weights1', lambda rng: random.normal(rng, (self.in_channels, self.out_channels, self.modes1, self.modes2)) * self.scale
        )
        self.weights2 = self.param(
            'weights2', lambda rng: random.normal(rng, (self.in_channels, self.out_channels, self.modes1, self.modes2)) * self.scale
        )

    def __call__(self, x):  # x: (batch, Nx, Ny, in_channels)
        batch, Nx, Ny, _ = x.shape
        # 转到频域
        x_ft = rfft2d(x.astype(jnp.float32).transpose((0,3,1,2))) # -> (batch, in_channels, Nx, Ny//2+1)

        # 构造输出频域张量
        out_ft = jnp.zeros((batch, self.out_channels, Nx, Ny//2+1), dtype=jnp.complex64)
        # 低频区卷积
        out_ft = out_ft.at[:, :, :self.modes1, :self.modes2].add(
            jnp.einsum("bcih,coih->boih", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        )
        out_ft = out_ft.at[:, :, -self.modes1:, :self.modes2].add(
            jnp.einsum("bcih,coih->boih", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        )

        # 逆 FFT 回时域
        x = irfft2d(out_ft, (Nx, Ny))
        return x.transpose((0,2,3,1))  # -> (batch, Nx, Ny, out_channels)

class SpectralConv1d(nn.Module):
    in_channels: int
    out_channels: int
    modes: int  # 保留前 modes 个低频
    @nn.compact
    def __call__(self, x):
        # x: (batch, N, in_channels)
        x_ft = jnp.fft.rfft(x, axis=1)
        # 只对前 modes 模式做变换
        W = self.param('W', jax.nn.initializers.normal(),
                       (self.in_channels, self.out_channels, self.modes))
        out_ft = jnp.zeros_like(x_ft)
        out_ft = out_ft.at[:, :self.modes, :].set(
            jnp.einsum('bic, ioc -> boc', x_ft[:, :self.modes, :], W)
        )
        x1 = jnp.fft.irfft(out_ft, n=x.shape[1], axis=1)
        return x1

