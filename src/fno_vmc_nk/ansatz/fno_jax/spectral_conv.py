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
        # x 的形状: (batch, N, in_channels)
        x_ft = jnp.fft.rfft(x, axis=1)
        # x_ft 的形状: (batch, N//2 + 1, in_channels)

        W_shape = (self.modes, self.in_channels, self.out_channels)
        W = self.param('W',
                       lambda key, shape, dtype=jnp.complex64: (
                           jax.random.normal(key, shape, jnp.float32) +
                           1j * jax.random.normal(key, shape, jnp.float32)
                       ) * jnp.sqrt(1.0 / self.in_channels), # 使用更稳健的初始化
                       W_shape)

        # --- 修改 2: 修正 einsum 下标 ---
        # 'bmc,mco->bmo' 的含义是:
        # b: batch (批次)
        # m: modes (频率模式)
        # c: in_channels (输入通道)
        # o: out_channels (输出通道)
        # 我们在 'c' 维度上进行矩阵乘法，并保留 b, m, o 维度。
        transformed_modes = jnp.einsum(
            'bmhc,mco->bmho',
            x_ft[:, :self.modes, :, :],
            W
        )

        # --- 修改 3: 正确初始化 out_ft ---
        # 输出的通道数应为 out_channels，并保持复数类型。
        out_ft_shape = list(x_ft.shape)
        out_ft_shape[-1] = self.out_channels
        out_ft = jnp.zeros(tuple(out_ft_shape), dtype=x_ft.dtype)

        # 将计算出的变换结果填充到输出张量的前 self.modes 个频率中
        out_ft = out_ft.at[:, :self.modes, :].set(transformed_modes)

        # 进行逆傅立叶变换
        x1 = jnp.fft.irfft(out_ft, n=x.shape[1], axis=1)

        return x1