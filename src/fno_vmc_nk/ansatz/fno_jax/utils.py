import jax.numpy as jnp
from jax import random

# 生成频率坐标 (x,y)
def get_grid(Nx, Ny):
    x = jnp.linspace(0, 1, Nx)
    y = jnp.linspace(0, 1, Ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    return jnp.stack([X, Y], axis=-1)  # shape (Nx, Ny, 2)

# 复数 FFT
def rfft2d(x):
    return jnp.fft.rfft2(x)

def irfft2d(x, s):
    return jnp.fft.irfft2(x, s)