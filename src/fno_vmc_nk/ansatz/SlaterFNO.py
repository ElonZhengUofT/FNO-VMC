import jax
import jax.random as random
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from src.fno_vmc_nk.ansatz.fno_ansatz_jax import FNOAnsatzFlax
import netket as nk

import flax.linen as nn
import jax.numpy as jnp
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import DType
from netket.nn.masked_linear import default_kernel_init


class SlaterDetFlax(nn.Module):
    """
    Flax Module for Slater determinant with dynamic dimensions inferred from hilbert.

    Attributes:
      hilbert: 一个 SpinOrbitalFermions 实例，包含粒子数、轨道数等信息。
    """
    hilbert: SpinOrbitalFermions
    dtype: DType = jnp.float32

    def setup(self):
        # 从 hilbert 属性中读取需要的维度
        n_sites = self.hilbert.size  # 基于格点数
        n_orbitals = self.hilbert.n_orbitals  # 轨道数
        n_particles = self.hilbert.n_fermions  # 粒子数

        # 定义一个 DenseGeneral，用于把输入映射到 Slater 矩阵
        # 输出维度为 [batch, n_sites, n_orbitals]
        self.orbital_layer = nn.DenseGeneral(
            features=(n_particles,),
            axis=-1,
            dtype=self.dtype,
            kernel_init=default_kernel_init
        )

    def __call__(self, config):
        """
        参数:
          config: shape [batch, n_sites] 的 0/1 占据数组
        返回:
          logψ: 波函数的对数幅度
          sign: 波函数的符号
        """
        # 1. 通过全连接层构造 Slater 矩阵
        #    先把 config 变成 [batch, n_sites, 1]，再映射到 [batch, n_sites, n_orbitals]
        x = config[..., None]
        orb_mat = self.orbital_layer(x)  # shape [batch, n_sites, n_orbitals]

        def build_slater(mat, conf):
            # top_k 会返回最大的 k 个值及它们的索引
            _, occ_idx = lax.top_k(conf, self.hilbert.n_fermions)
            return mat[occ_idx, :]

        slater_mats = jax.vmap(build_slater, in_axes=(0, 0))(orb_mat, config)

        signs, logdets = jax.vmap(lambda M: jnp.linalg.slogdet(M))(slater_mats)
        phase   = jnp.where(signs > 0, 0.0, jnp.pi)
        log_psi = logdets + 1j * phase

        return log_psi


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