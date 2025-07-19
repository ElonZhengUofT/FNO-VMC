import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.experimental.models import Slater2nd
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import NNInitFunc, DType
from netket import jax as nkjax
from src.fno_vmc_nk.ansatz.fno_ansatz_jax import FNOAnsatzFlax


#region Vanilla and FNO Backflow on Slater2nd
class NNBackflowSlater2nd(nn.Module):
    """
    在 Slater2nd 基础上加入 NN-backflow:
      M(n) = M_base + F(n)
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    generalized: bool = True
    restricted: bool = True
    hidden_units: int = 64
    kernel_init: NNInitFunc = default_kernel_init
    param_dtype: DType = float
    dim: int =2
    modes1: int = 4
    modes2: int = None
    width: int = 32
    channel: int = 1

    def setup(self):
        # 1) 基础 Slater2nd
        self.slater = Slater2nd(
            hilbert=self.hilbert,
            generalized=self.generalized,
            restricted=self.restricted,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
        )
        # 2) 计算每个轨道块的形状列表
        if self.generalized:
            self.shapes = [tuple(self.slater.orbitals.shape)]
        else:
            self.shapes = [tuple(M.shape) for M in self.slater.orbitals]
        # 3) 每块扁平长度 & 静态切分点
        sizes = [s[0] * s[1] for s in self.shapes]
        self.cuts = np.cumsum(sizes)[:-1].tolist()  # e.g. [512, 1024]
        self.total_size = sum(sizes)
        # 4) 定义背流 MLP
        self.backflow_mlp = nn.Sequential([
            nn.Dense(self.hidden_units),
            nn.tanh,
            nn.Dense(self.total_size),
        ])
        self.FNO = FNOAnsatzFlax(
            dim=self.dim,
            modes1=self.modes1,
            modes2=self.modes2,
            width=self.width,
            channel=self.channel,
            out_dim=self.total_size,
        )

    def __call__(self, n):
        # 确保是 0/1 整数张量
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.isclose(n, 1)

        # 1) 计算扁平修正向量 F_flat，形状 (..., total_size)
        F_flat = self.FNO(n)

        # 单样本下构建修正后的轨道块并计算 logdet
        def single_logdet(n_sample, F_flat_sample):
            # a) 根据模式（generalized / restricted）生成修正后轨道块
            if self.generalized:
                M_base = self.slater.orbitals           # (size, n_fermions)
                F = F_flat_sample.reshape(M_base.shape)
                M_blocks = [M_base + F]
            else:
                parts = jnp.split(F_flat_sample, self.cuts, axis=-1)
                M_blocks = [
                    M_i + p.reshape(shape)
                    for p, M_i, shape in zip(parts, self.slater.orbitals, self.shapes)
                ]

            # b) 提取占据行并累加各自的 det
            R = jnp.nonzero(n_sample, size=self.hilbert.n_fermions, fill_value=0)[0]
            logdet = 0.0
            offset = 0

            if self.generalized:
                A = M_blocks[0][R, :]                # (n_fermions, n_fermions)
                logdet = nkjax.logdet_cmplx(A)
            else:
                for i, (nf_i, M_i_block) in enumerate(
                    zip(self.hilbert.n_fermions_per_spin, M_blocks)
                ):
                    if nf_i == 0:
                        continue
                    R_i = R[offset : offset + nf_i] - i * self.hilbert.n_orbitals
                    A = M_i_block[R_i]               # (nf_i, nf_i)
                    logdet += nkjax.logdet_cmplx(A)
                    offset += nf_i

            return logdet

        # 2) 对批量输入用 vmap
        #    如果 n 是一维，就当单样本，否则映射到第一个维度
        if n.ndim == 1:
            return single_logdet(n, F_flat)
        else:
            batch_size = 16  # 2048 is max for 4090 GPU
            logdets = []
            # 按 batch_size 切块
            for i in range(0, n.shape[0], batch_size):
                ni = n[i: i + batch_size]
                Fi = F_flat[i: i + batch_size]
                # 对这一小块做 vmap
                logdets.append(jax.vmap(single_logdet, in_axes=(0, 0))(ni, Fi))
            # 最后拼回整个 batch
            return jnp.concatenate(logdets, axis=0)


class NNBackflowMLP(nn.Module):
    """
    在 Slater2nd 基础上加入 NN-backflow:
      M(n) = M_base + F(n)
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    generalized: bool = True
    restricted: bool = True
    hidden_units: int = 64
    kernel_init: NNInitFunc = default_kernel_init
    param_dtype: DType = float
    dim: int =2
    modes1: int = 4
    modes2: int = None
    width: int = 32
    channel: int = 1

    def setup(self):
        # 1) 基础 Slater2nd
        self.slater = Slater2nd(
            hilbert=self.hilbert,
            generalized=self.generalized,
            restricted=self.restricted,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
        )
        # 2) 计算每个轨道块的形状列表
        if self.generalized:
            self.shapes = [tuple(self.slater.orbitals.shape)]
        else:
            self.shapes = [tuple(M.shape) for M in self.slater.orbitals]
        # 3) 每块扁平长度 & 静态切分点
        sizes = [s[0] * s[1] for s in self.shapes]
        self.cuts = np.cumsum(sizes)[:-1].tolist()  # e.g. [512, 1024]
        self.total_size = sum(sizes)
        # 4) 定义背流 MLP
        self.backflow_mlp = nn.Sequential([
            nn.Dense(self.hidden_units),
            nn.tanh,
            nn.Dense(self.total_size),
        ])

    def __call__(self, n):
        # 确保是 0/1 整数张量
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.isclose(n, 1)

        # 1) 计算扁平修正向量 F_flat，形状 (..., total_size)
        F_flat = self.backflow_mlp(n.astype(jnp.float32))

        # 单样本下构建修正后的轨道块并计算 logdet
        def single_logdet(n_sample, F_flat_sample):
            # a) 根据模式（generalized / restricted）生成修正后轨道块
            if self.generalized:
                M_base = self.slater.orbitals           # (size, n_fermions)
                F = F_flat_sample.reshape(M_base.shape)
                M_blocks = [M_base + F]
            else:
                parts = jnp.split(F_flat_sample, self.cuts, axis=-1)
                M_blocks = [
                    M_i + p.reshape(shape)
                    for p, M_i, shape in zip(parts, self.slater.orbitals, self.shapes)
                ]

            # b) 提取占据行并累加各自的 det
            R = jnp.nonzero(n_sample, size=self.hilbert.n_fermions, fill_value=0)[0]
            logdet = 0.0
            offset = 0

            if self.generalized:
                A = M_blocks[0][R, :]                # (n_fermions, n_fermions)
                logdet = nkjax.logdet_cmplx(A)
            else:
                for i, (nf_i, M_i_block) in enumerate(
                    zip(self.hilbert.n_fermions_per_spin, M_blocks)
                ):
                    if nf_i == 0:
                        continue
                    R_i = R[offset : offset + nf_i] - i * self.hilbert.n_orbitals
                    A = M_i_block[R_i]               # (nf_i, nf_i)
                    logdet += nkjax.logdet_cmplx(A)
                    offset += nf_i

            return logdet

        # 2) 对批量输入用 vmap
        #    如果 n 是一维，就当单样本，否则映射到第一个维度
        if n.ndim == 1:
            return single_logdet(n, F_flat)
        else:
            batch_size = 16  # 2048 is max for 4090 GPU
            logdets = []
            # 按 batch_size 切块
            for i in range(0, n.shape[0], batch_size):
                ni = n[i: i + batch_size]
                Fi = F_flat[i: i + batch_size]
                # 对这一小块做 vmap
                logdets.append(jax.vmap(single_logdet, in_axes=(0, 0))(ni, Fi))
            # 最后拼回整个 batch
            return jnp.concatenate(logdets, axis=0)
#endregion


#region BackflowI and BackflowII
class BackflowI(nn.Module):
    """
    Backflow I:
    M_mod = M_base + F(n),
    where F(n) is given by a neural network of shape (B, K, n_orbitals, n_fermions).
    Wavefunction ψ(n) = ∑ₖ det[M_modₖ(n)].
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    base_orbitals: jnp.ndarray       # shape (K, n_orbitals, n_fermions)
    backflow_fn: nn.Module           # nn.Module mapping n -> (B, K, n_orbitals, n_fermions)

    def __call__(self, n):
        # Ensure binary occupancy (0/1)
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.where(n > 0.5, 1, 0).astype(jnp.int32)
        # Compute backflow correction F: (B, K, n_orbitals, n_fermions)
        F = self.backflow_fn(n.astype(jnp.float32))
        # Broadcast base_orbitals to batch: (B, K, n_orbitals, n_fermions)
        M_base = jnp.expand_dims(self.base_orbitals, axis=0)
        M_mod = M_base + F

        # Number of electrons
        n_fermions = self.hilbert.n_fermions

        def sample_logdets(M_mod_sample, n_sample):
            # Determine occupied indices
            R = jnp.nonzero(n_sample, size=n_fermions, fill_value=0)[0]
            # Compute logdet for each determinant k
            def logdet_k(Mk):
                A = Mk[R, :]
                return nkjax.logdet_cmplx(A)
            logdets = jax.vmap(logdet_k)(M_mod_sample)  # shape (K,)
            # Sum determinants: log ψ = log ∑ exp(logdets) (log-sum-exp for stability)
            return jax.scipy.special.logsumexp(logdets)

        # Vectorize over batch
        if n.ndim == 1:
            return sample_logdets(M_mod[0], n)
        else:
            return jax.vmap(sample_logdets)(M_mod, n)


class BackflowII(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    backflow_fn: nn.Module         # maps n -> (B,K,2N,Ne) or (B,rows,cols) or (B,total_size)
    generalized: bool = True
    restricted: bool = True
    a: float = -1.0  # scaling exponent for backflow modulation

    def setup(self):
        self.slater = Slater2nd(
            hilbert=self.hilbert,
            generalized=self.generalized,
            restricted=self.restricted,
            kernel_init=default_kernel_init,
            param_dtype=jnp.float32,
        )
        # 计算每块的形状
        if self.generalized:
            self.shapes = [tuple(self.slater.orbitals.shape)]
        else:
            self.shapes = [tuple(M.shape) for M in self.slater.orbitals]
        sizes = [r * c for (r, c) in self.shapes]
        self.total_size = sum(sizes)
        self.cuts = np.cumsum(sizes)[:-1].tolist()  # 不走 JAX tracing

    def __call__(self, n):
        # 1) 保证输入是 0/1 占据向量
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = (n > 0.5).astype(jnp.int32)

        # 2) 计算 backflow 输出
        F_out = self.backflow_fn(n.astype(jnp.float32))

        # 3) 如果直接输出了 (B, K, rows, cols)，那么直接广播相乘
        if F_out.ndim == 4:
            print("Using direct (B,K,rows,cols) backflow output.")
            # F_out: (B, K, rows, cols)
            N_sites = self.hilbert.n_orbitals // 2
            mod = 1.0 + F_out * (N_sites ** self.a)     # (B, K, rows, cols)
            M_base = self.slater.orbitals               # (rows, cols)
            # 广播到 (B, K, rows, cols)
            M_mod_all = M_base

        else:
            # 回退到原 flatten–split–reshape 逻辑
            ndim_block = len(self.shapes[0])
            B = F_out.shape[0]
            if F_out.ndim == ndim_block + 1:
                # e.g. (B, rows, cols) -> flatten
                F_flat = F_out.reshape((B, -1))
            else:
                # assume already (B, total_size)
                F_flat = F_out

            # 把每个样本的向量切分回若干块
            def build_blocks(F_vec):
                parts = (jnp.split(F_vec, self.cuts, axis=-1)
                         if len(self.shapes) > 1 else [F_vec])
                blocks = []
                N_sites = self.hilbert.n_orbitals // 2
                for (r, c), part in zip(self.shapes, parts):
                    F_block = part.reshape((r, c))
                    M_base = (self.slater.orbitals if self.generalized
                              else self.slater.orbitals[len(blocks)])
                    mod = 1.0 + F_block * (N_sites ** self.a)
                    blocks.append(M_base * mod)
                return jnp.stack(blocks, axis=0)  # (K, r, c) 或 (1, r, c)

            # 对整个 batch 向量化
            M_mod_all = jax.vmap(build_blocks)(F_flat)
            # 此时 M_mod_all.shape == (B, K, rows, cols) 或 (B,1,rows,cols)

        # 4) 计算 log-sum-exp of determinants
        n_fermions = self.hilbert.n_fermions

        def sample_logsumexp(M_blocks, n_sample):
            # M_blocks: (K, rows, cols)
            R = jnp.nonzero(n_sample, size=n_fermions, fill_value=0)[0]
            def logdet_k(Mk):
                A = Mk[R, :]
                return nkjax.logdet_cmplx(A)
            logdets = jax.vmap(logdet_k)(M_blocks)  # (K,)
            return jax.scipy.special.logsumexp(logdets)

        # 5) 根据输入是单样本还是 batch，返回结果
        if n.ndim == 1:
            return sample_logsumexp(M_mod_all[0], n)
        else:
            return jax.vmap(sample_logsumexp)(M_mod_all, n)



#endregion


