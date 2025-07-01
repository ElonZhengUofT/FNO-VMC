import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.experimental.models import Slater2nd
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import NNInitFunc, DType
from netket import jax as nkjax

class NNBackflowSlater2nd(nn.Module):
    """
    在 Slater2nd 基础上加入 NN-backflow:
      M(n) = M_base + F(n)
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    generalized: bool = False
    restricted: bool = True
    hidden_units: int = 64
    kernel_init: NNInitFunc = default_kernel_init
    param_dtype: DType = float

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
        F_flat = self.backflow_mlp(n)

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
            return jax.vmap(single_logdet, in_axes=(0, 0))(n, F_flat)

# —— 使用示例 ——
# hi = SpinOrbitalFermions(...); graph = ...

if __name__ == "__main__":
    hi = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=8, s=0.5, n_fermions_per_spin=(7, 7)
    )
    graph = nk.graph.Hypercube(length=8, n_dim=2, pbc=True)
    model = NNBackflowSlater2nd(hi, generalized=False, restricted=True,
                                hidden_units=128)
    sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048)
    opt = nk.optimizer.Sgd(learning_rate=1e-2)
    sr = nk.optimizer.SR(diag_shift=1e-2)
    t = 1.0
    U = 4.0
    H = 0
    for sz in (+1, -1):
        for u, v in graph.edges():
            H -= t * (nk.operator.fermion.create(hi, u, sz=sz) *
                      nk.operator.fermion.destroy(hi, v, sz=sz) +
                      nk.operator.fermion.create(hi, v, sz=sz) *
                      nk.operator.fermion.destroy(hi, u, sz=sz))
    for u in graph.nodes():
        H += U * nk.operator.fermion.number(hi, u, sz=+1) * \
             nk.operator.fermion.number(hi, u, sz=-1)
    gs = nk.VMC(H, opt, variational_state=vstate, preconditioner=sr)
    gs.run(n_iter=200)
