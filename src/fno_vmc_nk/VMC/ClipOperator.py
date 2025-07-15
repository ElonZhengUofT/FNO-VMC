import jax.numpy as jnp
import netket as nk
from netket.operator import LocalOperator
from nkx.operator import (
    ParticleNumberAndSpinConservingFermioperator2nd
)

class ClippedLocalOperator(LocalOperator):
    def __init__(self, base_op: LocalOperator, graph, threshold):
        super().__init__(
            hilbert=base_op.hilbert,
            operators=base_op._operators,
        )
        self.graph = graph
        self.threshold = threshold


    def _apply_locally(self, logpsi, x, *, chunk_size=None):
        # 1. 调用基类算符
        conns, weights = super()._apply_locally(
            logpsi, x, chunk_size=chunk_size
        )  # weights shape (batch, 1)

        # 2. 从 weights 中分离出 E_loc
        psi = jnp.exp(logpsi(x))               # shape (batch,)
        E_loc = weights / psi[:, None]         # shape (batch, 1)

        # 3. 计算 mean / std 并剪切
        E_mean = jnp.real(jnp.mean(E_loc))
        E_std = jnp.real(jnp.std(E_loc))
        δ = self.threshold(E_mean, E_std)      # 例如 δ=5.0

        E_loc_clipped = jnp.clip(
            E_loc,
            E_mean - δ,
            E_mean + δ
        )

        # 4. 重建被剪切后的 weights 并返回
        weights_clipped = E_loc_clipped * psi[:, None]
        return conns, weights_clipped
