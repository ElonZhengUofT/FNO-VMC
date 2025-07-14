import jax.numpy as jnp
import netket as nk
from netket.operator import LocalOperator

class ClippedLocalOperator(LocalOperator):
    def __init__(self, hilbert, hamiltonian, threshold):
        """
        base_op: 原始的 LocalOperator（比如 Hubbard Hamiltonian）
        threshold: 绝对值阈值，或一个函数 func(mean, std)->thresh
        """
        super().__init__(
            hilbert=hilbert,
            operators=hamiltonian
        )
        self.threshold = threshold

    def _apply_locally(self, logpsi, x, *, chunk_size=None):
        conns, weights = super()._apply_locally(logpsi, x, chunk_size=chunk_size)
        # 如果 threshold 是函数，就先计算 mean/std
        if callable(self.threshold):
            mean = jnp.mean(weights)
            std  = jnp.std(weights)
            thr  = self.threshold(mean, std)
        else:
            thr = self.threshold
        # clip 到 [mean-thr, mean+thr]
        # 也可以直接 clip 到 [-thr, +thr]：jnp.clip(weights, -thr, +thr)
        clipped = jnp.clip(weights, mean - thr, mean + thr)
        return conns, clipped
