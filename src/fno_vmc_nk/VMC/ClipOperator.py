import jax.numpy as jnp
import netket as nk
from netket.operator import LocalOperator

class ClippedLocalOperator(LocalOperator):
    def __init__(self, base_op, min_val, max_val):
        # 调用 LocalOperator 构造器，复用 base_op 的 hilbert、graph 等属性
        super().__init__(hilbert=base_op.hilbert,
                         graph=base_op.graph,
                         operators=base_op._operators,
                         parameters=base_op._parameters)
        self.min_val = min_val
        self.max_val = max_val

    def _apply_locally(self, logpsi, x, *, chunk_size=None):
        # 调用父类方法，得到原始的 (连接态列表, 权重列表)
        conns, weights = super()._apply_locally(logpsi, x, chunk_size=chunk_size)
        # 对局部能量（weights）做裁剪
        clipped = jnp.clip(weights, self.min_val, self.max_val)
        return conns, clipped