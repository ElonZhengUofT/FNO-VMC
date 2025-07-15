from netket.optimizer import LinearPreconditioner
from netket.vqs import VariationalState
import jax

class MARCH(LinearPreconditioner):
    def __init__(self,
                 diag_shift=1e-3,
                 beta=0.99,
                 mu=0.9,
                 lambda_reg=1e-4,
                 solver=jax.scipy.sparse.linalg.cg):
        # 让 QGTAuto 负责构造 Fisher 矩阵 S
        super().__init__(self.lhs_constructor,solver)
        self.diag_shift = diag_shift
        self.beta       = beta
        self.mu         = mu
        self.lambda_reg = lambda_reg
        # 用来存 history
        self.phi = None
        self.V   = None

    def lhs_constructor(self, vstate: VariationalState, step=None):
        # 就和 SR 一样，让 QGTAuto 用 diag_shift 构建 S + shift·I
        return QGTAuto(vstate, diag_shift=self.diag_shift)

    def apply(self, vstate: VariationalState, grad, step=None):
        # 1) 用 vstate 得到 O (jacobian) 和 local-energy 中心化后的 ε
        O    = vstate.jacobian()    #  shape (n_samples, n_params)
        eps  = vstate.local_energy_centered()  # shape (n_samples,)

        # 2) 初始化 phi, V
        if self.phi is None:
            self.phi = jax.tree_map(jnp.zeros_like, grad)
            self.V   = jax.tree_map(lambda g: jnp.ones_like(g),   grad)

        # 3) 计算新的动量和平滑量
        phi_new = jax.tree_map(lambda p: self.mu*p, self.phi)
        delta   = phi_new  # 为了写差分
        # 注意：真正的 delta_t-1 应该存上一次更新，不过简单起见可用 phi
        diff    = jax.tree_map(lambda d,p: d-p, grad, phi_new)
        V_new   = jax.tree_map(lambda v,d: self.beta*v + (1-self.beta)*d**2,
                              self.V, diff)

        # 4) 构造 adaptive 缩放 D⁻¹ = diag(V_new)^{-1/4}
        D14_inv = jax.tree_map(lambda v: v**(-0.25), V_new)

        # 5) 解正规方程：(O D^{-2} Oᵀ + λI) y = ε - O φ
        #    然后 δθ = D^{-1} Oᵀ y + φ
        #    下面是个简化示例，大小参数少时可以 to_dense
        O_scaled = O * jnp.stack([D14_inv]*O.shape[0], axis=0)
        A        = O_scaled @ O_scaled.T + self.lambda_reg * jnp.eye(O.shape[0])
        rhs      = eps - (O @ phi_new)
        y        = jnp.linalg.solve(A, rhs)
        update   = (O_scaled.T @ y) * D14_inv + phi_new

        # 6) 保存 history
        self.phi = phi_new
        self.V   = V_new

        return update
