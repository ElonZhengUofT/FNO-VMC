import flax.linen as nn
import jax.numpy as jnp
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import DType
from netket.nn.masked_linear import default_kernel_init
from netket.experimental.wavefunctions import WaveFunction


class LogNeuralBackflow(WaveFunction):
    hilbert: SpinOrbitalFermions
    hidden_dim: int = 64

    def setup(self):
        # Slater 行列式部分
        self.slater = nn.DenseGeneral(features=(self.hilbert.n_orbitals,),
                                      kernel_init=default_kernel_init)
        # Backflow 位移网络：MLP
        self.backflow_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hilbert.size)  # 每个格点一个位移标量
        ])

    def __call__(self, n_config):
        """
        n_config: [batch, N_sites] 占据数0/1或坐标
        """
        # 1. 预测位移 η: same shape as config
        eta = self.backflow_net(n_config)  # [batch, N_sites]

        # 2. 计算“流动后”特征
        #    对于格点模型，简单地把 eta 作为额外特征拼接
        bf_input = jnp.concatenate([n_config[..., None], eta[..., None]],
                                   axis=-1)

        # 3. 构造 Slater 矩阵
        #    DenseGeneral: [batch, N_sites, n_orbitals]
        slater_matrix = self.slater(bf_input)

        # 4. 计算行列式的 log and sign
        #    注意：需要批量化 slogdet
        def slogdet(m):
            return jnp.linalg.slogdet(m)

        batch_logdet = jnp.zeros(n_config.shape[0])
        batch_sign = jnp.ones(n_config.shape[0])
        for i in range(n_config.shape[0]):
            sign, logdet = slogdet(slater_matrix[i])
            batch_sign = batch_sign.at[i].set(sign)
            batch_logdet = batch_logdet.at[i].set(logdet)

        # 5. 返回 log ψ 和 sign
        return batch_logdet, batch_sign
