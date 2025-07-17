import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np

import netket as nk
from netket import jax as nkjax
from netket.sampler import MetropolisFermionHop
from netket.vqs import MCState
from netket.operator import LocalOperator
from netket.optimizer import SR

# --- 1) 定义 MLP-orbitals NNB ansatz ---
class NNBOrbitals(nn.Module):
    n_sites: int
    n_elec: int
    hidden_dim: int = 128

    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim)
        self.fc2 = nn.Dense(self.hidden_dim)
        self.fc_out = nn.Dense(self.n_sites * self.n_elec)

    def __call__(self, n):
        # 输入 n: (n_sites*2,) 布尔 or int -> float
        x = n.astype(jnp.float32)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        M_flat = self.fc_out(x)             # (n_sites*n_elec,)
        M = M_flat.reshape(self.n_sites, self.n_elec)
        return M

# --- 2) 把 orbitals -> wavefunction amplitude via det ---
class NNBWave(nn.Module):
    orbitals_model: NNBOrbitals

    def __call__(self, n):
        M = self.orbitals_model(n)
        # 选 occupied rows
        R = jnp.nonzero(n, size=self.orbitals_model.n_elec, fill_value=0)[0]
        A = M[R, :]                        # (n_elec, n_elec)
        return nkjax.logdet_cmplx(A).real

# --- 3) VMC 训练 NNB ansatz ---
def train_nnb(hi, graph, H, n_iter=1000):
    n_sites = hi.n_orbitals * 2
    n_elec = sum(hi.n_fermions_per_spin)
    # 构建 ansatz
    nnb_orb = NNBOrbitals(n_sites=n_sites, n_elec=n_elec)
    model_nnb = NNBWave(orbitals_model=nnb_orb)
    sampler = MetropolisFermionHop(hi, graph=graph)
    vstate = MCState(sampler, model_nnb, n_samples=1024)
    opt = nk.optimizer.Sgd(learning_rate=1e-2)
    sr_pre = SR(diag_shift=1e-3)
    vmc = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr_pre)
    vmc.run(n_iter=n_iter)
    return vstate, model_nnb

# --- 4) 生成监督数据集 ---
def generate_dataset(vstate, model_orb, N_pre=5000):
    # 从训练好的 vstate 采样
    samples = vstate.sample(N_pre)  # shape (N_pre, 2*n_sites)
    # 计算对应 orbitals
    params = vstate.parameters
    M_all = jax.vmap(lambda n: model_orb.apply(params, n))(samples)
    return samples, M_all  # samples:(N_pre,2N_s), M_all:(N_pre,N_s,N_e)

# --- 5) 监督式预训练 Transformer ---
class TransformerOrbitals(nn.Module):
    n_sites: int
    n_elec: int
    d_model: int = 128
    heads: int = 8
    layers: int = 4

    @nn.compact
    def __call__(self, n):
        # n: (2*N_s,) -> embed -> (2N_s,d_model)
        x = nn.Embed(num_embeddings=2, features=self.d_model)(n.astype(jnp.int32))
        pos = self.param('pos', nn.initializers.normal(), (2*self.n_sites, self.d_model))
        x = x + pos
        for _ in range(self.layers):
            x = nn.SelfAttention(num_heads=self.heads)(x)
            x = nn.Dense(self.d_model)(nn.relu(nn.Dense(self.d_model*2)(x)))
        # linear to orbitals
        M_flat = nn.Dense(self.n_sites*self.n_elec)(x)  # -> (2N_s, N_s*N_e)
        # average over spin channel: split first dim 2N_s -> (2,N_s)
        M = M_flat.reshape(2, self.n_sites, self.n_elec)
        M_stacked = jnp.concatenate(M, axis=0)  # (2*N_s,N_e)
        return M_stacked

def pretrain_transformer(samples, M_targets, n_sites, n_elec,
                         lr=1e-3, batch=256, epochs=20):
    # 初始化 Transformer
    model_t = TransformerOrbitals(n_sites=n_sites, n_elec=n_elec)
    rng = jax.random.PRNGKey(0)
    params = model_t.init(rng, samples[0])

    # optimizer
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, n_batch, M_batch):
        M_pred = jax.vmap(lambda n: model_t.apply(params, n))(n_batch)
        return jnp.mean((M_pred - M_batch)**2)

    @jax.jit
    def train_step(params, opt_state, n_batch, M_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, n_batch, M_batch)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    N = samples.shape[0]
    for ep in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            params, opt_state, l = train_step(params, opt_state,
                                              samples[idx], M_targets[idx])
        print(f"Pretrain Epoch {ep+1}, Loss: {l:.6f}")
    return params

# --- 使用示例 ---
# hi = SpinOrbitalFermions(...), graph = Hypercube(...)
# H = build_hubbard(hi, graph)
# vstate_nnb, model_nnb = train_nnb(hi, graph, H)
# samples, M_targets = generate_dataset(vstate_nnb, model_nnb)
# params_t = pretrain_transformer(samples, M_targets, hi.n_orbitals*2, sum(hi.n_fermions_per_spin))

