import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import initializers

import netket as nk
from netket.experimental.models import Slater2nd
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import NNInitFunc, DType
from netket import jax as nkjax
from src.fno_vmc_nk.ansatz.fno_jax import SpectralConv2d, SpectralConv1d
from src.fno_vmc_nk.ansatz.fno_ansatz_jax import FNOAnsatzFlax

SMALL_NORM = initializers.normal(1e-3)
ZERO_BIAS = initializers.zeros

#region Basic Utils
def f_reshape(x,dim, channel=2):
    """
    Netket samples x are flattened, (M, (2*spin+1)*n_sites) for spin=1/2.
    So we need to reshape them back to a grid format.
    In 1D, it becomes (M, L, 1, channel)
    in 2D, it becomes (M, L, L, channel)
    axis=0 is batch, axis=1 is sites for length, axis=2 is height (1 for 1D),
    axis=3 is channel.
    """
    batch, features = x.shape[0], x.shape[-1]
    if dim == 1:
        L = features // channel
        u = x.reshape(batch, L, 1, channel)
    if dim == 2:
        L = int((features / channel) ** 0.5)
        u = x.reshape(batch, L, L, channel)
    return u


def k_reshape(x, K=1):
    """
    Reshape from (B,L_x,L_y,DK) to (B,K, L_x,L_y,D)
    """
    batch, Lx, Ly, C = x.shape
    D = C // K
    u = x.reshape(batch, K, Lx, Ly, D)
    return u


def flatten(x, channel=2):
    """
    Reshape from (B,K,L_x,L_y,2D) to (B,K, 2L_x*L_y,D) for femions
    """
    batch, K, Lx, Ly, D = x.shape
    u = x.reshape(batch, K, channel * Lx * Ly, D // channel)
    return u


def index_encoding(batch, N_e: int, beta=0):
    """
    Generate power positional encoding for N_e positions with exponent beta.
    The shape is  (B, N_e)
    """
    idx = jnp.arange(1, N_e + 1)
    idx = idx * (1.0 / N_e) ** beta
    idx = idx[None, :, None]
    idx = jnp.broadcast_to(idx, (batch, N_e, 1))  # (B, Ne)
    return idx


def InnerProduct(context, index):
    # context: (B, K, N, P)
    # index:   (B, K, Ne, P)
    out = jnp.einsum('bknp,bkmp->bknm', context, index)  # (B, K, N, Ne)
    return out


#endregion


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


#region Basic Blocks
class Projector(nn.Module):
    """
    Position Embedding module for FNO.
    Maps input coordinates to a higher-dimensional space.
    """
    dim: int = 2
    hidden_features: int = 32

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, n_features)
        # output shape: (batch, n_features, width)
        return nn.Dense(
            self.hidden_features,
            kernel_init=SMALL_NORM,
            bias_init=ZERO_BIAS
        )(x)

class Lifting(nn.Module):
    """
    Lifting module for FNO.
    Maps input features to a higher-dimensional space.
    """
    hidden_features: int = 32
    # n_layers: int = 2 Maybe just 2 first

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, n_features)
        # output shape: (batch, n_features, width)
        x = nn.Dense(self.hidden_features,
                     kernel_init=SMALL_NORM,
                        bias_init=ZERO_BIAS)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_features,
                        kernel_init=SMALL_NORM,
                            bias_init=ZERO_BIAS)(x)
        return x


class FNOBlock1D(nn.Module):
    """
    A single FNO block consisting of spectral convolution and pointwise convolution.
    """
    width: int = 32  # Width of the hidden layers, typically the number of hidden features
    depth: int = 4
    modes: int = 16  # Number of Fourier modes

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x1 = SpectralConv1d(self.width, self.width, self.modes)(x)
            x2 = nn.Conv(self.width, kernel_size=(1,))(x)
            x = x1 + x2
            x = nn.gelu(x)
        return x


class FNOBlock2D(nn.Module):
    """
    A single FNO block consisting of spectral convolution and pointwise convolution.
    """
    width: int = 32 # Width of the hidden layers，typically the number of hidden features
    depth: int = 4
    modes1: int = 16  # Number of Fourier modes in the first dimension
    modes2: int = None  # Number of Fourier modes in the second dimension, if applicable

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)(x)
            x2 = nn.Conv(self.width, kernel_size=(1,1))(x)
            x = x1 + x2
            x = nn.gelu(x)
        return x


class ContextEncoder(nn.Module):
    """Context encoder module for FNO"""
    p_feature: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # x shape: (B, K, N, D)
        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.p_feature,
                         kernel_init=nn.initializers.normal(1e-2))(x)
            x = nn.gelu(x)
        x = nn.Dense(self.p_feature,
                     kernel_init=nn.initializers.normal(1e-2))(x)
        return x  # (B, K, N, P)


class IndexDecoder(nn.Module):
    """
    f：把 (context, index) -> per-electron weight
    j_vec(index):   (B, K, Ne, P)
    context: (B, K, N, P)
    输出:    (B, K, N, Ne)
    """
    hidden_dim:     int   # MLP hidden dimension

    @nn.compact
    def __call__(self, index, context):
        """
        context: (B, K, N, P)
        j_vec:   (B, K, Ne,P)
        returns: (B, K, N, Ne)
        """
        B, K, N, P = context.shape
        Ne = index.shape[-2]


        # 1) index (B, K, Ne, P) -> (B, K, N, Ne, P) embedding to P
        index = index[..., None, :, :]
        index = jnp.broadcast_to(index, (B, K, N, Ne, P))  # (B, K, N, Ne, P)


        # 2) 广播到 (B, K, N, Ne, P)
        #   - context 从 (B,K,N,P) -> (B,K,N,1,P) -> broadcast
        ctx = context[..., None, :]                  # (B, K, N, 1, P)
        ctx = jnp.broadcast_to(ctx, (B, K, N, Ne, P))
        # 3) 拼接 (B,K,N,Ne,2P)
        x = jnp.concatenate([ctx, index], axis=-1)

        # 4) 两层 MLP -> (B,K,N,Ne,1) -> squeeze -> (B,K,N,Ne)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=SMALL_NORM,
            bias_init=ZERO_BIAS
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            1,
            kernel_init=SMALL_NORM,
            bias_init=ZERO_BIAS
        )(x)
        return x.squeeze(-1)  # (B, K, N, Ne)
#endregion

# second-order Ansatz with FNO backflow
class NProcessor(nn.Module):
    """
    (B, 2N) -> (B, Lx, Ly, 2) -P-> (B, Lx, Ly, D) -FNO-> (B, Lx, Ly, D) -lifting
    -> (B, Lx, Ly, 2DK) -flatten-> (B, K, 2N, D)
    """
    D: int = 32
    K: int = 4

    dim: int = 1
    modes1: int = 8
    modes2: int = None
    channel: int = 2  # input channel, 2 for spin up/down

    @nn.compact
    def __call__(self, n):
        B, N_orbital = n.shape

        u = f_reshape(n, self.dim, self.channel)  # (B, Lx, Ly, 2)

        # 1) Positional Encoding
        PE = Projector(self.dim, self.D)(u)  # (B, Lx, Ly, D)

        u = Projector(self.dim, self.D)(u)  # (B, Lx, Ly, D)
        u = PE + u  # (B, Lx, Ly, P)

        # 2) FNO
        if self.dim == 1:
            u = FNOBlock1D(modes=self.modes1, width=self.D)(u)  # (B, Lx, 1, D)
        elif self.dim == 2:
            u = FNOBlock2D(modes1=self.modes1, modes2=self.modes2, width=self.D)(u)  # (B, Lx, Ly, D)

        # 3) Lifting
        u = Lifting(self.channel * self.D* self.K)(u)  # (B, Lx, Ly, 2DK)
        u = nn.tanh(u)
        u = k_reshape(u, self.K)  # (B, K, Lx, Ly, 2D)
        u = flatten(u, self.channel)  # (B, K, 2N, D)
        return u  # (B, K, 2N, D)


class NeProcessor(nn.Module):
    """
    Process Ne information to generate index encoding.
    From (B, Ne) to (B, Ne, P)
    """
    P: int = 32
    K: int = 4

    @nn.compact
    def __call__(self, index):
        B, Ne, L = index.shape
        index = index.reshape(B, Ne, L)  # (B, Ne, 1)

        #) Embedding to P
        index = Projector(1, self.P)(index)
        index = nn.tanh(index)
        x4d = index[..., None, :]  # (B, Ne, 1, P)
        x4d = FNOBlock1D(modes=8, width=self.P)(x4d)  # (B, Ne, 1, P)

        #) Lifting to (B, Ne, 1, P*K)
        x4d = nn.Dense(self.P * self.K)(x4d)
        x4d = k_reshape(x4d, self.K)  # (B, K, Ne,1 P)
        index = x4d.squeeze(-2)  # (B, K, Ne, P)
        return index  # (B, K, Ne, P)


class PositionalEmbedding(nn.Module):
    dim: int = 1 # spatial dimension
    Lx: int=16  # lattice x-dimension
    Ly: int=1  # lattice y-dimension
    embed_dim: int = 32  # output embedding dimension

    @nn.compact
    def __call__(self, n):
        """
        Args:
            n: (B, 2N) occupation vector, site-major flattened, spin-major last.
        Returns:
            embeddings: (B, Ne, embed_dim)
        """
        B, _ = n.shape
        N_sites = self.Lx * self.Ly

        occ_mask = n.astype(bool)
        Ne = jnp.sum(occ_mask, axis=-1)  # (B,)

        def extract_pos(n_sample):
            # For single sample: n_sample shape (2N,)
            idx = jnp.nonzero(n_sample, size=N_sites * 2, fill_value=-1)[
                0]  # pad with -1
            idx = idx[idx >= 0]  # filter valid ones
            return idx

        occ_indices = jax.vmap(extract_pos)(n)  # Ragged list: (B, <=2N)

        # Decode idx into (x, y, σ) or (site, σ)
        def decode(idx):
            site = idx // 2  # site index (0 ~ N-1)
            spin = idx % 2  # 0: up, 1: down
            if dim == 2:
                x = site % self.Lx
                y = site // self.Lx
                return jnp.stack([x, y, spin], axis=-1)  # shape (..., 3)
            return jnp.stack([site, spin], axis=-1)  # shape (..., 2)

        positions = jax.vmap(lambda ids: jax.vmap(decode)(ids))(
            occ_indices)  # (B, Ne, 3) or (B, Ne, 2)

        positions = positions.astype(jnp.float32)
        positions = positions.at[..., 0].set(positions[..., 0] / (self.Lx - 1))
        positions = positions.at[..., 1].set(positions[..., 1] / (self.Ly - 1))

        # 5️⃣ Lift to embedding dim
        emb = nn.Dense(
            self.embed_dim,
            kernel_init=SMALL_NORM,
            bias_init=ZERO_BIAS
        )(positions)  # (B, Ne, embed_dim)
        emb = nn.gelu(emb)
        return emb


#endregion


# region Ansatz Base
class AnsatzI(nn.Module):
    """
    Combine Nprocessor and Naive Ne index encoding to form a Slater Matrix
    NProcessor will output (B, K, 2N, D)
    Then build Naive Ne index encoding
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    D: int = 32
    P: int = 32
    K: int = 4

    dim: int = 1
    modes1: int = 8
    modes2: int = None
    channel: int = 2  # input channel, 2 for spin up/down

    def setup(self):
        self.nprocessor = NProcessor(
            D=self.D, K=self.K,
            dim=self.dim, modes1=self.modes1, modes2=self.modes2, channel=self.channel
        )
        self.neprocessor = NeProcessor(P=self.P, K=self.K)
        self.context_encoder = ContextEncoder(p_feature=self.P)
        self.index_decoder = IndexDecoder(hidden_dim=self.P)

    @nn.compact
    def __call__(self, n):
        B, N_orbital = n.shape

        Num_Ne = self.hilbert.n_fermions

        Y = self.nprocessor(n)  # (B, K, 2N, D)

        context = self.context_encoder(Y)  # (B,K, 2N, P)
        index = self.neprocessor(index_encoding(B, Num_Ne, beta=0))  # (B, K, Ne, P)

        orbitals = self.index_decoder(index, context)  # (B, K, 2N, Ne)
        return orbitals  # (B, K, 2N, Ne)


class AnsatzII(nn.Module):
    """
     Combine Nprocessor and Naive Ne index encoding to form a Slater Matrix
     Use Inner Product to combine,
     Then build Naive Ne index encoding
     """
    hilbert: nk.hilbert.SpinOrbitalFermions
    D: int = 32
    P: int = 32
    K: int = 4

    dim: int = 1
    modes1: int = 8
    modes2: int = None
    channel: int = 2  # input channel, 2 for spin up/down

    def setup(self):
        self.nprocessor = NProcessor(
            D=self.D, K=self.K,
            dim=self.dim, modes1=self.modes1, modes2=self.modes2,
            channel=self.channel
        )
        self.neprocessor = PostionEmbedding(P=self.P, K=self.K)
        self.context_encoder = ContextEncoder(p_feature=self.P)
        self.index_decoder = IndexDecoder(hidden_dim=self.P)

    @nn.compact
    def __call__(self, n):
        B, N_orbital = n.shape

        Ne = self.hilbert.n_fermions

        Y = self.nprocessor(n)  # (B, K, 2N, D)

        context = self.context_encoder(Y)  # (B,K, 2N, P)
        index = self.neprocessor(n)  # (B, K, Ne, P)

        orbitals = InnerProduct(context, index) + self.index_decoder(index, context)  # (B, K, 2N, Ne)
        return orbitals  # (B, K, 2N, Ne)


class AnsatzIII(nn.Module):
    """
     Combine Nprocessor and Naive Ne index encoding to form a Slater Matrix
     Use Index Mapping to combine,
     Then build Naive Ne index encoding
     """
    hilbert: nk.hilbert.SpinOrbitalFermions
    D: int = 64
    K: int = 4

    dim: int = 1
    modes1: int = 16
    modes2: int = None
    channel: int = 2  # input channel, 2 for spin up/down

    def setup(self):
        self.nprocessor = NProcessor(
            D=self.D, K=self.K,
            dim=self.dim, modes1=self.modes1, modes2=self.modes2,
            channel=self.channel
        )
        self.neprocessor = NeProcessor(P=self.D, K=self.K)
        self.index_decoder = IndexDecoder(hidden_dim=self.D)

    @nn.compact
    def __call__(self, n):
        B, N_orbital = n.shape

        Ne = self.hilbert.n_fermions

        Y = self.nprocessor(n)  # (B, K, 2N, D)

        index = self.neprocessor(index_encoding(B, Ne, beta=0))  # (B, K, Ne, D)

        orbitals = InnerProduct(context, index) + self.index_decoder(index, Y)  # (B, K, 2N, Ne)
        return orbitals  # (B, K, 2N, Ne)


class AnsatzIV(nn.Module):
    """
    A trivial ansatz that abandoned backflow structure
    """


class AnsatzV(nn.Module):
    """
    A trivial ansatz that abandoned backflow structure
    Refference: Autoregressive
    It starts from (B,2N) -> (B,N,2) -> (B,2N,D) -FNO-> (B,2N,D) -> (B,N,4)
    -softmax-> (B,N,4)
    Then for each site, add log probability
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    num_sites: int
    d_model: int = 64
    modes: int = 16
    depth: int = 4

    def setup(self):
        # Embedding from 2 channels (up/down) to d_model features
        self.embed = nn.Dense(self.d_model)
        # Stack of FNO blocks
        self.blocks = [FNO1dBlock(self.d_model, self.modes) for _ in
                       range(self.depth)]
        # Final projection to 4 logits per site
        self.project = nn.Dense(4)

    def __call__(self, occ):
        # occ: (B, 2*N) binary occupancy
        B = occ.shape[0]
        # reshape to (B, N, 2)
        x = occ.reshape(B, self.num_sites, 2)
        # embed: (B, N, d_model)
        x = self.embed(x)
        # prepare for FNO: (B, d_model, N)
        x = jnp.transpose(x, (0, 2, 1))
        # apply FNO blocks
        for block in self.blocks:
            x = block(x)
        # back to (B, N, d_model)
        x = jnp.transpose(x, (0, 2, 1))
        # project to logits: (B, N, 4)
        logits = self.project(x)
        # convert to log-probabilities along last axis
        log_probs = nn.log_softmax(logits)

        # log summing over all sites
        state_idx = occ_to_state_idx(occ)  # (B, N)
        lp = jnp.take_along_axis(log_probs, state_idx[..., None],
                                 axis=-1).squeeze(-1)
        return jnp.sum(lp, axis=1)  # (B,)
#endregion