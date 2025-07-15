import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.experimental.models import Slater2nd
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import NNInitFunc, DType
from netket import jax as nkjax
from src.fno_vmc_nk.ansatz.fno_jax import FNO2d
from src.fno_vmc_nk.ansatz.fno_ansatz_jax import FNOAnsatzFlax

class PE(nn.Module):
    """
    Position Embedding module for FNO.
    Maps input coordinates to a higher-dimensional space.
    """
    dim: int = 2
    hidden_features: int = 64

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, n_features)
        # output shape: (batch, n_features, width)
        return nn.Dense(self.hidden_features, kernel_init=nn.initializers.normal(1e-2))(x)

class Lifting(nn.Module):
    """
    Lifting module for FNO.
    Maps input features to a higher-dimensional space.
    """
    hidden_features: int = 64

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, n_features)
        # output shape: (batch, n_features, width)
        return nn.Dense(self.hidden_features, kernel_init=nn.initializers.normal(1e-2))(x)






# region MultiDetSlaterNNBackflow
class MultiDetSlaterNNBackflow(nn.Module):
    """
    Multi-determinant Ansatz combining Slater2nd base and NN-backflow correction.
    Wavefunction: psi(n) = sum_{k=1}^K coeffs[k] * det[ M_base_k + FNO_k(n) ]
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    K: int = 4
    hidden_units: int = 64
    general: bool = True
    restrict: bool = True
    kernel_init: NNInitFunc = default_kernel_init
    param_dtype: DType = float
    # FNO params
    dim: int = 2
    modes1: int = 4
    modes2: int = None
    width: int = 32
    channel: int = 1

    def setup(self):
        # Create K Slater blocks
        self.slater = [ Slater2nd(
                hilbert=self.hilbert,
                generalized=self.general,
                restricted=self.restrict,
                kernel_init=self.kernel_init,
                param_dtype=self.param_dtype,
            ) for _ in range(self.K)
        ]
        # Per-block FNO backflow
        self.fnos = [ FNOAnsatzFlax(
                dim=self.dim,
                modes1=self.modes1,
                modes2=self.modes2,
                width=self.width,
                channel=self.channel,
                out_dim=np.prod(self.slater[0].orbitals.shape),
            ) for _ in range(self.K)
        ]
        # learnable coefficients
        self.log_coeffs = self.param('log_coeffs', nn.initializers.zeros, (self.K,), self.param_dtype)

    def __call__(self, n):
        # ensure boolean occupancy
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.isclose(n, 1)

        def single_logpsi(n_sample):
            terms = []
            for k in range(self.K):
                M_base = self.slater[k].orbitals  # (nsites, nup+ndown)
                F_flat = self.fnos[k](n_sample)   # (nsites*(nup+ndown),)
                F = F_flat.reshape(M_base.shape)
                M = M_base + F
                # select occupied rows
                R = jnp.nonzero(n_sample, size=self.hilbert.n_fermions)[0]
                A = M[R, :]
                logdet = nkjax.logdet_cmplx(A)
                terms.append(logdet + self.log_coeffs[k])
            # log-sum-exp to combine
            return jax.scipy.special.logsumexp(jnp.stack(terms))

        if n.ndim == 1:
            return single_logpsi(n)
        else:
            return jax.vmap(single_logpsi)(n)
#endregion


#region MultiDetNNBackflowPure
class MultiDetNNBackflowPure(nn.Module):
    """
    Multi-determinant Ansatz using only NN backflow (no base Slater orbitals).
    psi(n) = sum_k coeffs[k] * det[ FNO_k(n) ]
    Each FNO outputs full orbitals.
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    K: int = 4
    # FNO parameters
    dim: int = 2
    modes1: int = 4
    modes2: int = None
    width: int = 32
    channel: int = 1

    def setup(self):
        nsites, nferm = self.hilbert.n_orbitals * 2, sum(self.hilbert.n_fermions_per_spin)
        self.fnos = [ FNOAnsatzFlax(
                dim=self.dim,
                modes1=self.modes1,
                modes2=self.modes2,
                width=self.width,
                channel=self.channel,
                out_dim=nsites * nferm,
            ) for _ in range(self.K)
        ]
        self.log_coeffs = self.param('log_coeffs', nn.initializers.zeros, (self.K,), jnp.float64)

    def __call__(self, n):
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.isclose(n, 1)

        def single_logpsi(n_sample):
            terms = []
            for k in range(self.K):
                F_flat = self.fnos[k](n_sample)
                M = F_flat.reshape((self.hilbert.n_sites, self.hilbert.n_fermions))
                R = jnp.nonzero(n_sample, size=self.hilbert.n_fermions)[0]
                A = M[R, :]
                logdet = nkjax.logdet_cmplx(A)
                terms.append(logdet + self.log_coeffs[k])
            return jax.scipy.special.logsumexp(jnp.stack(terms))

        if n.ndim == 1:
            return single_logpsi(n)
        else:
            return jax.vmap(single_logpsi)(n)
#endregion


#region MultiDetSingleFNO
class MultiDetSingleFNO(nn.Module):
    """
    Multi-determinant Ansatz with a single FNO generating K determinants' orbitals.
    psi(n) = sum_{k=1..K} exp(log_coeffs[k]) * det[ M_k(n) ],
    where each M_k(n) is extracted from one combined FNO output.
    """
    hilbert: nk.hilbert.SpinOrbitalFermions
    K: int = 4
    # FNO parameters
    dim: int = 2
    modes1: int = 4
    modes2: int = None
    width: int = 32
    channel: int = 1

    def setup(self):
        nsites = self.hilbert.n_sites
        nferm = sum(self.hilbert.n_fermions_per_spin)
        # Single FNO outputs K * (nsites * nferm) entries
        self.fno = FNOAnsatzFlax(
            dim=self.dim,
            modes1=self.modes1,
            modes2=self.modes2,
            width=self.width,
            channel=self.channel,
            out_dim=self.K * nsites * nferm,
        )
        # learnable log coefficients
        self.log_coeffs = self.param(
            'log_coeffs', nn.initializers.zeros, (self.K,), jnp.float64
        )

    def __call__(self, n):
        # ensure boolean occupancy
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.isclose(n, 1)

        def single_logpsi(n_sample):
            # F_all shape: (K * nsites * nferm,)
            F_all = self.fno(n_sample)
            # split into K blocks
            parts = jnp.split(F_all, self.K)
            log_terms = []
            # get occupied indices
            R = jnp.nonzero(n_sample, size=self.hilbert.n_fermions)[0]
            for k, part in enumerate(parts):
                # reshape to orbital matrix
                M = part.reshape((self.hilbert.n_sites, self.hilbert.n_fermions))
                A = M[R, :]  # pick occupied rows
                logdet = nkjax.logdet_cmplx(A)
                log_terms.append(logdet + self.log_coeffs[k])
            # combine via logsumexp
            return jax.scipy.special.logsumexp(jnp.stack(log_terms))

        if n.ndim == 1:
            return single_logpsi(n)
        else:
            return jax.vmap(single_logpsi)(n)
#endregion