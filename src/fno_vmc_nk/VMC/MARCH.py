import functools
from collections.abc import Callable
\import jax
import jax.numpy as jnp

from netket.vqs import VariationalState
from netket.utils.types import Scalar, ScalarOrSchedule
from netket.utils import struct
from netket.optimizer.preconditioner import AbstractLinearPreconditioner
from netket.optimizer.qgt import QGTAuto

class MARCH(AbstractLinearPreconditioner, mutable=True):
    r"""
    Moment-Adaptive ReConfiguration Heuristic (MARCH) preconditioner.

    This combines natural-gradient SR with Adam-style first/second moment adaptation.
    """
    # SR regularization
    diag_shift: ScalarOrSchedule = struct.field(serialize=False, default=1e-3)
    # Adaptive second-moment decay
    beta: float = struct.field(serialize=False, default=0.9)
    # Momentum coefficient
    mu: float = struct.field(serialize=False, default=0.9)
    # Additional ridge term inside SR solve
    lambda_reg: float = struct.field(serialize=False, default=1e-4)

    # QGT constructor args
    qgt_constructor: Callable = struct.static_field(default=None)
    qgt_kwargs: dict     = struct.field(serialize=False, default=None)

    def __init__(
        self,
        qgt: Callable | None = None,
        solver: Callable = jax.scipy.sparse.linalg.cg,
        *,
        diag_shift: ScalarOrSchedule = 1e-3,
        beta: float = 0.9,
        mu: float = 0.9,
        lambda_reg: float = 1e-4,
        solver_restart: bool = False,
        **kwargs,
    ):
        if qgt is None:
            qgt = QGTAuto(solver)
        self.diag_shift = diag_shift
        self.beta       = beta
        self.mu         = mu
        self.lambda_reg = lambda_reg
        self.qgt_constructor = qgt
        self.qgt_kwargs      = kwargs
        # history placeholders
        self.prev_delta = None
        self.V           = None

        super().__init__(solver, solver_restart=solver_restart)

    def lhs_constructor(self, vstate: VariationalState, step: int | None = None):
        # mirror SR: build QGT operator S + diag_shift*I
        diag = self.diag_shift(step) if callable(self.diag_shift) else self.diag_shift
        return self.qgt_constructor(vstate, diag_shift=diag, **self.qgt_kwargs)

    def apply(self, vstate: VariationalState, grad, step: int | None = None):
        # Flatten gradient pytree to vector
        flat_grad, unravel = jax.flatten_util.ravel_pytree(grad)
        n = flat_grad.shape[0]

        # Initialize history on first call
        if self.prev_delta is None:
            self.prev_delta = jnp.zeros(n)
            self.V           = jnp.ones(n)

        # Momentum term from previous delta
        phi = self.mu * self.prev_delta

        # Build scaled metric preconditioner: D = diag(V)^{1/4}
        D_inv = self.V ** (-0.25)

        # Construct QGT operator
        qgt = self.lhs_constructor(vstate, step)
        # Define matvec for (D^{-1} S D^{-1} + lambda_reg I)
        def matvec(x):
            # x is flat vector
            y = D_inv * qgt.matvec(D_inv * x)
            return y + self.lambda_reg * x
        linop = jax.scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec)

        # Solve linear system: (..) * y = flat_grad - phi
        rhs = flat_grad - phi
        y, _ = self.solver(linop, rhs)

        # Compute new delta
        delta = D_inv * y + phi

        # Update second moment: V = beta V + (1-beta)*(delta - prev_delta)^2
        diff     = delta - self.prev_delta
        self.V   = self.beta * self.V + (1 - self.beta) * (diff ** 2)
        # Store new delta
        self.prev_delta = delta

        # Unflatten back to pytree and return
        return unravel(delta)

    def __repr__(self):
        return (
            f"{type(self).__name__}(diag_shift={self.diag_shift}, beta={self.beta}, "
            f"mu={self.mu}, lambda_reg={self.lambda_reg})"
        )
