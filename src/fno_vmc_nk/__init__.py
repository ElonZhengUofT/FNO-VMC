"""FNO-VMC_Netket version package initialization."""
__version__ = "nk.1.0.1"

from .hamiltonian import make_hamiltonian
from .evaluate import evaluate_energy, evaluate_variance
from .AnsatzJax import make_ansatz_jax, BaseAnsatzJax

print(f"FNO-VMC version {__version__} initialized.")