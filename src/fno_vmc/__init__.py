"""FNO-VMC package initialization."""
__version__ = "0.1.1"

from .Ansatz import make_ansatz, BaseAnsatz
from .hamiltonian import make_hamiltonian
from .vmc import VMCTrainer
from .evaluate import evaluate_energy, evaluate_variance
from .AnsatzJax import make_ansatz_jax, BaseAnsatzJax

print(f"FNO-VMC version {__version__} initialized.")