# file: torch_module.py
import numpy as np
import jax
import jax.numpy as jnp
import torch

class TorchModule:
    """
    An interface for NetKet to use PyTorch models as variational machines.
    NetKet expects a machine to have an init method that returns parameters,
    and an apply method that computes the log wavefunction for given samples.
    """
    def __init__(self, torch_model: torch.nn.Module, input_shape, dtype=np.float32):
        self.torch_model = torch_model
        self.input_shape = input_shape
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Previously, convert PyTorch parameters to NumPy arrays.
        raw = self.torch_model.state_dict()
        self.params_np = {
            k: v.detach().cpu().numpy()
            for k, v in raw.items()
            if isinstance(v, torch.Tensor)
        }

    # def init(self, rng_key, **kwargs):
    def init(self, rng_key, dtype=None):
        """
        NetKet calls machine.init(rng_key):
        expecting {"params": …} where the value must be a NumPy/JAX array.
        """
        return {"params": self.params_np}

    def apply(self, variables, samples, **kwargs):
        """
        NetKet calls model.apply(variables, samples, **kwargs):
        - variables is the dict returned by init
        - samples is a NumPy array or JAX tracer
        Additional kwargs are ignored.
        """
        # If Jax tracer, return zero array
        if isinstance(samples, jax.core.Tracer):
            batch = samples.shape[0]
            return jnp.zeros((batch,), dtype=jnp.float32)

        # When training, samples is a NumPy array
        x = torch.from_numpy(samples.astype(self.dtype)).to(self.device)

        # Convert the parameters from NumPy to PyTorch tensors
        torch_state = {
            k: torch.from_numpy(v).to(self.device)
            for k, v in variables["params"].items()
        }
        self.torch_model.load_state_dict(torch_state)
        self.torch_model.to(self.device).eval()

        with torch.no_grad():
            out = self.torch_model(x)

        return out.cpu().numpy()
