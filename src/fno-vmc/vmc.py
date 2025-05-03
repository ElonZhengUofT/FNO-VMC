import netket as nk
import logging
from .sr_optimizer import make_sr_optimizer


class VMCTrainer:
    """Variational Monte Carlo training driver."""

    def __init__(
        self,
        hilbert,
        hamiltonian,
        ansatz_model,
        vmc_params: dict
    ):
        # Sampler
        sampler = nk.sampler.MetropolisLocal(hilbert)
        # MC state
        n_samples = vmc_params.get('n_samples', 1000)
        self.vstate = nk.vqs.MCState(sampler, ansatz_model, n_samples=n_samples)
        # Optimizer
        if vmc_params.get('sr', False):
            diag = vmc_params.get('diagshift', 0.01)
            optimizer = make_sr_optimizer(diagshift=diag)
        else:
            lr = vmc_params.get('lr', 1e-3)
            optimizer = nk.optimizer.Adam(learning_rate=lr)
        self.vstate.create_optimizer(optimizer)
        # VMC driver
        self.driver = nk.driver.VMC(hamiltonian, variational_state=self.vstate)
        self.n_iter = vmc_params.get('n_iter', 200)
        logging.info(f"VMCTrainer initialized: sr={vmc_params.get('sr', False)}, samples={n_samples}")

    def run(self, out: str = 'result', logfile: str = None):
        """
        Run VMC optimization.
        out: directory prefix for outputs
        logfile: optional path to log file
        """
        if logfile:
            logging.getLogger().addHandler(logging.FileHandler(logfile))
        self.driver.run(n_iter=self.n_iter, out=out)
