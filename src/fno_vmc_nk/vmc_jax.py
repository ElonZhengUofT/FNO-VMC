import netket as nk
import numpy as np
import jax
import optax
import matplotlib.pyplot as plt
import os
import wandb

ENERGY_MIN, ENERGY_MAX = -100000, 100000

class VMCTrainer:
    def __init__(self, hilbert, hamiltonian, ansatz_model, vmc_params, logger=None):
        # 1) prepare sampler
        sampler = nk.sampler.MetropolisLocal(
            hilbert,
            n_chains=128,
            n_sweeps=2
        )

        print("Hamiltonian:", hamiltonian)

        # 2) wrap the PyTorch model with TorchModule

        machine = ansatz_model

        # 3) prepare the MCState
        self.vstate = nk.vqs.MCState(
            sampler=sampler,
            model=machine,
            n_samples=vmc_params.get('n_samples', 1000),
            init_fun=hilbert.random_state
        )

        # 4) directly pass the optimizer to the VMC driver
        lr = float(vmc_params.get("lr", 1e-3))

        if vmc_params.get('sr', False):
            print(">>>>> VMCTrainer: Using SR optimizer")
            # jax_opt = optax.adam(learning_rate=vmc_params.get("lr", 1e-3))
            opt = nk.optimizer.Adam(learning_rate=lr)
            precond = nk.optimizer.SR(diag_shift=float(
                vmc_params.get("diagshift", 1e-3))) # 1e-4 is a common default value
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
                preconditioner=precond,
            )
        else:
            print(">>>>> VMCTrainer: Using Adam optimizer")
            opt = nk.optimizer.Adam(learning_rate=lr)
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
            )


        self.n_iter = vmc_params.get('n_iter', 200)

        self.logger = logger

        self.hilbert = hilbert
        self.hamiltonian = hamiltonian

        self.energy_list = []
        self.variance_list = []
        self.acceptance_list = []
        self.step_list = []

        self.log_freq = int(vmc_params.get("log_freq", 1))

        #         self._switch_at = int(vmc_params.get("switch_at", 150))
        #         self._new_diag = 1e-4  # New diagonal shift for SR optimizer after switch_at

    def run(self, out='result', logfile=None):
        if logfile:
            import logging
            logging.getLogger().addHandler(logging.FileHandler(logfile))

        def _wandb_callback(step, loss, params):
            #             if step == self._switch_at:
            #                 print(f"==> Step {step}: 增大 SR diagshift 到 {self._new_diag}")
            #                 self.driver.preconditioner = nk.optimizer.SR(
            #                     diag_shift=self._new_diag
            #                 )

            if step % self.log_freq != 0:
                return True

            print(f">>>> callback, step = {step}, energy = {loss.get('Energy')}")
            samples = self.vstate.samples

            acceptance = float(loss.get("acceptance", np.nan))
            stats = loss["Energy"]
            energy = float(stats.mean.real)  # e.g. 28.33
            variance = float(stats.variance.real)  # e.g. 77.66

            print(f">>>> callback, step = {step}, "
                  f"energy = {energy:.4f}, variance = {variance:.4f}, "
                  f"acceptance = {acceptance:.4f}")

            if energy < ENERGY_MIN or energy > ENERGY_MAX:
                energy = np.nan

            if self.logger is not None:
                # Log the things you want to track
                # push to wandb
                self.logger.log({
                    "train/energy": energy,
                    "train/variance": variance,
                    "train/acceptance": acceptance
                    # "params": params
                }, step=step)
            return True
        # check if we can pass some information of param to wandb

        print(">>>>> VMCTrainer.run() 开始执行 —— out =", out)
        print(f">>>>> VMCTrainer.run()：self.n_iter = {self.n_iter}")
        self.driver.run(
            self.n_iter,
            out=out,
            callback=_wandb_callback
        )

        print(">>>>> VMCTrainer.run() 执行结束")
