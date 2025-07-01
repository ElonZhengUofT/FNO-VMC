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
            n_chains=64,
            n_sweeps=2
        )
        self._key = jax.random.PRNGKey(vmc_params.get("seed", 42))
        self.local_energy_fn = nk.vqs.local_value(hamiltonian)

        print("Hamiltonian:", hamiltonian)
        self.sampler = sampler

        # 2) wrap the PyTorch model
        machine = ansatz_model
        self.machine = machine
        self.params = machine.parameters

        # 3) prepare the MCState
        self.vstate = nk.vqs.MCState(
            sampler=sampler,
            model=machine,
            n_samples=vmc_params.get('n_samples', 1000),
            init_fun=hilbert.random_state
        )

        # 4) directly pass the optimizer to the VMC driver
        lr = float(vmc_params.get("lr", 1e-3))
        decay_steps = 100
        decay_rate = 0.95

        if vmc_params.get('sr', False):
            print(">>>>> VMCTrainer: Using SR optimizer")
            # jax_opt = optax.adam(learning_rate=vmc_params.get("lr", 1e-3))
            lr_schedule = optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True,  # 如果 False 就是连续衰减；True 每 decay_steps 衰减一次
                end_value=1e-4  # 可选：下限
            )
            opt = nk.optimizer.Adam(learning_rate=lr_schedule)
            precond = nk.optimizer.SR(diag_shift=float(
                vmc_params.get("diagshift", 1e-5))) # 1e-4 is a common default value
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


        self.n_iter = vmc_params.get('n_iter', 2000)

        self.logger = logger

        self.hilbert = hilbert
        self.hamiltonian = hamiltonian

        self.energy_list = []
        self.variance_list = []
        self.acceptance_list = []
        self.step_list = []

        self.log_freq = int(vmc_params.get("log_freq", 2))

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

            # if energy < ENERGY_MIN or energy > ENERGY_MAX:energy = np.nan

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

    def estimate(self, n_block, block_size, burn_in: int=1000, log: bool=True):
        """
        Estimate the energy and variance of the current state.
        - `n_block`: Number of blocks to average over.
        - `block_size`: Size of each block.
        - burn_in: Number of initial samples to discard.

        """
        energies = []
        for i in range(n_blocks):
            # Burn-in phase
            self._key, subkey = jax.random.split(self._key)
            self.sampler.run(subkey, n_steps=burn_in, params=self.params)

            # Collect samples
            self._key, subkey = jax.random.split(self._key)
            samples = self.sampler.run(subkey, n_steps=block_size,
                                       params=self.params)

            # Estimate energy and variance
            e_loc = self.local_energy_fn(self.params, samples)
            energies.append(e_loc)

            if log:
                print(f"[Estimate] block {i + 1}/{n_blocks} done.")

            all_e = jnp.concatenate(energies)
            mean_e = jnp.mean(all_e).item()
            stderr = (jnp.std(all_e) / jnp.sqrt(all_e.shape[0])).item()
            print(
                f"\n=== Inference estimate: ⟨E⟩ = {mean_e:.6f} ± {stderr:.6f} ===")

            if self.logger is not None:
                self.logger.log({
                    "estimate/energy": mean_e,
                    "estimate/variance": stderr
                })
            return mean_e, stderr
