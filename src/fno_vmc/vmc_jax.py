import netket as nk
import numpy as np
import jax
import optax


def random_init(rng_key, n_chains):
    # 生成形状 (n_chains, L) 的随机 {+1, -1} 自旋阵列
    L = hilbert.size
    # uniform in {0,1}, 然后映射到 ±1
    bits = jax.random.randint(rng_key, (n_chains, L), 0, 2)
    return 2 * bits - 1


class VMCTrainer:
    def __init__(self, hilbert, hamiltonian, ansatz_model, vmc_params, logger=None):
        # 1) prepare sampler
        sampler = nk.sampler.MetropolisLocal(hilbert)

        # 2) wrap the PyTorch model with TorchModule

        machine = ansatz_model

        # 3) prepare the MCState
        self.vstate = nk.vqs.MCState(
            sampler=sampler,
            model=machine,
            n_samples=vmc_params.get('n_samples', 1000),
            init_fun=random_init,
        )

        # 4) directly pass the optimizer to the VMC driver
        lr = float(vmc_params.get("lr", 1e-3))

        if vmc_params.get('sr', False):
            precond = make_sr_optimizer(
                diagshift=vmc_params.get("diagshift", 0.01))
            # jax_opt = optax.adam(learning_rate=vmc_params.get("lr", 1e-3))
            opt = nk.optimizer.SR(
                gradient_transform=optax.adam(learning_rate=lr),
                diagshift=float(vmc_params.get("diagshift", 0.01)),
            )
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
                preconditioner=precond,
            )
        else:
            opt = nk.optimizer.Adam(learning_rate=lr)
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
            )


        self.n_iter = vmc_params.get('n_iter', 200)
        self.logger = logger

    def run(self, out='result', logfile=None):
        if logfile:
            import logging
            logging.getLogger().addHandler(logging.FileHandler(logfile))

        def _wandb_callback(step, loss, params):
            print(f">>>> callback, step = {step}, energy = {loss.get('energy')}")
            if self.logger is not None:
                # Log the things you want to track
                energy = loss.get("energy")
                variance = loss.get("variance")
                # push to wandb
                self.logger.log({
                    "step": step,
                    "energy": energy,
                    "variance": variance
                    # "params": params
                }, step=step)
            return True
        #TODO: check if we can pass some information of param to wandb

        print(">>>>> VMCTrainer.run() 开始执行 —— out =", out)
        print(f">>>>> VMCTrainer.run()：self.n_iter = {self.n_iter}")
        self.driver.run(
            self.n_iter,
            out=out,
            callback=_wandb_callback
        )
        print(">>>>> VMCTrainer.run() 执行结束")
