import netket as nk
import torch
import jax
from src.fno_vmc_torch.torch_module import TorchModule


def random_init(rng_key, n_chains):
    # 生成形状 (n_chains, L) 的随机 {+1, -1} 自旋阵列
    L = hilbert.size
    # uniform in {0,1}, 然后映射到 ±1
    bits = jax.random.randint(rng_key, (n_chains, L), 0, 2)
    return 2 * bits - 1


class VMCTrainer:
    def __init__(self, hilbert, hamiltonian, ansatz_model, vmc_params):
        # 1) prepare sampler
        sampler = nk.sampler.MetropolisLocal(hilbert)

        # 2) wrap the PyTorch model with TorchModule

        machine = TorchModule(
            ansatz_model,
            input_shape=(hilbert.size, ),
            dtype=torch.float32,
        )

        # 3) prepare the MCState
        self.vstate = nk.vqs.MCState(
            sampler=sampler,
            model=machine,
            n_samples=vmc_params.get('n_samples', 1000),
            init_fun=random_init,
        )

        # 4) directly pass the optimizer to the VMC driver
        if vmc_params.get('sr', False):
            precond = make_sr_optimizer(
                diagshift=vmc_params.get("diagshift", 0.01))
            torch_opt = topt.Adam(ansatz_model.parameters(),
                                  lr=vmc_params.get("lr", 1e-3))
            opt = Torch(torch_opt, preconditioner=precond)
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
                preconditioner=precond,
            )
        else:
            torch_opt = topt.Adam(ansatz_model.parameters(),
                                  lr=vmc_params.get("lr", 1e-3))
            opt = Torch(torch_opt)
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
            )


        self.n_iter = vmc_params.get('n_iter', 200)

    def run(self, out='result', logfile=None):
        if logfile:
            import logging
            logging.getLogger().addHandler(logging.FileHandler(logfile))
        self.driver.run(self.n_iter, out=out)
