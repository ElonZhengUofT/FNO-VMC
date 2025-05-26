import netket as nk
import torch
import numpy as np

class VMCTrainer:
    def __init__(self, hilbert, hamiltonian, ansatz_model, vmc_params):
        # 准备 sampler
        sampler = nk.sampler.MetropolisLocal(hilbert)
        self.torch_model = ansatz_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        raw = self.torch_model.state_dict()
        raw = self.torch_model.state_dict()
        self.params_np = {
            k: v.detach().cpu().numpy()
            for k, v in raw.items()
            if isinstance(v, torch.Tensor)
        }

        def init_fun(model, rng_key):
            return {"params": self.params_np}

        def apply_fun(variables, spins):
            params = variables['params']
            x = torch.from_numpy(spins).to(self.device)
            torch_state = {k: torch.from_numpy(v).to(self.device) for k, v in params.items()}
            self.torch_model.load_state_dict(torch_state)
            self.torch_model.to(self.device).eval()
            with torch.no_grad():
                out = self.torch_model(x)
            return out.cpu().numpy()

        # 2) 构造 MCState
        self.vstate = nk.vqs.MCState(
            sampler=sampler,
            model=None,
            n_samples=vmc_params.get('n_samples', 1000),
            init_fun=init_fun,
            apply_fun=apply_fun,
        )

        # 3) 直接把 optimizer 传给 VMC 驱动
        if vmc_params.get('sr', False):
            precond = make_sr_optimizer(diagshift=vmc_params.get('diagshift', 0.01))
            opt = nk.optimizer.Adam(learning_rate=vmc_params.get('lr', 1e-3))
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
                preconditioner=precond,
            )
        else:
            opt = nk.optimizer.Adam(learning_rate=vmc_params.get('lr', 1e-3))
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
