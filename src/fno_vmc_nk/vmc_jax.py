import netket as nk
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import os
import wandb
import optax
import flax
from flax.core.frozen_dict import freeze, unfreeze

ENERGY_MIN, ENERGY_MAX = -100000, 100000

SLATER_STEPS = 700

def label_fn(path, _):
    return "slater" if path[0] == "slater" else "backflow"


#region VMCTrainer Class
class VMCTrainer:
    def __init__(self, hilbert, hamiltonian, ansatz_model, vmc_params, logger=None, phase=None,variables=None):
        """
        VMCTrainer Initialization
        """
        # 1) prepare sampler
        self.phase = phase
        sampler = nk.sampler.MetropolisLocal(
            hilbert,
            n_chains=64,
            n_sweeps=2
        )
        self._key = jax.random.PRNGKey(vmc_params.get("seed", 42))

        print("Hamiltonian:", hamiltonian)
        self.sampler = sampler

        # 2) wrap the PyTorch model
        machine = ansatz_model

        rngs = {"params": self._key}
        params = machine.init(rngs, jnp.zeros((hilbert.size,)))["params"]
        param_labels = flax.traverse_util.path_aware_map(
            lambda path, _: label_fn(path, _),
            params,
        )

        self.machine = machine

        # 3) prepare the MCState
        if phase == 2 and variables is not None:
            self.vstate = nk.vqs.MCState(
                sampler=sampler,
                model=machine,
                n_samples=vmc_params.get('n_samples', 1000),
                variables=variables,  # ← 传入 pre_trainer.vstate.variables
            )
        else:
            self.vstate = nk.vqs.MCState(
                sampler=sampler,
                model=self.machine,
                n_samples=vmc_params.get('n_samples', 1000),
                init_fun=hilbert.random_state
            )

        # 4) directly pass the optimizer to the VMC driver
        decay_steps = 100
        decay_rate = 0.95

        if phase == 1:
            lr = float(vmc_params.get("lr_slater", 1e-2))
            lr_schedule = optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True,  # 如果 False 就是连续衰减；True 每 decay_steps 衰减一次
                end_value=1e-4  # 可选：下限
            )
            slater_opt = optax.adam(learning_rate=lr_schedule)
            transform = optax.multi_transform(
                {"slater": slater_opt,
                 "backflow": optax.set_to_zero()},
                param_labels,
            )
            opt = transform
            print(">>> Phase 1 optimizer: only SLATER")
        elif phase == 2:
            lr = float(vmc_params.get("lr", 5e-3))
            lr_schedule = optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True,  # 如果 False 就是连续衰减；True 每 decay_steps 衰减一次
                end_value=1e-4  # 可选：下限
            )
            backflow_opt = optax.adam(learning_rate=lr_schedule)
            transform = optax.multi_transform(
                {"slater": optax.set_to_zero(),
                 "backflow": backflow_opt},
                param_labels,
            )
            opt = transform
            print(">>> Phase 2 optimizer: only BACKFLOW")
        else:
            lr = float(vmc_params.get("lr", 5e-3))
            lr_schedule = optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True,  # 如果 False 就是连续衰减；True 每 decay_steps 衰减一次
                end_value=1e-4  # 可选：下限
            )
                # jax_opt = optax.adam(learning_rate=vmc_params.get("lr", 1e-3))
            opt = nk.optimizer.Adam(learning_rate=lr_schedule)

        if vmc_params.get('sr', False):
            precond = nk.optimizer.SR(diag_shift=float(vmc_params.get("diagshift",
                           1e-3)))  # 1e-4 is a common default value
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
                preconditioner=precond,
            )
        else:
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
            )
        if self.phase == 1:
            self.n_iter = SLATER_STEPS
        else:
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

    # region Run Method
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

            if self.phase == 2:
                step_revised = step + SLATER_STEPS
            else:
                step_revised = step

            if self.logger is not None:
                # Log the things you want to track
                # push to wandb
                self.logger.log({
                    "train/energy": energy,
                    "train/variance": variance,
                    "train/acceptance": acceptance
                    # "params": params
                }, step=step_revised)
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
        #endregion

    # region Estimate Method
    def estimate(
            self,
            n_blocks: int = 50,
            block_size: int = 2000,
            burn_in: int = 1000,
            log: bool = True
    ):
        """
        冻结模型参数后，分块运行 MCMC 并对局部能量做时间平均：
        - n_blocks: 分块次数
        - block_size: 每块正式采样的 MCMC 步数
        - burn_in: 每块前的热身步数（丢弃）
        - log: 是否打印每块的均值
        """
        import numpy as _np
        print(f"\n=== Inference estimate: {n_blocks} blocks, ")

        # 重置采样状态
        block_means = []

        for i in range(n_blocks):
            # 每个 block：先丢弃 burn_in，再采 block_size 个样本
            self.vstate.n_discard_per_chain = burn_in
            self.vstate.n_samples = block_size
            self.vstate.reset()
            stats = self.vstate.expect(self.hamiltonian)
            m = float(stats.mean.real)
            block_means.append(m)
            if log:
                print(f"[Estimate] block {i + 1}/{n_blocks} mean = {m:.6f}")

        # 计算总体均值和标准误
        bm = _np.array(block_means)
        mean_e = float(bm.mean())
        stderr = float(bm.std(ddof=1) / _np.sqrt(len(bm)))

        print(
            f"\n=== Inference estimate: ⟨E⟩ = {mean_e:.6f} ± {stderr:.6f} ===")

        # 如果使用了 WandB，顺便记录
        if self.logger is not None:
            self.logger.log({
                "inference/energy_mean": mean_e,
                "inference/energy_stderr": stderr
            })
        print(">>>>> VMCTrainer.estimate() 执行结束")

        return mean_e, stderr
    #endregion
#endregion

