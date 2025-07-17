import netket as nk
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import os
import wandb
import optax
import flax
import time
import functools
import jax.scipy.sparse.linalg as jsp
from netket.optimizer.qgt import QGTAuto,QGTOnTheFly
import netket.experimental as nkx
from netket.optimizer import SR, LinearPreconditioner
from src.fno_vmc_nk.VMC.MARCH import MARCH
# from nkx.driver import VMC_SRt
from flax.core.frozen_dict import freeze, unfreeze


ENERGY_MIN, ENERGY_MAX = -100000, 100000

SLATER_STEPS = 300

SPLIT = 4 # The number to split the batch of samples

GROUND_STATES = {(2,7): -20.35, (4,7):-17.664, (4,8): -13.768,
                (8,8): -8.32,(8,7): -11.984,(8,6): -14.92,
                (8,4): -16.46,(8,2): -11.32}

GROUND_STATE = -11.36

def label_fn(path, _):
    return "slater" if path[0] == "slater" else "backflow"


#region VMCTrainer Class
class VMCTrainer:
    def __init__(self, hilbert, hamiltonian,graph, ansatz_model, vmc_params, logger=None, phase=None,variables=None, ground_state=GROUND_STATE):
        """
        VMCTrainer Initialization
        """
        # 1) prepare sampler
        self.phase = phase

        self.split_batches = int(vmc_params.get("split_batches", SPLIT))
        chunk_size = vmc_params.get('n_samples', 1000) // self.split_batches
        print(f"chunk_size = {chunk_size}, split_batches = {self.split_batches}")

        if phase == 1:
            sampler = nk.sampler.MetropolisLocal(
                hilbert,
                n_chains_per_rank=64,
                sweep_size=1,
            )
        else:
            sampler = nk.sampler.MetropolisFermionHop(
                hilbert=hilbert,  # 必须
                graph=graph,  # 通常必须（除非 cluster 覆盖）
                d_max=1,  # 可选，默认=1
                spin_symmetric=True,  # 可选，默认=True
                n_chains=64,  # 可选
            )

        self._key = jax.random.PRNGKey(vmc_params.get("seed", 42))

        print("Hamiltonian:", hamiltonian)
        self.sampler = sampler

        # 2) wrap the PyTorch model
        machine = ansatz_model

        rngs = {"params": self._key}
        params = machine.init(rngs, jnp.zeros((1,hilbert.size)))["params"]
        param_labels = flax.traverse_util.path_aware_map(
            lambda path, _: label_fn(path, _),
            params,
        )

        self.machine = machine

        # 3) prepare the MCState
        if (phase == 2 or phase == 3) and variables is not None:
            self.vstate = nk.vqs.MCState(
                sampler=sampler,
                model=machine,
                n_samples=vmc_params.get('n_samples', 1000),
                variables=variables,  # ← 传入 pre_trainer.vstate.variables
                chunk_size=chunk_size,

            )
        else:
            self.vstate = nk.vqs.MCState(
                sampler=sampler,
                model=self.machine,
                n_samples=vmc_params.get('n_samples', 1000),
                init_fun=hilbert.random_state,
                chunk_size=chunk_size,
            )
################################################################################
        #     ----     ------    ---------
        # |       |   |      |       |
        # |       |   |------        |
        #   _____     |              |
################################################################################
        # 4) directly pass the optimizer to the VMC driver
        decay_steps = 100
        decay_rate = 0.85

        if phase == 1:
            lr = float(vmc_params.get("lr_slater", 1e-2))
            lr_schedule = optax.exponential_decay(
                init_value=lr,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True,  # 如果 False 就是连续衰减；True 每 decay_steps 衰减一次
                end_value=1e-5  # 可选：下限
            )
            slater_opt = optax.adamw(learning_rate=lr_schedule)
            transform = optax.multi_transform(
                {"slater": slater_opt,
                 "backflow": optax.set_to_zero()},
                param_labels,
            )
            opt = transform
            print(">>> Phase 1 optimizer: only SLATER")
        elif phase == 2:
            lr = float(vmc_params.get("lr", 5e-4))
            lr_schedule = optax.exponential_decay(
                init_value=1e-2,
                transition_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True,  # 如果 False 就是连续衰减；True 每 decay_steps 衰减一次
                end_value=1e-5  # 可选：下限
            )
            slater_opt = optax.adamw(learning_rate=lr_schedule)
            backflow_opt = optax.adamw(learning_rate=lr_schedule)
            transform = optax.multi_transform(
                {"slater": slater_opt,
                 "backflow": backflow_opt},
                param_labels,
            )
            opt = transform
            print(">>> Phase 2 optimizer: only BACKFLOW")
        else:
            lr = float(vmc_params.get("lr", 5e-3)) * 0.1  # 降低学习率
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
            diag_schedule = optax.exponential_decay(
                init_value=float(vmc_params.get("diagshift", 1e-2)),
                transition_steps=100,
                decay_rate=0.8,
                staircase=True,  # 如果 False 就是连续衰减；True 每 decay_steps 衰减一次
                end_value=1e-4  # 可选：下限
            )
            print(">>> Using SR preconditioner")
            precond = MARCH(
                qgt= QGTOnTheFly(),  # 或者 QGTAuto()
                # QFTOnTheFly() or QGTAuto()
                diag_shift=diag_schedule,
                solver=functools.partial(
                    jsp.cg,
                    tol=1e-4,  # 设置求解器的容忍度
                    maxiter=25,  # 最大迭代次数
                )
            )  # 1e-4 is a common default value
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
                preconditioner=precond,
            )
        else:
            print(">>> Not using SR preconditioner")
            self.driver = nk.driver.VMC(
                hamiltonian,
                opt,
                variational_state=self.vstate,
            )
        ########################################################################
        #     ----     ------    ---------
        # |       |   |      |       |
        # |       |   |------        |
        #   _____     |              |
        ########################################################################

        # 5) set the number of iterations
        if self.phase == 1:
            self.n_iter = SLATER_STEPS
        else:
            self.n_iter = vmc_params.get('n_iter', 2000)
        print(f"SLATER_STEPS = {SLATER_STEPS}")

        self.logger = logger

        self.hilbert = hilbert
        self.hamiltonian = hamiltonian

        self.energy_list = []
        self.variance_list = []
        self.acceptance_list = []
        self.step_list = []

        self.log_freq = int(vmc_params.get("log_freq", 2))

        self.ground_state = ground_state
        print(f"Ground state energy: {self.ground_state}")

        self._last_time = time.perf_counter()

        #         self._switch_at = int(vmc_params.get("switch_at", 150))
        #         self._new_diag = 1e-4  # New diagonal shift for SR optimizer after switch_at

        # Record the initial parameters
        if self.logger is not None:
            base = {
                "phase": self.phase,
                "n_iter": self.n_iter,
                "n_split": self.split_batches,
                "n_samples": vmc_params.get('n_samples', 1000),
                "split_batches": self.split_batches,
                "ground_state": ground_state,
                "log_freq": self.log_freq,
                "optimizer": str(opt),
                "sampler": str(sampler),
                "variational_state": str(self.vstate),
                "machine": str(self.machine),
                "hilbert": str(hilbert),
                "initial_parameters": unfreeze(self.vstate.variables)
            }
            suffix = f"_{self.phase}" if self.phase is not None else ""
            to_log = {f"{k}{suffix}": v for k, v in base.items()}
            self.logger.config.update(to_log)
        #endregion

            # self.logger.watch(self.vstate.model, log="all", log_freq=self.log_freq)
        # print the initial parameters
        print("=== VMCTrainer 初始化完成 ===")
        print(f"Phase: {self.phase}, n_iter: {self.n_iter}, "
                f"split_batches: {self.split_batches}, "
                f"ground_state: {self.ground_state}, log_freq: {self.log_freq}")
        print(f"Optimizer: {self.driver.optimizer}")
        print(f"Sampler: {self.sampler}")
        print(f"Variational State: {self.vstate}")
        print(f"Machine: {self.machine}")
        print(f"Hilbert space: {self.hilbert}")
        print(f"Hamiltonian: {self.hamiltonian}")
        # print(f"Initial parameters: {self.vstate.variables}")
        print("=====================================")

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

            now = time.perf_counter()
            dt = now - self._last_time
            # 这次 callback 覆盖了 log_freq 步
            sec_per_step = dt / self.log_freq if dt > 0 else float("nan")
            self._last_time = now

            acceptance = float(loss.get("acceptance", np.nan))
            stats = loss["Energy"]
            energy = float(stats.mean.real)  # e.g. 28.33
            variance = float(stats.variance.real)  # e.g. 77.66

            print(f">>>> callback, step = {step}, "
                  f"energy = {energy:.4f}, variance = {variance:.4f}, "
                  f"acceptance = {acceptance:.4f}")

            if energy < ENERGY_MIN or energy > ENERGY_MAX:
                energy = np.nan

            relative_error = (energy - self.ground_state) / abs(self.ground_state) if self.ground_state != 0 else np.nan

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
                    "train/acceptance": acceptance,
                    "train/relative_error": relative_error,
                    "train/log_relative_error": np.log10(abs(relative_error)) if relative_error != 0 else np.nan,
                    "train/velocity": sec_per_step,
                    "train/log_velocity": np.log10(sec_per_step) if sec_per_step > 0 else np.nan,
                    "train/log_variance": np.log10(variance) if variance > 0 else np.nan,
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
                print(f"[Estimate] block {i + 1}/{n_blocks} variance = {stats.variance.real:.6f}")

        # 计算总体均值和标准误
        bm = _np.array(block_means)
        mean_e = float(bm.mean())
        stderr = float(bm.std(ddof=1) / _np.sqrt(len(bm)))
        variance = float(bm.var(ddof=1))

        print(
            f"\n=== Inference estimate: ⟨E⟩ = {mean_e:.6f} ± {stderr:.6f} ===")

        relative_error = (mean_e - self.ground_state) / abs(self.ground_state) if self.ground_state != 0 else np.nan

        # 如果使用了 WandB，顺便记录
        if self.logger is not None:
            self.logger.log({
                "inference/energy_mean": mean_e,
                "inference/energy_stderr": stderr,
                "inference/energy_variance": variance,
                "inference/relative_error": relative_error
            })
        print(">>>>> VMCTrainer.estimate() 执行结束")

        return mean_e, stderr
    #endregion

    # region Save samples-psi pair
    def dump_orbitals_dataset(
            self,
            n_samples: int = 4096+1024,
            burn_in: int = 1024,
            out_path: str = "pretrain_dataset/slater_dataset.npz"
    ):
        """
        冻结当前模型参数，采样 n_samples 个配置 (丢弃 burn_in)，
        并针对每个配置计算 Slater orbitals 矩阵 M(n)，最后保存到 .npz。
        Freeze the current model parameters, sample n_samples configurations (discard burn_in),
        and calculate the Slater orbitals matrix M(n) for each configuration, and finally save it to .npz.
        """
        # 1) Reset sampler & discard热身
        self.vstate.n_discard_per_chain = burn_in
        self.vstate.n_samples = n_samples
        self.vstate.reset()

        # 2) 采样 n_samples 配置
        samples = self.vstate.sample()  # shape (n_samples, 2*N_sites)

        # 3) 用当前 Slater ansatz 计算 M(n)
        #    假设 self.vstate.model.apply 返回 shape (2*N_sites, N_e)
        slater_apply = self.vstate.model.apply
        params = self.vstate.parameters

        # jit + vmap
        @jax.jit
        def _get_M(n):
            return slater_apply({"params": params}, n)

        M_all = jax.vmap(_get_M)(samples)  # (n_samples, 2N_sites, N_e)

        # 4) 保存到磁盘
        np.savez_compressed(
            out_path,
            samples=np.array(samples),
            M_all=_np.array(M_all),
        )
        print(f"Saved slater dataset → {out_path}")
        return samples, M_all
    #endregion
#endregion

