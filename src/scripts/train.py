# scripts/train.py
import argparse
import logging
import os
import wandb
import torch.nn as nn
import netket as nk
from src.fno_vmc_nk.util import load_config, set_logger
from src.fno_vmc_nk.hamiltonian import make_hamiltonian
# from src.fno_vmc_nk.Ansatz import make_ansatz
from src.fno_vmc_nk.AnsatzJax import make_ansatz_jax
from src.fno_vmc_nk.vmc_jax import VMCTrainer
import jax.numpy as jnp
import jax
import time
import optax
import flax
from flax.core.frozen_dict import freeze, unfreeze

GROUND_STATES = {(2,7): -20.35, (4,7):-17.664, (4,8): -13.768,
                (8,8): -8.32,(8,7): -11.984,(8,6): -14.92,
                (8,4): -16.46,(8,2): -11.32}

GROUND_STATE = GROUND_STATES.get((2,7))  # Default value if not found

XLA_FLAGS="--xla_gpu_autotune_level=2"


def main():
    parser = argparse.ArgumentParser(description="Train VMC with different ansatz and models")
    parser.add_argument("--ansatz", choices=["fno", "tn", "SlaterFNO", "RBM", "Slater", "backflow"], required=True,
                        help="Type of variational ansatz to use: 'fno' or 'tn'.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Directory to save outputs (default: results)")
    parser.add_argument("--logfile", type=str, default=None,
                        help="Optional file to log training output")
    parser.add_argument("--wandb_project", type=str, default="FNO-VMC",
                        help="wandb project name")
    args = parser.parse_args()

    # setup output directory
    os.makedirs(args.outdir, exist_ok=True)

    # configure logging
    set_logger(logfile=args.logfile)
    logging.info(f"Starting training: ansatz={args.ansatz}, config={args.config}")

    # load configuration
    cfg = load_config(args.config)

    # initialize Weights & Biases
    wandb.login(key="9648574da18d2bf024ef72f8f5b196d410e674d4")
    wandb.init(
        project = args.wandb_project,
        config = cfg,
        name = f"{args.ansatz}-{os.path.basename(args.config)}",
        sync_tensorboard = False
    )
    wandb.watch_callable = lambda m: wandb.watch(m, log="all", log_freq=50)

    wandb.config.update(cfg, allow_val_change=True)

    artifact = wandb.Artifact(f"{args.ansatz}_config", type="config")
    artifact.add_file(args.config)
    wandb.log_artifact(artifact)

    # build Hamiltonian
    hilbert, hamiltonian, graph = make_hamiltonian(
        ham_type=cfg["hamiltonian"]["type"],
        params=cfg["hamiltonian"]["params"]
    )

    # 在 train.py 中 make_ansatz 之前
    # logging.info(f"Creating ansatz with params: {cfg.get('model_params')}")

    # build ansatz
    # rng = jax.random.PRNGKey(0)
    model = make_ansatz_jax(
        kind=args.ansatz if args.ansatz != None else "fno",
        dim=cfg["hamiltonian"]["params"]["dim"],
        hilbert=hilbert,
        **cfg.get("model_params", {})
    )
    # Lx = cfg["hamiltonian"]["params"]["L"]
    # Ly = cfg["hamiltonian"]["params"]["L"] if cfg["hamiltonian"]["params"]["dim"] == 2 else 1
    # dummy_x = jnp.ones((1, Lx * Ly)) if cfg["hamiltonian"]["params"]["dim"] == 2 else jnp.ones((Lx * Ly, 1))
    # params = model.init(rng, dummy_x)
    # states = hilbert.random_state(jax.random.PRNGKey(42), 100)  # 100组随机自旋
    # log_psi = model.apply(params, states)
    # print("log_psi mean:", log_psi.mean(), "std:", log_psi.std())
    # These sentences is for the test of init of flax model


    # Cold start: log ansatz type and parameters
    #     for p in model.parameters():
    #         p.data.zero_()
    # wandb.watch_callable(model)
    params = cfg["hamiltonian"]["params"]
    ground_state = GROUND_STATES.get((params.get("U"), params.get("n_particles")[0]), GROUND_STATE)

    # train the model
    # model = nk.models.RBM() # A Test model, replace when debug is done
    print("=== Stage 1: training only Slater parameters ===")
    pre_trainer = VMCTrainer(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        ansatz_model=model,
        phase=1,  # 指定为第一阶段
        vmc_params={**cfg.get("vmc", {})},
        logger=wandb,
        ground_state=ground_state,
    )
    pre_trainer.run(out=os.path.join(args.outdir, "phase1"),
                 logfile=args.logfile)

    print("=== Stage 2: training only Backflow MLP ===")
    trainer = VMCTrainer(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        ansatz_model=pre_trainer.vstate.model,  # 使用阶段一结束时的参数
        phase=2,  # 指定为第二阶段
        variables=pre_trainer.vstate.variables,
        vmc_params={**cfg.get("vmc", {})},
        logger=wandb,
        ground_state=ground_state,
    )
    trainer.run(out=os.path.join(args.outdir, "phase2"),
                 logfile=args.logfile)
    trainer.estimate()
    print("Training finished.")

    trainer = VMCTrainer(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        ansatz_model= model,
        vmc_params=cfg.get("vmc", {}),
        logger = wandb,
        ground_state=ground_state,
    )



if __name__ == '__main__':
    main()
