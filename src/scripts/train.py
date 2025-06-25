# scripts/train.py
import argparse
import logging
import os
import wandb
import torch.nn as nn
import netket as nk
from src.fno_vmc.util import load_config, set_logger
from src.fno_vmc.hamiltonian import make_hamiltonian
# from src.fno_vmc.Ansatz import make_ansatz
from src.fno_vmc.AnsatzJax import make_ansatz_jax
from src.fno_vmc.vmc_jax import VMCTrainer
import jax.numpy as jnp
import jax


def main():
    parser = argparse.ArgumentParser(description="Train VMC with different ansatz and models")
    parser.add_argument("--ansatz", choices=["fno", "tn"], required=True,
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
        name = f"{args.ansatz}-{os.path.basename(args.config)}"
    )
    wandb.watch_callable = lambda m: wandb.watch(m, log="all", log_freq=50)

    # build Hamiltonian
    hilbert, hamiltonian = make_hamiltonian(
        ham_type=cfg["hamiltonian"]["type"],
        params=cfg["hamiltonian"]["params"]
    )

    # 在 train.py 中 make_ansatz 之前
    # logging.info(f"Creating ansatz with params: {cfg.get('model_params')}")

    # build ansatz
    rng = jax.random.PRNGKey(0)
    model = make_ansatz_jax(
        kind=args.ansatz,
        dim=cfg["hamiltonian"]["params"]["dim"],
        **cfg.get("model_params", {})
    )
    Lx = cfg["hamiltonian"]["params"]["L"]
    Ly = cfg["hamiltonian"]["params"]["L"] if cfg["hamiltonian"]["params"]["dim"] == 2 else 1
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

    # train the model
    model = nk.models.RBM() # A Test model, replace when debug is done
    trainer = VMCTrainer(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        ansatz_model= model,
        vmc_params=cfg.get("vmc", {}),
        logger = wandb
    )
    trainer.run(out=os.path.join(args.outdir, args.ansatz), logfile=args.logfile)


if __name__ == '__main__':
    main()