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
from src.fno_vmc_nk.ansatz.FNOAnsatz import AnsatzI, AnsatzII, AnsatzIII, AnsatzIV, AnsatzV
from src.fno_vmc_nk.ansatz.BackflowFNO import BackflowI, BackflowII

from flax.serialization import to_bytes, from_bytes
from flax.core.frozen_dict import freeze, unfreeze

XLA_FLAGS="--xla_gpu_autotune_level=2"

# jax.config.update("jax_enable_x64", True)


def save_flax_params(variables, path):
    """把 Flax 的参数树序列化成二进制，然后写文件。"""
    param_bytes = to_bytes(variables)
    with open(path, "wb") as f:
        f.write(param_bytes)
    print(f"Saved parameters to {path}")

def load_flax_params(path, trainer=None):
    with open(path, "rb") as f:
        param_bytes = f.read()

    variables = from_bytes(trainer.vstate.variables, param_bytes)
    print(f"Loaded parameters from {path}")
    return variables

def TFS(size=16):
    parser = argparse.ArgumentParser(description="Train VMC with different ansatz and models")
    # parser.add_argument("--ansatz", choices=["fno", "tn", "SlaterFNO", "RBM", "Slater", "backflow"], required=True,
                        # help="Type of variational ansatz to use: 'fno' or 'tn'.")
    # parser.add_argument("--config", type=str, required=True,
                        # help="Path to YAML config file.")
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
    logfile_path = "logs/fno_run.log"
    set_logger(logfile=logfile_path)
    logging.info(f"Starting training: ansatz= Slater, config=./configs_pretrain/one_dim_hubbard_fno_{size}.yaml")

    # load configuration
    config_path = f'./configs_pretrain/one_dim_hubbard_fno_{size}.yaml'
    cfg = load_config(config_path)

    # initialize Weights & Biases
    wandb.login(key="9648574da18d2bf024ef72f8f5b196d410e674d4")
    wandb.init(
        project = args.wandb_project,
        config = cfg,
        name = f"Slater_pretrain_fno_{size}",
        sync_tensorboard = False,
        reinit = True,
        resume=False
    )
    wandb.watch_callable = lambda m: wandb.watch(m, log="all", log_freq=50)

    wandb.config.update(cfg, allow_val_change=True)

    artifact = wandb.Artifact(f"config_fno_{size}", type="config")
    artifact.add_file(config_path)
    wandb.log_artifact(artifact)

    # build Hamiltonian
    hilbert, hamiltonian, graph = make_hamiltonian(
        ham_type=cfg["hamiltonian"]["type"],
        params=cfg["hamiltonian"]["params"]
    )

    #     model = make_ansatz_jax(
    #         kind="slater",
    #         dim=cfg["hamiltonian"]["params"]["dim"],
    #         hilbert=hilbert,
    #         **cfg.get("model_params", {})
    #     )
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((hilbert.size,), dtype=jnp.int32)
    template_A = nk.models.Slater2nd(
        hilbert=hilbert,
        generalized=True,
        restricted=True,
    ).init(rng, dummy_input)

    with open(f'./pretrain_dataset/fno_slater_pretrain_{size}.flax', "rb") as f:
        param_bytes = f.read()

    loaded_A = from_bytes(template_A, param_bytes)

    rng2 = jax.random.PRNGKey(1)
    matrix = AnsatzI()
    model = BackflowII(backflow_fn=matrix, hilbert=hilbert).init(rng2, dummy_input)

    full = unfreeze(model)
    full['params']['slater'] = loaded_A['params']
    merged = freeze(full)

    params = cfg["hamiltonian"]["params"]
    ground_state = cfg.get("vmc", {}).get("GS", None)
    print(f"Using ground state energy: {ground_state}")

    # train the model
    # model = nk.models.RBM() # A Test model, replace when debug is done
    print("=== Stage 1: training only Slater parameters ===")
    pre_trainer = VMCTrainer(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        graph=graph,
        ansatz_model=model,
        phase=None,  # 指定为第一阶段
        vmc_params={**cfg.get("vmc", {})},
        logger=wandb,
        ground_state=ground_state,
        variables=merged,
    )
    template = trainer.vstate.variables
    loaded_vars = from_bytes(template, param_bytes)
    pre_trainer.vstate.replace(variables=loaded_vars)
    pre_trainer.run(out=os.path.join(args.outdir, "phase1"),
                 logfile=args.logfile)

    # trainer.estimate()
    # trainer.dump_orbitals_dataset(out_path=os.path.join(args.outdir, f"orbitals_dataset_fno_{size}.npz"))
    print("Training completed, saving model parameters...")
    variables = unfreeze(trainer.vstate.variables)
    save_flax_params(variables, os.path.join(args.outdir, f"fno_slater_pretrain_{size}.flax"))

    # Save the model parameters and upload to wan
    wandb.finish()
    print("Training finished.")



if __name__ == '__main__':
    for size in [16, 32, 64]:
        print(f"Starting training for size {size}...")
        TFS(size=size)
        print(f"Finished training for size {size}.")
