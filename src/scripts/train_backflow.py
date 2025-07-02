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
from src.fno_vmc_nk.ansatz.BackflowFNO import NNBackflowSlater2nd
from src.fno_vmc_nk.vmc_jax import VMCTrainer
import jax.numpy as jnp
import jax
import time
import optax
import flax
from flax.core.frozen_dict import freeze, unfreeze

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

    # build Hamiltonian
    hilbert, hamiltonian,graph = make_hamiltonian(
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

    # train the model
    # model = nk.models.RBM() # A Test model, replace when debug is done
    hi = hilbert
    model = NNBackflowSlater2nd(
        hilbert=hi, generalized=False, restricted=True, hidden_units=128
    )
    rng = jax.random.PRNGKey(0)
    dummy_n = jnp.zeros((hi.size,), dtype=jnp.int32)
    variables = model.init(rng, dummy_n)

    # variables 是一个 FrozenDict，结构大概是 {"params": { … }}
    params = variables["params"]

    # 递归打印各层级的 key 路径
    def print_tree(d, prefix=()):
        if isinstance(d, dict):
            for k, v in d.items():
                print(" / ".join(prefix + (k,)))
                print_tree(v, prefix + (k,))
        else:
            # 叶子节点，打印形状
            try:
                print(
                    f"{' / '.join(prefix)} : shape={v.shape}, dtype={v.dtype}"
                )
            except:
                pass

    print_tree(params)



if __name__ == '__main__':
    main()