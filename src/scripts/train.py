# scripts/train.py
import argparse
import logging
import os
import torch.nn as nn
from src.fno_vmc.util import load_config, set_logger
from src.fno_vmc.hamiltonian import make_hamiltonian
from src.fno_vmc.Ansatz import make_ansatz
from src.fno_vmc.vmc import VMCTrainer


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
    args = parser.parse_args()

    # setup output directory
    os.makedirs(args.outdir, exist_ok=True)

    # configure logging
    set_logger(logfile=args.logfile)
    logging.info(f"Starting training: ansatz={args.ansatz}, config={args.config}")

    # load configuration
    cfg = load_config(args.config)

    # build Hamiltonian
    hilbert, hamiltonian = make_hamiltonian(
        ham_type=cfg["hamiltonian"]["type"],
        params=cfg["hamiltonian"]["params"]
    )

    # 在 train.py 中 make_ansatz 之前
    # logging.info(f"Creating ansatz with params: {cfg.get('model_params')}")

    # build ansatz
    model = make_ansatz(
        kind=args.ansatz,
        dim=cfg["hamiltonian"]["params"]["dim"],
        **cfg.get("model_params", {})
    )

    # Cold start: log ansatz type and parameters
    for p in model.parameters():
        p.data.zero_()

    # train
    trainer = VMCTrainer(
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        ansatz_model=model,
        vmc_params=cfg.get("vmc", {})
    )
    trainer.run(out=os.path.join(args.outdir, args.ansatz), logfile=args.logfile)


if __name__ == '__main__':
    main()