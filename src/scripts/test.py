import argparse
import logging
import os
import torch
from src.fno_vmc_jax.util import load_config, set_logger
from src.fno_vmc_jax.hamiltonian import make_hamiltonian
from src.fno_vmc_jax.Ansatz import make_ansatz
from src.fno_vmc_jax.evaluate import evaluate_energy, evaluate_variance


def main():
    parser = argparse.ArgumentParser(description="Test trained VMC model")
    parser.add_argument("--ansatz", choices=["fno", "tn"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved model checkpoint (.pt) or NetKet state file.")
    parser.add_argument("--n_samples", type=int, default=2000)
    args = parser.parse_args()

    set_logger()
    logging.info(f"Testing checkpoint: {args.checkpoint}")

    # load config
    cfg = load_config(args.config)

    # build Hamiltonian
    hilbert, hamiltonian = make_hamiltonian(
        ham_type=cfg["hamiltonian"]["type"],
        params=cfg["hamiltonian"]["params"]
    )

    # build ansatz
    model = make_ansatz(
        kind=args.ansatz,
        dim=cfg["hamiltonian"]["params"]["dim"],
        **cfg.get("model_params", {})
    )

    # load checkpoint
    if args.checkpoint.endswith('.pt'):
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)
    else:
        # assume NetKet format: we rely on MCState.load
        pass

    # evaluate
    E = evaluate_energy(model, hamiltonian, hilbert, n_samples=args.n_samples)
    V = evaluate_variance(model, hamiltonian, hilbert, n_samples=args.n_samples)

    logging.info(f"Energy     = {E:.6f}")
    logging.info(f"Variance   = {V:.6f}")


if __name__ == '__main__':
    main()