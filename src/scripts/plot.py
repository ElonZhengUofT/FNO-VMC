import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot energy and variance from VMC results")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory where training outputs are stored")
    parser.add_argument("--ansatz", choices=["fno", "tn"], required=True)
    args = parser.parse_args()

    # assume energy & variance saved as NumPy arrays
    base = os.path.join(args.outdir, args.ansatz)
    energy_file = os.path.join(base, 'energy.npy')
    var_file    = os.path.join(base, 'variance.npy')

    energy   = np.load(energy_file)
    variance = np.load(var_file)
    steps     = np.arange(1, len(energy) + 1)

    # plot
    plt.figure()
    plt.plot(steps, energy, label='Energy')
    plt.plot(steps, variance, label='Variance')
    plt.xlabel('Iteration')
    plt.ylabel('Expectation')
    plt.title(f'VMC Training ({args.ansatz.upper()})')
    plt.legend()
    plt.grid(True)

    # save figure
    fig_path = os.path.join(base, 'vmc_training.png')
    plt.savefig(fig_path, dpi=300)
    print(f"Plot saved to {fig_path}")

if __name__ == '__main__':
    main()
