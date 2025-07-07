# FNO-VMC: Fourier Neural Operators in Variational Monte Carlo

## Overview
FNO-VMC is a research implementation that integrates Fourier Neural Operators (FNO) as variational ansatz within the Variational Monte Carlo (VMC) framework. This approach leverages the spectral capabilities of FNO to capture long-range correlations in strongly correlated quantum systems, offering improved energy convergence and generalization compared to traditional ansatz.

## Motivation
Variational Monte Carlo is a powerful method for estimating ground-state properties of quantum many-body systems, but conventional ansatz such as Restricted Boltzmann Machines (RBMs) and simple neural networks can struggle with long-range dependencies. FNOs, originally developed for solving partial differential equations, provide a natural mechanism to encode non-local interactions via global Fourier integrals, making them ideal candidates for representing complex quantum wavefunctions.

## Key Features
- **FNO Ansätze**: Two complementary strategies
  1. **Direct Amplitude Mapping**: Map spin configurations to wavefunction amplitudes \(\psi(s)\) using FNO.
  2. **Tensor Network Generation**: Use FNO to predict tensor network parameters for ansatz like MPS/PEPS.
- **Stochastic Reconfiguration (SR)**: Employ quantum natural gradient optimization to update variational parameters via the Fisher information matrix.
- **Modular Design**: Built on PyTorch and NetKet for GPU acceleration and easy experimentation.

## Repository Structure
```
FNO-VMC/
├── data/                # Sample configurations and ground-truth data
├── experiments/         # Scripts to run experiments on different models
├── fno-vmc              # FNO and auxiliary neural network definitions
├── scripts/             # Training, evaluation, and plotting utilities
├── requirements.txt     # Python package dependencies
└── README.md            # Project overview and instructions
```

## Installation


## Usage

### Training


### Evaluation


### Custom Experiments
Refer to `experiments/` for examples on running J1-J2 or custom Hamiltonians.

## Model Details
- **FNO Layer**: Implements spectral convolution with learnable global weights.
- **Positional Encoding**: Embeds grid coordinates for lattice symmetries.
- **Fully-Connected Readout**: Projects FNO outputs to log-amplitude and phase of \(\psi\).

## Optimization
- **Stochastic Reconfiguration**: Solve the linear system
  \[
    S_{\alpha\beta} \delta \theta_{\beta} = - \nabla_{\alpha} E(\theta)
  \]
  where \(S\) is the quantum Fisher information matrix.
- **Batching**: Monte Carlo samples are generated on-the-fly with NetKet samplers.
- **split_batches**: Set `vmc.split_batches` in a config file to split each training
  iteration into multiple sub-batches. Gradients from the sub-batches are averaged
  before applying an optimizer step, reducing memory usage. The default value is `1`.

## Experiments & Results
We validate FNO-VMC on benchmark models:
- **1D Ising**: Achieves energy error < 0.1% relative to exact diagonalization.
- **2D Heisenberg**: Captures ground-state properties with fewer parameters.
- **J1-J2 Model**: Demonstrates robust generalization across frustration regimes.

<!Plots and detailed metrics are available in `results/` after running `scripts/plot_results.py`.>

## Citation
If you use FNO-VMC in your research, please cite:
```
@article{YourName2025FNOVMC,
  title={Fourier Neural Operators as Variational Ansatz in Monte Carlo Simulations},
  author={Zheng, Shizhao and Collaborators},
  journal={Preprint},
  year={2025},
  arxiv={2405.00001}
}
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, new features, or discussion topics.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or collaborations, reach out to **Shizhao Zheng** at shizhao.zheng@mail.utoronto.ca

