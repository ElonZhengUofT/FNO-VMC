import netket as nk


def evaluate_energy(model, hamiltonian, hilbert, n_samples: int = 1000):
    """Return expectation value of energy."""
    sampler = nk.sampler.MetropolisLocal(hilbert)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)
    return vstate.expect(hamiltonian)


def evaluate_variance(model, hamiltonian, hilbert, n_samples: int = 1000):
    """Return variance of energy."""
    sampler = nk.sampler.MetropolisLocal(hilbert)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)
    return vstate.variance(hamiltonian)