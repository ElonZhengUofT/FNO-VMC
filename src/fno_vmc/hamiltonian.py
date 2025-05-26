import netket as nk


def make_hamiltonian(ham_type: str, params: dict):
    """
    Build Hilbert space and Hamiltonian operator.
    ham_type: 'ising' | 'heisenberg' | 'xxz'
    params: dict with keys 'L', 'dim', and model-specific params
    """
    L    = params.get("L", 16)
    dim  = params.get("dim", 1)
    pbc  = params.get("pbc", True)

    graph = nk.graph.Hypercube(length=L, n_dim=dim, pbc=pbc)
    N_sites = graph.n_nodes
    hilbert = nk.hilbert.Spin(s=0.5, N=N_sites)

    t = ham_type.lower()
    if t == "ising":
        J = params.get("J", 1.0)
        h = params.get("h", 1.0)
        op = nk.operator.Ising(hilbert, h=h, J=J, graph=graph)
    elif t == "heisenberg":
        J = params.get("J", 1.0)
        op = nk.operator.Heisenberg(hilbert, J=J, graph=graph)
    elif t == "xxz":
        Jx = params.get("Jx", 1.0)
        Jz = params.get("Jz", 1.0)
        op = nk.operator.XXZ(hilbert, Jx=Jx, Jz=Jz, graph=graph)
    else:
        raise ValueError(f"Unknown Hamiltonian type: {ham_type}")

    return hilbert, op