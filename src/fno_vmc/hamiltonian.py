import netket as nk

def generate_j2_edges(L, pbc=True):
    """
    Generates edges for a square lattice with periodic or open boundary conditions.
    L: Number of lattice points along each edge.
    pbc: If True, applies periodic boundary conditions.
    Returns a list of tuples representing edges (i, j) where i < j.
    """
    edges = []
    for y in range(L):
        for x in range(L):
            i = x + L * y  # Current site index

            x1 = (x + 1) % L if pbc else (x + 1)
            y1 = (y + 1) % L if pbc else (y + 1)
            if 0 <= x1 < L and 0 <= y1 < L:
                j = x1 + L * y1
                if i < j:
                    edges.append((i, j))

            x2 = (x + 1) % L if pbc else (x + 1)
            y2 = (y - 1) % L if pbc else (y - 1)
            if 0 <= x2 < L and 0 <= y2 < L:
                j = x2 + L * y2
                if i < j:
                    edges.append((i, j))

    return edges


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

    # spin Hamiltonians
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
    elif t == "j1j2":
        J1 = params.get("J1", 1.0)
        J2 = params.get("J2", 0.0)
        edges = generate_j2_edges(L, pbc=pbc)
        op = nk.operator.J1J2(hilbert, J1=J1, J2=J2, edges=edges)
    elif t == "j1j2_heisenberg":
        J1 = params.get("J1", 1.0)
        J2 = params.get("J2", 0.0)
        edges = generate_j2_edges(L, pbc=pbc)
        op = nk.operator.J1J2Heisenberg(hilbert, J1=J1, J2=J2, edges=edges)
    else:
        raise ValueError(f"Unknown Hamiltonian type: {ham_type}")

    return hilbert, op