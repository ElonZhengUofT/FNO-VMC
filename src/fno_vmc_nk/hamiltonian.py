import netket as nk
# from netket.experimental.operator import FermiHubbardJax
# import netket.experimental.operator.FermiHubbardJax as FermiHubbardJax
from netket.operator.fermion import destroy, create, number

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
    L    = params.get("L", 8)
    dim  = params.get("dim", 1)
    pbc  = params.get("pbc", True)

    graph = nk.graph.Hypercube(length=L, n_dim=dim, pbc=pbc)
    N_sites = graph.n_nodes

    t = ham_type.lower()
    # print(f"This model is a {t} model with {N_sites} sites, with L={L} and dim={dim}")
    if t in ["ising", "heisenberg", "xxz", "j1j2", "j1j2_heisenberg"]:
        hilbert = nk.hilbert.Spin(s=0.5, N=N_sites)
    elif t in ["hubbard", "fermihubbard"]:
        n_particles = params.get("n_particles", None)
        N_up, N_down = n_particles
        if n_particles is None:
            raise ValueError("n_particles must be specified for Hubbard model.")
        hilbert = nk.hilbert.SpinOrbitalFermions(
            n_orbitals=N_sites,
            s=0.5,
            n_fermions_per_spin=(N_up, N_down)
        )

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
    elif t == "hubbard":
        U = params.get("U", 4.0)
        t_hop = params.get("t", 1.0)
        op = 0.0

        def c(site, sz):
            return nk.operator.fermion.destroy(hilbert, site, sz=sz)

        def cdag(site, sz):
            return nk.operator.fermion.create(hilbert, site, sz=sz)

        def nc(site, sz):
            return nk.operator.fermion.number(hilbert, site, sz=sz)

        for sz in (0, 1):
            for u, v in graph.edges():
                op -= t_hop * (cdag(u, sz) * c(v, sz) + cdag(v, sz) * c(u, sz))
        for u in graph.nodes():
            op += U * nc(u, 0) * nc(u, 1)
    else:
        raise ValueError(f"Unknown Hamiltonian type: {ham_type}")

    return hilbert, op

if __name__ == "__main__":
    # Example usage
    hilbert, hamiltonian = make_hamiltonian(
        ham_type='heisenberg',
        params={
            'L': 4,
            'dim': 2,
            'J': 1.0,
            'pbc': True
        }
    )
    # print("Hilbert space:", hilbert)
    # print("Hamiltonian operator:", hamiltonian)

    matrix = hamiltonian.to_sparse()
    # print("Hamiltonian matrix shape:", matrix.shape)
    # print("Matrix data:\n", matrix.todense())

    # build a fermionic Hubbard model
    hilbert, hamiltonian = make_hamiltonian(
        ham_type='hubbard',
        params={
            'L': 4,
            'dim': 2,
            'n_particles': (2, 2),  # 2 up and 2 down
            'U': 4.0,
            't': 1.0,
            'pbc': True
        }
    )
    print("Fermionic Hubbard model Hilbert space:", hilbert)