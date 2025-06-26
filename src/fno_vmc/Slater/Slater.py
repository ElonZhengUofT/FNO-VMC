import netket.experimental as nkx

N_sites = 8  # Example size, can be adjusted
#             n_fermions_per_spin=(N_up, N_down)
N_up = 4  # Number of up fermions
N_down = 4  # Number of down fermions

hilbert = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=N_sites,
    s=0.5,
    n_fermions_per_spin=(N_up, N_down)
)

slater = nkx.models.Slater2nd(hilbert=hilbert)