import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

def build_hubbard_mpo(driver, L, t, U):
    """
    在 driver 上构造 1D Hubbard 链的 MPO（PBC），每个格点一个轨道、双自旋，半填充 N=L。
    """
    # 1) 初始化系统：L 格点，L 个电子（半填充），累计自旋投影 TWOSZ=0
    driver.initialize_system(n_sites=L, n_elec=L, spin=0)

    # 2) 构造哈密顿量表达式
    b = driver.expr_builder()

    #   a) 最近邻 hopping（PBC）
    for i in range(L):
        j = (i + 1) % L   # PBC：L-1 -> 0
        # c†_i c_j + h.c.
        b.add_term("cd",   [i, j], -t)
        b.add_term("cd",   [j, i], -t)
        # C†_i C_j + h.c. （Spin 自旋分离写法）
        b.add_term("CD",   [i, j], -t)
        b.add_term("CD",   [j, i], -t)

    #   b) onsite U 作用：c†_i c_i C†_i C_i
    for i in range(L):
        b.add_term("cdCD", [i, i, i, i], U)

    # 3) 完成表达式并生成 MPO
    return driver.get_mpo(b.finalize(), iprint=0)


def run_dmrg(driver, mpo, max_bond=250):
    """
    对给定的 MPO 运行一次 DMRG，返回基态能量。
    你可以根据 L 的大小适当调整 bond_dims、noises、n_sweeps 等参数以保证收敛。
    """
    ket = driver.get_random_mps(tag="KET", bond_dim=max_bond, nroots=1)

    # 示例：前几次 sweep 保持低 bond，再逐步加大
    bond_dims = [max_bond] * 4 + [max_bond * 2] * 4
    noises    = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds     = [1e-10] * 8

    energies = driver.dmrg(
        mpo, ket,
        n_sweeps=len(bond_dims),
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        cutoff=0,
        iprint=0
    )
    return energies


if __name__ == "__main__":
    t = 1.0
    U = 4.0
    sizes = [32, 64]

    results = {}
    for L in sizes:
        print(f"Running DMRG for L={L} …")
        # 每次都新建一个 driver，避免状态残留
        driver = DMRGDriver(scratch=f"./tmp/L{L}", symm_type=SymmetryTypes.SZ, n_threads=4)
        mpo    = build_hubbard_mpo(driver, L, t, U)

        # 对 L 越大，通常需要更大的 max_bond，更多 sweep，或者 OBC 以提高效率
        max_bond = 1000 if L <= 64 else 2000
        energy = run_dmrg(driver, mpo, max_bond=max_bond)
        results[L] = energy
        print(f" → L={L}, E0 = {energy:.15f}\n")

    print("All results:")
    for L, E in results.items():
        print(f"  L={L:3d}  E0={E:.15f}")
