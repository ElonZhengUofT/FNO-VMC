import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

L = 16
N = 16
TWOSZ = 0

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=4)
driver.initialize_system(n_sites=L, n_elec=N, spin=TWOSZ)

# 先创建表达式构造器
b = driver.expr_builder()

# 一些任意的 CD 项（示例）
b.add_term("CD", [[1, 3]], 0.7)
b.add_term("CD", [[3, 1]], 0.6)
b.add_term("CD", [[2, 2]], 0.5)
b.add_term("CD", [[2, 4]], 0.4)

t = 1.0
U = 4.0

# hopping: 分别 (i -> i+1) 和 (i+1 -> i)
hopping = [[i, (i + 1) % L] for i in range(L)]
hopping += [[j, i] for (i, j) in hopping]  # 反向
hopping = np.array(hopping)  # 形状 (2L, 2)
b.add_term("cd",      hopping, -t)
b.add_term("CD",      hopping, -t)

# onsite U term: cdCD 作用在同一个 i 上
onsite = np.array([[i, i, i, i] for i in range(L)])  # 形状 (L, 4)
b.add_term("cdCD", onsite, U)

# 构造 MPO
mpo = driver.get_mpo(b.finalize(), iprint=2)

def run_dmrg(driver, mpo):
    ket = driver.get_random_mps(tag="KET", bond_dim=250, nroots=1)
    bond_dims = [250] * 4 + [500] * 4
    noises    = [1e-4]  * 4 + [1e-5]   * 4 + [0]
    thrds     = [1e-10] * 8
    return driver.dmrg(
        mpo, ket,
        n_sweeps=20,
        bond_dims=bond_dims,
        noises=noises,
        thrds=thrds,
        cutoff=0,
        iprint=1
    )

energies = run_dmrg(driver, mpo)
print('DMRG energy = %20.15f' % energies)
