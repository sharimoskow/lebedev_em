"""
compute_reference.py — Fine reference solution, k=6 nodal, DZ=0.125m z-grid.
Run once; saves tilted_reference.npz for use by tilted_convergence.py.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import planar_interface_isotropic
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.postprocess import lebedev_E_at_point

SIGMA1, SIGMA2 = 0.1, 1.0
OMEGA = 2*np.pi*100.0
N_HAT = np.array([1.,0.,1.])/np.sqrt(2.)
D_PLANE = 1.0
Z_IFACE = D_PLANE*np.sqrt(2.)
X_SRC, Y_SRC, Z_SRC = 0., 0., -2.
GAMMA = 1./np.sqrt(2.)
TARGET_XMAX = 20.0
K_REF = 6

DZ = 0.125
n_inner = int(round(6.0 / DZ))       # 6m inner domain, 48 cells
z_grid  = hybrid_axial_grid(-2.5, 3.5, n_inner, 6, GAMMA)
if (len(z_grid)-1) % 2 != 0:
    z_grid = np.append(z_grid, 2*z_grid[-1] - z_grid[-2])

alpha = np.exp(GAMMA*np.pi/np.sqrt(K_REF))
h_min = TARGET_XMAX / sum(alpha**i for i in range(K_REF))
grid  = symmetric_optimal_grid(h_min, TARGET_XMAX*1.5, z_grid, GAMMA, k=K_REF)
print(f"k={K_REF}, Mx=My={grid.Mx}, Mz={grid.Mz}, N_R={grid.N_R}, h_min={h_min:.3f}m")

media  = planar_interface_isotropic(grid, N_HAT, D_PLANE, SIGMA1, SIGMA2)
t0     = time.time()
result = LebedevMaxwellSolver(grid, media, omega=OMEGA).solve(
             X_SRC, Y_SRC, Z_SRC, dipole_comp=2, moment=1.0)
print(f"Solve: {time.time()-t0:.1f}s")

# Evaluate on z-axis, well clear of the source (z=-2 → evaluate from z=-0.5)
z_eval = np.linspace(-0.5, 3.0, 150)
E_c    = result["E_c"]
Ez = np.array([lebedev_E_at_point(grid, E_c, 2, 0.,0.,z) for z in z_eval])
Ex = np.array([lebedev_E_at_point(grid, E_c, 0, 0.,0.,z) for z in z_eval])

outfile = os.path.join(os.path.dirname(__file__), "tilted_reference.npz")
np.savez(outfile, z_eval=z_eval, Ez=Ez, Ex=Ex,
         k_ref=K_REF, h_min=h_min, N_R=grid.N_R, Mx=grid.Mx, DZ=DZ)
print(f"Saved → {outfile}")
