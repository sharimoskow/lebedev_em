"""
run_k6.py — Run k=6 lateral refinement and save results to k6_results.npz.

After this script finishes, re-run plot_ddh03_fig7.py to include k=6
in the figure (it loads k6_results.npz automatically).

Typical run time: 3–5 minutes on a modern laptop.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import scipy.sparse.linalg as spla

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import EMMedia, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, extract_B_on_axis

FREQ  = 52650.0
OMEGA = 2.0 * np.pi * FREQ
SIGMA_BORE = 0.05; SIGMA_INV = 0.10; SIGMA_ISO = 0.50
SIGMA_N = 0.01;  SIGMA_T = 0.10
R_BORE = 0.1;    R_INV  = 0.6
DIP_RAD = np.radians(60.0)
N_HAT = np.array([np.sin(DIP_RAD), 0.0, np.cos(DIP_RAD)])
Z_IFACE = -0.5
D_PLANE = N_HAT[0]*0.0 + N_HAT[2]*Z_IFACE
SIGMA_ANISO = SIGMA_T*np.eye(3) + (SIGMA_N-SIGMA_T)*np.outer(N_HAT, N_HAT)

z_fd_grid = hybrid_axial_grid(-3.5, 2.5, 98, 8, 1.0/np.sqrt(2))
H_MIN=0.5; L_TRANS=300.0; GAMMA=1.0/np.sqrt(2)
Z_MEAS_MIN, Z_MEAS_MAX = -1.7, -0.05

def build_media(grid):
    N_R = grid.N_R
    sigma_R = np.zeros((N_R,3,3), dtype=complex)
    for seq,(i,j,k) in enumerate(grid.R_nodes):
        x=float(grid.x[i]); y=float(grid.y[j]); z=float(grid.z[k])
        r_xy=np.sqrt(x**2+y**2)
        if r_xy<R_BORE:   sigma_R[seq]=SIGMA_BORE*np.eye(3)
        elif r_xy<R_INV:  sigma_R[seq]=SIGMA_INV*np.eye(3)
        else:
            side=N_HAT[0]*x+N_HAT[2]*z
            if side<D_PLANE: sigma_R[seq]=SIGMA_ANISO
            else:            sigma_R[seq]=SIGMA_ISO*np.eye(3)
    mu_P  = np.full(grid.N_P, complex(MU0))
    eps_R = np.full(grid.N_R, complex(EPS0))
    return EMMedia(grid, sigma_R, mu_P, eps_R)

def build_rhs(grid, solver_obj, omega):
    Mx,My,Mz=grid.Mx,grid.My,grid.Mz
    i0=Mx//2; j0=My//2
    k_even=np.arange(0,Mz+1,2)
    k0=int(k_even[np.argmin(np.abs(grid.z[k_even]))])
    P_seq=int(grid.P_idx[i0,j0,k0])
    dx=float(grid.x[min(i0+1,Mx)]-grid.x[max(i0-1,0)])
    dy=float(grid.y[min(j0+1,My)]-grid.y[max(j0-1,0)])
    dz=float(grid.z[min(k0+1,Mz)]-grid.z[max(k0-1,0)])
    vol_P=abs(dx*dy*dz)
    print(f"    Hx P-node: ({i0},{j0},{k0}), pos=({grid.x[i0]:.4f},{grid.y[j0]:.4f},{grid.z[k0]:.4f}), vol_P={vol_P:.4e}", flush=True)
    M_P_vec=np.zeros(3*grid.N_P, dtype=complex)
    M_P_vec[0*grid.N_P+P_seq]=1.0/vol_P
    return 1j*omega*(solver_obj._C_PR @ M_P_vec)

print("=== k=6 run ===", flush=True)
t0=time.time()
k=6
grid=symmetric_optimal_grid(H_MIN, L_TRANS, z_fd_grid, GAMMA, k=k)
print(f"Grid: Mx={grid.Mx}, My={grid.My}, N_R={grid.N_R}, N_P={grid.N_P}  ({time.time()-t0:.1f}s)", flush=True)
med=build_media(grid)
print(f"Media done  ({time.time()-t0:.1f}s)", flush=True)
solver=LebedevMaxwellSolver(grid, med, OMEGA)
b_mag=build_rhs(grid, solver, OMEGA)
bc_dofs=_component_aware_bc_dofs(grid)
A_bc,b_bc=apply_electric_bc(solver._A.copy(), b_mag, bc_dofs)
print(f"System assembled, shape={A_bc.shape}, nnz={A_bc.nnz}  ({time.time()-t0:.1f}s)", flush=True)
E=spla.spsolve(A_bc, b_bc)
print(f"Solve done  ({time.time()-t0:.1f}s)", flush=True)
B_vec=compute_B_from_E(grid, E, OMEGA)
z_bx,Bx=extract_B_on_axis(grid, B_vec, comp=0, axis='z')
z_bz,Bz=extract_B_on_axis(grid, B_vec, comp=2, axis='z')
mask_x=(z_bx>=Z_MEAS_MIN)&(z_bx<=Z_MEAS_MAX)
mask_z=(z_bz>=Z_MEAS_MIN)&(z_bz<=Z_MEAS_MAX)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k6_results.npz")
np.savez(out_path,
    z_bx=z_bx[mask_x], Bxx=np.imag(Bx[mask_x]),
    z_bz=z_bz[mask_z], Bxz=np.imag(Bz[mask_z]))
print(f"Saved to {out_path}", flush=True)
print(f"Im(Bxx) range: {np.imag(Bx[mask_x]).min()*1e9:.2f} to {np.imag(Bx[mask_x]).max()*1e9:.2f} nT", flush=True)
print(f"Total time: {time.time()-t0:.1f}s", flush=True)
