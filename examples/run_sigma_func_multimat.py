"""
run_sigma_func_multimat.py  —  Validate from_sigma_func with multi-material
nodal homogenization for the DDH03 geometry (k=4, H=0.10m).

Compares crossing z against hmin010_k4_E.npz (strategy-E exact geometry).
Uses LGMRES + diagonal preconditioner for robustness with anisotropic tensors.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import scipy.sparse.linalg as spla
import warnings; warnings.filterwarnings('ignore')

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import from_sigma_func, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_multicl,
                                    extract_B_on_axis_multicl)

FREQ = 52650.; OMEGA = 2*np.pi*FREQ
SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50; SIGMA_N=0.01; SIGMA_T=0.10
R_BORE=0.1; R_INV=0.6; DIP_RAD=np.radians(60.)
N_HAT = np.array([np.sin(DIP_RAD), 0., np.cos(DIP_RAD)])
D_PLANE = N_HAT[2]*(-0.5)
SIGMA_ANISO = SIGMA_T*np.eye(3) + (SIGMA_N-SIGMA_T)*np.outer(N_HAT, N_HAT)
H_MIN = 0.10; K_VAL = 4; GAMMA = 1/2**0.5
Z_MIN, Z_MAX = -1.7, -0.05

def sigma_func(X, Y, Z):
    X=np.asarray(X,dtype=float); Y=np.asarray(Y,dtype=float); Z=np.asarray(Z,dtype=float)
    shape=np.broadcast(X,Y,Z).shape; out=np.zeros(shape+(3,3),dtype=complex)
    r=np.sqrt(X**2+Y**2)
    out[r<R_BORE] = SIGMA_BORE*np.eye(3)
    out[(r>=R_BORE)&(r<R_INV)] = SIGMA_INV*np.eye(3)
    side=N_HAT[0]*X+N_HAT[2]*Z
    out[(r>=R_INV)&(side<D_PLANE)] = SIGMA_ANISO
    out[(r>=R_INV)&(side>=D_PLANE)] = SIGMA_ISO*np.eye(3)
    return out

t0 = time.time()
z_fd = hybrid_axial_grid(-3.5, 2.5, 98, 8, GAMMA)
grid = symmetric_optimal_grid(H_MIN, 300., z_fd, GAMMA, k=K_VAL)
print(f'Grid N_R={grid.N_R}  t={time.time()-t0:.1f}s', flush=True)

med = from_sigma_func(grid, sigma_func, h_svd=0.025, method='nodal', svd_isotropy_tol=0.7)
print(f'Media built  t={time.time()-t0:.1f}s', flush=True)

solver = LebedevMaxwellSolver(grid, med, OMEGA)
b = build_rhs_multicl(grid, solver._C_PR, OMEGA)
bc = _component_aware_bc_dofs(grid)
A_bc, b_bc = apply_electric_bc(solver._A.copy(), b, bc)
print(f'System {A_bc.shape} nnz={A_bc.nnz}  t={time.time()-t0:.1f}s', flush=True)

# LGMRES with diagonal preconditioner
d = A_bc.diagonal()
d_inv = np.where(np.abs(d) > 1e-30, 1.0/d, 1.0)
M = spla.LinearOperator(A_bc.shape, matvec=lambda x: d_inv*x, dtype=complex)

iters = [0]
def cb(r):
    iters[0] += 1
    if iters[0] % 10 == 0:
        print(f'  iter {iters[0]}  res={np.linalg.norm(r):.3e}  t={time.time()-t0:.1f}s', flush=True)

print('Solving LGMRES...', flush=True)
E, info = spla.lgmres(A_bc, b_bc, M=M, rtol=1e-8, atol=0, maxiter=300,
                      inner_m=30, outer_k=10, callback=cb)
print(f'LGMRES info={info}  iters={iters[0]}  t={time.time()-t0:.1f}s', flush=True)

B = compute_B_from_E(grid, E, OMEGA)
z_bx, Bx = extract_B_on_axis_multicl(grid, B, comp=0)
z_bz, Bz = extract_B_on_axis_multicl(grid, B, comp=2)
mx = (z_bx>=Z_MIN)&(z_bx<=Z_MAX); mz = (z_bz>=Z_MIN)&(z_bz<=Z_MAX)
z_w = z_bx[mx]
bxx = np.imag(Bx[mx])*1e9
bxz = np.interp(z_w, z_bz[mz], np.imag(Bz[mz]))*1e9
diff = bxx-bxz; sc = np.where(np.diff(np.sign(diff)))[0]
if len(sc):
    zi=sc[0]; zc=z_w[zi]-diff[zi]/(diff[zi+1]-diff[zi])*(z_w[zi+1]-z_w[zi])
    print(f'\nCROSSING (lookup+multimat): z={zc:.4f} m')
else:
    zc = None; print('\nNo crossing')

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hmin010_k4_sigma_multimat.npz'),
         z_x=z_w, bxx=bxx, bxz=bxz, z_cross=np.array([zc if zc else np.nan]))

ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hmin010_k4_E.npz')
if os.path.exists(ref_path):
    ref = np.load(ref_path)
    ref_diff = ref['bxx']-ref['bxz']; sc_r = np.where(np.diff(np.sign(ref_diff)))[0]
    if len(sc_r):
        zi=sc_r[0]; zc_r=ref['z_x'][zi]-ref_diff[zi]/(ref_diff[zi+1]-ref_diff[zi])*(ref['z_x'][zi+1]-ref['z_x'][zi])
        print(f'Reference (strategy-E):     z={zc_r:.4f} m')
        if zc: print(f'Difference: {abs(zc-zc_r)*1000:.0f} mm')

print(f'Total t={time.time()-t0:.1f}s', flush=True)
