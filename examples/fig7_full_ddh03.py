"""
fig7_full_ddh03.py — Fig. 7 model with the COMPLETE DDH03 methodology:
  fine transverse grid (h_min=0.05 resolves the borehole, as in paper Fig. 6),
  k=6 optimal transverse grid (domain ±7.8 m),
  z-grid with source exactly at an even-index node (no snap),
  sub-cell homogenization via from_geometry_func (paper's eq. 8-9 averaging),
  4-cluster sources (eq. 7) + per-cluster mixed BCs + interpolate-then-average.

Usage: python fig7_full_ddh03.py <cluster_indices e.g. 0 1>   # solve stage
       python fig7_full_ddh03.py extract                       # final stage
"""
import sys, os, time
import numpy as np
import scipy.sparse.linalg as spla
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import os as _os
OUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'out')
_os.makedirs(OUT_DIR, exist_ok=True)


from lebedev_em.grid import (symmetric_optimal_grid, hybrid_axial_grid,
                             C000, C101, C110, C011)
from lebedev_em.media import from_geometry_func, MU0, EPS0
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver, _cluster_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    lebedev_B_on_z_axis)

FREQ = 52650.; OMEGA = 2*np.pi*FREQ
SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50; SIGMA_N=0.01; SIGMA_T=0.10
R_BORE=0.1; R_INV=0.6; DIP_RAD=np.radians(60.)
N_HAT = np.array([np.sin(DIP_RAD), 0., np.cos(DIP_RAD)])
D_PLANE = N_HAT[2]*(-0.5)
SIGMA_ANISO = SIGMA_T*np.eye(3) + (SIGMA_N-SIGMA_T)*np.outer(N_HAT, N_HAT)
H_MIN = 0.05; K_VAL = 6; GAMMA = 1/2**0.5
CLUSTERS = [C000, C101, C110, C011]
OUT = OUT_DIR + '/fig7_full_B_{c}.npz'

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

geo = GeometryStack([
    CylindricalBoundary(radius=R_BORE),
    CylindricalBoundary(radius=R_INV),
    PlanarBoundary(n_hat=N_HAT, d=D_PLANE),
])

def build():
    t0=time.time()
    # inner zone [-3.5, 2.5], 96 intervals -> z=0 at node 16+56=72 (even): no snap
    z_fd = hybrid_axial_grid(-3.5, 2.5, 96, 8, GAMMA)
    grid = symmetric_optimal_grid(H_MIN, 300., z_fd, GAMMA, k=K_VAL)
    print(f'Grid: Mx={grid.Mx} Mz={grid.Mz} N_R={grid.N_R} x_max={grid.x[-1]:.2f} '
          f't={time.time()-t0:.0f}s', flush=True)
    med = from_geometry_func(grid, sigma_func, geo.interface_func, h_svd=0.025)
    print(f'Homogenized media built t={time.time()-t0:.0f}s', flush=True)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    print(f'System assembled t={time.time()-t0:.0f}s', flush=True)
    return grid, solver

if sys.argv[1] == 'extract':
    grid, solver = build()
    Bc = {c: np.load(OUT.format(c=c))['B'] for c in CLUSTERS}
    z_bx, Bxx = lebedev_B_on_z_axis(grid, Bc, comp=0)
    z_bz, Bxz = lebedev_B_on_z_axis(grid, Bc, comp=2)
    z_bx=np.asarray(z_bx); Bxx=np.asarray(Bxx); Bxz=np.asarray(Bxz)
    np.savez(OUT_DIR + '/fig7_full_ddh03_k6.npz', z=z_bx, Bxx=Bxx, Bxz=Bxz)
    paper_xx = {-1.663:3.11,-1.541:3.66,-1.051:7.17,-0.929:8.46,-0.806:10.11,
                -0.684:12.21,-0.561:14.84,-0.194:32.47,-0.071:60.82}
    paper_xz = {-1.663:3.89,-1.541:4.20,-1.051:5.43,-0.929:5.67,-0.806:5.80,
                -0.684:5.97,-0.561:5.83,-0.439:5.58,-0.316:5.31,-0.194:4.70,-0.071:4.07}
    print("  z      ImBxx(nT)  paper  ratio |  ImBxz(nT)  paper  ratio")
    for zz in sorted(set(list(paper_xx)+list(paper_xz))):
        i = int(np.argmin(np.abs(z_bx - zz)))
        bx, bz = Bxx[i].imag*1e9, Bxz[i].imag*1e9
        px, pz = paper_xx.get(zz), paper_xz.get(zz)
        sx = f"{px:6.2f} {px/bx:6.2f}" if px else "   --     --"
        sz = f"{pz:6.2f} {pz/bz:6.2f}" if pz else "   --     --"
        print(f"{zz:7.3f}  {bx:8.3f}  {sx} | {bz:8.3f}  {sz}")
else:
    idxs = [int(a) for a in sys.argv[1:]]
    grid, solver = build()
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=0)
    t0=time.time()
    for ix in idxs:
        c = CLUSTERS[ix]
        bc = _cluster_bc_dofs(grid, c)
        A_bc, b_bc = apply_electric_bc(solver._A.copy(), rhs[c].copy(), bc)
        A_bc = A_bc.tocsr()
        d = A_bc.diagonal()
        d_inv = np.where(np.abs(d) > 1e-30, 1.0/d, 1.0)
        M = spla.LinearOperator(A_bc.shape, matvec=lambda x: d_inv*x, dtype=complex)
        E, info = spla.lgmres(A_bc, b_bc, M=M, rtol=1e-8, atol=0,
                              maxiter=400, inner_m=30, outer_k=10)
        print(f'cluster {c} LGMRES info={info} t={time.time()-t0:.0f}s', flush=True)
        B = compute_B_from_E(grid, E, OMEGA)
        np.savez(OUT.format(c=c), B=B)
        print(f'cluster {c} saved t={time.time()-t0:.0f}s', flush=True)
