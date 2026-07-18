"""
fig9_check.py — Reproduce DDH03 Fig. 9 (thin dipping resistive layer) as an
absolute-calibration test: the paper's "no layer" curve is verified against the
unit-moment analytic (ratio 0.92-0.99), so matching/missing the paper here
cleanly separates "our heterogeneous machinery" from "Fig-7-specific issues".

Model (DDH03 Fig. 8): background sigma=0.1; borehole sigma=0.05 R=0.1 m;
invasion sigma=0.1 R=0.6 m (same as background); thin 75-degree dipping layer,
0.25 m normal thickness, sigma_T=0.1, sigma_N=sigma_T/200, crossing the axis
over z in [-1.93, -0.95]. Source: z-directed magnetic dipole at origin,
52.65 kHz. Plot: Im Bz on the z-axis.

Usage: python fig9_check.py <nolayer|layer> <cluster idx...>
       python fig9_check.py extract
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
SIG_BG = 0.10; SIG_BORE = 0.05; R_BORE = 0.1
SIG_T = 0.10; SIG_N = SIG_T/200.0
DIP = np.radians(75.)
N_HAT = np.array([np.sin(DIP), 0., np.cos(DIP)])
# layer crosses the axis over z in [-1.93, -0.95] -> plane offsets:
D_TOP = N_HAT[2]*(-0.95)     # upper face
D_BOT = N_HAT[2]*(-1.93)     # lower face  (normal thickness ~0.127? see note)
# normal distance between faces: |D_TOP - D_BOT| = cos75 * 0.98 = 0.2536 -> 0.25 m thick, consistent
SIG_ANISO = SIG_T*np.eye(3) + (SIG_N-SIG_T)*np.outer(N_HAT, N_HAT)
H_MIN = 0.05; K_VAL = 6; GAMMA = 1/2**0.5
CLUSTERS = [C000, C101, C110, C011]

def make_model(with_layer):
    def sigma_func(X, Y, Z):
        X=np.asarray(X,float); Y=np.asarray(Y,float); Z=np.asarray(Z,float)
        shape=np.broadcast(X,Y,Z).shape; out=np.zeros(shape+(3,3),dtype=complex)
        out[...] = SIG_BG*np.eye(3)
        if with_layer:
            side = N_HAT[0]*X + N_HAT[2]*Z
            inlayer = (side < D_TOP) & (side > D_BOT)
            out[inlayer] = SIG_ANISO
        r=np.sqrt(X**2+Y**2)
        out[r<R_BORE] = SIG_BORE*np.eye(3)   # borehole column wins (crosses the layer)
        return out
    bounds = [CylindricalBoundary(radius=R_BORE)]
    if with_layer:
        bounds += [PlanarBoundary(n_hat=N_HAT, d=D_TOP), PlanarBoundary(n_hat=N_HAT, d=D_BOT)]
    return sigma_func, GeometryStack(bounds)

def build(tag):
    t0=time.time()
    z_fd = hybrid_axial_grid(-3.5, 2.5, 96, 8, GAMMA)
    grid = symmetric_optimal_grid(H_MIN, 300., z_fd, GAMMA, k=K_VAL)
    sf, geo = make_model(tag == 'layer')
    med = from_geometry_func(grid, sf, geo.interface_func, h_svd=0.025)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    print(f'{tag}: grid+media+assembly t={time.time()-t0:.0f}s', flush=True)
    return grid, solver

if sys.argv[1] == 'extract':
    z_fd = hybrid_axial_grid(-3.5, 2.5, 96, 8, GAMMA)
    grid = symmetric_optimal_grid(H_MIN, 300., z_fd, GAMMA, k=K_VAL)
    paper = {'nolayer': {-2.17:1.50,-1.93:1.75,-1.68:2.05,-1.43:2.47,-1.17:3.08,-0.95:3.95,
                         -0.77:4.90,-0.66:5.68,-0.57:6.55,-0.47:7.95,-0.38:9.95,-0.30:12.70,-0.25:14.90},
             'layer':   {-2.17:0.80,-1.93:0.95,-1.68:1.17,-1.43:1.47,-1.17:1.92,-0.95:2.63,
                         -0.77:3.52,-0.66:4.25,-0.57:5.10,-0.47:6.42,-0.38:8.42,-0.30:11.15,-0.25:13.40}}
    for tag in ('nolayer','layer'):
        Bc = {c: np.load(OUT_DIR + f'/fig9_B_{tag}_{c}.npz')['B'] for c in CLUSTERS}
        z_ax, Bz = lebedev_B_on_z_axis(grid, Bc, comp=2)
        z_ax=np.asarray(z_ax); Bz=np.asarray(Bz)
        np.savez(OUT_DIR + f'/fig9_{tag}_result.npz', z=z_ax, Bz=Bz)
        print(f"\n[{tag}]   z     FD ImBz(nT)   paper   paper/FD")
        for zz, p in paper[tag].items():
            i = int(np.argmin(np.abs(z_ax - zz)))
            b = Bz[i].imag*1e9
            print(f"  {zz:6.2f}  {b:9.3f}  {p:7.2f}  {p/b:7.2f}")
else:
    tag = sys.argv[1]; idxs = [int(a) for a in sys.argv[2:]]
    grid, solver = build(tag)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=2)   # z-dipole
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
        B = compute_B_from_E(grid, E, OMEGA)
        np.savez(OUT_DIR + f'/fig9_B_{tag}_{c}.npz', B=B)
        print(f'{tag} cluster {c} info={info} t={time.time()-t0:.0f}s', flush=True)
