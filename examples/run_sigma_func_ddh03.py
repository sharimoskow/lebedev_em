"""
run_sigma_func_ddh03.py  —  Validate from_sigma_func (tensor-valued callable)
against the exact-geometry strategy-E result for DDH03 Fig. 7.

The tensor sigma_func returns the full 3×3 conductivity tensor at any (x,y,z)
by evaluating the DDH03 piecewise geometry as a black box — no knowledge of
interface normals, borehole structure, or geometry is passed to from_sigma_func.

Expected:  crossing z_cross ≈ −1.29 m (matching hmin010_k4_E.npz)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import scipy.sparse.linalg as spla

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import from_sigma_func, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E



# ── Physical parameters (identical to exact-geometry runs) ────────────────────
FREQ   = 52650.0
OMEGA  = 2.0 * np.pi * FREQ

SIGMA_BORE = 0.05
SIGMA_INV  = 0.10
SIGMA_ISO  = 0.50
SIGMA_N    = 0.01
SIGMA_T    = 0.10
R_BORE     = 0.1
R_INV      = 0.6

DIP_DEG  = 60.0
DIP_RAD  = np.radians(DIP_DEG)
N_HAT    = np.array([np.sin(DIP_RAD), 0.0, np.cos(DIP_RAD)])
Z_IFACE  = -0.5
D_PLANE  = N_HAT[0]*0.0 + N_HAT[2]*Z_IFACE   # = −0.25

SIGMA_ANISO = SIGMA_T*np.eye(3) + (SIGMA_N - SIGMA_T)*np.outer(N_HAT, N_HAT)

# Grid parameters (k=4, H=0.10 m — matches hmin010_k4_E.npz)
K_VAL   = 4
H_MIN   = 0.10
L_TRANS = 300.0
GAMMA   = 1.0 / np.sqrt(2.0)

Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER = -3.5, 2.5, 98, 8
Z_MEAS_MIN, Z_MEAS_MAX = -1.7, -0.05


# ── Tensor-valued sigma_func ───────────────────────────────────────────────────
def sigma_func_ddh03(X, Y, Z):
    """
    Returns the 3×3 conductivity tensor at every point (X, Y, Z).
    X, Y, Z are broadcastable arrays; output shape is (..., 3, 3).
    No geometry metadata (normals, radii) is used by the caller.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    shape = np.broadcast(X, Y, Z).shape

    out = np.zeros(shape + (3, 3), dtype=complex)
    r_xy = np.sqrt(X**2 + Y**2)

    # Borehole
    m = r_xy < R_BORE
    out[m] = SIGMA_BORE * np.eye(3)

    # Invasion zone
    m = (r_xy >= R_BORE) & (r_xy < R_INV)
    out[m] = SIGMA_INV * np.eye(3)

    # Formation — anisotropic half-space
    side = N_HAT[0]*X + N_HAT[2]*Z
    m = (r_xy >= R_INV) & (side < D_PLANE)
    out[m] = SIGMA_ANISO

    # Formation — isotropic half-space
    m = (r_xy >= R_INV) & (side >= D_PLANE)
    out[m] = SIGMA_ISO * np.eye(3)

    return out


# ── Build grid ────────────────────────────────────────────────────────────────
print(f"Building grid k={K_VAL}, H={H_MIN} m ...", flush=True)
t0 = time.time()
z_fd = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
grid = symmetric_optimal_grid(H_MIN, L_TRANS, z_fd, GAMMA, k=K_VAL)
print(f"  grid: Mx={grid.Mx}, N_R={grid.N_R}  ({time.time()-t0:.1f}s)", flush=True)

# ── Build media via tensor sigma_func ─────────────────────────────────────────
print("Building media (from_sigma_func, method=nodal) ...", flush=True)
t1 = time.time()
med = from_sigma_func(grid, sigma_func_ddh03, h_svd=0.025, method="nodal",
                      svd_isotropy_tol=0.7)
print(f"  media: {time.time()-t1:.1f}s", flush=True)
print(f"  sigma_R shape: {med.sigma_R.shape}", flush=True)

# ── Solve ─────────────────────────────────────────────────────────────────────
from lebedev_em.postprocess import build_rhs_multicl, extract_B_on_axis_multicl

print("Solving ...", flush=True)
t2 = time.time()
solver = LebedevMaxwellSolver(grid, med, OMEGA)
b = build_rhs_multicl(grid, solver._C_PR, OMEGA)

bc_dofs = _component_aware_bc_dofs(grid)
A_bc, b_bc = apply_electric_bc(solver._A.copy(), b, bc_dofs)
E = spla.spsolve(A_bc, b_bc)
print(f"  solve: {time.time()-t2:.1f}s", flush=True)

# ── Extract B-field ───────────────────────────────────────────────────────────
B_vec = compute_B_from_E(grid, E, OMEGA)

z_bx, Bx = extract_B_on_axis_multicl(grid, B_vec, comp=0)
z_bz, Bz = extract_B_on_axis_multicl(grid, B_vec, comp=2)

# ── Filter and find crossing ──────────────────────────────────────────────────
mx = (z_bx >= Z_MEAS_MIN) & (z_bx <= Z_MEAS_MAX)
mz = (z_bz >= Z_MEAS_MIN) & (z_bz <= Z_MEAS_MAX)
z_w  = z_bx[mx]
bxx  = np.imag(Bx[mx]) * 1e9    # nT
bxz  = np.interp(z_w, z_bz[mz], np.imag(Bz[mz])) * 1e9

diff = bxx - bxz
sc = np.where(np.diff(np.sign(diff)))[0]
if len(sc):
    zi = sc[0]
    z_cross = z_w[zi] - diff[zi]/(diff[zi+1]-diff[zi])*(z_w[zi+1]-z_w[zi])
    print(f"\nCrossing Im(Bxx)=Im(Bxz) at z = {z_cross:.4f} m")
else:
    z_cross = None
    print("\nNo crossing found in window.")

# Save result
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hmin010_k4_sigma_func.npz")
np.savez(out_path, z_x=z_w, bxx=bxx/1e9, bxz=bxz/1e9,
         z_cross=np.array([z_cross if z_cross else np.nan]))
print(f"Saved: {out_path}")

# ── Compare with exact-geometry result ───────────────────────────────────────
ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hmin010_k4_E.npz")
if os.path.exists(ref_path):
    ref = np.load(ref_path)
    ref_bxx = ref['bxx']
    ref_bxz = ref['bxz']
    ref_z   = ref['z_x']
    # Find ref crossing
    ref_diff = ref_bxx - ref_bxz
    sc_r = np.where(np.diff(np.sign(ref_diff)))[0]
    if len(sc_r):
        zi = sc_r[0]
        z_cross_ref = ref_z[zi] - ref_diff[zi]/(ref_diff[zi+1]-ref_diff[zi])*(ref_z[zi+1]-ref_z[zi])
        print(f"Exact-geometry (strat E) crossing:  z = {z_cross_ref:.4f} m")
        if z_cross is not None:
            print(f"Difference: {abs(z_cross - z_cross_ref)*100:.0f} mm")
    # Max relative error in Im(Bxx)
    bxx_ref_interp = np.interp(z_w, ref_z, ref_bxx)
    rel_err = np.max(np.abs(bxx - bxx_ref_interp*1e9) / (np.abs(bxx_ref_interp*1e9) + 1e-20))
    print(f"Max relative error Im(Bxx): {rel_err*100:.2f}%")
