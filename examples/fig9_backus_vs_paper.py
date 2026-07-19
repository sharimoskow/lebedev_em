"""
fig9_backus_vs_paper.py — the corrected anisotropic Backus on the REAL DDH03
Fig. 9 configuration (optimal grid + borehole + thin 75-deg TI layer),
overlaid on the values digitized from the published figure.

NOTE: the DDH03 curve is extracted from the printed figure, so it is an
approximate target, not exact ground truth.

Uses from_geometry_exact (exact normals + exact volume/line fractions) with
method= 'pointwise' / 'backus' / 'nodal', the fully coupled single solve, and
lgmres (the optimal grid is ~10^5 complex unknowns — too large for a direct
solve).

Usage:
  python fig9_backus_vs_paper.py probe      # time one coupled solve
  python fig9_backus_vs_paper.py run        # backus + pointwise, both models, figure
"""
import os, sys, time, warnings
import numpy as np
import scipy.sparse.linalg as spla
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid, C000, C101, C110, C011
from lebedev_em.media import from_geometry_exact
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    lebedev_B_on_z_axis)

FREQ = 52650.0; OMEGA = 2 * np.pi * FREQ
SIG_BG = 0.10; SIG_BORE = 0.05; R_BORE = 0.1; R_INV = 0.6
SIG_T = 0.10; SIG_N = SIG_T / 200.0
DIP = np.radians(75.0); N_HAT = np.array([np.sin(DIP), 0., np.cos(DIP)])
D_TOP = N_HAT[2] * (-0.95); D_BOT = N_HAT[2] * (-1.93)
SIG_ANISO = SIG_T * np.eye(3) + (SIG_N - SIG_T) * np.outer(N_HAT, N_HAT)
H_MIN = 0.05; K_VAL = 6; GAMMA = 1 / 2 ** 0.5
CLUSTERS = [C000, C101, C110, C011]

# DDH03 Fig. 9 values digitized from the figure (approximate).
PAPER = {'nolayer': {-2.17: 1.50, -1.93: 1.75, -1.68: 2.05, -1.43: 2.47, -1.17: 3.08,
                     -0.95: 3.95, -0.77: 4.90, -0.66: 5.68, -0.57: 6.55, -0.47: 7.95,
                     -0.38: 9.95, -0.30: 12.70, -0.25: 14.90},
         'layer':   {-2.17: 0.80, -1.93: 0.95, -1.68: 1.17, -1.43: 1.47, -1.17: 1.92,
                     -0.95: 2.63, -0.77: 3.52, -0.66: 4.25, -0.57: 5.10, -0.47: 6.42,
                     -0.38: 8.42, -0.30: 11.15, -0.25: 13.40}}


def make_model(with_layer):
    def sigma_func(X, Y, Z):
        X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        shape = np.broadcast(X, Y, Z).shape
        out = np.zeros(shape + (3, 3), dtype=complex)
        out[...] = SIG_BG * np.eye(3)
        if with_layer:
            side = N_HAT[0] * X + N_HAT[2] * Z
            out[(side < D_TOP) & (side > D_BOT)] = SIG_ANISO
        r = np.sqrt(X ** 2 + Y ** 2)
        out[r < R_BORE] = SIG_BORE * np.eye(3)
        return out
    bounds = [CylindricalBoundary(radius=R_BORE)]
    if with_layer:
        bounds += [PlanarBoundary(n_hat=N_HAT, d=D_TOP), PlanarBoundary(n_hat=N_HAT, d=D_BOT)]
    return sigma_func, GeometryStack(bounds)


def build_grid():
    z_fd = hybrid_axial_grid(-3.5, 2.5, 96, 8, GAMMA)
    return symmetric_optimal_grid(H_MIN, 300., z_fd, GAMMA, k=K_VAL)


def solve(grid, with_layer, method):
    t0 = time.time()
    sf, geo = make_model(with_layer)
    # Backus/pointwise via the clean exact core (eq. 9 everywhere, exact
    # fractions); nodal via the exact-normal path.
    if method in ("pointwise", "backus"):
        med = from_geometry_exact(grid, sf, geo, method=method, h_svd=0.03)
    else:
        med = from_geometry_exact(grid, sf, geo, method=method, h_svd=0.025)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    print(f"    [{method}/{'layer' if with_layer else 'nolayer'}] media+assembly {time.time()-t0:.0f}s", flush=True)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=2)   # z magnetic dipole
    b = sum(rhs[c] for c in CLUSTERS)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b.copy(), _component_aware_bc_dofs(grid))
    A_bc = A_bc.tocsr()
    d = A_bc.diagonal(); d_inv = np.where(np.abs(d) > 1e-30, 1.0 / d, 1.0)
    M = spla.LinearOperator(A_bc.shape, matvec=lambda x: d_inv * x, dtype=complex)
    t1 = time.time()
    E, info = spla.lgmres(A_bc, b_bc, M=M, rtol=1e-8, atol=0, maxiter=400, inner_m=30, outer_k=10)
    B = compute_B_from_E(grid, E, OMEGA)
    z, Bz = lebedev_B_on_z_axis(grid, {c: B for c in CLUSTERS}, comp=2)
    print(f"    [{method}/{'layer' if with_layer else 'nolayer'}] lgmres info={info} {time.time()-t1:.0f}s", flush=True)
    return np.asarray(z), np.asarray(Bz).imag * 1e9


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    grid = build_grid()
    print(f"grid N_R={grid.N_R} 3N_R={3*grid.N_R}", flush=True)
    if mode == "probe":
        solve(grid, False, "pointwise")
        sys.exit(0)

    data = {}
    for method in ("pointwise", "backus"):
        for wl in (False, True):
            z, Bz = solve(grid, wl, method)
            data[(method, "layer" if wl else "nolayer")] = Bz
    data["z"] = z
    np.savez(os.path.join(OUT, "fig9_backus_vs_paper.npz"), **{f"{k}": v for k, v in
             {f"{a}_{b}": data[(a, b)] for a in ("pointwise", "backus") for b in ("nolayer", "layer")}.items()}, z=z)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for a, tag, ttl in [(ax[0], "nolayer", "No layer (σ_N = σ_T)"),
                        (ax[1], "layer", "Resistive layer (σ_N = σ_T/200)")]:
        zp = np.array(list(PAPER[tag].keys())); vp = np.array(list(PAPER[tag].values()))
        a.plot(zp, vp, "ks", ms=6, label="DDH03 (digitized)")
        a.plot(z, data[("pointwise", tag)], "-", color="tab:blue", label="ours: pointwise")
        a.plot(z, data[("backus", tag)], "--", color="tab:red", label="ours: anisotropic Backus")
        a.set_xlim(-2.3, -0.2); a.set_ylim(0, 16)
        a.set_xlabel("z (m)"); a.set_title(ttl, fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[0].set_ylabel(r"Im $B_z$  (nT)")
    fig.suptitle("DDH03 Fig. 9 vs corrected anisotropic Backus (coupled solve, k=6 optimal grid)", fontsize=11)
    fig.tight_layout()
    png = os.path.join(OUT, "fig9_backus_vs_paper.png"); fig.savefig(png, dpi=130)
    print("figure ->", png)
    # print ratio table
    print("\n z     paper   pointwise  backus   (layer)")
    for zz in PAPER["layer"]:
        idx = int(np.argmin(np.abs(z - zz)))
        print(f"{zz:6.2f} {PAPER['layer'][zz]:7.2f}  {data[('pointwise','layer')][idx]:8.2f}  {data[('backus','layer')][idx]:7.2f}")
