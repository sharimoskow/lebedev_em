"""
tilted_interface_convergence.py — a SINGLE 75-deg interface between two
isotropic half-spaces, refined until the answer stops moving.

No analytic solution is needed: refinement is the ground truth.  A tilted
interface makes the straddle-cell averaged tensor OFF-DIAGONAL (harmonic in the
tilted normal, arithmetic tangential -> rotated back to the grid frame), so this
is the first test where `backus` and `nodal` differ from each other and from
`pointwise`.  The question:

    as h -> 0, do pointwise / backus / nodal converge to the SAME field?

If yes, the oblique averaging is a consistent effective medium and any Fig-9
gap is under-resolution.  If backus/nodal converge to a different (attenuated)
limit than fine-grid pointwise, the oblique laminate cell is a biased effective
medium — the Fig-9 over-attenuation, reproduced with a known ground truth.

Resumable: each run does ONE method over a list of grid sizes and appends to
examples/out/tilted_iface_conv.npz.  Plot with `plot`.

Usage:
  python examples/tilted_interface_convergence.py pointwise 24 32 40
  python examples/tilted_interface_convergence.py backus    24 32 40
  python examples/tilted_interface_convergence.py nodal     24 32 40
  python examples/tilted_interface_convergence.py plot
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
SIG1 = 0.10                       # background (source side)
SIG2 = 0.10 / 200.0              # resistive side (200x contrast)
THETA = np.radians(75.0)
NH = np.array([np.sin(THETA), 0.0, np.cos(THETA)])
Z_CROSS = 1.0                     # interface crosses the z-axis here
D_PLANE = NH[2] * Z_CROSS
CLUSTERS = [C000, C101, C110, C011]
L_DOM = 3.0
RECV = np.linspace(0.5, 2.0, 13)  # on-axis receivers, straddling the interface
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
NPZ = os.path.join(OUT, "tilted_iface_conv.npz")


def sigma_func(X, Y, Z):
    X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
    shape = np.broadcast(X, Y, Z).shape
    out = np.zeros(shape + (3, 3), dtype=complex)
    side = NH[0] * X + NH[1] * Y + NH[2] * Z
    out[...] = SIG1 * np.eye(3)
    out[side >= D_PLANE] = SIG2 * np.eye(3)
    return out


def solve_one(M, method):
    grid = symmetric_uniform_grid(Mx=M, My=M, Mz=M, Lx=L_DOM, Ly=L_DOM, Lz=L_DOM)
    geo = GeometryStack([PlanarBoundary(n_hat=NH, d=D_PLANE)])
    t0 = time.time()
    med = from_geometry_exact(grid, sigma_func, geo, method=method, h_svd=0.02)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=2)  # z m-dipole
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
    # interpolate Im B_z (nT) onto the fixed receiver set
    order = np.argsort(z)
    val = np.interp(RECV, z[order], Bz[order].imag * 1e9)
    h = float(grid.x[1] - grid.x[0])
    print(f"  [{method}/M={M}] h={h:.4f} 3N_R={3*grid.N_R} info={info} "
          f"{time.time()-t0:.0f}s", flush=True)
    return h, val


def load_store():
    if os.path.exists(NPZ):
        d = dict(np.load(NPZ, allow_pickle=True))
        return d
    return {}


def save_store(store):
    os.makedirs(OUT, exist_ok=True)
    np.savez(NPZ, **store)


def run(method, Ms):
    store = load_store()
    store["recv"] = RECV
    for M in Ms:
        h, val = solve_one(M, method)
        store[f"{method}_M{M}"] = val
        store[f"{method}_M{M}_h"] = np.array([h])
        save_store(store)   # save after each (resumable)
    print(f"saved -> {NPZ}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    store = load_store()
    recv = store["recv"]
    methods = ("pointwise", "backus", "nodal")
    col = {"pointwise": "tab:blue", "backus": "tab:red", "nodal": "tab:green"}
    # collect available (method, M)
    avail = {m: sorted(int(k.split("_M")[1]) for k in store
                       if k.startswith(m + "_M") and not k.endswith("_h"))
             for m in methods}
    print("available:", {m: avail[m] for m in methods})

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    # finest grid common to all three methods -> the convergence "truth"
    Mtruth = min(avail[m][-1] for m in methods if avail[m])
    truth = np.mean([store[f"{m}_M{Mtruth}"] for m in methods], axis=0)
    peak = np.abs(truth).max()
    # error measured AWAY from the source (near-source receivers are erratic and
    # dominated by the source approximation); keep the interface + past-interface
    # band, magnitude-weighted so the tiny far-field tail adds no noise.
    Z_ERR_MIN = 0.9
    msk = (np.abs(truth) > 0.05 * peak) & (recv >= Z_ERR_MIN)

    # Panel A: Im B_z profiles at the truth grid per method
    for m in methods:
        if not avail[m]:
            continue
        ax[0].plot(recv, store[f"{m}_M{Mtruth}"], "o-", color=col[m], ms=4,
                   label=f"{m} (M={Mtruth})")
    ax[0].axvline(Z_CROSS, color="gray", ls=":", label="interface on axis")
    ax[0].set_xlabel("z (m)"); ax[0].set_ylabel(r"Im $B_z$ (nT)")
    ax[0].set_title(f"Profiles at the truth grid (M={Mtruth})")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    # Panel B: convergence of each method to the M=Mtruth reference.
    # error(h) = relative L2 over significant receivers vs the truth field.
    def relL2(a):
        return np.sqrt(np.sum(np.abs(a[msk] - truth[msk])**2) /
                       np.sum(np.abs(truth[msk])**2)) * 100
    for m in methods:
        if not avail[m]:
            continue
        Ms = [M for M in avail[m] if M < Mtruth]
        hs = [float(store[f"{m}_M{M}_h"][0]) for M in Ms]
        er = [relL2(store[f"{m}_M{M}"]) for M in Ms]
        ax[1].loglog(hs, er, "o-", color=col[m], label=m)
    # first-order reference slope
    hr = np.array([0.13, 0.06])
    ax[1].loglog(hr, 60 * hr, "k:", lw=1, label="O(h) ref")
    ax[1].set_xlabel("grid spacing h (m)")
    ax[1].set_ylabel(f"rel. L2 error vs M={Mtruth} truth (%)")
    ax[1].set_title(f"Convergence to truth, z ≥ {Z_ERR_MIN} m (away from source)")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8)

    fig.suptitle(f"Tilted 75° isotropic interface (σ1={SIG1}, σ2=σ1/200), "
                 "coupled solve — refinement as ground truth", fontsize=11)
    fig.tight_layout()
    png = os.path.join(OUT, "tilted_iface_conv.png")
    fig.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    if sys.argv[1] == "plot":
        plot()
    else:
        method = sys.argv[1]
        Ms = [int(x) for x in sys.argv[2:]] or [24, 32, 40]
        run(method, Ms)
