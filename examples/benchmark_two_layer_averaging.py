"""
benchmark_two_layer_averaging.py — the two-half-space Sommerfeld benchmark,
but now with the *averaging* methods and the *coupled* solve.

The original `benchmark_two_layer.py` uses `layered_isotropic` (exact
piecewise-constant sigma, no sub-cell averaging) with the contact landing on a
grid node, so it never exercises the `backus` / `nodal` homogenization at all.
Here we instead:

  * place the contact MID-CELL, so a dual cell genuinely straddles the
    interface and the averaging is actually invoked;
  * build the medium three ways with `from_geometry_exact`
    (method = pointwise / backus / nodal) via a `PlanarBoundary`;
  * use the fully coupled single solve (solver.solve default = 'coupled');
  * compare each to the exact two-layer Sommerfeld analytic.

For an axis-aligned (normal-z) interface the averaged tensor is diagonal
(transverse arithmetic, normal harmonic) with NO off-diagonal coupling, so this
is a clean baseline: if backus/nodal reproduce the analytic here, the averaging
+ coupled-solve machinery is validated, and the Fig-9 discrepancy is specific to
the *tilt* (off-diagonal coupling). Backus/nodal should also REDUCE the
interface error relative to pointwise, because the harmonic normal average is
the physically correct effective conductivity for E_z continuity.

Usage:
    python examples/benchmark_two_layer_averaging.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from lebedev_em.grid import (symmetric_optimal_grid, hybrid_axial_grid,
                             C000, C101, C110, C011)
from lebedev_em.media import from_geometry_exact, MU0, EPS0
from lebedev_em.geometry import PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.analytics import (electric_dipole_Ez_two_layer_onaxis,
                                  electric_dipole_Ez_homogeneous_onaxis)
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

SIGMA1 = 0.1
SIGMA2 = 1.0
Z_SRC = 0.0
FREQ = 2500.0
OMEGA = 2.0 * np.pi * FREQ
MU, EPS = MU0, EPS0
GAMMA = 1.0 / np.sqrt(2.0)

Z_INNER_MIN, Z_INNER_MAX, N_INNER = -0.25, 7.75, 128
K_OUTER = 8
H_MIN, L_TRANS = 0.5, 300.0
K_GRID = 3
Z_RECV_MIN, Z_RECV_MAX = 0.5, 7.5
METHODS = ("pointwise", "backus", "nodal")


def build_grid():
    z_fine = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
    return symmetric_optimal_grid(H_MIN, L_TRANS, z_fine, GAMMA, k=K_GRID)


def make_sigma_func(z_cont):
    def sigma_func(X, Y, Z):
        X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        shape = np.broadcast(X, Y, Z).shape
        out = np.zeros(shape + (3, 3), dtype=complex)
        out[...] = SIGMA1 * np.eye(3)
        out[Z >= z_cont] = SIGMA2 * np.eye(3)
        return out
    return sigma_func


def onaxis_ez_nodes(grid):
    """Return (z_eval, seq_c000) for C000-native on-axis E_z (VED) nodes."""
    Mx2, My2 = grid.Mx // 2, grid.My // 2
    nat = _native_type_for_cluster_comp(C000, 2)   # (0,0,1)
    zlist, seqlist = [], []
    for seq, (i, j, k) in enumerate(grid.R_nodes):
        if i == Mx2 and j == My2 and (i % 2, j % 2, k % 2) == nat:
            zv = float(grid.z[k])
            if Z_RECV_MIN <= zv <= Z_RECV_MAX:
                zlist.append(zv); seqlist.append(seq)
    z = np.array(zlist); seq = np.array(seqlist, dtype=int)
    o = np.argsort(z)
    return z[o], seq[o]


def lebedev_Ez(grid, result, z_eval, seq_c000):
    N_R = grid.N_R
    Ez_c000 = np.array([result["E_c"][C000][2 * N_R + s] for s in seq_c000])
    stacks = [Ez_c000]
    for c in (C101, C110, C011):
        stacks.append(np.array([
            interpolate_cluster_E(grid, result["E_c"][c], c, 2, 0.0, 0.0, zv)
            for zv in z_eval]))
    return np.mean(np.stack(stacks, axis=0), axis=0)


def rms_rel(fd, an):
    fd_re, an_re = np.real(fd), np.real(an)
    m = np.abs(an_re) > 0.01 * np.max(np.abs(an_re))
    return float(np.sqrt(np.mean(((fd_re[m] - an_re[m]) / an_re[m]) ** 2)))


def main():
    grid = build_grid()
    z_eval, seq_c000 = onaxis_ez_nodes(grid)

    # Contact mid-cell: between the node nearest z=4 and its upper neighbour.
    k0 = int(np.argmin(np.abs(grid.z - 4.0)))
    Z_CONT = 0.5 * (grid.z[k0] + grid.z[k0 + 1])
    print(f"grid Mx={grid.Mx} Mz={grid.Mz} N_R={grid.N_R} 3N_R={3*grid.N_R}")
    print(f"contact placed mid-cell at z_c = {Z_CONT:.5f} "
          f"(between z[{k0}]={grid.z[k0]:.4f} and z[{k0+1}]={grid.z[k0+1]:.4f})")
    print(f"on-axis E_z receiver nodes: {len(z_eval)} in [{Z_RECV_MIN},{Z_RECV_MAX}]")

    geo = GeometryStack([PlanarBoundary(n_hat=[0.0, 0.0, 1.0], d=Z_CONT)])
    sf = make_sigma_func(Z_CONT)

    # Exact two-layer Sommerfeld analytic (single VED at z=0).
    print("computing Sommerfeld analytic ...", flush=True)
    Ez_an = np.array([electric_dipole_Ez_two_layer_onaxis(
        zv, Z_SRC, Z_CONT, SIGMA1, SIGMA2, OMEGA, MU, EPS) for zv in z_eval])

    # Physical zones.  E_z is DISCONTINUOUS across the contact (normal-current
    # continuity: sigma1 E_z1 = sigma2 E_z2), so the one node whose dual cell
    # straddles the contact sits on a ~10x jump / zero-crossing and is not a
    # meaningful point comparison for ANY method — report it separately.
    dz_cell = float(grid.z[k0 + 1] - grid.z[k0])
    near_src = z_eval < 1.0
    iface = np.abs(z_eval - Z_CONT) < 1.5 * dz_cell
    bulk_L1 = (~near_src) & (~iface) & (z_eval < Z_CONT)
    bulk_L2 = (~iface) & (z_eval >= Z_CONT)

    results = {}
    print(f"\n  {'method':9s}  {'src(z<1)':>9} {'bulkL1':>8} {'iface':>8} "
          f"{'bulkL2':>8}   (RMS rel err %)")
    for method in METHODS:
        t0 = time.time()
        med = from_geometry_exact(grid, sf, geo, method=method, h_svd=0.02)
        solver = LebedevMaxwellSolver(grid, med, omega=OMEGA)
        res = solver.solve(0.0, 0.0, Z_SRC, dipole_comp=2, moment=1.0)  # coupled
        Ez_fd = lebedev_Ez(grid, res, z_eval, seq_c000)
        results[method] = Ez_fd
        e = {z: rms_rel(Ez_fd[m], Ez_an[m]) for z, m in
             (("s", near_src), ("l1", bulk_L1), ("if", iface), ("l2", bulk_L2))}
        print(f"  {method:9s}  {e['s']*100:8.3f}  {e['l1']*100:7.3f}  "
              f"{e['if']*100:7.3f}  {e['l2']*100:7.3f}   ({time.time()-t0:.0f}s)",
              flush=True)
    print("  bulk L1/L2 exclude the near-source zone and the single interface-"
          "straddling node.")

    # Per-node table near the interface
    print(f"\n  {'z':>7} {'analytic':>12}", end="")
    for m in METHODS:
        print(f" {m[:8]:>10}", end="")
    print()
    for i, zv in enumerate(z_eval):
        print(f"  {zv:>7.3f} {np.real(Ez_an[i]):>12.4e}", end="")
        for m in METHODS:
            r = np.real(results[m][i]) / np.real(Ez_an[i])
            print(f" {r:>10.4f}", end="")
        print()
    print("  (method columns are FD/analytic ratios)")

    np.savez(os.path.join(os.path.dirname(__file__), "out",
                          "two_layer_averaging.npz"),
             z=z_eval, analytic=Ez_an, **{m: results[m] for m in METHODS})

    # ---- plot ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"pointwise": "tab:blue", "backus": "tab:red", "nodal": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].semilogy(z_eval, np.abs(Ez_an), "k-", lw=1.6, label="Sommerfeld analytic")
    for m in METHODS:
        ax[0].semilogy(z_eval, np.abs(results[m]), "o", ms=3.5,
                       color=colors[m], alpha=0.8, label=f"coupled: {m}")
    ax[0].axvline(Z_CONT, color="gray", ls=":", label=f"contact z={Z_CONT:.3f}")
    ax[0].set_xlabel("z (m)"); ax[0].set_ylabel(r"$|E_z|$ (V/m)")
    ax[0].set_title("Two-layer VED: coupled solve vs Sommerfeld"); ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    for m in METHODS:
        dev = (np.real(results[m]) - np.real(Ez_an)) / np.abs(Ez_an) * 100
        ax[1].plot(z_eval, dev, "o-", ms=3.5, color=colors[m], alpha=0.8, label=m)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(Z_CONT, color="gray", ls=":")
    ax[1].set_ylim(-8, 8)
    ax[1].set_xlabel("z (m)"); ax[1].set_ylabel("% deviation from analytic")
    ax[1].set_title("Deviation (bulk); interface node off-scale"); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle("Axis-aligned two-half-space: averaging methods + coupled solve",
                 fontsize=11)
    fig.tight_layout()
    png = os.path.join(os.path.dirname(__file__), "out", "two_layer_averaging.png")
    fig.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
    main()
