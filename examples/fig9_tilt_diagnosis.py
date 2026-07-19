"""
fig9_tilt_diagnosis.py — Why the homogenization scheme changes the Fig-9 answer.

A *cheap, self-contained* reproduction (small uniform grid, direct solve) of the
DDH03 Fig-9 phenomenology that lives in the July-2026 handoff's Open Problem #1.
It does NOT reproduce the paper's absolute values (that needs the full optimal
grid + borehole + invasion); instead it isolates the MECHANISM by comparing the
three cell-homogenization schemes to each other on the SAME coarse geometry, with
an axis-aligned control that removes the tilt.

Model: isotropic background sigma=0.1; a thin (0.25 m normal) transversely
isotropic layer, sigma_T=0.1, sigma_N=sigma_T/200, dipping at `DIP` degrees,
crossing the borehole (z) axis near z=-1.2 m. Source: z-directed MAGNETIC dipole
at the origin, 52.65 kHz (built via C_PR, the coil RHS — an *electric* z-dipole
gives Bz=0 on axis by symmetry). Coupled single solve (correct for anisotropic
media). Quantity: Im Bz on the z-axis.

Findings (see docs/fig9_diagnosis_2026-07-19.md):
  * dip=0 control: pointwise ~= backus ~= nodal  (all schemes correct when the
    interface aligns with the grid; nodal -> Lemma 1 diag tensor).
  * dip=75 tilted: backus UNDER-attenuates (Bz too big), nodal OVER-attenuates
    (Bz too small), pointwise sits between. The split into diagonal vs
    off-diagonal tensor error is grid/path dependent.

Run:  python examples/fig9_tilt_diagnosis.py
"""
import os, sys, time, warnings
import numpy as np
import scipy.sparse.linalg as spla
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

from lebedev_em.grid import symmetric_uniform_grid, C000, C101, C110, C011
from lebedev_em.media import from_sigma_func, EMMedia
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    lebedev_B_on_z_axis)

FREQ = 52650.0; OMEGA = 2 * np.pi * FREQ
SIG_BG = 0.10; SIG_T = 0.10; SIG_N = SIG_T / 200.0
CLUSTERS = [C000, C101, C110, C011]
Mx = My = 16; Mz = 40; Lx = Ly = 5.0; Lz = 6.0
Z_CENTER = -1.2; THICK = 0.25


def make_sf(dip_deg, with_layer):
    dip = np.radians(dip_deg); NH = np.array([np.sin(dip), 0.0, np.cos(dip)])
    SA = SIG_T * np.eye(3) + (SIG_N - SIG_T) * np.outer(NH, NH); d0 = NH[2] * Z_CENTER

    def sf(X, Y, Z):
        X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        shp = np.broadcast(X, Y, Z).shape; out = np.zeros(shp + (3, 3), complex)
        out[...] = SIG_BG * np.eye(3)
        if with_layer:
            side = NH[0] * X + NH[2] * Z
            out[np.abs(side - d0) < THICK / 2] = SA
        return out
    return sf


def solve_media(grid, med):
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=2)  # z magnetic dipole
    b = sum(rhs[c] for c in CLUSTERS)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b.copy(), _component_aware_bc_dofs(grid))
    E = spla.spsolve(A_bc.tocsc(), b_bc)
    B = compute_B_from_E(grid, E, OMEGA)
    z, Bz = lebedev_B_on_z_axis(grid, {c: B for c in CLUSTERS}, comp=2)
    return np.asarray(z), np.asarray(Bz).imag * 1e9


def diag_only(grid, med):
    s = np.array(med.sigma_R)
    if s.ndim == 3:
        for a in range(3):
            for b in range(3):
                if a != b:
                    s[:, a, b] = 0.0
    return EMMedia(grid, s, np.array(med.mu_P), np.array(med.eps_R))


def build(grid, method, dip, with_layer=True):
    return from_sigma_func(grid, make_sf(dip, with_layer), h_svd=0.03,
                           n_line=30, n_vol=6, method=method)


def gm(a, b, mask):
    r = np.array(a)[mask] / np.array(b)[mask]; r = r[np.isfinite(r) & (r > 0)]
    return float(np.exp(np.mean(np.log(r))))


def main():
    grid = symmetric_uniform_grid(Mx, My, Mz, Lx, Ly, Lz)
    print(f"grid N_R={grid.N_R} 3N_R={3*grid.N_R} hx={Lx/Mx:.3f} hz={Lz/Mz:.3f}")
    t0 = time.time()
    z, Bnl = solve_media(grid, build(grid, "pointwise", 0, with_layer=False))
    data = {"z": z, "nolayer": Bnl}
    for dip, tag in [(0, "ctrl0"), (75, "tilt75")]:
        for m in ("pointwise", "backus", "nodal"):
            _, B = solve_media(grid, build(grid, m, dip)); data[f"{tag}_{m}"] = B
        _, Bd = solve_media(grid, diag_only(grid, build(grid, "nodal", dip)))
        data[f"{tag}_nodal_diag"] = Bd
    np.savez(os.path.join(OUT, "fig9_tilt_diagnosis.npz"), **data)
    print(f"all solves t={time.time()-t0:.0f}s")

    mask = (z >= -2.4) & (z <= -0.5)
    for tag in ("ctrl0", "tilt75"):
        pw = data[f"{tag}_pointwise"]
        print(f"\n[{tag}] geo-mean vs pointwise:  "
              f"backus={gm(data[f'{tag}_backus'], pw, mask):.3f}  "
              f"nodal={gm(data[f'{tag}_nodal'], pw, mask):.3f}  "
              f"nodal_diag={gm(data[f'{tag}_nodal_diag'], pw, mask):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        for a, tag, ttl in [(ax[0], "ctrl0", "Control: dip = 0 (grid-aligned)"),
                            (ax[1], "tilt75", "Tilted: dip = 75 (Fig-9 regime)")]:
            a.plot(z, data["nolayer"], "k:", lw=1.4, label="no layer")
            a.plot(z, data[f"{tag}_pointwise"], "o-", ms=3, label="pointwise (ref)")
            a.plot(z, data[f"{tag}_backus"], "s--", ms=3, label="Backus")
            a.plot(z, data[f"{tag}_nodal"], "^--", ms=3, label="nodal")
            a.set_xlim(-2.6, -0.4); a.set_ylim(0, 6)
            a.set_xlabel("z (m)"); a.set_title(ttl, fontsize=10)
            a.grid(alpha=0.3); a.legend(fontsize=8)
        ax[0].set_ylabel(r"Im $B_z$  (nT)")
        fig.suptitle("Thin 75 deg resistive layer: cell-homogenization scheme sets the attenuation",
                     fontsize=11)
        fig.tight_layout()
        png = os.path.join(OUT, "fig9_tilt_diagnosis.png")
        fig.savefig(png, dpi=130); print(f"figure -> {png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
