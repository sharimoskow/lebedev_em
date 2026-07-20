"""
fig9_refinement.py — refinement study of the ACTUAL DDH03 Fig. 9
configuration (borehole + tilted 75-deg TI layer, optimal transverse grid,
hybrid axial grid).

Motivation (see tilted_slab_convergence.py, LAYER=aniso): in the controlled
tilted-slab test with the exact Fig-9 layer material, refinement showed that
backus/nodal are accurate at every resolution while POINTWISE badly
under-attenuates on under-resolved grids and converges slowly from above.
That inverts the earlier reading of the Fig-9 comparison: pointwise's match
with the curve digitized from the printed figure is the suspect datum.

This script asks the same question on the real Fig-9 geometry: as the grid
refines (transverse h_min down, k up to keep the domain reach; axial inner
spacing down proportionally), does the pointwise layer curve drift DOWN toward
the backus/nodal curves and away from the digitized paper values, while
backus/nodal barely move?

Refinement levels (transverse h_min, optimal-grid k, axial inner steps over
[-3.5, 2.5]):

  L1: h=0.0500  k=6  n_inner=96    (the grid used for the published comparison)
  L2: h=0.0333  k=7  n_inner=144
  L3: h=0.0250  k=8  n_inner=192
  L4: h=0.0200  k=9  n_inner=240   (optional, if time allows)

k is bumped with each level so the transverse reach does not shrink as h_min
drops (k=6/h=.05 reaches 7.8 m ~ 1.1 skin depths; L2-L4 reach 9-14 m).
h_svd (sub-voxel classification spacing) scales with h_min so the relative
sub-cell resolution is constant across levels.

Resumable: one (method, level, model) solve per invocation, appended to
examples/out/fig9_refine.npz.  Analyze with `report`.

Usage:
  python examples/fig9_refinement.py solve METHOD LEVEL MODEL
      METHOD in {pointwise, backus, nodal}
      LEVEL  in {1, 2, 3, 4}
      MODEL  in {layer, nolayer}
  python examples/fig9_refinement.py report
"""
import os, sys, time, warnings
import numpy as np
import scipy.sparse.linalg as spla
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
NPZ = os.path.join(OUT, "fig9_refine.npz")

from lebedev_em.grid import (symmetric_optimal_grid, hybrid_axial_grid,
                             C000, C101, C110, C011)
from lebedev_em.media import from_geometry_exact
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    lebedev_B_on_z_axis)

# --- identical physics to fig9_backus_vs_paper.py ---------------------------
FREQ = 52650.0; OMEGA = 2 * np.pi * FREQ
SIG_BG = 0.10; SIG_BORE = 0.05; R_BORE = 0.1
SIG_T = 0.10; SIG_N = SIG_T / 200.0
DIP = np.radians(75.0); N_HAT = np.array([np.sin(DIP), 0., np.cos(DIP)])
D_TOP = N_HAT[2] * (-0.95); D_BOT = N_HAT[2] * (-1.93)
SIG_ANISO = SIG_T * np.eye(3) + (SIG_N - SIG_T) * np.outer(N_HAT, N_HAT)
GAMMA = 1 / 2 ** 0.5
CLUSTERS = [C000, C101, C110, C011]

PAPER = {'nolayer': {-2.17: 1.50, -1.93: 1.75, -1.68: 2.05, -1.43: 2.47, -1.17: 3.08,
                     -0.95: 3.95, -0.77: 4.90, -0.66: 5.68, -0.57: 6.55, -0.47: 7.95,
                     -0.38: 9.95, -0.30: 12.70, -0.25: 14.90},
         'layer':   {-2.17: 0.80, -1.93: 0.95, -1.68: 1.17, -1.43: 1.47, -1.17: 1.92,
                     -0.95: 2.63, -0.77: 3.52, -0.66: 4.25, -0.57: 5.10, -0.47: 6.42,
                     -0.38: 8.42, -0.30: 11.15, -0.25: 13.40}}
Z_EVAL = np.array(sorted(PAPER['layer'].keys()))      # receiver positions

# level -> (h_min, k, n_inner)
LEVELS = {1: (0.05, 6, 96),
          2: (1.0 / 30.0, 7, 144),
          3: (0.025, 8, 192),
          4: (0.02, 9, 240)}


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


def build_grid(level):
    h_min, k, n_inner = LEVELS[level]
    z_fd = hybrid_axial_grid(-3.5, 2.5, n_inner, 8, GAMMA)
    return symmetric_optimal_grid(h_min, 300., z_fd, GAMMA, k=k)


def solve(level, with_layer, method):
    h_min, k, n_inner = LEVELS[level]
    grid = build_grid(level)
    tag = f"{method}_L{level}_{'layer' if with_layer else 'nolayer'}"
    print(f"[{tag}] h_min={h_min:.4f} k={k} n_inner={n_inner} "
          f"Mx={len(grid.x)-1} Mz={len(grid.z)-1} 3N_R={3*grid.N_R}", flush=True)
    t0 = time.time()
    sf, geo = make_model(with_layer)
    # h_svd scaled so sub-cell resolution relative to h_min matches the base
    # run (0.03 for pointwise/backus, 0.025 for nodal at h_min=0.05).
    scale = h_min / 0.05
    h_svd = (0.03 if method in ("pointwise", "backus") else 0.025) * scale
    med = from_geometry_exact(grid, sf, geo, method=method, h_svd=h_svd)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    print(f"[{tag}] media+assembly {time.time()-t0:.0f}s", flush=True)
    rhs = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=2)   # z magnetic dipole
    b = sum(rhs[c] for c in CLUSTERS)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b.copy(), _component_aware_bc_dofs(grid))
    A_bc = A_bc.tocsr()
    d = A_bc.diagonal(); d_inv = np.where(np.abs(d) > 1e-30, 1.0 / d, 1.0)
    M = spla.LinearOperator(A_bc.shape, matvec=lambda x: d_inv * x, dtype=complex)
    t1 = time.time()
    E, info = spla.lgmres(A_bc, b_bc, M=M, rtol=1e-8, atol=0,
                          maxiter=1000, inner_m=30, outer_k=10)
    B = compute_B_from_E(grid, E, OMEGA)
    z, Bz = lebedev_B_on_z_axis(grid, {c: B for c in CLUSTERS}, comp=2)
    z = np.asarray(z); Bz = np.asarray(Bz).imag * 1e9
    print(f"[{tag}] lgmres info={info} {time.time()-t1:.0f}s", flush=True)
    if info != 0:
        print(f"[{tag}] WARNING: lgmres did not converge (info={info})", flush=True)
    order = np.argsort(z)
    vals = np.interp(Z_EVAL, z[order], Bz[order])
    return z[order], Bz[order], vals, info


def save(key, z, Bz, vals, info):
    data = {}
    if os.path.exists(NPZ):
        with np.load(NPZ) as d:
            data = {k: d[k] for k in d.files}
    data[f"{key}_z"] = z
    data[f"{key}_Bz"] = Bz
    data[f"{key}_vals"] = vals
    data[f"{key}_info"] = np.array(info)
    np.savez(NPZ, **data)
    print(f"saved {key} -> {NPZ}", flush=True)


def report():
    if not os.path.exists(NPZ):
        print("no data yet"); return
    with np.load(NPZ) as d:
        data = {k: d[k] for k in d.files}
    keys = sorted({k[:-5] for k in data if k.endswith("_vals")})
    print("completed solves:", ", ".join(keys) or "(none)")
    paper_l = np.array([PAPER['layer'][zz] for zz in Z_EVAL])
    paper_n = np.array([PAPER['nolayer'][zz] for zz in Z_EVAL])

    def gmean_ratio(vals, paper):
        r = vals / paper
        r = r[r > 0]
        return np.exp(np.mean(np.log(r)))

    for model, paper in (("layer", paper_l), ("nolayer", paper_n)):
        print(f"\n=== {model}: geo-mean ratio vs digitized paper ===")
        print("level  h_min   " + "".join(f"{m:>11s}" for m in ("pointwise", "backus", "nodal")))
        for lv in sorted(LEVELS):
            row = f"L{lv}    {LEVELS[lv][0]:.4f} "
            any_ = False
            for m in ("pointwise", "backus", "nodal"):
                key = f"{m}_L{lv}_{model}_vals"
                if key in data:
                    row += f"{gmean_ratio(data[key], paper):11.3f}"; any_ = True
                else:
                    row += f"{'--':>11s}"
            if any_:
                print(row)

    # pointwise-vs-mean(backus,nodal) gap on the layer curve, per level
    print("\n=== layer: geo-mean(pointwise / mean(backus,nodal)) ===")
    for lv in sorted(LEVELS):
        kp = f"pointwise_L{lv}_layer_vals"
        kb = f"backus_L{lv}_layer_vals"
        kn = f"nodal_L{lv}_layer_vals"
        have = [k for k in (kb, kn) if k in data]
        if kp in data and have:
            ref = np.mean([data[k] for k in have], axis=0)
            print(f"L{lv}: {gmean_ratio(data[kp], ref):.3f}   (ref = {len(have)} avg method(s))")

    # per-receiver table at the deepest level with all three methods (layer)
    for lv in sorted(LEVELS, reverse=True):
        ks = [f"{m}_L{lv}_layer_vals" for m in ("pointwise", "backus", "nodal")]
        if all(k in data for k in ks):
            print(f"\n=== layer, L{lv}: per-receiver Im Bz (nT) ===")
            print("  z      paper  pointwise   backus    nodal")
            for i, zz in enumerate(Z_EVAL):
                print(f"{zz:6.2f} {paper_l[i]:8.2f} {data[ks[0]][i]:9.3f} "
                      f"{data[ks[1]][i]:9.3f} {data[ks[2]][i]:9.3f}")
            break


def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    with np.load(NPZ) as d:
        data = {k: d[k] for k in d.files}
    colors = {"pointwise": "tab:blue", "backus": "tab:red", "nodal": "tab:green"}
    lstyle = {1: (0, (1, 2)), 2: (0, (4, 2)), 3: (0, (7, 2)), 4: "solid"}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Left: layer curves, all methods x levels, + digitized paper
    a = ax[0]
    zp = np.array(list(PAPER['layer'].keys())); vp = np.array(list(PAPER['layer'].values()))
    a.plot(zp, vp, "ks", ms=6, label="DDH03 Fig. 9 (digitized)", zorder=5)
    for m in ("pointwise", "backus", "nodal"):
        for lv in sorted(LEVELS):
            k = f"{m}_L{lv}_layer"
            if f"{k}_z" in data:
                lbl = f"{m} L{lv}" if lv in (1, max(v for v in sorted(LEVELS)
                        if f"{m}_L{v}_layer_z" in data)) else None
                a.plot(data[f"{k}_z"], data[f"{k}_Bz"], color=colors[m],
                       ls=lstyle[lv], lw=1.4, label=lbl)
    a.set_xlim(-2.3, -0.2); a.set_ylim(0, 16); a.grid(alpha=0.3)
    a.set_xlabel("z (m)"); a.set_ylabel(r"Im $B_z$ (nT)")
    a.set_title("Layer curves under refinement (dotted→solid = finer)", fontsize=10)
    a.legend(fontsize=7, ncol=2)

    # Right: geo-mean ratio vs paper as a function of h_min
    a = ax[1]
    paper_l = np.array([PAPER['layer'][zz] for zz in Z_EVAL])
    for m in ("pointwise", "backus", "nodal"):
        hs, rs = [], []
        for lv in sorted(LEVELS):
            k = f"{m}_L{lv}_layer_vals"
            if k in data:
                r = data[k] / paper_l
                hs.append(LEVELS[lv][0]); rs.append(np.exp(np.mean(np.log(r[r > 0]))))
        if hs:
            a.plot(hs, rs, "o-", color=colors[m], label=f"{m} (layer)")
    a.axhline(1.0, color="k", lw=0.8, ls="--", label="digitized paper")
    a.set_xlabel(r"transverse $h_{\min}$ (m)"); a.set_ylabel("geo-mean ratio vs paper")
    a.set_title("Refinement drift of each scheme", fontsize=10)
    a.invert_xaxis(); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("DDH03 Fig. 9 configuration: refinement study (borehole + tilted TI layer)",
                 fontsize=11)
    fig.tight_layout()
    png = os.path.join(OUT, "fig9_refinement.png")
    fig.savefig(png, dpi=130)
    print("figure ->", png)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "report":
        report(); sys.exit(0)
    if mode == "plot":
        plot(); sys.exit(0)
    if mode != "solve" or len(sys.argv) < 5:
        print(__doc__); sys.exit(1)
    method = sys.argv[2]; level = int(sys.argv[3]); model = sys.argv[4]
    assert method in ("pointwise", "backus", "nodal") and level in LEVELS \
        and model in ("layer", "nolayer")
    key = f"{method}_L{level}_{model}"
    if os.path.exists(NPZ):
        with np.load(NPZ) as d:
            if f"{key}_vals" in d.files:
                print(f"{key} already done; skipping"); sys.exit(0)
    z, Bz, vals, info = solve(level, model == "layer", method)
    save(key, z, Bz, vals, info)
