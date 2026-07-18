"""
plot_ddh03_fig7.py — DDH03 Figure 7 model: SELF-CONSISTENCY convergence test only.

STATUS NOTE (July 2026): this script demonstrates k-convergence of the optimal
grids (its original purpose) but is NOT a like-for-like reproduction of the
paper's method, and its absolute amplitudes should not be compared against the
published Fig. 7 curves. It uses a single-cluster source, a global BC
assignment with nearest-node extraction, a coarse transverse grid
(H_MIN = 0.5 m, so the R = 0.1 m borehole is under-resolved — and since all k
share H_MIN, the k-convergence test cannot detect the resulting bias), and
pointwise media with no sub-cell homogenization. For the complete DDH03
methodology (eq.-7 four-cluster sources, per-cluster mixed BCs,
interpolate-then-average, h_min = 0.05 m, nodal homogenization), see
fig7_full_ddh03.py. Even with the full method, the published Fig. 7 amplitudes
are not yet matched (flat factors ~1.8 in Bxx, ~2.7 in Bxz, localized to the
invasion-zone treatment by the variant scan in fig7_variants.py); the
absolutely calibrated benchmark against published curves is fig9_check.py.

Geometry (Figure 6 of DDH03):
  - Borehole   : σ = 0.05 S/m, R = 0.1 m
  - Invasion   : σ = 0.1  S/m, R = 0.6 m
  - Dipping anisotropic layer : σ_N = 0.01, σ_T = 0.1 S/m, 60° dip
      interface crosses borehole axis at z = –0.5 m
  - Isotropic layer : σ = 0.5 S/m

Source  : x-directed MAGNETIC 52.65 kHz dipole at the origin.
          Implemented as b = iω C_PR M_P_vec (magnetic dipole in E-field system).
Grid    : Mz = 98 (hybrid axial), Mx = My = 4k, k = 3, 4, 5 [, 6].
          NOTE: DDH03 uses My = 2k (half y-axis symmetry); we use My = 4k (full
          domain), which is physically equivalent but 2× more expensive in y.
Plotted : Im(Bxx) and Im(Bxz) vs z for z ∈ [–1.7, –0.05] m.

Usage
-----
Results for k=3,4,5 are cached in k345_results.npz (run once to compute).
k=6 (~3–5 min on a laptop) is cached in k6_results.npz — run run_k6.py first,
then re-run this script to include it in the figure.

DDH03: Davydycheva, Druskin & Habashy (2003), Geophysics 68(5):1525–1536.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import EMMedia, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, extract_B_on_axis

# ── Physical parameters ───────────────────────────────────────────────────────
FREQ  = 52650.0
OMEGA = 2.0 * np.pi * FREQ

# Conductivities
SIGMA_BORE = 0.05   # borehole
SIGMA_INV  = 0.10   # invasion zone
SIGMA_ISO  = 0.50   # isotropic formation layer
SIGMA_N    = 0.01   # anisotropic layer — normal direction (perpendicular to layer)
SIGMA_T    = 0.10   # anisotropic layer — tangential direction (parallel to layer)

# Borehole/invasion radii
R_BORE = 0.1        # m
R_INV  = 0.6        # m

# Dipping interface geometry (60° dip from horizontal)
DIP_DEG    = 60.0
DIP_RAD    = np.radians(DIP_DEG)
N_HAT      = np.array([np.sin(DIP_RAD), 0.0, np.cos(DIP_RAD)])   # layer normal
Z_IFACE    = -0.5   # z at which interface crosses borehole axis (x=0)
D_PLANE    = N_HAT[0] * 0.0 + N_HAT[2] * Z_IFACE   # = Z_IFACE * cos(60°) = –0.25
#   anisotropic layer: N_HAT . r < D_PLANE  →  sin(60°)x + cos(60°)z < –0.25

# Intrinsic conductivity tensor of the dipping anisotropic layer (Cartesian)
# σ = σ_T I + (σ_N – σ_T) n̂⊗n̂
SIGMA_ANISO = (SIGMA_T * np.eye(3)
               + (SIGMA_N - SIGMA_T) * np.outer(N_HAT, N_HAT))

# ── Z-grid (Mz=98 inner nodes, matches DDH03) ─────────────────────────────────
Z_INNER_MIN = -3.5
Z_INNER_MAX =  2.5
N_INNER     = 98
K_OUTER     = 8
GAMMA       = 1.0 / np.sqrt(2.0)
H_MIN       = 0.5
L_TRANS     = 300.0

z_fd_grid = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
print(f"Z-grid: N_inner={N_INNER}, total z-nodes={len(z_fd_grid)}")

# ── Measurement range ─────────────────────────────────────────────────────────
Z_MEAS_MIN, Z_MEAS_MAX = -1.7, -0.05


def build_media(grid):
    """
    Build EMMedia with anisotropic dipping layers for the given grid.
    Each R-node is assigned the bulk conductivity tensor of its region
    (pointwise assignment — no interface homogenization).
    """
    N_R = grid.N_R
    sigma_R = np.zeros((N_R, 3, 3), dtype=complex)

    for seq, (i, j, k) in enumerate(grid.R_nodes):
        x = float(grid.x[i])
        y = float(grid.y[j])
        z = float(grid.z[k])
        r_xy = np.sqrt(x**2 + y**2)

        if r_xy < R_BORE:
            sigma_R[seq] = SIGMA_BORE * np.eye(3)
        elif r_xy < R_INV:
            sigma_R[seq] = SIGMA_INV * np.eye(3)
        else:
            # Formation: check which side of the dipping interface
            side = N_HAT[0] * x + N_HAT[2] * z
            if side < D_PLANE:
                sigma_R[seq] = SIGMA_ANISO
            else:
                sigma_R[seq] = SIGMA_ISO * np.eye(3)

    mu_P  = np.full(grid.N_P, complex(MU0))
    eps_R = np.full(grid.N_R, complex(EPS0))
    return EMMedia(grid, sigma_R, mu_P, eps_R)


def build_magnetic_x_dipole_rhs(grid, solver_obj, omega, m_x=1.0):
    """
    Build RHS vector for a magnetic x-directed dipole at the origin.

    Physics derivation (e^{-iωt} convention, A = C_PR inv_mu C_RE - iω σ_dot):
        ∇ × E = iωμ₀(H + M)  →  b = iω C_PR @ M_P_vec
    where M_P_vec[Hx_dof] = m_x / vol_P at the Hx P-node nearest the origin.

    Hx P-nodes have type (0,0,0): all even (i,j,k).
    """
    Mx, My, Mz = grid.Mx, grid.My, grid.Mz

    # x: i = Mx//2 (must be even for symmetric grid with even Mx)
    i0 = Mx // 2
    assert i0 % 2 == 0, f"i0={i0} is odd (Mx={Mx}); grid must have even Mx"

    # y: j = My//2 (even for even My)
    j0 = My // 2
    if j0 % 2 != 0:
        raise ValueError(f"j0={j0} is odd (My={My}); need My divisible by 4 for centered Hx node")

    # z: find even k with z[k] nearest 0
    k_even = np.arange(0, Mz + 1, 2)
    k0 = int(k_even[np.argmin(np.abs(grid.z[k_even]))])

    # Validate P-node
    assert (i0 + j0 + k0) % 2 == 0, f"Not a P-node: ({i0},{j0},{k0})"
    P_seq = int(grid.P_idx[i0, j0, k0])
    assert P_seq >= 0, f"P_idx=-1 at ({i0},{j0},{k0})"

    # Dual cell volume (skip-2 spacings)
    dx = float(grid.x[min(i0 + 1, Mx)] - grid.x[max(i0 - 1, 0)])
    dy = float(grid.y[min(j0 + 1, My)] - grid.y[max(j0 - 1, 0)])
    dz = float(grid.z[min(k0 + 1, Mz)] - grid.z[max(k0 - 1, 0)])
    vol_P = abs(dx * dy * dz)

    print(f"    Hx P-node: ({i0},{j0},{k0}), seq={P_seq}, "
          f"pos=({grid.x[i0]:.4f},{grid.y[j0]:.4f},{grid.z[k0]:.4f}), "
          f"vol_P={vol_P:.4e}")

    M_P_vec = np.zeros(3 * grid.N_P, dtype=complex)
    M_P_vec[0 * grid.N_P + P_seq] = m_x / vol_P  # Hx component

    b = 1j * omega * (solver_obj._C_PR @ M_P_vec)
    return b


def run_k(k_val):
    t0 = time.time()
    grid = symmetric_optimal_grid(H_MIN, L_TRANS, z_fd_grid, GAMMA, k=k_val)
    t_grid = time.time() - t0
    print(f"\n  k={k_val}: Mx={grid.Mx}, My={grid.My}, N_R={grid.N_R}, N_P={grid.N_P}"
          f"  (grid={t_grid:.1f}s)")

    t1 = time.time()
    med = build_media(grid)
    t_med = time.time() - t1
    print(f"    media build: {t_med:.1f}s")

    t2 = time.time()
    solver = LebedevMaxwellSolver(grid, med, OMEGA)

    # Magnetic x-dipole source
    b_mag = build_magnetic_x_dipole_rhs(grid, solver, OMEGA, m_x=1.0)

    # Apply BCs and solve
    bc_dofs = _component_aware_bc_dofs(grid)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b_mag, bc_dofs)
    E = spla.spsolve(A_bc, b_bc)
    t_sol = time.time() - t2
    print(f"    solve:       {t_sol:.1f}s")

    # B-field from E-field (SI units: Tesla for 1 A·m² magnetic dipole)
    B_vec = compute_B_from_E(grid, E, OMEGA)

    # Extract Bx (comp=0) and Bz (comp=2) on the z-axis
    z_bx, Bx_axis = extract_B_on_axis(grid, B_vec, comp=0, axis='z')
    z_bz, Bz_axis = extract_B_on_axis(grid, B_vec, comp=2, axis='z')

    # Filter to measurement window
    mask_x = (z_bx >= Z_MEAS_MIN) & (z_bx <= Z_MEAS_MAX)
    mask_z = (z_bz >= Z_MEAS_MIN) & (z_bz <= Z_MEAS_MAX)

    print(f"    Bx pts in window: {mask_x.sum()},  Bz pts: {mask_z.sum()}")
    print(f"    Im(Bxx) range: {np.imag(Bx_axis[mask_x]).min()*1e9:.2f} to "
          f"{np.imag(Bx_axis[mask_x]).max()*1e9:.2f} nT")

    return (z_bx[mask_x], np.imag(Bx_axis[mask_x]),
            z_bz[mask_z], np.imag(Bz_axis[mask_z]))


# ── Load or compute k=3,4,5 ───────────────────────────────────────────────────
CACHE_345 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k345_results.npz")
CACHE_6   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k6_results.npz")

results = {}
if os.path.exists(CACHE_345):
    print(f"Loading cached k=3,4,5 from {os.path.basename(CACHE_345)}")
    d = np.load(CACHE_345)
    for k in [3, 4, 5]:
        results[k] = (d[f"z_bx_{k}"], d[f"Bxx_{k}"], d[f"z_bz_{k}"], d[f"Bxz_{k}"])
else:
    print("Computing k=3,4,5 (will cache to k345_results.npz) ...")
    for k in [3, 4, 5]:
        results[k] = run_k(k)
    np.savez(CACHE_345,
        z_bx_3=results[3][0], Bxx_3=results[3][1],
        z_bz_3=results[3][2], Bxz_3=results[3][3],
        z_bx_4=results[4][0], Bxx_4=results[4][1],
        z_bz_4=results[4][2], Bxz_4=results[4][3],
        z_bx_5=results[5][0], Bxx_5=results[5][1],
        z_bz_5=results[5][2], Bxz_5=results[5][3])
    print(f"Saved {CACHE_345}")

# Load k=6 if run_k6.py has been run
if os.path.exists(CACHE_6):
    print(f"Loading k=6 from {os.path.basename(CACHE_6)}")
    d6 = np.load(CACHE_6)
    results[6] = (d6["z_bx"], d6["Bxx"], d6["z_bz"], d6["Bxz"])
else:
    print("k=6 not cached — run 'python run_k6.py' to add it (takes ~3–5 min).")

# ── Plot ──────────────────────────────────────────────────────────────────────
markers = {3: "^", 4: "s", 5: "*", 6: "o"}
colors  = {3: "tab:blue", 4: "tab:orange", 5: "tab:green", 6: "tab:red"}
msize   = {3: 5, 4: 5, 5: 6, 6: 5}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle(
    "DDH03 Fig. 7 — Convergence test: borehole + invasion + 60° dipping anisotropic layer\n"
    "52.65 kHz x-directed magnetic dipole  |  Mx = My = 4k,  Mz = 98",
    fontsize=10
)

for ax_idx, (comp_key, comp_lbl, ylim) in enumerate([
        (0, "Bxx", (0.5, 200)),
        (2, "Bxz", (0.1, 5))]):
    ax = axes[ax_idx]
    ax.set_yscale("log")

    for k, (zx, Bxx, zz, Bxz) in sorted(results.items()):
        data = Bxx if comp_key == 0 else Bxz
        z    = zx  if comp_key == 0 else zz
        ax.plot(z, np.abs(data) * 1e9,
                marker=markers[k], color=colors[k],
                markersize=msize[k], linewidth=0.9,
                label=f"k = {k}  (Mx = {4*k})")

    ax.axvline(Z_IFACE, color="gray", lw=0.9, ls=":", alpha=0.7,
               label="interface  z = −0.5 m")
    ax.set_xlabel("z  (m)", fontsize=11)
    ax.set_ylabel(r"$|\,\mathrm{Im}\,B|$  (nT)  [per A·m² dipole]", fontsize=10)

    if comp_key == 0:
        ax.set_title(r"$\mathrm{Im}(B_{xx})$  — solid (self-consistent)", fontsize=11)
    else:
        ax.set_title(r"$\mathrm{Im}(B_{xz})$  — off-diagonal (dip-induced)", fontsize=11)

    ax.set_xlim(Z_MEAS_MIN, 0.0)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", alpha=0.3)

    ax.axvspan(Z_MEAS_MIN, Z_IFACE, alpha=0.04, color="tab:blue")
    ax.axvspan(Z_IFACE, 0.0,        alpha=0.04, color="tab:orange")
    ylo = ylim[0] * 1.5
    ax.text(-1.1, ylo, "Anisotropic layer\n"
            r"$\sigma_T=0.1,\;\sigma_N=0.01$ S/m",
            fontsize=7.5, color="navy", ha="center", va="bottom")
    ax.text(-0.13, ylo, "Isotropic\n" r"$\sigma=0.5$ S/m",
            fontsize=7.5, color="saddlebrown", ha="center", va="bottom")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ddh03_fig7.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
