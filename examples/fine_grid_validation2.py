"""
fine_grid_validation2.py
========================
Validate from_fine_grid against analytic / semi-analytic references using the
proper hybrid grid (identical setup to benchmark_two_layer.py).

Three test geometries
---------------------
1. Homogeneous        σ₁ = 0.1 S/m everywhere
2. Horizontal interface  n̂ = ê_z, z_c = 4 m  (Sommerfeld analytic)
3. 45° tilted interface  n̂ = [1,0,1]/√2, x+z=4  (planar_interface_isotropic ref)

Design assumption (mirroring real use):
    Fine σ grid is ALWAYS finer than the FD grid in ALL three dimensions.

    Both the x,y and z fine grids are built by subdividing the FD grid axes
    (NSUB intervals per FD interval), so:
      - the fine grid covers the full FD domain (no boundary-clamping artefacts)
      - every FD dual cell contains multiple fine-grid points
      - minimum fine spacing = minimum FD spacing / NSUB in each direction
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lebedev_em.grid import (
    symmetric_optimal_grid, hybrid_axial_grid,
    C000, C101, C110, C011,
)
from lebedev_em.media import (
    homogeneous_isotropic, layered_isotropic,
    planar_interface_isotropic, from_fine_grid,
)
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.analytics import (
    electric_dipole_Ez_homogeneous_onaxis,
    electric_dipole_Ez_two_layer_onaxis,
)
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

# ── Physical parameters ───────────────────────────────────────────────────────
SIGMA1  = 0.1;  SIGMA2 = 1.0
Z_SRC   = 0.0;  Z_CONT = 4.0
OMEGA   = 2.0 * np.pi * 2500.0

# ── FD grid (k=3, identical to benchmark_two_layer.py) ───────────────────────
DZ          = 0.0625        # inner-zone FD spacing
Z_INNER_MIN = -0.25
Z_INNER_MAX =  7.75
N_INNER     = int(round((Z_INNER_MAX - Z_INNER_MIN) / DZ))   # 128
K_OUTER     = 8
GAMMA       = 1.0 / np.sqrt(2.0)
H_MIN       = 0.5           # minimum transverse FD spacing
L_TRANS     = 300.0         # physical target domain half-length
K_GRID      = 3             # k=3 → domain ±8.8m, fast (0.4s/solve); increase for production

z_fd_grid = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
grid      = symmetric_optimal_grid(H_MIN, L_TRANS, z_fd_grid, GAMMA, k=K_GRID)

print(f"FD grid : Mx=My={grid.Mx}, Mz={grid.Mz}, N_R={grid.N_R}")
print(f"  z ∈ [{z_fd_grid[0]:.2f}, {z_fd_grid[-1]:.2f}] m,  inner dz_FD = {DZ} m")
print(f"  x,y domain ±{grid.x[-1]:.1f} m,  H_MIN = {H_MIN} m")

# ── Receiver nodes: C000-native Ez on-axis ───────────────────────────────────
Mx2, My2 = grid.Mx // 2, grid.My // 2
nat_c000  = _native_type_for_cluster_comp(C000, 2)   # Ez component for VED

z_eval, seq_c000 = [], []
for seq, (i, j, k) in enumerate(grid.R_nodes):
    if (i == Mx2 and j == My2 and (i%2, j%2, k%2) == nat_c000):
        zv = float(grid.z[k])
        if 0.5 <= zv <= 7.5:
            z_eval.append(zv); seq_c000.append(seq)
z_eval   = np.array(z_eval);  seq_c000 = np.array(seq_c000, dtype=int)
order    = np.argsort(z_eval); z_eval   = z_eval[order]; seq_c000 = seq_c000[order]
print(f"  {len(z_eval)} receiver nodes, z ∈ [{z_eval[0]:.3f}, {z_eval[-1]:.3f}] m")

def extract_Ez(result):
    """4-cluster average of Ez at on-axis receivers."""
    out = np.zeros(len(z_eval), dtype=complex)
    for idx, (zv, _) in enumerate(zip(z_eval, seq_c000)):
        vals = [interpolate_cluster_E(grid, result["E_c"][c], c, 2, 0.0, 0.0, zv)
                for c in (C000, C101, C110, C011)]
        out[idx] = np.mean(vals)
    return out

def run(med, label):
    t0 = time.time()
    result = LebedevMaxwellSolver(grid, med, OMEGA).solve(0.0, 0.0, Z_SRC, dipole_comp=2)
    Ez = extract_Ez(result)
    print(f"  [{label}] {time.time()-t0:.1f}s")
    return Ez

# ── Analytic references ───────────────────────────────────────────────────────
Ez_homo_anal = np.array([electric_dipole_Ez_homogeneous_onaxis(zr, Z_SRC, SIGMA1, OMEGA)
                          for zr in z_eval])
Ez_two_anal  = np.array([electric_dipole_Ez_two_layer_onaxis(zr, Z_SRC, Z_CONT,
                                                               SIGMA1, SIGMA2, OMEGA)
                          for zr in z_eval])

# ── Fine σ grids ──────────────────────────────────────────────────────────────
# Both x,y and z fine grids are built by subdividing the FD grid axes by NSUB.
# This guarantees:  fine spacing = FD spacing / NSUB  everywhere in the domain.
NSUB = 4   # subdivision factor in all three directions

def subdivide_grid_axis(g, nsub):
    """Insert (nsub-1) equally-spaced points in every FD interval."""
    pts = []
    for i in range(len(g) - 1):
        pts.extend(np.linspace(float(g[i]), float(g[i+1]), nsub + 1)[:-1])
    pts.append(float(g[-1]))
    return np.array(pts)

ffx = subdivide_grid_axis(grid.x, NSUB)
ffy = subdivide_grid_axis(grid.y, NSUB)
ffz = subdivide_grid_axis(z_fd_grid, NSUB)
NF_X, NF_Y, NF_Z = len(ffx), len(ffy), len(ffz)

assert np.any(np.abs(ffz - Z_CONT) < 1e-10), "Z_CONT must land on fine z-grid"

dx_min = float(np.diff(ffx).min())
dz_min = float(np.diff(ffz).min())
print(f"\nFine σ grid (FD subdivided ×{NSUB}):")
print(f"  {NF_X}×{NF_Y}×{NF_Z} pts,  "
      f"dx_min={dx_min:.4f} m,  dz_min={dz_min:.6f} m")
print(f"  array size: {NF_X}×{NF_Y}×{NF_Z} = {NF_X*NF_Y*NF_Z/1e6:.1f}M pts  "
      f"({NF_X*NF_Y*NF_Z*16/1e6:.0f} MB complex128)")

# σ on the single fine grid (same x,y,z axes for all cases)
FFX_3D = ffx[:, np.newaxis, np.newaxis]   # (Nx, 1,  1 )
FFZ_3D = ffz[np.newaxis, np.newaxis, :]   # (1,  1,  Nz)

sig_hom   = np.full((NF_X, NF_Y, NF_Z), SIGMA1, dtype=complex)
sig_horiz = np.broadcast_to(
    np.where(FFZ_3D >= Z_CONT, SIGMA2, SIGMA1).astype(complex),
    (NF_X, NF_Y, NF_Z)
).copy()
# 45° tilted: x + z ≥ Z_CONT  →  σ₂
sig_tilt45 = np.broadcast_to(
    np.where(FFX_3D + FFZ_3D >= Z_CONT, SIGMA2, SIGMA1).astype(complex),
    (NF_X, NF_Y, NF_Z)
).copy()

# ── Tilted-interface geometry ─────────────────────────────────────────────────
# n̂ = [1,0,1]/√2, interface: n̂·r = Z_CONT/√2
N_HAT_45   = np.array([1., 0., 1.]) / np.sqrt(2.)
D_PLANE_45 = float(Z_CONT / np.sqrt(2.))   # ≈ 2.828 m

# ── Build all media ───────────────────────────────────────────────────────────
print("\nBuilding media...")
t0 = time.time()

# References
med_homo_ref  = homogeneous_isotropic(grid, SIGMA1)
med_horiz_ref = layered_isotropic(grid, [Z_CONT], [SIGMA1, SIGMA2], direction="z")
med_horiz_pi  = planar_interface_isotropic(grid, [0,0,1], Z_CONT, SIGMA1, SIGMA2,
                                            method="nodal")
med_tilt_pi   = planar_interface_isotropic(grid, N_HAT_45, D_PLANE_45, SIGMA1, SIGMA2,
                                            method="nodal")

# All fine-grid media use the same (ffx, ffy, ffz) axes
def ffg(sig, method):
    return from_fine_grid(grid, ffx, ffy, ffz, sig, method=method)

# Homogeneous fine-grid media
med_hom_pw    = ffg(sig_hom, "pointwise")
med_hom_arith = ffg(sig_hom, "arithmetic")
med_hom_nodal = ffg(sig_hom, "nodal")

# Horizontal-interface fine-grid media
med_horiz_pw    = ffg(sig_horiz, "pointwise")
med_horiz_arith = ffg(sig_horiz, "arithmetic")
med_horiz_diag  = ffg(sig_horiz, "diagonal")
med_horiz_backus= ffg(sig_horiz, "backus")
med_horiz_nodal = ffg(sig_horiz, "nodal")

# 45°-tilted fine-grid media
med_tilt_diag  = ffg(sig_tilt45, "diagonal")
med_tilt_backus= ffg(sig_tilt45, "backus")
med_tilt_nodal = ffg(sig_tilt45, "nodal")

print(f"  all media built in {time.time()-t0:.1f}s")

# ── Solves ────────────────────────────────────────────────────────────────────
print("\nHomogeneous solves:")
Ez_homo_ref   = run(med_homo_ref,  "layered ref")
Ez_hom_pw     = run(med_hom_pw,    "fine-grid pointwise")
Ez_hom_arith  = run(med_hom_arith, "fine-grid arithmetic")
Ez_hom_nodal  = run(med_hom_nodal, "fine-grid nodal")

print("\nHorizontal interface solves:")
Ez_horiz_ref   = run(med_horiz_ref,   "layered_isotropic ref")
Ez_horiz_pi    = run(med_horiz_pi,    "planar_interface nodal")
Ez_horiz_pw    = run(med_horiz_pw,    "fine-grid pointwise")
Ez_horiz_arith = run(med_horiz_arith, "fine-grid arithmetic")
Ez_horiz_diag  = run(med_horiz_diag,  "fine-grid diagonal")
Ez_horiz_backus= run(med_horiz_backus,"fine-grid backus")
Ez_horiz_nodal = run(med_horiz_nodal, "fine-grid nodal")

print("\n45° tilted interface solves:")
Ez_tilt_pi    = run(med_tilt_pi,    "planar_interface nodal (ref)")
Ez_tilt_diag  = run(med_tilt_diag,  "fine-grid diagonal")
Ez_tilt_backus= run(med_tilt_backus,"fine-grid backus")
Ez_tilt_nodal = run(med_tilt_nodal, "fine-grid nodal")

# ── Error tables ──────────────────────────────────────────────────────────────
def err(Ez, ref):
    return np.abs(Ez - ref) / np.abs(ref) * 100

def print_table(label, ref_label, ref, **kwargs):
    print(f"\n{label}  [ref = {ref_label}]")
    names = list(kwargs.keys())
    w = max(10, max(len(n) for n in names))
    header = f"  {'z':>6}  {'|Ez| ref':>12}  " + "  ".join(f"{n:>{w}}" for n in names)
    print(header)
    print("  " + "-"*(len(header)-2))
    for j in range(len(z_eval)):
        row = f"  {z_eval[j]:>6.3f}  {abs(ref[j]):>12.4e}  "
        row += "  ".join(f"{err(v[j], ref[j]):>{w-1}.2f}%" for v in kwargs.values())
        print(row)

mask_away = np.abs(z_eval - Z_CONT) > 0.1

print_table("HOMOGENEOUS (rel error vs analytic):", "analytic", Ez_homo_anal,
            hom_ref=Ez_homo_ref, pointwise=Ez_hom_pw,
            arith=Ez_hom_arith, nodal=Ez_hom_nodal)

print_table("HORIZONTAL INTERFACE (rel error vs Sommerfeld):", "Sommerfeld", Ez_two_anal,
            horiz_ref=Ez_horiz_ref, pi_nodal=Ez_horiz_pi,
            pointwise=Ez_horiz_pw, fg_arith=Ez_horiz_arith,
            fg_diag=Ez_horiz_diag, fg_backus=Ez_horiz_backus, fg_nodal=Ez_horiz_nodal)

print_table("45° TILTED INTERFACE (rel error vs pi_nodal ref):", "planar_interface(nodal)",
            Ez_tilt_pi,
            fg_diag=Ez_tilt_diag, fg_backus=Ez_tilt_backus, fg_nodal=Ez_tilt_nodal)

print(f"\nSummary — max rel-error vs reference (|z-{Z_CONT}|>0.1):")
print(f"  {'Method':<18} {'Horizontal':>12}  {'Tilted 45°':>12}")
print("  " + "-"*45)
for lbl, Ez_h, Ez_t, ref_h, ref_t in [
    ("horiz_ref/—",   Ez_horiz_ref,    None,          Ez_two_anal, None),
    ("pi_nodal",      Ez_horiz_pi,     Ez_tilt_pi,    Ez_two_anal, Ez_tilt_pi),
    ("pointwise",     Ez_horiz_pw,     None,          Ez_two_anal, None),
    ("fg_arith",      Ez_horiz_arith,  None,          Ez_two_anal, None),
    ("fg_diagonal",   Ez_horiz_diag,   Ez_tilt_diag,  Ez_two_anal, Ez_tilt_pi),
    ("fg_backus",     Ez_horiz_backus, Ez_tilt_backus,Ez_two_anal, Ez_tilt_pi),
    ("fg_nodal",      Ez_horiz_nodal,  Ez_tilt_nodal, Ez_two_anal, Ez_tilt_pi),
]:
    h_str = f"{np.max(err(Ez_h, ref_h)[mask_away]):>11.2f}%" if Ez_h is not None and ref_h is not None else "         —"
    # for tilted, exclude pi_nodal vs itself (that's 0)
    if Ez_t is not None and ref_t is not None and lbl != "pi_nodal":
        t_str = f"{np.max(err(Ez_t, ref_t)[mask_away]):>11.2f}%"
    else:
        t_str = "         —"
    print(f"  {lbl:<18} {h_str}  {t_str}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# Panel 1: Homogeneous
ax = axes[0]
ax.semilogy(z_eval, np.abs(Ez_homo_anal), 'k-',  lw=2.5, label='Analytic')
ax.semilogy(z_eval, np.abs(Ez_homo_ref),  'b--', lw=1.5, label='layered_isotropic')
ax.semilogy(z_eval, np.abs(Ez_hom_pw),   'c:',  lw=1.5, label='pointwise')
ax.semilogy(z_eval, np.abs(Ez_hom_arith), 'r:',  lw=1.5, label='arithmetic')
ax.semilogy(z_eval, np.abs(Ez_hom_nodal), 'g-.', lw=1.5, label='nodal')
ax.axvline(Z_CONT, color='gray', lw=0.8, ls=':')
ax.set_xlabel('z (m)'); ax.set_ylabel('|Ez| (V/m)')
ax.set_title(f'Homogeneous  σ = {SIGMA1} S/m')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel 2: Horizontal interface
ax = axes[1]
ax.semilogy(z_eval, np.abs(Ez_two_anal),    'k-',  lw=2.5, label='Sommerfeld analytic')
ax.semilogy(z_eval, np.abs(Ez_horiz_ref),   'b--', lw=1.5, label='layered_isotropic')
ax.semilogy(z_eval, np.abs(Ez_horiz_pw),   'm:',  lw=1.5, label='pointwise')
ax.semilogy(z_eval, np.abs(Ez_horiz_arith), 'r:',  lw=1.5, label='arithmetic')
ax.semilogy(z_eval, np.abs(Ez_horiz_diag),  'y-',  lw=1.0, label='diagonal')
ax.semilogy(z_eval, np.abs(Ez_horiz_backus),'tab:orange', lw=1.0, ls='--', label='backus')
ax.semilogy(z_eval, np.abs(Ez_horiz_nodal), 'g-.', lw=1.5, label='nodal')
ax.axvline(Z_CONT, color='gray', lw=0.8, ls=':', label=f'z={Z_CONT} m')
ax.set_xlabel('z (m)'); ax.set_ylabel('|Ez| (V/m)')
ax.set_title(f'Horizontal interface  n̂=ê_z, σ₁={SIGMA1}, σ₂={SIGMA2}')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel 3: 45° tilted interface
ax = axes[2]
ax.semilogy(z_eval, np.abs(Ez_tilt_pi),    'k-',  lw=2.5, label='planar_interface nodal (ref)')
ax.semilogy(z_eval, np.abs(Ez_tilt_diag),  'y-',  lw=1.5, label='fine-grid diagonal')
ax.semilogy(z_eval, np.abs(Ez_tilt_backus),'tab:orange', lw=1.5, ls='--', label='fine-grid backus')
ax.semilogy(z_eval, np.abs(Ez_tilt_nodal), 'g-.', lw=1.5, label='fine-grid nodal')
ax.axvline(Z_CONT, color='gray', lw=0.8, ls=':', label=f'on-axis interface z={Z_CONT} m')
ax.set_xlabel('z (m)'); ax.set_ylabel('|Ez| (V/m)')
ax.set_title(f'45° tilted interface  n̂=[1,0,1]/√2, σ₁={SIGMA1}, σ₂={SIGMA2}')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

note = (f"Fine σ grid: FD axes subdivided ×{NSUB}  →  "
        f"dx_min={dx_min:.4f} m,  dz_min={dz_min:.6f} m  "
        f"({NF_X}×{NF_Y}×{NF_Z} pts)")
fig.suptitle(f'from_fine_grid validation — Mx=My={grid.Mx}, Mz={grid.Mz}, DZ_FD={DZ} m\n{note}',
             fontsize=10)
plt.tight_layout()
plt.savefig("examples/fine_grid_validation2.png", dpi=140)
print("\nSaved examples/fine_grid_validation2.png")
