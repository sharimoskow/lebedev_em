"""
Side-by-side: from_sigma_func vs from_fine_grid (NSUB=4) vs pi_nodal reference.
Same K=2 grid, tilted 45° interface.
"""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid, C000, C101, C110, C011
from lebedev_em.media import planar_interface_isotropic, from_sigma_func, from_fine_grid
from lebedev_em.solver import LebedevMaxwellSolver
from lebedev_em.sources import _native_type_for_cluster_comp
from lebedev_em.postprocess import interpolate_cluster_E

SIGMA1=0.1; SIGMA2=1.0; Z_CONT=4.0; OMEGA=2.0*np.pi*2500.0
DZ=0.0625; Z_INNER_MIN=-0.25; Z_INNER_MAX=7.75
N_INNER=int(round((Z_INNER_MAX-Z_INNER_MIN)/DZ))
K_OUTER=8; GAMMA=1.0/np.sqrt(2.0); H_MIN=0.5; L_TRANS=300.0; K_GRID=2

z_fd_grid = hybrid_axial_grid(Z_INNER_MIN, Z_INNER_MAX, N_INNER, K_OUTER, GAMMA)
grid      = symmetric_optimal_grid(H_MIN, L_TRANS, z_fd_grid, GAMMA, k=K_GRID)
print(f"Grid: N_R={grid.N_R}")

N_HAT_45   = np.array([1.,0.,1.])/np.sqrt(2.)
D_PLANE_45 = float(Z_CONT/np.sqrt(2.))

def sigma_func(X, Y, Z):
    return np.where(N_HAT_45[0]*X + N_HAT_45[2]*Z < D_PLANE_45, SIGMA1, SIGMA2)

# ── Fine grid (NSUB=4 subdivision, same as fine_grid_validation2.py) ─────────
def subdivide(g, nsub):
    pts = []
    for i in range(len(g)-1):
        pts.extend(np.linspace(float(g[i]), float(g[i+1]), nsub+1)[:-1])
    pts.append(float(g[-1]))
    return np.array(pts)

NSUB=4
ffx = subdivide(grid.x, NSUB)
ffy = subdivide(grid.y, NSUB)
ffz = subdivide(z_fd_grid, NSUB)
print(f"Fine grid: {len(ffx)}×{len(ffy)}×{len(ffz)} = {len(ffx)*len(ffy)*len(ffz)/1e6:.1f}M pts")
print(f"  dx_min={np.diff(ffx).min():.4f}  dz_min={np.diff(ffz).min():.6f}")

FFX_3D = ffx[:,None,None]; FFZ_3D = ffz[None,None,:]
sig_tilt = np.broadcast_to(
    np.where(FFX_3D + FFZ_3D >= Z_CONT, SIGMA2, SIGMA1).astype(complex),
    (len(ffx), len(ffy), len(ffz))
).copy()

# ── Receivers ──────────────────────────────────────────────────────────────
Mx2, My2 = grid.Mx//2, grid.My//2
nat_c000  = _native_type_for_cluster_comp(C000, 2)
z_eval, seq_c000 = [], []
for seq, (i,j,k) in enumerate(grid.R_nodes):
    if (i==Mx2 and j==My2 and (i%2,j%2,k%2)==nat_c000):
        zv = float(grid.z[k])
        if 0.5 <= zv <= 7.5:
            z_eval.append(zv); seq_c000.append(seq)
z_eval = np.array(z_eval)
order  = np.argsort(z_eval); z_eval = z_eval[order]
print(f"Receivers: {len(z_eval)} nodes")

def extract_Ez(result):
    out = np.zeros(len(z_eval), dtype=complex)
    for idx, zv in enumerate(z_eval):
        vals = [interpolate_cluster_E(grid, result["E_c"][c], c, 2, 0.0, 0.0, zv)
                for c in (C000, C101, C110, C011)]
        out[idx] = np.mean(vals)
    return out

def run(med, label):
    t0 = time.time()
    result = LebedevMaxwellSolver(grid, med, OMEGA).solve(0.0, 0.0, 0.0, dipole_comp=2)
    Ez = extract_Ez(result)
    print(f"  [{label}] solve={time.time()-t0:.1f}s")
    return Ez

print("\nBuilding media...")
t0=time.time(); med_pi = planar_interface_isotropic(grid, N_HAT_45, D_PLANE_45, SIGMA1, SIGMA2, method="nodal"); print(f"  pi_nodal:    {time.time()-t0:.2f}s")
t0=time.time(); med_sf = from_sigma_func(grid, sigma_func, h_svd=0.025, n_line=50, n_vol=8, method="nodal"); print(f"  sf_nodal:    {time.time()-t0:.2f}s")
t0=time.time(); med_fg = from_fine_grid(grid, ffx, ffy, ffz, sig_tilt, method="nodal"); print(f"  fg_nodal:    {time.time()-t0:.2f}s")

print("\nSolving...")
Ez_pi = run(med_pi, "pi_nodal")
Ez_sf = run(med_sf, "sf_nodal")
Ez_fg = run(med_fg, "fg_nodal")

def stats(Ez, ref, label):
    r = np.abs(Ez - ref) / (np.abs(ref) + 1e-30)
    print(f"  {label:20s}  max={r.max()*100:6.2f}%  mean={r.mean()*100:5.2f}%")

print("\n--- Field errors vs pi_nodal ---")
stats(Ez_sf, Ez_pi, "sf_nodal (callable)")
stats(Ez_fg, Ez_pi, "fg_nodal (fine grid)")
