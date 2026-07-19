"""
tilted_aniso_wholespace.py — does the COUPLED solve handle off-diagonal sigma
correctly?  A controlled test with an exact analytic reference and Fig-9-level
off-diagonal coupling, but NO interface and NO averaging.

Physics
-------
A magnetic dipole aligned with the symmetry axis n of a uniaxial (TI) medium
induces purely transverse eddy currents, so it never samples the axial
conductivity sigma_N: its field equals that of an ISOTROPIC medium with
sigma = sigma_T, *independent of sigma_N*.

We build a UNIFORM medium uniaxial about a tilted axis
    n = (sin th, 0, cos th),   sigma = sigma_T I + (sigma_N - sigma_T) n n^T,
which carries a large off-diagonal entry sigma_xz = (sigma_N - sigma_T) sin th cos th
(exactly the kind of coupling the Fig-9 tilted layer produces), place a magnetic
dipole with moment along n, and solve with the fully coupled single solve.

Predictions if the coupled operator treats off-diagonal sigma correctly:
  * the field is INVARIANT to sigma_N  (matches the sigma_N = sigma_T case), and
  * it matches the isotropic sigma_T analytic (magnetic_dipole_B).

If instead the field drifts as sigma_N shrinks (off-diagonal grows), the coupled
off-diagonal discretization is the culprit for the Fig-9 over-attenuation.  If it
stays invariant, the operator is fine and the Fig-9 issue lives in how the
averaged tilted CELL is built, not in the solve.

Usage:  python examples/tilted_aniso_wholespace.py [Mcells]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import scipy.sparse.linalg as spla

from lebedev_em.grid import symmetric_uniform_grid, C000, C101, C110, C011
from lebedev_em.media import EMMedia, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    interpolate_cluster_B)
from lebedev_em.analytics import magnetic_dipole_B

OMEGA = 2 * np.pi * 52650.0
SIG_T = 0.10
THETA = np.radians(75.0)
NH = np.array([np.sin(THETA), 0.0, np.cos(THETA)])
CLUSTERS = [C000, C101, C110, C011]
CONTRASTS = [1.0, 10.0, 200.0]          # sigma_T / sigma_N
# observation points along the dipole axis n (avoid the source)
RADII = [0.6, 0.9, 1.2, 1.5]
OBS = [tuple(r * NH) for r in RADII]


def uniform_media(grid, sig):
    sigma_R = np.zeros((grid.N_R, 3, 3), dtype=complex)
    sigma_R[:] = sig
    return EMMedia(grid, sigma_R,
                   np.full(grid.N_P, complex(MU0)),
                   np.full(grid.N_R, complex(EPS0)))


def solve_B(grid, sig, moment):
    """Coupled solve for a magnetic dipole with the given moment vector
    (in the x-z plane); return the assembled B-field vector."""
    med = uniform_media(grid, sig)
    solver = LebedevMaxwellSolver(grid, med, OMEGA)
    r0 = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=0)
    r2 = build_rhs_per_cluster(grid, solver._C_PR, OMEGA, hx_comp=2)
    b = sum(moment[0] * r0[c] + moment[2] * r2[c] for c in CLUSTERS)
    A_bc, b_bc = apply_electric_bc(solver._A.copy(), b.copy(),
                                   _component_aware_bc_dofs(grid))
    A_bc = A_bc.tocsr()
    d = A_bc.diagonal()
    d_inv = np.where(np.abs(d) > 1e-30, 1.0 / d, 1.0)
    Mprec = spla.LinearOperator(A_bc.shape, matvec=lambda x: d_inv * x,
                                dtype=complex)
    E, info = spla.lgmres(A_bc, b_bc, M=Mprec, rtol=1e-9, atol=0,
                          maxiter=600, inner_m=30, outer_k=10)
    if info != 0:
        print(f"    [warn] lgmres info={info}", flush=True)
    return compute_B_from_E(grid, E, OMEGA), med.has_offdiagonal_sigma


def B_at(grid, B, p):
    """Lebedev-averaged B vector at point p."""
    x, y, z = p
    return np.array([
        np.mean([interpolate_cluster_B(grid, B, c, comp, x, y, z)
                 for c in CLUSTERS])
        for comp in range(3)])


def analytic_B(p, moment):
    """Isotropic sigma_T magnetic-dipole field for the given moment vector."""
    x, y, z = p
    return (moment[0] * magnetic_dipole_B(x, y, z, SIG_T, OMEGA, dipole_comp=0)
            + moment[2] * magnetic_dipole_B(x, y, z, SIG_T, OMEGA, dipole_comp=2))


def sig_of(cinv):
    sigN = SIG_T / cinv
    return SIG_T * np.eye(3) + (sigN - SIG_T) * np.outer(NH, NH)


def run_orientation(grid, moment, proj, label):
    """Solve for each sigma_N; report |B.proj| and the ratio to the isotropic
    (sigma_N=sigma_T) result at each obs point."""
    print(f"\n=== moment along {label}  m={np.round(moment,4)} ===")
    val = {}
    for cinv in CONTRASTS:
        sig = sig_of(cinv)
        B, has_off = solve_B(grid, sig, moment)
        f = np.array([B_at(grid, B, p) for p in OBS])
        val[cinv] = np.array([abs(np.dot(f[i], proj)) for i in range(len(OBS))])
    an = np.array([abs(np.dot(analytic_B(p, moment), proj)) for p in OBS])

    print(f"  {'r':>5} {'analytic':>11} " +
          " ".join(f"{'sT/'+str(int(c)):>11}" for c in CONTRASTS))
    for i, r in enumerate(RADII):
        print(f"  {r:>5.2f} {an[i]:>11.3e} " +
              " ".join(f"{val[c][i]:>11.3e}" for c in CONTRASTS))
    print("  ratio to isotropic FD (sigma_N=sigma_T):")
    for i, r in enumerate(RADII):
        print(f"    r={r:.2f}: " +
              "  ".join(f"sT/{int(c)}={val[c][i]/val[1.0][i]:.6f}"
                       for c in CONTRASTS))
    return val, an


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    grid = symmetric_uniform_grid(Mx=M, My=M, Mz=M, Lx=3.0, Ly=3.0, Lz=3.0)
    h = float(grid.x[1] - grid.x[0])
    print(f"uniform grid M={M} h={h:.4f} N_R={grid.N_R} 3N_R={3*grid.N_R}")
    print(f"tilt theta={np.degrees(THETA):.0f} deg  n={np.round(NH,4)}  "
          f"sigma_T={SIG_T}   sigma_xz at sT/200 = {sig_of(200)[0,2]:+.5f}")

    # (1) dipole ALONG n: physics -> field must be INVARIANT to sigma_N
    run_orientation(grid, NH.copy(), NH.copy(), "n (uniaxial axis)")

    # (2) CONTROL: dipole PERPENDICULAR to n (in x-z plane).  Physics -> the
    #     currents now sample sigma_N, so the field SHOULD depend on sigma_N.
    m_perp = np.array([np.cos(THETA), 0.0, -np.sin(THETA)])
    run_orientation(grid, m_perp, m_perp, "n_perp (control, should vary)")


if __name__ == "__main__":
    main()
