"""Exact-geometry averaging core: clean data + bound-preserving Backus."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lebedev_em.grid import symmetric_uniform_grid
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
from lebedev_em.media import from_geometry_exact

NH = np.array([np.sin(np.radians(75.)), 0., np.cos(np.radians(75.))])
SIG_T, SIG_N = 0.1, 0.1/200.
SA = SIG_T*np.eye(3)+(SIG_N-SIG_T)*np.outer(NH, NH)

def _model():
    d0 = NH[2]*(-1.2)
    def sf(X,Y,Z):
        X=np.asarray(X,float);Y=np.asarray(Y,float);Z=np.asarray(Z,float)
        o=np.zeros(np.broadcast(X,Y,Z).shape+(3,3),complex); o[...]=0.1*np.eye(3)
        o[np.abs(NH[0]*X+NH[2]*Z-d0)<0.25/2]=SA
        o[np.hypot(X,Y)<0.1]=0.05*np.eye(3)     # borehole (3-material cells)
        return o
    geo=GeometryStack([CylindricalBoundary(0.1),
                       PlanarBoundary(NH, NH[2]*(-1.2)+NH[2]*0+0.125*1),  # placeholder
                       PlanarBoundary(NH, NH[2]*(-1.2)-0.125)])
    # exact layer faces: |NH·x - d0| < 0.125  -> planes at d0±0.125
    geo=GeometryStack([CylindricalBoundary(0.1),
                       PlanarBoundary(NH, d0+0.125),
                       PlanarBoundary(NH, d0-0.125)])
    return sf, geo

def test_backus_never_overshoots_on_real_geometry():
    grid=symmetric_uniform_grid(16,16,40,5.,5.,6.)
    sf,geo=_model()
    med=from_geometry_exact(grid, sf, geo, method="backus", h_svd=0.04)
    S=np.array(med.sigma_R)
    an=[s for s in range(len(S)) if np.abs(S[s]-np.diag(np.diag(S[s]))).max()>1e-9]
    assert an, "expected some anisotropic (interface) cells"
    mx=max(np.linalg.eigvals(S[s]).real.max() for s in an)
    assert mx <= 0.1*(1+1e-6), f"Backus overshoot: max eig {mx:.4f} > sigma_T"
    # blocking is actually present somewhere (layer resolved)
    snn=[float((NH@S[s]@NH).real) for s in an]
    assert min(snn) < 0.02

def test_pointwise_returns_node_material():
    grid=symmetric_uniform_grid(16,16,40,5.,5.,6.)
    sf,geo=_model()
    med=from_geometry_exact(grid, sf, geo, method="pointwise")
    S=np.array(med.sigma_R)
    # every node tensor equals sigma_func at that node
    bad=0
    for seq,(i,j,k) in enumerate(grid.R_nodes):
        exact=np.asarray(sf(np.array([[[grid.x[i]]]]),np.array([[[grid.y[j]]]]),
                            np.array([[[grid.z[k]]]])),complex).reshape(3,3)
        if np.abs(S[seq]-exact).max()>1e-9: bad+=1
    assert bad==0


def test_region_volume_is_exact():
    """Analytic dual-cell volumes must match closed forms, not sampled counts."""
    import numpy as np
    from lebedev_em.media import _region_volume
    from lebedev_em.geometry import PlanarBoundary, CylindricalBoundary
    bmin = np.array([-1., -1, -1]); bmax = np.array([1., 1, 1])
    # plane through origin -> exactly half the box
    P = PlanarBoundary([0, 0, 1], 0.0)
    assert abs(_region_volume(bmin, bmax, [(P, True)]) - 4.0) < 1e-9
    # cylinder R=0.5 -> pi*R^2 * height
    C = CylindricalBoundary(0.5)
    assert abs(_region_volume(bmin, bmax, [(C, True)]) - np.pi * 0.25 * 2) < 1e-6
    # unsupported (n_y != 0 plane) -> None (caller falls back)
    assert _region_volume(bmin, bmax, [(PlanarBoundary([0, 1, 0], 0.0), True)]) is None


def test_exact_nodal_axis_aligned_reduces_to_physical_diagonal():
    """Nodal via the exact core: for a grid-aligned layer (normal = ẑ) the
    tensor must reduce to the arith/harmonic DIAGONAL (notes' Lemma 1) — no
    off-diagonal, no overshoot."""
    import numpy as np
    from lebedev_em.grid import symmetric_uniform_grid
    from lebedev_em.geometry import PlanarBoundary, GeometryStack
    from lebedev_em.media import from_geometry_exact
    NZ = np.array([0., 0., 1.]); ST, SN = 0.1, 0.1 / 50
    SA = ST * np.eye(3) + (SN - ST) * np.outer(NZ, NZ)

    def sf(X, Y, Z):
        X = np.asarray(X, float); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        o = np.zeros(np.broadcast(X, Y, Z).shape + (3, 3), complex); o[...] = ST * np.eye(3)
        o[np.abs(Z + 1.0) < 0.25 / 2] = SA
        return o
    geo = GeometryStack([PlanarBoundary(NZ, -1.0 + 0.125), PlanarBoundary(NZ, -1.0 - 0.125)])
    grid = symmetric_uniform_grid(12, 12, 40, 4., 4., 6.)
    S = np.array(from_geometry_exact(grid, sf, geo, method="nodal", h_svd=0.05).sigma_R)
    # interface cells came out anisotropic in the DIAGONAL (an axis-aligned
    # layer produces NO off-diagonal, only σ_zz < σ_xx = σ_yy).
    diag = np.array([np.diag(S[s]).real for s in range(len(S))])
    an = [s for s in range(len(S)) if diag[s].max() - diag[s].min() > 1e-3]
    assert an, "expected anisotropic interface cells"
    for s in an:
        assert np.abs(S[s] - np.diag(np.diag(S[s]))).max() < 1e-6      # NO off-diagonal
        ev = np.linalg.eigvalsh(S[s].real)
        assert ev.max() <= ST * (1 + 1e-6) and ev.min() >= SN * (1 - 1e-3)  # physical (Lemma 1)
