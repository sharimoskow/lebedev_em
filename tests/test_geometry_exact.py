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
