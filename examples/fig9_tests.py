"""Fig. 9 sensitivity tests: layer position +-5 cm; standard vs nodal averaging."""
import sys, os, time
import numpy as np
import scipy.sparse.linalg as spla
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import os as _os
OUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'out')
_os.makedirs(OUT_DIR, exist_ok=True)

from lebedev_em.grid import (symmetric_optimal_grid, hybrid_axial_grid,
                             C000, C101, C110, C011)
from lebedev_em.media import from_geometry_exact, from_sigma_func, MU0, EPS0
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver, _cluster_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    interpolate_cluster_B)

FREQ=52650.; OMEGA=2*np.pi*FREQ
SIG_BG=0.10; SIG_BORE=0.05; R_BORE=0.1
SIG_T=0.10; SIG_N=SIG_T/200.
DIP=np.radians(75.); N_HAT=np.array([np.sin(DIP),0.,np.cos(DIP)])
H_MIN=0.05; K_VAL=6; GAMMA=1/2**0.5
CLUSTERS=[C000,C101,C110,C011]

def model(dz_shift):
    d_top=N_HAT[2]*(-0.95+dz_shift); d_bot=N_HAT[2]*(-1.93+dz_shift)
    SIG_A=SIG_T*np.eye(3)+(SIG_N-SIG_T)*np.outer(N_HAT,N_HAT)
    def f(X,Y,Z):
        X=np.asarray(X,float);Y=np.asarray(Y,float);Z=np.asarray(Z,float)
        out=np.zeros(np.broadcast(X,Y,Z).shape+(3,3),dtype=complex)
        out[...]=SIG_BG*np.eye(3)
        side=N_HAT[0]*X+N_HAT[2]*Z
        out[(side<d_top)&(side>d_bot)]=SIG_A
        r=np.sqrt(X**2+Y**2)
        out[r<R_BORE]=SIG_BORE*np.eye(3)
        return out
    geo=GeometryStack([CylindricalBoundary(radius=R_BORE),
                       PlanarBoundary(n_hat=N_HAT,d=d_top),
                       PlanarBoundary(n_hat=N_HAT,d=d_bot)])
    return f, geo

variant=sys.argv[1]
z_fd=hybrid_axial_grid(-3.5,2.5,96,8,GAMMA)
grid=symmetric_optimal_grid(H_MIN,300.,z_fd,GAMMA,k=K_VAL)

if sys.argv[2]=='extract':
    x0,y0=float(grid.x[grid.Mx//2]),float(grid.y[grid.My//2])
    paper={-2.17:0.80,-1.93:0.95,-1.68:1.17,-1.43:1.47,-1.17:1.92,-0.95:2.63,
           -0.77:3.52,-0.66:4.25,-0.57:5.10,-0.47:6.42,-0.38:8.42,-0.30:11.15,-0.25:13.40}
    Bc={c:np.load(OUT_DIR + f'/fig9t_{variant}_{c}.npz')['B'] for c in CLUSTERS}
    print(f"[{variant}]  z    FD ImBz(nT)  paper  paper/FD")
    rats=[]
    for zz,p in paper.items():
        vals=[interpolate_cluster_B(grid,Bc[c],c,2,x0,y0,zz) for c in CLUSTERS]
        b=np.mean(vals).imag*1e9; rats.append(p/b)
        print(f"  {zz:6.2f} {b:9.3f} {p:7.2f} {p/b:7.3f}")
    print(f"  geo-mean {np.exp(np.mean(np.log(rats))):.3f} log-scatter {np.std(np.log(rats)):.3f}")
else:
    t0=time.time()
    if variant=='shiftup':  sf,geo=model(+0.05); med=from_geometry_exact(grid,sf,geo,method="nodal",h_svd=0.025)
    elif variant=='shiftdn':sf,geo=model(-0.05); med=from_geometry_exact(grid,sf,geo,method="nodal",h_svd=0.025)
    elif variant=='std':    sf,_=model(0.0);     med=from_sigma_func(grid,sf,h_svd=0.025,method='backus')
    elif variant=='nodalsf':sf,_=model(0.0);     med=from_sigma_func(grid,sf,h_svd=0.025,method='nodal')
    print(f'{variant}: media t={time.time()-t0:.0f}s',flush=True)
    solver=LebedevMaxwellSolver(grid,med,OMEGA)
    rhs=build_rhs_per_cluster(grid,solver._C_PR,OMEGA,hx_comp=2)
    for ix in [int(a) for a in sys.argv[2:]]:
        c=CLUSTERS[ix]
        bc=_cluster_bc_dofs(grid,c)
        A_bc,b_bc=apply_electric_bc(solver._A.copy(),rhs[c].copy(),bc)
        A_bc=A_bc.tocsr(); d=A_bc.diagonal()
        d_inv=np.where(np.abs(d)>1e-30,1.0/d,1.0)
        M=spla.LinearOperator(A_bc.shape,matvec=lambda x:d_inv*x,dtype=complex)
        E,info=spla.lgmres(A_bc,b_bc,M=M,rtol=1e-8,atol=0,maxiter=400,inner_m=30,outer_k=10)
        np.savez(OUT_DIR + f'/fig9t_{variant}_{c}.npz',B=compute_B_from_E(grid,E,OMEGA))
        print(f'{variant} cluster {c} info={info} t={time.time()-t0:.0f}s',flush=True)
