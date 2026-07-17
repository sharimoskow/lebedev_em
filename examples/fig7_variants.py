"""Variant scan for the Fig. 7 model: which change yields flat x1.8 / x2.7?"""
import sys, os, time
import numpy as np
import scipy.sparse.linalg as spla
import warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from lebedev_em.grid import (symmetric_optimal_grid, hybrid_axial_grid,
                             C000, C101, C110, C011)
from lebedev_em.media import from_geometry_func, MU0, EPS0
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver, _cluster_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import (compute_B_from_E, build_rhs_per_cluster,
                                    lebedev_B_on_z_axis)

FREQ = 52650.; OMEGA = 2*np.pi*FREQ
DIP = np.radians(60.); N_HAT = np.array([np.sin(DIP),0.,np.cos(DIP)])
D_PLANE = N_HAT[2]*(-0.5)
SIG_ANISO = 0.10*np.eye(3) + (0.01-0.10)*np.outer(N_HAT,N_HAT)
H_MIN=0.05; K_VAL=6; GAMMA=1/2**0.5
CLUSTERS=[C000,C101,C110,C011]

def sigma_factory(variant):
    def f(X,Y,Z):
        X=np.asarray(X,float);Y=np.asarray(Y,float);Z=np.asarray(Z,float)
        out=np.zeros(np.broadcast(X,Y,Z).shape+(3,3),dtype=complex)
        side=N_HAT[0]*X+N_HAT[2]*Z
        if variant=='swap':   # anisotropic ABOVE the interface, isotropic 0.5 below
            out[side>=D_PLANE]=SIG_ANISO; out[side<D_PLANE]=0.5*np.eye(3)
        else:                 # standard assignment
            out[side<D_PLANE]=SIG_ANISO; out[side>=D_PLANE]=0.5*np.eye(3)
        r=np.sqrt(X**2+Y**2)
        if variant!='noinv': out[(r>=0.1)&(r<0.6)]=0.10*np.eye(3)
        out[r<0.1]=0.05*np.eye(3)
        return f2(out)
    def f2(o): return o
    return f

def geo_for(variant):
    b=[CylindricalBoundary(radius=0.1)]
    if variant!='noinv': b.append(CylindricalBoundary(radius=0.6))
    b.append(PlanarBoundary(n_hat=N_HAT, d=D_PLANE))
    return GeometryStack(b)

variant = sys.argv[1]
if sys.argv[2] == 'extract':
    z_fd = hybrid_axial_grid(-3.5,2.5,96,8,GAMMA)
    grid = symmetric_optimal_grid(H_MIN,300.,z_fd,GAMMA,k=K_VAL)
    Bc={c:np.load(f'/home/claude/fig7v_{variant}_{c}.npz')['B'] for c in CLUSTERS}
    z_bx,Bxx=lebedev_B_on_z_axis(grid,Bc,comp=0)
    z_bz,Bxz=lebedev_B_on_z_axis(grid,Bc,comp=2)
    z_bx=np.asarray(z_bx);Bxx=np.asarray(Bxx);Bxz=np.asarray(Bxz)
    np.savez(f'/home/claude/fig7v_{variant}_result.npz',z=z_bx,Bxx=Bxx,Bxz=Bxz)
    paper_xx={-1.663:3.11,-1.541:3.66,-1.051:7.17,-0.929:8.46,-0.806:10.11,-0.684:12.21,-0.561:14.84,-0.194:32.47,-0.071:60.82}
    paper_xz={-1.663:3.89,-1.541:4.20,-1.051:5.43,-0.929:5.67,-0.806:5.80,-0.684:5.97,-0.561:5.83,-0.439:5.58,-0.316:5.31,-0.194:4.70,-0.071:4.07}
    print(f"[{variant}]  z    ImBxx  paper ratio |  ImBxz  paper ratio")
    for zz in sorted(set(list(paper_xx)+list(paper_xz))):
        i=int(np.argmin(np.abs(z_bx-zz)))
        bx,bz=Bxx[i].imag*1e9,Bxz[i].imag*1e9
        px,pz=paper_xx.get(zz),paper_xz.get(zz)
        sx=f"{px:6.2f} {px/bx:5.2f}" if px else "  --    --"
        sz=f"{pz:6.2f} {pz/bz:5.2f}" if pz else "  --    --"
        print(f"{zz:7.3f} {bx:7.3f} {sx} | {bz:7.3f} {sz}")
else:
    idxs=[int(a) for a in sys.argv[2:]]
    z_fd=hybrid_axial_grid(-3.5,2.5,96,8,GAMMA)
    grid=symmetric_optimal_grid(H_MIN,300.,z_fd,GAMMA,k=K_VAL)
    med=from_geometry_func(grid,sigma_factory(variant),geo_for(variant).interface_func,h_svd=0.025)
    solver=LebedevMaxwellSolver(grid,med,OMEGA)
    rhs=build_rhs_per_cluster(grid,solver._C_PR,OMEGA,hx_comp=0)
    t0=time.time()
    for ix in idxs:
        c=CLUSTERS[ix]
        bc=_cluster_bc_dofs(grid,c)
        A_bc,b_bc=apply_electric_bc(solver._A.copy(),rhs[c].copy(),bc)
        A_bc=A_bc.tocsr(); d=A_bc.diagonal()
        d_inv=np.where(np.abs(d)>1e-30,1.0/d,1.0)
        M=spla.LinearOperator(A_bc.shape,matvec=lambda x:d_inv*x,dtype=complex)
        E,info=spla.lgmres(A_bc,b_bc,M=M,rtol=1e-8,atol=0,maxiter=400,inner_m=30,outer_k=10)
        B=compute_B_from_E(grid,E,OMEGA)
        np.savez(f'/home/claude/fig7v_{variant}_{c}.npz',B=B)
        print(f'{variant} cluster {c} info={info} t={time.time()-t0:.0f}s',flush=True)
