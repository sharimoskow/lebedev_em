import sys,time,warnings,numpy as np, scipy.sparse.linalg as spla
warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid, C000,C101,C110,C011
from lebedev_em.media import from_geometry_exact, from_geometry_func
from lebedev_em.geometry import CylindricalBoundary, PlanarBoundary, GeometryStack
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, build_rhs_per_cluster, lebedev_B_on_z_axis

FREQ=52650.; OMEGA=2*np.pi*FREQ; CL=[C000,C101,C110,C011]
SB,SINV,SISO,SN,ST=0.05,0.10,0.50,0.01,0.10; RB,RI=0.1,0.6
NH=np.array([np.sin(np.radians(60)),0,np.cos(np.radians(60))]); DP=NH[2]*(-0.5)
SANI=ST*np.eye(3)+(SN-ST)*np.outer(NH,NH); GAMMA=1/2**0.5
def sf(X,Y,Z):
    X=np.asarray(X,float);Y=np.asarray(Y,float);Z=np.asarray(Z,float)
    o=np.zeros(np.broadcast(X,Y,Z).shape+(3,3),complex); r=np.hypot(X,Y); side=NH[0]*X+NH[2]*Z
    o[(r>=RI)&(side<DP)]=SANI; o[(r>=RI)&(side>=DP)]=SISO*np.eye(3)
    o[(r>=RB)&(r<RI)]=SINV*np.eye(3); o[r<RB]=SB*np.eye(3); return o
geo=GeometryStack([CylindricalBoundary(RB),CylindricalBoundary(RI),PlanarBoundary(NH,DP)])
paper_xx={-1.663:3.11,-1.541:3.66,-1.051:7.17,-0.929:8.46,-0.806:10.11,-0.684:12.21,-0.561:14.84,-0.194:32.47,-0.071:60.82}
paper_xz={-1.663:3.89,-1.541:4.20,-1.051:5.43,-0.929:5.67,-0.806:5.80,-0.684:5.97,-0.561:5.83,-0.439:5.58,-0.316:5.31,-0.194:4.70,-0.071:4.07}

z_fd=hybrid_axial_grid(-3.5,2.5,96,8,GAMMA); grid=symmetric_optimal_grid(0.05,300.,z_fd,GAMMA,k=6)
print(f"N_R={grid.N_R}",flush=True)
def solve(med):
    s=LebedevMaxwellSolver(grid,med,OMEGA); rhs=build_rhs_per_cluster(grid,s._C_PR,OMEGA,hx_comp=0)
    b=sum(rhs[c] for c in CL); A,bb=apply_electric_bc(s._A.copy(),b.copy(),_component_aware_bc_dofs(grid)); A=A.tocsr()
    dd=A.diagonal(); di=np.where(np.abs(dd)>1e-30,1.0/dd,1.0); M=spla.LinearOperator(A.shape,matvec=lambda x:di*x,dtype=complex)
    E,info=spla.lgmres(A,bb,M=M,rtol=1e-8,atol=0,maxiter=400,inner_m=30,outer_k=10); B=compute_B_from_E(grid,E,OMEGA)
    z,Bxx=lebedev_B_on_z_axis(grid,{c:B for c in CL},comp=0); _,Bxz=lebedev_B_on_z_axis(grid,{c:B for c in CL},comp=2)
    return np.asarray(z),np.asarray(Bxx).imag*1e9,np.asarray(Bxz).imag*1e9,info
def factor(z,B,paper):
    r=np.array([paper[zz]/B[int(np.argmin(np.abs(z-zz)))] for zz in paper]); return r.mean(),r.std()/r.mean()
for name,med in [("exact-BACKUS",from_geometry_exact(grid,sf,geo,method="backus",h_svd=0.03)),
                 ("nodal",from_geometry_func(grid,sf,geo.interface_func,h_svd=0.025,method="nodal"))]:
    t0=time.time(); z,Bxx,Bxz,info=solve(med)
    mxx,vxx=factor(z,Bxx,paper_xx); mxz,vxz=factor(z,Bxz,paper_xz)
    print(f"{name:13s} info={info} t={time.time()-t0:.0f}s | Bxx paper/ours mean={mxx:.2f} spread={100*vxx:.1f}% | Bxz mean={mxz:.2f} spread={100*vxz:.1f}%",flush=True)
