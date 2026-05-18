"""Strategy E run using ILU-preconditioned LGMRES — faster for k=5."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import (EMMedia, MU0, EPS0,
    _nodal_eff_tensor_general, _multiregion_line_and_vol_fracs,
    _volume_frac_layer1_planar, _frac_1d_layer1)
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, build_rhs_multicl, extract_B_on_axis_multicl

FREQ=52650.; OMEGA=2*np.pi*FREQ
SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50
SIGMA_N=0.01; SIGMA_T=0.10; R_BORE=0.1; R_INV=0.6
DIP_RAD=1.04719755119660
N_HAT=np.array([np.sin(DIP_RAD),0.,np.cos(DIP_RAD)])
D_PLANE=N_HAT[0]*0.0+N_HAT[2]*(-0.5)
SIGMA_ANISO=SIGMA_T*np.eye(3)+(SIGMA_N-SIGMA_T)*np.outer(N_HAT,N_HAT)
z_fd=hybrid_axial_grid(-3.5,2.5,98,8,1.0/2**0.5)
GAMMA=1.0/2**0.5; Z_MIN,Z_MAX=-2.5,0.02; H=0.10

_S_BORE=SIGMA_BORE*np.eye(3,dtype=complex)
_S_INV =SIGMA_INV *np.eye(3,dtype=complex)
_S_ISO =SIGMA_ISO *np.eye(3,dtype=complex)
_S_ANI =SIGMA_ANISO.astype(complex)

def _pointwise_sigma(x,y,z):
    r=(x**2+y**2)**0.5
    if r<R_BORE: return _S_BORE.copy()
    elif r<R_INV: return _S_INV.copy()
    else: return _S_ANI.copy() if N_HAT[0]*x+N_HAT[2]*z<D_PLANE else _S_ISO.copy()

def _line_frac(bmin,bmax,n_hat,d_plane,node):
    n_dot_node=float(np.dot(n_hat,node)); f=np.empty(3,dtype=float)
    for a in range(3):
        d_rest=d_plane-n_dot_node+float(n_hat[a])*float(node[a])
        f[a]=_frac_1d_layer1(bmin[a],bmax[a],float(n_hat[a]),d_rest)
    return f

def _corner_vals(bmin,bmax,n_hat):
    return [n_hat[0]*cx+n_hat[1]*cy+n_hat[2]*cz
            for cx in [bmin[0],bmax[0]] for cy in [bmin[1],bmax[1]] for cz in [bmin[2],bmax[2]]]

def _backus_2region(sig1,sig2,f1,n_hat):
    f2=1.0-f1; n=n_hat/np.linalg.norm(n_hat)
    nn1=float(np.real(n@sig1@n)); nn2=float(np.real(n@sig2@n))
    nn_eff=1.0/(f1/nn1+f2/nn2)
    Tn1=sig1@n; Tn2=sig2@n
    avg=f1*Tn1/nn1+f2*Tn2/nn2
    S1=sig1-np.outer(Tn1,sig1.conj().T@n)/nn1
    S2=sig2-np.outer(Tn2,sig2.conj().T@n)/nn2
    return (f1*S1+f2*S2)+nn_eff*np.outer(avg,avg)

def build_media_E(grid):
    nx,ny,nz=len(grid.x),len(grid.y),len(grid.z)
    sr=np.zeros((grid.N_R,3,3),dtype=complex); n_double=0
    for seq,(i,j,kk) in enumerate(grid.R_nodes):
        x=float(grid.x[i]); y=float(grid.y[j]); z=float(grid.z[kk])
        r=(x**2+y**2)**0.5; node=np.array([x,y,z])
        bmin=np.array([float(grid.x[max(i-1,0)]),float(grid.y[max(j-1,0)]),float(grid.z[max(kk-1,0)])])
        bmax=np.array([float(grid.x[min(i+1,nx-1)]),float(grid.y[min(j+1,ny-1)]),float(grid.z[min(kk+1,nz-1)])])
        sr[seq]=_pointwise_sigma(x,y,z); averaged=False
        if r>1e-10:
            n_hat_r=np.array([x/r,y/r,0.0])
            cd=[n_hat_r[0]*cx+n_hat_r[1]*cy for cx in [bmin[0],bmax[0]] for cy in [bmin[1],bmax[1]]]
            if min(cd)<R_BORE<=max(cd):
                sr[seq]=_nodal_eff_tensor_general(_S_BORE,_S_INV,
                    _volume_frac_layer1_planar(bmin,bmax,n_hat_r,R_BORE),
                    _line_frac(bmin,bmax,n_hat_r,R_BORE,node),n_hat_r); averaged=True
        if not averaged and r>1e-10:
            n_hat_r=np.array([x/r,y/r,0.0])
            cd=[n_hat_r[0]*cx+n_hat_r[1]*cy for cx in [bmin[0],bmax[0]] for cy in [bmin[1],bmax[1]]]
            if min(cd)<R_INV<=max(cd):
                cdp=_corner_vals(bmin,bmax,N_HAT)
                if min(cdp)<D_PLANE<=max(cdp):
                    planes=[(n_hat_r,R_INV),(N_HAT,D_PLANE)]
                    lf,vf=_multiregion_line_and_vol_fracs(bmin,bmax,planes,node)
                    f_ani,f_iso=vf[1],vf[2]; f_outer=f_ani+f_iso
                    sigma_outer=_backus_2region(_S_ANI,_S_ISO,f_ani/f_outer,N_HAT) if f_outer>1e-12 else _S_ISO.copy()
                    vf_inv=_volume_frac_layer1_planar(bmin,bmax,n_hat_r,R_INV)
                    lf_inv=_line_frac(bmin,bmax,n_hat_r,R_INV,node)
                    sr[seq]=_nodal_eff_tensor_general(_S_INV,sigma_outer,vf_inv,lf_inv,n_hat_r); n_double+=1
                else:
                    d_out=R_INV+0.5*(bmax[0]-bmin[0])
                    x_out=x+(d_out-r)*n_hat_r[0]; y_out=y+(d_out-r)*n_hat_r[1]
                    sr[seq]=_nodal_eff_tensor_general(_S_INV,
                        _S_ANI.copy() if N_HAT[0]*x_out+N_HAT[2]*z<D_PLANE else _S_ISO.copy(),
                        _volume_frac_layer1_planar(bmin,bmax,n_hat_r,R_INV),
                        _line_frac(bmin,bmax,n_hat_r,R_INV,node),n_hat_r)
                averaged=True
        if not averaged and r>=R_INV:
            cdp=_corner_vals(bmin,bmax,N_HAT)
            if min(cdp)<D_PLANE<=max(cdp):
                sr[seq]=_nodal_eff_tensor_general(_S_ANI,_S_ISO,
                    _volume_frac_layer1_planar(bmin,bmax,N_HAT,D_PLANE),
                    _line_frac(bmin,bmax,N_HAT,D_PLANE,node),N_HAT)
    print(f"  Doubly-straddled cells: {n_double}",flush=True)
    return EMMedia(grid,sr,np.full(grid.N_P,complex(MU0)),np.full(grid.N_R,complex(EPS0)))

k=int(sys.argv[1]) if len(sys.argv)>1 else 5
t0=time.time()
print(f"=== Strategy E (iterative)  k={k}  H={H} ===",flush=True)
grid=symmetric_optimal_grid(H,300.,z_fd,GAMMA,k=k)
print(f"Grid N_R={grid.N_R}  t={time.time()-t0:.1f}s",flush=True)
med=build_media_E(grid)
print(f"Media built  t={time.time()-t0:.1f}s",flush=True)
solver=LebedevMaxwellSolver(grid,med,OMEGA)
bc=_component_aware_bc_dofs(grid)
b=build_rhs_multicl(grid,solver._C_PR,OMEGA)
A=solver._A.copy()
A_bc,b_bc=apply_electric_bc(A,b,bc)
print(f"System shape={A_bc.shape} nnz={A_bc.nnz}  t={time.time()-t0:.1f}s",flush=True)

# ILU preconditioner + LGMRES
print("Building ILU preconditioner...",flush=True)
tp=time.time()
ilu=spla.spilu(A_bc.tocsc(), drop_tol=1e-4, fill_factor=20)
M=spla.LinearOperator(A_bc.shape, ilu.solve)
print(f"ILU done  t={time.time()-tp:.1f}s  total={time.time()-t0:.1f}s",flush=True)

iters=[0]
def callback(r): iters[0]+=1
ts=time.time()
E,info=spla.lgmres(A_bc, b_bc, M=M, tol=1e-8, maxiter=200, inner_m=30, callback=callback)
print(f"LGMRES info={info} iters={iters[0]}  solve_t={time.time()-ts:.1f}s  total={time.time()-t0:.1f}s",flush=True)
if info!=0:
    print("LGMRES did not converge, falling back to direct...",flush=True)
    E=spla.spsolve(A_bc,b_bc)
    print(f"Direct solve done  t={time.time()-t0:.1f}s",flush=True)

B=compute_B_from_E(grid,E,OMEGA)
z_x,Bxx=extract_B_on_axis_multicl(grid,B,comp=0,axis='z')
z_z,Bxz=extract_B_on_axis_multicl(grid,B,comp=2,axis='z')
mx=(z_x>=Z_MIN)&(z_x<=Z_MAX); mz=(z_z>=Z_MIN)&(z_z<=Z_MAX)
bxx=np.imag(Bxx[mx])*1e9
bxz=np.interp(z_x[mx],z_z[mz],np.imag(Bxz[mz])*1e9)
diff=bxx-bxz; sc=np.where(np.diff(np.sign(diff)))[0]
if len(sc):
    zi=sc[0]; zc=z_x[mx][zi]-(diff[zi]/(diff[zi+1]-diff[zi]))*(z_x[mx][zi+1]-z_x[mx][zi])
    print(f"CROSSING at z={zc:.4f} m  t={time.time()-t0:.1f}s",flush=True)
else:
    print(f"No crossing  ratio={bxx[0]/bxz[0]:.3f}  t={time.time()-t0:.1f}s",flush=True)
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),f'hmin010_k{k}_E.npz')
np.savez(out,z_x=z_x[mx],bxx=bxx,z_z=z_x[mx],bxz=bxz)
print(f"Saved {out}",flush=True)
