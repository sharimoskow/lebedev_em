"""Lookup geometry (sigma_func nodal), k=5, h_min=dz=6/98.
Solver: LGMRES with block-diagonal Jacobi preconditioner, rtol=1e-10.
"""
import sys, os, time
sys.path.insert(0, '/sessions/wizardly-intelligent-feynman/mnt/outputs/lebedev_em/src')
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import from_sigma_func, MU0, EPS0
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, build_rhs_multicl, extract_B_on_axis_multicl

LOGFILE = '/sessions/wizardly-intelligent-feynman/mnt/outputs/lebedev_em/examples/run_lookup_k5.log'

def log(msg):
    print(msg, flush=True)
    with open(LOGFILE, 'a') as f:
        f.write(msg + '\n')

# Clear log
with open(LOGFILE, 'w') as f:
    f.write('')

FREQ=52650.; OMEGA=2*np.pi*FREQ
SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50
SIGMA_N=0.01; SIGMA_T=0.10; R_BORE=0.1; R_INV=0.6
DIP_RAD=np.radians(60.0)
N_HAT=np.array([np.sin(DIP_RAD),0.,np.cos(DIP_RAD)])
D_PLANE=N_HAT[0]*0.0+N_HAT[2]*(-0.5)
SIGMA_ANISO=SIGMA_T*np.eye(3)+(SIGMA_N-SIGMA_T)*np.outer(N_HAT,N_HAT)

K_VAL  = 5
H_MIN  = 6.0/98      # ~0.06122 m
GAMMA  = 1.0/np.sqrt(2.0)
Z_MIN, Z_MAX = -2.5, 0.02

def sigma_func_ddh03(X, Y, Z):
    X,Y,Z = np.asarray(X,float),np.asarray(Y,float),np.asarray(Z,float)
    shape = np.broadcast(X,Y,Z).shape
    out = np.zeros(shape+(3,3), dtype=complex)
    r_xy = np.sqrt(X**2+Y**2)
    out[r_xy < R_BORE] = SIGMA_BORE*np.eye(3)
    out[(r_xy>=R_BORE)&(r_xy<R_INV)] = SIGMA_INV*np.eye(3)
    side = N_HAT[0]*X+N_HAT[2]*Z
    out[(r_xy>=R_INV)&(side<D_PLANE)]  = SIGMA_ANISO
    out[(r_xy>=R_INV)&(side>=D_PLANE)] = SIGMA_ISO*np.eye(3)
    return out

t0=time.time()
log(f"=== Lookup geometry  k={K_VAL}  H={H_MIN:.5f} m  (LGMRES Jacobi rtol=1e-10) ===")
z_fd=hybrid_axial_grid(-3.5,2.5,98,8,GAMMA)
grid=symmetric_optimal_grid(H_MIN,300.,z_fd,GAMMA,k=K_VAL)
log(f"Grid N_R={grid.N_R}  Mx={grid.Mx}  t={time.time()-t0:.1f}s")

log("Building media (from_sigma_func, nodal) ...")
t1=time.time()
med=from_sigma_func(grid,sigma_func_ddh03,h_svd=0.025,method="nodal",svd_isotropy_tol=0.7)
log(f"  media: {time.time()-t1:.1f}s")

solver=LebedevMaxwellSolver(grid,med,OMEGA)
b=build_rhs_multicl(grid,solver._C_PR,OMEGA)
bc=_component_aware_bc_dofs(grid)
A_bc,b_bc=apply_electric_bc(solver._A.copy(),b,bc)
log(f"System {A_bc.shape} nnz={A_bc.nnz}  t={time.time()-t0:.1f}s")

# Block-diagonal Jacobi preconditioner: use block size 3 (E_r, E_phi, E_z per node)
log("Building block-diagonal Jacobi preconditioner (block size 3) ...")
n = A_bc.shape[0]
assert n % 3 == 0, f"DOF count {n} not divisible by 3"
nblocks = n // 3
# Extract 3x3 diagonal blocks
diag_blocks = []
A_csr = A_bc.tocsr()
for i in range(nblocks):
    s = slice(3*i, 3*i+3)
    blk = A_csr[s, s].toarray()
    try:
        diag_blocks.append(np.linalg.inv(blk))
    except np.linalg.LinAlgError:
        diag_blocks.append(np.eye(3, dtype=complex))
log(f"  Preconditioner built  t={time.time()-t0:.1f}s")

# Build full block-diagonal preconditioner as a LinearOperator
P_data = np.zeros((n, 3), dtype=complex)
P_inv_blocks = np.stack(diag_blocks, axis=0)  # (nblocks, 3, 3)

def matvec_prec(x):
    x = x.reshape(nblocks, 3)
    # Apply each 3x3 block: result[i] = P_inv_blocks[i] @ x[i]
    return np.einsum('ijk,ik->ij', P_inv_blocks, x).ravel()

M = spla.LinearOperator((n, n), matvec=matvec_prec, dtype=complex)

# LGMRES solve
iters = [0]
residuals = []
def callback(r):
    iters[0] += 1
    res = np.linalg.norm(r)
    residuals.append(res)
    if iters[0] % 10 == 0:
        log(f"  LGMRES iter {iters[0]}  res={res:.3e}  t={time.time()-t0:.1f}s")

log("Solving with LGMRES rtol=1e-10 ...")
t_solve = time.time()
E, info = spla.lgmres(A_bc, b_bc, M=M, rtol=1e-10, atol=0,
                      maxiter=500, inner_m=30, outer_k=10,
                      callback=callback)
log(f"LGMRES done: info={info}  iters={iters[0]}  t_solve={time.time()-t_solve:.1f}s  t_total={time.time()-t0:.1f}s")
if info != 0:
    log(f"WARNING: LGMRES did not converge (info={info})")

B=compute_B_from_E(grid,E,OMEGA)
z_x,Bxx=extract_B_on_axis_multicl(grid,B,comp=0,axis='z')
z_z,Bxz=extract_B_on_axis_multicl(grid,B,comp=2,axis='z')
mx=(z_x>=Z_MIN)&(z_x<=Z_MAX); mz=(z_z>=Z_MIN)&(z_z<=Z_MAX)
bxx=np.imag(Bxx[mx])*1e9; bxz=np.interp(z_x[mx],z_z[mz],np.imag(Bxz[mz])*1e9)
diff=bxx-bxz; sc=np.where(np.diff(np.sign(diff)))[0]
if len(sc):
    zi=sc[0]; zc=z_x[mx][zi]-(diff[zi]/(diff[zi+1]-diff[zi]))*(z_x[mx][zi+1]-z_x[mx][zi])
    log(f"CROSSING z={zc:.4f} m  t={time.time()-t0:.1f}s")
else:
    log(f"No crossing found  t={time.time()-t0:.1f}s")

out='/sessions/wizardly-intelligent-feynman/mnt/outputs/lebedev_em/examples/hmin_dz_k5_lookup.npz'
np.savez(out,z_x=z_x[mx],bxx=bxx,z_z=z_x[mx],bxz=bxz)
log(f"Saved {out}")
