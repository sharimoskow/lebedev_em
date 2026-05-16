import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import scipy.sparse.linalg as spla
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import (EMMedia, MU0, EPS0,
    _nodal_eff_tensor_general,
    _volume_frac_layer1_planar,
    _frac_1d_layer1)
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, build_rhs_multicl, extract_B_on_axis_multicl

FREQ=52650.; OMEGA=2*3.14159265358979*FREQ
SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50
SIGMA_N=0.01; SIGMA_T=0.10; R_BORE=0.1; R_INV=0.6
DIP_RAD=1.04719755119660; N_HAT=np.array([np.sin(DIP_RAD),0.,np.cos(DIP_RAD)])
D_PLANE=N_HAT[0]*0.0+N_HAT[2]*(-0.5)
SIGMA_ANISO=SIGMA_T*np.eye(3)+(SIGMA_N-SIGMA_T)*np.outer(N_HAT,N_HAT)
z_fd=hybrid_axial_grid(-3.5,2.5,98,8,1.0/2**0.5)
GAMMA=1.0/2**0.5; Z_MIN,Z_MAX=-1.7,0.02

# Pre-compute isotropic identity tensors for convenience
_S_BORE = SIGMA_BORE * np.eye(3, dtype=complex)
_S_INV  = SIGMA_INV  * np.eye(3, dtype=complex)
_S_ISO  = SIGMA_ISO  * np.eye(3, dtype=complex)
_S_ANI  = SIGMA_ANISO.astype(complex)


def _pointwise_sigma(x, y, z):
    """Pointwise (no averaging) conductivity tensor at (x, y, z)."""
    r = (x**2 + y**2)**0.5
    if r < R_BORE:
        return _S_BORE.copy()
    elif r < R_INV:
        return _S_INV.copy()
    else:
        if N_HAT[0]*x + N_HAT[2]*z < D_PLANE:
            return _S_ANI.copy()
        return _S_ISO.copy()


def _outer_sigma(x, y, z):
    """Sigma outside r = R_INV at location (x, y, z): aniso or iso."""
    if N_HAT[0]*x + N_HAT[2]*z < D_PLANE:
        return _S_ANI.copy()
    return _S_ISO.copy()


def _line_frac(box_min, box_max, n_hat, d_plane, node):
    """
    Per-axis line fractions for a planar interface  n̂·x = d_plane.

    For each grid axis α the fraction of [box_min[α], box_max[α]] in
    region 1 (n̂·x < d_plane) is computed analytically using
    _frac_1d_layer1, with the other two coordinates fixed at the node.

    Returns ndarray (3,) in [0, 1].
    """
    n_dot_node = float(np.dot(n_hat, node))
    f = np.empty(3, dtype=float)
    for a in range(3):
        # d_rest = d_plane - Σ_{β≠a} n̂[β]*node[β] = d_plane - n̂·node + n̂[a]*node[a]
        d_rest = d_plane - n_dot_node + float(n_hat[a]) * float(node[a])
        f[a] = _frac_1d_layer1(box_min[a], box_max[a], float(n_hat[a]), d_rest)
    return f


def build_media(grid):
    """
    Build EMMedia for the DDH03 borehole geometry with nodal homogenization
    at every interface (borehole wall r=R_BORE, invasion zone r=R_INV, and
    dipping formation plane N_HAT·x = D_PLANE).

    For each dual cell that straddles an interface we use
    _nodal_eff_tensor_general (generalised Moskow et al. 1999 formula that
    handles anisotropic σ tensors) with a local planar approximation of the
    cylindrical interface.  This makes the medium representation k-independent
    for cells that straddle r=R_INV even though node positions vary with k.
    """
    nx, ny, nz = len(grid.x), len(grid.y), len(grid.z)
    sr = np.zeros((grid.N_R, 3, 3), dtype=complex)

    for seq, (i, j, kk) in enumerate(grid.R_nodes):
        x = float(grid.x[i])
        y = float(grid.y[j])
        z = float(grid.z[kk])
        r = (x**2 + y**2)**0.5
        node = np.array([x, y, z])

        # Dual-cell bounds (clamp to grid edges)
        bmin = np.array([float(grid.x[max(i-1, 0)]),
                         float(grid.y[max(j-1, 0)]),
                         float(grid.z[max(kk-1, 0)])])
        bmax = np.array([float(grid.x[min(i+1, nx-1)]),
                         float(grid.y[min(j+1, ny-1)]),
                         float(grid.z[min(kk+1, nz-1)])])

        # ---------------------------------------------------------------
        # Start with pointwise assignment; override if a straddle is found
        # ---------------------------------------------------------------
        sr[seq] = _pointwise_sigma(x, y, z)
        averaged = False

        # ---------------------------------------------------------------
        # 1.  Check r = R_BORE  (borehole wall: bore / invasion)
        # ---------------------------------------------------------------
        if r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])   # radial unit vector
            # Dot product of n_hat_r with each x-y corner of the cell
            xs = [bmin[0], bmax[0]]
            ys = [bmin[1], bmax[1]]
            corner_d = [n_hat_r[0]*cx + n_hat_r[1]*cy for cx in xs for cy in ys]
            d_min = min(corner_d); d_max = max(corner_d)

            if d_min < R_BORE <= d_max:
                f_vol  = _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_BORE)
                f_line = _line_frac(bmin, bmax, n_hat_r, R_BORE, node)
                sr[seq] = _nodal_eff_tensor_general(
                    _S_BORE, _S_INV, f_vol, f_line, n_hat_r)
                averaged = True

        # ---------------------------------------------------------------
        # 2.  Check r = R_INV  (invasion zone outer boundary)
        # ---------------------------------------------------------------
        if not averaged and r > 1e-10:
            n_hat_r = np.array([x/r, y/r, 0.0])
            xs = [bmin[0], bmax[0]]
            ys = [bmin[1], bmax[1]]
            corner_d = [n_hat_r[0]*cx + n_hat_r[1]*cy for cx in xs for cy in ys]
            d_min = min(corner_d); d_max = max(corner_d)

            if d_min < R_INV <= d_max:
                f_vol  = _volume_frac_layer1_planar(bmin, bmax, n_hat_r, R_INV)
                f_line = _line_frac(bmin, bmax, n_hat_r, R_INV, node)
                # Outer medium: use the approximate outer-side position to
                # determine whether anisotropic or isotropic formation.
                # Project outward half a cell width beyond R_INV.
                d_out = R_INV + 0.5 * (bmax[0] - bmin[0])  # rough outer offset
                x_out = x + (d_out - r) * n_hat_r[0] if r > 1e-10 else x
                y_out = y + (d_out - r) * n_hat_r[1] if r > 1e-10 else y
                sigma2 = _outer_sigma(x_out, y_out, z)
                sr[seq] = _nodal_eff_tensor_general(
                    _S_INV, sigma2, f_vol, f_line, n_hat_r)
                averaged = True

        # ---------------------------------------------------------------
        # 3.  Check dipping plane  N_HAT·x = D_PLANE  (outside R_INV)
        # ---------------------------------------------------------------
        if not averaged and r >= R_INV:
            # Corner values of N_HAT·x (only x and z components matter
            # since N_HAT[1] = 0 for the 60° dip configuration)
            xs = [bmin[0], bmax[0]]
            zs = [bmin[2], bmax[2]]
            corner_dp = [N_HAT[0]*cx + N_HAT[2]*cz for cx in xs for cz in zs]
            dp_min = min(corner_dp); dp_max = max(corner_dp)

            if dp_min < D_PLANE <= dp_max:
                f_vol  = _volume_frac_layer1_planar(bmin, bmax, N_HAT, D_PLANE)
                f_line = _line_frac(bmin, bmax, N_HAT, D_PLANE, node)
                # region 1: N_HAT·x < D_PLANE → anisotropic
                # region 2: N_HAT·x >= D_PLANE → isotropic
                sr[seq] = _nodal_eff_tensor_general(
                    _S_ANI, _S_ISO, f_vol, f_line, N_HAT)

    return EMMedia(grid, sr, np.full(grid.N_P, complex(MU0)), np.full(grid.N_R, complex(EPS0)))

H=0.10
k = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # pass k as first argument, e.g. python run_k6_mc_bg.py 5
t0=time.time()
grid=symmetric_optimal_grid(H,300.,z_fd,GAMMA,k=k)
x1=grid.x[grid.Mx//2+1]
print("k=%d h_min=%.2f x1=%.4fm domain=%.2fm N_R=%d" % (k,H,x1,grid.x[-1],grid.N_R),flush=True)
med=build_media(grid)
solver=LebedevMaxwellSolver(grid,med,OMEGA)
bc=_component_aware_bc_dofs(grid)
b=build_rhs_multicl(grid,solver._C_PR,OMEGA)
A_bc,b_bc=apply_electric_bc(solver._A.copy(),b,bc)
print("system shape=%s nnz=%d t=%.1fs" % (A_bc.shape,A_bc.nnz,time.time()-t0),flush=True)
E=spla.spsolve(A_bc,b_bc)
print("solve done t=%.1fs" % (time.time()-t0),flush=True)
B=compute_B_from_E(grid,E,OMEGA)
z_x,Bxx=extract_B_on_axis_multicl(grid,B,comp=0,axis='z')
z_z,Bxz=extract_B_on_axis_multicl(grid,B,comp=2,axis='z')
mx=(z_x>=Z_MIN)&(z_x<=Z_MAX); mz=(z_z>=Z_MIN)&(z_z<=Z_MAX)
bxx=np.imag(Bxx[mx])*1e9; bxz=np.interp(z_x[mx],z_z[mz],np.imag(Bxz[mz])*1e9)
diff=bxx-bxz; sc=np.where(np.diff(np.sign(diff)))[0]
if len(sc):
    zi=sc[0]; zc=z_x[mx][zi]-(diff[zi]/(diff[zi+1]-diff[zi]))*(z_x[mx][zi+1]-z_x[mx][zi])
    print("CROSSING at z=%.4f m (t=%.1fs)" % (zc, time.time()-t0),flush=True)
else:
    print("no crossing ratio=%.3f at z=-1.7m (t=%.1fs)" % (bxx[0]/bxz[0],time.time()-t0),flush=True)
tag = "hmin%03d_k%d_avg" % (round(H*100), k)
out=os.path.join(os.path.dirname(os.path.abspath(__file__)), tag + '.npz')
np.savez(out, z_x=z_x[mx],bxx=bxx,z_z=z_z[mz],bxz=np.imag(Bxz[mz])*1e9)
print("DONE saved to %s" % out,flush=True)
