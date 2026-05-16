"""
diag_subsvd2.py — Validate masked per-pair sub-SVD for dual-interface cells.

For each pair of distinct sigma values (s_lo, s_hi), we run SVD using only
voxels that are within tolerance of s_lo OR s_hi — excluding any third material.
This cleanly isolates each interface's normal.

Key geometry (dip_inv cells): svals = [0.07(aniso), 0.10(inv), 0.50(iso)]
  - Pair (0.07, 0.10): only aniso↔inv crossings → radial normal (inv wall)
  - Pair (0.10, 0.50): only inv↔iso crossings  → radial normal (inv wall)
  - Pair (0.07, 0.50): only aniso↔iso crossings → dip normal  (dip plane)
"""
import sys, time
sys.path.insert(0, '../src')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid

# ── Geometry ─────────────────────────────────────────────────────────────────
R_BORE=0.1; R_INV=0.6
DIP_RAD=1.04719755119660
N_HAT_DIPPING=np.array([np.sin(DIP_RAD),0.,np.cos(DIP_RAD)])
D_PLANE=N_HAT_DIPPING[2]*(-0.5)
SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50
SIGMA_N=0.01; SIGMA_T=0.10
SIGMA_ANI_S=float(np.trace(SIGMA_T*np.eye(3)+(SIGMA_N-SIGMA_T)*np.outer(N_HAT_DIPPING,N_HAT_DIPPING))/3.)
H=0.095; GAMMA=1/2**0.5; H_SVD=0.025

def sigma_block(XS,YS,ZS):
    R=(XS**2+YS**2)**0.5
    side=N_HAT_DIPPING[0]*XS+N_HAT_DIPPING[2]*ZS
    out=np.full(XS.shape, SIGMA_ISO)
    out[R<R_BORE]=SIGMA_BORE
    out[(R>=R_BORE)&(R<R_INV)]=SIGMA_INV
    out[(R>=R_INV)&(side<D_PLANE)]=SIGMA_ANI_S
    return out

def which_interface(bmin,bmax):
    ifaces=[]
    xs=[bmin[0],bmax[0]]; ys=[bmin[1],bmax[1]]; zs=[bmin[2],bmax[2]]
    r_corners=[(cx**2+cy**2)**0.5 for cx in xs for cy in ys]
    plane_corners=[N_HAT_DIPPING[0]*cx+N_HAT_DIPPING[2]*cz for cx in xs for cy in ys for cz in zs]
    if min(r_corners)<R_BORE<max(r_corners): ifaces.append('bore')
    if min(r_corners)<R_INV<max(r_corners): ifaces.append('inv')
    if min(plane_corners)<D_PLANE<max(plane_corners): ifaces.append('dip')
    return ifaces

def build_block(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi):
    nx=max(3,int(np.ceil((x_hi-x_lo)/H_SVD))+1)
    ny=max(3,int(np.ceil((y_hi-y_lo)/H_SVD))+1)
    nz=max(3,int(np.ceil((z_hi-z_lo)/H_SVD))+1)
    xs=np.linspace(x_lo,x_hi,nx); ys=np.linspace(y_lo,y_hi,ny); zs=np.linspace(z_lo,z_hi,nz)
    XS,YS,ZS=np.meshgrid(xs,ys,zs,indexing='ij')
    return sigma_block(XS,YS,ZS), xs, ys, zs

def _pair_tols(svals, i, j):
    """Adaptive tolerance for masked sub-SVD of the (svals[i], svals[j]) pair."""
    n = len(svals)
    # Nearest other sval above s_lo (i.e., first sval strictly between s_lo and s_hi)
    inner_above_lo = svals[i+1] if i+1 < j else None
    inner_below_hi = svals[j-1] if j-1 > i else None
    gap_lo = (inner_above_lo - svals[i]) if inner_above_lo is not None else (svals[j]-svals[i])
    gap_hi = (svals[j] - inner_below_hi) if inner_below_hi is not None else (svals[j]-svals[i])
    ext_below = (svals[i] - svals[i-1]) if i > 0   else (svals[j]-svals[i])
    ext_above = (svals[j+1] - svals[j]) if j+1 < n else (svals[j]-svals[i])
    tol_lo = 0.4 * min(gap_lo, ext_below)
    tol_hi = 0.4 * min(gap_hi, ext_above)
    return tol_lo, tol_hi

def masked_svd_pair(block, xs, ys, zs, s_lo, s_hi, tol_lo, tol_hi):
    """
    SVD interface estimator: only count crossings between voxels near s_lo or s_hi,
    ignoring any other materials in the block.
    """
    sig = np.real(block).astype(float)
    mask_lo = np.abs(sig - s_lo) <= tol_lo
    mask_hi = np.abs(sig - s_hi) <= tol_hi
    mask = mask_lo | mask_hi
    if not mask.any(): return None, 1.0

    # binary: s_lo→0, s_hi→1, other→-1 (excluded)
    binary = np.full(sig.shape, -1, dtype=int)
    binary[mask_lo] = 0
    binary[mask_hi] = 1

    XX,YY,ZZ = np.meshgrid(xs,ys,zs,indexing='ij')
    pos_list = []
    for ax in range(3):
        sl_a=[slice(None)]*3; sl_b=[slice(None)]*3
        sl_a[ax]=slice(None,-1); sl_b[ax]=slice(1,None)
        sl_a,sl_b=tuple(sl_a),tuple(sl_b)
        both_valid = (binary[sl_a]>=0) & (binary[sl_b]>=0)
        crossing   = (binary[sl_a] != binary[sl_b]) & both_valid
        if crossing.any():
            mid_X=0.5*(XX[sl_a][crossing]+XX[sl_b][crossing])
            mid_Y=0.5*(YY[sl_a][crossing]+YY[sl_b][crossing])
            mid_Z=0.5*(ZZ[sl_a][crossing]+ZZ[sl_b][crossing])
            pos_list.append(np.column_stack([mid_X,mid_Y,mid_Z]))

    if not pos_list: return None, 1.0
    P=np.vstack(pos_list)
    if len(P)<3: return None, 1.0
    P=np.unique(P,axis=0)
    if len(P)<3: return None, 1.0
    P_c=P-P.mean(axis=0)
    _,s,Vt=np.linalg.svd(P_c,full_matrices=False)
    if s[0]<1e-14: return None, 1.0
    return Vt[-1], float(s[-1]/s[0])

def angle_to(n_est,n_true):
    if n_est is None: return np.nan
    return np.degrees(np.arccos(min(abs(float(np.dot(n_est,n_true))),1.0)))

# ── Build grid ────────────────────────────────────────────────────────────────
t0=time.time()
z_fd=hybrid_axial_grid(-3.5,2.5,98,8,GAMMA)
grid=symmetric_optimal_grid(H,300.,z_fd,GAMMA,k=4)
print(f'Grid N_R={grid.N_R}  t={time.time()-t0:.1f}s')
x_fd,y_fd,z_fd_arr=grid.x,grid.y,grid.z
nx,ny,nz=len(x_fd),len(y_fd),len(z_fd_arr)

# ── Collect dip_inv dual cells with 3 distinct materials ─────────────────────
records_3 = []   # dip_inv cells with 3 materials
records_2 = []   # all dual cells with 2 materials

for seq,(i,j,k) in enumerate(grid.R_nodes):
    x=float(x_fd[i]); y=float(y_fd[j]); z=float(z_fd_arr[k])
    x_lo=float(x_fd[max(i-1,0)]); x_hi=float(x_fd[min(i+1,nx-1)])
    y_lo=float(y_fd[max(j-1,0)]); y_hi=float(y_fd[min(j+1,ny-1)])
    z_lo=float(z_fd_arr[max(k-1,0)]); z_hi=float(z_fd_arr[min(k+1,nz-1)])
    bmin=np.array([x_lo,y_lo,z_lo]); bmax=np.array([x_hi,y_hi,z_hi])
    ifaces=which_interface(bmin,bmax)
    if len(ifaces)<2: continue

    block,xs,ys,zs=build_block(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi)
    svals=np.sort(np.unique(np.round(block,4)))
    n_dist=len(svals)
    pair='_'.join(sorted(ifaces))
    r=max((x**2+y**2)**0.5,1e-12)
    n_radial=np.array([x/r,y/r,0.])

    if n_dist==3 and pair=='dip_inv':
        # Run all 3 masked sub-SVDs
        pairs_idx=[(0,1),(1,2),(0,2)]
        pnorms=[]; pratios=[]
        for pi,pj in pairs_idx:
            tl,th=_pair_tols(svals,pi,pj)
            n_hat_p,ratio_p=masked_svd_pair(block,xs,ys,zs,svals[pi],svals[pj],tl,th)
            pnorms.append(n_hat_p); pratios.append(ratio_p)
        records_3.append(dict(svals=svals, pnorms=pnorms, pratios=pratios,
                               n_radial=n_radial, x=x, y=y, z=z))
    elif n_dist==2:
        records_2.append(dict(pair=pair, svals=svals, x=x, y=y, z=z))

# ── Report for 3-material dip_inv cells ──────────────────────────────────────
print(f'\n=== 3-material dip_inv cells (n={len(records_3)}) ===')
print(f'svals[0]=aniso({SIGMA_ANI_S:.4f}), svals[1]=inv({SIGMA_INV}), svals[2]=iso({SIGMA_ISO})')
print(f'Expected: pairs (0,1) and (1,2) → radial; pair (0,2) → dip')

pair_labels=['(aniso,inv)','(inv,iso)','(aniso,iso)']
pair_expected=['radial','radial','dip']
for pidx,(plbl,pexp) in enumerate(zip(pair_labels,pair_expected)):
    ang_r=[]  # angle to radial
    ang_d=[]  # angle to dip
    n_none=0
    for rec in records_3:
        n_hat_p=rec['pnorms'][pidx]
        if n_hat_p is None: n_none+=1; continue
        ang_r.append(angle_to(n_hat_p,rec['n_radial']))
        ang_d.append(angle_to(n_hat_p,N_HAT_DIPPING))
    ang_r=np.array(ang_r); ang_d=np.array(ang_d)
    if len(ang_r)==0:
        print(f'  Pair {plbl} (expected {pexp}): ALL NONE ({n_none} None)')
        continue
    print(f'  Pair {plbl} (expected {pexp}, n={len(ang_r)}, none={n_none}):')
    print(f'    ∠radial: med={np.median(ang_r):.1f}°  p95={np.percentile(ang_r,95):.1f}°  max={ang_r.max():.1f}°')
    print(f'    ∠dip:    med={np.median(ang_d):.1f}°  p95={np.percentile(ang_d,95):.1f}°  max={ang_d.max():.1f}°')

# Planarity ratios
print(f'\n  Planarity ratios (small = planar, reliable):')
for pidx,plbl in enumerate(pair_labels):
    ratios=np.array([rec['pratios'][pidx] for rec in records_3 if rec['pnorms'][pidx] is not None])
    if len(ratios)==0: continue
    print(f'    {plbl}: med={np.median(ratios):.3f}  p95={np.percentile(ratios,95):.3f}')

# Check: are (0,1) and (1,2) parallel to each other?
print(f'\n  Parallel check: are (aniso,inv) and (inv,iso) normals parallel?')
dot_01_12=[]
for rec in records_3:
    n01,n12=rec['pnorms'][0],rec['pnorms'][1]
    if n01 is None or n12 is None: continue
    dot_01_12.append(abs(float(np.dot(n01,n12))))
dot_01_12=np.array(dot_01_12)
print(f'    |n01·n12| med={np.median(dot_01_12):.3f}  p5={np.percentile(dot_01_12,5):.3f}')

# Check: is (0,2) different from (0,1)?
print(f'\n  Perpendicularity check: are (aniso,inv) and (aniso,iso) normals different?')
dot_01_02=[]
for rec in records_3:
    n01,n02=rec['pnorms'][0],rec['pnorms'][2]
    if n01 is None or n02 is None: continue
    dot_01_02.append(abs(float(np.dot(n01,n02))))
dot_01_02=np.array(dot_01_02)
print(f'    |n01·n02| med={np.median(dot_01_02):.3f}  p95={np.percentile(dot_01_02,95):.3f}')

print(f'\nTotal time: {time.time()-t0:.1f}s')
