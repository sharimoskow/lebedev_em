"""
diag_subsvd.py — For each dual-interface cell in the DDH03 geometry,
run a *per-pair* sub-SVD (binary threshold between each adjacent material pair)
and compare the resulting normals to the analytical ones.

This tests whether the pair-wise sub-SVD approach gives clean normals that
can be used to drive multi-material homogenization in from_sigma_func.

For dual cells: we look at whether we can distinguish the two interface normals.
We also report coverage (how many dual cells show 3 distinct sigma values
vs only 2 at h_svd=0.025).
"""
import sys, time
sys.path.insert(0, '../src')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import _estimate_normal_svd

# ── Geometry ─────────────────────────────────────────────────────────────────
FREQ=52650.; OMEGA=2*np.pi*FREQ
R_BORE=0.1; R_INV=0.6
DIP_RAD=1.04719755119660
N_HAT_DIPPING=np.array([np.sin(DIP_RAD),0.,np.cos(DIP_RAD)])
D_PLANE=N_HAT_DIPPING[2]*(-0.5)

SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50
SIGMA_N=0.01; SIGMA_T=0.10
SIGMA_ANISO_SCALAR=float(np.trace(SIGMA_T*np.eye(3)+(SIGMA_N-SIGMA_T)*np.outer(N_HAT_DIPPING,N_HAT_DIPPING))/3.)

H=0.095; GAMMA=1/2**0.5; H_SVD=0.025

def sigma_block(XS,YS,ZS):
    R=(XS**2+YS**2)**0.5
    side=N_HAT_DIPPING[0]*XS+N_HAT_DIPPING[2]*ZS
    out=np.full(XS.shape, SIGMA_ISO)
    out[R<R_BORE]=SIGMA_BORE
    out[(R>=R_BORE)&(R<R_INV)]=SIGMA_INV
    out[(R>=R_INV)&(side<D_PLANE)]=SIGMA_ANISO_SCALAR
    return out

def which_interface(x,y,z,bmin,bmax):
    ifaces=[]
    xs=[bmin[0],bmax[0]]; ys=[bmin[1],bmax[1]]; zs=[bmin[2],bmax[2]]
    r_corners=[(cx**2+cy**2)**0.5 for cx in xs for cy in ys]
    plane_corners=[N_HAT_DIPPING[0]*cx+N_HAT_DIPPING[2]*cz
                   for cx in xs for cy in ys for cz in zs]
    if min(r_corners)<R_BORE<max(r_corners): ifaces.append('bore')
    if min(r_corners)<R_INV<max(r_corners): ifaces.append('inv')
    if min(plane_corners)<D_PLANE<max(plane_corners): ifaces.append('dip')
    return ifaces

def build_svd_block(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi):
    _nx=max(3,int(np.ceil((x_hi-x_lo)/H_SVD))+1)
    _ny=max(3,int(np.ceil((y_hi-y_lo)/H_SVD))+1)
    _nz=max(3,int(np.ceil((z_hi-z_lo)/H_SVD))+1)
    xs=np.linspace(x_lo,x_hi,_nx); ys=np.linspace(y_lo,y_hi,_ny); zs=np.linspace(z_lo,z_hi,_nz)
    XS,YS,ZS=np.meshgrid(xs,ys,zs,indexing='ij')
    block=sigma_block(XS,YS,ZS)
    return block, xs, ys, zs

def sub_svd(block, xs, ys, zs, s_lo, s_hi):
    """Binary threshold between s_lo and s_hi, run SVD."""
    mid = 0.5*(s_lo+s_hi)
    b2 = np.where(block<=mid, 0.0, 1.0)
    return _estimate_normal_svd(b2, xs, ys, zs)

def angle_to(n_est, n_true):
    if n_est is None: return np.nan
    cos_a=min(abs(float(np.dot(n_est,n_true))),1.0)
    return np.degrees(np.arccos(cos_a))

# ── Build grid ────────────────────────────────────────────────────────────────
t0=time.time()
z_fd=hybrid_axial_grid(-3.5,2.5,98,8,GAMMA)
grid=symmetric_optimal_grid(H,300.,z_fd,GAMMA,k=4)
print(f'Grid N_R={grid.N_R}  t={time.time()-t0:.1f}s')

x_fd,y_fd,z_fd_arr=grid.x,grid.y,grid.z
nx,ny,nz=len(x_fd),len(y_fd),len(z_fd_arr)

# ── Data collection ───────────────────────────────────────────────────────────
# For dual cells, track whether we can resolve 3 distinct materials,
# and how well sub-SVD recovers each interface normal.
#
# Sigma value ordering: bore < inv < aniso_scalar < iso
# SIGMA_BORE=0.05, SIGMA_INV=0.10, SIGMA_ANISO_SCALAR≈0.07, SIGMA_ISO=0.50
# Actually aniso < inv! Let's check:
print(f'SIGMA values: bore={SIGMA_BORE}, inv={SIGMA_INV}, aniso_scalar={SIGMA_ANISO_SCALAR:.4f}, iso={SIGMA_ISO}')
# Ordering: bore(0.05) < aniso_scalar(0.07) < inv(0.10) < iso(0.50)
# Sorted: 0.05, 0.07, 0.10, 0.50

dual_records = []
# Fields: pair, n_distinct, sub_svd_quality, ...

for seq,(i,j,k) in enumerate(grid.R_nodes):
    x=float(x_fd[i]); y=float(y_fd[j]); z=float(z_fd_arr[k])
    x_lo=float(x_fd[max(i-1,0)]); x_hi=float(x_fd[min(i+1,nx-1)])
    y_lo=float(y_fd[max(j-1,0)]); y_hi=float(y_fd[min(j+1,ny-1)])
    z_lo=float(z_fd_arr[max(k-1,0)]); z_hi=float(z_fd_arr[min(k+1,nz-1)])
    bmin=np.array([x_lo,y_lo,z_lo]); bmax=np.array([x_hi,y_hi,z_hi])

    ifaces=which_interface(x,y,z,bmin,bmax)
    if len(ifaces)<2: continue  # only care about dual cells

    block, xs, ys, zs = build_svd_block(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi)
    svals = np.unique(np.round(block, 4))
    n_distinct = len(svals)

    pair='_'.join(sorted(ifaces))
    r=max((x**2+y**2)**0.5,1e-12)
    n_radial=np.array([x/r,y/r,0.])

    # Sub-SVD for each adjacent sigma pair
    # Pairs: (svals[0], svals[1]), (svals[1], svals[2]), ...
    sub_normals = []
    sub_ratios  = []
    for idx in range(len(svals)-1):
        n_hat_sub, ratio_sub = sub_svd(block, xs, ys, zs, svals[idx], svals[idx+1])
        sub_normals.append(n_hat_sub)
        sub_ratios.append(ratio_sub if ratio_sub is not None else np.nan)

    # Angles to analytical normals: radial (bore/inv wall) and dipping layer
    # For 'bore_dip': pair 1 = bore wall (bore↔inv boundary), pair 2 = dip layer (inv side, but for bore+dip cell it's bore↔dip directly)
    # For 'inv_dip':  pair 1 = inv wall (inv↔aniso), pair 2 = dip/iso boundary (aniso↔iso)
    # We need angles of each sub-normal to both the radial and dipping normals.
    ang_sub_to_radial = [angle_to(n, n_radial)      for n in sub_normals]
    ang_sub_to_dip    = [angle_to(n, N_HAT_DIPPING)  for n in sub_normals]

    dual_records.append(dict(
        pair=pair, n_distinct=n_distinct,
        svals=svals.tolist(),
        sub_normals=sub_normals,
        sub_ratios=sub_ratios,
        ang_sub_to_radial=ang_sub_to_radial,
        ang_sub_to_dip=ang_sub_to_dip,
        x=x, y=y, z=z
    ))

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import defaultdict
by_pair = defaultdict(list)
for d in dual_records:
    by_pair[d['pair']].append(d)

print(f'\nTotal dual cells: {len(dual_records)}')
for pair, entries in sorted(by_pair.items()):
    n3 = sum(1 for e in entries if e['n_distinct']>=3)
    n2 = sum(1 for e in entries if e['n_distinct']==2)
    n1 = sum(1 for e in entries if e['n_distinct']==1)
    print(f'\n--- {pair}  (n={len(entries)}) ---')
    print(f'  n_distinct: 3={n3}, 2={n2}, 1={n1}')

    # For cells with 3 distinct values: sub-SVD angles
    entries3 = [e for e in entries if e['n_distinct']>=3]
    if entries3:
        # sub-SVD pair 0: lower boundary (bore wall or inv wall)
        # sub-SVD pair 1: upper boundary (dip or iso boundary)
        for pidx, plabel in enumerate(['lower pair', 'upper pair']):
            ang_r = np.array([e['ang_sub_to_radial'][pidx] for e in entries3
                              if pidx < len(e['ang_sub_to_radial']) and not np.isnan(e['ang_sub_to_radial'][pidx])])
            ang_d = np.array([e['ang_sub_to_dip'][pidx] for e in entries3
                              if pidx < len(e['ang_sub_to_dip']) and not np.isnan(e['ang_sub_to_dip'][pidx])])
            if len(ang_r)==0: continue
            print(f'  Sub-SVD {plabel} (n={len(ang_r)}):')
            print(f'    ∠radial: mean={ang_r.mean():.1f}°  med={np.median(ang_r):.1f}°  p95={np.percentile(ang_r,95):.1f}°')
            print(f'    ∠dip:    mean={ang_d.mean():.1f}°  med={np.median(ang_d):.1f}°  p95={np.percentile(ang_d,95):.1f}°')

    # For cells with only 2 distinct values: report what values were seen
    entries2 = [e for e in entries if e['n_distinct']==2]
    if entries2:
        val_counts = defaultdict(int)
        for e in entries2:
            val_counts[tuple(round(s,4) for s in e['svals'])] += 1
        print(f'  2-material cells: sigma pairs seen:')
        for vals, cnt in sorted(val_counts.items(), key=lambda x: -x[1]):
            ang_r0 = np.array([e['ang_sub_to_radial'][0] for e in entries2
                               if tuple(round(s,4) for s in e['svals'])==vals and not np.isnan(e['ang_sub_to_radial'][0])])
            ang_d0 = np.array([e['ang_sub_to_dip'][0] for e in entries2
                               if tuple(round(s,4) for s in e['svals'])==vals and not np.isnan(e['ang_sub_to_dip'][0])])
            print(f'    {vals}: n={cnt}  ∠radial med={np.median(ang_r0):.1f}°  ∠dip med={np.median(ang_d0):.1f}°')

print(f'\nTotal time: {time.time()-t0:.1f}s')
