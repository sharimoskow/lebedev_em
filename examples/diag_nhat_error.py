"""
diag_nhat_error.py — Compare SVD-estimated n̂ (from_sigma_func "lookup")
against the analytical interface normal for each interface cell in the
DDH03 geometry at H=0.095, k=4.

Three interfaces:
  1. Borehole wall   r = R_BORE = 0.1 m   true n̂ = radial [x/r, y/r, 0]
  2. Invasion wall   r = R_INV  = 0.6 m   true n̂ = radial [x/r, y/r, 0]
  3. Dipping layer   n̂·x = D_PLANE        true n̂ = N_HAT

For each R-node whose dual cell straddles one of these interfaces we call
_estimate_normal_svd directly and record the angle between the SVD estimate
and the true normal.  We ignore "doubly-straddled" cells (more than one
interface) to keep comparisons clean.
"""
import sys, time
sys.path.insert(0, '../src')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lebedev_em.grid import symmetric_optimal_grid, hybrid_axial_grid
from lebedev_em.media import _estimate_normal_svd

# ── Geometry ──────────────────────────────────────────────────────────────────
FREQ=52650.; OMEGA=2*np.pi*FREQ
R_BORE=0.1; R_INV=0.6
DIP_RAD=1.04719755119660
N_HAT_DIPPING=np.array([np.sin(DIP_RAD),0.,np.cos(DIP_RAD)])
D_PLANE=N_HAT_DIPPING[2]*(-0.5)

SIGMA_BORE=0.05; SIGMA_INV=0.10; SIGMA_ISO=0.50
SIGMA_N=0.01; SIGMA_T=0.10
SIGMA_ANISO=SIGMA_T*np.eye(3)+(SIGMA_N-SIGMA_T)*np.outer(N_HAT_DIPPING,N_HAT_DIPPING)

H=0.095; GAMMA=1/2**0.5; H_SVD=0.025

def sigma_scalar(x,y,z):
    """Scalar proxy (trace/3) for the DDH03 conductivity tensor."""
    r=(x**2+y**2)**0.5
    if r<R_BORE: return SIGMA_BORE
    elif r<R_INV: return SIGMA_INV
    else:
        if N_HAT_DIPPING[0]*x+N_HAT_DIPPING[2]*z<D_PLANE:
            return float(np.trace(SIGMA_ANISO)/3.)
        return SIGMA_ISO

def sigma_block(XS,YS,ZS):
    """Vectorised scalar sigma on a meshgrid."""
    R=(XS**2+YS**2)**0.5
    side=N_HAT_DIPPING[0]*XS+N_HAT_DIPPING[2]*ZS
    out=np.full(XS.shape, SIGMA_ISO)
    out[R<R_BORE]=SIGMA_BORE
    out[(R>=R_BORE)&(R<R_INV)]=SIGMA_INV
    # anisotropic layer: use trace/3 as scalar proxy
    sigma_ani_scalar=float(np.trace(SIGMA_ANISO)/3.)
    out[(R>=R_INV)&(side<D_PLANE)]=sigma_ani_scalar
    return out

def which_interface(x,y,z,bmin,bmax):
    """
    Returns which interface(s) the dual cell [bmin,bmax] straddles.
    'bore'   : straddles r=R_BORE
    'inv'    : straddles r=R_INV
    'dip'    : straddles dipping layer plane
    Returns list of interface names; empty = uniform cell.
    """
    # Sample 8 corners for quick check
    ifaces=[]
    xs=[bmin[0],bmax[0]]; ys=[bmin[1],bmax[1]]; zs=[bmin[2],bmax[2]]
    r_corners=[(cx**2+cy**2)**0.5 for cx in xs for cy in ys]
    plane_corners=[N_HAT_DIPPING[0]*cx+N_HAT_DIPPING[2]*cz
                   for cx in xs for cy in ys for cz in zs]
    if min(r_corners)<R_BORE<max(r_corners): ifaces.append('bore')
    if min(r_corners)<R_INV<max(r_corners): ifaces.append('inv')
    if min(plane_corners)<D_PLANE<max(plane_corners): ifaces.append('dip')
    return ifaces

# ── Build grid ────────────────────────────────────────────────────────────────
t0=time.time()
z_fd=hybrid_axial_grid(-3.5,2.5,98,8,GAMMA)
grid=symmetric_optimal_grid(H,300.,z_fd,GAMMA,k=4)
print(f'Grid N_R={grid.N_R}  t={time.time()-t0:.1f}s')

x_fd,y_fd,z_fd_arr=grid.x,grid.y,grid.z
nx,ny,nz=len(x_fd),len(y_fd),len(z_fd_arr)

# ── Per-cell SVD diagnostic ───────────────────────────────────────────────────
single_records = []  # (iface, angle_deg, planarity, x, y, z)
dual_records   = []  # (iface_pair_str, angle_to_n1, angle_to_n2, planarity, x, y, z)

def run_svd(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi):
    _nx=max(3,int(np.ceil((x_hi-x_lo)/H_SVD))+1)
    _ny=max(3,int(np.ceil((y_hi-y_lo)/H_SVD))+1)
    _nz=max(3,int(np.ceil((z_hi-z_lo)/H_SVD))+1)
    xs=np.linspace(x_lo,x_hi,_nx); ys=np.linspace(y_lo,y_hi,_ny); zs=np.linspace(z_lo,z_hi,_nz)
    XS,YS,ZS=np.meshgrid(xs,ys,zs,indexing='ij')
    block=sigma_block(XS,YS,ZS)
    return _estimate_normal_svd(block,xs,ys,zs)

def angle_to(n_est, n_true):
    cos_a=min(abs(float(np.dot(n_est,n_true))),1.0)
    return np.degrees(np.arccos(cos_a))

for seq,(i,j,k) in enumerate(grid.R_nodes):
    x=float(x_fd[i]); y=float(y_fd[j]); z=float(z_fd_arr[k])
    x_lo=float(x_fd[max(i-1,0)]); x_hi=float(x_fd[min(i+1,nx-1)])
    y_lo=float(y_fd[max(j-1,0)]); y_hi=float(y_fd[min(j+1,ny-1)])
    z_lo=float(z_fd_arr[max(k-1,0)]); z_hi=float(z_fd_arr[min(k+1,nz-1)])
    bmin=np.array([x_lo,y_lo,z_lo]); bmax=np.array([x_hi,y_hi,z_hi])

    ifaces=which_interface(x,y,z,bmin,bmax)
    if len(ifaces)==0: continue   # uniform cell

    n_est,ratio=run_svd(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi)
    if n_est is None: continue

    r=max((x**2+y**2)**0.5,1e-12)
    n_radial=np.array([x/r,y/r,0.])

    if len(ifaces)==1:
        iface=ifaces[0]
        n_true=n_radial if iface in ('bore','inv') else N_HAT_DIPPING.copy()
        single_records.append((iface, angle_to(n_est,n_true), ratio, x, y, z))

    else:  # dual (2 interfaces)
        pair='_'.join(sorted(ifaces))
        # angle to each true normal separately
        n1=n_radial  # bore or inv normal (radial)
        n2=N_HAT_DIPPING.copy()  # dipping layer normal
        dual_records.append((pair, angle_to(n_est,n1), angle_to(n_est,n2), ratio, x, y, z))

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import defaultdict
by_iface=defaultdict(list)
for iface,angle,ratio,x,y,z in single_records:
    by_iface[iface].append((angle,ratio))
by_pair=defaultdict(list)
for pair,a1,a2,ratio,x,y,z in dual_records:
    by_pair[pair].append((a1,a2,ratio))

print(f'\n--- Singly-straddled cells: {len(single_records)} ---')
print(f'{"Interface":12s}  {"n_cells":>8}  {"mean":>8}  {"median":>8}  {"p95":>8}  {"max":>8}')
for iface in ['bore','inv','dip']:
    if iface not in by_iface: continue
    angles=np.array([a for a,r in by_iface[iface]])
    print(f'{iface:12s}  {len(angles):>8d}  {angles.mean():>7.2f}°  '
          f'{np.median(angles):>7.2f}°  {np.percentile(angles,95):>7.2f}°  {angles.max():>7.2f}°')

print(f'\n--- Dual-interface cells: {len(dual_records)} ---')
print(f'{"Pair":12s}  {"n_cells":>8}  {"med∠n1(radial)":>16}  {"med∠n2(dip)":>14}  {"med planarity":>14}')
for pair,entries in by_pair.items():
    a1s=np.array([a1 for a1,a2,r in entries])
    a2s=np.array([a2 for a1,a2,r in entries])
    rs =np.array([r  for a1,a2,r in entries])
    print(f'{pair:12s}  {len(entries):>8d}  {np.median(a1s):>14.2f}°  '
          f'{np.median(a2s):>13.2f}°  {np.median(rs):>14.3f}')

# ── Plot: 2 rows — single (top) and dual (bottom) ────────────────────────────
iface_labels={'bore':f'Borehole  r={R_BORE}m','inv':f'Invasion  r={R_INV}m','dip':'Dipping layer  60°'}
colors_map={'bore':'#1f77b4','inv':'#ff7f0e','dip':'#2ca02c'}

fig,axes=plt.subplots(2,3,figsize=(14,8))
fig.suptitle(f'SVD n̂ error: singly- vs dual-straddled cells  (H={H}m, h_svd={H_SVD}m, k=4)',
             fontsize=11)

# Row 0: singly-straddled
for ax,(iface,label) in zip(axes[0],iface_labels.items()):
    if iface not in by_iface: ax.set_visible(False); continue
    angles=np.array([a for a,r in by_iface[iface]])
    ax.hist(angles,bins=30,color=colors_map[iface],edgecolor='white',alpha=0.85)
    ax.axvline(np.median(angles),color='k',ls='--',lw=1.5,
               label=f'median {np.median(angles):.2f}°')
    ax.axvline(np.percentile(angles,95),color='k',ls=':',lw=1.2,
               label=f'p95 {np.percentile(angles,95):.2f}°')
    ax.set_title(f'Single: {label}\n(n={len(angles)})',fontsize=9)
    ax.set_xlabel('Angle error (°)',fontsize=9); ax.set_ylabel('Count',fontsize=9)
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25)

# Row 1: dual-straddled — one panel per pair, angle to each normal
dual_iface_cols = [('bore','inv'),('bore','dip'),('inv','dip')]
for ax,(i1,i2) in zip(axes[1],dual_iface_cols):
    pair='_'.join(sorted([i1,i2]))
    if pair not in by_pair: ax.set_visible(False); continue
    entries=by_pair[pair]
    a1s=np.array([a1 for a1,a2,r in entries])
    a2s=np.array([a2 for a1,a2,r in entries])
    bins=np.linspace(0,90,31)
    n1_label='radial' if i1 in ('bore','inv') else 'dip'
    n2_label='dip'    if i2=='dip'            else 'radial'
    ax.hist(a1s,bins=bins,color=colors_map[i1],alpha=0.6,label=f'∠ to {i1} normal')
    ax.hist(a2s,bins=bins,color=colors_map[i2],alpha=0.6,label=f'∠ to {i2} normal')
    ax.set_title(f'Dual: {i1}+{i2}  (n={len(entries)})',fontsize=9)
    ax.set_xlabel('Angle to true normal (°)',fontsize=9); ax.set_ylabel('Count',fontsize=9)
    ax.legend(fontsize=8); ax.grid(True,alpha=0.25)

plt.tight_layout()
out='diag_nhat_error.png'
plt.savefig(out,dpi=150)
print(f'\nPlot saved: {out}')
print(f'Total time: {time.time()-t0:.1f}s')
