import os,sys,time,warnings
import numpy as np, scipy.sparse.linalg as spla
warnings.filterwarnings("ignore")
sys.path.insert(0,"src")
OUT="examples/out"
from examples.fig9_backus_vs_paper import (build_grid, make_model, PAPER, OMEGA, CLUSTERS)
from lebedev_em.media import from_geometry_func
from lebedev_em.solver import LebedevMaxwellSolver, _component_aware_bc_dofs
from lebedev_em.operators import apply_electric_bc
from lebedev_em.postprocess import compute_B_from_E, build_rhs_per_cluster, lebedev_B_on_z_axis

d=np.load(os.path.join(OUT,"fig9_backus_vs_paper.npz"))
z=d["z"]; cached={k:d[k] for k in d.files}
grid=build_grid()
print(f"grid N_R={grid.N_R}",flush=True)
def solve(wl,method):
    t0=time.time(); sf,geo=make_model(wl)
    med=from_geometry_func(grid,sf,geo.interface_func,h_svd=0.025,method=method)
    solver=LebedevMaxwellSolver(grid,med,OMEGA)
    rhs=build_rhs_per_cluster(grid,solver._C_PR,OMEGA,hx_comp=2)
    b=sum(rhs[c] for c in CLUSTERS)
    A,bb=apply_electric_bc(solver._A.copy(),b.copy(),_component_aware_bc_dofs(grid)); A=A.tocsr()
    dd=A.diagonal(); di=np.where(np.abs(dd)>1e-30,1.0/dd,1.0)
    M=spla.LinearOperator(A.shape,matvec=lambda x:di*x,dtype=complex)
    E,info=spla.lgmres(A,bb,M=M,rtol=1e-8,atol=0,maxiter=400,inner_m=30,outer_k=10)
    B=compute_B_from_E(grid,E,OMEGA)
    _,Bz=lebedev_B_on_z_axis(grid,{c:B for c in CLUSTERS},comp=2)
    print(f"  nodal/{'layer' if wl else 'nolayer'} info={info} {time.time()-t0:.0f}s",flush=True)
    return np.asarray(Bz).imag*1e9
nod={'nolayer':solve(False,'nodal'),'layer':solve(True,'nodal')}
np.savez(os.path.join(OUT,"fig9_nodal_curves.npz"),z=z,nolayer=nod['nolayer'],layer=nod['layer'])

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
for a,tag,ttl in [(ax[0],"nolayer","No layer (σ_N = σ_T)"),(ax[1],"layer","Resistive layer (σ_N = σ_T/200)")]:
    zp=np.array(list(PAPER[tag].keys())); vp=np.array(list(PAPER[tag].values()))
    a.plot(zp,vp,"ks",ms=6,label="DDH03 (digitized)")
    a.plot(z,cached[f"pointwise_{tag}"],"-",color="tab:blue",label="pointwise")
    a.plot(z,cached[f"backus_{tag}"],"--",color="tab:red",label="anisotropic Backus")
    a.plot(z,nod[tag],"-.",color="tab:green",label="nodal")
    a.set_xlim(-2.3,-0.2); a.set_ylim(0,16); a.set_xlabel("z (m)"); a.set_title(ttl,fontsize=10); a.grid(alpha=0.3); a.legend(fontsize=8)
ax[0].set_ylabel(r"Im $B_z$ (nT)")
fig.suptitle("DDH03 Fig. 9 vs pointwise / anisotropic Backus / nodal (coupled, k=6)",fontsize=11)
fig.tight_layout(); png=os.path.join(OUT,"fig9_all_schemes_vs_paper.png"); fig.savefig(png,dpi=130)
print("figure ->",png)
# geo-mean ratios vs paper over the distinct digitized points (layer)
def ratio(curve):
    rs=[]
    for zz,pv in PAPER['layer'].items():
        rs.append(curve[int(np.argmin(np.abs(z-zz)))]/pv)
    rs=np.array(rs); return np.exp(np.mean(np.log(rs[rs>0])))
print(f"\nlayer geo-mean vs DDH03:  pointwise={ratio(cached['pointwise_layer']):.3f}  backus={ratio(cached['backus_layer']):.3f}  nodal={ratio(nod['layer']):.3f}")
