"""hmin010 k=3,4,5 convergence + strategy E overlay."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

base = os.path.dirname(os.path.abspath(__file__))

# ── Reference convergence data (strategy A) ───────────────────────────────────
refs = {}
for k in [3, 4, 5]:
    d  = np.load(os.path.join(base, f"hmin010_k{k}_avg.npz"))
    zz = d['z_z'] if 'z_z' in d else d['z_x']
    refs[k] = dict(zx=d['z_x'], bxx=d['bxx'], zz=zz, bxz=d['bxz'])

# Extend k=3 leftward with strategy-A data (same computation, wider z range)
d_abe = np.load(os.path.join(base, "media_compare_k3_ABE_H010.npz"))
mask  = d_abe['z_A'] < refs[3]['zx'][0]
refs[3]['zx']  = np.concatenate([d_abe['z_A'][mask],  refs[3]['zx']])
refs[3]['bxx'] = np.concatenate([d_abe['bxx_A'][mask], refs[3]['bxx']])
refs[3]['zz']  = refs[3]['zx']
refs[3]['bxz'] = np.concatenate([d_abe['bxz_A'][mask], refs[3]['bxz']])

# ── Strategy E data ────────────────────────────────────────────────────────────
strat_E = {}
# k=3 from media_compare_k3_ABE_H010
strat_E[3] = dict(z=d_abe['z_E'], bxx=d_abe['bxx_E'], bxz=d_abe['bxz_E'])
# k=4 from dedicated run
fn4 = os.path.join(base, 'hmin010_k4_E.npz')
if os.path.exists(fn4):
    d4 = np.load(fn4)
    strat_E[4] = dict(z=d4['z_x'], bxx=d4['bxx'], bxz=d4['bxz'])
# k=5 if available
fn5 = os.path.join(base, 'hmin010_k5_E.npz')
if os.path.exists(fn5):
    d5 = np.load(fn5)
    strat_E[5] = dict(z=d5['z_x'], bxx=d5['bxx'], bxz=d5['bxz'])

def find_crossing(z, bxx, bxz):
    d = bxx - bxz
    sc = np.where(np.diff(np.sign(d)))[0]
    if len(sc):
        zi = sc[0]
        return z[zi] - (d[zi]/(d[zi+1]-d[zi]))*(z[zi+1]-z[zi])
    return None

col_A  = {3: '#9E9E9E', 4: '#1976D2', 5: '#7B1FA2'}
col_E  = {3: '#FF8F00', 4: '#E65100', 5: '#BF360C'}   # amber/orange family
lw_A   = {3: 1.5,       4: 1.7,       5: 1.7}
lw_E   = 2.2

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_yscale("log")

# ── Strategy A reference curves ────────────────────────────────────────────────
for k, r in sorted(refs.items()):
    bxz_i = np.interp(r['zx'], r['zz'], r['bxz'])
    ax.plot(r['zx'], r['bxx'],  color=col_A[k], lw=lw_A[k], ls='-',  alpha=0.8)
    ax.plot(r['zx'], bxz_i,    color=col_A[k], lw=lw_A[k], ls='--', alpha=0.8)
    zc = find_crossing(r['zx'], r['bxx'], bxz_i)
    if zc:
        ax.axvline(zc, color=col_A[k], lw=0.8, ls=':', alpha=0.5)

# ── Strategy E curves ──────────────────────────────────────────────────────────
for k, s in sorted(strat_E.items()):
    ax.plot(s['z'], s['bxx'], color=col_E[k], lw=lw_E, ls='-')
    ax.plot(s['z'], s['bxz'], color=col_E[k], lw=lw_E, ls='--')
    zc = find_crossing(s['z'], s['bxx'], s['bxz'])
    if zc:
        ax.axvline(zc, color=col_E[k], lw=1.1, ls=':', alpha=0.75)

ax.axvline(-0.5, color='gray', lw=0.8, ls='-', alpha=0.35)

# Label families at right edge
r5 = refs[5]; bxz5 = np.interp(r5['zx'], r5['zz'], r5['bxz'])
ax.annotate(r"$B_{xx}$", xy=(r5['zx'][-1], r5['bxx'][-1]),
            xytext=(r5['zx'][-1]+0.05, r5['bxx'][-1]*1.25), fontsize=12)
ax.annotate(r"$B_{xz}$", xy=(r5['zx'][-1], bxz5[-1]),
            xytext=(r5['zx'][-1]+0.05, bxz5[-1]*0.78), fontsize=12)

ax.text(-1.55, 5.5, "60° dipping aniso layer\n"
        r"$\sigma_T=0.1$, $\sigma_N=0.01$ S/m", fontsize=8, ha='center')
ax.text(-0.22, 5.5, "Iso layer\n"r"$\sigma=0.5$ S/m", fontsize=8, ha='center')

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_elems = []
for k in [3, 4, 5]:
    r = refs[k]
    bxz_i = np.interp(r['zx'], r['zz'], r['bxz'])
    zc = find_crossing(r['zx'], r['bxx'], bxz_i)
    lbl = f"k={k}, strat A  ({zc:.3f} m)" if zc else f"k={k}, strat A"
    legend_elems.append(Line2D([0],[0], color=col_A[k], lw=lw_A[k], label=lbl))
for k, s in sorted(strat_E.items()):
    zc = find_crossing(s['z'], s['bxx'], s['bxz'])
    lbl = f"k={k}, strat E  ({zc:.3f} m)" if zc else f"k={k}, strat E"
    legend_elems.append(Line2D([0],[0], color=col_E[k], lw=lw_E, label=lbl))

ax.legend(handles=legend_elems, fontsize=9, framealpha=0.92, loc='upper left')
ax.set_xlim(-2.1, 0.02)
ax.set_ylim(0.5, 7)
ax.set_xlabel("z  (m)", fontsize=12)
ax.set_ylabel(r"Im $B_x$  (nT)", fontsize=12)
ax.set_title(
    "DDH03 Fig. 7 — strategy A (thin) vs strategy E (thick)  |  H = 0.10 m\n"
    r"Im$(B_{xx})$ solid, Im$(B_{xz})$ dashed  |  52.65 kHz, x-directed dipole",
    fontsize=10)
ax.grid(True, which='both', alpha=0.22)

plt.tight_layout()
out = os.path.join(base, "hmin010_fig7.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

# Summary table
print("\nCrossing positions:")
for k in [3,4,5]:
    r = refs[k]; bxz_i = np.interp(r['zx'], r['zz'], r['bxz'])
    zc = find_crossing(r['zx'], r['bxx'], bxz_i)
    print(f"  k={k} strat A: {zc:.4f} m" if zc else f"  k={k} strat A: (outside range)")
for k, s in sorted(strat_E.items()):
    zc = find_crossing(s['z'], s['bxx'], s['bxz'])
    print(f"  k={k} strat E: {zc:.4f} m" if zc else f"  k={k} strat E: no crossing")
print("  DDH03: ≈ −1.40 m")
