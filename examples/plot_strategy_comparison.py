"""
plot_strategy_comparison.py  —  DDH03 strategy / k-convergence comparison
H=0.10 m throughout.

Reference data: hmin010_k{3,4,5}_avg.npz  (from run_k6_mc_bg.py, strategy A)
  k=3 data starts at z=-1.663 m (Z_MIN=-1.7 clamp), crossing is at z≈-1.746 m
  → k=3 ref starts just past its own crossing; consistent with strategy A below.

Strategy data at k=3: media_compare_k3_ABE_H010.npz  (Z_MIN=-2.5, wider range)
  A: crossing -1.746 m   (identical to k=3 ref)
  B: crossing -1.845 m   (nodal homog with gradient n̂ at doubly-straddled cells)
  E: crossing -1.720 m   (Backus pre-homog of outer region, nodal n̂=r̂)
  DDH03 reference: ≈ -1.40 m
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

base = os.path.dirname(os.path.abspath(__file__))

# ── Load reference convergence data ───────────────────────────────────────────
refs = {}
for k in [3, 4, 5]:
    fn = os.path.join(base, f"hmin010_k{k}_avg.npz")
    if os.path.exists(fn):
        d = np.load(fn)
        zz = d['z_z'] if 'z_z' in d else d['z_x']
        refs[k] = dict(zx=d['z_x'], bxx=d['bxx'], zz=zz, bxz=d['bxz'])

# ── Load strategy curves ───────────────────────────────────────────────────────
d_abe = np.load(os.path.join(base, "media_compare_k3_ABE_H010.npz"))
strats = {
    'A': dict(z=d_abe['z_A'], bxx=d_abe['bxx_A'], bxz=d_abe['bxz_A']),
    'B': dict(z=d_abe['z_B'], bxx=d_abe['bxx_B'], bxz=d_abe['bxz_B']),
    'E': dict(z=d_abe['z_E'], bxx=d_abe['bxx_E'], bxz=d_abe['bxz_E']),
}

def find_crossing(z, bxx, bxz_on_z):
    diff = bxx - bxz_on_z
    sc   = np.where(np.diff(np.sign(diff)))[0]
    if len(sc):
        zi = sc[0]
        return z[zi] - (diff[zi]/(diff[zi+1]-diff[zi]))*(z[zi+1]-z[zi])
    return None

# Crossing summary
print("Crossing positions (m):")
for k, r in sorted(refs.items()):
    bxz_i = np.interp(r['zx'], r['zz'], r['bxz'])
    zc = find_crossing(r['zx'], r['bxx'], bxz_i)
    note = f"{zc:.4f}" if zc else f"<{r['zx'][0]:.3f} (outside saved range)"
    print(f"  k={k} ref (A): {note}")
for name, s in strats.items():
    zc = find_crossing(s['z'], s['bxx'], s['bxz'])
    print(f"  k=3 strat {name}: {zc:.4f}" if zc else f"  k=3 strat {name}: no crossing")
print("  DDH03: ≈ −1.40")

# ── Colours / styles ───────────────────────────────────────────────────────────
ref_col  = {3: '#9E9E9E', 4: '#1976D2', 5: '#7B1FA2'}
ref_lw   = {3: 1.4,       4: 1.6,       5: 1.6}
str_col  = {'B': '#D32F2F', 'E': '#388E3C'}
str_lw   = 2.2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    r"DDH03 Fig. 7 — Im$(B_{xx})$ and Im$(B_{xz})$ vs $z$, f = 52.65 kHz, H = 0.10 m"
    "\nSolid = $B_{xx}$,  dashed = $B_{xz}$  |  k-convergence (thin) + k=3 strategies (thick)",
    fontsize=10
)

crossing_info = {}   # for legend

for ax_idx, ax in enumerate(axes):
    ax.set_yscale("log")

    # ── Reference k-convergence ────────────────────────────────────────────────
    for k, r in sorted(refs.items()):
        bxz_i = np.interp(r['zx'], r['zz'], r['bxz'])
        ax.plot(r['zx'], r['bxx'],  color=ref_col[k], lw=ref_lw[k], ls='-',  alpha=0.85)
        ax.plot(r['zx'], bxz_i,    color=ref_col[k], lw=ref_lw[k], ls='--', alpha=0.85)
        zc = find_crossing(r['zx'], r['bxx'], bxz_i)
        if zc:
            ax.axvline(zc, color=ref_col[k], lw=0.9, ls=':', alpha=0.55)
        crossing_info[f'k={k}'] = zc

    # ── Strategy B and E (thick) ───────────────────────────────────────────────
    for name in ['B', 'E']:
        s = strats[name]
        ax.plot(s['z'], s['bxx'], color=str_col[name], lw=str_lw, ls='-')
        ax.plot(s['z'], s['bxz'], color=str_col[name], lw=str_lw, ls='--')
        zc = find_crossing(s['z'], s['bxx'], s['bxz'])
        if zc:
            ax.axvline(zc, color=str_col[name], lw=1.1, ls=':', alpha=0.8)
        crossing_info[f'strat {name}'] = zc

    # Also plot strategy A (thin, same colour as k=3 ref, extends further left)
    s = strats['A']
    mask_ext = s['z'] < refs[3]['zx'][0]   # only the part not covered by k=3 ref
    if mask_ext.any():
        ax.plot(s['z'][mask_ext], s['bxx'][mask_ext],
                color=ref_col[3], lw=ref_lw[3], ls='-',  alpha=0.85)
        ax.plot(s['z'][mask_ext], s['bxz'][mask_ext],
                color=ref_col[3], lw=ref_lw[3], ls='--', alpha=0.85)
    zc_A = find_crossing(s['z'], s['bxx'], s['bxz'])
    if zc_A:
        ax.axvline(zc_A, color=ref_col[3], lw=0.9, ls=':', alpha=0.55)
    crossing_info['k=3 (A)'] = zc_A

    # DDH03 reference crossing
    ax.axvline(-1.40, color='black', lw=1.8, ls='-', alpha=0.7)
    # Dipping-plane intercept
    ax.axvline(-0.5, color='gray', lw=0.7, ls='-', alpha=0.35)

    ax.set_xlabel("z  (m)", fontsize=12)
    ax.grid(True, which='both', alpha=0.2)

# ── Left panel: full window ────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-2.1, 0.02)
ax.set_ylim(0.5, 7)
ax.set_ylabel(r"Im $B_x$  (nT)", fontsize=12)

# Label the two curve families at right edge
r5 = refs[5]
bxz5_i = np.interp(r5['zx'], r5['zz'], r5['bxz'])
ax.annotate(r"$B_{xx}$", xy=(r5['zx'][-1], r5['bxx'][-1]),
            xytext=(r5['zx'][-1]+0.06, r5['bxx'][-1]*1.2),
            fontsize=11, arrowprops=dict(arrowstyle='-', lw=0.5))
ax.annotate(r"$B_{xz}$", xy=(r5['zx'][-1], bxz5_i[-1]),
            xytext=(r5['zx'][-1]+0.06, bxz5_i[-1]*0.80),
            fontsize=11, arrowprops=dict(arrowstyle='-', lw=0.5))

# Region labels
ax.text(-1.6, 6.0, "60° dipping\naniso layer", fontsize=8, ha='center', style='italic')
ax.text(-0.25, 6.0, "Iso\nlayer",                fontsize=8, ha='center', style='italic')

# Legend
zc3  = crossing_info.get('k=3 (A)', None)
zc4  = crossing_info.get('k=4',     None)
zc5  = crossing_info.get('k=5',     None)
zcB  = crossing_info.get('strat B', None)
zcE  = crossing_info.get('strat E', None)

legend_elems = [
    Line2D([0],[0], color=ref_col[3], lw=ref_lw[3], label=f"k=3, strat A  (cross {zc3:.3f} m)" if zc3 else "k=3, strat A"),
    Line2D([0],[0], color=ref_col[4], lw=ref_lw[4], label=f"k=4, strat A  (cross {zc4:.3f} m)" if zc4 else "k=4, strat A"),
    Line2D([0],[0], color=ref_col[5], lw=ref_lw[5], label=f"k=5, strat A  (cross {zc5:.3f} m)" if zc5 else "k=5, strat A"),
    Line2D([0],[0], color=str_col['B'], lw=str_lw, label=f"k=3, strat B  (cross {zcB:.3f} m)" if zcB else "k=3, strat B"),
    Line2D([0],[0], color=str_col['E'], lw=str_lw, label=f"k=3, strat E  (cross {zcE:.3f} m)" if zcE else "k=3, strat E"),
    Line2D([0],[0], color='black', lw=1.8, label="DDH03 ref crossing (≈ −1.40 m)"),
]
ax.legend(handles=legend_elems, fontsize=8.5, framealpha=0.92, loc='lower left')

# ── Right panel: crossing zoom ─────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(-2.05, -1.10)
ax.set_ylim(1.0, 3.5)
ax.set_ylabel(r"Im $B_x$  (nT)", fontsize=12)
ax.set_title("Crossing-region zoom", fontsize=10)

# Annotate crossings with vertical text
for label, zc, col in [
    ("A,k=3\n(−1.746)", zc3,  ref_col[3]),
    ("B,k=3\n(−1.845)", zcB,  str_col['B']),
    ("E,k=3\n(−1.720)", zcE,  str_col['E']),
    ("k=4,A\n(−1.300)", zc4,  ref_col[4]),
    ("k=5,A\n(−1.307)", zc5,  ref_col[5]),
]:
    if zc:
        ax.text(zc, 3.3, label, fontsize=7, ha='center', va='top', color=col,
                rotation=0, bbox=dict(fc='white', ec=col, lw=0.6, pad=1.5))

# DDH03 label
ax.text(-1.40, 1.08, "DDH03\n−1.40", fontsize=7.5, ha='center', va='bottom', color='black')

plt.tight_layout(rect=[0, 0, 1, 0.92])
outfile = os.path.join(base, "strategy_comparison_hmin010.png")
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outfile}")
