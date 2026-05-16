"""
plot_stratE_clean.py — Strategy E, k=3–6, no crossing lines, no DDH03 band.

Data files:
  k=3: hmin0166_k3_E.npz  (magic h_min = 0.166 m = R_INV/alpha)
  k=4: hmin010_k4_E.npz
  k=5: hmin010_k5_E.npz
  k=6: hmin010_k6_E.npz
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

base = os.path.dirname(os.path.abspath(__file__))

# ── Load data ─────────────────────────────────────────────────────────────────
files = {
    3: ('hmin0166_k3_E.npz', 'z_x', 'bxx', 'bxz'),
    4: ('hmin010_k4_E.npz',  'z_x', 'bxx', 'bxz'),
    5: ('hmin010_k5_E.npz',  'z_x', 'bxx', 'bxz'),
    6: ('hmin010_k6_E.npz',  'z_x', 'bxx', 'bxz'),
}

data = {}
for k, (fn, zkey, bxx_key, bxz_key) in files.items():
    path = os.path.join(base, fn)
    if not os.path.exists(path):
        print(f"Missing {fn} — skipping k={k}")
        continue
    d = np.load(path)
    data[k] = dict(z=d[zkey], bxx=d[bxx_key], bxz=d[bxz_key])
    print(f"k={k}: {len(d[zkey])} z-points, "
          f"z=[{d[zkey].min():.3f}, {d[zkey].max():.3f}]")

# ── Plot ──────────────────────────────────────────────────────────────────────
colors  = {3: '#1f77b4', 4: '#ff7f0e', 5: '#2ca02c', 6: '#d62728'}
markers = {3: '^',       4: 's',       5: '*',       6: 'o'}
msize   = {3: 5,         4: 5,         5: 7,         6: 5}
lw      = 1.6

fig, ax = plt.subplots(figsize=(7, 6))
ax.set_yscale("log")

for k, d in sorted(data.items()):
    col = colors[k]
    mk  = markers[k]
    ms  = msize[k]
    ax.plot(d['z'], d['bxx'],
            color=col, marker=mk, ms=ms, lw=lw, ls='-',
            markevery=4, label=f'k={k}  (Mx={4*k})')
    ax.plot(d['z'], d['bxz'],
            color=col, marker=mk, ms=ms, lw=lw, ls='--',
            markevery=4)

# Interface position
ax.axvline(-0.5, color='gray', lw=0.9, ls='-', alpha=0.4)

# Region labels
ax.text(-1.35, 0.75, "60° dipping anisotropic layer\n"
        r"$\sigma_T=0.1$, $\sigma_N=0.01$ S/m",
        fontsize=8.5, ha='center', color='navy')
ax.text(-0.22, 0.75, "Isotropic\n"r"$\sigma=0.5$ S/m",
        fontsize=8.5, ha='center', color='saddlebrown')

# Component labels at right edge (use k=5 for position reference)
d5 = data.get(5) or data[max(data)]
ax.text(d5['z'][-1] + 0.02, d5['bxx'][-1] * 1.3,
        r'$B_{xx}$', fontsize=11, va='center')
ax.text(d5['z'][-1] + 0.02, d5['bxz'][-1] * 0.75,
        r'$B_{xz}$', fontsize=11, va='center')

# Legend: k entries + line style legend
legend_k = [
    Line2D([0],[0], color=colors[k], lw=lw, marker=markers[k],
           ms=msize[k], label=f'k={k}  (Mx={4*k})')
    for k in sorted(data)
]
legend_style = [
    Line2D([0],[0], color='k', lw=lw, ls='-',  label=r'$\mathrm{Im}(B_{xx})$'),
    Line2D([0],[0], color='k', lw=lw, ls='--', label=r'$\mathrm{Im}(B_{xz})$'),
]
ax.legend(handles=legend_k + legend_style,
          fontsize=9, framealpha=0.9, loc='upper left')

ax.set_xlim(-1.75, 0.02)
ax.set_ylim(0.5, 200)
ax.set_xlabel("z  (m)", fontsize=12)
ax.set_ylabel(r"Im $B_x$  (nT)", fontsize=12)
ax.set_title(
    "DDH03 geometry — strategy E, k = 3–6\n"
    r"Im$(B_{xx})$ solid, Im$(B_{xz})$ dashed  $\cdot$  "
    "52.65 kHz x-directed magnetic dipole",
    fontsize=10)
ax.grid(True, which='both', alpha=0.22)

plt.tight_layout()
out = os.path.join(base, "ddh03_fig7_stratE_clean.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
