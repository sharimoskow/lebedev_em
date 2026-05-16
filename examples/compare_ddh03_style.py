"""
Plot Im(Bxx) and Im(Bxz) in DDH03 Fig. 7 style (log y scale, same z range)
to directly compare our results with DDH03.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(7, 6))

colors_A = {'k3': '#888888', 'k4': '#2244aa', 'k5': '#883399'}
colors_E = {'k3': '#dd7700', 'k4': '#cc4400', 'k5': '#882200'}
markers = {'k3': '^', 'k4': 's', 'k5': 'o'}

fn_abe = 'media_compare_k3_ABE_H010.npz'
d_abe = np.load(fn_abe) if os.path.exists(fn_abe) else None

# ── Strategy A curves ──────────────────────────────────────────────
for k in [3, 4, 5]:
    fn = f'hmin010_k{k}_avg.npz'
    if not os.path.exists(fn):
        print(f"Missing {fn}"); continue
    d = np.load(fn)
    zx = d['z_x']; bxx = d['bxx']
    zz = d.get('z_z', d['z_x']); bxz = d['bxz']

    # Extend k=3 leftward with ABE strategy A data
    if k == 3 and d_abe is not None:
        mask_left = d_abe['z_A'] < zx[0]
        zx  = np.concatenate([d_abe['z_A'][mask_left],  zx])
        bxx = np.concatenate([d_abe['bxx_A'][mask_left], bxx])
        zz  = np.concatenate([d_abe['z_A'][mask_left],  zz])
        bxz = np.concatenate([d_abe['bxz_A'][mask_left], bxz])

    ax.semilogy(zx, bxx, color=colors_A[f'k{k}'], marker=markers[f'k{k}'],
                ms=4, lw=1.0, label=f'k={k} strat A', zorder=3)
    ax.semilogy(zz, bxz, color=colors_A[f'k{k}'], marker=markers[f'k{k}'],
                ms=4, lw=1.0, ls='--', zorder=3)

    diff = bxx - bxz
    ci = np.where(np.diff(np.sign(diff)))[0]
    for c in ci:
        z_c = zx[c] + (zx[c+1]-zx[c])*(-diff[c]/(diff[c+1]-diff[c]))
        ax.axvline(z_c, color=colors_A[f'k{k}'], lw=0.6, ls=':', alpha=0.7)

# ── Strategy E curves ──────────────────────────────────────────────
# k=3: full range from ABE file
if d_abe is not None:
    ze = d_abe['z_E']; bxx_e = d_abe['bxx_E']; bxz_e = d_abe['bxz_E']
    ax.semilogy(ze, bxx_e, color=colors_E['k3'], marker='^',
                ms=4, lw=1.5, label='k=3 strat E', zorder=4)
    ax.semilogy(ze, bxz_e, color=colors_E['k3'], marker='^',
                ms=4, lw=1.5, ls='--', zorder=4)
    diff_e = bxx_e - bxz_e
    ci = np.where(np.diff(np.sign(diff_e)))[0]
    for c in ci:
        z_c = ze[c] + (ze[c+1]-ze[c])*(-diff_e[c]/(diff_e[c+1]-diff_e[c]))
        ax.axvline(z_c, color=colors_E['k3'], lw=1.0, ls='--', alpha=0.7)

# k=4, k=5: from dedicated E runs
for k, fn in [(4, 'hmin010_k4_E.npz'), (5, 'hmin010_k5_E.npz')]:
    if not os.path.exists(fn):
        print(f"Missing {fn}"); continue
    d = np.load(fn)
    zx_e = d['z_x']; bxx_e = d['bxx']; bxz_e = d['bxz']
    ax.semilogy(zx_e, bxx_e, color=colors_E[f'k{k}'], marker=markers[f'k{k}'],
                ms=4, lw=1.5, label=f'k={k} strat E', zorder=4)
    ax.semilogy(zx_e, bxz_e, color=colors_E[f'k{k}'], marker=markers[f'k{k}'],
                ms=4, lw=1.5, ls='--', zorder=4)
    diff_e = bxx_e - bxz_e
    ci = np.where(np.diff(np.sign(diff_e)))[0]
    for c in ci:
        z_c = zx_e[c] + (zx_e[c+1]-zx_e[c])*(-diff_e[c]/(diff_e[c+1]-diff_e[c]))
        ax.axvline(z_c, color=colors_E[f'k{k}'], lw=1.0, ls='--', alpha=0.7)

ax.set_xlabel('z (m)', fontsize=11)
ax.set_ylabel(r'Im $B_x$ ($\times 10^{-9}$ T)', fontsize=11)
ax.set_title('DDH03 Fig. 7 comparison — H=0.10m, f=52.65 kHz\n'
             'solid=Im(B$_{xx}$), dashed=Im(B$_{xz}$)', fontsize=10)
ax.set_xlim(-1.75, -0.05)
ax.set_ylim(1.0, 100.)
ax.grid(True, which='both', ls=':', alpha=0.4)

# Region labels
ax.axvline(-0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
ax.text(-1.3, 1.4, '60° dipping anisotropic\n'
        r'$\sigma_T$=0.1, $\sigma_N$=0.01 S/m', fontsize=8)
ax.text(-0.48, 1.4, 'Isotropic\n$\\sigma$=0.5 S/m', fontsize=8)

ax.legend(fontsize=7, loc='upper left', ncol=2)

# Add Bxx / Bxz labels on right
ax.text(-0.12, 22, r'$B_{xx}$', fontsize=10)
ax.text(-0.12, 2.7, r'$B_{xz}$', fontsize=10)

plt.tight_layout()
plt.savefig('ddh03_fig7_logstyle.png', dpi=150, bbox_inches='tight')
print("Saved ddh03_fig7_logstyle.png")
