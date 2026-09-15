"""fig_ridge_density_check — the ridge-column anomaly against the pure
density structure.

Writes figures/fig_ridge_density_check.png from the column_profiles_cache
(run scripts/column_profiles_cache.py first).

Rows STD / WAL; columns: temperature difference, density difference, and
the stress anomaly test — the ridge anomaly shifted by |ΔP| (the untilted
column) against the density-route prediction g∫(ρ_R−ρ_I)dz with its deep
constant removed. Because the fluidity mesh is undeformed (topography is a
surface-stress field), the density field carries no elevation mass and
only the deep-corrected SHAPE of the lithostatic difference is comparable.

What it establishes (2026-09-15 session): the untilted ridge anomaly IS
the density structure (rms 2–3 MPa on a ~45 MPa signal); the thermal and
density contrasts persist to ~110–135 km, so ridge-side isostasy completes
only at the thermal thickness — the measured curve's shallower apparent
equilibration is set by where the isostatic tail equals |ΔP|, an amplitude
coincidence, not a material thickness.

DRAFT CAPTION. Ridge column relative to the first isostatic column:
temperature difference (left), density difference (centre), and the
vertical normal stress anomaly test (right) for STD (top) and WAL
(bottom), averaged over the mid-run window (36–44 Myr). In the right
panels the measured anomaly shifted by the deep pressure anomaly |ΔP|
(dashed) tracks the pure density prediction (dash-dotted) to within a few
MPa: with the tilt removed, the ridge column is isostatic cooling
structure, equilibrating only at the base of the thermal contrast.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc

def main():
    d = cpc.load()
    sm = lambda a: gaussian_filter1d(a, 2)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6), sharey=True)
    for row, k in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, k)
        zkm, m = c['z'] / 1e3, c['mid']
        Td = sm((d[f'{k}_temp_R'] - d[f'{k}_temp_I'])[m].mean(axis=0))
        rd = sm((d[f'{k}_rho_R'] - d[f'{k}_rho_I'])[m].mean(axis=0))
        ps = sm(c['p_R'][m].mean(axis=0)) / 1e6
        dP = c['dP'][m].mean() / 1e6
        lith_c = sm(c['lith_R'][m].mean(axis=0)) / 1e6
        ax = axes[row, 0]
        ax.plot(Td, zkm, 'k-', lw=1.5)
        ax.set_title(f'{k}: $T_R - T_I$ [K]', fontsize=10)
        ax = axes[row, 1]
        ax.plot(rd, zkm, 'k-', lw=1.5)
        ax.set_title(f'{k}: $\\rho_R - \\rho_I$ [kg m$^{{-3}}$]', fontsize=10)
        ax = axes[row, 2]
        ax.plot(ps, zkm, 'k-', lw=1.6, label='measured anomaly')
        ax.plot(ps - dP, zkm, 'k--', lw=1.3, label='shifted by $|\\Delta P|$ (untilted)')
        ax.plot(lith_c, zkm, color='0.45', lw=1.6, ls='-.',
                label='density prediction\n$g\\int(\\rho_R-\\rho_I)dz$ $-$ deep const.')
        ax.set_title(f'{k}: column anomaly [MPa]', fontsize=10)
        if row == 0:
            ax.legend(fontsize=7.5, frameon=False, loc='lower right')
        sel = (zkm > 5) & (zkm < 220)
        rms = np.sqrt(np.mean((ps[sel] - dP - lith_c[sel]) ** 2))
        print(f'{k}: untilted vs density prediction rms {rms:.2f} MPa '
              f'(signal ~{np.abs(ps[sel] - dP).max():.0f} MPa); dP = {dP:+.2f} MPa')
        for a in axes[row]:
            a.axvline(0, color='0.7', lw=0.7)
            a.axhline(cpc.ZC_KM, color='0.5', lw=0.6, ls=':')
    axes[0, 0].set_ylim(250, 0)
    axes[0, 0].set_ylabel('Depth [km]')
    axes[1, 0].set_ylabel('Depth [km]')
    fig.suptitle('Ridge vs first isostatic column: the untilted stress anomaly against the\n'
                 'pure density structure (mid-run average, 36–44 Myr; dotted: $z_c$)',
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures', 'fig_ridge_density_check.png')
    fig.savefig(out, dpi=200)
    print('written:', out)

if __name__ == '__main__':
    main()
