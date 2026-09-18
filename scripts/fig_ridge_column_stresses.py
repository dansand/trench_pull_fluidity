"""fig_ridge_column_stresses — the two column stresses at the ridge.

Writes figures/fig_ridge_column_stresses.png and
tables/ridge_column_stresses.csv from the committed column-profile cache.

Only two parts of the stress field act on the vertical faces of a
column: the vertical normal stress, whose difference integrates to
Delta GPE*, and the normal-stress difference, whose integral is N_D.
(The third term in the balance, basal shear, acts on the horizontal
face.) This figure puts those two differences side by side for the same
column pair — ridge minus first isostatic column — so their depth
behaviour can be compared directly.

  (a) Delta sigma_zz. Positive through the lithosphere (the cooling
      topography), crossing zero near the base of the coherently
      translating plate, then holding a FINITE deep offset: the
      asthenospheric pressure gradient that tilts the plate.
  (b) Delta (sigma_xx - sigma_zz). Large in the lithosphere, where the
      ridge and the first isostatic column have very different thermal
      and rheological structure, and then DECAYING TO ZERO by about
      120 km — the asthenosphere has no shear strength with which to
      support a normal-stress difference.

The contrast is the point: below the plate one difference persists and
the other does not, which is why the deep part of Delta sigma_zz is a
flow pressure and not a strength signal.

The TRENCH column is deliberately excluded (Dan, 2026-09-18). Its
normal-stress difference carries the flexural fibre stress, hundreds of
MPa, which would swamp the comparison; the trench profiles have their
own figure.

Mid-run average (36-44 Myr, conventions 4b), models overlaid in the
brand colours per FIGURE_STYLE.md.

DRAFT CAPTION. The two column stresses that act on vertical faces,
differenced between the ridge and the first isostatic column and
averaged over the mid-run window, for STD (navy) and WAL (magenta).
(a) The vertical normal stress difference, whose integral is the
plate-wide driving term: positive through the plate, crossing zero near
its base, and holding a finite deep offset — the asthenospheric pressure
gradient. (b) The normal-stress difference, whose integral is N_D: large
within the lithosphere but decaying to zero by about 120 km, because the
asthenosphere cannot support a normal-stress difference. The deep part of
(a) is therefore a flow pressure, not a strength signal.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
ZERO_TOL_MPA = 1.0        # |value| below this counts as equilibrated

def main():
    d = cpc.load()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.4), sharey=True)
    rows = [('model', 'quantity', 'value')]
    for key in ('STD', 'WAL'):
        col = C[key]
        c = cpc.derive(d, key)
        m = c['mid']
        z, zkm = c['z'], c['z'] / 1e3
        sm = lambda a: gaussian_filter1d(a, 2)
        dzz = sm(c['p_R'][m].mean(axis=0)) / 1e6                      # pressure register
        nd = sm(((d[f'{key}_sxx_R'] - d[f'{key}_szz_R'])
                 - (d[f'{key}_sxx_I'] - d[f'{key}_szz_I']))[m].mean(axis=0)) / 1e6
        axes[0].plot(dzz, zkm, color=col, lw=2.0, label=key)
        axes[1].plot(nd, zkm, color=col, lw=2.0, label=key)
        # where each difference stops changing
        deep = (zkm >= 150) & (zkm <= 240)
        dP = dzz[deep].mean()
        below = np.where((zkm > 40) & (np.abs(nd) < ZERO_TOL_MPA))[0]
        z_nd0 = zkm[below[0]] if len(below) else np.nan
        s = np.where((dzz[:-1] > 0) & (dzz[1:] <= 0) & (zkm[:-1] > 20))[0]
        z_cross = zkm[s[0]] if len(s) else np.nan
        axes[0].axvline(dP, color=col, lw=1.0, ls='-.')
        axes[1].axhline(z_nd0, color=col, lw=1.0, ls=':')
        print(f'{key}: dSzz deep offset {dP:+.2f} MPa (sign change {z_cross:.0f} km); '
              f'd(Sxx-Szz) falls below {ZERO_TOL_MPA} MPa at {z_nd0:.0f} km, '
              f'deep value {nd[deep].mean():+.2f} MPa')
        rows += [(key, 'dSzz_deep_offset_MPa', f'{dP:.2f}'),
                 (key, 'dSzz_sign_change_km', f'{z_cross:.0f}'),
                 (key, 'dNSD_equilibration_depth_km', f'{z_nd0:.0f}'),
                 (key, 'dNSD_deep_value_MPa', f'{nd[deep].mean():.2f}'),
                 (key, 'dNSD_lithospheric_extreme_MPa', f'{nd[zkm < 60].min():.0f}')]
    axes[0].set_xlabel(r'(a) $\Delta\sigma_{zz}$, ridge $-\,x_I$ [MPa]'
                       '\n(dash-dot: deep offset)', fontsize=11)
    axes[1].set_xlabel(r'(b) $\Delta(\sigma_{xx}-\sigma_{zz})$, ridge $-\,x_I$ [MPa]'
                       '\n(dotted: equilibration depth)', fontsize=11)
    axes[0].set_ylabel('Depth [km]', fontsize=11)
    axes[0].set_ylim(250, 0)
    for ax in axes:
        ax.axvline(0, color='k', lw=1.6)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
    axes[0].legend(frameon=False, fontsize=10, loc='upper left')
    axes[0].text(0.03, 0.74, 'persists at depth:\nasthenospheric\npressure gradient',
                 transform=axes[0].transAxes, fontsize=8.5, color='0.3', va='top')
    axes[1].text(0.56, 0.74, 'equilibrates by ~100 km:\nno shear strength\nto support it',
                 transform=axes[1].transAxes, fontsize=8.5, color='0.3', va='top')
    fig.suptitle('The two column stresses acting on vertical faces, ridge minus first '
                 'isostatic column\n(mid-run average). One persists below the plate; '
                 'the other does not.', fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_ridge_column_stresses.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('ridge_column_stresses', rows[0], rows[1:],
                      script='fig_ridge_column_stresses.py',
                      figure='fig_ridge_column_stresses.png',
                      models=('STD', 'WAL'),
                      meta={'midrun_window_Myr': list(cpc.MIDRUN_MYR),
                            'deep_band_km': [150, 240],
                            'zero_tolerance_MPa': ZERO_TOL_MPA})
    print('written:', tab)

if __name__ == '__main__':
    main()
