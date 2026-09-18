"""fig_ridge_column_stresses — the two column stresses at the ridge.

Writes figures/fig_ridge_column_stresses.png and
tables/ridge_column_stresses.csv from the committed column-profile cache.

This is the decomposition of the total column force across the
ISOSTATIC DOMAIN (x_I to the ridge, in the sense of Figure 1); the
non-isostatic domain (trench to x_I) has its own figures.

Only two parts of the stress field act on the vertical faces of a
column: the vertical normal stress, whose difference integrates to
Delta GPE*, and the normal-stress difference, whose integral is N_D.
Together they are the total resultant, Delta sigma_xx-bar = Delta N_D +
Delta sigma_zz-bar. (The third term in the balance, basal shear, acts on
the horizontal face.) This figure puts those two differences side by
side for the same column pair, so their depth behaviour can be compared
directly.

WHAT IT ESTABLISHES. All of the net force across the isostatic domain
arises ABOVE the depth at which Delta sigma_zz changes sign — that is,
within the convective thermal boundary layer. Below it, Delta N_D plays
essentially no role (it moves by 0.06 TN/m between 100 and 240 km,
because the asthenosphere has no shear strength to support a
normal-stress difference), while Delta sigma_zz continues to act, as the
asthenospheric pressure gradient. The sign change is therefore not only
where the forcing reverses but the base of the region that generates net
force at all, which is what one would expect.

  (a) Delta sigma_zz. Positive through the lithosphere (the cooling
      topography), crossing zero near the base of the coherently
      translating plate, then holding a FINITE deep offset: the
      asthenospheric pressure gradient that tilts the plate.
  (c) The CUMULATIVE contribution of each to the driving force, signed
      so that positive drives the plate trench-ward. This is where the
      boundary layer and the asthenosphere separate: below about 100 km
      the N_D term stops changing (it moves by 0.06 TN/m between 100 and
      240 km), while the sigma_zz term keeps declining as the adverse
      gradient eats into it. The asthenosphere removes driving force
      through sigma_zz and supplies none through N_D.

  (b) Delta (sigma_xx - sigma_zz). Large in the lithosphere, where the
      ridge and the first isostatic column have very different thermal
      and rheological structure, and then DECAYING TO ZERO by about
      just below the Delta sigma_zz sign change — the asthenosphere has
      no shear strength with which to support a normal-stress difference.

The dotted horizontal line on ALL panels is the depth at which
Delta sigma_zz changes sign (74 km STD / 84 km WAL at mid-run). It is the
figure's single reference level, so the three panels can be read against
one another.

The contrast is the point: below the plate one difference persists and
the other does not, which is why the deep part of Delta sigma_zz is a
flow pressure and not a strength signal.

The TRENCH column is deliberately excluded (Dan, 2026-09-18). Its
normal-stress difference carries the flexural fibre stress, hundreds of
MPa, which would swamp the comparison; the trench profiles have their
own figure.

Mid-run average (36-44 Myr, conventions 4b), models overlaid in the
brand colours per FIGURE_STYLE.md.

DRAFT CAPTION. Decomposition of the total column force across the
isostatic domain: the two stresses that act on vertical faces,
differenced between the ridge and the first isostatic column and
averaged over the mid-run window, for STD (navy) and WAL (magenta).
(a) The vertical normal stress difference, whose integral is the
plate-wide driving term: positive through the plate, crossing zero near
its base, and holding a finite deep offset — the asthenospheric pressure
gradient. (b) The normal-stress difference, whose integral is N_D: large
within the lithosphere but decaying to zero just below that level,
because the asthenosphere cannot support a normal-stress difference. The deep part of
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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4), sharey=True)
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
        s_x = np.where((dzz[:-1] > 0) & (dzz[1:] <= 0) & (zkm[:-1] > 20))[0]
        z_cross = zkm[s_x[0]] if len(s_x) else np.nan
        # (c) cumulative contributions, SIGNED so positive drives the plate
        # trench-ward. The net force on the segment in +x is
        # d(sigma_xx-bar) = dN_D + d(sigma_zz-bar); driving is -x, so each
        # term's driving contribution is minus its own difference. In the
        # pressure register that makes the sigma_zz term +int p_R dz.
        cum = lambda a: np.concatenate(
            [[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(z))])
        C_zz = cum(dzz * 1e6) / 1e12
        C_nd = -cum(nd * 1e6) / 1e12
        axes[2].plot(C_zz, zkm, color=col, lw=1.4,
                     label=f'{key}  $\\int\\Delta\\sigma_{{zz}}\\,dz$')
        axes[2].plot(C_nd, zkm, color=col, lw=1.4, ls='--',
                     label=f'{key}  $-\\int\\Delta(\\sigma_{{xx}}-\\sigma_{{zz}})\\,dz$')
        # the sum, bold above the sign change and faded below it: only the
        # part above is net force generated within the boundary layer
        C_sum = C_zz + C_nd
        above = zkm <= z_cross
        axes[2].plot(C_sum[above], zkm[above], color=col, lw=3.4,
                     solid_capstyle='round', label=f'{key}  sum')
        axes[2].plot(C_sum[~above], zkm[~above], color=col, lw=3.4, alpha=0.3,
                     solid_capstyle='round')
        nd_settled = np.interp(240e3, z, C_nd) - np.interp(100e3, z, C_nd)
        zz_lost = np.interp(240e3, z, C_zz) - np.max(C_zz)
        print(f'   cumulative at 100 km: sigma_zz {np.interp(100e3, z, C_zz):+.2f}, '
              f'N_D {np.interp(100e3, z, C_nd):+.2f} TN/m; below 100 km the N_D term '
              f'moves {nd_settled:+.3f} while the sigma_zz term loses {zz_lost:+.2f}')
        # where each difference stops changing
        deep = (zkm >= 150) & (zkm <= 240)
        dP = dzz[deep].mean()
        below = np.where((zkm > 40) & (np.abs(nd) < ZERO_TOL_MPA))[0]
        z_nd0 = zkm[below[0]] if len(below) else np.nan
        axes[0].axvline(dP, color=col, lw=1.0, ls='-.')
        # the one horizontal marker, on ALL panels: where Delta sigma_zz
        # changes sign (Dan, 2026-09-18)
        for ax in axes:
            ax.axhline(z_cross, color=col, lw=1.0, ls=':')
        print(f'{key}: dSzz deep offset {dP:+.2f} MPa (sign change {z_cross:.0f} km); '
              f'd(Sxx-Szz) falls below {ZERO_TOL_MPA} MPa at {z_nd0:.0f} km, '
              f'deep value {nd[deep].mean():+.2f} MPa')
        rows += [(key, 'dSzz_deep_offset_MPa', f'{dP:.2f}'),
                 (key, 'dSzz_sign_change_km', f'{z_cross:.0f}'),
                 (key, 'dNSD_equilibration_depth_km', f'{z_nd0:.0f}'),
                 (key, 'dNSD_deep_value_MPa', f'{nd[deep].mean():.2f}'),
                 (key, 'dNSD_lithospheric_extreme_MPa', f'{nd[zkm < 60].min():.0f}'),
                 (key, 'cum_szz_driving_at_100km_TNm', f'{np.interp(100e3, z, C_zz):.3f}'),
                 (key, 'cum_nd_driving_at_100km_TNm', f'{np.interp(100e3, z, C_nd):.3f}'),
                 (key, 'cum_nd_change_100_to_240km_TNm', f'{nd_settled:.3f}'),
                 (key, 'cum_szz_loss_below_peak_TNm', f'{zz_lost:.3f}')]
    axes[0].set_xlabel(r'(a) $\Delta\sigma_{zz}$, ridge $-\,x_I$ [MPa]'
                       '\n(dotted: its sign change, all panels)', fontsize=10.5)
    axes[1].set_xlabel(r'(b) $\Delta(\sigma_{xx}-\sigma_{zz})$, ridge $-\,x_I$ [MPa]',
                       fontsize=11)
    axes[2].set_xlabel('(c) cumulative integral of (a) and (b) [TN/m]\n'
                       '(signed: positive drives trench-ward)', fontsize=10.5)
    axes[0].set_ylabel('Depth [km]', fontsize=11)
    axes[0].set_ylim(250, 0)
    for ax in axes:
        ax.axvline(0, color='k', lw=1.6)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
    axes[0].legend(frameon=False, fontsize=10, loc='lower right')
    axes[1].legend(frameon=False, fontsize=10, loc='upper right')
    axes[2].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Ridge minus first isostatic column, mid-run average', fontsize=11)
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
