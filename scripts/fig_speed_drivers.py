"""fig_speed_drivers — what the plate speed does and does not track.

Writes figures/fig_speed_drivers.png and tables/speed_drivers.csv from
the committed column-profile cache.

Requested by Dan 2026-09-21: show what leads to plate speed-ups and
slow-downs. The candidate drivers are the trench side (a deeper trench
means a bigger pressure deficit, a larger trench pull, and a less
compressional N_D at the trench) and the plate-wide GPE* -- and the
GPE* term splits into a part that can respond quickly and a part that
cannot.

THE DECOMPOSITION AND THE TEST. Across the isostatic domain the measured
ridge push is the sum of two things (conventions §6b):

    measured ridge push  =  NON-TILTING  -  TILTING
    non-tilting = int p_R_untilted dz     the isostatic cooling term: the
                                          density structure alone
    tilting     = |Delta P| * z_c         the adverse asthenospheric
                                          pressure gradient

The prediction being tested is that the non-tilting term is a thermal
quantity and must evolve on a thermal timescale, so it CANNOT couple to
rapid speed variations, while the tilting term is velocity-slaved
through the channel and must.

The discriminating statistic is the INCREMENT correlation, not the level
correlation (conventions §6b.5): both series carry strong secular trends
over 80 Myr, so levels correlate for reasons that have nothing to do with
a causal link. Differences between consecutive snapshots are immune to a
shared trend. Both are computed; the legends quote the increment one as
r', and the table carries both.

TRENCH DEPTH is the EQUIVALENT TOPOGRAPHY, -sigma_zz(z->0)/rho g,
referenced to the first isostatic column, NOT the free-surface field.
That route was validated on 2026-09-21 against the top-boundary
free-surface reading and agrees with it to 1 m across the whole plate,
and it has the practical advantage of coming straight from the cached
stress columns -- no dependence on the cache's surface reading, which is
still the superseded one.

Layout: columns STD | WAL, rows
  1  plate speed (|v_x| averaged over the plate, above 20 km)
  2  trench depth, equivalent topography relative to x_I
  3  the trench resultants: trench pull, and N_D at the trench
  4  the isostatic-domain decomposition: non-tilting, tilting, measured

DRAFT CAPTION. Plate speed and its candidate drivers through both runs,
STD (left) and WAL (right). From the top: the plate speed, averaged
horizontally between the trench and ridge and over the top 20 km; the
trench depth, as the equivalent topography of the trench column relative
to the first isostatic column; the trench pull and the
normal-stress-difference resultant at the trench; and the isostatic
domain split into its non-tilting part, the isostatic cooling term set by
the density structure, and its tilting part, |Delta P| z_c. Legends quote
r', the correlation of each quantity's increments with the increments of
the plate speed, which is insensitive to the shared secular trend.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHO_G = 3300.0 * 9.8        # no ocean in these models (conventions §6.1)
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain
C_RIDGE = '#D55E00'         # isostatic / ridge push domain
C_TILT = '#7B3294'
C_RULE = '#BFC3D1'


def corr(a, b):
    """Level and increment correlation of two series."""
    lev = float(np.corrcoef(a, b)[0, 1])
    inc = float(np.corrcoef(np.diff(a), np.diff(b))[0, 1])
    return lev, inc


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    fig, axes = plt.subplots(4, 2, figsize=(12.6, 11.4), sharex=True)
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        z, zkm, t = c['z'], c['z'] / 1e3, c['t']
        zc = c['zc']
        res = cpc.resultants(d, key)

        # plate speed: horizontally averaged v_x, plate interior only
        v = d[f'{key}_vx_TR']
        speed = np.abs(v[:, zkm < 20].mean(axis=1))

        # trench depth as EQUIVALENT TOPOGRAPHY relative to x_I:
        # extrapolate the pressure-register anomaly to z = 0 and divide by
        # rho g. p_T is a deficit (negative), so w_T comes out positive
        # downward.
        s0 = c['p_T'][:, 0] - 0.5 * (c['p_T'][:, 1] - c['p_T'][:, 0])
        w_T = -s0 / RHO_G

        trench_pull = -np.trapezoid(c['p_T'][:, zc], z[zc], axis=1)
        ridge_measured = np.trapezoid(c['p_R'][:, zc], z[zc], axis=1)
        tilting = -c['dP'] * (cpc.ZC_KM * 1e3)          # |Delta P| * z_c
        non_tilting = np.trapezoid(c['p_R_untilted'][:, zc], z[zc], axis=1)
        nd_T = res['nd_T']

        series = [('trench_depth_m', w_T), ('trench_pull_TNm', trench_pull / 1e12),
                  ('nd_trench_TNm', nd_T / 1e12),
                  ('ridge_measured_TNm', ridge_measured / 1e12),
                  ('tilting_TNm', tilting / 1e12),
                  ('non_tilting_TNm', non_tilting / 1e12),
                  ('total_gpe_TNm', (trench_pull + ridge_measured) / 1e12)]
        cc = {}
        for nm, a in series:
            lev, inc = corr(a, speed)
            cc[nm] = (lev, inc)
            rows += [(key, f'{nm}_corr_level_with_speed', f'{lev:.3f}'),
                     (key, f'{nm}_corr_increment_with_speed', f'{inc:.3f}'),
                     (key, f'{nm}_median', f'{np.median(a):.3f}')]
        rows += [(key, 'plate_speed_median_cm_yr', f'{np.median(speed):.3f}'),
                 (key, 'plate_speed_range_cm_yr',
                  f'{speed.min():.2f}-{speed.max():.2f}')]

        lab = lambda nm, s: f"{s}  ($r'$ = {cc[nm][1]:+.2f})"

        a0, a1, a2, a3 = (axes[r, col] for r in range(4))
        a0.plot(t, speed, '-', color='k', lw=2.0)
        a0.set_ylabel('plate speed [cm/yr]', fontsize=9.5)
        a0.set_title(key, fontsize=11)

        a1.plot(t, w_T, '-', color=C_TRENCH, lw=1.8,
                label=lab('trench_depth_m', 'trench depth'))
        a1.set_ylabel('trench depth [m]', fontsize=9.5)

        a2.plot(t, trench_pull / 1e12, '-', color=C_TRENCH, lw=1.8,
                label=lab('trench_pull_TNm', 'trench pull'))
        a2.plot(t, nd_T / 1e12, '-', color='k', lw=1.4,
                label=lab('nd_trench_TNm', '$N_D(x_T)$'))
        a2.axhline(0, color='k', lw=0.8)
        a2.set_ylabel('[TN/m]', fontsize=9.5)

        a3.plot(t, non_tilting / 1e12, '-', color=C_RIDGE, lw=2.0,
                label=lab('non_tilting_TNm', 'non-tilting (density)'))
        a3.plot(t, tilting / 1e12, '--', color=C_TILT, lw=1.8,
                label=lab('tilting_TNm', 'tilting $|\\Delta P|z_c$'))
        a3.plot(t, ridge_measured / 1e12, '-', color='0.45', lw=1.2,
                label=lab('ridge_measured_TNm', 'measured ridge push'))
        a3.set_ylabel('[TN/m]', fontsize=9.5)
        a3.set_xlabel('Model time [Myr]', fontsize=11)

        for ax in (a0, a1, a2, a3):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        for ax in (a1, a2, a3):
            ax.legend(frameon=False, fontsize=8, loc='best')

        print(f'\n=== {key} ===  speed {speed.min():.2f}-{speed.max():.2f} cm/yr')
        for nm, _ in series:
            print(f'  {nm:22s} level r {cc[nm][0]:+.2f}   increment r {cc[nm][1]:+.2f}')

    fig.suptitle("Plate speed and its candidate drivers  "
                 "($r'$ = correlation of INCREMENTS with the speed increments,\n"
                 'the statistic that is immune to the shared secular trend)',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_speed_drivers.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('\nwritten:', out)
    tab = write_table('speed_drivers', rows[0], rows[1:],
                      script='fig_speed_drivers.py',
                      figure='fig_speed_drivers.png', models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM,
                            'speed': '|v_x| averaged x_T..x_R and over z < 20 km',
                            'trench_depth': 'equivalent topography '
                                            '-sigma_zz(0)/rho_g relative to x_I',
                            'decomposition': 'measured = non_tilting - tilting'})
    print('written:', tab)


if __name__ == '__main__':
    main()
