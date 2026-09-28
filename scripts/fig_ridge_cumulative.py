"""fig_ridge_cumulative — what a fixed integration depth costs.

Writes figures/fig_ridge_cumulative.png and tables/ridge_cumulative.csv from
the committed column-profile cache.

SI FIGURE, one job only (Dan, 2026-09-27): the level of approximation
involved in taking a single fixed z_c rather than each snapshot's own
maximum. An earlier seven-panel version also carried the Delta N_D
cumulative and three panels on what SETS z*; all of that is out. The
Delta N_D case is now a sentence of text (numbers below), and the
"what sets z*" material is a separate argument that has not been given a
figure.

  a, b  the cumulative ridge push against depth, one panel per model.
        One light line per snapshot (t >= 8 Myr), the heavy line the
        mid-run average (36-44 Myr), a dot at each snapshot's maximum.
        Shaded: where the mid-run curve stays within 1 % of its maximum.
        Dashed: the decline predicted by the measured deep anomaly,
        peak + Delta P (z - z*), with no fitting -- below the maximum the
        integral falls at exactly the rate the asthenospheric pressure
        difference sets.
  c     the cost. Shortfall against each snapshot's own maximum for every
        candidate fixed depth, median and 10-90th percentile band.
        DEPTH IS THE VERTICAL AXIS here too, so the reader can carry a
        level straight across from (a, b).

WHAT IT SHOWS. The maxima scatter over 70-80 km of depth and the peak
force varies by a factor of two across the run, which on first sight looks
like a lot to replace with one constant. Panel (c) is the answer: the
cumulative is flat near its maximum, so over a broad range of candidate
depths the shortfall is a few per cent. At z_c = 75 km:

                    median   10-90th   worst   worst after 20 Myr
      STD            3.8 %   0.1-14.8 %  31.5 %       7.0 %
      WAL            2.9 %   0.6-7.2 %   14.2 %       8.7 %

Only the median is DRAWN; the spread is here and in the table.

The worst cases are the two or three earliest retained snapshots (t = 8
and 12 Myr), where the boundary layer is still thin and 75 km reaches well
past the peak. THE PENALTY IS ASYMMETRIC -- above the peak the curve is
flat, so stopping short costs ~1 %, while overshooting into the
counterflow costs 10-30 %. A fixed depth errs in the safe direction for
most of the run.

⚠ z_c = 75 km is the FIXED NOMINAL INTEGRATION DEPTH (the manuscript's
wording, Sections 2-3): one round depth standing in for the maximum across
both runs and the whole time span.

THE Delta N_D TERM, stated rather than plotted. It has no extremum -- it
swings through the top 40 km and then drifts -- so it inherits its
stopping depth from the sigma_zz criterion. The cost of that is small:
between 75 and 150 km the cumulative moves 0.09 TN/m (STD) and 0.03 (WAL),
which is 7 % and 2 % of the GPE-like term it is set against (quoted against
a 150 km reference; the figure now uses 200 km, where the flow reverses).
The same argument therefore holds for both terms in the balance.

DRAFT CAPTION. The cost of a fixed integration depth. (a, b) The
cumulative integral of the vertical normal stress difference across the
isostatic domain -- the ridge push accumulated from the surface to depth
z -- for STD and WAL. Light lines are individual snapshots at 2 Myr
spacing over 8-80 Myr, the heavy line is the mid-run average over
36-44 Myr, and dots mark each snapshot's maximum. Shading spans the depths
over which the mid-run curve stays within 1 per cent of its maximum; the
dashed line is the decline predicted by the measured deep pressure
anomaly, with no fitting. (c) The shortfall in ridge push incurred by
integrating to a fixed depth instead of to each snapshot's own maximum,
as a function of that depth, taken as the median over the 37 snapshots. At
the fixed nominal integration depth z_c = 75 km the median shortfall is
3.8 per cent (STD) and 2.9 (WAL); it exceeds 10 per cent only for the two
earliest retained snapshots (t = 8 and 12 Myr), and after 20 Myr the worst
case is 7.0 and 8.7 per cent.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_budget_time import T_MIN_MYR
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_MODEL = {'STD': '#002147', 'WAL': '#E5007D'}
C_RIDGE = '#D55E00'
C_RULE = '#BFC3D1'
Z_MAX_KM = 200.0
# candidate fixed depths for panels (c, f): the WHOLE plotted depth range,
# so the curves are not truncated at an arbitrary grid. What bounds them on
# the page is the x limit, which is a stated tolerance, not a hidden choice.
ZC_GRID = np.arange(20.0, 200.1)
# Deep reference for the N_D tolerance. 200 km because that is where the
# horizontal velocity changes sign (199 km in both runs, fig_lab_kinematics)
# -- the base of the material that moves with the plate at all, so it is a
# physical stopping point rather than an arbitrary one. N_D has no extremum
# of its own, so some reference has to be chosen; this is the defensible one.
Z_REF_KM = 200.0
C_ND = '0.15'                         # N_D is black in the suite's register


def cum(profile, z):
    """Cumulative integral from the surface, TN/m, on the cache grid."""
    return np.concatenate([[0.0], np.cumsum(
        0.5 * (profile[1:] + profile[:-1]) * np.diff(z))]) / 1e12


def flat_interval(z, c_, frac):
    """Contiguous depth interval about the maximum within `frac` of it."""
    k = int(np.argmax(c_))
    ok = c_ >= (1.0 - frac) * c_[k]
    lo = hi = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return z[lo] / 1e3, z[hi] / 1e3


def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 11.4), sharey=True,
                             layout='constrained',
                             gridspec_kw={'width_ratios': [1.0, 1.0, 1.05]})
    a_pen, a_pen_nd = axes[0, 2], axes[1, 2]
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        z, t = c['z'], c['t']
        m = t >= T_MIN_MYR
        mid = c['mid'] & m
        keep = z <= Z_MAX_KM * 1e3
        zk = z[keep]
        ax, ax_nd = axes[0, col], axes[1, col]

        C, zstar, peak = [], [], []
        for i in np.where(m)[0]:
            cu = cum(c['p_R'][i], z)[keep]
            k = int(np.argmax(cu))
            C.append(cu); zstar.append(zk[k]); peak.append(cu[k])
            ax.plot(cu, zk / 1e3, '-', color=C_RIDGE, lw=0.7, alpha=0.20, zorder=1)
            ax.plot(cu[k], zk[k] / 1e3, 'o', ms=3.0, color=C_RIDGE, alpha=0.55,
                    zorder=2)
        C = np.array(C)
        avg = np.mean([cum(c['p_R'][i], z)[keep] for i in np.where(mid)[0]], axis=0)
        ka = int(np.argmax(avg))
        f1 = flat_interval(zk, avg, 0.01)
        dPm = float(np.mean(c['dP'][mid]))

        ax.axhspan(*f1, color='0.6', alpha=0.13, lw=0, zorder=0,
                   label=f'within 1 % of the maximum ({f1[1]-f1[0]:.0f} km)')
        ax.plot(avg, zk / 1e3, '-', color=C_RIDGE, lw=2.4, zorder=4,
                label='mid-run average')
        below = zk >= zk[ka]
        ax.plot(avg[ka] + dPm * (zk[below] - zk[ka]) / 1e12, zk[below] / 1e3,
                '--', color='0.25', lw=1.4, zorder=5,
                label=f'slope $=\\Delta P$ ({dPm/1e6:.1f} MPa)')
        ax.plot(avg[ka], zk[ka] / 1e3, 'o', ms=8, color=C_RIDGE,
                markeredgecolor='white', markeredgewidth=1.2, zorder=6,
                label=f'$z^*$ = {zk[ka]/1e3:.0f} km')
        ax.axhline(cpc.ZC_KM, color='0.30', lw=1.2, zorder=3)
        ax.set_title(f'({"ab"[col]}) {key}', fontsize=12, loc='left')
        ax.set_xlabel('Cumulative ridge push [TN/m]', fontsize=11.5)
        ax.set_xlim(-0.5, 3.6)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.legend(frameon=False, fontsize=9, loc='lower right')

        # ---- panel (c): the cost, with DEPTH on the vertical axis --------
        pk = np.array(peak)
        pen = np.array([pk - C[:, np.argmin(np.abs(zk - g * 1e3))]
                        for g in ZC_GRID])              # TN/m, (n_depth, n_snap)
        # MEDIAN ONLY, no band (Dan, 2026-09-27). A 10-90th percentile band
        # is set by the two earliest retained snapshots and made the panel
        # say "this varies enormously" -- the opposite of its point. The
        # range is in the caption and the table instead.
        a_pen.plot(np.median(pen, axis=1), ZC_GRID, '-', color=C_MODEL[key],
                   lw=2.4, label=key)

        # ---- Delta N_D, same treatment (Dan, 2026-09-27) ---------------
        # It has NO extremum, so there is no per-snapshot optimum to measure
        # against. The cost is instead the departure from a DEEP REFERENCE
        # (Z_REF_KM), expressed as a percentage of the ridge push it is set
        # against in the balance -- the same units as panel (c), so the two
        # rows are directly comparable.
        fib = ((d[f'{key}_sxx_R'] - d[f'{key}_szz_R'])
               - (d[f'{key}_sxx_I'] - d[f'{key}_szz_I']))
        Cnd = np.array([cum(fib[i], z)[keep] for i in np.where(m)[0]])
        for cn in Cnd:
            ax_nd.plot(cn, zk / 1e3, '-', color=C_ND, lw=0.7, alpha=0.18, zorder=1)
        avg_nd = np.mean([cum(fib[i], z)[keep] for i in np.where(mid)[0]], axis=0)
        ax_nd.plot(avg_nd, zk / 1e3, '-', color=C_ND, lw=2.4, zorder=4,
                   label='mid-run average')
        ax_nd.axhline(cpc.ZC_KM, color='0.30', lw=1.2, zorder=3)
        ax_nd.set_title(f'({"de"[col]}) {key}', fontsize=12, loc='left')
        ax_nd.set_xlabel(r'Cumulative $N_D(x_R)-N_D(x_I)$ [TN/m]', fontsize=11.5)
        ax_nd.grid(alpha=0.25, color=C_RULE, lw=0.6)
        if col == 0:
            ax_nd.legend(frameon=False, fontsize=9, loc='lower left')
        iref = np.argmin(np.abs(zk - Z_REF_KM * 1e3))
        pen_nd = np.array([np.abs(Cnd[:, np.argmin(np.abs(zk - g * 1e3))]
                                  - Cnd[:, iref]) for g in ZC_GRID])      # TN/m
        a_pen_nd.plot(np.median(pen_nd, axis=1), ZC_GRID, '-', color=C_MODEL[key],
                      lw=2.4, label=key)
        izcg = np.argmin(np.abs(ZC_GRID - cpc.ZC_KM))
        print(f'      dN_D: departure from {Z_REF_KM:.0f} km at z_c = '
              f'{np.median(pen_nd[izcg]):.3f} TN/m '
              f'(worst {pen_nd[izcg].max():.3f})')
        rows += [(key, 'dnd_variation_at_zc_median_TNm',
                  f'{np.median(pen_nd[izcg]):.4f}'),
                 (key, 'dnd_variation_at_zc_worst_TNm',
                  f'{pen_nd[izcg].max():.4f}'),
                 (key, 'dnd_reference_depth_km', f'{Z_REF_KM:.0f}')]

        zs = np.array(zstar) / 1e3
        izc = np.argmin(np.abs(ZC_GRID - cpc.ZC_KM))
        p_zc = pen[izc]
        late = t[m] >= 20.0
        print(f'{key}: z* median {np.median(zs):.0f} km (range {zs.min():.0f}–{zs.max():.0f});'
              f'  peak {np.median(pk):.2f} TN/m (range {pk.min():.2f}–{pk.max():.2f})')
        print(f'      at z_c = {cpc.ZC_KM:.0f} km: median {np.median(p_zc):.3f} TN/m, '
              f'worst {p_zc.max():.3f} (t = {t[m][np.argmax(p_zc)]:.0f} Myr), '
              f'worst after 20 Myr {p_zc[late].max():.3f} TN/m')
        rows += [(key, 'zstar_median_km', f'{np.median(zs):.1f}'),
                 (key, 'zstar_min_km', f'{zs.min():.1f}'),
                 (key, 'zstar_max_km', f'{zs.max():.1f}'),
                 (key, 'peak_median_TNm', f'{np.median(pk):.3f}'),
                 (key, 'variation_at_zc_median_TNm', f'{np.median(p_zc):.4f}'),
                 (key, 'variation_at_zc_p10_TNm', f'{np.percentile(p_zc,10):.4f}'),
                 (key, 'variation_at_zc_p90_TNm', f'{np.percentile(p_zc,90):.4f}'),
                 (key, 'variation_at_zc_worst_TNm', f'{p_zc.max():.4f}'),
                 (key, 'variation_at_zc_worst_after20Myr_TNm', f'{p_zc[late].max():.4f}'),
                 (key, 'midrun_zstar_km', f'{zk[ka]/1e3:.1f}'),
                 (key, 'midrun_within1pct_lo_km', f'{f1[0]:.1f}'),
                 (key, 'midrun_within1pct_hi_km', f'{f1[1]:.1f}')]

    for ap, lab, ttl in ((a_pen, 'Estimate variation [TN/m]',
                          r'(c) fixed $z_c$ versus the maximum cumulative'),
                         (a_pen_nd, 'Estimate variation [TN/m]',
                          f'(f) the same for $N_D$, versus its {Z_REF_KM:.0f} km value')):
        ap.axhline(cpc.ZC_KM, color='0.30', lw=1.2)
        ap.set_xlabel(lab, fontsize=11.5)
        ap.set_title(ttl, fontsize=12, loc='left')
        ap.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ap.legend(frameon=False, fontsize=9, loc='lower right')
    a_pen_nd.set_xlim(0, 0.35)
    axes[1, 0].set_ylabel('Depth [km]', fontsize=12)
    a_pen.text(0.97, cpc.ZC_KM + 9, f'$z_c$ = {cpc.ZC_KM:.0f} km (fixed nominal)',
               transform=blended_transform_factory(a_pen.transAxes, a_pen.transData),
               fontsize=9, color='0.25', ha='right', va='top')
    a_pen.set_xlim(0, 0.35)
    axes[0, 0].set_ylabel('Depth [km]', fontsize=12)
    axes[0, 0].set_ylim(Z_MAX_KM, 0)
    out = os.path.join(ROOT, 'figures', 'fig_ridge_cumulative.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('ridge_cumulative', rows[0], rows[1:],
                      script='fig_ridge_cumulative.py',
                      figure='fig_ridge_cumulative.png', models=('STD', 'WAL'),
                      meta={'quantity': 'cumulative int_0^z [szz(x_I) - szz(x_R)] dz',
                            'panel_c': 'shortfall vs each snapshot own maximum, per '
                                       'candidate fixed depth, 40-140 km',
                            'midrun_window_Myr': list(cpc.MIDRUN_MYR),
                            't_min_Myr': T_MIN_MYR, 'zc_km': cpc.ZC_KM,
                            'zc': 'fixed nominal integration depth',
                            'dnd_note': 'the N_D term has no extremum; between 75 and '
                                        '150 km its cumulative moves 0.09 (STD) / 0.03 '
                                        '(WAL) TN/m, 7 % / 2 % of the GPE-like term'})
    print('written:', tab)


if __name__ == '__main__':
    main()
