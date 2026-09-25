"""fig_plate_thickness — four independent estimates of the plate's mechanical
thickness through time, and how far apart they are.

Writes figures/fig_plate_thickness.png and tables/plate_thickness.csv from
the committed column-profile cache.

Requested by Dan 2026-09-26. `fig_mechanical_thickness` carries six
definitions and uses them to test the h/2 scaling of the trench dipole;
this figure does something narrower and more basic -- it asks how well the
plate's base is defined at all, by putting four INDEPENDENT estimates on
one axis and plotting their spread.

THE FOUR ESTIMATES. Three are lifted unchanged from
fig_mechanical_thickness (imported, not re-implemented, so the two figures
cannot drift apart) and are evaluated at the column of MAXIMUM BENDING
MOMENT, ~20-25 km seaward of the trench:

  thermal      the 900 C isotherm
  2 h_np       twice the neutral-plane depth, the neutral plane taken as
               the extremum of the cumulative fibre stress
  yield 10 %   where the fibre-stress envelope falls to a tenth of its peak

The fourth is new here (Dan's request):

  sigma_zz=0   the depth at which the RIDGE column's pressure anomaly,
               taken relative to the first isostatic column, changes sign

⚠ THE FOURTH IS NOT A STRENGTH MEASURE, and that is the point of adding
it. The other three ask where the plate stops behaving elastically or
stops being cold. This one asks where the topographic pressure anomaly
carried by the plate gives way to the opposing asthenospheric one -- a
FORCE-BALANCE boundary, measured on the isostatic domain, with no
reference to rheology or temperature. That it lands in the same place is
the result; it is not built to.

⚠ It is also the only one of the four that is not evaluated near the
trench. It comes from the ridge column, a whole plate-length away, so the
agreement is not an artefact of a shared location either.

⚠ This is the depth the dashed line marks in fig_column_anomalies, and the
same caution applies: it is where the stress anomaly changes sign, not
where the flow reverses. The base of the coherently translating plate is
deeper (99 km STD / 93 WAL, fig_lab_kinematics).

  1  the four estimates through time, one panel per model, with the
     spread between them shaded.
  2  the spread itself -- max minus min across the four -- for both
     models on one axis, which is the quantity the figure exists to show
     and is a comparison, so it is not split across panels.

DRAFT CAPTION. (a, b) Four independent estimates of the mechanical
thickness of the subducting plate through the runs, for STD and WAL: the
900 C isotherm, twice the neutral-plane depth, the depth at which the
fibre-stress envelope falls to a tenth of its peak, and the depth at
which the ridge column's pressure anomaly relative to the first isostatic
column changes sign. The first three are strength- or
temperature-based and are evaluated at the column of maximum bending
moment; the fourth is a force-balance boundary evaluated at the ridge
column, a plate length away. Shading spans the four. (c) The spread, the
largest estimate minus the smallest, for both models.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_mechanical_thickness import thicknesses
from fig_budget_time import T_MIN_MYR
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_RULE = '#BFC3D1'
C_MODEL = {'STD': '#002147', 'WAL': '#E5007D'}
EST = [('thermal', '#0072B2', '-', f'thermal, 900 $^\\circ$C'),
       ('np2', '#009E73', '--', r'$2\,h_{np}$'),
       ('yield10', '#D55E00', '-.', 'yield envelope (10 % of peak)'),
       ('szz0', '#7B3294', ':', r'$\Delta\sigma_{zz}=0$ (ridge column)')]


def sign_change_depth(d, key):
    """Depth at which the ridge column's pressure anomaly changes sign.

    Per snapshot, smoothed exactly as fig_column_anomalies smooths it, and
    searched below 20 km so the near-surface structure cannot trigger it.
    """
    c = cpc.derive(d, key)
    z = c['z']
    out = []
    for pr in c['p_R']:
        s = gaussian_filter1d(pr, 2)
        j = np.where((s[:-1] > 0) & (s[1:] <= 0) & (z[:-1] > 20e3))[0]
        out.append(z[j[0]] if len(j) else np.nan)
    return np.array(out)


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    fig = plt.figure(figsize=(11.0, 7.4), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.62])
    axes = [fig.add_subplot(gs[0, c]) for c in range(2)]
    # The spread is a COMPARISON between the runs, so both go on one axis
    # rather than being split across the columns above.
    axr = fig.add_subplot(gs[1, :])
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        r = thicknesses(d, key)
        t = cpc.derive(d, key)['t']
        m = t >= T_MIN_MYR
        est = {'thermal': r['thermal'], 'np2': r['np2'],
               'yield10': r['yield10'], 'szz0': sign_change_depth(d, key)}
        A = np.vstack([est[k] for k, *_ in EST]) / 1e3        # km
        lo, hi = np.nanmin(A, axis=0), np.nanmax(A, axis=0)

        ax = axes[col]
        ax.fill_between(t[m], lo[m], hi[m], color='0.6', alpha=0.20, lw=0,
                        label='spread of the four')
        for k, c_, ls, lab in EST:
            ax.plot(t[m], est[k][m] / 1e3, ls, color=c_, lw=1.9, label=lab)
        ax.set_title(key, fontsize=11.5)
        ax.set_xlabel('Model time [Myr]', fontsize=10.5)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        if col == 0:
            ax.set_ylabel('Mechanical thickness [km]', fontsize=11)
            ax.legend(frameon=False, fontsize=8.5, loc='lower right')

        axr.plot(t[m], (hi - lo)[m], '-', color=C_MODEL[key], lw=2.2, label=key)

        med = lambda a: np.nanmedian(a[m]) / 1e3
        print(f'{key}: thermal {med(est["thermal"]):5.1f}, 2h_np '
              f'{med(est["np2"]):5.1f}, yield10 {med(est["yield10"]):5.1f}, '
              f'szz=0 {med(est["szz0"]):5.1f} km;  spread median '
              f'{np.nanmedian((hi - lo)[m]):5.1f} km '
              f'({100*np.nanmedian((hi - lo)[m]) / np.nanmedian(0.5*(hi+lo)[m]):.0f} % '
              f'of the mid-estimate)')
        for k, *_ in EST:
            rows.append((key, f'{k}_median_km', f'{med(est[k]):.2f}'))
        rows += [(key, 'spread_median_km', f'{np.nanmedian((hi - lo)[m]):.2f}'),
                 (key, 'spread_min_km', f'{np.nanmin((hi - lo)[m]):.2f}'),
                 (key, 'spread_max_km', f'{np.nanmax((hi - lo)[m]):.2f}'),
                 (key, 'spread_percent_of_mid',
                  f'{100*np.nanmedian((hi - lo)[m]) / np.nanmedian(0.5*(hi+lo)[m]):.1f}')]

    lo_, hi_ = axes[0].get_ylim(), axes[1].get_ylim()
    for ax in axes:
        ax.set_ylim(min(lo_[0], hi_[0]), max(lo_[1], hi_[1]))
    axr.set_ylabel('Spread of the four\nestimates [km]', fontsize=10.5)
    axr.set_xlabel('Model time [Myr]', fontsize=11)
    axr.set_ylim(bottom=0)
    axr.grid(alpha=0.25, color=C_RULE, lw=0.6)
    axr.legend(frameon=False, fontsize=9.5, loc='upper left')
    fig.suptitle('Four independent estimates of the mechanical thickness',
                 fontsize=10.5, color='0.35')
    out = os.path.join(ROOT, 'figures', 'fig_plate_thickness.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('plate_thickness', rows[0], rows[1:],
                      script='fig_plate_thickness.py',
                      figure='fig_plate_thickness.png', models=('STD', 'WAL'),
                      meta={'estimates': 'thermal 900 C, 2 h_np, yield 10 %, '
                                         'sigma_zz sign change (ridge column)',
                            'columns': 'first three at the maximum-bending-moment '
                                       'column; the fourth at the ridge column',
                            't_min_Myr': T_MIN_MYR,
                            'note': 'the sigma_zz sign change is a force-balance '
                                    'boundary, not a strength measure, and is not '
                                    'where the flow reverses'})
    print('written:', tab)


if __name__ == '__main__':
    main()
