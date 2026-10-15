"""fig_ridge_cumulative — why z_c is set by the sigma_zz structure.

Writes figures/fig_ridge_cumulative.png and tables/ridge_cumulative.csv
from the committed column-profile cache.

SI FIGURE. Rebuilt 2026-10-08 on Dan's ruling; supersedes a six-panel
version entirely (same output paths, so nothing of it survives).

WHY DELTA N_D IS NOT HERE. It has asymptoted by 76-94 km and moves 1-2 %
between 100 and 175 km. A term that has stopped changing is indifferent
to where you stop integrating, so it cannot constrain z_c. The choice is
governed entirely by Delta sigma_zz -- the one term that never
asymptotes: it peaks at the crossing and then declines at the rate
Delta P sets, because past the crossing the integral accumulates the
counterflow rather than the plate's driving gradient. State that in one
sentence; it does not need panels.

FOUR PANELS, ONE TEMPORAL TREATMENT EACH. The retired version carried
three at once -- per-snapshot curves, a mid-run average and a
37-snapshot median -- so a reader carrying a depth between panels was
silently changing what was averaged.

  a  mid-run cumulative -Delta sigma_zz (x_R - x_I), both models, with
     the crossing marked
  b  mid-run cost of a different depth, as % of the value at z_c
  c  the crossing through time, against the fixed z_c
  d  what z_c costs at each step, against that step's own crossing

THE ARGUMENT IS THE ASYMMETRY, NOT THE FLATNESS. "Flat near z_c" invites
"flat to within what tolerance?". Directional is stronger and physical:
shallower costs little, deeper costs 16 % at 125 km and 29 % at 150 (STD),
because past the crossing the integral subtracts the counterflow. So the
defensible stopping point is the crossing.

                              STD        WAL
      mid-run crossing       75.0 km    85.0 km
      deepens                +9.1       +10.1 km / 10 Myr
      crossing range         42-105     47-125 km
      cost of z_c, median    -0.3 %     -0.9 %
      worst                  -3.6       -4.8 %
      worst after 20 Myr     -2.7       -4.8 %

WHAT THE COST PANEL MEASURES (changed 2026-10-14, numbers corrected
2026-10-15). z* is the crossing of the RIDGE-SIDE cumulative -- the
crossing exists because that term reverses sign below the plate, while
the trench side has no turnover and simply closes. The COST, however, is
evaluated on the WHOLE trench-to-ridge force, because that is what the
balance reports and what a reader is anxious about. So the numerator and
the denominator are the plate-wide force at z_c and at z*(t), while z*
itself comes from the ridge side. The mismatch is deliberate and must be
stated wherever these numbers are quoted.

⚠ THE SIGN GOES BOTH WAYS, so this is a change, not a shortfall. Early,
z* sits near 42 km while the trench side is still accumulating, so
stopping there gives LESS force than z_c does.

WHY z_c IS 80 km AND NOT 75 (ruled 2026-10-09, conventions §5.1). The old
answer was that 75 km coincided with STD's mid-run crossing. That is a
coincidence with ONE model at ONE epoch, and it does not survive the
question "why not WAL's?". The new answer is measured: z_c swept 60-120 km
at 1 km, error against each snapshot's own crossing taken IN TN/m, run-wide
rms 0.175 (75), 0.165 (80), 0.175 (85), 0.201 (90) summed over the two
runs. 80 km is the joint minimum, on the run-wide rms and on the mid-run
window alike. STD's own optimum is 74 km and WAL's is 87, so 85 is WAL's
number and costs STD enough to land back where 75 sat.

⚠ WHY THE PANEL NO LONGER USES THE RIDGE-SIDE DENOMINATOR. It did, and
that peak falls to 0.70 TN/m early in STD, so the cost read -31 % at
12-14 Myr where the absolute error is 0.22 TN/m. The error was real but
the percentage was inflated by a small denominator -- conventions §5.1
warns against exactly that measure for this quantity. On the plate-wide
force the same choice costs under 5 % at every state in both runs.

TWO THINGS A READER WOULD OTHERWISE GET WRONG. The crossing's spread is
SECULAR, not scatter (r^2 0.95 and 0.90) -- panel (c) shows it; the old
dot-scatter presentation hid it. And WAL errs the OTHER WAY: its crossing
reaches 126 km, so late in that run z_c sits well above the crossing and
the error is the shallow-side one. No fixed depth is right at every state
(Dan, 2026-10-09) and the figure's job is to show by how much it is
wrong, not to claim it is not.

The manuscript's SI caption (si.tex S2) still describes the retired
six-panel version and must be rewritten with this figure.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from column_profiles_cache import T_MIN_MYR
from fluidity_helpers import depth_ticks
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_MODEL = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
Z_MAX_KM = 200.0


def cum(profile, z):
    """Cumulative integral from the surface: (edge depths [km], TN/m).

    ⚠ RETURNS ITS OWN DEPTH AXIS, and that is the point. This used to
    return values alone, built on cell EDGES and then trimmed to len(z) so
    it could be paired with the cell-CENTRE array the panels already had.
    The two are offset by half a cell, so `argmin(|z_centres - 80|)` chose
    index 79 -- centre 79.5 km -- holding the integral to the 79 km edge.
    A constant 1 MPa profile then integrated to 0.079 TN/m over a nominal
    0-80 km instead of 0.080: one cell short, which is exactly the defect
    the repo-wide quadrature invariant exists to prevent, reintroduced by
    pairing an edge array with centre coordinates. Found by external
    review, 2026-10-15.
    """
    ze, cu = cpc.cumulative_edges(profile, z)      # len N+1, both of them
    return ze / 1e3, cu / 1e12



def _cum_self_test():
    """The quadrature invariant, asserted at import.

    A constant 1 MPa profile must integrate to exactly ZC_KM x 1 MPa at
    the z_c EDGE. The defect this guards against is not an arithmetic
    error but a pairing error -- edge values indexed by centre
    coordinates -- which is silent, costs one cell, and had already been
    reintroduced here once after the repo-wide quadrature fix.
    """
    z = (np.arange(250) + 0.5) * 1000.0
    ze, cu = cum(np.ones_like(z) * 1e6, z)
    j = int(np.argmin(np.abs(ze - cpc.ZC_KM)))
    want = cpc.ZC_KM * 1e3 * 1e6 / 1e12
    assert abs(ze[j] - cpc.ZC_KM) < 1e-9, (
        f'the index chosen as z_c sits at {ze[j]} km, not {cpc.ZC_KM}')
    assert abs(cu[j] - want) < 1e-12, (
        f'1 MPa over 0-{cpc.ZC_KM:.0f} km integrated to {cu[j]:.4f} TN/m, '
        f'not {want:.4f} -- edge/centre pairing has slipped again')


_cum_self_test()


def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.6), layout='constrained')
    a_shape, a_cost, a_zstar, a_tcost = (axes[0, 0], axes[0, 1],
                                         axes[1, 0], axes[1, 1])
    rows = [('model', 'quantity', 'value')]

    for key in ('STD', 'WAL'):
        c = cpc.derive(d, key)
        z, t = c['z'], c['t']
        m = t >= T_MIN_MYR
        mid = c['mid'] & m
        keep = z <= Z_MAX_KM * 1e3
        zk = z[keep] / 1e3
        col = C_MODEL[key]

        # ---- per snapshot: the crossing and the cost of 75 km there -----
        # z* IS THE RIDGE-SIDE CROSSING, but the cost of the fixed depth is
        # measured on the WHOLE TRENCH-TO-RIDGE force (Dan, 2026-10-14).
        # Two reasons, and the mismatch is deliberate. The crossing exists
        # BECAUSE the ridge-side term reverses sign below the plate; the
        # trench side has no turnover of its own, it simply closes, so a
        # crossing defined on the sum mixes a quantity that has one with a
        # quantity that does not. But what the balance reports, and what a
        # reader cares about, is the plate-wide Delta GPE*, so that is what
        # the error should be quoted against.
        # Measured: ridge-side-only gives a median cost of 3.0 / 3.1 % with
        # excursions to 31 % (STD, 12-14 Myr) and 20 % (WAL, 8 Myr), driven
        # by a small early denominator -- the ridge-side force is 0.70 TN/m
        # there against 1.2 later. On the plate-wide force the same choice
        # costs a median 0.1 / 0.8 % and never more than 3.2 / 5.0 %.
        # ⚠ THE SIGN GOES BOTH WAYS, which is why this is not called a
        # shortfall. Early, z* sits near 42 km while the trench side is
        # still accumulating, so stopping there would give LESS force than
        # z_c does: the value is positive at 38 % of STD steps and 8 % of
        # WAL steps.
        zs, pen_pct = [], []
        for i in np.where(m)[0]:
            ze, cu = cum(c['p_R'][i], z)            # ridge side: sets z*
            _, ct = cum(c['p_T'][i], z)
            w = ze <= Z_MAX_KM
            ze, cu, cw = ze[w], cu[w], (-ct[w] + cu[w])   # plate-wide: the cost
            k = int(np.argmax(cu))
            zs.append(ze[k])
            izc = int(np.argmin(np.abs(ze - cpc.ZC_KM)))  # the 80 km EDGE
            pen_pct.append(100.0 * (cw[izc] - cw[k]) / cw[k])
        zs, pen_pct = np.array(zs), np.array(pen_pct)

        # ---- mid-run shape and its cost curve ---------------------------
        # panels (a) and (b) on the SAME edge axis as the statistics above
        zedge = cum(c['p_R'][0], z)[0]
        we = zedge <= Z_MAX_KM
        zedge = zedge[we]
        avg = np.mean([cum(c['p_R'][i], z)[1][we] for i in np.where(mid)[0]], axis=0)
        ka = int(np.argmax(avg))
        izc = int(np.argmin(np.abs(zedge - cpc.ZC_KM)))
        cost = 100.0 * (avg - avg[izc]) / abs(avg[izc])

        a_shape.plot(avg, zedge, '-', color=col, lw=2.2, label=key)
        a_shape.plot(avg[ka], zedge[ka], 'o', ms=8, color=col,
                     markeredgecolor='white', markeredgewidth=1.2, zorder=5)
        a_cost.plot(cost, zedge, '-', color=col, lw=2.2, label=key)
        a_zstar.plot(t[m], zs, '-o', color=col, lw=1.6, ms=3.5, label=key)
        a_tcost.plot(t[m], pen_pct, '-o', color=col, lw=1.6, ms=3.5, label=key)

        sl = np.polyfit(t[m], zs, 1)[0] * 10
        print(f'{key}: mid-run crossing {zedge[ka]:.1f} km; '
              f'z* deepens {sl:+.1f} km/10 Myr ({zs.min():.0f}-{zs.max():.0f}); '
              f'cost of z_c median {np.median(pen_pct):+.1f} %, '
              f'worst {pen_pct.min():+.1f} %, '
              f'after 20 Myr worst {pen_pct[t[m] >= 20].min():+.1f} %')
        for zq in (100.0, 125.0, 150.0):
            rows.append((key, f'cost_at_{zq:.0f}km_pct',
                         f'{np.interp(zq, zedge, cost):.2f}'))
        rows += [(key, 'midrun_crossing_km', f'{zedge[ka]:.1f}'),
                 (key, 'zstar_trend_km_per_10Myr', f'{sl:.2f}'),
                 (key, 'zstar_min_km', f'{zs.min():.1f}'),
                 (key, 'zstar_max_km', f'{zs.max():.1f}'),
                 (key, 'cost_of_zc_median_pct', f'{np.median(pen_pct):.2f}'),
                 (key, 'cost_of_zc_worst_pct', f'{pen_pct.min():.2f}'),
                 (key, 'cost_of_zc_worst_after20Myr_pct',
                  f'{pen_pct[t[m] >= 20].min():.2f}')]

    for ax in (a_shape, a_cost):
        ax.axhline(cpc.ZC_KM, color='0.30', lw=1.2)
        ax.set_ylim(Z_MAX_KM, 0)
        ax.set_ylabel('Depth [km]', fontsize=11)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
    a_shape.axvline(0, color='0.55', lw=1.3)
    a_shape.set_xlabel(r'Cumulative $-\Delta\sigma_{zz}$  ($x_R - x_I$)  [TN/m]',
                       fontsize=10.5)
    a_shape.set_title('(a) Cumulative ridge-side force, mid-run',
                      fontsize=10.5, loc='left')
    a_shape.legend(frameon=False, fontsize=9, loc='lower right')

    a_cost.axvline(0, color='0.55', lw=1.3)
    for v in (-10, 10):
        a_cost.axvline(v, color='0.45', lw=1.0, ls='--')
    a_cost.set_xlim(-45, 15)
    a_cost.set_xlabel('Change from the value at $z_c$  [%]', fontsize=10.5)
    a_cost.set_title('(b) Sensitivity to the integration depth',
                     fontsize=10.5, loc='left')

    for ax, loc in ((a_zstar, 'upper left'), (a_tcost, 'lower right')):
        ax.set_xlabel('Model time [Myr]', fontsize=11)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.legend(frameon=False, fontsize=9, loc=loc)
    a_zstar.axhline(cpc.ZC_KM, color='0.30', lw=1.4)
    a_zstar.set_ylabel(r'Crossing depth $z^{*}$  [km]', fontsize=11)
    a_zstar.set_title('(c) Crossing depth through time',
                      fontsize=10.5, loc='left')
    a_tcost.axhline(0, color='0.55', lw=1.3)
    # THE -10 % GUIDE IS GONE. It was drawn when the cost was measured on
    # the ridge side alone and reached 31 %; on the plate-wide force
    # nothing comes within a factor of two of it, so it only pushed the
    # data into the top third of the panel.
    for v in (-5,):
        a_tcost.axhline(v, color='0.45', lw=1.0, ls='--')
    a_tcost.set_ylabel(r'Change relative to $z^{*}(t)$  [%]',
                       fontsize=11)
    # A FIXED, GENEROUS Y RANGE. Autoscaled, a 5 % wiggle fills the panel
    # and the quietest metric in the figure reads as the most dramatic.
    # -15 % leaves the curves in the upper third, where their size is the
    # point (Dan, 2026-10-14).
    a_tcost.set_ylim(-15.0, 3.0)
    a_tcost.set_title('(d) Cost of the fixed depth, plate-wide',
                      fontsize=10.5, loc='left')

    out = os.path.join(ROOT, 'figures', 'fig_ridge_cumulative.png')
    depth_ticks(fig)   # 20 km depth ticks (FIGURE_STYLE)
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('ridge_cumulative', rows[0], rows[1:],
                      script='fig_ridge_cumulative.py',
                      figure='fig_ridge_cumulative.png', models=('STD', 'WAL'),
                      meta={'quantity': 'cumulative int_0^z [-Delta sigma_zz] dz, '
                                        'ridge minus first isostatic column',
                            'cost_panel_b': 'signed % change relative to the '
                                            'value at z_c, ridge side, mid-run',
                            'cost_of_zc_rows': 'per-step signed % change of the '
                                               'PLATE-WIDE force between z_c and '
                                               'z*(t); z* is the RIDGE-SIDE '
                                               'crossing and the denominator is '
                                               'the plate-wide value at z*',
                            'zstar': 'depth at which the anomaly changes sign, i.e. '
                                     'where the cumulative is stationary',
                            'dnd_omitted': 'Delta N_D is not shown: it has asymptoted '
                                           'by 76-94 km and moves 1-2 % between 100 '
                                           'and 175 km, so it cannot constrain z_c',
                            'midrun_window_Myr': list(cpc.MIDRUN_MYR),
                            't_min_Myr': T_MIN_MYR, 'zc_km': cpc.ZC_KM})
    print('written:', tab)


if __name__ == '__main__':
    main()
