"""fig_slab_descent — the velocity cascade from trench to lower mantle.

Writes figures/fig_slab_descent.png and tables/slab_descent.csv from the
slab-geometry cache (run scripts/slab_geometry_cache.py first).

Requested by Dan 2026-09-22. The trailing-plate force balance measures
every force correctly and still cannot predict the plate velocity,
because the boundary condition is imposed at the hinge. This figure
measures what the hinge is doing.

THE ARGUMENT, in Dan's words: convergence can never be faster than the
rate at which material is removed, and when it cannot be removed fast
enough the shortfall appears as compression at the trench.

  1  slab geometry: tip depth, with the 660 km discontinuity marked, and
     the dip of the cold anomaly.
  2  THE CASCADE. Convergence, the slab's vertical velocity in the upper
     mantle, the same below 660 km, and the rate at which the tip
     advances -- all in cm/yr on one axis. Each stage is slower than the
     one above it.
  3  THE MECHANISM, detrended: the slab's upper-mantle descent against
     the in-plane resultant at the trench, with the Delta N_D axis
     inverted so the anti-correlation reads as tracking.

⚠ Row 3 MUST be detrended. At levels almost everything here correlates
with everything else, because all of it trends monotonically: slab area
against Delta N_D gives r = +0.97 and means nothing (detrended it is
~0). An earlier version of this figure plotted exactly that pairing --
the most persuasive-looking panel carrying the least information, while
the relationship that survives detrending was not shown at all.

WHAT IT ESTABLISHES.
  * The tip advances at only ~18 % (STD) / ~20 % (WAL) of the convergence
    rate. Four fifths of what converges does NOT deepen the slab.
  * It is accommodated by the slab lengthening and flattening instead:
    dip 56 -> 26 deg (STD) and 74 -> 40 (WAL), with the cross-section
    growing five- to six-fold (tabulated; the dip panel carries the same
    story visually).
  * The removal rate below 660 km is CAPPED and inert: 0.80 +/- 0.06
    (STD) / 1.00 +/- 0.09 cm/yr, trending +0.02 / +0.03 per 10 Myr. It
    responds to nothing.
  * Above 660 km the slab and the plate are one kinematic unit --
    r(vz_upper, plate velocity) = +0.90 / +0.92 DETRENDED.

⚠ The compression link is STD-ONLY. r(vz_upper, Delta N_D) = -0.89
detrended in STD (-0.95 at levels): when the slab's descent slows, the
trench compresses. In WAL it is -0.13, i.e. absent. Write it as a
strong-channel result, not a general one.

THE SLAB IS THE COLD ANOMALY, T < 1300 K against a 1573 K ambient. The
material volume fractions cannot be used: NormalSP and NormalOP partition
the whole domain and are side labels, not plate markers (see the cache
builder).

DRAFT CAPTION. The slab through both runs, STD (left) and WAL (right).
Top: the depth of the slab tip, with the 660 km discontinuity marked, and
the dip of the cold anomaly between 200 and 400 km. Middle: the velocity
cascade -- the convergence rate at the trench, the slab's mean vertical
velocity in the upper mantle and below 660 km, and the rate at which the
tip advances. Each stage is slower than the one above it, so most of what
converges is accommodated by the slab lengthening and flattening rather
than deepening. Bottom: the slab's upper-mantle descent rate against the
normal-stress-difference resultant across the trailing plate, both with
their secular trends removed and the resultant's axis inverted, so that
episodes of slower descent coincide with greater compression at the
trench.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # the shared scripts/ dir
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from fig_budget_time import terms
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))   # repo root
C_CONV = 'k'                # convergence at the trench
C_UPPER = '#0072B2'         # slab, upper mantle
C_DEEP = '#D55E00'          # slab, below 660
C_TIP = '#7B3294'           # tip advance
C_AREA = '#1B9E77'
C_RULE = '#BFC3D1'
Z660 = 660.0


def main():
    d = cpc.load()
    s = sgc.load()
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 10.4), sharex=True,
                             gridspec_kw={'height_ratios': [1.15, 1.35, 1.0]})
    rows = [('model', 'quantity', 'value')]
    ld = lambda t, y: y - np.polyval(np.polyfit(t, y, 1), t)

    for col, key in enumerate(('STD', 'WAL')):
        ts = s[f'{key}_t']
        z_tip, vz_up = s[f'{key}_z_tip'], s[f'{key}_vz_upper']
        vz_dp, dip, area = s[f'{key}_vz_deep'], s[f'{key}_dip'], s[f'{key}_area']
        c = cpc.derive(d, key); q = terms(d, key); t = c['t']; zkm = c['z'] / 1e3
        n = min(len(t), len(ts))
        assert np.allclose(t[:n], ts[:n], atol=0.3), f'{key}: time axes differ'
        v = np.abs(d[f'{key}_vx_TR'][:, zkm < 20].mean(axis=1))[:n]
        roll = (np.gradient(d[f'{key}_xT'] / 1e3, t) / 10.0)[:n]
        conv = v + roll
        ND = q['d_nd'][:n] / 1e12
        ts, z_tip, vz_up, vz_dp, dip, area = (a[:n] for a in
                                              (ts, z_tip, vz_up, vz_dp, dip, area))
        tip_rate = np.gradient(z_tip, ts) / 10.0          # km/Myr -> cm/yr

        a0, a1, a2 = (axes[r, col] for r in range(3))
        a0.plot(ts, z_tip, '-', color='k', lw=2.0, label='slab tip depth')
        a0.axhline(Z660, color='0.5', lw=1.0, ls='--')
        a0.annotate('660 km', xy=(ts[1], Z660), fontsize=8, color='0.4',
                    va='bottom')
        a0.invert_yaxis()
        a0.set_ylabel('tip depth [km]', fontsize=9.5)
        a0.set_title(key, fontsize=11)
        a0b = a0.twinx()
        a0b.plot(ts, dip, '-', color='0.55', lw=1.3)
        a0b.set_ylabel('dip [deg] (grey)', fontsize=8.5, color='0.45')
        a0b.tick_params(axis='y', labelcolor='0.45', labelsize=8)

        for arr, c_, lw, nm in ((conv, C_CONV, 2.2, 'convergence at the trench'),
                                (vz_up, C_UPPER, 1.8, 'slab $v_z$, 200–600 km'),
                                (vz_dp, C_DEEP, 1.8, 'slab $v_z$, below 660 km'),
                                (tip_rate, C_TIP, 1.5, 'tip advance rate')):
            a1.plot(ts, arr, '-', color=c_, lw=lw,
                    label=f'{nm}  ({np.nanmean(arr):.2f})')
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel('Rate [cm/yr]', fontsize=10)
        a1.legend(frameon=False, fontsize=8, loc='upper right')

        # Row 3 is the MECHANISM, and it must be plotted DETRENDED. At
        # levels almost everything here correlates with everything else
        # because all of it trends monotonically -- slab area against
        # Delta N_D looks like r = +0.97 and is worth nothing (detrended
        # it is ~0). The relationship that survives detrending is the
        # slab's upper-mantle descent against the in-plane resultant.
        # Delta N_D's axis is INVERTED so the anti-correlation reads as
        # tracking; the label says so.
        rr = float(np.corrcoef(ld(ts, vz_up), ld(ts, ND))[0, 1])
        a2.plot(ts, ld(ts, vz_up), '-', color=C_UPPER, lw=1.9)
        a2.axhline(0, color='k', lw=0.8)
        a2.set_ylabel('detrended slab $v_z$\n200–600 km [cm/yr]', fontsize=9,
                      color=C_UPPER)
        a2.tick_params(axis='y', labelcolor=C_UPPER)
        a2.set_xlabel('Model time [Myr]', fontsize=11)
        a2b = a2.twinx()
        a2b.plot(ts, ld(ts, ND), '-', color='k', lw=1.6)
        a2b.invert_yaxis()
        a2b.set_ylabel('detrended $\\Delta N_D$ [TN/m]\n(axis INVERTED)', fontsize=8.5)
        a2.text(0.02, 0.06, f'$r$ = {rr:+.2f}' +
                ('  — slower descent, more compression' if rr < -0.5
                 else '  — no relationship'),
                transform=a2.transAxes, fontsize=9,
                color='k' if rr < -0.5 else '0.45')
        for ax in (a0, a1, a2):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        frac = 100 * np.nanmean(tip_rate) / np.nanmean(conv)
        r_v = float(np.corrcoef(ld(ts, vz_up), ld(ts, v))[0, 1])
        r_nd = float(np.corrcoef(ld(ts, vz_up), ld(ts, ND))[0, 1])
        print(f'{key}: tip {z_tip[0]:.0f}->{z_tip[-1]:.0f} km at '
              f'{np.nanmean(tip_rate):.2f} cm/yr = {frac:.0f} % of convergence '
              f'({np.nanmean(conv):.2f}); vz_up {np.nanmean(vz_up):.2f}, '
              f'vz_deep {np.nanmean(vz_dp):.2f}+/-{np.nanstd(vz_dp):.2f}; '
              f'dip {dip[0]:.0f}->{dip[-1]:.0f}; area x{area[-1]/area[0]:.1f}; '
              f'r(vz_up, v) {r_v:+.2f}, r(vz_up, dN_D) {r_nd:+.2f}')
        rows += [(key, 'tip_depth_start_km', f'{z_tip[0]:.0f}'),
                 (key, 'tip_depth_end_km', f'{z_tip[-1]:.0f}'),
                 (key, 'tip_advance_rate_cm_yr', f'{np.nanmean(tip_rate):.3f}'),
                 (key, 'convergence_mean_cm_yr', f'{np.nanmean(conv):.3f}'),
                 (key, 'tip_advance_percent_of_convergence', f'{frac:.1f}'),
                 (key, 'vz_upper_mean_cm_yr', f'{np.nanmean(vz_up):.3f}'),
                 (key, 'vz_deep_mean_cm_yr', f'{np.nanmean(vz_dp):.3f}'),
                 (key, 'vz_deep_sd_cm_yr', f'{np.nanstd(vz_dp):.3f}'),
                 (key, 'vz_deep_trend_per_10Myr',
                  f'{np.polyfit(ts, vz_dp, 1)[0]*10:.3f}'),
                 (key, 'dip_start_deg', f'{dip[0]:.0f}'),
                 (key, 'dip_end_deg', f'{dip[-1]:.0f}'),
                 (key, 'area_growth_factor', f'{area[-1]/area[0]:.2f}'),
                 (key, 'corr_detrended_vzupper_platevelocity', f'{r_v:.3f}'),
                 (key, 'corr_detrended_vzupper_dND', f'{r_nd:.3f}')]

    fig.suptitle('The slab: tip advances at a fifth of the convergence rate, and '
                 'the removal rate below 660 km is capped', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'kinematics', 'fig_slab_descent.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('slab_descent', rows[0], rows[1:],
                      script='fig_slab_descent.py',
                      figure='fig_slab_descent.png', models=('STD', 'WAL'),
                      meta={'slab_definition': f'T < {sgc.T_SLAB:.0f} K '
                                               f'(ambient {sgc.T_AMBIENT:.0f} K), '
                                               f'below {sgc.Z_MIN/1e3:.0f} km',
                            'convergence': 'plate speed + trench rollback'})
    print('written:', tab)


if __name__ == '__main__':
    main()
