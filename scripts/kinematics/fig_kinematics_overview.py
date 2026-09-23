"""fig_kinematics_overview — the four kinematic series, STD and WAL, through time.

Writes figures/kinematics/fig_kinematics_overview.png and
tables/kinematics_overview.csv. Needs the column cache
(scripts/column_profiles_cache.py) and the slab cache
(scripts/kinematics/slab_geometry_cache.py).

Requested by Dan 2026-09-23: trailing-plate velocity, convergence rate,
slab dip and slab vertical velocity, for each model, against time. A
plain orientation figure -- no detrending, no smoothing, no derived
statistic in the panels. Everything else in the kinematics set works on
detrended anomalies, and this is the figure that shows what those
anomalies are anomalies OF.

Layout follows FIGURE_STYLE.md encoding 1, the register's preferred form:
side-by-side columns, STD left | WAL right, with a shared y scale per row
so the two models are compared by eye rather than by reading ticks.

  1  trailing-plate velocity |v_x|, mean over the top 20 km from the
     trench to the ridge.
  2  convergence rate = plate velocity + trench rollback, with the two
     components drawn faintly beneath it so the partition is visible.
  3  slab dip, mean inclination of the cold anomaly over 200-400 km.
  4  slab vertical velocity, mean over 200-600 km (the upper-mantle
     descent rate), with the sub-660 km rate faint for comparison.

Run medians over the cached snapshots (t >= 8 Myr), full range in
brackets:

  quantity                     STD                    WAL
  plate velocity [cm/yr]       1.69  [1.06, 2.93]     2.48  [1.58, 4.83]
  trench rollback [cm/yr]      1.67  [1.00, 2.13]     1.05  [-0.28, 2.16]
  convergence    [cm/yr]       3.47  [2.63, 4.04]     3.41  [2.38, 5.53]
  slab dip       [deg]        37.0   [26.1, 70.0]    50.5   [32.9, 73.7]
  slab v_z 200-600 km [cm/yr]  1.41  [0.98, 2.27]     1.68  [1.39, 3.05]
  slab v_z below 660  [cm/yr]  0.84  [0.65, 0.85]     0.97  [0.89, 1.19]

(These are computed here from the cached snapshots and differ by a few
per cent from the medians in DYNAMICS_FINDINGS §2.1, which come from
fig_convergence_partition's own masking. The comparisons below are not
sensitive to the difference.)

WHAT TO READ OFF IT. Three things this figure establishes on its own,
each of which is used elsewhere in the set:

  - **Convergence is nearly identical between the runs** (medians 3.47
    and 3.41, 2 % apart) while the PLATE velocities differ by 47 %
    (1.69 against 2.48). The
    weak asthenospheric layer changes how convergence is PARTITIONED
    between plate motion and trench rollback, not how fast the margin
    converges (DYNAMICS_FINDINGS §2.1; fig_convergence_partition
    develops this).
  - **STD's plate velocity declines through the run and WAL's does not**
    (−1.11 cm/yr, −27 %, surrogate p < 0.0001; WAL flat). This is the
    contrast that lets §1.5 attribute F_B's secular decline to the
    shrinking plate rather than to slowing.
  - **The ~25 Myr cycle is visible by eye in STD's dip and descent rate**
    and is the subject of §1.8. WAL's is faster and less regular -- its
    descent-rate spectrum peaks at 14.8 Myr against STD's 24.7.

⚠ The dip is a mean inclination over 200-400 km, not a curvature. Per
Dan's ruling 2026-09-23 that is sufficient as an expression of buckling
(the interpretation is inherited from the earlier paper), but the series
should not be described as measuring fold amplitude.

⚠ Rollback is d(x_T)/dt in the MIRRORED analysis frame, in which the
subducting plate lies on the right and trench retreat is +x. It is
plotted as a positive quantity where the trench retreats. WAL advances
(negative rollback) at 7 of 37 snapshots, through roughly 48-62 Myr --
the regime STD never enters (0 of 37), and the interval that dominates
several of WAL's whole-run statistics (§5.9). The deep descent rate is
also visibly CAPPED in both runs -- 0.84 and 0.97 cm/yr with almost no
variance -- against upper-mantle rates that swing by a factor of two;
that contrast is the subject of fig_slab_descent.

DRAFT CAPTION. Kinematics of the two runs through time, STD (left) and
WAL (right), with a shared vertical scale per row. From top: trailing-
plate velocity; convergence rate, with plate velocity and trench rollback
shown faintly; mean slab dip over 200-400 km; and slab vertical velocity
averaged over 200-600 km, with the sub-660 km rate faint. No series is
detrended or smoothed.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import column_profiles_cache as cpc
import slab_geometry_cache as sgc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))
C_RULE = '#BFC3D1'
C_PLATE = '#1B9E77'
C_CONV = 'k'
C_ROLL = '#E7298A'
C_DIP = '#56B4E9'
C_VZ = '#0072B2'
C_VZD = '#9ecae1'


def main():
    d, s = cpc.load(), sgc.load()
    fig, axes = plt.subplots(4, 2, figsize=(12.6, 12.2), sharex='col')
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        tc, zkm = c['t'], c['z'] / 1e3
        ts = s[f'{key}_t']; n = min(len(tc), len(ts)); t = tc[:n]
        vp = np.abs(d[f'{key}_vx_TR'][:, zkm < 20].mean(axis=1))
        roll = np.gradient(d[f'{key}_xT'] / 1e3, tc) / 10.0      # km/Myr -> cm/yr
        conv = vp + roll
        dip = s[f'{key}_dip'][:n]
        vz = s[f'{key}_vz_upper'][:n]
        vzd = s[f'{key}_vz_deep'][:n]

        a0, a1, a2, a3 = (axes[r, col] for r in range(4))

        a0.plot(tc, vp, '-', color=C_PLATE, lw=2.2)
        a0.set_ylabel('trailing-plate\nvelocity [cm/yr]', fontsize=9.5)
        a0.set_title(key, fontsize=11.5)

        a1.plot(tc, conv, '-', color=C_CONV, lw=2.4, label='convergence')
        a1.plot(tc, vp, '-', color=C_PLATE, lw=1.2, alpha=0.7,
                label='plate velocity')
        a1.plot(tc, roll, '-', color=C_ROLL, lw=1.2, alpha=0.7,
                label='trench rollback')
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel('convergence\n[cm/yr]', fontsize=9.5)
        a1.legend(frameon=False, fontsize=8, loc='upper left', ncol=3)

        a2.plot(t, dip, '-', color=C_DIP, lw=2.2)
        a2.set_ylabel('slab dip [deg]\n(200–400 km)', fontsize=9.5)

        a3.plot(t, vz, '-', color=C_VZ, lw=2.2, label='200–600 km')
        a3.plot(t, vzd, '-', color=C_VZD, lw=1.3, alpha=0.9, label='below 660 km')
        a3.set_ylabel('slab vertical\nvelocity [cm/yr]', fontsize=9.5)
        a3.set_xlabel('Model time [Myr]', fontsize=11)
        a3.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

        for ax in axes[:, col]:
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        print(f'=== {key}')
        for nm, y, u in (('plate_velocity', vp, 'cm/yr'),
                         ('convergence', conv, 'cm/yr'),
                         ('rollback', roll, 'cm/yr'),
                         ('slab_dip', dip, 'deg'),
                         ('slab_vz_upper', vz, 'cm/yr'),
                         ('slab_vz_deep', vzd, 'cm/yr')):
            y = y[np.isfinite(y)]
            print(f'   {nm:16s} median {np.median(y):6.2f}  '
                  f'[{y.min():6.2f}, {y.max():6.2f}] {u}')
            rows += [(key, f'{nm}_median_{u.replace("/", "per")}', f'{np.median(y):.3f}'),
                     (key, f'{nm}_min_{u.replace("/", "per")}', f'{y.min():.3f}'),
                     (key, f'{nm}_max_{u.replace("/", "per")}', f'{y.max():.3f}')]
        adv = (roll < 0).sum()
        print(f'   trench ADVANCES at {adv} of {len(roll)} snapshots')
        rows.append((key, 'snapshots_with_trench_advance', str(int(adv))))

    # shared y per row (FIGURE_STYLE encoding 1: the columns are the models,
    # so the rows must be directly comparable)
    for row in range(4):
        lo = min(ax.get_ylim()[0] for ax in axes[row, :])
        hi = max(ax.get_ylim()[1] for ax in axes[row, :])
        for ax in axes[row, :]:
            ax.set_ylim(lo, hi)

    fig.suptitle('Kinematics of the two runs through time', fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'kinematics', 'fig_kinematics_overview.png')
    fig.savefig(out, bbox_inches='tight', dpi=200)
    print('written:', out)
    tab = write_table('kinematics_overview', rows[0], rows[1:],
                      script='kinematics/fig_kinematics_overview.py',
                      figure='kinematics/fig_kinematics_overview.png',
                      models=('STD', 'WAL'),
                      meta={'plate_velocity': '|v_x| mean over top 20 km, trench->ridge',
                            'convergence': 'plate velocity + trench rollback',
                            'rollback': 'd(x_T)/dt, positive = retreat (mirrored frame)',
                            'dip': 'mean inclination of the cold anomaly, 200-400 km',
                            'processing': 'none — raw series, no detrend or smoothing'})
    print('written:', tab)


if __name__ == '__main__':
    main()
