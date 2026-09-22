"""fig_convergence_partition — how the convergence splits between plate and trench.

Writes figures/fig_convergence_partition.png and
tables/convergence_partition.csv from the committed column-profile cache.

Requested by Dan 2026-09-22, after the reconnaissance that turned up the
trench migration nobody had looked at. The trailing plate is only half
the kinematic story: the trench migrates too, and

    convergence = plate speed + trench rollback

is the rate the slab is actually being fed. The question this figure
answers is which of the three is the stable quantity.

  top     the three rates through time. Convergence in black; the plate
          and rollback terms that sum to it beneath.
  bottom  the partition fraction f = plate speed / convergence. f = 0.5
          is an even split; f = 1 means the trench is STATIONARY and the
          plate carries all of it; f > 1 means the trench ADVANCES.

WHAT IT SHOWS. In STD the convergence is buffered -- CV 12 % against
29 % for the plate speed -- because plate motion and rollback trade off
(r = -0.62), and that negative covariance cancels about half of their
combined variance. In WAL the buffering largely fails (26 % vs 35 %,
r = -0.39) and the partition is both higher and far more variable
(f = 0.74 +/- 0.21, reaching 1.10).

So the buffering is a property of the STRONG-channel model. With a stiff
asthenosphere the plate is expensive to move and slab forcing is absorbed
as rollback; with a weak channel the plate is cheap to move and the
forcing passes through into plate speed.

THE NUMBER THAT MATTERS FOR THE PAPER: the models differ by +47 % in
plate speed (1.74 vs 2.55 cm/yr) and by only +3 % in convergence (3.40
vs 3.51). "WAL is faster" is almost entirely a statement about PARTITION,
not about driving force. Any argument resting on trailing-plate velocity
alone is describing a partition.

METHOD. Rollback is d(x_T)/dt by central difference, positive = the
trench retreating into the subducting plate. Sign convention: the plate
speed is |<v_x>| and both terms are positive when they close the trench,
so they add. The x_T pick jitters by only 1.7 km (STD) / 2.0 km (WAL)
about a local quadratic, a noise floor of 0.06-0.07 cm/yr on the
derivative, so the rollback signal-to-noise is 3-11 and the variability
is real. A Savitzky-Golay (7,2) derivative gives the same means, the same
convergence CVs and correlations within 0.06 -- the result does not
depend on the differencing.

DRAFT CAPTION. The kinematic partition through both runs, STD (left) and
WAL (right). Top: the convergence rate (black) and the two terms that sum
to it, the trailing-plate speed and the trench rollback rate; legends
give each quantity's coefficient of variation. Bottom: the fraction of
the convergence carried by plate motion, with the level at which the
trench is stationary marked. The convergence rate is nearly the same in
the two models while the plate speeds differ by half as much again, so
the weak asthenospheric layer changes how the convergence is partitioned
rather than how fast the system converges.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # the shared scripts/ dir
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(_HERE))   # repo root
C_CONV = 'k'                # the sum
C_PLATE = '#1B9E77'         # trailing-plate motion
C_ROLL = '#E7298A'          # trench rollback
C_RULE = '#BFC3D1'


def rates(d, key):
    c = cpc.derive(d, key)
    t, zkm = c['t'], c['z'] / 1e3
    vp = np.abs(d[f'{key}_vx_TR'][:, zkm < 20].mean(axis=1))
    xT = d[f'{key}_xT'] / 1e3
    vt = np.gradient(xT, t) / 10.0                 # km/Myr -> cm/yr
    dt = float(np.median(np.diff(t)))
    jit = float((xT - savgol_filter(xT, 7, 2)).std())
    return dict(t=t, vp=vp, vt=vt, conv=vp + vt, f=vp / (vp + vt),
                jitter_km=jit, noise=jit / (np.sqrt(2) * dt) / 10.0,
                vt_sg=savgol_filter(xT, 7, 2, deriv=1, delta=dt) / 10.0)


def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), sharex=True,
                             gridspec_kw={'height_ratios': [2.0, 1.0]})
    rows = [('model', 'quantity', 'value')]
    cv = lambda a: 100 * a.std() / abs(a.mean())

    for col, key in enumerate(('STD', 'WAL')):
        r = rates(d, key)
        t = r['t']
        a0, a1 = axes[0, col], axes[1, col]
        for arr, c_, lw, nm in ((r['conv'], C_CONV, 2.4, 'convergence'),
                                (r['vp'], C_PLATE, 1.8, 'plate speed'),
                                (r['vt'], C_ROLL, 1.8, 'trench rollback')):
            a0.plot(t, arr, '-', color=c_, lw=lw,
                    label=f'{nm}  (CV {cv(arr):.0f} %)')
        a0.axhline(0, color='k', lw=0.8)
        a0.set_title(key, fontsize=11)
        a0.set_ylabel('Rate [cm/yr]', fontsize=10.5)
        # headroom top and bottom so the legend never lands on the curves
        # or on the zero line (it did in STD at default limits)
        lo = min(0.0, r['vt'].min())
        hi = r['conv'].max()
        a0.set_ylim(lo - 0.22 * (hi - lo), hi + 0.14 * (hi - lo))
        a0.legend(frameon=False, fontsize=8.5, loc='lower left')

        a1.plot(t, r['f'], '-', color='k', lw=1.8)
        a1.axhline(1.0, color='0.45', lw=0.9, ls='--')
        a1.axhline(0.5, color='0.7', lw=0.8, ls=':')
        a1.annotate('trench stationary', xy=(t[1], 1.0), fontsize=8,
                    color='0.35', va='bottom')
        a1.set_ylabel('$f$ = plate / convergence', fontsize=9.5)
        a1.set_xlabel('Model time [Myr]', fontsize=11)
        for ax in (a0, a1):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        rr = float(np.corrcoef(r['vp'], r['vt'])[0, 1])
        cancel = 100 * (1 - r['conv'].var() / (r['vp'].var() + r['vt'].var()))
        print(f'{key}: plate {r["vp"].mean():.2f}+/-{r["vp"].std():.2f} '
              f'(CV {cv(r["vp"]):.0f} %), rollback {r["vt"].mean():.2f}+/-'
              f'{r["vt"].std():.2f} (CV {cv(r["vt"]):.0f} %), CONVERGENCE '
              f'{r["conv"].mean():.2f}+/-{r["conv"].std():.2f} '
              f'(CV {cv(r["conv"]):.0f} %); r(plate,rollback) {rr:+.2f}; '
              f'covariance cancels {cancel:.0f} % of the variance; '
              f'f {r["f"].mean():.2f}+/-{r["f"].std():.2f}; '
              f'jitter {r["jitter_km"]:.1f} km -> noise {r["noise"]:.2f} cm/yr')
        for nm, a in (('plate_speed', r['vp']), ('rollback', r['vt']),
                      ('convergence', r['conv']), ('partition_f', r['f'])):
            rows += [(key, f'{nm}_mean', f'{a.mean():.3f}'),
                     (key, f'{nm}_sd', f'{a.std():.3f}'),
                     (key, f'{nm}_cv_percent', f'{cv(a):.1f}')]
        rows += [(key, 'corr_plate_rollback', f'{rr:.3f}'),
                 (key, 'variance_cancelled_percent', f'{cancel:.1f}'),
                 (key, 'partition_f_min', f'{r["f"].min():.3f}'),
                 (key, 'partition_f_max', f'{r["f"].max():.3f}'),
                 (key, 'xT_jitter_km', f'{r["jitter_km"]:.2f}'),
                 (key, 'rollback_noise_floor_cm_yr', f'{r["noise"]:.3f}'),
                 (key, 'rollback_sd_savgol_cm_yr', f'{r["vt_sg"].std():.3f}')]

    fig.suptitle('The kinematic partition: convergence, and how it splits between '
                 'plate motion and trench rollback', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'kinematics', 'fig_convergence_partition.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('convergence_partition', rows[0], rows[1:],
                      script='fig_convergence_partition.py',
                      figure='fig_convergence_partition.png',
                      models=('STD', 'WAL'),
                      meta={'convergence': 'plate speed + trench rollback',
                            'rollback': 'd(x_T)/dt, central difference, '
                                        'positive = trench retreats into the plate',
                            'plate_speed': '|<v_x>| averaged x_T..x_R, z < 20 km'})
    print('written:', tab)


if __name__ == '__main__':
    main()
