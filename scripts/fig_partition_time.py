"""fig_partition_time — the driving-force partition through the runs.

Writes figures/fig_partition_time.png and tables/partition.csv from the
committed column-profile cache. Partition computed by the single shared
implementation, column_profiles_cache.partition() — one depth rule (z*,
the sign change of the ridge anomaly) for every term.

Top row, one panel per model: the three components of the plate-wide
term through time —
  dynamic ridge push   what the force balance contains (integral to z*)
  back-tilt            |Delta P| * z*, contained within it; removing it
                       makes ridge push larger
  static ridge push    the density structure alone, read from the plateau
                       of the cumulative integral of the shifted profile
with the trench pull drawn for scale.

Bottom row: the back-tilt as a FRACTION of the static ridge push — the
channel dial, model against model, through time.

DRAFT CAPTION. The driving-force partition through the runs for STD
(left) and WAL (right). Upper panels: the ridge-to-first-isostatic-column
term as the models produce it (dynamic ridge push, solid), the back-tilt
contribution contained within it (shaded), and the static ridge push that
the density structure alone would supply (dashed); the trench pull is
shown for scale (grey). Lower panels: the back-tilt as a fraction of the
static ridge push. The weak asthenospheric channel in WAL suppresses the
tilt throughout, so more of the cooling topography survives as usable
driving force.
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
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'

def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    rows = [('model', 'quantity', 'value')]
    for col_i, key in enumerate(('STD', 'WAL')):
        col = C[key]
        c = cpc.derive(d, key)
        p = cpc.partition(d, key)
        t = p['t']
        z = c['z']
        cum = lambda a: np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(z))])
        tp = np.array([np.interp(zs, z, -cum(pt))
                       for pt, zs in zip(c['p_T'], p['z_star'])]) / 1e12
        dyn, tilt = p['measured'] / 1e12, p['tilt'] / 1e12
        static = p['static'] / 1e12
        ax = axes[0, col_i]
        ax.plot(t, tp, color='0.55', lw=1.6, ls=':', label='trench pull (for scale)')
        ax.fill_between(t, dyn, dyn + tilt, color=col, alpha=0.22, lw=0,
                        label='back-tilt (contained in the dynamic term)')
        ax.plot(t, dyn, '-o', color=col, lw=2.0, ms=4.5, markeredgecolor='white',
                markeredgewidth=0.5, label='dynamic ridge push (in the balance)')
        ax.plot(t, static, '--', color=col, lw=1.5, label='static ridge push')
        ax.set_title(key, fontsize=11)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.axhline(0, color='k', lw=0.8)
        frac = tilt / static
        axl = axes[1, col_i]
        axl.plot(t, frac, '-o', color=col, lw=1.8, ms=4.0, markeredgecolor='white',
                 markeredgewidth=0.5)
        axl.grid(alpha=0.25, color=C_RULE, lw=0.6)
        axl.set_ylim(0, 0.55)
        axl.set_xlabel('Model time [Myr]', fontsize=12)
        print(f'{key}: run medians — trench pull {np.median(tp):+.2f}, dynamic '
              f'{np.median(dyn):+.2f}, back-tilt {np.median(tilt):+.2f}, static '
              f'{np.median(static):+.2f} TN/m; tilt fraction {np.median(frac):.3f} '
              f'(IQR {np.percentile(frac, 25):.3f}-{np.percentile(frac, 75):.3f})')
        for name, arr in (('z_star_km', p['z_star'] / 1e3), ('trench_pull', tp),
                          ('ridge_push_dynamic', dyn), ('back_tilt', tilt),
                          ('ridge_push_static', static), ('tail_below_zstar',
                                                          p['tail'] / 1e12),
                          ('tilt_fraction_of_static', frac),
                          ('ratio_dynamic_over_trench_pull', dyn / tp)):
            rows += [(key, f'{name}_median', f'{np.median(arr):.3f}'),
                     (key, f'{name}_q1', f'{np.percentile(arr, 25):.3f}'),
                     (key, f'{name}_q3', f'{np.percentile(arr, 75):.3f}')]
    axes[0, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    axes[1, 0].set_ylabel('back-tilt /\nstatic ridge push', fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.suptitle('The plate-wide driving term through time: dynamic ridge push, '
                 'the back-tilt it contains, and the static limit', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_partition_time.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('partition', rows[0], rows[1:],
                      script='fig_partition_time.py', figure='fig_partition_time.png',
                      models=('STD', 'WAL'), meta={'depth_rule': 'sign change of the ridge anomaly (= argmax of its cumulative integral)', 'window_km': 5, 't_min_Myr': 8})
    print('written:', tab)

if __name__ == '__main__':
    main()
