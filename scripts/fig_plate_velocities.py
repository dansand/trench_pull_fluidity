"""fig_plate_velocities — plate and trench kinematics through the runs.

Writes figures/fig_plate_velocities.png and tables/plate_velocities.csv.
Diagnostic figure: the point is to see whether either run changes regime,
so that any structure in the force partition can be checked against the
kinematics rather than read as a trend.

One panel per model. Three curves each:
  subducting plate   v_sp from the committed time-evolution cache
  overriding plate   v_op from the same cache
  trench migration   d x_T / dt, differentiated from the trench positions
                     in the column-profile cache (negative = retreat in
                     the analysis frame, i.e. moving with the subducting
                     plate; positive = advance)
plus the convergence rate |v_sp - v_op| as a light line.

Sign convention: the analysis frame has the subducting plate on the
right, so plate motion toward the trench is NEGATIVE.

DRAFT CAPTION. Plate and trench kinematics for STD (left) and WAL
(right): subducting-plate velocity (solid), overriding-plate velocity
(dashed), trench migration rate (dotted) and convergence rate (grey).
Velocities are negative in the direction of subducting-plate motion.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'
MYR = 1e6 * 3.15576e7          # seconds per Myr

def main():
    d = cpc.load()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, sharex=True)
    rows = [('model', 'quantity', 'value')]
    for ax, key in zip(axes, ('STD', 'WAL')):
        col = C[key]
        te = np.load(os.path.join(ROOT, 'notebooks', 'outputs',
                                  f'time_evolution_{key}.npz'))
        t = te['t_yr'] / 1e6
        m = t >= 8.0
        v_sp, v_op = te['v_sp'][m], te['v_op'][m]
        # trench migration from the column cache picks (cm/yr)
        tc, xT = d[f'{key}_t'], d[f'{key}_xT']
        v_tr = np.gradient(gaussian_filter1d(xT, 1), tc) * 100.0 / 1e6
        ax.axhline(0, color='k', lw=1.2)
        ax.plot(t[m], v_sp, '-o', color=col, lw=2.0, ms=4, markeredgecolor='white',
                markeredgewidth=0.5, label='subducting plate')
        ax.plot(t[m], v_op, '--', color=col, lw=1.6, label='overriding plate')
        ax.plot(tc, v_tr, ':', color=col, lw=1.8, label='trench migration')
        ax.plot(t[m], -np.abs(v_sp - v_op), '-', color='0.6', lw=1.2,
                label='convergence rate')
        ax.set_title(key, fontsize=11)
        ax.set_xlabel('Model time [Myr]', fontsize=11)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        print(f'{key}: v_sp median {np.median(v_sp):+.2f} '
              f'(range {v_sp.min():+.2f}..{v_sp.max():+.2f}) cm/yr; '
              f'v_op median {np.median(v_op):+.2f}; '
              f'trench migration median {np.median(v_tr):+.2f} '
              f'(range {v_tr.min():+.2f}..{v_tr.max():+.2f}); '
              f'convergence median {np.median(np.abs(v_sp - v_op)):.2f}')
        for name, arr in (('v_sp', v_sp), ('v_op', v_op), ('v_trench', v_tr),
                          ('convergence', np.abs(v_sp - v_op))):
            rows += [(key, f'{name}_median_cmyr', f'{np.median(arr):.3f}'),
                     (key, f'{name}_min_cmyr', f'{np.min(arr):.3f}'),
                     (key, f'{name}_max_cmyr', f'{np.max(arr):.3f}')]
    axes[0].set_ylabel('velocity [cm/yr]\n(negative = subducting-plate motion)',
                       fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    fig.suptitle('Plate and trench kinematics — a check for regime changes', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_plate_velocities.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    os.makedirs(os.path.join(ROOT, 'tables'), exist_ok=True)
    tab = os.path.join(ROOT, 'tables', 'plate_velocities.csv')
    with open(tab, 'w') as f:
        f.write('\n'.join(','.join(x) for x in rows) + '\n')
    print('written:', tab)

if __name__ == '__main__':
    main()
