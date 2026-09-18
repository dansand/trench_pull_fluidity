"""fig_balance_absolute — the trailing-plate balance in the ABSOLUTE form.

Writes figures/fig_balance_absolute.png. One panel per model, at the
reference snapshot (t = 40 Myr, conventions §4b), using the shared
fig_balance_snapshot.compute() so both forms of the figure come from one
computation.

Where fig_balance_snapshot shows the pure Delta form (every term zero at
the trench), this figure applies the shift used in the notebook: the
topographic pressure term is offset by N_D(x_T), so the black curve is
the ACTUAL normal-stress-difference resultant N_D(x) rather than its
change. The point is to read the raw column value of N_D — in particular
at the first isostatic column, where the plate-wide partition is taken.

Curves, in the balance palette (FIGURE_STYLE.md):
    black        N_D(x), the resultant itself
    blue thick   [Delta GPE*(x) + N_D(x_T)], the topographic pressure term
                 carried onto the same offset
    red          F_B(x), the accumulated basal traction
    green dashed N_D(x) - [Delta GPE*(x) + N_D(x_T)] + F_B(x), the closure

Seaward side only; x_T, x_I and x_R marked.

DRAFT CAPTION. The trailing-plate force balance at the reference
snapshot (t = 40 Myr) for STD (left) and WAL (right), plotted so that the
normal-stress-difference resultant appears as its actual value rather
than as a change: the topographic pressure term is offset by N_D at the
trench. Black, N_D(x); blue, the topographic pressure term on the same
offset; red, the accumulated basal traction; green dashed, the closure.
The value of N_D at the first isostatic column is marked.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_balance_snapshot import compute
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, sharex=True)
    rows = [('model', 'quantity', 'value')]
    for ax, key in zip(axes, ('STD', 'WAL')):
        r = compute(key)
        x, xT = r['x'], r['xT']
        xkm = (x - xT) / 1e3
        xi_km, xr_km = (r['xI'] - xT) / 1e3, (r['xR'] - xT) / 1e3
        Fd, FB = r['Fd'], r['FB']
        bracket = r['d_gpe'] + r['Fd_T']            # the notebook's shift
        residual = Fd - bracket + FB
        ax.plot(xkm, residual * 1e-12, color='g', ls='--', lw=3,
                label=r'$N_D - [\Delta\mathrm{GPE}^{*} + N_D(x_T)] + F_B$')
        ax.plot(xkm, FB * 1e-12, color='red', lw=2, label=r'$F_B(x)$')
        ax.plot(xkm, bracket * 1e-12, color='b', lw=4, alpha=0.6,
                label=r'$\Delta\mathrm{GPE}^{*}(x) + N_D(x_T)$')
        ax.plot(xkm, Fd * 1e-12, color='k', lw=1.5, label=r'$N_D(x)$')
        ax.axhline(0, color='k', lw=1.4)
        for xc, lab in ((0, r'$x_T$'), (xi_km, r'$x_I$'), (xr_km, r'$x_R$')):
            ax.axvline(xc, color='k', lw=0.5)
            ax.annotate(lab, xy=(xc, 1), xycoords=('data', 'axes fraction'),
                        xytext=(4, -4), textcoords='offset points',
                        ha='left', va='top', fontsize=11, fontweight='bold',
                        color='#002147' if lab == r'$x_T$' else 'k')
        nd_T = float(r['Fd_T'])
        nd_I = float(np.interp(r['xI'], x, Fd))
        nd_R = float(np.interp(r['xR'], x, Fd))
        ax.plot([xi_km], [nd_I * 1e-12], 'o', color='k', ms=7,
                markeredgecolor='white', markeredgewidth=1.2, zorder=5)
        ax.annotate(f'$N_D(x_I)$ = {nd_I/1e12:+.2f} TN/m',
                    xy=(xi_km, nd_I * 1e-12), xytext=(28, -34),
                    textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='-', color='0.4', lw=0.8))
        ax.set_title(f'{key},  $t = {r["t_myr"]:.1f}$ Myr', fontsize=12)
        ax.set_xlabel('Distance from trench [km]', fontsize=11)
        ax.set_xlim(-50, 3500)
        ax.set_ylim(-2.6, 3.4)
        print(f'{key}: N_D at x_T {nd_T/1e12:+.2f}, at x_I {nd_I/1e12:+.2f}, '
              f'at x_R {nd_R/1e12:+.2f} TN/m')
        rows += [(key, 'nd_trench_TNm', f'{nd_T/1e12:.3f}'),
                 (key, 'nd_first_isostatic_TNm', f'{nd_I/1e12:.3f}'),
                 (key, 'nd_ridge_TNm', f'{nd_R/1e12:.3f}'),
                 (key, 'reference_epoch_Myr', f'{r["t_myr"]:.1f}')]
    axes[0].set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    axes[0].legend(loc='lower right', fontsize=8, ncol=2)
    fig.suptitle('The trailing-plate balance in absolute form: the topographic pressure '
                 'term is offset by $N_D(x_T)$,\nso the black curve is $N_D$ itself',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_balance_absolute.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('balance_absolute', rows[0], rows[1:],
                      script='fig_balance_absolute.py',
                      figure='fig_balance_absolute.png', models=('STD', 'WAL'),
                      meta={'form': 'absolute (GPE* offset by N_D(x_T))',
                            'reference_epoch_Myr': 40.0})
    print('written:', tab)

if __name__ == '__main__':
    main()
