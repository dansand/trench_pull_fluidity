"""fig_balance_linear — the snapshot balance on a linear distance axis.

Writes figures/fig_balance_linear.png and tables/balance_linear.csv.

SI FIGURE (Dan, 2026-10-10). A direct linear-coordinate counterpart of the
main-text balance figure: same quantities, same colours, same reference
time, same computation (balance_chain.compute -- nothing is recalculated
here). It belongs in the methodological-validation group and is a CHECK ON
SPATIAL INTERPRETATION, not a further analysis.

WHY IT IS NEEDED. The main figure uses a square-root distance axis, which
is justified there: x_I sits ~90 km from the trench while the ridge is at
~2900 km, so on a linear axis the non-isostatic domain -- where the entire
trench pull is generated -- occupies under 3 per cent of the plot width and
is unreadable. The cost is that the transformation visually suppresses the
order-of-magnitude difference between the two domains, and a reader who
sees only the square-root version may carry away the wrong sense of where
each term accumulates.

THE ROWS ARE THE MAIN FIGURE'S ROWS, less one. Main-text rows are: the
free surface; the fundamental form; the decomposed three-term form; and
N_D itself. This figure carries the first, third and fourth. The
fundamental form is dropped because it restates the decomposed one
(Dan, 2026-10-14) and the SI does not need both.

⚠ IT SHOWS NOTHING THE MAIN FIGURE DOES NOT. That is the whole point: the
axis is the only difference, so a reader can confirm the square-root
transformation hid nothing. A panel here that has no counterpart in the
main figure defeats the comparison. One was added and removed -- a
normalised "fraction of the plate-wide value" row with reach markers,
which was a new analysis wearing an SI figure's clothes. Its NUMBERS are
kept (the manuscript cites gpe_fraction_by_x_I and FB_fraction_by_x_I),
they simply do not need a panel.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import balance_chain as bc
from tables_io import write_table

ROOT = os.path.dirname(_HERE)
C_RULE = '#BFC3D1'
# DISPLAY LIMIT ONLY. The plate-wide endpoint is the RIDGE, which is at
# 2978 (STD) / 2816 (WAL) km -- both inside this axis. Every fraction below
# is referenced to the ridge, never to the axis limit: 3000 km lies 22 km
# beyond the STD ridge but 184 km beyond the WAL one, and Delta GPE* is
# already decaying on the far flank, so normalising at the limit read
# 3.280 / 2.384 TN/m against the ridge values of 3.491 / 3.198 and made the
# x_I fraction 0.62 / 0.71 instead of 0.58 / 0.53. Caught 2026-10-10.
XMAX_KM = 3000.0


def reach(s_km, y, frac, end):
    """Seaward distance at which |y| first reaches `frac` of its ridge value."""
    if abs(end) < 1e-9:
        return np.nan
    j = np.where(np.abs(y) >= frac * abs(end))[0]
    return float(s_km[j[0]]) if len(j) else np.nan


def main():
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 8.6), sharex=True,
                             gridspec_kw={'height_ratios': [0.7, 1.3, 1.0]})
    rows = [('model', 'quantity', 'value')]
    for col, key in enumerate(('STD', 'WAL')):
        r = bc.compute(key, verbose=False)
        x, xT, t = r['x'], r['xT'], r['t_myr']
        s = (xT - x) / 1e3                       # seaward distance [km]
        o = np.argsort(s); s = s[o]
        g = lambda a: a[o]
        vis = (s >= 0) & (s <= XMAX_KM)
        sI, sR = (xT - r['xI']) / 1e3, (xT - r['xR']) / 1e3

        a0 = axes[0, col]
        a0.plot(s[vis], g(r['w'])[vis], 'k', lw=1.4)
        a0.set_title(f'({"ab"[col]})  {key},  $t$ = {t:.0f} Myr', fontsize=11,
                     loc='left')
        a0.invert_yaxis()

        a1 = axes[1, col]
        for y, c, lw, lab in ((g(r['d_gpe']), 'b', 3.5, r'$\Delta\mathrm{GPE}^{*}$'),
                              (g(r['d_Fd']), 'k', 1.6, r'$\Delta N_D$'),
                              (g(r['FB']), 'red', 2.0, '$F_B$'),
                              (g(r['res_pin']), 'g', 2.5, 'closure')):
            a1.plot(s[vis], y[vis] * 1e-12, color=c, lw=lw,
                    ls='--' if lab == 'closure' else '-',
                    alpha=0.6 if c == 'b' else 1.0, label=lab)

        # ROW 3 = THE MAIN FIGURE'S ROW 4: N_D itself, not a Delta form.
        # The reach/fraction numbers below are still measured and tabulated
        # -- the manuscript cites them -- but they are no longer plotted.
        a2 = axes[2, col]
        a2.plot(s[vis], g(r['Fd'])[vis] * 1e-12, color='k', lw=1.8,
                label='$N_D(x)$')
        a2.axhline(0, color='k', lw=0.5)
        a2.plot(0, r['Fd_T'] * 1e-12, 'o', color='k', ms=5, zorder=6)
        a2.set_xlabel('Seaward distance from the trench [km]', fontsize=10)
        for y, c in ((g(r['d_gpe']), 'b'), (g(r['FB']), 'red')):
            end = float(np.interp(sR, s, y))
            nm = 'gpe' if c == 'b' else 'FB'
            rows.append((key, f'{nm}_reach50_km',
                         f'{reach(s[vis], y[vis], 0.5, end):.0f}'))
            rows.append((key, f'{nm}_plate_wide_TNm', f'{end*1e-12:.3f}'))
            rows.append((key, f'{nm}_fraction_by_x_I',
                         f'{float(np.interp(sI, s[vis], y[vis] / end)):.3f}'))

        for ax in axes[:, col]:
            ax.set_xlim(0, XMAX_KM)
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
            for xc in (0, sI, sR):
                ax.axvline(xc, color='0.55', lw=0.8)
        rows += [(key, 'x_I_seaward_km', f'{sI:.0f}'),
                 (key, 'x_R_seaward_km', f'{sR:.0f}'),
                 (key, 'trench_pull_TNm',
                  f'{float(np.interp(sI, s, g(r["d_gpe"])))*1e-12:.3f}'),
                 (key, 'reference_Myr', f'{t:.1f}')]
        eg = float(np.interp(sR, s, g(r['d_gpe'])))
        eb = float(np.interp(sR, s, g(r['FB'])))
        print(f'{key}: x_I at {sI:.0f} km, ridge at {sR:.0f} km | '
              f'dGPE* reaches 50/90 % by '
              f'{reach(s[vis], g(r["d_gpe"])[vis], .5, eg):.0f}/'
              f'{reach(s[vis], g(r["d_gpe"])[vis], .9, eg):.0f} km; '
              f'F_B by {reach(s[vis], g(r["FB"])[vis], .5, eb):.0f}/'
              f'{reach(s[vis], g(r["FB"])[vis], .9, eb):.0f} km')

    axes[0, 0].set_ylabel('$w$ [m]\n(positive down)', fontsize=9.5)
    axes[1, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=10)
    axes[2, 0].set_ylabel('$N_D$  [TN/m]', fontsize=10)
    axes[1, 0].legend(frameon=False, fontsize=9, ncol=2, loc='upper left')
    fig.suptitle('The balance on a linear distance axis',
                 fontsize=10.5, color='0.35')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_balance_linear.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('written:', out)
    print('written:', write_table(
        'balance_linear', rows[0], rows[1:], script='fig_balance_linear.py',
        figure='fig_balance_linear.png', models=('STD', 'WAL'),
        meta={'role': 'linear-axis counterpart of the main balance figure; a '
                      'check on spatial interpretation, not a new analysis',
              'computation': 'balance_chain.compute, identical to the main figure',
              'plate_wide': 'every fraction is referenced to the value at the '
                            'RIDGE column, not to the 3000 km display limit',
              'rows': 'main-text rows 1, 3 and 4 (free surface; decomposed '
                      'balance; N_D). The fundamental form is dropped as a '
                      'restatement of the decomposed one (Dan, 2026-10-14)',
              'not_plotted': 'the reach and x_I-fraction numbers are measured '
                             'and tabulated but have no panel -- they are '
                             'cited by the manuscript, not displayed here',
              'zc_km': bc.ZC / 1e3}))


if __name__ == '__main__':
    main()
