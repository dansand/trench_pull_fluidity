"""fig_trench_resultants — the resultants through time, consolidated.

Writes figures/fig_trench_resultants.png and tables/trench_resultants.csv
from the committed column-profile cache, via the shared
column_profiles_cache.resultants(). Consolidated 2026-09-21 per Dan: this
figure absorbs what fig_nd_trench_ridge carried, and the bending moment
moves to the SI (fig_trench_moment).

  (a) N_D and V at the TRENCH. Together these show the partition at the
      trench column between the vertical load supported by shear stress
      (V) and the horizontal load transmitted by the slab (N_D) -- the
      comparison that matters for the conventional slab-pull expectation.
  (b) N_D at the TRENCH, the FIRST ISOSTATIC COLUMN and the RIDGE. The
      trench curve repeats from (a) deliberately, so the change between
      columns can be read through time.

Run medians (t >= 8 Myr): N_D is -0.70 (STD) / -1.83 (WAL) TN/m at the
trench, +1.27 / -0.14 at the first isostatic column and +0.19 / +0.13 at
the ridge; V at the trench is -1.60 / -1.11.

Styling per FIGURE_STYLE.md: model by brand colour, quantity or column by
linestyle.

Register note on V (SYMBOLOGY §7.5): the plotted V is the extracted
V = int tau_zx dz at the trench column -- the repo's engineering sense,
which is the NEGATIVE of the companion register's V. Any prose quoting a
V relation must say which V it means.

DRAFT CAPTION. Stress resultants through the runs for STD (navy) and WAL
(magenta). (a) At the trench column, the normal-stress-difference
resultant N_D (solid) and the vertical shear resultant V (dashed): the
horizontal load transmitted along the plate against the vertical load
carried by shear. (b) N_D at the trench (solid), the first isostatic
column (dashed) and the ridge (dotted), showing how the resultant changes
between columns through time.
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
T_MIN_MYR = 8.0
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'

def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    rows = [('model', 'quantity', 'value')]
    for key in ('STD', 'WAL'):
        col = C[key]
        r = cpc.resultants(d, key)
        t = r['t']
        m = t >= T_MIN_MYR
        kw = dict(color=col, lw=1.9)
        axes[0].plot(t[m], r['nd_T'][m] * 1e-12, '-', label=f'{key}  $N_D(x_T)$', **kw)
        axes[0].plot(t[m], r['v_T'][m] * 1e-12, '--', label=f'{key}  $V(x_T)$', **kw)
        axes[1].plot(t[m], r['nd_T'][m] * 1e-12, '-', label=f'{key}  $N_D(x_T)$', **kw)
        axes[1].plot(t[m], r['nd_I'][m] * 1e-12, '--', label=f'{key}  $N_D(x_I)$', **kw)
        axes[1].plot(t[m], r['nd_R'][m] * 1e-12, ':', label=f'{key}  $N_D(x_R)$',
                     color=col, lw=2.1)
        med = lambda a: np.median(a[m]) / 1e12
        print(f'{key}: N_D trench {med(r["nd_T"]):+.2f}, x_I {med(r["nd_I"]):+.2f}, '
              f'ridge {med(r["nd_R"]):+.2f} TN/m; V trench {med(r["v_T"]):+.2f}')
        for name, arr in (('nd_trench', r['nd_T']), ('nd_first_isostatic', r['nd_I']),
                          ('nd_ridge', r['nd_R']), ('v_trench', r['v_T'])):
            a = arr[m] / 1e12
            rows += [(key, f'{name}_median_TNm', f'{np.median(a):.3f}'),
                     (key, f'{name}_q1_TNm', f'{np.percentile(a, 25):.3f}'),
                     (key, f'{name}_q3_TNm', f'{np.percentile(a, 75):.3f}')]
        nd_t, v_t = r['nd_T'][m], r['v_T'][m]
        rows += [(key, 'nd_trench_tension_like_fraction', f'{(nd_t > 0).mean():.3f}'),
                 (key, 'nd_ridge_always_tension_like',
                  str(bool((r['nd_R'][m] > 0).all()))),
                 (key, 'abs_v_over_abs_nd_trench_median',
                  f'{np.median(np.abs(v_t) / np.abs(nd_t)):.2f}')]
    axes[0].set_title('(a) at the trench: horizontal load ($N_D$) and vertical '
                      'load carried by shear ($V$)', fontsize=10.5)
    axes[1].set_title('(b) $N_D$ at the trench, first isostatic column and ridge',
                      fontsize=10.5)
    axes[1].set_xlabel('Model time [Myr]', fontsize=12)
    for ax in axes:
        ax.axhline(0, color='k', lw=1.4)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.set_ylabel('Force per unit distance [TN/m]', fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='best')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_trench_resultants.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('trench_resultants', rows[0], rows[1:],
                      script='fig_trench_resultants.py',
                      figure='fig_trench_resultants.png', models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM, 't_min_Myr': T_MIN_MYR,
                            'V_sense': 'engineering, int tau_zx dz (negative of the '
                                       'companion register V)'})
    print('written:', tab)

if __name__ == '__main__':
    main()
