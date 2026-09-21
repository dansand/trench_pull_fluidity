"""fig_trench_resultants — the resultants through time, consolidated.

Writes figures/fig_trench_resultants.png and tables/trench_resultants.csv
from the committed column-profile cache, via the shared
column_profiles_cache.resultants(). Consolidated 2026-09-21 per Dan: this
figure absorbs what fig_nd_trench_ridge carried, and the bending moment
moves to the SI (fig_trench_moment).

  (a) N_D and -V at the TRENCH. Together these show the partition at the
      trench column between the vertical load supported by shear stress
      and the horizontal load transmitted by the slab (N_D) -- the
      comparison that matters for the conventional slab-pull expectation.
  (b) N_D at the TRENCH, the FIRST ISOSTATIC COLUMN and the RIDGE. The
      trench curve repeats from (a) deliberately, so the change between
      columns can be read through time. The light fill between the trench
      and first-isostatic curves is the TRENCH PULL INCREMENT -- the
      register-Delta of N_D across the non-isostatic domain, which is the
      quantity the balance actually uses (Dan, 2026-09-21). Its run
      medians are +1.94 (STD) / +1.71 (WAL) TN/m (medians of the
      difference, not the difference of the medians); the fill is per
      model rather than plotted as a curve so the two columns it is taken
      between stay visible, and is kept faint (alpha 0.06) so the STD and
      WAL bands stay separable where they overlap.

THE PLATE-WIDE DIFFERENCE, N_D(x_T) - N_D(x_R), is tabulated but not
plotted (it was the third curve of the retired fig_nd_trench_ridge). Its
sense is counter-intuitive and is the one thing here most likely to be
quoted backwards: the net x-force on the trench-to-ridge segment from
N_D is N_D(x_R) - N_D(x_T), so the tabulated quantity is MINUS the net
force, and DRIVING (a force toward -x, trench-ward) means it is
POSITIVE. Run medians -0.94 (STD) / -2.02 (WAL) TN/m, driving in 38 % /
0 % of steps -- i.e. the plate-wide N_D difference predominantly RESISTS.
Earlier drafts quoted +0.39 / +0.24 for the ridge median and 32 % / 0 %
for the driving fraction; those predate the flow-based ridge pick, and
the ridge medians are now +0.19 / +0.13.

WHY -V AND NOT V (Dan's ruling, 2026-09-21). V = int tau_zx dz is a
resultant, and a resultant's sign is only meaningful once paired with the
outward normal of the plane it acts on (SYMBOLOGY §4.6). In the mirrored
analysis frame the subducting plate lies seaward at x > x_T, so the
trailing plate's trench-side face has outward normal -x, and the shear
traction it carries is -V, positive downward in the z-down frame. That is
the physically meaningful quantity here -- the downward load the slab
applies to the trailing plate at the trench -- and it is what is plotted.
Plotting V itself gives a negative curve for a downward pull, which reads
backwards.

This also reconciles the display with the companion register, which
defines V as positive at the trench (SYMBOLOGY §7.5). The repo keeps its
own V in the frame it extracts in; only the sign shown here is flipped,
and the table rows are named for the face load, not for V, so nothing
downstream can quote the two senses interchangeably. The alternative --
un-mirroring the whole analysis back to the source study's native frame,
where V is positive at the trench by construction -- was considered and
deferred on 2026-09-21; it remains the clean fix if the frame is ever
revisited.

Run medians (t >= 8 Myr): N_D is -0.70 (STD) / -1.83 (WAL) TN/m at the
trench, +1.27 / -0.14 at the first isostatic column and +0.19 / +0.13 at
the ridge; the downward shear load at the trench face is +1.60 / +1.11.

Styling per FIGURE_STYLE.md: model by brand colour, quantity or column by
linestyle.

DRAFT CAPTION. Stress resultants through the runs for STD (navy) and WAL
(magenta). (a) At the trench column, the normal-stress-difference
resultant N_D (solid) and the downward shear load carried by the
trailing plate's trench-side face, -V (dashed): the horizontal load
transmitted along the plate against the vertical load carried by shear.
(b) N_D at the trench (solid), the first isostatic column (dashed) and
the ridge (dotted), showing how the resultant changes between columns
through time. The shaded band between the trench and first isostatic
curves is the trench pull increment, the change in N_D across the
non-isostatic domain.
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
        # the shear load on the trailing plate's trench-side face (outward
        # normal -x), positive downward -- see the module docstring
        face_load = -r['v_T']
        axes[0].plot(t[m], r['nd_T'][m] * 1e-12, '-', label=f'{key}  $N_D(x_T)$', **kw)
        axes[0].plot(t[m], face_load[m] * 1e-12, '--', label=f'{key}  $-V(x_T)$', **kw)
        # the gap between the trench and first-isostatic curves IS the
        # trench pull increment, Delta N_D across the non-isostatic domain
        axes[1].fill_between(t[m], r['nd_T'][m] * 1e-12, r['nd_I'][m] * 1e-12,
                             color=col, alpha=0.06, lw=0, zorder=0)
        # the TRENCH curve carries the argument; x_I and the ridge are
        # context, so they are faded and drawn beneath rather than
        # competing for attention (Dan, 2026-09-21). Plot order keeps the
        # trench first in the legend; zorder puts it on top.
        axes[1].plot(t[m], r['nd_T'][m] * 1e-12, '-', label=f'{key}  $N_D(x_T)$',
                     zorder=3, **kw)
        axes[1].plot(t[m], r['nd_I'][m] * 1e-12, '--', label=f'{key}  $N_D(x_I)$',
                     color=col, lw=1.9, alpha=0.45, zorder=2)
        axes[1].plot(t[m], r['nd_R'][m] * 1e-12, ':', label=f'{key}  $N_D(x_R)$',
                     color=col, lw=2.1, alpha=0.45, zorder=2)
        med = lambda a: np.median(a[m]) / 1e12
        print(f'{key}: N_D trench {med(r["nd_T"]):+.2f}, x_I {med(r["nd_I"]):+.2f}, '
              f'ridge {med(r["nd_R"]):+.2f} TN/m; downward shear load at the '
              f'trench face {med(face_load):+.2f} (extracted V {med(r["v_T"]):+.2f})')
        for name, arr in (('nd_trench', r['nd_T']), ('nd_first_isostatic', r['nd_I']),
                          ('nd_ridge', r['nd_R']),
                          ('shear_load_trench_face_down', face_load),
                          ('d_nd_trench_to_first_isostatic',
                           r['nd_I'] - r['nd_T'])):
            a = arr[m] / 1e12
            rows += [(key, f'{name}_median_TNm', f'{np.median(a):.3f}'),
                     (key, f'{name}_q1_TNm', f'{np.percentile(a, 25):.3f}'),
                     (key, f'{name}_q3_TNm', f'{np.percentile(a, 75):.3f}')]
        nd_t, v_t = r['nd_T'][m], r['v_T'][m]
        # the plate-wide difference across the trench-to-ridge segment.
        # CAREFUL with its sense: the net x-force on that segment from N_D
        # is N_D(x_R) - N_D(x_T), so the quantity below is MINUS the net
        # force. Driving means a force toward -x (trench-ward), which is
        # d_tr > 0. This is the opposite of the reading that the plotted
        # sign suggests, hence the explicit row name.
        d_tr = (r['nd_T'] - r['nd_R'])[m]
        rows += [(key, 'nd_trench_tension_like_fraction', f'{(nd_t > 0).mean():.3f}'),
                 (key, 'nd_ridge_always_tension_like',
                  str(bool((r['nd_R'][m] > 0).all()))),
                 (key, 'nd_trench_minus_ridge_median_TNm',
                  f'{np.median(d_tr) / 1e12:.3f}'),
                 (key, 'nd_trench_minus_ridge_driving_fraction',
                  f'{(d_tr > 0).mean():.3f}'),
                 (key, 'abs_v_over_abs_nd_trench_median',
                  f'{np.median(np.abs(v_t) / np.abs(nd_t)):.2f}')]
    axes[0].set_title('(a) at the trench: horizontal load ($N_D$) and the downward '
                      'shear load on the trench-side face ($-V$)', fontsize=10.5)
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
                            'V_sense': 'extracted V = int tau_zx dz in the mirrored '
                                       'analysis frame (negative of the companion '
                                       'register V)',
                            'shear_load_sense': 'shear_load_trench_face_down = -V, the '
                                                'traction resultant on the trailing '
                                                "plate's trench-side face (outward "
                                                'normal -x), positive DOWNWARD'})
    print('written:', tab)

if __name__ == '__main__':
    main()
