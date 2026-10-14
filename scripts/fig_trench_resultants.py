"""

⚠ SI FIGURE as of 2026-10-09 (Dan). It was main-text Figure 6. The
column-wise N_D panel moved into fig_budget_time_detrended, which now
carries the plate-wide balance and the column-wise resultant through
time as one figure; V and M were interrupting the main force-balance
argument. This figure keeps them as secondary diagnostics in the SI,
placed after the stress-state figure. Its content is unchanged apart
from today's +V correction.
fig_trench_resultants — the resultants through time, consolidated.

Writes figures/fig_trench_resultants.png and tables/trench_resultants.csv
from the committed column-profile cache, via the shared
column_profiles_cache.resultants(). Consolidated 2026-09-21 per Dan: this
figure absorbs what fig_nd_trench_ridge carried, and the bending moment
moves to the SI (fig_trench_moment).

  (a) N_D and V at the TRENCH. Together these show the partition at the
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
POSITIVE. Run medians -1.13 (STD) / -2.12 (WAL) TN/m, driving in 30 % /
0 % of steps -- i.e. the plate-wide N_D difference predominantly RESISTS.

⚠ THE RIDGE MEDIAN HAS MOVED TWICE AND THE SECOND MOVE WENT UNRECORDED.
The flow-based ridge pick took it from +0.39 / +0.24 to +0.19 / +0.13;
the CREST REFINEMENT of 2026-09-27 (conventions §3.4, commit 5b24c8f)
then took it back up to +0.37 / +0.22, and the plate-wide difference with
it (-0.94 -> -1.13, driving 38 % -> 30 %). Only the first move reached
PAPER_PLAN. The committed tables/trench_resultants.csv is the authority;
PAPER_PLAN §3 and W30 still instruct the writing agent to use the
intermediate +0.19 / +0.13, which no cache has produced since September.

WHY +V (settled by Dan, 2026-10-09). V = int tau_zx dz is a resultant,
and a resultant's sign only means something once paired with the outward
normal of the plane it acts on (SYMBOLOGY §4.6). The trailing plate lies
seaward at x < x_T, so its trench-side face has outward normal +x, and
the traction that face carries is t_z = tau_zx n_x = +tau_zx -- positive
downward in the z-down frame. The panel therefore plots V itself,
+1.61 / +1.11 TN/m.

The figure plotted -V until 2026-10-09, which was right in the MIRRORED
frame: there the plate sat at x > x_T, the normal was -x, and V came out
negative, so negative times negative gave a positive downward load. The
frame flipped on 2026-09-28 and this sign did not follow it, so the curve
read -1.60 TN/m under the label "downward shear load".

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
trench, +1.27 / -0.14 at the first isostatic column and +0.37 / +0.22 at
the ridge; the extracted V(x_T) is +1.60 / +1.11, and the curve plotted
as -V is therefore -1.60 / -1.11.

⚠ UNRESOLVED, flagged 2026-10-09 and NOT decided here. The prose below
calls the plotted -V "the downward shear load", which was true in the
mirrored frame, where the trailing plate lay seaward at x > x_T and V
came out negative. In the native frame (W31) the trailing plate is at
x < x_T, V is positive at every step, and -V is negative -- so either the
label or the plotted sign is now wrong. This is the same class of problem
as the three sign registers in W28 and needs Dan's ruling, not a silent
flip.

Styling per FIGURE_STYLE.md: model by brand colour, quantity or column by
linestyle.

DRAFT CAPTION. Stress resultants through the STD and WAL runs, all
quantities integrated to z_c and evaluated on +-5 km column windows.
(a) The three resultants at the trench: the normal-stress-difference
resultant N_D(x_T) (solid), the downward shear load on the trailing
plate's trench-side face V(x_T) (dashed), and the bending moment
-M(x_T) about the neutral plane (dash-dot, right axis). The shear load
is the traction on that face: its outward normal is +x, so
t_z = tau_zx n_x = +tau_zx and positive values act downward. The moment is plotted as -M because the extracted moment
is negative at every step of both runs. The left and right axes carry
different dimensions -- TN/m and 1e17 N -- but are given the same
numerical range and the same symmetric limits about a common zero, so
that the three curves share one baseline and their variations are drawn
to a common scale. Median values are N_D(x_T) -0.70 and -1.83 TN/m,
V(x_T) +1.61 and +1.11 TN/m, and M(x_T) -1.17 and -1.02 x 1e17 N for
STD and WAL respectively. (b) N_D at the trench (solid), the first
isostatic column (dashed) and the ridge (dotted), with the shading
marking the increment N_D(x_I) - N_D(x_T). Medians are +1.27 and -0.14
TN/m at the first isostatic column, +0.37 and +0.22 TN/m at the ridge,
and +1.94 and +1.71 TN/m for the increment; these are medians of the
per-step difference, not differences of the column medians. Both panels
use snapshots at t >= 8 Myr (conventions §7.5).
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
from column_profiles_cache import T_MIN_MYR   # the only spin-up cut
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_RULE = '#BFC3D1'

def main():
    d = cpc.load()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.6), sharex=True)
    # BENDING MOMENT back in panel (a) (Dan, 2026-09-25): that panel is the
    # trench itself -- the horizontal load, the vertical shear load and the
    # moment are the three resultants acting there. M is in N, not N/m, so
    # it needs its own axis.
    a0b = axes[0].twinx()
    rows = [('model', 'quantity', 'value')]
    for key in ('STD', 'WAL'):
        col = C[key]
        r = cpc.resultants(d, key)
        t = r['t']
        m = t >= T_MIN_MYR
        kw = dict(color=col, lw=1.9)
        # the shear load on the trailing plate's trench-side face (outward
        # normal -x), positive downward -- see the module docstring
        # THE DOWNWARD SHEAR LOAD IS +V IN THE NATIVE FRAME.
        # The trailing plate lies seaward at x < x_T, so its trench-side
        # face has outward normal +x and the traction it carries is
        # t_z = tau_zx n_x = +tau_zx, positive downward with z down.
        # This was -V, correct in the MIRRORED frame where the plate sat
        # at x > x_T and the normal was -x (negative normal times the then
        # negative V gave a positive, downward load). The frame flipped on
        # 2026-09-28 and the sign did not follow, so the figure plotted
        # -1.60 TN/m under the label "downward shear load" (Dan, 2026-10-09).
        face_load = r['v_T']
        axes[0].plot(t[m], r['nd_T'][m] * 1e-12, '-', label=f'{key}  $N_D(x_T)$', **kw)
        axes[0].plot(t[m], face_load[m] * 1e-12, '--', label=f'{key}  $V(x_T)$', **kw)
        # -M, not M: the extracted moment is negative at every step in both
        # runs (-0.67 to -1.77 x 1e17 N), so plotting its negation keeps the
        # panel's quantities positive alongside +V. The sign is constant
        # through both runs, so nothing is lost by the negation.
        a0b.plot(t[m], -r['m_T'][m] / 1e17, '-.', color=col, lw=1.7, alpha=0.85,
                 label=f'{key}  $-M(x_T)$')
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
              f'trench face {med(face_load):+.2f} (extracted V {med(r["v_T"]):+.2f}); '
              f'M(x_T) {np.median(r["m_T"][m])/1e17:+.2f} x 1e17 N')
        rows.append((key, 'moment_trench_median_1e17N',
                     f'{np.median(r["m_T"][m]) / 1e17:.3f}'))
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
    # Panel titles name the SUBJECT only; what each curve is belongs in the
    # legend and the caption, not spelled out across the top of the figure
    # (Dan, 2026-09-25).
    axes[0].set_title('(a) Stress resultants at the trench', fontsize=11)
    a0b.set_ylabel(r'Bending moment, $-M(x_T)$ [$10^{17}$ N]', fontsize=11)
    # SHARED NUMERICAL RANGE, SYMMETRIC ABOUT ZERO (Dan, 2026-09-25).
    # The two axes carry different dimensions -- TN/m and 1e17 N -- and are
    # nonetheless given the SAME numbers and the same symmetric limits, so
    # that the zero line is common and the three curves can be compared for
    # RELATIVE VARIABILITY. That is the claim the panel makes: the shear
    # load and the bending moment barely move while N_D sweeps from
    # tension-like to strongly compression-like. An axis pair chosen to
    # separate the curves into bands (tried first) hides exactly that, and
    # free-scaling the moment exaggerates its variation to fill the panel.
    # The coincidence of numbers is deliberate and declared, not implied
    # equivalence: 1 TN/m and 1e17 N are not the same quantity.
    fl, fh = axes[0].get_ylim()
    A = max(abs(fl), abs(fh))
    axes[0].set_ylim(-A, A)
    a0b.set_ylim(-A, A)
    axes[1].set_title('(b) $N_D$ at the three bounding columns',
                      fontsize=11)
    axes[1].set_xlabel('Model time [Myr]', fontsize=12)
    for ax in axes:
        ax.axhline(0, color='k', lw=1.4)
        ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        ax.set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    # panel (a)'s legend must gather both axes' handles
    h0, l0 = axes[0].get_legend_handles_labels()
    hb, lb = a0b.get_legend_handles_labels()
    axes[0].legend(h0 + hb, l0 + lb, frameon=False, fontsize=8.5, ncol=3,
                   loc='lower left')
    axes[1].legend(frameon=False, fontsize=8.5, ncol=2, loc='best')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_trench_resultants.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('trench_resultants', rows[0], rows[1:],
                      script='fig_trench_resultants.py',
                      figure='fig_trench_resultants.png', models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM, 't_min_Myr': T_MIN_MYR,
                            'V_sense': 'extracted V = int tau_zx dz in the '
                                       "model's own frame, which carries the "
                                       'sense the companion register defines: '
                                       'positive at the trench, dV/ds = -rho g w',
                            'shear_load_sense': 'shear_load_trench_face_down = +V, '
                                                'the traction resultant on the '
                                                "trailing plate's trench-side face "
                                                '(outward normal +x: the plate lies '
                                                'seaward at x < x_T), positive '
                                                'DOWNWARD'})
    print('written:', tab)

if __name__ == '__main__':
    main()
