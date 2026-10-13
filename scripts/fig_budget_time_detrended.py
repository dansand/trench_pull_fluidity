"""fig_budget_time_detrended — the three-term budget through time, levels and residual.

Writes figures/fig_budget_time_detrended.png and
tables/budget_time_detrended.csv from the committed column-profile cache.

Requested by Dan 2026-09-24: fig_budget_time with a detrended panel added
beneath each model, same lines, symmetric vertical axis, and the same
axis on both models so STD and WAL are directly comparable.

  a, b  the three balance terms and the closure, as in fig_budget_time,
        with each term's fitted secular trend drawn thin and the
        Delta N_D trend mirrored into the driving half so its slope can
        be laid against the drive's by eye.
  c, d  the COLUMN-WISE resultant at the trench, the first isostatic
        column and the ridge, on one zero line.
  e, f  the four balance series with their linear trends removed. Axis
        symmetric about zero and shared between the models.

THE MIDDLE ROW, added 2026-10-09 at Dan's request, makes Figures 4 and 5
a spatial/time pair: Figure 4 carries Delta N_D(x) and the column-wise
N_D(x) at the reference epoch, and this figure carries both through time.
Read (c, d) against (a, b): Delta N_D generally RESISTS, while N_D(x_T)
itself is compression-like in the median state of both runs (-0.79 STD,
-1.87 WAL; tension-like at 38 % and 0 % of snapshots). The positive
N_D(x_I) in STD is generated ACROSS THE TRENCH SIDE by the trench pull --
it is not a tension-like edge force arriving through the hinge.

V and M are deliberately absent. They were interrupting the main
force-balance argument and now live in fig_trench_resultants, which moves
to the SI.

⚠ THE STANDING WAL RIDGE-PICK CAVEAT IS RESOLVED, NOT CARRIED (measured
2026-10-09). The claim was that WAL ridge-referenced values after 70 Myr
are unreliable because the ridge pick flips. On the committed cache it
does not. The WAL ridge migrates MONOTONICALLY inward over the last
15 Myr -- 2569 -> 2532 -> 2485 -> 2462 -> 2418 -> 2347 -> 2324 km seaward
of the trench -- with no reversal and no excursion of the hundreds of km
the flow-based pick was adopted to avoid (conventions §3.4). The largest
single step after 66 Myr is 71 km (WAL, 76->78 Myr) and 53 km (STD), on a
ridge 2400 km away: about 3 %. N_D(x_R) is correspondingly quiet late
(+0.20 to +0.33 TN/m; its largest step anywhere is 0.108 TN/m and falls
at t = 12 Myr, not late). What IS real is a
single 0.48 TN/m step in GPE*(x_R) at 68 Myr -- inside the documented
62-70 Myr loading-and-release episode, which the warning below already
covers. No segment of (c, d) is masked or faded, because nothing in it
warrants it.

SIGN CONVENTION, unchanged from fig_budget_time (Dan's, 2026-09-20).
Every curve is that term's contribution to the net force in +x, the
analysis frame's rightward direction; the subducting plate lies to the
right and moves trench-ward, so

    negative = force to the left  = DRIVING
    positive = force to the right = RESISTING

WHY THE SECOND ROW EARNS ITS SPACE. The top row is dominated by the
secular rise and reads as three smooth, well-separated curves. Removing
the trends shows that what is left is NOT small and NOT the same in the
two runs:

  residual s.d. [TN/m]      STD     WAL
    -Delta GPE* (drive)     0.110   0.259
    Delta N_D               0.289   0.319
    F_B                     0.309   0.243
    closure                 0.019   0.023

The two resisting terms fluctuate about three times harder than the
driving term in STD, and the closure is 15x smaller than either of them
-- so the residual structure is signal, not extraction noise.

⚠ THE EXCHANGE IS THE DOMINANT MODE OF THE WHOLE RESIDUAL, not just of
one Fourier bin. Detrended, r(Delta N_D, F_B) = **-0.93 (STD)** and
**-0.58 (WAL)** across all frequencies. DYNAMICS_FINDINGS §1.8 measured
this at the 25 Myr buckling bin (178 degrees apart, amplitudes 0.296 and
0.243 TN/m); the broadband correlation shows that the same antiphase
trade accounts for essentially all of STD's short-timescale budget
variation, not only the buckling component. In the lower-left panel the
black and red curves are visibly mirror images.

⚠ WAL's residual is dominated by ONE EVENT, not by a cycle. Its
62-70 Myr excursion (diag_wal_episode) contributes most of the amplitude
in this panel, and during it the drive is NOT quiet -- it rises 56 % and
discharges. Do not read WAL's residual s.d. as the size of a cycle; read
STD's. The two runs carry different phenomena here (§5.9, §2b).

⚠ The residual retains everything that is not a straight line: the
~25 Myr oscillation, curvature, events and noise alike. It is not a
high-pass filter, and the choice matters for numbers though not for the
sign structure (§5.3).

DRAFT CAPTION. The three terms of the vertically integrated force balance
between the trench and ridge columns, for STD (left) and WAL (right).
Each curve is that term's contribution to the net horizontal force in the
analysis frame, signed so that negative acts toward the trench and drives
the plate while positive resists, so that the balance reads
Delta N_D + Delta sigma_zz-bar + F_B = 0 and the plotted curves sum to the
closure (green dashed). The driving term is plotted in the vertical
normal-stress register, Delta sigma_zz-bar, which is identically
-Delta GPE*; it is DASHED because the balance palette's blue denotes
Delta GPE* itself, and the linestyle rather than the colour carries the
change of sign. Secular trends per 10 Myr: Delta sigma_zz-bar -0.36 (STD)
/ -0.40 (WAL), Delta N_D +0.56 / +0.46, F_B -0.19 / -0.06. (a, b) The terms through time, with each fitted
secular trend drawn thin and the normal-stress-difference trend mirrored
into the driving half (dotted) so that its slope can be read against the
driving trend, the gap between them being the basal-traction trend.
(c, d) The same four series with their linear trends removed, on a common
symmetric axis. The light grey band spans +/-0.5 TN/m in all four panels:
in (c, d) it is the envelope of the perturbations, and in (a, b) the same
band acts as a scale bar, showing that those perturbations are about a
tenth of the GPE-like force they perturb. Through the residual the two
resisting terms vary in near mirror image -- detrended
r(Delta N_D, F_B) = -0.93 in STD -- while the driving term is three times
quieter (residual s.d. 0.110 against 0.289 and 0.309 TN/m), and the
closure is smaller again at 0.019, so the structure shown is signal
rather than extraction error. WAL's residual (d) is dominated by the
single 62-70 Myr episode rather than by a repeating cycle.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from column_profiles_cache import terms, T_MIN_MYR
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_RULE = '#BFC3D1'
# Very light band spanning +/- PERT_TNM about zero, drawn across all times
# on BOTH rows (Dan, 2026-09-24). On the residual row it is the envelope of
# the perturbations; on the levels row it is a scale bar -- it shows at a
# glance how small the perturbations are against the ~4 TN/m carried by the
# driving term, which no amount of caption wording conveys as directly.
PERT_TNM = 0.5


def main():
    d = cpc.load()
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.4), sharex='col',
                             gridspec_kw={'height_ratios': [1.6, 1.25, 1.0]})
    rows = [('model', 'quantity', 'value')]
    fit = lambda t, y: np.polyval(np.polyfit(t, y, 1), t)
    ld = lambda t, y: y - fit(t, y)

    for col, key in enumerate(('STD', 'WAL')):
        q = terms(d, key)
        m = q['t'] >= T_MIN_MYR
        t = q['t'][m]
        gpe, nd, fb = (q['d_gpe'][m] / 1e12, q['d_nd'][m] / 1e12,
                       q['f_b'][m] / 1e12)
        # CLOSURE IN THE DISPLAYED REGISTER: Delta GPE* - Delta N_D - F_B.
        # terms() returns d_nd - d_gpe + f_b, which is its negative.
        clo = -q['closure'][m] / 1e12
        sl = lambda y: np.polyfit(t, y, 1)[0] * 10          # per 10 Myr

        # ---------------- top: levels, as in fig_budget_time -------------
        a0 = axes[0, col]
        a0.axhspan(-PERT_TNM, PERT_TNM, color='0.55', alpha=0.13, lw=0,
                   zorder=0,
                   label=f'$\\pm${PERT_TNM:.1f} TN/m (perturbation scale)')
        a0.axhline(0, color='k', lw=1.6)
        # DISPLAY REGISTER ALIGNED with fig_balance_snapshot and
        # fig_gpe_partition (paper-agent handoff, 2026-10-08): Delta GPE*
        # is plotted POSITIVE, solid blue, and the closure is written
        #     Delta GPE* - Delta N_D - F_B = 0.
        # Delta N_D and F_B keep their own signs. Nothing is recomputed;
        # only the displayed register changes.
        a0.plot(t, gpe, color='b', lw=4, alpha=0.6,
                label=r'$\Delta\mathrm{GPE}^{*}$')
        a0.plot(t, nd, color='k', lw=1.6, label=r'$\Delta N_D$')
        a0.plot(t, fb, color='red', lw=2, label='$F_B$')
        a0.plot(t, clo, color='g', ls='--', lw=2.5, label='closure')
        a0.plot(t, fit(t, gpe), color='b', lw=1.0, alpha=0.9)
        a0.plot(t, fit(t, nd), color='k', lw=1.0, alpha=0.9)
        a0.plot(t, fit(t, fb), color='red', lw=1.0, alpha=0.9)
        # The mirrored Delta N_D trend is gone. In this register the gap
        # between the ordinary Delta GPE* and Delta N_D trends IS F_B, so
        # the arrow spans those two directly and needs no reflected line.
        a0.annotate('', xy=(t[-3], fit(t, nd)[-3]),
                    xytext=(t[-3], fit(t, gpe)[-3]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
        # Label ABOVE the arrow (Dan, 2026-09-24): centred on it and offset
        # by 20 % of the arrow's own height, which puts it in the empty
        # region between the mirrored trend and zero and leaves room for a
        # readable size. Placing it beside the arrow cramped it against the
        # right spine and forced an 8 pt label.
        _lo, _hi = sorted((fit(t, nd)[-3], fit(t, gpe)[-3]))
        a0.text(t[-3], _hi + 0.20 * (_hi - _lo),
                f'$F_B$ trend\n{sl(fb):+.2f}', color='red', fontsize=10.5,
                va='bottom', ha='center')
        # PANEL LETTERS (Dan, 2026-10-01). Without them the manuscript
        # cannot point a reader at one panel, which is what the results
        # prose has to do. Top row (a, b) levels, bottom row (c, d)
        # residuals; letter in the title so it sits clear of the curves.
        a0.set_title(f'({"ab"[col]})  {key}', fontsize=11)
        a0.grid(alpha=0.25, color=C_RULE, lw=0.6)

        # ---------------- bottom: the same lines, detrended ---------------
        # ---------------- middle: N_D at the three columns ---------------
        # Dan's rebuild, 2026-10-09: Figures 4 and 5 become a spatial/time
        # pair, so the time figure must carry the COLUMN-WISE resultant
        # beside the plate-wide difference. Read against row (a, b): Delta
        # N_D generally RESISTS, while N_D(x_T) itself is compression-like
        # in the median state of both runs. A positive N_D(x_I) in STD is
        # generated across the trench side by the trench pull; it is not a
        # tension-like edge force arriving through the hinge.
        rcol = cpc.resultants(d, key)
        nd_T = rcol['nd_T'][m] / 1e12
        nd_I = rcol['nd_I'][m] / 1e12
        nd_R = rcol['nd_R'][m] / 1e12
        a1 = axes[1, col]
        a1.axhline(0, color='k', lw=1.3)
        a1.plot(t, nd_T, '-', color='k', lw=2.0, label=r'$N_D(x_T)$')
        a1.plot(t, nd_I, '--', color='k', lw=1.5, label=r'$N_D(x_I)$')
        a1.plot(t, nd_R, ':', color='k', lw=1.5, label=r'$N_D(x_R)$')
        a1.grid(alpha=0.25, color=C_RULE, lw=0.6)
        a1.set_title(f'({"cd"[col]})  {key} — resultant at each column',
                     fontsize=11)
        for lab, yy, va in (('TENSION-LIKE', 0.95, 'top'),
                            ('COMPRESSION-LIKE', 0.05, 'bottom')):
            a1.text(0.5, yy, lab, transform=a1.transAxes, ha='center',
                    va=va, fontsize=10.5, fontweight='bold', color='0.82',
                    zorder=0)
        for nm, y in (('nd_trench', nd_T), ('nd_first_isostatic', nd_I),
                      ('nd_ridge', nd_R)):
            rows += [(key, f'{nm}_median_TNm', f'{np.median(y):.3f}'),
                     (key, f'{nm}_q1_TNm', f'{np.percentile(y, 25):.3f}'),
                     (key, f'{nm}_q3_TNm', f'{np.percentile(y, 75):.3f}')]
        rows += [(key, 'nd_trench_tension_like_fraction',
                  f'{np.mean(nd_T > 0):.3f}'),
                 (key, 'nd_ridge_always_tension_like', str(bool((nd_R > 0).all()))),
                 (key, 'ridge_seaward_max_step_after_66Myr_km',
                  f'{np.abs(np.diff(((d[f"{key}_xT"] - d[f"{key}_xR"])[m])[t >= 66])).max() / 1e3:.0f}')]

        a2 = axes[2, col]
        a2.set_title(f'({"ef"[col]})  {key} — residual about the trend',
                     fontsize=10, loc='left', color='0.35')
        a2.axhspan(-PERT_TNM, PERT_TNM, color='0.55', alpha=0.13, lw=0,
                   zorder=0)
        res = {r'$\Delta\mathrm{GPE}^{*}$': (ld(t, gpe), 'b', 4, '-', 0.6),
               r'$\Delta N_D$': (ld(t, nd), 'k', 1.6, '-', 1.0),
               '$F_B$': (ld(t, fb), 'red', 2.0, '-', 1.0),
               'closure': (ld(t, clo), 'g', 2.2, '--', 1.0)}
        for nm, (y, c_, lw, ls, al) in res.items():
            a2.plot(t, y, color=c_, lw=lw, ls=ls, alpha=al, label=nm)
        a2.axhline(0, color='k', lw=1.0)
        a2.set_xlabel('Model time [Myr]', fontsize=11)
        a2.grid(alpha=0.25, color=C_RULE, lw=0.6)
        a2.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

        print(f'{key}: slopes/10 Myr  dGPE* {sl(gpe):+.3f} = dN_D {sl(nd):+.3f} '
              f'+ F_B {sl(fb):+.3f} (sum {sl(nd)+sl(fb):+.3f}); '
              f'residual s.d. drive {ld(t, gpe).std():.3f}, '
              f'dN_D {ld(t, nd).std():.3f}, F_B {ld(t, fb).std():.3f}, '
              f'closure {ld(t, clo).std():.3f}; '
              f'r(res dN_D, res F_B) '
              f'{np.corrcoef(ld(t, nd), ld(t, fb))[0, 1]:+.2f}')
        for nm, y in (('gpe_like_force', gpe), ('delta_nd', nd),
                      ('basal_traction', fb), ('closure', clo)):
            rows += [(key, f'{nm}_trend_per_10Myr', f'{sl(y):.3f}'),
                     (key, f'{nm}_residual_sd_TNm', f'{ld(t, y).std():.4f}')]
        rows.append((key, 'corr_residual_nd_fb',
                     f'{np.corrcoef(ld(t, nd), ld(t, fb))[0, 1]:.3f}'))

        # ------------------------------------------------------------------
        # REQUESTED BY THE MANUSCRIPT AUDIT, 2026-10-11 (batch 1). Each row
        # is emitted here rather than quoted in conversation, and the
        # amplitude measure is DEFINED rather than left as "about".
        # ------------------------------------------------------------------
        # plate-wide levels: median and quartiles
        for nm, y in (('gpe_like_force', gpe), ('delta_nd', nd),
                      ('basal_traction', fb)):
            rows += [(key, f'{nm}_median_TNm', f'{np.median(y):.3f}'),
                     (key, f'{nm}_q1_TNm', f'{np.percentile(y, 25):.3f}'),
                     (key, f'{nm}_q3_TNm', f'{np.percentile(y, 75):.3f}'),
                     (key, f'{nm}_min_TNm', f'{y.min():.3f}'),
                     (key, f'{nm}_max_TNm', f'{y.max():.3f}'),
                     # TRANSIENT AMPLITUDE, defined: half the full range of
                     # the DETRENDED series. The manuscript says "about
                     # 0.5 TN/m" with no stated measure; this is one, and it
                     # is the one these rows support.
                     (key, f'{nm}_transient_half_range_TNm',
                      f'{0.5 * (ld(t, y).max() - ld(t, y).min()):.3f}')]
        # column resultants: trends and extrema
        for nm, y in (('nd_trench', nd_T), ('nd_first_isostatic', nd_I),
                      ('nd_ridge', nd_R)):
            rows += [(key, f'{nm}_trend_per_10Myr', f'{sl(y):.3f}'),
                     (key, f'{nm}_min_TNm', f'{y.min():.3f}'),
                     (key, f'{nm}_max_TNm', f'{y.max():.3f}')]
        # sign fractions, for the "always/dominant" formulations
        marg = gpe - np.maximum(nd, fb)
        rows += [(key, 'gpe_like_force_driving_fraction',
                  f'{np.mean(gpe > 0):.3f}'),
                 (key, 'basal_traction_resisting_fraction',
                  f'{np.mean(fb > 0):.3f}'),
                 (key, 'delta_nd_resisting_fraction', f'{np.mean(nd > 0):.3f}'),
                 # "dominant" is only safe at every state if this is positive
                 (key, 'gpe_like_force_min_margin_over_others_TNm',
                  f'{marg.min():.3f}'),
                 (key, 'gpe_like_force_dominant_all_steps',
                  str(bool((marg > 0).all()))),
                 (key, 'max_total_driving_force_TNm', f'{gpe.max():.3f}'),
                 (key, 'n_snapshots', f'{len(t)}')]
        print(f'   audit rows: dGPE* driving {100*np.mean(gpe>0):.0f} % of steps, '
              f'F_B resisting {100*np.mean(fb>0):.0f} %, min dominance margin '
              f'{marg.min():+.3f} TN/m, max driving {gpe.max():.2f} TN/m, '
              f'transient half-range dGPE* {0.5*(ld(t,gpe).max()-ld(t,gpe).min()):.3f}')

    # --- shared, symmetric vertical axes (Dan, 2026-09-24) ---------------
    # Top row: common limits so the two runs are read against one scale.
    # Bottom row: the same, and forced symmetric about zero — an anomaly
    # axis that is not symmetric misrepresents which way the excursions go.
    for row in range(3):
        lo = min(ax.get_ylim()[0] for ax in axes[row, :])
        hi = max(ax.get_ylim()[1] for ax in axes[row, :])
        if row == 0:
            lo, hi = lo - 0.30 * abs(lo), hi + 0.30 * abs(hi)
        else:
            # rows 1 and 2 are both signed quantities read about zero, so
            # both are forced symmetric -- an unsymmetric axis on either
            # misrepresents which way the excursions go
            hi = max(abs(lo), abs(hi)) * 1.05; lo = -hi
        for ax in axes[row, :]:
            ax.set_ylim(lo, hi)

    axes[0, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    axes[1, 0].set_ylabel('$N_D$ at a column [TN/m]', fontsize=11)
    # upper right: the only corner free of a curve in BOTH panels, and
    # clear of the tension/compression watermarks, which stay centred
    axes[1, 0].legend(frameon=False, fontsize=9.5, loc='upper right')
    axes[2, 0].set_ylabel('residual about\nthe trend [TN/m]', fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9.5, loc='upper left', ncol=2)
    # NO GLOBAL DRIVING/RESISTING LABELS (paper-agent handoff, 2026-10-08).
    # In this register positive Delta GPE* drives while positive Delta N_D
    # and F_B resist, so the vertical axis has no single physical sense and
    # a background wash asserting one would be wrong for two of three curves.
    # Plain, descriptive title (Dan, 2026-09-24) — the figure argues for
    # itself; the suptitle should only say what is plotted.
    fig.suptitle('Trench-to-ridge force balance and column resultants through time', fontsize=10.5, color='0.35')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_budget_time_detrended.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('budget_time_detrended', rows[0], rows[1:],
                      script='fig_budget_time_detrended.py',
                      figure='fig_budget_time_detrended.png',
                      models=('STD', 'WAL'),
                      meta={'domain': 'trench to ridge columns',
                            'zc_km': cpc.ZC_KM, 't_min_Myr': T_MIN_MYR,
                            'sign': 'Delta GPE* plotted POSITIVE (driving); '
                                    'Delta N_D and F_B keep their own signs and '
                                    'resist when positive. The axis has no single '
                                    'physical sense, which is why the figure carries '
                                    'no driving/resisting labels.',
                            'closure': 'Delta GPE* - Delta N_D - F_B = 0; its trend '
                                       'row changed sign on 2026-10-08 with the '
                                       'display register, magnitude unchanged',
                            'detrend': 'fitted linear model per series, removed',
                            'axes': 'shared per row; residual row symmetric about 0'})
    print('written:', tab)


if __name__ == '__main__':
    main()
