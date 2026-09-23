"""fig_budget_time_detrended — the three-term budget through time, levels and residual.

Writes figures/fig_budget_time_detrended.png and
tables/budget_time_detrended.csv from the committed column-profile cache.

Requested by Dan 2026-09-24: fig_budget_time with a detrended panel added
beneath each model, same lines, symmetric vertical axis, and the same
axis on both models so STD and WAL are directly comparable.

  top     the three balance terms and the closure, as in fig_budget_time,
          with each term's fitted secular trend drawn thin and the
          Delta N_D trend mirrored into the driving half so its slope can
          be laid against the drive's by eye.
  bottom  the same four series with their linear trends removed. Axis
          symmetric about zero and shared between the models.

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
Each curve is that term's contribution to the net horizontal force,
signed so that negative acts toward the trench and drives the plate while
positive resists. Top: the terms through time, with fitted secular trends
(thin) and the normal-stress-difference trend mirrored into the driving
half (dotted), the gap between it and the driving trend being the basal
traction trend. Bottom: the same series with linear trends removed, on a
common symmetric axis. The residual is carried by the two resisting
terms, which vary in antiphase, while the driving term is comparatively
quiet; the closure (green) is several times smaller than any of them.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_budget_time import terms, T_MIN_MYR
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
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), sharex='col',
                             gridspec_kw={'height_ratios': [1.6, 1.0]})
    rows = [('model', 'quantity', 'value')]
    fit = lambda t, y: np.polyval(np.polyfit(t, y, 1), t)
    ld = lambda t, y: y - fit(t, y)

    for col, key in enumerate(('STD', 'WAL')):
        q = terms(d, key)
        m = q['t'] >= T_MIN_MYR
        t = q['t'][m]
        gpe, nd, fb = (q['d_gpe'][m] / 1e12, q['d_nd'][m] / 1e12,
                       q['f_b'][m] / 1e12)
        clo = q['closure'][m] / 1e12
        sl = lambda y: np.polyfit(t, y, 1)[0] * 10          # per 10 Myr

        # ---------------- top: levels, as in fig_budget_time -------------
        a0 = axes[0, col]
        a0.axhspan(-PERT_TNM, PERT_TNM, color='0.55', alpha=0.13, lw=0,
                   zorder=0,
                   label=f'$\\pm${PERT_TNM:.1f} TN/m (perturbation scale)')
        a0.axhline(0, color='k', lw=1.6)
        a0.plot(t, -gpe, color='b', lw=4, alpha=0.6,
                label=r'$-\Delta\mathrm{GPE}^{*}$' + f'  ({sl(gpe):+.2f}/10 Myr)')
        a0.plot(t, nd, color='k', lw=1.6,
                label=r'$\Delta N_D$' + f'  ({sl(nd):+.2f})')
        a0.plot(t, fb, color='red', lw=2, label='$F_B$' + f'  ({sl(fb):+.2f})')
        a0.plot(t, clo, color='g', ls='--', lw=2.5, label='closure')
        a0.plot(t, -fit(t, gpe), color='b', lw=1.0, alpha=0.9)
        a0.plot(t, fit(t, nd), color='k', lw=1.0, alpha=0.9)
        a0.plot(t, fit(t, fb), color='red', lw=1.0, alpha=0.9)
        a0.plot(t, -fit(t, nd), color='k', ls=':', lw=1.8,
                label=r'$\Delta N_D$ trend, mirrored')
        a0.annotate('', xy=(t[-3], -fit(t, nd)[-3]),
                    xytext=(t[-3], -fit(t, gpe)[-3]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
        a0.text(t[-3], 0.5 * (-fit(t, nd)[-3] - fit(t, gpe)[-3]),
                f'  $F_B$ trend\n  {sl(fb):+.2f}', color='red', fontsize=8,
                va='center', ha='left')
        a0.set_title(key, fontsize=11)
        a0.grid(alpha=0.25, color=C_RULE, lw=0.6)

        # ---------------- bottom: the same lines, detrended ---------------
        a1 = axes[1, col]
        a1.axhspan(-PERT_TNM, PERT_TNM, color='0.55', alpha=0.13, lw=0,
                   zorder=0)
        res = {r'$-\Delta\mathrm{GPE}^{*}$': (ld(t, -gpe), 'b', 4, '-', 0.6),
               r'$\Delta N_D$': (ld(t, nd), 'k', 1.6, '-', 1.0),
               '$F_B$': (ld(t, fb), 'red', 2.0, '-', 1.0),
               'closure': (ld(t, clo), 'g', 2.2, '--', 1.0)}
        for nm, (y, c_, lw, ls, al) in res.items():
            a1.plot(t, y, color=c_, lw=lw, ls=ls, alpha=al,
                    label=f'{nm}  (s.d. {y.std():.3f})')
        a1.axhline(0, color='k', lw=1.0)
        a1.set_xlabel('Model time [Myr]', fontsize=11)
        a1.grid(alpha=0.25, color=C_RULE, lw=0.6)
        a1.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)

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

    # --- shared, symmetric vertical axes (Dan, 2026-09-24) ---------------
    # Top row: common limits so the two runs are read against one scale.
    # Bottom row: the same, and forced symmetric about zero — an anomaly
    # axis that is not symmetric misrepresents which way the excursions go.
    for row in range(2):
        lo = min(ax.get_ylim()[0] for ax in axes[row, :])
        hi = max(ax.get_ylim()[1] for ax in axes[row, :])
        if row == 0:
            lo, hi = lo - 0.30 * abs(lo), hi + 0.30 * abs(hi)
        else:
            hi = max(abs(lo), abs(hi)); lo = -hi
        for ax in axes[row, :]:
            ax.set_ylim(lo, hi)

    axes[0, 0].set_ylabel('Force per unit distance [TN/m]', fontsize=11)
    axes[1, 0].set_ylabel('residual about\nthe trend [TN/m]', fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9.5, loc='lower left', ncol=2)
    for ax in axes[0, :]:
        for lab, dy, va in (('RESISTING', 0.975, 'top'),
                            ('DRIVING', 0.025, 'bottom')):
            ax.text(0.74, dy, lab, transform=ax.transAxes, ha='center', va=va,
                    fontsize=15, fontweight='bold', color='0.78', zorder=0)
    # Plain, descriptive title (Dan, 2026-09-24) — the figure argues for
    # itself; the suptitle should only say what is plotted.
    fig.suptitle('Balance terms, trench to ridge', fontsize=10.5, color='0.35')
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
                            'sign': 'positive = +x = resisting; negative = driving',
                            'detrend': 'fitted linear model per series, removed',
                            'axes': 'shared per row; residual row symmetric about 0'})
    print('written:', tab)


if __name__ == '__main__':
    main()
