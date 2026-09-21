"""fig_gpe_decomposition — the driving term split by domain, trend and residual.

Writes figures/fig_gpe_decomposition.png and
tables/gpe_decomposition.csv from the committed column-profile cache.

Step 2 of the 2026-09-22 investigation. The plate-wide driving term is
the sum of the two additive domains of the schematic,

    Delta GPE* (trench -> ridge) = trench pull + ridge push
                                   (non-isostatic)  (isostatic)

and the point of separating them is that the two domains carry different
timescales.

  top     levels, with the fitted secular trend of each shown dashed.
  bottom  the residual about that trend.

DETRENDING: A FITTED LINEAR MODEL, NOT A FILTER (Dan, 2026-09-22). The
first version used a Gaussian high-pass at ~24 Myr. A linear fit is
better here for four reasons:

  1. It has no free cutoff. The filter's 24 Myr was chosen to sit between
     the oscillation and the run length, which is defensible but is still
     a tuned parameter a reader has to accept.
  2. The trend really is close to linear where it matters: the fit gives
     R2 = 0.96 (STD) / 0.92 (WAL) on ridge push and 0.98 / 0.92 on the
     total, and a quadratic buys 0-31 % more.
  3. The trend can be DRAWN. The decomposition becomes visible in the
     figure instead of asserted in the caption.
  4. It keeps genuine multi-decadal structure that the filter attenuated
     -- notably the WAL 60-70 Myr excursion, which is an event, not a
     trend, and should survive detrending rather than be smoothed away.

Where the linear fit is poor (WAL trench pull, R2 = 0.43) that is because
there is no real secular trend to remove -- the series is flat and then
spikes -- so a straight line is the right null, not a bad fit.

⚠ THE CONCLUSION IS ROBUST TO THE CHOICE, THE HEADLINE NUMBER IS NOT.
Under both treatments the two domains oppose each other against the
plate speed, trench pull positive and ridge push negative, and the total
is less sensitive than either part. But how MUCH less depends on the
method: high-passed, the total collapses to +0.16 (STD); linearly
detrended it only falls to +0.31. Quote the sign structure, not the
collapse factor, unless the timescale is stated with it.

  r(plate speed, X), linear-detrended [high-passed in brackets]:
      trench pull   STD +0.52 [+0.57]   WAL +0.63 [+0.54]
      ridge push    STD -0.39 [-0.50]   WAL -0.23 [-0.33]
      total         STD +0.31 [+0.16]   WAL +0.50 [+0.36]

WHAT IT ESTABLISHES. The secular rise is isostatic: ridge push supplies
64 % (STD) / 72 % (WAL) of it, and what the trench pull contributes is
mostly its growing moment arm (h +34 %) rather than a deepening trench
(+13 %). A thermal quantity cannot accelerate a plate whose feed rate is
set elsewhere, so the growing drive is absorbed as in-plane resistance --
which is what the secular rise of Delta N_D is.

The residual is where the two domains part company: a faster plate
deepens the trench and simultaneously suppresses the ridge push through
the tilt, so they act in opposition and partly cancel in the total.

The plate-speed series is NOT plotted (Dan, 2026-09-22) -- the velocity
correlations above are carried in the table and the caption instead. For
the kinematics see fig_convergence_partition.

DRAFT CAPTION. The plate-wide driving term separated into its two
domains, for STD (left) and WAL (right). Top: the trench pull across the
non-isostatic domain (blue), the ridge push across the isostatic domain
(orange), and their sum (black), each with its fitted secular trend
(dashed). Bottom: the residual about those trends. The secular growth is
supplied mainly by the isostatic domain. In the residual the two domains
respond to changes in plate speed with opposite sign -- the trench pull
rising as the trench deepens, the ridge push falling as the tilt grows --
so they partly cancel and the total is left less sensitive to plate speed
than either of its parts.
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
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain
C_RIDGE = '#D55E00'         # isostatic / ridge push domain
C_TOTAL = 'k'
C_RULE = '#BFC3D1'


def detrend(t, y):
    """Fitted linear secular model, and the residual about it."""
    trend = np.polyval(np.polyfit(t, y, 1), t)
    return trend, y - trend


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True,
                             gridspec_kw={'height_ratios': [1.55, 1.0]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        z, t, zc, zkm = c['z'], c['t'], c['zc'], c['z'] / 1e3
        TP = -np.trapezoid(c['p_T'][:, zc], z[zc], axis=1) / 1e12
        RP = np.trapezoid(c['p_R'][:, zc], z[zc], axis=1) / 1e12
        TOT = TP + RP
        vp = np.abs(d[f'{key}_vx_TR'][:, zkm < 20].mean(axis=1))
        _, rv = detrend(t, vp)

        span = t[-1] - t[0]
        parts = {}
        for nm, y in (('trench_pull', TP), ('ridge_push', RP), ('total', TOT)):
            tr, res = detrend(t, y)
            parts[nm] = dict(y=y, trend=tr, res=res,
                             rise=np.polyfit(t, y, 1)[0] * span,
                             r2=1 - res.var() / y.var(),
                             rv=float(np.corrcoef(rv, res)[0, 1]))
        share = 100 * parts['ridge_push']['rise'] / (parts['trench_pull']['rise']
                                                     + parts['ridge_push']['rise'])
        tsd = parts['trench_pull']['res'].std() + parts['ridge_push']['res'].std()
        tshare = 100 * parts['trench_pull']['res'].std() / tsd

        a0, a1 = axes[0, col], axes[1, col]
        style = (('total', C_TOTAL, 2.2, f'total $\\Delta$GPE*'),
                 ('trench_pull', C_TRENCH, 1.8,
                  f'trench pull ({100 - share:.0f} % of the rise)'),
                 ('ridge_push', C_RIDGE, 1.8,
                  f'ridge push ({share:.0f} % of the rise)'))
        for nm, c_, lw, lab in style:
            p = parts[nm]
            a0.plot(t, p['y'], '-', color=c_, lw=lw, label=lab)
            a0.plot(t, p['trend'], '--', color=c_, lw=1.0, alpha=0.75)
        a0.set_title(key, fontsize=11)
        a0.set_ylabel('[TN/m]', fontsize=10)
        a0.legend(frameon=False, fontsize=8.5, loc='upper left')

        for nm, c_, lw, _ in style:
            p = parts[nm]
            lab = ('total' if nm == 'total' else
                   f'{nm.replace("_", " ")} '
                   f'({tshare if nm == "trench_pull" else 100 - tshare:.0f} % '
                   'of the residual)')
            a1.plot(t, p['res'], '-', color=c_, lw=lw * 0.9, label=lab)
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel('residual about\nthe trend [TN/m]', fontsize=9.5)
        a1.set_xlabel('Model time [Myr]', fontsize=11)
        a1.legend(frameon=False, fontsize=8, loc='upper left')
        for ax in (a0, a1):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        print(f'{key}: rise TP {parts["trench_pull"]["rise"]:+.2f} '
              f'(R2 {parts["trench_pull"]["r2"]:.2f}), '
              f'RP {parts["ridge_push"]["rise"]:+.2f} '
              f'(R2 {parts["ridge_push"]["r2"]:.2f}) TN/m, ridge {share:.0f} %; '
              f'residual sd TP {parts["trench_pull"]["res"].std():.3f}, '
              f'RP {parts["ridge_push"]["res"].std():.3f}, '
              f'TOT {parts["total"]["res"].std():.3f}; '
              f'r with detrended speed: TP {parts["trench_pull"]["rv"]:+.2f}, '
              f'RP {parts["ridge_push"]["rv"]:+.2f}, '
              f'TOT {parts["total"]["rv"]:+.2f}')
        for nm, p in parts.items():
            rows += [(key, f'secular_rise_{nm}_TNm', f'{p["rise"]:.3f}'),
                     (key, f'linear_fit_r2_{nm}', f'{p["r2"]:.3f}'),
                     (key, f'residual_sd_{nm}_TNm', f'{p["res"].std():.4f}'),
                     (key, f'corr_detrended_speed_{nm}', f'{p["rv"]:.3f}')]
        rows += [(key, 'ridge_share_of_secular_rise_percent', f'{share:.1f}'),
                 (key, 'trench_share_of_residual_percent', f'{tshare:.1f}')]

    fig.suptitle('The driving term by domain: the secular rise is isostatic; '
                 'in the residual the two domains oppose each other', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_gpe_decomposition.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('gpe_decomposition', rows[0], rows[1:],
                      script='fig_gpe_decomposition.py',
                      figure='fig_gpe_decomposition.png', models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM,
                            'total': 'trench pull + ridge push, both over 0..z_c',
                            'detrend': 'fitted linear model, removed; residual '
                                       'retains all curvature and events'})
    print('written:', tab)


if __name__ == '__main__':
    main()
