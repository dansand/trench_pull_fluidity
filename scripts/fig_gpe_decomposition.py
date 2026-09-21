"""fig_gpe_decomposition — the driving term split by domain, secular vs transient.

Writes figures/fig_gpe_decomposition.png and
tables/gpe_decomposition.csv from the committed column-profile cache.

Step 2 of the 2026-09-22 investigation. The plate-wide driving term is
the sum of the two additive domains of the schematic,

    Delta GPE* (trench -> ridge) = trench pull + ridge push
                                   (non-isostatic)  (isostatic)

and the point of separating them is that THE TWO DOMAINS CARRY DIFFERENT
TIMESCALES. Rows:

  1  levels. The secular growth, and which domain supplies it.
  2  the same three high-passed at ~24 Myr, between the oscillation
     period (~20-25 Myr) and the run-length thermal trend. This is the
     transient part, and it lives somewhere else.
  3  the transient plate speed and the kinematic partition f, for
     comparison against row 2 (see fig_convergence_partition).

WHAT IT ESTABLISHES. The secular rise is a RIDGE-side phenomenon: the
ridge push supplies 64 % (STD) / 72 % (WAL) of it, and what rise the
trench pull does contribute is mostly its growing moment arm (h +34 %)
rather than a deepening trench (+13 %). A thermal quantity cannot
accelerate a plate whose feed rate is set elsewhere, so the growing drive
is absorbed as in-plane resistance instead -- which is what the secular
rise of Delta N_D is.

The transient part is NOT simply trench-side -- the trench carries only
53 % (STD) / 59 % (WAL) of it, which is barely more than half. What
separates the domains on this timescale is the SIGN. Against the
high-passed plate speed the trench pull correlates +0.57 / +0.54 while
the ridge push correlates -0.50 / -0.33: a faster plate deepens the
trench and simultaneously suppresses the ridge push through the tilt.
The two therefore largely cancel: the TOTAL correlates with the plate
speed at only +0.16 (STD) / +0.36 (WAL), against +-0.5 for either part
alone. THAT COLLAPSE is the result, not a variance share -- the sd ratio
understates it (the total's transient sd is only 21 % / 13 % below the
larger part), because the cancellation is in the phase, not the
amplitude.

WHY HIGH-PASS AND NOT DIFFERENCING. Measured against the plate speed,
first differencing -- which is what the earlier increment correlations
used -- is the WEAKEST of the detrending options, because it
over-amplifies the highest frequencies. On the trench-side loading it
gives r = +0.36 against +0.57 for a high-pass; on trench depth, +0.57
against +0.76. Regressing out the non-tilting term fails outright (+0.06)
because that term and the moment arm are collinear, both being thermal,
so it removes the signal with the trend. The filter uses mode='nearest':
reflect padding distorts a strongly trending series at the endpoints.

DRAFT CAPTION. The plate-wide driving term separated into its two
domains, for STD (left) and WAL (right). Top: the trench pull across the
non-isostatic domain (blue), the ridge push across the isostatic domain
(orange), and their sum (black). Middle: the same three high-pass
filtered at 24 Myr to isolate variability faster than the thermal trend.
Bottom: the high-passed plate speed and the fraction of the convergence
carried by plate motion. The secular growth of the driving term is
supplied mainly by the isostatic domain. On the transient timescale the
two domains respond to changes in plate speed with opposite sign — the
trench pull rising as the trench deepens, the ridge push falling as the
tilt grows — so they largely cancel: the total correlates with the plate
speed at only +0.16 (STD) and +0.36 (WAL), against ±0.5 for either part
on its own.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain
C_RIDGE = '#D55E00'         # isostatic / ridge push domain
C_TOTAL = 'k'
C_PLATE = '#1B9E77'         # plate motion (fig_convergence_partition)
C_RULE = '#BFC3D1'
HP_SIGMA = 6                # samples; 2 Myr cadence -> ~24 Myr cutoff


def hp(y):
    """High-pass. mode='nearest' -- reflect padding wrecks a trending series."""
    return y - gaussian_filter1d(y, HP_SIGMA, mode='nearest')


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.4), sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1.2, 1.0]})
    rows = [('model', 'quantity', 'value')]

    for col, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        z, t, zc, zkm = c['z'], c['t'], c['zc'], c['z'] / 1e3
        TP = -np.trapezoid(c['p_T'][:, zc], z[zc], axis=1) / 1e12
        RP = np.trapezoid(c['p_R'][:, zc], z[zc], axis=1) / 1e12
        TOT = TP + RP
        vp = np.abs(d[f'{key}_vx_TR'][:, zkm < 20].mean(axis=1))
        vt = np.gradient(d[f'{key}_xT'] / 1e3, t) / 10.0
        f = vp / (vp + vt)

        rise = lambda y: np.polyfit(t, y, 1)[0] * (t[-1] - t[0])
        share = 100 * rise(RP) / (rise(TP) + rise(RP))
        hTP, hRP, hTOT = hp(TP), hp(RP), hp(TOT)
        tshare = 100 * hTP.std() / (hTP.std() + hRP.std())

        a0, a1, a2 = (axes[r, col] for r in range(3))
        for arr, c_, lw, nm in ((TOT, C_TOTAL, 2.2, 'total $\\Delta$GPE*'),
                                (TP, C_TRENCH, 1.8,
                                 f'trench pull ({100-share:.0f} % of the rise)'),
                                (RP, C_RIDGE, 1.8,
                                 f'ridge push ({share:.0f} % of the rise)')):
            a0.plot(t, arr, '-', color=c_, lw=lw, label=nm)
        a0.set_title(key, fontsize=11)
        a0.set_ylabel('[TN/m]', fontsize=10)
        a0.legend(frameon=False, fontsize=8.5, loc='upper left')

        for arr, c_, lw, nm in ((hTOT, C_TOTAL, 2.0, 'total'),
                                (hTP, C_TRENCH, 1.7,
                                 f'trench pull ({tshare:.0f} % of the transient)'),
                                (hRP, C_RIDGE, 1.7,
                                 f'ridge push ({100-tshare:.0f} %)')):
            a1.plot(t, arr, '-', color=c_, lw=lw, label=nm)
        a1.axhline(0, color='k', lw=0.8)
        a1.set_ylabel('high-passed [TN/m]', fontsize=9.5)
        a1.legend(frameon=False, fontsize=8, loc='upper left', ncol=1)

        rv_tp = float(np.corrcoef(hp(vp), hTP)[0, 1])
        rv_rp = float(np.corrcoef(hp(vp), hRP)[0, 1])
        rv_tot = float(np.corrcoef(hp(vp), hTOT)[0, 1])
        # the cancellation: the total varies less than its larger part
        cancel = 100 * (1 - hTOT.std() / max(hTP.std(), hRP.std()))
        a2.plot(t, hp(vp), '-', color=C_PLATE, lw=1.8,
                label=f'plate speed  ($r$: trench pull {rv_tp:+.2f}, ridge push '
                      f'{rv_rp:+.2f}, TOTAL {rv_tot:+.2f})')
        a2.axhline(0, color='k', lw=0.8)
        a2.set_ylabel('high-passed\n[cm/yr]', fontsize=9.5)
        a2.set_xlabel('Model time [Myr]', fontsize=11)
        a2.legend(frameon=False, fontsize=8, loc='upper left')
        a2b = a2.twinx()
        a2b.plot(t, f, '-', color='0.55', lw=1.2)
        a2b.set_ylabel('$f$ (grey)', fontsize=9, color='0.45')
        a2b.tick_params(axis='y', labelcolor='0.45', labelsize=8)

        for ax in (a0, a1, a2):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        print(f'{key}: secular rise TP {rise(TP):+.2f}, RP {rise(RP):+.2f} TN/m '
              f'(ridge {share:.0f} %); transient sd TP {hTP.std():.3f}, '
              f'RP {hRP.std():.3f} (trench {tshare:.0f} %); '
              f'r(hp speed, hp TP) {rv_tp:+.2f}, RP {rv_rp:+.2f}, '
              f'TOTAL {rv_tot:+.2f}; total transient sd {hTOT.std():.3f} '
              f'-> cancellation {cancel:.0f} %')
        rows += [(key, 'secular_rise_trench_pull_TNm', f'{rise(TP):.3f}'),
                 (key, 'secular_rise_ridge_push_TNm', f'{rise(RP):.3f}'),
                 (key, 'ridge_share_of_secular_rise_percent', f'{share:.1f}'),
                 (key, 'transient_sd_trench_pull_TNm', f'{hTP.std():.4f}'),
                 (key, 'transient_sd_ridge_push_TNm', f'{hRP.std():.4f}'),
                 (key, 'trench_share_of_transient_percent', f'{tshare:.1f}'),
                 (key, 'corr_hp_speed_hp_trench_pull', f'{rv_tp:.3f}'),
                 (key, 'corr_hp_speed_hp_ridge_push', f'{rv_rp:.3f}'),
                 (key, 'corr_hp_speed_hp_total', f'{rv_tot:.3f}'),
                 (key, 'transient_sd_total_TNm', f'{hTOT.std():.4f}'),
                 (key, 'cancellation_percent', f'{cancel:.1f}'),
                 (key, 'highpass_cutoff_Myr', f'{2*HP_SIGMA*2:.0f}')]

    fig.suptitle('The driving term by domain: the secular rise is isostatic; '
                 'on transients the two domains oppose each other', fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_gpe_decomposition.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('gpe_decomposition', rows[0], rows[1:],
                      script='fig_gpe_decomposition.py',
                      figure='fig_gpe_decomposition.png', models=('STD', 'WAL'),
                      meta={'zc_km': cpc.ZC_KM,
                            'total': 'trench pull + ridge push, both over 0..z_c',
                            'highpass': f'Gaussian sigma {HP_SIGMA} samples '
                                        '(mode=nearest), ~24 Myr cutoff'})
    print('written:', tab)


if __name__ == '__main__':
    main()
