"""fig_thickness_definitions — what the mechanical-thickness definitions
measure, and how they sit against the two depths the force balance uses.

Writes figures/fig_thickness_definitions.png and
tables/thickness_definitions.csv from the committed column-profile cache
(run scripts/column_profiles_cache.py first).

SI FIGURE requested by Dan 2026-10-08, to back the thickness definitions
used in the main text. One row per model, mid-run average (36-44 Myr):

  left   the FIBRE STRESS sigma_xx - sigma_zz at the column of maximum
         bending moment -- the profile every strength-based definition is
         read from. Marked on it: the neutral plane h_np (extremum of the
         cumulative fibre stress), twice it, the depth at which the
         envelope last exceeds the 25 MPa strength cap, and the 900 C
         isotherm.
         The three proxies the main text retains (2 h_np, 25 MPa cap,
         900 C) are spanned by a shaded band.

  right  the vertical normal stress anomalies of the trench and ridge
         columns relative to the first isostatic column, as in the
         main-text column-anomaly figure, so that the band can be read
         against where each anomaly actually equilibrates. The same band
         is drawn, with the two depths the force balance uses:

             z_c  = 80 km, the FIXED NOMINAL integration depth
             z*         the depth at which the ridge-side anomaly changes
                        sign (the manuscript's definition), which is also
                        where the cumulative ridge push is stationary

WHY THE TWO PANELS BELONG TOGETHER. The thickness definitions are read
from the fibre stress, which is a flexural quantity at one column; the
force balance integrates the vertical normal stress over the whole plate.
They are different measurements and nothing requires them to agree. The
figure's point is that the strength-based band sits well above both z_c
and z*, i.e. the plate the force balance integrates over is thicker than
the plate the flexural definitions return -- which is why z_c is NOT
presented as a mechanical thickness.

The fibre-stress profile and the thickness definitions follow
fig_mechanical_thickness exactly (same helper); nothing is recomputed
here with a different convention.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fig_mechanical_thickness import thicknesses, YIELD_CAP_MPA, T_MECH_C
from fluidity_helpers import depth_ticks
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {'STD': '#002147', 'WAL': '#E5007D'}
C_TRENCH, C_RIDGE = '#0072B2', '#D55E00'     # domain colours, FIGURE_STYLE.md
C_BAND, C_RULE = '0.55', '#BFC3D1'
Z_MAX_KM = 150.0


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    sm = lambda a: gaussian_filter1d(a, 2)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.0), sharey=True)
    rows = [('model', 'quantity', 'value')]

    for r, k in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, k)
        z, zk, m = c['z'], c['z'] / 1e3, c['mid']
        th = thicknesses(d, k)
        # mid-run medians of the three retained proxies
        prox = {n: np.nanmedian(np.asarray(th[n])[m]) / 1e3
                for n in ('np2', 'yield_cap', 'thermal')}
        h_np = np.nanmedian(np.asarray(th['np2'])[m]) / 2e3
        lo, hi = min(prox.values()), max(prox.values())

        # EVERY depth marker is named in a LEGEND, with its value in the
        # legend entry (Dan, 2026-10-08). Inline labels were tried and are
        # unusable here: the three proxies fall within 5 km of one another,
        # h_np sits on its own line, and z_c and z* coincide in STD, so no
        # placement avoids text sitting on a rule. Both panels have a large
        # empty lower half, which is where the legends go.

        # ---- left: the fibre stress the definitions are read from -------
        a0 = axes[r, 0]
        fib = sm((d[f'{k}_sxx_M'] - d[f'{k}_szz_M'])[m].mean(axis=0)) / 1e6
        a0.axhspan(lo, hi, color=C_BAND, alpha=0.16, lw=0, zorder=0,
                   label=f'proxy span  {lo:.0f}–{hi:.0f} km')
        a0.plot(fib, zk, '-', color=C[k], lw=1.9)
        a0.axvline(0, color='0.55', lw=1.3)
        # THE GUIDE LINES MUST BE THE CRITERION THAT IS ACTUALLY PICKED.
        # These were drawn at +-10 % of the peak and labelled as such, which
        # stopped being the criterion when the cap became an absolute
        # 25 MPa on 2026-10-10: the computation moved and the visual
        # reference did not, so the figure about the thickness definitions
        # was drawing the retired one. Caught by Dan, 2026-10-10.
        for n, v in enumerate((-YIELD_CAP_MPA, YIELD_CAP_MPA)):
            a0.axvline(v, color='0.45', lw=0.8, ls=':',
                       label=f'{YIELD_CAP_MPA:.0f} MPa cap' if n == 0 else None)
        for zz, ls, lab in (
                # ONE DECIMAL (Dan, 2026-10-08). h_np is picked at a cell
                # centre on the 1 km grid, so it always lands on a half
                # kilometre; rounding it to whole km while 2h_np stays
                # exact printed 28 and 55 in WAL, 32 and 63 in STD.
                (h_np, (0, (6, 3)), rf'$h_{{np}}$  {h_np:.1f} km'),
                (prox['np2'], '-', rf'$2h_{{np}}$  {prox["np2"]:.0f} km'),
                (prox['yield_cap'], '-.',
                 f'{YIELD_CAP_MPA:.0f} MPa envelope base  '
                 f'{prox["yield_cap"]:.0f} km'),
                (prox['thermal'], (0, (1, 2)),
                 rf'900 $^\circ$C  {prox["thermal"]:.0f} km')):
            a0.axhline(zz, color='0.35', lw=1.1, ls=ls, label=lab)
        a0.set_xlabel(r'$\sigma_{xx}-\sigma_{zz}$  at max $M$  [MPa]', fontsize=10)
        a0.legend(frameon=False, fontsize=8, loc='lower left',
                  handlelength=2.4, handletextpad=0.7, labelspacing=0.5,
                  borderaxespad=0.8)

        # ---- right: where the anomalies equilibrate ---------------------
        a1 = axes[r, 1]
        pT = sm(c['p_T'][m].mean(axis=0)) / 1e6
        pR = sm(c['p_R'][m].mean(axis=0)) / 1e6
        a1.axhspan(lo, hi, color=C_BAND, alpha=0.16, lw=0, zorder=0,
                   label=f'proxy span  {lo:.0f}–{hi:.0f} km')
        a1.plot(pT, zk, '-', color=C_TRENCH, lw=1.7, label='trench column')
        a1.plot(pR, zk, '-', color=C_RIDGE, lw=1.5, label='ridge column')
        a1.axvline(0, color='0.55', lw=1.3)
        # z*: where the RIDGE-side anomaly changes sign (manuscript's
        # definition); also where the cumulative ridge push is stationary,
        # necessarily, since the integrand vanishes there.
        j = np.where(np.diff(np.sign(pR)))[0]
        zstar = zk[j[0]] + (zk[j[0]+1]-zk[j[0]]) * (0-pR[j[0]])/(pR[j[0]+1]-pR[j[0]])
        a1.axhline(cpc.ZC_KM, color='0.35', lw=1.1, ls=(0, (6, 3)),
                   label=rf'$z_c$  {cpc.ZC_KM:.0f} km (nominal)')
        a1.axhline(zstar, color=C_RIDGE, lw=1.1, ls=(0, (1, 2)),
                   label=rf'$z_{{\ast}}$  {zstar:.0f} km')
        a1.set_xlabel(r'$-\Delta\sigma_{zz}$  (column $-\ x_I$)  [MPa]', fontsize=10)
        a1.legend(frameon=False, fontsize=8, loc='lower left',
                  handlelength=2.4, handletextpad=0.7, labelspacing=0.5,
                  borderaxespad=0.8)

        for ax in (a0, a1):
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        a0.set_ylabel('Depth [km]')
        a0.set_title(f'({"ac"[r]})  {k}   fibre stress at max $M$', fontsize=10,
                     loc='left')
        a1.set_title(f'({"bd"[r]})  {k}   column anomalies', fontsize=10,
                     loc='left')

        print(f'{k}: 2h_np {prox["np2"]:.0f}, 25 MPa cap {prox["yield_cap"]:.0f}, '
              f'900 C {prox["thermal"]:.0f} km  -> band {lo:.0f}–{hi:.0f} km; '
              f'z_c {cpc.ZC_KM:.0f}, z* {zstar:.0f} km')
        rows += [(k, 'h_np_km', f'{h_np:.1f}'),
                 (k, 'two_h_np_km', f'{prox["np2"]:.1f}'),
                 (k, 'yield_cap_km', f'{prox["yield_cap"]:.1f}'),
                 (k, 'isotherm_900C_km', f'{prox["thermal"]:.1f}'),
                 (k, 'proxy_band_low_km', f'{lo:.1f}'),
                 (k, 'proxy_band_high_km', f'{hi:.1f}'),
                 (k, 'zc_nominal_km', f'{cpc.ZC_KM:.1f}'),
                 (k, 'z_star_km', f'{zstar:.1f}')]

    axes[0, 0].set_ylim(Z_MAX_KM, 0)
    fig.suptitle('Mechanical-thickness proxies and integration depths',
                 fontsize=10.5, color='0.35')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_thickness_definitions.png')
    depth_ticks(fig)   # 20 km depth ticks (FIGURE_STYLE)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('written:', out)
    tab = write_table('thickness_definitions', rows[0], rows[1:],
                      script='fig_thickness_definitions.py',
                      figure='fig_thickness_definitions.png',
                      models=('STD', 'WAL'),
                      meta={'midrun_window_Myr': list(cpc.MIDRUN_MYR),
                            'isotherm_C': T_MECH_C,
                            'z_star': 'depth at which the ridge-minus-x_I '
                                      'anomaly changes sign'})
    print('written:', tab)


if __name__ == '__main__':
    main()
