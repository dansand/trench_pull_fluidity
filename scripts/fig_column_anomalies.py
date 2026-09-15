"""fig_column_anomalies — trench and ridge column stress anomalies vs depth.

Writes figures/fig_column_anomalies.png from the column_profiles_cache
(run scripts/column_profiles_cache.py first to build/refresh the cache).

Two panels (STD | WAL), dimensional, pressure-register (positive = higher
vertical normal pressure than the first isostatic column). Solid black =
trench − x_I, mid-run average; solid grey = ridge − x_I, mid-run average;
dashed = the ridge anomaly with the asthenospheric part removed (shifted
by |ΔP| so its deep asymptote is zero): the untilted ridge column. Light bands =
full-run range (8–80 Myr). Dotted line: z_c = 75 km; shaded band: the deep
ΔP band (150–220 km).

Sign pin (asserted before rendering): the trench-lobe integral over 0..z_c
must reproduce the committed f10 trench pulls (1.71 STD / 1.74 WAL TN/m)
to within 3 %.

DRAFT CAPTION. Vertical normal stress anomalies of the trench column
(black) and ridge column (grey) relative to the first isostatic column,
for the Fluidity models STD (left) and WAL (right); curves are averages
over the mid-run window (36–44 Myr) and bands show the full range through
the run (8–80 Myr). The trench deficit is confined to the boundary layer,
closing by z_c = 75 km (dotted); its area is the trench pull. The ridge
column crosses zero and holds a finite deep deficit ΔP (shaded band) — the
adverse asthenospheric pressure gradient. The dashed curve removes the
asthenospheric part (shifts the curve by |ΔP| onto a zero deep
asymptote): the ridge column in the absence of the tilt, whose enlarged
area is the untilted ridge push — the difference between the areas is the
tilting force F_tilt = |ΔP|·z_c by definition.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc

COMMITTED_F10_TP = {'STD': 1.71e12, 'WAL': 1.74e12}   # N/m, ±5 km, f10

def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    sm = lambda a: gaussian_filter1d(a, 2, axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.6), sharey=True, sharex=True)
    for ax, k in zip(axes, ('STD', 'WAL')):
        c = cpc.derive(d, k)
        z, zkm = c['z'], c['z'] / 1e3
        pT, pR = sm(c['p_T']) / 1e6, sm(c['p_R']) / 1e6
        pRd = sm(c['p_R_untilted']) / 1e6
        # sign pin against the committed values
        tp = -np.trapezoid(c['p_T'][:, c['zc']], z[c['zc']], axis=1)
        i10 = np.argmin(np.abs(c['t'] - 20.0))
        rel = abs(tp[i10] - COMMITTED_F10_TP[k]) / COMMITTED_F10_TP[k]
        assert rel < 0.03, f'{k}: trench-lobe integral {tp[i10]/1e12:.2f} TN/m vs committed — sign chain broken'
        print(f'{k}: sign pin OK — trench-lobe integral f10 {tp[i10]/1e12:+.2f} TN/m '
              f'(committed {COMMITTED_F10_TP[k]/1e12:.2f}); '
              f'dP band {cpc.DP_BAND_KM} km: f10 {c["dP"][i10]/1e6:+.2f} MPa, '
              f'median {np.median(c["dP"])/1e6:+.2f} MPa')
        m = c['mid']
        ax.fill_betweenx(zkm, pT.min(axis=0), pT.max(axis=0), color='0.85', lw=0)
        ax.fill_betweenx(zkm, pR.min(axis=0), pR.max(axis=0), color='0.92', lw=0)
        ax.plot(pT[m].mean(axis=0), zkm, 'k-', lw=1.7, label='trench $-$ first isostatic')
        ax.plot(pR[m].mean(axis=0), zkm, '-', color='0.4', lw=1.5, label='ridge $-$ first isostatic')
        ax.plot(pRd[m].mean(axis=0), zkm, 'k--', lw=1.2,
                label='ridge, tilt ($\\Delta P$) removed')
        ax.axvline(0, color='0.7', lw=0.7)
        ax.axhline(cpc.ZC_KM, color='0.5', lw=0.8, ls=':')
        ax.axhspan(*cpc.DP_BAND_KM, color='0.6', alpha=0.12, lw=0)
        ax.set_title(f'{k}  (avg {c["t"][m].min():.0f}–{c["t"][m].max():.0f} Myr, n={m.sum()})',
                     fontsize=10)
        ax.set_xlabel('Vertical normal stress anomaly [MPa]\n(pressure-positive)')
    axes[0].set_ylabel('Depth [km]')
    axes[0].set_ylim(230, 0)
    axes[0].text(0.03, 0.34, '$z_c$', transform=axes[0].get_yaxis_transform(),
                 fontsize=8, color='0.4')
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Column stress anomalies relative to the first isostatic column\n'
                 '(bands: full-run range, 8–80 Myr)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures', 'fig_column_anomalies.png')
    fig.savefig(out, dpi=200)
    print('written:', out)

if __name__ == '__main__':
    main()
