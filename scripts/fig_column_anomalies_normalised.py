"""fig_column_anomalies_normalised — run-averaged column anomalies scaled
by the trench pressure deficit.

Writes figures/fig_column_anomalies_normalised.png from the
column_profiles_cache (run scripts/column_profiles_cache.py first).

Each snapshot's trench AND ridge anomaly curves are divided by that
snapshot's trench pressure deficit ΔP_T (the peak shallow trench anomaly,
= Δρ g w_T to good approximation) before averaging over the whole run —
one common scale, so the two curves' areas over 0..z_c display the
ridge-push : trench-pull proportion directly. The dashed curve is the
ridge anomaly with the asthenospheric part removed (shifted by |ΔP|),
normalised the same way: the untilted ridge column. Lobe areas above z_c are shaded; the area ratio is printed
in each panel title.

DRAFT CAPTION. Run-averaged vertical normal stress anomalies of the trench
(black) and ridge (grey) columns relative to the first isostatic column,
normalised per snapshot by the trench pressure deficit ΔP_T (37 snapshots,
8–80 Myr). Shaded areas above z_c = 75 km show the two lobes on a common
scale: their ratio is the model-averaged ridge-push to trench-pull
proportion. The dashed curve removes the asthenospheric part
(|ΔP| shift): the untilted ridge column, whose enlarged area is the
ridge push in the absence of the tilting force.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc

def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    sm = lambda a: gaussian_filter1d(a, 2, axis=-1)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.6), sharey=True, sharex=True)
    for ax, k in zip(axes, ('STD', 'WAL')):
        c = cpc.derive(d, k)
        z, zkm, zc = c['z'], c['z'] / 1e3, c['zc']
        n = c['dP_T'][:, None]
        mT = (sm(c['p_T']) / n).mean(axis=0)
        mR = (sm(c['p_R']) / n).mean(axis=0)
        mRd = (sm(c['p_R_untilted']) / n).mean(axis=0)
        ax.fill_betweenx(zkm[zc], 0, mT[zc], color='0.55', alpha=0.45, lw=0)
        ax.fill_betweenx(zkm[zc], 0, mR[zc], color='0.80', alpha=0.55, lw=0)
        ax.plot(mT, zkm, 'k-', lw=1.7, label='trench $-$ first isostatic')
        ax.plot(mR, zkm, '-', color='0.4', lw=1.5, label='ridge $-$ first isostatic')
        ax.plot(mRd, zkm, 'k--', lw=1.2, label='ridge, tilt ($\\Delta P$) removed')
        ax.axhline(cpc.ZC_KM, color='0.5', lw=0.8, ls=':')
        ax.axvline(0, color='0.7', lw=0.7)
        aT = -np.trapezoid(mT[zc], z[zc])
        aR = np.trapezoid(mR[zc], z[zc])
        ax.set_title(f'{k}:  ridge-lobe area / trench-lobe area = {aR/aT:.2f}', fontsize=10)
        ax.set_xlabel('Stress anomaly / $\\Delta P_T$')
        print(f'{k}: area ratio {aR/aT:.2f}; mean dP_T {c["dP_T"].mean()/1e6:.1f} MPa; '
              f'n={len(c["t"])} snapshots')
    axes[0].set_ylabel('Depth [km]')
    axes[0].set_ylim(230, 0)
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Run-averaged column anomalies, normalised per snapshot by the\n'
                 'trench pressure deficit $\\Delta P_T$ (8–80 Myr)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures', 'fig_column_anomalies_normalised.png')
    fig.savefig(out, dpi=200)
    print('written:', out)

if __name__ == '__main__':
    main()
