"""fig_column_anomalies — trench and ridge column stress anomalies vs depth.

Writes figures/fig_column_anomalies.png from the column_profiles_cache
(run scripts/column_profiles_cache.py first to build/refresh the cache).

Two panels (STD | WAL), dimensional, pressure-register (positive = higher
vertical normal pressure than the first isostatic column). Solid blue =
trench − x_I, mid-run average; solid orange = ridge − x_I, mid-run average;
dashed orange = the ridge anomaly with the asthenospheric part removed
(shifted by |ΔP| so its deep asymptote is zero): the untilted ridge
column. Light bands = full-run range (8–80 Myr). Dotted line: z_c = 75 km;
shaded band: the deep ΔP band (150–220 km).

DOMAIN COLOURS (Dan, 2026-09-21). Blue and orange are NOT decorative and
are not this figure's own choice: they are the manuscript schematic's two
domain colours, carried over so the reader meets the same pair in the
schematic and in the data. Blue is the NON-ISOSTATIC DOMAIN (x_T to x_I),
also called the TRENCH PULL DOMAIN; orange is the ISOSTATIC DOMAIN (x_I to
x_R), also called the RIDGE PUSH DOMAIN. Each curve is drawn in the colour
of the domain whose force its area represents. Values and provenance are
in FIGURE_STYLE.md; the canonical source is the schematic's TikZ
(trench_pull_ferrite/schematic/ridge_trench_overview_v2.tex, cboxA/cboxB).

⚠ "Ridge push domain" is a name for the INTERVAL, not a claim that its
force balance is purely isostatic ridge push. The orange curve's deep
part is the asthenospheric pressure gradient ΔP, which is exactly what
the dashed curve removes — the interval is isostatic in the sense that
the topography there is compensated, while the Δσ_zz that the balance
integrates over it still carries the tilt.

Sign pin (asserted before rendering): the trench-lobe integral over 0..z_c
must reproduce the committed f10 trench pulls (1.71 STD / 1.74 WAL TN/m)
to within 3 %.

DRAFT CAPTION. Vertical normal stress anomalies of the trench column
(blue) and ridge column (orange) relative to the first isostatic column,
in the domain colours of the schematic — blue the non-isostatic or trench
pull domain, orange the isostatic or ridge push domain —
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

# DOMAIN COLOURS, taken from the manuscript schematic (Dan, 2026-09-21).
# Source of truth: trench_pull_ferrite/schematic/ridge_trench_overview_v2.tex
# (cboxA / cboxB), the TikZ behind figures/ridge_trench_overview_v2.pdf; its
# caption reads "the non-isostatic domain (x_T to x_I, blue) ... the
# isostatic domain (x_I to x_R, orange)". Okabe-Ito, colourblind-safe.
# Using them here ties each curve to the domain whose force it carries, so
# the reader meets the same two colours in the schematic and in the data.
C_TRENCH = '#0072B2'        # cboxA — non-isostatic / TRENCH PULL domain
C_RIDGE = '#D55E00'         # cboxB — isostatic / RIDGE PUSH domain

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
        ax.plot(pT[m].mean(axis=0), zkm, '-', color=C_TRENCH, lw=1.7,
                label='trench $-$ first isostatic')
        ax.plot(pR[m].mean(axis=0), zkm, '-', color=C_RIDGE, lw=1.5,
                label='ridge $-$ first isostatic')
        ax.plot(pRd[m].mean(axis=0), zkm, '--', color=C_RIDGE, lw=1.2, alpha=0.45,
                label='ridge, tilt ($\\Delta P$) removed')
        ax.axvline(0, color='0.7', lw=0.7)
        ax.set_title(f'{k}  (avg {c["t"][m].min():.0f}–{c["t"][m].max():.0f} Myr, n={m.sum()})',
                     fontsize=10)
        # regime annotations (Dan, 2026-09-18): above the sign change the
        # ridge column carries the plate's own topographic pressure
        # gradient; below it the anomaly is the asthenospheric gradient,
        # which stabilises onto Delta P.
        pr = sm(c['p_R'][m].mean(axis=0)) / 1e6
        sgn = np.where((pr[:-1] > 0) & (pr[1:] <= 0) & (zkm[:-1] > 20))[0]
        z_x = zkm[sgn[0]] if len(sgn) else np.nan
        ax.axhline(z_x, color='0.35', lw=0.9, ls='--')
        # name the two forces where their amplitude is largest, in the
        # corridor between the trench and ridge bands (Dan, 2026-09-18)
        ax.text(-9, 10, 'trench\npull', ha='right', va='center', fontsize=9.5,
                color=C_TRENCH, fontweight='bold', linespacing=1.3)
        ax.text(9, 10, 'ridge\npush', ha='left', va='center', fontsize=9.5,
                color=C_RIDGE, fontweight='bold', linespacing=1.3)
        # Regime labels: two lines each, set WELL CLEAR of the sign change
        # (one shallower, one much deeper), each with an arrow giving the
        # direction of the force it produces (Dan, 2026-09-18).
        # Each label sits on ITS OWN SIDE of the sign change -- plate above,
        # asthenosphere below -- with the arrow beside it giving the
        # direction of the force. Two balanced lines, not a one-word first
        # line (Dan, 2026-09-18).
        # Plate label in the empty LEFT region above the sign change,
        # asthenosphere label in the empty RIGHT region below it; each with
        # its force-direction arrow beneath. Trench-ward is left.
        for xc, z_lab, z_arr, txt, head in (
                (0.24, z_x - 18, z_x - 4, 'plate: topographic\npressure gradient', -1),
                (0.755, z_x + 26, z_x + 42, 'asthenosphere: adverse\npressure gradient', +1)):
            ax.text(xc, z_lab, txt, transform=ax.get_yaxis_transform(),
                    ha='center', va='center', fontsize=8.5, color='0.25',
                    linespacing=1.4)
            ax.annotate('', xy=(xc + head * 0.06, z_arr), xytext=(xc - head * 0.06, z_arr),
                        xycoords=ax.get_yaxis_transform(),
                        textcoords=ax.get_yaxis_transform(),
                        arrowprops=dict(arrowstyle='-|>', color='0.25', lw=2.2,
                                        mutation_scale=16))
        ax.set_xlabel('Vertical normal stress anomaly [MPa]\n(pressure-positive)')
    axes[0].set_ylabel('Depth [km]')
    axes[0].set_ylim(200, 0)
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Column stress anomalies relative to the first isostatic column\n'
                 '(bands: full-run range, 8–80 Myr; dashed: $\\Delta\\sigma_{zz}$ sign change)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures', 'fig_column_anomalies.png')
    fig.savefig(out, dpi=200)
    print('written:', out)

if __name__ == '__main__':
    main()
