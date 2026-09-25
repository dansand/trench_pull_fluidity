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

DRAFT CAPTION. Vertical normal stress anomalies of the trench and ridge
columns relative to the first isostatic column, for STD (left) and WAL
(right), in the pressure-positive register. Solid curves are means over
the mid-run window (36-44 Myr, n = 5, and 38-44 Myr, n = 4); the grey
bands are the full range over 8-80 Myr. Blue is the anomaly of the trench
column and orange that of the ridge column; the areas they enclose are
the trench pull and the ridge push, and the domains they span are the
non-isostatic domain between the trench and the first isostatic column
and the isostatic domain between that column and the ridge. The dagger on
'ridge push' marks that the name refers to the interval and not to its
content: the balance across it carries the adverse asthenospheric
pressure gradient as well as the isostatic cooling signal, so the force
delivered there is smaller than static ridge push. The
dashed orange curve is the ridge anomaly with the deep offset Delta P
subtracted, so that its deep asymptote is zero. The horizontal dashed
line marks the depth at which the ridge anomaly changes sign, 80 km in
STD and 86 km in WAL; it is a change in the sign of the stress anomaly
and not a change in the sense of flow, which occurs deeper, at the base
of the coherently translating plate. The two arrows give the direction
of the force each gradient exerts on the plate. Delta P is the mean of
the ridge anomaly over 150-220 km, the interval over which it asymptotes.
Quantities are +-5 km column means and all snapshots at t >= 8 Myr enter
the range bands.
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
                label='trench column')
        ax.plot(pR[m].mean(axis=0), zkm, '-', color=C_RIDGE, lw=1.5,
                label='ridge column')
        ax.plot(pRd[m].mean(axis=0), zkm, '--', color=C_RIDGE, lw=1.2, alpha=0.45,
                label='ridge, $\\Delta P$ removed')
        ax.axvline(0, color='0.55', lw=1.3)
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
        # DOMAIN names first, mechanism in brackets (Dan, 2026-09-26): the
        # schematic's two additive domains are what the colours encode, and
        # "trench pull"/"ridge push" are the forces they carry.
        # SHORT names, daggered (Dan, 2026-09-26). The full domain names
        # ('non-isostatic domain (trench pull)') were tried and reverted:
        # at this width they cross both the curves and the range bands, and
        # no placement fixes that without moving them away from the lobes
        # they name. The dagger sends the reader to the caption, where the
        # domains are named. A DAGGER not an asterisk -- '*' already carries
        # a specific meaning in this paper (GPE*), and reusing it as a
        # footnote marker invites a misreading.
        # The dagger goes on RIDGE PUSH ONLY (Dan, 2026-09-26). It is not a
        # generic "see caption" marker: it flags the one label that needs a
        # caveat. SYMBOLOGY §4.4 -- the ridge push domain names the INTERVAL,
        # never its content, because the balance across it carries the
        # adverse asthenospheric gradient as well as the isostatic cooling
        # signal. 'Trench pull' carries no such ambiguity and takes no mark.
        ax.text(-9, 10, 'trench\npull', ha='right', va='center',
                fontsize=9.5, color=C_TRENCH, fontweight='bold',
                linespacing=1.3)
        ax.text(9, 10, 'ridge\npush$\\,\\dagger$', ha='left', va='center',
                fontsize=9.5, color=C_RIDGE, fontweight='bold',
                linespacing=1.3)
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
        # The two gradients are now named PLAINLY (Dan, 2026-09-26). The
        # old wording ("plate: topographic" above, "asthenosphere: adverse"
        # below, both hugging the dashed line) implied that the FLOW
        # reverses at the sigma_zz sign change. It does not: the sign change
        # sits at 80 km (STD) / 86 (WAL) while the base of the coherently
        # translating plate -- the dynamical LAB -- is at 99 / 93 km
        # (fig_lab_kinematics, W17). The dashed line marks where the stress
        # anomaly changes sign, nothing more.
        # The lower arrow is also moved well below it, into the interval
        # where the anomaly has asymptoted, so that neither label can be
        # read as locating a transition.
        # FIXED depths, not offsets from z_x (Dan, 2026-09-26): the sign
        # change sits at 75.5 km in STD and 85.5 in WAL, so anchoring the
        # plate label to it put the two panels' annotations at different
        # heights. Fixed values put them level, and 62 km still clears the
        # dashed line in both. The arrow now sits 12 km below its label
        # rather than 16-20, so it reads as part of the label.
        for xc, z_lab, z_arr, txt, head in (
                (0.24, 50, 62, 'plate pressure\ngradient', -1),
                (0.755, 112, 124, 'counterflow pressure\ngradient', +1)):
            ax.text(xc, z_lab, txt, transform=ax.get_yaxis_transform(),
                    ha='center', va='center', fontsize=8.5, color='0.25',
                    linespacing=1.4)
            ax.annotate('', xy=(xc + head * 0.06, z_arr), xytext=(xc - head * 0.06, z_arr),
                        xycoords=ax.get_yaxis_transform(),
                        textcoords=ax.get_yaxis_transform(),
                        arrowprops=dict(arrowstyle='-|>', color='0.25', lw=2.2,
                                        mutation_scale=16))
        # The deep asthenospheric anomaly, labelled with its SIGN, rounded
        # to whole MPa, and stated with the band it is sampled over -- the
        # interval in which it asymptotes (Dan, 2026-09-26).
        # Sampled over 150-200 km, matching the plotted depth range, rather
        # than the cache's 150-220 band (Dan, 2026-09-26). The anomaly has
        # asymptoted by 150 km, so the two differ by 0.02 MPa and round to
        # the same integer -- the figure stays consistent with every force
        # quantity computed from cpc.DP_BAND_KM while quoting a band the
        # reader can see.
        zlo, zhi = 150.0, 200.0
        band = (zkm >= zlo) & (zkm <= zhi)
        dP_mid = c['p_R'][m].mean(axis=0)[band].mean() / 1e6
        # number inside mathtext so the sign renders as a minus, not a hyphen
        ax.annotate(f'$\\Delta P \\approx {dP_mid:.0f}$ MPa\n'
                    f'({zlo:.0f}–{zhi:.0f} km)',
                    xy=(dP_mid, 175), xytext=(-46, 175),
                    ha='center', va='center', fontsize=8.5, color=C_RIDGE,
                    linespacing=1.35,
                    arrowprops=dict(arrowstyle='-', color=C_RIDGE, lw=0.9,
                                    shrinkA=2, shrinkB=2))
        ax.set_xlabel('Vertical normal stress anomaly [MPa]\n(pressure-positive)')
    axes[0].set_ylabel('Depth [km]')
    # 200 km (Dan, 2026-09-26). The curves are flat well before it, so the
    # asymptote is established on-plot; the Delta P band runs to 220 km and
    # its upper part is off-plot, which is why the annotation states the
    # band explicitly rather than relying on the axis to show it.
    axes[0].set_ylim(200, 0)
    # lower RIGHT: the deep left side now carries the Delta P annotation,
    # and below ~150 km the right side is empty in both models.
    # Short labels carry the clearance, so the size does not have to: at
    # 8.5 pt the block still ends near x = +14, clear of the curves at
    # -8..0. The title already says the anomalies are relative to the first
    # isostatic column, so the legend need only name which column each
    # curve is.
    axes[0].legend(fontsize=8.5, frameon=False, loc='lower right')
    # Title names the subject only; the bands, the dashed line and the
    # averaging windows are caption material (Dan, 2026-09-26).
    fig.suptitle('Column stress anomalies relative to the first isostatic column',
                 fontsize=10.5, color='0.35')
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures', 'fig_column_anomalies.png')
    fig.savefig(out, dpi=200)
    print('written:', out)

if __name__ == '__main__':
    main()
