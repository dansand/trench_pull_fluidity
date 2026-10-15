"""fig_column_anomalies_cumulative — column anomalies AND what they integrate to.

Writes figures/fig_column_anomalies_cumulative.png and
tables/column_anomalies_cumulative.csv from the committed column-profile
cache (run scripts/column_profiles_cache.py first).

Requested by Dan 2026-09-21 as a variant of fig_column_anomalies, which
stays as it is. Four panels, one ROW PER MODEL (STD, WAL):

  left   Delta sigma_zz vs depth, exactly as fig_column_anomalies: trench
         - x_I (blue) and ridge - x_I (orange), plus the ridge curve with
         the asthenospheric tilt removed (dashed), mid-run average with
         the full-run range as a light band.
  right  what those curves INTEGRATE TO: the cumulative GPE* difference
         from the surface down, so the reader can see the force
         accumulate with depth rather than having to integrate the left
         panel by eye. Three curves:
             trench pull  = -int p_T dz   (blue, the NON-ISOSTATIC domain)
             ridge push   = +int p_R dz   (orange, the ISOSTATIC domain)
             total        = their sum     (black), the whole
                            trench-to-ridge GPE* difference
         The two components carry the DOMAIN COLOURS of the left panel
         and of the schematic (FIGURE_STYLE.md), so a curve keeps its
         meaning across the row.

WHY THE COMPONENTS ADD. The two domains are additive and share the x_I
plane: GPE*(x_R) - GPE*(x_T) = [GPE*(x_I) - GPE*(x_T)] + [GPE*(x_R) -
GPE*(x_I)]. The first bracket is the trench pull, the second the ridge
push, and the resultants on the shared plane cancel -- the same
statement the manuscript schematic makes with its two boxes.

SIGNS follow fig_column_anomalies: the cache's p_T / p_R are
pressure-register anomalies relative to x_I, so trench pull is
-int p_T dz (the trench is a DEFICIT) and ridge push is +int p_R dz.
Both come out positive, i.e. both drive. The trench-lobe integral is
pinned against the committed f10 values before anything is drawn.

READ THE RIGHT-HAND PANELS AS A DEPTH BUDGET. The trench curve flattens
by ~75 km -- the deficit is equilibrated and the force is complete. The
ridge curve does not: it keeps climbing to the thermal thickness and
then turns over as the asthenospheric pressure gradient eats into it,
which is why the ridge push is the term that inherits the choice of
integration depth and the trench pull is not.

DRAFT CAPTION. Column stress anomalies and the forces they integrate to,
for STD (top) and WAL (bottom), averaged over the mid-run window
(36-44 Myr) with the full-run range shaded. Left: the vertical normal
stress anomaly of the trench column (blue) and ridge column (orange)
relative to the first isostatic column, with the ridge column's
asthenospheric tilt removed (dashed). Right: the cumulative GPE*
difference from the surface down -- the trench pull across the
non-isostatic domain (blue), the ridge push across the isostatic domain
(orange), and their sum, the whole trench-to-ridge difference (black).
The trench contribution is complete by the compensation depth (dotted);
the ridge contribution continues to accumulate well below it.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import column_profiles_cache as cpc
from fluidity_helpers import depth_ticks
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# N/m, +-5 km windows, t = 20 Myr. UPDATED 2026-10-10 from 1.71/1.74e12
# after the surface-read fix: the cache picked x_I from the free-surface
# field sampled 500 m below the boundary, where it retains the spatial
# pattern but only ~60 % of the amplitude. x_I moved +9 km (STD) and
# -4 km (WAL). This pin TRIPPED on the rebuild, which is what it is for --
# it is a tripwire on the sign chain and the column picks, not a target.
# RE-PINNED 2026-10-15 for the x_I rule change (conventions §3.2:
# x_I is the dV/dx zero, not the topographic pick). Was 1.93 / 1.75.
# The guard fired, which is what it is for; it was checked before
# being moved -- the values fell 5.1 % (STD) and 5.0 % (WAL) at this
# epoch with sign and magnitude intact, which is a rule-change-sized
# move, not a broken sign chain.
COMMITTED_F10_TP = {'STD': 1.86e12, 'WAL': 1.69e12}
# domain colours — the schematic's cboxA / cboxB, see FIGURE_STYLE.md
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain
C_RIDGE = '#D55E00'         # isostatic / ridge push domain
C_TOTAL = 'k'
C_RULE = '#BFC3D1'
Z_MAX_KM = 200.0
X_LIM_MPA = 80.0            # symmetric; must match fig_column_anomalies
ZC_RANGE_KM = (50.0, 125.0)  # the latitude in a fixed z_c the box represents


def cumulative(a, z):
    """Running integral of a(z) from the surface down, same length as z."""
    # CELL RULE, cumulative on CELL EDGES (2026-10-10). This was a
    # centre-to-centre trapezoid: its value at index i is the integral
    # from 0.5 km, not from 0, so a curve read at z_c gave the 79 km
    # integral under an 80 km label. Edge k now holds the integral over
    # cells 0..k, so the curve read at the z_c edge IS the reported
    # scalar. See column_profiles_cache.cumulative_to_zc.
    # edges 0..N*h, trimmed to len(z) so it pairs with the panels' depth
    # array: index k is the integral from the SURFACE to k*h, so the value
    # at the z_c edge is the reported scalar.
    return cpc.cumulative_edges(a, z)[1][:len(z)]


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    sm = lambda a: gaussian_filter1d(a, 2, axis=-1)
    # sharex='col' so the two models sit on identical scales within each
    # column — the rows are the comparison, and unequal axes would hide it
    # The right column is the ADD-ON (Dan, 2026-09-30): the anomalies are the
    # subject, the integrals are what they amount to, so it gets about half
    # the width.
    # PANEL aspect matched to fig_column_anomalies (Dan, 2026-09-30), whose
    # proportions were right: ~3.5 x 4.2 in of axes per model. The right
    # column is the add-on at half that width, so the FIGURE narrows rather
    # than the left panels widening.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 9.6), sharey=True, sharex='col',
                             gridspec_kw={'width_ratios': [1.0, 0.5]})
    rows = [('model', 'quantity', 'value')]

    for r, k in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, k)
        z, zkm = c['z'], c['z'] / 1e3
        m = c['mid']
        pT, pR = sm(c['p_T']) / 1e6, sm(c['p_R']) / 1e6

        # sign pin against the committed values, before anything is drawn
        # the shared cell rule, 0 -> z_c -- this is the SIGN-PINNED trench
        # pull, so it must be the same integral cpc.partition reports
        tp = -cpc.integrate_to_zc(c['p_T'], z, axis=1)
        i10 = np.argmin(np.abs(c['t'] - 20.0))
        rel = abs(tp[i10] - COMMITTED_F10_TP[k]) / COMMITTED_F10_TP[k]
        assert rel < 0.03, (f'{k}: trench-lobe integral {tp[i10]/1e12:.2f} TN/m '
                            'vs committed — sign chain broken')

        # ---- left: the anomalies themselves -------------------------------
        # THIS PANEL IS fig_column_anomalies (Dan, 2026-09-30). It is the
        # same panel, so it carries the same bands, colours, line weights,
        # labels, annotations and axis text; a reader moving between the two
        # figures must not have to re-learn it. Anything changed here must
        # be changed there.
        a0 = axes[r, 0]
        a0.fill_betweenx(zkm, pT.min(axis=0), pT.max(axis=0), color='0.85', lw=0, zorder=0)
        a0.fill_betweenx(zkm, pR.min(axis=0), pR.max(axis=0), color='0.92', lw=0, zorder=0)
        a0.plot(pT[m].mean(axis=0), zkm, '-', color=C_TRENCH, lw=1.7,
                label='trench column')
        a0.plot(pR[m].mean(axis=0), zkm, '-', color=C_RIDGE, lw=1.5,
                label='ridge column')
        # THE AREA IS THE FORCE (Dan, 2026-10-01). GPE* is the column
        # integral of the vertical normal stress, so the area each curve
        # encloses with the zero line IS the trench pull / ridge push --
        # the quantity plotted cumulatively in the companion panel.
        # SHADED TO z_c ONLY. The shading is the force as the paper reports
        # it and the paper integrates to z_c, so a fill running to the base
        # of the panel would shade area that no quoted number contains.
        # What lies below stays visible as unshaded curve.
        # zorder 1: ABOVE the opaque grey full-run bands (zorder 0), below
        # the curves. Drawn underneath them the bands hid the fills
        # completely -- the bands carry no alpha.
        zfill = zkm <= cpc.ZC_KM
        a0.fill_betweenx(zkm[zfill], 0, pT[m].mean(axis=0)[zfill],
                        color=C_TRENCH, alpha=0.30, lw=0, zorder=1)
        a0.fill_betweenx(zkm[zfill], 0, pR[m].mean(axis=0)[zfill],
                        color=C_RIDGE, alpha=0.30, lw=0, zorder=1)
        # LITHOSTATIC APPROXIMATION per column, surface-referenced:
        #   Delta P(z) = Delta P(0) + g int_0^z (rho_X - rho_I) dz'
        # The trench curve is terminated at 125 km: under this approximation
        # the trench anomaly never equilibrates, so its integral grows
        # without bound with the chosen depth and a curve drawn to the axis
        # floor would imply a limit that does not exist.
        for col, rk, cc in (('T', 'rho_T', C_TRENCH), ('R', 'rho_R', C_RIDGE)):
            meas = (sm(c[f'p_{col}'])[m].mean(axis=0)) / 1e6
            rd = (d[f'{k}_{rk}'] - d[f'{k}_rho_I'])[m].mean(axis=0)
            lith = meas[0] + np.concatenate(
                [[0.0], np.cumsum(rd) * float(np.diff(z)[0])])[:len(z)] * 9.8 / 1e6
            if col == 'T':
                w = zkm <= 125.0
                # VERTICAL REFERENCE at the surface value (Dan, 2026-09-30).
                # The dashed curve leans right with depth and the lean is
                # real, not a rendering artefact: the trench column runs
                # ~2 kg/m3 denser than the first isostatic column through
                # the plate, so the integral accumulates rho g dz ~ 2.5 MPa
                # over 125 km. This hairline is what NO density contrast
                # would look like, so the reader can see the lean against
                # true vertical instead of guessing at it.
                a0.plot([lith[0], lith[0]], [0.0, 125.0], '-', color=cc,
                        lw=0.6, alpha=0.55)
                # NO label on the curve itself (Dan, 2026-09-30): there are
                # TWO lithostatic curves, one per column, and labelling only
                # this one put a blue handle in the legend against a phrase
                # that covers both. A neutral grey proxy is added below.
                a0.plot(lith[w], zkm[w], '--', color=cc, lw=1.5, alpha=0.85)
                a0.text(lith[w][-1], 133.0, r'$\ldots$?', color=cc,
                        fontsize=13, ha='center', va='top')
            else:
                a0.plot(lith, zkm, '--', color=cc, lw=1.5, alpha=0.85)
        # grey proxy: the dashed style belongs to both columns, so the
        # legend key carries no column colour
        a0.plot([], [], '--', color='0.45', lw=1.5, label='lithostatic approx.')
        # SYMBOL-LED (Dan, 2026-10-01). The plotted quantity is exactly the
        # Delta GPE* integrand, so writing it as -Delta sigma_zz makes the
        # panel beside it visibly its integral, and the leading minus does
        # the work the '(pressure-positive)' parenthetical was doing.
        a0.set_xlabel(r'$-\Delta\sigma_{zz}$   (column $-\ x_I$)  [MPa]', fontsize=10)

        # --- the two intervals, named ---------------------------------
        # TRENCH SIDE / RIDGE SIDE, not trench pull / ridge push (Dan,
        # 2026-10-09). These label the two INTERVALS of the plate, which
        # is what the schematic labels, so the figure and the schematic
        # now read against each other directly.
        #
        # It also removes a caveat rather than annotating one. The old
        # 'ridge push' carried a dagger pointing at a caption note that
        # the interval's balance is NOT pure ridge push -- it contains the
        # adverse asthenospheric gradient as well as the cooling signal
        # (SYMBOLOGY §4.4: the domain name must never imply its content).
        # 'Ridge side' makes no claim about content, so the dagger has
        # nothing left to warn about and goes.
        a0.text(-9, 10, 'trench\nside', ha='right', va='center',
                fontsize=9.5, color=C_TRENCH, fontweight='bold',
                linespacing=1.3)
        a0.text(9, 10, 'ridge\nside', ha='left', va='center',
                fontsize=9.5, color=C_RIDGE, fontweight='bold',
                linespacing=1.3)
        # The two pressure gradients, each on its own side of the anomaly's
        # sign change and well clear of it, with the direction of the force
        # it produces. Trench-ward is left. Fixed depths so the two rows sit
        # level.
        for xc, z_lab, z_arr, txt, head in (
                (0.755, 50, 62, 'plate pressure\ngradient', -1),
                # 'ADVERSE', not 'counterflow' (ruled 2026-10-04). The
                # gradient opposes plate motion at every depth below the
                # crossing, but the flow does not reverse until ~200 km --
                # and that depth is set by the plate velocity, not by the
                # mantle, so naming the region for it names an accident.
                # 'Counterflow' is reserved for the reversed flow itself.
                (0.755, 140, 152, 'adverse pressure\ngradient', +1)):
            a0.text(xc, z_lab, txt, transform=a0.get_yaxis_transform(),
                    ha='center', va='center', fontsize=8.5, color='0.25',
                    linespacing=1.4)
            a0.annotate('', xy=(xc + head * 0.06, z_arr),
                        xytext=(xc - head * 0.06, z_arr),
                        xycoords=a0.get_yaxis_transform(),
                        textcoords=a0.get_yaxis_transform(),
                        arrowprops=dict(arrowstyle='-|>', color='0.25', lw=2.2,
                                        mutation_scale=16))
        # The deep asthenospheric anomaly, quoted over the interval where it
        # has asymptoted and which the reader can see on the axis (the
        # cache's band runs to 220 km; the two agree to 0.02 MPa).
        zlo, zhi = 150.0, 200.0
        band = (zkm >= zlo) & (zkm <= zhi)
        dP_mid = c['p_R'][m].mean(axis=0)[band].mean() / 1e6
        a0.annotate(f'$\\Delta P \\approx {dP_mid:.0f}$ MPa\n'
                    f'({zlo:.0f}–{zhi:.0f} km)',
                    xy=(dP_mid, 182), xytext=(46, 182),
                    ha='center', va='center', fontsize=8.5, color=C_RIDGE,
                    linespacing=1.35,
                    arrowprops=dict(arrowstyle='-', color=C_RIDGE, lw=0.9,
                                    shrinkA=2, shrinkB=2))
        rows.append((k, 'delta_P_150_200km_MPa', f'{dP_mid:.2f}'))

        # ---- right: what they integrate to --------------------------------
        a1 = axes[r, 1]
        cT = np.array([-cumulative(a, z) for a in c['p_T']]) / 1e12
        cR = np.array([cumulative(a, z) for a in c['p_R']]) / 1e12
        cTot = cT + cR
        for arr, col in ((cT, C_TRENCH), (cR, C_RIDGE), (cTot, C_TOTAL)):
            # NO full-run bands here (Dan, 2026-09-30): three overlapping
            # ranges in one panel obscure the curves they belong to. The
            # left column carries the spread; this one carries the means.
            a1.plot(arr[m].mean(axis=0), zkm, '-', color=col,
                    lw=2.2 if col == C_TOTAL else 1.8)
        # WHAT A FIXED z_c COSTS (Dan, 2026-10-01). The box spans the depth
        # range over which z_c might reasonably be fixed, and in force the
        # values the total takes across it: from the LESSER of the two
        # endpoint values to the maximum the curve reaches inside. The
        # maximum is where the integrand changes sign, which is close to but
        # not the same as z_c, so the box is the full spread rather than a
        # difference between two arbitrary depths.
        # ON THE TOTAL ONLY. Drawn on all three curves it triples the ink
        # and invites the reader to compare component sensitivities, which
        # is a different question with a different answer -- over this window
        # the two components are comparably sensitive (they diverge only
        # BELOW it, where the ridge curve turns over). The figure's claim is
        # about the budget as a whole.
        ct = cTot[m].mean(axis=0)
        wz = (zkm >= ZC_RANGE_KM[0]) & (zkm <= ZC_RANGE_KM[1])
        v_lo = min(float(np.interp(ZC_RANGE_KM[0], zkm, ct)),
                   float(np.interp(ZC_RANGE_KM[1], zkm, ct)))
        v_hi = float(ct[wz].max())
        a1.add_patch(plt.Rectangle((v_lo, ZC_RANGE_KM[0]), v_hi - v_lo,
                                   ZC_RANGE_KM[1] - ZC_RANGE_KM[0],
                                   facecolor='0.45', alpha=0.22, lw=0, zorder=0))
        # The capped bar below the box carries the same extent (Dan,
        # 2026-10-01). It is not redundant with the shading: the box says
        # which depths are in play, the bar states the force interval alone
        # and gives it two definite ends to read against the axis.
        zb = ZC_RANGE_KM[1] + 6.0
        a1.plot([v_lo, v_hi], [zb, zb], '-', color=C_TOTAL, lw=1.4)
        for e in (v_lo, v_hi):
            a1.plot([e, e], [zb - 3.0, zb + 3.0], '-', color=C_TOTAL, lw=1.4)
        # The bar's extent as a percentage of the largest value it spans
        # (Dan, 2026-10-01) -- the reader should not have to measure it
        # against the axis. LEVEL WITH THE BAR and set to its left, toward
        # zero: placed below the bar it sat on the thick total curve, which
        # sweeps back across that depth.
        a1.text(v_lo, zb, f'{100 * (v_hi - v_lo) / v_hi:.0f} %  ',
                ha='right', va='center', fontsize=8.5, color=C_TOTAL)
        rows += [(k, f'total_spread_over_zc_{ZC_RANGE_KM[0]:.0f}_'
                     f'{ZC_RANGE_KM[1]:.0f}km_TNm', f'{v_hi - v_lo:.3f}'),
                 (k, 'total_spread_over_zc_range_fraction_of_max',
                  f'{(v_hi - v_lo) / v_hi:.3f}')]
        print(f'   {k}: total over z_c {ZC_RANGE_KM[0]:.0f}-{ZC_RANGE_KM[1]:.0f} km '
              f'spans {v_lo:.2f}-{v_hi:.2f} TN/m '
              f'({v_hi - v_lo:.2f}, {100*(v_hi-v_lo)/v_hi:.0f} % of max)')
        # NO LEGEND in this panel (Dan, 2026-09-30). Every placement that
        # fits three entries lands on a curve: the panel is half-width, all
        # three curves sweep from x = 0 to the right-hand edge, and the only
        # clear corner is too narrow for the labels. It is not needed -- blue
        # and orange are the SAME two columns as the panel beside it, named
        # there in bold in the same colours, which is the link the two
        # figures are meant to have. Only black is new, so only black is
        # labelled, on the curve itself where it is furthest from the others.
        a1.annotate('total', xy=(float(np.interp(30e3, z, cTot[m].mean(axis=0))), 30),
                    xytext=(14, -2), textcoords='offset points',
                    fontsize=8.5, color=C_TOTAL, fontweight='bold',
                    ha='left', va='center')
        # two lines: the full phrase overruns a half-width panel
        a1.set_xlabel('cumulative\n' r'$\Delta\mathrm{GPE}^{*}$ [TN/m]',
                      fontsize=10)

        # shared reference levels on BOTH panels of the row
        zc_km = cpc.ZC_KM
        # ONE reference level (Dan, 2026-09-30): z_c, the fixed nominal
        # integration depth. The per-panel sign-change line is gone -- it sat
        # at a different depth in each panel and read as a measured boundary.
        # Zero line and grid match fig_column_anomalies: its weight, and no
        # grid. The right panel follows the left rather than carrying a grid
        # of its own, which read as an oversight across the row.
        for ax in (a0, a1):
            ax.axhline(cpc.ZC_KM, color='0.35', lw=0.9, ls='--')
            ax.axvline(0, color='0.55', lw=1.3)
        a0.set_ylabel('Depth [km]')
        a0.set_title(f'{k}  (avg {c["t"][m].min():.0f}–{c["t"][m].max():.0f} Myr, '
                     f'n={m.sum()})', fontsize=10)

        at = lambda arr, zz: float(np.interp(zz * 1e3, z, arr[m].mean(axis=0)))
        print(f'{k}: at z_c = {zc_km:.0f} km — trench pull {at(cT, zc_km):+.2f}, '
              f'ridge push {at(cR, zc_km):+.2f}, total {at(cTot, zc_km):+.2f} TN/m; '
              f'at 150 km total {at(cTot, 150):+.2f}')
        for nm, arr in (('trench_pull', cT), ('ridge_push', cR), ('total', cTot)):
            for zz in (zc_km, 100.0, 150.0):
                rows.append((k, f'{nm}_at_{zz:.0f}km_TNm', f'{at(arr, zz):.3f}'))
        # The trench curve is monotonic and flattens, so 'fraction complete
        # by z_c' is well defined for it. The RIDGE curve rises and then
        # turns over as the adverse gradient eats into it, so the same
        # ratio against the 200 km value exceeds 1 and is not a fraction of
        # anything -- it was reported as one until 2026-10-01. The honest
        # quantities for the ridge are its maximum, the depth of that
        # maximum, and how much of it z_c has captured.
        cRm = cR[m].mean(axis=0)
        j = int(np.argmax(cRm))
        rows += [
                 (k, 'trench_pull_fraction_complete_by_zc',
                  f'{at(cT, zc_km) / at(cT, Z_MAX_KM):.3f}'),
                 (k, 'ridge_push_max_TNm', f'{cRm[j]:.3f}'),
                 (k, 'ridge_push_depth_of_max_km', f'{z[j]/1e3:.0f}'),
                 (k, 'ridge_push_at_zc_over_max', f'{at(cR, zc_km)/cRm[j]:.3f}')]

    axes[0, 0].set_ylim(Z_MAX_KM, 0)
    # SYMMETRIC about zero (Dan, 2026-09-30). The trench and ridge lobes sit
    # either side of the zero line and the figure is about their comparison,
    # so an asymmetric axis makes one lobe look larger than it is. Matched in
    # fig_column_anomalies.
    axes[0, 0].set_xlim(-X_LIM_MPA, X_LIM_MPA)
    # legend placement and styling as in fig_column_anomalies: lower left of
    # the STD panel, the corner the terminated trench lithostatic curve
    # clears
    axes[0, 0].legend(fontsize=8, frameon=False, loc='lower left',
                      handlelength=1.8, handletextpad=0.6, borderaxespad=0.6,
                      labelspacing=0.4)
    # Title in the register of fig_column_anomalies: the locked term for the
    # subject, plus what the added column contributes. Bands, the dashed
    # level and the averaging window are caption material, not title material.
    # PARED BACK (Dan, 2026-10-01): the axis label now names the column
    # pair, so the title need not.
    fig.suptitle('Cumulative topographic pressure anomalies', fontsize=10.5, color='0.35')
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_column_anomalies_cumulative.png')
    depth_ticks(fig)   # 20 km depth ticks (FIGURE_STYLE)
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('column_anomalies_cumulative', rows[0], rows[1:],
                      script='fig_column_anomalies_cumulative.py',
                      figure='fig_column_anomalies_cumulative.png',
                      models=('STD', 'WAL'),
                      meta={'midrun_window_Myr': list(cpc.MIDRUN_MYR),
                            'zc_km': cpc.ZC_KM,
                            'sign': 'trench pull = -int p_T dz; ridge push = '
                                    '+int p_R dz; both positive = driving'})
    print('written:', tab)


if __name__ == '__main__':
    main()
