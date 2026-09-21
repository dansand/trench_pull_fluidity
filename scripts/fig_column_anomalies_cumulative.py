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
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED_F10_TP = {'STD': 1.71e12, 'WAL': 1.74e12}   # N/m, +/-5 km, f10
# domain colours — the schematic's cboxA / cboxB, see FIGURE_STYLE.md
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain
C_RIDGE = '#D55E00'         # isostatic / ridge push domain
C_TOTAL = 'k'
C_RULE = '#BFC3D1'
Z_MAX_KM = 200.0


def cumulative(a, z):
    """Running integral of a(z) from the surface down, same length as z."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(z))])


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    sm = lambda a: gaussian_filter1d(a, 2, axis=-1)
    # sharex='col' so the two models sit on identical scales within each
    # column — the rows are the comparison, and unequal axes would hide it
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.4), sharey=True, sharex='col')
    rows = [('model', 'quantity', 'value')]

    for r, k in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, k)
        z, zkm = c['z'], c['z'] / 1e3
        m = c['mid']
        pT, pR = sm(c['p_T']) / 1e6, sm(c['p_R']) / 1e6
        pRd = sm(c['p_R_untilted']) / 1e6

        # sign pin against the committed values, before anything is drawn
        tp = -np.trapezoid(c['p_T'][:, c['zc']], z[c['zc']], axis=1)
        i10 = np.argmin(np.abs(c['t'] - 20.0))
        rel = abs(tp[i10] - COMMITTED_F10_TP[k]) / COMMITTED_F10_TP[k]
        assert rel < 0.03, (f'{k}: trench-lobe integral {tp[i10]/1e12:.2f} TN/m '
                            'vs committed — sign chain broken')

        # ---- left: the anomalies themselves -------------------------------
        a0 = axes[r, 0]
        a0.fill_betweenx(zkm, pT.min(axis=0), pT.max(axis=0), color=C_TRENCH,
                         alpha=0.12, lw=0)
        a0.fill_betweenx(zkm, pR.min(axis=0), pR.max(axis=0), color=C_RIDGE,
                         alpha=0.12, lw=0)
        a0.plot(pT[m].mean(axis=0), zkm, '-', color=C_TRENCH, lw=1.8,
                label='trench $-$ first isostatic')
        a0.plot(pR[m].mean(axis=0), zkm, '-', color=C_RIDGE, lw=1.6,
                label='ridge $-$ first isostatic')
        a0.plot(pRd[m].mean(axis=0), zkm, '--', color=C_RIDGE, lw=1.2,
                alpha=0.45, label='ridge, tilt ($\\Delta P$) removed')
        a0.set_xlabel(r'$\Delta\sigma_{zz}$ [MPa] (pressure-positive)',
                      fontsize=10)

        # ---- right: what they integrate to --------------------------------
        a1 = axes[r, 1]
        cT = np.array([-cumulative(a, z) for a in c['p_T']]) / 1e12
        cR = np.array([cumulative(a, z) for a in c['p_R']]) / 1e12
        cTot = cT + cR
        for arr, col, lab in ((cT, C_TRENCH, 'trench pull (non-isostatic)'),
                              (cR, C_RIDGE, 'ridge push (isostatic)'),
                              (cTot, C_TOTAL, 'total, trench $\\to$ ridge')):
            a1.fill_betweenx(zkm, arr.min(axis=0), arr.max(axis=0), color=col,
                             alpha=0.10, lw=0)
            a1.plot(arr[m].mean(axis=0), zkm, '-', color=col,
                    lw=2.2 if col == C_TOTAL else 1.8, label=lab)
        a1.set_xlabel(r'cumulative $\Delta\mathrm{GPE}^{*}$ from the surface'
                      ' [TN/m]', fontsize=10)

        # shared reference levels on BOTH panels of the row
        zc_km = cpc.ZC_KM
        prm = sm(c['p_R'][m].mean(axis=0)) / 1e6
        sgn = np.where((prm[:-1] > 0) & (prm[1:] <= 0) & (zkm[:-1] > 20))[0]
        z_x = zkm[sgn[0]] if len(sgn) else np.nan
        for ax in (a0, a1):
            ax.axhline(zc_km, color='0.35', lw=0.9, ls=':')
            ax.axhline(z_x, color='0.55', lw=0.9, ls='--')
            ax.axvline(0, color='k', lw=1.0)
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)
        a0.set_ylabel('Depth [km]', fontsize=11)
        a0.set_title(f'{k}   (avg {c["t"][m].min():.0f}–{c["t"][m].max():.0f} Myr, '
                     f'n={m.sum()})', fontsize=10.5, loc='left')

        at = lambda arr, zz: float(np.interp(zz * 1e3, z, arr[m].mean(axis=0)))
        print(f'{k}: at z_c = {zc_km:.0f} km — trench pull {at(cT, zc_km):+.2f}, '
              f'ridge push {at(cR, zc_km):+.2f}, total {at(cTot, zc_km):+.2f} TN/m; '
              f'sign change {z_x:.0f} km; at 150 km total {at(cTot, 150):+.2f}')
        for nm, arr in (('trench_pull', cT), ('ridge_push', cR), ('total', cTot)):
            for zz in (zc_km, 100.0, 150.0):
                rows.append((k, f'{nm}_at_{zz:.0f}km_TNm', f'{at(arr, zz):.3f}'))
        rows += [(k, 'dSzz_sign_change_km', f'{z_x:.0f}'),
                 (k, 'trench_pull_fraction_complete_by_zc',
                  f'{at(cT, zc_km) / at(cT, Z_MAX_KM):.3f}'),
                 (k, 'ridge_push_fraction_complete_by_zc',
                  f'{at(cR, zc_km) / at(cR, Z_MAX_KM):.3f}')]

    axes[0, 0].set_ylim(Z_MAX_KM, 0)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc='lower left')
    axes[0, 1].legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.suptitle('Column anomalies relative to the first isostatic column, and the '
                 'forces they integrate to\n'
                 '(dotted: $z_c$;  dashed: the $\\Delta\\sigma_{zz}$ sign change)',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_column_anomalies_cumulative.png')
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
