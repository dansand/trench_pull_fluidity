"""fig_column_kinematics — column stress anomalies beside the mean velocity profile.

Writes figures/fig_column_kinematics.png and tables/column_kinematics.csv
from the committed column-profile cache (run
scripts/column_profiles_cache.py first).

Requested by Dan 2026-09-21. Same layout as fig_column_anomalies_
cumulative, one ROW PER MODEL, but the right column carries a kinematic
measurement instead of an integral of the left one.

  left   Delta sigma_zz vs depth relative to the first isostatic column:
         trench (blue) and ridge (orange) in the schematic's domain
         colours, plus the ridge curve with the asthenospheric tilt
         removed (dashed). SAME CONTENT AS fig_column_anomalies.
  right  ONE black line: the horizontally averaged horizontal velocity
         v_x, taken along vertical profiles and averaged over every
         column from the trench to the ridge. Just the mean velocity
         profile of the trailing plate -- no column comparison, no
         domain split, nothing else on the panel.

WHAT GOES INTO THE LINES AND THE BANDS (Dan, 2026-09-21). They are
DIFFERENT populations and the caption must say so:
  solid line   the four snapshots around the reference epoch --
               t = 38, 40, 42, 44 Myr, matched across both models. The
               WAL archive has no 36 Myr snapshot, so a 36-44 window
               would average STD over five and WAL over four; four in
               each keeps the rows comparable.
  shaded band  the range across the ENTIRE model run (37 snapshots,
               8-80 Myr) -- grey on the right panel, per-curve colour on
               the left.

The velocity profile comes from the cache key vx_TR, added 2026-09-21,
in cm/yr with negative = trench-ward in the mirrored analysis frame.

DRAFT CAPTION. Column stress anomalies and the mean velocity profile of
the trailing plate, for STD (top) and WAL (bottom). Solid curves average
the four snapshots around the reference epoch (38-44 Myr); shading spans
the full run (8-80 Myr). Left: the vertical normal stress anomaly of the
trench column (blue) and ridge column (orange) relative to the first
isostatic column, with the ridge column's asthenospheric tilt removed
(dashed). Right: the horizontal velocity averaged over every column
between the trench and the ridge, showing the plate translating
coherently above its decoupling depth and reversing into the return flow
beneath it. The dashed horizontal line on both panels is the first
zero-crossing of the isostatic-domain stress anomaly, so the stress
structure and the velocity structure can be read against a common
depth.
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
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain
C_RIDGE = '#D55E00'         # isostatic / ridge push domain
C_RULE = '#BFC3D1'
SNAP_MYR = (38.0, 40.0, 42.0, 44.0)     # matched across both models
Z_PLOT_KM = 250.0


def main():
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    d = cpc.load()
    sm = lambda a: gaussian_filter1d(a, 2, axis=-1)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.4), sharey=True, sharex='col')
    rows = [('model', 'quantity', 'value')]

    for r, key in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, key)
        z, zkm, t = c['z'], c['z'] / 1e3, c['t']
        # the four snapshots around the reference epoch, matched
        mid = np.zeros_like(t, dtype=bool)
        for tt in SNAP_MYR:
            mid[int(np.argmin(np.abs(t - tt)))] = True
        assert mid.sum() == len(SNAP_MYR), f'{key}: matched snapshots not found'

        pT, pR = sm(c['p_T']) / 1e6, sm(c['p_R']) / 1e6
        pRd = sm(c['p_R_untilted']) / 1e6
        # sign pin against the committed values, before anything is drawn
        tp = -np.trapezoid(c['p_T'][:, c['zc']], z[c['zc']], axis=1)
        i10 = np.argmin(np.abs(t - 20.0))
        assert abs(tp[i10] - COMMITTED_F10_TP[key]) / COMMITTED_F10_TP[key] < 0.03, \
            f'{key}: trench-lobe integral vs committed — sign chain broken'

        # ---- left: the anomalies, as fig_column_anomalies ---------------
        a0 = axes[r, 0]
        for arr, col, lab in ((pT, C_TRENCH, 'trench $-$ first isostatic'),
                              (pR, C_RIDGE, 'ridge $-$ first isostatic')):
            a0.fill_betweenx(zkm, arr.min(axis=0), arr.max(axis=0), color=col,
                             alpha=0.15, lw=0)
            a0.plot(arr[mid].mean(axis=0), zkm, '-', color=col, lw=1.8, label=lab)
        a0.plot(pRd[mid].mean(axis=0), zkm, '--', color=C_RIDGE, lw=1.2,
                alpha=0.45, label='ridge, tilt ($\\Delta P$) removed')
        a0.set_xlabel(r'$\Delta\sigma_{zz}$ [MPa] (pressure-positive)', fontsize=10)

        # ---- right: ONE black line, the mean velocity profile -----------
        a1 = axes[r, 1]
        v = d[f'{key}_vx_TR']
        a1.fill_betweenx(zkm, v.min(axis=0), v.max(axis=0), color='0.55',
                         alpha=0.30, lw=0, label='full run (8–80 Myr)')
        vm = v[mid].mean(axis=0)
        a1.plot(vm, zkm, '-', color='k', lw=2.0,
                label=f'{SNAP_MYR[0]:.0f}–{SNAP_MYR[-1]:.0f} Myr mean')
        a1.set_xlabel(r'$\langle v_x\rangle$ [cm/yr]'
                      '\n(horizontal average, trench to ridge; '
                      'negative = trench-ward)', fontsize=10)

        # The one shared reference level (Dan, 2026-09-21): the first
        # zero-crossing of Delta sigma_zz in the ISOSTATIC domain, i.e. of
        # the ridge - x_I curve. Drawn on BOTH panels so the stress
        # structure and the velocity structure can be read against the
        # same depth. Taken from the same mid-run mean that is plotted.
        prm = sm(c['p_R'][mid].mean(axis=0)) / 1e6
        sgn = np.where((prm[:-1] > 0) & (prm[1:] <= 0) & (zkm[:-1] > 20))[0]
        z_x = zkm[sgn[0]] if len(sgn) else np.nan

        a0.set_ylabel('Depth [km]', fontsize=11)
        a0.set_title(f'{key}', fontsize=11, loc='left')
        for ax in (a0, a1):
            ax.axhline(z_x, color='0.35', lw=0.9, ls='--')
            ax.axvline(0, color='k', lw=1.0)
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        surf = vm[zkm < 20].mean()
        below = np.where(np.abs(vm) < 0.5 * abs(surf))[0]
        z_half = zkm[below[0]] if len(below) else np.nan
        rev = np.where((np.sign(vm[:-1]) != np.sign(vm[1:])) & (zkm[:-1] > 50))[0]
        z_rev = zkm[rev[0]] if len(rev) else np.nan
        vs = v[:, zkm < 20].mean(axis=1)
        print(f'{key}: plate <v_x> {surf:+.2f} cm/yr; halves at {z_half:.0f} km; '
              f'reverses at {z_rev:.0f} km; dSzz sign change {z_x:.0f} km; '
              f'<v_x> there {np.interp(z_x*1e3, z, vm):+.2f} cm/yr '
              f'({100*np.interp(z_x*1e3, z, vm)/surf:.0f} % of surface); '
              f'full-run surface range {vs.min():+.2f}..{vs.max():+.2f}')
        rows += [(key, 'plate_vx_surface_cm_yr', f'{surf:.3f}'),
                 (key, 'depth_vx_half_surface_km', f'{z_half:.0f}'),
                 (key, 'depth_vx_reversal_km', f'{z_rev:.0f}'),
                 (key, 'vx_surface_fullrun_min_cm_yr', f'{vs.min():.3f}'),
                 (key, 'vx_surface_fullrun_max_cm_yr', f'{vs.max():.3f}'),
                 (key, 'dSzz_sign_change_km', f'{z_x:.0f}'),
                 (key, 'vx_at_sign_change_cm_yr', f'{np.interp(z_x*1e3, z, vm):.3f}'),
                 (key, 'vx_at_sign_change_fraction_of_surface',
                  f'{np.interp(z_x*1e3, z, vm)/surf:.3f}'),
                 (key, 'n_snapshots_line', str(len(SNAP_MYR))),
                 (key, 'n_snapshots_band', str(len(t)))]

    axes[0, 0].set_ylim(Z_PLOT_KM, 0)
    axes[0, 0].legend(frameon=False, fontsize=9, loc='lower left')
    axes[0, 1].legend(frameon=False, fontsize=9, loc='lower left')
    fig.suptitle('Column stress anomalies and the mean velocity profile of the '
                 'trailing plate\n'
                 f'(lines: {SNAP_MYR[0]:.0f}–{SNAP_MYR[-1]:.0f} Myr;  '
                 'shading: full run, 8–80 Myr)', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_column_kinematics.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('column_kinematics', rows[0], rows[1:],
                      script='fig_column_kinematics.py',
                      figure='fig_column_kinematics.png', models=('STD', 'WAL'),
                      meta={'line_snapshots_Myr': list(SNAP_MYR),
                            'band': 'full run, all cached snapshots',
                            'velocity': 'v_x averaged over all columns x_T..x_R, '
                                        'cm/yr, negative = trench-ward'})
    print('written:', tab)


if __name__ == '__main__':
    main()
