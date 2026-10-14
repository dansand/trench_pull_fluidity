"""fig_ridge_density_check — the ridge-column anomaly against the pure
density structure.

Writes figures/fig_ridge_density_check.png from the column_profiles_cache
(run scripts/column_profiles_cache.py first).

Rows STD / WAL; columns: temperature difference, density difference, and
the stress anomaly test — the ridge anomaly shifted by |ΔP| (the untilted
column) against the density-route prediction g∫(ρ_R−ρ_I)dz with its deep
constant removed. Because the fluidity mesh is undeformed (topography is a
surface-stress field), the density field carries no elevation mass and
only the deep-corrected SHAPE of the lithostatic difference is comparable.

What it establishes (2026-09-15 session): the untilted ridge anomaly IS
the density structure (rms 2–3 MPa on a ~45 MPa signal); the thermal and
density contrasts persist to ~110–135 km, so ridge-side isostasy completes
only at the thermal thickness — the measured curve's shallower apparent
equilibration is set by where the isostatic tail equals |ΔP|, an amplitude
coincidence, not a material thickness.

DRAFT CAPTION. Ridge column relative to the first isostatic column:
temperature difference (left), density difference (centre), and the
vertical normal stress anomaly test (right) for STD (top) and WAL
(bottom), averaged over the mid-run window (36–44 Myr). In the right
panels the measured anomaly shifted by the deep pressure anomaly |ΔP|
(dashed) tracks the pure density prediction (dash-dotted) to within a few
MPa: with the tilt removed, the ridge column is isostatic cooling
structure, equilibrating only at the base of the thermal contrast.
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

def main():
    d = cpc.load()
    rows = [('model', 'quantity', 'value')]
    sm = lambda a: gaussian_filter1d(a, 2)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6), sharey=True)
    for row, k in enumerate(('STD', 'WAL')):
        c = cpc.derive(d, k)
        zkm, m = c['z'] / 1e3, c['mid']
        Td = sm((d[f'{k}_temp_R'] - d[f'{k}_temp_I'])[m].mean(axis=0))
        rd = sm((d[f'{k}_rho_R'] - d[f'{k}_rho_I'])[m].mean(axis=0))
        ps = sm(c['p_R'][m].mean(axis=0)) / 1e6
        dP = c['dP'][m].mean() / 1e6
        lith_c = sm(c['lith_R'][m].mean(axis=0)) / 1e6
        ax = axes[row, 0]
        ax.plot(Td, zkm, 'k-', lw=1.5)
        ax.set_title(f'({"ad"[row]})  {k}', fontsize=10.5, loc='left')
        ax = axes[row, 1]
        ax.plot(rd, zkm, 'k-', lw=1.5)
        ax.set_title(f'({"be"[row]})  {k}', fontsize=10.5, loc='left')
        ax = axes[row, 2]
        ax.plot(ps, zkm, 'k-', lw=1.6, label='measured anomaly')
        ax.plot(ps - dP, zkm, 'k--', lw=1.3, label='tilting contribution removed')
        ax.plot(lith_c, zkm, color='0.45', lw=1.6, ls='-.',
                label='density prediction\n$g\\int(\\rho_R-\\rho_I)dz$ $-$ deep const.')
        ax.set_title(f'({"cf"[row]})  {k}', fontsize=10.5, loc='left')
        if row == 0:
            ax.legend(fontsize=7.5, frameon=False, loc='lower right')
        sel = (zkm > 5) & (zkm < 220)
        rms = np.sqrt(np.mean((ps[sel] - dP - lith_c[sel]) ** 2))
        sig = float(np.abs(ps[sel] - dP).max())
        print(f'{k}: untilted vs density prediction rms {rms:.2f} MPa '
              f'(signal ~{sig:.0f} MPa, {100*rms/sig:.0f} %); dP = {dP:+.2f} MPa')
        # TABULATED 2026-10-09. This was the only paper figure with no
        # table, and both its SI text and its caption asserted that the
        # untilted column matches the density structure without saying
        # how well -- in a section headed 'sensitivity tests and
        # methodological validation'. The misfit IS the result.
        rows += [(k, 'untilted_vs_density_rms_MPa', f'{rms:.2f}'),
                 (k, 'signal_amplitude_MPa', f'{sig:.0f}'),
                 (k, 'misfit_pct_of_signal', f'{100*rms/sig:.0f}'),
                 (k, 'deep_pressure_anomaly_MPa', f'{dP:.2f}')]
        for a in axes[row]:
            a.axvline(0, color='0.7', lw=0.7)
            a.axhline(cpc.ZC_KM, color='0.5', lw=0.6, ls=':')
    # THE QUANTITY AND ITS UNITS BELONG ON THE AXIS, not in the title. The
    # titles read '{model}: $T_R - T_I$ [K]' and so on -- a formula and a
    # unit bracket doing the x-label's job, in the one figure of the set
    # that also had no panel letters.
    for a, lab in zip(axes[1], (r'$T_R - T_I$  [K]',
                                r'$\rho_R - \rho_I$  [kg m$^{-3}$]',
                                r'Column anomaly  [MPa]')):
        a.set_xlabel(lab, fontsize=10.5)
    axes[0, 0].set_ylim(250, 0)
    axes[0, 0].set_ylabel('Depth [km]')
    axes[1, 0].set_ylabel('Depth [km]')
    # NOT "untilted" (Dan, 2026-10-14). At a glance it reads as "untitled",
    # and the companion figure fig_ridge_push_anatomy already says "tilt
    # removed" for the same operation -- one phrase, both figures.
    fig.suptitle('Ridge anomaly with the tilting contribution removed, '
                 'against density structure', fontsize=11)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures', 'fig_ridge_density_check.png')
    depth_ticks(fig)   # 20 km depth ticks (FIGURE_STYLE)
    fig.savefig(out, dpi=200)
    tab = write_table('ridge_density_check', rows[0], rows[1:],
                      script='fig_ridge_density_check.py',
                      figure='fig_ridge_density_check.png',
                      models=('STD', 'WAL'),
                      meta={'test': 'ridge anomaly shifted by |dP| against '
                                    'g*int(rho_R - rho_I)dz with its deep '
                                    'constant removed',
                            'zc_km': cpc.ZC_KM,
                            'window': 'mid-run average'})
    print('written:', tab)
    print('written:', out)

if __name__ == '__main__':
    main()
