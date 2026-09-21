"""fig_column_kinematics — column stress anomalies beside the column VELOCITIES.

Writes figures/fig_column_kinematics.png and tables/column_kinematics.csv.

Requested by Dan 2026-09-21 as the kinematic counterpart of
fig_column_anomalies_cumulative: same layout, one ROW PER MODEL, but the
right column carries a measurement rather than an integral of the left
one.

  left   Delta sigma_zz vs depth relative to the first isostatic column:
         trench (blue) and ridge (orange), in the schematic's domain
         colours (FIGURE_STYLE.md).
  right  the horizontal velocity v_x, HORIZONTALLY AVERAGED over each
         domain (Dan, 2026-09-21): x_T -> x_I in blue and x_I -> x_R in
         orange. Column samples were tried first and rejected -- the
         depth structure of the flow beneath the plate is a regional
         property, and single columns carry local detail that has nothing
         to do with it. This is an INDEPENDENT measurement: the stress
         panels say what force the columns carry, the velocity panels say
         how the material is actually moving, and nothing in one is
         derived from the other.

         The whole-plate average x_T -> x_R is computed and tabulated but
         NOT drawn: the isostatic domain is 2878 of 2972 km (STD), so its
         average and the whole-plate average agree to 0.01 cm/yr and the
         two curves sit on top of each other. Plotting both would imply a
         distinction the data does not contain.

MATCHED SNAPSHOTS (Dan, 2026-09-21). Both columns, both the lines and the
bands, use the SAME FOUR snapshots per model: t = 38, 40, 42, 44 Myr.
fig_column_anomalies and its cumulative variant average STD over five
(36-44) and WAL over four (38-44), because the WAL archive has no 36 Myr
snapshot -- so the two models were being averaged over different epochs
and their bands were full-run envelopes over 37 snapshots, a different
population again. Here line and band describe the same four epochs in
both models, so every difference between the rows is a difference
between the MODELS.

  line = mean of the four;  band = their min-to-max spread.

The band therefore shows epoch-to-epoch scatter within the mid-run
window, NOT the full-run range. For the full-run envelope see
fig_column_anomalies.

⚠ This script reads the archive directly and uses the CORRECTED surface
reading (fluidity_helpers.surface_fs, top-boundary nodes), so its x_I is
the corrected one. The column cache still holds the old reading, so x_I
here differs by a few km from any cache-derived figure until that is
rebuilt.

DRAFT CAPTION. Column stress anomalies and column velocities for STD
(top) and WAL (bottom), averaged over four mid-run snapshots
(38-44 Myr), with the shaded band spanning those four. Left: the
vertical normal stress anomaly of the trench column (blue) and ridge
column (orange) relative to the first isostatic column. Right: the
horizontal velocity averaged over each domain: the non-isostatic domain
x_T to x_I (blue) and the isostatic domain x_I to x_R (orange). The
near-trench domain moves slightly faster than the plate as a whole
(-1.33 against -1.18 cm/yr in STD, -2.07 against -1.97 in WAL), and both
decouple downward over the same depth range, falling to half the surface
rate near 140 km and reversing into the return flow at about 200 km in
both models.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fluidity_helpers import (make_field_extractor, mirror_fields_in_x,
                              pick_trench_3step, find_first_isostatic_column,
                              find_ridge_x_flow, surface_fs)
from tables_io import write_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser('~/DATA/numerical_models/OUTPUTS/')
DX, Z_MAX, Y, W = 1000.0, 250e3, 2_900_000.0, 5
SNAP_MYR = (38.0, 40.0, 42.0, 44.0)      # matched across both models
DT_MYR = 2.0                             # archive cadence (index = t / DT)
C_TRENCH = '#0072B2'        # non-isostatic / trench pull domain
C_RIDGE = '#D55E00'         # isostatic / ridge push domain
C_ISO = '0.35'              # the first isostatic column itself
C_RULE = '#BFC3D1'
Z_PLOT_KM = 250.0


def one_snapshot(key, t_target):
    """Column profiles at x_T, x_I, x_R for one snapshot."""
    files = natsort.natsorted(glob.glob(DATA + f'{key}_RefModel/subduction_with_LM*.pvtu'))
    v = pv.read(files[int(round(t_target / DT_MYR))])
    t = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
    assert abs(t - t_target) < 0.5, f'{key}: expected t={t_target}, got {t:.2f}'
    v.point_data['p'] = v['NormalSP::Pressure']
    v.point_data['tzz'] = v['NormalSP::Stress'][:, 4]
    v.point_data['txx'] = v['NormalSP::Stress'][:, 0]
    v.point_data['txz'] = -v['NormalSP::Stress'][:, 1]
    v.point_data['T'] = v['NormalSP::Temperature']
    v.point_data['fs'] = v['NormalSP::FreeSurface']
    vel = v['NormalSP::Velocity'] / 3.17098e-10          # m/s -> cm/yr
    v.point_data['vx'] = vel[:, 0]; v.point_data['vy'] = vel[:, 1]
    x0, x1, _, _, _, _ = v.bounds
    nx, nz = int((x1 - x0) / DX), int(Z_MAX / DX)
    x = x0 + (np.arange(nx) + 0.5) * DX
    z = (np.arange(nz) + 0.5) * DX
    X, Z = np.meshgrid(x, z, indexing='xy')
    g = pv.StructuredGrid(X.T, (Y - Z).T, np.zeros_like(X.T)).sample(v)
    nxp, nzp, _ = g.dimensions
    val = g['vtkValidPointMask'].reshape((nxp, nzp), order='F').astype(bool)
    gf = make_field_extractor(g, nxp, nzp, val)
    p, txx, tzz, txz = gf('p'), gf('txx'), gf('tzz'), gf('txz')
    T, fs, vx, vz = gf('T'), gf('fs'), gf('vx'), -gf('vy')
    p, txx, tzz, txz, T, fs, vx, vz = mirror_fields_in_x(p, txx, tzz, txz, T,
                                                         fs, vx, vz)
    szz = np.nan_to_num(tzz - p)
    vx = np.nan_to_num(vx)
    fs_top = surface_fs(v, x, mirror_x=True)             # CORRECTED reading
    xT, _ = pick_trench_3step(x, z, p, vx, subducting_side='right')
    ti = int(np.argmin(np.abs(x - xT)))
    iI, _ = find_first_isostatic_column(x, fs_top, xT, ti, DX, seaward_sign=+1)
    iz10 = int(np.argmin(np.abs(z - 10e3)))
    _, iR = find_ridge_x_flow(x, vx[iz10], xT, seaward_sign=+1)
    ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)   # +/-5 km
    out = dict(z=z, t=t, xT=xT, xI=x[iI], xR=x[iR])
    for nm, j in (('T', ti), ('I', iI), ('R', iR)):
        out[f'szz_{nm}'] = ca(szz, j)
    # velocity is taken as a HORIZONTAL AVERAGE over each domain, not
    # column-sampled (Dan, 2026-09-21): the depth structure of the flow
    # beneath the plate is a regional property, and single columns sample
    # local detail that has nothing to do with it.
    span = lambda f, j0, j1: f[:, min(j0, j1):max(j0, j1) + 1].mean(axis=1)
    out['vx_TI'] = span(vx, ti, iI)        # non-isostatic / trench pull domain
    out['vx_IR'] = span(vx, iI, iR)        # isostatic / ridge push domain
    out['vx_TR'] = span(vx, ti, iR)        # whole trailing plate
    out['len_TI_km'] = abs(x[iI] - xT) / 1e3
    out['len_IR_km'] = abs(x[iR] - x[iI]) / 1e3
    return out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.4), sharey=True, sharex='col')
    rows = [('model', 'quantity', 'value')]
    sm = lambda a: gaussian_filter1d(a, 2, axis=-1)

    for r, key in enumerate(('STD', 'WAL')):
        snaps = [one_snapshot(key, tt) for tt in SNAP_MYR]
        z = snaps[0]['z']; zkm = z / 1e3
        stack = lambda f: np.array([f(s) for s in snaps])
        # pressure-register anomalies relative to x_I, as in the cache
        pT = sm(stack(lambda s: -(s['szz_T'] - s['szz_I']))) / 1e6
        pR = sm(stack(lambda s: -(s['szz_R'] - s['szz_I']))) / 1e6
        vTI = stack(lambda s: s['vx_TI'])
        vIR = stack(lambda s: s['vx_IR'])
        vTR = stack(lambda s: s['vx_TR'])

        a0, a1 = axes[r, 0], axes[r, 1]
        for arr, col, lab in ((pT, C_TRENCH, 'trench $-$ first isostatic'),
                              (pR, C_RIDGE, 'ridge $-$ first isostatic')):
            a0.fill_betweenx(zkm, arr.min(axis=0), arr.max(axis=0), color=col,
                             alpha=0.15, lw=0)
            a0.plot(arr.mean(axis=0), zkm, '-', color=col, lw=1.8, label=lab)
        L_TI = np.mean([s['len_TI_km'] for s in snaps])
        L_IR = np.mean([s['len_IR_km'] for s in snaps])
        for arr, col, lw, lab in (
                (vTI, C_TRENCH, 1.8, f'$x_T\\to x_I$  ({L_TI:.0f} km)'),
                (vIR, C_RIDGE, 1.8, f'$x_I\\to x_R$  ({L_IR:.0f} km)')):
            a1.fill_betweenx(zkm, arr.min(axis=0), arr.max(axis=0), color=col,
                             alpha=0.15, lw=0)
            a1.plot(arr.mean(axis=0), zkm, '-', color=col, lw=lw, label=lab)

        a0.set_xlabel(r'$\Delta\sigma_{zz}$ [MPa] (pressure-positive)', fontsize=10)
        a1.set_xlabel(r'$\langle v_x\rangle$ [cm/yr], horizontal average'
                      '\n(negative = trench-ward)', fontsize=10)
        a0.set_ylabel('Depth [km]', fontsize=11)
        a0.set_title(f'{key}   (mean of {len(SNAP_MYR)} snapshots, '
                     f'{SNAP_MYR[0]:.0f}–{SNAP_MYR[-1]:.0f} Myr)',
                     fontsize=10.5, loc='left')
        for ax in (a0, a1):
            ax.axvline(0, color='k', lw=1.0)
            ax.grid(alpha=0.25, color=C_RULE, lw=0.6)

        vm = vTR.mean(axis=0)
        surf = vm[zkm < 20].mean()
        # base of coherent translation: where the horizontally averaged
        # |v_x| has fallen to half the plate value -- a MEASURED level,
        # not an assumed thickness
        below = np.where(np.abs(vm) < 0.5 * abs(surf))[0]
        z_half = zkm[below[0]] if len(below) else np.nan
        rev = np.where((np.sign(vm[:-1]) != np.sign(vm[1:])) & (zkm[:-1] > 50))[0]
        z_rev = zkm[rev[0]] if len(rev) else np.nan
        sfc = lambda a: a.mean(axis=0)[zkm < 20].mean()
        print(f'{key}: plate <v_x> {surf:+.2f} cm/yr; |v| halves at {z_half:.0f} km; '
              f'flow reverses at {z_rev:.0f} km; domain means at the surface '
              f'x_T->x_I {sfc(vTI):+.2f} ({L_TI:.0f} km), '
              f'x_I->x_R {sfc(vIR):+.2f} ({L_IR:.0f} km)')
        rows += [(key, 'plate_vx_surface_cm_yr', f'{surf:.3f}'),
                 (key, 'depth_vx_half_plate_km', f'{z_half:.0f}'),
                 (key, 'depth_vx_reversal_km', f'{z_rev:.0f}'),
                 (key, 'vx_trench_to_isostatic_surface_cm_yr', f'{sfc(vTI):.3f}'),
                 (key, 'vx_isostatic_to_ridge_surface_cm_yr', f'{sfc(vIR):.3f}'),
                 (key, 'domain_length_trench_to_isostatic_km', f'{L_TI:.0f}'),
                 (key, 'domain_length_isostatic_to_ridge_km', f'{L_IR:.0f}'),
                 (key, 'n_snapshots', str(len(SNAP_MYR)))]

    axes[0, 0].set_ylim(Z_PLOT_KM, 0)
    axes[0, 0].legend(frameon=False, fontsize=9, loc='lower left')
    axes[0, 1].legend(frameon=False, fontsize=9, loc='lower left',
                      title='horizontal average over')
    fig.suptitle('Column stress anomalies and the velocity structure beneath the plate '
                 '— the same four '
                 f'snapshots ({SNAP_MYR[0]:.0f}–{SNAP_MYR[-1]:.0f} Myr) in both models\n'
                 '(band: spread across those four)', fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, 'figures', 'fig_column_kinematics.png')
    fig.savefig(out, bbox_inches='tight', dpi=220)
    print('written:', out)
    tab = write_table('column_kinematics', rows[0], rows[1:],
                      script='fig_column_kinematics.py',
                      figure='fig_column_kinematics.png', models=('STD', 'WAL'),
                      meta={'snapshots_Myr': list(SNAP_MYR),
                            'window_km': 5,
                            'surface': 'top-boundary nodes (corrected reading)',
                            'band': 'min-to-max across the four snapshots'})
    print('written:', tab)


if __name__ == '__main__':
    main()
