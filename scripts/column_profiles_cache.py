"""Deep column-profile cache for the sigma_zz anomaly figure family.

Per snapshot (t >= T_MIN_MYR), STD and WAL: vertical profiles to 250 km of
the full vertical normal stress sigma_zz, density and temperature at the
trench, first-isostatic and ridge columns (+-5 km window means,
conventions §4.1), plus the picks. Raw per-column profiles are stored so
downstream scripts form their own differences.

Feeds (one script -> one figure, the fig_ convention):
  fig_column_anomalies.py            -> figures/fig_column_anomalies.png
  fig_column_anomalies_normalised.py -> figures/fig_column_anomalies_normalised.png
  fig_ridge_density_check.py         -> figures/fig_ridge_density_check.png

Cache: notebooks/outputs/column_profiles_deep.npz. Run this script directly
to (re)build (~20 min; reads every archive snapshot of both runs).

Method notes (2026-09-15 session):
- Analysis frame per the repo conventions: x-mirrored so the subducting
  plate is on the right; sigma_zz = tzz - p (tension-positive); scalars
  flip only under the mirror (cerpa_helpers.mirror_fields_in_x register).
- The fluidity mesh is UNDEFORMED (no true free surface): topography lives
  in the surface-stress field, so column density integrals carry no
  elevation mass -- the topographic counterweight enters sigma_zz as a
  surface load. Density-route comparisons must remove the deep constant
  (see fig_ridge_density_check.py).
- Deep pressure-anomaly band for Delta P: 150-220 km, deliberately deeper
  than the 120-190 km back-tilt band (Dan, 2026-09-15). The ridge-column
  thermal contrast persists to ~110-135 km, so a shallower band mixes
  isostatic tail into Delta P and UNDERESTIMATES it. Verified on the data:
  median dP converges by 150 km (STD -9.05/-9.46/-9.47, WAL
  -4.63/-4.85/-4.80 MPa over 120-190/150-220/200-250 km) -- the chosen
  band sits on the plateau.
- Sign pin: the trench-lobe integral over 0..z_c must reproduce the
  committed trench pulls (1.71 STD / 1.74 WAL TN/m at f10) -- asserted by
  fig_column_anomalies.py before rendering.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cerpa_helpers import (make_field_extractor, pick_trench_3step,
                           find_first_isostatic_column, find_ridge_x)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser('~/DATA/numerical_models/OUTPUTS/')
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'column_profiles_deep.npz')

DX, Z_MAX, Y, W = 1000.0, 250e3, 2_900_000.0, 5
T_MIN_MYR = 8.0
DP_BAND_KM = (150.0, 220.0)        # deep Delta P band (see module docstring)

def build():
    flip = lambda a: np.flip(a, axis=1)
    out = {}
    for KEY in ('STD', 'WAL'):
        rows = []
        for ff in natsort.natsorted(glob.glob(DATA + f'{KEY}_RefModel/subduction_with_LM*.pvtu'))[1:]:
            v = pv.read(ff)
            t_myr = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
            if t_myr < T_MIN_MYR:
                del v
                continue
            v.point_data['p'] = v['NormalSP::Pressure']
            v.point_data['tzz'] = v['NormalSP::Stress'][:, 4]
            v.point_data['rho'] = v['NormalSP::Density']
            v.point_data['T'] = v['NormalSP::Temperature']
            v.point_data['fs'] = v['NormalSP::FreeSurface']
            v.point_data['vx'] = (v['NormalSP::Velocity'] / 3.17098e-10)[:, 0]
            x0, x1, _, _, _, _ = v.bounds
            nx, nz = int((x1 - x0) / DX), int(Z_MAX / DX)
            x = x0 + (np.arange(nx) + 0.5) * DX
            z = (np.arange(nz) + 0.5) * DX
            X, Z = np.meshgrid(x, z, indexing='xy')
            g = pv.StructuredGrid(X.T, (Y - Z).T, np.zeros_like(X.T)).sample(v)
            nxp, nzp, _ = g.dimensions
            val = g['vtkValidPointMask'].reshape((nxp, nzp), order='F').astype(bool)
            gf = make_field_extractor(g, nxp, nzp, val)
            # scalars: mirror = flip only; vx needs the sign flip
            p, tzz = flip(gf('p')), flip(gf('tzz'))
            rho, T = flip(gf('rho')), flip(gf('T'))
            fs, vx = flip(gf('fs')), -flip(gf('vx'))
            szz = np.nan_to_num(tzz - p)
            rho = np.nan_to_num(rho)           # void rows above the surface -> 0
            T = np.nan_to_num(T)
            fs_top = np.nan_to_num(fs[0, :])
            xT, _ = pick_trench_3step(x, z, p, vx, subducting_side='right')
            ti = int(np.argmin(np.abs(x - xT)))
            iI, _ = find_first_isostatic_column(x, fs_top, xT, ti, DX, seaward_sign=+1)
            xR, iR = find_ridge_x(x, fs_top, xT, seaward_sign=+1)
            ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)
            rows.append(dict(t=t_myr,
                             szz_T=ca(szz, ti), szz_I=ca(szz, iI), szz_R=ca(szz, iR),
                             rho_T=ca(rho, ti), rho_I=ca(rho, iI), rho_R=ca(rho, iR),
                             temp_T=ca(T, ti), temp_I=ca(T, iI), temp_R=ca(T, iR),
                             xT=xT, xI=x[iI], xR=xR))
            print(f'{KEY} t={t_myr:6.2f} Myr  xT={xT/1e3:7.1f} xI={x[iI]/1e3:7.1f} '
                  f'xR={xR/1e3:7.1f}', flush=True)
            del v, g
        out[KEY] = rows
    keys = ('t', 'szz_T', 'szz_I', 'szz_R', 'rho_T', 'rho_I', 'rho_R',
            'temp_T', 'temp_I', 'temp_R', 'xT', 'xI', 'xR')
    np.savez(OUT,
             **{f'{k}_{q}': np.array([r[q] for r in rows]) for k, rows in out.items()
                for q in keys},
             z=(np.arange(int(Z_MAX / DX)) + 0.5) * DX,
             dp_band_km=np.array(DP_BAND_KM))
    print('cache written:', OUT)

GRAV = 9.8
ZC_KM = 75.0                       # column-integral depth (conventions §5.1)
MIDRUN_MYR = (36.0, 44.0)          # candidate mid-run reference window (W12 pending)

def load():
    if not os.path.exists(OUT):
        raise SystemExit(f'cache missing: {OUT} — run column_profiles_cache.py first')
    return np.load(OUT)

def derive(d, key):
    """Pressure-register anomaly curves and scalars for one model.

    Everything is relative to the first isostatic column, pressure-positive
    (a positive value = higher vertical normal pressure than under x_I).
    Returns dict with per-snapshot arrays:
      p_T, p_R      trench / ridge stress anomaly profiles [Pa]
      lith_R        density-route ridge anomaly, deep constant removed [Pa]
                    (the fluidity mesh is undeformed: no elevation mass in
                    the density field, so only the deep-corrected SHAPE is
                    comparable — see module docstring)
      p_R_untilted  ridge anomaly with the asthenospheric part removed
                    (shifted by |Delta P| so the deep asymptote is zero):
                    the ridge column in the absence of the tilt [Pa]
      dP            deep pressure anomaly, band mean [Pa] (negative=deficit)
      dP_T          trench pressure deficit (peak shallow anomaly) [Pa]
      t, z, mid (mid-run snapshot mask), zc (z <= z_c mask), band mask
    """
    z = d['z']
    t = d[f'{key}_t']
    band = (z >= DP_BAND_KM[0] * 1e3) & (z <= DP_BAND_KM[1] * 1e3)
    zc = z <= ZC_KM * 1e3
    p_T = -(d[f'{key}_szz_T'] - d[f'{key}_szz_I'])
    p_R = -(d[f'{key}_szz_R'] - d[f'{key}_szz_I'])
    rho_d = d[f'{key}_rho_R'] - d[f'{key}_rho_I']
    lith = np.cumsum(np.concatenate(
        [np.zeros((len(t), 1)), 0.5 * (rho_d[:, 1:] + rho_d[:, :-1]) * np.diff(z)], axis=1),
        axis=1) * GRAV
    lith_R = lith - lith[:, band].mean(axis=1, keepdims=True)
    dP = p_R[:, band].mean(axis=1)
    dP_T = np.abs(p_T[:, zc]).max(axis=1)
    return dict(t=t, z=z, band=band, zc=zc,
                p_T=p_T, p_R=p_R, lith_R=lith_R,
                p_R_untilted=p_R - dP[:, None],
                dP=dP, dP_T=dP_T,
                mid=(t >= MIDRUN_MYR[0]) & (t <= MIDRUN_MYR[1]))

PLATEAU_KM = (150.0, 240.0)    # where cum(D) has flattened (spread < 0.3 %)

def partition(d, key):
    """Per-snapshot decomposition of the x_I-to-ridge sigma_zz term.

    Single implementation, used by every figure and table that quotes the
    partition (Dan's rule, 2026-09-16). One depth throughout: z*, the sign
    change of the measured anomaly — equivalently the maximum of its
    cumulative integral. Returns, per snapshot (N/m unless noted):

      z_star    integration depth [m]: argmax of the cumulative integral
      measured  the term as it enters the force balance, integral to z*
      tilt      |Delta P| * z*  -- the back-tilt contribution contained in
                `measured`, evaluated at the SAME depth (using any other
                thickness would redefine the plate midway through the
                derivation: a different, static model)
      dynamic_isostatic  measured + tilt: the isostatic part over the same
                depth. Closes exactly by construction.
      static    the STATIC RIDGE PUSH (Dan's term): the asymptote of the
                cumulative integral of the shifted profile D = anomaly -
                Delta P, i.e. what the density structure alone would supply
                with no transition from lithospheric boundary layer to
                asthenospheric counterflow. Read off the plateau, NOT by
                integrating to a zero crossing -- D is flat there, so its
                crossing is ill-conditioned while the integral is bounded
                (plateau spread 0.1-0.3 %).
      tail      static - dynamic_isostatic: the density-structure tail
                below z*, excluded from the balance (6-10 % of static).
    """
    c = derive(d, key)
    z = c['z']
    plateau = (z >= PLATEAU_KM[0] * 1e3) & (z <= PLATEAU_KM[1] * 1e3)
    sel = (z > 30e3) & (z < 200e3)
    cum = lambda a: np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(z))])
    out = {k: [] for k in ('z_star', 'measured', 'tilt', 'dynamic_isostatic',
                           'static', 'tail')}
    for p_r, dP in zip(c['p_R'], c['dP']):
        cr = cum(p_r)
        j = int(np.argmax(cr[sel])) + int((z <= 30e3).sum())
        zs = z[j]
        meas = cr[j]
        tilt = -dP * zs
        static = cum(p_r - dP)[plateau].mean()
        out['z_star'].append(zs)
        out['measured'].append(meas)
        out['tilt'].append(tilt)
        out['dynamic_isostatic'].append(meas + tilt)
        out['static'].append(static)
        out['tail'].append(static - (meas + tilt))
    out = {k: np.array(v) for k, v in out.items()}
    out['t'] = c['t']
    out['mid'] = c['mid']
    return out

if __name__ == '__main__':
    build()
