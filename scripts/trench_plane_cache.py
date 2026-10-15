"""Trench-plane cache — the stress state at the trench on two planes, through time.

Per snapshot (t >= 8 Myr, conventions §7.5), STD and WAL: a narrow 2-D patch
of the normal-stress difference, the shear stress and temperature straddling
the trench column, plus the local plate dip and the trench resultant taken
on two planes:

  VERTICAL      the plane the manuscript uses -- trace vertical at x_T
  SLAB-NORMAL   the plane perpendicular to the plate, tilted from vertical
                by the local dip theta

WHY A TIME SERIES. At the reference epoch alone the comparison is
uninterpretable: STD's trench resultant crosses zero near t = 35 Myr, so at
t = 40 Myr any absolute shift reads as a huge percentage. Dan's ruling
(2026-10-09): the shift is to be judged against the run-scale amplitude of
N_D(x_T), not against its instantaneous value, and the sign agreement is to
be tested where the stress state is unambiguous -- STD early (tension-like,
N_D(x_T) ~ +1.4 TN/m) and WAL late (compression-like, ~ -3.6 TN/m).

THE ALGEBRA, which is why the shift is a shear-load term and not a dip term:

    sigma_xi.xi - sigma_eta.eta = (sigma_xx - sigma_zz) cos 2theta
                                  + 2 tau_zx sin 2theta

so, integrated on the same vertical trace,

    N_D^rot - N_D = (cos 2theta - 1) N_D + 2 sin 2theta * V

with V = int tau_zx dz the trench-face shear load. At theta ~ 4 deg the
first term is 1 % of a small number; the second is 14 % of a large one.
The rotation shift is carried almost entirely by V.

Dip: the 800 C isotherm fitted over +/-10 km at x_T, the method worked out
in notebooks/further_analysis/fluidity_slab_normal_FD.ipynb. Do not
re-derive it.

Columns are +/-5 km window means (conventions §4.1) so N_D(x_T) here
reproduces the committed column_profiles_deep values.

Cache: notebooks/outputs/trench_plane.npz. Run directly (~15 min).
"""
import os, sys, glob, warnings, logging
import numpy as np, natsort, pyvista as pv
warnings.filterwarnings('ignore'); logging.disable(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fluidity_helpers import (pick_trench_3step, find_first_isostatic_column,
                               surface_fs, archive_root)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RESOLVED, NOT VALIDATED, AT IMPORT (2026-10-15). This called
# archive_root() with its default required=True, so merely importing this
# module demanded the full 1.4 GB archive -- and every figure script
# imports it for ZC_KM, T_MIN_MYR and the integrator. A clone with no
# archive could therefore render nothing, while the error it raised said
# "You do NOT need the archive to reproduce any figure in the paper".
# The README promises standalone reproduction; this broke it. The check
# now happens in build(), the only thing that reads the archive.
# Found by external review.
DATA = archive_root(required=False)
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'trench_plane.npz')

Y = 2_900_000.0
DX_PICK = 1000.0          # grid for the trench / x_I pickers (as the suite)
Z_PICK = 250e3
from column_profiles_cache import (T_MIN_MYR, ZC_KM,   # the only spin-up cut,
                                   integrate_to_zc)  # z_c and integrator
SPY = 31_557_600.0

# the narrow patch: seaward is -x in the native frame, and the tilted plane
# drifts seaward as it deepens, so the patch is asymmetric
PATCH_SEAWARD, PATCH_LANDWARD = 40e3, 20e3
PATCH_DX, PATCH_DZ, PATCH_ZMAX = 1e3, 500.0, 120e3
W_KM = 5.0                # column half-width, conventions §4.1
ISO_DIP_C = 800.0         # isotherm the dip is fitted to
DIP_WINDOW_KM = 10.0


def _dip(temp_C, x, z, xT, window_km=DIP_WINDOW_KM, t_iso_C=ISO_DIP_C):
    """Local plate dip from the t_iso_C isotherm, fitted over +/- window_km."""
    w = np.where(np.abs(x - xT) <= window_km * 1e3)[0]
    zi = np.array([np.interp(t_iso_C, temp_C[i, :], z)
                   if temp_C[i, 0] < t_iso_C < temp_C[i, -1] else np.nan
                   for i in w])
    g = np.isfinite(zi)
    if g.sum() < 3:
        return np.nan
    m = np.polyfit(x[w][g], zi[g], 1)[0]
    return float(np.arctan(m))


def build():
    archive_root()   # the validation deferred from import time
    out = {}
    for KEY in ('STD', 'WAL'):
        rows, patches = [], {'ns': [], 'tzx': [], 'temp': []}
        files = natsort.natsorted(
            glob.glob(DATA + f'{KEY}_RefModel/subduction_with_LM*.pvtu'))[1:]
        zp = (np.arange(int(PATCH_ZMAX / PATCH_DZ)) + 0.5) * PATCH_DZ
        for ff in files:
            v = pv.read(ff)
            t_myr = float(np.asarray(v['NormalSP::Time']).flat[0]) / SPY / 1e6
            if t_myr < T_MIN_MYR:
                del v; continue
            # --- picks, on the suite's 1 km grid -------------------------
            v.point_data['p'] = v['NormalSP::Pressure']
            v.point_data['tzz'] = v['NormalSP::Stress'][:, 4]
            v.point_data['txx'] = v['NormalSP::Stress'][:, 0]
            v.point_data['txz'] = -v['NormalSP::Stress'][:, 1]   # z-down, §5
            v.point_data['T'] = v['NormalSP::Temperature']
            v.point_data['fs'] = v['NormalSP::FreeSurface']
            v.point_data['vx'] = (v['NormalSP::Velocity'] / 3.17098e-10)[:, 0]
            x0, x1 = v.bounds[0], v.bounds[1]
            xg = x0 + (np.arange(int((x1 - x0) / DX_PICK)) + 0.5) * DX_PICK
            zg = (np.arange(int(Z_PICK / DX_PICK)) + 0.5) * DX_PICK
            Xg, Zg = np.meshgrid(xg, zg, indexing='xy')
            g = pv.StructuredGrid(Xg.T, (Y - Zg).T, np.zeros_like(Xg.T)).sample(v)
            nxp, nzp, _ = g.dimensions
            gf = lambda n: np.nan_to_num(
                np.asarray(g[n]).reshape((nxp, nzp), order='F')).T
            p_, vx_ = gf('p'), gf('vx')
            xT, _ = pick_trench_3step(xg, zg, p_, vx_, subducting_side='left')
            ti = int(np.argmin(np.abs(xg - xT)))
            # THE SURFACE IS THE TOP-BOUNDARY NODES, not the top row of the
            # sampled grid. This read fs_[0, :], the row at z = DX/2 = 500 m,
            # where NormalSP::FreeSurface -- a boundary field carried into the
            # domain and decaying with depth -- retains the pattern but only
            # ~60 % of the amplitude. The same slip in column_profiles_cache
            # moved x_I by 9 km and the trench pull by 7.6 % (fixed 2026-10-10,
            # commit 804e28a); this copy survived it because its x_I feeds no
            # figure. Fixed anyway -- a stale extraction is a latent bug.
            fs_top = surface_fs(v, xg, mirror_x=False)
            # x_I IS THE dV/dx ZERO (conventions §3.2, 2026-10-15):
            # the local minimum of the vertical shear resultant, which is
            # blind to compensated topography and so carries none of the
            # cooling-plus-tilt bias the old topographic reference did.
            _m = zg <= ZC_KM * 1e3
            V_pick = integrate_to_zc(gf('txz')[_m].T, zg[_m],
                                     ZC_KM * 1e3, axis=1)
            iI = find_first_isostatic_column(xg, V_pick, xT,
                                             seaward_sign=-1)
            xI = float(xg[iI])
            del g

            # --- the narrow patch ----------------------------------------
            xp = np.arange(xT - PATCH_SEAWARD, xT + PATCH_LANDWARD,
                           PATCH_DX) + PATCH_DX / 2
            X, Z = np.meshgrid(xp, zp, indexing='ij')
            s = pv.PolyData(np.c_[X.ravel(), (Y - Z).ravel(),
                                  np.zeros(X.size)]).sample(v)
            r = lambda n, j=None: np.asarray(
                s[n] if j is None else s[n][:, j]).reshape(X.shape)
            pp = r('NormalSP::Pressure')
            ns = (r('NormalSP::Stress', 0) - r('NormalSP::Stress', 4))
            tzx = -r('NormalSP::Stress', 1)
            tC = r('NormalSP::Temperature') - 273.15
            patches['ns'].append(ns); patches['tzx'].append(tzx)
            patches['temp'].append(tC)
            rows.append((t_myr, xT, xI, _dip(tC, xp, zp, xT),
                         float(xp[0] - xT)))
            print(f'  {KEY} t = {t_myr:5.1f} Myr  x_T = {xT/1e3:7.1f} km  '
                  f'dip = {np.degrees(rows[-1][3]):5.2f} deg', flush=True)
            del v, s
        a = np.array(rows)
        out[f'{KEY}_t'], out[f'{KEY}_xT'] = a[:, 0], a[:, 1]
        out[f'{KEY}_xI'], out[f'{KEY}_dip'] = a[:, 2], a[:, 3]
        out[f'{KEY}_x_rel'] = (np.arange(len(patches['ns'][0]))
                               * PATCH_DX + PATCH_DX / 2 - PATCH_SEAWARD)
        for k, vv in patches.items():
            out[f'{KEY}_{k}'] = np.array(vv, dtype=np.float32)
    out['z'] = (np.arange(int(PATCH_ZMAX / PATCH_DZ)) + 0.5) * PATCH_DZ
    out['w_km'] = W_KM
    out['iso_dip_C'] = ISO_DIP_C
    out['dip_window_km'] = DIP_WINDOW_KM
    np.savez_compressed(OUT, **out)
    print('written:', OUT)


def load():
    if not os.path.exists(OUT):
        raise SystemExit(f'run {__file__} to build the cache first')
    return np.load(OUT)


if __name__ == '__main__':
    build()
