"""slab_geometry_cache — CACHE BUILDER for slab descent and geometry.

Writes notebooks/outputs/slab_geometry.npz. Run before fig_slab_descent.
Takes a few minutes; reads every snapshot from t >= 8 Myr in both models.

WHY THIS EXISTS (2026-09-22). The column cache stops at 250 km, so
nothing in the analysis could see the slab. The question it cannot answer
is the one the whole force-balance investigation kept arriving at: the
trailing-plate balance measures every force correctly and still cannot
predict the plate velocity, because the boundary condition is set at the
hinge by whatever the slab is doing.

THE SLAB IS TRACKED BY ITS THERMAL ANOMALY, T < T_SLAB, not by the
material volume fraction. NormalSP:: and NormalOP::MaterialVolumeFraction
PARTITION THE WHOLE DOMAIN -- they sum to 1 at every depth and are side
labels, not plate markers, so SP > 0.5 selects half the mantle. That cost
one wasted extraction; do not try it again. Temperature works: against a
1573 K ambient the slab core reaches 911 K at 300 km and 1179 K at
700 km, with nothing cold at 1000 km.

Per snapshot, in the mirrored analysis frame:
  z_tip     deepest point of the cold anomaly below Z_MIN [km]
  vz_upper  mean vertical velocity of slab material, 200-600 km [cm/yr]
  vz_deep   the same below 600 km -- the LOWER-MANTLE removal rate
  dip       mean dip of the cold anomaly between 200 and 400 km [deg]
  area      cold-anomaly cross-section below Z_MIN [10^6 km^2 per unit
            length along strike]

Z_MAX is 2500 km deliberately: an earlier 1200 km grid clipped the WAL
slab from t = 68 Myr onward, so its late tip depth and area were
underestimates.

Vertical velocity is positive DOWNWARD, consistent with the z-down
analysis frame.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # the shared scripts/ dir
from fluidity_helpers import make_field_extractor

ROOT = os.path.dirname(os.path.dirname(_HERE))   # repo root
DATA = os.path.expanduser('~/DATA/numerical_models/OUTPUTS/')
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'slab_geometry.npz')
Y = 2_900_000.0
DX = 5000.0                 # slab geometry does not need the 1 km grid
Z_MAX = 2_500_000.0         # deep enough not to clip either slab
T_SLAB = 1300.0             # K; ambient is 1573 K
T_AMBIENT = 1573.0
Z_MIN = 150e3               # below the plates, so the lithosphere is excluded
BAND_UPPER = (200e3, 600e3)
BAND_DEEP = 600e3
DIP_BAND = (200e3, 400e3)
T_MIN_MYR = 8.0


def one_snapshot(v):
    v.point_data['T'] = v['NormalSP::Temperature']
    v.point_data['vz'] = -(v['NormalSP::Velocity'] / 3.17098e-10)[:, 1]
    x0, x1, _, _, _, _ = v.bounds
    nx, nz = int((x1 - x0) / DX), int(Z_MAX / DX)
    x = x0 + (np.arange(nx) + 0.5) * DX
    z = (np.arange(nz) + 0.5) * DX
    X, Z = np.meshgrid(x, z, indexing='xy')
    g = pv.StructuredGrid(X.T, (Y - Z).T, np.zeros_like(X.T)).sample(v)
    nxp, nzp, _ = g.dimensions
    val = g['vtkValidPointMask'].reshape((nxp, nzp), order='F').astype(bool)
    gf = make_field_extractor(g, nxp, nzp, val)
    T = gf('T')
    T = np.where(np.isnan(T), T_AMBIENT, T)      # outside the mesh = ambient
    vz = np.nan_to_num(gf('vz'))
    T, vz = np.flip(T, 1), np.flip(vz, 1)        # x-mirror; both invariant
    slab = (T < T_SLAB) & (z[:, None] > Z_MIN)

    rows = np.where(slab.any(axis=1))[0]
    z_tip = z[rows].max() / 1e3 if len(rows) else np.nan
    up = slab & (z[:, None] > BAND_UPPER[0]) & (z[:, None] < BAND_UPPER[1])
    dp = slab & (z[:, None] > BAND_DEEP)
    vz_upper = vz[up].mean() if up.any() else np.nan
    vz_deep = vz[dp].mean() if dp.any() else np.nan
    area = slab.sum() * DX * DX / 1e12           # 10^6 km^2
    cen = [(z[i], x[slab[i]].mean())
           for i in np.where((z > DIP_BAND[0]) & (z < DIP_BAND[1]))[0] if slab[i].any()]
    dip = np.nan
    if len(cen) > 2:
        zz = np.array([c[0] for c in cen]); xx = np.array([c[1] for c in cen])
        dip = np.degrees(np.arctan2(1.0, abs(np.polyfit(zz, xx, 1)[0])))
    return z_tip, vz_upper, vz_deep, dip, area


def build():
    out = {}
    for key in ('STD', 'WAL'):
        rows = []
        for ff in natsort.natsorted(
                glob.glob(DATA + f'{key}_RefModel/subduction_with_LM*.pvtu')):
            v = pv.read(ff)
            t = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
            if t < T_MIN_MYR - 0.1:
                del v
                continue
            rows.append((t,) + one_snapshot(v))
            print(f'{key} t={t:6.2f} Myr  z_tip {rows[-1][1]:7.0f} km  '
                  f'vz_up {rows[-1][2]:+5.2f}  vz_deep {rows[-1][3]:+5.2f}  '
                  f'dip {rows[-1][4]:4.0f}  area {rows[-1][5]:.3f}', flush=True)
            del v
        a = np.array(rows)
        for i, nm in enumerate(('t', 'z_tip', 'vz_upper', 'vz_deep', 'dip', 'area')):
            out[f'{key}_{nm}'] = a[:, i]
        if np.nanmax(a[:, 1]) > Z_MAX / 1e3 - 3 * DX / 1e3:
            print(f'  ** WARNING {key}: slab reaches the grid bottom — CLIPPED')
    np.savez(OUT, **out)
    print('cache written:', OUT)


def load():
    return np.load(OUT)


if __name__ == '__main__':
    build()
