"""Stress-section cache — 2D stress and temperature fields at the reference epoch.

For the SI stress-state figure: sigma_xx, sigma_zz, tau_zx and T on a 2 km
grid around the trench and down the slab, both models, at the declared
reference snapshot (40 Myr).

REGION. Native x from x_T - 500 km (seaward, over the outer rise and
trailing plate) to x_T + 700 km (landward, down the slab), 0-500 km depth.
Wide enough for the top panels to show the analysis box to 150 km and for
the lower panels to follow the slab.

Cache: notebooks/outputs/stress_section.npz. Run directly (~1 min).
"""
import os, sys, glob, warnings, logging
import numpy as np, natsort, pyvista as pv
warnings.filterwarnings('ignore'); logging.disable(logging.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from fluidity_helpers import archive_root
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
COLS = os.path.join(ROOT, 'notebooks', 'outputs', 'column_profiles_deep.npz')
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'stress_section.npz')
Y = 2_900_000.0
DX, ZMAX = 2e3, 500e3
SEAWARD, LANDWARD = 500e3, 700e3
REF_MYR = 40.0
SPY = 31_557_600.0


def build():
    archive_root()   # the validation deferred from import time
    cache = np.load(COLS)
    z = np.arange(0, ZMAX, DX) + DX / 2
    out = {'z': z}
    for KEY in ('STD', 'WAL'):
        tc = cache[f'{KEY}_t']
        files = natsort.natsorted(
            glob.glob(DATA + f'{KEY}_RefModel/subduction_with_LM*.pvtu'))
        best, bt = None, 1e9
        for ff in files:
            v = pv.read(ff)
            t = float(np.asarray(v['NormalSP::Time']).flat[0]) / SPY / 1e6
            if abs(t - REF_MYR) < bt:
                best, bt, btime = ff, abs(t - REF_MYR), t
            del v
        i = int(np.argmin(np.abs(tc - btime)))
        xT = float(cache[f'{KEY}_xT'][i])
        x = np.arange(xT - SEAWARD, xT + LANDWARD, DX) + DX / 2
        X, Z = np.meshgrid(x, z, indexing='ij')
        v = pv.read(best)
        s = pv.PolyData(np.c_[X.ravel(), (Y - Z).ravel(),
                              np.zeros(X.size)]).sample(v)
        p = np.asarray(s['NormalSP::Pressure']).reshape(X.shape)
        st = np.asarray(s['NormalSP::Stress'])
        out[f'{KEY}_sxx'] = -p + st[:, 0].reshape(X.shape)
        out[f'{KEY}_szz'] = -p + st[:, 4].reshape(X.shape)
        out[f'{KEY}_tzx'] = -st[:, 1].reshape(X.shape)     # z-down, SYMBOLOGY §5
        out[f'{KEY}_temp'] = np.asarray(s['NormalSP::Temperature']).reshape(X.shape)
        out[f'{KEY}_x'] = x
        out[f'{KEY}_xT'] = xT
        out[f'{KEY}_xI'] = float(cache[f'{KEY}_xI'][i])
        out[f'{KEY}_t'] = btime
        print(f'  {KEY} t = {btime:.1f} Myr, x_T = {xT/1e3:.0f} km, '
              f'{len(x)} x {len(z)} grid', flush=True)
        del v, s
    # FIELDS AS float32 (2026-10-09). These are plotted, not integrated --
    # pcolormesh and contour -- and float64 doubled the cache for no
    # visible difference. The repo ships its caches so every figure can be
    # re-rendered without the Zenodo data; that only works if they are a
    # sane size. Coordinate axes and times stay float64.
    out = {k: (a.astype(np.float32)
               if (np.asarray(a).dtype == np.float64 and np.asarray(a).ndim >= 2)
               else a)
           for k, a in out.items()}
    np.savez_compressed(OUT, **out)
    print('written:', OUT)


def load():
    if not os.path.exists(OUT):
        raise SystemExit(f'run {__file__} to build the cache first')
    return np.load(OUT)


if __name__ == '__main__':
    build()
