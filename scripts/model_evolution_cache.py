"""Model-evolution cache — viscosity and temperature fields at four epochs.

Samples log10(viscosity) and temperature over the upper 1500 km at four
times per model, for the orientation figure that introduces STD and WAL.

FRAME. Sampling is done in the model's NATIVE coordinates (subducting
plate at x < x_T, ANALYSIS_CONVENTIONS §1.1). The figure displays the
mirrored view -- trench at the left, seaward increasing rightward -- and
that flip happens at plot time in fig_model_evolution.py, nowhere else.

TIMES. 8, 32, 56 and 68 Myr (paper-agent spec, 2026-10-09):
  8   the first retained snapshot, after the declared spin-up cut
  32  mid-early, slab established in the lower mantle
  56  mid-late
  68  inside WAL's 62-70 Myr episode, which the earlier six-panel
      notebook figure did not resolve
The retired notebook version used 2 and 4 Myr, both inside the excluded
spin-up window; nothing here does.

Cache: notebooks/outputs/model_evolution.npz. Run directly (~2 min).
"""
import os, sys, glob, warnings, logging
import numpy as np
import natsort
import pyvista as pv

warnings.filterwarnings('ignore')
logging.disable(logging.ERROR)

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
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'model_evolution.npz')

Y, XMAX = 2_900_000.0, 8_000_000.0
DX = 5e3                    # 5 km: whole-domain view, fast
ZMAX = 1_500e3
X0, X1 = 2_400e3, 6_200e3   # native x window containing trench and slab at all four times
TARGETS_MYR = (8.0, 32.0, 56.0, 68.0)
SECONDS_PER_YEAR = 31_557_600.0


def build():
    archive_root()   # the validation deferred from import time
    cache = np.load(COLS)
    x = np.arange(X0, X1, DX) + DX / 2
    z = np.arange(0, ZMAX, DX) + DX / 2
    X, Z = np.meshgrid(x, z, indexing='ij')
    pts = np.c_[X.ravel(), (Y - Z).ravel(), np.zeros(X.size)]
    out = {'x': x, 'z': z}
    for KEY in ('STD', 'WAL'):
        tc = cache[f'{KEY}_t']
        files = natsort.natsorted(
            glob.glob(DATA + f'{KEY}_RefModel/subduction_with_LM*.pvtu'))
        times = []
        for ff in files:
            v = pv.read(ff)
            times.append(float(np.asarray(v['NormalSP::Time']).flat[0])
                         / SECONDS_PER_YEAR / 1e6)
            del v
        times = np.array(times)
        eta, temp, xT, ts = [], [], [], []
        for target in TARGETS_MYR:
            i = int(np.argmin(np.abs(times - target)))
            v = pv.read(files[i])
            s = pv.PolyData(pts).sample(v)
            # viscosity is a 3x3 tensor in this dataset; the xx diagonal is
            # the isotropic value
            e = np.asarray(s['NormalSP::Viscosity'])
            e = e[:, 0] if e.ndim > 1 else e
            eta.append(np.log10(np.abs(e).reshape(X.shape) + 1e-30))
            temp.append(np.asarray(s['NormalSP::Temperature']).reshape(X.shape))
            j = int(np.argmin(np.abs(tc - times[i])))
            xT.append(float(cache[f'{KEY}_xT'][j]))
            ts.append(times[i])
            print(f'  {KEY} target {target:.0f} -> t = {times[i]:.1f} Myr, '
                  f'x_T = {xT[-1]/1e3:.0f} km', flush=True)
            del v, s
        out[f'{KEY}_log_eta'] = np.array(eta)
        out[f'{KEY}_temp'] = np.array(temp)
        out[f'{KEY}_xT'] = np.array(xT)
        out[f'{KEY}_t'] = np.array(ts)
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
