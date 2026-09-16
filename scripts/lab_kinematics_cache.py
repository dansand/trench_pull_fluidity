"""Plate-averaged kinematics cache — the lithosphere–asthenosphere
evidence base for the trailing plate.

Per snapshot (t >= T_MIN_MYR), STD and WAL: profiles to 300 km of the
horizontal velocity, strain-rate invariant, viscosity, shear stress and
temperature, AVERAGED ACROSS THE TRAILING PLATE from the first isostatic
column x_I to the ridge column x_R. Averaging across the span is what
makes the shear-stress test meaningful: averaging horizontal momentum
over x gives d<tau_zx>/dz = -(sigma_xx(x_R) - sigma_xx(x_I))/span, so the
channel value of d<tau_zx>/dz must equal the measured plate-wide pressure
gradient dP/dx (verified: ratios 0.93 STD / 0.85 WAL, 2026-09-16).

Picks (x_I, x_R) are READ FROM the committed column-profile cache
(column_profiles_cache.py) so both caches share one set of pickers;
build that first. Sampling is done in NATIVE coordinates with explicit
sign flips for the mirrored analysis frame (vx, tau_zx), so no array
mirroring is applied here.

Cache: notebooks/outputs/lab_kinematics.npz. Run directly to build
(~10 min). Consumed by fig_lab_kinematics.py.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser('~/DATA/numerical_models/OUTPUTS/')
COLS = os.path.join(ROOT, 'notebooks', 'outputs', 'column_profiles_deep.npz')
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'lab_kinematics.npz')

Y, XMAX = 2_900_000.0, 8_000_000.0
ZMAX, DZ = 300e3, 2e3
DXS = 10e3                 # sampling spacing across the plate span
T_MIN_MYR = 8.0
SECONDS_PER_YEAR = 31_557_600.0
PER_YEAR = 3.17098e-10     # m/s -> cm/yr conversion used suite-wide

def build():
    if not os.path.exists(COLS):
        raise SystemExit(f'build column_profiles_cache.py first: {COLS}')
    cache = np.load(COLS)
    z = np.arange(0, ZMAX, DZ) + DZ / 2
    out = {}
    for KEY in ('STD', 'WAL'):
        tc = cache[f'{KEY}_t']
        rows = []
        for ff in natsort.natsorted(glob.glob(DATA + f'{KEY}_RefModel/subduction_with_LM*.pvtu')):
            v = pv.read(ff)
            t = float(np.asarray(v['NormalSP::Time']).flat[0]) / SECONDS_PER_YEAR / 1e6
            if t < T_MIN_MYR:
                del v
                continue
            i = int(np.argmin(np.abs(tc - t)))
            if abs(tc[i] - t) > 0.5:
                del v
                continue
            xI_a, xR_a = float(cache[f'{KEY}_xI'][i]), float(cache[f'{KEY}_xR'][i])
            xs = np.arange(XMAX - xR_a + DXS, XMAX - xI_a, DXS)   # native coords
            P, Z = np.meshgrid(xs, z, indexing='ij')
            s = pv.PolyData(np.c_[P.ravel(), (Y - Z).ravel(),
                                  np.zeros(P.size)]).sample(v)

            def sq(name, ncol=0):
                a = np.asarray(s[name])
                if a.ndim > 1:
                    a = a[:, ncol]
                return a.reshape(P.shape)

            vx = -sq('NormalSP::Velocity', 0) / PER_YEAR      # cm/yr, analysis frame
            txz = sq('NormalSP::Stress', 1)                   # analysis-frame tau_zx
            eii = sq('NormalSP::SecondInvariant_StrainRate')
            eta = sq('NormalSP::Viscosity')
            T = sq('NormalSP::Temperature')
            rows.append(dict(t=t, span=xR_a - xI_a,
                             vx=np.nanmean(vx, axis=0),
                             txz=np.nanmean(txz, axis=0),
                             log_eii=np.nanmean(np.log10(np.clip(eii, 1e-22, None)), axis=0),
                             log_eta=np.nanmean(np.log10(np.clip(eta, 1.0, None)), axis=0),
                             temp=np.nanmean(T, axis=0)))
            print(f'{KEY} t={t:6.2f} Myr  span {xI_a/1e3:.0f}–{xR_a/1e3:.0f} km '
                  f'({len(xs)} columns)', flush=True)
            del v, s
        out[KEY] = rows
    keys = ('t', 'span', 'vx', 'txz', 'log_eii', 'log_eta', 'temp')
    np.savez(OUT, **{f'{k}_{q}': np.array([r[q] for r in rows])
                     for k, rows in out.items() for q in keys}, z=z)
    print('cache written:', OUT)

MIDRUN_MYR = (36.0, 44.0)      # conventions §4b mid-run window

def load():
    if not os.path.exists(OUT):
        raise SystemExit(f'cache missing: {OUT} — run lab_kinematics_cache.py first')
    return np.load(OUT)

if __name__ == '__main__':
    build()
