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
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'lab_kinematics.npz')

Y, XMAX = 2_900_000.0, 8_000_000.0
ZMAX, DZ = 300e3, 2e3
DXS = 10e3                 # sampling spacing across the plate span
from column_profiles_cache import T_MIN_MYR   # the only spin-up cut
SECONDS_PER_YEAR = 31_557_600.0
PER_YEAR = 3.17098e-10     # m/s -> cm/yr conversion used suite-wide

def build():
    archive_root()   # the validation deferred from import time
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
            # x_I and x_R come from the column cache, which is in the
            # MODEL'S OWN coordinates since W31 (2026-09-28), so no
            # conversion is needed. Natively x_R < x_I (the ridge is at
            # small x), and this arange must run increasing. The old line
            # applied the mirrored->native map a second time and returned an
            # EMPTY array, i.e. zero sampled columns per snapshot.
            xs = np.arange(xR_a + DXS, xI_a, DXS)
            P, Z = np.meshgrid(xs, z, indexing='ij')
            s = pv.PolyData(np.c_[P.ravel(), (Y - Z).ravel(),
                                  np.zeros(P.size)]).sample(v)

            def sq(name, ncol=0):
                a = np.asarray(s[name])
                if a.ndim > 1:
                    a = a[:, ncol]
                return a.reshape(P.shape)

            vx = sq('NormalSP::Velocity', 0) / PER_YEAR       # cm/yr, native frame (W31)
            # SIGN FIX 2026-10-04. The z-down flip tau_zx = -sigma_xy(y-up)
            # (SYMBOLOGY §5) was missing here, so this cache carried the
            # OPPOSITE shear-stress convention to column_profiles_cache.py,
            # which applies it. Decisive test: in a sheared channel
            # tau_zx = eta dv_x/dz, and over 120-220 km the cached value was
            # +0.53 MPa against eta dv_x/dz of -0.57 (correlation -1.00,
            # magnitudes within 8 %).
            # It stayed hidden because `span` = x_R - x_I was positive in the
            # retired mirrored frame and is negative natively, so two sign
            # errors cancelled in the one place the cache was checked --
            # fig_lab_kinematics panel (d). The frame change broke the
            # cancellation and exposed this one.
            txz = -sq('NormalSP::Stress', 1)                  # tau_zx, z-down
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

# THE MID-RUN WINDOW IS NOT DEFINED HERE. It was -- as a bare
# MIDRUN_MYR = (36.0, 44.0), compared without the 0.1 Myr tolerance the
# archive times need, which cost fig_ridge_push_anatomy one WAL snapshot.
# Single-sourced in column_profiles_cache, like z_c and the spin-up cut.
from column_profiles_cache import MIDRUN_MYR, MIDRUN_TOL, midrun_mask   # noqa: F401

def load():
    if not os.path.exists(OUT):
        raise SystemExit(f'cache missing: {OUT} — run lab_kinematics_cache.py first')
    return np.load(OUT)

if __name__ == '__main__':
    build()
