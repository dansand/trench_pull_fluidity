"""Deep column-profile cache for the sigma_zz anomaly figure family.

Per snapshot (t >= T_MIN_MYR), STD and WAL: vertical profiles to 250 km of
the full vertical normal stress sigma_zz, the horizontal normal stress
sigma_xx (so the fibre stress sigma_xx - sigma_zz is available for the
mechanical-thickness analysis), density and temperature at the
trench, first-isostatic, ridge AND maximum-bending-moment columns (+-5 km window means,
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
  flip only under the mirror (fluidity_helpers.mirror_fields_in_x register).
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
- RIDGE PICK (2026-09-16): the ridge column is the maximum of the surface
  divergence dvx/dx seaward of the trench, REFINED to the local
  topographic crest within +/-20 km (`find_ridge_x_flow`, conventions
  §3.4, 2026-09-27), NOT the
  shallowest surface point. The topographic pick is stored alongside as
  `xR_topo` for comparison. See the helper's docstring for why.
- Sign pin: the trench-lobe integral over 0..z_c must reproduce the
  committed trench pulls (1.71 STD / 1.74 WAL TN/m at f10) -- asserted by
  fig_column_anomalies.py before rendering.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fluidity_helpers import (make_field_extractor, pick_trench_3step,
                           find_first_isostatic_column, find_ridge_x,
                           find_ridge_x_flow, surface_fs, archive_root)

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
OUT = os.path.join(ROOT, 'notebooks', 'outputs', 'column_profiles_deep.npz')

DX, Z_MAX, Y, W = 1000.0, 250e3, 2_900_000.0, 5
T_MIN_MYR = 8.0                    # spin-up cut (conventions §7.5)
# THE ONLY SPIN-UP CUT IN THE REPO, as of 2026-10-10. It had been written
# out separately in eight scripts -- the same failure as z_c, and the one
# ANALYSIS_CONVENTIONS §7.5 already warned about: "this constant existed
# only as T_MIN_MYR = 8.0 repeated across seven scripts, undeclared".
# Everything imports it from here now. Changing it requires rebuilding
# the caches: the excluded snapshots are dropped at build, not masked.
# ---------------------------------------------------------------------------
# THE ONE VERTICAL INTEGRATOR  (added 2026-10-10)
# ---------------------------------------------------------------------------
# Every quantity this project reports as an integral "from the surface to
# z_c" goes through here. It exists because they did not, and the obvious
# thing to write was wrong.
#
# THE DEFECT IT REPLACES. The profile grid is CELL-CENTRED: nodes at 0.5,
# 1.5, ... 79.5 km, representing 80 cells of 1 km that together span
# exactly 0-80 km. `np.trapz(values, z)` integrates BETWEEN THE FIRST AND
# LAST SAMPLE -- 0.5 to 79.5 km, a span of 79 km -- while every label,
# caption and register entry said 0 to z_c = 80 km. Measured over the 37
# retained snapshots of both runs, that omitted
#
#     the SURFACE half-cell   +0.0324 (STD) / +0.0288 (WAL) TN/m
#     the BASE half-cell      +0.0012        / +0.0007
#
# i.e. 96 % of it at the surface, where the trench deficit is at its
# maximum (64.8 / 57.6 MPa at the first node). Absolute forces were 1.5 %
# low; all tested qualitative conclusions were unaffected and the
# dimensionless quantities moved by under 1 % (the trench-pull:ridge-push
# ratio by 0.08 %). THIS IS THE THIRD INSTANCE of one root cause -- the
# cell-centred grid's first node is not the boundary. The other two were
# the free-surface read (fs[0, :] at z = 500 m) and the cell dropped at
# the z_c split in fig_ridge_push_anatomy.
#
# WHAT THE RULE IS, stated precisely rather than called "exact": the cell
# rule integrates the REPRESENTED CELL WIDTHS exactly, is second-order
# accurate for point samples, and is exactly conservative if the values
# are read as cell averages. It is NOT exact for an arbitrary curved
# profile. It agrees with endpoint-extrapolated trapezoid to 1 part in
# 1e4 on this data, which is the independent check, not the definition.
#
# IT REFUSES ANY OTHER GRID. Exterior cell edges cannot be recovered
# uniquely from non-uniform centres, so a non-uniform grid must supply its
# own widths rather than have them guessed.

def _cell_widths(z, zc_m):
    """Cells of a uniform, boundary-aligned centred grid, clipped to zc_m.

    Returns (index_mask, widths). Raises if `z` is not uniform or not
    half-a-cell off the origin -- do not relax this to accept centres
    alone on a non-uniform grid.
    """
    z = np.asarray(z, float)
    dz = np.diff(z)
    if not np.allclose(dz, dz[0], rtol=1e-9, atol=1e-6):
        raise ValueError('column integrator needs a uniform grid; pass '
                         'explicit cell widths for anything else')
    h = float(dz[0])
    if abs(z[0] - h / 2) > 1e-6:
        raise ValueError(f'grid is not boundary-aligned: first centre '
                         f'{z[0]} m on a {h} m spacing (expected {h/2})')
    n = int(round(zc_m / h))
    if abs(n * h - zc_m) > 1e-6:
        raise ValueError(f'z_c = {zc_m} m is not a whole number of {h} m '
                         f'cells; the integrator will not fake a part cell')
    if n > len(z):
        raise ValueError(f'z_c = {zc_m} m is deeper than the stored profile')
    m = np.zeros(len(z), bool); m[:n] = True
    return m, h


def integrate_to_zc(values, z, zc_m=None, axis=-1):
    """Integral of `values` over the cells spanning 0 to z_c. THE rule.

    `values` is sampled at the centres `z`; the integral covers the full
    0..z_c interval, not the span between the first and last sample.
    """
    zc_m = ZC_KM * 1e3 if zc_m is None else zc_m
    m, h = _cell_widths(z, zc_m)
    return np.take(values, np.where(m)[0], axis=axis).sum(axis=axis) * h


def cumulative_to_zc(values, z, zc_m=None, axis=-1):
    """Cumulative integral ON CELL EDGES: 0, h, 2h, ... z_c.

    Returns (z_edges, cumulative) with len = n_cells + 1, cumulative[0] = 0
    and cumulative[-1] == integrate_to_zc(...) BY CONSTRUCTION. Returning
    an N-long cumulative against the CENTRE depths is what re-introduces
    the half-cell ambiguity this module exists to remove, so it is not
    offered.
    """
    zc_m = ZC_KM * 1e3 if zc_m is None else zc_m
    m, h = _cell_widths(z, zc_m)
    v = np.take(values, np.where(m)[0], axis=axis)
    c = np.cumsum(v, axis=axis) * h
    pad = [(0, 0)] * c.ndim; pad[axis if axis >= 0 else c.ndim + axis] = (1, 0)
    return np.arange(int(m.sum()) + 1) * h, np.pad(c, pad)


def cumulative_edges(values, z, axis=-1):
    """Cumulative integral over the WHOLE stored profile, on cell edges.

    Returns (z_edges, cumulative) with len = len(z) + 1: edges at
    0, h, 2h, ... len(z)*h, and cumulative[k] = the integral over cells
    0..k-1, i.e. the integral from the SURFACE to edge k. cumulative[0] = 0.

    Use this for the cumulative-versus-depth panels, which run past z_c to
    150-250 km. Plot it against the returned edges, NOT against `z` -- the
    centres are one shorter, and pairing an edge cumulative with centre
    depths is the half-cell ambiguity this module exists to remove.
    """
    z = np.asarray(z, float)
    h = float(np.diff(z)[0])
    c = np.cumsum(values, axis=axis) * h
    pad = [(0, 0)] * c.ndim
    pad[axis if axis >= 0 else c.ndim + axis] = (1, 0)
    return np.concatenate([[0.0], z + h / 2]), np.pad(c, pad)


def _self_test():
    """Invariants, asserted at import. Prose is what failed last time.

    1. a constant 1 MPa over 0-80 km must integrate to exactly 0.080 TN/m
       -- the check that fails if the interval is ever 79 km again;
    2. the cumulative must land on its own scalar at the last edge;
    3. the grid guards must actually refuse.
    """
    h = 1e3
    zt = np.arange(80) * h + h / 2
    one = np.full(80, 1e6)
    got = integrate_to_zc(one, zt, 80e3)
    assert abs(got - 0.080e12) < 1.0, f'1 MPa over 0-80 km gave {got:.6e}'
    ze, cu = cumulative_to_zc(one, zt, 80e3)
    assert ze[0] == 0.0 and abs(ze[-1] - 80e3) < 1e-9
    assert abs(cu[-1] - got) < 1e-3, 'cumulative endpoint != scalar'
    assert cu.shape[0] == 81
    ze2, cu2 = cumulative_edges(one, zt)
    assert len(ze2) == 81 and len(cu2) == 81
    assert abs(ze2[-1] - 80e3) < 1e-9, 'edge array must end at the profile base'
    assert abs(cu2[-1] - got) < 1e-3, 'full-profile cumulative endpoint != scalar'
    for bad, lbl in (((np.arange(80) + 1.0) * h, 'not boundary-aligned'),
                     (np.cumsum(np.arange(1, 81) * h), 'non-uniform')):
        try:
            integrate_to_zc(one, bad, 80e3)
        except ValueError:
            pass
        else:
            raise AssertionError(f'integrator accepted a {lbl} grid')


_self_test()      # RUNS AT IMPORT. Anything that reads a column runs it.

DP_BAND_KM = (150.0, 220.0)        # deep Delta P band (see module docstring)
GRAD_SMOOTH_KM = 10.0              # x-smoothing before differentiating tau_zx

def build():
    archive_root()   # the validation deferred from import time
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
            v.point_data['txx'] = v['NormalSP::Stress'][:, 0]
            # z-down shear: tau_zx = -sigma_xy(y-up). Under the x-mirror it
            # flips sign again, so the analysis-frame field is +flip(sigma_xy).
            v.point_data['txz'] = -v['NormalSP::Stress'][:, 1]
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
            # NATIVE ORIENTATION (W31, 2026-09-28): the x-mirror is gone.
            # The model's own frame has the subducting plate on the LEFT and
            # the trench near x = 4000 km, so seaward is -x. Verified on a
            # snapshot: N_D, GPE*, Delta N_D, Delta GPE* and F_B come back
            # bit-identical, V and v_x exactly negated.
            p, tzz, txx = gf('p'), gf('tzz'), gf('txx')
            txz = gf('txz')
            rho, T = gf('rho'), gf('T')
            fs, vx = gf('fs'), gf('vx')
            szz = np.nan_to_num(tzz - p)
            sxx = np.nan_to_num(txx - p)
            txz = np.nan_to_num(txz)
            # along-strike gradient of the vertical shear stress, tau_zx,x --
            # the quantity whose depth distribution supports the trench
            # pressure deficit (the companion's equivalent density is
            # rho_hat = tau_zx,x / g). Smoothed over GRAD_SMOOTH_KM in x
            # before differencing; the raw field is too noisy to differentiate.
            dtxz_dx = np.gradient(
                gaussian_filter1d(txz, GRAD_SMOOTH_KM * 1e3 / DX, axis=1), DX, axis=1)
            rho = np.nan_to_num(rho)           # void rows above the surface -> 0
            T = np.nan_to_num(T)
            # THE SURFACE IS THE BOUNDARY NODES, not the top row of the
            # sampled grid, which sits at z = DX/2 = 500 m. The mesh does
            # not deform: FreeSurface is a BOUNDARY field and carries
            # something else immediately below, stepping from +2842 m at
            # z = 0 to -133 m at z = 500 m down the trench column.
            # surface_fs was written for this in September and applied to
            # balance_chain; THIS cache was never updated, so every pick
            # it makes has been reading a heavily low-passed surface.
            # Measured at the reference snapshot: the two fields differ by
            # rms 324 m (STD) / 501 m (WAL), x_I moves +9 / -4 km, and the
            # trench pull moves +7.6 / -3.5 %. Found by code review,
            # 2026-10-10.
            fs_top = surface_fs(v, x, mirror_x=False)
            xT, _ = pick_trench_3step(x, z, p, vx, subducting_side='left')
            ti = int(np.argmin(np.abs(x - xT)))
            # x_I IS THE dV/dx ZERO (conventions §3.2, 2026-10-15):
            # the local minimum of the vertical shear resultant, which is
            # blind to compensated topography and so carries none of the
            # cooling-plus-tilt bias the old topographic reference did.
            V_pick = integrate_to_zc(txz[z <= ZC_KM * 1e3].T,
                                     z[z <= ZC_KM * 1e3], ZC_KM * 1e3, axis=1)
            iI = find_first_isostatic_column(x, V_pick, xT,
                                             seaward_sign=-1)
            # ridge from the FLOW (max surface divergence) -- the
            # topographic pick is retained alongside for comparison
            iz10 = int(np.argmin(np.abs(z - 10e3)))
            # flow pick REFINED to the local topographic crest within
            # +/-20 km (conventions §3.4, 2026-09-27): the divergence
            # maximum is the spreading axis and the spreading axis sits in
            # the axial valley
            xR, iR = find_ridge_x_flow(x, vx[iz10], xT, seaward_sign=-1,
                                       fs_top=fs_top)
            xR_topo, _ = find_ridge_x(x, fs_top, xT, seaward_sign=-1)
            ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)
            # HORIZONTALLY AVERAGED HORIZONTAL VELOCITY (Dan, 2026-09-21):
            # one mean vertical profile of v_x per snapshot, averaged over
            # every column from the trench to the ridge. The flow structure
            # beneath the plate is a regional property; single columns
            # carry local detail that has nothing to do with it. Units are
            # cm/yr (the velocity is converted at extraction), negative =
            # trench-ward in the mirrored analysis frame.
            j0, j1 = min(ti, iR), max(ti, iR)
            vx_TR = vx[:, j0:j1 + 1].mean(axis=1)
            # accumulated basal traction along the plate: tau_zx on the
            # horizontal plane at z_c, integrated in x from the trench.
            # Needed for the three-term budget through time (Dan,
            # 2026-09-20); the balance is dN_D - dGPE* + F_B = 0.
            # tau_b ON THE LOWER FACE OF THE CONTROL VOLUME, z = z_c, not at
            # the centre of the nearest cell (2026-10-10). The grid is
            # cell-centred, so argmin picked 79.5 km while the balance is
            # defined for the 0..z_c volume and its basal traction belongs at
            # 80 km. Interpolated between the centres either side. The
            # horizontal integration in x below stays trapezoidal: its limits
            # ARE sampled x positions, so it has no half-cell problem.
            iz_lo = int(np.where(z <= ZC_KM * 1e3)[0][-1])
            w_hi = (ZC_KM * 1e3 - z[iz_lo]) / (z[iz_lo + 1] - z[iz_lo])
            tau_b = (1 - w_hi) * txz[iz_lo] + w_hi * txz[iz_lo + 1]
            FB_x = np.concatenate(
                [[0.0], np.cumsum(0.5 * (tau_b[1:] + tau_b[:-1]) * DX)])
            FB_x = FB_x - FB_x[ti]
            # column of MAXIMUM BENDING MOMENT (Dan, 2026-09-16): the
            # flexure-based thickness definitions belong here, not at the
            # trench. Fixed pivot = the trench neutral plane; searched from
            # 50 km landward of the trench to 500 km SEAWARD of it, smoothed
            # 5 km before the argmax. In the native frame (W31) seaward is
            # -x, so the window is x_T - 500 km .. x_T + 50 km. Getting this
            # backwards put x_M ~420 km on the wrong side and took szz_M,
            # sxx_M, temp_M, txz_M and M_max with it -- caught by the
            # invariance test, which is what it is for.
            fib = sxx - szz
            Ccol = np.cumsum(ca(fib, ti)) * DX
            wnp = (z >= 5e3) & (z <= 60e3)
            z_np0 = z[wnp][int(np.argmax(np.abs(Ccol[wnp])))]
            # 0 -> z_c BY THE CELL RULE (2026-10-10). This selects x_m, so
            # x_m could move; measured after the rebuild rather than assumed.
            Mx = integrate_to_zc((fib * (z[:, None] - z_np0)).T, z,
                                 ZC_KM * 1e3, axis=1)
            sea = np.where((x >= xT - 500e3) & (x <= xT + 50e3))[0]
            iM = sea[int(np.argmax(np.abs(gaussian_filter1d(Mx[sea], 5))))]
            rows.append(dict(t=t_myr, xM=x[iM], M_max=Mx[iM],
                             szz_M=ca(szz, iM), sxx_M=ca(sxx, iM), temp_M=ca(T, iM),
                             txz_M=ca(txz, iM), dtxz_M=ca(dtxz_dx, iM),
                             txz_T=ca(txz, ti), dtxz_T=ca(dtxz_dx, ti),
                             txz_I=ca(txz, iI), dtxz_I=ca(dtxz_dx, iI),
                             szz_T=ca(szz, ti), szz_I=ca(szz, iI), szz_R=ca(szz, iR),
                             sxx_T=ca(sxx, ti), sxx_I=ca(sxx, iI), sxx_R=ca(sxx, iR),
                             rho_T=ca(rho, ti), rho_I=ca(rho, iI), rho_R=ca(rho, iR),
                             temp_T=ca(T, ti), temp_I=ca(T, iI), temp_R=ca(T, iR),
                             vx_TR=vx_TR,
                             xT=xT, xI=x[iI], xR=xR, xR_topo=xR_topo,
                             FB_I=float(FB_x[iI]), FB_R=float(FB_x[iR]),
                             # the SAME tau_b the F_B accumulation uses, i.e. on the
                             # z_c face, not at the nearest cell centre
                             tau_b_rms=float(np.sqrt(np.mean(tau_b ** 2)))))
            print(f'{KEY} t={t_myr:6.2f} Myr  xT={xT/1e3:7.1f} xI={x[iI]/1e3:7.1f} '
                  f'xR={xR/1e3:7.1f} (topo {xR_topo/1e3:7.1f})  '
                  f'xM-xT={(x[iM]-xT)/1e3:+6.1f}', flush=True)
            del v, g
        out[KEY] = rows
    keys = ('t', 'xM', 'M_max', 'szz_M', 'sxx_M', 'temp_M',
            'txz_M', 'dtxz_M', 'txz_T', 'dtxz_T', 'txz_I', 'dtxz_I',
            'szz_T', 'szz_I', 'szz_R', 'sxx_T', 'sxx_I', 'sxx_R',
            'rho_T', 'rho_I', 'rho_R',
            'temp_T', 'temp_I', 'temp_R', 'xT', 'xI', 'xR', 'xR_topo', 'FB_I', 'FB_R',
            'tau_b_rms', 'vx_TR')
    # this tuple is the SAVE LIST, not derived from the row dicts — a new
    # quantity added to rows.append() is silently dropped unless it is
    # named here too (cost me one full rebuild, 2026-09-21)
    missing = set(next(iter(out.values()))[0]) - set(keys)
    assert not missing, f'computed but not saved: {sorted(missing)}'
    np.savez(OUT,
             **{f'{k}_{q}': np.array([r[q] for r in rows]) for k, rows in out.items()
                for q in keys},
             z=(np.arange(int(Z_MAX / DX)) + 0.5) * DX,
             dp_band_km=np.array(DP_BAND_KM))
    print('cache written:', OUT)

GRAV = 9.8
ZC_KM = 80.0                       # column-integral depth (conventions §5.1)
# THE ONLY z_c IN THE REPO. Raised from 75 km on 2026-10-09 (Dan's ruling,
# conventions §5.1): 80 km is the joint minimum of the rms error against
# the measured per-snapshot sigma_zz crossing in both runs. Three other
# scripts used to carry their own copy of this number -- balance_chain,
# fig_neutral_plane and fig_stress_state -- and they now import it.
# No fixed depth is right at every state; the crossing moves 42-126 km
# across the run.
MIDRUN_MYR = (36.0, 44.0)          # candidate mid-run reference window (W12 pending)
MIDRUN_TOL = 0.1                   # see midrun_mask(): archive times are not exact


def midrun_mask(t):
    """The mid-run window as a boolean mask over an array of model times.

    THE ONLY PLACE THE WINDOW IS APPLIED. It exists as a function rather
    than two constants because the bare comparison is wrong and was
    written twice: `lab_kinematics_cache` carried its own
    MIDRUN_MYR = (36.0, 44.0) and fig_ridge_push_anatomy compared against
    it with a bare `>=`/`<=`, so its velocity panel averaged four WAL
    snapshots while every other mid-run average in the repo used five.
    Importing a constant does not stop that; importing the mask does.
    """
    t = np.asarray(t)
    return (t >= MIDRUN_MYR[0] - MIDRUN_TOL) & (t <= MIDRUN_MYR[1] + MIDRUN_TOL)

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
    # CELL RULE, cumulative on cell edges -- the same convention as every
    # other cumulative in the repo as of 2026-10-10. This was a
    # centre-to-centre trapezoid, so the density route it feeds started
    # accumulating half a cell below the surface while the sigma_zz curve it
    # is compared against now starts at 0. Only the SHAPE of this curve is
    # used (its deep constant is removed below), so the effect is small --
    # but a comparison of two curves must not use two rules.
    lith = np.concatenate(
        [np.zeros((len(t), 1)),
         np.cumsum(rho_d, axis=1) * float(np.diff(z)[0])], axis=1)[:, :len(z)] * GRAV
    lith_R = lith - lith[:, band].mean(axis=1, keepdims=True)
    dP = p_R[:, band].mean(axis=1)
    dP_T = np.abs(p_T[:, zc]).max(axis=1)
    return dict(t=t, z=z, band=band, zc=zc,
                p_T=p_T, p_R=p_R, lith_R=lith_R,
                p_R_untilted=p_R - dP[:, None],
                dP=dP, dP_T=dP_T,
                # HALF-OPEN BY A TOLERANCE, not by a bare comparison.
                # WAL's 36 Myr snapshot is archived at t = 35.9905 and
                # STD's at 36.0035, so a bare `t >= 36.0` put five
                # snapshots in STD's mid-run window and four in WAL's --
                # the declared window (conventions §4b) is 36-44 for both,
                # and every mid-run average silently disagreed with it for
                # one model. The archive cadence is 2 Myr, so 0.1 Myr is
                # unambiguous. Found 2026-10-09 from a panel title reading
                # 'avg 38-44 Myr, n=4'.
                mid=midrun_mask(t))

# THE REFERENCE SNAPSHOT (conventions §4b, Dan's mid-run rule 2026-09-15):
# t = 40.0 Myr for BOTH runs. Single-sourced here beside z_c and the spin-up
# cut, for the same reason -- the epoch was a literal in each script that
# quoted it.
REF_MYR = 40.0


def reference_index(t, ref_myr=None):
    """Index of the reference snapshot in a cache time array.

    Nearest to REF_MYR, with the same 0.1 Myr archive tolerance the mid-run
    window needs (the archive stamps 40 Myr as 39.9905 in one run).
    """
    ref_myr = REF_MYR if ref_myr is None else ref_myr
    i = int(np.argmin(np.abs(np.asarray(t) - ref_myr)))
    if abs(np.asarray(t)[i] - ref_myr) > 1.0:
        raise ValueError(f'no snapshot within 1 Myr of t = {ref_myr}')
    return i


PLATEAU_KM = (150.0, 240.0)    # where cum(D) has flattened (spread < 0.3 %)

def partition(d, key):
    """Per-snapshot decomposition of the x_I-to-ridge sigma_zz term.

    Single implementation, used by every figure and table that quotes the
    partition. ONE DEPTH THROUGHOUT, AND IT IS THE FIXED z_c.

    ⚠ CORRECTED 2026-10-09. This function used to integrate to z*, the
    per-snapshot sign change of the measured anomaly, while
    fig_gpe_partition integrated to the fixed z_c -- so the repo reported
    two different trench pulls for the same quantity (2.082 against
    2.1215 TN/m in STD) and the manuscript quoted both, in different
    sections. Dan's ruling: everything the paper reports uses the fixed
    integration depth. The analysis chooses one z_c, computes the balance,
    and SHOWS that the choice does not matter much; a reader who prefers
    another depth is welcome to it. A per-snapshot depth belongs only in
    the diagnostic that displays that sensitivity (fig_ridge_cumulative),
    never in a reported force.

    z_star is still returned, because it is a real measured property of
    the model and the sensitivity figure needs it -- but NOTHING here
    integrates to it. Returns, per snapshot (N/m unless noted):

      z_star    the crossing depth [m], REPORTED ONLY: argmax of the
                cumulative integral. Not an integration limit.
      measured  the term as it enters the force balance, integral to z_c
      tilt      |Delta P| * z_c  -- the back-tilt contribution contained
                in `measured`, evaluated at the SAME depth (using any
                other thickness would redefine the plate midway through
                the derivation: a different, static model)
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
                below z_c, excluded from the balance.
    """
    c = derive(d, key)
    z = c['z']
    plateau = (z >= PLATEAU_KM[0] * 1e3) & (z <= PLATEAU_KM[1] * 1e3)
    sel = (z > 30e3) & (z < 200e3)
    # ON CELL EDGES (2026-10-10): cumulative_to_zc returns 0, h, ... z_c with
    # its endpoint equal to integrate_to_zc by construction, so `meas` below
    # is the 0-z_c integral and not the 0.5-79.5 km one. The old centre-to-
    # centre trapezoid, read at the last centre, was both.
    cum_zc = lambda a: cumulative_to_zc(a, z)[1]
    # the DEEP cumulative still needs the full 250 km profile for `static`,
    # and over that range the same cell rule applies
    cum_all = lambda a: np.concatenate([[0.0], np.cumsum(a) * float(np.diff(z)[0])])
    out = {k: [] for k in ('z_star', 'measured', 'tilt', 'dynamic_isostatic',
                           'static', 'tail')}
    zcm = z <= ZC_KM * 1e3                           # the fixed z_c
    jc = int(np.where(zcm)[0][-1])
    for p_r, dP in zip(c['p_R'], c['dP']):
        cr_all = cum_all(p_r)
        # z* is measured and reported; the INTEGRAL stops at z_c
        zs = z[int(np.argmax(cr_all[1:][sel])) + int((z <= 30e3).sum())]
        meas = cum_zc(p_r)[-1]                  # exactly 0 -> z_c
        tilt = -dP * ZC_KM * 1e3                # z_c, not the last centre
        static = cum_all(p_r - dP)[1:][plateau].mean()
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

def resultants(d, key, zc_m=None):
    """Column resultants at the trench and ridge, per snapshot.

    Single implementation, so every figure quoting N_D, V or M uses the
    same columns, the same pickers and the same integration depth. Added
    2026-09-17 to take fig_nd_trench_ridge and fig_trench_resultants off
    the notebook-built time-evolution cache, which predates the
    flow-based ridge pick and placed the ridge ~300 km away.

    Returns (N/m unless noted):
      nd_T, nd_I, nd_R  the normal-stress-difference resultant at the
                  trench, first-isostatic and ridge columns, integrated to
                  zc_m (default z_c)
      v_T         the vertical shear resultant at the trench, int tau_zx dz
      m_T         the bending moment at the trench about its neutral
                  plane [N], with h_np from the extremum of the
                  cumulative fibre stress over 5-60 km
      h_np        that neutral-plane depth [m]
      t           model time [Myr]
    """
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    z = d['z']
    zcm = ZC_KM * 1e3 if zc_m is None else zc_m
    zc = z <= zcm
    fib_T = d[f'{key}_sxx_T'] - d[f'{key}_szz_T']
    fib_R = d[f'{key}_sxx_R'] - d[f'{key}_szz_R']
    fib_I = d[f'{key}_sxx_I'] - d[f'{key}_szz_I']
    C = np.cumsum(fib_T * np.gradient(z), axis=1)
    w = (z >= 5e3) & (z <= 60e3)
    h_np = z[w][np.argmax(np.abs(C[:, w]), axis=1)]
    return dict(
        t=d[f'{key}_t'],
        # THE CELL RULE, not trapz over the centres -- see integrate_to_zc.
        nd_T=integrate_to_zc(fib_T, z, zcm, axis=1),
        nd_R=integrate_to_zc(fib_R, z, zcm, axis=1),
        nd_I=integrate_to_zc(fib_I, z, zcm, axis=1),
        v_T=integrate_to_zc(d[f'{key}_txz_T'], z, zcm, axis=1),
        m_T=integrate_to_zc(fib_T * (z[None, :] - h_np[:, None]), z, zcm,
                            axis=1),
        h_np=h_np)

if __name__ == '__main__':
    build()


def terms(d, key):
    """The three balance terms between the trench and ridge columns,
    each as its contribution to the net force in +x.

    Moved here from fig_budget_time.py on 2026-10-10. It is a reduction
    of the cache, not a figure, and four figure scripts imported it --
    which meant a non-paper figure script had to ship just to carry it,
    and running it wrote a table the paper does not use.
    """
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    z = d['z']
    zc = z <= ZC_KM * 1e3
    r = resultants(d, key)
    szz_T = integrate_to_zc(d[f'{key}_szz_T'], z, axis=1)
    szz_R = integrate_to_zc(d[f'{key}_szz_R'], z, axis=1)
    d_gpe = -(szz_R - szz_T)                 # the GPE-like force
    d_nd = r['nd_R'] - r['nd_T']
    f_b = d[f'{key}_FB_R']
    return dict(t=r['t'], d_nd=d_nd, gpe_term=-d_gpe, f_b=f_b,
                closure=d_nd - d_gpe + f_b, d_gpe=d_gpe)
