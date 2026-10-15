"""balance_chain — the trailing-plate balance at one snapshot, computed once.

THE SINGLE IMPLEMENTATION of the extraction and force chain behind every
snapshot-level balance figure. It was duplicated for months:
fig_balance_snapshot.py and fig_balance_snapshot_absolute.py each carried
a byte-identical 115-line `compute` and 9-line `load_snapshot`, and their
`draw_direction_glyph` had DRIFTED -- the _absolute copy still rendered
the header Dan replaced on 2026-09-28. Two main-text figures, one of them
showing a rejected label. Hoisted here 2026-10-09; conventions §7.1 asks
for one implementation per repo and this is now it.

Consumers:
  fig_balance_snapshot.py           the Delta form (main text)
  fig_balance_snapshot_absolute.py  the same with the absolute N_D row
  fig_balance_absolute.py           the shifted form
  anim_balance.py                   every snapshot of the run, as a GIF

TIME IS A PARAMETER. `compute(key, t_target)` defaults to the declared
reference epoch (conventions §4b), so the figure scripts call it exactly
as they did. The animation walks the archive instead.

TWO PINS, and only one of them is epoch-specific. The identity
N_D = N_D(x_T) + dGPE* - F_B is a wiring check that must hold at every
snapshot and is asserted always. The comparison against the committed
t = 40 trench pull is a sign pin for the reference epoch only, so it is
skipped when a different time is asked for -- `pin_committed` makes that
explicit rather than silent.
"""
import os, sys, glob
import numpy as np
import natsort
import pyvista as pv
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fluidity_helpers import (make_field_extractor, pick_trench_3step,
                              find_first_isostatic_column, find_ridge_x_flow,
                              surface_fs, archive_root)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# THE BUNDLED SNAPSHOT FIRST, the full archive second. The public
# repository ships the reference snapshot of both runs in data/, so the
# figure built from it regenerates with no downloads; a working copy with
# the whole archive keeps using the archive. Without this, the main-text
# balance figure was the one paper figure that could NOT be reproduced
# from the public repo (caught by running it with ~/DATA hidden,
# 2026-10-10).
_BUNDLED = os.path.join(ROOT, 'data')
DATA = (_BUNDLED + os.sep if os.path.isdir(_BUNDLED)
        else archive_root())      # $TRENCH_PULL_DATA or the default
from column_profiles_cache import ZC_KM, integrate_to_zc   # the only z_c and the only integrator
DX, ZC, Y, W = 1000.0, ZC_KM * 1e3, 2_900_000.0, 5
# ONE EXTRA SAMPLED ROW (2026-10-10). The grid is cell-centred, so rows run
# 0.5 .. ZC-0.5 km and `txz[-1]` -- the basal traction for F_B -- sat at the
# CENTRE of the last cell, 79.5 km, not on the control volume's lower face at
# z_c. The balance is defined for the 0..z_c volume, so its basal traction
# belongs at z_c; one additional row at ZC + DX/2 lets tau_b be interpolated
# there. The column integrals themselves use only the first NZ_C rows.
NZ_C = int(ZC / DX)
T_REF_MYR = 40.0                      # conventions §4b (mid-run rule)
from column_profiles_cache import T_MIN_MYR   # the only spin-up cut
# sqrt-x tick set: round numbers chosen to be near-EVENLY spaced once
# square-rooted (0, 7.1, 14.1, 22.4, 31.6, 44.7, 54.8), so the labels do
# not collide near the trench the way a linear-looking set does
SQRT_TICKS_KM = [0, 50, 200, 500, 1000, 2000, 3000]
PIN_KM = (1000.0, 2000.0)             # closure pinning window, SEAWARD of x_T (conventions §2.3)
RHO_M, G = 3300.0, 9.8                # Fluidity mantle density; no ocean (conventions §6.1)
SHEAR_SMOOTH_KM = 10.0                # smoothing of V before differentiating (notebook value)
# N/m at t = 40, from the committed column cache. RAISED 2026-10-10 from
# {'STD': 1.98e12, 'WAL': 1.75e12} when the vertical quadrature was fixed:
# the integral had spanned 0.5-79.5 km under a 0-80 km label, so every
# column integral was ~1.5 % low. The pin TRIPPED on the corrected chain,
# which is what it is for -- it is a tripwire, not a target, so it is
# updated with the reason rather than relaxed.
COMMITTED_TP_T40 = {'STD': 2.09e12, 'WAL': 1.78e12}


def snapshot_files(key):
    """Every archive snapshot of one run, in time order."""
    return natsort.natsorted(
        glob.glob(DATA + f'{key}_RefModel/subduction_with_LM*.pvtu'))


def load_snapshot(key, t_target=T_REF_MYR):
    for ff in snapshot_files(key):
        v = pv.read(ff)
        t = float(np.asarray(v['NormalSP::Time']).flat[0]) / 31557600.0 / 1e6
        if abs(t - t_target) < 1.0:
            return v, t
        del v
    raise SystemExit(f'{key}: no snapshot within 1 Myr of t = {t_target}')


def draw_direction_glyph(ax, terms, x0=0.365, y_top=1.0):
    """Force-direction glyph for THE SNAPSHOT BEING PLOTTED.

    `terms` is a list of (value, positive_means, colour, symbol, lw), one
    row per term, drawn top-down. `positive_means` is +1 if a positive
    value of that term is a force to the RIGHT (seaward) and -1 if it is
    a force to the LEFT (trench-ward); the arrow flips when the value is
    negative. See the module docstring for the per-term conventions.

    x0 is an AXES FRACTION, so on the sqrt scale 0.365 is ~465 km from
    the trench -- moved out from ~315 km (Dan, 2026-09-21) because the
    header and the GPE* label were running across the x_I line.
    """
    L = 0.09
    # MINIMAL GLYPH HEADER (Dan, 2026-09-28, option 5). The old header
    # named the figure ("trailing-plate force balance"), not the arrows,
    # so the glyph read as a legend; and "(current time step)" gestured at
    # variability without saying what varies. The header now describes the
    # arrows and nothing else. What varies is quantified in the caption:
    # Delta GPE* and F_B hold their direction at 100 % of snapshots in both
    # runs; Delta N_D reverses on 30 % of STD snapshots (11 of 37) and on
    # none in WAL.
    # header dropped ~0.1 in below the axes top (Dan, 2026-09-28); the rows
    # below it are placed relative to the same y_top, so they follow
    y_hdr = y_top - 0.10 / (ax.get_position().height
                            * ax.get_figure().get_figheight())
    ax.text(x0, y_hdr, 'direction at this step',
            transform=ax.transAxes, fontsize=11, style='italic',
            color='0.25', ha='center', va='top', linespacing=1.3)
    # Header clearance and row spacing are set in INCHES and converted to
    # this axes' own fraction. The rows sit in different-height panels
    # (height_ratios 1 : 1.4 : 1.8) while the text stays a fixed point
    # size, so a single hard-coded fraction that clears the header in the
    # tall panel runs straight through it in the short one.
    h_in = ax.get_position().height * ax.get_figure().get_figheight()
    y0 = y_hdr - 0.24 / h_in                 # first row, below the header
    ys = [y0 - (0.17 / h_in) * i for i in range(len(terms))]
    ax.plot([x0, x0], [min(ys) - 0.02, max(ys) + 0.02], color='0.5', lw=0.8,
            transform=ax.transAxes)
    for y, (val, pos_dir, c, lab, lw) in zip(ys, terms):
        s = pos_dir if val >= 0 else -pos_dir      # the actual direction
        ax.annotate('', xy=(x0 + s * L, y), xytext=(x0, y),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                    mutation_scale=14))
        ax.text(x0 + s * (L + 0.015), y, lab, transform=ax.transAxes,
                fontsize=10.5, color=c, va='center',
                ha='right' if s < 0 else 'left')


def compute(key, t_target=T_REF_MYR, pin_committed=None, verbose=True):
    """The balance at the snapshot nearest t_target. Scans the archive.

    ⚠ THIS SCANS FROM THE START OF THE RUN on every call. That is fine for
    a figure script, which asks once. A caller that wants EVERY snapshot
    must walk `snapshot_files` itself and call `compute_from` -- calling
    this in a loop is quadratic in reads and turns a 15-minute job into a
    four-hour one (measured, 2026-10-09).
    """
    v, t_myr = load_snapshot(key, t_target)
    return compute_from(v, t_myr, key, pin_committed=pin_committed,
                        verbose=verbose)


def compute_from(v, t_myr, key, pin_committed=None, verbose=True):
    """Every term of the trailing-plate balance from an ALREADY-READ mesh.

    Single implementation, shared by fig_balance_snapshot (the Delta form)
    and fig_balance_absolute (the form shifted by N_D(x_T), which shows
    the actual N_D rather than its change). Returns a dict of arrays in
    the analysis frame.
    """
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    if pin_committed is None:
        pin_committed = abs(t_myr - T_REF_MYR) < 1.0
    v.point_data['p'] = v['NormalSP::Pressure']
    v.point_data['tzz'] = v['NormalSP::Stress'][:, 4]
    v.point_data['txx'] = v['NormalSP::Stress'][:, 0]
    v.point_data['txz'] = -v['NormalSP::Stress'][:, 1]
    v.point_data['T'] = v['NormalSP::Temperature']
    v.point_data['fs'] = v['NormalSP::FreeSurface']
    vel = v['NormalSP::Velocity'] / 3.17098e-10
    v.point_data['vx'] = vel[:, 0]; v.point_data['vy'] = vel[:, 1]
    x0, x1, _, _, _, _ = v.bounds
    nx, nz = int((x1 - x0) / DX), NZ_C + 1      # +1: the row below z_c, for tau_b
    x = x0 + (np.arange(nx) + 0.5) * DX
    z = (np.arange(nz) + 0.5) * DX
    X, Z = np.meshgrid(x, z, indexing='xy')
    g = pv.StructuredGrid(X.T, (Y - Z).T, np.zeros_like(X.T)).sample(v)
    nxp, nzp, _ = g.dimensions
    val = g['vtkValidPointMask'].reshape((nxp, nzp), order='F').astype(bool)
    gf = make_field_extractor(g, nxp, nzp, val)
    p, txx, tzz, txz = gf('p'), gf('txx'), gf('tzz'), gf('txz')
    T, fs, vx, vz = gf('T'), gf('fs'), gf('vx'), -gf('vy')
    # native orientation (W31, 2026-09-28) -- no mirror
    p, txx, tzz, txz = (np.nan_to_num(a) for a in (p, txx, tzz, txz))
    # The free surface comes from the TOP-BOUNDARY NODES, not from the
    # top row of the sampled grid (which sits at z = 500 m and is not the
    # surface at all). See fluidity_helpers.surface_fs. `fs` is extracted
    # above only for the legacy chain; its top row is no longer used.
    # NATIVE ORIENTATION (2026-09-30): no reflection. Until this fix the
    # call passed mirror_x=True, left over from the mirrored frame, which
    # reflected the surface about the box centre while the stress fields
    # stayed native -- x_I landed ~400 km too far seaward and the ridge
    # pick ran into the sidewall.
    fs_top = surface_fs(v, x, mirror_x=False)
    xT, _ = pick_trench_3step(x, z, p, vx, subducting_side='left')
    ti = int(np.argmin(np.abs(x - xT)))
    iI, _ = find_first_isostatic_column(x, fs_top, xT, ti, DX, seaward_sign=-1)
    # THE CANONICAL RIDGE PICKER (conventions §3.4): the maximum surface
    # divergence, refined to the local topographic crest within +/-20 km.
    # This called find_ridge_x, the retired GLOBAL topographic pick, so the
    # snapshot figures used a different ridge from the time series and the
    # caches. On the corrected surface the two agree to a median 6 km (STD)
    # / 13 km (WAL) -- max 11 / 22 -- and 7 / 5 km at the reference epoch,
    # so nothing was badly wrong; but one quantity gets one routine, and
    # find_ridge_x is kept only as the xR_topo diagnostic in the column
    # cache. Fixed 2026-10-10.
    iz10 = int(np.argmin(np.abs(z - 10e3)))
    xR, iR = find_ridge_x_flow(x, vx[iz10], xT, seaward_sign=-1,
                               fs_top=fs_top)
    ca = lambda f, j: f[..., max(0, j - W):j + W + 1].mean(axis=-1)

    # resultants (notebook §7 chain, verbatim logic)
    # THE CELL RULE over the cells spanning 0..z_c (conventions §5.1; the
    # integrator and the defect it replaces are documented in
    # column_profiles_cache.integrate_to_zc). These were trapz over the
    # centres, i.e. 0.5..79.5 km under a 0..80 km label.
    zc_ = z[:NZ_C]
    col = lambda a: integrate_to_zc(a[:NZ_C, :].T, zc_, ZC, axis=1)
    Fd = col(txx - tzz)
    Sxx = col(-p + txx)
    gpe = -col(-p + tzz)
    # tau_b ON THE LOWER FACE at z_c, interpolated between the centres either
    # side of it (79.5 and 80.5 km) rather than taken at the last centre.
    tau_b = 0.5 * (txz[NZ_C - 1, :] + txz[NZ_C, :])
    FB_ = cumulative_trapezoid(tau_b, x, initial=0.0)

    # SHEAR-SUPPORTED TOPOGRAPHY w_tau (SYMBOLOGY §2), lifted from
    # fluidity_single_step.ipynb §w_actual_vs_w_tau: the vertical shear
    # resultant V = int tau_zx dz, smoothed 10 km before differentiating
    # (the raw field is too noisy to differentiate), then
    #     w_tau = (dV/dx) / (rho_m g)
    # the deflection that a static balance against the vertical shear load
    # would require. SIGN IS PINNED ON THE DATA below, not asserted. (The
    # reason used to be that this repo's V was the negative of the companion
    # register's -- the SYMBOLOGY §7.5 divergence. That is retired: V here
    # is int tau_zx dz in the model's own frame and carries the register's
    # sense, V(x_T) > 0. The ritual stays because an empirical pin is right
    # regardless, and it is what caught the frame change.)
    V = col(txz)
    dVdx = np.gradient(gaussian_filter1d(V, SHEAR_SMOOTH_KM * 1e3 / DX,
                                         mode='nearest'), x)
    w_tau = dVdx / (RHO_M * G)
    # sign ritual (conventions §1.2): pick the sign against the measured
    # surface over the flexural interval and PRINT it; never assert it
    w_meas = -fs_top
    # SEAWARD windows, written as s = xT - x (native orientation, 2026-09-30).
    # Both of these were still written x > xT from the mirrored frame, which
    # in the native frame samples the OVERRIDING plate.
    s_ = xT - x
    seg = (s_ > 0) & (s_ < 400e3)
    cc = float(np.corrcoef(w_tau[seg], w_meas[seg])[0, 1])
    if cc < 0:
        w_tau, cc = -w_tau, -cc
    # BOTH curves zero-referenced at the FIRST ISOSTATIC COLUMN (Dan,
    # 2026-09-21). x_I is where dV/dx = 0, so it is the natural zero for a
    # deflection curve and for its shear-supported counterpart alike, and
    # it matches the notebooks' own convention, w = fs_top[iI] - fs_top.
    w_meas = w_meas - w_meas[iI]
    w_tau = w_tau - w_tau[iI]
    if verbose:
        print(f'{key}: w_T measured {w_meas[ti]:+.0f} m, shear-supported '
              f'{w_tau[ti]:+.0f} m ({100*w_tau[ti]/w_meas[ti]:.0f} % of it); '
              f'r = {cc:+.2f} over s = 0..400 km seaward  [both zeroed at x_I]')
    d_Fd = Fd - ca(Fd, ti)
    d_Sxx = Sxx - ca(Sxx, ti)
    d_gpe = gpe - ca(gpe, ti)
    FB = FB_ - ca(FB_, ti)

    # sign pin: trench pull at x_I vs the committed series
    tp = ca(d_gpe, iI)
    if pin_committed:
        rel = abs(tp - COMMITTED_TP_T40[key]) / COMMITTED_TP_T40[key]
        assert rel < 0.05, f'{key}: ΔGPE*(x_I) = {tp/1e12:.2f} TN/m vs committed — chain broken'

    # pinned closure (conventions §2.3)
    res = d_Fd - d_gpe + FB
    pin = (s_ > PIN_KM[0] * 1e3) & (s_ < PIN_KM[1] * 1e3)
    res_const = res[pin].mean()
    res_pin = res - res_const
    if verbose:
        print(f'{key} t={t_myr:.1f}: ΔGPE*(x_I) {tp/1e12:+.2f} TN/m '
              f'({"pin OK" if pin_committed else "committed-value pin skipped: "
                 "not the reference epoch"}); '
              f'closure constant removed {res_const/1e12:+.3f} TN/m '
              f'(window s = {PIN_KM[0]:.0f}..{PIN_KM[1]:.0f} km seaward); '
              f'rms about pinned closure {res_pin[pin].std()/1e12:.3f} TN/m')


    return dict(x=x, xT=xT, xI=x[iI], xR=xR, t_myr=t_myr, z=z,
                fs_top=fs_top, Fd=Fd, Sxx=Sxx, gpe=gpe, V=V, w_tau=w_tau, w=w_meas,
                d_Fd=d_Fd, d_Sxx=d_Sxx, d_gpe=d_gpe, FB=FB,
                # COLUMN RESULTANTS AT ALL THREE COLUMNS, window-averaged
                # the same way (conventions §4.1). Only the trench one was
                # returned, so a figure wanting N_D(x_I) had to re-derive it
                # and would not have used the same window.
                Fd_T=ca(Fd, ti), Fd_I=ca(Fd, iI), Fd_R=ca(Fd, iR),
                res_pin=res_pin, res_const=res_const)
