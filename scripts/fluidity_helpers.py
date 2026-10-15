"""Shared helpers for the trench_pull_fluidity analysis.

Renamed from `fluidity_helpers.py` on 2026-09-16 to match the repository's
naming (one module per code, not per author). The model OUTPUTS analysed
here are the work of Cerpa et al. and are cited as such throughout —
see MODEL.md, the README, and `zenodo_materials/`; the rename is a
code-organisation change only and carries no implication about
attribution.

Provides field extraction, x-mirror, trench / first-isostatic-column / ridge
pickers, column-window averaging, simple [0, 1] normalisation, and the npz
cache loader for the time-evolution analysis.

Convention: x rightward positive, z downward positive (z = Y_SURFACE − y).
The off-diagonal stress sign flip σ_xz_zdown = −σ_xy_yup is applied at
extraction inside each notebook's load cell, not buried in any helper here.
"""

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Grid extraction
# ---------------------------------------------------------------------------

def make_field_extractor(interp, nx_pv, nz_pv, valid):
    """Build a `get_field(name)` closure for the supplied sampled grid.

    Used in each notebook's load cell, after `grid.sample(vtk_data)`:

        interp = grid.sample(vtk_data)
        nx_pv, nz_pv, _ = interp.dimensions
        valid = interp["vtkValidPointMask"].reshape((nx_pv, nz_pv), order="F").astype(bool)
        get_field = make_field_extractor(interp, nx_pv, nz_pv, valid)
        p = get_field("p")
        ...
    """
    def get_field(name):
        f = interp[name].reshape((nx_pv, nz_pv), order="F")
        f[~valid] = np.nan
        return f.T  # (z, x)
    return get_field


def mirror_fields_in_x(p, txx, tzz, txz, T, fs, vx, vz):
    """Mirror all 2D fields along x so subduction matches the MDOODZ convention
    (subducting plate on the right, slab descending leftward).

    Under x → −x reflection (z untouched):
      - scalars and diagonals (p, T, σ_xx, σ_zz, fs, vz) — np.flip only
      - off-diagonals (σ_xz) and x-velocity (vx)         — np.flip + sign flip
    """
    flip = lambda a: np.flip(a, axis=1)
    return (flip(p), flip(txx), flip(tzz), -flip(txz),
            flip(T), flip(fs), -flip(vx), flip(vz))


# ---------------------------------------------------------------------------
# WHERE THE MODEL ARCHIVE LIVES
# ---------------------------------------------------------------------------

def archive_root(required=True):
    """Root of the Cerpa et al. archive: $TRENCH_PULL_DATA, or the default.

    ONE RESOLVER, because the path was written out in eight places -- seven
    as `~/DATA/numerical_models/OUTPUTS/` and one, in
    fig_headline_tracking, as an ABSOLUTE path under the author's home
    directory, which no other machine could satisfy (2026-10-10).

    Resolution order:
      1. $TRENCH_PULL_DATA, if set -- what a second user sets;
      2. ~/DATA/numerical_models/OUTPUTS/, the author's layout and the
         default so nothing here needed changing.

    The directory must contain STD_RefModel/ and WAL_RefModel/, each a
    directory of subduction_with_LM*.pvtu snapshots, exactly as unpacked
    from the Zenodo record (https://zenodo.org/records/6817177).

    ⚠ ONLY THE CACHE BUILDERS NEED THIS. Every figure in the paper renders
    from the committed caches and the bundled reference snapshot in
    `data/`, so a reader reproducing the figures never sets it. It is
    needed to rebuild a cache or to work at another epoch.

    `required=False` returns the resolved path without checking it exists,
    for callers that only want to report it.
    """
    env = os.environ.get('TRENCH_PULL_DATA')
    root = os.path.expanduser(env or '~/DATA/numerical_models/OUTPUTS/')
    root = root.rstrip(os.sep) + os.sep
    if required and not os.path.isdir(root):
        raise SystemExit(
            f'model archive not found: {root}\n'
            f'  This script rebuilds a cache, so it needs the full Cerpa et al.\n'
            f'  archive (1.4 GB): https://zenodo.org/records/6817177\n'
            f'  Unpack it so that STD_RefModel/ and WAL_RefModel/ sit inside one\n'
            f'  directory, then point this at it:\n'
            f'      export TRENCH_PULL_DATA=/path/to/OUTPUTS\n'
            + ('' if env else '  (TRENCH_PULL_DATA is unset; the default above was tried.)\n')
            + f'  You do NOT need the archive to reproduce any figure in the paper --\n'
              f'  those render from the committed caches and data/.')
    return root


# ---------------------------------------------------------------------------
# Pickers
# ---------------------------------------------------------------------------

def pick_trench_3step(x, z, p, vx,
                     p_depth_m=0.0,
                     vx_depth_m=5_000.0,
                     Wp1_km=800.0,       # ACCEPTED AND IGNORED -- see below
                     Wv_km=800.0,
                     Wp3_km=300.0,
                     expected_offset_km=50.0,
                     smooth_km=20.0,
                     subducting_side="left"):
    """Three-step non-stateful trench picker:
      1. coarse surface-pressure minimum -- GLOBAL, not windowed
      2. velocity sign-change (plate boundary), mirror-aware
      3. directional, windowed local pressure minimum

    Returns (trench_x [m], info_dict).

    ⚠ `Wp1_km` IS ACCEPTED AND IGNORED. Step 1 takes the global argmin of
    the pressure row; there is no window on it, and there has not been for
    as long as the signature has advertised one. The name survives because
    callers pass it. Do not read it as a guard on stage 1 -- it is a fossil
    of a three-stage algorithm whose first stage no longer constrains
    anything (noted 2026-10-10).

    WHY THAT IS TOLERABLE, measured over all 37 retained snapshots of both
    runs rather than argued: the global pressure minimum typically lands
    318-327 km from the trench and as far as 439 km away, so stage 1 is a
    poor estimate on its own -- but the 800 km velocity window of stage 2
    comfortably contains the true plate-boundary crossing from there, and
    the FINAL pick agrees with the trench's own topographic minimum, taken
    on the boundary nodes, to within 2 km at the 90th percentile and 3 km
    at worst. The pick is sound; its first stage is just not doing the job
    its parameter list implies.

    The more direct definition -- guard on the velocity crossing, then take
    the boundary-topography minimum in a directional window, keeping the
    pressure minimum as an independent check -- became available only when
    the surface read was corrected (2026-10-10). It is not adopted here:
    it would move x_T by at most 3 km, and every cache would need
    rebuilding to gain it.
    """
    iz_p  = 0 if p_depth_m == 0.0 else int(np.argmin(np.abs(z - p_depth_m)))
    iz_vx = int(np.argmin(np.abs(z - vx_depth_m)))

    p_row  = p[iz_p, :].copy()
    vx_row = vx[iz_vx, :].copy()

    # ---- step 1: coarse pressure minimum --------------------------------
    j0 = int(np.nanargmin(p_row))
    x_p0 = float(x[j0])

    # ---- step 2: velocity sign-change in a window around x_p0 ----------
    Wv = Wv_km * 1000.0
    in_v = (x >= x_p0 - Wv) & (x <= x_p0 + Wv)
    x_vw  = x[in_v]
    vx_vw = vx_row[in_v]

    if x_vw.size < 5:
        raise RuntimeError("Velocity window too small")

    dx = np.median(np.diff(x))
    k = max(1, int((smooth_km * 1000.0) / dx))
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k) / k
    vx_s = np.convolve(vx_vw, kernel, mode="same")

    s = np.sign(vx_s)
    s[s == 0] = np.nan
    cross = np.where(
        np.isfinite(s[:-1]) &
        np.isfinite(s[1:]) &
        (s[:-1] * s[1:] < 0)
    )[0]

    if cross.size == 0:
        jv = int(np.nanargmin(np.abs(vx_s)))
        x_v = float(x_vw[jv])
        vx_method = "min|vx|"
    else:
        xc = []
        for i in cross:
            x1, x2 = x_vw[i], x_vw[i+1]
            v1, v2 = vx_s[i], vx_s[i+1]
            xc.append(x1 - v1 * (x2 - x1) / (v2 - v1))
        xc = np.array(xc)

        # Mirror-aware: pick the velocity zero-crossing on the
        # subducting-plate side.  Without this, mirrored runs (subducting
        # plate on the right) lock onto the wrong crossing.
        if subducting_side == "right":
            x_v = float(np.max(xc))
            vx_method = "zero_cross_rightmost"
        else:
            x_v = float(np.min(xc))
            vx_method = "zero_cross_leftmost"

    # ---- step 3: directional, windowed pressure minimum ----------------
    Wp3 = Wp3_km * 1000.0
    expected_offset = expected_offset_km * 1000.0

    if subducting_side == "left":
        mask = (x < x_v) & (x >= x_v - Wp3)
        x_expected = x_v - expected_offset
    else:
        mask = (x > x_v) & (x <= x_v + Wp3)
        x_expected = x_v + expected_offset

    xw = x[mask]
    pw = p_row[mask]

    if xw.size < 5:
        raise RuntimeError("Pressure window too small")

    good = np.isfinite(pw)
    xw, pw = xw[good], pw[good]

    if xw.size < 5:
        raise RuntimeError("Too many NaNs in pressure window")

    is_min = (pw[1:-1] < pw[:-2]) & (pw[1:-1] < pw[2:])
    idx = np.where(is_min)[0] + 1

    if idx.size == 0:
        j = int(np.argmin(pw))
        trench_x = float(xw[j])
        p_method = "fallback_global_min"
    else:
        j = idx[np.argmin(np.abs(xw[idx] - x_expected))]
        trench_x = float(xw[j])
        p_method = "local_min_windowed"

    info = {
        "x_p0":            x_p0,
        "x_v":             x_v,
        "trench_x":        trench_x,
        "vx_method":       vx_method,
        "p_method":        p_method,
        "subducting_side": subducting_side,
    }
    return trench_x, info


def find_first_isostatic_column(x, V, x_trench, seaward_sign=+1):
    """The column seaward of the trench at which V is least.

    V is the vertical shear resultant int tau_zx dz to z_c, on the x grid;
    the caller forms it (one integrator,
    `column_profiles_cache.integrate_to_zc`).

    Integrated vertical equilibrium gives dV/ds = -rho_m g w, so dV/dx = 0
    marks the column carrying no non-isostatic load -- the first isostatic
    column. Compensated topography produces no shear gradient, so the
    criterion is blind to the plate's cooling subsidence and to the
    basin-scale tilt, and needs no reference elevation.

    ⚠ TAKEN AS THE MINIMUM OF V, NOT AS A ZERO OF ITS DERIVATIVE. V falls
    from the trench, turns at x_I and climbs again over the outer rise, and
    an extremum is far better conditioned than a zero of a numerical
    derivative: on the raw field the minimum is stable, while the
    derivative of that same field changes sign ~30 times across the plate.

    Returns the index of the first isostatic column.
    """
    sea = (x - x_trench) * seaward_sign > 0
    if sea.sum() < 10:
        raise RuntimeError('find_first_isostatic_column: only '
                           f'{sea.sum()} columns seaward of the trench — '
                           'check seaward_sign')
    idx = np.where(sea)[0]
    return int(idx[np.argmin(np.nan_to_num(V)[sea])])


def find_first_isostatic_column_topo(x, fs_top, x_trench, tindx, DX,
                                     buffer_km=30, window_km=500,
                                     regional_ref_km=500, seaward_sign=+1):
    """RETIRED AS THE PICK, kept as the cross-check (conventions §3.2).

    Walks seaward of the trench and returns the first index at which the
    topography is back up to the topography `regional_ref_km` seaward.
    Carries the cooling-plus-tilt bias documented above; use it only to
    quote the offset against the dV/dx pick, never to pick a column.

    Returns (i_first_isostatic, i_far_reference).
    """
    i_far = int(np.argmin(np.abs(x - (x_trench + seaward_sign * regional_ref_km * 1000.0))))
    target = float(fs_top[i_far])

    buffer_grid = int(buffer_km * 1000.0 / DX)
    window_grid = int(window_km * 1000.0 / DX)
    i_start = max(0, min(len(x) - 1, tindx + seaward_sign * buffer_grid))
    i_end   = max(0, min(len(x) - 1, tindx + seaward_sign * window_grid))
    step    = int(np.sign(seaward_sign)) or +1

    for i in range(i_start, i_end, step):
        if fs_top[i] >= target:
            return i, i_far
    return i_end, i_far


def surface_fs(vtk_data, x, mirror_x=False, tol_m=1.0):
    """Free-surface field read at the TOP-BOUNDARY NODES, on the x grid.

    ``mirror_x`` DEFAULTS TO FALSE since the move to the model's native
    orientation (2026-09-30). It reflects the boundary-node x-coordinates
    about the box centre and is correct ONLY for the retired mirrored
    frame. Left in place for the legacy notebooks; passing True from a
    native-frame script reflects the surface away from the fields it is
    read alongside, which mislocates every topography-based pick.

    Returns fs(x) in the analysis frame, the same sense as the raw
    `NormalSP::FreeSurface` field (so the downward-positive deflection is
    ``w = -fs``).

    WHY THIS EXISTS (2026-09-21). The notebooks and every script before
    this read the free surface as ``fs[0, :]`` -- the top row of the
    sampled analysis grid, which sits at z = DX/2 = 500 m. That is NOT the
    surface. This model has NO deforming mesh: the top is flat at
    y = y_max and `FreeSurface` is a BOUNDARY field, meaningful on the top
    nodes and carrying something else immediately below. Down the trench
    column the field steps from +2842 m at z = 0 to -133 m at z = 500 m.

    Reading at 500 m acted as a heavy low-pass filter: it understated the
    reference-snapshot trench by 1130 m (STD, 629 vs 1761) and 390 m (WAL),
    put an ~11 km wiggle into the profile that the surface does not have,
    and biased the far field by ~80 m. Verified against a fully
    independent route -- the equivalent topography sigma_zz(0)/rho_g --
    which agrees with this reading to 1 m across the whole plate.

    No filtering is applied and none is needed: a 9 km median filter moves
    the trench value by 2 m and a +/-5 km window mean by 6 m.
    """
    pts = np.asarray(vtk_data.points)
    fsf = np.asarray(vtk_data['NormalSP::FreeSurface'])
    top = np.abs(pts[:, 1] - pts[:, 1].max()) < tol_m
    if top.sum() < 10:
        raise RuntimeError('surface_fs: no flat top boundary found — this '
                           'model may have a deforming mesh, in which case '
                           'the surface must come from geometry instead')
    xb = pts[top, 0]
    if mirror_x:
        xb = (pts[:, 0].min() + pts[:, 0].max()) - xb
    o = np.argsort(xb)
    return np.interp(x, xb[o], fsf[top][o])


def find_ridge_x_flow(x, vx_row, x_trench, seaward_sign=+1,
                      smooth_km=15.0, min_offset_km=300.0,
                      fs_top=None, crest_half_km=20.0, crest_smooth_km=3.0):
    """Ridge column from the FLOW, refined to the local topographic crest.

    TWO STEPS (ANALYSIS_CONVENTIONS §3.4, ruled by Dan 2026-09-26,
    specified 2026-09-27), because each fixes what the other gets wrong:

      locate  the spreading axis as the maximum surface divergence
              dvx/dx seaward of the trench -- robust, defined at every
              snapshot, and it does not wander (see below).
      refine  to the highest point of the smoothed free surface within
              +/- `crest_half_km`. The divergence maximum IS the
              spreading axis, and the spreading axis sits in the AXIAL
              VALLEY: measured on six snapshots per run, the local crest
              is 50-350 m higher and lies a median 5-20 km away,
              systematically INBOARD (negative offset on 9 of 12).

    +/- 20 km measured, not assumed: the gain from 20 to 30 km is under
    40 m in every snapshot tested, while +/- 50 km escapes to a
    neighbouring high in 1 of 6 STD and 3 of 6 WAL snapshots -- the
    failure mode the flow pick exists to avoid.

    The refinement is SKIPPED if `fs_top` is None, which preserves the
    pre-2026-09-27 behaviour for any caller that has not been updated.

    The ridge is the spreading centre: the maximum of the surface
    divergence dvx/dx seaward of the trench. Adopted 2026-09-16 after the
    surface-based pick (`find_ridge_x`) flipped 313 km in a single step in
    WAL at 70 Myr — the late-run surface carries several topographic highs
    within 350 m of one another and a shallowest-point pick resolves the
    ambiguity discontinuously. Dan's ParaView inspection settled it: the
    ridge is clearly identifiable in the velocity field and does not
    migrate appreciably.

    Measured behaviour over both runs (37 snapshots each): largest single
    step 15 km (STD) / 25 km (WAL), against 102 / 313 km for the
    topographic pick, and defined at every snapshot (the vx = 0 crossing
    is not). Sits a median 15-25 km seaward of the topographic pick at
    epochs where that pick is sane.

    Parameters
    ----------
    x        : column positions [m], analysis frame
    vx_row   : horizontal velocity along x at a shallow level (analysis
               frame; a row of the sampled grid, ~10 km depth)
    x_trench : trench position [m]
    min_offset_km : ignore the near-trench zone, where the flexural
               velocity field has its own structure
    fs_top   : free surface along x [m]. If given, the pick is refined to
               the local topographic crest; if None, it is not.
    crest_half_km : half-width of the crest search about the flow pick

    Returns (x_ridge, index).
    """
    dx = float(np.median(np.diff(x)))
    v = gaussian_filter1d(np.nan_to_num(vx_row), max(1.0, smooth_km * 1e3 / dx))
    div = np.gradient(v, x)
    # The surface divergence dv_x/dx is INVARIANT under x -> -x: v_x and
    # d/dx each change sign. An earlier version negated it on the
    # seaward_sign < 0 branch, which turned the search into a minimum and
    # returned the exclusion boundary instead of the ridge. That branch had
    # never been exercised (fluidity mirrored before picking); fixed
    # 2026-09-28 while un-mirroring the analysis frame (W31).
    if seaward_sign > 0:
        sea = x > x_trench + min_offset_km * 1e3
    else:
        sea = x < x_trench - min_offset_km * 1e3
    idx = np.where(sea)[0]
    j = idx[int(np.argmax(div[idx]))]
    if fs_top is None:
        return float(x[j]), int(j)
    # refine to the local topographic crest (conventions §3.4)
    s_fs = gaussian_filter1d(np.nan_to_num(fs_top),
                             max(1.0, crest_smooth_km * 1e3 / dx))
    w = np.where(np.abs(x - x[j]) <= crest_half_km * 1e3)[0]
    k = int(w[int(np.argmax(s_fs[w]))])
    return float(x[k]), k


def find_ridge_x(x, fs_top, x_trench, seaward_sign=+1,
                 buffer_km=200, smooth_km=50):
    """Locate the mid-ocean ridge as the highest point of the free-surface
    field, restricted to columns seaward of the trench.

    Returns (x_ridge [m], i_ridge).
    """
    DX = float(np.median(np.diff(x)))
    sigma_pts = max(1, int(smooth_km * 1000.0 / DX))
    fs_smooth = gaussian_filter1d(fs_top, sigma=sigma_pts, mode="nearest")

    buf_m = buffer_km * 1000.0
    if seaward_sign > 0:
        seaward = x > (x_trench + buf_m)
    else:
        seaward = x < (x_trench - buf_m)

    if not seaward.any():
        raise RuntimeError("No columns seaward of trench (after buffer)")

    fs_search = np.where(seaward, fs_smooth, -np.inf)
    i_ridge = int(np.argmax(fs_search))
    return float(x[i_ridge]), i_ridge


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Thermal structure: the effective age of a column
# ---------------------------------------------------------------------------

# READ FROM THE ARCHIVE, NOT CHOSEN (zenodo_materials/constants_weakasth2.py
# lines 134 and 141). Declared once here because two copies of a physical
# constant is how this repo got two integration depths and two spin-up cuts.
KAPPA = 1.0e-6              # thermal diffusivity, m^2/s
T_SURFACE_K = 273.0
T_MANTLE_K = 1573.0
SEC_PER_MYR = 365.0 * 86400.0 * 1e6   # the MODEL's year (365 d), per the
                                      # same file's Age_SP conversion. The
                                      # Julian year used elsewhere in the
                                      # suite differs by 0.07 %, below any
                                      # digit this quantity is quoted to.
AGE_ISO_C = 900.0           # the inversion isotherm (conventions §6.3) --
                            # the same isotherm h_m calibrates to, so the
                            # two quantities refer to one level
# TWO ISOTHERM SETS, because they answer different questions. The CORE set
# is where the profile is genuinely half-space cooling; the FULL set adds
# the two ends, 400 C (shallow, where the surface boundary condition and the
# flexural depression matter) and 1100 C (close enough to the 1573 K mantle
# that the error function is flattening out). The full spread is roughly
# twice the core spread and the ends are what drive it -- which is a
# property of the profile, not a defect of the inversion, and is tabulated
# rather than hidden by quoting only the core.
AGE_ISO_CORE_C = (600.0, 800.0, 900.0, 1000.0)
AGE_ISO_CHECK_C = (400.0, 600.0, 800.0, 900.0, 1000.0, 1100.0)


def hsc_age(temp_K, z, iso_C=None, kappa=KAPPA):
    """Effective thermal age [Myr] of ONE column, by half-space cooling.

    Inverts the column's own temperature profile for the age that would put
    the `iso_C` isotherm at the depth where it is actually found:

        z_iso = 2 erf^-1(theta) sqrt(kappa t),
        theta = (T_iso - T_s) / (T_m - T_s)

    `temp_K` is the profile in KELVIN on the depth axis `z` [m], surface
    first. Returns NaN if the isotherm is not reached inside the profile.

    ⚠ This is NOT elapsed model time. At the trench the column is
    continuously resupplied from further up the plate, so the age grows more
    slowly than the clock -- 0.72 Myr per Myr in both fluidity runs.

    ⚠ The result scales as 1/kappa. It inherits the archive's value and is a
    declared quantity, not a correct one (conventions §6.3).
    """
    from scipy.special import erfinv
    iso_C = AGE_ISO_C if iso_C is None else iso_C
    # the model's surface IS T_SURFACE_K, so degrees above the surface and
    # degrees Celsius are the same number here
    prof = np.asarray(temp_K, dtype=float) - T_SURFACE_K
    z = np.asarray(z, dtype=float)
    j = np.where(prof >= iso_C)[0]
    if not len(j):
        return np.nan
    k = j[0]
    z_iso = np.interp(iso_C, prof[:k + 1], z[:k + 1])
    eta = erfinv(iso_C / (T_MANTLE_K - T_SURFACE_K))
    return (z_iso / (2.0 * eta)) ** 2 / kappa / SEC_PER_MYR


def hsc_age_spread(temp_K, z, isotherms=AGE_ISO_CHECK_C):
    """Fractional disagreement between ages inverted at several isotherms.

    THE TEST THAT MAKES THE AGE A MEASUREMENT. If the column really is
    half-space cooling, every isotherm returns the same age; if it is not,
    the quantity is not well defined and no single isotherm rescues it.
    Returns (spread, ages) with spread = (max - min) / median.
    """
    a = np.array([hsc_age(temp_K, z, iso) for iso in isotherms])
    return float((np.nanmax(a) - np.nanmin(a)) / np.nanmedian(a)), a


def col_avg(arr, ix, win=1):
    """Average over [ix − win, ix + win] along the last (x) axis.

    Works for 1D (depth-integrated) and 2D (z, x) arrays.  `win = 0` returns
    the single column at `ix`.
    """
    lo = max(0, ix - win)
    hi = min(arr.shape[-1] - 1, ix + win)
    return arr[..., lo:hi+1].mean(axis=-1)


def norm01(arr):
    """Linear rescale to [0, 1]:  0 ← arr.min(),  1 ← arr.max().
    Returns zeros if the array is constant.
    """
    arr = np.asarray(arr, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def norm_TR(arr, i_T, i_R):
    """Linear rescale anchored at two specific indices:
    `arr[i_T] → 0` and `arr[i_R] → 1`.

    Used for the trench-to-ridge normalisation in the column-comparison
    figures.  Distinct from `norm01`, which uses min/max.
    """
    arr = np.asarray(arr, dtype=float)
    a_T, a_R = arr[i_T], arr[i_R]
    return (arr - a_T) / (a_R - a_T)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_records(key, outputs_dir=Path("outputs")):
    """Load a previously-saved time-evolution npz cache for the given model
    key.  Returns the loaded NpzFile or None (with a warning) if the cache
    doesn't exist yet — re-run §4 with that MODEL_KEY first.
    """
    path = outputs_dir / f"time_evolution_{key}.npz"
    if not path.exists():
        print(f"⚠  {path.name} not found — re-run §4 with MODEL_KEY = {key!r}")
        return None
    return np.load(path)


def depth_ticks(fig, step=20.0, max_span_km=300.0):
    """Put every shallow depth axis in a figure on `step`-km ticks.

    Dan's ruling 2026-10-09. Matplotlib's default on a 0-200 km depth axis
    is 25 km, which with z_c = 80 km never puts a tick on the integration
    depth -- the one depth the reader most wants to locate. 20 km does,
    and it also lands on the 40/60/100 km marks the thickness discussion
    uses.

    Applies to INVERTED axes (depth increases downward) spanning no more
    than max_span_km, so whole-mantle panels -- slab buoyancy to 1800 km,
    the model-evolution sections to 1500 -- keep their own sensible ticks.
    Colourbars are skipped: their axes are not inverted, but the guard is
    explicit because a colourbar's range can look like a depth span.
    """
    from matplotlib.ticker import MultipleLocator
    for ax in fig.axes:
        if ax.get_label() == '<colorbar>':
            continue
        lo, hi = ax.get_ylim()
        if lo > hi and (lo - hi) <= max_span_km:      # inverted = depth
            ax.yaxis.set_major_locator(MultipleLocator(step))
