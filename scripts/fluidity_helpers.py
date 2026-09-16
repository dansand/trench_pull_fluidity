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
# Pickers
# ---------------------------------------------------------------------------

def pick_trench_3step(x, z, p, vx,
                     p_depth_m=0.0,
                     vx_depth_m=5_000.0,
                     Wp1_km=800.0,
                     Wv_km=800.0,
                     Wp3_km=300.0,
                     expected_offset_km=50.0,
                     smooth_km=20.0,
                     subducting_side="left"):
    """Three-step non-stateful trench picker:
      1. coarse surface-pressure minimum
      2. velocity sign-change (plate boundary), mirror-aware
      3. directional, windowed local pressure minimum

    Returns (trench_x [m], info_dict).
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


def find_first_isostatic_column(x, fs_top, x_trench, tindx, DX,
                                buffer_km=30, window_km=500,
                                regional_ref_km=500, seaward_sign=+1):
    """Walk seaward of the trench (with a buffer) and return the first index
    where the free-surface elevation has recovered to the regional reference
    taken `regional_ref_km` seaward of the trench.

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


def find_ridge_x_flow(x, vx_row, x_trench, seaward_sign=+1,
                      smooth_km=15.0, min_offset_km=300.0):
    """Ridge column from the FLOW, not the topography.

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

    Returns (x_ridge, index).
    """
    dx = float(np.median(np.diff(x)))
    v = gaussian_filter1d(np.nan_to_num(vx_row), max(1.0, smooth_km * 1e3 / dx))
    div = np.gradient(v, x)
    if seaward_sign > 0:
        sea = x > x_trench + min_offset_km * 1e3
    else:
        sea = x < x_trench - min_offset_km * 1e3
        div = -div
    idx = np.where(sea)[0]
    j = idx[int(np.argmax(div[idx]))]
    return float(x[j]), int(j)


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
